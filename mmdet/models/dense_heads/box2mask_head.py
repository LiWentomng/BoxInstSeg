import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import Conv2d, build_plugin_layer, caffe2_xavier_init
from mmcv.cnn.bricks.transformer import (build_positional_encoding,
                                         build_transformer_layer_sequence)
from mmcv.ops import point_sample
from mmcv.runner import ModuleList

from mmdet.core import build_assigner, build_sampler, reduce_mean, multi_apply
from mmdet.models.utils import preprocess_panoptic_gt
from mmdet.models.utils import _scale_target
from ..builder import HEADS, build_loss
from .anchor_free_head import AnchorFreeHead
from mmdet.ops.tree_filter.modules.tree_filter import MinimumSpanningTree, TreeFilter2D
from mmcv.runner import force_fp32
from mmdet.models.losses import LCM


@HEADS.register_module()
class Box2MaskHead(AnchorFreeHead):
    """Implements the Box2Mask head.
    """

    def __init__(self,
                 in_channels,
                 feat_channels,
                 out_channels,
                 num_things_classes=80,
                 num_stuff_classes=53,
                 num_queries=100,
                 num_transformer_feat_level=3,
                 pixel_decoder=None,
                 enforce_decoder_input_project=False,
                 transformer_decoder=None,
                 positional_encoding=None,
                 loss_cls=None,
                 loss_box=None,
                 loss_mask=None,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None,
                 **kwargs):
        super(AnchorFreeHead, self).__init__(init_cfg)
        self.num_things_classes = num_things_classes
        self.num_stuff_classes = num_stuff_classes
        self.num_classes = self.num_things_classes + self.num_stuff_classes
        self.num_queries = num_queries
        self.num_transformer_feat_level = num_transformer_feat_level
        self.num_heads = transformer_decoder.transformerlayers.\
            attn_cfgs.num_heads
        self.num_transformer_decoder_layers = transformer_decoder.num_layers
        assert pixel_decoder.encoder.transformerlayers.\
            attn_cfgs.num_levels == num_transformer_feat_level
        pixel_decoder_ = copy.deepcopy(pixel_decoder)
        pixel_decoder_.update(
            in_channels=in_channels,
            feat_channels=feat_channels,
            out_channels=out_channels)
        self.pixel_decoder = build_plugin_layer(pixel_decoder_)[1]
        self.transformer_decoder = build_transformer_layer_sequence(
            transformer_decoder)
        self.decoder_embed_dims = self.transformer_decoder.embed_dims

        self.decoder_input_projs = ModuleList()
        # from low resolution to high resolution
        for _ in range(num_transformer_feat_level):
            if (self.decoder_embed_dims != feat_channels or enforce_decoder_input_project):
                self.decoder_input_projs.append(
                    Conv2d(feat_channels, self.decoder_embed_dims, kernel_size=1))
            else:
                self.decoder_input_projs.append(nn.Identity()) #
        self.decoder_positional_encoding = build_positional_encoding(
            positional_encoding)

        # nn.Embedding
        self.query_embed = nn.Embedding(self.num_queries, feat_channels)  #
        self.query_feat = nn.Embedding(self.num_queries, feat_channels) #

        # from low resolution to high resolution
        self.level_embed = nn.Embedding(self.num_transformer_feat_level, feat_channels)

        self.cls_embed = nn.Linear(feat_channels, self.num_classes + 1)
        self.mask_embed = nn.Sequential(
            nn.Linear(feat_channels, feat_channels), nn.ReLU(inplace=True),
            nn.Linear(feat_channels, feat_channels), nn.ReLU(inplace=True),
            nn.Linear(feat_channels, out_channels))

        self.levelset_bottom = nn.Conv2d(256, 1, 3, padding=1)

        self.test_cfg = test_cfg
        self.train_cfg = train_cfg
        if train_cfg:
            self.assigner = build_assigner(self.train_cfg.assigner)
            self.sampler = build_sampler(self.train_cfg.sampler, context=self)

        self.class_weight = loss_cls.class_weight
        self.loss_cls = build_loss(loss_cls)
        self.loss_box = build_loss(loss_box)
        self.loss_mask = build_loss(loss_mask)
        #tree filter
        self.mst = MinimumSpanningTree(TreeFilter2D.norm2_distance)
        self.tree_filter = TreeFilter2D()

    def init_weights(self):
        for m in self.decoder_input_projs:
            if isinstance(m, Conv2d):
                caffe2_xavier_init(m, bias=0)

        self.pixel_decoder.init_weights()

        for p in self.transformer_decoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)


    def preprocess_gt(self, gt_labels_list, gt_masks_list, gt_semantic_segs,
                      img_metas):
        """Preprocess the ground truth for all images.
        """
        num_things_list = [self.num_things_classes] * len(gt_labels_list)
        num_stuff_list = [self.num_stuff_classes] * len(gt_labels_list)
        if gt_semantic_segs is None:
            gt_semantic_segs = [None] * len(gt_labels_list)

        targets = multi_apply(preprocess_panoptic_gt, gt_labels_list,
                              gt_masks_list, gt_semantic_segs, num_things_list,
                              num_stuff_list, img_metas)
        labels, masks = targets
        return labels, masks


    def get_targets(self, cls_scores_list, mask_preds_list, gt_labels_list,
                    gt_masks_list, img_metas):
        """Compute classification and mask targets for all images for a decoder
        layer.
        """
        (labels_list, label_weights_list, mask_targets_list, mask_weights_list,
         pos_inds_list,
         neg_inds_list) = multi_apply(self._get_target_single, cls_scores_list,
                                      mask_preds_list, gt_labels_list,
                                      gt_masks_list, img_metas)

        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return (labels_list, label_weights_list, mask_targets_list,
                mask_weights_list, num_total_pos, num_total_neg)


    def _get_target_single(self, cls_score, mask_pred, gt_labels, gt_masks,
                           img_metas):
        """Compute classification and mask targets for one image.
        """

        target_shape = gt_masks.shape[-2:]

        mask_pred = F.interpolate(
            mask_pred.unsqueeze(1),
            target_shape,
            mode='bilinear',
            align_corners=False)

        gt_masks = gt_masks.unsqueeze(dim=1)
        # assign and sample
        assign_result = self.assigner.assign(cls_score, mask_pred,
                                             gt_labels, gt_masks,
                                             img_metas)

        sampling_result = self.sampler.sample(assign_result, mask_pred,
                                              gt_masks)
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds

        # label target
        labels = gt_labels.new_full((self.num_queries, ),
                                    self.num_classes,
                                    dtype=torch.long)
        labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        label_weights = gt_labels.new_ones((self.num_queries, ))

        # mask target
        mask_targets = gt_masks[sampling_result.pos_assigned_gt_inds]
        mask_weights = mask_pred.new_zeros((self.num_queries, ))
        mask_weights[pos_inds] = 1.0

        return (labels, label_weights, mask_targets, mask_weights, pos_inds,
                neg_inds)


    @force_fp32(apply_to=('all_cls_scores', 'all_mask_preds'))
    def loss(self, all_cls_scores, all_mask_preds, all_lst_feats, gt_labels_list,
             gt_masks_list, img_metas, norm_img):
        """Loss function.
        """
        num_dec_layers = len(all_cls_scores)
        all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]
        all_gt_masks_list = [gt_masks_list for _ in range(num_dec_layers)]
        img_metas_list = [img_metas for _ in range(num_dec_layers)]
        img_list = [norm_img for _ in range(num_dec_layers)]

        losses_cls, loss_project, loss_levelset = multi_apply(
            self.loss_single, all_cls_scores, all_mask_preds, all_lst_feats,
            all_gt_labels_list, all_gt_masks_list, img_metas_list, img_list)

        loss_dict = dict()
        # loss from the last decoder layer
        loss_dict['loss_cls'] = losses_cls[-1]
        loss_dict['loss_project'] = loss_project[-1]
        loss_dict['loss_levelset'] = loss_levelset[-1]

        # loss from other decoder layers
        num_dec_layer = 0
        for loss_cls_i, loss_project_i, loss_levelset_i in zip(losses_cls[:-1], loss_project[:-1], loss_levelset[:-1]):
            loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_project'] = loss_project_i
            loss_dict[f'd{num_dec_layer}.loss_levelset'] = loss_levelset_i

            num_dec_layer += 1
        return loss_dict


    def loss_single(self, cls_scores, mask_preds, lst_feat, gt_labels_list,
                    gt_masks_list, img_metas, norm_img):

        """Loss function for outputs from a single decoder layer.
        """

        pred_shape = mask_preds.shape[-2:]

        norm_img = F.interpolate(norm_img, pred_shape, mode='bilinear', align_corners=False)
        lst_feat = F.interpolate(lst_feat, pred_shape, mode='bilinear', align_corners=False)

        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        mask_preds_list = [mask_preds[i] for i in range(num_imgs)]
        (labels_list, label_weights_list, mask_targets_list, mask_weights_list,
         num_total_pos,num_total_neg) = self.get_targets(cls_scores_list, mask_preds_list,
                                           gt_labels_list, gt_masks_list,
                                           img_metas)

        labels = torch.stack(labels_list, dim=0)
        label_weights = torch.stack(label_weights_list, dim=0)
        mask_targets = torch.cat(mask_targets_list, dim=0)
        mask_weights = torch.stack(mask_weights_list, dim=0)

        # classfication loss
        cls_scores = cls_scores.flatten(0, 1)
        labels = labels.flatten(0, 1)
        label_weights = label_weights.flatten(0, 1)

        class_weight = cls_scores.new_tensor(self.class_weight)
        loss_cls = self.loss_cls(
            cls_scores,
            labels,
            label_weights,
            avg_factor=class_weight[labels].sum())

        mask_preds = mask_preds[mask_weights > 0]

        if mask_targets.shape[0] == 0:
            # zero match
            # continue
            loss_project = mask_preds.sum()
            loss_levelset = mask_preds.sum()
            return loss_cls, loss_project, loss_levelset

        scale_img = _scale_target(norm_img)
        scale_feat = _scale_target(lst_feat)
        norm_img_tree = self.mst(scale_img)
        lst_feat_tree = self.mst(scale_feat)

        target_img_idx = mask_weights.sum(dim=1).cpu().numpy().tolist()
        target_img_list = []
        target_lst_list = []
        target_img_tree_list = []
        target_feat_tree_list = []

        for i in range(num_imgs):
            repeat_num = int(target_img_idx[i])
            if repeat_num == 0:
                continue
            norm_imgs = norm_img[i:i+1].repeat(repeat_num, 1, 1, 1)
            lst_feats = lst_feat[i:i+1].repeat(repeat_num, 1, 1, 1)
            target_img_tree = norm_img_tree[i:i+1].repeat(repeat_num, 1, 1)
            target_lstfeat_tree = lst_feat_tree[i:i+1].repeat(repeat_num, 1, 1)

            target_img_list.append(norm_imgs)
            target_lst_list.append(lst_feats)
            target_img_tree_list.append(target_img_tree)
            target_feat_tree_list.append(target_lstfeat_tree)

        img_targets = torch.cat(target_img_list, dim=0)
        lst_targets = torch.cat(target_lst_list, dim=0)
        target_img_tree = torch.cat(target_img_tree_list, dim=0)
        target_feat_tree = torch.cat(target_feat_tree_list, dim=0)

        pred_shape = mask_preds.shape[-2:]
        box_mask_targets = F.interpolate(mask_targets.to(mask_preds.dtype), pred_shape, mode='bilinear', align_corners=False)
        mask_preds = torch.sigmoid(mask_preds.unsqueeze(dim=1))

        loss_project = self.loss_box(mask_preds, box_mask_targets).mean()

        back_scores = 1.0 - mask_preds
        mask_scores_concat = torch.cat((mask_preds, back_scores), dim=1)
        mask_scores_phi = mask_scores_concat * box_mask_targets
        img_target_wbox = img_targets * box_mask_targets
        pixel_num = box_mask_targets.sum((1, 2, 3))
        pixel_num = torch.clamp(pixel_num, min=1)
        # level set on img
        loss_img_lst = self.loss_mask(mask_scores_phi, img_target_wbox, pixel_num).mean() * 0.05

        #scale the targets
        img_targets = _scale_target(img_targets)
        lst_targets = _scale_target(lst_targets)
        mask_preds = _scale_target(mask_preds)

        _stru_feature_img = self.tree_filter(feature_in=mask_preds, embed_in=img_targets,
                                                 tree=target_img_tree)
        _stru_feature_lst = self.tree_filter(_stru_feature_img, lst_targets, target_feat_tree,
                                                 low_tree=False)
        stru_feature_img = F.interpolate(_stru_feature_img, pred_shape, mode='bilinear', align_corners=False)
        stru_feature_lst = F.interpolate(_stru_feature_lst, pred_shape, mode='bilinear', align_corners=False)
        deep_stru_feats = torch.cat((stru_feature_img, stru_feature_lst), dim=1) * box_mask_targets
        # level set on deep feature
        loss_feat_lst = self.loss_mask(mask_scores_phi, deep_stru_feats, pixel_num).mean() * 5.0

        box_mask_targets = _scale_target(box_mask_targets)
        # local consistency module
        loss_lcm = 0.2 * LCM(img_targets, mask_preds, box_mask_targets)

        loss_levelset = loss_img_lst + loss_feat_lst + loss_lcm

        return loss_cls, loss_project, loss_levelset


    def forward_head(self, decoder_out, mask_feature, attn_mask_target_size):
        """Forward for head part which is called after every decoder layer.
        """
        decoder_out = self.transformer_decoder.post_norm(decoder_out)
        decoder_out = decoder_out.transpose(0, 1)
        cls_pred = self.cls_embed(decoder_out)
        mask_embed = self.mask_embed(decoder_out)
        mask_pred = torch.einsum('bqc,bchw->bqhw', mask_embed, mask_feature)
        attn_mask = F.interpolate(
            mask_pred,
            attn_mask_target_size,
            mode='bilinear',
            align_corners=False)

        attn_mask = attn_mask.flatten(2).unsqueeze(1).repeat(
            (1, self.num_heads, 1, 1)).flatten(0, 1)
        attn_mask = attn_mask.sigmoid() < 0.5
        attn_mask = attn_mask.detach()

        levelset_feat = self.levelset_bottom(mask_feature)

        return cls_pred, mask_pred, attn_mask, levelset_feat

    def forward(self, feats, img_metas):

        batch_size = len(img_metas)
        # pixel decoder MSDeformAttnPixelDecoder
        mask_features, multi_scale_memorys = self.pixel_decoder(feats)

        # multi_scale_memorys (from low resolution to high resolution)
        decoder_inputs = []
        decoder_positional_encodings = []
        for i in range(self.num_transformer_feat_level):
            decoder_input = self.decoder_input_projs[i](multi_scale_memorys[i])
            decoder_input = decoder_input.flatten(2).permute(2, 0, 1)
            level_embed = self.level_embed.weight[i].view(1, 1, -1)
            decoder_input = decoder_input + level_embed
            mask = decoder_input.new_zeros((batch_size, ) + multi_scale_memorys[i].shape[-2:],
                dtype=torch.bool)
            decoder_positional_encoding = self.decoder_positional_encoding(mask)
            decoder_positional_encoding = decoder_positional_encoding.flatten(2).permute(2, 0, 1)
            decoder_inputs.append(decoder_input)
            decoder_positional_encodings.append(decoder_positional_encoding)

        query_feat = self.query_feat.weight.unsqueeze(1).repeat((1, batch_size, 1))
        query_embed = self.query_embed.weight.unsqueeze(1).repeat((1, batch_size, 1))

        cls_pred_list = []
        mask_pred_list = []
        lst_feat_list = []

        cls_pred, mask_pred, attn_mask, lst_feat = self.forward_head(query_feat, mask_features, multi_scale_memorys[0].shape[-2:])
        cls_pred_list.append(cls_pred)
        mask_pred_list.append(mask_pred)
        lst_feat_list.append(lst_feat)

        for i in range(self.num_transformer_decoder_layers):
            level_idx = i % self.num_transformer_feat_level
            # if a mask is all True(all background), then set it all False.
            attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False

            # cross_attn + self_attn
            layer = self.transformer_decoder.layers[i]
            attn_masks = [attn_mask, None]
            query_feat = layer(
                query=query_feat,
                key=decoder_inputs[level_idx],
                value=decoder_inputs[level_idx],
                query_pos=query_embed,
                key_pos=decoder_positional_encodings[level_idx],
                attn_masks=attn_masks,
                query_key_padding_mask=None,
                key_padding_mask=None)
            cls_pred, mask_pred, attn_mask, lst_feat = self.forward_head(
                query_feat, mask_features, multi_scale_memorys[
                    (i + 1) % self.num_transformer_feat_level].shape[-2:])

            cls_pred_list.append(cls_pred)
            mask_pred_list.append(mask_pred)
            lst_feat_list.append(lst_feat)

        return cls_pred_list, mask_pred_list, lst_feat_list


    def forward_train(self,
                      feats,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_masks,
                      gt_semantic_seg,
                      gt_bboxes_ignore=None,
                      norm_img=None):
        """Forward function for training mode.
        """
        # not consider ignoring bboxes
        assert gt_bboxes_ignore is None
        # forward
        all_cls_scores, all_mask_preds, all_lst_feats = self(feats, img_metas)
        # preprocess ground truth
        gt_labels, gt_masks = self.preprocess_gt(gt_labels, gt_masks, gt_semantic_seg, img_metas)
        # loss
        losses = self.loss(all_cls_scores, all_mask_preds, all_lst_feats,gt_labels, gt_masks,
                           img_metas, norm_img)

        return losses

    def simple_test(self, feats, img_metas, **kwargs):
        """Test without augmentaton.
        """
        all_cls_scores, all_mask_preds, all_levelset_feats = self(feats, img_metas)
        mask_cls_results = all_cls_scores[-1]
        mask_pred_results = all_mask_preds[-1]

        img_shape = img_metas[0]['batch_input_shape']
        mask_pred_results = F.interpolate(
            mask_pred_results,
            size=(img_shape[0], img_shape[1]),
            mode='bilinear',
            align_corners=False)

        return mask_cls_results, mask_pred_results

