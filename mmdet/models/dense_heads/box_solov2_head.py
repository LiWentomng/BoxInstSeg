import mmcv
import torch
import cv2
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import normal_init
from mmdet.core import InstanceData, multi_apply, mask_matrix_nms
from ..builder import build_loss, HEADS
from mmcv.cnn import bias_init_with_prob, ConvModule
from mmcv.runner import BaseModule
import numpy as np
from scipy import ndimage
from mmdet.ops.tree_filter.modules.tree_filter import MinimumSpanningTree, TreeFilter2D

@HEADS.register_module()
class BoxSOLOv2Head(BaseModule):
    def __init__(self,
                 num_classes,
                 in_channels,
                 seg_feat_channels=256,
                 stacked_convs=4,
                 strides=(4, 8, 16, 32, 64),
                 base_edge_list=(16, 32, 64, 128, 256),
                 scale_ranges=((8, 32), (16, 64), (32, 128), (64, 256), (128, 512)),
                 sigma=0.2,
                 num_grids=None,
                 cate_down_pos=0,
                 loss_cate=None,
                 loss_boxpro=None,
                 loss_levelset=None,
                 conv_cfg=None,
                 norm_cfg=None,
                 use_dcn_in_tower=False,
                 type_dcn=None,
                 init_cfg=None):
        super(BoxSOLOv2Head, self).__init__(init_cfg=init_cfg)
        self.num_classes = num_classes
        self.seg_num_grids = num_grids
        self.cate_out_channels = self.num_classes
        self.in_channels = in_channels
        self.seg_feat_channels = seg_feat_channels
        self.stacked_convs = stacked_convs
        self.strides = strides
        self.sigma = sigma
        self.stacked_convs = stacked_convs
        self.cate_down_pos = cate_down_pos
        self.base_edge_list = base_edge_list
        self.scale_ranges = scale_ranges

        self.loss_cate = build_loss(loss_cate)
        self.loss_boxpro = build_loss(loss_boxpro)
        self.loss_levelset = build_loss(loss_levelset)

        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.use_dcn_in_tower = use_dcn_in_tower
        self.type_dcn = type_dcn
        self.mst = MinimumSpanningTree(TreeFilter2D.norm2_distance)
        self.tree_filter = TreeFilter2D()
        self._init_layers()

    def _init_layers(self):
        norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)
        self.feature_convs = nn.ModuleList()
        self.cate_convs = nn.ModuleList()
        self.kernel_convs = nn.ModuleList()

        if self.use_dcn_in_tower:
            cfg_conv = dict(type=self.type_dcn)
        else:
            cfg_conv = self.conv_cfg

        # mask feature
        for i in range(4):
            convs_per_level = nn.Sequential()
            if i == 0:
                one_conv = ConvModule(
                    self.in_channels,
                    self.seg_feat_channels,
                    3,
                    padding=1,
                    conv_cfg=cfg_conv,
                    norm_cfg=self.norm_cfg,
                    bias=norm_cfg is None)
                convs_per_level.add_module('conv' + str(i), one_conv)
                self.feature_convs.append(convs_per_level)
                continue
            for j in range(i):
                if j == 0:
                    if i == 3:
                        in_channel = self.in_channels + 2
                    else:
                        in_channel = self.in_channels
                    one_conv = ConvModule(
                        in_channel,
                        self.seg_feat_channels,
                        3,
                        padding=1,
                        conv_cfg=cfg_conv,
                        norm_cfg=self.norm_cfg,
                        bias=norm_cfg is None)
                    convs_per_level.add_module('conv' + str(j), one_conv)
                    one_upsample = nn.Upsample(
                        scale_factor=2, mode='bilinear', align_corners=False)
                    convs_per_level.add_module(
                        'upsample' + str(j), one_upsample)
                    continue
                one_conv = ConvModule(
                    self.seg_feat_channels,
                    self.seg_feat_channels,
                    3,
                    padding=1,
                    conv_cfg=cfg_conv,
                    norm_cfg=self.norm_cfg,
                    bias=norm_cfg is None)
                convs_per_level.add_module('conv' + str(j), one_conv)
                one_upsample = nn.Upsample(
                    scale_factor=2,
                    mode='bilinear',
                    align_corners=False)
                convs_per_level.add_module('upsample' + str(j), one_upsample)
            self.feature_convs.append(convs_per_level)

        for i in range(self.stacked_convs):
            chn = self.in_channels + 2 if i == 0 else self.seg_feat_channels
            self.kernel_convs.append(
                ConvModule(
                    chn,
                    self.seg_feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=cfg_conv,
                    norm_cfg=norm_cfg,
                    bias=norm_cfg is None))

            chn = self.in_channels if i == 0 else self.seg_feat_channels
            self.cate_convs.append(
                ConvModule(
                    chn,
                    self.seg_feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=cfg_conv,
                    norm_cfg=norm_cfg,
                    bias=norm_cfg is None))

        self.solo_cate = nn.Conv2d(
            self.seg_feat_channels, self.cate_out_channels, 3, padding=1)

        self.solo_kernel = nn.Conv2d(
            self.seg_feat_channels, self.seg_feat_channels, 1, padding=0)
        self.solo_mask = nn.Conv2d(
            self.seg_feat_channels, self.seg_feat_channels, 1, padding=0)

        self.levelset_bottom = nn.Conv2d(self.seg_feat_channels, 5, 3, padding=1)

    def init_weights(self):
        super(BoxSOLOv2Head, self).init_weights()
        for m in self.feature_convs:
            s=len(m)
            for i in range(s):
                if i%2 == 0:
                    normal_init(m[i].conv, std=0.01)

        for m in self.cate_convs:
            normal_init(m.conv, std=0.01)
        for m in self.kernel_convs:
            normal_init(m.conv, std=0.01)

        bias_cate = bias_init_with_prob(0.01)
        normal_init(self.solo_cate, std=0.01, bias=bias_cate)
        normal_init(self.solo_kernel, std=0.01)

        normal_init(self.levelset_bottom, std=0.01)
        normal_init(self.solo_mask, std=0.01)

    def forward(self, feats, eval=False):

        feats = [feat.float() for feat in feats]
        new_feats = self.split_feats(feats)
        featmap_sizes = [featmap.size()[-2:] for featmap in new_feats]
        upsampled_size = (feats[0].shape[-2], feats[0].shape[-3])
        kernel_pred, cate_pred = multi_apply(self.forward_single, new_feats,
                                             list(range(len(self.seg_num_grids))),
                                             eval=eval)

        # add coord for p5
        x_range = torch.linspace(-1, 1, feats[-2].shape[-1], device=feats[-2].device)
        y_range = torch.linspace(-1, 1, feats[-2].shape[-2], device=feats[-2].device)
        y, x = torch.meshgrid(y_range, x_range)

        y = y.expand([feats[-2].shape[0], 1, -1, -1])
        x = x.expand([feats[-2].shape[0], 1, -1, -1])
        coord_feat = torch.cat([x, y], 1)
        feature_add_all_level = self.feature_convs[0](feats[0])

        for i in range(1, 3):
            feature_add_all_level = feature_add_all_level + self.feature_convs[i](feats[i])
        feature_add_all_level = feature_add_all_level + self.feature_convs[3](torch.cat([feats[3], coord_feat], 1))
        feature_pred = self.solo_mask(feature_add_all_level)

        levelset_feats = self.levelset_bottom(feature_pred) #5 channel
        N, c, h, w = feature_pred.shape
        feature_pred = feature_pred.view(-1, h, w).unsqueeze(0)
        ins_pred = []

        for i in range(5):
            kernel = kernel_pred[i].permute(0, 2, 3, 1).contiguous().view(-1, c).unsqueeze(-1).unsqueeze(-1)
            ins_i = F.conv2d(feature_pred, kernel, groups=N).view(N, self.seg_num_grids[i] ** 2, h, w)
            if not eval:
                ins_i = F.interpolate(ins_i, size=(featmap_sizes[i][0] * 2, featmap_sizes[i][1] * 2), mode='bilinear')
            if eval:
                ins_i = ins_i.sigmoid()
            ins_pred.append(ins_i)
        return ins_pred, cate_pred, levelset_feats

    def split_feats(self, feats):
        return (F.interpolate(feats[0], scale_factor=0.5, mode='bilinear'),
                feats[1],
                feats[2],
                feats[3],
                F.interpolate(feats[4], size=feats[3].shape[-2:], mode='bilinear'))

    def forward_single(self, x, idx, eval=False):
        kernel_feat = x
        cate_feat = x

        x_range = torch.linspace(-1, 1, kernel_feat.shape[-1], device=kernel_feat.device)
        y_range = torch.linspace(-1, 1, kernel_feat.shape[-2], device=kernel_feat.device)
        y, x = torch.meshgrid(y_range, x_range)
        y = y.expand([kernel_feat.shape[0], 1, -1, -1])
        x = x.expand([kernel_feat.shape[0], 1, -1, -1])
        coord_feat = torch.cat([x, y], 1)

        kernel_feat = torch.cat([kernel_feat, coord_feat], 1)
        for i, kernel_layer in enumerate(self.kernel_convs):
            if i == self.cate_down_pos:
                seg_num_grid = self.seg_num_grids[idx]
                kernel_feat = F.interpolate(kernel_feat, size=seg_num_grid, mode='bilinear')
            kernel_feat = kernel_layer(kernel_feat)
        kernel_pred = self.solo_kernel(kernel_feat)

        # cate branch
        for i, cate_layer in enumerate(self.cate_convs):
            if i == self.cate_down_pos:
                seg_num_grid = self.seg_num_grids[idx]
                cate_feat = F.interpolate(cate_feat, size=seg_num_grid, mode='bilinear')
            cate_feat = cate_layer(cate_feat)

        cate_pred = self.solo_cate(cate_feat)

        if eval:
            cate_pred = cate_pred.sigmoid()
            local_max = F.max_pool2d(cate_pred, (2,2), stride=1, padding=1)
            keep_mask = local_max[:, :, :-1, :-1] == cate_pred
            cate_pred = cate_pred * keep_mask
            cate_pred = cate_pred.permute(0, 2, 3, 1)
        return kernel_pred, cate_pred

    def loss(self,
             ins_preds,
             cate_preds,
             levelset_feats,
             gt_bbox_list,
             gt_label_list,
             gt_mask_list,
             img,
             img_metas,
             cfg,
             gt_bboxes_ignore=None):

        num_imgs = len(gt_label_list)
        img_list = []
        lst_feats_list = []
        for i in range(num_imgs):
            img_list.append(img[i])
            lst_feats_list.append(levelset_feats[i])

        featmap_sizes = [featmap.size()[-2:] for featmap in
                         ins_preds]

        ins_label_list, cate_label_list, ins_ind_label_list, scale_img_list, scale_lstfeat_list = multi_apply(
            self.solo_target_single,
            gt_bbox_list,
            gt_label_list,
            gt_mask_list,
            img_list,
            lst_feats_list,
            featmap_sizes=featmap_sizes)

        # ins
        ins_labels = [torch.cat([ins_labels_level_img[ins_ind_labels_level_img, ...]
                                 for ins_labels_level_img, ins_ind_labels_level_img in
                                 zip(ins_labels_level, ins_ind_labels_level)], 0)
                      for ins_labels_level, ins_ind_labels_level in zip(zip(*ins_label_list), zip(*ins_ind_label_list))]

        # img_target
        img_target = [
            torch.cat([ins_img_level_img.repeat(ins_labels_level_img[ins_ind_labels_level_img, ...].shape[0], 1, 1, 1)
                       for ins_labels_level_img, ins_ind_labels_level_img, ins_img_level_img in
                       zip(ins_labels_level, ins_ind_labels_level, ins_img_level)], 0)
            for ins_labels_level, ins_ind_labels_level, ins_img_level in
            zip(zip(*ins_label_list), zip(*ins_ind_label_list), zip(*scale_img_list))]

        # deep_feats_target
        lst_target = [torch.cat([ins_lstfeat_level_img.expand(
            ins_labels_level_img[ins_ind_labels_level_img, ...].shape[0], ins_lstfeat_level_img.shape[-3],
            ins_lstfeat_level_img.shape[-2], ins_lstfeat_level_img.shape[-1])
                                 for ins_labels_level_img, ins_ind_labels_level_img, ins_lstfeat_level_img in
                                 zip(ins_labels_level, ins_ind_labels_level, ins_lstfeat_level)], 0)
                      for ins_labels_level, ins_ind_labels_level, ins_lstfeat_level in
                      zip(zip(*ins_label_list), zip(*ins_ind_label_list), zip(*scale_lstfeat_list))]


        ins_preds = [torch.cat([ins_preds_level_img[ins_ind_labels_level_img, ...]
                                for ins_preds_level_img, ins_ind_labels_level_img in
                                zip(ins_preds_level, ins_ind_labels_level)], 0)
                     for ins_preds_level, ins_ind_labels_level in zip(ins_preds, zip(*ins_ind_label_list))]

        ins_ind_labels = [
            torch.cat([ins_ind_labels_level_img.flatten()
                       for ins_ind_labels_level_img in ins_ind_labels_level])
            for ins_ind_labels_level in zip(*ins_ind_label_list)
        ]
        flatten_ins_ind_labels = torch.cat(ins_ind_labels)

        num_ins = flatten_ins_ind_labels.int().sum()

        loss_project = []
        loss_levelset = []

        for ins_pred, box_mask, img_target, lst_target in zip(ins_preds, ins_labels, img_target, lst_target):
            if ins_pred.size()[0] == 0:
                continue

            mask_pred = torch.sigmoid(ins_pred.unsqueeze(dim=1))
            box_mask_target = box_mask.unsqueeze(dim=1).to(dtype=mask_pred.dtype)
            loss_project.append(self.loss_boxpro(mask_pred, box_mask_target))

            back_scores = 1.0 - mask_pred
            mask_scores_concat = torch.cat((mask_pred, back_scores), dim=1)

            mask_scores_phi = mask_scores_concat * box_mask_target
            img_target_wbox = img_target * box_mask_target

            pixel_num = box_mask_target.sum((1, 2, 3))
            pixel_num = torch.clamp(pixel_num, min=1)

            loss_img_lst = self.loss_levelset(mask_scores_phi, img_target_wbox, pixel_num) * 0.05

            img_mst_tree = self.mst(img_target)
            deep_stru_feature_img = self.tree_filter(mask_pred, img_target, img_mst_tree)

            lst_mst_tree = self.mst(lst_target)
            deep_stru_feature_lst = self.tree_filter(deep_stru_feature_img, lst_target, lst_mst_tree, low_tree=False)
            high_feature = torch.cat((deep_stru_feature_img, deep_stru_feature_lst), dim=1) * box_mask_target

            loss_feat_lst = self.loss_levelset(mask_scores_phi, high_feature, pixel_num) * 5.0

            # img levelset loss and high-level feature levelset loss
            loss_levelset_term = loss_img_lst + loss_feat_lst
            loss_levelset.append(loss_levelset_term)

        loss_project = torch.cat(loss_project).mean()
        loss_levelset = torch.cat(loss_levelset).mean()

        # cate
        cate_labels = [
            torch.cat([cate_labels_level_img.flatten()
                       for cate_labels_level_img in cate_labels_level])
            for cate_labels_level in zip(*cate_label_list)
        ]
        flatten_cate_labels = torch.cat(cate_labels)

        cate_preds = [
            cate_pred.permute(0, 2, 3, 1).reshape(-1, self.cate_out_channels)
            for cate_pred in cate_preds
        ]
        flatten_cate_preds = torch.cat(cate_preds)

        loss_cate = self.loss_cate(flatten_cate_preds, flatten_cate_labels, avg_factor=num_ins + 1)

        return dict(
            loss_boxpro=loss_project,
            loss_levelset=loss_levelset,
            loss_cate=loss_cate)

    def solo_target_single(self,
                           gt_bboxes_raw,
                           gt_labels_raw,
                           gt_masks_raw,
                           norm_img,
                           lst_feats,
                           featmap_sizes=None):

        device = gt_labels_raw[0].device

        # ins
        gt_areas = torch.sqrt((gt_bboxes_raw[:, 2] - gt_bboxes_raw[:, 0]) * (
                gt_bboxes_raw[:, 3] - gt_bboxes_raw[:, 1]))

        ins_label_list = []
        cate_label_list = []
        ins_ind_label_list = []
        scale_img_list = []
        scale_lst_feat_list = []
        for (lower_bound, upper_bound), stride, featmap_size, num_grid \
                in zip(self.scale_ranges, self.strides, featmap_sizes, self.seg_num_grids):

            scale_img = F.interpolate(norm_img.unsqueeze(dim=0), size=featmap_size, mode='bilinear')
            scale_img_list.append(scale_img)

            scale_lst_feat = F.interpolate(lst_feats.unsqueeze(dim=0), size=featmap_size, mode='bilinear')
            scale_lst_feat_list.append(scale_lst_feat)

            ins_label = torch.zeros([num_grid ** 2, featmap_size[0], featmap_size[1]], dtype=torch.uint8, device=device)
            cate_label = torch.zeros([num_grid, num_grid], dtype=torch.int64, device=device) + self.num_classes
            ins_ind_label = torch.zeros([num_grid ** 2], dtype=torch.bool, device=device)

            hit_indices = ((gt_areas >= lower_bound) & (gt_areas <= upper_bound)).nonzero().flatten()
            # hit_indices = torch.nonzero(((gt_areas >= lower_bound) & (gt_areas <= upper_bound)), as_tuple=False).flatten()

            if len(hit_indices) == 0:
                ins_label_list.append(ins_label)
                cate_label_list.append(cate_label)
                ins_ind_label_list.append(ins_ind_label)
                continue
            gt_bboxes = gt_bboxes_raw[hit_indices]
            gt_labels = gt_labels_raw[hit_indices]
            gt_masks = gt_masks_raw[hit_indices.cpu().numpy(), ...]

            half_ws = 0.5 * (gt_bboxes[:, 2] - gt_bboxes[:, 0]) * self.sigma
            half_hs = 0.5 * (gt_bboxes[:, 3] - gt_bboxes[:, 1]) * self.sigma

            output_stride = stride / 2

            for seg_mask, gt_label, half_h, half_w in zip(gt_masks, gt_labels, half_hs, half_ws):
                if seg_mask.sum() < 10:
                    continue
                # mass center
                upsampled_size = (featmap_sizes[0][0] * 4, featmap_sizes[0][1] * 4)
                center_h, center_w = ndimage.measurements.center_of_mass(seg_mask)
                coord_w = int((center_w / upsampled_size[1]) // (1. / num_grid))
                coord_h = int((center_h / upsampled_size[0]) // (1. / num_grid))

                # left, top, right, down
                top_box = max(0, int(((center_h - half_h) / upsampled_size[0]) // (1. / num_grid)))
                down_box = min(num_grid - 1, int(((center_h + half_h) / upsampled_size[0]) // (1. / num_grid)))
                left_box = max(0, int(((center_w - half_w) / upsampled_size[1]) // (1. / num_grid)))
                right_box = min(num_grid - 1, int(((center_w + half_w) / upsampled_size[1]) // (1. / num_grid)))

                top = max(top_box, coord_h - 1)
                down = min(down_box, coord_h + 1)
                left = max(coord_w - 1, left_box)
                right = min(right_box, coord_w + 1)

                cate_label[top:(down + 1), left:(right + 1)] = gt_label
                seg_mask = mmcv.imrescale(seg_mask, scale=1. / output_stride)
                seg_mask = torch.from_numpy(seg_mask).to(device=device)

                for i in range(top, down + 1):
                    for j in range(left, right + 1):
                        label = int(i * num_grid + j)
                        ins_label[label, :seg_mask.shape[0], :seg_mask.shape[1]] = seg_mask
                        ins_ind_label[label] = True

            ins_label_list.append(ins_label)
            cate_label_list.append(cate_label)
            ins_ind_label_list.append(ins_ind_label)
        return ins_label_list, cate_label_list, ins_ind_label_list, scale_img_list, scale_lst_feat_list


    def get_seg(self, seg_preds, cate_preds, levelset_feats, img_metas, cfg, rescale=None):

        assert len(seg_preds) == len(cate_preds)
        num_levels = len(cate_preds)
        featmap_size = seg_preds[0].size()[-2:]

        result_list = []
        for img_id in range(len(img_metas)):
            cate_pred_list = [
                cate_preds[i][img_id].view(-1, self.cate_out_channels).detach() for i in range(num_levels)
            ]
            seg_pred_list = [
                seg_preds[i][img_id].detach() for i in range(num_levels)
            ]

            cate_pred_list = torch.cat(cate_pred_list, dim=0)
            seg_pred_list = torch.cat(seg_pred_list, dim=0)

            result = self.get_seg_single(
                cate_pred_list, 
                seg_pred_list,
                featmap_size,
                img_meta=img_metas[img_id],
                cfg=cfg)
            result_list.append(result)
        return result_list


    def get_seg_single(self,
                       cate_preds,
                       seg_preds,
                       featmap_size,
                       img_meta,
                       cfg=None):

        def empty_results(results, cls_scores):
            results.scores = cls_scores.new_ones(0)
            results.masks = cls_scores.new_zeros(0, *results.ori_shape[:2])
            results.labels = cls_scores.new_ones(0)
            return results

        assert len(cate_preds) == len(seg_preds)
        results = InstanceData(img_meta)

        img_shape = results.img_shape
        ori_shape = results.ori_shape

        h, w, _ = img_shape
        upsampled_size_out = (featmap_size[0] * 4, featmap_size[1] * 4)

        inds = (cate_preds > cfg.score_thr)
        # category scores.
        cate_scores = cate_preds[inds]

        if len(cate_scores) == 0:
            return empty_results(results, cate_scores)

        # category labels.
        inds = inds.nonzero()
        cate_labels = inds[:, 1]

        # strides.
        size_trans = cate_labels.new_tensor(self.seg_num_grids).pow(2).cumsum(0)
        strides = cate_scores.new_ones(size_trans[-1])
        n_stage = len(self.seg_num_grids)
        strides[:size_trans[0]] *= self.strides[0]
        for ind_ in range(1, n_stage):
            strides[size_trans[ind_ - 1]:size_trans[ind_]] *= self.strides[ind_]
        strides = strides[inds[:, 0]]

        # masks.
        seg_preds = seg_preds[inds[:, 0]]
        seg_masks = seg_preds > cfg.mask_thr
        sum_masks = seg_masks.sum((1, 2)).float()

        # filter.
        keep = sum_masks > strides
        if keep.sum() == 0:
            return empty_results(results, cate_scores)
        seg_masks = seg_masks[keep, ...]
        seg_preds = seg_preds[keep, ...]
        sum_masks = sum_masks[keep]
        cate_scores = cate_scores[keep]
        cate_labels = cate_labels[keep]

        # mask scoring.
        seg_scores = (seg_preds * seg_masks.float()).sum((1, 2)) / sum_masks
        cate_scores *= seg_scores

        # Matrix NMS
        scores, labels, _, keep_inds  = mask_matrix_nms(
            seg_masks, 
            cate_labels, 
            cate_scores,
            filter_thr=cfg.filter_thr,
            nms_pre=cfg.nms_pre,
            max_num=cfg.max_per_img,
            kernel=cfg.kernel, 
            sigma=cfg.sigma, 
            mask_area=sum_masks)

        seg_preds = seg_preds[keep_inds]
        seg_preds = F.interpolate(seg_preds.unsqueeze(0),
                                  size=upsampled_size_out,
                                  mode='bilinear')[:, :, :h, :w]
        seg_masks = F.interpolate(seg_preds,
                                  size=ori_shape[:2],
                                  mode='bilinear').squeeze(0)

        masks = seg_masks > cfg.mask_thr

        results.masks = masks
        results.labels = labels
        results.scores = scores

        return results





