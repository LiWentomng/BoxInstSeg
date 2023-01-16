import torch
import numpy as np
import cv2
from mmdet.core import bbox2result
from ..builder import DETECTORS, build_head
from .single_stage import SingleStageDetector
import mmcv
from mmdet.core.visualization.image import imshow_det_bboxes


@DETECTORS.register_module()
class CondInst(SingleStageDetector):
    """Implementation of `CondInst <https://arxiv.org/abs/2003.05664>
    and BoxInst <https://arxiv.org/abs/2012.02310> `_"""

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 mask_branch,
                 mask_head,
                 segm_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(CondInst, self).__init__(backbone, neck, bbox_head, train_cfg,
                                       test_cfg, pretrained, init_cfg)
        self.mask_branch = build_head(mask_branch)
        self.mask_head = build_head(mask_head)
        self.segm_head = None if segm_head is None else \
            build_head(segm_head)

    def forward_dummy(self, img):
        raise NotImplementedError

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None):
        if gt_masks is not None:
            H, W = img.size(2), img.size(3)
            tensor_masks = []
            for masks in gt_masks:
                masks = masks.expand(H, W, 0, 0)
                tensor_masks.append(
                    masks.to_tensor(dtype=torch.uint8, device=img.device))
            gt_masks = tensor_masks

        x = self.extract_feat(img)
        cls_score, bbox_pred, centerness, param_pred = \
            self.bbox_head(x, self.mask_head.param_conv)
        bbox_head_loss_inputs = (cls_score, bbox_pred, centerness) + (
            gt_bboxes, gt_labels, img_metas)
        losses, coors, level_inds, img_inds, gt_inds = self.bbox_head.loss(
            *bbox_head_loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)

        mask_feat = self.mask_branch(x)
        if self.segm_head is not None:
            segm_pred = self.segm_head(x[0])
            loss_segm = self.segm_head.loss(segm_pred, gt_masks, gt_labels)
            losses.update(loss_segm)

        inputs = (cls_score, centerness, param_pred,
                  coors, level_inds, img_inds, gt_inds)
        param_pred, coors, level_inds, img_inds, gt_inds = self.mask_head.training_sample(
            *inputs)
        mask_pred = self.mask_head(
            mask_feat, param_pred, coors, level_inds, img_inds)
        loss_mask = self.mask_head.loss(img, img_metas, mask_pred, gt_inds, gt_bboxes,
                                        gt_masks, gt_labels)
        losses.update(loss_mask)
        return losses

    def simple_test(self, img, img_metas, rescale=False):
        feat = self.extract_feat(img)
        outputs = self.bbox_head.simple_test(
            feat, self.mask_head.param_conv, img_metas, rescale=rescale)
        det_bboxes, det_labels, det_params, det_coors, det_level_inds = zip(
            *outputs)
        bbox_results = [
            bbox2result(det_bbox, det_label, self.bbox_head.num_classes)
            for det_bbox, det_label in zip(det_bboxes, det_labels)
        ]

        mask_feat = self.mask_branch(feat)
        mask_results = self.mask_head.simple_test(
            mask_feat,
            det_labels,
            det_params,
            det_coors,
            det_level_inds,
            img_metas,
            self.bbox_head.num_classes,
            rescale=rescale)

        return list(zip(bbox_results, mask_results))

    def aug_test(self, imgs, img_metas, rescale=False):
        raise NotImplementedError

    def show_result(self,
                    img,
                    result,
                    score_thr=0.3,
                    bbox_color=(72, 101, 241),
                    text_color=(72, 101, 241),
                    mask_color=None,
                    thickness=2,
                    font_size=13,
                    win_name='',
                    show=False,
                    wait_time=0,
                    out_file=None,
                    **kwargs):
        img = mmcv.imread(img)
        img = img.copy()
        if isinstance(result, tuple):
            bbox_result, segm_result = result
            if isinstance(segm_result, tuple):
                segm_result = segm_result[0]
        else:
            bbox_result, segm_result = result, None
        bboxes = np.vstack(bbox_result)
        labels = [
            np.full(bbox.shape[0], i, dtype=np.int32)
            for i, bbox in enumerate(bbox_result)
        ]
        labels = np.concatenate(labels)
        # draw segmentation masks
        segms = None
        if segm_result is not None and len(labels) > 0:  # non empty
            segms = mmcv.concat_list(segm_result)
            if isinstance(segms[0], torch.Tensor):
                segms = torch.stack(segms, dim=0).detach().cpu().numpy()
            else:
                segms = np.stack(segms, axis=0)
        # if out_file specified, do not show image in window
        if out_file is not None:
            show = False
        # draw bounding boxes
        img = imshow_det_bboxes(
            img,
            bboxes,
            labels,
            segms,
            class_names=self.CLASSES,
            score_thr=score_thr,
            bbox_color=bbox_color,
            text_color=text_color,
            mask_color=mask_color,
            thickness=thickness,
            font_size=font_size,
            win_name=win_name,
            show=show,
            wait_time=wait_time,
            out_file=out_file)

        if not (show or out_file):
            return img
