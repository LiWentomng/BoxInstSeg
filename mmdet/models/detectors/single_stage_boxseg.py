import torch.nn as nn
import torch
import warnings
from ..builder import DETECTORS, build_backbone, build_head, build_neck
from .base import BaseDetector
from mmdet.core import bbox2result
import numpy as np


@DETECTORS.register_module()
class SingleStageBoxInsDetector(BaseDetector):

    def __init__(self,
                 backbone,
                 neck=None,
                 bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):

        if pretrained:
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            backbone.pretrained = pretrained
        super(SingleStageBoxInsDetector, self).__init__(init_cfg=init_cfg)
        self.backbone = build_backbone(backbone)
        if neck is not None:
            self.neck = build_neck(neck)

        if bbox_head is not None:
            self.bbox_head = build_head(bbox_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg


    def extract_feat(self, img):
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward_dummy(self, img):
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        return outs

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None):

        x = self.extract_feat(img)
        outs = self.bbox_head(x)

        loss_inputs = outs + (gt_bboxes, gt_labels, gt_masks, img, img_metas, self.train_cfg)
        losses = self.bbox_head.loss(*loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        return losses

    def simple_test(self, img, img_meta, rescale=False):
        x = self.extract_feat(img)
        outs = self.bbox_head(x, eval=True)

        seg_inputs = outs + (img_meta, self.test_cfg, rescale)
        results_list = self.bbox_head.get_seg(*seg_inputs)
        format_results_list = []
        for results in results_list:
            format_results_list.append(self.format_results(results))
        return format_results_list
    
    def format_results(self, results):
        bbox_results = [[] for _ in range(self.bbox_head.num_classes)]
        mask_results = [[] for _ in range(self.bbox_head.num_classes)]
        score_results = [[] for _ in range(self.bbox_head.num_classes)]

        for cate_label, cate_score, seg_mask in zip(results.labels, results.scores, results.masks):
            if seg_mask.sum() > 0:
                mask_results[cate_label].append(seg_mask.cpu())
                score_results[cate_label].append(cate_score.cpu())
                ys, xs = torch.where(seg_mask)
                min_x, min_y, max_x, max_y = xs.min().cpu().data.numpy(), ys.min().cpu().data.numpy(), xs.max().cpu().data.numpy(), ys.max().cpu().data.numpy()
                bbox_results[cate_label].append([min_x, min_y, max_x+1, max_y+1, cate_score.cpu().data.numpy()])

        bbox_results = [np.array(bbox_result) if len(bbox_result) > 0 else np.zeros((0, 5)) for bbox_result in bbox_results]

        return bbox_results, (mask_results, score_results)

    def aug_test(self, imgs, img_metas, rescale=False):
        raise NotImplementedError


