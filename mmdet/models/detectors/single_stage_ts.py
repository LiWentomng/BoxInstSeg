import torch.nn as nn
import torch
import numpy as np
import warnings
from .. import builder
from ..builder import DETECTORS
from .base import BaseDetector
from collections import OrderedDict

from mmcv.runner import auto_fp16


@DETECTORS.register_module()
class SingleStageWSInsDetector(BaseDetector):

    def __init__(self,
                 backbone,
                 neck=None,
                 bbox_head=None,
                 mask_feat_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        if pretrained:
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            backbone.pretrained = pretrained
        super(SingleStageWSInsDetector, self).__init__(init_cfg=init_cfg)
        self.backbone = builder.build_backbone(backbone)
        if neck is not None:
            self.neck = builder.build_neck(neck)
        if mask_feat_head is not None:
            self.mask_feat_head = builder.build_head(mask_feat_head)

        self.bbox_head = builder.build_head(bbox_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.cnt = 0
        self.avg_loss_ins = 2

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
        self.cnt += 1
        x = self.extract_feat(img)
        outs = self.bbox_head(x)

        mask_feat_pred = self.mask_feat_head(
            x[self.mask_feat_head.
              start_level:self.mask_feat_head.end_level + 1])
        loss_inputs = outs + (mask_feat_pred, gt_bboxes, gt_labels, gt_masks, img_metas, self.train_cfg)
        losses = self.bbox_head.loss(
            *loss_inputs, img=img, gt_bboxes_ignore=gt_bboxes_ignore,
            use_loss_ts=self.avg_loss_ins<0.4)
        self.avg_loss_ins = self.avg_loss_ins * 0.99 + float(losses['loss_ins']) * 0.01
        return losses

    def simple_test(self, img, img_meta, rescale=False):
        x = self.extract_feat(img)
        outs = self.bbox_head(x, eval=True)

        mask_feat_pred = self.mask_feat_head(
            x[self.mask_feat_head.
              start_level:self.mask_feat_head.end_level + 1])
        seg_inputs = outs + (mask_feat_pred, img_meta, self.test_cfg, rescale)
        results_list = self.bbox_head.get_seg(*seg_inputs, img=img)
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



class SingleStageWSInsTeacherDetectorWrapper:

    def __init__(self, teacher):
        self.teacher = teacher


class SingleStageWSInsTeacherDetector(SingleStageWSInsDetector):

    @torch.no_grad()
    def momentum_update(self, cur_model, teacher_momentum=None):
        """
        Momentum update of the key encoder
        """
        if teacher_momentum is None:
            teacher_momentum = self.teacher_momentum
        cur_state_dict = cur_model.state_dict()
        self_state_dict = self.state_dict()
        weight_keys = list(cur_state_dict.keys())
        fed_state_dict = OrderedDict()
        for key in weight_keys:
            fed_state_dict[key] = cur_state_dict[key]* (1. - teacher_momentum)+self_state_dict[key]* teacher_momentum
        self.load_state_dict(fed_state_dict)

@DETECTORS.register_module()
class SingleStageWSInsTSDetector(SingleStageWSInsDetector):

    def __init__(self,
                 backbone,
                 neck=None,
                 bbox_head=None,
                 mask_feat_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        if pretrained:
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            backbone.pretrained = pretrained
        super(SingleStageWSInsTSDetector, self).__init__(backbone, neck, bbox_head, mask_feat_head,
                                                         train_cfg, test_cfg, pretrained, init_cfg)

        self.use_ind_teacher = bbox_head['loss_ts']['use_ind_teacher']
        if self.use_ind_teacher:
            self.tsw = SingleStageWSInsTeacherDetectorWrapper(
                            SingleStageWSInsTeacherDetector(backbone, neck, bbox_head, mask_feat_head,
                                                            train_cfg, test_cfg, pretrained, init_cfg))
            self.tsw.teacher.eval()
            self.tsw.teacher.teacher_momentum = bbox_head['loss_ts']['momentum']
        self.use_corr = bbox_head.get('loss_corr', None) is not None
        self.cnt = 0
        self.avg_loss_ins = 2
        self.turn_on_teacher = False

    def to(self, *args, **kwargs):
        if self.use_ind_teacher:
            self.tsw.teacher = self.tsw.teacher.to(*args, **kwargs)
        return super(SingleStageWSInsTSDetector, self).to(*args, **kwargs)

    def cuda(self, *args, **kwargs):
        if self.use_ind_teacher:
            self.tsw.teacher = self.tsw.teacher.cuda(*args, **kwargs)
        return super(SingleStageWSInsTSDetector, self).cuda(*args, **kwargs)

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

        if self.turn_on_teacher and self.use_ind_teacher:
            self.tsw.teacher.momentum_update(self)

        # Not sure why the previous flip doesn't work
        student_x = self.extract_feat(img)
        student_outs = self.bbox_head(student_x)
        _, s_kernel_preds = student_outs

        with torch.no_grad():
            if self.turn_on_teacher and self.use_ind_teacher:
                teacher_x = self.tsw.teacher.extract_feat(img)
                teacher_outs = self.tsw.teacher.bbox_head(teacher_x)
                _, t_kernel_preds = teacher_outs

        s_mask_feat_pred = self.mask_feat_head(
            student_x[self.mask_feat_head.
                start_level:self.mask_feat_head.end_level + 1]
        )
        if self.turn_on_teacher and self.use_ind_teacher:
            with torch.no_grad():
                t_mask_feat_pred = self.tsw.teacher.mask_feat_head(
                    teacher_x[self.tsw.teacher.mask_feat_head.
                        start_level:self.tsw.teacher.mask_feat_head.end_level + 1]
                )
        else:
            teacher_x = student_x

        if self.turn_on_teacher and self.use_ind_teacher:
            loss_inputs = student_outs + (t_kernel_preds, s_mask_feat_pred, t_mask_feat_pred, gt_bboxes, gt_labels, gt_masks, img_metas, self.train_cfg)
        else:
            loss_inputs = student_outs + (None, s_mask_feat_pred, None, gt_bboxes, gt_labels, gt_masks, img_metas, self.train_cfg)

        losses = self.bbox_head.loss(
            *loss_inputs,
            img=img,
            gt_bboxes_ignore=gt_bboxes_ignore,
            use_loss_ts=self.avg_loss_ins < 0.3,
            use_ind_teacher=self.use_ind_teacher and self.turn_on_teacher,
            use_corr=self.avg_loss_ins < 0.2 and self.use_corr,
            s_feat=student_x[self.mask_feat_head.start_level:self.mask_feat_head.end_level + 1],
            t_feat=teacher_x[self.mask_feat_head.start_level:self.mask_feat_head.end_level + 1]
        )

        self.avg_loss_ins = self.avg_loss_ins * 0.9 + 0.1 * float(losses['loss_ins'])
        if self.use_ind_teacher and not self.turn_on_teacher and self.cnt > 13000:
            self.tsw.teacher.momentum_update(self, 0)
            self.turn_on_teacher = True
            print("Turn on teacher.")
        else:
            self.cnt += 1
        return losses

    def aug_test(self, imgs, img_metas, rescale=False):
        raise NotImplementedError


