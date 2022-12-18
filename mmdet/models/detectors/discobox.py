# ---------------------------------------------------------------
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for DiscoBox. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

from .single_stage_ts import SingleStageWSInsDetector, SingleStageWSInsTSDetector
from ..builder import DETECTORS
import mmcv
import numpy as np
import torch


@DETECTORS.register_module()
class DiscoBoxSOLOv2(SingleStageWSInsTSDetector):
    """Implementation of `DiscoBox <https://arxiv.org/abs/2105.06464v2>`_"""
    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 mask_feat_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(DiscoBoxSOLOv2, self).__init__(backbone, neck, bbox_head, mask_feat_head, train_cfg,
                                   test_cfg, pretrained, init_cfg)
