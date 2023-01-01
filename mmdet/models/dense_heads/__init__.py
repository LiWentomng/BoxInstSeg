# Copyright (c) OpenMMLab. All rights reserved.
from .anchor_free_head import AnchorFreeHead
from .anchor_head import AnchorHead
from .condinst_head import CondInstBoxHead, CondInstSegmHead, CondInstMaskBranch, CondInstMaskHead
from .box_solov2_head import BoxSOLOv2Head
from .discobox_head import DiscoBoxSOLOv2Head, DiscoBoxMaskFeatHead
from .box2mask_head import Box2MaskHead

__all__ = [
    'AnchorFreeHead', 'AnchorHead', 'CondInstBoxHead', 'CondInstMaskBranch',
    'CondInstMaskHead', 'CondInstSegmHead', 'BoxSOLOv2Head', 'DiscoBoxSOLOv2Head',
    'DiscoBoxMaskFeatHead', 'Box2MaskHead'
]
