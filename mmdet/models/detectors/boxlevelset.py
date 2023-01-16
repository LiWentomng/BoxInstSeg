from .single_stage_boxseg import SingleStageBoxInsDetector
from ..builder import DETECTORS

@DETECTORS.register_module()
class BoxLevelSet(SingleStageBoxInsDetector):
    r"""Implementation of `Box-supervised Instance Segmentation
    with Level Set Evolution <https://arxiv.org/abs/2207.09055.pdf>`_."""

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(BoxLevelSet, self).__init__(backbone, neck, bbox_head, train_cfg,
                                   test_cfg, pretrained, init_cfg)
