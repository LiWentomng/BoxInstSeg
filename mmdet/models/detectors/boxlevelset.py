from .single_stage_boxseg import SingleStageBoxInsDetector
from ..builder import DETECTORS

@DETECTORS.register_module()
class BoxLevelSet(SingleStageBoxInsDetector):

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
