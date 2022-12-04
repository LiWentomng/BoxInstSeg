
## Introduction
**BoxInstSeg** is a toolbox that aims to provide state-of-the-art box-supervised instance segmentation algorithms. 
It is built on top of [mmdetection](https://github.com/open-mmlab/mmdetection).
The main branch works with Pytorch 1.6+ or higher (we recommend Pytorch **1.9.0**)


#### Major features

- **Support of instance segmentation with only box annotations**

   We implement multiple box-supervised instance segmentation methods in this toolbox,(*e.g.* BoxInst, DiscoBox). This toolbox can achieve the similar performance as the original paper. 

- **MMdetection feature inheritance**

  This toolbox doesn't change the structure and logic of mmdetection. It inherits all features from MMdetection.

## Model Zoo
<summary> Supported methods </summary>

- [x] [BoxInst (CVPR2021)](https://arxiv.org/abs/2012.02310)
- [x] [DiscoBox (ICCV2021)](https://arxiv.org/abs/2105.06464v2)
- [x] [BoxLevelset (ECCV2022)](https://arxiv.org/abs/2207.09055)
- [ ] Box2Mask (to be appear)

### COCO (val)
|     method      | Backbone | Models    | sched.  |config   | AP      | original paper |
|:---------------:|----------|-----------|:-------:|:-------:|:-------:|:-------:|
|   BoxInst       | R-50     | [model](https://drive.google.com/file/d/1dVOmUGsvnORcUGpFRXTxknepvsUYiV34/view?usp=sharing)     |   1x    |[config](https://github.com/LiWentomng/BoxInstSeg/blob/main/configs/boxinst/boxinst_r50_fpn_1x_coco.py)   |  30.7   | 30.7 |
|   BoxInst       | R-50     | model     |   3x    |[config](https://github.com/LiWentomng/BoxInstSeg/blob/main/configs/boxinst/boxinst_r50_fpn_3x_coco.py)   |         |      |
|   DiscoBox      | R-50     | model     |   3x    |[config](https://github.com/LiWentomng/BoxInstSeg/blob/main/configs/discobox/discobox_solov2_coco_r50_fpn_3x.py)   |  31.7   |   |
|   DiscoBox      | R-101    | model     |   3x    |[config](https://github.com/LiWentomng/BoxInstSeg/blob/main/configs/discobox/discobox_solov2_coco_r101_fpn_3x.py)   |  33.1   |   |

### Pascal VOC
|     method      | Backbone | Models    | sched.  |config   | AP      | 
|:---------------:|----------|-----------|:-------:|:-------:|:-------:|
|   BoxInst       | R-50     | model     |   3x    |config   |         |

**_Multiple pretrained models based on the Pascal VOC and COCO and more datasets are coming as soon as possible._**


## Installation and Getting Started
This is built on the MMdetection (recent V2.25.0). Please refer to [Installation](https://github.com/open-mmlab/mmdetection/blob/master/docs/en/get_started.md/#Installation) and [Getting Started](https://github.com/open-mmlab/mmdetection/blob/master/docs/en/get_started.md) for the details of installation and the basic usage.


## License

This project is released under the [Apache 2.0 license](LICENSE).


## Acknowledgement

This project is built based on [MMdetection](https://github.com/open-mmlab/mmdetection) and part of module is borrowed from the original rep of [Adelaidet](https://github.com/aim-uofa/AdelaiDet) and [DiscoBox](https://github.com/NVlabs/DiscoBox).

## More
- This [repo](https://github.com/LiWentomng/Box-supervised-instance-segmentation) will update the **survey** of _box-supervised instance segmentation_, we highly welcome the user to develop more algorithms in this toolbox.

- **_If this rep is helpful for your work, please give me a star._**

