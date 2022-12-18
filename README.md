
## Introduction
**BoxInstSeg** is a toolbox that aims to provide state-of-the-art box-supervised instance segmentation algorithms. 
It is built on top of [mmdetection](https://github.com/open-mmlab/mmdetection).
The main branch works with Pytorch 1.6+ or higher (we recommend Pytorch **1.9.0**)


https://user-images.githubusercontent.com/32033843/207869884-5254b2ba-5e3d-44fb-bbd6-b26d07c9b404.mp4


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
- [ ] [Box2Mask](https://arxiv.org/pdf/2212.01579.pdf) (to be released)

**_More  models and datasets are under preparation, which are coming as soon as possible._**

### COCO (val)
|     method      | Backbone | GPUs| Models                                                                                        | sched.  |config   | AP (this rep) | AP(original rep/paper) |
|:---------------:|----------|-----------|-----------------------------------------------------------------------------------------------|:-------:|:-------:|:-------------:|:----------------------:|
|   BoxInst       | R-50     |  8  | [model](https://drive.google.com/file/d/1dVOmUGsvnORcUGpFRXTxknepvsUYiV34/view?usp=sharing)   |   1x    |[config](https://github.com/LiWentomng/BoxInstSeg/blob/main/configs/boxinst/boxinst_r50_fpn_1x_coco.py)   |     30.7      |          30.7          |
|   BoxInst       | R-50     |  8  | [model](https://drive.google.com/drive/folders/1dbBM6EMA_8lFnrMzHCAV4X7ecxYzjl4w?usp=sharing) |   3x    |[config](https://github.com/LiWentomng/BoxInstSeg/blob/main/configs/boxinst/boxinst_r50_fpn_3x_coco.py)   |     32.1      |          31.8          |
|   BoxInst       | R-101     |  8  | [model](https://drive.google.com/drive/folders/1RCFqb15bVlNaI7AxKerP6hRmJ8AnexKN?usp=sharing) |   1x    |[config](https://github.com/LiWentomng/BoxInstSeg/blob/main/configs/boxinst/boxinst_r101_fpn_1x_coco.py)   |     32.0      |          32.2          |
|   BoxInst       | R-101     |  8  | [model](https://drive.google.com/file/d/1tlXLL5Ba9_o5V7zn18KZPCIyEJdpZFV1/view?usp=sharing)   |   3x    |[config](https://github.com/LiWentomng/BoxInstSeg/blob/main/configs/boxinst/boxinst_r101_fpn_3x_coco.py)   |     33.1      |          33.0          |
|   DiscoBox      | R-50     |  8 | [model](https://drive.google.com/file/d/1ifhmjXbFCBsn6wLBeOM6BWOm11LDmecr/view?usp=sharing)   |   3x    |[config](https://github.com/LiWentomng/BoxInstSeg/blob/main/configs/discobox/discobox_solov2_coco_r50_fpn_3x.py)   |     32.2      |      31.4(wo ms)       |
|   DiscoBox      | R-101    |  8  | [model](https://drive.google.com/file/d/12yNKThkQK3yV8B5YHkkgyMOa0e5-uUpJ/view?usp=sharing)     |   3x    |[config](https://github.com/LiWentomng/BoxInstSeg/blob/main/configs/discobox/discobox_solov2_coco_r101_fpn_3x.py)   |     33.4      |           -            |
| Box2Mask-T      | R-50     |  8  | model                                                                                         |   50e   | config |               |                        |
| Box2Mask-T      | R-101     |  8  | model                                                                                         |   50e   | config |               |                        |

The above models are trained with `ms` to make a performance comparison.

### Pascal VOC
|     method      | Backbone | Models    | sched.  |config   |
|:---------------:|----------|-----------|:-------:|:-------:|
|   BoxInst       | R-50     | model     |   3x    |config   | 


## Installation and Getting Started
This is built on the MMdetection (V2.25.0). Please refer to [Installation](https://github.com/LiWentomng/BoxInstSeg/blob/main/docs/install.md) and [Getting Started](https://github.com/LiWentomng/BoxInstSeg/blob/main/docs/get_started.md) for the details of installation and basic usage. We also recommend the user to refer the office [introduction](https://github.com/open-mmlab/mmdetection/blob/master/docs/en/get_started.md/#Installation) of MMdetection.


## License

This project is released under the [Apache 2.0 license](LICENSE).


## Acknowledgement

This project is built based on [MMdetection](https://github.com/open-mmlab/mmdetection) and part of module is borrowed from the original rep of [Adelaidet](https://github.com/aim-uofa/AdelaiDet) and [DiscoBox](https://github.com/NVlabs/DiscoBox).

## More
- This [repo](https://github.com/LiWentomng/Box-supervised-instance-segmentation) will update the **survey** of _box-supervised instance segmentation_, we highly welcome the user to develop more algorithms in this toolbox.

- **_If this rep is helpful for your work, please give me a star._**

