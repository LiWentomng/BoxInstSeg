## BoxInstSeg
## Introduction
BoxInstSeg is an open-source toolbox that aims to provide state-of-the-art box-supervised instance segmentation algorithms. 
It is built on top of [MMdetection](https://github.com/open-mmlab/mmdetection).
The main branch works with Pytorch 1.6+ or higher (we recommend Pytorch 1.9)

Welcome the contribution.

### Major features
- **MMdetection feature inheritance**

  This toolbox doesn't change the structure and codes of original MMdetection and the additive codes are under MMdetection logic. Therefore, it inherits all features from MMdetection.

- **Support of instance segmentation with only box annotations**

   We implement multiple  box-supervised instance segmentation methods in this toolbox,(*e.g.* BoxInst, DiscoBox).

## Model Zoo
<summary> Supported methods </summary>

- [x] [BoxInst (CVPR2021)]()
- [x] [DiscoBox (ICCV2021)]()
- [x] [BoxLevelset (ECCV2022)]()
- [ ] [Box2Mask (ArXiv2022)]() (To be appear)

### COCO
|     method      | Backbone | Models    | config  | AP  | AP_50 | AP_75 | 
|:---------------:|----------|-----------|:-------:|:---:|:-----:|:-----:|
|   [BoxInst]()   | R-50     | [model]() | config  |     |       |       |
|   [BoxInst]()   | R-101    | [model]() | config  |     |       |       |
|  [DiscoBox]()   | R-50     | [model]() | config  |     |       |       |
|  [DiscoxBox]()  | R-101    | [model]() | config  |     |       |       | 
| [BoxLevelset]() | R-50     | [model]() | config  |     |       |       | 
| [BoxLevelset]() | R-101    | [model]() | config  |     |       |       | 
| [Box2Mask-C]()  | R-101    | [model]() | config  |     |       |       |   
| [Box2Mask-T]()  | R-50     | [model]() | config  |     |       |       | 
| [Box2Mask-T]()  | R-101    | [model]() | config  |     |       |       |    


### Pascal VOC

|     method      | Backbone | Models    | config  | AP  | AP_50 | AP_75 | 
|:---------------:|----------|-----------|:-------:|:---:|:-----:|:-----:|
|   [BoxInst]()   | R-50     | [model]() | config  |     |       |       |
|   [BoxInst]()   | R-101    | [model]() | config  |     |       |       |
|  [DiscoBox]()   | R-50     | [model]() | config  |     |       |       |
|  [DiscoxBox]()  | R-101    | [model]() | config  |     |       |       | 
| [BoxLevelSet]() | R-50     | [model]() | config  |     |       |       | 
| [BoxLevelSet]() | R-101    | [model]() | config  |     |       |       | 
| [Box2Mask-C]()  | R-101    | [model]() | config  |     |       |       |
| [Box2Mask-T]()  | R-50     | [model]() | config  |     |       |       | 
| [Box2Mask-T]()  | R-101    | [model]() | config  |     |       |       |    


## Installation and Getting Started
Please refer to [Installation]() and [Getting Started]() for the details of installation and the basic usage.
This is built on the MMdetection V2.25.0. We recommend the user to see the detailed instroductions of MMdetection.


## License

This project is released under the [Apache 2.0 license](LICENSE).


## Acknowledgement

This project is built based on [MMdetection](https://github.com/open-mmlab/mmdetection) and part of module is borrowed from the original rep of [Adelaidet](https://github.com/aim-uofa/AdelaiDet) and [DiscoBox](https://github.com/NVlabs/DiscoBox).

## More
This [repo](https://github.com/LiWentomng/Box-supervised-instance-segmentation) updates the **survey** of _box-supervised instance segmentation_, we highly welcome the user to develop more algorithms in this toolbox.

## Citation
If you find the projects useful in your research, please consider cite:







