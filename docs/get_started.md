## Getting Started


This page provides basic usage based on MMdetection (V2.25.0). For installation instructions, please see [install.md](./install.md).


# Train a model
MMDetection implements distributed training and non-distributed training,
which uses `MMDistributedDataParallel` and `MMDataParallel` respectively.

All outputs (log files and checkpoints) will be saved to the working directory,
which is specified by `work_dir` in the config file.

1. Train  with a single GPU 

```shell
CUDA_VISIBLE_DEVICES=0 python tools/train.py ${CONFIG_FILE}  #

Example:
CUDA_VISIBLE_DEVICES=0 python toos/train.py configs/boxinst/boxinst_r50_fpn_1x_coco.py 
```

2. Train with multiple GPUs

```shell
./tools/dist_train.sh ${CONFIG_FILE} ${GPU_NUM} [optional arguments]

Example:
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ./tools/dist_train.sh configs/coco/boxinst_r50_fpn_1x_coco.py 8
```

# Inference with pretrained models
We provide the testing scripts to evaluate the trained models.

Examples for boxinst:
Assume that you have already downloaded the checkpoints to `work_dirs/boxinst_r50_1x_coco/`.

1. Test with single GPU and get mask AP values.

```shell
CUDA_VISIBLE_DEVICES=0 python tools/test.py configs/coco/boxinst_r50_fpn_1x_coco.py \
    work_dirs/boxinst_r50_1x_coco/xxx.pth  --eval segm

```
2. Test with 8 GPUs and get mask AP values on `val` dataset.
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ./tools/dist_test.sh configs/coco/boxinst_r50_fpn_1x_coco.py \
    work_dirs/boxinst_r50_1x_coco/xxx.pth 8 --eval segm 
```

3. Test with 8 GPUs and get mask AP values on `test-dev` dataset.

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ./tools/dist_test.sh configs/coco/boxinst_r50_fpn_1x_coco.py \
    work_dirs/coco_r50_3x/xxx.pth 8 --format-only --eval-options "jsonfile_prefix=work_dirs/r50_coco_dev" 
```
Generate the json results, and submit to the [COCO challenge server](https://codalab.lisn.upsaclay.fr/competitions/7383#participate-submit_results) for `test-dev` performance evaluation.


# Inference for visual results


1. Test for COCO

   ```shell
    CUDA_VISIBLE_DEVICES=0 python tools/test.py configs/coco/boxinst_r50_fpn_1x_coco.py \
    work_dirs/boxinst_r50_1x_coco/xxx.pth  --show-dir work_dirs/vis_coco_r50/
    ```

Note: The visual results is in `show-dir`. 



# Data preparation

1. **Pascal VOC**(Augmented) is the extension of the training set of VOC 2012 with SBD following [BBTP](https://github.com/chengchunhsu/WSIS_BBTP) 
   The link of whole dataset with coco json format is [here](https://drive.google.com/file/d/16Mz13NSZBbhwPuRxiwi7ZA2Qvt9DaKtN/view?usp=sharing)(GoogleDrive)

More dataset will be updated.

