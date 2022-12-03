model = dict(
    type='BoxLevelSet',
    backbone=dict(
        type='ResNet',
        depth=101,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='https://download.pytorch.org/models/resnet101-b641f3a9.pth')),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=0,
        num_outs=5),
    bbox_head=dict(
        type='BoxSOLOv2Head',
        num_classes=20,
        in_channels=256,
        stacked_convs=4,
        seg_feat_channels=256,
        strides=[8, 8, 16, 32, 32],
        scale_ranges=((1, 96), (48, 192), (96, 384), (192, 768), (384, 2048)),
        sigma=0.2,
        num_grids=[40, 36, 24, 16, 12],
        cate_down_pos=0,
        loss_cate=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_boxpro=dict(
            type='BoxProjectionLoss',
            loss_weight=3.0),
        loss_levelset=dict(
            type='LevelsetLoss',
            loss_weight=1.0)),
    train_cfg=dict(),
    test_cfg=dict(
        nms_pre=500,
        score_thr=0.05,
        mask_thr=0.7,
        filter_thr=0.025,
        kernel='gaussian',  # gaussian/linear
        sigma=2.0,
        max_per_img=100))

# dataset settings
dataset_type = 'PascalVOCDataset'
data_root = '/data/VOC/'
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=False), # without GT mask
    dict(type='GenerateBoxMask'), #generate box mask
    dict(type='Resize',
         img_scale=[(1333, 800), (1333, 768), (1333, 736),
                    (1333, 704), (1333, 672), (1333, 640)],
         multiscale_mode='value',
         keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/voc_2012_train.json',
        img_prefix=data_root + 'train/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/voc_2012_val.json',
        img_prefix=data_root + 'val/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/voc_2012_val.json',
        img_prefix=data_root + 'val/',
        pipeline=test_pipeline))


optimizer = dict(
    type='AdamW',
    lr=0.00005,
    weight_decay=0.1,
    paramwise_cfg=dict(norm_decay_mult=0., bypass_duplicate=True))
optimizer_config = dict(grad_clip=dict(max_norm=1, norm_type=2))
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=2000,
    warmup_ratio=0.001,
    step=[27, 33])
checkpoint_config = dict(interval=2)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
    ])
# yapf:enable
# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=36)
evaluation = dict(interval=2, metric=['segm'])
device_ids = range(4)
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/boxlevelset_voc_r101_3x'
load_from = None
resume_from = None
workflow = [('train', 1)]
