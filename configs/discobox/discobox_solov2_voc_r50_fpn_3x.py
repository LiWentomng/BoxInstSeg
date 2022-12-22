
fp16 = dict(loss_scale=512.)
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375], to_rgb=True)
model = dict(
    type='DiscoBoxSOLOv2',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3), # C2, C3, C4, C5
        frozen_stages=1,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=0,
        num_outs=5),
    bbox_head=dict(
        type='DiscoBoxSOLOv2Head',
        num_classes=20,
        in_channels=256,
        stacked_convs=4,
        seg_feat_channels=512,
        strides=[8, 8, 16, 32, 32],
        scale_ranges=((1, 96), (48, 192), (96, 384), (192, 768), (384, 2048)),
        sigma=0.2,
        num_grids=[40, 36, 24, 16, 12],
        ins_out_channels=256,
        loss_ins=dict(
            type='DiceLoss',
            use_sigmoid=True,
            loss_weight=1.0),
        loss_ts=dict(
            type='DiceLoss',
            momentum=0.999,
            use_ind_teacher=True,
            loss_weight=1.0,
            kernel=3,
            max_iter=10,
            alpha0=2.0,
            theta0=0.5,
            theta1=30.0,
            theta2=20.0,
            base=0.10,
            crf_height=28,
            crf_width=28,
        ),
        loss_cate=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_corr=dict(
            type='InfoNCE',
            loss_weight=1,
            corr_exp=1.0,
            corr_eps=0.05,
            gaussian_filter_size=3,
            low_score=0.3,
            corr_num_iter=10,
            corr_num_smooth_iter=1,
            save_corr_img=False,
            dist_kernel=9,
            obj_bank=dict(
                img_norm_cfg=img_norm_cfg,
                len_object_queues=100,
                fg_iou_thresh=0.7,
                bg_iou_thresh=0.7,
                ratio_range=[0.9, 1.2],
                appear_thresh=0.7,
                min_retrieval_objs=2,
                max_retrieval_objs=5,
                feat_height=7,
                feat_width=7,
                mask_height=28,
                mask_width=28,
                img_height=200,
                img_width=200,
                min_size=32,
                num_gpu_bank=20,
            )
        )
    ),
    mask_feat_head=dict(
            type='DiscoBoxMaskFeatHead',
            in_channels=256,
            out_channels=128,
            start_level=0,
            end_level=3,
            num_classes=256,
            norm_cfg=dict(type='GN', num_groups=32, requires_grad=True)),
    train_cfg=dict(),
    test_cfg=dict(
        nms_pre=500,
        score_thr=0.1,
        mask_thr=0.5,
        filter_thr=0.05,
        kernel='gaussian',  # gaussian/linear
        sigma=2.0,
        max_per_img=100))

# dataset settings
dataset_type = 'PascalVOCDataset'
data_root = '/data/VOC/'
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='GenerateBoxMask'),
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

# optimizer
optimizer = dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=2000,
    warmup_ratio=0.01,
    step=[33, 35])
checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=36)
evaluation = dict(interval=1, metric=['bbox', 'segm'])
device_ids = range(4)
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/discobox_r50_fpn_3x_voc/'
load_from = None
resume_from = None
workflow = [('train', 1)]
