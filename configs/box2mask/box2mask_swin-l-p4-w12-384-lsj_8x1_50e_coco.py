_base_ = ['./box2mask_swin-t-p4-w7-224_lsj_8x2_50e_coco.py']
pretrained = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window12_384_22k.pth'

depths = [2, 2, 18, 2]
model = dict(
    backbone=dict(
        pretrain_img_size=384,
        embed_dims=192,
        depths=depths,
        num_heads=[6, 12, 24, 48],
        window_size=12,
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
    panoptic_head=dict(num_queries=100, in_channels=[192, 384, 768, 1536]))


work_dir = './work_dirs/box2mask_swin_large_100query_50e'
