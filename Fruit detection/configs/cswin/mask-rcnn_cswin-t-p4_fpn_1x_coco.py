_base_ = [
    '../_base_/models/mask-rcnn_r50_fpn.py',
    '../_base_/datasets/coco_instance.py',
    '../_base_/schedules/schedule_1x.py', 
    '../_base_/default_runtime.py'
]
pretrained = 'D:/build/cswin_tiny_224.pth'
model = dict(
    type='MaskRCNN',
    backbone=dict(
        _delete_=True,
        type='CSWin',
        embed_dim=96,
        depth=[2, 4, 6, 2],
        num_heads=[2, 4, 8, 16],
        split_size=[1,2,7,7],
        drop_path_rate=0.6,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        use_chk=False,
        #init_cfg=None 
        init_cfg = dict(type='Pretrained', checkpoint=pretrained)
    ),
    roi_head=dict(
        bbox_head=dict(num_classes=22), mask_head=dict(num_classes=22)),
    neck=dict(in_channels=[96, 192, 384, 768]))


max_epochs = 12
train_cfg = dict(max_epochs=max_epochs)

# learning rate
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0,
        end=1000),
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[8, 11],
        gamma=0.1)
]

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }),
    optimizer=dict(
        _delete_=True,
        type='AdamW',
        lr=0.0001,
        betas=(0.9, 0.999),
        weight_decay=0.05))
