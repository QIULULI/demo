# # 基于原Cityscapes配置
# _base_ = ['../cityscapes/diffusion_detector_cityscapes.py']

# # 1) 单类
# classes = ('drone',)

# dataset_type = 'CocoDataset'
# data_root = 'data/'
# backend_args = None

# # 2) 训练与验证集（用刚转好的 COCO）
# train_dataloader = dict(
#     batch_size=8,
#     num_workers=8,
#     persistent_workers=True,
#     sampler=dict(type='DefaultSampler', shuffle=True),
#     batch_sampler=dict(type='AspectRatioBatchSampler'),
#     dataset=dict(
#         type=dataset_type,
#         data_root=data_root,
#         metainfo=dict(classes=classes),
#         ann_file='drone_sim_coco/train.json',
#         data_prefix=dict(img='/userhome/liqiulu/data/drone_rgb_clear_day'),  # 若没做软链，这里改成绝对根：例如 '/userhome/liqiulu/.../carla_data/'
#         filter_cfg=dict(filter_empty_gt=True),
#         pipeline={{_base_.train_pipeline}}  # 若 train_pipeline 里有 AlbuDomainAdaption，建议去掉
#     )
# )

# val_dataloader = dict(
#     batch_size=1,
#     num_workers=8,
#     persistent_workers=True,
#     drop_last=False,
#     sampler=dict(type='DefaultSampler', shuffle=False),
#     dataset=dict(
#         type=dataset_type,
#         data_root=data_root,
#         metainfo=dict(classes=classes),
#         ann_file='drone_sim_coco/val.json',
#         data_prefix=dict(img='/userhome/liqiulu/data/drone_rgb_clear_day'),
#         test_mode=True,
#         filter_cfg=dict(filter_empty_gt=True),
#         pipeline={{_base_.test_pipeline}}
#     )
# )
# test_dataloader = val_dataloader

# val_evaluator = dict(
#     type='CocoMetric',
#     ann_file=data_root + 'drone_sim_coco/val.json',
#     metric='bbox',
#     format_only=False
# )
# test_evaluator = val_evaluator

# # 3) 调整模型类别数（一定要改）
# model = dict(
#     roi_head=dict(
#         bbox_head=dict(
#             num_classes=1  # 单类drone
#         )
#     ),
#     # 让扩散分支也知道类别列表（你原配置里 backbone.diff_config 有 classes）
#     backbone=dict(
#         diff_config=dict(
#             classes=classes
#         )
#     )
# )

# # 4) 可选：把训练pipeline里“域自适应”那步去掉（Cityscapes专用），更干净
# # 如果你 _base_ 的 train_pipeline 有 'AlbuDomainAdaption'，可以重写：
# train_pipeline = [
#     dict(type='LoadImageFromFile', backend_args=backend_args),
#     dict(type='LoadAnnotations', with_bbox=True),
#     dict(type='Resize', scale=(1333, 800), keep_ratio=True),
#     dict(type='RandomCrop', crop_type='absolute', crop_size=(512, 512),
#          recompute_bbox=True, allow_negative_crop=True),
#     dict(type='FilterAnnotations', min_gt_bbox_wh=(1e-2, 1e-2)),
#     dict(type='RandomFlip', prob=0.5),
#     dict(type='RandAugment', aug_space=[[dict(type='ColorTransform')]], aug_num=1),
#     dict(type='RandomErasing', n_patches=(1, 3), ratio=(0, 0.2)),
#     dict(type='PackDetInputs',
#         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape','scale_factor',
#                    'flip', 'flip_direction', 'homography_matrix')),
# ]

##——————————————————————————————————————————————————————————————————————————————————————————————————————————————————

# P0 fixed config: single-class 'drone' + small-object anchors + higher input resolution
# 基于原 Cityscapes 扩散检测器配置
_base_ = ['../cityscapes/diffusion_detector_cityscapes.py']

# ---------- 任务类别 ----------
classes = ('drone',)

# ---------- 数据 ----------
dataset_type = 'CocoDataset'
data_root = 'data/'
backend_args = None

# 训练集 / 验证集（确保你的 COCO json 与 file_name 前缀匹配 data_prefix.img）
train_dataloader = dict(
    batch_size=8,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=dict(classes=classes),
        ann_file='drone_sim_coco/train.json',
        data_prefix=dict(img='/userhome/liqiulu/data/drone_rgb_clear_day'),  # 若没做软链，这里改成绝对根：例如 '/userhome/liqiulu/.../carla_data/'
        filter_cfg=dict(filter_empty_gt=True),
        pipeline={{_base_.train_pipeline}}
    )
)

val_dataloader = dict(
    batch_size=1,
    num_workers=8,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=dict(classes=classes),
        ann_file='drone_sim_coco/val.json',
        data_prefix=dict(img='/userhome/liqiulu/data/drone_rgb_clear_day'),
        test_mode=True,
        filter_cfg=dict(filter_empty_gt=True),
        pipeline={{_base_.test_pipeline}}
    )
)
test_dataloader = val_dataloader

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'drone_sim_coco/val.json',
    metric='bbox',
    format_only=False
)
test_evaluator = val_evaluator

# ---------- 模型覆写（仅 P0 必要项） ----------
model = dict(
    # 1) 单类：ROI Head 调整
    roi_head=dict(
        bbox_head=dict(num_classes=1)
    ),
    # 2) 扩散分支知晓类别（用于 mask+prompt）
    backbone=dict(
        diff_config=dict(classes=classes)
    ),
    # 3) 小目标友好 RPN 锚框（P0 关键）
    rpn_head=dict(
        anchor_generator=dict(
            type='AnchorGenerator',
            # 更小更密的尺度 + 增加“瘦高”比例
            scales=[2, 4, 8],
            ratios=[0.33, 0.5, 1.0, 2.0],
            # strides 与基类一致（通常为 [4,8,16,32,64]）
        )
    )
)

# ---------- 提升小目标可见性：提高输入分辨率 ----------
# 仅重写 Resize 到更高分辨率（其余步骤仍沿用基类的 train/test_pipeline）
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),   
    dict(type='Resize', scale=(1600, 960), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'flip', 'flip_direction',
                   'homography_matrix'))
]

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=(1600, 960), keep_ratio=True),
    # 如果没有 gt，请删掉下一行；这里保留以便离线评测
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor'))
]
