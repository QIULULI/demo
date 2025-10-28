# # P1+: single-class 'drone' + small-object anchors + higher input res
# #      + FilterAnnotations + LIGHT color augment  (real pipeline override)
# _base_ = ['../cityscapes/diffusion_detector_cityscapes.py']

# # ---------- 类别 ----------
# classes = ('drone',)

# # ---------- 数据 ----------
# dataset_type = 'CocoDataset'
# data_root = 'data/'
# backend_args = None

# # ---------- 轻量颜色增强空间（排除 Solarize/Posterize） ----------
# color_space_light = [
#     [dict(type='AutoContrast')],
#     [dict(type='Equalize')],
#     [dict(type='Color')],       # 饱和度
#     [dict(type='Contrast')],    # 对比度
#     [dict(type='Brightness')],  # 亮度
#     [dict(type='Sharpness')],   # 锐度
# ]

# # ---------- 训练/测试流水线（高分辨率；不裁剪/不擦除） ----------
# train_pipeline = [
#     dict(type='LoadImageFromFile', backend_args=backend_args),
#     dict(type='LoadAnnotations', with_bbox=True),

#     # 更高的可见性
#     dict(type='Resize', scale=(1600, 960), keep_ratio=True),

#     # 稳妥地过滤“极小框”，避免数值/抖动问题
#     dict(type='FilterAnnotations', min_gt_bbox_wh=(1e-2, 1e-2)),

#     dict(type='RandomFlip', prob=0.5),

#     # 轻量颜色增强：每次取一个，强度温和
#     dict(type='RandAugment', aug_space=color_space_light, aug_num=1),

#     dict(
#         type='PackDetInputs',
#         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
#                    'scale_factor', 'flip', 'flip_direction', 'homography_matrix')),
# ]

# test_pipeline = [
#     dict(type='LoadImageFromFile', backend_args=backend_args),
#     dict(type='Resize', scale=(1600, 960), keep_ratio=True),
#     # 如果 val/test 没有 GT，请把下一行注释掉
#     dict(type='LoadAnnotations', with_bbox=True),
#     dict(
#         type='PackDetInputs',
#         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor')),
# ]

# # ---------- DataLoaders（显式使用本文件的 pipeline） ----------
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
#         ann_file='drone_ann/single_clear_day_rgb/train.json', # need to change
#         data_prefix=dict(img='/userhome/liqiulu/data/drone_rgb_clear_day/00001'), # need to change
#         filter_cfg=dict(filter_empty_gt=True),
#         pipeline={{_base_.train_pipeline}},
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
#         ann_file='drone_ann/single_clear_day_rgb/val.json',
#         data_prefix=dict(img='/userhome/liqiulu/data/drone_rgb_clear_day/00001'), # need to change
#         test_mode=True,
#         filter_cfg=dict(filter_empty_gt=True),
#         pipeline={{_base_.test_pipeline}},
#     )
# )
# test_dataloader = val_dataloader

# # ---------- 评测器 ----------
# val_evaluator = dict(
#     type='CocoMetric',
#     ann_file=data_root + 'drone_ann/single_clear_day_rgb/val.json', # need to change
#     metric='bbox',
#     format_only=False
# )
# test_evaluator = val_evaluator

# # ---------- 模型（单类 + 小目标友好 RPN 锚框） ----------
# model = dict(
#     # 单类
#     roi_head=dict(bbox_head=dict(num_classes=1)),
#     backbone=dict(diff_config=dict(classes=classes)),

#     # 小目标锚框：更小尺度 + 加瘦高比例（其余沿用基类）
#     rpn_head=dict(
#         anchor_generator=dict(
#             type='AnchorGenerator',
#             scales=[2, 4, 8],
#             ratios=[0.33, 0.5, 1.0, 2.0],
#             # strides 走基类即可
#         )
#     )
# )

# P1+: single-class 'drone' + small-object anchors + higher input res
#      + FilterAnnotations + LIGHT color augment  (real pipeline override)
_base_ = ['../cityscapes/diffusion_detector_cityscapes.py']

# ---------- 类别 ----------
classes = ('drone',)

# ---------- 数据 ----------
dataset_type = 'CocoDataset'
data_root = 'data/'
backend_args = None

# ---------- 轻量颜色增强空间（排除 Solarize/Posterize） ----------
color_space_light = [
    [dict(type='AutoContrast')],
    [dict(type='Equalize')],
    [dict(type='Color')],       # 饱和度
    [dict(type='Contrast')],    # 对比度
    [dict(type='Brightness')],  # 亮度
    [dict(type='Sharpness')],   # 锐度
]

# ---------- 训练/测试流水线（当前：高分辨率；不裁剪/不擦除） ----------
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    # 更高的可见性
    dict(type='Resize', scale=(1600, 960), keep_ratio=True),
    dict(type='RandomCrop', crop_size=(640, 640), allow_negative_crop=True, recompute_bbox=True),
    # 稳妥地过滤“极小框”，避免数值/抖动问题
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1e-2, 1e-2)),
    dict(type='RandomFlip', prob=0.5),
    # 轻量颜色增强：每次取一个，强度温和
    dict(type='RandAugment', aug_space=color_space_light, aug_num=1),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'flip', 'flip_direction', 'homography_matrix')),
]

# test_pipeline = [
#     dict(type='LoadImageFromFile', backend_args=backend_args),
#     dict(type='Resize', scale=(1600, 960), keep_ratio=True),
#     # 如果 val/test 没有 GT，请把下一行注释掉
#     dict(type='LoadAnnotations', with_bbox=True),
#     dict(
#         type='PackDetInputs',
#         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor')),
# ]

# ============================ 可逐步启用的改动（默认全部注释） ============================

# --- STEP 1（评估期更稳）：轻量测试分辨率，减少评估显存峰值（默认注释）
# test_pipeline_light = [
#     dict(type='LoadImageFromFile', backend_args=backend_args),
#     dict(type='Resize', scale=(1333, 800), keep_ratio=True),  # 从 1600x960 降一档，更稳
#     # 若无 GT，注释下一行
#     dict(type='LoadAnnotations', with_bbox=True),
#     dict(type='PackDetInputs',
#          meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor')),
# ]
# 启用方法：把 val/test_dataloader.dataset.pipeline 的 {{_base_.test_pipeline}} 改成 test_pipeline_light

# --- STEP 2（训练更省显存但保留小目标可见性）：放大后裁剪成小贴片（默认注释）
# train_pipeline_choice = [
#     dict(type='LoadImageFromFile', backend_args=backend_args),
#     dict(type='LoadAnnotations', with_bbox=True),
#     dict(
#         type='RandomChoice',  # 两路策略随机选：A 稳定分辨率；B 放大+裁剪提高小目标尺度
#         transforms=[
#             [dict(type='Resize', scale=(1333, 800), keep_ratio=True)],            # A：COCO 常规分辨率
#             [dict(type='Resize', scale=(1600, 960), keep_ratio=True),             # B1：先放大
#              dict(type='RandomCrop', crop_size=(640, 640), allow_negative_crop=True, recompute_bbox=True)],  # B2：再裁剪，小贴片控显存
#         ],
#     ),
#     dict(type='FilterAnnotations', min_gt_bbox_wh=(1e-2, 1e-2)),
#     dict(type='RandomFlip', prob=0.5),
#     dict(type='RandAugment', aug_space=color_space_light, aug_num=1),
#     dict(type='PackDetInputs',
#          meta_keys=('img_id','img_path','ori_shape','img_shape','scale_factor','flip','flip_direction','homography_matrix')),
# ]
# 启用方法：把 train_dataloader.dataset.pipeline 改成 train_pipeline_choice（见下方 dataloader 注释）

# --- STEP 3（优化器/AMP/梯度累积）：不改等效 batch 的前提下降显存（默认注释）
# optim_wrapper = dict(
#     type='AmpOptimWrapper',         # 开启混合精度，通常省 30%~40% 显存
#     loss_scale='dynamic',
#     accumulative_counts=4,          # 结合 batch_size=2 → 等效总 batch=8
#     optimizer=dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001),
# )
# auto_scale_lr = dict(enable=True, base_batch_size=8)  # 等效总 batch=8 时自动缩放学习率

# --- STEP 4（进一步降峰值）：减少填充/提案数量（默认注释，影响极小）
# model = dict(
#     data_preprocessor=dict(pad_size_divisor=32),  # 基类常为 64 → 改 32，减少 padding 带来的无效计算
#     train_cfg=dict(rpn_proposal=dict(max_per_img=512), rcnn=dict(sampler=dict(num=256))),  # 训练期显存更稳
#     test_cfg=dict(rpn=dict(max_per_img=600), rcnn=dict(max_per_img=100)),  # 评估期显存更稳
# )
# 启用方法：在本文件末尾追加/放开该 block 即可按字典方式 merge 基类

# ============================ /可逐步启用的改动 ============================

# ---------- DataLoaders（当前：显式使用“基类”的 pipeline） ----------
train_dataloader = dict(
    batch_size=8,
    # --- 如果按 STEP 3 启用梯度累积，建议把 batch_size 改为 2（显存更稳）
    # batch_size=2,  # ← STEP 3：配合 accumulative_counts=4，等效总 batch 仍为 8
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=dict(classes=classes),
        ann_file='drone_ann/single_clear_day_ir/train.json', # need to change
        data_prefix=dict(img='/userhome/liqiulu/data/drone_ir_clear_day/00001'), # need to change
        filter_cfg=dict(filter_empty_gt=True),
        # pipeline={{_base_.train_pipeline}},  # 现状：用“基类”训练流水线
        pipeline=train_pipeline,      # ← STEP 1：改为本文件的训练流水线
        # pipeline=train_pipeline_choice,     # ← STEP 2：改为本文件的“放大+裁剪/随机二选一”训练流水线
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
        ann_file='drone_ann/single_clear_day_ir/val.json',
        data_prefix=dict(img='/userhome/liqiulu/data/drone_ir_clear_day/00001'), # need to change
        test_mode=True,
        filter_cfg=dict(filter_empty_gt=True),
        pipeline={{_base_.test_pipeline}},  # 现状：用“基类”测试流水线
        # pipeline=test_pipeline_light,      # ← STEP 1：改为本文件的轻量测试流水线（1333x800）
    )
)
test_dataloader = val_dataloader

# ---------- 评测器 ----------
val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'drone_ann/single_clear_day_ir/val.json', # need to change
    metric='bbox',
    format_only=False
)
test_evaluator = val_evaluator

# ---------- 模型（单类 + 小目标友好 RPN 锚框） ----------
model = dict(
    # 单类
    roi_head=dict(bbox_head=dict(num_classes=1)),
    backbone=dict(diff_config=dict(classes=classes)),

    # 小目标锚框：更小尺度 + 加瘦高比例（其余沿用基类）
    rpn_head=dict(
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[2, 4, 8],
            ratios=[0.33, 0.5, 1.0, 2.0],
            # strides 走基类即可
        )
    )
)

# ------------- 运行小贴士（非代码）-------------
# 启动时推荐（抗碎片）：export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
# 如果某张卡空闲不多，先：export CUDA_VISIBLE_DEVICES=<id>
# 逐步启用建议顺序：
#   STEP 1 → STEP 3 → STEP 2 → STEP 4
# 原因：先稳评估显存，再用 AMP/累积降训练显存，然后再“放大+裁剪”提升小目标可见性，最后微调 proposals/padding。
# 其中 STEP 3（AMP/累积）对结果无影响，STEP 1/2/4 影响极小，可根据显存情况选择性启用。