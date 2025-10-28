# dataset_type = 'CocoDataset'
# data_root = '/userhome/liqiulu/code/Fitness-Generalization-Transferability/data/'
# classes = ('drone',)

# # Keep image roots consistent with the single-domain IR recipe.
# ir_img_prefix = '/userhome/liqiulu/data/drone_ir_clear_day/00001'
# rgb_img_prefix = '/userhome/liqiulu/data/drone_rgb_clear_day/00001'

# backend_args = None

# # Light-weight color augmentations used in the IR training recipe
# color_space_light = [
#     [dict(type='AutoContrast')],
#     [dict(type='Equalize')],
#     [dict(type='Color')],
#     [dict(type='Contrast')],
#     [dict(type='Brightness')],
#     [dict(type='Sharpness')],
# ]

# branch_field = ['sup', 'unsup_teacher', 'unsup_student']

# # Pipelines closely follow the single-domain diffusion detector recipe:
# # resize to 1600x960, crop 640x640 tiles, light color jitter, no random erasing.
# sup_aug_pipeline = [
#     dict(type='RandAugment', aug_space=color_space_light, aug_num=1),
#     dict(
#         type='PackDetInputs',
#         meta_keys=(
#             'img_id',
#             'img_path',
#             'ori_shape',
#             'img_shape',
#             'scale_factor',
#             'flip',
#             'flip_direction',
#             'homography_matrix',
#         ),
#     ),
# ]

# strong_pipeline = [
#     dict(type='RandAugment', aug_space=color_space_light, aug_num=1),
#     dict(
#         type='PackDetInputs',
#         meta_keys=(
#             'img_id',
#             'img_path',
#             'ori_shape',
#             'img_shape',
#             'scale_factor',
#             'flip',
#             'flip_direction',
#             'homography_matrix',
#         ),
#     ),
# ]

# weak_pipeline = [
#     dict(
#         type='PackDetInputs',
#         meta_keys=(
#             'img_id',
#             'img_path',
#             'ori_shape',
#             'img_shape',
#             'scale_factor',
#             'flip',
#             'flip_direction',
#             'homography_matrix',
#         ),
#     ),
# ]

# sup_pipeline = [
#     dict(type='LoadImageFromFile', backend_args=backend_args),
#     dict(type='LoadAnnotations', with_bbox=True),
#     dict(type='Resize', scale=(1600, 960), keep_ratio=True),
#     dict(
#         type='RandomCrop',
#         crop_type='absolute',
#         crop_size=(640, 640),
#         recompute_bbox=True,
#         allow_negative_crop=True,
#     ),
#     dict(type='FilterAnnotations', min_gt_bbox_wh=(1e-2, 1e-2)),
#     dict(type='RandomFlip', prob=0.5),
#     dict(
#         type='MultiBranch',
#         branch_field=branch_field,
#         sup=sup_aug_pipeline,
#     ),
# ]

# unsup_pipeline = [
#     dict(type='LoadImageFromFile', backend_args=backend_args),
#     dict(type='LoadEmptyAnnotations'),
#     dict(type='Resize', scale=(1600, 960), keep_ratio=True),
#     dict(
#         type='RandomCrop',
#         crop_type='absolute',
#         crop_size=(640, 640),
#         recompute_bbox=True,
#         allow_negative_crop=True,
#     ),
#     dict(type='RandomFlip', prob=0.5),
#     dict(
#         type='MultiBranch',
#         branch_field=branch_field,
#         unsup_teacher=weak_pipeline,
#         unsup_student=strong_pipeline,
#     ),
# ]

# test_pipeline = [
#     dict(type='LoadImageFromFile', backend_args=backend_args),
#     dict(type='Resize', scale=(1600, 960), keep_ratio=True),
#     dict(type='LoadAnnotations', with_bbox=True),
#     dict(
#         type='PackDetInputs',
#         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor'),
#     ),
# ]

# batch_size = 8
# num_workers = 8

# labeled_dataset = dict(
#     type=dataset_type,
#     data_root=data_root,
#     metainfo=dict(classes=classes),
#     ann_file='drone_ann/single_clear_day_ir/train.json',
#     data_prefix=dict(img=ir_img_prefix),
#     filter_cfg=dict(filter_empty_gt=True),
#     pipeline=sup_pipeline,
# )

# unlabeled_dataset = dict(
#     type=dataset_type,
#     data_root=data_root,
#     metainfo=dict(classes=classes),
#     ann_file='drone_ann/single_clear_day_rgb/train.json',
#     data_prefix=dict(img=rgb_img_prefix),
#     pipeline=unsup_pipeline,
# )

# train_dataloader = dict(
#     batch_size=batch_size,
#     num_workers=num_workers,
#     persistent_workers=True,
#     sampler=dict(
#         type='GroupMultiSourceSampler',
#         batch_size=batch_size,
#         source_ratio=[1, 1],
#     ),
#     dataset=dict(type='ConcatDataset', datasets=[labeled_dataset, unlabeled_dataset]),
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
#         data_prefix=dict(img=rgb_img_prefix),
#         test_mode=True,
#         filter_cfg=dict(filter_empty_gt=True),
#         pipeline=test_pipeline,
#     ),
# )

# test_dataloader = val_dataloader

# val_evaluator = dict(
#     type='CocoMetric',
#     ann_file=data_root + 'drone_ann/single_clear_day_rgb/val.json',
#     metric='bbox',
#     format_only=False,
# )

# test_evaluator = val_evaluator

dataset_type = 'CocoDataset'  # 指定数据集格式为COCO
data_root = 'data/'  # 设置标注文件根目录
classes = ('drone',)  # 定义单类别列表
backend_args = None  # 使用默认文件读取后端
branch_field = ['sup', 'unsup_teacher', 'unsup_student']  # 定义多分支流水线的键名
# ----------------------------------------------------------------------------------------------------
ir_img_prefix = 'real_drone_ir/train/'  # 真实红外训练图像目录
rgb_day_img_prefix = 'real_drone_rgb/train/'  # 真实可见光训练图像目录
rgb_night_img_prefix = 'real_drone_rgb/val/'  # 真实可见光验证图像目录
rgb_snow_day_img_prefix = 'real_drone_rgb/test/'  # 真实可见光测试图像目录
# ----------------------------------------------------------------------------------------------------
color_space_light = [  # 定义轻量颜色增强空间
    [dict(type='AutoContrast')],  # 自动对比度增强
    [dict(type='Equalize')],  # 直方图均衡化
    [dict(type='Color')],  # 调整饱和度
    [dict(type='Contrast')],  # 调整对比度
    [dict(type='Brightness')],  # 调整亮度
    [dict(type='Sharpness')],  # 调整锐度
]  # 结束颜色增强定义
# ----------------------------------------------------------------------------------------------------
strong_pipeline = [  # 定义学生分支的强增强流水线
    dict(  # 使用随机顺序组合增强
        type='RandomOrder',  # 指定随机顺序执行
        transforms=[  # 定义增强列表
            dict(type='RandAugment', aug_space=color_space_light, aug_num=1),  # 颜色增强
        ]),  # 结束增强列表
    dict(type='RandomErasing', n_patches=(1, 5), ratio=(0, 0.2)),  # 随机擦除增加扰动
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1e-2, 1e-2)),  # 过滤极小框
    dict(  # 打包输入数据
        type='PackDetInputs',  # 使用检测打包器
        meta_keys=(  # 指定元信息键
            'img_id',  # 图像编号
            'img_path',  # 图像路径
            'ori_shape',  # 原始尺寸
            'img_shape',  # 处理后尺寸
            'scale_factor',  # 缩放比例
            'flip',  # 是否翻转
            'flip_direction',  # 翻转方向
            'homography_matrix')),  # 仿射矩阵
]  # 结束强增强流水线
# ----------------------------------------------------------------------------------------------------
weak_pipeline = [  # 定义教师分支的弱增强流水线
    dict(  # 直接打包输入
        type='PackDetInputs',  # 检测打包器
        meta_keys=(  # 元信息键
            'img_id',  # 图像编号
            'img_path',  # 图像路径
            'ori_shape',  # 原始尺寸
            'img_shape',  # 处理后尺寸
            'scale_factor',  # 缩放比例
            'flip',  # 是否翻转
            'flip_direction',  # 翻转方向
            'homography_matrix')),  # 仿射矩阵
]  # 结束弱增强流水线
# ----------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------
sup_output_pipeline = [  # 定义监督分支的输出打包流程
    dict(  # 打包输入数据
        type='PackDetInputs',  # 指定检测任务打包器
        meta_keys=(  # 设置需要保留的元信息键
            'img_id',  # 图像编号
            'img_path',  # 图像路径
            'ori_shape',  # 原始尺寸
            'img_shape',  # 处理后尺寸
            'scale_factor',  # 缩放比例
            'flip',  # 是否翻转
            'flip_direction',  # 翻转方向
            'homography_matrix'))  # 仿射矩阵
]  # 结束监督输出打包流程
# ----------------------------------------------------------------------------------------------------
sup_pipeline = [  # 定义有标签分支流水线
    dict(type='LoadImageFromFile', backend_args=backend_args),  # 读取图像
    dict(type='LoadAnnotations', with_bbox=True),  # 载入标注
    dict(type='Resize', scale=(1600, 960), keep_ratio=True),  # 高分辨率缩放
    dict(  # 定义随机裁剪
        type='RandomCrop',  # 随机裁剪
        crop_size=(640, 640),  # 裁剪尺寸
        allow_negative_crop=True,  # 允许无目标
        recompute_bbox=True),  # 重新计算边界框
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1e-2, 1e-2)),  # 过滤极小框
    dict(type='RandomFlip', prob=0.5),  # 随机水平翻转
    dict(type='RandAugment', aug_space=color_space_light, aug_num=1),  # 颜色增强
    dict(  # 拆分多分支输出
        type='MultiBranch',  # 使用多分支模块
        branch_field=branch_field,  # 指定分支名称
        sup=sup_output_pipeline),  # 绑定监督输出流程
]  # 结束有标签流水线
# ----------------------------------------------------------------------------------------------------
unsup_pipeline = [  # 定义无标签分支流水线
    dict(type='LoadImageFromFile', backend_args=backend_args),  # 读取RGB图像
    dict(type='LoadEmptyAnnotations'),  # 构造空标注以兼容流程
    dict(type='Resize', scale=(1600, 960), keep_ratio=True),  # 高分辨率缩放
    dict(  # 定义随机裁剪
        type='RandomCrop',  # 随机裁剪
        crop_size=(640, 640),  # 裁剪尺寸
        allow_negative_crop=True,  # 允许无目标
        recompute_bbox=True),  # 重新计算边界框
    dict(type='RandomFlip', prob=0.5),  # 随机水平翻转
    dict(  # 拆分多分支输出
        type='MultiBranch',  # 使用多分支模块
        branch_field=branch_field,  # 分支键名
        unsup_teacher=weak_pipeline,  # 指定教师弱增强
        unsup_student=strong_pipeline),  # 指定学生强增强
]  # 结束无标签流水线
# ----------------------------------------------------------------------------------------------------
test_pipeline = [  # 定义测试流水线
    dict(type='LoadImageFromFile', backend_args=backend_args),  # 读取图像
    dict(type='Resize', scale=(1600, 960), keep_ratio=True),  # 与训练分辨率对齐
    dict(type='LoadAnnotations', with_bbox=True),  # 载入标注
    dict(  # 打包输入
        type='PackDetInputs',  # 检测打包器
        meta_keys=(  # 元信息键
            'img_id',  # 图像编号
            'img_path',  # 图像路径
            'ori_shape',  # 原始尺寸
            'img_shape',  # 处理后尺寸
            'scale_factor')),  # 缩放比例
]  # 结束测试流水线
# ----------------------------------------------------------------------------------------------------
batch_size = 8  # 定义单个分支的批大小
num_workers = 8  # 定义数据加载进程数
# ----------------------------------------------------------------------------------------------------
labeled_dataset = dict(  # 定义有标签数据集
    type=dataset_type,  # COCO格式
    data_root=data_root,  # 标注根目录
    metainfo=dict(classes=classes),  # 类别信息
    ann_file='real_drone_ann/train_infrared.json',  # IR训练标注
    data_prefix=dict(img=ir_img_prefix),  # IR图像目录
    filter_cfg=dict(filter_empty_gt=True),  # 过滤无目标
    pipeline=sup_pipeline)  # 使用监督流水线
# ----------------------------------------------------------------------------------------------------
unlabeled_dataset = dict(  # 定义无标签数据集
    type=dataset_type,  # COCO格式
    data_root=data_root,  # 标注根目录
    metainfo=dict(classes=classes),  # 类别信息
    ann_file='real_drone_ann/train_visible.json',  # RGB伪标注占位文件
    data_prefix=dict(img=rgb_day_img_prefix),  # RGB图像目录
    pipeline=unsup_pipeline)  # 使用无标签流水线
# ----------------------------------------------------------------------------------------------------
train_dataloader = dict(  # 定义训练数据加载器
    batch_size=batch_size,  # 批大小
    num_workers=num_workers,  # 进程数
    persistent_workers=True,  # 复用线程
    sampler=dict(type='GroupMultiSourceSampler', batch_size=batch_size, source_ratio=[1, 1]),  # 定义源目标1:1采样
    dataset=dict(type='ConcatDataset', datasets=[labeled_dataset, unlabeled_dataset]))  # 拼接两种数据集
# ----------------------------------------------------------------------------------------------------
val_dataloader = dict(  # 定义真实域验证数据加载器
    batch_size=1,  # 单张评估
    num_workers=8,  # 工作进程
    persistent_workers=True,  # 复用线程
    drop_last=False,  # 不丢弃尾批
    sampler=dict(type='DefaultSampler', shuffle=False),  # 顺序采样
    dataset=dict(  # 定义验证数据集
        type=dataset_type,  # COCO格式
        data_root=data_root,  # 标注根目录
        metainfo=dict(classes=classes),  # 类别信息
        ann_file='real_drone_ann/val_visible.json',  # 可见光验证标注
        data_prefix=dict(img=rgb_night_img_prefix),  # 可见光验证图像目录
        test_mode=True,  # 测试模式
        filter_cfg=dict(filter_empty_gt=True),  # 过滤无目标
        pipeline=test_pipeline))  # 使用测试流水线
# ----------------------------------------------------------------------------------------------------
test_dataloader = dict(  # 定义真实域测试数据加载器
    batch_size=1,  # 单张评估
    num_workers=8,  # 工作进程
    persistent_workers=True,  # 复用线程
    drop_last=False,  # 不丢弃尾批
    sampler=dict(type='DefaultSampler', shuffle=False),  # 顺序采样
    dataset=dict(  # 定义测试数据集
        type=dataset_type,  # COCO格式
        data_root=data_root,  # 标注根目录
        metainfo=dict(classes=classes),  # 类别信息
        ann_file='real_drone_ann/test_visible.json',  # 可见光测试标注
        data_prefix=dict(img=rgb_snow_day_img_prefix),  # 可见光测试图像目录
        test_mode=True,  # 测试模式
        filter_cfg=dict(filter_empty_gt=True),  # 过滤无目标
        pipeline=test_pipeline))  # 使用测试流水线
# ----------------------------------------------------------------------------------------------------
val_evaluator = dict(  # 定义验证评估器
    type='CocoMetric',  # COCO指标
    ann_file=data_root + 'real_drone_ann/val_visible.json',  # 验证标注路径
    metric='bbox',  # 边界框指标
    format_only=False)  # 直接计算指标
# ----------------------------------------------------------------------------------------------------
test_evaluator = dict(  # 定义测试评估器
    type='CocoMetric',  # COCO指标
    ann_file=data_root + 'real_drone_ann/test_visible.json',  # 测试标注路径
    metric='bbox',  # 边界框指标
    format_only=False)  # 直接计算指标
# test_evaluator = dict(  # 定义夜间测试评估器
#     type='CocoMetric',  # COCO指标
#     ann_file=data_root + 'drone_ann/single_snow_day_rgb/val.json',  # 夜间标注路径
#     metric='bbox',  # 边界框指标
#     format_only=False)  # 直接计算指标