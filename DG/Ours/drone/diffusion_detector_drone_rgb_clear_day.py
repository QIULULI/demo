# _base_ = [
#     '../cityscapes/diffusion_detector_cityscapes.py',
# ]

# classes = ('drone',)

# dataset_type = 'CocoDataset'
# data_root = '/userhome/liqiulu/code/Fitness-Generalization-Transferability/data/'

# img_prefix = '/userhome/liqiulu/data/drone_rgb_clear_day/00001'
# ann_prefix = 'drone_ann/single_clear_day_rgb/'
# backend_args = None

# color_space_light = [
#     [dict(type='AutoContrast')],
#     [dict(type='Equalize')],
#     [dict(type='Color')],
#     [dict(type='Contrast')],
#     [dict(type='Brightness')],
#     [dict(type='Sharpness')],
# ]

# train_pipeline = [
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

# test_pipeline = [
#     dict(type='LoadImageFromFile', backend_args=backend_args),
#     dict(type='Resize', scale=(1600, 960), keep_ratio=True),
#     dict(type='LoadAnnotations', with_bbox=True),
#     dict(
#         type='PackDetInputs',
#         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor'),
#     ),
# ]

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
#         ann_file=ann_prefix+'train.json',
#         data_prefix=dict(img=img_prefix),
#         filter_cfg=dict(filter_empty_gt=True),
#         pipeline=train_pipeline,
#     ),
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
#         ann_file=ann_prefix+'val.json',
#         data_prefix=dict(img=img_prefix),
#         test_mode=True,
#         filter_cfg=dict(filter_empty_gt=True),
#         pipeline=test_pipeline,
#     ),
# )

# test_dataloader = val_dataloader

# val_evaluator = dict(
#     type='CocoMetric',
#     ann_file=data_root + ann_prefix + 'val.json',
#     metric='bbox',
#     format_only=False,
# )

# test_evaluator = val_evaluator

# model = dict(
#     roi_head=dict(bbox_head=dict(num_classes=1)),
#     backbone=dict(diff_config=dict(classes=classes)),
#     rpn_head=dict(
#         anchor_generator=dict(
#             type='AnchorGenerator',
#             scales=[2, 4, 8],
#             ratios=[0.33, 0.5, 1.0, 2.0],
#             strides=[4, 8, 16, 32, 64],
#         ),
#     ),
# )
_base_ = [  # 指定需要继承的基础配置列表
    '../../_base_/models/faster-rcnn_diff_fpn.py',  # 继承扩散检测器结构
    '../../_base_/dg_setting/dg_20k.py',  # 继承DG训练调度
]  # 结束基础配置列表
# ----------------------------------------------------------------------------------------------------
classes = ('drone',)  # 定义类别为无人机
# ----------------------------------------------------------------------------------------------------
dataset_type = 'CocoDataset'  # 指定数据集格式
data_root = 'data/'  # 指定标注根目录
rgb_day_img_prefix = '/userhome/liqiulu/data/drone_rgb_clear_day/00001'  # 定义晴天RGB图像目录
rgb_night_img_prefix = '/userhome/liqiulu/data/drone_rgb_clear_night/00001'  # 定义夜晚RGB图像目录
backend_args = None  # 使用默认后端
# ----------------------------------------------------------------------------------------------------
color_space_light = [  # 定义颜色增强空间
    [dict(type='AutoContrast')],  # 自动对比度增强
    [dict(type='Equalize')],  # 直方图均衡化
    [dict(type='Color')],  # 调整饱和度
    [dict(type='Contrast')],  # 调整对比度
    [dict(type='Brightness')],  # 调整亮度
    [dict(type='Sharpness')],  # 调整锐度
]  # 结束颜色增强定义
# ----------------------------------------------------------------------------------------------------
train_pipeline = [  # 构建训练流水线
    dict(type='LoadImageFromFile', backend_args=backend_args),  # 读取RGB图像
    dict(type='LoadAnnotations', with_bbox=True),  # 载入标注
    dict(type='Resize', scale=(1600, 960), keep_ratio=True),  # 高分辨率缩放
    dict(  # 定义随机裁剪
        type='RandomCrop',  # 随机裁剪操作
        crop_size=(640, 640),  # 裁剪尺寸
        allow_negative_crop=True,  # 允许无目标裁剪
        recompute_bbox=True),  # 重新计算边界框
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1e-2, 1e-2)),  # 过滤极小框
    dict(type='RandomFlip', prob=0.5),  # 随机水平翻转
    dict(type='RandAugment', aug_space=color_space_light, aug_num=1),  # 颜色增强
    dict(  # 打包输入
        type='PackDetInputs',  # 检测打包器
        meta_keys=(  # 元信息键
            'img_id',  # 图像编号
            'img_path',  # 图像路径
            'ori_shape',  # 原始尺寸
            'img_shape',  # 增强后尺寸
            'scale_factor',  # 缩放比例
            'flip',  # 是否翻转
            'flip_direction',  # 翻转方向
            'homography_matrix')),  # 仿射矩阵
]  # 结束训练流水线
# ----------------------------------------------------------------------------------------------------
test_pipeline = [  # 构建验证与测试阶段的数据处理流水线
    dict(type='LoadImageFromFile', backend_args=backend_args),  # 读取图像文件
    dict(type='Resize', scale=(1600, 960), keep_ratio=True),  # 按照训练同样的高分辨率缩放
    dict(type='LoadAnnotations', with_bbox=True),  # 加载标注用于评估指标
    dict(  # 打包评估输入
        type='PackDetInputs',  # 使用检测任务打包器
        meta_keys=(  # 指定需要保留的元信息键
            'img_id',  # 图像编号
            'img_path',  # 图像路径
            'ori_shape',  # 原始图像尺寸
            'img_shape',  # 处理后图像尺寸
            'scale_factor')),  # 缩放比例用于恢复坐标
]  # 结束测试流水线定义
# ----------------------------------------------------------------------------------------------------
train_dataloader = dict(  # 配置训练加载器
    batch_size=8,  # 批大小
    num_workers=8,  # 工作进程
    persistent_workers=True,  # 复用线程
    sampler=dict(type='DefaultSampler', shuffle=True),  # 随机采样
    batch_sampler=dict(type='AspectRatioBatchSampler'),  # 宽高比分组
    dataset=dict(  # 定义训练数据集
        type=dataset_type,  # COCO格式
        data_root=data_root,  # 标注根目录
        metainfo=dict(classes=classes),  # 类别信息
        ann_file='drone_ann/single_clear_day_rgb/train.json',  # 晴天训练标注
        data_prefix=dict(img=rgb_day_img_prefix),  # 晴天图像目录
        filter_cfg=dict(filter_empty_gt=True),  # 过滤无标注
        pipeline=train_pipeline))  # 使用训练流水线
# ----------------------------------------------------------------------------------------------------
val_dataloader = dict(  # 配置验证加载器
    batch_size=1,  # 单张评估
    num_workers=8,  # 工作进程
    persistent_workers=True,  # 复用线程
    drop_last=False,  # 不丢弃尾批
    sampler=dict(type='DefaultSampler', shuffle=False),  # 顺序采样
    dataset=dict(  # 定义验证数据集
        type=dataset_type,  # COCO格式
        data_root=data_root,  # 标注根目录
        metainfo=dict(classes=classes),  # 类别信息
        ann_file='drone_ann/single_clear_day_rgb/val.json',  # 晴天验证标注
        data_prefix=dict(img=rgb_day_img_prefix),  # 晴天图像目录
        test_mode=True,  # 测试模式
        filter_cfg=dict(filter_empty_gt=True),  # 过滤无目标
        pipeline=test_pipeline))  # 使用基础测试流水线
# ----------------------------------------------------------------------------------------------------
test_dataloader = dict(  # 配置夜间测试加载器
    batch_size=1,  # 单张测试
    num_workers=8,  # 工作进程
    persistent_workers=True,  # 复用线程
    drop_last=False,  # 不丢弃尾批
    sampler=dict(type='DefaultSampler', shuffle=False),  # 顺序采样
    dataset=dict(  # 定义夜间数据集
        type=dataset_type,  # COCO格式
        data_root=data_root,  # 标注根目录
        metainfo=dict(classes=classes),  # 类别信息
        ann_file='drone_ann/single_clear_night_rgb/val.json',  # 夜间验证标注
        data_prefix=dict(img=rgb_night_img_prefix),  # 夜间图像目录
        test_mode=True,  # 测试模式
        filter_cfg=dict(filter_empty_gt=True),  # 过滤无目标
        pipeline=test_pipeline))  # 使用基础测试流水线
# ----------------------------------------------------------------------------------------------------
val_evaluator = dict(  # 定义晴天验证评估器
    type='CocoMetric',  # COCO指标
    ann_file=data_root + 'drone_ann/single_clear_day_rgb/val.json',  # 晴天验证标注路径
    metric='bbox',  # 边界框指标
    format_only=False)  # 直接计算指标
# ----------------------------------------------------------------------------------------------------
test_evaluator = dict(  # 定义夜间测试评估器
    type='CocoMetric',  # COCO指标
    ann_file=data_root + 'drone_ann/single_clear_night_rgb/val.json',  # 夜间标注路径
    metric='bbox',  # 边界框指标
    format_only=False)  # 直接计算指标
# ----------------------------------------------------------------------------------------------------
model = dict(  # 调整模型设置
    roi_head=dict(bbox_head=dict(num_classes=1)),  # 设置ROI头类别数
    backbone=dict(diff_config=dict(classes=classes)),  # 将类别信息写入扩散主干
    rpn_head=dict(  # 调整RPN配置
        anchor_generator=dict(  # 修改锚框生成器
            type='AnchorGenerator',  # 指定生成器类型
            scales=[2, 4, 8],  # 设置更小尺度捕获微小目标
            ratios=[0.33, 0.5, 1.0, 2.0],  # 增加纵横比覆盖细长目标
            strides=[4, 8, 16, 32, 64])))  # 指定FPN层对应的步长