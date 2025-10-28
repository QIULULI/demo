_base_ = [  # 指定需要继承的基础配置列表
    '../../_base_/models/faster-rcnn_r101_fpn.py',  # 继承常规Faster R-CNN检测器结构
    '../../_base_/dg_setting/dg_20k.py',  # 继承通用DG训练调度和日志设置
]  # 结束基础配置列表
# ----------------------------------------------------------------------------------------------------
classes = ('drone',)  # 定义单类别列表
# ----------------------------------------------------------------------------------------------------
dataset_type = 'CocoDataset'  # 指明数据集格式为COCO
data_root = 'data/'  # 设置标注文件相对根目录
rgb_day_img_prefix = '/userhome/liqiulu/data/drone_rgb_clear_day/00001'  # 指定晴天RGB图像所在目录
rgb_night_img_prefix = '/userhome/liqiulu/data/drone_rgb_clear_night/00001'  # 指定夜晚RGB图像所在目录
backend_args = None  # 使用默认文件读取后端
# ----------------------------------------------------------------------------------------------------
color_space_light = [  # 定义轻量颜色增强空间
    [dict(type='AutoContrast')],  # 自动对比度增强
    [dict(type='Equalize')],  # 直方图均衡化
    [dict(type='Color')],  # 调整饱和度
    [dict(type='Contrast')],  # 调整对比度
    [dict(type='Brightness')],  # 调整亮度
    [dict(type='Sharpness')],  # 调整锐度
]  # 结束增强空间定义
# ----------------------------------------------------------------------------------------------------
train_pipeline = [  # 构建训练数据流水线
    dict(type='LoadImageFromFile', backend_args=backend_args),  # 读取RGB图像
    dict(type='LoadAnnotations', with_bbox=True),  # 载入标注框
    dict(type='Resize', scale=(1600, 960), keep_ratio=True),  # 放大图像到高分辨率
    dict(  # 定义随机裁剪步骤
        type='RandomCrop',  # 使用随机裁剪
        crop_size=(640, 640),  # 裁剪为640x640小贴片
        allow_negative_crop=True,  # 允许裁剪后无目标
        recompute_bbox=True),  # 重新计算边界框
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1e-2, 1e-2)),  # 过滤极小框
    dict(type='RandomFlip', prob=0.5),  # 随机水平翻转
    dict(type='RandAugment', aug_space=color_space_light, aug_num=1),  # 随机选择一种颜色增强
    dict(  # 打包输入数据
        type='PackDetInputs',  # 使用检测任务打包器
        meta_keys=(  # 指定元信息键
            'img_id',  # 图像编号
            'img_path',  # 图像路径
            'ori_shape',  # 原始尺寸
            'img_shape',  # 增强后尺寸
            'scale_factor',  # 缩放比例
            'flip',  # 是否翻转
            'flip_direction',  # 翻转方向
            'homography_matrix')),  # 仿射矩阵记录
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
train_dataloader = dict(  # 配置训练数据加载器
    batch_size=8,  # 设置批大小
    num_workers=8,  # 设置工作进程数
    persistent_workers=True,  # 复用线程
    sampler=dict(type='DefaultSampler', shuffle=True),  # 随机采样
    batch_sampler=dict(type='AspectRatioBatchSampler'),  # 宽高比分组
    dataset=dict(  # 指定训练数据集
        type=dataset_type,  # COCO格式
        data_root=data_root,  # 标注根目录
        metainfo=dict(classes=classes),  # 类别信息
        ann_file='drone_ann/single_clear_day_rgb/train.json',  # 晴天RGB训练标注
        data_prefix=dict(img=rgb_day_img_prefix),  # 晴天RGB图像目录
        filter_cfg=dict(filter_empty_gt=True),  # 过滤无目标样本
        pipeline=train_pipeline))  # 使用训练流水线
# ----------------------------------------------------------------------------------------------------
val_dataloader = dict(  # 配置验证数据加载器
    batch_size=1,  # 单张评估
    num_workers=8,  # 工作进程数
    persistent_workers=True,  # 复用线程
    drop_last=False,  # 不丢弃尾批
    sampler=dict(type='DefaultSampler', shuffle=False),  # 顺序采样
    dataset=dict(  # 定义晴天验证集
        type=dataset_type,  # COCO格式
        data_root=data_root,  # 标注根目录
        metainfo=dict(classes=classes),  # 类别信息
        ann_file='drone_ann/single_clear_day_rgb/val.json',  # 晴天RGB验证标注
        data_prefix=dict(img=rgb_day_img_prefix),  # 晴天验证图像目录
        test_mode=True,  # 标记测试模式
        filter_cfg=dict(filter_empty_gt=True),  # 过滤无目标
        pipeline=test_pipeline))  # 使用基础测试流水线
# ----------------------------------------------------------------------------------------------------
test_dataloader = dict(  # 配置夜间测试数据加载器
    batch_size=1,  # 单张测试
    num_workers=8,  # 工作进程数
    persistent_workers=True,  # 复用线程
    drop_last=False,  # 不丢弃尾批
    sampler=dict(type='DefaultSampler', shuffle=False),  # 顺序采样
    dataset=dict(  # 定义夜间评估集
        type=dataset_type,  # COCO格式
        data_root=data_root,  # 标注根目录
        metainfo=dict(classes=classes),  # 类别信息
        ann_file='drone_ann/single_clear_night_rgb/val.json',  # 夜间RGB验证标注
        data_prefix=dict(img=rgb_night_img_prefix),  # 夜间图像目录
        test_mode=True,  # 测试模式
        filter_cfg=dict(filter_empty_gt=True),  # 过滤无目标
        pipeline=test_pipeline))  # 使用基础测试流水线
# ----------------------------------------------------------------------------------------------------
val_evaluator = dict(  # 定义晴天验证评估器
    type='CocoMetric',  # 使用COCO指标
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
model = dict(  # 调整模型结构
    roi_head=dict(bbox_head=dict(num_classes=1)),  # 将检测头类别数设为1
    rpn_head=dict(  # 调整RPN配置
        anchor_generator=dict(  # 修改锚框生成器
            type='AnchorGenerator',  # 指定生成器类型
            scales=[2, 4, 8],  # 设置更小尺度捕获微小目标
            ratios=[0.33, 0.5, 1.0, 2.0])))  # 增加纵横比覆盖细长目标