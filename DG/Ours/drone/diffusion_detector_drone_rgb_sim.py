_base_ = [  # 声明需要继承的基础配置
    '../../_base_/models/faster-rcnn_diff_fpn.py',  # 继承扩散版 Faster R-CNN 模型定义
    '../../_base_/dg_setting/dg_20k.py',  # 继承 20k 迭代的训练调度
]  # 结束基础配置列表
classes = ('drone',)  # 指定单类别为无人机
dataset_type = 'CocoDataset'  # 使用 COCO 数据集格式
data_root = 'data/'  # 指向数据软链接根目录
rgb_img_prefix = 'sim_drone_rgb/Town01_Opt/carla_data/'  # 仿真无人机 RGB 图像根目录
backend_args = None  # 使用默认后端读取图像
color_space_light = [  # 定义较轻量的颜色增强空间
    [dict(type='AutoContrast')],  # 自动对比度调节
    [dict(type='Equalize')],  # 直方图均衡化增强
    [dict(type='Color')],  # 调整饱和度
    [dict(type='Contrast')],  # 调整对比度
    [dict(type='Brightness')],  # 调整亮度
    [dict(type='Sharpness')],  # 调整锐度
]  # 结束颜色增强定义
train_pipeline = [  # 定义训练时的增广流水线
    dict(type='LoadImageFromFile', backend_args=backend_args),  # 读取仿真图像
    dict(type='LoadAnnotations', with_bbox=True),  # 加载边界框标注
    dict(type='Resize', scale=(1600, 900), keep_ratio=True),  # 调整图像分辨率
    dict(  # 定义随机裁剪步骤
        type='RandomCrop',  # 使用随机裁剪操作
        crop_size=(640, 640),  # 设置裁剪尺寸
        allow_negative_crop=True,  # 允许裁剪后无目标
        recompute_bbox=True),  # 裁剪后重新计算标注
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1e-2, 1e-2)),  # 过滤极小目标
    dict(type='RandomFlip', prob=0.5),  # 以 50% 概率水平翻转
    dict(type='RandAugment', aug_space=color_space_light, aug_num=1),  # 随机选择一种颜色增强
    dict(  # 打包训练样本
        type='PackDetInputs',  # 使用检测任务打包器
        meta_keys=(  # 指定需要保留的元信息
            'img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor', 'flip', 'flip_direction', 'homography_matrix')),  # 列出所有元信息键
]  # 结束训练流水线
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
train_dataloader = dict(  # 定义训练数据加载器
    batch_size=8,  # 设置批大小
    num_workers=8,  # 设置工作线程数
    persistent_workers=True,  # 启用线程持久化以提升性能
    sampler=dict(type='DefaultSampler', shuffle=True),  # 使用默认随机采样器
    batch_sampler=dict(type='AspectRatioBatchSampler'),  # 按长宽比分组组成 batch
    dataset=dict(  # 配置训练数据集
        type=dataset_type,  # 指定数据集类型
        data_root=data_root,  # 数据根目录
        metainfo=dict(classes=classes),  # 注入类别元信息
        ann_file='sim_drone_ann/rgb/train.json',  # 指向仿真训练标注文件
        data_prefix=dict(img=rgb_img_prefix),  # 指向仿真图像路径前缀
        filter_cfg=dict(filter_empty_gt=True),  # 过滤无标注图像
        pipeline=train_pipeline))  # 绑定训练增广流水线
val_dataloader = dict(  # 定义验证数据加载器
    batch_size=1,  # 单张图像评估
    num_workers=8,  # 工作线程数
    persistent_workers=True,  # 启用线程持久化
    drop_last=False,  # 不丢弃最后一个 batch
    sampler=dict(type='DefaultSampler', shuffle=False),  # 顺序采样
    dataset=dict(  # 配置验证数据集
        type=dataset_type,  # 使用 COCO 数据集格式
        data_root=data_root,  # 数据根目录
        metainfo=dict(classes=classes),  # 类别元信息
        ann_file='sim_drone_ann/rgb/val.json',  # 指向仿真验证标注文件
        data_prefix=dict(img=rgb_img_prefix),  # 图像目录前缀
        # ann_file='real_drone_ann/val_visible.json',  # 真实域测试标注文件
        # data_prefix=dict(img='real_drone_rgb/'),  # 真实域图像根目录
        test_mode=True,  # 启用测试模式以跳过数据增强
        filter_cfg=dict(filter_empty_gt=True),  # 过滤无标注图像
        pipeline=test_pipeline))  # 复用基础测试流水线
test_dataloader = val_dataloader  # 将测试加载器与验证加载器保持一致
val_evaluator = dict(  # 定义验证阶段评估器
    type='CocoMetric',  # 使用 COCO 指标
    ann_file=data_root + 'sim_drone_ann/rgb/val.json',  # 指向验证标注文件
    # ann_file=data_root + 'real_drone_ann/val_visible.json',  # 指向验证标注文件
    metric='bbox',  # 计算边界框指标
    format_only=False)  # 直接输出评估结果
test_evaluator = val_evaluator  # 复用验证评估器作为测试评估器
model = dict(  # 重写模型特定参数
    roi_head=dict(bbox_head=dict(num_classes=1)),  # 将 ROI Head 的类别数设置为 1
    backbone=dict(diff_config=dict(classes=classes)),  # 将类别信息写入扩散主干配置
    rpn_head=dict(  # 自定义 RPN 头以适应小目标
        anchor_generator=dict(  # 修改锚框生成策略
            type='AnchorGenerator',  # 指定生成器类型
            scales=[2, 4, 8],  # 使用较小尺度捕捉小型无人机
            ratios=[0.33, 0.5, 1.0, 2.0], # 扩展宽高比以覆盖细长目标
            strides=[4, 8, 16, 32, 64]))) # 明确各层步长以匹配特征图