# -*- coding: utf-8 -*-  # 指明文件编码防止中文注释出现乱码
"""仿真IR与仿真RGB双模态扩散教师互学习配置。"""  # 顶部文档字符串概述配置目的

_base_ = [  # 指定需要继承的基础配置列表
    '../../_base_/models/faster-rcnn_diff_fpn.py',  # 继承扩散版Faster R-CNN骨干与头部定义
    '../../_base_/dg_setting/dg_20k.py',  # 继承通用20k迭代训练调度与日志可视化
]  # 结束基础配置列表

classes = ('drone',)  # 定义训练任务涉及的类别元组，仅包含无人机

dataset_type = 'CocoDataset'  # 指定训练与验证数据采用COCO格式标注

data_root = 'data/'  # 定义数据根目录，方便拼接相对路径

backend_args = None  # 使用默认后端读取方式，不额外传递文件系统参数

ir_img_prefix = 'sim_drone_ir/Town01_Opt/carla_data/'  # 仿真IR图像所在目录

rgb_img_prefix = 'sim_drone_rgb/Town01_Opt/carla_data/'  # 仿真RGB图像所在目录

real_img_prefix = 'real_drone_rgb/'  # 真实RGB图像所在目录用于验证

color_space_rgb = [  # 定义仿真RGB分支的颜色增强候选集合
    [dict(type='ColorTransform')],  # 候选1：颜色空间扰动
    [dict(type='AutoContrast')],  # 候选2：自动对比度增强
    [dict(type='Equalize')],  # 候选3：直方图均衡化
    [dict(type='Sharpness')],  # 候选4：锐化操作强化边缘
    [dict(type='Posterize')],  # 候选5：色调分层模拟压缩伪影
    [dict(type='Solarize')],  # 候选6：曝光反转模拟极端光照
    [dict(type='Color')],  # 候选7：饱和度调整
    [dict(type='Contrast')],  # 候选8：对比度调整
    [dict(type='Brightness')],  # 候选9：亮度调整
]  # RGB颜色增强空间定义结束

color_space_ir = [  # 定义仿真IR分支的温和颜色增强集合
    [dict(type='AutoContrast')],  # 候选1：自动对比度增强
    [dict(type='Equalize')],  # 候选2：直方图均衡化
    [dict(type='Color')],  # 候选3：伪彩上色模拟不同热力映射
    [dict(type='Contrast')],  # 候选4：对比度调整
    [dict(type='Brightness')],  # 候选5：亮度调整
    [dict(type='Sharpness')],  # 候选6：锐度调整
]  # IR颜色增强空间定义结束

def _build_sup_pipeline(color_space, sensor_tag, random_erasing_cfg):  # 定义辅助函数用于构建监督流水线
    pipeline = [  # 初始化流水线步骤列表
        dict(type='LoadImageFromFile', backend_args=backend_args),  # 读取图像文件
        dict(type='LoadAnnotations', with_bbox=True),  # 加载边界框标注
        dict(type='Resize', scale=(1600, 900), keep_ratio=True),  # 首先缩放到较高分辨率保留小目标
        dict(  # 定义随机裁剪步骤
            type='RandomCrop',  # 使用随机裁剪
            crop_type='absolute',  # 采用绝对尺寸裁剪
            crop_size=(640, 640),  # 裁剪窗口大小
            allow_negative_crop=True,  # 允许裁剪后无目标以提升鲁棒性
            recompute_bbox=True,  # 重新计算裁剪后的边界框
        ),  # 随机裁剪配置结束
        dict(type='RandomFlip', prob=0.5),  # 以50%概率执行水平翻转
        dict(  # 在增强之前写入传感器标签
            type='SetSensorTag',  # 调用自定义传感器标记变换
            sensor=sensor_tag,  # 指定当前流水线对应的传感器名称
        ),  # SetSensorTag配置结束
        dict(  # 随机顺序执行单个RandAugment增强
            type='RandomOrder',  # 使用随机顺序模块
            transforms=[dict(type='RandAugment', aug_space=color_space, aug_num=1)],  # 在给定增强空间中采样一次
        ),  # RandAugment配置结束
        random_erasing_cfg,  # 插入随机擦除配置以提升遮挡鲁棒性
        dict(type='FilterAnnotations', min_gt_bbox_wh=(1e-2, 1e-2)),  # 过滤面积过小的标注框
        dict(  # 打包检测输入
            type='PackDetInputs',  # 使用检测任务打包器
            meta_keys=(  # 指定需要保留的元信息键
                'img_id',  # 图像编号
                'img_path',  # 图像路径
                'ori_shape',  # 原始图像尺寸
                'img_shape',  # 当前图像尺寸
                'scale_factor',  # 缩放比例
                'flip',  # 是否进行了翻转
                'flip_direction',  # 翻转方向
                'homography_matrix',  # 单应矩阵记录仿射操作
                'sensor',  # 传感器标签用于多教师路由
            ),  # 元信息键定义结束
        ),  # PackDetInputs配置结束
    ]  # 监督流水线定义完成
    return pipeline  # 返回构建好的流水线

rgb_random_erasing = dict(type='RandomErasing', n_patches=(1, 5), ratio=(0, 0.2))  # 为RGB分支定义较强的随机擦除

ir_random_erasing = dict(type='RandomErasing', n_patches=(1, 3), ratio=(0, 0.1))  # 为IR分支定义较温和的随机擦除

rgb_train_pipeline = _build_sup_pipeline(color_space_rgb, 'sim_rgb', rgb_random_erasing)  # 构建RGB监督流水线

ir_train_pipeline = _build_sup_pipeline(color_space_ir, 'sim_ir', ir_random_erasing)  # 构建IR监督流水线

val_test_pipeline = [  # 构建验证与测试共用流水线
    dict(type='LoadImageFromFile', backend_args=backend_args),  # 读取图像文件
    dict(type='LoadAnnotations', with_bbox=True),  # 加载真实标注以计算指标
    dict(type='Resize', scale=(1600, 900), keep_ratio=True),  # 调整分辨率保持纵横比
    dict(  # 写入真实域传感器标签
        type='SetSensorTag',  # 调用自定义传感器标记变换
        sensor='real_rgb',  # 标记真实RGB域
    ),  # SetSensorTag配置结束
    dict(  # 打包验证数据
        type='PackDetInputs',  # 使用检测任务打包器
        meta_keys=(  # 指定需要保留的元信息
            'img_id',  # 图像编号
            'img_path',  # 图像路径
            'ori_shape',  # 原始尺寸
            'img_shape',  # 当前尺寸
            'scale_factor',  # 缩放比例
            'sensor',  # 传感器标签
        ),  # 元信息键定义结束
    ),  # PackDetInputs配置结束
]  # 验证与测试流水线定义完成

train_dataloader = dict(  # 构建训练阶段的数据加载器
    batch_size=16,  # 每个迭代批量大小
    num_workers=16,  # 数据加载工作进程数
    persistent_workers=True,  # 启用worker复用以减少初始化开销
    sampler=dict(type='DefaultSampler', shuffle=True),  # 使用默认随机采样器打乱数据
    batch_sampler=dict(type='AspectRatioBatchSampler'),  # 按宽高比分组以减少填充
    dataset=dict(  # 指定训练使用的数据集
        type='ConcatDataset',  # 采用拼接方式合并IR与RGB监督数据
        datasets=[  # 定义需要拼接的两个子数据集
            dict(  # 配置仿真RGB数据集
                type=dataset_type,  # 使用COCO格式数据集类
                data_root=data_root,  # 指定数据根目录
                metainfo=dict(classes=classes),  # 传入类别信息
                ann_file='sim_drone_ann/rgb/train.json',  # RGB训练标注文件
                data_prefix=dict(img=rgb_img_prefix),  # RGB图像目录前缀
                filter_cfg=dict(filter_empty_gt=True),  # 过滤无标注图片
                pipeline=rgb_train_pipeline,  # 绑定RGB监督流水线
            ),  # 仿真RGB数据集配置结束
            dict(  # 配置仿真IR数据集
                type=dataset_type,  # 同样使用COCO格式
                data_root=data_root,  # 指定根目录
                metainfo=dict(classes=classes),  # 提供类别信息
                ann_file='sim_drone_ann/ir/train.json',  # IR训练标注文件
                data_prefix=dict(img=ir_img_prefix),  # IR图像目录前缀
                filter_cfg=dict(filter_empty_gt=True),  # 过滤无标注图片
                pipeline=ir_train_pipeline,  # 绑定IR监督流水线
            ),  # 仿真IR数据集配置结束
        ],  # 子数据集列表定义结束
    ),  # 数据集配置结束
)  # 训练数据加载器定义完成

val_dataloader = dict(  # 构建验证阶段的数据加载器
    batch_size=1,  # 单张图像评估以避免显存峰值
    num_workers=8,  # 验证阶段数据加载线程数
    persistent_workers=True,  # 启用worker复用
    drop_last=False,  # 不丢弃最后一个批次
    sampler=dict(type='DefaultSampler', shuffle=False),  # 顺序采样保证评估稳定
    dataset=dict(  # 定义验证数据集
        type=dataset_type,  # COCO格式数据集
        data_root=data_root,  # 指定数据根目录
        metainfo=dict(classes=classes),  # 传入类别信息
        ann_file='real_drone_ann/val_visible.json',  # 真实RGB验证标注文件
        data_prefix=dict(img=real_img_prefix),  # 真实RGB图像目录
        test_mode=True,  # 开启测试模式
        pipeline=val_test_pipeline,  # 使用验证流水线
    ),  # 验证数据集配置结束
)  # 验证数据加载器定义完成

test_dataloader = dict(  # 构建测试阶段的数据加载器
    batch_size=1,  # 单张图像推理
    num_workers=8,  # 测试阶段线程数
    persistent_workers=True,  # 启用worker复用
    drop_last=False,  # 保留所有样本
    sampler=dict(type='DefaultSampler', shuffle=False),  # 顺序采样
    dataset=dict(  # 指定测试数据集
        type=dataset_type,  # COCO格式数据集
        data_root=data_root,  # 指定根目录
        metainfo=dict(classes=classes),  # 传入类别信息
        ann_file='real_drone_ann/test_visible.json',  # 真实RGB测试标注文件
        data_prefix=dict(img=real_img_prefix),  # 测试图像目录
        test_mode=True,  # 设置为测试模式
        pipeline=val_test_pipeline,  # 使用与验证一致的流水线
    ),  # 测试数据集配置结束
)  # 测试数据加载器定义完成

val_evaluator = dict(  # 定义验证评估器
    type='CocoMetric',  # 使用COCO检测指标
    ann_file=data_root + 'real_drone_ann/val_visible.json',  # 指定验证标注路径
    metric='bbox',  # 评估边界框指标
    format_only=False,  # 直接计算指标
)  # 验证评估器定义完成

test_evaluator = dict(  # 定义测试评估器
    type='CocoMetric',  # 采用COCO检测指标
    ann_file=data_root + 'real_drone_ann/test_visible.json',  # 指定测试标注路径
    metric='bbox',  # 评估边界框指标
    format_only=False,  # 直接输出指标
)  # 测试评估器定义完成

model = dict(  # 对模型结构进行必要调整
    roi_head=dict(bbox_head=dict(num_classes=len(classes))),  # 将ROI头类别数设为无人机数量
    backbone=dict(diff_config=dict(classes=classes)),  # 将类别信息传入扩散主干
    rpn_head=dict(  # 调整RPN锚框生成器
        anchor_generator=dict(  # 自定义锚框生成参数
            type='AnchorGenerator',  # 指定锚框生成器类型
            scales=[2, 4, 8],  # 使用小尺度锚框捕获无人机
            ratios=[0.33, 0.5, 1.0, 2.0],  # 设置多种纵横比提升召回
            strides=[4, 8, 16, 32, 64],  # 对应FPN层级的步长
        ),  # 锚框生成器配置结束
    ),  # RPN头配置结束
)  # 模型结构调整完成

auto_scale_lr = dict(enable=True, base_batch_size=16)  # 启用自动学习率缩放以适配不同GPU数量
