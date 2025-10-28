_base_ = [  # 指定需要继承的基础配置列表
    '../../_base_/models/diffusion_guided_generalization_faster_rcnn_r101_fpn.py',  # 复用扩散引导的一阶段检测器结构
    '../../_base_/dg_setting/semi_20k.py'  # 复用半监督20k迭代训练调度
]  # 结束基础配置列表定义

classes = ('drone',)  # 定义训练和评估只包含无人机这一类别

dataset_type = 'CocoDataset'  # 指定数据集类型为COCO格式

data_root = 'data/'  # 定义默认数据根目录

ir_img_prefix = '/userhome/liqiulu/data/drone_ir_clear_day/00001'  # 指定红外图像所在路径前缀

backend_args = None  # 指定图像加载后端参数为None以使用默认行为

color_space_light = [  # 定义轻量级颜色增强搜索空间
    [dict(type='AutoContrast')],  # 使用自动对比度调整增强
    [dict(type='Equalize')],  # 使用直方图均衡增强
    [dict(type='Color')],  # 调整色彩饱和度增强
    [dict(type='Contrast')],  # 调整对比度增强
    [dict(type='Brightness')],  # 调整亮度增强
    [dict(type='Sharpness')],  # 调整锐度增强
]  # 结束颜色增强空间定义

train_pipeline = [  # 构建训练数据处理流水线
    dict(type='LoadImageFromFile', backend_args=backend_args),  # 从文件加载图像并可指定后端
    dict(type='LoadAnnotations', with_bbox=True),  # 加载带边界框的标注
    dict(type='Resize', scale=(1600, 960), keep_ratio=True),  # 调整图像尺寸并保持长宽比
    dict(  # 随机裁剪操作配置开始
        type='RandomCrop',  # 指定操作类型为随机裁剪
        crop_type='absolute',  # 使用绝对尺寸裁剪方式
        crop_size=(640, 640),  # 指定裁剪窗口大小
        recompute_bbox=True,  # 裁剪后重新计算边界框
        allow_negative_crop=True,  # 允许出现没有目标的裁剪结果
    ),  # 随机裁剪配置结束
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1e-2, 1e-2)),  # 过滤过小的标注框
    dict(type='RandomFlip', prob=0.5),  # 以50%概率进行随机翻转
    dict(type='RandAugment', aug_space=color_space_light, aug_num=1),  # 在轻量增强空间中随机选一种增强
    dict(  # 打包检测输入配置开始
        type='PackDetInputs',  # 指定打包操作类型
        meta_keys=(  # 定义需要保留的元信息键
            'img_id',  # 图像编号
            'img_path',  # 图像路径
            'ori_shape',  # 原始尺寸
            'img_shape',  # 当前尺寸
            'scale_factor',  # 缩放因子
            'flip',  # 是否翻转
            'flip_direction',  # 翻转方向
            'homography_matrix',  # 单应矩阵信息
        ),  # 元信息元组结束
    ),  # 打包配置结束
]  # 训练流水线定义结束

test_pipeline = [  # 构建验证与测试数据处理流水线
    dict(type='LoadImageFromFile', backend_args=backend_args),  # 从文件加载图像
    dict(type='Resize', scale=(1600, 960), keep_ratio=True),  # 调整图像尺寸并保持比例
    dict(type='LoadAnnotations', with_bbox=True),  # 加载标注用于评估
    dict(  # 打包检测输入配置开始
        type='PackDetInputs',  # 指定打包操作类型
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor'),  # 指定保留的元信息键
    ),  # 打包配置结束
]  # 测试流水线定义结束

train_dataloader = dict(  # 定义训练数据加载器
    batch_size=8,  # 设置每批次样本数量
    num_workers=8,  # 设置数据加载线程数
    persistent_workers=True,  # 启用持久化工作线程加速加载
    sampler=dict(type='DefaultSampler', shuffle=True),  # 使用默认采样器并启用打乱
    batch_sampler=dict(type='AspectRatioBatchSampler'),  # 按长宽比分组采样
    dataset=dict(  # 构建训练数据集配置
        type=dataset_type,  # 使用前面定义的COCO数据集类型
        data_root=data_root,  # 指定数据根目录
        metainfo=dict(classes=classes),  # 写入类别元信息
        ann_file='drone_ann/single_clear_day_ir/train.json',  # 指定训练标注文件
        data_prefix=dict(img=ir_img_prefix),  # 设置图像路径前缀
        filter_cfg=dict(filter_empty_gt=True),  # 过滤掉没有目标的图片
        pipeline=train_pipeline,  # 引用训练数据处理流水线
    ),  # 数据集配置结束
)  # 训练数据加载器定义结束

val_dataloader = dict(  # 定义验证数据加载器
    batch_size=1,  # 设置验证批大小为1
    num_workers=8,  # 设置数据加载线程数
    persistent_workers=True,  # 启用持久化工作线程
    drop_last=False,  # 不丢弃最后一个不完整批次
    sampler=dict(type='DefaultSampler', shuffle=False),  # 使用默认采样器且不打乱
    dataset=dict(  # 构建验证数据集配置
        type=dataset_type,  # 指定数据集类型
        data_root=data_root,  # 指定数据根目录
        metainfo=dict(classes=classes),  # 写入类别元信息
        ann_file='drone_ann/single_clear_day_ir/val.json',  # 指定验证标注文件
        data_prefix=dict(img=ir_img_prefix),  # 设置图像路径前缀
        test_mode=True,  # 启用测试模式以跳过某些训练增强
        filter_cfg=dict(filter_empty_gt=True),  # 过滤没有目标的样本
        pipeline=test_pipeline,  # 引用测试流水线
    ),  # 数据集配置结束
)  # 验证数据加载器定义结束

test_dataloader = val_dataloader  # 复用验证数据加载器作为测试加载器

val_evaluator = dict(  # 定义验证评估器
    type='CocoMetric',  # 指定评估器类型为COCO指标
    ann_file=data_root + 'drone_ann/single_clear_day_ir/val.json',  # 指定验证标注路径
    metric='bbox',  # 指定评估指标为边界框
    format_only=False,  # 指定同时输出指标而非仅格式化结果
)  # 验证评估器定义结束

test_evaluator = val_evaluator  # 测试评估器与验证评估器保持一致

# 配置扩散教师与学生联合训练的封装器
detector = _base_.model  # 从基础配置中获取扩散引导检测器结构

detector.data_preprocessor = dict(  # 重设数据预处理模块以适配当前任务
    type='DetDataPreprocessor',  # 指定预处理器类型
    mean=[123.675, 116.28, 103.53],  # 设置图像归一化均值
    std=[58.395, 57.12, 57.375],  # 设置图像归一化标准差
    bgr_to_rgb=True,  # 将输入图像从BGR转换为RGB
    pad_size_divisor=64,  # 将图像填充到64的倍数
)  # 数据预处理模块定义结束

detector.detector.roi_head.bbox_head.num_classes = len(classes)  # 调整ROI头类别数量为无人机类数量

detector.detector.rpn_head.anchor_generator = dict(  # 重设RPN锚框生成器
    type='AnchorGenerator',  # 指定锚框生成器类型
    scales=[2, 4, 8],  # 使用适合无人机小目标的尺度
    ratios=[0.33, 0.5, 1.0, 2.0],  # 配置多种纵横比增强鲁棒性
    strides=[4, 8, 16, 32, 64],  # 对应FPN各层的步长
)  # 锚框生成器配置结束

detector.diff_model.config = 'DG/Ours/drone/diffusion_detector_drone_ir_clear_day.py'  # 指定扩散教师的配置文件路径

detector.diff_model.pretrained_model = 'work_dirs/DD_IR.pth'  # 指定扩散教师的权重路径

model = dict(  # 重建域泛化检测器包装器
    _delete_=True,  # 删除基础配置中原有模型定义
    type='DomainGeneralizationDetector',  # 指定模型类型为域泛化检测器
    detector=detector,  # 注入带教师的检测器结构
    data_preprocessor=detector.data_preprocessor,  # 复用检测器的数据预处理模块
    train_cfg=dict(  # 配置训练阶段超参数
        burn_up_iters=2000,  # 设置蒸馏预热迭代数
        cross_loss_cfg=dict(enable_cross_loss=True, cross_loss_weight=0.5),  # 配置交叉蒸馏损失并设定权重
        feature_loss_cfg=dict(  # 配置特征蒸馏损失
            enable_feature_loss=True,  # 启用特征损失
            feature_loss_type='mse',  # 指定特征损失类型为均方误差
            feature_loss_weight=0.5,  # 设置特征损失权重
        ),  # 特征损失配置结束
        kd_cfg=dict(  # 配置知识蒸馏相关损失
            loss_cls_kd=dict(  # 分类蒸馏损失定义
                type='KnowledgeDistillationKLDivLoss',  # 指定损失类型为KL散度
                class_reduction='sum',  # 指定类别维度归约方式
                T=3,  # 设置蒸馏温度
                loss_weight=1.0,  # 设置分类蒸馏损失权重
            ),  # 分类蒸馏损失配置结束
            loss_reg_kd=dict(type='L1Loss', loss_weight=1.0),  # 配置回归蒸馏损失为L1并设置权重
        ),  # 蒸馏损失配置结束
    ),  # 训练配置结束
)  # 模型定义结束

auto_scale_lr = dict(enable=True, base_batch_size=8)  # 启用自动学习率缩放并设置基准批量大小
