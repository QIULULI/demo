_base_ = [  # 指定继承的基础配置列表
    '../../_base_/models/diffusion_guided_adaptation_faster_rcnn_r101_fpn.py',  # 复用扩散引导域适应检测器结构
    '../../_base_/da_setting/semi_20k.py',  # 复用半监督20k迭代训练调度
    '../../_base_/datasets/sim_to_real/semi_drone_rgb_aug.py',  # 引入无人机仿真到真实的数据增强配置
]  # 结束基础配置列表

classes = ('drone',)  # 定义任务只包含无人机类别

detector = _base_.model  # 从基础配置中提取域适应检测器结构

detector.data_preprocessor = dict(  # 重设学生模型的数据预处理模块
    type='DetDataPreprocessor',  # 指定检测数据预处理器类型
    mean=[123.675, 116.28, 103.53],  # 设置归一化均值
    std=[58.395, 57.12, 57.375],  # 设置归一化标准差
    bgr_to_rgb=True,  # 将输入图像从BGR转换为RGB
    pad_size_divisor=64,  # 将图像填充到64的倍数
)  # 数据预处理模块定义结束

detector.detector.roi_head.bbox_head.num_classes = len(classes)  # 将ROI头类别数设置为无人机类别数量

detector.detector.rpn_head.anchor_generator = dict(  # 调整RPN锚框生成器超参数
    type='AnchorGenerator',  # 指定锚框生成器类型
    scales=[2, 4, 8],  # 使用小尺度更好匹配无人机尺寸
    ratios=[0.33, 0.5, 1.0, 2.0],  # 设定多种纵横比增强适应性
    strides=[4, 8, 16, 32, 64],  # 对应FPN各层的步长
)  # 锚框生成器配置结束

# 扩散教师模型在红外数据上完成训练
detector.diff_model.config = 'DG/Ours/drone/diffusion_detector_drone_ir_clear_day.py'  # 指定扩散教师配置路径

detector.diff_model.pretrained_model = 'work_dirs/DD_IR.pth'  # 指定扩散教师权重路径

model = dict(  # 重建域适应检测器包装器
    _delete_=True,  # 删除基础配置中的默认模型定义
    type='DomainAdaptationDetector',  # 指定模型类型为域适应检测器
    detector=detector,  # 注入包含教师的检测器结构
    data_preprocessor=dict(  # 构建多分支数据预处理器
        type='MultiBranchDataPreprocessor',  # 指定多分支预处理器类型
        data_preprocessor=detector.data_preprocessor,  # 嵌入检测器的单分支预处理模块
    ),  # 多分支预处理器配置结束
    train_cfg=dict(  # 配置训练阶段超参数
        detector_cfg=dict(type='SemiBaseDiff', burn_up_iters=2000),  # 指定半监督扩散训练流程与预热步数
        feature_loss_cfg=dict(feature_loss_type='mse', feature_loss_weight=1.0),  # 指定特征蒸馏损失类型与权重
    ),  # 训练配置结束
)  # 模型定义结束

auto_scale_lr = dict(enable=True, base_batch_size=8)  # 启用自动学习率缩放并设置基准批量大小
