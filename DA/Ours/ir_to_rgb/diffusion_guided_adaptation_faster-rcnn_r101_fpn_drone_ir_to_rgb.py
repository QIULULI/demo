_base_ = [  # 指定基础配置列表
    '../../_base_/models/diffusion_guided_adaptation_faster_rcnn_r101_fpn.py',  # 继承扩散引导域适应检测器结构
    '../../_base_/da_setting/semi_20k.py',  # 继承半监督20k迭代训练调度
    '../../_base_/datasets/ir_to_rgb/semi_drone_ir_to_rgb.py',  # 继承刚刚定义的IR→RGB数据配置
]  # 结束基础配置列表
# ----------------------------------------------------------------------------------------------------
detector = _base_.model  # 读取基础配置中的模型定义
# ----------------------------------------------------------------------------------------------------
classes = ('drone',)  # 定义数据集类别为“drone”
# ----------------------------------------------------------------------------------------------------
detector.data_preprocessor = dict(  # 重设数据预处理模块
    type='DetDataPreprocessor',  # 指定检测数据预处理器类型
    mean=[123.675, 116.28, 103.53],  # 设置均值用于归一化
    std=[58.395, 57.12, 57.375],  # 设置标准差用于归一化
    bgr_to_rgb=True,  # 将输入从BGR转换为RGB
    pad_size_divisor=64)  # 设置填充对齐到64
# ----------------------------------------------------------------------------------------------------
detector.detector.roi_head.bbox_head.num_classes = len(classes) # 设置ROI头的边界框头类别数
# ----------------------------------------------------------------------------------------------------
detector.detector.rpn_head.anchor_generator = dict(  # 调整学生RPN锚框生成器
    type='AnchorGenerator',  # 指定生成器类型
    scales=[2, 4, 8],  # 使用更小尺度捕获微小目标
    ratios=[0.33, 0.5, 1.0, 2.0],  # 设定多种纵横比覆盖细长目标
    strides=[4, 8, 16, 32, 64])  # 指定与FPN各层对应的步长列表
# ----------------------------------------------------------------------------------------------------
# Diffusion teacher trained on IR data.
detector.diff_model.config = None#'DG/Ours/drone/diffusion_detector_drone_ir_clear_day.py'
detector.diff_model.pretrained_model = None#'/userhome/liqiulu/code/Fitness-Generalization-Transferability/work_dirs/resultpth/ir_rgb_improve.pth'
# ----------------------------------------------------------------------------------------------------
model = dict(  # 重建域适应包装器
    _delete_=True,  # 删除基础配置中的默认定义
    type='DomainAdaptationDetector',  # 指定域适应检测器类型
    detector=detector,  # 注入刚刚修改的学生与教师结构
    data_preprocessor=dict(  # 配置多分支数据预处理器
        type='MultiBranchDataPreprocessor',  # 指定多分支预处理器
        data_preprocessor=detector.data_preprocessor),  # 嵌入检测预处理器
    train_cfg=dict(  # 设置训练阶段参数
        detector_cfg=dict(type='SemiBaseDiff', burn_up_iters=_base_.burn_up_iters),  # 指定半监督扩散训练流程和预热步数
        feature_loss_cfg=dict(feature_loss_type='mse', feature_loss_weight=1.0)),  # 设置特征对齐损失类型与权重
)  # 结束模型定义
# ----------------------------------------------------------------------------------------------------
# Match the effective batch size (8) from the IR training recipe for LR scaling
# auto_scale_lr = dict(enable=True, base_batch_size=8)