_base_ = [  # 指定要继承的基础配置列表
    '../../_base_/models/diffusion_guided_adaptation_faster_rcnn_r101_fpn.py',  # 基础模型结构定义
    '../../_base_/da_setting/semi_20k.py',  # 半监督域自适应训练调度
    '../../_base_/datasets/sim_to_real/semi_drone_ir_rgb_aug.py'  # 新增的仿真到真实数据配置
]  # 结束基础配置定义
detector = _base_.model  # 从基础配置拷贝域自适应检测器设置
detector.data_preprocessor = dict(  # 重写数据预处理模块
    type='DetDataPreprocessor',  # 使用常规检测预处理器
    mean=[123.675, 116.28, 103.53],  # 指定均值用于归一化
    std=[58.395, 57.12, 57.375],  # 指定标准差用于归一化
    bgr_to_rgb=True,  # 将 BGR 转换为 RGB 排序
    pad_size_divisor=64)  # 将输入尺寸补齐到 64 的倍数以适配 FPN
detector.detector.roi_head.bbox_head.num_classes = 1  # 将 ROI 头类别数设置为单类无人机
detector.diff_model.config = None #'DG/Ours/drone/diffusion_detector_drone_rgb_sim.py'  # 指向仿真域训练的扩散检测器配置
detector.diff_model.pretrained_model = None #'/mnt/ssd/lql/Fitness-Generalization-Transferability/work_dirs/diffusion_detector_drone_rgb_sim/best_coco_bbox_mAP_50_iter_20000.pth'  # 指向扩散教师的预训练权重
model = dict(  # 重写最外层模型配置
    _delete_=True,  # 删除基础配置的同名字段后重新定义
    type='DomainAdaptationDetector',  # 使用域自适应检测器封装
    detector=detector,  # 注入学生-教师-扩散联合模型
    data_preprocessor=dict(  # 指定多分支数据预处理
        type='MultiBranchDataPreprocessor',  # 使用多分支包装器区分监督与无监督分支
        data_preprocessor=detector.data_preprocessor),  # 共享同一归一化配置
    train_cfg=dict(  # 配置训练相关参数
        detector_cfg=dict(type='SemiBaseDiff', burn_up_iters=_base_.burn_up_iters),  # 设置半监督扩散框架与热身迭代
        feature_loss_cfg=dict(feature_loss_type='mse', feature_loss_weight=1.0))  # 启用 MSE 特征蒸馏并设置权重
)  # 结束模型定义