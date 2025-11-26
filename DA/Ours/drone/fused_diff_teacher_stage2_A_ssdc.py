# -*- coding: utf-8 -*-
# 注意：每行均提供中文注释，方便对照需求与默认值
_base_ = [  # 继承基础模型、训练日程与数据集配置
    '../../_base_/models/diffusion_guided_adaptation_faster_rcnn_r101_fpn.py',  # 基础扩散引导检测器
    '../../_base_/da_setting/semi_20k.py',  # 20000迭代半监督/域自适应日程
    '../../_base_/datasets/sim_to_real/semi_drone_rgb_aug.py'  # 仿真 RGB 到真实 RGB 数据管线
]

detector = _base_.model  # 复制基础 detector 配置以便覆写
detector.data_preprocessor = dict(  # 统一数据预处理
    type='DetDataPreprocessor',  # 常规检测预处理器
    mean=[123.675, 116.28, 103.53],  # Imagenet 均值
    std=[58.395, 57.12, 57.375],  # Imagenet 方差
    bgr_to_rgb=True,  # BGR 转 RGB
    pad_size_divisor=64)  # 保持与 FPN 对齐的填充因子

detector.detector.roi_head.bbox_head.num_classes = 1  # 目标类别仅无人机
detector.detector.enable_ssdc = True  # 启用谱-空解耦模块
detector.detector.use_ds_tokens = True  # 允许注入 DS token
detector.detector.said_cfg = dict(  # SAID 频率掩码配置（可根据实验再调）
    shared_mask=True,  # 所有层共享径向掩码
    mask_mode='soft')  # 默认软掩码便于可导
detector.detector.coupling_cfg = dict(  # 耦合颈部配置
    levels=('P2', 'P3', 'P4', 'P5'),  # 作用的 FPN 层级
    use_ds_tokens=True,  # 再次声明 token 启用
    num_ds_tokens=4)  # DS token 数，默认 4
detector.diff_model.config = 'DG/Ours/drone/fused_diff_teacher_stage1_A_rpn_roi.py'  # Stage-1 教师/扩散配置
detector.diff_model.pretrained_model = 'rgb_fused1111.pth'  # Stage-1 权重

model = dict(  # 最外层包装 DomainAdaptationDetector
    _delete_=True,  # 删除基础同名字段
    type='DomainAdaptationDetector',  # 域自适应封装
    detector=detector,  # 注入学生-教师-扩散联合模型
    data_preprocessor=dict(  # 多分支预处理封装
        type='MultiBranchDataPreprocessor',  # 区分 sup/unsup 分支
        data_preprocessor=detector.data_preprocessor),
    train_cfg=dict(  # 训练阶段控制
        detector_cfg=dict(type='SemiBaseDiff', burn_up_iters=2000),  # 2k 热身仅监督
        ssdc_cfg=dict(  # SS-DC 损失与调度配置
            w_decouple=[(0, 0.1), (6000, 0.5)],  # 谱解耦权重线性升高
            w_couple=[(2000, 0.2), (10000, 0.5)],  # 耦合对齐从热身后开始
            w_di_consistency=0.3,  # DI 一致性常数权重
            consistency_gate=[(0, 0.9), (12000, 0.6)],  # 伪标签 DI 余弦阈值线性下降
            freeze_coupling_iters=2000),  # 前 2k 迭代冻结耦合颈以保持稳定
        feature_loss_cfg=dict(feature_loss_type='mse', feature_loss_weight=1.0))  # 沿用基础特征蒸馏
)
