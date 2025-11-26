# DA/Ours/drone/fused_diff_teacher_stage2_A_ssdc.py
_base_ = [  # 继承基础配置列表
    '../../_base_/models/diffusion_guided_adaptation_faster_rcnn_r101_fpn.py',  # 基础扩散-检测模型结构
    '../../_base_/da_setting/semi_20k.py',  # 2 万迭代半监督/域自适应训练日程
    '../../_base_/datasets/sim_to_real/semi_drone_rgb_aug.py',  # 仿真 RGB → 真实 RGB 数据集定义
]  # 结束基础配置

detector = _base_.model  # 从基础配置拷贝域自适应检测器
detector.data_preprocessor = dict(  # 覆盖数据预处理模块
    type='DetDataPreprocessor',  # 常规检测预处理器
    mean=[123.675, 116.28, 103.53],  # 归一化均值
    std=[58.395, 57.12, 57.375],  # 归一化方差
    bgr_to_rgb=True,  # BGR 转 RGB
    pad_size_divisor=64)  # 输入尺寸补齐倍数

detector.detector.roi_head.bbox_head.num_classes = 1  # 单类别无人机检测
detector.detector.enable_ssdc = True  # 启用谱-空解耦 SS-DC 模块
detector.detector.use_ds_tokens = True  # 开启 DS token 注入
detector.detector.num_ds_tokens = 4  # DS token 数量，建议 4~8 之间
detector.detector.init_cfg = dict(  # 学生模型初始化
    type='Pretrained',  # 使用预训练权重
    checkpoint='best_coco_bbox_mAP_50_iter_20000.pth')  # Stage-1 学生权重路径
detector.diff_model.config = 'DG/Ours/drone/fused_diff_teacher_stage1_A_rpn_roi.py'  # 扩散教师配置（沿用 Stage-1）
detector.diff_model.pretrained_model = 'rgb_fused1111.pth'  # 扩散教师权重

model = dict(  # 最外层模型封装
    _delete_=True,  # 删除并重写基础字段
    type='DomainAdaptationDetector',  # 域自适应检测封装
    detector=detector,  # 注入学生-教师-扩散联合模型
    data_preprocessor=dict(  # 多分支数据预处理
        type='MultiBranchDataPreprocessor',  # 区分监督/无监督分支
        data_preprocessor=detector.data_preprocessor),  # 复用相同归一化
    train_cfg=dict(  # 训练阶段设置
        detector_cfg=dict(type='SemiBaseDiff', burn_up_iters=2000),  # 半监督扩散框架与 2k 热身
        ssdc_cfg=dict(  # SS-DC 专用超参与调度
            w_decouple=[(0, 0.1), (6000, 0.5)],  # 频域解耦损失权重线性爬升
            w_couple=[(2000, 0.2), (10000, 0.5)],  # 谱-空耦合对齐权重热身后启动
            w_di_consistency=0.3,  # 学生/教师 DI 一致性恒定权重
            consistency_gate=[(0, 0.9), (12000, 0.6)],  # 伪标签 DI 余弦阈值由严到松
            freeze_coupling_until=2000),  # 耦合颈在热身前冻结/旁路
        feature_loss_cfg=dict(  # 兼容原有特征蒸馏开关
            feature_loss_type='mse',  # 蒸馏类型
            feature_loss_weight=1.0)))  # 蒸馏权重
