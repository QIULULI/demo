_base_ = [  # 指定需要继承的基础配置
    '../../_base_/models/diffusion_guided_generalization_faster_rcnn_r101_fpn.py',  # 继承扩散引导的师生框架
    '../../_base_/dg_setting/semi_20k.py',  # 继承半监督训练调度
    '../../_base_/datasets/domain_generalization_coco/drone_dual_ir_rgb.py',  # 继承双模态数据配置
]  # 基础配置列表定义完毕

classes = ('drone',)  # 定义任务仅包含无人机类别

detector = _base_.model  # 从基础配置中获取师生检测器结构

detector.data_preprocessor = dict(  # 重设数据预处理器保证与当前数据一致
    type='DetDataPreprocessor',  # 指定预处理器类型
    mean=[123.675, 116.28, 103.53],  # 归一化均值
    std=[58.395, 57.12, 57.375],  # 归一化标准差
    bgr_to_rgb=True,  # 转换色彩通道
    pad_size_divisor=64)  # 填充尺寸对齐

detector.detector.roi_head.bbox_head.num_classes = len(classes)  # 设置ROI头类别数

detector.detector.rpn_head.anchor_generator = dict(  # 调整RPN锚框以适应小目标
    type='AnchorGenerator',  # 指定锚框生成器
    scales=[2, 4, 8],  # 小尺度锚框
    ratios=[0.33, 0.5, 1.0, 2.0],  # 丰富纵横比
    strides=[4, 8, 16, 32, 64])  # 对齐FPN步长

detector.diff_model = dict(  # 配置扩散教师信息
    main_teacher='sim_rgb',  # 指定主教师为仿真RGB分支
    teachers=[  # 列出全部扩散教师
        dict(  # 红外教师配置
            name='ir',  # 教师名称
            sensor='sim_ir',  # 指定与数据集标注一致的传感器标签
            config='DG/Ours/drone/diffusion_detector_drone_ir_clear_day.py',  # 教师配置路径
            pretrained_model='/mnt/ssd/lql/Fitness-Generalization-Transferability/work_dirs/diffusion_detector_drone_ir_clear_day/best_coco_bbox_mAP_50_iter_5000.pth'),  # 红外教师权重
        dict(  # 可见光教师配置
            name='rgb',  # 教师名称
            sensor='sim_rgb',  # 指定与数据集标注一致的传感器标签
            config='DG/Ours/drone/diffusion_detector_drone_rgb_sim.py',  # 教师配置路径
            pretrained_model='/mnt/ssd/lql/Fitness-Generalization-Transferability/work_dirs/diffusion_detector_drone_rgb_sim/best_coco_bbox_mAP_50_iter_20000.pth')  # 可见光教师权重
    ])  # 教师列表定义完毕

model = dict(  # 构建域泛化训练包装器
    _delete_=True,  # 删除基础模型定义
    type='DomainGeneralizationDetector',  # 指定顶层模型类型
    detector=detector,  # 注入扩散师生检测器
    data_preprocessor=detector.data_preprocessor,  # 复用预处理器
    train_cfg=dict(  # 配置训练阶段策略
        burn_up_iters=5000,  # 前5000迭代仅更新学生主干
        cross_loss_cfg=dict(  # 交叉蒸馏配置
            enable_cross_loss=True,  # 启用交叉蒸馏
            cross_loss_weight=0.4,  # 默认交叉蒸馏权重
            schedule=[  # 阶段性调度表
                dict(start_iter=0, active_teacher='sim_rgb', cross_loss_weight=0.0),  # 初始阶段仅依赖仿真RGB教师
                dict(start_iter=8000, active_teacher='sim_ir', cross_loss_weight=0.4),  # 进入交叉阶段启用仿真IR教师
                dict(start_iter=14000, active_teacher='sim_rgb', cross_loss_weight=0.5),  # 后期回归仿真RGB教师并加大权重
            ]),  # 调度表定义完毕
        feature_loss_cfg=dict(  # 特征蒸馏配置
            enable_feature_loss=True,  # 启用特征蒸馏
            feature_loss_type='mse',  # 采用均方误差
            feature_loss_weight=0.5),  # 特征蒸馏权重
        kd_cfg=dict(  # ROI蒸馏配置
            loss_cls_kd=dict(  # 分类蒸馏损失
                type='KnowledgeDistillationKLDivLoss',  # KL散度损失
                class_reduction='sum',  # 类别维度求和
                T=3,  # 蒸馏温度
                loss_weight=1.0),  # 分类蒸馏权重
            loss_reg_kd=dict(type='L1Loss', loss_weight=1.0))  # 边框回归蒸馏损失
    ))  # 训练配置结束

auto_scale_lr = dict(enable=True, base_batch_size=16)  # 启用自动学习率缩放
