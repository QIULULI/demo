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
            pretrained_model='/mnt/ssd/lql/Fitness-Generalization-Transferability/work_dirs/diffusion_detector_drone_rgb_sim/best_coco_bbox_mAP_50_iter_20000.pth'),  # 可见光教师权重
        dict(  # 新增真实RGB教师配置
            name='dual_rgb_teacher',  # 为可训练教师命名以便后续引用
            sensor='dual_rgb',  # 将可训练教师的传感器标签改为双RGB以匹配新增数据流水线
            config='DG/Ours/drone/diffusion_detector_drone_rgb_sim.py',  # 复用仿真RGB结构作为初始架构
            pretrained_model='/mnt/ssd/lql/Fitness-Generalization-Transferability/work_dirs/diffusion_detector_drone_rgb_sim/best_coco_bbox_mAP_50_iter_20000.pth',  # 加载已收敛的基础权重作为初始化
            trainable=True,  # 显式标记该教师需参与训练以便后续逻辑开启梯度
            requires_training=True  # 再次强调该分支应被调度器视为待优化对象
            )  # 可训练教师配置结束
    ])  # 教师列表定义完毕

detector.semi_test_cfg = dict(  # 重写半监督测试配置以便互学习后的学生承担推理职责
    predict_on='student',  # 指定推理阶段使用学生分支输出预测结果
    forward_on='student',  # 指定推理阶段的forward调用走学生分支以维持一致性
    extract_feat_on='student')  # 指定特征抽取阶段调用学生分支以便后续蒸馏或可视化

model = dict(  # 构建域泛化训练包装器
    _delete_=True,  # 删除基础模型定义
    type='DomainGeneralizationDetector',  # 指定顶层模型类型
    detector=detector,  # 注入扩散师生检测器
    data_preprocessor=detector.data_preprocessor,  # 复用预处理器
    train_cfg=dict(  # 配置训练阶段策略
        burn_up_iters=2000,  # 前2000迭代仅更新学生主干
        warmup_start_iters=2000,
        warmup_ramp_iters=3000,
        cross_loss_cfg=dict(  # 交叉蒸馏配置
            enable_cross_loss=True,  # 启用交叉蒸馏
            cross_loss_weight=0.4,  # 默认交叉蒸馏权重
            cross_feature_loss_weight=0.3,  # 设置交叉特征蒸馏损失的相对权重便于平衡特征对齐
            cls_consistency_weight=0.1,  # 设置分类一致性正则的权重帮助约束学生教师分类输出
            reg_consistency_weight=0.1,  # 设置边框回归一致性正则权重以稳定定位
            cross_roi_kd_weight=0.2,  # 设置交叉ROI级蒸馏的额外权重兼容需要该项的训练逻辑
            schedule=[  # 阶段性调度表
                dict(  # 阶段一配置
                    start_iter=0,  # 迭代0开始进入阶段一
                    active_teacher='sim_rgb',  # 阶段一使用仿真RGB教师
                    cross_loss_weight=0.0,  # 阶段一关闭交叉蒸馏避免扰动初始学生
                    trainable_teacher_loss_weight=0.0),  # 阶段一禁用可训练教师损失避免无梯度阶段浪费计算
                dict(  # 阶段二配置
                    start_iter=5000,  # 迭代5000开始进入阶段二
                    active_teacher='sim_ir',  # 阶段二切换仿真IR教师以提供跨模态信息
                    cross_loss_weight=0.4,  # 阶段二恢复交叉蒸馏权重以进行互学习
                    trainable_teacher_loss_weight=0.8),  # 阶段二为可训练教师分支提供较高损失权重促进收敛
                dict(  # 阶段三配置
                    start_iter=10000,  # 迭代10000开始进入阶段三
                    active_teacher='sim_rgb',  # 阶段三回归仿真RGB教师巩固性能
                    cross_loss_weight=0.5,  # 阶段三进一步提升交叉蒸馏强度
                    trainable_teacher_loss_weight=1.0),  # 阶段三将可训练教师损失权重恢复至基准实现充分训练
            ]),  # 调度表定义完毕
        feature_loss_cfg=dict(  # 特征蒸馏配置
            enable_feature_loss=True,  # 启用特征蒸馏
            feature_loss_type='mse',  # 采用均方误差
            feature_loss_weight=0.5,  # 特征蒸馏权重
            cross_feature_loss_weight=0.3,  # 同步定义交叉特征蒸馏权重以兼容旧版读取逻辑
            cross_consistency_cfg=dict(  # 兼容旧字段的交叉一致性配置
                cls_weight=0.1,  # 旧字段中的分类一致性权重与新配置保持一致
                reg_weight=0.1  # 旧字段中的回归一致性权重与新配置保持一致
            )),  # 特征蒸馏配置结束
        kd_cfg=dict(  # ROI蒸馏配置
            loss_cls_kd=dict(  # 分类蒸馏损失
                type='KnowledgeDistillationKLDivLoss',  # KL散度损失
                class_reduction='sum',  # 类别维度求和
                T=3,  # 蒸馏温度
                loss_weight=1.0),  # 分类蒸馏权重
            loss_reg_kd=dict(type='L1Loss', loss_weight=1.0))  # 边框回归蒸馏损失
    ))  # 训练配置结束

auto_scale_lr = dict(enable=True, base_batch_size=16)  # 启用自动学习率缩放
