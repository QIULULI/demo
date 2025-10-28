# # model settings
# model = dict(
#     type='DiffusionDetector',
#     data_preprocessor=dict(
#         type='DetDataPreprocessor',
#         mean=[123.675, 116.28, 103.53],
#         std=[58.395, 57.12, 57.375],
#         bgr_to_rgb=True,
#         pad_size_divisor=64),
#     backbone=dict(
#         type='DIFF',
#         diff_config=dict(aggregation_type="direct_aggregation",
#                          fine_type = 'deep_fusion',
#                          projection_dim=[2048, 2048, 1024, 512],
#                          projection_dim_x4=256,
#                          model_id="/mnt/ssd/lql/Fitness-Generalization-Transferability/stable-diffusion-1-5",
#                          diffusion_mode="inversion",
#                          input_resolution=[512, 512],
#                          prompt="",
#                          negative_prompt="",
#                          guidance_scale=-1,
#                          scheduler_timesteps=[50, 25],
#                          save_timestep=[0],
#                          num_timesteps=1,
#                          idxs_resnet=[[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [
#                              1, 2], [2, 0], [2, 1], [2, 2], [3, 0], [3, 1], [3, 2]],
#                          idxs_ca=[[1, 0], [1, 1], [1, 2], [2, 0], [2, 1], [2, 2], [3, 0], [3, 1], [3, 2]],
#                          s_tmin=10,
#                          s_tmax=250,
#                          do_mask_steps=True,
#                          classes=('bicycle', 'bus', 'car', 'motorcycle',
#                                   'person', 'rider', 'train', 'truck')
#                          )
#     ),
#     neck=dict(
#         type='FPN',
#         in_channels=[256, 512, 1024, 2048],
#         out_channels=256,
#         num_outs=5),
#     rpn_head=dict(
#         type='RPNHead',
#         in_channels=256,
#         feat_channels=256,
#         anchor_generator=dict(
#             type='AnchorGenerator',
#             scales=[8],
#             ratios=[0.5, 1.0, 2.0],
#             strides=[4, 8, 16, 32, 64]),
#         bbox_coder=dict(
#             type='DeltaXYWHBBoxCoder',
#             target_means=[.0, .0, .0, .0],
#             target_stds=[1.0, 1.0, 1.0, 1.0]),
#         loss_cls=dict(
#             type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
#         loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
#     roi_head=dict(
#         type='StandardRoIHead',
#         bbox_roi_extractor=dict(
#             type='SingleRoIExtractor',
#             roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
#             out_channels=256,
#             featmap_strides=[4, 8, 16, 32]),
#         bbox_head=dict(
#             type='Shared2FCBBoxHead',
#             in_channels=256,
#             fc_out_channels=1024,
#             roi_feat_size=7,
#             num_classes=80,
#             bbox_coder=dict(
#                 type='DeltaXYWHBBoxCoder',
#                 target_means=[0., 0., 0., 0.],
#                 target_stds=[0.1, 0.1, 0.2, 0.2]),
#             reg_class_agnostic=False,
#             loss_cls=dict(
#                 type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
#             loss_bbox=dict(type='L1Loss', loss_weight=1.0))),
#     # model training and testing settings
#     train_cfg=dict(
#         rpn=dict(
#             assigner=dict(
#                 type='MaxIoUAssigner',
#                 pos_iou_thr=0.7,
#                 neg_iou_thr=0.3,
#                 min_pos_iou=0.3,
#                 match_low_quality=True,
#                 ignore_iof_thr=-1),
#             sampler=dict(
#                 type='RandomSampler',
#                 num=256,
#                 pos_fraction=0.5,
#                 neg_pos_ub=-1,
#                 add_gt_as_proposals=False),
#             allowed_border=-1,
#             pos_weight=-1,
#             debug=False),
#         rpn_proposal=dict(
#             nms_pre=2000,
#             max_per_img=1000,
#             nms=dict(type='nms', iou_threshold=0.7),
#             min_bbox_size=0),
#         rcnn=dict(
#             assigner=dict(
#                 type='MaxIoUAssigner',
#                 pos_iou_thr=0.5,
#                 neg_iou_thr=0.5,
#                 min_pos_iou=0.5,
#                 match_low_quality=False,
#                 ignore_iof_thr=-1),
#             sampler=dict(
#                 type='RandomSampler',
#                 num=512,
#                 pos_fraction=0.25,
#                 neg_pos_ub=-1,
#                 add_gt_as_proposals=True),
#             pos_weight=-1,
#             debug=False)),
#     test_cfg=dict(
#         rpn=dict(
#             nms_pre=1000,
#             max_per_img=1000,
#             nms=dict(type='nms', iou_threshold=0.7),
#             min_bbox_size=0),
#         rcnn=dict(
#             score_thr=0.05,
#             nms=dict(type='nms', iou_threshold=0.5),
#             max_per_img=100)),
#     auxiliary_branch_cfg = dict(
#             apply_auxiliary_branch = True,
#             loss_cls_kd=dict(type='KnowledgeDistillationKLDivLoss', class_reduction='sum', T=3, loss_weight=1.0),
#             loss_reg_kd=dict(type='L1Loss', loss_weight=1.0),
#         ),
#         # soft-nms is also supported for rcnn testing
#         # e.g., nms=dict(type='soft_nms', iou_threshold=0.5, min_score=0.05)
#     )





# model settings  # 模型总体配置：定义检测器类型、骨干网络、FPN、RPN、ROI 以及训练/测试超参
model = dict(  # 顶层 model 配置字典
    type='DiffusionDetector',  # 使用自定义的扩散特征引导检测器（两阶段架构）
    data_preprocessor=dict(  # 数据预处理模块（对输入图片做归一化、通道顺序等）
        type='DetDataPreprocessor',  # MMDet 通用检测预处理
        mean=[123.675, 116.28, 103.53],  # ImageNet 均值（像素级，0~255）
        std=[58.395, 57.12, 57.375],  # ImageNet 标准差（像素级，0~255）
        bgr_to_rgb=True,  # 读图通常是 BGR，这里转换为 RGB
        pad_size_divisor=64),  # 将尺寸 pad 到 64 的倍数（便于下采样金字塔对齐）
    backbone=dict(  # 骨干网络（此处为 DIFF 扩散特征编码器）
        type='DIFF',  # 自定义 DIFF 编码器（将 SD 的中间特征抽取为多尺度表征）
        diff_config=dict(  # DIFF 的内部配置
            aggregation_type="direct_aggregation",  # 特征聚合方式（direct_aggregation：按 stride 直接聚合）
            fine_type='deep_fusion',  # 细粒度解码器类型（deep_fusion：深度融合细化多尺度）
            projection_dim=[2048, 2048, 1024, 512],  # 各尺度（1/64,1/32,1/16,1/8）投影通道数
            projection_dim_x4=256,  # 1/4 分辨率的隐藏通道（细化头用）
            model_id="/mnt/ssd/lql/Fitness-Generalization-Transferability/stable-diffusion-1-5",  # 本地 Stable Diffusion 权重路径
            diffusion_mode="inversion",  # 扩散工作模式（如 inversion 反演以提取条件特征）
            input_resolution=[512, 512],  # SD UNet 期望的输入分辨率（内部会对特征对齐）
            prompt="",  # 文本提示（为空表示不使用文本条件）
            negative_prompt="",  # 负向提示（为空）
            guidance_scale=-1,  # 引导系数（-1 表示关闭 CFG，引导由结构逻辑决定）
            scheduler_timesteps=[50, 25],  # 采样/调度步（可用于中间时刻特征）
            save_timestep=[0],  # 需要保留的时间步索引（0 代表最终/或指定时刻）
            num_timesteps=1,  # 实际抽取的时间步个数
            idxs_resnet=[[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2], [2, 0], [2, 1], [2, 2], [3, 0], [3, 1], [3, 2]],  # 选择的 UNet ResBlock 索引
            idxs_ca=[[1, 0], [1, 1], [1, 2], [2, 0], [2, 1], [2, 2], [3, 0], [3, 1], [3, 2]],  # 选择的 Cross-Attn 索引
            s_tmin=10,  # 采样最小时间步（过滤极早期噪声）
            s_tmax=250,  # 采样最大时间步（控制可视语义程度）
            do_mask_steps=True,  # 是否在指定时间步做掩码处理（用于对象引导）
            classes=('bicycle', 'bus', 'car', 'motorcycle',  # 训练/计算掩码时的类名映射（与数据集一致）
                     'person', 'rider', 'train', 'truck')
        )
    ),
    neck=dict(  # 颈部网络（特征金字塔）
        type='FPN',  # 标准 FPN
        in_channels=[256, 512, 1024, 2048],  # 来自骨干（或聚合输出）各层输入通道
        out_channels=256,  # 各 FPN 层统一输出通道
        num_outs=5),  # 输出尺度数（一般 P3~P7 共 5 层）
    rpn_head=dict(  # 区域建议网络（RPN）
        type='RPNHead',  # 标准 RPN 头
        in_channels=256,  # 输入通道（与 FPN out_channels 对齐）
        feat_channels=256,  # RPN 内部特征通道
        anchor_generator=dict(  # 锚框生成器
            type='AnchorGenerator',
            scales=[8],  # 基础尺寸缩放（与 stride 结合得到实际 anchor 大小）
            ratios=[0.5, 1.0, 2.0],  # 锚框宽高比集合
            strides=[4, 8, 16, 32, 64]),  # 对应 P3~P7 的步长
        bbox_coder=dict(  # 边框编码器（Δx,Δy,Δw,Δh）
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],  # 目标均值
            target_stds=[1.0, 1.0, 1.0, 1.0]),  # 目标标准差
        loss_cls=dict(  # RPN 分类损失（前景/背景）
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),  # Sigmoid 二分类
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)),  # RPN 回归损失 L1
    roi_head=dict(  # ROI 阶段（第二阶段）
        type='StandardRoIHead',  # 标准两层 FC ROI 头
        bbox_roi_extractor=dict(  # ROI 特征对齐模块
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),  # RoIAlign 输出 7x7
            out_channels=256,  # 输出通道
            featmap_strides=[4, 8, 16, 32]),  # 对应使用的金字塔层步长
        bbox_head=dict(  # 边框分类与回归头
            type='Shared2FCBBoxHead',  # 两层全连接共享头
            in_channels=256,  # 输入通道（与 ROI extractor 对齐）
            fc_out_channels=1024,  # FC 隐藏维度
            roi_feat_size=7,  # 与 RoIAlign 输出一致
            num_classes=80,  # 类别数（如 COCO=80；若自定义需与数据集一致）
            bbox_coder=dict(  # 二阶段边框编码器
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],  # 目标均值
                target_stds=[0.1, 0.1, 0.2, 0.2]),  # 更严格的 std（阶段二回归更精细）
            reg_class_agnostic=False,  # 回归是否与类别无关（False：每类一组回归）
            loss_cls=dict(  # ROI 分类损失
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),  # Softmax 多类交叉熵
            loss_bbox=dict(type='L1Loss', loss_weight=1.0))),  # ROI 回归 L1
    # model training and testing settings  # 训练与测试策略
    train_cfg=dict(  # 训练阶段配置
        rpn=dict(  # RPN 训练策略
            assigner=dict(  # 正负样本匹配器
                type='MaxIoUAssigner',  # 基于 IoU 的最大匹配
                pos_iou_thr=0.7,  # IoU≥0.7 判定为前景
                neg_iou_thr=0.3,  # IoU<0.3 判定为背景
                min_pos_iou=0.3,  # 最小正样本 IoU（低于此不当正样本）
                match_low_quality=True,  # 低质量匹配（保证每个 GT 有匹配）
                ignore_iof_thr=-1),  # 忽略区域阈值（-1 关闭）
            sampler=dict(  # 采样器
                type='RandomSampler',  # 随机正负采样
                num=256,  # 每张图采样 256 个 RPN anchors
                pos_fraction=0.5,  # 正样本比例 50%
                neg_pos_ub=-1,  # 负/正上界（-1 不限制）
                add_gt_as_proposals=False),  # 是否把 GT 作为 proposals 加入
            allowed_border=-1,  # 允许框越界像素（-1 表示不限制）
            pos_weight=-1,  # 正样本权重（-1 走默认）
            debug=False),  # 调试开关
        rpn_proposal=dict(  # 生成 proposals 的后处理
            nms_pre=2000,  # NMS 前保留 top-k
            max_per_img=1000,  # 每图最多 proposals 数
            nms=dict(type='nms', iou_threshold=0.7),  # RPN 的 NMS 阈值
            min_bbox_size=0),  # 过滤过小框
        rcnn=dict(  # ROI（RCNN）阶段训练策略
            assigner=dict(  # 二阶段 IoU 匹配
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,  # IoU≥0.5 为正
                neg_iou_thr=0.5,  # IoU<0.5 为负
                min_pos_iou=0.5,  # 最小正样本 IoU
                match_low_quality=False,  # 二阶段一般不需要低质量匹配
                ignore_iof_thr=-1),  # 忽略阈值
            sampler=dict(  # ROI 采样
                type='RandomSampler',
                num=512,  # 每图采样 512 个 RoIs
                pos_fraction=0.25,  # 正样本 25%
                neg_pos_ub=-1,  # 负正上界
                add_gt_as_proposals=True),  # 将 GT 加入候选（提升 recall）
            pos_weight=-1,  # 正样本权重
            debug=False)),  # 调试
    test_cfg=dict(  # 测试阶段配置
        rpn=dict(
            nms_pre=1000,  # NMS 前保留 top-k
            max_per_img=1000,  # 每图最多 proposals
            nms=dict(type='nms', iou_threshold=0.7),  # RPN NMS 阈值
            min_bbox_size=0),  # 最小框尺寸过滤
        rcnn=dict(
            score_thr=0.05,  # 最低置信度阈值
            nms=dict(type='nms', iou_threshold=0.5),  # ROI NMS 阈值
            max_per_img=100)),  # 每图最多最终检测框数
    auxiliary_branch_cfg = dict(  # 辅助蒸馏分支配置（用“参考引导”与“无参考”做 KD）
        apply_auxiliary_branch = True,  # 是否启用辅助分支（开启则计算 KD 与特征约束）
        loss_cls_kd=dict(type='KnowledgeDistillationKLDivLoss', class_reduction='sum', T=3, loss_weight=1.0),  # 分类蒸馏（KL 散度，温度 T=3）
        loss_reg_kd=dict(type='L1Loss', loss_weight=1.0),  # 回归蒸馏（L1）
    ),
    # soft-nms is also supported for rcnn testing  # 备注：RCNN 测试也可改用 soft-nms
    # e.g., nms=dict(type='soft_nms', iou_threshold=0.5, min_score=0.05)  # 若需要可在 test_cfg.rcnn.nms 处替换
)  # 配置字典结束
