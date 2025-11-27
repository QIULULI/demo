# model settings
model = dict(
    type='DiffusionDetector',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=64),
    backbone=dict(
        type='DIFF',
        diff_config=dict(aggregation_type="direct_aggregation",
                         fine_type = 'deep_fusion',
                         projection_dim=[2048, 2048, 1024, 512],
                         projection_dim_x4=256,
                         model_id="/userhome/liqiulu/code/FGT-stage2/stable-diffusion-1-5",
                         diffusion_mode="inversion",
                         input_resolution=[512, 512],
                         prompt="",
                         negative_prompt="",
                         guidance_scale=-1,
                         scheduler_timesteps=[50, 25],
                         save_timestep=[0],
                         num_timesteps=1,
                         idxs_resnet=[[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [
                             1, 2], [2, 0], [2, 1], [2, 2], [3, 0], [3, 1], [3, 2]],
                         idxs_ca=[[1, 0], [1, 1], [1, 2], [2, 0], [2, 1], [2, 2], [3, 0], [3, 1], [3, 2]],
                         s_tmin=10,
                         s_tmax=250,
                         do_mask_steps=True,
                         classes=('bicycle', 'bus', 'car', 'motorcycle',
                                  'person', 'rider', 'train', 'truck')
                         ),
        enable_ssdc=False,  # 默认关闭SS-DC训练以保持现有行为
        ssdc_cfg=dict(  # 提供SS-DC模块的最小可用默认配置
            enable_ssdc=False,  # 通过配置开关控制是否启用SS-DC流程
            skip_local_loss=False,  # 中文注释：默认不跳过本地SS-DC损失，包装器可按需覆盖以避免重复累加
            said_filter=dict(type='SAIDFilterBank'),  # 使用默认SAID滤波器参数便于快速启用
            coupling_neck=dict(type='SSDCouplingNeck'),  # 使用默认耦合颈部参数便于快速启用
            loss_decouple=dict(  # 解耦损失配置模块化支持超参数调整
                type='LossDecouple',  # 指定使用LossDecouple实现
                idem_weight=1.0,  # 建议默认幂等性权重1.0
                orth_weight=1.0,  # 建议默认正交性权重1.0
                energy_weight=1.0,  # 建议默认能量守恒权重1.0
                loss_weight=1.0  # 总体缩放因子默认1.0
            ),
            loss_couple=dict(  # 耦合损失配置模块化支持超参数调整
                type='LossCouple',  # 指定使用LossCouple实现
                align_weight=1.0,  # 建议默认耦合对齐权重1.0
                ds_weight=1.0,  # 建议默认域特异抑制权重1.0
                ds_margin=0.2,  # 建议默认域特异阈值0.2
                loss_weight=1.0  # 总体缩放因子默认1.0
            )
        )
    ),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
    roi_head=dict(
        type='StandardRoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=dict(
            type='Shared2FCBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=80,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0))),
    # model training and testing settings
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=-1,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_pre=2000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=False,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            pos_weight=-1,
            debug=False)),
    test_cfg=dict(
        rpn=dict(
            nms_pre=1000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=100)),
    auxiliary_branch_cfg = dict(
            apply_auxiliary_branch = True,
            loss_cls_kd=dict(type='KnowledgeDistillationKLDivLoss', class_reduction='sum', T=3, loss_weight=1.0),
            loss_reg_kd=dict(type='L1Loss', loss_weight=1.0),
        ),
        # soft-nms is also supported for rcnn testing
        # e.g., nms=dict(type='soft_nms', iou_threshold=0.5, min_score=0.05)
    )
