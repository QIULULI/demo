# Stage-2 UDA配置，融合扩散教师并启用SS-DC模块
# 引用Stage-1基础配置（扩散引导UDA、20k半监督日程与Sim→Real无人机数据集），相对路径需从当前文件起算
import os  # 引入os模块以便通过环境变量灵活切换扩散教师路径
import copy  # 引入deepcopy工具以在合并模板时避免原地修改导致的引用污染

_base_ = [  # 定义基础配置列表以便mmengine按顺序合并
    '../../../../DA/_base_/models/diffusion_guided_adaptation_faster_rcnn_r101_fpn.py',  # 半监督扩散基础检测器（保留唯一含模型的基线）
    '../../../../DA/_base_/da_setting/semi_20k.py',  # 20k迭代的半监督训练调度
    '../../../../DA/_base_/datasets/sim_to_real/semi_drone_rgb_aug.py'  # Sim→Real无人机数据集配置
]

stage1_diff_teacher_config = os.environ.get(  # 读取Stage-1扩散教师配置路径
    'STAGE1_DIFF_TEACHER_CONFIG',  # 环境变量名称
    'DG/Ours/drone/fused_diff_teacher_stage1_A_rpn_roi.py'  # 默认路径
)  # 获取配置路径结束
stage1_diff_teacher_ckpt = os.environ.get(  # 读取Stage-1扩散教师权重路径
    'STAGE1_DIFF_TEACHER_CKPT',  # 环境变量名称
    '/userhome/liqiulu/code/FGT-stage2/rgb_fused1111.pth'  # 默认权重文件
)  # 获取权重路径结束

classes = ('drone',)  # 明确声明单类别元组供Diffusion骨干使用

ssdc_schedule = dict(  # 定义SS-DC损失调度以便前向与训练阶段复用
    w_decouple=[(0, 0.1), (6000, 0.5)],  # 解耦损失权重线性升档
    w_couple=[(0, 0.0), (2000, 0.2), (10000, 0.5)],  # 耦合损失权重分段提升
    w_di_consistency=0.3,  # 域不变一致性损失固定权重
    consistency_gate=[(0, 0.9), (12000, 0.6)],  # 一致性阈值逐步放宽
    burn_in_iters=2000  # 耦合损失预热步数
)  # 调度定义结束

diffusion_detector_template = dict(  # 手动移植DiffusionDetector模板避免重复_base_合并
    type='DiffusionDetector',  # 指定主干检测器类型
    data_preprocessor=dict(  # 定义标准检测数据预处理器
        type='DetDataPreprocessor',  # 使用检测任务预处理基类
        mean=[123.675, 116.28, 103.53],  # 设置像素归一化均值
        std=[58.395, 57.12, 57.375],  # 设置像素归一化标准差
        bgr_to_rgb=True,  # 将输入由BGR转换为RGB
        pad_size_divisor=64  # 将图像填充到64的倍数
    ),
    backbone=dict(  # 配置扩散骨干网络
        type='DIFF',  # 使用自定义DIFF骨干
        diff_config=dict(  # 传递扩散骨干的详细参数
            aggregation_type='direct_aggregation',  # 聚合方式设为直接聚合
            fine_type='deep_fusion',  # 微调策略使用深度融合
            projection_dim=[2048, 2048, 1024, 512],  # 各层投影维度设置
            projection_dim_x4=256,  # 第四层投影维度
            model_id='/userhome/liqiulu/code/FGT-stage2/stable-diffusion-1-5',  # 指定稳定扩散模型路径
            diffusion_mode='inversion',  # 扩散模式设为反演
            input_resolution=[512, 512],  # 输入分辨率设定
            prompt='',  # 默认提示词留空
            negative_prompt='',  # 默认负向提示词留空
            guidance_scale=-1,  # 禁用指导比例以走纯反演流程
            scheduler_timesteps=[50, 25],  # 调度时间步设置
            save_timestep=[0],  # 需要保存的时间步索引
            num_timesteps=1,  # 扩散步数设为1以保持推理高效
            idxs_resnet=[[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [
                1, 2], [2, 0], [2, 1], [2, 2], [3, 0], [3, 1], [3, 2]],  # ResNet层索引映射
            idxs_ca=[[1, 0], [1, 1], [1, 2], [2, 0], [2, 1], [2, 2], [3, 0], [3, 1], [3, 2]],  # 交叉注意力层索引映射
            s_tmin=10,  # 反演时间步下界
            s_tmax=250,  # 反演时间步上界
            do_mask_steps=True,  # 启用掩码步骤支持局部编辑
            classes=('bicycle', 'bus', 'car', 'motorcycle',  # 占位类别列表后续会覆盖
                     'person', 'rider', 'train', 'truck')
        )
    ),
    neck=dict(  # FPN颈部配置
        type='FPN',  # 使用特征金字塔结构
        in_channels=[256, 512, 1024, 2048],  # 输入通道列表
        out_channels=256,  # 输出通道数
        num_outs=5  # 输出特征层数
    ),
    rpn_head=dict(  # RPN头配置
        type='RPNHead',  # 使用标准RPN头
        in_channels=256,  # 输入通道与FPN输出一致
        feat_channels=256,  # 特征通道数
        anchor_generator=dict(  # 锚框生成器设置
            type='AnchorGenerator',  # 生成器类型
            scales=[8],  # 锚框尺度列表
            ratios=[0.5, 1.0, 2.0],  # 锚框纵横比列表
            strides=[4, 8, 16, 32, 64]  # 对应特征层的步长
        ),
        bbox_coder=dict(  # 边框编码器设置
            type='DeltaXYWHBBoxCoder',  # 使用Delta编码
            target_means=[.0, .0, .0, .0],  # 均值为0
            target_stds=[1.0, 1.0, 1.0, 1.0]  # 方差为1
        ),
        loss_cls=dict(  # 分类损失设置
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0  # 采用Sigmoid交叉熵
        ),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)  # 边框回归使用L1损失
    ),
    roi_head=dict(  # ROI头配置
        type='StandardRoIHead',  # 使用标准ROIHead
        bbox_roi_extractor=dict(  # ROI特征抽取器
            type='SingleRoIExtractor',  # 使用单ROI抽取器
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),  # 采用RoIAlign
            out_channels=256,  # 输出通道数
            featmap_strides=[4, 8, 16, 32]  # 特征层步长
        ),
        bbox_head=dict(  # 边框头配置
            type='Shared2FCBBoxHead',  # 使用共享双全连接头
            in_channels=256,  # 输入通道
            fc_out_channels=1024,  # 全连接层输出通道
            roi_feat_size=7,  # ROI特征尺寸
            num_classes=80,  # 默认类别数占位后续覆盖
            bbox_coder=dict(  # 边框编码器
                type='DeltaXYWHBBoxCoder',  # 使用Delta编码
                target_means=[0., 0., 0., 0.],  # 编码均值
                target_stds=[0.1, 0.1, 0.2, 0.2]  # 编码标准差
            ),
            reg_class_agnostic=False,  # 回归不区分类别
            loss_cls=dict(  # 分类损失
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0  # 采用Softmax交叉熵
            ),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0)  # 回归损失使用L1
        )
    ),
    train_cfg=dict(  # 训练阶段配置
        rpn=dict(  # RPN训练配置
            assigner=dict(  # 分配器设置
                type='MaxIoUAssigner',  # 使用最大IoU分配
                pos_iou_thr=0.7,  # 正样本阈值
                neg_iou_thr=0.3,  # 负样本阈值
                min_pos_iou=0.3,  # 最小正样本IoU
                match_low_quality=True,  # 允许低质量匹配
                ignore_iof_thr=-1  # 忽略阈值
            ),
            sampler=dict(  # 采样器设置
                type='RandomSampler',  # 随机采样正负样本
                num=256,  # 每批采样数量
                pos_fraction=0.5,  # 正样本比例
                neg_pos_ub=-1,  # 负样本上限
                add_gt_as_proposals=False  # 不将GT作为建议框
            ),
            allowed_border=-1,  # 允许越界像素
            pos_weight=-1,  # 正样本权重默认
            debug=False  # 关闭调试
        ),
        rpn_proposal=dict(  # RPN建议框配置
            nms_pre=2000,  # NMS前保留建议框数量
            max_per_img=1000,  # 每张图最大建议框数
            nms=dict(type='nms', iou_threshold=0.7),  # NMS参数
            min_bbox_size=0  # 最小边框尺寸
        ),
        rcnn=dict(  # RCNN阶段配置
            assigner=dict(  # RCNN分配器
                type='MaxIoUAssigner',  # 使用最大IoU分配
                pos_iou_thr=0.5,  # 正样本阈值
                neg_iou_thr=0.5,  # 负样本阈值
                min_pos_iou=0.5,  # 最小正样本IoU
                match_low_quality=False,  # 不匹配低质量样本
                ignore_iof_thr=-1  # 忽略阈值
            ),
            sampler=dict(  # RCNN采样器
                type='RandomSampler',  # 随机采样
                num=512,  # 每批次采样数量
                pos_fraction=0.25,  # 正样本比例
                neg_pos_ub=-1,  # 负样本上限
                add_gt_as_proposals=True  # 将GT加入建议框
            ),
            pos_weight=-1,  # 正样本权重默认
            debug=False  # 关闭调试
        )
    ),
    test_cfg=dict(  # 测试阶段配置
        rpn=dict(  # RPN测试配置
            nms_pre=1000,  # NMS前保留框数
            max_per_img=1000,  # 每张图最大建议框数
            nms=dict(type='nms', iou_threshold=0.7),  # NMS参数
            min_bbox_size=0  # 最小边框尺寸
        ),
        rcnn=dict(  # RCNN测试配置
            score_thr=0.05,  # 置信度阈值
            nms=dict(type='nms', iou_threshold=0.5),  # NMS参数
            max_per_img=100  # 每张图最大检测数
        )
    ),
    auxiliary_branch_cfg=dict(  # 辅助分支配置
        apply_auxiliary_branch=True,  # 启用辅助分支
        loss_cls_kd=dict(type='KnowledgeDistillationKLDivLoss', class_reduction='sum', T=3, loss_weight=1.0),  # 分类蒸馏损失
        loss_reg_kd=dict(type='L1Loss', loss_weight=1.0)  # 回归蒸馏损失
    )
)  # 模板定义结束

diffusion_detector = copy.deepcopy(diffusion_detector_template)  # 深拷贝模板以便安全覆写无人机专属配置
diffusion_detector['roi_head']['bbox_head']['num_classes'] = len(classes)  # 调整类别数为无人机单类任务
diffusion_detector['backbone']['diff_config']['classes'] = classes  # 将类别元组写入扩散骨干以生成正确语义嵌入
diffusion_detector['init_cfg'] = dict(  # 注入学生初始化权重
    type='Pretrained',  # 使用预训练权重
    checkpoint='/userhome/liqiulu/code/FGT-stage2/best_coco_bbox_mAP_50_iter_20000.pth'  # 指向Stage-1学生权重
)  # 初始化配置结束
diffusion_detector['enable_ssdc'] = True  # 开启DiffusionDetector内部SS-DC支持
diffusion_detector['ssdc_cfg'] = dict(  # 写入SS-DC模块配置
    enable_ssdc=True,  # 明确开启以确保开关生效
    said_filter=dict(type='SAIDFilterBank'),  # 指定SAID滤波器族
    coupling_neck=dict(  # 耦合颈配置
        type='SSDCouplingNeck',  # 选择SSDC耦合颈实现
        use_ds_tokens=True,  # 启用域特异token保留跨域信息
        in_channels=256,  # 输入通道与FPN输出对齐
        levels=('P2', 'P3', 'P4', 'P5')  # 覆盖FPN多层特征
    ),
    loss_decouple=dict(type='LossDecouple', loss_weight=1.0),  # 解耦损失定义
    loss_couple=dict(type='LossCouple', loss_weight=1.0),  # 耦合损失定义
    w_decouple=ssdc_schedule['w_decouple'],  # 绑定解耦权重调度
    w_couple=ssdc_schedule['w_couple'],  # 绑定耦合权重调度
    w_di_consistency=ssdc_schedule['w_di_consistency'],  # 绑定域不变一致性权重
    consistency_gate=ssdc_schedule['consistency_gate'],  # 绑定一致性阈值调度
    burn_in_iters=ssdc_schedule['burn_in_iters']  # 对齐预热步数
)  # SSDC配置结束

semi_base_diff_detector = dict(  # 构建包含扩散教师的半监督检测器
    type='SemiBaseDiffDetector',  # 使用半监督扩散检测器封装学生与教师
    detector=diffusion_detector,  # 注入经过无人机特化的DiffusionDetector主体
    diff_model=dict(  # 配置主扩散教师来源
        config=stage1_diff_teacher_config,  # 指定Stage-1扩散教师配置文件路径
        pretrained_model=stage1_diff_teacher_ckpt  # 指定Stage-1扩散教师权重路径
    ),
    data_preprocessor=dict(  # 为学生与教师配置多分支预处理器
        type='MultiBranchDataPreprocessor',  # 采用多分支预处理以分离有监督与无监督数据
        data_preprocessor=diffusion_detector['data_preprocessor']  # 复用DiffusionDetector的标准预处理配置
    ),
    semi_train_cfg=dict(  # 保留基础半监督训练超参
        student_pretrained=None,  # 学生额外预训练权重默认空
        freeze_teacher=True,  # 默认冻结主教师参数
        sup_weight=1.0,  # 有监督损失权重
        unsup_weight=1.0,  # 无监督损失权重
        cls_pseudo_thr=0.5,  # 分类伪标签阈值
        min_pseudo_bbox_wh=(1e-2, 1e-2)  # 最小伪标签宽高
    ),
    semi_test_cfg=dict(predict_on='teacher')  # 推理阶段默认使用教师预测
)  # 半监督扩散检测器定义结束

model = dict(  # 通过层级覆盖更新模型结构
    _delete_=True,  # 顶层删除基础模型配置以避免遗留键混入
    type='DomainAdaptationDetector',  # 将Stage-2模型包装为域自适应检测器
    detector=semi_base_diff_detector,  # 注入包含扩散教师与SS-DC配置的半监督检测器
    data_preprocessor=dict(  # 维持多分支预处理器以兼容UDA流程
        type='MultiBranchDataPreprocessor',  # 预处理器类型
        data_preprocessor=diffusion_detector['data_preprocessor']  # 直接复用DiffusionDetector的预处理配置
    ),
    train_cfg=dict(  # 训练阶段配置
        detector_cfg=dict(type='SemiBaseDiff', burn_up_iters=2000),  # 半监督扩散训练器并设置预热步数
        ssdc_cfg=dict(  # 训练阶段的SS-DC调度
            w_decouple=ssdc_schedule['w_decouple'],  # 解耦损失调度
            w_couple=ssdc_schedule['w_couple'],  # 耦合损失调度
            w_di_consistency=ssdc_schedule['w_di_consistency'],  # 域不变一致性权重
            consistency_gate=ssdc_schedule['consistency_gate'],  # 一致性阈值调度
            burn_in_iters=ssdc_schedule['burn_in_iters']  # 耦合预热步数
        ),
        feature_loss_cfg=dict(  # 特征蒸馏损失配置
            feature_loss_type='mse',  # 主教师特征损失类型
            feature_loss_weight=1.0,  # 主教师特征损失权重
            cross_feature_loss_weight=0.0,  # 交叉特征蒸馏权重
            cross_consistency_cfg=dict(  # 交叉一致性子配置
                cls_weight=0.0,  # 分类一致性权重
                reg_weight=0.0  # 回归一致性权重
            )
        )
    )
)  # 模型配置结束

default_hooks = dict(  # 追加默认钩子配置以在Stage-2训练中启用SSDC监控
    _delete_=False,  # 保留基础钩子并补充自定义钩子
    ssdc_monitor=dict(  # 注册SSDC监控钩子
        type='SSDCMonitorHook',  # 钩子类型
        interval=100,  # 日志打印间隔
        vis_interval=1000,  # 可视化保存间隔
        max_vis_samples=1  # 每次可视化样本数
    )
)  # 钩子配置结束

# 小型自检代码（仅导入与前向张量）
if __name__ == '__main__':  # 当作为脚本运行时执行自检
    from mmengine import Config  # 导入配置解析器
    cfg = Config.fromfile(__file__)  # 载入当前配置文件
    print('cfg.model.type:', cfg.model['type'])  # 打印模型类型确认解析成功
    print('cfg.model.detector keys:', cfg.model.detector.keys())  # 打印检测器键集合确认未混入基础键
    print('cfg.model.detector.detector.enable_ssdc:', cfg.model.detector.detector.enable_ssdc)  # 检查DiffusionDetector层级SS-DC开关
    print('cfg.model.detector.diff_model.config:', cfg.model.detector.diff_model.config)  # 输出扩散教师配置路径
    print('cfg.model.detector.detector.init_cfg:', cfg.model.detector.detector.init_cfg)  # 打印初始化配置确认权重设置
