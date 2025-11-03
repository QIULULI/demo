# -*- coding: utf-8 -*-  # 指明编码防止中文注释乱码
"""仿真IR+RGB到真实RGB的双教师扩散引导域适应配置。"""  # 顶部文档字符串描述配置用途

_base_ = [  # 指定继承的基础配置列表
    '../../_base_/models/diffusion_guided_adaptation_faster_rcnn_r101_fpn.py',  # 继承扩散引导学生结构
    '../../_base_/da_setting/semi_20k.py',  # 继承半监督20k训练调度
    '../../_base_/datasets/sim_to_real/semi_drone_ir_rgb_aug.py',  # 继承包含SetSensorTag写入传感器元信息的RGB+IR数据配置
]  # 基础配置列表结束

classes = ('drone',)  # 定义任务类别，仅包含无人机

detector = _base_.model  # 从基础配置中提取域适应检测器结构

detector.data_preprocessor = dict(  # 重设学生模型使用的数据预处理器
    type='DetDataPreprocessor',  # 指定检测任务预处理器
    mean=[123.675, 116.28, 103.53],  # 设置图像归一化均值
    std=[58.395, 57.12, 57.375],  # 设置图像归一化方差
    bgr_to_rgb=True,  # 将输入通道从BGR转换为RGB
    pad_size_divisor=64,  # 将图像填充到64的倍数以匹配FPN
)  # 数据预处理器配置结束

detector.detector.roi_head.bbox_head.num_classes = len(classes)  # 将ROI头类别数量设为无人机类别数

detector.detector.rpn_head.anchor_generator = dict(  # 调整RPN锚框生成器参数
    type='AnchorGenerator',  # 指定锚框生成器类型
    scales=[2, 4, 8],  # 使用小尺度锚框捕获微小目标
    ratios=[0.33, 0.5, 1.0, 2.0],  # 设置多种纵横比提升召回率
    strides=[4, 8, 16, 32, 64],  # 对应FPN层级步长
)  # RPN锚框生成器配置结束

# 组建扩散教师字典，分别载入仿真IR与仿真RGB单模态教师权重
# 若已训练完成双模态互学习教师，可将pretrained_model路径替换为Dual_Diffusion_Teacher权重
# 为保证训练正常启动，请先准备好DD_IR.pth与DD_RGB.pth或等效文件

detector.diff_model = dict(  # 使用字典形式同时声明教师池与主教师标识
    main_teacher='dual_real_rgb',  # 指定默认主教师为真实域双模态教师以便覆盖真实样本
    teachers=[  # 构建扩散教师列表确保每个传感器均有对应权重
        dict(  # 第一名教师：仿真IR扩散检测器
            name='sim_ir',  # 唯一名称用于从教师池中检索模型
            sensor='sim_ir',  # 指定服务的传感器标签对应仿真IR样本
            config='DG/Ours/drone/diffusion_detector_drone_ir_clear_day.py',  # 指向仿真IR教师的模型配置文件
            pretrained_model='work_dirs/diffusion_detector_drone_ir_clear_day/best_coco_bbox_mAP_50_iter_5000.pth',  # 明确仿真IR教师的检查点路径方便权重加载
        ),  # 仿真IR教师配置结束
        dict(  # 第二名教师：仿真RGB扩散检测器
            name='sim_rgb',  # 唯一名称用于仿真RGB教师
            sensor='sim_rgb',  # 指定服务的传感器标签对应仿真RGB样本
            config='DG/Ours/drone/diffusion_detector_drone_rgb_sim.py',  # 指向仿真RGB教师的模型配置文件
            pretrained_model='work_dirs/diffusion_detector_drone_rgb_sim/best_coco_bbox_mAP_50_iter_20000.pth',  # 明确仿真RGB教师的检查点路径确保加载成功
        ),  # 仿真RGB教师配置结束
        dict(  # 第三名教师：真实域双模态扩散检测器
            name='dual_real_rgb',  # 唯一名称用于真实域双模态教师
            sensor='real_rgb',  # 指定服务的传感器标签对应真实RGB样本
            config='DG/Ours/drone/diffusion_detector_drone_dual_ir_rgb.py',  # 指向双模态教师的模型配置文件以提供互补信息
            pretrained_model='work_dirs/Dual_Diffusion_Teacher.pth',  # 明确双模态教师的检查点路径便于权重准备与加载
        ),  # 真实域双模态教师配置结束
    ],  # 扩散教师列表定义完成
)  # 扩散教师字典配置完成

# 如果希望采用互学习后导出的单一双模态教师，请将上方列表替换为
# dict(sensor='sim_rgb', config='DG/Ours/drone/diffusion_detector_drone_dual_ir_rgb.py', pretrained_model='work_dirs/Dual_Diffusion_Teacher.pth')
# 并同步调整sensor标签，以便与SetSensorTag写入的值保持一致

model = dict(  # 构建域适应检测器包装器
    _delete_=True,  # 删除基础配置中的默认模型定义
    type='DomainAdaptationDetector',  # 指定模型类型
    detector=detector,  # 注入包含多教师的检测器结构
    data_preprocessor=dict(  # 配置多分支数据预处理器
        type='MultiBranchDataPreprocessor',  # 指定多分支预处理器类型
        data_preprocessor=detector.data_preprocessor,  # 嵌入学生预处理模块
    ),  # 数据预处理器配置结束
    train_cfg=dict(  # 配置训练阶段超参数
        detector_cfg=dict(type='SemiBaseDiff', burn_up_iters=2000),  # 指定半监督扩散框架与预热步数
        feature_loss_cfg=dict(  # 配置特征蒸馏与交叉一致性损失
            feature_loss_type='mse',  # 主教师特征蒸馏采用MSE
            feature_loss_weight=1.0,  # 主教师特征蒸馏权重
            cross_feature_loss_weight=0.5,  # 交叉教师特征蒸馏权重，可根据实验调节
            cross_consistency_cfg=dict(  # 交叉教师分类与回归一致性损失设置
                cls_weight=0.1,  # 分类一致性损失权重
                reg_weight=0.1,  # 回归一致性损失权重
            ),  # 交叉一致性配置结束
        ),  # 特征损失配置结束
    ),  # 训练配置结束
)  # 模型定义完成

auto_scale_lr = dict(enable=True, base_batch_size=16)  # 启用自动学习率缩放便于不同GPU规模复现
