# -*- coding: utf-8 -*-  # 指定文件编码确保中文注释在不同平台下保持一致
"""无人机仿真RGB学生与冻结IR教师的第一阶段扩散融合训练配置。"""  # 顶部文档字符串简述配置功能

from copy import deepcopy  # 中文注释：导入deepcopy以便在不共享引用的情况下复用基础模型配置

_base_ = [  # 中文注释：列出需要继承的基础配置文件
    '../../_base_/models/faster-rcnn_diff_fpn.py',  # 中文注释：复用两阶段DIFF检测器结构定义
    '../../_base_/dg_setting/dg_20k.py',  # 中文注释：继承20k迭代训练调度与默认钩子设置
]  # 中文注释：基础配置列表结束

classes = ('drone',)  # 中文注释：定义单类别任务仅包含无人机目标
dataset_type = 'CocoDataset'  # 中文注释：指定数据集格式遵循COCO标注
data_root = 'data/'  # 中文注释：设定数据根目录与项目默认软链接保持一致
rgb_img_prefix = 'sim_drone_rgb/Town01_Opt/carla_data/'  # 中文注释：指向仿真RGB图像所在的相对路径
backend_args = None  # 中文注释：使用默认文件后端读取图像数据

def _apply_drone_specialization(detector_cfg):  # 中文注释：封装无人机场景下的通用模型微调逻辑
    detector_cfg['roi_head']['bbox_head']['num_classes'] = len(classes)  # 中文注释：将ROI分类头类别数替换为无人机数量
    detector_cfg['backbone']['diff_config']['classes'] = classes  # 中文注释：将类别信息写入扩散骨干以生成正确的语义嵌入
    detector_cfg['rpn_head']['anchor_generator'] = dict(  # 中文注释：重写锚框生成器以适配微小无人机目标
        type='AnchorGenerator',  # 中文注释：指定生成器类型为标准AnchorGenerator
        scales=[2, 4, 8],  # 中文注释：使用更小的基础尺度提升小目标召回
        ratios=[0.33, 0.5, 1.0, 2.0],  # 中文注释：扩展纵横比覆盖瘦长与宽扁无人机外形
        strides=[4, 8, 16, 32, 64],  # 中文注释：与FPN层级保持一致的步长定义
    )  # 中文注释：锚框配置结束
    return detector_cfg  # 中文注释：返回修改后的检测器配置供调用方继续使用

det_data_preprocessor = dict(  # 中文注释：定义与仿真RGB模型一致的数据预处理器
    type='DetDataPreprocessor',  # 中文注释：指定检测任务通用预处理模块
    mean=[123.675, 116.28, 103.53],  # 中文注释：采用ImageNet均值进行像素归一化
    std=[58.395, 57.12, 57.375],  # 中文注释：采用ImageNet标准差配合归一化
    bgr_to_rgb=True,  # 中文注释：读取图像后从BGR转换为RGB通道顺序
    pad_size_divisor=64,  # 中文注释：将图像边长填充到64的倍数以利于多尺度特征对齐
)  # 中文注释：预处理器配置结束

teacher_ir = _apply_drone_specialization(deepcopy(_base_.model))  # 中文注释：基于基础DIFF模型深拷贝并套用无人机特化配置构建冻结教师
teacher_ir_default_ckpt = 'work_dirs/pretrained/sim_ir_diff_detector.pth'  # 中文注释：默认教师权重占位路径可通过 --cfg-options model.teacher_ir.init_cfg.checkpoint=xxx 覆盖
teacher_ir['init_cfg'] = dict(type='Pretrained', checkpoint=teacher_ir_default_ckpt)  # 中文注释：使用Pretrained初始化教师扩散检测器权重
teacher_ir['data_preprocessor'] = det_data_preprocessor  # 中文注释：将教师的数据预处理器与学生保持一致避免分布差异

student_rgb = _apply_drone_specialization(deepcopy(_base_.model))  # 中文注释：深拷贝基础模型构建学生分支并应用同样的无人机特化修改
student_rgb_default_ckpt = 'work_dirs/pretrained/sim_rgb_diff_detector.pth'  # 中文注释：默认学生预热权重占位路径可通过 --cfg-options model.student_rgb.init_cfg.checkpoint=xxx 覆盖
student_rgb['init_cfg'] = dict(type='Pretrained', checkpoint=student_rgb_default_ckpt)  # 中文注释：指定学生扩散检测器的预训练权重
student_rgb['data_preprocessor'] = det_data_preprocessor  # 中文注释：指定学生的数据预处理器确保输入管线一致

model = dict(  # 中文注释：构建DualDiffFusionStage1检测器顶层配置
    type='DualDiffFusionStage1',  # 中文注释：指定使用第一阶段扩散融合蒸馏框架
    teacher_ir=teacher_ir,  # 中文注释：注入冻结的红外教师分支配置
    student_rgb=student_rgb,  # 中文注释：注入可训练的仿真RGB学生分支配置
    data_preprocessor=det_data_preprocessor,  # 中文注释：顶层模型沿用学生预处理器保持训练数据一致
    train_cfg=dict(  # 中文注释：定义蒸馏阶段的核心超参数
        w_sup=1.0,  # 中文注释：学生监督损失权重建议保持1.0作为基准
        w_cross=1.0,  # 中文注释：交叉蒸馏损失权重默认开启便于融合教师特征
        w_feat_kd=0.0,  # 中文注释：特征蒸馏默认关闭按需通过配置文件覆盖
        enable_roi_kd=False,  # 中文注释：ROI级蒸馏默认关闭避免早期不稳定
        w_roi_kd=1.0,  # 中文注释：ROI蒸馏损失权重设置为1.0当启用时直接生效
        cross_warmup_iters=0,  # 中文注释：交叉蒸馏预热迭代默认0表示立即启用
        freeze_teacher=True,  # 中文注释：冻结教师权重确保第一阶段仅训练学生
    ),  # 中文注释：蒸馏配置结束
)  # 中文注释：模型总配置结束

train_pipeline = [  # 中文注释：定义仿真RGB无人机训练阶段的数据处理流水线
    dict(type='LoadImageFromFile', backend_args=backend_args),  # 中文注释：从磁盘读取图像文件
    dict(type='LoadAnnotations', with_bbox=True),  # 中文注释：加载边界框标注用于监督
    dict(type='Resize', scale=(1600, 900), keep_ratio=True),  # 中文注释：统一缩放至1600x900保持纵横比
    dict(  # 中文注释：配置随机裁剪增强
        type='RandomCrop',  # 中文注释：使用随机裁剪算子提升尺度多样性
        crop_size=(640, 640),  # 中文注释：裁剪区域尺寸设为640平方
        allow_negative_crop=True,  # 中文注释：允许裁剪后无目标以提升鲁棒性
        recompute_bbox=True,  # 中文注释：裁剪后重算边框坐标保持准确
    ),  # 中文注释：随机裁剪配置结束
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1e-2, 1e-2)),  # 中文注释：过滤极小边框防止数值不稳定
    dict(type='RandomFlip', prob=0.5),  # 中文注释：以50%概率水平翻转增强方向泛化
    dict(  # 中文注释：配置轻量随机颜色增强
        type='RandAugment',  # 中文注释：使用RandAugment统一调度颜色变换
        aug_space=[  # 中文注释：定义颜色操作候选集合
            [dict(type='AutoContrast')],  # 中文注释：自动对比度增强单独作为一个子操作
            [dict(type='Equalize')],  # 中文注释：直方图均衡化提升暗部细节
            [dict(type='Color')],  # 中文注释：调整色彩饱和度
            [dict(type='Contrast')],  # 中文注释：调整对比度幅度
            [dict(type='Brightness')],  # 中文注释：调整亮度水平
            [dict(type='Sharpness')],  # 中文注释：调整锐度表现
        ],  # 中文注释：颜色增强候选结束
        aug_num=1,  # 中文注释：每张图随机选择一个增强操作
    ),  # 中文注释：RandAugment配置结束
    dict(  # 中文注释：打包训练样本结构
        type='PackDetInputs',  # 中文注释：使用检测任务专用打包器
        meta_keys=(  # 中文注释：列出需要保留的元信息字段
            'img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor', 'flip', 'flip_direction', 'homography_matrix',  # 中文注释：包含训练过程中常用的图像属性
        ),  # 中文注释：元信息字段结束
    ),  # 中文注释：打包步骤结束
]  # 中文注释：训练流水线定义完成

test_pipeline = [  # 中文注释：定义验证与测试阶段的数据处理流水线
    dict(type='LoadImageFromFile', backend_args=backend_args),  # 中文注释：读取图像输入
    dict(type='Resize', scale=(1600, 960), keep_ratio=True),  # 中文注释：调整到1600x960用于评估
    dict(type='LoadAnnotations', with_bbox=True),  # 中文注释：加载标注以计算评估指标
    dict(  # 中文注释：打包评估输入结构
        type='PackDetInputs',  # 中文注释：使用检测任务打包器
        meta_keys=(  # 中文注释：保留关键元信息字段
            'img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor',  # 中文注释：这些字段用于恢复原始坐标
        ),  # 中文注释：元信息字段列表结束
    ),  # 中文注释：打包步骤结束
]  # 中文注释：测试流水线定义完成

train_dataloader = dict(  # 中文注释：配置训练数据加载器参数
    batch_size=8,  # 中文注释：每个迭代批量大小设为8以平衡显存与稳定性
    num_workers=8,  # 中文注释：使用8个数据加载线程提升吞吐
    persistent_workers=True,  # 中文注释：开启持久化worker避免重复创建线程
    sampler=dict(type='DefaultSampler', shuffle=True),  # 中文注释：按默认策略随机打乱样本
    batch_sampler=dict(type='AspectRatioBatchSampler'),  # 中文注释：按长宽比分组组成批次减小填充浪费
    dataset=dict(  # 中文注释：定义训练数据集细节
        type=dataset_type,  # 中文注释：使用COCO格式数据解析器
        data_root=data_root,  # 中文注释：指定数据根目录
        metainfo=dict(classes=classes),  # 中文注释：注入类别元信息便于评估解析
        ann_file='sim_drone_ann/rgb/train.json',  # 中文注释：仿真RGB训练标注文件路径
        data_prefix=dict(img=rgb_img_prefix),  # 中文注释：图像前缀目录
        filter_cfg=dict(filter_empty_gt=True),  # 中文注释：过滤无标注图片保持监督有效
        pipeline=train_pipeline,  # 中文注释：绑定上方训练增广流水线
    ),  # 中文注释：训练数据集配置结束
)  # 中文注释：训练数据加载器定义完成

val_dataloader = dict(  # 中文注释：配置验证阶段数据加载器
    batch_size=1,  # 中文注释：单张图像评估保证结果精确
    num_workers=8,  # 中文注释：使用8个线程读取数据
    persistent_workers=True,  # 中文注释：开启线程持久化减少切换开销
    drop_last=False,  # 中文注释：不丢弃最后一个未满批次
    sampler=dict(type='DefaultSampler', shuffle=False),  # 中文注释：顺序遍历验证集确保结果稳定
    dataset=dict(  # 中文注释：定义验证数据集细节
        type=dataset_type,  # 中文注释：COCO格式解析
        data_root=data_root,  # 中文注释：指定数据根目录
        metainfo=dict(classes=classes),  # 中文注释：注入类别信息
        ann_file='sim_drone_ann/rgb/val.json',  # 中文注释：仿真RGB验证标注路径
        data_prefix=dict(img=rgb_img_prefix),  # 中文注释：验证图像目录前缀
        test_mode=True,  # 中文注释：启用测试模式禁用训练增广
        filter_cfg=dict(filter_empty_gt=True),  # 中文注释：过滤无标注图像保持评估一致
        pipeline=test_pipeline,  # 中文注释：绑定测试流水线
    ),  # 中文注释：验证数据集配置结束
)  # 中文注释：验证数据加载器定义完成

test_dataloader = val_dataloader  # 中文注释：测试阶段复用验证加载器配置保持一致

val_evaluator = dict(  # 中文注释：配置验证指标
    type='CocoMetric',  # 中文注释：使用COCO标准评估器
    ann_file=data_root + 'sim_drone_ann/rgb/val.json',  # 中文注释：指向验证标注文件
    metric='bbox',  # 中文注释：评估边界框检测性能
    format_only=False,  # 中文注释：直接输出完整COCO指标
)  # 中文注释：验证评估器配置结束

test_evaluator = val_evaluator  # 中文注释：测试评估器与验证阶段保持一致

optim_wrapper = dict(  # 中文注释：优化器封装配置
    type='OptimWrapper',  # 中文注释：采用MMEngine标准优化器封装
    optimizer=dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001),  # 中文注释：使用SGD优化学生分支参数
    paramwise_cfg=dict(custom_keys={'teacher_ir': dict(lr_mult=0.0, decay_mult=0.0)}),  # 中文注释：将教师参数学习率与权重衰减置零确保仅更新学生
    clip_grad=dict(max_norm=2, norm_type=2),  # 中文注释：启用梯度裁剪提升训练稳定性
)  # 中文注释：优化器配置结束

default_hooks = dict(  # 中文注释：重写默认钩子以加入融合教师导出逻辑
    timer=dict(type='IterTimerHook'),  # 中文注释：记录迭代耗时
    logger=dict(type='LoggerHook', interval=50),  # 中文注释：每50次迭代输出日志
    param_scheduler=dict(type='ParamSchedulerHook'),  # 中文注释：负责参数调度器步进
    checkpoint=dict(  # 中文注释：配置常规权重保存策略
        type='CheckpointHook', interval=1000, by_epoch=False, max_keep_ckpts=3, save_best=['coco/bbox_mAP_50'],  # 中文注释：每千次迭代保存一次并追踪最优mAP
    ),  # 中文注释：CheckpointHook配置结束
    sampler_seed=dict(type='DistSamplerSeedHook'),  # 中文注释：多卡训练时同步采样器随机种子
    visualization=dict(  # 中文注释：训练过程中定期可视化预测结果
        type='DetVisualizationHook', draw=True, interval=1000, test_out_dir='drone_vis',  # 中文注释：每千次迭代保存可视化结果
    ),  # 中文注释：可视化钩子配置结束
    fused_teacher_export=dict(  # 中文注释：新增定期导出融合教师权重的钩子
        type='FusedTeacherExportHook', interval=1000, by_epoch=False, filename='student_rgb_fused.pth',  # 中文注释：每千次迭代调用export_fused_teacher
    ),  # 中文注释：融合教师导出钩子配置结束
)  # 中文注释：默认钩子配置结束

log_processor = dict(type='LogProcessor', window_size=50, by_epoch=False)  # 中文注释：日志处理器保持与基础设置一致

train_cfg = dict(type='IterBasedTrainLoop', max_iters=5000, val_interval=1000)  # 中文注释：迭代式训练循环共5000步并每千步验证
val_cfg = dict(type='ValLoop')  # 中文注释：使用默认验证循环实现
test_cfg = dict(type='TestLoop')  # 中文注释：使用默认测试循环实现

auto_scale_lr = dict(enable=True, base_batch_size=16)  # 中文注释：允许按照总批量自动缩放学习率
find_unused_parameters = True  # 中文注释：在分布式训练中启用查找未使用参数以避免梯度同步错误

if __name__ == '__main__':  # 中文注释：提供最小化自检脚本方便快速验证配置可用性
    import torch  # 中文注释：导入PyTorch以构造虚拟输入张量
    from mmengine.config import Config  # 中文注释：导入Config类用于读取当前配置文件
    from mmdet.registry import MODELS  # 中文注释：导入模型注册表以实例化检测器

    cfg = Config.fromfile(__file__)  # 中文注释：加载当前配置文件内容
    detector = MODELS.build(cfg.model)  # 中文注释：根据配置构建DualDiffFusionStage1模型实例
    dummy_inputs = torch.randn(1, 3, 640, 640)  # 中文注释：创建单张虚拟RGB图像作为输入
    with torch.no_grad():  # 中文注释：关闭梯度计算以进行快速前向测试
        _ = detector.extract_feat_student(dummy_inputs)  # 中文注释：调用学生分支特征提取验证前向流程
    print('提示：在正式训练前请将 teacher_ir/student_rgb 的 init_cfg.checkpoint 更新为真实权重路径')  # 中文注释：提醒用户替换真实权重避免占位符导致加载失败
    print('DualDiffFusionStage1 配置自检通过')  # 中文注释：输出自检成功提示
