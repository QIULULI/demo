_base_ = [  # 指定需要继承的基础配置列表
    '../../_base_/models/faster-rcnn_diff_fpn.py',  # 继承扩散检测器的两阶段结构
    '../../_base_/dg_setting/dg_20k.py',  # 继承DG训练调度和可视化设置
]  # 结束基础配置列表
# ----------------------------------------------------------------------------------------------------
classes = ('drone',)  # 定义单类别列表
# ----------------------------------------------------------------------------------------------------
dataset_type = 'CocoDataset'  # 指明数据集采用COCO标注格式
data_root = 'data/'  # 设置标注文件的相对根目录
ir_img_prefix = 'sim_drone_ir/Town01_Opt/carla_data/'  # 定义IR图像所在的绝对目录
backend_args = None  # 关闭后端参数配置，使用默认文件读取方式
# ----------------------------------------------------------------------------------------------------
color_space_light = [  # 定义轻量级颜色增强操作空间
    [dict(type='AutoContrast')],  # 自动对比度增强以提升动态范围
    [dict(type='Equalize')],  # 直方图均衡化提升亮度分布
    [dict(type='Color')],  # 调整色彩饱和度以模拟光照变化
    [dict(type='Contrast')],  # 调整对比度以增加纹理对比
    [dict(type='Brightness')],  # 调整亮度以覆盖曝光差异
    [dict(type='Sharpness')],  # 调整锐度突出边缘细节
]  # 结束颜色增强空间定义
# ----------------------------------------------------------------------------------------------------
train_pipeline = [  # 构建训练阶段的数据处理流水线
    dict(type='LoadImageFromFile', backend_args=backend_args),  # 读取原始图像
    dict(type='LoadAnnotations', with_bbox=True),  # 载入边界框标注信息
    dict(type='Resize', scale=(1600, 960), keep_ratio=True),  # 首先将图像缩放到高分辨率以保留小目标
    dict(  # 定义随机裁剪步骤
        type='RandomCrop',  # 使用随机裁剪
        crop_size=(640, 640),  # 裁剪成640x640的小贴片
        allow_negative_crop=True,  # 允许裁剪后没有目标以提升鲁棒性
        recompute_bbox=True),  # 重新计算裁剪后的边界框
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1e-2, 1e-2)),  # 过滤掉面积极小的框以减少噪声
    dict(type='RandomFlip', prob=0.5),  # 以50%概率水平翻转以扩充样本
    dict(type='RandAugment', aug_space=color_space_light, aug_num=1),  # 随机选取一种轻量颜色增强操作
    dict(  # 打包输入数据
        type='PackDetInputs',  # 使用检测任务打包器
        meta_keys=(  # 指定需要保留的元信息键
            'img_id',  # 图像编号
            'img_path',  # 图像路径
            'ori_shape',  # 原始图像尺寸
            'img_shape',  # 增强后图像尺寸
            'scale_factor',  # 缩放比例
            'flip',  # 是否翻转
            'flip_direction',  # 翻转方向
            'homography_matrix')),  # 仿射矩阵用于记录变换
]  # 结束训练流水线定义
# ----------------------------------------------------------------------------------------------------
test_pipeline = [  # 构建验证与测试阶段的数据处理流水线
    dict(type='LoadImageFromFile', backend_args=backend_args),  # 读取图像文件
    dict(type='Resize', scale=(1600, 960), keep_ratio=True),  # 按照训练同样的高分辨率缩放
    dict(type='LoadAnnotations', with_bbox=True),  # 加载标注用于评估指标
    dict(  # 打包评估输入
        type='PackDetInputs',  # 使用检测任务打包器
        meta_keys=(  # 指定需要保留的元信息键
            'img_id',  # 图像编号
            'img_path',  # 图像路径
            'ori_shape',  # 原始图像尺寸
            'img_shape',  # 处理后图像尺寸
            'scale_factor')),  # 缩放比例用于恢复坐标
]  # 结束测试流水线定义
# ----------------------------------------------------------------------------------------------------
train_dataloader = dict(  # 配置训练数据加载器
    batch_size=8,  # 每个迭代批量大小为8
    num_workers=8,  # 使用8个工作进程加速数据加载
    persistent_workers=True,  # 复用dataloader线程以降低重启开销
    sampler=dict(type='DefaultSampler', shuffle=True),  # 使用默认采样器并开启随机打乱
    batch_sampler=dict(type='AspectRatioBatchSampler'),  # 根据宽高比分组以减少填充
    dataset=dict(  # 指定实际使用的数据集
        type=dataset_type,  # 使用COCO格式数据集
        data_root=data_root,  # 设置标注根目录
        metainfo=dict(classes=classes),  # 传入类别元信息
        ann_file='sim_drone_ann/ir/train.json',  # 指定IR训练标注文件
        data_prefix=dict(img=ir_img_prefix),  # 指定图像目录
        filter_cfg=dict(filter_empty_gt=True),  # 过滤无标注图片
        pipeline=train_pipeline))  # 使用前面定义的训练流水线
# ----------------------------------------------------------------------------------------------------
val_dataloader = dict(  # 配置验证数据加载器
    batch_size=1,  # 验证阶段使用单张图像
    num_workers=8,  # 使用8个进程读取数据
    persistent_workers=True,  # 复用线程避免重复创建
    drop_last=False,  # 不丢弃最后一个小批次
    sampler=dict(type='DefaultSampler', shuffle=False),  # 顺序采样确保评估稳定
    dataset=dict(  # 定义验证数据集
        type=dataset_type,  # 同样使用COCO格式
        data_root=data_root,  # 指定标注根目录
        metainfo=dict(classes=classes),  # 指定类别信息
        ann_file='sim_drone_ann/ir/val.json',  # 读取IR验证标注
        data_prefix=dict(img=ir_img_prefix),  # 指定IR验证图像目录
        test_mode=True,  # 标记为测试模式
        filter_cfg=dict(filter_empty_gt=True),  # 过滤无标注样本
        pipeline=test_pipeline))  # 沿用基础配置的测试流水线
# ----------------------------------------------------------------------------------------------------
test_dataloader = val_dataloader  # 将测试配置与验证保持一致，方便复用IR验证集
# ----------------------------------------------------------------------------------------------------
val_evaluator = dict(  # 定义验证指标
    type='CocoMetric',  # 使用COCO检测指标
    ann_file=data_root + 'sim_drone_ann/ir/val.json',  # 指定验证标注文件完整路径
    metric='bbox',  # 评估边界框指标
    format_only=False)  # 直接计算指标而非仅导出结果
# ----------------------------------------------------------------------------------------------------
test_evaluator = val_evaluator  # 复用同样的评估器以保持一致
# ----------------------------------------------------------------------------------------------------
model = dict(  # 调整模型超参数
    roi_head=dict(bbox_head=dict(num_classes=1)),  # 将ROI头的类别数设置为1
    backbone=dict(diff_config=dict(classes=classes)),  # 将类别信息传入扩散主干以对齐训练类别
    rpn_head=dict(  # 调整RPN以适配小目标
        anchor_generator=dict(  # 修改锚框生成器
            type='AnchorGenerator',  # 指定生成器类型
            scales=[2, 4, 8],  # 使用更小的尺度捕获微小目标
            ratios=[0.33, 0.5, 1.0, 2.0],  # 引入更多纵横比提升召回
            strides=[4, 8, 16, 32, 64])))  # 明确FPN各层的步长以正确匹配特征图
# ----------------------------------------------------------------------------------------------------