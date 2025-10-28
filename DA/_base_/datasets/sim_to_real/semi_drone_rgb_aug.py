# dataset settings
dataset_type = 'CocoDataset'  # 指定数据集类型为 COCO 格式
data_root = 'data/'  # 指定数据根目录，配合 COCO json 中的相对路径使用
classes = ('drone',)  # 定义单类别元信息，仅包含无人机
backend_args = None  # 默认使用本地文件后端读取图像
branch_field = ['sup', 'unsup_teacher', 'unsup_student']  # 定义多分支字段名称，确保增广流水线输出一致
color_space = [  # 定义颜色增强搜索空间
    [dict(type='ColorTransform')],  # 色彩变换操作提升风格多样性
    [dict(type='AutoContrast')],  # 自动对比度增强改善亮度分布
    [dict(type='Equalize')],  # 直方图均衡化提升对比度
    [dict(type='Sharpness')],  # 锐化操作突出细节
    [dict(type='Posterize')],  # 色阶压缩增强风格差异
    [dict(type='Solarize')],  # 反相增强制造极端风格
    [dict(type='Color')],  # 饱和度调整模拟不同光照
    [dict(type='Contrast')],  # 对比度调整强化目标
    [dict(type='Brightness')],  # 亮度调整提升鲁棒性
]  # 结束颜色增强定义
geometric = [  # 定义几何增强搜索空间
    [dict(type='Rotate')],  # 旋转操作模拟姿态变化
    [dict(type='ShearX')],  # 水平错切增强视角
    [dict(type='ShearY')],  # 垂直错切增强视角
    [dict(type='TranslateX')],  # 水平平移模拟跟拍误差
    [dict(type='TranslateY')],  # 垂直平移模拟高度变化
]  # 结束几何增强定义
sup_aug_pipeline = [  # 定义监督分支的额外增广
    dict(  # 采用随机顺序组合增强
        type='RandomOrder',  # 在多个增强之间随机排序
        transforms=[dict(type='RandAugment', aug_space=color_space, aug_num=1)]),  # 随机抽取一种颜色增强
    dict(type='RandomErasing', n_patches=(1, 5), ratio=(0, 0.2)),  # 在图像上随机擦除模拟遮挡
    dict(  # 引入域风格自适应
        type='AlbuDomainAdaption',  # 使用 Albumentations 域迁移模块
        domain_adaption_type='ALL',  # 组合所有风格迁移策略
        target_dir='data/real_drone_rgb/style_bank',  # 指向真实域风格图像仓库
        p=0),  # 以 50% 概率执行风格迁移
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1e-2, 1e-2)),  # 删除过小目标避免噪声
    dict(  # 打包监督分支数据
        type='PackDetInputs',  # 转换为检测任务输入格式
        meta_keys=(  # 需要保留的元信息键
            'img_id',  # 图像编号
            'img_path',  # 图像路径
            'ori_shape',  # 原始尺寸
            'img_shape',  # 当前尺寸
            'scale_factor',  # 缩放比例
            'flip',  # 翻转标记
            'flip_direction',  # 翻转方向
            'homography_matrix')),  # 单应矩阵信息
]  # 结束监督增广流水线
strong_pipeline = [  # 定义强增广流水线供学生模型使用
    dict(  # 首先执行随机顺序增强
        type='RandomOrder',  # 随机排列增强操作
        transforms=[  # 同时包含颜色与几何增强
            dict(type='RandAugment', aug_space=color_space, aug_num=1),  # 颜色增强
            dict(type='RandAugment', aug_space=geometric, aug_num=1),  # 几何增强
        ]),  # 结束增强定义
    dict(type='RandomErasing', n_patches=(1, 5), ratio=(0, 0.2)),  # 随机擦除提升鲁棒性
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1e-2, 1e-2)),  # 过滤微小框
    dict(  # 打包输出供学生分支使用
        type='PackDetInputs',  # 统一打包逻辑
        meta_keys=(  # 所需元信息键
            'img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor', 'flip', 'flip_direction', 'homography_matrix')),  # 汇总元信息
]  # 结束强增广流水线
weak_pipeline = [  # 定义弱增广流水线供教师模型使用
    dict(  # 直接打包当前视角
        type='PackDetInputs',  # 使用基础打包器
        meta_keys=(  # 所需元信息
            'img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor', 'flip', 'flip_direction', 'homography_matrix')),  # 保留基础元数据
]  # 结束弱增广流水线
sup_pipeline_rgb = [  # 定义监督数据基础流水线（RGB）
    dict(type='LoadImageFromFile', backend_args=backend_args),  # 读取源域图像
    dict(type='LoadAnnotations', with_bbox=True),  # 加载边界框标注
    dict(type='Resize', scale=(1600, 900), keep_ratio=True),  # 调整分辨率并保持纵横比
    dict(  # 执行随机裁剪
        type='RandomCrop',  # 随机裁剪操作
        crop_type='absolute',  # 使用绝对尺寸
        crop_size=(640, 640),  # 裁剪大小兼顾 720p 与 1080p 图像
        recompute_bbox=True,  # 裁剪后重新计算边界框
        allow_negative_crop=True),  # 允许产生无目标图像
    dict(type='RandomFlip', prob=0.5),  # 随机水平翻转
    dict(  # 将监督图像送入多分支增广
        type='MultiBranch',  # 多分支包装器
        branch_field=branch_field,  # 指定分支名称
        sup=sup_aug_pipeline),  # 监督分支使用额外增广
]  # 结束监督流水线
sup_pipeline_ir = sup_pipeline_rgb.copy()
sup_pipeline_ir[0] = dict(  # IR 图像加载需要显式展开通道
    type='LoadImageFromFile',
    backend_args=backend_args,
    color_type='color_ignore_orientation')
unsup_pipeline = [  # 定义无监督数据流水线
    dict(type='LoadImageFromFile', backend_args=backend_args),  # 读取目标域图像
    dict(type='LoadEmptyAnnotations'),  # 不加载标注，仅占位
    dict(type='Resize', scale=(1600, 900), keep_ratio=True),  # 调整目标域分辨率
    dict(  # 执行随机裁剪
        type='RandomCrop',  # 随机裁剪操作
        crop_type='absolute',  # 使用绝对尺寸
        crop_size=(640, 640),  # 与监督分支保持一致
        recompute_bbox=True,  # 裁剪后更新框
        allow_negative_crop=True),  # 允许无目标裁剪
    dict(type='RandomFlip', prob=0.5),  # 随机翻转增强
    dict(  # 将图像拆分为教师与学生视角
        type='MultiBranch',  # 使用多分支包装器
        branch_field=branch_field,  # 指定输出分支
        unsup_teacher=weak_pipeline,  # 教师使用弱增广
        unsup_student=strong_pipeline),  # 学生使用强增广
]  # 结束无监督流水线
test_pipeline = [  # 定义验证测试流水线
    dict(type='LoadImageFromFile', backend_args=backend_args),  # 读取评估图像
    dict(type='Resize', scale=(1600, 900), keep_ratio=True),  # 调整尺寸保持比例
    dict(  # 打包测试输入
        type='PackDetInputs',  # 使用检测打包器
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor')),  # 保留必要元信息
]  # 结束测试流水线
batch_size = 16  # 设置训练批大小
num_workers = 16  # 设置加载线程数
labeled_dataset_rgb = dict(  # 定义仿真RGB带标注数据集
    type=dataset_type,  # 使用 COCO 数据集类
    data_root=data_root,  # 指定数据根目录
    metainfo=dict(classes=classes),  # 设置类别信息
    ann_file='sim_drone_ann/rgb/train.json',  # 指向仿真RGB训练标注
    data_prefix=dict(img='sim_drone_rgb/Town01_Opt/carla_data/images_rgb/'),  # 指向RGB图像目录
    filter_cfg=dict(filter_empty_gt=True),  # 过滤无标注图像
    pipeline=sup_pipeline_rgb)  # 采用RGB监督流水线
labeled_dataset_ir = dict(  # 定义仿真IR带标注数据集
    type=dataset_type,  # 使用 COCO 数据集类
    data_root=data_root,  # 指定数据根目录
    metainfo=dict(classes=classes),  # 设置类别信息
    ann_file='sim_drone_ann/ir/train.json',  # 指向仿真IR训练标注
    data_prefix=dict(img='sim_drone_ir/Town01_Opt/carla_data/images_ir/'),  # 指向IR图像目录
    filter_cfg=dict(filter_empty_gt=True),  # 过滤无标注图像
    pipeline=sup_pipeline_ir)  # 采用IR监督流水线
labeled_dataset = dict(  # 组合仿真RGB与IR数据集
    type='ConcatDataset',
    datasets=[labeled_dataset_rgb, labeled_dataset_ir])
unlabeled_dataset = dict(  # 定义无标注目标域数据集
    type=dataset_type,  # 使用 COCO 数据集类
    data_root=data_root,  # 指定数据根目录
    metainfo=dict(classes=classes),  # 设置类别信息
    ann_file='real_drone_ann/train_visible.json',  # 指向真实域训练标注（仅使用图像路径）
    data_prefix=dict(img='real_drone_rgb/'),  # 指向真实图像根目录
    pipeline=unsup_pipeline)  # 使用无监督流水线
train_dataloader = dict(  # 配置训练数据加载器
    batch_size=batch_size,  # 设定批大小
    num_workers=num_workers,  # 工作线程数
    persistent_workers=True,  # 复用数据加载进程
    sampler=dict(  # 多源采样器配置
        type='GroupMultiSourceSampler',  # 维持源与目标域平衡
        batch_size=batch_size,  # 指定批大小
        source_ratio=[1, 1]),  # 源域与目标域比例 1:1
    dataset=dict(  # 将源与目标拼接
        type='ConcatDataset',  # 使用拼接数据集
        datasets=[labeled_dataset, unlabeled_dataset]))  # 提供两域数据
val_dataloader = dict(  # 配置验证加载器
    batch_size=1,  # 单图像评估
    num_workers=8,  # 工作线程
    persistent_workers=True,  # 复用加载器
    drop_last=False,  # 保留尾批
    sampler=dict(type='DefaultSampler', shuffle=False),  # 按序采样
    dataset=dict(  # 定义真实域验证集
        type=dataset_type,  # 使用 COCO 数据格式
        data_root=data_root,  # 数据根目录
        metainfo=dict(classes=classes),  # 类别信息
        ann_file='real_drone_ann/val_visible.json',  # 真实域验证标注文件
        data_prefix=dict(img='real_drone_rgb/'),  # 真实域图像根目录
        test_mode=True,  # 开启测试模式
        filter_cfg=dict(filter_empty_gt=True),  # 过滤无标注图像
        pipeline=test_pipeline))  # 使用测试流水线
test_dataloader = dict(  # 配置测试加载器
    batch_size=1,  # 单图像测试
    num_workers=8,  # 工作线程
    persistent_workers=True,  # 复用加载器
    drop_last=False,  # 保留尾批
    sampler=dict(type='DefaultSampler', shuffle=False),  # 顺序采样
    dataset=dict(  # 定义真实域测试集
        type=dataset_type,  # 使用 COCO 数据格式
        data_root=data_root,  # 数据根目录
        metainfo=dict(classes=classes),  # 类别信息
        ann_file='real_drone_ann/test_visible.json',  # 真实域测试标注文件
        data_prefix=dict(img='real_drone_rgb/'),  # 真实域图像根目录
        test_mode=True,  # 开启测试模式
        filter_cfg=dict(filter_empty_gt=True),  # 过滤无标注图像
        pipeline=test_pipeline))  # 使用测试流水线
val_evaluator = dict(  # 定义验证评估器
    type='CocoMetric',  # 使用 COCO 指标
    ann_file=data_root + 'real_drone_ann/val_visible.json',  # 指向验证标注文件
    metric='bbox',  # 计算边界框精度
    format_only=False)  # 直接输出指标
test_evaluator = dict(  # 定义测试评估器
    type='CocoMetric',  # 使用 COCO 指标
    ann_file=data_root + 'real_drone_ann/test_visible.json',  # 指向测试标注文件
    metric='bbox',  # 计算边界框精度
    format_only=False)  # 直接输出指标