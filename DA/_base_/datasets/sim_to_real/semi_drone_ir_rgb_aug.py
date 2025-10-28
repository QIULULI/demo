# 半监督仿真到真实无人机检测配置，融合RGB与IR监督样本并定义无监督增强  # 顶部中文注释说明文件作用
from copy import deepcopy  # 从copy模块导入deepcopy以便深拷贝复杂结构
# ------------------------------  # 分隔线，保持可读性
# 数据集基础参数设定  # 说明接下来定义常量
dataset_type = 'CocoDataset'  # 指定采用COCO格式数据集
data_root = 'data/'  # 设置数据根目录
classes = ('drone',)  # 设定单一类别为无人机
backend_args = None  # 数据加载后端保持默认
branch_field = ['sup', 'unsup_teacher', 'unsup_student']  # 多分支流水线需要的字段列表
# ------------------------------  # 分隔线
# 颜色增强空间定义，复用RGB-only方案  # 说明即将列出候选增强
color_space = [  # 初始化颜色增强候选列表
    [dict(type='ColorTransform')],  # 候选1：颜色空间变换
    [dict(type='AutoContrast')],  # 候选2：自动对比度调整
    [dict(type='Equalize')],  # 候选3：直方图均衡
    [dict(type='Sharpness')],  # 候选4：锐化增强
    [dict(type='Posterize')],  # 候选5：色调分层
    [dict(type='Solarize')],  # 候选6：曝光反转
    [dict(type='Color')],  # 候选7：颜色饱和度调节
    [dict(type='Contrast')],  # 候选8：对比度调节
    [dict(type='Brightness')],  # 候选9：亮度调节
]  # 结束颜色增强候选定义
# ------------------------------  # 分隔线
# 定义子集构造函数以便为IR域选择更温和的增强组合  # 说明函数用途
# ------------------------------  # 占位分隔防止空行
def _subset(space, indices):  # 定义函数从增强空间中按索引挑选子集
    return [deepcopy(space[idx]) for idx in indices]  # 返回指定索引的浅拷贝列表
# ------------------------------  # 分隔线
# 为RGB和IR分别构造颜色增强空间  # 说明不同域策略
color_space_rgb = deepcopy(color_space)  # RGB保持完整颜色增强空间
color_space_ir = _subset(color_space, [1, 2, 3, 6, 7, 8])  # IR选取温和增强避免过度扰动
# ------------------------------  # 分隔线
# 定义RGB监督分支的域自适应风格迁移配置  # 说明域自适应作用
domain_style_rgb = dict(  # 构造RGB风格迁移配置字典
    type='AlbuDomainAdaption',  # 指定使用Albumentations域自适应算子
    domain_adaption_type='ALL',  # 对所有样本启用风格迁移
    target_dir='data/real_drone_rgb/style_bank',  # 风格库路径指向真实RGB图像
    p=0,  # 当前迁移概率设为零保持外观稳定
)  # RGB风格迁移配置结束
# ------------------------------  # 分隔线
# 构建监督分支增强流水线辅助函数，允许覆盖随机擦除和域自适应参数  # 说明函数功能
# ------------------------------  # 占位分隔防止空行
def build_sup_aug_pipeline(  # 定义监督增强流水线构建函数
    color_aug_space,  # 颜色增强候选空间参数
    *,  # 强制后续参数使用关键字形式
    random_erasing_cfg=None,  # 可选随机擦除配置
    domain_adaption_cfg=None,  # 可选域自适应配置
):  # 函数头结束
    aug_pipeline = [  # 初始化增强步骤列表
        dict(  # 新增随机顺序组合模块
            type='RandomOrder',  # 指定随机执行顺序
            transforms=[dict(type='RandAugment', aug_space=color_aug_space, aug_num=1)],  # 使用RandAugment按空间采样一次
        )  # RandomOrder配置结束
    ]  # 增强步骤列表完成初始设置
    if random_erasing_cfg is None:  # 若未显式提供随机擦除配置
        random_erasing_cfg = dict(type='RandomErasing', n_patches=(1, 5), ratio=(0, 0.2))  # 采用默认较强擦除策略
    if random_erasing_cfg:  # 当存在随机擦除配置时
        aug_pipeline.append(deepcopy(random_erasing_cfg))  # 深拷贝配置后加入流水线避免共享
    if domain_adaption_cfg:  # 若提供域自适应配置
        aug_pipeline.append(deepcopy(domain_adaption_cfg))  # 深拷贝后添加以独立控制
    aug_pipeline.extend([  # 追加通用后处理步骤
        dict(type='FilterAnnotations', min_gt_bbox_wh=(1e-2, 1e-2)),  # 过滤过小标注框
        dict(  # 打包检测输入
            type='PackDetInputs',  # 使用PackDetInputs整理输出
            meta_keys=(  # 指定需要保留的元信息字段
                'img_id',  # 元信息：图像ID
                'img_path',  # 元信息：图像路径
                'ori_shape',  # 元信息：原始尺寸
                'img_shape',  # 元信息：处理后尺寸
                'scale_factor',  # 元信息：缩放系数
                'flip',  # 元信息：是否翻转
                'flip_direction',  # 元信息：翻转方向
                'homography_matrix',  # 元信息：单应矩阵
            ),  # 元信息字段元组结束
        ),  # PackDetInputs配置结束
    ])  # 通用后处理步骤添加完成
    return aug_pipeline  # 返回构造好的增强流水线
# ------------------------------  # 分隔线
# 为RGB与IR构建具体的监督增强流水线  # 说明以下变量含义
rgb_sup_aug_pipeline = build_sup_aug_pipeline(  # 构建RGB监督增强流水线
    color_space_rgb, domain_adaption_cfg=domain_style_rgb  # 使用完整颜色空间并附加风格迁移
)  # RGB监督增强流水线构建完成
ir_sup_aug_pipeline = build_sup_aug_pipeline(  # 构建IR监督增强流水线
    color_space_ir,  # 使用温和颜色增强空间
    random_erasing_cfg=dict(type='RandomErasing', n_patches=(1, 3), ratio=(0, 0.1)),  # 覆盖随机擦除使扰动更轻
)  # IR监督增强流水线构建完成
# ------------------------------  # 分隔线
# 定义几何增强空间用于无监督学生分支  # 说明即将列出几何操作
geometric = [  # 初始化几何增强候选列表
    [dict(type='Rotate')],  # 候选1：旋转操作
    [dict(type='ShearX')],  # 候选2：X轴错切
    [dict(type='ShearY')],  # 候选3：Y轴错切
    [dict(type='TranslateX')],  # 候选4：X轴平移
    [dict(type='TranslateY')],  # 候选5：Y轴平移
]  # 几何增强候选列表结束
# ------------------------------  # 分隔线
# 构建无监督学生分支的强增强流水线  # 说明以下列表
strong_pipeline = [  # 定义学生分支强增强流水线
    dict(  # 使用随机顺序组合颜色与几何RandAugment
        type='RandomOrder',  # 指定随机执行顺序
        transforms=[  # 内含两个RandAugment操作
            dict(type='RandAugment', aug_space=color_space_rgb, aug_num=1),  # 颜色增强采样一次
            dict(type='RandAugment', aug_space=geometric, aug_num=1),  # 几何增强采样一次
        ],  # RandAugment操作列表结束
    ),  # RandomOrder配置结束
    dict(type='RandomErasing', n_patches=(1, 5), ratio=(0, 0.2)),  # 应用较强随机擦除丰富扰动
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1e-2, 1e-2)),  # 维持接口一致过滤无效框
    dict(  # 打包增强后的图像供模型使用
        type='PackDetInputs',  # 调用PackDetInputs组件
        meta_keys=(  # 指定保留元信息字段
            'img_id',  # 元信息：图像ID
            'img_path',  # 元信息：图像路径
            'ori_shape',  # 元信息：原始尺寸
            'img_shape',  # 元信息：增强后尺寸
            'scale_factor',  # 元信息：缩放系数
            'flip',  # 元信息：是否翻转
            'flip_direction',  # 元信息：翻转方向
            'homography_matrix',  # 元信息：单应矩阵
        ),  # 元信息字段元组结束
    ),  # PackDetInputs配置结束
]  # 学生分支强增强流水线定义完成
# ------------------------------  # 分隔线
# 构建无监督教师分支的弱增强流水线  # 说明以下列表
weak_pipeline = [  # 定义教师分支弱增强流水线
    dict(  # 保持最小扰动仅执行打包
        type='PackDetInputs',  # 使用PackDetInputs组件
        meta_keys=(  # 指定保留元信息字段
            'img_id',  # 元信息：图像ID
            'img_path',  # 元信息：图像路径
            'ori_shape',  # 元信息：原始尺寸
            'img_shape',  # 元信息：处理后尺寸
            'scale_factor',  # 元信息：缩放系数
            'flip',  # 元信息：是否翻转
            'flip_direction',  # 元信息：翻转方向
            'homography_matrix',  # 元信息：单应矩阵
        ),  # 元信息字段元组结束
    ),  # PackDetInputs配置结束
]  # 教师分支弱增强流水线定义完成
# ------------------------------  # 分隔线
# 定义监督分支的通用前处理模板  # 说明模板作用
sup_pipeline_template = [  # 构建监督分支通用流程
    dict(type='LoadImageFromFile', backend_args=backend_args),  # 读取图像文件
    dict(type='LoadAnnotations', with_bbox=True),  # 加载边界框标注
    dict(type='Resize', scale=(1600, 900), keep_ratio=True),  # 缩放并保持纵横比
    dict(  # 随机裁剪操作配置
        type='RandomCrop',  # 指定使用随机裁剪
        crop_type='absolute',  # 采用绝对尺寸裁剪
        crop_size=(640, 640),  # 设置裁剪尺寸
        recompute_bbox=True,  # 裁剪后重新计算标注框
        allow_negative_crop=True,  # 允许裁剪后无目标
    ),  # RandomCrop配置结束
    dict(type='RandomFlip', prob=0.5),  # 以0.5概率执行随机翻转
]  # 监督分支通用流程定义完成
# ------------------------------  # 分隔线
# 通过模板构建完整监督流水线并插入多分支增强  # 说明下方函数
# ------------------------------  # 占位分隔防止空行
def build_sup_pipeline(branch_aug_pipeline):  # 定义函数根据增强流水线生成完整配置
    pipeline = deepcopy(sup_pipeline_template)  # 深拷贝模板防止共享引用
    pipeline.append(  # 向流水线末尾追加多分支模块
        dict(type='MultiBranch', branch_field=branch_field, sup=branch_aug_pipeline)  # 指定监督分支增强配置
    )  # 追加操作结束
    return pipeline  # 返回完整监督流水线
# ------------------------------  # 分隔线
# 生成RGB和IR监督流水线实例  # 说明下面赋值
rgb_sup_pipeline = build_sup_pipeline(rgb_sup_aug_pipeline)  # 获取RGB监督流水线
ir_sup_pipeline = build_sup_pipeline(ir_sup_aug_pipeline)  # 获取IR监督流水线
# ------------------------------  # 分隔线
# 定义无监督分支整体流水线  # 说明下方列表
unsup_pipeline = [  # 构建无监督流水线
    dict(type='LoadImageFromFile', backend_args=backend_args),  # 读取无标注图像
    dict(type='LoadEmptyAnnotations'),  # 填充空标注保持接口一致
    dict(type='Resize', scale=(1600, 900), keep_ratio=True),  # 缩放无监督图像
    dict(  # 随机裁剪模块
        type='RandomCrop',  # 指定随机裁剪
        crop_type='absolute',  # 采用绝对尺寸裁剪
        crop_size=(640, 640),  # 设置裁剪尺寸
        recompute_bbox=True,  # 虽无标注仍执行重算以兼容接口
        allow_negative_crop=True,  # 允许裁剪后无目标
    ),  # RandomCrop配置结束
    dict(type='RandomFlip', prob=0.5),  # 以0.5概率翻转
    dict(  # 多分支包装器
        type='MultiBranch',  # 指定多分支模块
        branch_field=branch_field,  # 指定分支标签
        unsup_teacher=weak_pipeline,  # 教师分支采用弱增强
        unsup_student=strong_pipeline,  # 学生分支采用强增强
    ),  # MultiBranch配置结束
]  # 无监督流水线定义完成
# ------------------------------  # 分隔线
# 定义测试评估流水线  # 说明下方列表
test_pipeline = [  # 构建测试阶段流水线
    dict(type='LoadImageFromFile', backend_args=backend_args),  # 读取测试图像
    dict(type='Resize', scale=(1600, 900), keep_ratio=True),  # 调整尺寸保持比例
    dict(  # 打包推理输入
        type='PackDetInputs',  # 使用PackDetInputs组件
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor'),  # 指定保留元信息
    ),  # PackDetInputs配置结束
]  # 测试流水线定义完成
# ------------------------------  # 分隔线
# 数据加载器通用参数设置  # 说明即将定义批量等参数
batch_size = 16  # 训练批量大小
num_workers = 16  # 训练阶段数据加载线程数
# ------------------------------  # 分隔线
# 定义RGB标注数据集配置  # 说明下方字典
rgb_labeled_dataset = dict(  # 构造RGB监督数据集
    type=dataset_type,  # 指定数据集类型
    data_root=data_root,  # 指定根目录
    metainfo=dict(classes=classes),  # 声明类别信息
    ann_file='sim_drone_ann/rgb/train.json',  # 标注文件路径
    data_prefix=dict(img='sim_drone_rgb/Town01_Opt/carla_data/'),  # 图像前缀目录
    filter_cfg=dict(filter_empty_gt=True),  # 过滤无标注样本
    pipeline=rgb_sup_pipeline,  # 绑定RGB监督流水线
)  # RGB监督数据集配置结束
# ------------------------------  # 分隔线
# 定义IR标注数据集配置  # 说明下方字典
ir_labeled_dataset = dict(  # 构造IR监督数据集
    type=dataset_type,  # 指定数据集类型
    data_root=data_root,  # 指定根目录
    metainfo=dict(classes=classes),  # 声明类别信息
    ann_file='sim_drone_ann/ir/train.json',  # 标注文件路径
    data_prefix=dict(img='sim_drone_ir/Town01_Opt/carla_data/'),  # 图像前缀目录
    filter_cfg=dict(filter_empty_gt=True),  # 过滤无标注样本
    pipeline=ir_sup_pipeline,  # 绑定IR监督流水线
)  # IR监督数据集配置结束
# ------------------------------  # 分隔线
# 合并RGB与IR监督数据集  # 说明下方字典
labeled_dataset = dict(  # 定义监督数据集合并配置
    type='ConcatDataset',  # 使用ConcatDataset拼接
    datasets=[rgb_labeled_dataset, ir_labeled_dataset],  # 指定要拼接的数据集列表
)  # 监督数据集拼接配置结束
# ------------------------------  # 分隔线
# 定义真实无人机无标注数据集  # 说明下方字典
unlabeled_dataset = dict(  # 构造真实域无监督数据集
    type=dataset_type,  # 指定数据集类型
    data_root=data_root,  # 指定根目录
    metainfo=dict(classes=classes),  # 声明类别信息
    ann_file='real_drone_ann/train_visible.json',  # 无监督标注文件
    data_prefix=dict(img='real_drone_rgb/'),  # 图像前缀目录
    pipeline=unsup_pipeline,  # 绑定无监督流水线
)  # 无监督数据集配置结束
# ------------------------------  # 分隔线
# 构建训练数据加载器并采用多源采样保持平衡  # 说明下方字典
train_dataloader = dict(  # 定义训练数据加载器
    batch_size=batch_size,  # 指定批量大小
    num_workers=num_workers,  # 指定线程数量
    persistent_workers=True,  # 启用持久化线程提升性能
    sampler=dict(  # 配置采样器
        type='GroupMultiSourceSampler', batch_size=batch_size, source_ratio=[1, 1]  # 多源采样保持监督与无监督均衡
    ),  # 采样器配置结束
    dataset=dict(type='ConcatDataset', datasets=[labeled_dataset, unlabeled_dataset]),  # 拼接监督与无监督数据集
)  # 训练数据加载器配置结束
# ------------------------------  # 分隔线
# 构建验证数据加载器  # 说明下方字典
val_dataloader = dict(  # 定义验证数据加载器
    batch_size=1,  # 验证阶段批量大小
    num_workers=8,  # 验证线程数量
    persistent_workers=True,  # 启用持久化线程
    drop_last=False,  # 不丢弃最后一个批次
    sampler=dict(type='DefaultSampler', shuffle=False),  # 使用默认采样器且不打乱顺序
    dataset=dict(  # 验证数据集配置
        type=dataset_type,  # 指定数据集类型
        data_root=data_root,  # 指定根目录
        metainfo=dict(classes=classes),  # 声明类别信息
        ann_file='real_drone_ann/val_visible.json',  # 验证标注文件
        data_prefix=dict(img='real_drone_rgb/'),  # 图像前缀目录
        test_mode=True,  # 启用测试模式
        filter_cfg=dict(filter_empty_gt=True),  # 过滤无标注样本
        pipeline=test_pipeline,  # 绑定测试流水线
    ),  # 验证数据集配置结束
)  # 验证数据加载器配置结束
# ------------------------------  # 分隔线
# 构建测试数据加载器  # 说明下方字典
test_dataloader = dict(  # 定义测试数据加载器
    batch_size=1,  # 测试阶段批量大小
    num_workers=8,  # 测试线程数量
    persistent_workers=True,  # 启用持久化线程
    drop_last=False,  # 不丢弃最后一个批次
    sampler=dict(type='DefaultSampler', shuffle=False),  # 使用默认采样器且不打乱顺序
    dataset=dict(  # 测试数据集配置
        type=dataset_type,  # 指定数据集类型
        data_root=data_root,  # 指定根目录
        metainfo=dict(classes=classes),  # 声明类别信息
        ann_file='real_drone_ann/test_visible.json',  # 测试标注文件
        data_prefix=dict(img='real_drone_rgb/'),  # 图像前缀目录
        test_mode=True,  # 启用测试模式
        filter_cfg=dict(filter_empty_gt=True),  # 过滤无标注样本
        pipeline=test_pipeline,  # 绑定测试流水线
    ),  # 测试数据集配置结束
)  # 测试数据加载器配置结束
# ------------------------------  # 分隔线
# 定义验证评估器  # 说明下方字典
val_evaluator = dict(  # 构造验证评估器
    type='CocoMetric',  # 指定使用COCO指标
    ann_file=data_root + 'real_drone_ann/val_visible.json',  # 指定验证标注文件路径
    metric='bbox',  # 计算边界框mAP
    format_only=False,  # 不仅导出结果同时计算指标
)  # 验证评估器配置结束
# ------------------------------  # 分隔线
# 定义测试评估器  # 说明下方字典
test_evaluator = dict(  # 构造测试评估器
    type='CocoMetric',  # 指定使用COCO指标
    ann_file=data_root + 'real_drone_ann/test_visible.json',  # 指定测试标注文件路径
    metric='bbox',  # 计算边界框mAP
    format_only=False,  # 不仅导出结果同时计算指标
)  # 测试评估器配置结束
