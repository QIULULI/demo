# 数据集基础配置文件：定义无人机IR/RGB双模态训练集与验证集
from copy import deepcopy  # 引入deepcopy用于生成独立流水线副本
dataset_type = 'CocoDataset'  # 指定数据集类型为COCO格式
data_root = 'data/'  # 定义统一的数据根目录
classes = ('drone',)  # 仅包含无人机单一类别

backend_args = None  # 图像加载默认使用mmcv后端

ir_img_prefix = '/mnt/ssd/lql/Fitness-Generalization-Transferability/data/sim_drone_ir/Town01_Opt/carla_data'  # 红外图像前缀路径
rgb_img_prefix = '/mnt/ssd/lql/Fitness-Generalization-Transferability/data/sim_drone_rgb/Town01_Opt/carla_data'  # 可见光图像前缀路径

ir_repeat = 1  # 红外数据重复次数，可通过修改该值控制采样比例
rgb_repeat = 1  # 可见光数据重复次数，可通过修改该值控制采样比例

color_space_light = [  # 轻量级颜色增强空间定义
    [dict(type='AutoContrast')],  # 自动对比度增强
    [dict(type='Equalize')],  # 直方图均衡化增强
    [dict(type='Color')],  # 饱和度增强
    [dict(type='Contrast')],  # 对比度增强
    [dict(type='Brightness')],  # 亮度增强
    [dict(type='Sharpness')],  # 锐度增强
]  # 结束颜色增强空间

train_pipeline_template = [  # 定义训练阶段通用数据处理模板流水线
    dict(type='LoadImageFromFile', backend_args=backend_args),  # 加载图像文件
    dict(type='LoadAnnotations', with_bbox=True),  # 加载带边界框标注
    dict(type='Resize', scale=(1600, 960), keep_ratio=True),  # 等比例缩放到统一分辨率
    dict(type='RandomCrop', crop_type='absolute', crop_size=(640, 640),  # 随机裁剪窗口
         recompute_bbox=True, allow_negative_crop=True),  # 裁剪后重算标注并允许无目标块
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1e-2, 1e-2)),  # 过滤过小目标
    dict(type='RandomFlip', prob=0.5),  # 随机水平翻转
    dict(type='RandAugment', aug_space=color_space_light, aug_num=1),  # 轻量随机增强
]  # 结束训练通用流水线模板定义

def build_train_pipeline(sensor_tag):  # 定义函数用于根据传感器标签生成完整训练流水线
    pipeline = deepcopy(train_pipeline_template)  # 深拷贝模板确保不同模态互不影响
    pipeline.append(dict(type='SetSensorTag', sensor=sensor_tag))  # 插入SetSensorTag以写入当前样本的传感器标记
    pipeline.append(  # 在流水线末尾追加打包组件并扩展元信息字段
        dict(type='PackDetInputs',  # 打包检测模型输入
             meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',  # 保留关键元信息
                        'scale_factor', 'flip', 'flip_direction',  # 记录几何变换信息
                        'homography_matrix', 'sensor'))  # 同时保留单应矩阵与传感器标签
    )  # PackDetInputs配置结束
    return pipeline  # 返回构建完成的流水线

train_pipeline_ir = build_train_pipeline(sensor_tag='sim_ir')  # 构建红外训练流水线并写入仿真IR标签
train_pipeline_rgb = build_train_pipeline(sensor_tag='sim_rgb')  # 构建可见光训练流水线并写入仿真RGB标签

test_pipeline = [  # 定义验证/测试流水线
    dict(type='LoadImageFromFile', backend_args=backend_args),  # 加载图像
    dict(type='Resize', scale=(1600, 960), keep_ratio=True),  # 调整尺寸
    dict(type='LoadAnnotations', with_bbox=True),  # 加载标注便于评估
    dict(type='SetSensorTag', sensor='sim_rgb'),  # 写入验证集默认仿真RGB传感器标签
    dict(type='PackDetInputs',  # 打包评估输入
         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor', 'sensor')),  # 保留必要元信息并包含传感器字段
]  # 结束验证流水线

train_dataset = dict(  # 构建训练数据集组合
    type='ConcatDataset',  # 使用串联数据集拼接两种模态
    #separate_eval=False,  # 训练阶段无需区分评估指标
    datasets=[  # 定义具体子数据集
        dict(  # 红外数据分支配置
            type='RepeatDataset',  # 通过重复实现采样比例控制
            times=ir_repeat,  # 红外重复次数
            dataset=dict(  # 红外COCO数据集配置
                type=dataset_type,  # 指定数据集类型
                data_root=data_root,  # 指定数据根目录
                metainfo=dict(classes=classes),  # 写入类别信息
                ann_file='sim_drone_ann/ir/train.json',  # 红外训练标注文件
                data_prefix=dict(img=ir_img_prefix),  # 红外图像前缀
                filter_cfg=dict(filter_empty_gt=True),  # 过滤无目标图像
                pipeline=train_pipeline_ir)),  # 引用红外专属流水线
        dict(  # 可见光数据分支配置
            type='RepeatDataset',  # 使用重复机制
            times=rgb_repeat,  # 可见光重复次数
            dataset=dict(  # 可见光COCO数据集
                type=dataset_type,  # 指定数据集类型
                data_root=data_root,  # 指定数据根目录
                metainfo=dict(classes=classes),  # 写入类别信息
                ann_file='sim_drone_ann/rgb/train.json',  # 可见光训练标注文件
                data_prefix=dict(img=rgb_img_prefix),  # 可见光图像前缀
                filter_cfg=dict(filter_empty_gt=True),  # 过滤无目标图像
                pipeline=train_pipeline_rgb))  # 引用可见光专属流水线
    ])  # 结束数据集列表定义

val_dataset = dict(  # 构建验证数据集
    type=dataset_type,  # 指定数据集类型
    data_root=data_root,  # 指定数据根目录
    metainfo=dict(classes=classes),  # 写入类别信息
    ann_file='sim_drone_ann/rgb/val.json',  # 验证标注文件路径
    data_prefix=dict(img=rgb_img_prefix),  # 默认使用红外模态进行评估
    test_mode=True,  # 以测试模式加载
    filter_cfg=dict(filter_empty_gt=True),  # 过滤无目标图片
    pipeline=test_pipeline)  # 引用验证流水线

train_dataloader = dict(  # 定义训练数据加载器
    batch_size=8,  # 单卡批大小
    num_workers=8,  # 数据加载线程数
    persistent_workers=True,  # 启用持久化线程
    sampler=dict(type='DefaultSampler', shuffle=True),  # 默认随机采样
    batch_sampler=dict(type='AspectRatioBatchSampler'),  # 保持长宽比分组
    dataset=train_dataset)  # 传入构建好的组合数据集

val_dataloader = dict(  # 定义验证数据加载器
    batch_size=1,  # 验证阶段单张推理
    num_workers=8,  # 数据加载线程数
    persistent_workers=True,  # 启用持久化线程
    drop_last=False,  # 不丢弃最后一批
    sampler=dict(type='DefaultSampler', shuffle=False),  # 顺序采样
    dataset=val_dataset)  # 传入验证数据集

test_dataloader = val_dataloader  # 复用验证加载器作为测试加载器

val_evaluator = dict(  # 构建验证评估器
    type='CocoMetric',  # 使用COCO指标
    ann_file=data_root + 'sim_drone_ann/rgb/val.json',  # 指定评估标注路径
    metric='bbox',  # 关注检测框指标
    format_only=False)  # 同时输出指标结果

test_evaluator = val_evaluator  # 测试阶段复用验证评估器
