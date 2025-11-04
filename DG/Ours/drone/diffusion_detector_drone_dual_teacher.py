# -*- coding: utf-8 -*-  # 指定文件编码防止中文注释产生乱码
"""真实RGB双模态扩散教师的独立模型配置，仅保留教师结构供蒸馏阶段加载。"""  # 顶部文档字符串概述配置用途

_base_ = [  # 声明需要继承的基础配置列表
    '../../_base_/models/faster-rcnn_diff_fpn.py',  # 继承扩散版Faster R-CNN模型定义以复用主干结构
    '../../_base_/dg_setting/dg_20k.py',  # 继承20k迭代的训练调度以便单独训练教师时复用默认策略
]  # 结束基础配置列表定义

classes = ('drone',)  # 指定单类别为无人机以匹配无人机场景

model = dict(  # 重写模型结构以适配双模态教师的输出需求
    roi_head=dict(  # 调整ROI头相关配置
        bbox_head=dict(num_classes=len(classes))),  # 将ROI分类头的类别数设置为无人机类别数量
    backbone=dict(  # 重设骨干网络的扩散配置
        diff_config=dict(classes=classes)),  # 将类别信息写入扩散骨干以对齐类别嵌入
    rpn_head=dict(  # 自定义RPN头以提升小目标召回
        anchor_generator=dict(  # 调整锚框生成策略
            type='AnchorGenerator',  # 指定锚框生成器类型
            scales=[2, 4, 8],  # 使用较小尺度捕捉微小目标
            ratios=[0.33, 0.5, 1.0, 2.0],  # 提供多样纵横比应对不同外形
            strides=[4, 8, 16, 32, 64],  # 明确各FPN层级的步长对应关系
        ),  # 结束锚框生成器配置
    ),  # 结束RPN头配置
)  # 结束教师模型配置

# 当该配置被扩散教师加载器读取时，将直接构建与阶段一训练一致的教师结构
