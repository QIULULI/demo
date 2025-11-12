# DA/Ours/drone/diffusion_detector_drone.py

# 继承两阶段 DIFF 检测器的基础结构
_base_ = [
    '../../_base_/models/faster-rcnn_diff_fpn.py',
]

# 从基础模型中取出定义
detector = _base_.model

# 修改类别为无人机单类
detector.roi_head.bbox_head.num_classes = 1
detector.backbone.diff_config.classes = ('drone',)

# 调整 RPN 锚框生成器以适应小目标无人机
detector.rpn_head.anchor_generator = dict(
    type='AnchorGenerator',
    scales=[2, 4, 8],
    ratios=[0.33, 0.5, 1.0, 2.0],
    strides=[4, 8, 16, 32, 64],
)

# 保留其它字段（如 model_id）由基础配置定义，不做更改
model = detector
