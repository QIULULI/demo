_base_ = [  # 中文注释：继承基础扩散Faster R-CNN模型定义
    '../_base_/models/faster-rcnn_diff_fpn.py',  # 中文注释：复用标准单分支扩散检测器结构
]  # 中文注释：列表结束

load_from = 'rgb_fused_teacher_only.pth'  # 中文注释：Stage1提取后的教师权重路径

model = dict(  # 中文注释：模型配置总字典
    type='DiffusionDetector',  # 中文注释：使用单分支扩散检测器类
    init_cfg=dict(type='Pretrained', checkpoint=load_from),  # 中文注释：通过init_cfg加载冻结教师权重
    auxiliary_branch_cfg=dict(  # 中文注释：禁用辅助分支同时保留必需字段以兼容构造函数
        apply_auxiliary_branch=False,  # 中文注释：关闭参考分支与蒸馏逻辑确保单分支推理
        loss_cls_kd=dict(  # 中文注释：占位的分类蒸馏损失配置避免缺键报错
            type='KnowledgeDistillationKLDivLoss',  # 中文注释：沿用默认损失类型
            class_reduction='mean',  # 中文注释：保持与基线一致的归约方式
            T=1,  # 中文注释：占位温度参数未启用
            loss_weight=0.0),  # 中文注释：权重设0确保即便被调用也不影响结果
        loss_reg_kd=dict(type='L1Loss', loss_weight=0.0),  # 中文注释：占位回归蒸馏损失同样设为0权重
    ),  # 中文注释：辅助配置结束
)  # 中文注释：模型配置结束

# 小型自检示例（供REPL复制）：  # 中文注释：提供快速配置加载与前向检查参考
# >>> from mmengine import Config  # 中文注释：导入配置解析器
# >>> from mmdet.utils import register_all_modules  # 中文注释：注册组件确保构建成功
# >>> from mmdet.registry import MODELS  # 中文注释：导入模型注册表
# >>> import torch  # 中文注释：导入PyTorch构造假输入
# >>> cfg = Config.fromfile('configs/diff/fused_teacher_deploy.py')  # 中文注释：加载当前部署配置
# >>> register_all_modules()  # 中文注释：注册模型与算子
# >>> model = MODELS.build(cfg.model)  # 中文注释：构建单分支扩散检测器
# >>> dummy = torch.randn(1, 3, 224, 224)  # 中文注释：构造小分辨率假输入降低显存占用
# >>> with torch.no_grad():  # 中文注释：关闭梯度以便快速验证
# ...     _ = model.extract_feat(dummy)  # 中文注释：执行一次特征提取确认链路连通
