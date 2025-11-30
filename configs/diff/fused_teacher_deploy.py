_base_ = [  # 中文注释：继承基础扩散Faster R-CNN模型定义
    '../_base_/models/faster-rcnn_diff_fpn.py',  # 中文注释：复用标准单分支扩散检测器结构
]  # 中文注释：列表结束

load_from = 'rgb_fused_teacher_only.pth'  # 中文注释：Stage1提取后的教师权重路径
classes = ('drone',)
model = dict(  # 中文注释：模型配置总字典
    type='DiffusionDetector',  # 中文注释：使用单分支扩散检测器类
    init_cfg=dict(type='Pretrained', checkpoint=load_from),  # 中文注释：通过init_cfg加载冻结教师权重
    # ==== 新增：让 diffusion teacher 跑 SS-DC，只用来算 F_inv ====
    enable_ssdc=True,
    ssdc_cfg=dict(
        enable_ssdc=True,        # 打开 SS-DC 流程
        skip_local_loss=True,    # teacher 自己不在内部算 SS-DC 的 loss
        w_decouple=1.0,          # >0 才会在 extract_feat 里真正跑 SAID
        w_couple=0.0,            # 0 -> 只分解，不耦合，检测头输入还是原始 FPN
        w_di_consistency=0.0,    # teacher 自己不做 DI 一致性
        burn_in_iters=0,         # 不需要 burn-in
        # said_filter / coupling_neck / loss_decouple / loss_couple
        # 不写就沿用 _base_ 里的默认配置
    ),
    # ★ 1）backbone 里把类别标签改成 drone 单类
    backbone=dict(
        diff_config=dict(
            classes=classes,
        )
    ),
    # ★ 2）RPN 的 anchor 配置改成 Stage1 的无人机特化版本
    rpn_head=dict(
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[2, 4, 8],
            ratios=[0.33, 0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64],
        )
    ),
    # ★ 3）RoI Head 的 num_classes 改成 1
    roi_head=dict(
        bbox_head=dict(
            num_classes=len(classes),
        )
    ),
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
