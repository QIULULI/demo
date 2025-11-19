# 中文注释：Stage-2 UDA配置，融合扩散教师并启用SS-DC模块
# 中文注释：引用Stage-1基础配置（扩散引导UDA、20k半监督日程与Sim→Real无人机数据集），相对路径需从当前文件起算
import os  # 中文注释：引入os模块以便通过环境变量灵活切换扩散教师路径

_base_ = [
    '../../../../DA/_base_/models/diffusion_guided_adaptation_faster_rcnn_r101_fpn.py',  # 中文注释：基础检测与扩散蒸馏结构（实际存在的Stage-1模型配置）
    '../../../../DA/_base_/da_setting/semi_20k.py',  # 中文注释：20k迭代的半监督训练调度（实际存在的Stage-1训练日程）
    '../../../../DA/_base_/datasets/sim_to_real/semi_drone_rgb_aug.py'  # 中文注释：Sim→Real无人机数据集配置（实际存在的Stage-1数据设置）
]

stage1_diff_teacher_config = os.environ.get(  # 中文注释：优先读取环境变量以适配不同服务器路径
    'STAGE1_DIFF_TEACHER_CONFIG',  # 中文注释：可选环境变量名称，部署时可export来自定义
    'DG/Ours/drone/fused_diff_teacher_stage1_A_rpn_roi.py'  # 中文注释：默认指向Stage-1扩散教师配置文件（需按需替换）
)  # 中文注释：配置路径变量定义结束
stage1_diff_teacher_ckpt = os.environ.get(  # 中文注释：同理读取环境变量为权重路径提供默认值
    'STAGE1_DIFF_TEACHER_CKPT',  # 中文注释：可选环境变量名称，未设置时回落至示例权重
    'work_dirs/DG/Ours/drone/fused_teacher_stage1_A/best_coco_bbox_mAP_50_iter_20000.pth'  # 中文注释：Stage-1教师最佳权重示例
)  # 中文注释：权重路径变量定义结束

# 中文注释：读取基础模型配置并覆盖关键字段
detector = _base_.model  # 中文注释：从基础配置中取得模型字典
classes = ('drone',)  # 中文注释：显式声明类别元组方便下游组件复用
detector.detector.roi_head.bbox_head.num_classes = 1  # 中文注释：任务为单类无人机检测
detector.detector.init_cfg = dict(type='Pretrained', checkpoint='work_dirs/DG/Ours/drone/student_rgb_fused.pth')  # 中文注释：加载Stage-1学生权重作为初始化
detector.diff_model.config = stage1_diff_teacher_config  # 中文注释：指向Stage-1扩散教师配置（默认值可通过环境变量或直接修改替换）
detector.diff_model.pretrained_model = stage1_diff_teacher_ckpt  # 中文注释：指向Stage-1扩散教师权重（默认值为示例路径）
ssdc_schedule = dict(  # 中文注释：集中定义SS-DC损失调度以便骨干与训练阶段共享
    w_decouple=[(0, 0.1), (6000, 0.5)],  # 中文注释：解耦损失权重在0到6000迭代间线性由0.1升至0.5
    w_couple=[(2000, 0.2), (10000, 0.5)],  # 中文注释：耦合损失权重在2000到10000迭代间从0.2提升到0.5
    w_di_consistency=0.3,  # 中文注释：域不变一致性损失采用固定0.3权重
    consistency_gate=[(0, 0.9), (12000, 0.6)]  # 中文注释：DI一致性阈值从0.9逐步降至0.6以放宽伪标签筛选
)  # 中文注释：调度字典定义结束

detector.detector.backbone.enable_ssdc = True  # 中文注释：直接在骨干网络启用SS-DC路径以满足新版要求
detector.detector.backbone.ssdc_cfg = dict(  # 中文注释：为骨干提供完整SS-DC子配置
    enable_ssdc=True,  # 中文注释：再次显式打开子配置中的开关以兼容多重来源
    said_filter=dict(type='SAIDFilterBank'),  # 中文注释：使用默认SAID滤波器实现即可在FPN特征上提取频段
    coupling_neck=dict(type='SSDCouplingNeck', use_ds_tokens=True),  # 中文注释：耦合颈设置启用域特异token满足既有逻辑
    loss_decouple=dict(type='LossDecouple', loss_weight=1.0),  # 中文注释：保持解耦损失类型与权重为通用默认值
    loss_couple=dict(type='LossCouple', loss_weight=1.0),  # 中文注释：保持耦合损失类型与权重为通用默认值
    w_decouple=ssdc_schedule['w_decouple'],  # 中文注释：引用共享调度以确保前向构建与训练调度一致
    w_couple=ssdc_schedule['w_couple'],  # 中文注释：耦合权重调度亦复用共享定义保持一致
    w_di_consistency=ssdc_schedule['w_di_consistency'],  # 中文注释：域不变一致性权重同步骨干与训练阶段
    consistency_gate=ssdc_schedule['consistency_gate']  # 中文注释：一致性阈值调度保持统一来源防止偏差
)  # 中文注释：骨干SS-DC配置结束

# 中文注释：包装DomainAdaptationDetector并指定训练超参
model = dict(
    _delete_=True,  # 中文注释：删除基础同名字段以避免重复
    type='DomainAdaptationDetector',  # 中文注释：使用域自适应包装器管理学生/教师
    detector=detector,  # 中文注释：传入上方定义的检测模型
    data_preprocessor=dict(
        type='MultiBranchDataPreprocessor',  # 中文注释：多分支预处理以兼容监督与非监督输入
        data_preprocessor=detector.data_preprocessor  # 中文注释：复用检测器的标准化配置
    ),
    train_cfg=dict(
        detector_cfg=dict(type='SemiBaseDiff', burn_up_iters=2000),  # 中文注释：采用半监督扩散流程并设置2000步预热
        ssdc_cfg=dict(  # 中文注释：训练阶段复用骨干共享调度以保持损失权重一致
            w_decouple=ssdc_schedule['w_decouple'],  # 中文注释：解耦损失权重调度引用共享字典避免重复配置
            w_couple=ssdc_schedule['w_couple'],  # 中文注释：耦合损失权重调度同样引用共享字典
            w_di_consistency=ssdc_schedule['w_di_consistency'],  # 中文注释：域不变一致性权重同步共享定义
            consistency_gate=ssdc_schedule['consistency_gate']  # 中文注释：一致性门控阈值亦保持共享配置
        ),
        feature_loss_cfg=dict(  # 中文注释：补充特征蒸馏配置以满足DomainAdaptationDetector初始化需求
            feature_loss_type='mse',  # 中文注释：设置主教师特征蒸馏损失类型为MSE保证稳定
            feature_loss_weight=1.0,  # 中文注释：指定主教师特征蒸馏损失权重为1.0作为安全默认值
            cross_feature_loss_weight=0.0,  # 中文注释：默认关闭交叉特征蒸馏可按需升高该权重
            cross_consistency_cfg=dict(  # 中文注释：为交叉一致性分支提供显式字典避免读取空值
                cls_weight=0.0,  # 中文注释：默认分类一致性权重为0保持关闭状态
                reg_weight=0.0  # 中文注释：默认回归一致性权重为0保持关闭状态
            )
        )
    )
)

# 中文注释：小型自检代码（仅导入与前向张量）
if __name__ == '__main__':
    from mmengine import Config  # 中文注释：导入配置解析器
    cfg = Config.fromfile(__file__)  # 中文注释：载入当前配置文件
    print(cfg.model['type'])  # 中文注释：打印模型类型确认解析成功
    print(cfg.model.detector.detector.backbone.enable_ssdc)  # 中文注释：额外打印骨干SS-DC开关确保已按需开启
    print(cfg.model.detector.diff_model.config)  # 中文注释：打印扩散教师配置路径确认注入成功
    print(cfg.model.detector.diff_model.pretrained_model)  # 中文注释：打印扩散教师权重路径确认注入成功
