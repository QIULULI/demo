# DA/Ours/drone/fused_diff_teacher_stage2_A_ssdc.py  # 定义阶段2使用融合Diff教师的Drone配置
import copy  # 导入copy用于深拷贝配置避免原始对象被修改

_base_ = [  # 继承基础配置列表
    '../../_base_/models/diffusion_guided_adaptation_faster_rcnn_r101_fpn.py',  # 基础扩散-检测模型结构
    '../../_base_/da_setting/semi_20k.py',  # 2 万迭代半监督/域自适应训练日程
    '../../_base_/datasets/sim_to_real/semi_drone_rgb_aug.py',  # 仿真 RGB → 真实 RGB 数据集定义
]  # 结束基础配置
burn_cross = 0  # 定义burn_up_iters以便后续引用
burn_ssdc = 0  # 定义SS-DC烧入步数以便后续引用
detector = _base_.model  # 从基础配置拷贝域自适应检测器
inner_det = detector['detector']  # 获取半监督框架内部的学生/教师检测器配置
inner_det['roi_head']['bbox_head']['num_classes'] = 1  # 将类别数设置为无人机单类任务

ssdc_runtime_cfg = dict(  # 整理SS-DC运行期配置以匹配SSDCFasterRCNN签名
    enable_ssdc=True,  # 显式开启SS-DC以触发相关模块构建
    compute_in_wrapper=True,          # -> DomainAdaptationDetector 来调度
    skip_student_ssdc_loss=True,      # 学生内部不再自己算 LossDecouple/LossCouple
    skip_local_loss=True,             # 会在 _propagate_ssdc_skip_flags 里下发到 student/teacher/diff_detector
    said=dict(type='SAIDFilterBank',),  # 指定SAID滤波器类型保持最小可用配置
    coupling=dict(  # 指定耦合颈部配置
        type='SSDCouplingNeck',  # 使用SS-DC耦合颈部实现
        levels=('P4','P5'),  # 指定处理的FPN层级以节省显存
        use_ds_tokens=True,  # 关闭域特异令牌以降低显存
        num_heads=64,  # 多头注意力头数可按显存调整
        max_q_chunk=256),  # 查询分块大小平衡显存与速度
    loss_decouple=dict(type='LossDecouple'),  # 设置解耦损失最小默认配置
    loss_couple=dict(type='LossCouple'),  # 设置耦合损失最小默认配置
    w_decouple=[(0, 0.1), (6000, 0.5)],  # 复制阶段性解耦权重调度
    w_couple=[(2000, 0.2), (10000, 0.5)],  # 复制阶段性耦合权重调度
    w_di_consistency=0.3,  # 设置域不变一致性损失权重
    consistency_gate=[(0, 0.3), (12000, 0.5)],  # 设置伪标签余弦门限调度
    burn_in_iters=burn_ssdc,  # 默认无额外SS-DC烧入步保持向后兼容
    use_coupled_feature=True)  # 下游检测头使用耦合后的特征
inner_det.update(  # 将内部检测器切换为支持SS-DC的实现
    type='SSDCFasterRCNN',  # 指定自定义检测器类型
    enable_ssdc=ssdc_runtime_cfg['enable_ssdc'],  # 透传SS-DC开关
    ssdc_cfg=copy.deepcopy(ssdc_runtime_cfg))  # 深拷贝运行期配置避免共享引用
inner_det.setdefault('train_cfg', {}).setdefault(  # 确保训练配置存在
    'ssdc_cfg', copy.deepcopy(ssdc_runtime_cfg))  # 在训练配置中同步写入SS-DC字段便于优先级合并
inner_det['rpn_head']['anchor_generator'] = dict(
    type='AnchorGenerator',
    scales=[2, 4, 8],               # 与 Stage-1 保持一致
    ratios=[0.33, 0.5, 1.0, 2.0],   # 与 Stage-1 保持一致
    strides=[4, 8, 16, 32, 64]
)
detector.data_preprocessor = inner_det['data_preprocessor']  # 复用内部检测器的数据预处理配置保持一致

detector['diff_model'].update(  # 配置扩散教师路径与冻结策略
    # config='configs/diff/fused_teacher_deploy.py',  # Stage-1 融合DIFF教师配置
    # pretrained_model='rgb_fused_teacher_only.pth',  # Stage-1教师权重
    config='work_dirs/DG/Ours/drone/fused_teacher_stage1_A/fused_diff_teacher_stage1_A_rpn_roi.py',  # Stage-1 融合DIFF教师配置
    pretrained_model='work_dirs/DG/Ours/drone/fused_teacher_stage1_A/rgb_fused1111.pth',  # Stage-1教师权重
    freeze_grad=True)  # 完全冻结DIFF教师梯度以节省显存

# # NOTE:
# #   Base config defaults to predict_on='teacher'. After removing the student-EMA
# #   teacher hook in Stage-2, the teacher branch will stay at its initial weights
# #   (almost never what you want for validation). So we explicitly set it here.
# #   You can switch to 'student' if you prefer evaluating the student detector.
# detector.setdefault('semi_test_cfg', {})
# detector['semi_test_cfg']['predict_on'] = 'diff_detector'

model = dict(  # 最外层模型封装
    _delete_=True,  # 删除并重写基础字段
    type='DomainAdaptationDetector',  # 域自适应检测封装
    detector=detector,  # 注入学生-教师-扩散联合模型
    data_preprocessor=dict(  # 多分支数据预处理
        type='MultiBranchDataPreprocessor',  # 区分监督/无监督分支
        data_preprocessor=detector.data_preprocessor),  # 复用相同归一化
    train_cfg=dict(  # 训练阶段设置
        detector_cfg=dict(type='SemiBaseDiff', burn_up_iters=burn_cross),  # 半监督扩散框架与零热身
        ssdc_cfg=copy.deepcopy(ssdc_runtime_cfg),
        feature_loss_cfg=dict(  # 兼容原有特征蒸馏开关
            feature_loss_type='mse',  # 蒸馏类型
            feature_loss_weight=1.0)))  # 蒸馏权重

# -------------------------------------------------------------------------
# Stage-2 core refactor (IMPORTANT):
#
# Replace the default AdaptiveTeacherHook (EMA on model.teacher)
# with an EMA update on the *real diffusion teacher* (diff_detector).
# Only update detection heads (RPN/ROI); keep diffusion encoder + fused FPN
# untouched to preserve IR prior.

custom_hooks = [
    # ① 保留：student copy 的 EMA teacher（model.teacher）
    dict(type='AdaptiveTeacherHook', momentum=0.0004, interval=1, skip_buffer=True),

    # ② 新增：对 diff teacher 做 EMA（只更新 head）
    dict(
        type='DiffTeacherHeadEMAHook',
        momentum=0.0004,
        interval=1,
        burn_up_iters=burn_cross,  # 让 student 先预热，再开始更新 diff_teacher head（你要的预热逻辑）
        target_modules=('rpn_head', 'roi_head'),
        skip_buffers=True,
        init_from_student=False,
        update_all_diff_teachers=False,
        strict=True,
    ),

    # ③ 可选：保持你项目里原本的导出逻辑
    dict(type='StudentToTeacherExportHook'),
]

__all__ = ['_base_', 'model', 'custom_hooks']  # 导出自定义钩子以覆盖base schedule的默认hook
