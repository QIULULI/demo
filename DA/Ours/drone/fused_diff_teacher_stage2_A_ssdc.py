# DA/Ours/drone/fused_diff_teacher_stage2_A_ssdc.py  # 中文注释：定义阶段2使用融合Diff教师的Drone配置
import copy  # 中文注释：导入copy用于深拷贝配置避免原始对象被修改

_base_ = [  # 中文注释：继承基础配置列表
    '../../_base_/models/diffusion_guided_adaptation_faster_rcnn_r101_fpn.py',  # 中文注释：基础扩散-检测模型结构
    '../../_base_/da_setting/semi_20k.py',  # 中文注释：2 万迭代半监督/域自适应训练日程
    '../../_base_/datasets/sim_to_real/semi_drone_rgb_aug.py',  # 中文注释：仿真 RGB → 真实 RGB 数据集定义
]  # 中文注释：结束基础配置

detector = _base_.model  # 中文注释：从基础配置拷贝域自适应检测器
inner_det = detector['detector']  # 中文注释：获取半监督框架内部的学生/教师检测器配置
inner_det['roi_head']['bbox_head']['num_classes'] = 1  # 中文注释：将类别数设置为无人机单类任务
inner_det['init_cfg'] = dict(  # 中文注释：设置学生初始化权重保持与Stage-1对齐
    type='Pretrained',  # 中文注释：使用预训练权重初始化学生
    checkpoint='best_coco_bbox_mAP_50_iter_20000.pth')  # 中文注释：指定Stage-1学生权重路径
levels = ('P5',) # 中文注释：定义要处理的FPN层级
ssdc_runtime_cfg = dict(  # 中文注释：整理SS-DC运行期配置以匹配SSDCFasterRCNN签名
    enable_ssdc=True,  # 中文注释：显式开启SS-DC以触发相关模块构建
    skip_local_loss=False,  # 中文注释：默认不跳过本地SS-DC损失可由包装器覆盖
    said=dict(type='SAIDFilterBank', levels=levels),  # 中文注释：指定SAID滤波器类型保持最小可用配置
    coupling=dict(  # 中文注释：指定耦合颈部配置
        type='SSDCouplingNeck',  # 中文注释：使用SS-DC耦合颈部实现
        use_ds_tokens=False,  # 中文注释：关闭域特异令牌以降低显存
        num_heads=1,  # 中文注释：多头注意力头数可按显存调整
        max_q_chunk=8),  # 中文注释：查询分块大小平衡显存与速度
    loss_decouple=dict(type='LossDecouple'),  # 中文注释：设置解耦损失最小默认配置
    loss_couple=dict(type='LossCouple'),  # 中文注释：设置耦合损失最小默认配置
    w_decouple=[(0, 0.1), (6000, 0.5)],  # 中文注释：复制阶段性解耦权重调度
    w_couple=[(2000, 0.2), (10000, 0.5)],  # 中文注释：复制阶段性耦合权重调度
    w_di_consistency=0.3,  # 中文注释：设置域不变一致性损失权重
    consistency_gate=[(0, 0.9), (12000, 0.6)],  # 中文注释：设置伪标签余弦门限调度
    burn_in_iters=500,  # 中文注释：默认无额外SS-DC烧入步保持向后兼容
    use_coupled_feature=True)  # 中文注释：下游检测头使用耦合后的特征
inner_det.update(  # 中文注释：将内部检测器切换为支持SS-DC的实现
    type='SSDCFasterRCNN',  # 中文注释：指定自定义检测器类型
    enable_ssdc=ssdc_runtime_cfg['enable_ssdc'],  # 中文注释：透传SS-DC开关
    ssdc_cfg=copy.deepcopy(ssdc_runtime_cfg))  # 中文注释：深拷贝运行期配置避免共享引用
inner_det.setdefault('train_cfg', {}).setdefault(  # 中文注释：确保训练配置存在
    'ssdc_cfg', copy.deepcopy(ssdc_runtime_cfg))  # 中文注释：在训练配置中同步写入SS-DC字段便于优先级合并
detector.data_preprocessor = inner_det['data_preprocessor']  # 中文注释：复用内部检测器的数据预处理配置保持一致

detector['diff_model'].update(  # 中文注释：配置扩散教师路径与冻结策略
    config='configs/diff/fused_teacher_deploy.py',  # 中文注释：Stage-1 融合DIFF教师配置
    pretrained_model='rgb_fused_teacher_only.pth',  # 中文注释：Stage-1教师权重
    freeze_grad=True)  # 中文注释：完全冻结DIFF教师梯度以节省显存

model = dict(  # 中文注释：最外层模型封装
    _delete_=True,  # 中文注释：删除并重写基础字段
    type='DomainAdaptationDetector',  # 中文注释：域自适应检测封装
    detector=detector,  # 中文注释：注入学生-教师-扩散联合模型
    data_preprocessor=dict(  # 中文注释：多分支数据预处理
        type='MultiBranchDataPreprocessor',  # 中文注释：区分监督/无监督分支
        data_preprocessor=detector.data_preprocessor),  # 中文注释：复用相同归一化
    train_cfg=dict(  # 中文注释：训练阶段设置
        detector_cfg=dict(type='SemiBaseDiff', burn_up_iters=500),  # 中文注释：半监督扩散框架与零热身
        ssdc_cfg=dict(  # 中文注释：SS-DC 专用超参与调度
            w_decouple=[(0, 0.1), (6000, 0.5)],  # 中文注释：频域解耦损失权重线性爬升
            w_couple=[(2000, 0.2), (10000, 0.5)],  # 中文注释：谱-空耦合对齐权重热身后启动
            w_di_consistency=0.3,  # 中文注释：学生/教师 DI 一致性恒定权重
            consistency_gate=[(0, 0.9), (12000, 0.6)]),  # 中文注释：伪标签 DI 余弦阈值由严到松
        feature_loss_cfg=dict(  # 中文注释：兼容原有特征蒸馏开关
            feature_loss_type='mse',  # 中文注释：蒸馏类型
            feature_loss_weight=1.0)))  # 中文注释：蒸馏权重
__all__ = ['_base_', 'model']  # 中文注释：仅导出基础列表与模型字典避免额外符号泄漏

# 中文注释：以下为快速自检示例，验证配置可被解析
# from mmengine import Config
# cfg = Config.fromfile('DA/Ours/drone/fused_diff_teacher_stage2_A_ssdc.py')
# print(cfg.model['detector']['detector']['type'])
