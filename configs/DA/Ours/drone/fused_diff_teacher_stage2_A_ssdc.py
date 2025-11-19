# 中文注释：Stage-2 UDA配置，融合扩散教师并启用SS-DC模块
# 中文注释：引用Stage-1基础配置（扩散引导UDA、20k半监督日程与Sim→Real无人机数据集），相对路径需从当前文件起算
_base_ = [
    '../../../../DA/_base_/models/diffusion_guided_adaptation_faster_rcnn_r101_fpn.py',  # 中文注释：基础检测与扩散蒸馏结构（实际存在的Stage-1模型配置）
    '../../../../DA/_base_/da_setting/semi_20k.py',  # 中文注释：20k迭代的半监督训练调度（实际存在的Stage-1训练日程）
    '../../../../DA/_base_/datasets/sim_to_real/semi_drone_rgb_aug.py'  # 中文注释：Sim→Real无人机数据集配置（实际存在的Stage-1数据设置）
]

# 中文注释：读取基础模型配置并覆盖关键字段
detector = _base_.model  # 中文注释：从基础配置中取得模型字典
classes = ('drone',)  # 中文注释：显式声明类别元组方便下游组件复用
detector.detector.roi_head.bbox_head.num_classes = 1  # 中文注释：任务为单类无人机检测
detector.detector.init_cfg = dict(type='Pretrained', checkpoint='work_dirs/DG/Ours/drone/student_rgb_fused.pth')  # 中文注释：加载Stage-1学生权重作为初始化
detector.detector.enable_ssdc = True  # 中文注释：开启SS-DC模块构建SAID与耦合颈
detector.detector.use_ds_tokens = True  # 中文注释：启用域特异token以支撑解耦
detector.detector.backbone.setdefault('ssdc_cfg', dict())  # 中文注释：确保骨干网络具备SS-DC子配置容器方便写入跳过开关
detector.detector.backbone.ssdc_cfg['skip_local_loss'] = True  # 中文注释：显式要求学生模型在本地损失阶段跳过SS-DC计算

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
        ssdc_cfg=dict(  # 中文注释：SS-DC相关超参及调度
            w_decouple=[(0, 0.1), (6000, 0.5)],  # 中文注释：解耦损失权重线性从0.1升至0.5
            w_couple=[(2000, 0.2), (10000, 0.5)],  # 中文注释：耦合损失权重在预热后逐步增大
            w_di_consistency=0.3,  # 中文注释：域不变一致性损失的常数权重
            consistency_gate=[(0, 0.9), (12000, 0.6)]  # 中文注释：伪标签DI相似度阈值线性下降
        )
    )
)

# 中文注释：小型自检代码（仅导入与前向张量）
if __name__ == '__main__':
    from mmengine import Config  # 中文注释：导入配置解析器
    cfg = Config.fromfile(__file__)  # 中文注释：载入当前配置文件
    print(cfg.model['type'])  # 中文注释：打印模型类型确认解析成功
