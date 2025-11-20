# 中文注释：Stage-2 UDA配置，融合扩散教师并启用SS-DC模块
# 中文注释：引用Stage-1基础配置（扩散引导UDA、20k半监督日程与Sim→Real无人机数据集），相对路径需从当前文件起算
import os  # 中文注释：引入os模块以便通过环境变量灵活切换扩散教师路径
import copy  # 中文注释：引入copy模块以便安全复制基础检测器配置防止原地污染
import importlib.util  # 中文注释：引入importlib工具模块以便按文件路径动态加载带连字符的配置文件
from mmengine import Config  # 中文注释：引入Config解析器以便按路径读取基础配置对象

module_path = os.path.join(  # 中文注释：组合当前文件所在目录到基础模型配置的相对路径
    os.path.dirname(__file__),  # 中文注释：获取当前配置文件的目录
    '../../../../DA/_base_/models/faster-rcnn_diff_fpn.py'  # 中文注释：指向仓库根目录下DA/_base_/models的DiffusionDetector模板配置文件路径
)
spec = importlib.util.spec_from_file_location(  # 中文注释：基于文件路径创建模块规范以支持合法的模块名加载
    'faster_rcnn_diff_fpn',  # 中文注释：为动态模块指定合法的Python模块名称
    module_path  # 中文注释：提供实际的配置文件路径
)
faster_rcnn_module = importlib.util.module_from_spec(spec)  # 中文注释：根据模块规范创建模块对象
spec.loader.exec_module(faster_rcnn_module)  # 中文注释：执行模块以填充对象内容，确保model变量可用
diff_detector_template = faster_rcnn_module.model  # 中文注释：从动态加载的模块中取出包含DiffusionDetector与diff_config的基础模板

base_model_cfg_relative = '../../../../DA/_base_/models/diffusion_guided_adaptation_faster_rcnn_r101_fpn.py'  # 中文注释：定义基础SemiBaseDiffDetector配置的相对路径以便复用
base_model_cfg_path = os.path.join(os.path.dirname(__file__), base_model_cfg_relative)  # 中文注释：将相对路径转换为绝对路径确保Config.fromfile可用
base_cfg = Config.fromfile(base_model_cfg_path)  # 中文注释：读取基础配置对象以便安全取得model字段
_base_ = [
    base_model_cfg_relative,  # 中文注释：基础SemiBaseDiffDetector封装（保留半监督扩散蒸馏结构）
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
classes = ('drone',)  # 中文注释：显式声明类别元组方便下游组件复用
# 中文注释：从半监督扩散包装基础配置复制整体检测器配置
semibase_detector = copy.deepcopy(base_cfg.model)  # 中文注释：深拷贝基础配置中的SemiBaseDiffDetector定义避免污染原对象
# 中文注释：深拷贝DiffusionDetector模板以便在学生/教师内部复用且不污染原模板
ssdc_ready_diff_detector = copy.deepcopy(diff_detector_template)  # 中文注释：复制包含diff_config的扩散检测器
ssdc_ready_diff_detector.roi_head.bbox_head.num_classes = 1  # 中文注释：任务为单类无人机检测
ssdc_ready_diff_detector.backbone.diff_config['classes'] = classes  # 中文注释：更新扩散骨干的类别标签以匹配无人机任务
ssdc_ready_diff_detector.init_cfg = dict(  # 中文注释：在扩散检测器上设置权重初始化信息
    type='Pretrained',  # 中文注释：声明初始化方式为预训练模型加载
    checkpoint='work_dirs/DG/Ours/drone/student_rgb_fused.pth'  # 中文注释：加载Stage-1学生权重作为初始化
)
ssdc_ready_diff_detector.enable_ssdc = True  # 中文注释：直接在DiffusionDetector层级开启SS-DC开关
ssdc_schedule = dict(  # 中文注释：集中定义SS-DC损失调度以便骨干与训练阶段共享
    w_decouple=[(0, 0.1), (6000, 0.5)],  # 中文注释：解耦损失权重在0到6000迭代间线性由0.1升至0.5
    w_couple=[(0, 0.0), (2000, 0.2), (10000, 0.5)],  # 中文注释：耦合损失权重在0到1999迭代为0，2000到10000迭代由0.2提升至0.5
    w_di_consistency=0.3,  # 中文注释：域不变一致性损失采用固定0.3权重
    consistency_gate=[(0, 0.9), (12000, 0.6)],  # 中文注释：DI一致性阈值从0.9逐步降至0.6以放宽伪标签筛选
    burn_in_iters=2000  # 中文注释：设置耦合损失的预热迭代数为2000步以匹配Stage-2规划
)  # 中文注释：调度字典定义结束
ssdc_ready_diff_detector.ssdc_cfg = dict(  # 中文注释：将SS-DC相关配置显式挂载到DiffusionDetector层级
    enable_ssdc=True,  # 中文注释：在SS-DC子配置中开启模块开关以确保merge时生效
    said_filter=dict(type='SAIDFilterBank'),  # 中文注释：使用默认SAID滤波器实现即可在FPN特征上提取频段
    coupling_neck=dict(type='SSDCouplingNeck', use_ds_tokens=True),  # 中文注释：耦合颈设置启用域特异token满足既有逻辑
    loss_decouple=dict(type='LossDecouple', loss_weight=1.0),  # 中文注释：保持解耦损失类型与权重为通用默认值
    loss_couple=dict(type='LossCouple', loss_weight=1.0),  # 中文注释：保持耦合损失类型与权重为通用默认值
    w_decouple=ssdc_schedule['w_decouple'],  # 中文注释：引用共享调度以确保前向构建与训练调度一致
    w_couple=ssdc_schedule['w_couple'],  # 中文注释：耦合权重调度亦复用共享定义保持一致
    w_di_consistency=ssdc_schedule['w_di_consistency'],  # 中文注释：域不变一致性权重同步骨干与训练阶段
    consistency_gate=ssdc_schedule['consistency_gate']  # 中文注释：一致性阈值调度保持统一来源防止偏差
)  # 中文注释：SS-DC配置结束
# 中文注释：将准备好的DiffusionDetector嵌入半监督扩散包装器中以保证学生/教师使用DIFF骨干
semibase_detector.detector = ssdc_ready_diff_detector  # 中文注释：确保学生/教师骨干均为DIFF类型并携带SS-DC配置
semibase_detector.diff_model.config = stage1_diff_teacher_config  # 中文注释：指向Stage-1扩散教师配置（默认值可通过环境变量或直接修改替换）
semibase_detector.diff_model.pretrained_model = stage1_diff_teacher_ckpt  # 中文注释：指向Stage-1扩散教师权重（默认值为示例路径）

# 中文注释：包装DomainAdaptationDetector并指定训练超参
model = dict(
    _delete_=True,  # 中文注释：删除基础同名字段以避免重复
    type='DomainAdaptationDetector',  # 中文注释：使用域自适应包装器管理学生/教师
    detector=semibase_detector,  # 中文注释：传入上方定义的半监督扩散检测模型
    data_preprocessor=dict(
        type='MultiBranchDataPreprocessor',  # 中文注释：多分支预处理以兼容监督与非监督输入
        data_preprocessor=semibase_detector.data_preprocessor  # 中文注释：复用检测器的标准化配置
    ),
    train_cfg=dict(
        detector_cfg=dict(type='SemiBaseDiff', burn_up_iters=2000),  # 中文注释：采用半监督扩散流程并设置2000步预热
        ssdc_cfg=dict(  # 中文注释：训练阶段复用共享调度以保持损失权重一致
            w_decouple=ssdc_schedule['w_decouple'],  # 中文注释：解耦损失权重调度引用共享字典避免重复配置
            w_couple=ssdc_schedule['w_couple'],  # 中文注释：耦合损失权重调度同样引用共享字典
            w_di_consistency=ssdc_schedule['w_di_consistency'],  # 中文注释：域不变一致性权重同步共享定义
            consistency_gate=ssdc_schedule['consistency_gate'],  # 中文注释：一致性门控阈值亦保持共享配置
            burn_in_iters=ssdc_schedule['burn_in_iters']  # 中文注释：训练配置携带预热步数确保学生与教师耦合控制一致
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
    print(cfg.model.detector.detector.enable_ssdc)  # 中文注释：额外打印DiffusionDetector层级SS-DC开关确保已按需开启
    print(cfg.model.detector.diff_model.config)  # 中文注释：打印扩散教师配置路径确认注入成功
    print(cfg.model.detector.detector.init_cfg)  # 中文注释：打印扩散检测器初始化配置确认Stage-1权重路径正确传递
