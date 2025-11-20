# Stage-2 UDA配置，融合扩散教师并启用SS-DC模块
# 引用Stage-1基础配置（扩散引导UDA、20k半监督日程与Sim→Real无人机数据集），相对路径需从当前文件起算
import os  # 引入os模块以便通过环境变量灵活切换扩散教师路径

_base_ = [  # 定义基础配置列表以便mmengine按顺序合并
    '../../../../DA/_base_/models/diffusion_guided_adaptation_faster_rcnn_r101_fpn.py',  # 半监督扩散基础检测器
    '../../../../DA/_base_/da_setting/semi_20k.py',  # 20k迭代的半监督训练调度
    '../../../../DA/_base_/datasets/sim_to_real/semi_drone_rgb_aug.py',  # Sim→Real无人机数据集配置
    '../../../../DA/_base_/models/faster-rcnn_diff_fpn.py'  # 追加DiffusionDetector模板避免动态加载
]

stage1_diff_teacher_config = os.environ.get(  # 读取Stage-1扩散教师配置路径
    'STAGE1_DIFF_TEACHER_CONFIG',  # 环境变量名称
    'DG/Ours/drone/fused_diff_teacher_stage1_A_rpn_roi.py'  # 默认路径
)  # 获取配置路径结束
stage1_diff_teacher_ckpt = os.environ.get(  # 读取Stage-1扩散教师权重路径
    'STAGE1_DIFF_TEACHER_CKPT',  # 环境变量名称
    '/userhome/liqiulu/code/FGT-stage2/rgb_fused1111.pth'  # 默认权重文件
)  # 获取权重路径结束

classes = ('drone',)  # 明确声明单类别元组供Diffusion骨干使用

ssdc_schedule = dict(  # 定义SS-DC损失调度以便前向与训练阶段复用
    w_decouple=[(0, 0.1), (6000, 0.5)],  # 解耦损失权重线性升档
    w_couple=[(0, 0.0), (2000, 0.2), (10000, 0.5)],  # 耦合损失权重分段提升
    w_di_consistency=0.3,  # 域不变一致性损失固定权重
    consistency_gate=[(0, 0.9), (12000, 0.6)],  # 一致性阈值逐步放宽
    burn_in_iters=2000  # 耦合损失预热步数
)  # 调度定义结束

model = dict(  # 通过层级覆盖更新模型结构
    _delete_=True,  # 顶层删除基础模型配置以避免遗留键混入
    type='DomainAdaptationDetector',  # 将Stage-2模型包装为域自适应检测器
    detector=dict(  # 指定内部半监督扩散检测器配置
        detector=dict(  # 覆盖DiffusionDetector关键字段
            roi_head=dict(  # 更新ROI头配置
                bbox_head=dict(num_classes=1)  # 设置BBoxHead类别数为1
            ),
            backbone=dict(  # 更新扩散骨干配置
                diff_config=dict(classes=classes)  # 写入无人机类别元组
            ),
            init_cfg=dict(  # 设置DiffusionDetector初始化权重
                type='Pretrained',  # 使用预训练权重方式
                checkpoint='/userhome/liqiulu/code/FGT-stage2/best_coco_bbox_mAP_50_iter_20000.pth'  # Stage-1学生权重路径
            ),
            enable_ssdc=True,  # 启用SS-DC开关
            ssdc_cfg=dict(  # 注入SS-DC模块配置
                enable_ssdc=True,  # 再次声明开启以便合并生效
                said_filter=dict(type='SAIDFilterBank'),  # 指定SAID滤波器类型
                coupling_neck=dict(type='SSDCouplingNeck', use_ds_tokens=True),  # 设置耦合颈结构并启用域特异token
                loss_decouple=dict(type='LossDecouple', loss_weight=1.0),  # 解耦损失配置
                loss_couple=dict(type='LossCouple', loss_weight=1.0),  # 耦合损失配置
                w_decouple=ssdc_schedule['w_decouple'],  # 绑定解耦权重调度
                w_couple=ssdc_schedule['w_couple'],  # 绑定耦合权重调度
                w_di_consistency=ssdc_schedule['w_di_consistency'],  # 绑定域不变一致性权重
                consistency_gate=ssdc_schedule['consistency_gate']  # 绑定一致性阈值调度
            )
        ),
        diff_model=dict(  # 替换扩散教师配置
            config=stage1_diff_teacher_config,  # 指向Stage-1教师配置文件
            pretrained_model=stage1_diff_teacher_ckpt  # 指向Stage-1教师权重
        )
    ),
    data_preprocessor=dict(  # 维持多分支预处理器以兼容UDA流程
        type='MultiBranchDataPreprocessor',  # 预处理器类型
        data_preprocessor=dict(_delete_=False)  # 保留基础规范化配置
    ),
    train_cfg=dict(  # 训练阶段配置
        detector_cfg=dict(type='SemiBaseDiff', burn_up_iters=2000),  # 半监督扩散训练器并设置预热步数
        ssdc_cfg=dict(  # 训练阶段的SS-DC调度
            w_decouple=ssdc_schedule['w_decouple'],  # 解耦损失调度
            w_couple=ssdc_schedule['w_couple'],  # 耦合损失调度
            w_di_consistency=ssdc_schedule['w_di_consistency'],  # 域不变一致性权重
            consistency_gate=ssdc_schedule['consistency_gate'],  # 一致性阈值调度
            burn_in_iters=ssdc_schedule['burn_in_iters']  # 耦合预热步数
        ),
        feature_loss_cfg=dict(  # 特征蒸馏损失配置
            feature_loss_type='mse',  # 主教师特征损失类型
            feature_loss_weight=1.0,  # 主教师特征损失权重
            cross_feature_loss_weight=0.0,  # 交叉特征蒸馏权重
            cross_consistency_cfg=dict(  # 交叉一致性子配置
                cls_weight=0.0,  # 分类一致性权重
                reg_weight=0.0  # 回归一致性权重
            )
        )
    )
)  # 模型配置结束

default_hooks = dict(  # 追加默认钩子配置以在Stage-2训练中启用SSDC监控
    _delete_=False,  # 保留基础钩子并补充自定义钩子
    ssdc_monitor=dict(  # 注册SSDC监控钩子
        type='SSDCMonitorHook',  # 钩子类型
        interval=100,  # 日志打印间隔
        vis_interval=1000,  # 可视化保存间隔
        max_vis_samples=1  # 每次可视化样本数
    )
)  # 钩子配置结束

# 小型自检代码（仅导入与前向张量）
if __name__ == '__main__':  # 当作为脚本运行时执行自检
    from mmengine import Config  # 导入配置解析器
    cfg = Config.fromfile(__file__)  # 载入当前配置文件
    print('cfg.model.type:', cfg.model['type'])  # 打印模型类型确认解析成功
    print('cfg.model.detector keys:', cfg.model.detector.keys())  # 打印检测器键集合确认未混入基础键
    print('cfg.model.detector.detector.enable_ssdc:', cfg.model.detector.detector.enable_ssdc)  # 检查DiffusionDetector层级SS-DC开关
    print('cfg.model.detector.diff_model.config:', cfg.model.detector.diff_model.config)  # 输出扩散教师配置路径
    print('cfg.model.detector.detector.init_cfg:', cfg.model.detector.detector.init_cfg)  # 打印初始化配置确认权重设置
