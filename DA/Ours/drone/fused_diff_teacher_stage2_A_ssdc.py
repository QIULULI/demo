# DA/Ours/drone/fused_diff_teacher_stage2_A_ssdc.py
import copy  # 中文注释：导入copy用于安全深拷贝配置字典
from pathlib import Path  # 中文注释：用于构造绝对路径避免相对导入问题

from mmengine.config import Config  # 中文注释：使用Config绕过连字符模块名的直接import限制

_ROOT = Path(__name__).resolve().parents[3]  # 中文注释：定位仓库根目录
diffusion_detector_template = Config.fromfile(
    _ROOT / '/userhome/liqiulu/code/FGT-stage2/DA/_base_/models/faster-rcnn_diff_fpn.py'
).model  # 中文注释：加载支持SS-DC的DiffusionDetector模板

_base_ = [  # 继承基础配置列表
    '../../_base_/models/diffusion_guided_adaptation_faster_rcnn_r101_fpn.py',  # 基础扩散-检测模型结构
    '../../_base_/da_setting/semi_20k.py',  # 2 万迭代半监督/域自适应训练日程
    '../../_base_/datasets/sim_to_real/semi_drone_rgb_aug.py',  # 仿真 RGB → 真实 RGB 数据集定义
]  # 结束基础配置

detector = _base_.model  # 从基础配置拷贝域自适应检测器
diffusion_detector = copy.deepcopy(diffusion_detector_template)  # 中文注释：深拷贝DiffusionDetector模板确保后续覆写不影响原始配置
diffusion_detector['roi_head']['bbox_head']['num_classes'] = 1  # 中文注释：将类别数收敛为无人机单类任务
diffusion_detector['backbone']['diff_config']['classes'] = ('drone',)  # 中文注释：为扩散骨干指定单类别语义标签以匹配上游数据集
diffusion_detector['init_cfg'] = dict(  # 中文注释：设置学生初始化权重保持与Stage-1对齐
    type='Pretrained',  # 中文注释：使用预训练权重初始化学生
    checkpoint='best_coco_bbox_mAP_50_iter_20000.pth')  # 中文注释：指定Stage-1学生权重路径
ssdc_runtime_cfg = dict(  # 中文注释：整理SS-DC运行期配置以直传支持该签名的检测器
    enable_ssdc=True,  # 中文注释：显式开启SS-DC以触发相关模块构建
    skip_local_loss=False,  # 中文注释：默认不跳过本地SS-DC损失，包装器可按需覆盖
    said_filter=dict(type='SAIDFilterBank'),  # 中文注释：指定SAID滤波器类型保持最小可用配置
    coupling_neck=dict(type='SSDCouplingNeck', use_ds_tokens=True),  # 中文注释：启用带DS token的耦合颈模块
    loss_decouple=dict(type='LossDecouple', loss_weight=1.0),  # 中文注释：设置解耦损失默认权重
    loss_couple=dict(type='LossCouple', loss_weight=1.0),  # 中文注释：设置耦合损失默认权重
    w_decouple=[(0, 0.1), (6000, 0.5)],  # 中文注释：复制阶段性解耦权重调度
    w_couple=[(2000, 0.2), (10000, 0.5)],  # 中文注释：复制阶段性耦合权重调度
    w_di_consistency=0.3,  # 中文注释：设置域不变一致性损失权重
    consistency_gate=[(0, 0.9), (12000, 0.6)],  # 中文注释：设置伪标签余弦门限调度
    burn_in_iters=0)  # 中文注释：默认无额外SS-DC烧入步保持向后兼容
diffusion_detector['enable_ssdc'] = ssdc_runtime_cfg['enable_ssdc']  # 中文注释：将开关写入DiffusionDetector以匹配构造函数签名
diffusion_detector['use_ds_tokens'] = True  # 中文注释：显式传递DS token开关以匹配目标检测器构造参数
diffusion_detector['ssdc_cfg'] = copy.deepcopy(ssdc_runtime_cfg)  # 中文注释：深拷贝SS-DC配置供检测器合并多源参数
diffusion_detector['train_cfg'].setdefault('ssdc_cfg', copy.deepcopy(ssdc_runtime_cfg))  # 中文注释：在训练配置中同步写入SS-DC字段便于优先级合并
diffusion_detector['train_cfg']['enable_ssdc'] = ssdc_runtime_cfg['enable_ssdc']  # 中文注释：训练配置层面也声明开关避免缺省路径遗漏
detector.detector = diffusion_detector  # 中文注释：将半监督框架内部的学生/教师替换为支持SS-DC的DiffusionDetector
detector.data_preprocessor = diffusion_detector['data_preprocessor']  # 中文注释：复用DiffusionDetector的数据预处理配置保持一致

detector.diff_model.config = 'configs/diff/fused_teacher_deploy.py'  # 扩散教师配置（沿用 Stage-1）
detector.diff_model.pretrained_model = 'rgb_fused_teacher_only.pth'  # 扩散教师权重

model = dict(  # 最外层模型封装
    _delete_=True,  # 删除并重写基础字段
    type='DomainAdaptationDetector',  # 域自适应检测封装
    detector=detector,  # 注入学生-教师-扩散联合模型
    data_preprocessor=dict(  # 多分支数据预处理
        type='MultiBranchDataPreprocessor',  # 区分监督/无监督分支
        data_preprocessor=detector.data_preprocessor),  # 复用相同归一化
    train_cfg=dict(  # 训练阶段设置
        detector_cfg=dict(type='SemiBaseDiff', burn_up_iters=2000),  # 半监督扩散框架与 2k 热身
        ssdc_cfg=dict(  # SS-DC 专用超参与调度
            w_decouple=[(0, 0.1), (6000, 0.5)],  # 频域解耦损失权重线性爬升
            w_couple=[(2000, 0.2), (10000, 0.5)],  # 谱-空耦合对齐权重热身后启动
            w_di_consistency=0.3,  # 学生/教师 DI 一致性恒定权重
            consistency_gate=[(0, 0.9), (12000, 0.6)]),  # 伪标签 DI 余弦阈值由严到松
        feature_loss_cfg=dict(  # 兼容原有特征蒸馏开关
            feature_loss_type='mse',  # 蒸馏类型
            feature_loss_weight=1.0)))  # 蒸馏权重
__all__ = ['_base_', 'model']  # 中文注释：仅导出基础列表与模型字典避免额外符号泄漏

del Config  # 中文注释：删除仅用于模板加载的Config符号防止暴露到外部作用域
del Path  # 中文注释：删除仅用于路径解析的Path符号防止暴露到外部作用域
del _ROOT  # 中文注释：删除根路径中间变量保证最终字典精简
del diffusion_detector_template  # 中文注释：删除模板配置中间变量避免用户误用

# 自检代码（复制到REPL快速验证构建与前向占位）：
# from mmengine.config import Config  # 中文注释：导入配置解析器
# from mmdet.registry import MODELS  # 中文注释：导入注册表用于实例化模型
# import torch  # 中文注释：导入PyTorch构造假输入
# cfg = Config.fromfile('DA/Ours/drone/fused_diff_teacher_stage2_A_ssdc.py')  # 中文注释：载入当前配置文件
# cfg.model.detector['diff_model']['pretrained_model'] = None  # 中文注释：关闭权重加载避免路径缺失报错
# cfg.model.detector['detector']['init_cfg']['checkpoint'] = None  # 中文注释：同样关闭学生预训练权重读取
# model = MODELS.build(cfg.model)  # 中文注释：按照配置构建DomainAdaptationDetector
# model = model.cpu()  # 中文注释：将模型迁移到CPU便于本地快速测试
# dummy_inputs = torch.randn(1, 3, 512, 512)  # 中文注释：构造单张假图像
# _ = model.model.student.forward_dummy(dummy_inputs)  # 中文注释：调用学生分支的前向占位以确保接口连通
