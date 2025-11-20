import pytest  # 中文注释：导入pytest以处理可选依赖跳过

torch = pytest.importorskip('torch', reason='需要PyTorch以运行DiffusionDetector构造与前向自检')  # 中文注释：缺少torch时跳过本测试文件

from mmengine import ConfigDict  # 中文注释：导入ConfigDict以便构造轻量配置对象
from mmdet.registry import MODELS  # 中文注释：导入模型注册表以注册与构建桩模块
from mmdet.models.detectors.Z_diffusion_detector import DiffusionDetector  # 中文注释：导入待测试的DiffusionDetector实现


@MODELS.register_module()  # 中文注释：注册轻量骨干以便通过配置构建
class TinySSDCBackbone(torch.nn.Module):  # 中文注释：定义仅输出单层特征的骨干网络桩
    def __init__(self, diff_config, ssdc_cfg=None, enable_ssdc=False, out_channels=1):  # 中文注释：接受SS-DC相关参数确保兼容真实接口
        super().__init__()  # 中文注释：调用父类构造函数完成基本初始化
        self.diff_config = diff_config  # 中文注释：缓存类别映射以满足DiffusionDetector读取需求
        self.out_channels = out_channels  # 中文注释：声明输出通道用于推断耦合层输入维度

    def forward(self, x, ref_masks=None, ref_labels=None):  # 中文注释：前向传播忽略参考信息直接返回单层特征
        return (x,)  # 中文注释：以元组形式返回以符合后续SAID与耦合模块的输入假设


@MODELS.register_module()  # 中文注释：注册轻量SAID滤波器以模拟域特征分解
class TinySAIDFilter(torch.nn.Module):  # 中文注释：定义输出移位特征的SAID桩模块
    def __init__(self):  # 中文注释：初始化父类
        super().__init__()  # 中文注释：执行基础构造逻辑

    def forward(self, features):  # 中文注释：对输入特征执行简单偏置以生成域不变与域特异分量
        inv = [feat + 1 for feat in features]  # 中文注释：域不变特征通过加一获得
        ds = [feat + 2 for feat in features]  # 中文注释：域特异特征通过加二获得
        return inv, ds  # 中文注释：返回分解后的两组特征


@MODELS.register_module()  # 中文注释：注册轻量耦合颈部以模拟特征重组
class TinyCouplingNeck(torch.nn.Module):  # 中文注释：定义将多分支特征重新相加的耦合桩模块
    def __init__(self, in_channels=1, use_ds_tokens=False, num_feature_levels=1, levels=None):  # 中文注释：接受必要参数以兼容真实接口
        super().__init__()  # 中文注释：执行父类构造逻辑
        self.in_channels = in_channels  # 中文注释：保存输入通道数用于调试
        self.use_ds_tokens = use_ds_tokens  # 中文注释：记录是否启用域特异令牌
        self.num_feature_levels = num_feature_levels  # 中文注释：记录特征层级数以验证配置写回
        self.levels = levels  # 中文注释：缓存层级名称列表供断言使用

    def forward(self, features, inv_list, ds_list):  # 中文注释：对多分支特征执行逐层求和
        coupled = [feat + inv + ds for feat, inv, ds in zip(features, inv_list, ds_list)]  # 中文注释：相加得到耦合后的特征输出
        return coupled, {'levels': self.levels}  # 中文注释：返回耦合特征以及层级标签统计


@MODELS.register_module()  # 中文注释：注册恒等标量损失以满足KD分支构建
class TinyScalarLoss(torch.nn.Module):  # 中文注释：定义总是返回零张量的损失桩
    def __init__(self):  # 中文注释：初始化父类
        super().__init__()  # 中文注释：执行基础构造逻辑

    def forward(self, *args, **kwargs):  # 中文注释：忽略输入直接输出零标量
        return torch.tensor(0.0)  # 中文注释：返回用于兼容损失接口的标量张量


def test_constructor_explicit_ssdc_precedence():  # 中文注释：验证构造函数显式SS-DC参数优先级及最小前向流程
    backbone_cfg = {  # 中文注释：准备骨干配置包含基础类别映射与默认关闭的SS-DC开关
        'type': 'TinySSDCBackbone',  # 中文注释：指定使用轻量骨干桩
        'diff_config': {'classes': ['cls']},  # 中文注释：提供类别列表供检测器记录
        'ssdc_cfg': {'skip_local_loss': False},  # 中文注释：设置骨干SS-DC子配置默认关闭本地损失
        'enable_ssdc': False,  # 中文注释：骨干层面关闭SS-DC以测试显式入参优先级
    }
    train_cfg = ConfigDict(enable_ssdc=False, ssdc_cfg={'skip_local_loss': False})  # 中文注释：训练配置同样关闭SS-DC作为回退来源
    auxiliary_cfg = {  # 中文注释：构造辅助分支配置使用轻量损失桩
        'loss_cls_kd': {'type': 'TinyScalarLoss'},  # 中文注释：分类KD损失使用恒等桩
        'loss_reg_kd': {'type': 'TinyScalarLoss'},  # 中文注释：回归KD损失使用恒等桩
        'apply_auxiliary_branch': False,  # 中文注释：关闭辅助分支计算以简化前向
    }
    explicit_ssdc_cfg = {  # 中文注释：构造函数级别的SS-DC配置覆盖其他来源
        'enable_ssdc': True,  # 中文注释：显式开启SS-DC
        'skip_local_loss': True,  # 中文注释：启用跳过本地损失以验证覆盖行为
        'said_filter': {'type': 'TinySAIDFilter'},  # 中文注释：指定轻量SAID滤波器
        'coupling_neck': {'type': 'TinyCouplingNeck', 'in_channels': 1, 'use_ds_tokens': False},  # 中文注释：指定轻量耦合颈部并关闭配置内令牌
        'loss_decouple': {'type': 'TinyScalarLoss', 'loss_weight': 1.0},  # 中文注释：使用恒等损失构建解耦损失模块
        'loss_couple': {'type': 'TinyScalarLoss', 'loss_weight': 1.0},  # 中文注释：使用恒等损失构建耦合损失模块
    }
    detector = DiffusionDetector(  # 中文注释：构建DiffusionDetector实例验证参数解析
        backbone=backbone_cfg,  # 中文注释：传入骨干配置
        neck=None,  # 中文注释：省略颈部以保持最小化依赖
        rpn_head=None,  # 中文注释：省略RPN头以避免额外配置
        roi_head=None,  # 中文注释：省略ROI头以保持轻量
        train_cfg=train_cfg,  # 中文注释：传入训练配置用于回退
        test_cfg=ConfigDict(rcnn=None),  # 中文注释：提供空测试配置满足参数签名
        auxiliary_branch_cfg=auxiliary_cfg,  # 中文注释：传入辅助分支配置
        enable_ssdc=True,  # 中文注释：显式开启SS-DC以验证优先级
        ssdc_cfg=explicit_ssdc_cfg,  # 中文注释：传入显式SS-DC配置供解析
        use_ds_tokens=True,  # 中文注释：显式要求启用域特异令牌覆盖子配置
    )
    assert detector.enable_ssdc is True  # 中文注释：确保最终开关已开启体现优先级
    assert detector.ssdc_cfg.get('skip_local_loss') is True  # 中文注释：确认显式配置覆盖骨干与训练配置
    assert detector.use_ds_tokens is True  # 中文注释：确认构造函数参数覆盖耦合配置中的开关
    dummy_img = torch.zeros(1, 1, 4, 4)  # 中文注释：准备简单输入张量用于最小前向验证
    feats = detector.extract_feat(dummy_img, current_iter=0, ssdc_cfg=detector.ssdc_cfg)  # 中文注释：执行一次特征提取确保无意外参数报错
    assert isinstance(feats, tuple)  # 中文注释：验证返回类型符合预期接口
