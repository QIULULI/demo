import pytest  # 中文注释：引入PyTest工具以便根据依赖情况动态跳过测试
from unittest import mock  # 中文注释：导入mock用于替换重量级DiffEncoder以便在单测中构造桩

torch = pytest.importorskip('torch', reason='需要PyTorch以构建扩散检测器')  # 中文注释：若缺少PyTorch则跳过此测试以避免环境问题

from mmengine import Config  # 中文注释：导入配置解析器以读取Stage-2配置文件
from mmdet.registry import MODELS  # 中文注释：导入注册表以构建DiffusionDetector实例


class _FakeDiffEncoder:  # 中文注释：定义轻量级DiffEncoder桩以绕过真实权重加载
    def __init__(self, config, batch_size=2, mode='float', rank='cuda'):  # 中文注释：存储关键配置并忽略其余参数
        self.config = config  # 中文注释：保存传入配置供调试时使用
        self.batch_size = batch_size  # 中文注释：记录批大小保持接口一致
        self.mode = mode  # 中文注释：记录计算精度模式保持接口一致
        self.rank = rank  # 中文注释：记录设备标识保持接口一致

    def change_batchsize(self, batch_size):  # 中文注释：模拟批大小切换接口以满足骨干调用
        self.batch_size = batch_size  # 中文注释：更新内部批大小记录

    def change_mode(self, mode):  # 中文注释：模拟模式切换接口
        self.mode = mode  # 中文注释：更新内部模式记录

    def change_precision(self, mode):  # 中文注释：模拟精度切换接口
        self.mode = mode  # 中文注释：沿用同一字段记录当前精度

    def forward(self, img_tensor, ref_masks=None, ref_labels=None):  # 中文注释：提供前向桩返回多尺度张量列表
        return [img_tensor, img_tensor, img_tensor, img_tensor]  # 中文注释：返回重复张量以满足特征层数假设


def test_stage2_ssdc_backbone_flag(monkeypatch=None):  # 中文注释：验证Stage-2配置可构建DiffusionDetector且启用SS-DC
    assert torch is not None  # 中文注释：显式断言PyTorch已成功导入以满足后续构建依赖
    cfg = Config.fromfile('configs/DA/Ours/drone/fused_diff_teacher_stage2_A_ssdc.py')  # 中文注释：加载目标配置文件
    student_cfg = cfg.model.detector.detector  # 中文注释：提取学生DiffusionDetector配置
    with mock.patch('mmdet.models.backbones.diff_encoder.DIFFEncoder', _FakeDiffEncoder):  # 中文注释：用轻量桩替换真实DiffEncoder
        detector = MODELS.build(student_cfg)  # 中文注释：构建DiffusionDetector实例
    assert detector.enable_ssdc is True  # 中文注释：断言构建结果中的SS-DC开关已成功开启
