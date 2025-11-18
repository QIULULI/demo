import pytest  # 中文注释：引入pytest以便使用条件跳过机制

torch = pytest.importorskip('torch')  # 中文注释：若环境缺少PyTorch则跳过测试以避免报错
import torch.nn as nn  # 中文注释：引入nn模块以便定义仿真SAID模块

from mmdet.models.losses.ssdc_losses import LossDecouple  # 中文注释：从项目中导入待测试的解耦损失模块


class DummySaid(nn.Module):  # 中文注释：定义模拟的SAID模块用于返回输入以简化测试
    def forward(self, feats):  # 中文注释：实现前向接口接受特征序列
        return feats, None  # 中文注释：直接返回输入与占位符以匹配调用签名


def _build_feature_triplet(requires_grad: bool):  # 中文注释：辅助函数创建包含原始、域不变、域特异特征的序列
    raw = [torch.randn(2, 3, 4, 4, requires_grad=requires_grad) for _ in range(2)]  # 中文注释：生成原始特征列表
    inv = [feature * 0.5 for feature in raw]  # 中文注释：将原始特征缩放得到域不变特征保持梯度链路
    ds = [feature - inv_feature for feature, inv_feature in zip(raw, inv)]  # 中文注释：利用差值构造域特异特征保持梯度关联
    return raw, inv, ds  # 中文注释：返回构建好的三个序列


def test_loss_decouple_student_path_retains_gradients():  # 中文注释：测试学生路径下梯度应正常保留
    said = DummySaid()  # 中文注释：实例化模拟SAID模块
    raw, inv, ds = _build_feature_triplet(requires_grad=True)  # 中文注释：创建需要梯度的特征序列
    loss_module = LossDecouple()  # 中文注释：实例化待测损失模块
    outputs = loss_module(raw, inv, ds, said, require_grad=True)  # 中文注释：调用损失并显式允许梯度
    total = sum(outputs.values())  # 中文注释：聚合所有损失用于反向传播
    total.backward()  # 中文注释：执行反向传播检查梯度链路
    assert inv[0].grad is not None  # 中文注释：确认域不变特征获得梯度
    assert ds[0].grad is not None  # 中文注释：确认域特异特征获得梯度


def test_loss_decouple_teacher_path_no_grad():  # 中文注释：测试教师路径下不应产生梯度
    said = DummySaid()  # 中文注释：实例化模拟SAID模块
    raw, inv, ds = _build_feature_triplet(requires_grad=True)  # 中文注释：创建需要梯度的特征序列
    loss_module = LossDecouple()  # 中文注释：实例化待测损失模块
    outputs = loss_module(raw, inv, ds, said, require_grad=False)  # 中文注释：调用损失并显式禁止梯度
    for value in outputs.values():  # 中文注释：遍历各损失分量
        assert value.requires_grad is False  # 中文注释：确认输出张量不跟踪梯度
    assert inv[0].grad is None  # 中文注释：确认调用后域不变特征未产生梯度信息
    assert ds[0].grad is None  # 中文注释：确认调用后域特异特征未产生梯度信息
