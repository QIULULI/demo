import pytest  # 中文注释：引入pytest以便控制测试条件和断言

torch = pytest.importorskip('torch', reason='PyTorch is required for metric aggregation test')  # 中文注释：若环境缺少PyTorch则跳过本测试以避免导入错误

from mmdet.models.detectors.Z_domain_generalization_detector import DomainGeneralizationDetector  # 中文注释：导入待测试的域泛化检测器类以访问辅助函数


def test_merge_metrics_with_average():  # 中文注释：验证辅助函数在双传感器场景下的准确率统计逻辑
    detector = DomainGeneralizationDetector.__new__(DomainGeneralizationDetector)  # 中文注释：使用__new__绕过复杂初始化以获得类实例
    aggregated_losses = dict()  # 中文注释：准备一个空字典模拟全局损失累加容器
    accuracy_buffers = dict()  # 中文注释：初始化准确率缓存字典存放加权计数
    first_sensor_losses = {'cross_acc': torch.tensor(80.0), 'cross_loss_cls': torch.tensor(1.0)}  # 中文注释：构造首个传感器的损失与准确率
    detector._merge_metrics_with_average(aggregated_losses, first_sensor_losses, 2, accuracy_buffers)  # 中文注释：以两条样本累积首个传感器的指标
    second_sensor_losses = {'cross_acc': torch.tensor(60.0), 'cross_loss_cls': torch.tensor(2.0)}  # 中文注释：构造第二个传感器的损失与准确率
    detector._merge_metrics_with_average(aggregated_losses, second_sensor_losses, 3, accuracy_buffers)  # 中文注释：以三条样本累积第二个传感器的指标
    detector._finalize_accuracy_metrics(aggregated_losses, accuracy_buffers)  # 中文注释：在全部传感器处理结束后回写平均准确率
    expected_acc = (80.0 * 2 + 60.0 * 3) / 5  # 中文注释：手工计算基于样本数量的期望平均准确率
    assert torch.isclose(aggregated_losses['cross_acc'], torch.tensor(expected_acc))  # 中文注释：断言辅助函数输出的准确率符合期望值
    assert 0.0 <= aggregated_losses['cross_acc'].item() <= 100.0  # 中文注释：确保最终准确率保持在百分制范围内
    assert aggregated_losses['cross_loss_cls'] == torch.tensor(3.0)  # 中文注释：确认普通损失项按加法正常累计
