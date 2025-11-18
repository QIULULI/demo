import torch  # 中文注释：引入PyTorch用于创建张量和计算损失

from mmdet.models.detectors.Z_diffusion_detector import DiffusionDetector  # 中文注释：导入扩散检测器以验证调度工具函数
from mmdet.models.detectors.Z_domain_adaptation_detector import DomainAdaptationDetector  # 中文注释：导入域自适应检测器以测试包装器损失汇总
from mmdet.models.utils import rename_loss_dict  # 中文注释：导入工具函数用于前缀化损失字典
from mmdet.models.utils import reweight_loss_dict  # 中文注释：导入工具函数用于加权损失字典


class DummyLoss:  # 中文注释：定义简易损失模块用于构造假数据流
    def __init__(self, key_name: str):  # 中文注释：接收键名以便生成唯一的损失键
        self.key_name = key_name  # 中文注释：保存键名供调用时使用

    def __call__(self, raw, inv, ds, said_filter=None, require_grad=True):  # 中文注释：模拟解耦损失的调用接口
        raw_tensor = raw[0] if isinstance(raw, (list, tuple)) else raw  # 中文注释：提取张量便于计算均值
        loss_val = raw_tensor.mean()  # 中文注释：使用简单均值作为伪损失值
        return {f'loss_{self.key_name}': loss_val}  # 中文注释：返回带键名的损失字典


class DummyStudent:  # 中文注释：构造学生检测器桩对象以承载SS-DC缓存和损失函数
    def __init__(self):  # 中文注释：初始化缓存与损失模块
        base_feat = torch.zeros(1, 1, 2, 2)  # 中文注释：创建基础特征张量
        inv_feat = torch.ones(1, 1, 2, 2)  # 中文注释：创建域不变特征张量
        ds_feat = torch.ones(1, 1, 2, 2) * 2  # 中文注释：创建域特异特征张量
        coupled_feat = torch.ones(1, 1, 2, 2) * 3  # 中文注释：创建耦合后特征张量
        self.ssdc_feature_cache = {  # 中文注释：准备SS-DC缓存结构模拟真实前向结果
            'noref': {  # 中文注释：无参考分支缓存
                'raw': [base_feat],  # 中文注释：保存原始特征列表
                'inv': [inv_feat],  # 中文注释：保存域不变特征列表
                'ds': [ds_feat],  # 中文注释：保存域特异特征列表
                'coupled': [coupled_feat],  # 中文注释：保存耦合后特征列表
                'stats': {},  # 中文注释：占位统计信息字典
            }
        }
        self.loss_decouple = DummyLoss('decouple')  # 中文注释：使用桩损失模拟解耦损失计算
        self.loss_couple = lambda coupled, inv, stats: {'loss_couple': coupled[0].mean()}  # 中文注释：使用lambda模拟耦合损失输出
        self.said_filter = None  # 中文注释：占位的SAID模块保持接口一致


class DummyTeacher(DummyStudent):  # 中文注释：教师桩对象复用学生实现并添加detach行为
    def __call__(self, *args, **kwargs):  # 中文注释：保持可调用签名以防外部误用
        return super().__call__(*args, **kwargs)  # 中文注释：直接复用父类行为


def test_interp_schedule_midpoint():  # 中文注释：验证线性插值调度在中点返回合理权重
    weight = DiffusionDetector._interp_schedule([(0, 0.0), (10, 1.0)], 5, 1.0)  # 中文注释：在0到10的中点计算权重
    assert abs(weight - 0.5) < 1e-6  # 中文注释：确保插值结果符合预期


def test_wrapper_ssdc_loss_toggle():  # 中文注释：验证包装器汇总SS-DC损失并支持跳过学生内部计算
    wrapper = DomainAdaptationDetector.__new__(DomainAdaptationDetector)  # 中文注释：绕过基类初始化手动构造实例
    wrapper.ssdc_cfg = {'w_decouple': 1.0, 'w_couple': 1.0}  # 中文注释：提供基础SS-DC权重配置
    wrapper.ssdc_compute_in_wrapper = True  # 中文注释：打开包装器级别损失汇总
    wrapper.ssdc_skip_student_loss = True  # 中文注释：指示学生内部应跳过SS-DC损失
    wrapper.burn_up_iters = 0  # 中文注释：关闭烧入阶段以便立即计算
    wrapper.model = type('obj', (object,), {})()  # 中文注释：动态创建包含师生的容器对象
    wrapper.model.student = DummyStudent()  # 中文注释：绑定学生桩对象
    wrapper.model.teacher = DummyTeacher()  # 中文注释：绑定教师桩对象
    losses = wrapper._compute_ssdc_loss(current_iter=1)  # 中文注释：调用包装器内部的SS-DC损失计算
    wrapped_losses = reweight_loss_dict(rename_loss_dict('check_', losses), 1.0)  # 中文注释：通过重命名和重权验证返回格式
    assert wrapped_losses, '包装器SS-DC损失应当非空以证明路径正常'  # 中文注释：确保损失字典包含条目
