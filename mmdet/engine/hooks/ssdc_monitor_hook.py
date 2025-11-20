# -*- coding: utf-8 -*-  # 指定文件编码确保中文注释正常显示
"""SS-DC训练过程监控钩子实现。"""  # 提供模块级文档字符串说明用途

from typing import Dict, Optional, Sequence  # 中文注释：引入类型提示便于阅读

import torch  # 中文注释：引入PyTorch以便张量计算
import torch.nn.functional as F  # 中文注释：用于余弦相似度与归一化操作
from mmengine.hooks import Hook  # 中文注释：导入Hook基类以便继承
from mmengine.model import is_model_wrapper  # 中文注释：用于解封装分布式模型
from mmengine.runner import Runner  # 中文注释：用于类型注解和访问运行器属性

from mmdet.registry import HOOKS  # 中文注释：注册表用于注册自定义钩子


@HOOKS.register_module()  # 中文注释：将钩子注册到MMDetection以便在配置中启用
class SSDCMonitorHook(Hook):  # 中文注释：定义SS-DC监控钩子类
    def __init__(self,
                 interval: int = 100,  # 中文注释：数值日志输出的迭代间隔默认100
                 vis_interval: int = 1000,  # 中文注释：可视化输出的迭代间隔默认1000
                 max_vis_samples: int = 1  # 中文注释：单次可视化的样本数量上限避免生成过多图像
                 ) -> None:  # 中文注释：构造函数声明返回None
        super().__init__()  # 中文注释：调用父类初始化
        self.interval = interval  # 中文注释：保存数值日志间隔
        self.vis_interval = vis_interval  # 中文注释：保存可视化间隔
        self.max_vis_samples = max_vis_samples  # 中文注释：保存最大可视化样本数

    def after_train_iter(self, runner: Runner, batch_idx: int, data_batch: Optional[Dict] = None, outputs: Optional[Dict] = None) -> None:
        """中文注释：在每个训练迭代结束后收集并记录SS-DC统计信息。"""
        if runner.iter is None:  # 中文注释：若当前迭代未知则直接返回避免误判
            return  # 中文注释：无有效迭代索引无法调度
        detector, teacher = self._locate_detectors(runner.model)  # 中文注释：定位学生与教师检测器实例
        if detector is None or not getattr(detector, 'enable_ssdc', False):  # 中文注释：若未找到SS-DC检测器或模块未启用则跳过
            return  # 中文注释：无需记录
        if (runner.iter + 1) % self.interval != 0 and (runner.iter + 1) % self.vis_interval != 0:  # 中文注释：当未到达任一间隔时提前返回减少开销
            return  # 中文注释：保持训练效率
        student_cache = self._get_latest_cache(detector)  # 中文注释：读取学生最新一次前向缓存
        teacher_cache = self._get_latest_cache(teacher) if teacher is not None else None  # 中文注释：若教师存在则同步读取教师缓存
        if student_cache is None:  # 中文注释：缓存缺失时无法计算统计量
            return  # 中文注释：安全退出
        log_payload = self._compute_scalar_stats(student_cache, teacher_cache)  # 中文注释：计算标量统计信息用于日志
        if log_payload and (runner.iter + 1) % self.interval == 0:  # 中文注释：仅在数值间隔时写入日志
            logger = getattr(runner, 'logger', None)  # 中文注释：尝试获取日志记录器
            if logger is not None:  # 中文注释：日志器存在时输出统计信息
                logger.info(f"SSDCMonitor iter={runner.iter + 1} stats={log_payload}")  # 中文注释：格式化输出迭代与统计字典
        if (runner.iter + 1) % self.vis_interval == 0:  # 中文注释：到达可视化间隔时生成图像
            self._log_visualizations(runner, detector, student_cache)  # 中文注释：调用内部方法生成并记录可视化

    def _locate_detectors(self, model: torch.nn.Module) -> tuple:
        """中文注释：解封装模型并返回学生与教师检测器引用。"""
        base_model = model.module if is_model_wrapper(model) else model  # 中文注释：若模型被封装则取出内部真实模型
        if hasattr(base_model, 'model'):  # 中文注释：DomainAdaptationDetector场景下的再次解包
            base_model = base_model.model  # 中文注释：提取内部包含师生的实例
        student = getattr(base_model, 'student', None)  # 中文注释：尝试获取学生分支
        teacher = getattr(base_model, 'teacher', None)  # 中文注释：尝试获取教师分支
        if student is None and hasattr(base_model, 'ssdc_feature_cache'):  # 中文注释：当不存在学生分支但模型本身就是检测器时直接返回
            student = base_model  # 中文注释：将基础模型视作学生侧
        return student, teacher  # 中文注释：返回学生与教师引用

    def _get_latest_cache(self, detector: Optional[torch.nn.Module]) -> Optional[Dict]:
        """中文注释：从检测器读取最近一次的SS-DC缓存。"""
        if detector is None or not hasattr(detector, 'ssdc_feature_cache'):  # 中文注释：若对象不存在缓存属性则返回None
            return None  # 中文注释：无法提供统计信息
        cache = detector.ssdc_feature_cache.get('noref', None)  # 中文注释：优先使用无参考分支的缓存
        if cache is None:  # 中文注释：若无参考缓存不存在则尝试参考分支
            cache = detector.ssdc_feature_cache.get('ref', None)  # 中文注释：读取参考分支缓存
        return cache  # 中文注释：返回可能为空的缓存字典

    def _compute_scalar_stats(self, student_cache: Dict, teacher_cache: Optional[Dict]) -> Dict[str, float]:
        """中文注释：基于缓存计算能量比例、域特异占比与一致性等标量。"""
        stats: Dict[str, float] = {}  # 中文注释：初始化结果字典
        inv_feats: Optional[Sequence[torch.Tensor]] = student_cache.get('inv') if isinstance(student_cache, dict) else None  # 中文注释：读取学生域不变特征序列
        ds_feats: Optional[Sequence[torch.Tensor]] = student_cache.get('ds') if isinstance(student_cache, dict) else None  # 中文注释：读取学生域特异特征序列
        if isinstance(inv_feats, (list, tuple)) and isinstance(ds_feats, (list, tuple)) and len(inv_feats) == len(ds_feats):  # 中文注释：确认两类特征均存在且层数一致
            energy_ratios = []  # 中文注释：初始化能量比例列表
            for inv_feat, ds_feat in zip(inv_feats, ds_feats):  # 中文注释：逐层遍历域不变与域特异特征
                if torch.is_tensor(inv_feat) and torch.is_tensor(ds_feat):  # 中文注释：仅在均为张量时计算
                    inv_energy = inv_feat.pow(2).mean().detach()  # 中文注释：计算域不变特征能量
                    ds_energy = ds_feat.pow(2).mean().detach()  # 中文注释：计算域特异特征能量
                    total_energy = (inv_energy + ds_energy).clamp_min(1e-6)  # 中文注释：总能量加上安全下限防止除零
                    energy_ratios.append((inv_energy / total_energy).item())  # 中文注释：记录当前层的能量比例
            if energy_ratios:  # 中文注释：仅当存在有效层级时写入统计
                stats['energy_ratio_mean'] = float(sum(energy_ratios) / len(energy_ratios))  # 中文注释：记录平均能量比例
        if isinstance(student_cache, dict) and isinstance(student_cache.get('stats', None), dict):  # 中文注释：当耦合统计存在时读取域特异占比
            ds_ratios = student_cache['stats'].get('ds_ratios', None)  # 中文注释：提取域特异注意力比例张量
            if torch.is_tensor(ds_ratios):  # 中文注释：确保为张量后计算均值
                stats['ds_token_ratio'] = float(ds_ratios.mean().detach().item())  # 中文注释：记录域特异token注意力均值
        if teacher_cache is not None and isinstance(teacher_cache.get('inv', None), (list, tuple)) and isinstance(student_cache.get('inv', None), (list, tuple)):  # 中文注释：当教师与学生域不变特征均可用时计算一致性
            student_inv = student_cache['inv']  # 中文注释：提取学生域不变特征序列
            teacher_inv = teacher_cache['inv']  # 中文注释：提取教师域不变特征序列
            cos_values = []  # 中文注释：初始化余弦相似度列表
            for stu_feat, tea_feat in zip(student_inv, teacher_inv):  # 中文注释：逐层遍历对应特征
                if torch.is_tensor(stu_feat) and torch.is_tensor(tea_feat):  # 中文注释：仅在两者均为张量时计算
                    stu_vec = F.normalize(stu_feat.flatten(1), dim=1)  # 中文注释：将学生特征展开并归一化
                    tea_vec = F.normalize(tea_feat.flatten(1), dim=1)  # 中文注释：将教师特征展开并归一化
                    cos_val = (stu_vec * tea_vec.detach()).sum(dim=1).mean()  # 中文注释：计算批次平均余弦相似度
                    cos_values.append(cos_val.item())  # 中文注释：记录当前层结果
            if cos_values:  # 中文注释：存在有效层级时写入统计
                stats['di_consistency'] = float(sum(cos_values) / len(cos_values))  # 中文注释：计算层间平均余弦一致性
        return stats  # 中文注释：返回标量统计结果

    def _log_visualizations(self, runner: Runner, detector: torch.nn.Module, cache: Dict) -> None:
        """中文注释：生成频域掩码与解耦特征热力图并提交到可视化器。"""
        visualizer = getattr(runner, 'visualizer', None)  # 中文注释：获取可视化器实例
        if visualizer is None:  # 中文注释：未配置可视化器时跳过图像生成
            return  # 中文注释：直接退出以避免不必要开销
        inv_feats: Optional[Sequence[torch.Tensor]] = cache.get('inv') if isinstance(cache, dict) else None  # 中文注释：读取域不变特征
        ds_feats: Optional[Sequence[torch.Tensor]] = cache.get('ds') if isinstance(cache, dict) else None  # 中文注释：读取域特异特征
        if not isinstance(inv_feats, (list, tuple)) or not isinstance(ds_feats, (list, tuple)):  # 中文注释：无有效特征时返回
            return  # 中文注释：无需可视化
        num_samples = min(self.max_vis_samples, len(inv_feats))  # 中文注释：限制可视化层级数以避免生成过多图像
        for level_idx in range(num_samples):  # 中文注释：遍历需要可视化的层级
            inv_map = inv_feats[level_idx]  # 中文注释：选取当前层域不变特征
            ds_map = ds_feats[level_idx]  # 中文注释：选取当前层域特异特征
            if not (torch.is_tensor(inv_map) and torch.is_tensor(ds_map)):  # 中文注释：若任一不是张量则跳过当前层
                continue  # 中文注释：进入下一层
            inv_heat = inv_map.mean(dim=1, keepdim=True)  # 中文注释：在通道维度求均值形成单通道热力图
            ds_heat = ds_map.mean(dim=1, keepdim=True)  # 中文注释：同样处理域特异特征
            inv_norm = (inv_heat - inv_heat.min()) / (inv_heat.max() - inv_heat.min() + 1e-6)  # 中文注释：归一化到0-1范围避免亮度溢出
            ds_norm = (ds_heat - ds_heat.min()) / (ds_heat.max() - ds_heat.min() + 1e-6)  # 中文注释：归一化域特异热力图
            visualizer.add_image(  # 中文注释：将域不变热力图写入可视化器
                name=f'SSDC/Inv_L{level_idx}',  # 中文注释：指定图像名称包含层级索引
                img=inv_norm[0].detach(),  # 中文注释：取批次第一张图并detach以避免梯度
                step=runner.iter + 1)  # 中文注释：记录当前迭代步数
            visualizer.add_image(  # 中文注释：将域特异热力图写入可视化器
                name=f'SSDC/DS_L{level_idx}',  # 中文注释：指定域特异图像名称
                img=ds_norm[0].detach(),  # 中文注释：同样取第一张图用于可视化
                step=runner.iter + 1)  # 中文注释：使用相同步数方便对齐
        said_module = getattr(detector, 'said_filter', None)  # 中文注释：尝试获取SAID滤波器以可视化频率掩码
        if said_module is not None and hasattr(said_module, 'cutoff_logit'):  # 中文注释：仅当滤波器存在且含可学习掩码时处理
            cutoff = torch.sigmoid(said_module.cutoff_logit.detach())  # 中文注释：将logit转换为截止比例
            visualizer.add_scalar('SSDC/Cutoff', float(cutoff.mean().item()), runner.iter + 1)  # 中文注释：记录平均截止比例便于监控


# 中文注释：小型自检代码（仅执行导入与空缓存调用）
if __name__ == '__main__':  # 中文注释：仅在直接运行本文件时执行以下逻辑
    dummy_hook = SSDCMonitorHook()  # 中文注释：实例化钩子以验证构造函数无异常
    dummy_detector = type('Dummy', (), {'enable_ssdc': True, 'ssdc_feature_cache': {'noref': {}}})()  # 中文注释：构造简易检测器对象
    dummy_runner = type('DummyRunner', (), {'iter': 0, 'model': dummy_detector})()  # 中文注释：构造包含迭代计数与模型的伪Runner
    dummy_hook.after_train_iter(dummy_runner, 0)  # 中文注释：调用钩子方法确保空缓存时能安全退出
