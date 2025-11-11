import copy  # 中文注释：引入copy模块以便安全复制配置与样本对象
from typing import Dict, Iterable, List, Optional, Tuple, Union  # 中文注释：导入类型注解工具以提高代码可读性

import torch  # 中文注释：导入PyTorch主库用于张量操作与模型导出
import torch.nn.functional as F  # 中文注释：导入函数式API以计算特征蒸馏损失
from torch import Tensor  # 中文注释：从PyTorch中导入Tensor类型便于类型提示

from mmdet.models.detectors.base import BaseDetector  # 中文注释：导入基础检测器父类以便继承
from mmdet.models.detectors.Z_diffusion_detector import DiffusionDetector  # 中文注释：导入扩散检测器类型用于类型提示
from mmdet.models.utils import rename_loss_dict  # 中文注释：导入辅助函数用于统一损失命名
from mmdet.registry import MODELS  # 中文注释：导入注册表以支持模块化构建
from mmdet.structures import SampleList  # 中文注释：导入样本列表类型以明确接口
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig  # 中文注释：导入配置类型别名以保持与框架兼容


@MODELS.register_module()  # 中文注释：将当前检测器注册到框架注册表中
class DualDiffFusionStage1(BaseDetector):  # 中文注释：定义第一阶段红外可见光扩散融合检测器

    def __init__(self,  # 中文注释：初始化函数负责构建教师与学生检测器并处理配置
                 teacher_ir: Union[DiffusionDetector, ConfigType],  # 中文注释：教师分支可直接传实例或配置字典
                 student_rgb: Union[DiffusionDetector, ConfigType],  # 中文注释：学生分支同样支持实例或配置字典
                 train_cfg: OptConfigType = None,  # 中文注释：训练阶段附加配置控制蒸馏权重与调度
                 data_preprocessor: OptConfigType = None,  # 中文注释：数据预处理配置沿用父类约定
                 init_cfg: OptMultiConfig = None) -> None:  # 中文注释：权重初始化配置保持可选
        super().__init__(data_preprocessor=data_preprocessor, init_cfg=init_cfg)  # 中文注释：调用父类初始化保持接口一致
        self.teacher_ir = self._build_branch(teacher_ir, 'teacher_ir')  # 中文注释：通过私有方法构建或直接引用教师模型
        self.student_rgb = self._build_branch(student_rgb, 'student_rgb')  # 中文注释：通过私有方法构建或直接引用学生模型
        for param in self.teacher_ir.parameters():  # 中文注释：遍历教师模型全部参数以冻结梯度
            param.requires_grad_(False)  # 中文注释：显式禁止教师参数梯度更新
        self.teacher_ir.eval()  # 中文注释：将教师切换到评估模式避免训练态扰动
        self.train_cfg = copy.deepcopy(train_cfg) if train_cfg is not None else dict()  # 中文注释：深拷贝训练配置避免外部副作用
        self.cross_loss_cfg = self._init_cross_cfg(self.train_cfg.get('cross_loss_cfg', dict()))  # 中文注释：初始化交叉蒸馏配置并补齐默认值
        self.feature_loss_cfg = self._init_feature_cfg(self.train_cfg.get('feature_loss_cfg', dict()))  # 中文注释：初始化特征蒸馏配置与默认参数
        self.cross_loss_weight = float(self.cross_loss_cfg.get('loss_weight', 1.0))  # 中文注释：读取交叉蒸馏整体权重默认1.0
        self.cross_rpn_weight = float(self.cross_loss_cfg.get('rpn_weight', 1.0))  # 中文注释：读取RPN蒸馏权重默认1.0
        self.cross_roi_weight = float(self.cross_loss_cfg.get('roi_weight', 1.0))  # 中文注释：读取ROI蒸馏权重默认1.0
        self.feature_loss_type = str(self.feature_loss_cfg.get('type', 'mse')).lower()  # 中文注释：记录特征蒸馏损失类型默认mse
        self.feature_loss_weight = float(self.feature_loss_cfg.get('loss_weight', 0.0))  # 中文注释：读取特征蒸馏权重默认0关闭
        self.feature_loss_per_level = self.feature_loss_cfg.get('per_level_weight', None)  # 中文注释：可选逐层权重列表用于细粒度调节
        self.feature_loss_eps = float(self.feature_loss_cfg.get('eps', 1e-6))  # 中文注释：KL散度等损失的稳定项默认1e-6
        self.cross_loss_schedule = sorted(self.cross_loss_cfg.get('schedule', []), key=lambda item: item.get('start_iter', 0))  # 中文注释：根据起始迭代排序调度表
        self._schedule_stage = -1  # 中文注释：记录当前调度阶段索引以便增量更新
        self.local_iter = 0  # 中文注释：追踪内部迭代计数支持调度生效

    @property
    def with_rpn(self) -> bool:  # 中文注释：暴露是否包含RPN头部以复用学生检测器逻辑
        return getattr(self.student_rgb, 'with_rpn', False)  # 中文注释：直接查询学生检测器的with_rpn属性

    @property
    def with_roi_head(self) -> bool:  # 中文注释：暴露是否包含ROI头部以复用学生检测器逻辑
        return getattr(self.student_rgb, 'with_roi_head', False)  # 中文注释：直接查询学生检测器的with_roi_head属性

    def extract_feat_student(self, batch_inputs: Tensor) -> Tuple[Tensor, ...]:  # 中文注释：封装学生分支特征提取接口
        student_feats = self.student_rgb.extract_feat(batch_inputs)  # 中文注释：调用学生模型提取FPN多尺度特征
        if isinstance(student_feats, (list, tuple)):  # 中文注释：若返回列表或元组则直接转换为元组
            return tuple(student_feats)  # 中文注释：转换为不可变元组便于后续处理
        return (student_feats,)  # 中文注释：单尺度时包装成单元素元组保持接口一致

    def extract_feat_teacher(self, batch_inputs: Tensor) -> Tuple[Tensor, ...]:  # 中文注释：封装教师分支特征提取接口
        with torch.no_grad():  # 中文注释：在无梯度上下文中运行确保教师始终冻结
            teacher_feats = self.teacher_ir.extract_feat(batch_inputs)  # 中文注释：调用教师模型提取FPN多尺度特征
        if isinstance(teacher_feats, (list, tuple)):  # 中文注释：若返回列表或元组则直接转换
            return tuple(teacher_feats)  # 中文注释：转换为元组便于后续统一遍历
        return (teacher_feats,)  # 中文注释：单尺度时包装成元组保持兼容

    def loss(self, batch_inputs: Tensor, batch_data_samples: SampleList) -> Dict[str, Tensor]:  # 中文注释：实现整体损失计算接口
        self._update_schedule()  # 中文注释：根据当前迭代动态调整蒸馏权重
        losses: Dict[str, Tensor] = dict()  # 中文注释：初始化损失容器
        student_losses = self.student_rgb.loss(batch_inputs, batch_data_samples)  # 中文注释：计算学生分支常规监督损失
        losses.update(student_losses)  # 中文注释：将监督损失合并进最终返回字典
        teacher_feats = self.extract_feat_teacher(batch_inputs)  # 中文注释：提取冻结教师的特征图
        student_feats = self.extract_feat_student(batch_inputs)  # 中文注释：提取学生当前梯度下的特征图
        cross_losses = self._compute_cross_losses(teacher_feats, batch_data_samples)  # 中文注释：基于教师特征计算交叉蒸馏损失
        if cross_losses:  # 中文注释：仅当存在交叉损失时合并
            losses.update(cross_losses)  # 中文注释：合并交叉蒸馏损失项
        feature_losses = self._compute_feature_losses(student_feats, teacher_feats)  # 中文注释：计算逐层特征蒸馏损失
        if feature_losses:  # 中文注释：存在特征损失时再合并
            losses.update(feature_losses)  # 中文注释：合并特征蒸馏损失项
        self.local_iter += 1  # 中文注释：迭代计数加一以驱动调度推进
        return losses  # 中文注释：返回组合后的损失字典

    def predict(self, batch_inputs: Tensor, batch_data_samples: SampleList, rescale: bool = True) -> SampleList:  # 中文注释：推理接口完全委托学生模型
        return self.student_rgb.predict(batch_inputs, batch_data_samples, rescale=rescale)  # 中文注释：直接调用学生分支预测确保部署一致

    def _forward(self, batch_inputs: Tensor, batch_data_samples: SampleList) -> Tuple:  # 中文注释：特征前向接口复用学生实现
        return self.student_rgb._forward(batch_inputs, batch_data_samples)  # 中文注释：直接复用学生模型的前向逻辑

    def export_fused_teacher(self, path: str) -> None:  # 中文注释：导出融合教师权重供第二阶段使用
        torch.save(self.student_rgb.state_dict(), path)  # 中文注释：仅保存学生模型参数作为融合教师快照

    def _build_branch(self, module_or_cfg: Union[DiffusionDetector, ConfigType], name: str) -> DiffusionDetector:  # 中文注释：根据传入配置构建检测分支
        if isinstance(module_or_cfg, DiffusionDetector):  # 中文注释：若已是实例则直接返回
            return module_or_cfg  # 中文注释：无需重复构建直接复用现有模型
        if isinstance(module_or_cfg, BaseDetector):  # 中文注释：允许传入其他检测器实例用于快速调试
            return module_or_cfg  # 中文注释：直接返回保持兼容
        if isinstance(module_or_cfg, dict):  # 中文注释：当传入配置字典时使用注册表构建
            return MODELS.build(module_or_cfg)  # 中文注释：利用注册表生成对应检测器
        raise TypeError(f'{name} must be DiffusionDetector or config dict, but got {type(module_or_cfg)!r}')  # 中文注释：类型不匹配时抛出异常提示

    def _init_cross_cfg(self, cfg: Dict) -> Dict:  # 中文注释：为交叉蒸馏配置补齐默认字段
        default_cfg = {  # 中文注释：定义交叉蒸馏的默认超参数集合
            'loss_weight': 1.0,  # 中文注释：建议默认整体权重为1.0保持蒸馏与监督平衡
            'rpn_weight': 1.0,  # 中文注释：建议默认RPN蒸馏权重为1.0可按需缩放
            'roi_weight': 1.0,  # 中文注释：建议默认ROI蒸馏权重为1.0确保候选框质量
            'schedule': []  # 中文注释：默认无调度可在配置中指定按迭代调整权重
        }  # 中文注释：结束默认配置字典定义
        merged_cfg = copy.deepcopy(default_cfg)  # 中文注释：复制默认值防止原地修改
        merged_cfg.update(copy.deepcopy(cfg))  # 中文注释：合并用户配置以覆盖默认项
        return merged_cfg  # 中文注释：返回合并后的配置字典

    def _init_feature_cfg(self, cfg: Dict) -> Dict:  # 中文注释：初始化特征蒸馏配置并填充默认值
        default_cfg = {  # 中文注释：定义特征蒸馏的默认设置
            'type': 'mse',  # 中文注释：默认采用均方误差作为特征蒸馏损失
            'loss_weight': 0.0,  # 中文注释：默认权重为0表示不开启特征蒸馏
            'per_level_weight': None,  # 中文注释：可选逐层权重列表用于强调特定尺度
            'eps': 1e-6  # 中文注释：KL散度等损失的数值稳定项
        }  # 中文注释：结束特征默认配置定义
        merged_cfg = copy.deepcopy(default_cfg)  # 中文注释：复制默认设置以便安全更新
        merged_cfg.update(copy.deepcopy(cfg))  # 中文注释：合并外部配置覆盖默认值
        return merged_cfg  # 中文注释：返回特征蒸馏配置

    def _update_schedule(self) -> None:  # 中文注释：根据迭代数更新交叉蒸馏与特征蒸馏权重
        while (self._schedule_stage + 1 < len(self.cross_loss_schedule)  # 中文注释：判断是否存在后续调度阶段
               and self.local_iter >= self.cross_loss_schedule[self._schedule_stage + 1].get('start_iter', 0)):  # 中文注释：当达到阶段起始迭代时推进调度
            self._schedule_stage += 1  # 中文注释：更新当前调度阶段索引
            stage_cfg = self.cross_loss_schedule[self._schedule_stage]  # 中文注释：读取当前阶段配置字典
            if 'loss_weight' in stage_cfg:  # 中文注释：若配置指定整体权重则覆盖当前数值
                self.cross_loss_weight = float(stage_cfg['loss_weight'])  # 中文注释：更新交叉蒸馏整体权重
            if 'rpn_weight' in stage_cfg:  # 中文注释：若配置指定RPN权重则覆盖
                self.cross_rpn_weight = float(stage_cfg['rpn_weight'])  # 中文注释：更新RPN蒸馏权重
            if 'roi_weight' in stage_cfg:  # 中文注释：若配置指定ROI权重则覆盖
                self.cross_roi_weight = float(stage_cfg['roi_weight'])  # 中文注释：更新ROI蒸馏权重
            if 'feature_loss_weight' in stage_cfg:  # 中文注释：若调度包含特征蒸馏权重则同步更新
                self.feature_loss_weight = float(stage_cfg['feature_loss_weight'])  # 中文注释：更新特征蒸馏整体权重

    def _compute_cross_losses(self, teacher_feats: Tuple[Tensor, ...], batch_data_samples: SampleList) -> Dict[str, Tensor]:  # 中文注释：计算交叉蒸馏损失
        if self.cross_loss_weight <= 0:  # 中文注释：若整体权重为零直接返回空字典
            return dict()  # 中文注释：无损失情况下返回空容器
        cross_losses: Dict[str, Tensor] = dict()  # 中文注释：初始化交叉损失容器
        rpn_results_list: Optional[List] = None  # 中文注释：提前声明RPN候选框结果供ROI复用
        if self.with_rpn and self.cross_rpn_weight > 0:  # 中文注释：仅在存在RPN且权重大于零时计算RPN蒸馏
            rpn_data_samples = self._prepare_rpn_samples(batch_data_samples)  # 中文注释：复制样本并重置标签适配RPN训练
            proposal_cfg = self._get_proposal_cfg()  # 中文注释：读取候选框生成配置
            rpn_losses, rpn_results_list = self.student_rgb.rpn_head.loss_and_predict(teacher_feats, rpn_data_samples, proposal_cfg=proposal_cfg)  # 中文注释：基于教师特征计算RPN损失并输出候选框
            prefixed = rename_loss_dict('cross_', rpn_losses)  # 中文注释：为RPN损失添加交叉前缀防止与监督损失冲突
            for key, value in prefixed.items():  # 中文注释：遍历损失项以乘以相应权重
                cross_losses[key] = value * self.cross_loss_weight * self.cross_rpn_weight  # 中文注释：累积加权后的RPN蒸馏损失
        if self.with_roi_head and self.cross_roi_weight > 0:  # 中文注释：若存在ROI头且权重大于零则继续计算ROI蒸馏
            if rpn_results_list is None:  # 中文注释：当未执行RPN蒸馏时需要准备候选框
                rpn_results_list = self._prepare_roi_inputs(batch_data_samples, teacher_feats)  # 中文注释：通过辅助方法获取候选框列表
            roi_losses = self.student_rgb.roi_head.loss(teacher_feats, rpn_results_list, batch_data_samples)  # 中文注释：基于教师特征与候选框计算ROI损失
            prefixed_roi = rename_loss_dict('cross_', roi_losses)  # 中文注释：统一添加交叉前缀
            for key, value in prefixed_roi.items():  # 中文注释：遍历ROI损失项以乘以权重
                cross_losses[key] = value * self.cross_loss_weight * self.cross_roi_weight  # 中文注释：累积加权后的ROI蒸馏损失
        return cross_losses  # 中文注释：返回整理好的交叉蒸馏损失字典

    def _compute_feature_losses(self, student_feats: Tuple[Tensor, ...], teacher_feats: Tuple[Tensor, ...]) -> Dict[str, Tensor]:  # 中文注释：计算逐层特征蒸馏损失
        if self.feature_loss_weight <= 0:  # 中文注释：若整体权重为零则直接跳过
            return dict()  # 中文注释：返回空字典表示无特征蒸馏
        total_loss = None  # 中文注释：初始化累加器
        total_weight = 0.0  # 中文注释：初始化有效权重总和
        for level_idx, (student_feature, teacher_feature) in enumerate(zip(student_feats, teacher_feats)):  # 中文注释：逐尺度遍历学生与教师特征
            if not torch.is_tensor(student_feature) or not torch.is_tensor(teacher_feature):  # 中文注释：若任意一侧不是张量则跳过
                continue  # 中文注释：直接处理下一尺度
            per_level_weight = 1.0  # 中文注释：默认逐层权重为1
            if isinstance(self.feature_loss_per_level, Iterable):  # 中文注释：若用户提供逐层权重序列
                weights_list = list(self.feature_loss_per_level)  # 中文注释：将序列转为列表方便索引
                if level_idx < len(weights_list):  # 中文注释：确保当前索引在列表范围内
                    per_level_weight = float(weights_list[level_idx])  # 中文注释：读取对应尺度的权重
            if per_level_weight <= 0:  # 中文注释：当权重小于等于零时跳过该尺度
                continue  # 中文注释：继续处理下一个尺度
            detached_teacher = teacher_feature.detach()  # 中文注释：显式分离教师特征避免梯度反传
            if self.feature_loss_type == 'kl':  # 中文注释：当选择KL散度时需要概率化处理
                student_log_prob = F.log_softmax(student_feature, dim=1)  # 中文注释：对学生特征施加log softmax得到对数概率
                teacher_prob = F.softmax(detached_teacher, dim=1)  # 中文注释：对教师特征施加softmax得到概率分布
                loss_value = F.kl_div(student_log_prob, teacher_prob, reduction='batchmean')  # 中文注释：计算批均值KL散度
            else:  # 中文注释：默认执行均方误差损失
                loss_value = F.mse_loss(student_feature, detached_teacher)  # 中文注释：对齐学生与教师特征的均方误差
            scaled_loss = loss_value * per_level_weight  # 中文注释：乘以逐层权重以强化特定尺度
            total_loss = scaled_loss if total_loss is None else total_loss + scaled_loss  # 中文注释：将当前尺度损失累加到总损失
            total_weight += per_level_weight  # 中文注释：累计有效权重以便后续归一化
        if total_loss is None or total_weight <= 0:  # 中文注释：若没有有效尺度则返回空字典
            return dict()  # 中文注释：无特征蒸馏时返回空字典
        normalized_loss = total_loss / total_weight  # 中文注释：对累积损失按权重求平均
        return {'cross_feature_loss': normalized_loss * self.feature_loss_weight}  # 中文注释：返回加权后的特征蒸馏损失字典

    def _prepare_rpn_samples(self, batch_data_samples: SampleList) -> SampleList:  # 中文注释：复制样本并重置RPN标签
        copied_samples = copy.deepcopy(batch_data_samples)  # 中文注释：深拷贝样本避免修改原始数据
        for data_sample in copied_samples:  # 中文注释：遍历复制后的每个样本
            if hasattr(data_sample, 'gt_instances') and hasattr(data_sample.gt_instances, 'labels'):  # 中文注释：确保存在标签张量
                data_sample.gt_instances.labels = torch.zeros_like(data_sample.gt_instances.labels)  # 中文注释：将标签全部置零匹配RPN二分类
        return copied_samples  # 中文注释：返回处理后的样本列表

    def _prepare_roi_inputs(self, batch_data_samples: SampleList, teacher_feats: Tuple[Tensor, ...]) -> List:  # 中文注释：准备ROI阶段所需候选框
        if self.with_rpn:  # 中文注释：若学生包含RPN则直接使用教师特征重新生成候选框
            proposal_cfg = self._get_proposal_cfg()  # 中文注释：获取RPN候选框配置
            _, proposals = self.student_rgb.rpn_head.loss_and_predict(teacher_feats, batch_data_samples, proposal_cfg=proposal_cfg)  # 中文注释：在不关心损失的情况下生成候选框
            return proposals  # 中文注释：返回候选框列表供ROI使用
        return [sample.proposals for sample in batch_data_samples]  # 中文注释：无RPN时读取样本自带的候选框

    def _get_proposal_cfg(self):  # 中文注释：获取RPN候选框配置对象
        train_cfg = getattr(self.student_rgb, 'train_cfg', None)  # 中文注释：优先读取学生训练配置
        if train_cfg is not None and hasattr(train_cfg, 'rpn_proposal'):  # 中文注释：若训练配置定义了rpn_proposal则优先返回
            return train_cfg.rpn_proposal  # 中文注释：返回训练态使用的候选框配置
        test_cfg = getattr(self.student_rgb, 'test_cfg', None)  # 中文注释：否则尝试读取测试配置
        if test_cfg is not None and hasattr(test_cfg, 'rpn'):  # 中文注释：若测试配置包含rpn字段则返回
            return test_cfg.rpn  # 中文注释：返回测试态候选框配置
        return None  # 中文注释：若均未配置则返回None表示使用默认行为


if __name__ == '__main__':  # 中文注释：提供最小自检示例便于快速验证导入
    class _ToyRPNHead(torch.nn.Module):  # 中文注释：定义简化版RPN头用于自检
        def loss_and_predict(self, feats, samples, proposal_cfg=None):  # 中文注释：实现最小接口返回损失与候选框
            loss = {'loss_cls': sum(feat.sum() for feat in feats) * 0.0}  # 中文注释：构造恒为零的占位损失
            proposals = [torch.zeros((1, 4), device=feats[0].device)] * len(samples)  # 中文注释：返回全零候选框占位符
            return loss, proposals  # 中文注释：返回损失与候选框列表

    class _ToyROIHead(torch.nn.Module):  # 中文注释：定义简化版ROI头用于自检
        def loss(self, feats, proposals, samples):  # 中文注释：实现最小损失接口
            return {'loss_roi': sum(feat.sum() for feat in feats) * 0.0}  # 中文注释：返回恒零占位损失

    class _ToyDetector(BaseDetector):  # 中文注释：定义简化版扩散检测器用于自检
        def __init__(self):  # 中文注释：初始化简化检测器
            super().__init__(data_preprocessor=None)  # 中文注释：调用父类初始化
            self.rpn_head = _ToyRPNHead()  # 中文注释：附加简化RPN头
            self.roi_head = _ToyROIHead()  # 中文注释：附加简化ROI头
            self.train_cfg = type('cfg', (), {'rpn_proposal': None})()  # 中文注释：构造最小训练配置对象
            self.test_cfg = type('cfg', (), {'rpn': None})()  # 中文注释：构造最小测试配置对象

        @property
        def with_rpn(self):  # 中文注释：指示存在RPN头
            return True  # 中文注释：返回True以匹配主类逻辑

        @property
        def with_roi_head(self):  # 中文注释：指示存在ROI头
            return True  # 中文注释：返回True以匹配主类逻辑

        def extract_feat(self, batch_inputs: Tensor):  # 中文注释：实现最小特征提取接口
            return (batch_inputs.mean(dim=1, keepdim=True),)  # 中文注释：返回单尺度特征张量

        def loss(self, batch_inputs: Tensor, batch_data_samples: SampleList):  # 中文注释：实现最小监督损失接口
            return {'loss_dummy': batch_inputs.sum() * 0.0}  # 中文注释：返回零损失保持兼容

        def predict(self, batch_inputs: Tensor, batch_data_samples: SampleList, rescale: bool = True):  # 中文注释：实现最小预测接口
            return []  # 中文注释：返回空列表作为占位预测

        def _forward(self, batch_inputs: Tensor, batch_data_samples: SampleList):  # 中文注释：实现最小前向接口
            return tuple()  # 中文注释：返回空元组表示无附加输出

    class _ToySample:  # 中文注释：定义最小数据样本结构
        def __init__(self):  # 中文注释：初始化样本
            self.gt_instances = type('gt', (), {'labels': torch.zeros(1, dtype=torch.long)})()  # 中文注释：创建带标签字段的占位实例
            self.proposals = torch.zeros((1, 4))  # 中文注释：提供占位候选框

    teacher = _ToyDetector()  # 中文注释：实例化简化教师
    student = _ToyDetector()  # 中文注释：实例化简化学生
    model = DualDiffFusionStage1(teacher, student, train_cfg=dict())  # 中文注释：构建第一阶段融合模型
    dummy_inputs = torch.randn(1, 3, 8, 8)  # 中文注释：创建随机输入张量
    dummy_samples = [_ToySample()]  # 中文注释：构建单个占位样本列表
    output = model.loss(dummy_inputs, dummy_samples)  # 中文注释：执行一次损失前向以验证接口
    print(sorted(output.keys()))  # 中文注释：打印损失键列表确认输出结构
