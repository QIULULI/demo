# Copyright (c) OpenMMLab. All rights reserved.
import copy
import inspect  # 新增中文注释：引入inspect模块以便动态检查函数签名
from typing import Any, Dict, List, Optional, Sequence, Tuple  # 中文注释：引入类型注解工具以便描述复杂结构
import torch
import torch.nn as nn  # 中文注释：用于类型注解和潜在模块构建操作
import torch.nn.functional as F  # 中文注释：引入函数式API以便计算特征间的均方误差等损失
from torch import Tensor

from mmdet.models.utils import (rename_loss_dict,
                                reweight_loss_dict)
from mmdet.registry import MODELS
from mmdet.structures import SampleList
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from .base import BaseDetector
from ..utils import unpack_gt_instances
from mmdet.structures.bbox import bbox2roi
from ..losses import KDLoss

@MODELS.register_module()
class DomainGeneralizationDetector(BaseDetector):
    """Base class for semi-supervised detectors.

    Semi-supervised detectors typically consisting of a teacher model
    updated by exponential moving average and a student model updated
    by gradient descent.

    Args:
        detector (:obj:`ConfigDict` or dict): The detector config.
        semi_train_cfg (:obj:`ConfigDict` or dict, optional):
            The semi-supervised training config.
        semi_test_cfg (:obj:`ConfigDict` or dict, optional):
            The semi-supervised testing config.
        data_preprocessor (:obj:`ConfigDict` or dict, optional): Config of
            :class:`DetDataPreprocessor` to process the input data.
            Defaults to None.
        init_cfg (:obj:`ConfigDict` or list[:obj:`ConfigDict`] or dict or
            list[dict], optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self,
                 detector: ConfigType,
                 train_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None) -> None:
        assert train_cfg is not None, "train_cfg is must not None"
        super().__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        self.model = MODELS.build(detector)
        self.detector_name = detector.get('type')
        self.train_cfg = train_cfg
            
        # cross model setting
        self.cross_loss_cfg = self.train_cfg.cross_loss_cfg  # 中文注释：缓存交叉蒸馏配置以便复用子项
        self.cross_loss_weight = self.cross_loss_cfg.get('cross_loss_weight')  # 中文注释：读取交叉蒸馏初始权重
        self.cross_feature_loss_weight = self.cross_loss_cfg.get('cross_feature_loss_weight', 0.0)  # 中文注释：读取交叉教师特征蒸馏权重默认0
        if self.cross_feature_loss_weight <= 0:  # 中文注释：当交叉特征蒸馏权重未显式配置时尝试兼容旧配置
            legacy_cross_weight = self.train_cfg.feature_loss_cfg.get('cross_feature_loss_weight', 0.0)  # 中文注释：读取旧版配置中的交叉特征蒸馏权重
            if legacy_cross_weight > 0:  # 中文注释：当旧配置存在有效权重时沿用该值
                self.cross_feature_loss_weight = legacy_cross_weight  # 中文注释：覆盖默认值确保向后兼容
        self.cross_roi_kd_weight = self.cross_loss_cfg.get('cross_roi_kd_weight', 0.0)  # 中文注释：读取交叉ROI蒸馏额外权重默认0
        self.cross_cls_loss_weight = self.cross_loss_cfg.get('cls_consistency_weight', 0.0)  # 中文注释：读取交叉分类一致性权重默认0
        if self.cross_cls_loss_weight <= 0:  # 中文注释：当未设置交叉分类一致性权重时尝试读取旧版配置
            legacy_cls_weight = (self.train_cfg.feature_loss_cfg.get('cross_consistency_cfg', {}) or {}).get('cls_weight', 0.0)  # 中文注释：从旧配置的子字典中获取分类权重
            if legacy_cls_weight > 0:  # 中文注释：存在有效旧权重时启用
                self.cross_cls_loss_weight = legacy_cls_weight  # 中文注释：使用旧权重保持行为一致
        self.cross_reg_loss_weight = self.cross_loss_cfg.get('reg_consistency_weight', 0.0)  # 中文注释：读取交叉回归一致性权重默认0
        if self.cross_reg_loss_weight <= 0:  # 中文注释：当未设置交叉回归一致性权重时尝试读取旧版配置
            legacy_reg_weight = (self.train_cfg.feature_loss_cfg.get('cross_consistency_cfg', {}) or {}).get('reg_weight', 0.0)  # 中文注释：从旧配置的子字典中获取回归权重
            if legacy_reg_weight > 0:  # 中文注释：存在有效旧权重时沿用
                self.cross_reg_loss_weight = legacy_reg_weight  # 中文注释：覆盖默认值保证向后兼容
        raw_schedule = self.cross_loss_cfg.get('schedule', [])  # 中文注释：获取阶段性调度配置
        self.cross_loss_schedule = sorted(raw_schedule, key=lambda item: item.get('start_iter', 0))  # 按起始迭代排序调度表
        self.cross_schedule_stage = -1  # 记录当前处于的调度阶段索引
        # feature loss setting
        self.feature_loss_type = self.train_cfg.feature_loss_cfg.get(
            'feature_loss_type')
        self.feature_loss_weight = self.train_cfg.feature_loss_cfg.get(
            'feature_loss_weight')
        self.feature_loss = KDLoss(
            loss_weight=self.feature_loss_weight, loss_type=self.feature_loss_type)
        self.cross_feature_loss = None  # 中文注释：初始化交叉教师特征蒸馏损失为None避免未配置时误用
        if self.cross_feature_loss_weight > 0:  # 中文注释：仅当交叉特征蒸馏权重大于0时构建损失实例
            self.cross_feature_loss = KDLoss(  # 中文注释：基于配置的损失类型与权重创建知识蒸馏损失
                loss_weight=self.cross_feature_loss_weight, loss_type=self.feature_loss_type)  # 中文注释：沿用主特征蒸馏的损失类型确保一致

        self.loss_cls_kd = MODELS.build(self.train_cfg.kd_cfg['loss_cls_kd'])
        self.loss_reg_kd = MODELS.build(self.train_cfg.kd_cfg['loss_reg_kd'])
        
        self.burn_up_iters = self.train_cfg.get('burn_up_iters', 0)
        self.local_iter = 0

    def _update_cross_schedule(self) -> None:
        """动态调整交叉蒸馏权重以及底层扩散教师的启用状态。"""
        if not self.cross_loss_schedule:  # 若未配置调度策略则直接返回
            return  # 不执行任何更新
        while (self.cross_schedule_stage + 1 < len(self.cross_loss_schedule)
               and self.local_iter >= self.cross_loss_schedule[self.cross_schedule_stage + 1].get('start_iter', 0)):  # 检查是否需要推进阶段
            self.cross_schedule_stage += 1  # 推进到下一个阶段
            stage_cfg = self.cross_loss_schedule[self.cross_schedule_stage]  # 取出当前阶段配置
            new_weight = stage_cfg.get('cross_loss_weight', None)  # 读取阶段交叉蒸馏权重
            if new_weight is not None:  # 若显式提供权重
                self.cross_loss_weight = new_weight  # 更新交叉蒸馏权重
            target_teacher = stage_cfg.get('active_teacher', None)  # 读取期望启用的扩散教师名称
            if target_teacher and hasattr(self.model, 'set_active_diff_detector'):  # 当提供教师名称且底层模型支持切换
                self.model.set_active_diff_detector(target_teacher)  # 切换到底层指定教师
            dynamic_teacher_weight = stage_cfg.get('trainable_teacher_loss_weight', None)  # 中文注释：读取阶段性可训练教师损失权重
            if dynamic_teacher_weight is not None:  # 中文注释：若调度表指定了新权重则更新
                self.trainable_teacher_loss_weight = dynamic_teacher_weight  # 中文注释：同步可训练教师损失权重以配合阶段调度

    @property
    def with_rpn(self):
        if self.with_student:
            return hasattr(self.model.student, 'rpn_head')
        else:
            return hasattr(self.student, 'rpn_head')

    @property
    def with_student(self):
        return hasattr(self.model, 'student')

    def loss(self, batch_inputs: Dict[str, Tensor],
             batch_data_samples: Dict[str, SampleList]) -> dict:
        """Calculate losses from multi-branch inputs and data samples.

        Args:

        Returns:
            dict: A dictionary of loss components
        """
        self._update_cross_schedule()  # 在每次迭代前更新调度状态
        losses = dict()  # 初始化损失容器
        if self.local_iter >= self.burn_up_iters:
            # losses.update(**self.model.student.loss(batch_inputs, batch_data_samples))
            losses.update(**self.loss_cross(batch_inputs, batch_data_samples))
        else:
            losses.update(**self.model.student.loss(batch_inputs, batch_data_samples))
        self.local_iter += 1
        return losses

    def predict(self, batch_inputs: Tensor,  # 新增中文注释：输入的图像张量
                batch_data_samples: SampleList,  # 新增中文注释：输入的样本信息列表
                rescale: bool = True,  # 新增中文注释：是否对最终预测结果进行还原缩放
                return_feature: bool = False) -> SampleList:  # 新增中文注释：是否返回特征用于后续蒸馏等操作
        """Predict results from a batch of inputs and data samples with post-
        processing.

        Args:
            batch_inputs (Tensor): Inputs with shape (N, C, H, W).
            batch_data_samples (List[:obj:`DetDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.
            rescale (bool): Whether to rescale the results.
                Defaults to True.
            return_feature (bool): Whether to return additional feature
                representations from detector heads when supported.
                Defaults to False.

        Returns:
            list[:obj:`DetDataSample`] or tuple: When ``return_feature`` is
            False, returns only the detection results. When True, returns
            a tuple containing detection results list and corresponding
            feature representations compatible with
            ``SemiBaseDiffDetector.get_pseudo_instances_diff``.

                - scores (Tensor): Classification scores, has a shape
                    (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                    (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                    the last dimension 4 arrange as (x1, y1, x2, y2).
                - masks (Tensor): Has a shape (num_instances, H, W).
        """
        result = None  # 新增中文注释：初始化预测结果变量
        def _predict_with_signature(target_module):  # 新增中文注释：定义内部函数用于根据签名安全调用预测接口
            predict_method = getattr(target_module, 'predict', None)  # 新增中文注释：尝试从目标模块上获取predict方法
            if predict_method is None:  # 新增中文注释：若目标模块未实现predict方法
                raise AttributeError(f'{target_module.__class__.__name__} does not implement predict method')  # 新增中文注释：抛出异常提示缺少预测接口
            signature = inspect.signature(predict_method)  # 新增中文注释：使用inspect获取predict方法的参数签名
            accepts_return_feature = 'return_feature' in signature.parameters  # 新增中文注释：判断签名中是否包含return_feature参数
            if accepts_return_feature:  # 新增中文注释：若支持return_feature参数
                return predict_method(batch_inputs, batch_data_samples, rescale=rescale, return_feature=return_feature)  # 新增中文注释：直接传入rescale和return_feature执行预测
            prediction = predict_method(batch_inputs, batch_data_samples, rescale=rescale)  # 新增中文注释：当不支持return_feature时仅传入rescale执行预测
            if return_feature:  # 新增中文注释：若调用者期望返回特征但模块不支持
                return prediction, None  # 新增中文注释：补充None占位以满足接口约定
            return prediction  # 新增中文注释：返回原始预测结果

        if self.with_student:  # 新增中文注释：判断当前模型是否包含学生模型分支
            predict_on = self.model.semi_test_cfg.get('predict_on', 'teacher')  # 新增中文注释：获取推理时应使用的模型分支
            if predict_on == 'teacher':  # 新增中文注释：若选择教师模型进行预测
                result = _predict_with_signature(self.model.teacher)  # 新增中文注释：调用教师模型的安全预测包装函数
            elif predict_on == 'student':  # 新增中文注释：若选择学生模型进行预测
                result = _predict_with_signature(self.model.student)  # 新增中文注释：调用学生模型的安全预测包装函数
            elif predict_on == 'diff_detector':  # 新增中文注释：若选择扩散检测器进行预测
                result = _predict_with_signature(self.model.diff_detector)  # 新增中文注释：调用扩散检测器的安全预测包装函数
            else:  # 新增中文注释：若配置中的predict_on值未被识别
                raise ValueError(f'Unsupported predict_on value: {predict_on}')  # 新增中文注释：抛出异常提示非法配置
        else:  # 新增中文注释：若不存在师生结构则直接调用底层模型
            result = _predict_with_signature(self.model)  # 新增中文注释：调用基础模型的安全预测包装函数

        if return_feature and not isinstance(result, tuple):  # 新增中文注释：当需要特征且返回值非元组时进行兼容性处理
            result = (result, None)  # 新增中文注释：包装为包含预测结果及占位特征的元组
        return result  # 新增中文注释：返回最终的预测结果（及可选特征）

    def _forward(self, batch_inputs: Tensor,
                 batch_data_samples: SampleList) -> SampleList:
        """Network forward process. Usually includes backbone, neck and head
        forward without any post-processing.

        Args:
            batch_inputs (Tensor): Inputs with shape (N, C, H, W).

        Returns:
            tuple: A tuple of features from ``rpn_head`` and ``roi_head``
            forward.
        """
        return self.model(
            batch_inputs, batch_data_samples, mode='tensor')

    def extract_feat(self, batch_inputs: Tensor):
        """Extract features.

        Args:
            batch_inputs (Tensor): Image tensor with shape (N, C, H ,W).

        Returns:
            tuple[Tensor]: Multi-level features that may have
            different resolutions.
        """
        if not self.with_student:
            x_backbone = self.model.backbone(batch_inputs)
            x_neck = self.model.neck(x_backbone)
        else:
            x_backbone = self.model.student.backbone(batch_inputs)
            x_neck = self.model.student.neck(x_backbone)
        return x_neck

    def extract_feat_from_diff(self, batch_inputs: Tensor):
        """Extract features.

        Args:
            batch_inputs (Tensor): Image tensor with shape (N, C, H ,W).

        Returns:
            tuple[Tensor]: Multi-level features that may have
            different resolutions.
        """
        x_backbone = self.model.diff_detector.backbone(batch_inputs)
        x_neck = self.model.diff_detector.neck(x_backbone)

        return x_neck

    def _group_indices_by_sensor(self, batch_data_samples: SampleList) -> Dict[str, list]:  # 中文注释：根据传感器标签将批次样本索引分组
        sensor_to_indices = dict()  # 中文注释：初始化传感器到样本索引的映射字典
        for idx, data_sample in enumerate(batch_data_samples):  # 中文注释：遍历批次中的每一个数据样本
            if not hasattr(data_sample, 'metainfo') or 'sensor' not in data_sample.metainfo:  # 中文注释：校验样本是否包含必要的传感器元信息
                raise KeyError(f'Data sample at index {idx} is missing sensor tag for domain generalization training.')  # 中文注释：缺失传感器标记时抛出明确的错误提示
            sensor_tag = data_sample.metainfo['sensor']  # 中文注释：读取当前样本对应的传感器标识
            sensor_to_indices.setdefault(sensor_tag, []).append(idx)  # 中文注释：将样本索引加入对应传感器的列表中
        return sensor_to_indices  # 中文注释：返回按照传感器聚合的索引映射

    def _group_indices_by_sensor_tags(self, sensor_tags: Sequence[str]) -> Dict[str, List[int]]:  # 中文注释：依据传感器标签序列生成索引分组
        sensor_to_indices: Dict[str, List[int]] = dict()  # 中文注释：初始化空字典存放传感器到索引的映射
        for idx, sensor_tag in enumerate(sensor_tags):  # 中文注释：遍历每一个样本位置与对应的传感器标签
            if sensor_tag is None:  # 中文注释：若传感器标签为空则直接报错提示数据异常
                raise KeyError(f'Sensor tag is missing at batch index {idx}.')  # 中文注释：抛出键错误以便快速定位问题
            sensor_to_indices.setdefault(sensor_tag, []).append(idx)  # 中文注释：将当前索引加入对应传感器的索引列表
        return sensor_to_indices  # 中文注释：返回构建好的传感器索引映射

    def _normalize_group_feature_payload(self, feature_payload: Any, group_size: int) -> Tuple[Any, List[Any]]:  # 中文注释：规范化教师特征结构并生成逐样本列表
        if feature_payload is None:  # 中文注释：若输入特征为空直接返回空结果
            return None, [None] * group_size  # 中文注释：返回空特征及填充的None列表保持长度一致
        if isinstance(feature_payload, (list, tuple)) and feature_payload:  # 中文注释：当输入为非空列表或元组时进入详细判断
            first_item = feature_payload[0]  # 中文注释：提取第一个元素用于判断当前组织形式
            if torch.is_tensor(first_item):  # 中文注释：若首元素为张量说明结构按尺度存储
                level_tensors = [level_feat for level_feat in feature_payload]  # 中文注释：逐层复制张量列表保持原有排列
                per_sample_features: List[List[Tensor]] = []  # 中文注释：初始化逐样本特征容器
                level_count = len(level_tensors)  # 中文注释：记录尺度数量便于后续遍历
                for sample_idx in range(group_size):  # 中文注释：遍历组内每个样本索引
                    sample_levels = [level_feat[sample_idx] for level_feat in level_tensors]  # 中文注释：抽取当前样本在各尺度的特征切片
                    per_sample_features.append(sample_levels)  # 中文注释：将切片结果追加到逐样本列表
                return level_tensors, per_sample_features  # 中文注释：返回规范化后的多尺度张量与逐样本特征
            if isinstance(first_item, (list, tuple)) and first_item:  # 中文注释：若首元素本身为列表或元组说明特征已按样本展开
                per_sample_features = [list(sample_feat) for sample_feat in feature_payload]  # 中文注释：逐样本复制以避免共享引用
                level_count = len(first_item)  # 中文注释：读取单个样本包含的尺度数量
                stacked_levels: List[Tensor] = []  # 中文注释：初始化按尺度堆叠的特征列表
                for level_idx in range(level_count):  # 中文注释：遍历每一个尺度索引
                    level_stack = torch.stack([sample_feat[level_idx] for sample_feat in per_sample_features], dim=0)  # 中文注释：沿批维堆叠同尺度的所有样本特征
                    stacked_levels.append(level_stack)  # 中文注释：将堆叠后的张量保存以供后续前向使用
                return stacked_levels, per_sample_features  # 中文注释：返回多尺度堆叠张量与逐样本特征
        if torch.is_tensor(feature_payload):  # 中文注释：若输入直接为张量则视作单尺度特征
            per_sample_features = [feature_payload[sample_idx:sample_idx + 1] for sample_idx in range(group_size)]  # 中文注释：逐样本切片生成单尺度特征
            return [feature_payload], per_sample_features  # 中文注释：将张量包装成列表并返回逐样本特征
        per_sample_features = [feature_payload for _ in range(group_size)]  # 中文注释：对于其他类型直接复制引用以保持长度
        return feature_payload, per_sample_features  # 中文注释：返回原始特征对象与逐样本列表

    def _aggregate_sample_level_features(self, sample_feature_list: Sequence[Any]) -> Optional[List[Tensor]]:  # 中文注释：将逐样本特征重排为按尺度组织的批量张量
        if not isinstance(sample_feature_list, (list, tuple)) or not sample_feature_list:  # 中文注释：当输入不是序列或为空时无需处理
            return None  # 中文注释：直接返回None表示缺少有效特征
        level_count = 0  # 中文注释：初始化统计最大尺度数量的变量
        for sample_entry in sample_feature_list:  # 中文注释：遍历每个样本以确定潜在的尺度数量
            if isinstance(sample_entry, (list, tuple)) and sample_entry:  # 中文注释：当样本提供多尺度列表时参与统计
                level_count = max(level_count, len(sample_entry))  # 中文注释：更新最大尺度数量
            elif torch.is_tensor(sample_entry):  # 中文注释：当样本仅提供单个张量时至少存在一个尺度
                level_count = max(level_count, 1)  # 中文注释：记录单尺度情况
        if level_count == 0:  # 中文注释：若所有样本均未提供有效张量
            return None  # 中文注释：无法构建尺度聚合时返回None
        aggregated_levels: List[Optional[Tensor]] = []  # 中文注释：初始化按尺度存放批量张量的列表
        for level_idx in range(level_count):  # 中文注释：逐尺度构建批量张量
            reference_tensor: Optional[Tensor] = None  # 中文注释：记录当前尺度的参考张量以便填充缺失项
            ordered_candidates: List[Optional[Tensor]] = []  # 中文注释：按样本顺序收集候选张量
            for sample_entry in sample_feature_list:  # 中文注释：遍历每个样本提取当前尺度
                candidate_tensor = None  # 中文注释：初始化候选张量占位
                if isinstance(sample_entry, (list, tuple)) and len(sample_entry) > level_idx:  # 中文注释：当样本具有足够尺度时读取对应项
                    candidate_value = sample_entry[level_idx]  # 中文注释：获取当前尺度的对象
                    if torch.is_tensor(candidate_value):  # 中文注释：仅处理真实张量
                        candidate_tensor = candidate_value  # 中文注释：记录可用张量
                elif torch.is_tensor(sample_entry) and level_idx == 0:  # 中文注释：当样本为单张量且仅能贡献第一个尺度
                    candidate_tensor = sample_entry  # 中文注释：直接使用该张量
                if reference_tensor is None and candidate_tensor is not None:  # 中文注释：若尚未确定参考张量则记录首个有效张量
                    reference_tensor = candidate_tensor  # 中文注释：用于后续填充缺失样本
                ordered_candidates.append(candidate_tensor)  # 中文注释：保持样本顺序存放候选结果
            if reference_tensor is None:  # 中文注释：若当前尺度不存在任何有效张量
                aggregated_levels.append(None)  # 中文注释：以None占位保持索引一致
                continue  # 中文注释：继续处理下一个尺度
            normalized_slices: List[Tensor] = []  # 中文注释：存放填充后的样本张量
            for candidate_tensor in ordered_candidates:  # 中文注释：遍历每个样本的候选
                if candidate_tensor is None:  # 中文注释：当该样本缺失当前尺度时
                    normalized_slices.append(torch.zeros_like(reference_tensor))  # 中文注释：填充同形状零张量保持尺寸一致
                else:  # 中文注释：当样本提供该尺度时直接使用其原始张量
                    normalized_slices.append(candidate_tensor)  # 中文注释：保留原始张量
            aggregated_levels.append(torch.stack(normalized_slices, dim=0))  # 中文注释：将所有样本在该尺度的张量按batch维堆叠
        has_valid_level = any(torch.is_tensor(level_tensor) for level_tensor in aggregated_levels if level_tensor is not None)  # 中文注释：检查是否存在有效的尺度张量
        return aggregated_levels if has_valid_level else None  # 中文注释：若存在有效张量则返回聚合结果否则返回None

    def _average_cross_peer_feature_map(self, peer_feature_map: Any) -> Optional[List[Tensor]]:  # 中文注释：对同一样本内来自多教师的特征求平均
        if not isinstance(peer_feature_map, dict) or not peer_feature_map:  # 中文注释：当结构不是字典或为空时无法求平均
            return None  # 中文注释：直接返回None表示缺失有效同伴特征
        level_count = 0  # 中文注释：初始化最大尺度数量统计
        for feature_entry in peer_feature_map.values():  # 中文注释：遍历每个同伴教师提供的特征
            if isinstance(feature_entry, (list, tuple)) and feature_entry:  # 中文注释：当特征为多尺度列表时参与统计
                level_count = max(level_count, len(feature_entry))  # 中文注释：更新尺度数量
            elif torch.is_tensor(feature_entry):  # 中文注释：当特征为单张量时至少提供一个尺度
                level_count = max(level_count, 1)  # 中文注释：记录单尺度情况
        if level_count == 0:  # 中文注释：若所有同伴均未提供可用张量
            return None  # 中文注释：返回None表示无法构建平均特征
        averaged_levels: List[Optional[Tensor]] = []  # 中文注释：初始化按尺度存放平均结果的列表
        for level_idx in range(level_count):  # 中文注释：逐尺度聚合同伴教师特征
            candidates: List[Tensor] = []  # 中文注释：记录当前尺度的有效张量集合
            for feature_entry in peer_feature_map.values():  # 中文注释：遍历每个教师的特征条目
                candidate_tensor = None  # 中文注释：初始化候选张量占位
                if isinstance(feature_entry, (list, tuple)) and len(feature_entry) > level_idx:  # 中文注释：当教师提供足够尺度时提取对应张量
                    tensor_value = feature_entry[level_idx]  # 中文注释：读取当前尺度的张量
                    if torch.is_tensor(tensor_value):  # 中文注释：仅处理张量类型
                        candidate_tensor = tensor_value  # 中文注释：记录有效张量
                elif torch.is_tensor(feature_entry) and level_idx == 0:  # 中文注释：当教师仅提供单尺度张量时
                    candidate_tensor = feature_entry  # 中文注释：默认该张量对应首个尺度
                if candidate_tensor is not None:  # 中文注释：若找到有效张量则加入集合
                    candidates.append(candidate_tensor)  # 中文注释：累计同尺度的所有教师张量
            if candidates:  # 中文注释：当至少一个教师提供当前尺度时
                stacked = torch.stack(candidates, dim=0)  # 中文注释：沿新增维度堆叠全部候选张量
                averaged_levels.append(stacked.mean(dim=0))  # 中文注释：对堆叠结果取平均得到统一的交叉特征
            else:  # 中文注释：当所有教师都缺失该尺度时仅以None占位
                averaged_levels.append(None)  # 中文注释：若所有教师缺失该尺度则以None占位
        has_valid = any(level is not None for level in averaged_levels)  # 中文注释：检查是否存在有效的平均结果
        return averaged_levels if has_valid else None  # 中文注释：仅在存在有效尺度时返回平均后的特征列表

    def _prepare_cross_teacher_features(self, cross_teacher_info: Any, sensor_to_indices: Dict[str, List[int]], batch_size: int) -> Tuple[Optional[List[Tensor]], Dict[str, Any]]:  # 中文注释：解析交叉教师信息并返回聚合特征及按传感器划分的结构
        cross_payload = None  # 中文注释：初始化交叉教师原始特征容器
        if isinstance(cross_teacher_info, dict):  # 中文注释：当交叉教师信息为字典时按约定键读取特征
            cross_payload = cross_teacher_info.get('features', cross_teacher_info.get('main_feature'))  # 中文注释：优先从features或main_feature键中取值
        else:  # 中文注释：当交叉教师信息非字典时直接将其视作原始特征
            cross_payload = cross_teacher_info  # 中文注释：非字典结构直接视作特征载体
        per_sample_features: List[Any] = [None] * batch_size  # 中文注释：预先分配逐样本特征列表保持与批量索引一致
        if isinstance(cross_payload, (list, tuple)) and len(cross_payload) == batch_size:  # 中文注释：当提供逐样本列表时逐个解析
            for sample_idx, entry in enumerate(cross_payload):  # 中文注释：遍历每个样本的交叉特征条目
                if isinstance(entry, dict):  # 中文注释：条目为同伴教师字典时需要求平均
                    per_sample_features[sample_idx] = self._average_cross_peer_feature_map(entry)  # 中文注释：对同伴特征求平均生成统一表示
                elif isinstance(entry, (list, tuple)):  # 中文注释：当条目直接为多尺度列表时保留原样
                    per_sample_features[sample_idx] = [tensor for tensor in entry]  # 中文注释：复制列表避免共享引用
                elif torch.is_tensor(entry):  # 中文注释：当条目是单个张量时包装成单尺度列表
                    per_sample_features[sample_idx] = [entry]  # 中文注释：确保后续处理统一
                else:  # 中文注释：当条目结构未知或为空时直接置为None
                    per_sample_features[sample_idx] = None  # 中文注释：未知结构时以None占位
        elif isinstance(cross_payload, (list, tuple)) and cross_payload and all(torch.is_tensor(item) for item in cross_payload):  # 中文注释：当交叉特征以尺度优先列表形式提供
            sample_limit = min(batch_size, cross_payload[0].shape[0])  # 中文注释：确定可切片的样本数量以防越界
            per_sample_features = [[level_feat[sample_idx] for level_feat in cross_payload] for sample_idx in range(sample_limit)]  # 中文注释：沿批维切片构造逐样本尺度列表
            if len(per_sample_features) < batch_size:  # 中文注释：若教师批量小于学生批量
                per_sample_features.extend([None] * (batch_size - len(per_sample_features)))  # 中文注释：使用None填充缺失样本避免索引错误
        elif torch.is_tensor(cross_payload):  # 中文注释：当交叉教师仅返回单个批量张量
            sample_limit = min(batch_size, cross_payload.shape[0])  # 中文注释：确定可用的样本数量
            per_sample_features = [[cross_payload[sample_idx]] for sample_idx in range(sample_limit)]  # 中文注释：逐样本切片并包装成列表
            if len(per_sample_features) < batch_size:  # 中文注释：若批量不足则补齐剩余位置
                per_sample_features.extend([None] * (batch_size - len(per_sample_features)))  # 中文注释：填充None保持长度一致
        aggregated_levels = self._aggregate_sample_level_features(per_sample_features)  # 中文注释：将逐样本特征聚合为尺度优先结构
        grouped_cross_features = dict()  # 中文注释：初始化按传感器划分的交叉特征字典
        if aggregated_levels is not None and sensor_to_indices:  # 中文注释：存在有效聚合结果且给定传感器索引时才执行切分
            grouped_cross_features = self._slice_teacher_features_by_sensor(aggregated_levels, sensor_to_indices)  # 中文注释：按照传感器索引切分交叉特征
        return aggregated_levels, grouped_cross_features  # 中文注释：返回尺度优先张量列表与按传感器整理的特征

    def loss_cross_consistency(self, cross_teacher_info: Any) -> dict:  # 中文注释：根据交叉教师提供的一致性指标计算损失
        losses = dict()  # 中文注释：初始化损失字典
        if not isinstance(cross_teacher_info, dict):  # 中文注释：当交叉教师信息非字典时认为没有可用的一致性指标
            return losses  # 中文注释：直接返回空字典保持兼容
        if self.cross_cls_loss_weight > 0:  # 中文注释：仅在分类一致性权重大于0时才计算该项
            cls_value = cross_teacher_info.get('cls_consistency')  # 中文注释：优先读取顶层的分类一致性标量
            if cls_value is None:  # 中文注释：若顶层缺失则尝试从嵌套字典中获取
                consistency_block = cross_teacher_info.get('consistency') or {}  # 中文注释：读取可能的consistency子字典
                cls_value = consistency_block.get('cls')  # 中文注释：从子字典中提取分类一致性数值
            if cls_value is not None:  # 中文注释：成功取得数值后才写入损失
                losses['cross_cls_consistency_loss'] = cls_value * self.cross_cls_loss_weight  # 中文注释：按配置权重缩放分类一致性损失
        if self.cross_reg_loss_weight > 0:  # 中文注释：仅在回归一致性权重大于0时计算该项
            reg_value = cross_teacher_info.get('reg_consistency')  # 中文注释：优先读取顶层的回归一致性标量
            if reg_value is None:  # 中文注释：若顶层缺失则尝试从嵌套结构读取
                consistency_block = cross_teacher_info.get('consistency') or {}  # 中文注释：读取可能存在的consistency子字典
                reg_value = consistency_block.get('reg')  # 中文注释：从子字典中提取回归一致性数值
            if reg_value is not None:  # 中文注释：成功取得数值后才写入损失
                losses['cross_reg_consistency_loss'] = reg_value * self.cross_reg_loss_weight  # 中文注释：按权重缩放回归一致性损失
        return losses  # 中文注释：返回交叉教师一致性损失字典

    def _distribute_group_roi_outputs(self, group_outputs: Dict[str, Tensor],
                                      storage: List[Dict[str, Tensor]],
                                      sample_indices: List[int],
                                      roi_counts: List[int]) -> Dict[str, Tensor]:
        distribution_cache: Dict[str, Tensor] = dict()  # 中文注释：初始化缓存用于记录空张量模板
        for key, value in group_outputs.items():  # 中文注释：遍历教师前向结果中的每个键值对
            if value is None:  # 中文注释：若值为空则跳过该键
                continue  # 中文注释：继续处理下一个输出项
            distribution_cache.setdefault(key, value.new_empty((0,) + value.shape[1:]))  # 中文注释：为当前键构建零长度模板以备样本无ROI时使用
            start = 0  # 中文注释：初始化切片起始位置
            for local_idx, (sample_idx, roi_num) in enumerate(zip(sample_indices, roi_counts)):  # 中文注释：遍历组内样本及其对应的ROI数量
                if roi_num == 0:  # 中文注释：当某个样本没有ROI时直接填充零长度张量
                    storage[sample_idx][key] = distribution_cache[key]  # 中文注释：写入预构建的空张量保持结构一致
                    continue  # 中文注释：跳过切片操作处理下一个样本
                end = start + roi_num  # 中文注释：计算当前样本对应的结束位置
                storage[sample_idx][key] = value[start:end]  # 中文注释：将对应片段赋值给该样本位置
                start = end  # 中文注释：更新起始位置准备处理下一段
        return distribution_cache  # 中文注释：返回键到空模板张量的映射供外部补全缺失键

    def _finalize_grouped_outputs(self, aggregated: Dict[str, List[Tensor]],
                                  fallback_cache: Dict[str, Tensor]) -> Dict[str, Tensor]:  # 中文注释：将逐样本存储的张量按批次顺序拼接
        finalized: Dict[str, Tensor] = dict()  # 中文注释：初始化最终输出字典
        for key, sample_tensors in aggregated.items():  # 中文注释：遍历每一个输出键
            tensors_in_order: List[Tensor] = []  # 中文注释：按批次顺序收集张量片段
            for tensor in sample_tensors:  # 中文注释：遍历该键对应的逐样本张量
                if tensor is None:  # 中文注释：若当前样本未写入结果则补齐零长度模板
                    tensor = fallback_cache.get(key)  # 中文注释：读取预先缓存的空张量
                if tensor is None:  # 中文注释：若仍未获取到有效张量则跳过该样本
                    continue  # 中文注释：避免将None加入拼接序列导致错误
                tensors_in_order.append(tensor)  # 中文注释：将张量加入拼接列表
            if tensors_in_order:  # 中文注释：当存在可拼接的张量时执行拼接
                finalized[key] = torch.cat(tensors_in_order, dim=0)  # 中文注释：沿ROI维拼接形成批次输出
        return finalized  # 中文注释：返回已拼接好的输出字典

    def _slice_teacher_features_by_sensor(self, teacher_feature_payload, sensor_to_indices: Dict[str, list]):  # 中文注释：根据传感器索引切分教师特征
        grouped_features = dict()  # 中文注释：初始化以传感器为键的特征字典
        if teacher_feature_payload is None:  # 中文注释：若教师特征为空则直接返回空字典
            return grouped_features  # 中文注释：无可用特征时直接返回
        if isinstance(teacher_feature_payload, (list, tuple)) and teacher_feature_payload and all(torch.is_tensor(item) for item in teacher_feature_payload):  # 中文注释：当教师特征以多层张量列表形式提供时执行逐层切片
            for sensor_tag, indices in sensor_to_indices.items():  # 中文注释：遍历每一个传感器分组
                index_tensor = torch.tensor(indices, device=teacher_feature_payload[0].device, dtype=torch.long)  # 中文注释：将样本索引转换为当前设备上的长整型张量
                grouped_features[sensor_tag] = [level_features.index_select(0, index_tensor) for level_features in teacher_feature_payload]  # 中文注释：对每一层特征按照索引选择出属于当前传感器的样本
            return grouped_features  # 中文注释：返回切分后的特征字典
        if isinstance(teacher_feature_payload, (list, tuple)) and teacher_feature_payload and isinstance(teacher_feature_payload[0], (list, tuple)):  # 中文注释：当教师特征已经按样本打包且每个样本包含多尺度特征时处理
            for sensor_tag, indices in sensor_to_indices.items():  # 中文注释：遍历传感器与对应样本索引
                sample_features = [teacher_feature_payload[idx] for idx in indices]  # 中文注释：取出当前传感器的全部样本特征包
                if not sample_features:  # 中文注释：若该传感器下无样本则跳过
                    continue  # 中文注释：继续处理其他传感器
                level_count = max(len(sample_feature) if isinstance(sample_feature, (list, tuple)) else 0 for sample_feature in sample_features)  # 中文注释：统计样本内可用的最大尺度数量以兼容缺失情况
                if level_count <= 0:  # 中文注释：当无法识别任何尺度时跳过当前传感器
                    continue  # 中文注释：继续处理下一传感器
                grouped_levels = []  # 中文注释：初始化当前传感器的尺度特征列表
                for level_idx in range(level_count):  # 中文注释：遍历每一个尺度
                    level_candidates = []  # 中文注释：为当前尺度收集所有样本的特征切片
                    for sample_feature in sample_features:  # 中文注释：逐个样本提取当前尺度特征
                        if isinstance(sample_feature, (list, tuple)) and len(sample_feature) > level_idx:  # 中文注释：当样本提供足够尺度时直接取值
                            level_candidates.append(sample_feature[level_idx])  # 中文注释：记录当前样本的尺度特征
                        else:  # 中文注释：若样本缺失该尺度则使用占位符稍后补齐
                            level_candidates.append(None)  # 中文注释：以None占位保持索引稳定
                    reference_tensor = next((candidate for candidate in level_candidates if torch.is_tensor(candidate)), None)  # 中文注释：寻找首个有效张量以确定占位形状
                    if reference_tensor is None:  # 中文注释：若全部样本均缺失当前尺度则在输出中以None占位
                        grouped_levels.append(None)  # 中文注释：保持尺度位置以防错位
                        continue  # 中文注释：继续处理下一个尺度
                    normalized_slices = []  # 中文注释：准备存放格式统一的样本切片
                    for candidate in level_candidates:  # 中文注释：遍历原始候选列表
                        if torch.is_tensor(candidate):  # 中文注释：若候选为有效张量直接使用
                            normalized_slices.append(candidate)  # 中文注释：追加到规范化切片列表
                        else:  # 中文注释：当候选缺失时以零张量占位保证维度一致
                            normalized_slices.append(torch.zeros_like(reference_tensor))  # 中文注释：生成同形状零张量占位
                    grouped_levels.append(torch.stack(normalized_slices, dim=0))  # 中文注释：将所有样本在batch维度堆叠得到当前尺度张量
                grouped_features[sensor_tag] = grouped_levels  # 中文注释：登记当前传感器的多尺度特征
            return grouped_features  # 中文注释：返回按传感器切分的多尺度特征
        if torch.is_tensor(teacher_feature_payload):  # 中文注释：当教师特征为单一张量时直接按索引切片
            for sensor_tag, indices in sensor_to_indices.items():  # 中文注释：遍历传感器与索引
                index_tensor = torch.tensor(indices, device=teacher_feature_payload.device, dtype=torch.long)  # 中文注释：构建与特征同设备的索引张量
                grouped_features[sensor_tag] = teacher_feature_payload.index_select(0, index_tensor)  # 中文注释：切分出当前传感器对应的子张量
            return grouped_features  # 中文注释：返回切分后的张量字典
        for sensor_tag in sensor_to_indices:  # 中文注释：针对无法识别的数据结构直接复制引用
            grouped_features[sensor_tag] = teacher_feature_payload  # 中文注释：将原始特征对象挂载到每个传感器键下
        return grouped_features  # 中文注释：返回兜底处理结果

    def _merge_grouped_features(self, grouped_features: Dict[str, list], sensor_to_indices: Dict[str, list], total_count: int):  # 中文注释：按照原始顺序合并按传感器切分的特征
        if not grouped_features:  # 中文注释：若特征字典为空则无需合并
            return None  # 中文注释：直接返回空表示无可用特征
        routing = dict()  # 中文注释：初始化样本索引到传感器局部索引的映射
        for sensor_tag, indices in sensor_to_indices.items():  # 中文注释：遍历每个传感器的样本索引列表
            for local_idx, global_idx in enumerate(indices):  # 中文注释：逐个样本建立局部到全局索引映射
                routing[global_idx] = (sensor_tag, local_idx)  # 中文注释：记录全局索引对应的传感器与组内位置
        reference_sensor = None  # 中文注释：初始化用于推断特征结构的参考传感器
        for sensor_tag in sensor_to_indices:  # 中文注释：按照传感器顺序查找存在特征的参考项
            if sensor_tag in grouped_features:  # 中文注释：若当前传感器拥有有效特征
                reference_sensor = sensor_tag  # 中文注释：选定该传感器作为参考
                break  # 中文注释：找到后立即结束循环
        if reference_sensor is None:  # 中文注释：若没有任何传感器提供特征
            return None  # 中文注释：返回空以标识缺失
        reference_feature = grouped_features[reference_sensor]  # 中文注释：获取参考传感器的特征结构
        if isinstance(reference_feature, (list, tuple)):  # 中文注释：当参考特征为多尺度列表时执行逐尺度拼接
            merged_levels = []  # 中文注释：初始化合并后的多尺度特征容器
            level_count = len(reference_feature)  # 中文注释：读取尺度数量用于遍历
            for level_idx in range(level_count):  # 中文注释：逐个尺度进行拼接
                level_slices = []  # 中文注释：初始化当前尺度下的样本片段列表
                for sample_idx in range(total_count):  # 中文注释：按照原批次顺序遍历样本
                    sensor_tag, local_idx = routing[sample_idx]  # 中文注释：查找当前样本所属传感器及其组内索引
                    current_sensor_features = grouped_features.get(sensor_tag)  # 中文注释：获取该传感器的特征集合
                    if current_sensor_features is None or level_idx >= len(current_sensor_features):  # 中文注释：若缺少当前尺度特征则占位
                        level_slices.append(None)  # 中文注释：以None占位保持序列对齐
                        continue  # 中文注释：继续处理下一个样本
                    level_tensor = current_sensor_features[level_idx]  # 中文注释：提取当前尺度的特征张量
                    if level_tensor is None:  # 中文注释：若该尺度为空则占位
                        level_slices.append(None)  # 中文注释：以None保持索引稳定
                    else:  # 中文注释：当尺度张量有效时切片
                        level_slices.append(level_tensor[local_idx:local_idx + 1])  # 中文注释：截取与当前样本对应的切片并保留批维
                reference_slice = next((slice_tensor for slice_tensor in level_slices if torch.is_tensor(slice_tensor)), None)  # 中文注释：寻找参考切片以便填充占位
                if reference_slice is None:  # 中文注释：若所有切片均缺失则以None记录该尺度
                    merged_levels.append(None)  # 中文注释：将当前尺度标记为空以便后续过滤
                    continue  # 中文注释：进入下一尺度处理
                normalized_slices = []  # 中文注释：用于存放填充后的切片列表
                for slice_tensor in level_slices:  # 中文注释：遍历原始切片
                    if torch.is_tensor(slice_tensor):  # 中文注释：若切片有效则直接使用
                        normalized_slices.append(slice_tensor)  # 中文注释：保存原始切片
                    else:  # 中文注释：当切片缺失时以零张量填充
                        normalized_slices.append(torch.zeros_like(reference_slice))  # 中文注释：生成与参考切片形状一致的零张量填补
                merged_levels.append(torch.cat(normalized_slices, dim=0))  # 中文注释：将规范化切片在批维拼接恢复原顺序
            return merged_levels  # 中文注释：返回合并后的多尺度特征列表
        if torch.is_tensor(reference_feature):  # 中文注释：当参考特征为单张量时直接按顺序拼接
            ordered_slices = []  # 中文注释：初始化顺序切片列表
            for sample_idx in range(total_count):  # 中文注释：遍历原始批次索引
                sensor_tag, local_idx = routing[sample_idx]  # 中文注释：确定样本对应的传感器及其局部位置
                sensor_tensor = grouped_features.get(sensor_tag)  # 中文注释：获取当前传感器的张量特征
                if sensor_tensor is None:  # 中文注释：若无特征则跳过
                    continue  # 中文注释：继续处理下一样本
                ordered_slices.append(sensor_tensor[local_idx:local_idx + 1])  # 中文注释：截取当前样本的张量片段
            return torch.cat(ordered_slices, dim=0) if ordered_slices else None  # 中文注释：拼接所有片段并返回
        return reference_feature  # 中文注释：针对非常规结构直接返回参考对象

    def _is_accuracy_metric(self, metric_name: str) -> bool:
        """判断给定的指标名称是否代表准确率信息。"""
        return metric_name.endswith('_acc') or ('acc' in metric_name and 'loss' not in metric_name)  # 中文注释：满足_acc结尾或包含acc但不含loss时视为准确率

    def _merge_metrics_with_average(self, aggregated_losses: Dict[str, Any], sensor_losses: Dict[str, Any], sample_count: int,
                                    accuracy_buffers: Dict[str, Tuple[Any, int]]) -> None:
        """累积损失项并对准确率指标记录样本加权正确数。"""
        for metric_name, metric_value in sensor_losses.items():  # 中文注释：遍历单个传感器产生的所有损失或指标
            if self._is_accuracy_metric(metric_name):  # 中文注释：当指标属于准确率时单独进入加权统计
                correct_increment = metric_value * sample_count  # 中文注释：将准确率乘以样本数量得到正确样本数
                if metric_name not in accuracy_buffers:  # 中文注释：首次遇到该准确率指标时初始化缓存
                    accuracy_buffers[metric_name] = (correct_increment, sample_count)  # 中文注释：记录当前正确样本数与样本总数
                else:  # 中文注释：若已存在缓存则进行累加
                    accumulated_correct, accumulated_total = accuracy_buffers[metric_name]  # 中文注释：取出已累计的正确样本数与样本数量
                    accuracy_buffers[metric_name] = (accumulated_correct + correct_increment,
                                                     accumulated_total + sample_count)  # 中文注释：更新缓存为新的累计值
            else:  # 中文注释：对于普通损失直接累加
                if metric_name in aggregated_losses:  # 中文注释：当总损失中已存在相同键时执行加法
                    aggregated_losses[metric_name] = aggregated_losses[metric_name] + metric_value  # 中文注释：累加相同损失项的值
                else:  # 中文注释：首次出现的损失项直接写入
                    aggregated_losses[metric_name] = metric_value  # 中文注释：将新损失项记录到总损失字典

    def _finalize_accuracy_metrics(self, aggregated_losses: Dict[str, Any], accuracy_buffers: Dict[str, Tuple[Any, int]]) -> None:
        """根据累计的正确样本数恢复平均准确率并写回损失字典。"""
        for metric_name, (correct_count, total_count) in accuracy_buffers.items():  # 中文注释：遍历所有准确率缓存条目
            if total_count > 0:  # 中文注释：仅在累计样本数量大于零时计算平均值
                aggregated_losses[metric_name] = correct_count / total_count  # 中文注释：将正确样本数除以总样本数得到平均准确率

    def cross_loss_diff_to_student(self, grouped_samples: Dict[str, SampleList], grouped_features: Dict[str, list]):  # 中文注释：按传感器分组计算交叉蒸馏损失
        losses = dict()  # 中文注释：初始化损失字典用于累计各传感器的结果
        accuracy_buffers: Dict[str, Tuple[Any, int]] = dict()  # 中文注释：初始化准确率缓存字典用于记录加权结果
        for sensor_tag, samples in grouped_samples.items():  # 中文注释：遍历每个传感器下的样本列表
            if not samples:  # 中文注释：若该传感器下无样本则跳过
                continue  # 中文注释：继续处理其他传感器
            diff_fpn = grouped_features.get(sensor_tag)  # 中文注释：获取当前传感器匹配的教师特征
            if diff_fpn is None:  # 中文注释：若缺少特征则无法计算交叉蒸馏
                continue  # 中文注释：跳过该传感器以保持训练稳定
            sensor_losses = dict()  # 中文注释：初始化当前传感器的损失容器
            if not self.with_rpn:  # 中文注释：当学生模型无RPN分支时直接计算BBox头损失
                detector_loss = self.model.student.bbox_head.loss(diff_fpn, samples)  # 中文注释：使用教师特征与当前样本计算BBox头损失
                sensor_losses.update(rename_loss_dict('cross_', detector_loss))  # 中文注释：为损失项增加交叉蒸馏前缀并记录
            else:  # 中文注释：当学生包含RPN分支时需先计算RPN损失
                proposal_cfg = self.model.student.train_cfg.get('rpn_proposal', self.model.student.test_cfg.rpn)  # 中文注释：读取RPN候选框配置
                rpn_data_samples = copy.deepcopy(samples)  # 中文注释：深拷贝样本以避免修改原始标注
                for data_sample in rpn_data_samples:  # 中文注释：遍历复制后的样本以重写标签
                    data_sample.gt_instances.labels = torch.zeros_like(data_sample.gt_instances.labels)  # 中文注释：将RPN使用的标签全部置零保证二分类训练
                rpn_losses, rpn_results_list = self.model.student.rpn_head.loss_and_predict(diff_fpn, rpn_data_samples, proposal_cfg=proposal_cfg)  # 中文注释：基于教师特征计算RPN损失并获得候选框
                keys = rpn_losses.keys()  # 中文注释：提取RPN损失字典的所有键
                for key in list(keys):  # 中文注释：遍历键以避免与ROI损失重名
                    if 'loss' in key and 'rpn' not in key:  # 中文注释：仅重命名非rpn前缀的损失项
                        rpn_losses[f'rpn_{key}'] = rpn_losses.pop(key)  # 中文注释：为损失键增加rpn前缀
                sensor_losses.update(rename_loss_dict('cross_', rpn_losses))  # 中文注释：汇总带有交叉蒸馏前缀的RPN损失
                roi_losses = self.model.student.roi_head.loss(diff_fpn, rpn_results_list, samples)  # 中文注释：基于教师候选框计算ROI头损失
                sensor_losses.update(rename_loss_dict('cross_', roi_losses))  # 中文注释：记录ROI蒸馏损失
            self._merge_metrics_with_average(losses, sensor_losses, len(samples), accuracy_buffers)  # 中文注释：通过辅助函数合并损失并累积准确率指标
        self._finalize_accuracy_metrics(losses, accuracy_buffers)  # 中文注释：在处理完所有传感器后恢复准确率的平均值
        losses = reweight_loss_dict(losses=losses, weight=self.cross_loss_weight)  # 中文注释：按照调度权重重标定交叉蒸馏损失
        return losses  # 中文注释：返回累计后的交叉蒸馏损失

    def loss_cross(self, batch_inputs: Tensor,
                batch_data_samples: SampleList) -> dict:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            batch_inputs (Tensor): Input images of shape (N, C, H, W).
                These should usually be mean centered and std scaled.
            batch_data_samples (List[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.

        Returns:
            dict: A dictionary of loss components
        """
        pseudo_data_samples, batch_info, diff_feature = self.model.get_pseudo_instances_diff(batch_inputs, batch_data_samples)  # 中文注释：调用多教师接口生成伪标签并返回教师特征
        parsed_diff_feature = self.model._parse_diff_feature(diff_feature, batch_info)  # 中文注释：解析教师输出以获得标准化的特征结构
        batch_data_samples = pseudo_data_samples  # 中文注释：使用带伪标签的样本列表继续后续训练流程
        parsed_sensor_tags: Optional[List[str]] = parsed_diff_feature.get('sensor_map') if isinstance(parsed_diff_feature, dict) else None  # 中文注释：尝试从教师返回中读取逐样本传感器标签
        if parsed_sensor_tags is not None:  # 中文注释：若教师显式返回传感器标签映射
            sensor_to_indices = self._group_indices_by_sensor_tags(parsed_sensor_tags)  # 中文注释：直接根据映射构建索引分组
            sensor_tags_per_sample = parsed_sensor_tags  # 中文注释：保存逐样本传感器标签供后续使用
        else:  # 中文注释：若教师未返回传感器标签则退回到样本元信息
            sensor_to_indices = self._group_indices_by_sensor(batch_data_samples)  # 中文注释：根据传感器信息划分批次索引
            sensor_tags_per_sample = [sample.metainfo['sensor'] for sample in batch_data_samples]  # 中文注释：从样本元信息中提取逐样本传感器标签
        grouped_samples = {sensor_tag: [batch_data_samples[idx] for idx in indices] for sensor_tag, indices in sensor_to_indices.items()}  # 中文注释：按照传感器重组样本列表用于局部蒸馏
        grouped_teacher_features = dict()  # 中文注释：初始化教师特征分组字典
        raw_grouped_feature = parsed_diff_feature.get('grouped_main_teacher') if isinstance(parsed_diff_feature, dict) else None  # 中文注释：尝试直接获取按传感器划分的特征
        if isinstance(raw_grouped_feature, dict) and raw_grouped_feature:  # 中文注释：若教师已提供按传感器整理的特征
            for sensor_tag, indices in sensor_to_indices.items():  # 中文注释：遍历所需的传感器键
                if sensor_tag in raw_grouped_feature:  # 中文注释：仅当教师提供对应特征时才复制
                    grouped_teacher_features[sensor_tag] = raw_grouped_feature[sensor_tag]  # 中文注释：写入当前传感器的多尺度特征
        if not grouped_teacher_features:  # 中文注释：若教师未直接提供分组特征则退回切片逻辑
            grouped_teacher_features = self._slice_teacher_features_by_sensor(parsed_diff_feature.get('main_teacher'), sensor_to_indices)  # 中文注释：提取并切分主教师特征
        student_x = self.model.student.extract_feat(batch_inputs)  # 中文注释：前向学生模型获取自身特征表示
        student_feature_levels = list(student_x) if isinstance(student_x, (list, tuple)) else [student_x]  # 中文注释：将学生特征统一整理成列表以便逐尺度处理
        diff_x = self._merge_grouped_features(grouped_teacher_features, sensor_to_indices, len(batch_data_samples))  # 中文注释：将教师特征按照原批次顺序重新拼接
        cross_teacher_info = parsed_diff_feature.get('cross_teacher') if isinstance(parsed_diff_feature, dict) else None  # 中文注释：提取交叉教师信息块供后续蒸馏使用
        if cross_teacher_info is not None:  # 中文注释：仅在交叉教师信息存在时解析其特征
            cross_x, grouped_cross_features = self._prepare_cross_teacher_features(cross_teacher_info, sensor_to_indices, len(batch_data_samples))  # 中文注释：对交叉教师特征进行聚合并按传感器划分
        else:  # 中文注释：当未启用交叉教师时提供空的特征占位符
            cross_x, grouped_cross_features = None, dict()  # 中文注释：缺少交叉教师时使用占位结果保持接口一致
        cross_consistency_losses = self.loss_cross_consistency(cross_teacher_info)  # 中文注释：计算交叉教师提供的一致性损失项
        losses = dict()  # 中文注释：初始化损失累加字典

        # cross model loss
        ##############################################################################################################
        losses.update(
            **self.cross_loss_diff_to_student(grouped_samples, grouped_teacher_features))  # 中文注释：计算按模态分组的交叉蒸馏损失
        ##############################################################################################################

        # feature kd loss
        ##############################################################################################################
        feature_loss = dict()
        feature_loss['pkd_feature_loss'] = 0
        if isinstance(diff_x, (list, tuple)) and diff_x:  # 中文注释：仅在教师特征有效时计算特征蒸馏
            valid_pairs = [(stu_feat, tea_feat) for stu_feat, tea_feat in zip(student_feature_levels, diff_x) if tea_feat is not None]  # 中文注释：过滤掉缺失教师特征的尺度对
            valid_count = len(valid_pairs)  # 中文注释：统计可参与蒸馏的尺度数量
            if valid_count > 0:  # 中文注释：仅当存在有效尺度时才执行蒸馏
                for student_feature, diff_feature in valid_pairs:  # 中文注释：逐个尺度计算特征损失
                    layer_loss = self.feature_loss(  # 中文注释：调用蒸馏损失函数
                        student_feature, diff_feature)  # 中文注释：传入学生与教师的对应特征
                    feature_loss['pkd_feature_loss'] += layer_loss/valid_count  # 中文注释：将各尺度损失按数量求平均后累加
        if self.cross_feature_loss is not None and isinstance(cross_x, (list, tuple)) and cross_x:  # 中文注释：仅在配置交叉特征蒸馏且存在聚合特征时计算额外损失
            cross_pairs = [(stu_feat, cross_feat) for stu_feat, cross_feat in zip(student_x, cross_x) if torch.is_tensor(cross_feat)]  # 中文注释：筛选学生与交叉教师同时存在的尺度对
            if cross_pairs:  # 中文注释：存在有效尺度对时执行交叉蒸馏
                normalizer = 1.0 / len(cross_pairs)  # 中文注释：计算均值系数以平衡不同尺度数量
                cross_loss_total = None  # 中文注释：初始化交叉特征蒸馏损失累加器
                for student_feature, cross_feature in cross_pairs:  # 中文注释：遍历每个有效尺度对
                    layer_loss = self.cross_feature_loss(student_feature, cross_feature)  # 中文注释：调用交叉教师特征蒸馏损失函数
                    scaled_layer_loss = layer_loss * normalizer  # 中文注释：按照均值系数缩放单尺度损失
                    cross_loss_total = scaled_layer_loss if cross_loss_total is None else cross_loss_total + scaled_layer_loss  # 中文注释：累积各尺度的蒸馏损失
                feature_loss['loss_cross_feature'] = cross_loss_total  # 中文注释：记录交叉教师特征蒸馏损失项
        losses.update(feature_loss)  # 中文注释：将特征蒸馏损失写入总损失
        if cross_consistency_losses:  # 中文注释：若交叉教师提供了一致性指标则合并相应损失
            losses.update(cross_consistency_losses)  # 中文注释：将一致性损失累加进总体损失字典
        ##############################################################################################################

        # trainable teacher alignment loss
        ##############################################################################################################
        trainable_teacher_losses = dict()  # 中文注释：初始化可训练教师损失字典
        trainable_payload = parsed_diff_feature.get('trainable_teachers') if isinstance(parsed_diff_feature, dict) else None  # 中文注释：获取可训练教师的前向输出包
        trainable_keys = getattr(self.model, 'trainable_diff_teacher_keys', [])  # 中文注释：读取底层模型登记的可训练教师列表
        if trainable_keys and isinstance(trainable_payload, dict) and student_feature_levels:  # 中文注释：仅当存在可训练教师与有效特征时才计算对齐损失
            reference_tensor = student_feature_levels[0]  # 中文注释：选取首个学生特征作为创建标量的参考张量
            mse_accumulator = reference_tensor.new_zeros(())  # 中文注释：创建标量用于累积均方误差
            pair_counter = 0  # 中文注释：记录参与损失计算的样本尺度对数量
            for sensor_key in trainable_keys:  # 中文注释：遍历全部可训练教师
                sample_indices = sensor_to_indices.get(sensor_key, [])  # 中文注释：获取当前教师对应的样本索引列表
                if not sample_indices:  # 中文注释：若该教师本批次无样本则跳过
                    continue  # 中文注释：继续处理下一个教师
                teacher_entry = trainable_payload.get(sensor_key, {})  # 中文注释：提取教师输出条目
                sample_feature_list = None  # 中文注释：初始化逐样本特征列表
                if isinstance(teacher_entry, dict):  # 中文注释：当输出为字典时读取features字段
                    raw_features = teacher_entry.get('features')  # 中文注释：获取逐样本特征列表
                    if isinstance(raw_features, list) and raw_features:  # 中文注释：确认特征列表有效
                        sample_feature_list = raw_features  # 中文注释：使用教师提供的逐样本特征
                elif isinstance(teacher_entry, list):  # 中文注释：当条目直接是列表时视为逐样本特征
                    sample_feature_list = teacher_entry  # 中文注释：直接使用列表作为特征集合
                if sample_feature_list is None and sensor_key in grouped_teacher_features:  # 中文注释：若缺失逐样本特征则退回按传感器聚合的结构
                    grouped_feature = grouped_teacher_features[sensor_key]  # 中文注释：获取聚合特征
                    if isinstance(grouped_feature, (list, tuple)) and grouped_feature:  # 中文注释：仅在结构有效时继续处理
                        reconstructed_samples: List[List[Tensor]] = []  # 中文注释：初始化重建的逐样本特征容器
                        for local_idx in range(len(sample_indices)):  # 中文注释：遍历当前教师负责的样本数量
                            per_sample_levels: List[Tensor] = []  # 中文注释：初始化当前样本的多尺度特征列表
                            for level_feat in grouped_feature:  # 中文注释：遍历各尺度特征张量
                                if torch.is_tensor(level_feat):  # 中文注释：仅处理张量类型的尺度特征
                                    per_sample_levels.append(level_feat[local_idx])  # 中文注释：提取当前样本在该尺度的特征
                            if per_sample_levels:  # 中文注释：仅在成功提取时加入结果
                                reconstructed_samples.append(per_sample_levels)  # 中文注释：记录当前样本的多尺度特征
                        if reconstructed_samples:  # 中文注释：若成功重建则使用该列表
                            sample_feature_list = reconstructed_samples  # 中文注释：将重建结果作为逐样本特征
                if not sample_feature_list:  # 中文注释：若仍无法获得特征则跳过
                    continue  # 中文注释：继续处理下一位教师
                for local_offset, global_idx in enumerate(sample_indices):  # 中文注释：遍历该教师负责的每个样本
                    if local_offset >= len(sample_feature_list):  # 中文注释：防止索引越界
                        continue  # 中文注释：跳过不存在的条目
                    teacher_levels = sample_feature_list[local_offset]  # 中文注释：取得当前样本的多尺度特征列表
                    if not isinstance(teacher_levels, (list, tuple)):  # 中文注释：若结构异常则跳过
                        continue  # 中文注释：继续下一个样本
                    for level_idx, teacher_feature in enumerate(teacher_levels):  # 中文注释：遍历每个尺度的教师特征
                        if level_idx >= len(student_feature_levels):  # 中文注释：当学生特征尺度不足时终止当前样本的遍历
                            break  # 中文注释：防止访问越界
                        if not torch.is_tensor(teacher_feature):  # 中文注释：若当前尺度不是张量则跳过
                            continue  # 中文注释：跳过该尺度
                        student_slice = student_feature_levels[level_idx][global_idx:global_idx + 1]  # 中文注释：提取学生对应尺度的单样本特征
                        aligned_teacher_feature = teacher_feature.to(device=student_slice.device, dtype=student_slice.dtype)  # 中文注释：将教师特征转换到学生特征相同的设备与精度
                        mse_accumulator = mse_accumulator + F.mse_loss(student_slice.squeeze(0), aligned_teacher_feature, reduction='mean')  # 中文注释：累加当前尺度的均方误差
                        pair_counter += 1  # 中文注释：更新有效样本尺度对数量
            if pair_counter > 0:  # 中文注释：仅在存在有效比较对时才生成损失项
                trainable_teacher_losses['loss_trainable_teacher_mse'] = (mse_accumulator / pair_counter) * self.trainable_teacher_loss_weight  # 中文注释：平均化均方误差并乘以权重写入损失字典
        if trainable_teacher_losses:  # 中文注释：当存在可训练教师损失时合并到总损失中
            losses.update(trainable_teacher_losses)  # 中文注释：确保损失字典包含可训练教师对齐项
        ##############################################################################################################

        # student training
        ##############################################################################################################
        # RPN forward
        if self.with_rpn:
            proposal_cfg = self.model.student.train_cfg.get(
                'rpn_proposal', self.model.student.test_cfg.rpn)
            rpn_data_samples = copy.deepcopy(batch_data_samples)
            # set cat_id of gt_labels to 0 in RPN
            for data_sample in rpn_data_samples:
                data_sample.gt_instances.labels = torch.zeros_like(
                    data_sample.gt_instances.labels)
            rpn_losses, rpn_results_list = self.model.student.rpn_head.loss_and_predict(student_x, rpn_data_samples,
                                                                            proposal_cfg=proposal_cfg)
                        # avoid get same name with roi_head loss
            keys = rpn_losses.keys()
            for key in list(keys):
                if 'loss' in key and 'rpn' not in key:
                    rpn_losses[f'rpn_{key}'] = rpn_losses.pop(key)
            losses.update(rpn_losses)
        else:
            assert batch_data_samples[0].get('proposals', None) is not None
            # use pre-defined proposals in InstanceData for the second stage
            # to extract ROI features.
            rpn_results_list = [
                data_sample.proposals for data_sample in batch_data_samples
            ]
        
        roi_losses = self.model.student.roi_head.loss(student_x, rpn_results_list,
                                        batch_data_samples)
        losses.update(roi_losses)
        ##############################################################################################################

        # object kd loss
        ##############################################################################################################
        # Apply cross-kd in ROI head
        if isinstance(diff_x, (list, tuple)) and diff_x and all(item is not None for item in diff_x):  # 中文注释：确保全部尺度具备教师特征后再执行ROI蒸馏
            roi_losses_kd = self.roi_head_loss_with_kd(
                student_x, grouped_teacher_features, rpn_results_list, batch_data_samples, sensor_tags_per_sample)  # 中文注释：基于分组教师特征与逐样本传感器标签计算ROI知识蒸馏损失
            losses.update(roi_losses_kd)  # 中文注释：累加ROI蒸馏损失
        if self.cross_roi_kd_weight > 0 and isinstance(grouped_cross_features, dict) and grouped_cross_features:  # 中文注释：当配置交叉ROI蒸馏且存在交叉特征时执行额外蒸馏
            cross_roi_losses = self.roi_head_loss_with_kd(  # 中文注释：复用ROI蒸馏流程评估交叉教师特征
                student_x, grouped_cross_features, rpn_results_list, batch_data_samples, sensor_tags_per_sample)  # 中文注释：使用交叉教师特征替代主教师执行ROI蒸馏
            scaled_cross_roi_losses = {f'cross_{key}': value * self.cross_roi_kd_weight for key, value in cross_roi_losses.items()}  # 中文注释：为交叉ROI损失添加前缀并按配置权重缩放
            losses.update(scaled_cross_roi_losses)  # 中文注释：将交叉ROI蒸馏损失并入总损失
        ##############################################################################################################

        return losses

    def get_trainable_diff_teacher_parameters(self) -> List[nn.Parameter]:
        """中文注释：暴露底层可训练扩散教师的参数列表以便优化器构建额外参数组。"""
        if hasattr(self.model, 'get_trainable_diff_teacher_parameters'):  # 中文注释：确认底层模型实现了参数收集接口
            return self.model.get_trainable_diff_teacher_parameters()  # 中文注释：直接复用底层实现返回的参数列表
        return []  # 中文注释：默认情况下返回空列表表示不存在额外教师参数

    def roi_head_loss_with_kd(self,
                              student_x: Tuple[Tensor],
                              grouped_diff_x: Dict[str, Any],
                              rpn_results_list,
                            batch_data_samples,
                            sensor_tags: List[str]):  # 中文注释：基于传感器分组信息计算ROI级别的知识蒸馏损失
        assert len(rpn_results_list) == len(batch_data_samples)  # 中文注释：确保RPN输出与样本数量一致
        outputs = unpack_gt_instances(batch_data_samples)
        batch_gt_instances, batch_gt_instances_ignore, _ = outputs
        roi_head = self.model.student.roi_head

        # assign gts and sample proposals
        num_imgs = len(batch_data_samples)
        sampling_results = []
        for i in range(num_imgs):
            # rename rpn_results.bboxes to rpn_results.priors
            rpn_results = rpn_results_list[i]
            # rpn_results.priors = rpn_results.pop('bboxes')

            assign_result = roi_head.bbox_assigner.assign(
                rpn_results, batch_gt_instances[i],
                batch_gt_instances_ignore[i])
            sampling_result = roi_head.bbox_sampler.sample(
                assign_result,
                rpn_results,
                batch_gt_instances[i],
                feats=[lvl_feat[i][None] for lvl_feat in student_x])
            sampling_results.append(sampling_result)

        losses = dict()
        # bbox head loss
        if roi_head.with_bbox:
            bbox_results = self.bbox_loss_with_kd(
                student_x, grouped_diff_x, sampling_results, sensor_tags)
            losses.update(bbox_results['loss_bbox_kd'])

        return losses

    def bbox_loss_with_kd(self, student_x, grouped_diff_x, sampling_results, sensor_tags):  # 中文注释：按传感器划分执行ROI知识蒸馏前向与损失计算
        if len(sensor_tags) != len(sampling_results):  # 中文注释：确保传感器标签数量与采样结果数量一致
            raise ValueError('传感器标签数量与采样结果数量不一致，无法执行知识蒸馏。')  # 中文注释：抛出错误提示以便排查
        rois = bbox2roi([res.priors for res in sampling_results])  # 中文注释：基于全部样本的提议框拼接成批次ROI
        student_head = self.model.student.roi_head  # 中文注释：获取学生模型的ROI头以执行前向
        reference_diff_head = self.model.diff_detector.roi_head  # 中文注释：获取主扩散教师的ROI头用于读取结构信息
        stu_bbox_results = student_head._bbox_forward(student_x, rois)  # 中文注释：使用学生特征计算ROI头输出
        sensor_to_indices = self._group_indices_by_sensor_tags(sensor_tags)  # 中文注释：根据传感器标签将样本索引划分分组
        teacher_storage: List[Dict[str, Tensor]] = [dict() for _ in sampling_results]  # 中文注释：初始化教师前向结果的逐样本存储
        reused_storage: List[Dict[str, Tensor]] = [dict() for _ in sampling_results]  # 中文注释：初始化教师头复用学生特征的存储
        teacher_fallback_cache: Dict[str, Tensor] = dict()  # 中文注释：缓存教师输出各键的零长度模板以便补齐
        reused_fallback_cache: Dict[str, Tensor] = dict()  # 中文注释：缓存复用输出各键的零长度模板
        for sensor_tag, indices in sensor_to_indices.items():  # 中文注释：遍历每个传感器分组
            if not indices:  # 中文注释：若当前传感器下无样本则跳过
                continue  # 中文注释：继续处理下一个传感器
            if sensor_tag not in self.model.diff_teacher_bank:  # 中文注释：确保存在对应的扩散教师
                raise KeyError(f'未找到传感器"{sensor_tag}"对应的扩散教师，无法执行ROI蒸馏。')  # 中文注释：抛出详细错误提示
            group_feature_payload = grouped_diff_x.get(sensor_tag) if isinstance(grouped_diff_x, dict) else None  # 中文注释：读取当前传感器的教师特征
            if group_feature_payload is None:  # 中文注释：若缺少对应特征则无法继续
                raise KeyError(f'传感器"{sensor_tag}"缺少主教师特征，无法执行ROI蒸馏。')  # 中文注释：抛出错误提示
            normalized_group_feature, _ = self._normalize_group_feature_payload(group_feature_payload, len(indices))  # 中文注释：统一当前传感器特征结构
            if normalized_group_feature is None:  # 中文注释：若规范化结果为空则跳过后续计算
                continue  # 中文注释：继续处理其他传感器
            if isinstance(normalized_group_feature, (list, tuple)):  # 中文注释：若特征以序列形式提供
                teacher_features_ready = [feat for feat in normalized_group_feature]  # 中文注释：复制多尺度张量列表
            elif torch.is_tensor(normalized_group_feature):  # 中文注释：若规范化结果为单张量则包装成列表
                teacher_features_ready = [normalized_group_feature]  # 中文注释：构造单尺度张量列表
            else:  # 中文注释：遇到无法识别的结构时直接报错
                raise TypeError(f'不支持的教师特征类型：{type(normalized_group_feature)}')  # 中文注释：提醒开发者检查特征格式
            teacher_features_ready = [feat for feat in teacher_features_ready]  # 中文注释：复制列表以避免共享引用
            teacher_features_ready = tuple(teacher_features_ready)  # 中文注释：转换为元组以适配ROI头前向接口
            feature_device = student_x[0].device if isinstance(student_x, (list, tuple)) else student_x.device  # 中文注释：获取学生特征所在的设备
            index_tensor = torch.tensor(indices, device=feature_device, dtype=torch.long)  # 中文注释：将样本索引转换成张量以便切片
            student_group_features = [level_feat.index_select(0, index_tensor) for level_feat in student_x]  # 中文注释：提取当前传感器对应的学生特征
            sampling_subset = [sampling_results[idx] for idx in indices]  # 中文注释：收集当前传感器的采样结果
            sensor_rois = bbox2roi([res.priors for res in sampling_subset])  # 中文注释：基于子集采样结果生成局部ROI
            roi_counts = [res.priors.shape[0] for res in sampling_subset]  # 中文注释：统计每个样本的ROI数量用于结果拆分
            teacher_model = self.model.diff_teacher_bank[sensor_tag]  # 中文注释：获取当前传感器的教师模型
            teacher_roi_head = teacher_model.roi_head  # 中文注释：读取教师ROI头执行前向
            teacher_forward = teacher_roi_head._bbox_forward(teacher_features_ready, sensor_rois)  # 中文注释：使用教师特征计算ROI输出
            reused_forward = teacher_roi_head._bbox_forward(student_group_features, sensor_rois)  # 中文注释：使用教师头对学生特征前向
            teacher_cache_update = self._distribute_group_roi_outputs(teacher_forward, teacher_storage, indices, roi_counts)  # 中文注释：将教师输出拆分回逐样本存储
            reused_cache_update = self._distribute_group_roi_outputs(reused_forward, reused_storage, indices, roi_counts)  # 中文注释：将复用输出拆分回逐样本存储
            for key, value in teacher_cache_update.items():  # 中文注释：更新教师空模板缓存
                teacher_fallback_cache.setdefault(key, value)  # 中文注释：仅在未存在同名模板时写入
            for key, value in reused_cache_update.items():  # 中文注释：更新复用输出空模板缓存
                reused_fallback_cache.setdefault(key, value)  # 中文注释：仅在未存在同名模板时写入
        sample_count = len(sampling_results)  # 中文注释：记录样本数量用于构建拼接列表
        teacher_keys = set(teacher_fallback_cache.keys())  # 中文注释：初始化教师输出键集合
        reused_keys = set(reused_fallback_cache.keys())  # 中文注释：初始化复用输出键集合
        for sample_outputs in teacher_storage:  # 中文注释：遍历逐样本教师输出
            teacher_keys.update(sample_outputs.keys())  # 中文注释：补充键集合
        for sample_outputs in reused_storage:  # 中文注释：遍历逐样本复用输出
            reused_keys.update(sample_outputs.keys())  # 中文注释：补充键集合
        aggregated_teacher: Dict[str, List[Tensor]] = {key: [None] * sample_count for key in teacher_keys}  # 中文注释：创建教师输出的逐样本列表结构
        aggregated_reused: Dict[str, List[Tensor]] = {key: [None] * sample_count for key in reused_keys}  # 中文注释：创建复用输出的逐样本列表结构
        for sample_idx, sample_outputs in enumerate(teacher_storage):  # 中文注释：遍历教师逐样本输出
            for key, value in sample_outputs.items():  # 中文注释：遍历当前样本的所有输出项
                aggregated_teacher[key][sample_idx] = value  # 中文注释：将张量放回对应位置
        for sample_idx, sample_outputs in enumerate(reused_storage):  # 中文注释：遍历复用逐样本输出
            for key, value in sample_outputs.items():  # 中文注释：遍历当前样本的所有输出项
                aggregated_reused[key][sample_idx] = value  # 中文注释：将张量放回对应位置
        for key in teacher_keys:  # 中文注释：为缺少模板的教师键补齐空张量
            if key not in teacher_fallback_cache:  # 中文注释：当缓存中不存在该键时
                for sample_outputs in teacher_storage:  # 中文注释：遍历样本寻找可用张量
                    if key in sample_outputs:  # 中文注释：找到首个包含该键的样本
                        template_tensor = sample_outputs[key]  # 中文注释：获取对应张量
                        teacher_fallback_cache[key] = template_tensor.new_empty((0,) + template_tensor.shape[1:])  # 中文注释：基于该张量创建零长度模板
                        break  # 中文注释：完成模板创建后跳出循环
        for key in reused_keys:  # 中文注释：为缺少模板的复用键补齐空张量
            if key not in reused_fallback_cache:  # 中文注释：当缓存中不存在该键时
                for sample_outputs in reused_storage:  # 中文注释：遍历样本寻找可用张量
                    if key in sample_outputs:  # 中文注释：找到首个包含该键的样本
                        template_tensor = sample_outputs[key]  # 中文注释：获取对应张量
                        reused_fallback_cache[key] = template_tensor.new_empty((0,) + template_tensor.shape[1:])  # 中文注释：创建零长度模板
                        break  # 中文注释：模板创建完成后跳出循环
        diff_bbox_results = self._finalize_grouped_outputs(aggregated_teacher, teacher_fallback_cache)  # 中文注释：按批次顺序拼接教师输出
        reused_bbox_results = self._finalize_grouped_outputs(aggregated_reused, reused_fallback_cache)  # 中文注释：按批次顺序拼接复用输出
        if 'cls_score' not in reused_bbox_results or 'cls_score' not in diff_bbox_results:  # 中文注释：确保分类输出存在
            raise KeyError('教师或复用分支缺少cls_score，无法计算分类蒸馏损失。')  # 中文注释：抛出错误提示
        if 'bbox_pred' not in reused_bbox_results or 'bbox_pred' not in diff_bbox_results:  # 中文注释：确保回归输出存在
            raise KeyError('教师或复用分支缺少bbox_pred，无法计算回归蒸馏损失。')  # 中文注释：抛出错误提示
        losses_kd = dict()  # 中文注释：初始化蒸馏损失容器
        reused_cls_scores = reused_bbox_results['cls_score']  # 中文注释：提取复用分支的分类预测
        diff_cls_scores = diff_bbox_results['cls_score']  # 中文注释：提取教师分支的分类预测
        avg_factor = sum([res.avg_factor for res in sampling_results])  # 中文注释：汇总正负样本权重用于归一化
        loss_cls_kd = self.loss_cls_kd(  # 中文注释：计算分类蒸馏损失
            reused_cls_scores,
            diff_cls_scores,
            avg_factor=avg_factor)
        losses_kd['loss_cls_kd'] = loss_cls_kd  # 中文注释：记录分类蒸馏损失
        assert student_head.bbox_head.reg_class_agnostic == reference_diff_head.bbox_head.reg_class_agnostic  # 中文注释：验证学生与教师在回归模式上保持一致
        num_classes = student_head.bbox_head.num_classes  # 中文注释：读取类别数量用于回归通道选择
        reused_bbox_preds = reused_bbox_results['bbox_pred']  # 中文注释：提取复用分支的回归预测
        diff_bbox_preds = diff_bbox_results['bbox_pred']  # 中文注释：提取教师分支的回归预测
        diff_prob = diff_cls_scores.softmax(dim=1)[:, :num_classes]  # 中文注释：对教师分类结果做Softmax并截取目标类别范围
        reg_weights, reg_distill_idx = diff_prob.max(dim=1)  # 中文注释：获取最大概率及其对应类别索引用于加权回归
        if not student_head.bbox_head.reg_class_agnostic:  # 中文注释：当回归分支按类别划分时需根据最优类别挑选通道
            reg_distill_idx = reg_distill_idx[:, None, None].repeat(1, 1, 4)  # 中文注释：扩展索引形状匹配回归张量
            reused_bbox_preds = reused_bbox_preds.reshape(-1, num_classes, 4)  # 中文注释：将复用回归结果重塑为类别维度
            reused_bbox_preds = reused_bbox_preds.gather(dim=1, index=reg_distill_idx).squeeze(1)  # 中文注释：按照选定类别抽取对应回归预测
            diff_bbox_preds = diff_bbox_preds.reshape(-1, num_classes, 4)  # 中文注释：重塑教师回归结果
            diff_bbox_preds = diff_bbox_preds.gather(dim=1, index=reg_distill_idx).squeeze(1)  # 中文注释：抽取教师在目标类别下的回归预测
        loss_reg_kd = self.loss_reg_kd(  # 中文注释：计算回归蒸馏损失
            reused_bbox_preds,
            diff_bbox_preds,
            weight=reg_weights[:, None],
            avg_factor=reg_weights.sum() * 4)
        losses_kd['loss_reg_kd'] = loss_reg_kd  # 中文注释：记录回归蒸馏损失
        bbox_results = dict()  # 中文注释：初始化返回字典
        for key, value in stu_bbox_results.items():  # 中文注释：遍历学生分支输出
            bbox_results['stu_' + key] = value  # 中文注释：为学生输出添加前缀后存储
        for key, value in diff_bbox_results.items():  # 中文注释：遍历教师分支拼接结果
            bbox_results['diff_' + key] = value  # 中文注释：为教师输出添加前缀后存储
        for key, value in reused_bbox_results.items():  # 中文注释：遍历复用分支拼接结果
            bbox_results['reused_' + key] = value  # 中文注释：为复用输出添加前缀后存储
        bbox_results['loss_bbox_kd'] = losses_kd  # 中文注释：写入蒸馏损失项
        return bbox_results  # 中文注释：返回包含三分支输出与蒸馏损失的字典
