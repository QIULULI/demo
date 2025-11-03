# Copyright (c) OpenMMLab. All rights reserved.
import copy
import inspect  # 新增中文注释：引入inspect模块以便动态检查函数签名
from typing import Dict, Tuple
import torch
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
        self.cross_loss_weight = self.train_cfg.cross_loss_cfg.get('cross_loss_weight')  # 读取交叉蒸馏初始权重
        raw_schedule = self.train_cfg.cross_loss_cfg.get('schedule', [])  # 获取阶段性调度配置
        self.cross_loss_schedule = sorted(raw_schedule, key=lambda item: item.get('start_iter', 0))  # 按起始迭代排序调度表
        self.cross_schedule_stage = -1  # 记录当前处于的调度阶段索引
        # feature loss setting
        self.feature_loss_type = self.train_cfg.feature_loss_cfg.get(
            'feature_loss_type')
        self.feature_loss_weight = self.train_cfg.feature_loss_cfg.get(
            'feature_loss_weight')
        self.feature_loss = KDLoss(
            loss_weight=self.feature_loss_weight, loss_type=self.feature_loss_type)

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
                level_count = len(sample_features[0])  # 中文注释：读取单个样本所含的尺度数量
                grouped_levels = []  # 中文注释：初始化当前传感器的尺度特征列表
                for level_idx in range(level_count):  # 中文注释：遍历每一个尺度
                    level_stack = torch.stack([sample_feature[level_idx] for sample_feature in sample_features], dim=0)  # 中文注释：在batch维度堆叠当前尺度的所有样本特征
                    grouped_levels.append(level_stack)  # 中文注释：将堆叠结果加入传感器特征列表
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
                    if current_sensor_features is None:  # 中文注释：若缺少特征则跳过
                        continue  # 中文注释：继续处理下一个样本
                    level_tensor = current_sensor_features[level_idx]  # 中文注释：提取当前尺度的特征张量
                    level_slices.append(level_tensor[local_idx:local_idx + 1])  # 中文注释：截取与当前样本对应的切片并保留批维
                merged_levels.append(torch.cat(level_slices, dim=0) if level_slices else None)  # 中文注释：将切片按原顺序拼接成完整批次
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

    def cross_loss_diff_to_student(self, grouped_samples: Dict[str, SampleList], grouped_features: Dict[str, list]):  # 中文注释：按传感器分组计算交叉蒸馏损失
        losses = dict()  # 中文注释：初始化损失字典用于累计各传感器的结果
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
            for key, value in sensor_losses.items():  # 中文注释：遍历当前传感器的损失项
                losses[key] = losses[key] + value if key in losses else value  # 中文注释：将损失累加到总字典中
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
        sensor_to_indices = self._group_indices_by_sensor(batch_data_samples)  # 中文注释：根据传感器信息划分批次索引
        grouped_samples = {sensor_tag: [batch_data_samples[idx] for idx in indices] for sensor_tag, indices in sensor_to_indices.items()}  # 中文注释：按照传感器重组样本列表用于局部蒸馏
        grouped_teacher_features = self._slice_teacher_features_by_sensor(parsed_diff_feature.get('main_teacher'), sensor_to_indices)  # 中文注释：提取并切分主教师特征
        student_x = self.model.student.extract_feat(batch_inputs)  # 中文注释：前向学生模型获取自身特征表示
        diff_x = self._merge_grouped_features(grouped_teacher_features, sensor_to_indices, len(batch_data_samples))  # 中文注释：将教师特征按照原批次顺序重新拼接
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
            valid_pairs = [(stu_feat, tea_feat) for stu_feat, tea_feat in zip(student_x, diff_x) if tea_feat is not None]  # 中文注释：过滤掉缺失教师特征的尺度对
            valid_count = len(valid_pairs)  # 中文注释：统计可参与蒸馏的尺度数量
            if valid_count > 0:  # 中文注释：仅当存在有效尺度时才执行蒸馏
                for student_feature, diff_feature in valid_pairs:  # 中文注释：逐个尺度计算特征损失
                    layer_loss = self.feature_loss(  # 中文注释：调用蒸馏损失函数
                        student_feature, diff_feature)  # 中文注释：传入学生与教师的对应特征
                    feature_loss['pkd_feature_loss'] += layer_loss/valid_count  # 中文注释：将各尺度损失按数量求平均后累加
        losses.update(feature_loss)  # 中文注释：将特征蒸馏损失写入总损失
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
                student_x, diff_x, rpn_results_list, batch_data_samples)  # 中文注释：基于教师特征计算ROI知识蒸馏损失
            losses.update(roi_losses_kd)  # 中文注释：累加ROI蒸馏损失
        ##############################################################################################################

        return losses

    def roi_head_loss_with_kd(self,
                            student_x: Tuple[Tensor],
                            diff_x: Tuple[Tensor],
                            rpn_results_list,
                            batch_data_samples):
        assert len(rpn_results_list) == len(batch_data_samples)
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
                student_x, diff_x, sampling_results)
            losses.update(bbox_results['loss_bbox_kd'])

        return losses

    def bbox_loss_with_kd(self, student_x, diff_x, sampling_results):
        rois = bbox2roi([res.priors for res in sampling_results])

        student_head, diff_head = self.model.student.roi_head, self.model.diff_detector.roi_head
        stu_bbox_results = student_head._bbox_forward(student_x, rois)
        diff_bbox_results = diff_head._bbox_forward(diff_x, rois)
        reused_bbox_results = diff_head._bbox_forward(student_x, rois)

        losses_kd = dict()
        # classification KD
        reused_cls_scores = reused_bbox_results['cls_score']
        diff_cls_scores = diff_bbox_results['cls_score']
        avg_factor = sum([res.avg_factor for res in sampling_results])
        loss_cls_kd = self.loss_cls_kd(
            reused_cls_scores,
            diff_cls_scores,
            avg_factor=avg_factor)
        losses_kd['loss_cls_kd'] = loss_cls_kd
        # l1 loss
        assert student_head.bbox_head.reg_class_agnostic \
            == diff_head.bbox_head.reg_class_agnostic
        num_classes = student_head.bbox_head.num_classes
        reused_bbox_preds = reused_bbox_results['bbox_pred']
        diff_bbox_preds = diff_bbox_results['bbox_pred']
        diff_cls_scores = diff_cls_scores.softmax(dim=1)[:, :num_classes]
        reg_weights, reg_distill_idx = diff_cls_scores.max(dim=1)
        if not student_head.bbox_head.reg_class_agnostic:
            reg_distill_idx = reg_distill_idx[:, None, None].repeat(1, 1, 4)
            reused_bbox_preds = reused_bbox_preds.reshape(-1, num_classes, 4)
            reused_bbox_preds = reused_bbox_preds.gather(
                dim=1, index=reg_distill_idx)
            reused_bbox_preds = reused_bbox_preds.squeeze(1)
            diff_bbox_preds = diff_bbox_preds.reshape(-1, num_classes, 4)
            diff_bbox_preds = diff_bbox_preds.gather(
                dim=1, index=reg_distill_idx)
            diff_bbox_preds = diff_bbox_preds.squeeze(1)

        loss_reg_kd = self.loss_reg_kd(
            reused_bbox_preds,
            diff_bbox_preds,
            weight=reg_weights[:, None],
            avg_factor=reg_weights.sum() * 4)
        losses_kd['loss_reg_kd'] = loss_reg_kd

        bbox_results = dict()
        for key, value in stu_bbox_results.items():
            bbox_results['stu_' + key] = value
        for key, value in diff_bbox_results.items():
            bbox_results['diff_' + key] = value
        for key, value in reused_bbox_results.items():
            bbox_results['reused_' + key] = value
        bbox_results['loss_bbox_kd'] = losses_kd
        return bbox_results