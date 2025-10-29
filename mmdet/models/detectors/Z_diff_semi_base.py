# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import Dict
from typing import Any, List, Optional, Union, Tuple
import torch
import torch.nn as nn
from torch import Tensor

from mmdet.models.utils import (filter_gt_instances, rename_loss_dict, reweight_loss_dict)
from mmdet.registry import MODELS
from mmdet.structures import SampleList
from mmdet.structures.bbox import bbox_project
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from .base import BaseDetector

from pathlib import Path
from mmengine.config import Config
from mmengine.runner import load_checkpoint


@MODELS.register_module()
class SemiBaseDiffDetector(BaseDetector):
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
                 diff_model: ConfigType,
                 semi_train_cfg: OptConfigType = None,
                 semi_test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)

        self.student = MODELS.build(detector.deepcopy())
        self.teacher = MODELS.build(detector.deepcopy())
        self.diff_detector = None  # 初始化扩散教师占位符，默认为空
        self.diff_teacher_bank = {}  # 初始化扩散教师字典，后续按传感器标签填充
        parsed_diff_configs = self._normalize_diff_teacher_configs(diff_model)  # 解析输入配置，统一为{sensor:配置}格式
        for sensor_tag, teacher_cfg in parsed_diff_configs.items():  # 遍历每个传感器的教师配置，逐一构建模型
            config_path = teacher_cfg.get('config')  # 读取配置文件路径，用于后续构建模型
            if not config_path:  # 若未提供配置文件，则跳过该教师，保持兼容旧逻辑
                continue  # 跳过无效配置，避免错误构建
            teacher_config = Config.fromfile(config_path)  # 根据配置文件加载完整的模型配置
            teacher_model = MODELS.build(teacher_config['model'])  # 使用配置构建对应的扩散教师模型
            pretrained_path = teacher_cfg.get('pretrained_model')  # 读取预训练权重路径，便于恢复已训练的教师
            if pretrained_path:  # 若提供了预训练权重
                load_checkpoint(teacher_model, pretrained_path, map_location='cpu', strict=True)  # 严格加载权重，确保结构匹配
            teacher_model.cuda()  # 将教师模型搬移到GPU，以便推理时使用
            self.freeze(teacher_model)  # 冻结教师参数，防止其在训练过程中被更新
            self.diff_teacher_bank[sensor_tag] = teacher_model  # 将构建完成的教师注册到字典，键为传感器标签
        if self.diff_teacher_bank:  # 若成功构建了至少一个扩散教师
            self.diff_detector = next(iter(self.diff_teacher_bank.values()))  # 默认使用首个教师作为diff_detector，兼容旧接口
        if self.diff_detector is None:  # 若未构建任何教师，则回退使用学生模型
            self.diff_detector = self.student  # 保持旧逻辑，在无教师时直接使用学生模型

        self.semi_train_cfg = semi_train_cfg
        self.semi_test_cfg = semi_test_cfg
        if self.semi_train_cfg.get('freeze_teacher', True) is True:
            self.freeze(self.teacher)

    @staticmethod
    def freeze(model: nn.Module):
        """Freeze the model."""
        model.eval()
        for param in model.parameters():
            param.requires_grad = False

    @staticmethod
    def _fetch_config_value(config_entry: Any, key: str, default=None):  # 工具函数：统一读取配置对象中的指定字段
        if isinstance(config_entry, dict) and key in config_entry:  # 若配置是字典且存在对应键
            return config_entry.get(key, default)  # 直接从字典获取对应值
        if hasattr(config_entry, key):  # 若配置对象以属性形式存储
            return getattr(config_entry, key)  # 通过属性访问取出目标值
        if hasattr(config_entry, 'get'):  # 若配置对象实现了get方法（如ConfigDict）
            try:  # 使用try避免异常导致流程中断
                return config_entry.get(key, default)  # 调用get方法获取值
            except Exception:  # 捕获潜在异常，保证稳健性
                return default  # 出现异常时返回默认值
        return default  # 若无法读取则返回默认值

    @staticmethod
    def _is_single_diff_config(config_entry: Any) -> bool:  # 判断给定对象是否描述单个教师配置
        return any(  # 只要存在关键字段即可视为单个教师配置
            SemiBaseDiffDetector._fetch_config_value(config_entry, key) is not None  # 检查关键字段是否存在
            for key in ('config', 'pretrained_model', 'pretrained', 'sensor')  # 关键字段集合
        )

    def _normalize_diff_teacher_configs(self, diff_model: ConfigType) -> Dict[str, dict]:  # 解析diff教师配置并输出标准字典
        normalized_configs = {}  # 存放解析后的结果，键为传感器标签
        if diff_model is None:  # 若未提供扩散教师配置
            return normalized_configs  # 直接返回空字典
        parsing_queue: List[Tuple[Any, Optional[str]]] = []  # 初始化解析队列，元素包含配置对象及备用传感器标签
        if isinstance(diff_model, (list, tuple)):  # 若输入为列表/元组
            for item in diff_model:  # 遍历每个元素
                parsing_queue.append((item, None))  # 加入队列，暂不指定传感器标签
        else:  # 输入非序列
            parsing_queue.append((diff_model, None))  # 统一加入队列处理
        while parsing_queue:  # 逐个弹出进行解析
            current_cfg, fallback_sensor = parsing_queue.pop(0)  # 取出当前配置及备用传感器标签
            if isinstance(current_cfg, dict) and not self._is_single_diff_config(current_cfg):  # 若是嵌套字典且非单教师描述
                for sensor_key, nested_cfg in current_cfg.items():  # 遍历子配置
                    parsing_queue.append((nested_cfg, sensor_key))  # 将子配置加入队列，并携带上层传感器标签
                continue  # 继续处理队列中的其他元素
            if isinstance(current_cfg, (list, tuple)):  # 若当前项仍是列表/元组
                for nested_cfg in current_cfg:  # 遍历内部元素
                    parsing_queue.append((nested_cfg, fallback_sensor))  # 保留备用传感器标签递归处理
                continue  # 跳过后续步骤，等待下次循环
            sensor_key = self._fetch_config_value(current_cfg, 'sensor', fallback_sensor)  # 优先从配置中读取传感器标签
            if sensor_key is None:  # 若仍未获得标签
                sensor_key = 'default'  # 使用default作为占位标签，兼容旧逻辑
            config_path = self._fetch_config_value(current_cfg, 'config')  # 提取配置文件路径
            pretrained_path = self._fetch_config_value(current_cfg, 'pretrained_model')  # 提取预训练权重路径
            if pretrained_path is None:  # 若未使用新字段名
                pretrained_path = self._fetch_config_value(current_cfg, 'pretrained')  # 兼容旧字段名称
            normalized_configs[sensor_key] = {  # 汇总解析结果
                'config': config_path,  # 存储配置文件路径
                'pretrained_model': pretrained_path  # 存储预训练权重路径
            }
        return normalized_configs  # 返回标准化后的配置字典

    def cuda(self, device: Optional[str] = None) -> nn.Module:  # 重载cuda方法，确保所有扩散教师迁移到指定设备
        """Since teacher is registered as a plain object, it is necessary to
        put the teacher model to cuda when calling ``cuda`` function."""
        if self.diff_teacher_bank:  # 若存在多传感器扩散教师
            for teacher_model in self.diff_teacher_bank.values():  # 遍历所有教师模型
                teacher_model.cuda(device=device)  # 将每个教师迁移到指定设备
        else:  # 若仅存在单一路径教师
            self.diff_detector.cuda(device=device)  # 仅迁移默认的diff_detector，保持旧逻辑
        return super().cuda(device=device)

    def to(self, device: Optional[str] = None) -> nn.Module:  # 重载to方法，兼容多教师场景下的设备转换
        """Since teacher is registered as a plain object, it is necessary to
        put the teacher model to other device when calling ``to`` function."""
        if self.diff_teacher_bank:  # 若使用教师字典
            for teacher_model in self.diff_teacher_bank.values():  # 遍历教师集合
                teacher_model.to(device=device)  # 将教师迁移到目标设备
        else:  # 若仍沿用旧逻辑
            self.diff_detector.to(device=device)  # 转移默认的diff_detector
        return super().to(device=device)

    def train(self, mode: bool = True) -> None:  # 重载train方法，统一保持教师处于评估模式
        """Set the same train mode for teacher and student model."""
        if self.diff_teacher_bank:  # 若启用了多教师架构
            for teacher_model in self.diff_teacher_bank.values():  # 遍历所有教师
                teacher_model.train(False)  # 强制教师维持评估模式，避免梯度更新
        else:  # 若只有单教师
            self.diff_detector.train(False)  # 保持默认教师在评估模式
        super().train(mode)

    def __setattr__(self, name: str, value: Any) -> None:
        """Set attribute, i.e. self.name = value

        This reloading prevent the teacher model from being registered as a
        nn.Module. The teacher module is registered as a plain object, so that
        the teacher parameters will not show up when calling
        ``self.parameters``, ``self.modules``, ``self.children`` methods.
        """
        if name == 'diff_detector':
            object.__setattr__(self, name, value)
        else:
            super().__setattr__(name, value)

    def loss(self, multi_batch_inputs: Dict[str, Tensor],
             multi_batch_data_samples: Dict[str, SampleList]) -> dict:
        """Calculate losses from multi-branch inputs and data samples.

        Args:
            multi_batch_inputs (Dict[str, Tensor]): The dict of multi-branch
                input images, each value with shape (N, C, H, W).
                Each value should usually be mean centered and std scaled.
            multi_batch_data_samples (Dict[str, List[:obj:`DetDataSample`]]):
                The dict of multi-branch data samples.

        Returns:
            dict: A dictionary of loss components
        """
        losses = dict()
        losses.update(**self.loss_by_gt_instances(
            multi_batch_inputs['sup'], multi_batch_data_samples['sup']))

        origin_pseudo_data_samples, batch_info = self.get_pseudo_instances(
            multi_batch_inputs['unsup_teacher'], multi_batch_data_samples['unsup_teacher'])

        multi_batch_data_samples['unsup_student'] = self.project_pseudo_instances(
            origin_pseudo_data_samples, multi_batch_data_samples['unsup_student'])

        losses.update(**self.loss_by_pseudo_instances(multi_batch_inputs['unsup_student'],
                                                      multi_batch_data_samples['unsup_student'], batch_info))
        return losses
    
    def loss_diff_adaptation(self, multi_batch_inputs: Dict[str, Tensor],
                             multi_batch_data_samples: Dict[str, SampleList]) -> dict:
        """Calculate losses from multi-branch inputs and data samples.

        Args:
            multi_batch_inputs (Dict[str, Tensor]): The dict of multi-branch
                input images, each value with shape (N, C, H, W).
                Each value should usually be mean centered and std scaled.
            multi_batch_data_samples (Dict[str, List[:obj:`DetDataSample`]]):
                The dict of multi-branch data samples.

        Returns:
            dict: A dictionary of loss components
        """
        losses = dict()
        losses.update(**self.loss_by_gt_instances(
            multi_batch_inputs['sup'], multi_batch_data_samples['sup']))
        origin_pseudo_data_samples, batch_info, diff_feature = self.get_pseudo_instances_diff(
            multi_batch_inputs['unsup_teacher'], multi_batch_data_samples['unsup_teacher'])  # 中文注释：获取伪标签、批次信息以及原始的特征打包结果
        parsed_diff_feature = self._parse_diff_feature(diff_feature, batch_info)  # 中文注释：对返回的特征结构进行标准化解析并在必要时补充批次信息
        multi_batch_data_samples['unsup_student'] = self.project_pseudo_instances(
            origin_pseudo_data_samples, multi_batch_data_samples['unsup_student'])

        losses.update(**self.loss_by_pseudo_instances(multi_batch_inputs['unsup_student'],
                                                      multi_batch_data_samples['unsup_student'], batch_info))  # 中文注释：计算学生分支伪标签损失并合并
        return losses, parsed_diff_feature  # 中文注释：返回损失字典与解析后的特征包

    def loss_by_gt_instances(self, batch_inputs: Tensor,
                             batch_data_samples: SampleList) -> dict:
        """Calculate losses from a batch of inputs and ground-truth data
        samples.

        Args:
            batch_inputs (Tensor): Input images of shape (N, C, H, W).
                These should usually be mean centered and std scaled.
            batch_data_samples (List[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.

        Returns:
            dict: A dictionary of loss components
        """

        losses = self.student.loss(batch_inputs, batch_data_samples)
        sup_weight = self.semi_train_cfg.get('sup_weight', 1.)
        return rename_loss_dict('sup_', reweight_loss_dict(losses, sup_weight))

    def loss_by_pseudo_instances(self,
                                 batch_inputs: Tensor,
                                 batch_data_samples: SampleList,
                                 batch_info: Optional[dict] = None) -> dict:
        """Calculate losses from a batch of inputs and pseudo data samples.

        Args:
            batch_inputs (Tensor): Input images of shape (N, C, H, W).
                These should usually be mean centered and std scaled.
            batch_data_samples (List[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`,
                which are `pseudo_instance` or `pseudo_panoptic_seg`
                or `pseudo_sem_seg` in fact.
            batch_info (dict): Batch information of teacher model
                forward propagation process. Defaults to None.

        Returns:
            dict: A dictionary of loss components
        """
        batch_data_samples = filter_gt_instances(
            batch_data_samples, score_thr=self.semi_train_cfg.cls_pseudo_thr)
        losses = self.student.loss(batch_inputs, batch_data_samples)
        pseudo_instances_num = sum([
            len(data_samples.gt_instances)
            for data_samples in batch_data_samples
        ])
        unsup_weight = self.semi_train_cfg.get(
            'unsup_weight', 1.) if pseudo_instances_num > 0 else 0.
        return rename_loss_dict('unsup_',
                                reweight_loss_dict(losses, unsup_weight))

    @torch.no_grad()
    def get_pseudo_instances(
            self, batch_inputs: Tensor, batch_data_samples: SampleList
    ) -> Tuple[SampleList, Optional[dict]]:
        """Get pseudo instances from teacher model."""
        self.teacher.eval()
        results_list = self.teacher.predict(
            batch_inputs, batch_data_samples, rescale=False)
        batch_info = {}
        for data_samples, results in zip(batch_data_samples, results_list):
            data_samples.gt_instances = results.pred_instances
            data_samples.gt_instances.bboxes = bbox_project(
                data_samples.gt_instances.bboxes,
                torch.from_numpy(data_samples.homography_matrix).inverse().to(
                    self.data_preprocessor.device), data_samples.ori_shape)
        return batch_data_samples, batch_info
    
    @torch.no_grad()
    def get_pseudo_instances_diff(
            self, batch_inputs: Tensor, batch_data_samples: SampleList
    ) -> Tuple[SampleList, Optional[dict]]:
        """Get pseudo instances from teacher model."""
        if not self.diff_teacher_bank:  # 若未启用多教师，直接沿用默认diff_detector逻辑
            self.diff_detector.eval()  # 将默认教师设置为评估模式
            results_list, diff_feature = self.diff_detector.predict(  # 使用默认教师生成伪标签与特征
                batch_inputs, batch_data_samples, rescale=False, return_feature=True)  # 传入批量数据并要求返回特征
            batch_info = {}  # 初始化批处理信息占位符
            for data_samples, results in zip(batch_data_samples, results_list):  # 遍历样本与预测结果
                data_samples.gt_instances = results.pred_instances  # 写入伪实例以供后续训练
                data_samples.gt_instances.bboxes = bbox_project(  # 将预测框映射回原图坐标系
                    data_samples.gt_instances.bboxes,
                    torch.from_numpy(data_samples.homography_matrix).inverse().to(
                        self.data_preprocessor.device), data_samples.ori_shape)
            if isinstance(diff_feature, (list, tuple)):  # 若返回的是按样本排列的特征序列
                primary_features = list(diff_feature)  # 转换为列表以便后续统一封装
            else:  # 若返回单一张量
                primary_features = [diff_feature for _ in range(len(batch_data_samples))]  # 为每个样本复制一份引用
            distill_feature = {  # 构造与多教师一致的返回结构
                'primary': primary_features,  # 主教师特征列表
                'peer': [dict() for _ in range(len(batch_data_samples))]  # 同伴特征占位，保持兼容
            }
            return batch_data_samples, batch_info, distill_feature  # 返回伪标签、批信息与特征

        for teacher_model in self.diff_teacher_bank.values():  # 若存在多教师，先统一切换至评估模式
            teacher_model.eval()  # 设置教师为评估模式，确保推理一致性
        batch_info = {}  # 初始化批处理信息占位符
        sample_count = len(batch_data_samples)  # 记录样本数量，便于构造输出容器
        primary_feature_list: List[Any] = [None] * sample_count  # 为主教师特征预留位置
        peer_feature_list: List[Dict[str, Any]] = [dict() for _ in range(sample_count)]  # 初始化同伴特征字典列表
        sensor_to_indices: Dict[str, List[int]] = {}  # 构建传感器到样本索引的映射
        for idx, data_samples in enumerate(batch_data_samples):  # 遍历批次中的每个样本
            if not hasattr(data_samples, 'metainfo') or 'sensor' not in data_samples.metainfo:  # 若样本缺少传感器标签
                raise KeyError(f'伪标签生成失败：第{idx}个数据样本缺失sensor标签。')  # 抛出清晰错误提示
            sensor_tag = data_samples.metainfo['sensor']  # 读取传感器标签
            if sensor_tag not in self.diff_teacher_bank:  # 若未找到对应教师
                raise KeyError(f'伪标签生成失败：未找到传感器"{sensor_tag}"对应的扩散教师。')  # 抛出错误提示
            sensor_to_indices.setdefault(sensor_tag, []).append(idx)  # 将样本索引归类到对应传感器
        for sensor_tag, sample_indices in sensor_to_indices.items():  # 遍历每个传感器分组
            teacher_model = self.diff_teacher_bank[sensor_tag]  # 获取当前传感器的主教师
            group_inputs = batch_inputs[sample_indices]  # 按索引提取对应图像张量
            group_samples_for_pred = [copy.deepcopy(batch_data_samples[i]) for i in sample_indices]  # 复制数据样本供教师推理使用
            primary_results, primary_feature = teacher_model.predict(  # 使用主教师进行预测并返回特征
                group_inputs, group_samples_for_pred, rescale=False, return_feature=True)
            for local_idx, sample_idx in enumerate(sample_indices):  # 遍历组内样本
                origin_sample = batch_data_samples[sample_idx]  # 获取原始数据样本
                origin_sample.gt_instances = primary_results[local_idx].pred_instances  # 写入伪标签实例
                origin_sample.gt_instances.bboxes = bbox_project(  # 逆映射预测框到原图空间
                    origin_sample.gt_instances.bboxes,
                    torch.from_numpy(origin_sample.homography_matrix).inverse().to(
                        self.data_preprocessor.device), origin_sample.ori_shape)
                primary_feature_list[sample_idx] = primary_feature[local_idx]  # 记录主教师特征，按原批序填充
            for peer_sensor, peer_teacher in self.diff_teacher_bank.items():  # 遍历其它教师以获取交叉特征
                if peer_sensor == sensor_tag:  # 跳过主教师自身
                    continue  # 仅处理同伴教师
                peer_inputs = batch_inputs[sample_indices]  # 使用相同图像张量作为输入
                peer_samples = [copy.deepcopy(batch_data_samples[i]) for i in sample_indices]  # 复制样本以避免状态污染
                _, peer_feature = peer_teacher.predict(  # 调用同伴教师获取特征
                    peer_inputs, peer_samples, rescale=False, return_feature=True)
                for local_idx, sample_idx in enumerate(sample_indices):  # 将同伴特征写入对应位置
                    peer_feature_list[sample_idx][peer_sensor] = peer_feature[local_idx]  # 按传感器标签归档特征
        distill_feature = {  # 汇总主教师与同伴教师特征
            'primary': primary_feature_list,  # 主教师特征列表，顺序与批输入一致
            'peer': peer_feature_list  # 同伴特征字典列表，每个元素对应一个样本
        }
        return batch_data_samples, batch_info, distill_feature  # 返回伪标签、批信息与多教师特征

    def _parse_diff_feature(self, diff_feature: Any, batch_info: dict) -> dict:
        """中文注释：将扩散教师返回的特征结构统一整理便于后续蒸馏使用。"""
        main_teacher_feature = diff_feature  # 中文注释：默认主教师特征直接等于原始特征
        cross_teacher_info = None  # 中文注释：初始化交叉教师信息为空
        if isinstance(diff_feature, dict):  # 中文注释：当特征以字典形式提供时按约定键解析
            main_teacher_feature = diff_feature.get('main_teacher', diff_feature.get('teacher_feature'))  # 中文注释：优先读取主教师特征键
            cross_teacher_info = diff_feature.get('cross_teacher')  # 中文注释：获取交叉教师相关信息块
        parsed_feature = {'main_teacher': main_teacher_feature}  # 中文注释：构建标准化返回字典并写入主教师特征
        if cross_teacher_info is not None:  # 中文注释：若存在交叉教师信息则进一步整理
            if isinstance(cross_teacher_info, dict):  # 中文注释：仅在结构为字典时尝试补充批次信息
                cross_sensor = cross_teacher_info.get('sensor', 'cross_teacher')  # 中文注释：读取交叉教师的传感器标识若缺省则给默认值
                cross_predictions = cross_teacher_info.get('predictions')  # 中文注释：获取交叉教师的预测结果用于下游组件
                if cross_predictions is not None:  # 中文注释：当存在预测结果时写入批次信息供后续使用
                    batch_info.setdefault('cross_teacher', {})  # 中文注释：确保批次信息中存在交叉教师条目
                    batch_info['cross_teacher']['predictions'] = cross_predictions  # 中文注释：记录交叉教师的预测数据
                    batch_info['cross_teacher']['sensor'] = cross_sensor  # 中文注释：补充交叉教师的传感器名称便于诊断
            parsed_feature['cross_teacher'] = cross_teacher_info  # 中文注释：在标准化结果中保留完整的交叉教师信息
        return parsed_feature  # 中文注释：返回解析后的特征字典

    def project_pseudo_instances(self, batch_pseudo_instances: SampleList,
                                 batch_data_samples: SampleList) -> SampleList:
        """Project pseudo instances."""
        for pseudo_instances, data_samples in zip(batch_pseudo_instances,
                                                  batch_data_samples):
            data_samples.gt_instances = copy.deepcopy(
                pseudo_instances.gt_instances)
            data_samples.gt_instances.bboxes = bbox_project(
                data_samples.gt_instances.bboxes,
                torch.tensor(data_samples.homography_matrix).to(
                    self.data_preprocessor.device), data_samples.img_shape)
        wh_thr = self.semi_train_cfg.get('min_pseudo_bbox_wh', (1e-2, 1e-2))
        return filter_gt_instances(batch_data_samples, wh_thr=wh_thr)

    def predict(self, batch_inputs: Tensor,
                batch_data_samples: SampleList) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing.

        Args:
            batch_inputs (Tensor): Inputs with shape (N, C, H, W).
            batch_data_samples (List[:obj:`DetDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.
            rescale (bool): Whether to rescale the results.
                Defaults to True.

        Returns:
            list[:obj:`DetDataSample`]: Return the detection results of the
            input images. The returns value is DetDataSample,
            which usually contain 'pred_instances'. And the
            ``pred_instances`` usually contains following keys.

                - scores (Tensor): Classification scores, has a shape
                    (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                    (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                    the last dimension 4 arrange as (x1, y1, x2, y2).
                - masks (Tensor): Has a shape (num_instances, H, W).
        """
        if self.semi_test_cfg.get('predict_on', 'teacher') == 'teacher':
            return self.teacher(
                batch_inputs, batch_data_samples, mode='predict')
        elif self.semi_test_cfg.get('predict_on', 'teacher') == 'student':
            return self.student(
                batch_inputs, batch_data_samples, mode='predict')
        else:
            return self.diff_detector(
                batch_inputs, batch_data_samples, mode='predict')

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
        if self.semi_test_cfg.get('forward_on', 'teacher') == 'teacher':
            return self.teacher(batch_inputs, batch_data_samples, mode='tensor')
        elif self.semi_test_cfg.get('forward_on', 'teacher') == 'student':
            return self.student(batch_inputs, batch_data_samples, mode='tensor')
        elif self.semi_test_cfg.get('forward_on', 'teacher') == 'diff_detector':
            return self.diff_detector(batch_inputs, batch_data_samples, mode='tensor')

    def extract_feat(self, batch_inputs: Tensor) -> Tuple[Tensor]:
        """Extract features.

        Args:
            batch_inputs (Tensor): Image tensor with shape (N, C, H ,W).

        Returns:
            tuple[Tensor]: Multi-level features that may have
            different resolutions.
        """
        if self.semi_test_cfg.get('extract_feat_on', 'teacher') == 'teacher':
            return self.teacher.extract_feat(batch_inputs)
        elif self.semi_test_cfg.get('extract_feat_on', 'teacher') == 'student':
            return self.student.extract_feat(batch_inputs)
        elif self.semi_test_cfg.get('extract_feat_on', 'teacher') == 'diff_detector':
            return self.diff_detector.extract_feat(batch_inputs)

    def _load_from_state_dict(self, state_dict: dict, prefix: str,
                              local_metadata: dict, strict: bool,
                              missing_keys: Union[List[str], str],
                              unexpected_keys: Union[List[str], str],
                              error_msgs: Union[List[str], str]) -> None:
        """Add teacher and student prefixes to model parameter names."""
        if not any([
            'student' in key or 'teacher' in key
            for key in state_dict.keys()
        ]):
            keys = list(state_dict.keys())
            state_dict.update({'teacher.' + k: state_dict[k] for k in keys})
            state_dict.update({'student.' + k: state_dict[k] for k in keys})
            for k in keys:
                state_dict.pop(k)
        return super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
