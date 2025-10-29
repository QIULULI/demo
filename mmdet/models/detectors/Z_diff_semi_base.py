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

        self.student = MODELS.build(detector.deepcopy())  # 构建学生模型副本
        self.teacher = MODELS.build(detector.deepcopy())  # 构建教师模型副本
        self.diff_detector = None  # 初始化当前启用的扩散教师指针
        self.diff_detectors = dict()  # 构建一个字典用于缓存所有扩散教师
        self.active_diff_key = None  # 记录当前激活的扩散教师标识
        teacher_pool = None  # 初始化教师配置池占位
        if isinstance(diff_model, dict):  # 当diff_model为常规字典或ConfigDict时提取多教师配置
            teacher_pool = diff_model.get('teachers', None)  # 读取teachers列表
        if teacher_pool:  # 若提供多教师配置列表
            for teacher_cfg in teacher_pool:  # 遍历每个教师配置条目
                teacher_name = teacher_cfg.get('name')  # 读取教师名称（区分IR/RGB）
                if not teacher_name:  # 若缺失名称则跳过以避免键冲突
                    continue  # 直接跳过当前教师
                teacher_config_path = teacher_cfg.get('config', '')  # 读取教师对应的配置文件路径
                built_teacher = None  # 初始化教师模型占位
                if teacher_config_path:  # 若提供独立配置文件
                    teacher_config = Config.fromfile(teacher_config_path)  # 通过配置文件构建Config对象
                    built_teacher = MODELS.build(teacher_config['model'])  # 根据配置构建教师模型
                else:  # 若未提供配置则退回当前检测器结构
                    built_teacher = MODELS.build(detector.deepcopy())  # 直接拷贝主检测器结构
                pretrained_path = teacher_cfg.get('pretrained_model') or teacher_cfg.get('checkpoint')  # 读取权重路径
                if pretrained_path:  # 若提供预训练权重
                    load_checkpoint(built_teacher, pretrained_path, map_location='cpu', strict=True)  # 加载对应权重
                built_teacher.cuda()  # 将教师模型放置到GPU上
                self.freeze(built_teacher)  # 冻结教师模型参数
                self.diff_detectors[teacher_name] = built_teacher  # 将教师模型缓存到字典中
            if self.diff_detectors:  # 若成功构建至少一个教师
                self.active_diff_key = diff_model.get('main_teacher') if isinstance(diff_model, dict) else None  # 读取首选教师标识
                if not self.active_diff_key or self.active_diff_key not in self.diff_detectors:  # 若未指定或名称非法
                    self.active_diff_key = next(iter(self.diff_detectors.keys()))  # 回退到字典第一个教师
                self.diff_detector = self.diff_detectors[self.active_diff_key]  # 设置当前激活教师
        if self.diff_detector is None and getattr(diff_model, 'config', None):  # 若尚未成功加载教师且存在旧式配置
            teacher_config = Config.fromfile(diff_model.config)  # 解析旧式单教师配置
            self.diff_detector = MODELS.build(teacher_config['model'])  # 构建扩散教师模型
            if diff_model.pretrained_model:  # 若提供对应权重
                load_checkpoint(self.diff_detector, diff_model.pretrained_model, map_location='cpu', strict=True)  # 加载权重
                self.diff_detector.cuda()  # 将教师迁移到GPU
                self.freeze(self.diff_detector)  # 冻结教师参数
            self.diff_detectors['default'] = self.diff_detector  # 将单教师也放入缓存字典
            self.active_diff_key = 'default'  # 记录当前教师标识
        if self.diff_detector is None:  # 若仍未找到有效教师
            self.diff_detector = self.student  # 使用学生模型作为退路教师
            self.diff_detectors['student'] = self.student  # 将学生注册进教师字典
            self.active_diff_key = 'student'  # 记录当前教师标识

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

    def cuda(self, device: Optional[str] = None) -> nn.Module:
        """Since teacher is registered as a plain object, it is necessary to
        put the teacher model to cuda when calling ``cuda`` function."""
        for detector_name, detector_module in getattr(self, 'diff_detectors', {}).items():  # 遍历所有扩散教师模块
            detector_module.cuda(device=device)  # 将每个扩散教师迁移到指定设备
        return super().cuda(device=device)

    def to(self, device: Optional[str] = None) -> nn.Module:
        """Since teacher is registered as a plain object, it is necessary to
        put the teacher model to other device when calling ``to`` function."""
        for detector_name, detector_module in getattr(self, 'diff_detectors', {}).items():  # 遍历全部扩散教师
            detector_module.to(device=device)  # 将每个扩散教师迁移到目标设备
        return super().to(device=device)

    def train(self, mode: bool = True) -> None:
        """Set the same train mode for teacher and student model."""
        for detector_name, detector_module in getattr(self, 'diff_detectors', {}).items():  # 遍历所有扩散教师
            detector_module.train(False)  # 强制所有扩散教师保持评估模式
        super().train(mode)

    def __setattr__(self, name: str, value: Any) -> None:
        """Set attribute, i.e. self.name = value

        This reloading prevent the teacher model from being registered as a
        nn.Module. The teacher module is registered as a plain object, so that
        the teacher parameters will not show up when calling
        ``self.parameters``, ``self.modules``, ``self.children`` methods.
        """
        if name in ['diff_detector', 'diff_detectors']:
            object.__setattr__(self, name, value)
        else:
            super().__setattr__(name, value)

    def set_active_diff_detector(self, name: str) -> None:
        """根据名称切换当前使用的扩散教师模型。"""
        if not hasattr(self, 'diff_detectors'):  # 若尚未初始化教师字典则直接返回
            return  # 无需进一步处理
        if name not in self.diff_detectors:  # 若指定名称不存在
            return  # 忽略非法请求
        object.__setattr__(self, 'diff_detector', self.diff_detectors[name])  # 直接替换当前扩散教师引用
        self.active_diff_key = name  # 更新当前教师标识
        self.diff_detector.train(False)  # 确保新教师保持评估模式

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
            multi_batch_inputs['unsup_teacher'], multi_batch_data_samples['unsup_teacher'])
        multi_batch_data_samples['unsup_student'] = self.project_pseudo_instances(
            origin_pseudo_data_samples, multi_batch_data_samples['unsup_student'])

        losses.update(**self.loss_by_pseudo_instances(multi_batch_inputs['unsup_student'],
                                                      multi_batch_data_samples['unsup_student'], batch_info))
        return losses, diff_feature

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
        self.diff_detector.eval()
        results_list, diff_feature = self.diff_detector.predict(
            batch_inputs, batch_data_samples, rescale=False, return_feature=True)
        batch_info = {}
        for data_samples, results in zip(batch_data_samples, results_list):
            data_samples.gt_instances = results.pred_instances
            data_samples.gt_instances.bboxes = bbox_project(
                data_samples.gt_instances.bboxes,
                torch.from_numpy(data_samples.homography_matrix).inverse().to(
                    self.data_preprocessor.device), data_samples.ori_shape)
        return batch_data_samples, batch_info, diff_feature

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
