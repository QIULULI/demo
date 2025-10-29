# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import Any, Dict, Tuple  # 中文注释：引入Any用于处理动态结构的特征信息
import torch
from torch import Tensor

from mmdet.models.utils import (rename_loss_dict,
                                reweight_loss_dict)
from mmdet.registry import MODELS
from mmdet.structures import SampleList
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from .base import BaseDetector
from ..losses import KDLoss


@MODELS.register_module()
class DomainAdaptationDetector(BaseDetector):
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
        assert train_cfg.detector_cfg.get('type',
                                          None) is not None, "train_cfg.detector_cfg must use type select one detector"
        assert train_cfg.detector_cfg.get('type') in ['UDA', 'SemiBase', 'SoftTeacher', 'SemiBaseDiff'], \
            "da_cfg type must select in ['UDA','SemiBase','SoftTeacher', 'SemiBaseDiff]"
        super().__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        self.model = MODELS.build(detector)
        self.detector_name = detector.get('type')
        self.train_cfg = train_cfg
        
        feature_loss_cfg = self.train_cfg.feature_loss_cfg  # 中文注释：提取特征蒸馏配置方便后续多次使用
        self.feature_loss_type = feature_loss_cfg.get(
            'feature_loss_type')  # 中文注释：读取主教师特征蒸馏所采用的损失类型
        self.feature_loss_weight = feature_loss_cfg.get(
            'feature_loss_weight')  # 中文注释：读取主教师特征蒸馏的损失权重
        self.feature_loss = KDLoss(
            loss_weight=self.feature_loss_weight, loss_type=self.feature_loss_type)  # 中文注释：构建主教师特征蒸馏损失实例
        self.cross_feature_loss_weight = feature_loss_cfg.get(
            'cross_feature_loss_weight', 0.0)  # 中文注释：读取交叉教师特征蒸馏的权重默认0保持兼容
        self.cross_feature_loss = None  # 中文注释：初始化交叉教师特征蒸馏损失为空以便按需启用
        if self.cross_feature_loss_weight > 0:  # 中文注释：仅当配置权重大于0时才实例化交叉教师损失
            self.cross_feature_loss = KDLoss(
                loss_weight=self.cross_feature_loss_weight, loss_type=self.feature_loss_type)  # 中文注释：构建交叉教师特征蒸馏损失
        self.cross_consistency_cfg = feature_loss_cfg.get(
            'cross_consistency_cfg', {})  # 中文注释：读取交叉教师分类与回归一致性配置默认空字典
        self.cross_cls_loss_weight = self.cross_consistency_cfg.get(
            'cls_weight', 0.0)  # 中文注释：获取交叉教师分类一致性损失的权重默认关闭
        self.cross_reg_loss_weight = self.cross_consistency_cfg.get(
            'reg_weight', 0.0)  # 中文注释：获取交叉教师回归一致性损失的权重默认关闭
        self.burn_up_iters = self.train_cfg.detector_cfg.get('burn_up_iters', 0)
        self.local_iter = 0

    @property
    def with_rpn(self):
        if self.with_student:
            return hasattr(self.model.student, 'rpn_head')
        else:
            return hasattr(self.student, 'rpn_head')

    @property
    def with_student(self):
        return hasattr(self.model, 'student')

    def loss(self, multi_batch_inputs: Dict[str, Tensor],
             multi_batch_data_samples: Dict[str, SampleList]) -> dict:
        """Calculate losses from multi-branch inputs and data samples.

        Args:
        Returns:
            dict: A dictionary of loss components
        """
        losses = dict() 
        if self.train_cfg.detector_cfg.get('type') in ['SemiBase']:
            if self.local_iter >= self.burn_up_iters:
                losses.update(**self.model.loss(multi_batch_inputs, multi_batch_data_samples))
            else:
                losses.update(**self.model.loss_by_gt_instances(multi_batch_inputs['sup'], multi_batch_data_samples['sup']))
            self.local_iter += 1
            
        elif self.train_cfg.detector_cfg.get('type') in ['SemiBaseDiff']:
            if self.local_iter >= self.burn_up_iters:
                semi_loss, diff_feature = self.model.loss_diff_adaptation(multi_batch_inputs, multi_batch_data_samples)
                feature_loss = self.loss_feature(multi_batch_inputs['unsup_teacher'], diff_feature)
                losses.update(**semi_loss)
                losses.update(**feature_loss)
            else:
                losses.update(**self.model.loss_by_gt_instances(multi_batch_inputs['sup'], multi_batch_data_samples['sup']))
                
            self.local_iter += 1
        else:
            raise "detector type not in ['SemiBase','SoftTeacher','SemiBaseDiff'] "
        return losses

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
        if self.with_student:
            if self.model.semi_test_cfg.get('predict_on', 'teacher') == 'teacher':
                return self.model.teacher(batch_inputs, batch_data_samples, mode='predict')
            elif self.model.semi_test_cfg.get('predict_on', 'teacher') == 'student':
                return self.model.student(batch_inputs, batch_data_samples, mode='predict')
            elif self.model.semi_test_cfg.get('predict_on', 'teacher') == 'diff_detector':
                return self.model.diff_detector(batch_inputs, batch_data_samples, mode='predict')
        else:
            return self.model(batch_inputs, batch_data_samples, mode='predict')

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

    
    def cross_loss(self, batch_inputs: Tensor, batch_data_samples: SampleList):
        losses = dict()

        diff_x = self.model.diff_detector.extract_feat(batch_inputs)
          
        if not self.with_rpn:
            detector_loss = self.model.student.bbox_head.loss(
                diff_x, batch_data_samples)
            losses.update(rename_loss_dict(
                'cross_', detector_loss))
        else:
            proposal_cfg = self.model.student.train_cfg.get(
                'rpn_proposal', self.model.student.test_cfg.rpn)
            rpn_data_samples = copy.deepcopy(batch_data_samples)
            # set cat_id of gt_labels to 0 in RPN
            for data_sample in rpn_data_samples:
                data_sample.gt_instances.labels = torch.zeros_like(
                    data_sample.gt_instances.labels)
            rpn_losses, rpn_results_list = self.model.student.rpn_head.loss_and_predict(
                diff_x, rpn_data_samples, proposal_cfg=proposal_cfg)
            # avoid get same name with roi_head loss
            keys = rpn_losses.keys()
            for key in list(keys):
                if 'loss' in key and 'rpn' not in key:
                    rpn_losses[f'rpn_{key}'] = rpn_losses.pop(key)
            losses.update(rename_loss_dict(
                'cross_', rpn_losses))
            roi_losses = self.model.student.roi_head.loss(
                diff_x, rpn_results_list, batch_data_samples)
            losses.update(rename_loss_dict(
                'cross_', roi_losses))
        return losses, diff_x


    def loss_feature(self, batch_inputs: Tensor, diff_feature) -> dict:
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
        student_x = self.model.student.extract_feat(batch_inputs)  # 中文注释：提取学生模型在输入图像上的多尺度特征
        main_teacher_feature = diff_feature  # 中文注释：默认将传入特征视作主教师特征
        cross_teacher_info = None  # 中文注释：初始化交叉教师信息为None
        if isinstance(diff_feature, dict):  # 中文注释：当传入结构为字典时按约定键拆分主教师与交叉教师
            main_teacher_feature = diff_feature.get('main_teacher')  # 中文注释：提取主教师的多尺度特征集合
            cross_teacher_info = diff_feature.get('cross_teacher')  # 中文注释：提取交叉教师相关信息块
        losses = dict()  # 中文注释：初始化最终损失字典
        feature_loss = dict()  # 中文注释：准备记录特征蒸馏相关损失
        feature_loss['pkd_feature_loss'] = 0  # 中文注释：初始化主教师特征蒸馏损失累积值
        if main_teacher_feature is not None:  # 中文注释：仅在主教师特征有效时计算蒸馏损失
            if isinstance(main_teacher_feature, (list, tuple)):  # 中文注释：当主教师特征为序列时直接使用
                teacher_feature_list = main_teacher_feature  # 中文注释：记录主教师特征列表
            else:  # 中文注释：若主教师特征非序列则包装成单元素列表保持统一
                teacher_feature_list = [main_teacher_feature]  # 中文注释：构建仅含单层特征的列表
            num_teacher_layers = max(len(teacher_feature_list), 1)  # 中文注释：计算主教师特征层数避免除零
            for student_feature, teacher_feature in zip(student_x, teacher_feature_list):  # 中文注释：遍历对应层特征计算蒸馏损失
                layer_loss = self.feature_loss(
                    student_feature, teacher_feature)  # 中文注释：计算学生特征与主教师特征的蒸馏损失
                feature_loss['pkd_feature_loss'] = feature_loss['pkd_feature_loss'] + layer_loss / num_teacher_layers  # 中文注释：将损失按层数平均累积
        if cross_teacher_info is not None and self.cross_feature_loss is not None:  # 中文注释：若交叉教师信息存在且开启权重则计算额外特征蒸馏
            cross_features = None  # 中文注释：初始化交叉教师特征容器
            if isinstance(cross_teacher_info, dict):  # 中文注释：交叉教师信息为字典时尝试读取标准键
                cross_features = cross_teacher_info.get('features', cross_teacher_info.get('main_feature'))  # 中文注释：读取交叉教师提供的多尺度特征
            elif isinstance(cross_teacher_info, (list, tuple)):  # 中文注释：若直接给出序列则直接使用
                cross_features = cross_teacher_info  # 中文注释：直接记录交叉教师特征序列
            if cross_features is not None:  # 中文注释：仅当获取到有效的交叉教师特征时计算蒸馏损失
                cross_feature_list = cross_features if isinstance(cross_features, (list, tuple)) else [cross_features]  # 中文注释：确保交叉特征以列表形式处理
                num_cross_layers = max(len(cross_feature_list), 1)  # 中文注释：记录交叉教师特征层数避免除零
                cross_loss_value = 0  # 中文注释：初始化交叉特征损失累计值
                for student_feature, cross_feature in zip(student_x, cross_feature_list):  # 中文注释：遍历学生与交叉教师对应层
                    layer_loss = self.cross_feature_loss(
                        student_feature, cross_feature)  # 中文注释：计算每层的交叉教师特征蒸馏损失
                    cross_loss_value = cross_loss_value + layer_loss / num_cross_layers  # 中文注释：按层数平均累积交叉特征损失
                feature_loss['pkd_cross_feature_loss'] = cross_loss_value  # 中文注释：将交叉教师特征蒸馏损失写入结果字典
        if cross_teacher_info is not None:  # 中文注释：若存在交叉教师附加信息则进一步计算一致性损失
            feature_loss.update(self.loss_cross_feature(cross_teacher_info))  # 中文注释：调用辅助函数计算分类与回归一致性损失
        losses.update(feature_loss)  # 中文注释：将所有特征相关损失合并到最终输出
        return losses  # 中文注释：返回损失字典供训练流程使用

    def loss_cross_feature(self, cross_teacher_info: Any) -> dict:
        """中文注释：计算交叉教师提供的分类与回归一致性损失。"""
        losses = dict()  # 中文注释：初始化一致性损失字典
        if not isinstance(cross_teacher_info, dict):  # 中文注释：当交叉教师信息非字典时直接返回空损失
            return losses  # 中文注释：保持接口兼容返回空结果
        if self.cross_cls_loss_weight > 0:  # 中文注释：仅在分类一致性权重大于0时计算
            cls_value = cross_teacher_info.get('cls_consistency')  # 中文注释：优先读取顶层分类一致性数值
            if cls_value is None:  # 中文注释：若顶层缺失则尝试从嵌套结构读取
                consistency_block = cross_teacher_info.get('consistency', {})  # 中文注释：获取可能的嵌套一致性字典
                cls_value = consistency_block.get('cls')  # 中文注释：从嵌套字典读取分类一致性项
            if cls_value is not None:  # 中文注释：只有在成功取得数值时才写入损失
                losses['cross_cls_consistency_loss'] = cls_value * self.cross_cls_loss_weight  # 中文注释：按配置权重缩放分类一致性损失
        if self.cross_reg_loss_weight > 0:  # 中文注释：仅在回归一致性权重大于0时计算
            reg_value = cross_teacher_info.get('reg_consistency')  # 中文注释：优先读取顶层回归一致性数值
            if reg_value is None:  # 中文注释：若未提供则尝试从嵌套结构读取
                consistency_block = cross_teacher_info.get('consistency', {})  # 中文注释：获取可能的嵌套一致性字典
                reg_value = consistency_block.get('reg')  # 中文注释：从嵌套字典读取回归一致性项
            if reg_value is not None:  # 中文注释：确保存在有效数值再写入结果
                losses['cross_reg_consistency_loss'] = reg_value * self.cross_reg_loss_weight  # 中文注释：按权重缩放回归一致性损失
        return losses  # 中文注释：返回交叉教师一致性损失字典



