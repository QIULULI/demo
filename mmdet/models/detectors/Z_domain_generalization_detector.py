# Copyright (c) OpenMMLab. All rights reserved.
import copy
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
        if self.with_student:  # 新增中文注释：判断当前模型是否包含学生模型分支
            predict_on = self.model.semi_test_cfg.get('predict_on', 'teacher')  # 新增中文注释：获取推理时应使用的模型分支
            if predict_on == 'teacher':  # 新增中文注释：若选择教师模型进行预测
                result = self.model.teacher(  # 新增中文注释：调用教师模型并传递缩放及特征返回参数
                    batch_inputs,  # 新增中文注释：教师模型的图像输入
                    batch_data_samples,  # 新增中文注释：教师模型的样本数据输入
                    mode='predict',  # 新增中文注释：指定教师模型处于预测模式
                    rescale=rescale,  # 新增中文注释：传递是否还原缩放的标志
                    return_feature=return_feature)  # 新增中文注释：传递是否返回特征的开关
            elif predict_on == 'student':  # 新增中文注释：若选择学生模型进行预测
                result = self.model.student(  # 新增中文注释：调用学生模型并传递相同参数
                    batch_inputs,  # 新增中文注释：学生模型的图像输入
                    batch_data_samples,  # 新增中文注释：学生模型的样本数据输入
                    mode='predict',  # 新增中文注释：指定学生模型处于预测模式
                    rescale=rescale,  # 新增中文注释：传递是否还原缩放的标志
                    return_feature=return_feature)  # 新增中文注释：传递是否返回特征的开关
            elif predict_on == 'diff_detector':  # 新增中文注释：若选择扩散检测器进行预测
                result = self.model.diff_detector(  # 新增中文注释：调用扩散检测器并传递相同参数
                    batch_inputs,  # 新增中文注释：扩散检测器的图像输入
                    batch_data_samples,  # 新增中文注释：扩散检测器的样本数据输入
                    mode='predict',  # 新增中文注释：指定扩散检测器处于预测模式
                    rescale=rescale,  # 新增中文注释：传递是否还原缩放的标志
                    return_feature=return_feature)  # 新增中文注释：传递是否返回特征的开关
        else:  # 新增中文注释：若不存在师生结构则直接调用底层模型
            result = self.model(  # 新增中文注释：调用基础模型并传递缩放与特征开关参数
                batch_inputs,  # 新增中文注释：基础模型的图像输入
                batch_data_samples,  # 新增中文注释：基础模型的样本数据输入
                mode='predict',  # 新增中文注释：指定基础模型处于预测模式
                rescale=rescale,  # 新增中文注释：传递是否还原缩放的标志
                return_feature=return_feature)  # 新增中文注释：传递是否返回特征的开关

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

    def cross_loss_diff_to_student(self, batch_data_samples: SampleList, diff_fpn):
            losses = dict()
            if not self.with_rpn:
                detector_loss = self.model.student.bbox_head.loss(
                    diff_fpn, batch_data_samples)
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
                    diff_fpn, rpn_data_samples, proposal_cfg=proposal_cfg)
                # avoid get same name with roi_head loss
                keys = rpn_losses.keys()
                for key in list(keys):
                    if 'loss' in key and 'rpn' not in key:
                        rpn_losses[f'rpn_{key}'] = rpn_losses.pop(key)
                losses.update(rename_loss_dict(
                    'cross_', rpn_losses))
                roi_losses = self.model.student.roi_head.loss(
                    diff_fpn, rpn_results_list, batch_data_samples)
                losses.update(rename_loss_dict(
                    'cross_', roi_losses))
            losses = reweight_loss_dict(losses=losses, weight=self.cross_loss_weight)
            return losses

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
        student_x = self.model.student.extract_feat(batch_inputs)
        diff_x = self.model.diff_detector.extract_feat(batch_inputs)
        losses = dict()

        # cross model loss
        ##############################################################################################################
        losses.update(
            **self.cross_loss_diff_to_student(batch_data_samples, diff_x))
        ##############################################################################################################
        
        # feature kd loss
        ##############################################################################################################
        feature_loss = dict()
        feature_loss['pkd_feature_loss'] = 0
        for i, (student_feature, diff_feature) in enumerate(zip(student_x, diff_x)):
            layer_loss = self.feature_loss(
                student_feature, diff_feature)
            feature_loss['pkd_feature_loss'] += layer_loss/len(diff_x)
        losses.update(feature_loss)
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
        roi_losses_kd = self.roi_head_loss_with_kd(
            student_x, diff_x, rpn_results_list, batch_data_samples)
        losses.update(roi_losses_kd)
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