# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import Any, Dict, Tuple, Optional  # 中文注释：引入Optional以便在函数签名中携带current_iter开关
import torch
import torch.nn.functional as F  # 中文注释：引入函数式接口用于计算一致性与MSE损失
from torch import Tensor
from torchvision.ops import roi_align  # 中文注释：导入ROIAlign用于在域不变特征上执行ROI采样

from mmdet.models.utils import (rename_loss_dict,
                                reweight_loss_dict)
from mmdet.registry import MODELS
from mmdet.structures import SampleList
from mmdet.structures.bbox import bbox_overlaps  # 中文注释：导入IoU计算工具以匹配教师与学生伪框
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from mmengine.logging import MMLogger  # 中文注释：引入日志记录器以便在调试阶段输出一致性损失信息
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
        self.train_cfg = train_cfg  # 中文注释：缓存训练配置方便后续读取蒸馏调度参数
        self.ssdc_cfg = self.train_cfg.get('ssdc_cfg', {})  # 中文注释：读取SS-DC相关的训练配置便于后续动态调度
        self.ssdc_compute_in_wrapper = bool(self.ssdc_cfg.get('compute_in_wrapper', True))  # 中文注释：控制SS-DC损失是否仅由包装器统一计算默认开启避免重复
        self.ssdc_skip_student_loss = bool(self.ssdc_cfg.get('skip_student_ssdc_loss', self.ssdc_compute_in_wrapper))  # 中文注释：当包装器统一计算时默认跳过学生内部损失
        self.ssdc_cfg.setdefault('skip_local_loss', self.ssdc_skip_student_loss)  # 中文注释：向学生侧传递跳过开关保持配置一致
        self._propagate_ssdc_skip_flags()  # 中文注释：在模型构建后立即下发跳过本地SS-DC损失的控制位避免重复累加
        self._cached_main_teacher_inv = None  # 中文注释：缓存扩散教师提供的域不变特征以便在SS-DC阶段复用
        detector_cfg = self.train_cfg.detector_cfg  # 中文注释：提取检测器子配置以便访问蒸馏相关阈值
        self.warmup_start_iters = self.train_cfg.get(  # 中文注释：读取蒸馏预热起始迭代默认0保持兼容
            'warmup_start_iters', detector_cfg.get('warmup_start_iters', 0))
        self.warmup_ramp_iters = self.train_cfg.get(  # 中文注释：读取蒸馏预热线性爬坡长度默认0表示无预热
            'warmup_ramp_iters', detector_cfg.get('warmup_ramp_iters', 0))
        
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
        self.cross_consistency_debug_log = self.cross_consistency_cfg.get(
            'debug_log', False)  # 中文注释：读取是否启用交叉一致性损失的调试日志输出开关
        self.burn_up_iters = detector_cfg.get('burn_up_iters', 0)  # 中文注释：保留原有烧入阶段迭代控制逻辑
        self.local_iter = 0  # 中文注释：初始化局部迭代计数器供预热调度使用

    @property
    def with_rpn(self):
        if self.with_student:
            return hasattr(self.model.student, 'rpn_head')
        else:
            return hasattr(self.student, 'rpn_head')

    @property
    def with_student(self):
        return hasattr(self.model, 'student')

    def _propagate_ssdc_skip_flags(self) -> None:
        """中文注释：遍历学生/教师/扩散教师模块并同步跳过SS-DC本地损失的配置。"""
        skip_local_loss = bool(self.ssdc_cfg.get('skip_local_loss', self.ssdc_skip_student_loss))  # 中文注释：解析应当下发的跳过标志并确保布尔化避免歧义
        self.ssdc_cfg['skip_local_loss'] = skip_local_loss  # 中文注释：回写最终布尔值便于后续组件引用统一语义
        target_modules = []  # 中文注释：收集需要同步开关的模块实例
        for attr_name in ('student', 'teacher', 'diff_detector'):  # 中文注释：依次检查学生、主教师以及默认扩散教师
            module = getattr(self.model, attr_name, None)  # 中文注释：安全地获取对应属性避免未定义时报错
            if module is not None:  # 中文注释：仅在模块存在时才加入同步列表
                target_modules.append(module)  # 中文注释：缓存需要更新的模块实例
        diff_teacher_bank = getattr(self.model, 'diff_detectors', None)  # 中文注释：尝试读取扩散教师字典以便同步所有分支
        if isinstance(diff_teacher_bank, dict):  # 中文注释：仅当教师仓库为字典时才遍历其中的模块
            for bank_module in diff_teacher_bank.values():  # 中文注释：遍历所有扩散教师实例
                if bank_module is not None:  # 中文注释：过滤掉空占位符
                    target_modules.append(bank_module)  # 中文注释：将额外教师加入待同步列表
        for module in target_modules:  # 中文注释：依次同步每个模块的开关状态
            module_ssdc_cfg = getattr(module, 'ssdc_cfg', None)  # 中文注释：尝试读取模块内部的SS-DC配置引用
            if module_ssdc_cfg is not None and hasattr(module_ssdc_cfg, 'setdefault'):  # 中文注释：确保配置结构支持setdefault以避免不兼容类型
                module_ssdc_cfg.setdefault('skip_local_loss', skip_local_loss)  # 中文注释：若模块尚未定义跳过字段则写入默认值
                try:  # 中文注释：尝试以映射形式直接覆盖字段
                    module_ssdc_cfg['skip_local_loss'] = skip_local_loss  # 中文注释：确保模块配置内的跳过字段与包装器保持一致
                except Exception:  # 中文注释：兼容无法通过下标赋值的配置对象
                    setattr(module_ssdc_cfg, 'skip_local_loss', skip_local_loss)  # 中文注释：回退到属性赋值方式写入布尔值
            if hasattr(module, 'ssdc_skip_local_loss'):  # 中文注释：若模块拥有运行期跳过标志则直接覆盖
                module.ssdc_skip_local_loss = skip_local_loss  # 中文注释：将布尔值写入模块实例确保本地loss逻辑立即生效

    def _get_distill_warmup_weight(self, current_iter: int) -> float:
        """中文注释：根据当前迭代计算蒸馏损失对应的预热权重。"""
        if current_iter < self.warmup_start_iters:  # 中文注释：在预热起点之前蒸馏权重保持为0
            return 0.0  # 中文注释：完全关闭蒸馏分支
        if self.warmup_ramp_iters <= 0:  # 中文注释：当未设置线性爬坡长度时直接启用完整蒸馏权重
            return 1.0  # 中文注释：立即返回满权重
        ramp_progress = current_iter - self.warmup_start_iters  # 中文注释：计算相对预热起点的进度
        ramp_progress = max(ramp_progress, 0)  # 中文注释：确保进度不为负值
        ramp_progress = min(ramp_progress, self.warmup_ramp_iters)  # 中文注释：将进度限制在爬坡区间内
        return ramp_progress / self.warmup_ramp_iters  # 中文注释：按比例映射到0到1之间的权重

    def loss(self, multi_batch_inputs: Dict[str, Tensor],
             multi_batch_data_samples: Dict[str, SampleList]) -> dict:
        """Calculate losses from multi-branch inputs and data samples.

        Args:
        Returns:
            dict: A dictionary of loss components
        """
        losses = dict() 
        current_iter = self.local_iter  # 中文注释：在进入分支前统一记录当前迭代索引用于所有调用透传
        if self.train_cfg.detector_cfg.get('type') in ['SemiBase']:
            if self.local_iter >= self.burn_up_iters:
                losses.update(**self.model.loss(multi_batch_inputs, multi_batch_data_samples))
            else:
                losses.update(**self.model.loss_by_gt_instances(multi_batch_inputs['sup'], multi_batch_data_samples['sup']))
            self.local_iter += 1

        elif self.train_cfg.detector_cfg.get('type') in ['SemiBaseDiff']:
            distill_blocked = (current_iter < self.burn_up_iters  # 中文注释：遵循旧版烧入阶段限制
                               or current_iter < self.warmup_start_iters)  # 中文注释：在预热起点前禁止蒸馏
            if distill_blocked:  # 中文注释：当蒸馏尚未开放时仅执行有监督分支
                losses.update(**self.model.loss_by_gt_instances(
                    multi_batch_inputs['sup'], multi_batch_data_samples['sup'], current_iter=current_iter))  # 中文注释：计算监督学习损失并同步传递迭代索引
            else:  # 中文注释：当满足蒸馏开放条件时执行跨模型与特征蒸馏分支
                warmup_weight = self._get_distill_warmup_weight(current_iter)  # 中文注释：根据迭代进度获取预热权重
                pseudo_payload = None  # 中文注释：初始化伪标签负载占位符，便于后续插入门控过滤
                semi_output = self.model.loss_diff_adaptation(
                    multi_batch_inputs, multi_batch_data_samples, ssdc_cfg=self.ssdc_cfg, current_iter=current_iter)  # 中文注释：获取跨模型损失与教师特征，必要时携带伪标签信息
                semi_loss, diff_feature = self._unpack_diff_adaptation_output(semi_output)  # 中文注释：通过统一解析函数解包损失与教师特征
                pseudo_payload = self._extract_pseudo_payload(semi_output, diff_feature)  # 中文注释：尝试从返回结果中抽取伪标签与批次信息
                if pseudo_payload is not None:  # 中文注释：当存在伪标签负载时执行域不变特征门控
                    self._apply_pseudo_consistency_gate(
                        pseudo_payload, multi_batch_inputs, current_iter)  # 中文注释：基于余弦相似度对伪标签进行过滤或衰减
                    if not self._has_pseudo_loss(semi_loss):  # 中文注释：若底层尚未计算伪标签损失则在包装器中补算
                        gated_student_samples = pseudo_payload.get('student_pseudo_samples', None)  # 中文注释：读取经过门控后的学生伪标签
                        batch_info = pseudo_payload.get('batch_info', None)  # 中文注释：提取批次信息便于投影使用
                        if gated_student_samples is not None and hasattr(self.model, 'loss_by_pseudo_instances'):  # 中文注释：确保存在学生伪标签与对应loss接口
                            pseudo_losses = self.model.loss_by_pseudo_instances(
                                multi_batch_inputs['unsup_student'], gated_student_samples, batch_info, current_iter=current_iter)  # 中文注释：调用底层loss计算门控后的伪标签损失
                            semi_loss.update(**pseudo_losses)  # 中文注释：将补算的伪标签损失合并到蒸馏损失中
                self._update_ssdc_teacher_override(diff_feature)  # 中文注释：尝试从扩散教师返回包中提取域不变特征供SS-DC损失阶段直接复用
                ssdc_losses = self._compute_ssdc_loss(current_iter) if self.ssdc_compute_in_wrapper else {}  # 中文注释：根据配置决定是否在包装器内汇总SS-DC损失
                feature_loss = self.loss_feature(
                    multi_batch_inputs['unsup_teacher'], diff_feature, current_iter=current_iter)  # 中文注释：在读取SS-DC损失后再计算特征蒸馏并同步当前迭代索引，避免额外前向刷新ssdc_feature_cache造成缓存漂移
                losses.update(**semi_loss)  # 中文注释：合并跨模型损失项
                losses.update(**ssdc_losses)  # 中文注释：提前合并SS-DC损失以保留与缓存读取顺序一致的日志
                losses.update(**feature_loss)  # 中文注释：最后合并特征蒸馏损失，保持接口输出结构不变
                if warmup_weight < 1.0:  # 中文注释：仅在预热阶段对蒸馏损失执行线性缩放
                    warmup_targets = dict()  # 中文注释：准备收集需要缩放的蒸馏损失条目
                    for key, value in semi_loss.items():  # 中文注释：遍历跨模型损失字典
                        if key.startswith('cross_') or key == 'pkd_feature_loss':  # 中文注释：筛选符合规则的蒸馏项
                            warmup_targets[key] = value  # 中文注释：记录待缩放的损失条目
                    for key, value in feature_loss.items():  # 中文注释：遍历特征蒸馏损失字典
                        if key.startswith('cross_') or key == 'pkd_feature_loss':  # 中文注释：保持与跨模型一致的筛选条件
                            warmup_targets[key] = value  # 中文注释：将匹配的特征蒸馏损失纳入缩放集合
                    scaled_losses = reweight_loss_dict(warmup_targets, warmup_weight)  # 中文注释：调用通用工具执行统一缩放
                    for key, value in scaled_losses.items():  # 中文注释：遍历缩放后的损失条目
                        if key in losses:  # 中文注释：仅在最终损失字典存在对应键时才写回
                            losses[key] = value  # 中文注释：用缩放结果覆盖原始损失确保预热策略生效

            self.local_iter += 1  # 中文注释：更新局部迭代计数器以驱动后续调度
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

    
    def cross_loss(self, batch_inputs: Tensor, batch_data_samples: SampleList, current_iter: Optional[int] = None):
        losses = dict()

        diff_x = self.model.diff_detector.extract_feat(batch_inputs, current_iter=current_iter)
          
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


    def loss_feature(self, batch_inputs: Tensor, diff_feature, current_iter: Optional[int] = None) -> dict:
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
        student_x = self.model.student.extract_feat(batch_inputs, current_iter=current_iter)  # 中文注释：提取学生模型在输入图像上的多尺度特征并同步记录迭代索引
        main_teacher_feature = diff_feature  # 中文注释：默认将传入特征视作主教师特征
        cross_teacher_info = None  # 中文注释：初始化交叉教师信息为None
        if isinstance(diff_feature, dict):  # 中文注释：当传入结构为字典时按约定键拆分主教师与交叉教师
            main_teacher_feature = diff_feature.get('main_teacher', diff_feature.get('teacher_feature', diff_feature))  # 中文注释：优先读取主教师特征若缺失则回退
            cross_teacher_info = diff_feature.get('cross_teacher')  # 中文注释：提取交叉教师相关信息块
        def _aggregate_sample_level_features(sample_feature_list):  # 中文注释：内部工具函数将按样本排列的多尺度特征转换为尺度优先张量
            if not isinstance(sample_feature_list, (list, tuple)) or not sample_feature_list:  # 中文注释：若输入不是序列或为空则直接返回None
                return None  # 中文注释：无有效特征时返回None
            level_count = 0  # 中文注释：初始化可用尺度数量统计
            for sample_entry in sample_feature_list:  # 中文注释：遍历每个样本的特征条目
                if isinstance(sample_entry, (list, tuple)):  # 中文注释：仅统计序列形式的样本特征
                    level_count = max(level_count, len(sample_entry))  # 中文注释：更新最大尺度数量用于后续聚合
            if level_count == 0:  # 中文注释：当所有样本均缺失尺度信息时
                return None  # 中文注释：直接返回None表示无法聚合
            aggregated_levels = []  # 中文注释：初始化尺度优先的输出列表
            for level_idx in range(level_count):  # 中文注释：逐个尺度执行堆叠
                level_candidates = []  # 中文注释：收集当前尺度的样本切片
                for sample_entry in sample_feature_list:  # 中文注释：遍历每个样本
                    candidate_tensor = None  # 中文注释：为当前样本的尺度特征初始化占位
                    if isinstance(sample_entry, (list, tuple)) and len(sample_entry) > level_idx:  # 中文注释：当样本提供该尺度特征时取出
                        candidate_tensor = sample_entry[level_idx]  # 中文注释：记录当前样本对应尺度的张量
                    elif torch.is_tensor(sample_entry) and level_idx == 0:  # 中文注释：兼容直接返回单张量的情况并仅对首尺度有效
                        candidate_tensor = sample_entry  # 中文注释：将张量直接视为尺度特征
                    level_candidates.append(candidate_tensor)  # 中文注释：无论是否缺失都追加占位保持顺序
                reference_tensor = next((item for item in level_candidates if torch.is_tensor(item)), None)  # 中文注释：寻找首个有效张量用于构造占位
                if reference_tensor is None:  # 中文注释：若当前尺度所有样本均缺失则以None占位
                    aggregated_levels.append(None)  # 中文注释：记录None保持尺度索引一致
                    continue  # 中文注释：继续处理下一尺度
                normalized_slices = []  # 中文注释：用于存放处理后的样本切片
                for candidate_tensor in level_candidates:  # 中文注释：遍历当前尺度的全部样本切片
                    if torch.is_tensor(candidate_tensor):  # 中文注释：若样本提供有效张量直接使用
                        normalized_slices.append(candidate_tensor)  # 中文注释：记录原始张量
                    else:  # 中文注释：当样本缺失该尺度时需要填充占位
                        normalized_slices.append(torch.zeros_like(reference_tensor))  # 中文注释：使用零张量补齐以保持对齐
                aggregated_levels.append(torch.stack(normalized_slices, dim=0))  # 中文注释：按batch维度堆叠得到尺度优先张量
            has_valid_level = any(torch.is_tensor(level_tensor) for level_tensor in aggregated_levels if level_tensor is not None)  # 中文注释：检测是否存在有效尺度张量
            return aggregated_levels if has_valid_level else None  # 中文注释：若存在有效张量则返回聚合结果否则返回None
        if isinstance(main_teacher_feature, (list, tuple)) and main_teacher_feature and isinstance(main_teacher_feature[0], (list, tuple)):  # 中文注释：当主教师特征以样本列表形式提供时执行尺度聚合
            main_teacher_feature = _aggregate_sample_level_features(main_teacher_feature)  # 中文注释：将样本级特征转换为尺度优先结构
        elif torch.is_tensor(main_teacher_feature):  # 中文注释：当主教师仅返回单个张量时统一包装为列表
            main_teacher_feature = [main_teacher_feature]  # 中文注释：保持与多尺度结构一致的接口
        elif isinstance(main_teacher_feature, (list, tuple)) and main_teacher_feature and all(torch.is_tensor(item) for item in main_teacher_feature):  # 中文注释：当主教师已提供尺度优先张量列表时直接使用
            main_teacher_feature = list(main_teacher_feature)  # 中文注释：显式转换为列表以便后续修改
        else:  # 中文注释：其余情况视为缺失主教师特征
            main_teacher_feature = None  # 中文注释：将无效结构置为None
        losses = dict()  # 中文注释：初始化最终损失字典
        feature_loss = dict()  # 中文注释：准备记录特征蒸馏相关损失
        feature_loss['pkd_feature_loss'] = 0  # 中文注释：初始化主教师特征蒸馏损失累积值
        if isinstance(main_teacher_feature, (list, tuple)) and main_teacher_feature:  # 中文注释：仅在主教师提供有效尺度张量时计算蒸馏
            valid_pairs = [(stu_feat, tea_feat) for stu_feat, tea_feat in zip(student_x, main_teacher_feature) if torch.is_tensor(tea_feat)]  # 中文注释：筛选出教师与学生都具备的尺度组合
            if valid_pairs:  # 中文注释：存在至少一个有效尺度时执行蒸馏
                normalization_factor = 1.0 / len(valid_pairs)  # 中文注释：计算均值系数避免层数不同造成权重偏移
                for student_feature, teacher_feature in valid_pairs:  # 中文注释：遍历每个有效尺度对
                    layer_loss = self.feature_loss(student_feature, teacher_feature)  # 中文注释：调用蒸馏损失函数评估该尺度差异
                    feature_loss['pkd_feature_loss'] = feature_loss['pkd_feature_loss'] + layer_loss * normalization_factor  # 中文注释：按均值系数累计主教师蒸馏损失
        cross_features = None  # 中文注释：初始化交叉教师特征容器
        cross_has_consistency = False  # 中文注释：标记交叉教师是否提供一致性信息
        if isinstance(cross_teacher_info, dict):  # 中文注释：交叉教师信息为字典时解析标准字段
            cross_features = cross_teacher_info.get('features', cross_teacher_info.get('main_feature'))  # 中文注释：读取交叉教师提供的特征集合并兼容旧键名
            consistency_block = cross_teacher_info.get('consistency')  # 中文注释：获取可能存在的一致性信息字典
            if consistency_block is not None:  # 中文注释：当一致性字段显式存在时视为具备信息
                cross_has_consistency = True  # 中文注释：记录存在一致性数据
            if cross_teacher_info.get('cls_consistency') is not None or cross_teacher_info.get('reg_consistency') is not None:  # 中文注释：检测顶层是否直接提供一致性项
                cross_has_consistency = True  # 中文注释：若存在显式一致性项则同样标记可用
        elif isinstance(cross_teacher_info, (list, tuple)):  # 中文注释：若交叉教师以序列形式直接提供特征
            cross_features = cross_teacher_info  # 中文注释：直接使用该特征序列
        aggregated_cross_features = None  # 中文注释：初始化聚合后的交叉教师特征容器
        if isinstance(cross_features, (list, tuple)) and cross_features and isinstance(cross_features[0], dict):  # 中文注释：当交叉教师按样本提供并区分传感器时执行重组
            sensor_feature_map = dict()  # 中文注释：创建传感器到样本特征的映射
            for sample_entry in cross_features:  # 中文注释：遍历每个样本的交叉特征字典
                for sensor_tag, sensor_feature in sample_entry.items():  # 中文注释：遍历该样本中的各个传感器特征
                    sensor_feature_map.setdefault(sensor_tag, []).append(sensor_feature)  # 中文注释：将特征按传感器累积
            aggregated_cross_features = dict()  # 中文注释：准备存放按传感器聚合后的尺度张量
            for sensor_tag, sensor_sample_list in sensor_feature_map.items():  # 中文注释：遍历每个传感器及其对应的样本特征
                aggregated_cross_features[sensor_tag] = _aggregate_sample_level_features(sensor_sample_list)  # 中文注释：对每个传感器执行尺度聚合
            cross_features = aggregated_cross_features  # 中文注释：将交叉特征替换为聚合后的结构
            if isinstance(cross_teacher_info, dict):  # 中文注释：在保持字典结构时同步更新原信息块
                cross_teacher_info['features'] = aggregated_cross_features  # 中文注释：写回聚合结果供后续流程复用
        elif isinstance(cross_features, (list, tuple)) and cross_features and isinstance(cross_features[0], (list, tuple)):  # 中文注释：当交叉特征为样本列表但未区分传感器时直接聚合
            aggregated_cross_features = _aggregate_sample_level_features(cross_features)  # 中文注释：对整体样本列表执行尺度聚合
            cross_features = aggregated_cross_features  # 中文注释：使用聚合后的列表进行后续蒸馏
        elif isinstance(cross_features, (list, tuple)) and all(torch.is_tensor(item) for item in cross_features):  # 中文注释：当交叉特征已是尺度优先列表时保持不变
            aggregated_cross_features = list(cross_features)  # 中文注释：显式转换为列表以便后续处理
            cross_features = aggregated_cross_features  # 中文注释：保持变量一致
        elif torch.is_tensor(cross_features):  # 中文注释：当交叉特征为单个张量时包装成列表
            aggregated_cross_features = [cross_features]  # 中文注释：构造单尺度列表供蒸馏使用
            cross_features = aggregated_cross_features  # 中文注释：更新交叉特征引用
        if isinstance(cross_features, dict):  # 中文注释：若交叉特征以传感器字典形式存在则额外生成合并列表
            sensor_level_values = [levels for levels in cross_features.values() if isinstance(levels, (list, tuple))]  # 中文注释：提取所有传感器的尺度列表
            if sensor_level_values:  # 中文注释：当至少存在一个传感器提供有效尺度时
                max_level_count = max(len(levels) for levels in sensor_level_values)  # 中文注释：计算最大尺度数量
                aggregated_cross_features = []  # 中文注释：初始化跨传感器合并的尺度容器
                for level_idx in range(max_level_count):  # 中文注释：逐尺度汇总所有传感器
                    level_slices = []  # 中文注释：收集当前尺度下的传感器张量
                    for levels in sensor_level_values:  # 中文注释：遍历各个传感器的尺度列表
                        if level_idx < len(levels):  # 中文注释：仅在传感器提供该尺度时参与合并
                            level_tensor = levels[level_idx]  # 中文注释：取出对应尺度张量
                            if torch.is_tensor(level_tensor):  # 中文注释：仅保留有效张量
                                level_slices.append(level_tensor)  # 中文注释：加入待拼接列表
                    aggregated_cross_features.append(torch.cat(level_slices, dim=0) if level_slices else None)  # 中文注释：按批维合并所有传感器的该尺度特征
        if aggregated_cross_features is None and isinstance(cross_features, (list, tuple)):  # 中文注释：若尚未构建跨传感器合并结果而交叉特征为列表
            aggregated_cross_features = list(cross_features)  # 中文注释：直接复制列表作为聚合输出
        if aggregated_cross_features is not None and self.cross_feature_loss is not None:  # 中文注释：具备交叉特征且配置了蒸馏权重时计算交叉蒸馏
            valid_cross_pairs = [(stu_feat, cross_feat) for stu_feat, cross_feat in zip(student_x, aggregated_cross_features) if torch.is_tensor(cross_feat)]  # 中文注释：筛选存在有效交叉教师张量的尺度
            if valid_cross_pairs:  # 中文注释：当存在有效尺度时执行交叉蒸馏
                normalization_factor = 1.0 / len(valid_cross_pairs)  # 中文注释：计算均值系数保持损失规模稳定
                cross_loss_value = 0  # 中文注释：初始化交叉蒸馏损失累计值
                for student_feature, cross_feature in valid_cross_pairs:  # 中文注释：遍历每个有效尺度对
                    layer_loss = self.cross_feature_loss(student_feature, cross_feature)  # 中文注释：计算该尺度的交叉蒸馏损失
                    cross_loss_value = cross_loss_value + layer_loss * normalization_factor  # 中文注释：按均值系数累积交叉蒸馏损失
                feature_loss['pkd_cross_feature_loss'] = cross_loss_value  # 中文注释：记录交叉教师特征蒸馏损失
        if cross_teacher_info is not None and cross_has_consistency:  # 中文注释：仅在确实存在一致性信息时才计算相应损失
            feature_loss.update(self.loss_cross_feature(cross_teacher_info))  # 中文注释：调用辅助函数计算分类与回归一致性损失
        if self.cross_consistency_debug_log:  # 中文注释：当启用调试日志时输出交叉一致性损失数值
            logger = MMLogger.get_current_instance()  # 中文注释：获取当前训练过程绑定的日志记录器实例
            if logger is not None:  # 中文注释：仅在日志记录器存在时才尝试写入信息
                if 'cross_cls_consistency_loss' in feature_loss:  # 中文注释：检查分类一致性损失是否已计算
                    logger.info(f"cross_cls_consistency_loss={feature_loss['cross_cls_consistency_loss'].item():.6f}")  # 中文注释：记录分类一致性损失的标量数值便于观测
                if 'cross_reg_consistency_loss' in feature_loss:  # 中文注释：检查回归一致性损失是否已计算
                    logger.info(f"cross_reg_consistency_loss={feature_loss['cross_reg_consistency_loss'].item():.6f}")  # 中文注释：记录回归一致性损失的标量数值便于观测
        losses.update(feature_loss)  # 中文注释：将所有特征相关损失合并到最终输出
        return losses  # 中文注释：返回损失字典供训练流程使用

    @staticmethod
    def _interp_schedule(schedule_cfg: Any, current_iter: int, default: float = 0.0) -> float:
        """中文注释：线性插值读取迭代调度的权重/阈值。"""
        if schedule_cfg is None:  # 中文注释：当未提供调度配置时直接返回默认值
            return default  # 中文注释：保持兼容的零值
        if isinstance(schedule_cfg, (int, float)):  # 中文注释：当提供常数时直接返回
            return float(schedule_cfg)  # 中文注释：转换为浮点便于统一计算
        if not isinstance(schedule_cfg, (list, tuple)) or not schedule_cfg:  # 中文注释：当结构非法或为空时返回默认
            return default  # 中文注释：避免异常
        sorted_points = sorted(list(schedule_cfg), key=lambda item: item[0])  # 中文注释：按迭代索引对调度点排序
        if current_iter <= sorted_points[0][0]:  # 中文注释：当当前迭代早于首个调度点时
            return float(sorted_points[0][1])  # 中文注释：直接返回起始权重
        if current_iter >= sorted_points[-1][0]:  # 中文注释：当当前迭代晚于最后调度点时
            return float(sorted_points[-1][1])  # 中文注释：返回末尾权重
        for (iter_a, val_a), (iter_b, val_b) in zip(sorted_points[:-1], sorted_points[1:]):  # 中文注释：遍历相邻调度区间
            if iter_a <= current_iter <= iter_b:  # 中文注释：定位当前迭代所在区间
                ratio = (current_iter - iter_a) / max(float(iter_b - iter_a), 1.0)  # 中文注释：计算线性插值系数避免除零
                return float(val_a + ratio * (val_b - val_a))  # 中文注释：返回插值结果
        return default  # 中文注释：兜底返回默认值避免逻辑遗漏

    def _unpack_diff_adaptation_output(self, semi_output: Any) -> Tuple[dict, Any]:
        """中文注释：解析diff自适应分支的返回值以获得损失与特征包。"""
        semi_loss = dict()  # 中文注释：初始化蒸馏损失字典
        diff_feature = None  # 中文注释：初始化教师特征占位符
        if isinstance(semi_output, tuple):  # 中文注释：当返回值为元组时按位置解析
            if len(semi_output) >= 1:  # 中文注释：确认存在损失输出
                semi_loss = semi_output[0] if isinstance(semi_output[0], dict) else dict()  # 中文注释：确保返回值为字典类型
            if len(semi_output) >= 2:  # 中文注释：当提供第二个元素时将其视为教师特征
                diff_feature = semi_output[1]  # 中文注释：记录教师特征包
        elif isinstance(semi_output, dict):  # 中文注释：当返回值本身为字典时直接视作损失
            semi_loss = semi_output  # 中文注释：将字典作为损失输出
        return semi_loss, diff_feature  # 中文注释：返回解析得到的损失与教师特征

    def _extract_pseudo_payload(self, semi_output: Any, diff_feature: Any) -> Optional[dict]:
        """中文注释：从返回结果与教师特征中提取伪标签与批次信息。"""
        pseudo_payload = None  # 中文注释：初始化负载为空
        candidate = None  # 中文注释：准备候选伪标签容器
        batch_info = None  # 中文注释：初始化批次信息
        if isinstance(semi_output, tuple) and len(semi_output) >= 3:  # 中文注释：当元组包含第三个元素时将其视为伪标签候选
            candidate = semi_output[2]  # 中文注释：读取潜在伪标签负载
        if candidate is None and isinstance(diff_feature, dict):  # 中文注释：若教师特征为字典尝试直接提取伪标签键
            candidate = diff_feature.get('pseudo_samples', diff_feature.get('pseudo_data_samples'))  # 中文注释：兼容旧键名
            batch_info = diff_feature.get('batch_info', batch_info)  # 中文注释：同时提取批次信息
        if candidate is None:  # 中文注释：若仍未找到伪标签则直接返回空
            return None  # 中文注释：无伪标签时不执行门控
        teacher_samples = None  # 中文注释：初始化教师伪标签列表
        student_samples = None  # 中文注释：初始化学生伪标签列表
        if isinstance(candidate, dict):  # 中文注释：伪标签负载为字典时按键提取
            teacher_samples = candidate.get('teacher_pseudo_samples', candidate.get('teacher'))  # 中文注释：兼容不同键名的教师伪标签
            student_samples = candidate.get('student_pseudo_samples', candidate.get('student'))  # 中文注释：兼容不同键名的学生伪标签
            batch_info = candidate.get('batch_info', batch_info)  # 中文注释：更新批次信息
        elif isinstance(candidate, (list, tuple)) and len(candidate) == 2:  # 中文注释：当候选为长度为2的序列时按顺序视作教师与学生伪标签
            teacher_samples, student_samples = candidate  # 中文注释：解包教师与学生伪标签
        elif isinstance(candidate, (list, tuple)):  # 中文注释：若仅提供学生伪标签列表则直接赋值
            student_samples = candidate  # 中文注释：默认序列代表学生伪标签
        if student_samples is None:  # 中文注释：没有学生伪标签则无法继续
            return None  # 中文注释：返回空
        pseudo_payload = {  # 中文注释：构造统一的伪标签负载字典
            'teacher_pseudo_samples': teacher_samples,  # 中文注释：教师侧伪标签列表
            'student_pseudo_samples': student_samples,  # 中文注释：学生侧伪标签列表
            'batch_info': batch_info  # 中文注释：批次信息
        }
        return pseudo_payload  # 中文注释：返回标准化的伪标签负载

    @staticmethod
    def _has_pseudo_loss(loss_dict: dict) -> bool:
        """中文注释：检测损失字典中是否已包含伪标签相关的损失项。"""
        if not isinstance(loss_dict, dict):  # 中文注释：非字典直接返回False
            return False  # 中文注释：缺少损失键时需要补算
        for key in loss_dict.keys():  # 中文注释：遍历所有损失键
            if 'pseudo' in key or 'unsup' in key:  # 中文注释：通过关键词判断是否已计算伪标签损失
                return True  # 中文注释：存在相关条目时返回True
        return False  # 中文注释：未发现伪标签损失则返回False

    @staticmethod
    def _get_cached_inv_feature(detector: Any) -> Optional[Tensor]:
        """中文注释：从检测器的SS-DC缓存中获取首层域不变特征。"""
        if detector is None or not hasattr(detector, 'ssdc_feature_cache'):  # 中文注释：无缓存时直接返回
            return None  # 中文注释：保持稳健
        inv_cache = detector.ssdc_feature_cache.get('noref', None)  # 中文注释：优先读取无参考分支缓存
        if inv_cache is None:  # 中文注释：若无参考缓存为空则尝试读取参考分支
            inv_cache = detector.ssdc_feature_cache.get('ref', None)  # 中文注释：回退到参考分支
        if inv_cache is None:  # 中文注释：依然缺失时返回None
            return None  # 中文注释：无法提供域不变特征
        inv_feature = inv_cache.get('inv', None)  # 中文注释：提取域不变特征列表
        if isinstance(inv_feature, (list, tuple)) and inv_feature:  # 中文注释：当缓存为序列时取首层特征图
            return inv_feature[0]  # 中文注释：返回首层域不变特征
        if torch.is_tensor(inv_feature):  # 中文注释：若直接为张量则原样返回
            return inv_feature  # 中文注释：返回张量形式的域不变特征
        return None  # 中文注释：其余情况返回空

    def _apply_pseudo_consistency_gate(self, pseudo_payload: dict, multi_batch_inputs: Dict[str, Tensor], current_iter: int) -> None:
        """中文注释：利用教师/学生域不变特征对伪标签执行余弦相似度门控。"""
        gate_cfg = self.ssdc_cfg.get('consistency_gate', None)  # 中文注释：读取门控配置
        if gate_cfg is None:  # 中文注释：未配置门控时直接返回
            return  # 中文注释：保持现有伪标签
        tau_schedule = gate_cfg.get('tau', gate_cfg) if isinstance(gate_cfg, dict) else gate_cfg  # 中文注释：兼容直接给定阈值或调度
        tau_value = self._interp_schedule(tau_schedule, current_iter, 0.0)  # 中文注释：计算当前迭代的阈值
        if tau_value <= 0:  # 中文注释：阈值无效时不做处理
            return  # 中文注释：保持伪标签
        decay_mode = bool(gate_cfg.get('decay', False)) if isinstance(gate_cfg, dict) else False  # 中文注释：读取是否采用权重衰减模式
        teacher_map = self._get_cached_inv_feature(getattr(self.model, 'teacher', None))  # 中文注释：提取教师域不变特征图
        student_map = self._get_cached_inv_feature(getattr(self.model, 'student', None))  # 中文注释：提取学生域不变特征图
        if teacher_map is None or student_map is None:  # 中文注释：缺失任一特征则无法计算相似度
            return  # 中文注释：直接退出
        teacher_inputs = multi_batch_inputs.get('unsup_teacher', None)  # 中文注释：读取教师分支输入
        student_inputs = multi_batch_inputs.get('unsup_student', None)  # 中文注释：读取学生分支输入
        if teacher_inputs is None or student_inputs is None:  # 中文注释：未提供输入张量无法获取空间尺度
            return  # 中文注释：结束门控
        if teacher_inputs.shape[-1] > 0:  # 中文注释：计算教师特征图与输入的尺度比
            teacher_scale = float(teacher_map.shape[-1]) / float(teacher_inputs.shape[-1])  # 中文注释：特征宽度除以输入宽度得到空间比例
        else:  # 中文注释：宽度异常时
            teacher_scale = 1.0  # 中文注释：使用单位比例避免除零
        if student_inputs.shape[-1] > 0:  # 中文注释：计算学生侧空间比例
            student_scale = float(student_map.shape[-1]) / float(student_inputs.shape[-1])  # 中文注释：特征宽度除以输入宽度
        else:  # 中文注释：宽度异常时
            student_scale = 1.0  # 中文注释：使用单位比例
        teacher_pseudo_samples = pseudo_payload.get('teacher_pseudo_samples', None)  # 中文注释：获取教师伪标签列表
        student_pseudo_samples = pseudo_payload.get('student_pseudo_samples', None)  # 中文注释：获取学生伪标签列表
        if student_pseudo_samples is None:  # 中文注释：没有学生伪标签无法执行过滤
            return  # 中文注释：直接返回
        filtered_samples = []  # 中文注释：准备存放过滤后的学生伪标签
        total_instances = 0  # 中文注释：统计总伪标签数量
        filtered_instances = 0  # 中文注释：统计被过滤或衰减的伪标签数量
        for batch_idx, student_sample in enumerate(student_pseudo_samples):  # 中文注释：逐个样本处理学生伪标签
            teacher_sample = None  # 中文注释：初始化对应的教师样本
            if isinstance(teacher_pseudo_samples, (list, tuple)) and batch_idx < len(teacher_pseudo_samples):  # 中文注释：教师伪标签可用时按索引取出
                teacher_sample = teacher_pseudo_samples[batch_idx]  # 中文注释：获取对应教师样本
            filtered_sample, removed_count, sample_total = self._gate_single_sample(
                teacher_sample, student_sample, teacher_map, student_map, batch_idx, teacher_scale, student_scale, tau_value, decay_mode)  # 中文注释：对单个样本执行门控并返回过滤结果
            filtered_samples.append(filtered_sample)  # 中文注释：收集处理后的学生伪标签
            total_instances += sample_total  # 中文注释：累计当前样本的伪标签数量
            filtered_instances += removed_count  # 中文注释：累计过滤或衰减的伪标签数量
        pseudo_payload['student_pseudo_samples'] = filtered_samples  # 中文注释：将过滤结果写回伪标签负载
        if total_instances > 0:  # 中文注释：仅在存在伪标签时记录过滤比例
            filter_ratio = float(filtered_instances) / float(total_instances)  # 中文注释：计算过滤比例
            logger = MMLogger.get_current_instance()  # 中文注释：获取当前日志记录器
            if logger is not None:  # 中文注释：日志记录器可用时写入过滤信息
                logger.info(f"consistency_gate_filtered_ratio={filter_ratio:.4f}")  # 中文注释：输出过滤比例便于监控

    def _gate_single_sample(self, teacher_sample: Any, student_sample: Any, teacher_map: Tensor, student_map: Tensor,
                             batch_idx: int, teacher_scale: float, student_scale: float, tau_value: float,
                             decay_mode: bool) -> Tuple[Any, int, int]:
        """中文注释：对单个样本的伪标签执行ROI对齐与余弦相似度过滤。"""
        if student_sample is None or not hasattr(student_sample, 'gt_instances') or student_sample.gt_instances is None:  # 中文注释：缺少学生伪标签时直接返回
            return student_sample, 0, 0  # 中文注释：无过滤发生
        student_boxes = student_sample.gt_instances.bboxes  # 中文注释：读取学生伪框
        if student_boxes.numel() == 0:  # 中文注释：当学生伪框为空时返回
            return student_sample, 0, 0  # 中文注释：无过滤
        teacher_boxes = None  # 中文注释：初始化教师伪框
        if teacher_sample is not None and hasattr(teacher_sample, 'gt_instances') and teacher_sample.gt_instances is not None:  # 中文注释：教师伪标签存在时读取
            teacher_boxes = getattr(teacher_sample.gt_instances, 'teacher_view_bboxes', teacher_sample.gt_instances.bboxes)  # 中文注释：优先使用教师视角坐标
        if teacher_boxes is None or teacher_boxes.numel() == 0:  # 中文注释：若教师伪框缺失则与学生框数量对齐使用自身
            teacher_boxes = student_boxes.to(device=teacher_map.device, dtype=teacher_map.dtype)  # 中文注释：直接使用学生框作为教师参考
        if teacher_boxes.shape[0] != student_boxes.shape[0]:  # 中文注释：当教师与学生伪框数量不一致时按IoU匹配
            iou_matrix = bbox_overlaps(teacher_boxes, student_boxes, mode='iou')  # 中文注释：计算IoU矩阵用于匹配
            best_iou, best_teacher_idx = iou_matrix.max(dim=0)  # 中文注释：为每个学生框找到最佳教师框
            valid_mask = best_iou > 0  # 中文注释：仅保留IoU大于0的匹配
            if not valid_mask.any():  # 中文注释：无有效匹配时直接返回原样本
                return student_sample, 0, int(student_boxes.shape[0])  # 中文注释：无过滤但记录总数
            matched_teacher_boxes = teacher_boxes[best_teacher_idx[valid_mask]]  # 中文注释：按匹配索引抽取教师框
            matched_student_boxes = student_boxes[valid_mask]  # 中文注释：同步抽取学生框
            matched_mask = valid_mask  # 中文注释：保留有效匹配掩码
        else:  # 中文注释：数量一致时直接按索引对应
            matched_teacher_boxes = teacher_boxes  # 中文注释：教师框按顺序对应
            matched_student_boxes = student_boxes  # 中文注释：学生框按顺序对应
            matched_mask = torch.ones(student_boxes.shape[0], dtype=torch.bool, device=student_boxes.device)  # 中文注释：掩码全为True
        teacher_batch_index = torch.full((matched_teacher_boxes.shape[0], 1), batch_idx, device=teacher_map.device, dtype=teacher_map.dtype)  # 中文注释：构造教师ROI批次索引
        teacher_roi = torch.cat([teacher_batch_index, matched_teacher_boxes.to(device=teacher_map.device, dtype=teacher_map.dtype)], dim=1)  # 中文注释：拼接批次索引与教师框
        student_batch_index = torch.full((matched_student_boxes.shape[0], 1), batch_idx, device=student_map.device, dtype=student_map.dtype)  # 中文注释：构造学生ROI批次索引
        student_roi = torch.cat([student_batch_index, matched_student_boxes.to(device=student_map.device, dtype=student_map.dtype)], dim=1)  # 中文注释：拼接批次索引与学生框
        pooled_teacher = roi_align(teacher_map, teacher_roi, output_size=1, spatial_scale=teacher_scale, aligned=True)  # 中文注释：在教师域不变特征上采样ROI
        pooled_student = roi_align(student_map, student_roi, output_size=1, spatial_scale=student_scale, aligned=True)  # 中文注释：在学生域不变特征上采样ROI
        cosine_scores = (F.normalize(pooled_teacher.flatten(1), dim=1) * F.normalize(pooled_student.flatten(1), dim=1)).sum(dim=1)  # 中文注释：计算归一化后向量的余弦相似度
        keep_mask = torch.zeros(student_boxes.shape[0], dtype=torch.bool, device=student_boxes.device)  # 中文注释：初始化全局保留掩码
        keep_mask[matched_mask] = cosine_scores >= tau_value  # 中文注释：仅对已匹配的框应用阈值判断
        removed_count = int((~keep_mask[matched_mask]).sum().item())  # 中文注释：统计当前样本被剔除的伪标签数量
        total_count = int(student_boxes.shape[0])  # 中文注释：记录当前样本伪标签总数
        if decay_mode:  # 中文注释：当启用衰减模式时不直接删除低相似度伪标签
            decay_scale = torch.ones_like(keep_mask, dtype=cosine_scores.dtype)  # 中文注释：初始化衰减系数
            safe_denom = max(float(tau_value), 1e-6)  # 中文注释：设置安全除数避免除零
            decay_values = torch.clamp(cosine_scores / safe_denom, max=1.0)  # 中文注释：将相似度归一化到[0,1]
            decay_scale[matched_mask] = decay_values  # 中文注释：仅对匹配框写入衰减权重
            if hasattr(student_sample.gt_instances, 'scores') and student_sample.gt_instances.scores is not None:  # 中文注释：当伪标签包含分类分数时直接缩放
                scores = student_sample.gt_instances.scores.to(decay_scale.device)  # 中文注释：确保分数张量与权重在同一设备
                student_sample.gt_instances.scores = scores * decay_scale[:scores.shape[0]]  # 中文注释：按衰减权重缩放分数
            else:  # 中文注释：若缺少分数字段则额外存储权重
                student_sample.gt_instances.score_factors = decay_scale.to(student_sample.gt_instances.bboxes.device)  # 中文注释：以score_factors形式提供衰减权重
        else:  # 中文注释：未启用衰减时直接剔除低相似度伪标签
            if not keep_mask.all():  # 中文注释：仅当存在需要剔除的伪标签时更新实例
                student_sample.gt_instances = student_sample.gt_instances[keep_mask]  # 中文注释：应用掩码过滤伪标签
        return student_sample, removed_count, total_count  # 中文注释：返回过滤后的样本及统计信息

    def _compute_ssdc_loss(self, current_iter: int) -> dict:
        """中文注释：汇总学生与教师SS-DC模块产生的额外损失。"""
        losses = dict()  # 中文注释：初始化损失容器
        if not self.ssdc_cfg:  # 中文注释：若未配置SS-DC则直接返回空字典
            return losses  # 中文注释：保持兼容
        if current_iter < self.burn_up_iters:  # 中文注释：在烧入阶段不引入SS-DC损失避免干扰基础学习
            return losses  # 中文注释：直接退出
        student_detector = getattr(self.model, 'student', None)  # 中文注释：获取学生检测器引用
        teacher_detector = getattr(self.model, 'teacher', None)  # 中文注释：获取教师检测器引用
        if student_detector is None or not hasattr(student_detector, 'ssdc_feature_cache'):  # 中文注释：若学生不支持SS-DC则退出
            return losses  # 中文注释：返回空损失
        student_cache = student_detector.ssdc_feature_cache.get('noref', None)  # 中文注释：读取学生的无参考分支缓存
        if student_cache is None:  # 中文注释：若无参考缓存缺失则尝试回退到参考分支
            student_cache = student_detector.ssdc_feature_cache.get('ref', None)  # 中文注释：使用参考分支缓存避免空指针
        teacher_cache = None  # 中文注释：初始化教师缓存占位符
        if teacher_detector is not None and hasattr(teacher_detector, 'ssdc_feature_cache'):  # 中文注释：若教师具备缓存则读取
            teacher_cache = teacher_detector.ssdc_feature_cache.get('noref', None)  # 中文注释：优先读取教师无参考分支缓存
            if teacher_cache is None:  # 中文注释：当教师无参考缓存为空时回退到参考分支
                teacher_cache = teacher_detector.ssdc_feature_cache.get('ref', None)  # 中文注释：尝试使用参考分支的缓存特征
        teacher_cache_override = self._build_teacher_cache_override()  # 中文注释：若扩散教师已提供域不变特征则构造伪教师缓存供耦合与一致性损失复用
        w_decouple = self._interp_schedule(self.ssdc_cfg.get('w_decouple', 0.0), current_iter, 0.0)  # 中文注释：插值获得解耦损失权重
        w_couple = self._interp_schedule(self.ssdc_cfg.get('w_couple', 0.0), current_iter, 0.0)  # 中文注释：插值获得耦合损失权重
        w_di = self._interp_schedule(self.ssdc_cfg.get('w_di_consistency', 0.0), current_iter, 0.0)  # 中文注释：获取域不变一致性权重
        if w_decouple > 0 and student_cache is not None and student_cache.get('inv') is not None and getattr(student_detector, 'loss_decouple', None) is not None:  # 中文注释：仅在权重与缓存合法时计算学生解耦损失
            student_decouple = student_detector.loss_decouple(  # 中文注释：调用学生解耦损失并显式保持梯度
                student_cache.get('raw'),  # 中文注释：传入学生原始特征序列
                student_cache.get('inv'),  # 中文注释：传入学生域不变特征序列
                student_cache.get('ds'),  # 中文注释：传入学生域特异特征序列
                getattr(student_detector, 'said_filter', None),  # 中文注释：传入学生SAID模块实例
                require_grad=True)  # 中文注释：显式开启梯度以支持学生反向
            student_decouple = rename_loss_dict('ssdc_student_decouple_', student_decouple)  # 中文注释：为日志加上前缀区分来源
            student_decouple = reweight_loss_dict(student_decouple, w_decouple)  # 中文注释：按调度权重缩放损失
            losses.update(student_decouple)  # 中文注释：合并学生解耦损失
        if w_decouple > 0 and teacher_cache is not None and teacher_cache.get('inv') is not None and getattr(teacher_detector, 'loss_decouple', None) is not None:  # 中文注释：当教师具备解耦特征时计算教师解耦损失
            def _detach_feature_block(block):  # 中文注释：定义辅助函数用于安全地detach不同结构的特征块
                if isinstance(block, (list, tuple)):  # 中文注释：若为序列则逐个元素处理
                    return [item.detach() if torch.is_tensor(item) else item for item in block]  # 中文注释：张量执行detach防止梯度累积，其他类型原样返回
                if torch.is_tensor(block):  # 中文注释：单个张量直接detach
                    return block.detach()  # 中文注释：切断计算图避免无效梯度
                return block  # 中文注释：非张量直接返回保持兼容

            with torch.no_grad():  # 中文注释：教师分支不反向传播仅用于稳定分解
                teacher_decouple = teacher_detector.loss_decouple(  # 中文注释：调用教师解耦损失并显式禁止梯度
                    _detach_feature_block(teacher_cache.get('raw')),  # 中文注释：传入已detach的教师原始特征
                    _detach_feature_block(teacher_cache.get('inv')),  # 中文注释：传入已detach的教师域不变特征
                    _detach_feature_block(teacher_cache.get('ds')),  # 中文注释：传入已detach的教师域特异特征
                    getattr(teacher_detector, 'said_filter', None),  # 中文注释：传入教师SAID模块实例
                    require_grad=False)  # 中文注释：显式禁止梯度构建
            teacher_decouple = rename_loss_dict('ssdc_teacher_decouple_', teacher_decouple)  # 中文注释：添加教师前缀便于区分
            teacher_decouple = reweight_loss_dict(teacher_decouple, w_decouple)  # 中文注释：应用相同调度权重
            losses.update(teacher_decouple)  # 中文注释：合并教师解耦损失
        if w_couple > 0 and student_cache is not None and (teacher_cache is not None or teacher_cache_override is not None):  # 中文注释：当耦合权重大于0且任意教师缓存可用时计算耦合损失
            student_coupled = student_cache.get('coupled')  # 中文注释：读取学生耦合后特征
            teacher_inv = teacher_cache.get('inv') if teacher_cache is not None else None  # 中文注释：优先使用EMA教师缓存中的域不变特征
            if teacher_inv is None and teacher_cache_override is not None:  # 中文注释：若EMA教师未提供域不变特征则回退到扩散教师缓存
                teacher_inv = teacher_cache_override.get('inv')  # 中文注释：直接使用扩散教师返回的域不变特征
            if student_coupled is not None and teacher_inv is not None and getattr(student_detector, 'loss_couple', None) is not None:  # 中文注释：确保必要组件存在
                # detached_teacher_inv = [feat.detach() if torch.is_tensor(feat) else feat for feat in teacher_inv]  # 中文注释：对教师域不变特征执行detach防止反向传播至EMA教师
                couple_loss = student_detector.loss_couple(student_coupled, teacher_inv, student_cache.get('stats', {}))  # 中文注释：计算耦合特征对齐损失
                couple_loss = rename_loss_dict('ssdc_couple_', couple_loss)  # 中文注释：添加日志前缀
                couple_loss = reweight_loss_dict(couple_loss, w_couple)  # 中文注释：应用耦合调度权重
                losses.update(couple_loss)  # 中文注释：合并耦合损失
        if w_di > 0 and student_cache is not None and (teacher_cache is not None or teacher_cache_override is not None) and student_cache.get('inv') is not None:  # 中文注释：当域不变特征齐全时计算一致性
            student_inv = student_cache.get('inv')  # 中文注释：获取学生域不变特征序列
            teacher_inv = teacher_cache.get('inv') if teacher_cache is not None else None  # 中文注释：优先读取EMA教师缓存中的域不变特征序列
            if teacher_inv is None and teacher_cache_override is not None:  # 中文注释：当EMA教师缺少域不变特征时使用扩散教师提供的伪缓存
                teacher_inv = teacher_cache_override.get('inv')  # 中文注释：取出扩散教师提供的域不变特征序列
            if teacher_inv is None:  # 中文注释：若仍然缺失域不变特征则无法执行一致性损失
                return losses  # 中文注释：直接返回当前累计的SS-DC损失
            if isinstance(student_inv, (list, tuple)) and isinstance(teacher_inv, (list, tuple)):
                valid_pairs = [(s, t) for s, t in zip(student_inv, teacher_inv) if torch.is_tensor(s) and torch.is_tensor(t)]  # 中文注释：筛选双方均为张量的层级
                if valid_pairs:  # 中文注释：存在有效层级才计算
                    mse_total = 0.0  # 中文注释：初始化MSE累积值
                    for stu_feat, tea_feat in valid_pairs:  # 中文注释：遍历每个层级
                        mse_total = mse_total + F.mse_loss(stu_feat, tea_feat.detach())  # 中文注释：计算每层MSE并累加，教师分支停止梯度
                    mse_total = mse_total / float(len(valid_pairs))  # 中文注释：对层级数量取均值保持尺度稳定
                    losses['ssdc_di_consistency_loss'] = mse_total * w_di  # 中文注释：记录域不变一致性损失并按权重缩放
        return losses  # 中文注释：返回SS-DC综合损失

    def _update_ssdc_teacher_override(self, diff_feature: Any) -> None:
        """中文注释：从扩散教师输出中提取域不变特征并缓存供SS-DC损失复用。"""
        candidate_inv = None  # 中文注释：初始化候选域不变特征占位符
        if isinstance(diff_feature, dict):  # 中文注释：仅在扩散教师返回字典时尝试读取域不变特征
            candidate_inv = diff_feature.get('main_teacher_inv')  # 中文注释：主教师域不变特征存放在main_teacher_inv键下
        if candidate_inv is None and isinstance(self.ssdc_cfg, dict):  # 中文注释：若扩散教师未返回则回退到SS-DC配置查找
            candidate_inv = self.ssdc_cfg.get('main_teacher_inv')  # 中文注释：允许外部组件通过配置直接写入域不变特征
        self._cached_main_teacher_inv = self._sanitize_teacher_inv(candidate_inv)  # 中文注释：对候选特征执行detach并缓存

    @staticmethod
    def _sanitize_teacher_inv(candidate_inv: Any) -> Optional[Tuple[Tensor, ...]]:
        """中文注释：将任意结构的域不变特征转换为detach后的张量元组。"""
        if candidate_inv is None:  # 中文注释：当缺失候选特征时直接返回None
            return None  # 中文注释：无需进一步处理
        if isinstance(candidate_inv, dict):  # 中文注释：兼容以字典形式传递的域不变特征
            candidate_inv = candidate_inv.get('inv', candidate_inv.get('main_teacher_inv'))  # 中文注释：优先读取inv字段否则回退main_teacher_inv
        if torch.is_tensor(candidate_inv):  # 中文注释：当输入为单个张量时
            return (candidate_inv.detach(),)  # 中文注释：detach后封装为元组返回
        if isinstance(candidate_inv, (list, tuple)):  # 中文注释：当输入为序列时
            sanitized = []  # 中文注释：初始化处理结果列表
            for entry in candidate_inv:  # 中文注释：遍历每个尺度特征
                if torch.is_tensor(entry):  # 中文注释：仅处理真实张量
                    sanitized.append(entry.detach())  # 中文注释：detach以断开扩散教师计算图
            if not sanitized:  # 中文注释：若序列中不存在有效张量
                return None  # 中文注释：直接返回None
            return tuple(sanitized)  # 中文注释：将处理后的张量列表转换为元组
        return None  # 中文注释：对无法解析的结构返回None

    def _build_teacher_cache_override(self) -> Optional[Dict[str, Any]]:
        """中文注释：将缓存的域不变特征封装为伪教师缓存供耦合/一致性损失调用。"""
        teacher_inv = getattr(self, '_cached_main_teacher_inv', None)  # 中文注释：读取上一次扩散教师返回的域不变特征
        if teacher_inv is None and isinstance(self.ssdc_cfg, dict):  # 中文注释：若缓存为空则尝试从配置读取
            teacher_inv = self._sanitize_teacher_inv(self.ssdc_cfg.get('main_teacher_inv'))  # 中文注释：对配置中的特征执行标准化
            if teacher_inv is not None:  # 中文注释：当从配置成功提取时同步写入缓存
                self._cached_main_teacher_inv = teacher_inv  # 中文注释：保留以避免重复解析
        if teacher_inv is None:  # 中文注释：若最终仍无域不变特征则返回None
            return None  # 中文注释：外层逻辑将回退到EMA教师缓存
        return {'inv': teacher_inv, 'raw': None, 'ds': None}  # 中文注释：构造仅包含域不变特征的伪教师缓存以符合LossCouple接口需求

    def loss_cross_feature(self, cross_teacher_info: Any) -> dict:
        """中文注释：计算交叉教师提供的分类与回归一致性损失。"""
        losses = dict()  # 中文注释：初始化一致性损失字典
        if not isinstance(cross_teacher_info, dict):  # 中文注释：当交叉教师信息非字典时直接返回空损失
            return losses  # 中文注释：保持接口兼容返回空结果
        if self.cross_cls_loss_weight > 0:  # 中文注释：仅在分类一致性权重大于0时计算
            cls_value = cross_teacher_info.get('cls_consistency')  # 中文注释：优先读取顶层分类一致性数值
            if cls_value is None:  # 中文注释：若顶层缺失则尝试从嵌套结构读取
                consistency_block = cross_teacher_info.get('consistency') or {}  # 中文注释：获取可能的嵌套一致性字典并在缺失时返回空字典
                cls_value = consistency_block.get('cls')  # 中文注释：从嵌套字典读取分类一致性项
            if cls_value is not None:  # 中文注释：只有在成功取得数值时才写入损失
                losses['cross_cls_consistency_loss'] = cls_value * self.cross_cls_loss_weight  # 中文注释：按配置权重缩放分类一致性损失
        if self.cross_reg_loss_weight > 0:  # 中文注释：仅在回归一致性权重大于0时计算
            reg_value = cross_teacher_info.get('reg_consistency')  # 中文注释：优先读取顶层回归一致性数值
            if reg_value is None:  # 中文注释：若未提供则尝试从嵌套结构读取
                consistency_block = cross_teacher_info.get('consistency') or {}  # 中文注释：获取可能的嵌套一致性字典并在缺失时返回空字典
                reg_value = consistency_block.get('reg')  # 中文注释：从嵌套字典读取回归一致性项
            if reg_value is not None:  # 中文注释：确保存在有效数值再写入结果
                losses['cross_reg_consistency_loss'] = reg_value * self.cross_reg_loss_weight  # 中文注释：按权重缩放回归一致性损失
        return losses  # 中文注释：返回交叉教师一致性损失字典



