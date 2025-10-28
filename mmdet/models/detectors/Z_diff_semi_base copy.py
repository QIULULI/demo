# Copyright (c) OpenMMLab. All rights reserved.  # 版权声明：来自 OpenMMLab
import copy  # 标准库 copy，用于深拷贝对象
from typing import Dict  # 类型注解：字典
from typing import Any, List, Optional, Union, Tuple  # 类型注解：通用 Any、列表、可选、联合、元组
import torch  # PyTorch 主包
import torch.nn as nn  # 神经网络模块
from torch import Tensor  # 张量类型别名

from mmdet.models.utils import (filter_gt_instances, rename_loss_dict, reweight_loss_dict)  # MMDet 工具：过滤/重命名/重加权损失
from mmdet.registry import MODELS  # MMDet 注册表，用于构建模型
from mmdet.structures import SampleList  # 数据样本列表类型
from mmdet.structures.bbox import bbox_project  # 盒子投影（坐标变换）
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig  # 配置类型别名
from .base import BaseDetector  # 本目录下的基础检测器基类

from pathlib import Path  # 路径工具（当前文件未直接使用）
from mmengine.config import Config  # MMEngine 配置类
from mmengine.runner import load_checkpoint  # 加载权重接口


@MODELS.register_module()  # 将该类注册到 MODELS，便于通过配置构建
class SemiBaseDiffDetector(BaseDetector):  # 定义半监督基础检测器，包含 teacher/student/diff 三分支
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
    """  # 英文文档字符串：说明半监督框架（teacher/student）及参数

    def __init__(self,  # 构造函数：实例化 student/teacher/diff_detector，并处理冻结策略
                 detector: ConfigType,  # 检测器配置（会构建 student/teacher）
                 diff_model: ConfigType,  # diff 分支配置（可选，指向另一个检测模型或权重）
                 semi_train_cfg: OptConfigType = None,  # 半监督训练配置
                 semi_test_cfg: OptConfigType = None,  # 半监督测试配置
                 data_preprocessor: OptConfigType = None,  # 数据预处理配置
                 init_cfg: OptMultiConfig = None) -> None:  # 初始化配置（权重等）
        super().__init__(  # 调用基类构造
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)

        self.student = MODELS.build(detector.deepcopy())  # 构建 student 模型（通过配置深拷贝）
        self.teacher = MODELS.build(detector.deepcopy())  # 构建 teacher 模型（参数初始化同 student）
        self.diff_detector = None  # 额外 diff 分支（可作为 teacher 的替代/增强）
        if diff_model.config:  # 若提供 diff 模型的配置文件路径
            teacher_config = Config.fromfile(diff_model.config)  # 从文件加载配置
            self.diff_detector = MODELS.build(teacher_config['model'])  # 按配置构建 diff_detector
            if diff_model.pretrained_model:  # 若提供预训练权重
                load_checkpoint(self.diff_detector, diff_model.pretrained_model, map_location='cpu', strict=True)  # 加载权重（严格匹配）
                self.diff_detector.cuda()  # 将 diff_detector 放到 GPU
                self.freeze(self.diff_detector)  # 冻结 diff_detector 参数，不参与训练
        if self.diff_detector is None:  # 若没有单独的 diff_detector
            self.diff_detector = self.student  # 回退使用 student 作为 diff 分支（保证接口一致）

        self.semi_train_cfg = semi_train_cfg  # 保存训练配置
        self.semi_test_cfg = semi_test_cfg  # 保存测试配置
        if self.semi_train_cfg.get('freeze_teacher', True) is True:  # 若配置要求冻结 teacher（默认 True）
            self.freeze(self.teacher)  # 冻结 teacher 参数，不参与反传

    @staticmethod
    def freeze(model: nn.Module):  # 静态方法：冻结任意 nn.Module
        """Freeze the model."""  # 文档：冻结模型
        model.eval()  # 设为 eval 模式（如 BN/Dropout 固定）
        for param in model.parameters():  # 遍历所有参数
            param.requires_grad = False  # 关闭梯度

    def cuda(self, device: Optional[str] = None) -> nn.Module:  # 重载 cuda：确保 diff_detector 也被迁移
        """Since teacher is registered as a plain object, it is necessary to
        put the teacher model to cuda when calling ``cuda`` function."""  # 注：本类中 teacher/diff_detector 需手动处理
        self.diff_detector.cuda(device=device)  # 将 diff_detector 放入指定设备
        return super().cuda(device=device)  # 调用基类 cuda（会处理其他子模块）

    def to(self, device: Optional[str] = None) -> nn.Module:  # 重载 to：同上，处理 diff_detector
        """Since teacher is registered as a plain object, it is necessary to
        put the teacher model to other device when calling ``to`` function."""  # 注：确保 diff_detector 迁移
        self.diff_detector.to(device=device)  # 迁移 diff_detector
        return super().to(device=device)  # 基类迁移

    def train(self, mode: bool = True) -> None:  # 重载 train：保持 diff_detector 处于 eval
        """Set the same train mode for teacher and student model."""  # 文档：训练模式设置
        self.diff_detector.train(False)  # 强制 diff_detector 不训练（即使外部调用 train(True)）
        super().train(mode)  # 其余模块按传入的 mode 设置（student 通常为 True）

    def __setattr__(self, name: str, value: Any) -> None:  # 重载属性设置：控制 diff_detector 的注册方式
        """Set attribute, i.e. self.name = value

        This reloading prevent the teacher model from being registered as a
        nn.Module. The teacher module is registered as a plain object, so that
        the teacher parameters will not show up when calling
        ``self.parameters``, ``self.modules``, ``self.children`` methods.
        """  # 文档：避免将某些子模型注册为 nn.Module（从而不计入参数/优化器）
        if name == 'diff_detector':  # 对 diff_detector 特殊处理
            object.__setattr__(self, name, value)  # 直接用对象的 __setattr__（不走 nn.Module 的注册逻辑）
        else:
            super().__setattr__(name, value)  # 其他属性按常规处理（可能注册为子模块）

    def loss(self, multi_batch_inputs: Dict[str, Tensor],
             multi_batch_data_samples: Dict[str, SampleList]) -> dict:  # 计算半监督损失：有监督 + 伪标签
        """Calculate losses from multi-branch inputs and data samples.

        Args:
            multi_batch_inputs (Dict[str, Tensor]): The dict of multi-branch
                input images, each value with shape (N, C, H, W).
                Each value should usually be mean centered and std scaled.
            multi_batch_data_samples (Dict[str, List[:obj:`DetDataSample`]]):
                The dict of multi-branch data samples.

        Returns:
            dict: A dictionary of loss components
        """  # 文档：输入包含 sup、unsup_teacher、unsup_student 三个分支

        losses = dict()  # 初始化损失字典
        losses.update(**self.loss_by_gt_instances(  # 计算有监督分支损失
            multi_batch_inputs['sup'], multi_batch_data_samples['sup']))

        origin_pseudo_data_samples, batch_info = self.get_pseudo_instances(  # 调 teacher 产生伪标签（未投影）
            multi_batch_inputs['unsup_teacher'], multi_batch_data_samples['unsup_teacher'])

        multi_batch_data_samples['unsup_student'] = self.project_pseudo_instances(  # 将伪标签从 teacher 视角投影到 student 视角
            origin_pseudo_data_samples, multi_batch_data_samples['unsup_student'])

        losses.update(**self.loss_by_pseudo_instances(multi_batch_inputs['unsup_student'],  # 计算无监督（伪标签）损失
                                                      multi_batch_data_samples['unsup_student'], batch_info))
        return losses  # 返回合并后的损失
    
    def loss_diff_adaptation(self, multi_batch_inputs: Dict[str, Tensor],
                             multi_batch_data_samples: Dict[str, SampleList]) -> dict:  # 计算含 diff 分支的蒸馏/自适应损失
        """Calculate losses from multi-branch inputs and data samples.

        Args:
            multi_batch_inputs (Dict[str, Tensor]): The dict of multi-branch
                input images, each value with shape (N, C, H, W).
                Each value should usually be mean centered and std scaled.
            multi_batch_data_samples (Dict[str, List[:obj:`DetDataSample`]]):
                The dict of multi-branch data samples.

        Returns:
            dict: A dictionary of loss components
        """  # 文档：与 loss 类似，但会返回 diff 特征以便外部使用

        losses = dict()  # 初始化损失
        losses.update(**self.loss_by_gt_instances(  # 有监督损失
            multi_batch_inputs['sup'], multi_batch_data_samples['sup']))
        origin_pseudo_data_samples, batch_info, diff_feature = self.get_pseudo_instances_diff(  # 用 diff_detector 生成伪标签 + 特征
            multi_batch_inputs['unsup_teacher'], multi_batch_data_samples['unsup_teacher'])
        multi_batch_data_samples['unsup_student'] = self.project_pseudo_instances(  # 伪标签坐标投影到 student
            origin_pseudo_data_samples, multi_batch_data_samples['unsup_student'])

        losses.update(**self.loss_by_pseudo_instances(multi_batch_inputs['unsup_student'],  # 无监督损失
                                                      multi_batch_data_samples['unsup_student'], batch_info))
        return losses, diff_feature  # 额外返回 diff_feature 供外部蒸馏/对齐

    def loss_by_gt_instances(self, batch_inputs: Tensor,
                             batch_data_samples: SampleList) -> dict:  # 仅用 GT 计算损失（供 sup 分支）
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
        """  # 文档：封装 student.loss 并加权重/重命名

        losses = self.student.loss(batch_inputs, batch_data_samples)  # 调用 student 的损失计算
        sup_weight = self.semi_train_cfg.get('sup_weight', 1.)  # 读取有监督损失权重（默认 1.0）
        return rename_loss_dict('sup_', reweight_loss_dict(losses, sup_weight))  # 前缀重命名 + 加权

    def loss_by_pseudo_instances(self,
                                 batch_inputs: Tensor,
                                 batch_data_samples: SampleList,
                                 batch_info: Optional[dict] = None) -> dict:  # 用伪标签计算无监督损失
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
        """  # 文档：将低分伪框过滤，并按数量决定权重

        batch_data_samples = filter_gt_instances(  # 按置信度阈值过滤伪标签实例
            batch_data_samples, score_thr=self.semi_train_cfg.cls_pseudo_thr)
        losses = self.student.loss(batch_inputs, batch_data_samples)  # student 基于伪标签计算损失
        pseudo_instances_num = sum([  # 统计批次内伪实例数量（用于是否置零权重）
            len(data_samples.gt_instances)
            for data_samples in batch_data_samples
        ])
        unsup_weight = self.semi_train_cfg.get(  # 若无伪实例则置权重为 0，否则取配置中的权重（默认 1.0）
            'unsup_weight', 1.) if pseudo_instances_num > 0 else 0.
        return rename_loss_dict('unsup_',  # 添加前缀标识无监督损失
                                reweight_loss_dict(losses, unsup_weight))  # 应用权重

    @torch.no_grad()  # 关闭梯度：生成伪标签不需反传
    def get_pseudo_instances(
            self, batch_inputs: Tensor, batch_data_samples: SampleList
    ) -> Tuple[SampleList, Optional[dict]]:  # 使用 teacher 预测伪标签，并投影回原图
        """Get pseudo instances from teacher model."""  # 文档：teacher 推理得到伪实例
        self.teacher.eval()  # 确保 teacher 处于 eval 模式
        results_list = self.teacher.predict(  # 直接使用 predict（返回 DetDataSample 列表）
            batch_inputs, batch_data_samples, rescale=False)
        batch_info = {}  # 可扩展的批次信息（此处占位）
        for data_samples, results in zip(batch_data_samples, results_list):  # 遍历样本与对应结果
            data_samples.gt_instances = results.pred_instances  # 将预测结果作为伪标签写回 gt_instances
            data_samples.gt_instances.bboxes = bbox_project(  # 将 box 从 teacher 视角投影到原图坐标（逆单应性矩阵）
                data_samples.gt_instances.bboxes,
                torch.from_numpy(data_samples.homography_matrix).inverse().to(
                    self.data_preprocessor.device), data_samples.ori_shape)
        return batch_data_samples, batch_info  # 返回伪标签样本与批次信息
    
    @torch.no_grad()  # 关闭梯度：diff 分支推理仅用于伪标签/特征
    def get_pseudo_instances_diff(
            self, batch_inputs: Tensor, batch_data_samples: SampleList
    ) -> Tuple[SampleList, Optional[dict]]:  # 使用 diff_detector 预测伪标签并返回特征
        """Get pseudo instances from teacher model."""  # 文档：与上类似，但走 diff_detector
        self.diff_detector.eval()  # 设置 diff_detector 为 eval
        results_list, diff_feature = self.diff_detector.predict(  # 运行预测，并要求返回中间特征
            batch_inputs, batch_data_samples, rescale=False, return_feature=True)
        batch_info = {}  # 占位的批次信息
        for data_samples, results in zip(batch_data_samples, results_list):  # 遍历结果
            data_samples.gt_instances = results.pred_instances  # 写入伪标签
            data_samples.gt_instances.bboxes = bbox_project(  # 投影回原图坐标
                data_samples.gt_instances.bboxes,
                torch.from_numpy(data_samples.homography_matrix).inverse().to(
                    self.data_preprocessor.device), data_samples.ori_shape)
        return batch_data_samples, batch_info, diff_feature  # 返回伪标签、批次信息以及 diff 特征

    def project_pseudo_instances(self, batch_pseudo_instances: SampleList,
                                 batch_data_samples: SampleList) -> SampleList:  # 将伪标签从原图再投影到 student 视角
        """Project pseudo instances."""  # 文档：伪框坐标系转换
        for pseudo_instances, data_samples in zip(batch_pseudo_instances,
                                                  batch_data_samples):  # 遍历对应的伪标签与目标样本
            data_samples.gt_instances = copy.deepcopy(  # 深拷贝伪标签，避免原对象被修改
                pseudo_instances.gt_instances)
            data_samples.gt_instances.bboxes = bbox_project(  # 将 box 由原图投影到 student 图像坐标
                data_samples.gt_instances.bboxes,
                torch.tensor(data_samples.homography_matrix).to(
                    self.data_preprocessor.device), data_samples.img_shape)
        wh_thr = self.semi_train_cfg.get('min_pseudo_bbox_wh', (1e-2, 1e-2))  # 伪框的最小宽高阈值（过小剔除）
        return filter_gt_instances(batch_data_samples, wh_thr=wh_thr)  # 应用尺寸过滤并返回

    def predict(self, batch_inputs: Tensor,
                batch_data_samples: SampleList) -> SampleList:  # 对外预测接口：可选择 teacher/student/diff 分支
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
        """  # 文档：标准 MMDet 预测输出

        if self.semi_test_cfg.get('predict_on', 'teacher') == 'teacher':  # 若配置指定在 teacher 上预测（默认）
            return self.teacher(  # 直接调用 teacher 的前向（predict 模式）
                batch_inputs, batch_data_samples, mode='predict')
        elif self.semi_test_cfg.get('predict_on', 'teacher') == 'student':  # 若选择 student
            return self.student(
                batch_inputs, batch_data_samples, mode='predict')
        else:  # 否则走 diff_detector
            return self.diff_detector(
                batch_inputs, batch_data_samples, mode='predict')

    def _forward(self, batch_inputs: Tensor,
                 batch_data_samples: SampleList) -> SampleList:  # 纯前向（tensor 模式），无后处理
        """Network forward process. Usually includes backbone, neck and head
        forward without any post-processing.

        Args:
            batch_inputs (Tensor): Inputs with shape (N, C, H, W).

        Returns:
            tuple: A tuple of features from ``rpn_head`` and ``roi_head``
            forward.
        """  # 文档：用于分析/蒸馏等场景的中间张量

        if self.semi_test_cfg.get('forward_on', 'teacher') == 'teacher':  # 选择 teacher 的前向
            return self.teacher(batch_inputs, batch_data_samples, mode='tensor')
        elif self.semi_test_cfg.get('forward_on', 'teacher') == 'student':  # 选择 student 的前向
            return self.student(batch_inputs, batch_data_samples, mode='tensor')
        elif self.semi_test_cfg.get('forward_on', 'teacher') == 'diff_detector':  # 选择 diff_detector 的前向
            return self.diff_detector(batch_inputs, batch_data_samples, mode='tensor')

    def extract_feat(self, batch_inputs: Tensor) -> Tuple[Tensor]:  # 仅提取 backbone/neck 特征
        """Extract features.

        Args:
            batch_inputs (Tensor): Image tensor with shape (N, C, H ,W).

        Returns:
            tuple[Tensor]: Multi-level features that may have
            different resolutions.
        """  # 文档：多尺度特征输出（供 RPN/ROI 使用）

        if self.semi_test_cfg.get('extract_feat_on', 'teacher') == 'teacher':  # 从 teacher 提特征
            return self.teacher.extract_feat(batch_inputs)
        elif self.semi_test_cfg.get('extract_feat_on', 'teacher') == 'student':  # 从 student 提特征
            return self.student.extract_feat(batch_inputs)
        elif self.semi_test_cfg.get('extract_feat_on', 'teacher') == 'diff_detector':  # 从 diff_detector 提特征
            return self.diff_detector.extract_feat(batch_inputs)

    def _load_from_state_dict(self, state_dict: dict, prefix: str,
                              local_metadata: dict, strict: bool,
                              missing_keys: Union[List[str], str],
                              unexpected_keys: Union[List[str], str],
                              error_msgs: Union[List[str], str]) -> None:  # 加载权重时为参数名自动加上 teacher/student 前缀
        """Add teacher and student prefixes to model parameter names."""  # 文档：兼容无前缀的权重
        if not any([  # 若权重键中既没有 'student' 也没有 'teacher'
            'student' in key or 'teacher' in key
            for key in state_dict.keys()
        ]):
            keys = list(state_dict.keys())  # 保存原始键列表
            state_dict.update({'teacher.' + k: state_dict[k] for k in keys})  # 复制一份加 teacher. 前缀
            state_dict.update({'student.' + k: state_dict[k] for k in keys})  # 复制一份加 student. 前缀
            for k in keys:  # 删除无前缀的原键
                state_dict.pop(k)
        return super()._load_from_state_dict(  # 调用基类加载（此时已经有前缀）
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )  # 结束：完成自定义加载逻辑并回退到父类实现
