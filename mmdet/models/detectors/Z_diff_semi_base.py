# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import Dict
from typing import Any, List, Optional, Union, Tuple
import torch  # 中文注释：导入PyTorch基础库用于张量运算
import torch.nn as nn  # 中文注释：导入神经网络模块以便构建模型与操作层
from torch import Tensor  # 中文注释：从PyTorch中显式导入Tensor类型便于类型标注

from mmdet.models.utils import (filter_gt_instances, rename_loss_dict, reweight_loss_dict)
from mmdet.registry import MODELS
from mmdet.structures import SampleList
from mmdet.structures.bbox import bbox_project, bbox_overlaps  # 中文注释：导入边界框投影与IoU计算函数用于伪标签与匹配
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

        self.student = MODELS.build(detector.deepcopy())  # 构建学生模型副本用于梯度更新
        self.teacher = MODELS.build(detector.deepcopy())  # 构建教师模型副本用于生成伪标签
        self.diff_detector = None  # 初始化扩散教师占位符
        self.diff_detectors = dict()  # 创建扩散教师存储字典
        self.diff_teacher_bank = self.diff_detectors  # 将教师资源库引用到同一字典以便统一维护
        self.active_diff_key = None  # 记录当前激活的扩散教师标识
        normalized_teachers = self._normalize_diff_teacher_configs(diff_model)  # 调用解析函数获取标准化教师配置
        main_teacher_hint = self._fetch_config_value(diff_model, 'main_teacher') if diff_model is not None else None  # 读取主教师提示
        for teacher_key, teacher_meta in normalized_teachers.items():  # 遍历标准化后的教师配置字典
            teacher_model_cfg = None  # 初始化教师模型结构占位符
            teacher_config_path = teacher_meta.get('config')  # 获取教师独立配置文件路径
            raw_config_entry = teacher_meta.get('raw')  # 保留原始配置对象便于回退
            if teacher_config_path:  # 若提供独立配置文件
                teacher_config_obj = Config.fromfile(teacher_config_path)  # 载入配置文件生成Config对象
                teacher_model_cfg = teacher_config_obj['model']  # 提取模型结构配置
            elif isinstance(raw_config_entry, dict) and 'model' in raw_config_entry:  # 若原始配置直接给出模型字段
                teacher_model_cfg = raw_config_entry['model']  # 直接使用内嵌模型配置
            else:  # 当未提供专用配置时回退到学生检测器结构
                teacher_model_cfg = detector.deepcopy()  # 复制学生模型配置以保持一致结构
            teacher_instance = MODELS.build(copy.deepcopy(teacher_model_cfg))  # 根据模型配置构建扩散教师实例
            pretrained_path = teacher_meta.get('pretrained_model')  # 读取预训练权重路径
            if pretrained_path:  # 若存在对应权重
                load_checkpoint(teacher_instance, pretrained_path, map_location='cpu', strict=True)  # 加载预训练权重确保参数一致
            teacher_instance.cuda()  # 将教师模型迁移至GPU以加速推理
            self.freeze(teacher_instance)  # 冻结教师模型参数避免训练阶段被更新
            self.diff_detectors[teacher_key] = teacher_instance  # 将实例化教师写入字典并以标识符索引
            alias_set = teacher_meta.get('aliases', set())  # 取出可选别名集合用于主教师匹配
            if self.active_diff_key is None and main_teacher_hint is not None and (main_teacher_hint == teacher_key or main_teacher_hint in alias_set):  # 若尚未确定主教师且提示匹配当前教师
                self.active_diff_key = teacher_key  # 将当前教师设为主教师
        if self.diff_detectors:  # 若成功加载至少一名扩散教师
            if self.active_diff_key is None:  # 若尚未确定主教师
                self.active_diff_key = next(iter(self.diff_detectors.keys()))  # 默认选择字典中的第一名教师
            self.diff_detector = self.diff_detectors[self.active_diff_key]  # 将当前激活教师指向主教师实例
        elif diff_model is not None and self._fetch_config_value(diff_model, 'config') is not None:  # 若未能解析教师但存在旧式配置字段
            teacher_config = Config.fromfile(self._fetch_config_value(diff_model, 'config'))  # 载入旧式配置文件
            self.diff_detector = MODELS.build(teacher_config['model'])  # 构建单一扩散教师实例
            pretrained_path = self._fetch_config_value(diff_model, 'pretrained_model') or self._fetch_config_value(diff_model, 'pretrained')  # 获取旧式权重字段
            if pretrained_path:  # 若存在旧式权重
                load_checkpoint(self.diff_detector, pretrained_path, map_location='cpu', strict=True)  # 加载旧式教师权重
            self.diff_detector.cuda()  # 将旧式教师迁移至GPU
            self.freeze(self.diff_detector)  # 冻结旧式教师参数
            self.diff_detectors['default'] = self.diff_detector  # 将旧式教师登记到字典中
            self.active_diff_key = 'default'  # 记录当前主教师标识
        else:  # 当未提供任何有效教师配置时
            self.diff_detector = self.student  # 使用学生模型作为退路教师
            self.diff_detectors['student'] = self.student  # 将学生登记到扩散教师字典
            self.active_diff_key = 'student'  # 标记学生为当前主教师
        self.diff_teacher_bank = self.diff_detectors  # 确保教师资源库最终指向完整教师字典

        self.semi_train_cfg = semi_train_cfg  # 中文注释：记录半监督训练阶段的配置字典以便各类损失读取参数
        cross_cfg_candidate = self._fetch_config_value(self.semi_train_cfg, 'cross_consistency_cfg', {}) if self.semi_train_cfg is not None else {}  # 中文注释：尝试从训练配置中抽取交叉一致性相关的子配置
        self.cross_consistency_cfg = cross_cfg_candidate if cross_cfg_candidate is not None else {}  # 中文注释：若未设置则回退为空字典确保后续访问安全
        self.cross_consistency_iou_thr = float(self.cross_consistency_cfg.get('iou_thr', 0.5))  # 中文注释：读取用于匹配候选框的IoU阈值默认0.5
        self.cross_consistency_min_conf = float(self.cross_consistency_cfg.get('min_conf', 0.0))  # 中文注释：读取主教师候选框的最小置信度筛选阈值默认0
        self.cross_consistency_max_pairs = self.cross_consistency_cfg.get('max_pairs', None)  # 中文注释：读取每幅图像允许用于一致性计算的最大匹配对数量
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
            if not isinstance(current_cfg, dict) and not self._is_single_diff_config(current_cfg):  # 若当前元素既非字典也不含关键字段
                continue  # 直接跳过无效条目以避免误解析
            sensor_key = self._fetch_config_value(current_cfg, 'sensor', fallback_sensor)  # 优先从配置中读取传感器标签
            if sensor_key is None:  # 若仍未获得标签
                sensor_key = self._fetch_config_value(current_cfg, 'name', fallback_sensor)  # 回退使用名称字段作为标识
            if sensor_key is None:  # 若依旧无法确定标识
                sensor_key = 'default'  # 使用default作为占位标签兼容旧逻辑
            config_path = self._fetch_config_value(current_cfg, 'config')  # 提取配置文件路径
            pretrained_path = self._fetch_config_value(current_cfg, 'pretrained_model')  # 提取预训练权重路径
            if pretrained_path is None:  # 若未使用新字段名
                pretrained_path = self._fetch_config_value(current_cfg, 'pretrained')  # 兼容旧字段名称
            if pretrained_path is None:  # 若仍未找到预训练字段
                pretrained_path = self._fetch_config_value(current_cfg, 'checkpoint')  # 兼容checkpoint字段
            alias_candidates = {sensor_key}  # 初始化别名集合并包含主键
            alias_sensor = self._fetch_config_value(current_cfg, 'sensor')  # 再次读取传感器字段便于补充别名
            if alias_sensor is not None:  # 若存在传感器字段
                alias_candidates.add(alias_sensor)  # 将传感器字段加入别名集合
            alias_name = self._fetch_config_value(current_cfg, 'name')  # 读取名称字段
            if alias_name is not None:  # 若存在名称字段
                alias_candidates.add(alias_name)  # 将名称字段加入别名集合
            if fallback_sensor is not None:  # 若存在备用标签
                alias_candidates.add(fallback_sensor)  # 将备用标签加入别名集合
            normalized_configs[sensor_key] = {  # 汇总解析结果
                'config': config_path,  # 存储配置文件路径
                'pretrained_model': pretrained_path,  # 存储预训练权重路径
                'raw': current_cfg,  # 保留原始配置项便于后续构建
                'aliases': alias_candidates  # 记录全部可匹配的别名集合
            }
        return normalized_configs  # 返回标准化后的配置字典

    @staticmethod
    def _binary_kl_div(p: Tensor, q: Tensor, eps: float = 1e-6) -> Tensor:
        """中文注释：计算二元概率分布之间的KL散度以衡量分类一致性。"""
        p = p.clamp(min=eps, max=1 - eps)  # 中文注释：限制主教师概率范围避免对数计算发生数值溢出
        q = q.clamp(min=eps, max=1 - eps)  # 中文注释：限制同伴教师概率范围保持运算稳定
        return p * torch.log(p / q) + (1 - p) * torch.log((1 - p) / (1 - q))  # 中文注释：按照KL散度定义累加正类与负类的贡献

    def cuda(self, device: Optional[str] = None) -> nn.Module:  # 重载cuda方法，确保所有扩散教师迁移到指定设备
        """Since teacher is registered as a plain object, it is necessary to
        put the teacher model to cuda when calling ``cuda`` function."""
        for detector_name, detector_module in getattr(self, 'diff_detectors', {}).items():  # 遍历所有扩散教师模块
            detector_module.cuda(device=device)  # 将每个扩散教师迁移到指定设备
        return super().cuda(device=device)

    def to(self, device: Optional[str] = None) -> nn.Module:  # 重载to方法，兼容多教师场景下的设备转换
        """Since teacher is registered as a plain object, it is necessary to
        put the teacher model to other device when calling ``to`` function."""
        for detector_name, detector_module in getattr(self, 'diff_detectors', {}).items():  # 遍历全部扩散教师
            detector_module.to(device=device)  # 将每个扩散教师迁移到目标设备
        return super().to(device=device)

    def train(self, mode: bool = True) -> None:  # 重载train方法，统一保持教师处于评估模式
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
        if not self.diff_teacher_bank or len(self.diff_teacher_bank) <= 1:  # 若未启用多教师或仅存在单一教师则沿用默认逻辑
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
            if isinstance(diff_feature, (list, tuple)):  # 若返回的是按层排列的特征序列
                primary_features = list(diff_feature)  # 转换为列表以便后续统一封装
            else:  # 若返回单一张量
                primary_features = [diff_feature for _ in range(len(batch_data_samples))]  # 为每个样本复制一份引用
            distill_feature = {  # 构造主教师输出结构
                'main_teacher': primary_features  # 主教师特征列表
            }
            return batch_data_samples, batch_info, distill_feature  # 返回伪标签、批信息与特征

        for teacher_model in self.diff_teacher_bank.values():  # 若存在多教师，先统一切换至评估模式
            teacher_model.eval()  # 设置教师为评估模式，确保推理一致性
        batch_info = {}  # 初始化批处理信息占位符
        sample_count = len(batch_data_samples)  # 记录样本数量，便于构造输出容器
        primary_feature_list: List[Any] = [None] * sample_count  # 为主教师特征预留位置
        peer_feature_list: List[Dict[str, Any]] = [dict() for _ in range(sample_count)]  # 初始化同伴特征字典列表
        peer_prediction_list: List[Dict[str, Any]] = [dict() for _ in range(sample_count)]  # 初始化同伴预测字典列表
        device_for_stats = batch_inputs.device  # 中文注释：记录输入张量所在设备以保证统计量与主干数据保持一致
        dtype_for_stats = batch_inputs.dtype  # 中文注释：记录输入张量的数据类型以便创建数值统计张量
        sample_cls_stats = [{'sum': torch.zeros((), device=device_for_stats, dtype=dtype_for_stats), 'count': 0} for _ in range(sample_count)]  # 中文注释：为每个样本初始化分类一致性累加器与计数器
        sample_reg_stats = [{'sum': torch.zeros((), device=device_for_stats, dtype=dtype_for_stats), 'count': 0} for _ in range(sample_count)]  # 中文注释：为每个样本初始化回归一致性累加器与计数器
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
                if isinstance(primary_feature, (list, tuple)) and primary_feature and all(torch.is_tensor(level_feat) for level_feat in primary_feature):  # 中文注释：当主教师以多尺度张量列表返回时按样本提取每个尺度
                    primary_feature_list[sample_idx] = [level_feat[local_idx] for level_feat in primary_feature]  # 中文注释：逐尺度切片并构成当前样本的特征列表
                elif torch.is_tensor(primary_feature):  # 中文注释：当主教师特征为单个张量时直接按样本索引切片
                    primary_feature_list[sample_idx] = [primary_feature[local_idx]]  # 中文注释：将单尺度特征包装成列表保持后续接口一致
                elif isinstance(primary_feature, (list, tuple)) and primary_feature:  # 中文注释：当主教师返回的结构为样本列表等其他序列时兜底处理
                    extracted_entry = primary_feature[local_idx]  # 中文注释：提取当前样本对应的条目
                    if isinstance(extracted_entry, (list, tuple)):  # 中文注释：若条目本身为多尺度序列则直接转为列表
                        primary_feature_list[sample_idx] = list(extracted_entry)  # 中文注释：保持原有尺度排列
                    elif torch.is_tensor(extracted_entry):  # 中文注释：若条目为张量则统一包装成列表
                        primary_feature_list[sample_idx] = [extracted_entry]  # 中文注释：确保后续处理始终基于列表结构
                    else:  # 中文注释：对于无法识别的类型直接存储原始对象以便后续兼容逻辑处理
                        primary_feature_list[sample_idx] = extracted_entry  # 中文注释：保留原始数据避免信息丢失
                else:  # 中文注释：当主教师特征结构异常时直接透传以防止流程中断
                    primary_feature_list[sample_idx] = primary_feature  # 中文注释：在未知结构下保留原引用供后续判断
            for peer_sensor, peer_teacher in self.diff_teacher_bank.items():  # 遍历其它教师以获取交叉特征
                if peer_sensor == sensor_tag:  # 跳过主教师自身
                    continue  # 仅处理同伴教师
                peer_inputs = batch_inputs[sample_indices]  # 使用相同图像张量作为输入
                peer_samples = [copy.deepcopy(batch_data_samples[i]) for i in sample_indices]  # 复制样本以避免状态污染
                peer_results, peer_feature = peer_teacher.predict(  # 调用同伴教师获取预测与特征
                    peer_inputs, peer_samples, rescale=False, return_feature=True)
                for local_idx, sample_idx in enumerate(sample_indices):  # 将同伴特征写入对应位置
                    if isinstance(peer_feature, (list, tuple)) and peer_feature and all(torch.is_tensor(level_feat) for level_feat in peer_feature):  # 中文注释：当同伴教师返回多尺度列表时逐尺度提取
                        peer_feature_list[sample_idx][peer_sensor] = [level_feat[local_idx] for level_feat in peer_feature]  # 中文注释：将当前样本的所有尺度特征整理成列表
                    elif torch.is_tensor(peer_feature):  # 中文注释：当同伴教师返回单张量时按样本切片
                        peer_feature_list[sample_idx][peer_sensor] = [peer_feature[local_idx]]  # 中文注释：统一包装成列表保持接口一致
                    elif isinstance(peer_feature, (list, tuple)) and peer_feature:  # 中文注释：当同伴教师输出其他序列结构时执行兼容处理
                        extracted_peer = peer_feature[local_idx]  # 中文注释：取出当前样本对应的条目
                        if isinstance(extracted_peer, (list, tuple)):  # 中文注释：若条目自身为多尺度序列则转为列表
                            peer_feature_list[sample_idx][peer_sensor] = list(extracted_peer)  # 中文注释：保留原始尺度顺序
                        elif torch.is_tensor(extracted_peer):  # 中文注释：若条目为张量则包装成列表
                            peer_feature_list[sample_idx][peer_sensor] = [extracted_peer]  # 中文注释：确保后续处理一致
                        else:  # 中文注释：在遇到未知类型时直接存储原始对象
                            peer_feature_list[sample_idx][peer_sensor] = extracted_peer  # 中文注释：保留原始结构供后续逻辑自行判断
                    else:  # 中文注释：在缺失或结构异常时直接记录原始返回值
                        peer_feature_list[sample_idx][peer_sensor] = peer_feature  # 中文注释：透传异常结构防止信息损失
                    peer_prediction_list[sample_idx][peer_sensor] = peer_results[local_idx].pred_instances  # 记录同伴教师预测实例
                    primary_instances = primary_results[local_idx].pred_instances  # 中文注释：读取当前样本主教师的预测实例用于匹配
                    peer_instances = peer_results[local_idx].pred_instances  # 中文注释：读取当前样本同伴教师的预测实例
                    main_bboxes = getattr(primary_instances, 'bboxes', None)  # 中文注释：获取主教师预测框张量
                    peer_bboxes = getattr(peer_instances, 'bboxes', None)  # 中文注释：获取同伴教师预测框张量
                    if main_bboxes is None or peer_bboxes is None or main_bboxes.numel() == 0 or peer_bboxes.numel() == 0:  # 中文注释：若任一教师缺少有效候选框则跳过一致性计算
                        continue  # 中文注释：跳过当前样本的交叉一致性评估
                    peer_bboxes = peer_bboxes.to(device=main_bboxes.device, dtype=main_bboxes.dtype)  # 中文注释：将同伴教师预测框转换到与主教师一致的设备与数据类型
                    main_scores = getattr(primary_instances, 'scores', None)  # 中文注释：读取主教师的分类置信度
                    peer_scores = getattr(peer_instances, 'scores', None)  # 中文注释：读取同伴教师的分类置信度
                    if main_scores is None:  # 中文注释：当主教师未提供置信度时使用单位张量代替
                        main_scores = torch.ones(main_bboxes.shape[0], device=main_bboxes.device, dtype=main_bboxes.dtype)  # 中文注释：构造全为1的置信度以保持流程连续
                    else:
                        main_scores = main_scores.to(device=main_bboxes.device, dtype=main_bboxes.dtype)  # 中文注释：确保主教师置信度与边界框处于相同设备与精度
                    if peer_scores is None:  # 中文注释：当同伴教师未提供置信度时使用单位张量代替
                        peer_scores = torch.ones(peer_bboxes.shape[0], device=main_bboxes.device, dtype=main_bboxes.dtype)  # 中文注释：构造全为1的置信度避免出现空值
                    else:
                        peer_scores = peer_scores.to(device=main_bboxes.device, dtype=main_bboxes.dtype)  # 中文注释：确保同伴教师置信度张量与主教师匹配
                    selection_mask = torch.ones(main_bboxes.shape[0], dtype=torch.bool, device=main_bboxes.device)  # 中文注释：初始化主教师候选框选择掩码
                    if self.cross_consistency_min_conf > 0:  # 中文注释：若配置了最小置信度则依据该阈值筛选候选框
                        selection_mask = selection_mask & (main_scores >= self.cross_consistency_min_conf)  # 中文注释：仅保留置信度满足要求的主教师候选框
                    selected_indices = torch.nonzero(selection_mask, as_tuple=False).squeeze(1)  # 中文注释：提取满足筛选条件的候选框索引
                    if selected_indices.numel() == 0:  # 中文注释：若没有候选框通过筛选则无法计算一致性
                        continue  # 中文注释：跳过当前样本的交叉一致性评估
                    if isinstance(self.cross_consistency_max_pairs, (int, float)) and self.cross_consistency_max_pairs > 0 and selected_indices.numel() > int(self.cross_consistency_max_pairs):  # 中文注释：当设置了匹配对上限且候选过多时执行裁剪
                        max_pair_limit = int(self.cross_consistency_max_pairs)  # 中文注释：将配置的匹配数量转换为整数以便索引操作
                        selected_scores = main_scores[selected_indices]  # 中文注释：收集候选框对应的置信度用于排序
                        _, topk_indices = torch.topk(selected_scores, k=max_pair_limit, largest=True)  # 中文注释：选择置信度最高的若干候选框以控制匹配数量
                        selected_indices = selected_indices[topk_indices]  # 中文注释：根据排序结果截取最终参与匹配的索引
                    overlaps = bbox_overlaps(main_bboxes[selected_indices], peer_bboxes)  # 中文注释：计算主教师候选框与同伴教师候选框之间的IoU矩阵
                    if overlaps.numel() == 0:  # 中文注释：若IoU矩阵为空说明无法建立匹配关系
                        continue  # 中文注释：跳过当前样本的交叉一致性评估
                    max_iou, matched_peer_indices = overlaps.max(dim=1)  # 中文注释：为每个主教师候选框选取IoU最高的同伴候选框
                    if self.cross_consistency_iou_thr > 0:  # 中文注释：当设定了IoU阈值时依据该阈值过滤弱匹配
                        valid_mask = max_iou >= self.cross_consistency_iou_thr  # 中文注释：仅保留满足IoU门槛的匹配结果
                    else:
                        valid_mask = torch.ones_like(max_iou, dtype=torch.bool)  # 中文注释：未设置阈值时默认全部匹配有效
                    if valid_mask.sum() == 0:  # 中文注释：若没有匹配通过阈值则无法继续计算
                        continue  # 中文注释：跳过当前样本的交叉一致性评估
                    matched_main_indices = selected_indices[valid_mask]  # 中文注释：提取有效匹配对应的主教师候选框索引
                    matched_peer_indices = matched_peer_indices[valid_mask]  # 中文注释：提取有效匹配对应的同伴教师候选框索引
                    matched_main_scores = main_scores[matched_main_indices]  # 中文注释：获取参与一致性计算的主教师置信度
                    matched_peer_scores = peer_scores[matched_peer_indices]  # 中文注释：获取对应的同伴教师置信度
                    cls_losses = self._binary_kl_div(matched_main_scores, matched_peer_scores)  # 中文注释：计算主教师与同伴教师在匹配候选上的分类KL散度
                    matched_main_boxes = main_bboxes[matched_main_indices]  # 中文注释：收集主教师匹配候选框的坐标
                    matched_peer_boxes = peer_bboxes[matched_peer_indices]  # 中文注释：收集同伴教师匹配候选框的坐标
                    reg_losses = torch.abs(matched_main_boxes - matched_peer_boxes).mean(dim=1)  # 中文注释：对匹配候选框的坐标差异计算L1损失
                    sample_cls_stats[sample_idx]['sum'] = sample_cls_stats[sample_idx]['sum'] + cls_losses.sum().to(device=device_for_stats, dtype=dtype_for_stats)  # 中文注释：将当前匹配的分类损失累加到对应样本的统计量
                    sample_cls_stats[sample_idx]['count'] += int(cls_losses.numel())  # 中文注释：更新当前样本的分类匹配数量
                    sample_reg_stats[sample_idx]['sum'] = sample_reg_stats[sample_idx]['sum'] + reg_losses.sum().to(device=device_for_stats, dtype=dtype_for_stats)  # 中文注释：将当前匹配的回归损失累加到对应样本的统计量
                    sample_reg_stats[sample_idx]['count'] += int(reg_losses.numel())  # 中文注释：更新当前样本的回归匹配数量
        distill_feature = {  # 汇总主教师与同伴教师特征
            'main_teacher': primary_feature_list  # 主教师特征列表，顺序与批输入一致
        }
        has_cross_feature = any(len(feature_map) > 0 for feature_map in peer_feature_list)  # 判断是否存在有效的同伴特征
        has_cross_prediction = any(len(pred_map) > 0 for pred_map in peer_prediction_list)  # 中文注释：判断是否存在同伴教师的预测结果
        cls_per_sample: List[Optional[Tensor]] = []  # 中文注释：初始化分类一致性逐样本统计列表
        reg_per_sample: List[Optional[Tensor]] = []  # 中文注释：初始化回归一致性逐样本统计列表
        for cls_stats, reg_stats in zip(sample_cls_stats, sample_reg_stats):  # 中文注释：遍历每个样本的分类与回归统计量
            if cls_stats['count'] > 0:  # 中文注释：当分类匹配数量大于零时计算平均KL散度
                cls_per_sample.append(cls_stats['sum'] / cls_stats['count'])  # 中文注释：记录该样本的平均分类一致性值
            else:
                cls_per_sample.append(None)  # 中文注释：当无匹配对时以None占位便于后续判断
            if reg_stats['count'] > 0:  # 中文注释：当回归匹配数量大于零时计算平均L1损失
                reg_per_sample.append(reg_stats['sum'] / reg_stats['count'])  # 中文注释：记录该样本的平均回归一致性值
            else:
                reg_per_sample.append(None)  # 中文注释：无匹配对时占位以避免误用
        valid_cls_values = [value for value in cls_per_sample if value is not None]  # 中文注释：筛选出真实存在的分类一致性结果
        valid_reg_values = [value for value in reg_per_sample if value is not None]  # 中文注释：筛选出真实存在的回归一致性结果
        has_cls_metric = len(valid_cls_values) > 0  # 中文注释：判断是否存在可用的分类一致性度量
        has_reg_metric = len(valid_reg_values) > 0  # 中文注释：判断是否存在可用的回归一致性度量
        zero_scalar = torch.zeros((), device=device_for_stats, dtype=dtype_for_stats)  # 中文注释：预先构造零标量便于在缺失时回退
        cls_consistency_value = torch.stack(valid_cls_values).mean() if has_cls_metric else zero_scalar  # 中文注释：汇总所有样本的分类一致性并在缺失时返回零
        reg_consistency_value = torch.stack(valid_reg_values).mean() if has_reg_metric else zero_scalar.clone()  # 中文注释：汇总所有样本的回归一致性并在缺失时返回零并拷贝一份零标量避免共享引用
        if has_cross_feature or has_cross_prediction or has_cls_metric or has_reg_metric:  # 中文注释：当至少存在一种交叉信息或一致性指标时构造返回包
            cross_teacher_payload: Dict[str, Any] = {}  # 中文注释：初始化交叉教师返回字典
            if has_cross_feature:  # 中文注释：存在同伴特征时写入列表
                cross_teacher_payload['features'] = peer_feature_list  # 中文注释：记录每个样本对应的同伴教师特征
            if has_cross_prediction:  # 中文注释：存在同伴预测时一并写入
                cross_teacher_payload['predictions'] = peer_prediction_list  # 中文注释：记录同伴教师输出的预测实例
            cross_teacher_payload['cls_consistency'] = cls_consistency_value  # 中文注释：写入跨教师分类一致性的平均结果
            cross_teacher_payload['reg_consistency'] = reg_consistency_value  # 中文注释：写入跨教师回归一致性的平均结果
            distill_feature['cross_teacher'] = cross_teacher_payload  # 中文注释：将交叉教师信息补充到整体返回字典
        return batch_data_samples, batch_info, distill_feature  # 返回伪标签、批信息与多教师特征

    def _parse_diff_feature(self, diff_feature: Any, batch_info: dict) -> dict:
        """中文注释：将扩散教师返回的特征结构统一整理便于后续蒸馏使用。"""
        main_teacher_feature = diff_feature  # 中文注释：默认主教师特征直接等于原始特征
        cross_teacher_info = None  # 中文注释：初始化交叉教师信息为空
        if isinstance(diff_feature, dict):  # 中文注释：当特征以字典形式提供时按约定键解析
            main_teacher_feature = diff_feature.get('main_teacher', diff_feature.get('teacher_feature', diff_feature))  # 中文注释：优先读取主教师特征键并在缺失时回退
            cross_teacher_info = diff_feature.get('cross_teacher')  # 中文注释：获取交叉教师相关信息块
        parsed_feature = {'main_teacher': main_teacher_feature}  # 中文注释：构建标准化返回字典并写入主教师特征
        if isinstance(cross_teacher_info, dict):  # 中文注释：若交叉教师信息为字典则分发特定字段
            cross_features = cross_teacher_info.get('features')  # 中文注释：读取交叉教师提供的特征集合
            cross_predictions = cross_teacher_info.get('predictions')  # 中文注释：读取交叉教师给出的预测集合
            if cross_features is not None:  # 中文注释：仅在存在交叉特征时才写入批次信息
                batch_info.setdefault('cross_teacher', {})  # 中文注释：确保批次信息包含交叉教师分支
                batch_info['cross_teacher']['features'] = cross_features  # 中文注释：记录交叉教师特征供后续损失使用
            if cross_predictions is not None:  # 中文注释：仅在存在交叉预测时记录
                batch_info.setdefault('cross_teacher', {})  # 中文注释：防止批次信息缺少交叉教师条目
                batch_info['cross_teacher']['predictions'] = cross_predictions  # 中文注释：存储交叉教师的预测结果
            parsed_feature['cross_teacher'] = cross_teacher_info  # 中文注释：在解析结果中保留交叉教师完整信息
        elif cross_teacher_info is not None:  # 中文注释：若交叉教师信息存在但非字典则直接透传
            parsed_feature['cross_teacher'] = cross_teacher_info  # 中文注释：保持原始结构避免信息丢失
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
