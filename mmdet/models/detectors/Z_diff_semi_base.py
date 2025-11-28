# Copyright (c) OpenMMLab. All rights reserved.
import copy
import inspect  # 中文注释：引入inspect以便在运行时检测学生loss是否支持current_iter参数
from typing import Dict
from typing import Any, List, Optional, Union, Tuple
import torch  # 中文注释：导入PyTorch基础库用于张量运算
import torch.nn as nn  # 中文注释：导入神经网络模块以便构建模型与操作层
import torch.nn.functional as F  # 中文注释：导入函数式接口以便进行归一化等运算
from torch import Tensor  # 中文注释：从PyTorch中显式导入Tensor类型便于类型标注
from torchvision.ops import roi_align  # 中文注释：导入ROIAlign函数以便在特征图上对齐采样

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
        self.trainable_diff_teacher_keys: List[str] = []  # 中文注释：记录需要训练的扩散教师标识列表以便后续逻辑快速判断
        self.trainable_diff_teachers: List[nn.Module] = []  # 中文注释：缓存所有可训练扩散教师模块对象方便外部优化器访问
        self._trainable_diff_teacher_modules = nn.ModuleDict()  # 中文注释：使用ModuleDict注册可训练教师以便state_dict与优化器捕获参数
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
            #——————
            # teacher_instance.half()  # 将教师模型转换为半精度以节省显存占用
            # for m in teacher_instance.modules():
            #     # 找到带 diff_model 且有 change_precision 接口的模块
            #     if hasattr(m, 'diff_model') and hasattr(m.diff_model, 'change_precision'):
            #         # 1. 把 aggregation_network / finecoder 转成 half
            #         m.diff_model.change_precision('half')
            #         # 2. 让 forward 走 self.mode == "half" 分支，不再 .to(torch.float)
            #         if hasattr(m.diff_model, 'mode'):
            #             m.diff_model.mode = 'half'            
            #——————
            is_trainable_teacher = bool(teacher_meta.get('trainable', False) or teacher_meta.get('requires_training', False))  # 中文注释：根据配置标识判断当前教师是否需要参与训练
            if is_trainable_teacher:  # 中文注释：当教师需要训练时跳过冻结流程
                for param in teacher_instance.parameters():  # 中文注释：遍历所有参数确保梯度开关处于激活状态
                    param.requires_grad = True  # 中文注释：显式开启梯度以防加载权重过程中被关闭
                teacher_instance.train(True)  # 中文注释：设置为训练模式以便批归一化等层更新统计量
                self.trainable_diff_teacher_keys.append(teacher_key)  # 中文注释：记录可训练教师的键值方便后续检索
                self.trainable_diff_teachers.append(teacher_instance)  # 中文注释：将教师实例加入可训练列表供优化器构建参数组
                self._trainable_diff_teacher_modules[teacher_key] = teacher_instance  # 中文注释：将可训练教师注册到ModuleDict以确保参数写入state_dict
            else:  # 中文注释：对于仅推理使用的教师仍保持冻结逻辑
                self.freeze(teacher_instance)  # 中文注释：冻结教师模型参数避免训练阶段被更新
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
            trainable_flag = bool(self._fetch_config_value(current_cfg, 'trainable', False))  # 中文注释：解析可训练标识并在缺省情况下回退为False确保逻辑稳定
            requires_training_flag = bool(self._fetch_config_value(current_cfg, 'requires_training', False))  # 中文注释：解析是否需要训练的标识以便后续teacher_meta直接读取
            gradient_flag_overrides = {}  # 中文注释：准备额外的梯度开关字段集合便于透传自定义键
            if isinstance(current_cfg, dict):  # 中文注释：仅在当前配置为字典时才遍历其键值
                for candidate_key, candidate_value in current_cfg.items():  # 中文注释：遍历所有键值对以筛选梯度相关开关
                    if not isinstance(candidate_key, str):  # 中文注释：若键非字符串则跳过避免lower操作报错
                        continue  # 中文注释：继续处理下一个键值对
                    lowered_key = candidate_key.lower()  # 中文注释：将键名转为小写以便统一匹配关键字
                    if any(marker in lowered_key for marker in ('train', 'grad')) and isinstance(candidate_value, bool):  # 中文注释：仅捕获名称中包含train或grad且值为布尔的开关字段
                        gradient_flag_overrides[candidate_key] = bool(candidate_value)  # 中文注释：将捕获的开关字段记录下来并显式转换为布尔值
            normalized_configs[sensor_key] = {  # 汇总解析结果
                'config': config_path,  # 存储配置文件路径
                'pretrained_model': pretrained_path,  # 存储预训练权重路径
                'raw': current_cfg,  # 保留原始配置项便于后续构建
                'aliases': alias_candidates,  # 记录全部可匹配的别名集合
                'trainable': trainable_flag,  # 中文注释：直接记录可训练开关使teacher_meta可以快速读取
                'requires_training': requires_training_flag  # 中文注释：记录是否需要训练的标识以匹配旧逻辑
            }
            for extra_flag_key, extra_flag_value in gradient_flag_overrides.items():  # 中文注释：遍历额外的梯度开关并写回标准化配置
                if extra_flag_key in normalized_configs[sensor_key]:  # 中文注释：若该键已存在则跳过以保留更明确的字段值
                    continue  # 中文注释：避免覆盖已有字段
                normalized_configs[sensor_key][extra_flag_key] = extra_flag_value  # 中文注释：将额外的梯度开关写入标准化配置方便后续直接读取
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
            if detector_name in self.trainable_diff_teacher_keys:  # 中文注释：当教师被标记为可训练时根据传入mode设置训练状态
                detector_module.train(mode)  # 中文注释：保持与外部训练模式一致以参与梯度更新
            else:  # 中文注释：普通教师仍固定在评估模式以稳定伪标签生成
                detector_module.train(False)  # 中文注释：确保仅推理教师不会受到train()调用影响
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
        if name in self.trainable_diff_teacher_keys:  # 中文注释：若当前激活教师需要训练则维持训练模式
            self.diff_detector.train(True)  # 中文注释：允许激活教师继续更新统计量并参与梯度
        else:  # 中文注释：否则强制其处于评估模式
            self.diff_detector.train(False)  # 中文注释：确保纯推理教师不会切换到训练模式

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
                             multi_batch_data_samples: Dict[str, SampleList],
                             ssdc_cfg: Optional[dict] = None,
                             current_iter: Optional[int] = None) -> dict:
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
            multi_batch_inputs['sup'], multi_batch_data_samples['sup'], current_iter=current_iter))  # 中文注释：先计算有监督分支损失并透传迭代索引驱动调度
        origin_pseudo_data_samples, batch_info, diff_feature = self.get_pseudo_instances_diff(
            multi_batch_inputs['unsup_teacher'], multi_batch_data_samples['unsup_teacher'])  # 中文注释：获取伪标签、批次信息以及原始的特征打包结果
        parsed_diff_feature = self._parse_diff_feature(diff_feature, batch_info)  # 中文注释：对返回的特征结构进行标准化解析并在必要时补充批次信息
        multi_batch_data_samples['unsup_student'] = self.project_pseudo_instances(
            origin_pseudo_data_samples, multi_batch_data_samples['unsup_student'])  # 中文注释：将伪标签投影到学生视角
        if ssdc_cfg is not None:  # 中文注释：当提供SS-DC配置时尝试执行域不变相似度门控
            teacher_inv_override = parsed_diff_feature.get('main_teacher_inv') if isinstance(parsed_diff_feature, dict) else None  # 中文注释：从扩散教师特征包中提取域不变特征以替代均值教师缓存
            self._apply_di_gate(origin_pseudo_data_samples, multi_batch_data_samples['unsup_student'], multi_batch_inputs['unsup_teacher'], multi_batch_inputs['unsup_student'], ssdc_cfg, current_iter, teacher_inv_override=teacher_inv_override)  # 中文注释：根据域不变相似度过滤低可信度伪标签

        losses.update(**self.loss_by_pseudo_instances(multi_batch_inputs['unsup_student'],
                                                      multi_batch_data_samples['unsup_student'], batch_info, current_iter=current_iter))  # 中文注释：计算学生分支伪标签损失并合并且同步传递迭代信息
        return losses, parsed_diff_feature  # 中文注释：返回损失字典与解析后的特征包

    def loss_by_gt_instances(self, batch_inputs: Tensor,
                             batch_data_samples: SampleList,
                             current_iter: Optional[int] = None) -> dict:
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

        losses = self._call_student_loss(batch_inputs, batch_data_samples, current_iter=current_iter)  # 中文注释：通过统一入口调用学生loss并在支持时传递current_iter
        sup_weight = self.semi_train_cfg.get('sup_weight', 1.)
        return rename_loss_dict('sup_', reweight_loss_dict(losses, sup_weight))

    def loss_by_pseudo_instances(self,
                                 batch_inputs: Tensor,
                                 batch_data_samples: SampleList,
                                 batch_info: Optional[dict] = None,
                                 current_iter: Optional[int] = None) -> dict:
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
        losses = self._call_student_loss(batch_inputs, batch_data_samples, current_iter=current_iter)  # 中文注释：通过统一入口确保current_iter被安全透传
        pseudo_instances_num = sum([
            len(data_samples.gt_instances)
            for data_samples in batch_data_samples
        ])
        unsup_weight = self.semi_train_cfg.get(
            'unsup_weight', 1.) if pseudo_instances_num > 0 else 0.
        return rename_loss_dict('unsup_',
                                reweight_loss_dict(losses, unsup_weight))

    def _call_student_loss(self,
                           batch_inputs: Tensor,
                           batch_data_samples: SampleList,
                           current_iter: Optional[int] = None) -> dict:
        """中文注释：封装学生loss调用并在其支持current_iter时安全传参。"""
        loss_func = getattr(self.student, 'loss')  # 中文注释：读取学生loss方法用于后续调用
        support_flag = getattr(self, '_student_loss_supports_current_iter', None)  # 中文注释：尝试读取缓存的参数支持标记
        if support_flag is None:  # 中文注释：当尚未缓存检测结果时执行一次签名分析
            try:  # 中文注释：捕获inspect过程中可能出现的异常
                signature = inspect.signature(loss_func)  # 中文注释：获取loss方法的函数签名
                support_flag = 'current_iter' in signature.parameters  # 中文注释：判断签名中是否包含current_iter形参
            except (ValueError, TypeError):  # 中文注释：当无法获取签名时回退为不支持
                support_flag = False  # 中文注释：将支持标记置为False保证兼容
            self._student_loss_supports_current_iter = support_flag  # 中文注释：缓存检测结果避免重复开销
        if support_flag:  # 中文注释：若学生loss支持current_iter参数
            return loss_func(batch_inputs, batch_data_samples, current_iter=current_iter)  # 中文注释：携带当前迭代索引调用loss
        return loss_func(batch_inputs, batch_data_samples)  # 中文注释：否则保持原始调用方式

    @staticmethod  # 中文注释：使用静态方法封装签名检测逻辑便于在任意实例上下文复用
    def _supports_current_iter_arg(target_callable) -> bool:  # 中文注释：判断目标可调用是否支持current_iter关键字
        """中文注释：检测目标可调用对象是否声明了current_iter参数以便按需传参。"""
        try:  # 中文注释：通过inspect读取函数签名可能触发异常因此使用try保证稳健
            signature = inspect.signature(target_callable)  # 中文注释：获取可调用对象的参数签名
            return 'current_iter' in signature.parameters  # 中文注释：判断签名中是否存在current_iter参数
        except (ValueError, TypeError):  # 中文注释：若无法获取签名则视为不支持
            return False  # 中文注释：返回False防止误传递额外参数

    def _extract_feat_with_optional_iter(self,  # 中文注释：封装extract_feat调用以在需要时透传current_iter参数
                                         module: Optional[nn.Module],  # 中文注释：目标模块可以是教师或学生模型
                                         batch_inputs: Tensor,  # 中文注释：输入图像张量
                                         current_iter: Optional[int]):  # 中文注释：当前训练迭代索引用于驱动burn-in调度
        """中文注释：在支持current_iter时携带该参数调用extract_feat并对检测结果进行缓存。"""
        if module is None or not hasattr(module, 'extract_feat'):  # 中文注释：若模块不存在或缺少特征提取接口则直接返回
            return None  # 中文注释：保持接口安全
        extract_func = getattr(module, 'extract_feat')  # 中文注释：读取模块的特征提取函数
        support_flag = getattr(module, '_extract_feat_supports_current_iter', None)  # 中文注释：尝试读取先前缓存的支持标记
        if support_flag is None:  # 中文注释：当未缓存检测结果时执行一次签名分析
            support_flag = self._supports_current_iter_arg(extract_func)  # 中文注释：复用静态函数检测是否支持current_iter
            setattr(module, '_extract_feat_supports_current_iter', support_flag)  # 中文注释：将检测结果挂载到模块以便复用
        if support_flag:  # 中文注释：若模块支持current_iter参数则携带该关键字调用
            return extract_func(batch_inputs, current_iter=current_iter)  # 中文注释：传入当前迭代索引驱动burn-in调度
        return extract_func(batch_inputs)  # 中文注释：不支持时沿用旧式调用避免异常

    @staticmethod
    def _interp_schedule_value(schedule_cfg: Any, current_iter: Optional[int], default: float = 0.0) -> float:
        """中文注释：根据迭代步线性插值获取调度值。"""
        if current_iter is None:  # 中文注释：当未提供当前迭代时直接返回默认
            return float(schedule_cfg if isinstance(schedule_cfg, (int, float)) else default)  # 中文注释：兼容常数与默认
        if isinstance(schedule_cfg, (int, float)):  # 中文注释：常量调度直接返回
            return float(schedule_cfg)  # 中文注释：转换为浮点
        if not isinstance(schedule_cfg, (list, tuple)) or not schedule_cfg:  # 中文注释：非法配置返回默认
            return default  # 中文注释：保持稳健
        sorted_points = sorted(list(schedule_cfg), key=lambda item: item[0])  # 中文注释：按迭代索引排序调度节点
        if current_iter <= sorted_points[0][0]:  # 中文注释：早于首节点
            return float(sorted_points[0][1])  # 中文注释：返回起始值
        if current_iter >= sorted_points[-1][0]:  # 中文注释：晚于末节点
            return float(sorted_points[-1][1])  # 中文注释：返回结束值
        for (iter_a, val_a), (iter_b, val_b) in zip(sorted_points[:-1], sorted_points[1:]):  # 中文注释：遍历区间
            if iter_a <= current_iter <= iter_b:  # 中文注释：找到所在区间
                ratio = (current_iter - iter_a) / max(float(iter_b - iter_a), 1.0)  # 中文注释：线性比例防止除零
                return float(val_a + ratio * (val_b - val_a))  # 中文注释：返回插值结果
        return default  # 中文注释：兜底返回默认值

    @staticmethod
    def _as_homography_tensor(matrix: Any, device: torch.device, dtype: torch.dtype) -> Tensor:
        """中文注释：将多种格式的单应性矩阵安全地转换为指定设备与精度的张量。"""
        if matrix is None:  # 中文注释：当未提供矩阵时默认使用单位矩阵代表无几何变换
            return torch.eye(3, device=device, dtype=dtype)  # 中文注释：单位矩阵表示原图坐标与目标坐标一致
        if torch.is_tensor(matrix):  # 中文注释：若输入已为张量则直接转换设备与精度
            return matrix.to(device=device, dtype=dtype)  # 中文注释：返回调整后的单应性矩阵
        return torch.as_tensor(matrix, device=device, dtype=dtype)  # 中文注释：将numpy或列表转换为张量格式

    @staticmethod
    def _project_with_homography(boxes: Tensor, homography: Tensor, target_shape: Optional[Tuple[int, int]]) -> Tensor:
        """中文注释：使用提供的单应性矩阵将边界框投影到目标坐标系。"""
        return bbox_project(boxes, homography, target_shape)  # 中文注释：直接调用bbox_project完成坐标变换并可选裁剪

    @torch.no_grad()
    def _apply_di_gate(self, teacher_pseudo_samples: SampleList,
                       student_pseudo_samples: SampleList,
                       teacher_inputs: Tensor,
                       student_inputs: Tensor,
                       ssdc_cfg: dict,
                       current_iter: Optional[int],
                       teacher_inv_override: Optional[Any] = None) -> None:
        """中文注释：依据域不变特征相似度过滤低置信度伪标签。"""
        tau_schedule = ssdc_cfg.get('consistency_gate', None)  # 中文注释：读取相似度阈值调度
        if tau_schedule is None:  # 中文注释：未配置阈值则直接返回
            return  # 中文注释：保持原始伪标签
        tau_value = self._interp_schedule_value(tau_schedule, current_iter, 0.0)  # 中文注释：计算当前迭代的阈值
        if tau_value <= 0:  # 中文注释：阈值无效时跳过过滤
            return  # 中文注释：保持伪标签完整
        teacher_inv = teacher_inv_override  # 中文注释：优先使用扩散教师在伪标签阶段缓存的域不变特征
        student_inv = None  # 中文注释：初始化学生域不变特征占位符
        if teacher_inv is None and hasattr(self.teacher, 'extract_feat'):  # 中文注释：当扩散教师未提供特征时回退均值教师提取流程
            getattr(self.teacher, 'ssdc_feature_cache', {}).clear()  # 中文注释：在提取前清空缓存避免跨batch残留影响本次相似度
            _ = self._extract_feat_with_optional_iter(self.teacher, teacher_inputs, current_iter)  # 中文注释：调用带有current_iter适配的特征提取以保持burn-in一致
            teacher_cache = getattr(self.teacher, 'ssdc_feature_cache', {}).get('noref', {})  # 中文注释：读取教师缓存
            teacher_inv = teacher_cache.get('inv')  # 中文注释：获取域不变特征
        if hasattr(self.student, 'extract_feat'):  # 中文注释：确保学生模型支持特征提取
            getattr(self.student, 'ssdc_feature_cache', {}).clear()  # 中文注释：同样清空学生缓存以保证当前迭代使用的特征最新
            _ = self._extract_feat_with_optional_iter(self.student, student_inputs, current_iter)  # 中文注释：通过统一入口传递current_iter以维持burn-in调度一致
            student_cache = getattr(self.student, 'ssdc_feature_cache', {}).get('noref', {})  # 中文注释：读取学生缓存
            student_inv = student_cache.get('inv')  # 中文注释：获取学生域不变特征
        if teacher_inv is None or student_inv is None:  # 中文注释：任一特征缺失则无法过滤
            return  # 中文注释：保持伪标签
        if not isinstance(teacher_inv, (list, tuple)) or not isinstance(student_inv, (list, tuple)):  # 中文注释：要求特征序列
            return  # 中文注释：结构不匹配时跳过
        if len(teacher_inv) == 0 or len(student_inv) == 0:  # 中文注释：空特征直接返回
            return  # 中文注释：无有效特征
        teacher_map = teacher_inv[0]  # 中文注释：使用首层域不变特征进行区域对齐
        student_map = student_inv[0]  # 中文注释：学生端同样使用首层
        if teacher_map is None or student_map is None:  # 中文注释：首层缺失则跳过
            return  # 中文注释：保持伪标签
        teacher_input_hw = (int(teacher_inputs.shape[-2]), int(teacher_inputs.shape[-1]))  # 中文注释：记录教师输入的高宽便于投影兜底
        student_input_hw = (int(student_inputs.shape[-2]), int(student_inputs.shape[-1]))  # 中文注释：记录学生输入的高宽信息
        if teacher_inputs.shape[-1] > 0:  # 中文注释：当输入宽度大于0时计算教师缩放比
            teacher_scale = float(teacher_map.shape[-1]) / float(teacher_inputs.shape[-1])  # 中文注释：用特征图宽度除以输入宽度得到缩放系数
        else:  # 中文注释：输入宽度异常时退回默认缩放比
            teacher_scale = 1.0  # 中文注释：默认缩放比为1避免除零
        if student_inputs.shape[-1] > 0:  # 中文注释：当学生输入宽度有效时计算学生缩放比
            student_scale = float(student_map.shape[-1]) / float(student_inputs.shape[-1])  # 中文注释：用学生特征图宽度除以输入宽度得到缩放
        else:  # 中文注释：学生输入宽度异常时
            student_scale = 1.0  # 中文注释：默认缩放比防止除零
        filtered_samples = []  # 中文注释：准备过滤后的学生伪标签列表
        for batch_idx in range(len(student_pseudo_samples)):  # 中文注释：逐个batch索引独立处理
            teacher_sample = (teacher_pseudo_samples[batch_idx]  # 中文注释：当索引合法时取出教师样本
                              if batch_idx < len(teacher_pseudo_samples)  # 中文注释：确保索引不会越界
                              else None)  # 中文注释：否则返回空以跳过过滤
            student_sample = student_pseudo_samples[batch_idx]  # 中文注释：获取当前学生样本
            if teacher_sample is None:  # 中文注释：若不存在对应的教师样本则直接保留
                filtered_samples.append(student_sample)  # 中文注释：直接加入结果列表
                continue  # 中文注释：处理下一个样本
            if not hasattr(teacher_sample, 'gt_instances') or teacher_sample.gt_instances is None:  # 中文注释：教师缺少伪标签时跳过过滤
                filtered_samples.append(student_sample)  # 中文注释：直接加入结果
                continue  # 中文注释：继续后续样本
            if not hasattr(student_sample, 'gt_instances') or student_sample.gt_instances is None:  # 中文注释：学生缺少伪标签时无需过滤
                filtered_samples.append(student_sample)  # 中文注释：直接加入结果
                continue  # 中文注释：继续后续处理
            teacher_boxes = teacher_sample.gt_instances.bboxes  # 中文注释：读取教师伪框坐标
            student_boxes = student_sample.gt_instances.bboxes  # 中文注释：读取学生伪框坐标
            if teacher_boxes.numel() == 0 or student_boxes.numel() == 0:  # 中文注释：若任一侧无框则不做相似度过滤
                filtered_samples.append(student_sample)  # 中文注释：直接加入结果列表
                continue  # 中文注释：继续处理下一个样本
            teacher_origin_boxes = getattr(teacher_sample.gt_instances, 'origin_bboxes', teacher_boxes).to(  # 中文注释：优先读取显式缓存的原图坐标
                device=teacher_map.device, dtype=teacher_map.dtype)  # 中文注释：对齐到教师特征所在设备与精度
            teacher_view_boxes = getattr(teacher_sample.gt_instances, 'teacher_view_bboxes', None)  # 中文注释：尝试读取教师特征视角下的边界框
            if teacher_view_boxes is None:  # 中文注释：当缓存缺失时根据单应性矩阵重新映射
                teacher_homography = self._as_homography_tensor(  # 中文注释：转换单应性矩阵为张量
                    getattr(teacher_sample, 'homography_matrix', None), teacher_map.device, teacher_map.dtype)
                teacher_target_shape = getattr(teacher_sample, 'img_shape', teacher_input_hw)  # 中文注释：确定教师输入空间大小
                teacher_view_boxes = self._project_with_homography(  # 中文注释：将原图框映射到教师视角
                    teacher_origin_boxes, teacher_homography, teacher_target_shape)
            else:
                teacher_view_boxes = teacher_view_boxes.to(device=teacher_map.device, dtype=teacher_map.dtype)  # 中文注释：确保缓存框与特征同设备精度
            student_view_boxes = getattr(student_sample.gt_instances, 'student_view_bboxes', None)  # 中文注释：尝试读取学生视角下的边界框
            student_homography = self._as_homography_tensor(  # 中文注释：转换学生单应性矩阵
                getattr(student_sample, 'homography_matrix', None), student_map.device, student_map.dtype)
            student_target_shape = getattr(student_sample, 'img_shape', student_input_hw)  # 中文注释：确定学生输入空间尺寸
            if student_view_boxes is None:  # 中文注释：若未缓存学生视角框则从原图框推回
                origin_boxes_hint = getattr(student_sample.gt_instances, 'origin_bboxes', None)  # 中文注释：仅在存在原图坐标缓存时才执行投影
                if origin_boxes_hint is not None:  # 中文注释：确认原图坐标可用
                    origin_boxes_hint = origin_boxes_hint.to(device=student_map.device, dtype=student_map.dtype)  # 中文注释：对齐设备与精度
                    student_view_boxes = self._project_with_homography(  # 中文注释：根据学生单应性矩阵映射到学生视角
                        origin_boxes_hint, student_homography, student_target_shape)
                else:
                    student_view_boxes = student_boxes.to(device=student_map.device, dtype=student_map.dtype)  # 中文注释：若无法回推则默认当前框已在学生视角
            else:
                student_view_boxes = student_view_boxes.to(device=student_map.device, dtype=student_map.dtype)  # 中文注释：将缓存框迁移到学生特征设备
            student_inverse_h = student_homography.inverse()  # 中文注释：求逆矩阵以便将学生框投影回原图
            student_origin_shape = getattr(student_sample, 'ori_shape', student_target_shape)  # 中文注释：确定原图尺寸用于裁剪
            student_origin_boxes = self._project_with_homography(  # 中文注释：得到学生框在原图坐标下的表示
                student_view_boxes, student_inverse_h, student_origin_shape)
            student_origin_boxes = student_origin_boxes.to(device=teacher_origin_boxes.device, dtype=teacher_origin_boxes.dtype)  # 中文注释：对齐设备与精度以便IoU计算
            if teacher_origin_boxes.shape[0] == student_origin_boxes.shape[0]:  # 中文注释：当教师与学生框数量一致时按索引对齐
                matched_teacher_boxes = teacher_origin_boxes  # 中文注释：直接使用原图坐标的教师框
                matched_teacher_view = teacher_view_boxes  # 中文注释：同步记录教师特征坐标下的框
                matched_student_boxes = student_origin_boxes  # 中文注释：直接使用原图坐标的学生框
                matched_student_view = student_view_boxes  # 中文注释：同步学生特征坐标下的框
                matched_mask = torch.ones(  # 中文注释：创建匹配掩码张量
                    student_origin_boxes.shape[0], dtype=torch.bool, device=student_origin_boxes.device)  # 中文注释：默认全保留便于后续覆盖
            else:  # 中文注释：数量不一致时使用IoU进行匹配
                iou_matrix = bbox_overlaps(teacher_origin_boxes, student_origin_boxes, mode='iou')  # 中文注释：计算教师与学生框的IoU矩阵
                best_iou, best_teacher_idx = iou_matrix.max(dim=0)  # 中文注释：对每个学生框选择最佳教师框及其IoU
                matched_mask = best_iou > 0  # 中文注释：仅保留存在重叠的学生框参与过滤
                if not matched_mask.any():  # 中文注释：若无任何有效匹配则直接跳过过滤
                    filtered_samples.append(student_sample)  # 中文注释：保持学生伪标签
                    continue  # 中文注释：进入下一张图像
                matched_teacher_boxes = teacher_origin_boxes[best_teacher_idx[matched_mask]]  # 中文注释：根据匹配索引提取对应教师框
                matched_teacher_view = teacher_view_boxes[best_teacher_idx[matched_mask]]  # 中文注释：同步抽取教师特征坐标下的框
                matched_student_boxes = student_origin_boxes[matched_mask]  # 中文注释：提取参与匹配的学生框
                matched_student_view = student_view_boxes[matched_mask]  # 中文注释：同步抽取学生特征坐标下的框
            teacher_batch_index = torch.full(  # 中文注释：构造教师ROI对应的批次索引列
                (matched_teacher_view.shape[0], 1), batch_idx, device=matched_teacher_view.device, dtype=matched_teacher_view.dtype)  # 中文注释：批次索引用于对齐
            teacher_roi = torch.cat([teacher_batch_index, matched_teacher_view], dim=1)  # 中文注释：拼接索引与教师框形成ROI
            student_batch_index = torch.full(  # 中文注释：构造学生ROI对应的批次索引列
                (matched_student_view.shape[0], 1), batch_idx, device=matched_student_view.device, dtype=matched_student_view.dtype)  # 中文注释：批次索引用于对齐
            student_roi = torch.cat([student_batch_index, matched_student_view], dim=1)  # 中文注释：拼接索引与学生框形成ROI
            pooled_teacher = roi_align(  # 中文注释：在教师特征图上对齐当前样本ROI
                teacher_map, teacher_roi, output_size=1, spatial_scale=teacher_scale, aligned=True)  # 中文注释：采样得到教师ROI特征
            pooled_student = roi_align(  # 中文注释：在学生特征图上对齐当前样本ROI
                student_map, student_roi, output_size=1, spatial_scale=student_scale, aligned=True)  # 中文注释：采样得到学生ROI特征
            pooled_teacher = pooled_teacher.flatten(1)  # 中文注释：将教师对齐结果展平为(N, C)
            pooled_student = pooled_student.flatten(1)  # 中文注释：将学生对齐结果展平为(N, C)
            teacher_norm = F.normalize(pooled_teacher, dim=1)  # 中文注释：教师特征进行L2归一化
            student_norm = F.normalize(pooled_student, dim=1)  # 中文注释：学生特征进行L2归一化
            cosine_scores = (teacher_norm * student_norm).sum(dim=1)  # 中文注释：计算匹配框对的余弦相似度
            sample_keep_mask = torch.ones(  # 中文注释：初始化当前样本的保留掩码默认全部保留
                student_boxes.shape[0], dtype=torch.bool, device=student_boxes.device)  # 中文注释：使用与学生框相同设备与数据类型
            if matched_mask.any():  # 中文注释：当存在有效匹配时才更新阈值掩码
                sample_keep_mask[matched_mask] = cosine_scores >= tau_value  # 中文注释：将匹配到的学生框按相似度阈值过滤
            if sample_keep_mask.all():  # 中文注释：若所有伪框均被保留则无需修改
                filtered_samples.append(student_sample)  # 中文注释：直接加入结果
                continue  # 中文注释：继续处理下一样本
            if sample_keep_mask.any():  # 中文注释：存在部分保留时应用掩码
                student_sample.gt_instances = student_sample.gt_instances[sample_keep_mask]  # 中文注释：按掩码筛选学生伪标签实例
            else:  # 中文注释：当全部被过滤时
                student_sample.gt_instances = student_sample.gt_instances[:0]  # 中文注释：清空伪标签保持结构完整
            filtered_samples.append(student_sample)  # 中文注释：将更新后的样本追加到列表
        student_pseudo_samples[:] = filtered_samples  # 中文注释：用过滤后的结果替换原学生伪标签列表

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
    def _get_pseudo_instances_diff_inference(
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
                teacher_view_boxes = data_samples.gt_instances.bboxes.detach().clone()  # 中文注释：缓存教师视角下的边界框以便后续对齐
                teacher_view_boxes = teacher_view_boxes.to(device=self.data_preprocessor.device)  # 中文注释：将教师视角框迁移到预处理器所在设备
                data_samples.gt_instances.teacher_view_bboxes = teacher_view_boxes.clone()  # 中文注释：显式区分教师特征坐标系下的边界框
                homography_tensor = self._as_homography_tensor(  # 中文注释：将单应性矩阵转换为张量方便求逆
                    data_samples.homography_matrix, self.data_preprocessor.device, teacher_view_boxes.dtype)
                inverse_homography = homography_tensor.inverse()  # 中文注释：求逆得到从教师视角映射回原图的矩阵
                projected_boxes = self._project_with_homography(  # 中文注释：利用逆矩阵将教师框映射回原图坐标
                    teacher_view_boxes, inverse_homography, data_samples.ori_shape)
                data_samples.gt_instances.bboxes = projected_boxes  # 中文注释：更新原图坐标下的伪标签
                data_samples.gt_instances.origin_bboxes = projected_boxes.detach().clone()  # 中文注释：缓存原图坐标供后续再映射
            if isinstance(diff_feature, (list, tuple)):  # 若返回的是按层排列的特征序列
                primary_features = list(diff_feature)  # 转换为列表以便后续统一封装
            else:  # 若返回单一张量
                primary_features = [diff_feature for _ in range(len(batch_data_samples))]  # 为每个样本复制一份引用
            teacher_cache = getattr(self.diff_detector, 'ssdc_feature_cache', {}).get('noref', {})  # 中文注释：读取扩散教师缓存的域不变特征
            teacher_inv_feature = teacher_cache.get('inv')  # 中文注释：尝试获取域不变特征
            if isinstance(teacher_inv_feature, (list, tuple)) and teacher_inv_feature:  # 中文注释：仅当结构合法且非空时才使用
                sample_count = len(batch_data_samples)  # 中文注释：记录当前批次的样本数量以便拆分特征
                per_level_slices: List[List[Tensor]] = []  # 中文注释：初始化按层切分的特征存储容器
                for level_feat in teacher_inv_feature:  # 中文注释：遍历每一层的特征张量
                    per_level_slices.append([level_feat[idx] for idx in range(sample_count)])  # 中文注释：按样本索引切片并缓存
                stacked_inv: List[Tensor] = []  # 中文注释：准备堆叠后的域不变特征列表
                for sample_stack in per_level_slices:  # 中文注释：遍历每一层的切片结果
                    stacked_inv.append(torch.stack(sample_stack, dim=0))  # 中文注释：沿批次维度堆叠恢复原有排列
                teacher_inv_feature = tuple(stacked_inv)  # 中文注释：将堆叠结果转换为元组以保持缓存格式一致
            else:  # 中文注释：当缓存缺失或格式不符时回退占位
                teacher_inv_feature = None  # 中文注释：标记域不变特征不可用
            distill_feature = {  # 构造主教师输出结构
                'main_teacher': primary_features,  # 主教师特征列表
                'main_teacher_inv': teacher_inv_feature  # 中文注释：同步返回扩散教师缓存的域不变特征以供相似度门控使用
            }
            return batch_data_samples, batch_info, distill_feature  # 返回伪标签、批信息与特征

        for teacher_model in self.diff_teacher_bank.values():  # 若存在多教师，先统一切换至评估模式
            teacher_model.eval()  # 设置教师为评估模式，确保推理一致性
        batch_info = {}  # 初始化批处理信息占位符
        sample_count = len(batch_data_samples)  # 记录样本数量，便于构造输出容器
        primary_feature_list: List[Any] = [None] * sample_count  # 为主教师特征预留位置
        main_inv_slices: Optional[List[List[Optional[Tensor]]]] = None  # 中文注释：初始化域不变特征分片容器用于重组批次顺序
        sensor_tag_list: List[Optional[str]] = [None] * sample_count  # 中文注释：初始化逐样本传感器标签列表
        grouped_primary_features: Dict[str, Any] = dict()  # 中文注释：记录按传感器划分的主教师多尺度特征
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
            normalized_group_feature, sample_level_features = self._normalize_group_feature_payload(primary_feature, len(sample_indices))  # 中文注释：统一当前传感器的教师特征结构并拆分逐样本特征
            grouped_primary_features[sensor_tag] = normalized_group_feature  # 中文注释：登记当前传感器对应的多尺度特征
            teacher_cache = getattr(teacher_model, 'ssdc_feature_cache', {}).get('noref', {})  # 中文注释：读取当前传感器教师的SS-DC缓存
            cache_inv = teacher_cache.get('inv')  # 中文注释：尝试获取域不变特征
            if isinstance(cache_inv, (list, tuple)) and cache_inv:  # 中文注释：仅在结构合法时才参与重组
                level_count = len(cache_inv)  # 中文注释：记录层级数量以初始化容器
                if main_inv_slices is None:  # 中文注释：首次遇到有效特征时创建批次重排容器
                    main_inv_slices = [[None for _ in range(sample_count)] for _ in range(level_count)]  # 中文注释：为每个尺度与样本准备占位
                for level_idx, level_feat in enumerate(cache_inv):  # 中文注释：遍历所有尺度特征
                    for local_idx, sample_idx in enumerate(sample_indices):  # 中文注释：遍历当前批次内的样本索引
                        main_inv_slices[level_idx][sample_idx] = level_feat[local_idx]  # 中文注释：按全局顺序填充域不变特征
            for local_idx, sample_idx in enumerate(sample_indices):  # 遍历组内样本
                origin_sample = batch_data_samples[sample_idx]  # 获取原始数据样本
                origin_sample.gt_instances = primary_results[local_idx].pred_instances  # 写入伪标签实例
                teacher_view_boxes = origin_sample.gt_instances.bboxes.detach().clone()  # 中文注释：缓存教师视角坐标的边界框
                teacher_view_boxes = teacher_view_boxes.to(device=self.data_preprocessor.device)  # 中文注释：将教师视角框迁移至预处理器设备
                origin_sample.gt_instances.teacher_view_bboxes = teacher_view_boxes.clone()  # 中文注释：记录教师特征坐标系下的框
                sensor_tag_list[sample_idx] = sensor_tag  # 中文注释：记录当前样本的传感器标签供后续蒸馏使用
                homography_tensor = self._as_homography_tensor(  # 中文注释：转换单应性矩阵为张量形式
                    origin_sample.homography_matrix, self.data_preprocessor.device, teacher_view_boxes.dtype)
                inverse_homography = homography_tensor.inverse()  # 中文注释：求逆以从教师视角回到原图坐标
                projected_boxes = self._project_with_homography(  # 中文注释：将教师视角框映射到原图
                    teacher_view_boxes, inverse_homography, origin_sample.ori_shape)
                origin_sample.gt_instances.bboxes = projected_boxes  # 中文注释：覆盖为原图坐标的伪标签
                origin_sample.gt_instances.origin_bboxes = projected_boxes.detach().clone()  # 中文注释：缓存原图坐标副本以便后续再投影
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
        if any(tag is None for tag in sensor_tag_list):  # 中文注释：检测是否存在未被赋值的传感器标签
            raise KeyError('存在缺失传感器标签的样本，无法完成多教师伪标签生成。')  # 中文注释：抛出明确错误提示
        distill_feature = {  # 汇总主教师与同伴教师特征
            'main_teacher': primary_feature_list,  # 主教师特征列表，顺序与批输入一致
            'sensor_map': sensor_tag_list,  # 中文注释：记录逐样本的传感器来源便于后续蒸馏分组
            'grouped_main_teacher': grouped_primary_features  # 中文注释：按传感器组织的主教师多尺度特征
        }
        if main_inv_slices is not None and all(all(level_sample is not None for level_sample in level_list) for level_list in main_inv_slices):  # 中文注释：仅当所有样本的域不变特征齐全时才组装堆叠结果
            stacked_inv: List[Tensor] = []  # 中文注释：初始化堆叠后的域不变特征列表
            for level_list in main_inv_slices:  # 中文注释：遍历每个尺度的采样结果
                stacked_inv.append(torch.stack(level_list, dim=0))  # 中文注释：沿批次维度堆叠确保顺序与输入一致
            distill_feature['main_teacher_inv'] = tuple(stacked_inv)  # 中文注释：写入与ssdc_feature_cache一致的域不变特征格式
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

    def get_pseudo_instances_diff(
            self, batch_inputs: Tensor, batch_data_samples: SampleList
    ) -> Tuple[SampleList, Optional[dict]]:
        """中文注释：联合推理与可训练教师前向以生成伪标签并返回可用于蒸馏的特征。"""
        pseudo_samples, batch_info, distill_feature = self._get_pseudo_instances_diff_inference(batch_inputs, batch_data_samples)  # 中文注释：先运行无梯度推理分支生成伪标签与基础特征
        if not self.trainable_diff_teacher_keys:  # 中文注释：若不存在可训练教师则直接返回推理结果
            return pseudo_samples, batch_info, distill_feature  # 中文注释：无需附加信息时直接退出
        sensor_map = distill_feature.get('sensor_map')  # 中文注释：尝试获取逐样本传感器标签用于匹配可训练教师
        if sensor_map is None:  # 中文注释：当推理阶段未返回传感器映射时使用样本元信息兜底
            sensor_map = [sample.metainfo.get('sensor') if hasattr(sample, 'metainfo') else None for sample in pseudo_samples]  # 中文注释：从数据样本元数据提取传感器标签
        trainable_payload = self._forward_trainable_diff_teachers(batch_inputs, pseudo_samples, sensor_map, distill_feature)  # 中文注释：执行可训练教师的正常前向以获取带梯度的特征与预测
        if trainable_payload:  # 中文注释：仅在存在可训练教师输出时才扩展特征字典
            distill_feature['trainable_teachers'] = trainable_payload  # 中文注释：将可训练教师的详细输出写入特征包
        return pseudo_samples, batch_info, distill_feature  # 中文注释：返回融合后的伪标签、批信息与特征

    def _forward_trainable_diff_teachers(self,
                                         batch_inputs: Tensor,
                                         batch_data_samples: SampleList,
                                         sensor_map: List[Optional[str]],
                                         distill_feature: dict) -> Dict[str, dict]:
        """中文注释：对标记为可训练的扩散教师执行常规前向并返回梯度可追踪的特征。"""
        trainable_outputs: Dict[str, dict] = {}  # 中文注释：初始化返回字典用于存放各个可训练教师的输出
        grouped_features: Dict[str, Any] = distill_feature.get('grouped_main_teacher', {})  # 中文注释：获取或初始化按传感器组织的特征字典
        main_feature_list: List[Any] = distill_feature.get('main_teacher', [])  # 中文注释：获取逐样本主教师特征列表以便替换为带梯度结果
        for teacher_key in self.trainable_diff_teacher_keys:  # 中文注释：遍历所有可训练教师标识
            if teacher_key not in self.diff_teacher_bank:  # 中文注释：若教师未注册则跳过防止异常
                continue  # 中文注释：继续处理下一个教师
            sample_indices = [idx for idx, tag in enumerate(sensor_map) if tag == teacher_key]  # 中文注释：根据传感器映射筛选当前教师负责的样本索引
            if not sample_indices:  # 中文注释：若当前批次无对应样本则无需前向
                continue  # 中文注释：跳过当前教师
            teacher_model = self.diff_teacher_bank[teacher_key]  # 中文注释：获取教师实例
            teacher_model.train(True)  # 中文注释：确保教师处于训练模式以便统计更新
            group_inputs = batch_inputs[sample_indices]  # 中文注释：按索引提取对应的输入图像张量
            group_samples = [copy.deepcopy(batch_data_samples[i]) for i in sample_indices]  # 中文注释：深拷贝样本避免前向过程中修改原对象
            predictions, feature_payload = teacher_model.predict(group_inputs, group_samples, rescale=False, return_feature=True)  # 中文注释：执行常规预测以获取特征并保留梯度信息
            normalized_features, per_sample_features = self._normalize_group_feature_payload(feature_payload, len(sample_indices))  # 中文注释：将返回特征整理为统一结构并拆分为逐样本列表
            grouped_features[teacher_key] = normalized_features  # 中文注释：以传感器键存储多尺度特征供后续蒸馏使用
            for local_idx, global_idx in enumerate(sample_indices):  # 中文注释：遍历当前教师负责的样本索引
                main_feature_list[global_idx] = per_sample_features[local_idx]  # 中文注释：使用带梯度的特征覆盖推理阶段的占位
            trainable_outputs[teacher_key] = {  # 中文注释：组装当前教师的输出包
                'features': per_sample_features,  # 中文注释：逐样本多尺度特征列表
                'group_features': normalized_features,  # 中文注释：按尺度整理的整体特征
                'predictions': [result.pred_instances for result in predictions]  # 中文注释：保存预测实例以便构造自监督损失
            }
        distill_feature['grouped_main_teacher'] = grouped_features  # 中文注释：回写最新的按传感器特征映射
        distill_feature['main_teacher'] = main_feature_list  # 中文注释：更新主教师特征列表确保外部读取到带梯度的张量
        return trainable_outputs  # 中文注释：返回可训练教师的输出详情供上层模块构建损失

    def get_trainable_diff_teacher_parameters(self) -> List[nn.Parameter]:
        """中文注释：收集所有可训练扩散教师的参数以便优化器创建独立参数组。"""
        parameters: List[nn.Parameter] = []  # 中文注释：初始化参数容器
        for teacher_module in self.trainable_diff_teachers:  # 中文注释：遍历全部可训练教师模块
            for param in teacher_module.parameters():  # 中文注释：遍历教师内部所有参数张量
                if param.requires_grad:  # 中文注释：仅保留已启用梯度的参数以避免冗余
                    parameters.append(param)  # 中文注释：将参数加入结果列表
        return parameters  # 中文注释：返回聚合后的参数列表供外部优化器使用

    def _parse_diff_feature(self, diff_feature: Any, batch_info: dict) -> dict:
        """中文注释：将扩散教师返回的特征结构统一整理便于后续蒸馏使用。"""
        main_teacher_feature = diff_feature  # 中文注释：默认主教师特征直接等于原始特征
        cross_teacher_info = None  # 中文注释：初始化交叉教师信息为空
        sensor_map = None  # 中文注释：初始化传感器标签映射占位
        grouped_main_teacher = None  # 中文注释：初始化按传感器划分的主教师特征结构
        trainable_teacher_info = None  # 中文注释：初始化可训练教师信息占位
        if isinstance(diff_feature, dict):  # 中文注释：当特征以字典形式提供时按约定键解析
            main_teacher_feature = diff_feature.get('main_teacher', diff_feature.get('teacher_feature', diff_feature))  # 中文注释：优先读取主教师特征键并在缺失时回退
            main_teacher_inv = diff_feature.get('main_teacher_inv')  # 中文注释：尝试读取主教师域不变特征以供相似度门控
            cross_teacher_info = diff_feature.get('cross_teacher')  # 中文注释：获取交叉教师相关信息块
            sensor_map = diff_feature.get('sensor_map')  # 中文注释：尝试读取逐样本传感器标签列表
            grouped_main_teacher = diff_feature.get('grouped_main_teacher')  # 中文注释：尝试读取按传感器整理的主教师特征
            trainable_teacher_info = diff_feature.get('trainable_teachers')  # 中文注释：提取可训练教师的附加输出以便上层构建损失
        parsed_feature = {'main_teacher': main_teacher_feature}  # 中文注释：构建标准化返回字典并写入主教师特征
        if 'main_teacher_inv' in locals():  # 中文注释：确保当存在域不变特征条目时同步写入解析结果
            parsed_feature['main_teacher_inv'] = main_teacher_inv  # 中文注释：将域不变特征透传给上层逻辑
        if sensor_map is not None:  # 中文注释：若提供逐样本传感器映射则保留
            parsed_feature['sensor_map'] = sensor_map  # 中文注释：记录传感器标签列表
        if grouped_main_teacher is not None:  # 中文注释：若存在按传感器分组的特征则附加
            parsed_feature['grouped_main_teacher'] = grouped_main_teacher  # 中文注释：保存分组后的主教师特征结构
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
        if trainable_teacher_info is not None:  # 中文注释：若可训练教师提供了额外输出则保留
            parsed_feature['trainable_teachers'] = trainable_teacher_info  # 中文注释：在解析结果中记录可训练教师信息供上层损失读取
        return parsed_feature  # 中文注释：返回解析后的特征字典

    def project_pseudo_instances(self, batch_pseudo_instances: SampleList,
                                 batch_data_samples: SampleList) -> SampleList:
        """Project pseudo instances."""
        for pseudo_instances, data_samples in zip(batch_pseudo_instances,
                                                  batch_data_samples):
            data_samples.gt_instances = copy.deepcopy(
                pseudo_instances.gt_instances)
            projected_student_boxes = bbox_project(  # 中文注释：将原图伪框根据学生单应性矩阵投影到学生视角
                data_samples.gt_instances.bboxes,
                torch.tensor(data_samples.homography_matrix).to(
                    self.data_preprocessor.device), data_samples.img_shape)
            data_samples.gt_instances.bboxes = projected_student_boxes  # 中文注释：更新学生视角下的伪框坐标
            data_samples.gt_instances.student_view_bboxes = projected_student_boxes.detach().clone()  # 中文注释：缓存学生输入坐标系下的伪框以便ROI对齐
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

    def _normalize_group_feature_payload(self, feature_payload: Any, group_size: int) -> Tuple[Any, List[Any]]:  # 中文注释：规范化教师特征结构以兼容不同调用场景
        if feature_payload is None:  # 中文注释：当输入特征为空时直接返回占位结果
            return None, [None] * group_size  # 中文注释：返回空特征以及长度匹配的None列表
        if isinstance(feature_payload, (list, tuple)) and feature_payload:  # 中文注释：针对列表或元组结构分别展开处理
            first_item = feature_payload[0]  # 中文注释：读取首个元素用于判断当前存储形式
            if torch.is_tensor(first_item):  # 中文注释：若首元素为张量说明结构按尺度排列
                level_tensors = [level_feat for level_feat in feature_payload]  # 中文注释：复制各尺度张量以保持与输入一致
                per_sample_features: List[List[Tensor]] = []  # 中文注释：初始化逐样本特征列表
                level_count = len(level_tensors)  # 中文注释：记录尺度数量用于遍历
                for sample_idx in range(group_size):  # 中文注释：遍历组内每一个样本索引
                    sample_levels = [level_feat[sample_idx] for level_feat in level_tensors]  # 中文注释：提取该样本在各尺度的切片
                    per_sample_features.append(sample_levels)  # 中文注释：将切片加入逐样本列表
                return level_tensors, per_sample_features  # 中文注释：返回按尺度排列的张量与逐样本特征
            if isinstance(first_item, (list, tuple)) and first_item:  # 中文注释：若首元素也是序列说明特征已按样本拆分
                per_sample_features = [list(sample_feat) for sample_feat in feature_payload]  # 中文注释：逐样本复制避免共享底层引用
                level_count = len(first_item)  # 中文注释：获取每个样本包含的尺度数量
                stacked_levels: List[Tensor] = []  # 中文注释：初始化按尺度堆叠的张量列表
                for level_idx in range(level_count):  # 中文注释：遍历所有尺度索引
                    level_stack = torch.stack([sample_feat[level_idx] for sample_feat in per_sample_features], dim=0)  # 中文注释：沿批次维度堆叠该尺度的所有样本
                    stacked_levels.append(level_stack)  # 中文注释：保存堆叠后的张量便于后续前向
                return stacked_levels, per_sample_features  # 中文注释：返回堆叠张量及逐样本列表
        if torch.is_tensor(feature_payload):  # 中文注释：若输入直接为张量则视作单尺度特征
            per_sample_features = [feature_payload[sample_idx:sample_idx + 1] for sample_idx in range(group_size)]  # 中文注释：逐样本切片保持张量结构
            return [feature_payload], per_sample_features  # 中文注释：返回包装后的张量和逐样本特征
        per_sample_features = [feature_payload for _ in range(group_size)]  # 中文注释：其他类型直接复制引用以补齐长度
        return feature_payload, per_sample_features  # 中文注释：返回原始特征对象与逐样本列表

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

