# Copyright (c) OpenMMLab. All rights reserved.
import copy
import warnings
from typing import List, Tuple, Union

import torch
from torch import Tensor

from mmdet.models.utils import (rename_loss_dict,
                                reweight_loss_dict)
from mmdet.structures.bbox import bbox2roi
from ..utils import unpack_gt_instances
from mmdet.models.ssdc import SAIDFilterBank, SSDCouplingNeck  # 导入SS-DC相关模块用于特征解耦与耦合


from mmdet.registry import MODELS
from mmdet.structures import SampleList
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from .base import BaseDetector
from ..losses import KDLoss

def bbox_to_mask(batch_data_samples, N, H, W, class_names):
    batch_masks = torch.full((N, H, W), 0, dtype=torch.long)
    batch_labels = []
    for i in range(N):
        gt_instance = batch_data_samples[i].gt_instances
        bboxes = gt_instance["bboxes"]
        labels = gt_instance["labels"]
        sample_labels = set([class_names[label.item()] for label in labels])
        if sample_labels:
            label_string = "A photo of " + ", ".join(sample_labels)
        else:
            label_string = ""
        batch_labels.append(label_string)
        bbox_areas = [(bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) for bbox in bboxes]
        sorted_indices = sorted(range(len(bboxes)), key=lambda idx: bbox_areas[idx], reverse=True)
        for idx in sorted_indices:
            bbox = bboxes[idx]
            x1, y1, x2, y2 = bbox.int()
            batch_masks[i, y1:y2, x1:x2] = 1

    return batch_masks, batch_labels


@MODELS.register_module()
class DiffusionDetector(BaseDetector):
    """Base class for two-stage detectors.

    Two-stage detectors typically consisting of a region proposal network and a
    task-specific regression head.
    """

    def __init__(self,
                 backbone: ConfigType,
                 neck: OptConfigType = None,
                 rpn_head: OptConfigType = None,
                 roi_head: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 auxiliary_branch_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        self.backbone = MODELS.build(backbone)

        if neck is not None:
            self.neck = MODELS.build(neck)

        if rpn_head is not None:
            rpn_train_cfg = train_cfg.rpn if train_cfg is not None else None
            rpn_head_ = rpn_head.copy()
            rpn_head_.update(train_cfg=rpn_train_cfg, test_cfg=test_cfg.rpn)
            rpn_head_num_classes = rpn_head_.get('num_classes', None)
            if rpn_head_num_classes is None:
                rpn_head_.update(num_classes=1)
            else:
                if rpn_head_num_classes != 1:
                    warnings.warn(
                        'The `num_classes` should be 1 in RPN, but get '
                        f'{rpn_head_num_classes}, please set '
                        'rpn_head.num_classes = 1 in your config file.')
                    rpn_head_.update(num_classes=1)
            self.rpn_head = MODELS.build(rpn_head_)

        if roi_head is not None:
            # update train and test cfg here for now
            # TODO: refactor assigner & sampler
            rcnn_train_cfg = train_cfg.rcnn if train_cfg is not None else None
            roi_head.update(train_cfg=rcnn_train_cfg)
            roi_head.update(test_cfg=test_cfg.rcnn)
            self.roi_head = MODELS.build(roi_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.auxiliary_branch_cfg = auxiliary_branch_cfg
        
        self.loss_cls_kd = MODELS.build(self.auxiliary_branch_cfg['loss_cls_kd'])
        self.loss_reg_kd = MODELS.build(self.auxiliary_branch_cfg['loss_reg_kd'])
        self.apply_auxiliary_branch = self.auxiliary_branch_cfg['apply_auxiliary_branch']
        
        self.loss_feature = KDLoss(loss_weight=1.0, loss_type='mse')  # 初始化特征蒸馏损失用于辅助训练

        self.enable_ssdc = False  # 初始化标志位默认关闭光谱空间解耦模块
        self.use_ds_tokens = False  # 初始化是否启用域特异令牌的标志位
        self.said_filter = None  # 初始化SAID滤波器引用占位以便条件构建
        self.coupling_neck = None  # 初始化耦合颈部引用占位以便条件构建
        self.ssdc_feature_cache = {}  # 初始化缓存字典用于保存最近一次的SS-DC特征
        self.loss_decouple = None  # 初始化解耦损失模块引用占位符
        self.loss_decouple_weight = 1.0  # 初始化解耦损失整体权重默认1.0
        self.loss_couple = None  # 初始化耦合损失模块引用占位符
        self.loss_couple_weight = 1.0  # 初始化耦合损失整体权重默认1.0
        self._ssdc_num_feature_levels = None  # 初始化特征层级数量占位便于运行时校验
        self._ssdc_level_names = None  # 初始化特征层级名称列表占位符用于配置同步
        self._ssdc_start_level = None  # 初始化特征层级起始索引占位符用于动态生成
        self._ssdc_level_prefix = None  # 初始化特征层级前缀占位符用于统一命名
        ssdc_cfg = {}  # 初始化SS-DC配置字典收集不同来源的参数
        backbone_enable_ssdc = False  # 初始化来自骨干网络配置的开关标志
        if isinstance(backbone, dict):  # 若骨干配置为字典则读取SS-DC相关配置
            ssdc_cfg = copy.deepcopy(backbone.get('ssdc_cfg', {}))  # 从骨干配置中提取SS-DC子配置并深拷贝避免原地修改
            backbone_enable_ssdc = backbone.get('enable_ssdc', False)  # 读取骨干配置中的SS-DC开关默认关闭
        train_cfg_enable_ssdc = False  # 初始化来自训练配置的开关标志
        ssdc_cfg_from_train = None  # 初始化训练阶段提供的SS-DC配置占位
        if train_cfg is not None:  # 当训练配置存在时尝试读取附加配置
            if hasattr(train_cfg, 'get'):  # 当训练配置实现get方法时优先使用字典式访问
                train_cfg_enable_ssdc = train_cfg.get('enable_ssdc', False)  # 读取训练配置中的SS-DC开关默认关闭
                ssdc_cfg_from_train = train_cfg.get('ssdc_cfg', None)  # 读取训练配置中的SS-DC详细参数若未提供则为空
            else:  # 当训练配置不支持get方法时使用属性访问回退
                train_cfg_enable_ssdc = getattr(train_cfg, 'enable_ssdc', False)  # 通过属性访问获取SS-DC开关默认False
                ssdc_cfg_from_train = getattr(train_cfg, 'ssdc_cfg', None)  # 通过属性访问获取SS-DC配置默认None
        if isinstance(ssdc_cfg_from_train, dict):  # 若训练配置提供了字典形式的SS-DC参数则合并到最终配置中
            ssdc_cfg.update(copy.deepcopy(ssdc_cfg_from_train))  # 使用深拷贝合并训练阶段覆盖的SS-DC参数
        enable_flags = (  # 组合多个来源的开关标志
            backbone_enable_ssdc,  # 来自骨干配置的开关标志
            train_cfg_enable_ssdc,  # 来自训练配置的开关标志
            ssdc_cfg.get('enable_ssdc', False),  # 来自SS-DC配置自身的开关标志
        )
        self.enable_ssdc = any(enable_flags)  # 只要任一来源开启则启用SS-DC
        if self.enable_ssdc:  # 当确定启用SS-DC时实例化对应模块
            said_cfg_ref = ssdc_cfg.setdefault('said_filter', {})  # 初始化或获取SAID滤波器配置以便写入层级标签
            coupling_cfg_ref = ssdc_cfg.setdefault('coupling_neck', {})  # 初始化或获取耦合颈部配置以便写入层级标签
            num_feature_levels, level_names = self._infer_num_feature_levels(said_cfg_ref, coupling_cfg_ref)  # 调用内部方法优先基于实网结构推断层级数量与名称
            coupling_cfg_ref['num_feature_levels'] = num_feature_levels  # 将推断得到的特征层数写入耦合颈部配置供模块校验
            coupling_cfg_ref['levels'] = level_names  # 将统一生成的层级列表写入耦合颈部配置确保内部模块一致
            said_cfg_ref['levels'] = level_names  # 将统一生成的层级列表写入SAID滤波器配置确保与特征数量匹配
            said_cfg = copy.deepcopy(said_cfg_ref)  # 深拷贝SAID滤波器子配置避免副作用
            if said_cfg and 'type' in said_cfg:  # 若提供了类型字段则通过注册表动态构建
                self.said_filter = MODELS.build(said_cfg)  # 使用注册表根据配置构建SAID滤波器模块
            else:  # 若未提供类型字段则直接实例化默认实现
                self.said_filter = SAIDFilterBank(**said_cfg)  # 以关键字参数构建SAID滤波器模块使用合理默认值
            coupling_cfg = copy.deepcopy(coupling_cfg_ref)  # 深拷贝耦合颈部配置以便修改默认参数
            inferred_channels = None  # 初始化根据现有网络推断的通道数占位
            if self.with_neck and hasattr(self.neck, 'out_channels'):  # 若存在颈部且暴露输出通道则直接采用
                inferred_channels = self.neck.out_channels  # 记录从颈部推断得到的通道数
            elif hasattr(self.backbone, 'out_channels'):  # 若颈部不可用则尝试从骨干网络读取通道数
                inferred_channels = self.backbone.out_channels  # 使用骨干输出通道作为耦合模块的输入维度
            has_custom_channels = (  # 初始化标志用于判断配置是否提供自定义输入通道
                'in_channels' in coupling_cfg  # 检查配置字典是否包含输入通道键
                and coupling_cfg.get('in_channels') is not None  # 确认提供的输入通道值有效
            )
            if not has_custom_channels:  # 当配置未显式提供输入通道时设置默认值
                default_channels = inferred_channels if inferred_channels is not None else 256  # 优先使用推断通道否则采用常见默认256
                coupling_cfg['in_channels'] = default_channels  # 将确定的通道数写入耦合配置供构造函数使用
            if coupling_cfg and 'type' in coupling_cfg:  # 若配置中包含类型字段则通过注册表构建
                self.coupling_neck = MODELS.build(coupling_cfg)  # 使用注册表构建耦合颈部模块以支持灵活替换
            else:  # 否则直接实例化默认实现
                self.coupling_neck = SSDCouplingNeck(**coupling_cfg)  # 以关键字参数构建耦合颈部模块使用合理默认值
            self.use_ds_tokens = bool(coupling_cfg.get('use_ds_tokens', False))  # 读取配置中是否启用域特异令牌的标志
            loss_decouple_cfg = copy.deepcopy(ssdc_cfg.get('loss_decouple', {}))  # 深拷贝解耦损失配置以避免污染原配置
            self.loss_decouple_weight = float(loss_decouple_cfg.pop('loss_weight', 1.0))  # 读取解耦损失总权重缺省为1.0
            loss_decouple_cfg.setdefault('type', 'LossDecouple')  # 若未指定类型则默认使用LossDecouple实现
            self.loss_decouple = MODELS.build(loss_decouple_cfg)  # 通过注册表构建解耦损失模块实例
            loss_couple_cfg = copy.deepcopy(ssdc_cfg.get('loss_couple', {}))  # 深拷贝耦合损失配置以避免污染原配置
            self.loss_couple_weight = float(loss_couple_cfg.pop('loss_weight', 1.0))  # 读取耦合损失总权重缺省为1.0
            loss_couple_cfg.setdefault('type', 'LossCouple')  # 若未指定类型则默认使用LossCouple实现
            self.loss_couple = MODELS.build(loss_couple_cfg)  # 通过注册表构建耦合损失模块实例

        self.class_maps = backbone['diff_config']['classes']

    def _infer_num_feature_levels(self, said_cfg_ref, coupling_cfg_ref):
        """基于已构建的骨干和颈部模块推断特征层级数量并生成层级名称列表。"""
        self._ssdc_start_level = coupling_cfg_ref.get('start_level', said_cfg_ref.get('start_level', 2))  # 读取层级起始索引若无则使用默认值2
        self._ssdc_level_prefix = coupling_cfg_ref.get('level_prefix', said_cfg_ref.get('level_prefix', 'P'))  # 读取层级命名前缀若无则默认P
        num_feature_levels = None  # 初始化特征层级数量占位符优先依据实际模块属性推断
        if hasattr(self, 'neck'):  # 当模型包含颈部模块时优先从颈部读取层级信息
            neck_num_outs = getattr(self.neck, 'num_outs', None)  # 读取颈部声明的输出层数
            if isinstance(neck_num_outs, int) and neck_num_outs > 0:  # 当颈部明确提供正整数层数时直接采用
                num_feature_levels = neck_num_outs  # 使用颈部声明的层数作为特征层级数量
            if num_feature_levels is None:  # 当颈部未声明层数时尝试根据输出通道列表长度推断
                neck_out_channels = getattr(self.neck, 'out_channels', None)  # 读取颈部输出通道配置
                if isinstance(neck_out_channels, (list, tuple)):  # 当输出通道为序列时可利用其长度作为层级数量
                    num_feature_levels = len(neck_out_channels)  # 使用颈部输出通道数量作为特征层级数量
        if num_feature_levels is None and hasattr(self.backbone, 'num_outs'):  # 当颈部推断失败且骨干声明输出层数时使用骨干信息
            backbone_num_outs = getattr(self.backbone, 'num_outs')  # 读取骨干网络声明的输出层级数量
            if isinstance(backbone_num_outs, int) and backbone_num_outs > 0:  # 当骨干提供合法层数时采用该值
                num_feature_levels = backbone_num_outs  # 使用骨干声明的输出层数
        if num_feature_levels is None and hasattr(self.backbone, 'out_channels'):  # 当仍未确定时检查骨干输出通道列表
            backbone_out_channels = getattr(self.backbone, 'out_channels')  # 读取骨干输出通道配置
            if isinstance(backbone_out_channels, (list, tuple)):  # 当骨干输出通道为序列时可利用其长度推断层级
                num_feature_levels = len(backbone_out_channels)  # 使用骨干输出通道数量作为特征层级数量
        if num_feature_levels is None:  # 当模块属性均无法提供信息时回退到配置字段
            preset_levels = coupling_cfg_ref.get('levels') or said_cfg_ref.get('levels')  # 尝试读取SAID或耦合配置中已有的层级列表
            if isinstance(preset_levels, (list, tuple)):  # 当配置提供合法列表时使用其长度
                num_feature_levels = len(preset_levels)  # 根据配置层级列表长度确定特征层级数量
        if num_feature_levels is None:  # 当仍无法获取层级数量时根据输入通道配置推断
            in_channels_cfg = coupling_cfg_ref.get('in_channels')  # 读取耦合颈部输入通道字段
            if isinstance(in_channels_cfg, (list, tuple)):  # 当输入通道为序列时利用其长度
                num_feature_levels = len(in_channels_cfg)  # 使用输入通道序列长度作为特征层级数量
        if num_feature_levels is None and hasattr(self.backbone, 'out_indices'):  # 若仍未知则根据骨干输出索引长度进行估计
            backbone_out_indices = getattr(self.backbone, 'out_indices')  # 读取骨干输出索引配置
            if isinstance(backbone_out_indices, (list, tuple)):  # 当输出索引为序列时可利用其长度
                num_feature_levels = len(backbone_out_indices)  # 使用输出索引长度作为特征层级数量
        if num_feature_levels is None:  # 若经过所有推断仍无结果则使用配置中显式给定的数值
            cfg_num_levels = coupling_cfg_ref.get('num_feature_levels')  # 读取耦合颈部配置中的层级数量字段
            if isinstance(cfg_num_levels, int) and cfg_num_levels > 0:  # 当配置中提供合法正整数时采用
                num_feature_levels = cfg_num_levels  # 使用配置显式给定的层级数量
        if num_feature_levels is None:  # 最后兜底使用安全默认值避免运行时崩溃
            num_feature_levels = 4  # 设置默认四层输出与常见FPN结构保持一致
        level_names = [f'{self._ssdc_level_prefix}{self._ssdc_start_level + idx}' for idx in range(num_feature_levels)]  # 根据推断数量生成层级名称列表
        self._ssdc_num_feature_levels = num_feature_levels  # 将最终确定的层级数量缓存在实例属性中便于后续校验
        self._ssdc_level_names = level_names  # 将生成的层级名称缓存在实例属性中便于复用
        return num_feature_levels, level_names  # 返回层级数量与名称列表供初始化流程写回配置

    def _load_from_state_dict(self, state_dict: dict, prefix: str,
                              local_metadata: dict, strict: bool,
                              missing_keys: Union[List[str], str],
                              unexpected_keys: Union[List[str], str],
                              error_msgs: Union[List[str], str]) -> None:
        """Exchange bbox_head key to rpn_head key when loading single-stage
        weights into two-stage model."""
        bbox_head_prefix = prefix + '.bbox_head' if prefix else 'bbox_head'
        bbox_head_keys = [
            k for k in state_dict.keys() if k.startswith(bbox_head_prefix)
        ]
        rpn_head_prefix = prefix + '.rpn_head' if prefix else 'rpn_head'
        rpn_head_keys = [
            k for k in state_dict.keys() if k.startswith(rpn_head_prefix)
        ]
        if len(bbox_head_keys) != 0 and len(rpn_head_keys) == 0:
            for bbox_head_key in bbox_head_keys:
                rpn_head_key = rpn_head_prefix + \
                    bbox_head_key[len(bbox_head_prefix):]
                state_dict[rpn_head_key] = state_dict.pop(bbox_head_key)
        super()._load_from_state_dict(state_dict, prefix, local_metadata,
                                      strict, missing_keys, unexpected_keys,
                                      error_msgs)

    @property
    def with_rpn(self) -> bool:
        """bool: whether the detector has RPN"""
        return hasattr(self, 'rpn_head') and self.rpn_head is not None

    @property
    def with_roi_head(self) -> bool:
        """bool: whether the detector has a RoI head"""
        return hasattr(self, 'roi_head') and self.roi_head is not None

    def extract_feat(self, batch_inputs: Tensor, ref_masks=None, ref_labels=None) -> Tuple[Tensor]:
        """Extract features.

        Args:
            batch_inputs (Tensor): Image tensor with shape (N, C, H ,W).

        Returns:
            tuple[Tensor]: Multi-level features that may have
            different resolutions.
        """
        if ref_masks != None and ref_labels != None:
            x = self.backbone(batch_inputs, ref_masks, ref_labels)
        else:
            x = self.backbone(batch_inputs)
        if self.with_neck:
            x = self.neck(x)
        storage_key = 'ref' if (ref_masks is not None and ref_labels is not None) else 'noref'  # 根据参考信息选择缓存键
        if isinstance(x, tuple):  # 当颈部或骨干直接返回元组特征时直接沿用保持结构不变
            features = x  # 记录元组特征供后续SS-DC与检测头使用
        elif isinstance(x, list):  # 当返回列表特征时转换为元组以保证接口一致性
            features = tuple(x)  # 将列表转换为元组避免后续修改影响缓存内容
        else:  # 当返回单个Tensor时包装为单元素元组以兼容多层特征接口
            features = (x,)  # 将单层特征封装为元组便于统一处理
        feature_count = len(features)  # 记录当前实际特征层级数量便于与配置对齐
        if self.enable_ssdc and self.said_filter is not None and self.coupling_neck is not None:  # 当启用SS-DC模块时执行层级数量校验
            expected_levels = self._ssdc_num_feature_levels  # 读取初始化阶段推断的预期层级数
            if expected_levels is None and hasattr(self.coupling_neck, 'num_feature_levels'):  # 当预期值缺失时尝试从耦合颈部读取声明
                coupling_levels = getattr(self.coupling_neck, 'num_feature_levels')  # 获取耦合颈部声明的层级数量
                if isinstance(coupling_levels, int) and coupling_levels > 0:  # 当耦合颈部提供合法值时使用
                    expected_levels = coupling_levels  # 使用耦合颈部层级数量作为预期
            if expected_levels is None:  # 若仍未获得预期层级数量则以实际输出为准并同步缓存
                expected_levels = feature_count  # 将当前输出层级数量视为预期值
                self._ssdc_num_feature_levels = feature_count  # 将预期值缓存以便后续复用
                level_prefix = self._ssdc_level_prefix if self._ssdc_level_prefix is not None else 'P'  # 读取或回退层级命名前缀
                start_level = self._ssdc_start_level if self._ssdc_start_level is not None else 2  # 读取或回退层级起始索引
                self._ssdc_level_names = [f'{level_prefix}{start_level + idx}' for idx in range(feature_count)]  # 根据真实层级数量生成层级名称列表
            if feature_count != expected_levels:  # 当实际层级数量与预期不一致时立即报错提示配置问题
                raise RuntimeError(
                    f'检测到SS-DC模块收到的特征层数为{feature_count}，但配置/推断期望{expected_levels}层，请检查backbone/neck输出或ssdc配置的levels/num_feature_levels设置。'
                )  # 抛出运行时异常以避免隐式形状错误
        if self.enable_ssdc and self.said_filter is not None and self.coupling_neck is not None:  # 启用SS-DC时执行解耦与耦合
            f_inv_list, f_ds_list = self.said_filter(features)  # 调用SAID滤波器获取域不变与域特异特征列表
            coupled_feats, ssdc_stats = self.coupling_neck(features, f_inv_list, f_ds_list)  # 将解耦特征送入耦合颈部融合并返回统计量
            coupled_tuple = tuple(coupled_feats)  # 将融合后的特征转换为元组以符合检测头输入要求
            self.ssdc_feature_cache[storage_key] = {  # 为当前分支缓存SS-DC相关特征供损失计算
                'raw': features,  # 缓存原始未耦合的FPN特征
                'inv': tuple(f_inv_list),  # 缓存域不变特征列表
                'ds': tuple(f_ds_list),  # 缓存域特异特征列表
                'coupled': coupled_tuple,  # 缓存耦合后的特征结果
                'stats': ssdc_stats,  # 缓存耦合过程的统计信息如域特异占比
            }  # 完成缓存字典的构造
            x = coupled_tuple  # 将耦合结果作为后续检测模块输入
        else:  # 当未启用SS-DC或模块未构建时直接返回原始特征
            self.ssdc_feature_cache[storage_key] = {  # 缓存基础特征以保持接口一致
                'raw': features,  # 缓存原始FPN特征
                'inv': None,  # 占位符表明未计算域不变特征
                'ds': None,  # 占位符表明未计算域特异特征
                'coupled': features,  # 将原始特征视作耦合结果保持一致性
                'stats': None,  # 占位符表明无附加统计量
            }  # 完成基础缓存的构造
            x = features  # 将原始特征作为输出
        return x

    def _forward(self, batch_inputs: Tensor,
                 batch_data_samples: SampleList) -> tuple:
        """Network forward process. Usually includes backbone, neck and head
        forward without any post-processing.

        Args:
            batch_inputs (Tensor): Inputs with shape (N, C, H, W).
            batch_data_samples (list[:obj:`DetDataSample`]): Each item contains
                the meta information of each image and corresponding
                annotations.

        Returns:
            tuple: A tuple of features from ``rpn_head`` and ``roi_head``
            forward.
        """
        self.ssdc_feature_cache.clear()  # 在正式前向推理前清空SS-DC特征缓存以防止跨调用残留影响当前特征流
        results = ()
        x = self.extract_feat(batch_inputs)

        if self.with_rpn:
            rpn_results_list = self.rpn_head.predict(
                x, batch_data_samples, rescale=False)
        else:
            assert batch_data_samples[0].get('proposals', None) is not None
            rpn_results_list = [
                data_sample.proposals for data_sample in batch_data_samples
            ]
        roi_outs = self.roi_head.forward(x, rpn_results_list,
                                         batch_data_samples)
        results = results + (roi_outs,)
        return results

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList,
             return_feature=False) -> dict:
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

        # 在每次计算损失前清空SS-DC特征缓存以避免跨batch残留导致错误配对
        ###########################################################################
        self.ssdc_feature_cache.clear()

        # Extract feature with mask input
        ###########################################################################
        if self.apply_auxiliary_branch:
            N, _, H, W = batch_inputs.shape
            ref_masks, ref_labels = bbox_to_mask(batch_data_samples, N, H, W, self.class_maps)
            x_w_ref = self.extract_feat(batch_inputs, ref_masks, ref_labels)
            x_wo_ref = self.extract_feat(batch_inputs)
        ###########################################################################
        else:
            x_wo_ref = self.extract_feat(batch_inputs)
            # 当未启用辅助分支时复制无参考分支的SS-DC缓存为参考分支防止后续读取缺失
            #######################################################################
            if 'noref' in self.ssdc_feature_cache:
                self.ssdc_feature_cache['ref'] = copy.deepcopy(self.ssdc_feature_cache['noref'])
            x_w_ref = x_wo_ref  # 当未启用辅助分支时将参考分支特征指向主分支特征以避免后续引用未定义

        losses = dict()

        
         # noref branch
        ###########################################################################
        # RPN forward and loss
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            rpn_data_samples = copy.deepcopy(batch_data_samples)
            # set cat_id of gt_labels to 0 in RPN
            for data_sample in rpn_data_samples:
                data_sample.gt_instances.labels = \
                    torch.zeros_like(data_sample.gt_instances.labels)

            rpn_losses, rpn_results_list_noref = self.rpn_head.loss_and_predict(
                x_wo_ref, rpn_data_samples, proposal_cfg=proposal_cfg)
            # avoid get same name with roi_head loss
            keys = rpn_losses.keys()
            for key in list(keys):
                if 'loss' in key and 'rpn' not in key:
                    rpn_losses[f'rpn_{key}'] = rpn_losses.pop(key)
            losses.update(rename_loss_dict('noref_', rpn_losses))
        else:
            assert batch_data_samples[0].get('proposals', None) is not None
            # use pre-defined proposals in InstanceData for the second stage
            # to extract ROI features.
            rpn_results_list_noref = [
                data_sample.proposals for data_sample in batch_data_samples
            ]

        roi_losses = self.roi_head.loss(x_wo_ref, rpn_results_list_noref,
                                        batch_data_samples)
        losses.update(rename_loss_dict('noref_', roi_losses))
        ###########################################################################
        
        # ref branch
        ###########################################################################
        # RPN forward and loss
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            rpn_data_samples = copy.deepcopy(batch_data_samples)
            # set cat_id of gt_labels to 0 in RPN
            for data_sample in rpn_data_samples:
                data_sample.gt_instances.labels = \
                    torch.zeros_like(data_sample.gt_instances.labels)

            rpn_losses, rpn_results_list_ref = self.rpn_head.loss_and_predict(
                x_w_ref, rpn_data_samples, proposal_cfg=proposal_cfg)
            # avoid get same name with roi_head loss
            keys = rpn_losses.keys()
            for key in list(keys):
                if 'loss' in key and 'rpn' not in key:
                    rpn_losses[f'rpn_{key}'] = rpn_losses.pop(key)
            losses.update(rename_loss_dict('ref_', rpn_losses))
        else:
            assert batch_data_samples[0].get('proposals', None) is not None
            # use pre-defined proposals in InstanceData for the second stage
            # to extract ROI features.
            rpn_results_list_ref = [
                data_sample.proposals for data_sample in batch_data_samples
            ]

        roi_losses = self.roi_head.loss(x_w_ref, rpn_results_list_ref,
                                        batch_data_samples)
        losses.update(rename_loss_dict('ref_', roi_losses))
        ##########################################################################

        # object-kd loss
        ##############################################################################################################
        if self.apply_auxiliary_branch:
            # Apply cross-kd in ROI head
            roi_losses_kd = self.roi_head_loss_with_kd(
                x_wo_ref, x_w_ref, rpn_results_list_ref, batch_data_samples)
            losses.update(roi_losses_kd)
        ##############################################################################################################

        # feature kd loss
        ##############################################################################################################
        if self.apply_auxiliary_branch:
            feature_loss = dict()
            feature_loss['pkd_feature_loss'] = 0
            for i, (x_wo, x_w) in enumerate(zip(x_wo_ref, x_w_ref)):
                layer_loss = self.loss_feature(x_wo, x_w)
                feature_loss['pkd_feature_loss'] += layer_loss/len(x_wo_ref)
            losses.update(feature_loss)
        ##############################################################################################################

        # ssdc loss
        ##############################################################################################################
        if self.training and self.enable_ssdc:  # 仅在训练阶段且开启SS-DC时计算额外损失
            cache_noref = self.ssdc_feature_cache.get('noref')  # 读取无参考分支的SS-DC缓存
            cache_ref = self.ssdc_feature_cache.get('ref')  # 读取有参考分支的SS-DC缓存
            if cache_noref is not None and cache_ref is not None:  # 确保两个分支缓存均存在
                ssdc_loss_dict = {}  # 初始化SS-DC损失累积字典
                if (self.loss_decouple is not None  # 确认解耦损失模块已构建
                        and cache_noref['inv'] is not None  # 确认无参考分支解耦域不变特征可用
                        and cache_noref['ds'] is not None):  # 确认无参考分支解耦域特异特征可用
                    decouple_noref = self.loss_decouple(  # 调用解耦损失模块计算无参考分支损失
                        cache_noref['raw'],  # 传入无参考分支原始特征供能量约束
                        cache_noref['inv'],  # 传入无参考分支域不变特征用于幂等与正交约束
                        cache_noref['ds'],  # 传入无参考分支域特异特征用于正交与能量约束
                        self.said_filter)  # 传入SAID模块用于幂等性重计算
                    decouple_noref = rename_loss_dict(  # 重命名无参考解耦损失键避免冲突
                        'ssdc_decouple_noref_',  # 传入无参考解耦损失前缀
                        decouple_noref)  # 传入原始无参考解耦损失字典
                    decouple_noref = reweight_loss_dict(  # 根据配置调整无参考解耦损失权重
                        decouple_noref,  # 传入已重命名的无参考解耦损失字典
                        self.loss_decouple_weight)  # 传入无参考解耦损失总缩放系数
                    ssdc_loss_dict.update(decouple_noref)  # 合并无参考解耦损失
                if (self.loss_decouple is not None  # 确认解耦损失模块已构建
                        and cache_ref['inv'] is not None  # 确认有参考分支解耦域不变特征可用
                        and cache_ref['ds'] is not None):  # 确认有参考分支解耦域特异特征可用
                    decouple_ref = self.loss_decouple(  # 调用解耦损失模块计算有参考分支损失
                        cache_ref['raw'],  # 传入有参考分支原始特征供能量约束
                        cache_ref['inv'],  # 传入有参考分支域不变特征用于幂等与正交约束
                        cache_ref['ds'],  # 传入有参考分支域特异特征用于正交与能量约束
                        self.said_filter)  # 传入SAID模块用于幂等性重计算
                    decouple_ref = rename_loss_dict(  # 重命名有参考解耦损失键避免冲突
                        'ssdc_decouple_ref_',  # 传入有参考解耦损失前缀
                        decouple_ref)  # 传入原始有参考解耦损失字典
                    decouple_ref = reweight_loss_dict(  # 根据配置调整有参考解耦损失权重
                        decouple_ref,  # 传入已重命名的有参考解耦损失字典
                        self.loss_decouple_weight)  # 传入有参考解耦损失总缩放系数
                    ssdc_loss_dict.update(decouple_ref)  # 合并有参考解耦损失
                if (self.loss_couple is not None  # 确认耦合损失模块已构建
                        and cache_noref['coupled'] is not None  # 确认无参考分支耦合特征可用
                        and cache_ref['inv'] is not None):  # 确认有参考分支域不变特征可用
                    couple_loss = self.loss_couple(  # 调用耦合损失模块计算跨分支对齐损失
                        cache_noref['coupled'],  # 传入无参考分支耦合后特征用于对齐监督
                        cache_ref['inv'],  # 传入有参考分支域不变特征作为教师信号
                        cache_noref.get('stats', {}))  # 传入无参考分支统计信息以约束域特异比例
                    couple_loss = rename_loss_dict(  # 重命名耦合损失键避免冲突
                        'ssdc_couple_',  # 传入耦合损失前缀
                        couple_loss)  # 传入原始耦合损失字典
                    couple_loss = reweight_loss_dict(  # 根据配置调整耦合损失权重
                        couple_loss,  # 传入已重命名的耦合损失字典
                        self.loss_couple_weight)  # 传入耦合损失总缩放系数
                    ssdc_loss_dict.update(couple_loss)  # 合并耦合损失
                losses.update(ssdc_loss_dict)  # 将SS-DC相关损失写入总损失字典
        ##############################################################################################################

        if not return_feature:
            return losses
        else:
            return losses, x_wo_ref

    def roi_head_loss_with_kd(self,
                              x_wo_ref, x_w_ref, rpn_results_list_ref, batch_data_samples):
        assert len(rpn_results_list_ref) == len(batch_data_samples)
         
        outputs = unpack_gt_instances(batch_data_samples)
        batch_gt_instances, batch_gt_instances_ignore, _ = outputs
        roi_head = self.roi_head

        # assign gts and sample proposals
        num_imgs = len(batch_data_samples)
        sampling_results_ref = []
        for i in range(num_imgs):
            # rename rpn_results.bboxes to rpn_results.priors
            rpn_results = rpn_results_list_ref[i]
            # rpn_results.priors = rpn_results.pop('bboxes')

            assign_result = roi_head.bbox_assigner.assign(
                rpn_results, batch_gt_instances[i],
                batch_gt_instances_ignore[i])
            sampling_result = roi_head.bbox_sampler.sample(
                assign_result,
                rpn_results,
                batch_gt_instances[i],
                feats=[lvl_feat[i][None] for lvl_feat in x_w_ref])
            sampling_results_ref.append(sampling_result)
                      
        losses = dict()
        # bbox head loss
        if roi_head.with_bbox:
            bbox_results = self.bbox_loss_with_kd(
                x_wo_ref, x_w_ref, sampling_results_ref)
            losses.update(bbox_results['loss_bbox_kd'])

        return losses

    def bbox_loss_with_kd(self, x_wo_ref, x_w_ref, sampling_results_ref):
        rois_ref = bbox2roi([res.priors for res in sampling_results_ref])

        roi_head = self.roi_head
        ref_bbox_results = roi_head._bbox_forward(x_w_ref, rois_ref)
        reused_bbox_results = roi_head._bbox_forward(x_wo_ref, rois_ref)

        losses_kd = dict()
        # classification KD
        reused_cls_scores = reused_bbox_results['cls_score']
        ref_cls_scores = ref_bbox_results['cls_score']
        avg_factor = sum([res.avg_factor for res in sampling_results_ref])
        loss_cls_kd = self.loss_cls_kd(
            ref_cls_scores,
            reused_cls_scores,
            avg_factor=avg_factor)
        losses_kd['loss_cls_kd'] = loss_cls_kd

        # l1 loss
        num_classes = roi_head.bbox_head.num_classes
        reused_bbox_preds = reused_bbox_results['bbox_pred']
        ref_bbox_preds = ref_bbox_results['bbox_pred']
        ref_cls_scores = ref_cls_scores.softmax(dim=1)[:, :num_classes]
        reg_weights, reg_distill_idx = ref_cls_scores.max(dim=1)
        if not roi_head.bbox_head.reg_class_agnostic:
            reg_distill_idx = reg_distill_idx[:, None, None].repeat(1, 1, 4)
            reused_bbox_preds = reused_bbox_preds.reshape(-1, num_classes, 4)
            reused_bbox_preds = reused_bbox_preds.gather(
                dim=1, index=reg_distill_idx)
            reused_bbox_preds = reused_bbox_preds.squeeze(1)
            ref_bbox_preds = ref_bbox_preds.reshape(-1, num_classes, 4)
            ref_bbox_preds = ref_bbox_preds.gather(
                dim=1, index=reg_distill_idx)
            ref_bbox_preds = ref_bbox_preds.squeeze(1)

        loss_reg_kd = self.loss_reg_kd(
            ref_bbox_preds,
            reused_bbox_preds,
            weight=reg_weights[:, None],
            avg_factor=reg_weights.sum() * 4)
        losses_kd['loss_reg_kd'] = loss_reg_kd

        bbox_results = dict()
        for key, value in ref_bbox_results.items():
            bbox_results['ref_' + key] = value
        for key, value in reused_bbox_results.items():
            bbox_results['reused_' + key] = value
        bbox_results['loss_bbox_kd'] = losses_kd
        return bbox_results

    def predict(self,
                batch_inputs: Tensor,
                batch_data_samples: SampleList,
                rescale: bool = True,
                return_feature=False):
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

        assert self.with_bbox, 'Bbox head must be implemented.'
        self.ssdc_feature_cache.clear()  # 在推理阶段开始时清空SS-DC缓存避免训练阶段遗留数据干扰预测
        x = self.extract_feat(batch_inputs)
        # If there are no pre-defined proposals, use RPN to get proposals
        if batch_data_samples[0].get('proposals', None) is None:
            rpn_results_list = self.rpn_head.predict(
                x, batch_data_samples, rescale=False)
        else:
            rpn_results_list = [
                data_sample.proposals for data_sample in batch_data_samples
            ]

        results_list = self.roi_head.predict(
            x, rpn_results_list, batch_data_samples, rescale=rescale)

        batch_data_samples = self.add_pred_to_datasample(
            batch_data_samples, results_list)
        if not return_feature:
            return batch_data_samples
        else:
            return batch_data_samples, x
