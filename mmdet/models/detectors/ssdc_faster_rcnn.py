"""中文注释：定义插入SS-DC流程的Faster R-CNN检测器。"""
import copy  # 中文注释：导入copy模块用于安全地深拷贝配置字典避免引用共享
from typing import Any, Dict, List, Optional, Sequence  # 中文注释：导入类型注解以提高可读性并辅助静态检查

from torch import Tensor  # 中文注释：从PyTorch中导入Tensor类型便于类型标注

from mmdet.registry import MODELS  # 中文注释：导入MMDetection注册表以便注册与构建自定义模块
from mmdet.models.detectors import FasterRCNN  # 中文注释：导入基础Faster R-CNN实现以便继承
from mmdet.models.ssdc.said_filter_bank import SAIDFilterBank  # 中文注释：导入SAID滤波器用于频域分解
from mmdet.models.ssdc.ss_coupling import SSDCouplingNeck  # 中文注释：导入SS-DC耦合颈部模块用于特征融合
from mmdet.models.losses.ssdc_losses import LossCouple, LossDecouple  # 中文注释：导入SS-DC相关损失实现供外部访问


@MODELS.register_module()  # 中文注释：将自定义检测器注册到MMDetection框架以支持配置化构建
class SSDCFasterRCNN(FasterRCNN):  # 中文注释：在标准Faster R-CNN上扩展光谱-空间解耦与耦合逻辑
    def __init__(self,
                 enable_ssdc: bool = False,  # 中文注释：控制是否启用SS-DC流程的布尔开关
                 ssdc_cfg: Optional[Dict[str, Any]] = None,  # 中文注释：承载SAID与耦合等子模块配置的字典
                 **kwargs) -> None:  # 中文注释：透传其他关键字参数给基类构造函数保持接口一致
        self.enable_ssdc = bool(enable_ssdc)  # 中文注释：提前缓存开关状态以便在父类初始化后使用
        self.ssdc_cfg: Dict[str, Any] = (  # 中文注释：初始化SS-DC配置字典并深拷贝外部输入
            copy.deepcopy(ssdc_cfg) if isinstance(ssdc_cfg, dict) else {}  # 中文注释：仅当传入为字典时深拷贝否则回退为空
        )  # 中文注释：结束配置初始化括号
        self.ssdc_feature_cache: Dict[str, Any] = {}  # 中文注释：初始化特征缓存字典供包装器或损失计算读取
        self.said_filter: Optional[SAIDFilterBank] = None  # 中文注释：初始化SAID滤波器引用占位符
        self.coupling_neck: Optional[SSDCouplingNeck] = None  # 中文注释：初始化耦合颈部引用占位符
        self.loss_decouple: Optional[LossDecouple] = None  # 中文注释：初始化解耦损失模块引用占位符
        self.loss_couple: Optional[LossCouple] = None  # 中文注释：初始化耦合损失模块引用占位符
        self.ssdc_skip_local_loss: bool = bool(self.ssdc_cfg.get('skip_local_loss', False))  # 中文注释：读取跳过本地SS-DC损失的布尔开关
        self.use_coupled_feature: bool = bool(self.ssdc_cfg.get('use_coupled_feature', True))  # 中文注释：决定下游检测头是否使用耦合后的特征
        super().__init__(**kwargs)  # 中文注释：调用父类构造函数构建骨干、颈部、RPN与ROI头等标准组件
        if self.enable_ssdc:  # 中文注释：仅在显式开启SS-DC时才实例化附加模块
            said_cfg = copy.deepcopy(self.ssdc_cfg.get('said', {}))  # 中文注释：读取并拷贝SAID滤波器配置
            coupling_cfg = copy.deepcopy(self.ssdc_cfg.get('coupling', {}))  # 中文注释：读取并拷贝耦合颈部配置
            loss_decouple_cfg = copy.deepcopy(self.ssdc_cfg.get('loss_decouple', {}))  # 中文注释：读取并拷贝解耦损失配置
            loss_couple_cfg = copy.deepcopy(self.ssdc_cfg.get('loss_couple', {}))  # 中文注释：读取并拷贝耦合损失配置
            num_feature_levels = coupling_cfg.get('num_feature_levels', None)  # 中文注释：尝试读取外部显式设置的特征层数
            has_neck_levels = hasattr(self, 'neck') and hasattr(self.neck, 'num_outs')  # 中文注释：记录FPN是否声明输出层数
            if num_feature_levels is None and has_neck_levels:  # 中文注释：未显式指定时优先使用FPN的num_outs
                declared_levels = getattr(self.neck, 'num_outs', None)  # 中文注释：安全地获取FPN层级数量
                if isinstance(declared_levels, int) and declared_levels > 0:  # 中文注释：确认读取到合法整数后采用
                    num_feature_levels = declared_levels  # 中文注释：使用FPN声明的输出层数
            if num_feature_levels is None and hasattr(self, 'neck') and hasattr(self.neck, 'out_channels'):
                neck_channels = getattr(self.neck, 'out_channels')  # 中文注释：尝试从FPN输出通道推断层数
                if isinstance(neck_channels, (list, tuple)) and len(neck_channels) > 0:  # 中文注释：当通道列表可用时使用其长度
                    num_feature_levels = len(neck_channels)  # 中文注释：以通道列表长度作为层级数量
            if num_feature_levels is None:  # 中文注释：兜底回退到典型的四层FPN配置
                num_feature_levels = len(said_cfg.get('levels', ('P2', 'P3', 'P4', 'P5')))  # 中文注释：根据默认层级名称数量推断
            level_prefix = coupling_cfg.get('level_prefix', 'P')  # 中文注释：读取层级前缀缺省为P
            start_level = coupling_cfg.get('start_level', 2)  # 中文注释：读取起始层索引缺省为2
            level_names = [  # 中文注释：按照前缀与起始索引生成层级名称列表
                f"{level_prefix}{start_level + idx}" for idx in range(num_feature_levels)  # 中文注释：遍历层编号生成名称
            ]  # 中文注释：结束层级名称列表构造
            coupling_cfg.setdefault('levels', level_names)  # 中文注释：将生成的层级名称写回耦合配置确保内部一致
            coupling_cfg.setdefault('num_feature_levels', num_feature_levels)  # 中文注释：将推断出的层级数量写回配置
            said_cfg.setdefault('levels', level_names)  # 中文注释：同步层级名称到SAID配置避免维度不匹配
            inferred_channels: Optional[int] = None  # 中文注释：初始化耦合颈部输入通道占位符
            if hasattr(self, 'neck') and hasattr(self.neck, 'out_channels'):
                neck_out_channels = getattr(self.neck, 'out_channels')  # 中文注释：读取颈部输出通道参数
                if isinstance(neck_out_channels, int) and neck_out_channels > 0:  # 中文注释：直接使用整数通道数
                    inferred_channels = neck_out_channels  # 中文注释：记录推断的通道数
                elif isinstance(neck_out_channels, (list, tuple)) and neck_out_channels:  # 中文注释：当为列表时尽量使用首个通道数
                    first_channel = neck_out_channels[0]  # 中文注释：读取首层通道数作为统一输入尺寸
                    if isinstance(first_channel, int) and first_channel > 0:  # 中文注释：确认首层通道数合法
                        inferred_channels = first_channel  # 中文注释：使用首层通道数作为默认输入通道
            if inferred_channels is None:  # 中文注释：若无法从颈部推断则使用常见默认值256
                inferred_channels = 256  # 中文注释：设置耦合颈部输入通道的兜底值
            coupling_cfg.setdefault('in_channels', inferred_channels)  # 中文注释：确保耦合颈部具备明确的输入通道参数
            if 'type' in said_cfg:  # 中文注释：当配置包含type字段时使用注册表构建SAID滤波器
                self.said_filter = MODELS.build(said_cfg)  # 中文注释：通过注册表创建SAID滤波器实例
            else:  # 中文注释：否则直接实例化默认SAID实现
                self.said_filter = SAIDFilterBank(**said_cfg)  # 中文注释：使用关键字参数构建SAID滤波器
            if 'type' in coupling_cfg:  # 中文注释：当耦合配置包含type字段时使用注册表构建
                self.coupling_neck = MODELS.build(coupling_cfg)  # 中文注释：通过注册表创建耦合颈部实例
            else:  # 中文注释：否则直接实例化默认耦合颈部
                self.coupling_neck = SSDCouplingNeck(**coupling_cfg)  # 中文注释：使用关键字参数构建耦合颈部
            if loss_decouple_cfg:  # 中文注释：当提供解耦损失配置时通过注册表构建
                self.loss_decouple = MODELS.build(loss_decouple_cfg)  # 中文注释：注册表创建自定义解耦损失
            else:  # 中文注释：否则使用默认LossDecouple实现
                self.loss_decouple = LossDecouple()  # 中文注释：实例化默认解耦损失模块
            if loss_couple_cfg:  # 中文注释：当提供耦合损失配置时通过注册表构建
                self.loss_couple = MODELS.build(loss_couple_cfg)  # 中文注释：注册表创建自定义耦合损失
            else:  # 中文注释：否则使用默认LossCouple实现
                self.loss_couple = LossCouple()  # 中文注释：实例化默认耦合损失模块

    def extract_feat(self,
                     img: Tensor,  # 中文注释：输入图像张量形状为(N,C,H,W)
                     img_metas: Optional[List[dict]] = None,  # 中文注释：可选的图像元信息列表保留与DiffusionDetector一致的签名
                     is_teacher: bool = False,  # 中文注释：指示当前前向是否来自教师分支供缓存记录
                     is_source: bool = True,  # 中文注释：指示当前样本是否来自源域供外部统计使用
                     current_iter: Optional[int] = None,  # 中文注释：可选的迭代索引用于保持接口兼容
                     **kwargs) -> Sequence[Tensor]:  # 中文注释：返回多尺度特征列表或元组
        features = self.backbone(img)  # 中文注释：通过骨干网络提取初步特征
        if self.with_neck:  # 中文注释：若定义了特征金字塔则继续处理
            features = self.neck(features)  # 中文注释：将骨干输出送入FPN获得多尺度特征
        feature_tuple = tuple(features) if isinstance(features, (list, tuple)) else (features,)  # 中文注释：统一封装为元组便于遍历
        storage_key = 'noref'  # 中文注释：当前实现仅缓存无参考分支，保持与DomainAdaptationDetector接口一致
        if not self.enable_ssdc or self.said_filter is None or self.coupling_neck is None:  # 中文注释：未启用SS-DC时直接返回原始特征
            self.ssdc_feature_cache = {  # 中文注释：仍按期望的分支键存储基础特征以保持兼容
                storage_key: {
                    'raw': feature_tuple,  # 中文注释：缓存原始FPN特征
                    'inv': None,  # 中文注释：未启用SS-DC时域不变特征为空
                    'ds': None,  # 中文注释：未启用SS-DC时域特异特征为空
                    'coupled': feature_tuple,  # 中文注释：耦合特征退化为原始特征
                    'stats': None,  # 中文注释：无附加统计信息
                    'is_teacher': bool(is_teacher),  # 中文注释：记录教师标记以便调试
                    'is_source': bool(is_source),  # 中文注释：记录域来源标记
                    'current_iter': current_iter  # 中文注释：记录前向迭代索引便于追踪
                }
            }
            return features  # 中文注释：返回未处理的特征以供后续检测头使用
        f_inv_list, f_ds_list = self.said_filter(feature_tuple)  # 中文注释：使用SAID滤波器生成域不变与域特异特征
        coupled_feats, ssdc_stats = self.coupling_neck(feature_tuple, f_inv_list, f_ds_list)  # 中文注释：耦合颈部融合特征并返回统计信息
        coupled_tuple = (  # 中文注释：将耦合输出统一封装为元组
            tuple(coupled_feats) if isinstance(coupled_feats, (list, tuple)) else (coupled_feats,)
        )  # 中文注释：结束耦合输出封装
        self.ssdc_feature_cache = {  # 中文注释：按分支键缓存全部中间结果供域自适应包装器计算SS-DC损失
            storage_key: {
                'raw': feature_tuple,  # 中文注释：缓存原始FPN特征
                'inv': tuple(f_inv_list),  # 中文注释：缓存域不变特征
                'ds': tuple(f_ds_list),  # 中文注释：缓存域特异特征
                'coupled': coupled_tuple,  # 中文注释：缓存耦合后特征
                'stats': ssdc_stats,  # 中文注释：缓存耦合阶段统计信息
                'is_teacher': bool(is_teacher),  # 中文注释：记录当前缓存是否来自教师分支
                'is_source': bool(is_source),  # 中文注释：记录当前缓存对应的数据域标识
                'current_iter': current_iter  # 中文注释：记录前向迭代索引便于调试
            }
        }
        return coupled_tuple if self.use_coupled_feature else feature_tuple  # 中文注释：按配置决定返回耦合或原始特征


# 中文注释：以下为快速自检示例，确保模块能被正常构建与前向
# import torch
# from mmdet.registry import MODELS
# dummy_backbone = dict(  # 中文注释：最小骨干配置
#     type='ResNet', depth=50, num_stages=4, out_indices=(0, 1, 2, 3), frozen_stages=1,
#     norm_cfg=dict(type='BN', requires_grad=True), norm_eval=True, style='pytorch',
#     init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50'))
# dummy_neck = dict(  # 中文注释：最小FPN配置
#     type='FPN', in_channels=[256, 512, 1024, 2048], out_channels=256, num_outs=5)
# dummy_rpn = dict(  # 中文注释：最小RPN配置
#     type='RPNHead', in_channels=256, feat_channels=256,
#     anchor_generator=dict(type='AnchorGenerator', scales=[8], ratios=[0.5, 1.0, 2.0], strides=[4, 8, 16, 32, 64]),
#     bbox_coder=dict(type='DeltaXYWHBBoxCoder', target_means=[0.0, 0.0, 0.0, 0.0], target_stds=[1.0, 1.0, 1.0, 1.0]),
#     loss_cls=dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
#     loss_bbox=dict(type='L1Loss', loss_weight=1.0))
# dummy_roi = dict(  # 中文注释：最小ROIHead配置
#     type='StandardRoIHead',
#     bbox_roi_extractor=dict(  # 中文注释：最小ROI特征提取配置
#         type='SingleRoIExtractor',
#         roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
#         out_channels=256,
#         featmap_strides=[4, 8, 16, 32]),
#     bbox_head=dict(  # 中文注释：最小bbox_head配置
#         type='Shared2FCBBoxHead', in_channels=256, fc_out_channels=1024, roi_feat_size=7, num_classes=1,
#         bbox_coder=dict(  # 中文注释：最小bbox编码器配置
#             type='DeltaXYWHBBoxCoder', target_means=[0.0, 0.0, 0.0, 0.0],
#             target_stds=[0.1, 0.1, 0.2, 0.2]),
#         reg_class_agnostic=False,
#         loss_cls=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
#         loss_bbox=dict(type='L1Loss', loss_weight=1.0)))
# dummy_train_cfg = dict(  # 中文注释：最小训练配置
#     rpn=dict(
#         assigner=dict(  # 中文注释：RPN分配器配置
#             type='MaxIoUAssigner', pos_iou_thr=0.7, neg_iou_thr=0.3, min_pos_iou=0.3,
#             match_low_quality=True, ignore_iof_thr=-1),
#         sampler=dict(type='RandomSampler', num=256, pos_fraction=0.5, neg_pos_ub=-1, add_gt_as_proposals=False),
#         allowed_border=-1, pos_weight=-1, debug=False),
#     rpn_proposal=dict(nms_pre=2000, max_per_img=1000, nms=dict(type='nms', iou_threshold=0.7), min_bbox_size=0),
#     rcnn=dict(
#         assigner=dict(  # 中文注释：RCNN分配器配置
#             type='MaxIoUAssigner', pos_iou_thr=0.5, neg_iou_thr=0.5, min_pos_iou=0.5,
#             match_low_quality=False, ignore_iof_thr=-1),
#         sampler=dict(type='RandomSampler', num=512, pos_fraction=0.25, neg_pos_ub=-1, add_gt_as_proposals=True),
#         pos_weight=-1, debug=False))
# dummy_test_cfg = dict(  # 中文注释：最小测试配置
#     rpn=dict(nms_pre=1000, max_per_img=1000, nms=dict(type='nms', iou_threshold=0.7), min_bbox_size=0),
#     rcnn=dict(score_thr=0.05, nms=dict(type='nms', iou_threshold=0.5), max_per_img=100))
# dummy_cfg = dict(type='SSDCFasterRCNN', enable_ssdc=False, backbone=dummy_backbone, neck=dummy_neck,
#                  rpn_head=dummy_rpn, roi_head=dummy_roi, train_cfg=dummy_train_cfg, test_cfg=dummy_test_cfg)
# model = MODELS.build(dummy_cfg)
# model.eval()
# dummy_input = torch.randn(1, 3, 224, 224)
# _ = model.extract_feat(dummy_input)
