"""定义插入SS-DC流程且区分高层/低层耦合策略的Faster R-CNN检测器。"""
import copy  # 导入copy模块用于安全地深拷贝配置字典避免引用共享
from typing import Any, Dict, List, Optional, Sequence  # 导入类型注解以提高可读性并辅助静态检查

import torch  # 导入torch模块以便张量运算与参数注册
import torch.nn as nn  # 导入torch.nn以便定义卷积与可学习参数容器
from torch import Tensor  # 从PyTorch中导入Tensor类型便于类型标注

from mmdet.registry import MODELS  # 导入MMDetection注册表以便注册与构建自定义模块
from mmdet.models.detectors import FasterRCNN  # 导入基础Faster R-CNN实现以便继承
from mmdet.models.ssdc.said_filter_bank import SAIDFilterBank  # 导入SAID滤波器用于频域分解
from mmdet.models.ssdc.ss_coupling import SSDCouplingNeck  # 导入SS-DC耦合颈部模块用于特征融合
from mmdet.models.losses.ssdc_losses import LossCouple, LossDecouple  # 导入SS-DC相关损失实现供外部访问


@MODELS.register_module()  # 将自定义检测器注册到MMDetection框架以支持配置化构建
class SSDCFasterRCNN(FasterRCNN):  # 在标准Faster R-CNN上扩展区分高层/低层的光谱-空间解耦与耦合逻辑
    def __init__(self,
                 enable_ssdc: bool = False,  # 控制是否启用SS-DC流程的布尔开关
                 ssdc_cfg: Optional[Dict[str, Any]] = None,  # 承载SAID与耦合等子模块配置的字典
                 **kwargs) -> None:  # 透传其他关键字参数给基类构造函数保持接口一致
        self.enable_ssdc = bool(enable_ssdc)  # 提前缓存开关状态以便在父类初始化后使用
        self.ssdc_cfg: Dict[str, Any] = (  # 初始化SS-DC配置字典并深拷贝外部输入
            copy.deepcopy(ssdc_cfg) if isinstance(ssdc_cfg, dict) else {}  # 仅当传入为字典时深拷贝否则回退为空
        )  # 结束配置初始化括号
        self.ssdc_feature_cache: Dict[str, Any] = {}  # 初始化特征缓存字典供包装器或损失计算读取
        self.said_filter: Optional[SAIDFilterBank] = None  # 初始化SAID滤波器引用占位符
        self.coupling_neck: Optional[SSDCouplingNeck] = None  # 初始化耦合颈部引用占位符
        self.loss_decouple: Optional[LossDecouple] = None  # 初始化解耦损失模块引用占位符
        self.loss_couple: Optional[LossCouple] = None  # 初始化耦合损失模块引用占位符
        self.ssdc_all_level_names: List[str] = []  # 记录全部FPN层级名称以便索引映射
        self.ssdc_heavy_levels: List[str] = []  # 记录需要耦合注意力的高层名称列表
        self.ssdc_heavy_indices: List[int] = []  # 记录高层在特征序列中的全局索引
        self.ssdc_light_indices: List[int] = []  # 记录低层在特征序列中的全局索引
        self.ssdc_light_convs: Optional[nn.ModuleList] = None  # 初始化低层注意力卷积容器占位符
        self.ssdc_alpha: Optional[nn.Parameter] = None  # 初始化低层融合权重参数占位符
        self.ssdc_skip_local_loss: bool = bool(self.ssdc_cfg.get('skip_local_loss', False))  # 读取跳过本地SS-DC损失的布尔开关
        self.use_coupled_feature: bool = bool(self.ssdc_cfg.get('use_coupled_feature', True))  # 决定下游检测头是否使用耦合后的特征
        super().__init__(**kwargs)  # 调用父类构造函数构建骨干、颈部、RPN与ROI头等标准组件
        if self.enable_ssdc:  # 仅在显式开启SS-DC时才实例化附加模块
            said_cfg = copy.deepcopy(self.ssdc_cfg.get('said', {}))  # 读取并拷贝SAID滤波器配置
            coupling_cfg = copy.deepcopy(self.ssdc_cfg.get('coupling', {}))  # 读取并拷贝耦合颈部配置
            loss_decouple_cfg = copy.deepcopy(self.ssdc_cfg.get('loss_decouple', {}))  # 读取并拷贝解耦损失配置
            loss_couple_cfg = copy.deepcopy(self.ssdc_cfg.get('loss_couple', {}))  # 读取并拷贝耦合损失配置
            num_feature_levels: Optional[int] = None  # 初始化全部FPN层级数量占位符
            level_prefix = coupling_cfg.get('level_prefix', 'P')  # 读取层级前缀缺省为P
            start_level = coupling_cfg.get('start_level', 2)  # 读取起始层索引缺省为2
            explicit_levels = said_cfg.get('levels', None)  # 尝试读取外部显式指定的层级名称
            if isinstance(explicit_levels, (list, tuple)) and explicit_levels:  # 当层级名称被直接提供时优先使用
                all_level_names = list(explicit_levels)  # 深拷贝层级名称列表以避免外部修改
                num_feature_levels = len(all_level_names)  # 同步记录层级数量
            else:  # 未显式提供层级名称时依据颈部配置推断
                num_feature_levels = coupling_cfg.get('num_feature_levels', None)  # 优先读取耦合配置中的层级数
                has_neck_levels = hasattr(self, 'neck') and hasattr(self.neck, 'num_outs')  # 记录FPN是否声明输出层数
                if num_feature_levels is None and has_neck_levels:  # 未显式指定时优先使用FPN的num_outs
                    declared_levels = getattr(self.neck, 'num_outs', None)  # 安全地获取FPN层级数量
                    if isinstance(declared_levels, int) and declared_levels > 0:  # 确认读取到合法整数后采用
                        num_feature_levels = declared_levels  # 使用FPN声明的输出层数
                if num_feature_levels is None and hasattr(self, 'neck') and hasattr(self.neck, 'out_channels'):
                    neck_channels = getattr(self.neck, 'out_channels')  # 尝试从FPN输出通道推断层数
                    if isinstance(neck_channels, (list, tuple)) and len(neck_channels) > 0:  # 当通道列表可用时使用其长度
                        num_feature_levels = len(neck_channels)  # 以通道列表长度作为层级数量
                if num_feature_levels is None:  # 兜底回退到典型的四层FPN配置
                    num_feature_levels = len(said_cfg.get('levels', ('P2', 'P3', 'P4', 'P5')))  # 根据默认层级名称数量推断
                all_level_names = [  # 按照前缀与起始索引生成全量层级名称列表
                    f"{level_prefix}{start_level + idx}" for idx in range(num_feature_levels)  # 遍历层编号生成名称
                ]  # 结束层级名称列表构造
            said_cfg['levels'] = all_level_names  # 强制将SAID层级配置覆盖为全量层级名称
            heavy_levels_cfg = coupling_cfg.get('levels', [])  # 读取外部指定需重耦合的高层列表
            heavy_levels_cfg = list(heavy_levels_cfg) if isinstance(heavy_levels_cfg, (list, tuple)) else []  # 确保高层列表类型正确
            if heavy_levels_cfg and all(level in all_level_names for level in heavy_levels_cfg):  # 当配置合法且层级存在时直接采用
                heavy_levels = heavy_levels_cfg  # 使用用户显式指定的高层列表
            else:  # 未指定或不合法时采用默认高层集合
                heavy_levels = [name for name in all_level_names if name in ('P4', 'P5')]  # 默认使用P4与P5作为高层
                if not heavy_levels:  # 若默认集合仍为空则使用最高层兜底
                    heavy_levels = [all_level_names[-1]]  # 保证至少存在一个高层供耦合使用
            heavy_indices = [all_level_names.index(level) for level in heavy_levels]  # 计算高层在特征序列中的索引
            light_indices = [idx for idx in range(len(all_level_names)) if idx not in heavy_indices]  # 低层索引为全集减去高层索引
            self.ssdc_all_level_names = all_level_names  # 缓存全量层级名称便于前向阶段使用
            self.ssdc_heavy_levels = heavy_levels  # 缓存高层名称列表
            self.ssdc_heavy_indices = heavy_indices  # 缓存高层索引列表
            self.ssdc_light_indices = light_indices  # 缓存低层索引列表
            coupling_cfg['levels'] = heavy_levels  # 仅将高层名称传递给耦合颈部以执行多头注意力
            coupling_cfg['num_feature_levels'] = len(heavy_levels)  # 同步耦合颈部使用的层级数量
            inferred_channels: Optional[int] = None  # 初始化耦合颈部输入通道占位符
            if hasattr(self, 'neck') and hasattr(self.neck, 'out_channels'):
                neck_out_channels = getattr(self.neck, 'out_channels')  # 读取颈部输出通道参数
                if isinstance(neck_out_channels, int) and neck_out_channels > 0:  # 直接使用整数通道数
                    inferred_channels = neck_out_channels  # 记录推断的通道数
                elif isinstance(neck_out_channels, (list, tuple)) and neck_out_channels:  # 当为列表时尽量使用首个通道数
                    first_channel = neck_out_channels[0]  # 读取首层通道数作为统一输入尺寸
                    if isinstance(first_channel, int) and first_channel > 0:  # 确认首层通道数合法
                        inferred_channels = first_channel  # 使用首层通道数作为默认输入通道
            if inferred_channels is None:  # 若无法从颈部推断则使用常见默认值256
                inferred_channels = 256  # 设置耦合颈部输入通道的兜底值
            coupling_cfg.setdefault('in_channels', inferred_channels)  # 确保耦合颈部具备明确的输入通道参数
            num_levels = len(all_level_names)  # 记录全量层级数量以便构建低层模块
            fpn_channels = getattr(self.neck, 'out_channels', inferred_channels)  # 尝试从FPN读取输出通道
            if isinstance(fpn_channels, (list, tuple)) and fpn_channels:  # 当为列表时使用首层通道数作为统一通道
                fpn_channels = fpn_channels[0]  # 统一低层卷积输入输出通道
            self.ssdc_light_convs = nn.ModuleList()  # 实例化存放低层注意力卷积的容器
            for _ in range(num_levels):  # 遍历每个层级创建对应卷积
                self.ssdc_light_convs.append(  # 向容器添加1x1卷积
                    nn.Conv2d(fpn_channels, fpn_channels, kernel_size=1, padding=0)  # 使用轻量1x1卷积生成A_inv
                )  # 结束卷积添加
            self.ssdc_alpha = nn.Parameter(torch.ones(num_levels) * 0.5)  # 为每个层级初始化可学习融合权重0.5
            if 'type' in said_cfg:  # 当配置包含type字段时使用注册表构建SAID滤波器
                self.said_filter = MODELS.build(said_cfg)  # 通过注册表创建SAID滤波器实例
            else:  # 否则直接实例化默认SAID实现
                self.said_filter = SAIDFilterBank(**said_cfg)  # 使用关键字参数构建SAID滤波器
            if 'type' in coupling_cfg:  # 当耦合配置包含type字段时使用注册表构建
                self.coupling_neck = MODELS.build(coupling_cfg)  # 通过注册表创建耦合颈部实例
            else:  # 否则直接实例化默认耦合颈部
                self.coupling_neck = SSDCouplingNeck(**coupling_cfg)  # 使用关键字参数构建耦合颈部
            if loss_decouple_cfg:  # 当提供解耦损失配置时通过注册表构建
                self.loss_decouple = MODELS.build(loss_decouple_cfg)  # 注册表创建自定义解耦损失
            else:  # 否则使用默认LossDecouple实现
                self.loss_decouple = LossDecouple()  # 实例化默认解耦损失模块
            if loss_couple_cfg:  # 当提供耦合损失配置时通过注册表构建
                self.loss_couple = MODELS.build(loss_couple_cfg)  # 注册表创建自定义耦合损失
            else:  # 否则使用默认LossCouple实现
                self.loss_couple = LossCouple()  # 实例化默认耦合损失模块

    def extract_feat(self,
                     img: Tensor,  # 输入图像张量形状为(N,C,H,W)
                     img_metas: Optional[List[dict]] = None,  # 可选的图像元信息列表保留与DiffusionDetector一致的签名
                     is_teacher: bool = False,  # 指示当前前向是否来自教师分支供缓存记录
                     is_source: bool = True,  # 指示当前样本是否来自源域供外部统计使用
                     current_iter: Optional[int] = None,  # 可选的迭代索引用于保持接口兼容
                     **kwargs) -> Sequence[Tensor]:  # 返回多尺度特征列表或元组
        features = self.backbone(img)  # 通过骨干网络提取初步特征
        if self.with_neck:  # 若定义了特征金字塔则继续处理
            features = self.neck(features)  # 将骨干输出送入FPN获得多尺度特征
        feature_tuple = tuple(features) if isinstance(features, (list, tuple)) else (features,)  # 统一封装为元组便于遍历
        storage_key = 'noref'  # 当前实现仅缓存无参考分支，保持与DomainAdaptationDetector接口一致
        if (not self.enable_ssdc) or (self.said_filter is None) or (self.coupling_neck is None):  # 未启用SS-DC时直接返回原始特征
            self.ssdc_feature_cache = {  # 仍按期望的分支键存储基础特征以保持兼容
                storage_key: {
                    'raw': feature_tuple,  # 缓存原始FPN特征
                    'inv': None,  # 未启用SS-DC时域不变特征为空
                    'ds': None,  # 未启用SS-DC时域特异特征为空
                    'coupled': feature_tuple,  # 耦合特征退化为原始特征
                    'stats': None,  # 无附加统计信息
                    'is_teacher': bool(is_teacher),  # 记录教师标记以便调试
                    'is_source': bool(is_source),  # 记录域来源标记
                    'current_iter': current_iter  # 记录前向迭代索引便于追踪
                }
            }
            return feature_tuple  # 返回未处理的特征以供后续检测头使用
        f_inv_full, f_ds_full = self.said_filter(list(feature_tuple))  # 使用SAID滤波器生成域不变与域特异特征
        heavy_indices = getattr(self, 'ssdc_heavy_indices', list(range(len(feature_tuple))))  # 获取需要耦合的高层索引
        heavy_raw = [feature_tuple[idx] for idx in heavy_indices]  # 提取对应高层的原始特征
        heavy_inv = [f_inv_full[idx] for idx in heavy_indices]  # 提取对应高层的域不变特征
        heavy_ds = [f_ds_full[idx] for idx in heavy_indices]  # 提取对应高层的域特异特征
        coupled_heavy, ssdc_stats = self.coupling_neck(heavy_raw, heavy_inv, heavy_ds)  # 在高层执行耦合颈部的多头注意力
        if not isinstance(coupled_heavy, (list, tuple)):  # 确保耦合输出统一为列表格式
            coupled_heavy = [coupled_heavy]  # 当返回单个张量时封装为列表
        full_inv = list(f_inv_full)  # 复制全量域不变特征列表
        full_ds = list(f_ds_full)  # 复制全量域特异特征列表
        full_coupled = list(feature_tuple)  # 初始化耦合特征为原始特征
        for local_idx, global_idx in enumerate(heavy_indices):  # 遍历高层索引将耦合结果写回对应位置
            full_coupled[global_idx] = coupled_heavy[local_idx]  # 更新高层耦合特征
        light_indices = getattr(self, 'ssdc_light_indices', [])  # 获取低层索引列表
        for li in light_indices:  # 遍历每个低层执行DI注意力融合
            F_B = feature_tuple[li]  # 读取原始FPN特征
            F_inv = f_inv_full[li]  # 读取SAID解耦得到的域不变特征
            if self.ssdc_light_convs is None or self.ssdc_alpha is None:  # 若低层模块未正确初始化则安全回退
                full_coupled[li] = F_B  # 直接保留原始特征避免异常
                continue  # 跳过后续融合逻辑
            A_inv = self.ssdc_light_convs[li](F_inv)  # 使用1x1卷积生成低层注意力图
            alpha_l = torch.clamp(self.ssdc_alpha[li], 0.0, 1.0)  # 将可学习融合权重限制在[0,1]
            F_out = alpha_l * A_inv + (1.0 - alpha_l) * F_B  # 依据公式F_out=alpha*A_inv+(1-alpha)*F_B进行融合
            full_coupled[li] = F_out  # 写回融合后的低层特征
        coupled_tuple = tuple(full_coupled)  # 将完整耦合特征列表封装为元组
        self.ssdc_feature_cache = {  # 按分支键缓存全部中间结果供域自适应包装器计算SS-DC损失
            storage_key: {
                'raw': feature_tuple,  # 缓存原始FPN特征
                'inv': tuple(full_inv),  # 缓存域不变特征
                'ds': tuple(full_ds),  # 缓存域特异特征
                'coupled': coupled_tuple,  # 缓存耦合后特征
                'stats': ssdc_stats,  # 缓存耦合阶段统计信息
                'is_teacher': bool(is_teacher),  # 记录当前缓存是否来自教师分支
                'is_source': bool(is_source),  # 记录当前缓存对应的数据域标识
                'current_iter': current_iter  # 记录前向迭代索引便于调试
            }
        }
        return coupled_tuple if self.use_coupled_feature else feature_tuple  # 按配置决定返回耦合或原始特征