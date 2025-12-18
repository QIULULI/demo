from typing import List, Sequence, Tuple  # 导入类型注解工具以提高代码可读性

import torch  # 导入PyTorch主库以支撑张量与傅里叶变换操作
import torch.nn as nn  # 导入神经网络模块以构建可学习的滤波器

from mmdet.registry import MODELS  # 导入MMDetection注册器以登记自定义模块


@MODELS.register_module()  # 注册SAID滤波器组以便在配置中构建
class SAIDFilterBank(nn.Module):  # 定义用于光谱自适应幂等分离的滤波器组
    def __init__(self,
                 levels: Sequence[str] = ('P2', 'P3', 'P4', 'P5'),  # 定义FPN层级名称以保持与特征列表对齐
                 share_mask: bool = True,  # 是否在所有层之间共享截止参数
                 init_cutoff: float = 0.6,  # 设置初始截止频率比例建议值0.6
                 temperature: float = 0.1,  # 设置软掩码温度以控制过渡平滑度
                 use_hard_mask: bool = False,  # 是否采用硬阈值掩码用于消融实验
                 invert_bands: bool = False  # 是否反转频带划分
                 ) -> None:  # 构造函数返回None
        super().__init__()  # 调用父类初始化保障nn.Module正确注册参数
        self.levels = list(levels)  # 保存层级名称列表以便索引与调试
        self.share_mask = share_mask  # 记录是否共享截止参数的布尔配置
        self.temperature = temperature  # 存储软掩码温度便于前向使用
        self.use_hard_mask = use_hard_mask  # 存储是否使用硬掩码
        self.invert_bands = invert_bands  # 是否反转频带划分
        init_cutoff = float(min(max(init_cutoff, 1e-3), 0.999))  # 限制初始截止比例避免sigmoid饱和
        cutoff_logit = torch.logit(torch.tensor(init_cutoff))  # 将初始截止比例转换为可训练logit
        if self.share_mask:  # 根据是否共享掩码决定参数形状
            self.cutoff_logit = nn.Parameter(cutoff_logit.unsqueeze(0))  # 创建共享logit参数
        else:
            repeated = cutoff_logit.repeat(len(self.levels))  # 为每个层级复制logit
            self.cutoff_logit = nn.Parameter(repeated)  # 创建逐层独立logit参数

    def _compute_radius_grid(self,
                             height: int,
                             width: int,
                             device: torch.device,
                             dtype: torch.dtype) -> torch.Tensor:  # 计算归一化半径网格用于频域掩码
        freq_y = torch.fft.fftfreq(height, device=device, dtype=dtype).view(height, 1)  # 计算垂直频率分布并调整形状
        freq_x = torch.fft.fftfreq(width, device=device, dtype=dtype).view(1, width)  # 计算水平频率分布并调整形状
        radius = torch.sqrt(freq_y.square() + freq_x.square())  # 通过勾股定理得到频率半径
        max_radius = radius.max().clamp_min(1e-6)  # 计算最大半径并避免除零
        norm_radius = radius / max_radius  # 将半径归一化到0到1之间
        return norm_radius  # 返回归一化半径网格

    def forward(self, feats: Sequence[torch.Tensor]) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:  # 定义前向传播返回低频与高频列表
        f_inv_list: List[torch.Tensor] = []  # 初始化存储低频重建特征的列表
        f_ds_list: List[torch.Tensor] = []  # 初始化存储高频重建特征的列表
        for level_idx, feat in enumerate(feats):  # 遍历每个FPN层级的输入特征
            height, width = feat.shape[-2:]  # 获取当前特征的空间尺寸
            device = feat.device  # 记录当前特征所在设备
            dtype = feat.dtype  # 记录当前特征的数据类型
            radius = self._compute_radius_grid(height, width, device, dtype)  # 生成该尺寸下的半径网格
            if self.share_mask:  # 根据配置选择掩码参数
                cutoff = torch.sigmoid(self.cutoff_logit[0])  # 若共享掩码则使用单一logit
            else:
                cutoff = torch.sigmoid(self.cutoff_logit[level_idx])  # 若不共享则选取对应层级logit
            if self.use_hard_mask:  # 判断是否启用硬掩码
                low_mask = (radius <= cutoff).to(dtype=feat.dtype)  # 使用阈值生成二值低频掩码
            else:
                scale = max(self.temperature, 1e-6)  # 确保温度非零避免数值问题
                low_mask = torch.sigmoid((cutoff - radius) / scale).to(dtype=feat.dtype)  # 使用软阈值构建平滑掩码
            high_mask = 1.0 - low_mask  # 高频掩码为一减低频掩码确保频谱划分
            spec = torch.fft.fft2(feat, norm='ortho')  # 对特征做二维傅里叶变换得到频谱
            low_spec = spec * low_mask.unsqueeze(0).unsqueeze(0)  # 应用低频掩码筛选频谱能量
            high_spec = spec * high_mask.unsqueeze(0).unsqueeze(0)  # 应用高频掩码获取高频频谱
            f_inv = torch.fft.ifft2(low_spec, norm='ortho').real  # 对低频频谱逆变换恢复空间域特征并取实部
            f_ds = torch.fft.ifft2(high_spec, norm='ortho').real  # 对高频频谱逆变换恢复空间域特征并取实部
            if self.invert_bands:  # 检查是否需要交换频带
                f_inv, f_ds = f_ds, f_inv  # 交换低频与高频特征以实现反转
            f_inv_list.append(f_inv)  # 记录当前层级的低频特征
            f_ds_list.append(f_ds)  # 记录当前层级的高频特征
        return f_inv_list, f_ds_list  # 返回低频与高频特征列表
