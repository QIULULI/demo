import math  # 导入数学库用于平方根等基本数学运算
from typing import Dict, List, Sequence, Tuple, Optional  # 导入类型注解提升可读性并包含可选类型

import torch  # 导入PyTorch主库以便进行张量运算
import torch.nn as nn  # 导入神经网络模块基类
import torch.nn.functional as F  # 导入常用函数模块以便使用softmax和dropout

from mmdet.registry import MODELS  # 导入MMDetection注册表以注册自定义模块


@MODELS.register_module()  # 将耦合颈部模块注册至MMDetection框架
class SSDCouplingNeck(nn.Module):  # 定义光谱-空间耦合颈部模块
    def __init__(self,
                 in_channels: int,  # 指定FPN各层特征通道数通常为256
                 levels: Sequence[str] = ('P2', 'P3', 'P4', 'P5'),  # 指定要处理的FPN层名称顺序
                 num_heads: int = 64,  # 指定多头注意力的头数建议保持能被通道数整除
                 use_ds_tokens: bool = True,  # 是否启用域特异性令牌注入
                 num_ds_tokens: int = 32,  # 指定每层生成的域特异性令牌数量
                 attn_dropout: float = 0.0,  # 注意力权重的丢弃概率建议默认0避免不稳定
                 num_feature_levels: Optional[int] = None,  # 显式指定期望的特征层数量用于配置校验
                 max_q_chunk: int = 256  # 控制按查询维度分块的大小默认256以平衡显存与性能
                 ) -> None:  # 构造函数返回None
        super().__init__()  # 调用父类初始化确保参数注册
        assert in_channels % num_heads == 0, 'in_channels must be divisible by num_heads'  # 断言通道数可被头数整除
        self.in_channels = in_channels  # 记录输入通道数供前向使用
        self.levels = list(levels)  # 保存层级名称列表便于调试
        self.feature_level_count = (  # 若未指定则使用层级列表长度
            num_feature_levels if num_feature_levels is not None else len(self.levels)  # 从配置或层级推断数量
        )  # 使用括号保持可读性
        assert len(self.levels) == self.feature_level_count, 'levels length mismatches feature count'  # 校验层级数量与期望特征层数一致
        self.num_heads = num_heads  # 保存多头数量
        self.use_ds_tokens = use_ds_tokens  # 保存是否启用域特异性令牌
        self.num_ds_tokens = num_ds_tokens  # 保存域特异性令牌数量
        self.attn_dropout = attn_dropout  # 保存注意力丢弃概率
        self.max_q_chunk = max_q_chunk  # 保存查询分块大小以用于分块注意力计算
        head_dim = in_channels // num_heads  # 计算每个注意力头的维度
        self.head_dim = head_dim  # 存储头维度供前向使用
        self.query_convs = nn.ModuleList()  # 创建用于生成查询的卷积层列表
        self.key_convs = nn.ModuleList()  # 创建用于生成键的卷积层列表
        self.value_convs = nn.ModuleList()  # 创建用于生成值的卷积层列表
        self.output_convs = nn.ModuleList()  # 创建用于整合注意力输出的卷积层列表
        self.ds_key_proj = nn.ModuleList()  # 创建用于生成域特异键的线性层列表
        self.ds_value_proj = nn.ModuleList()  # 创建用于生成域特异值的线性层列表
        self.ds_gate = nn.Parameter(torch.zeros(len(self.levels)))  # 创建可学习门控抑制域特异贡献
        for _ in self.levels:  # 遍历每个层级以构建对应模块
            self.query_convs.append(nn.Conv2d(in_channels, in_channels, kernel_size=1))  # 追加1x1卷积生成查询
            self.key_convs.append(nn.Conv2d(in_channels, in_channels, kernel_size=1))  # 追加1x1卷积生成键
            self.value_convs.append(nn.Conv2d(in_channels, in_channels, kernel_size=1))  # 追加1x1卷积生成值
            self.output_convs.append(nn.Conv2d(in_channels, in_channels, kernel_size=1))  # 追加1x1卷积整合输出
            self.ds_key_proj.append(nn.Linear(in_channels, num_heads * head_dim * num_ds_tokens))  # 追加线性层生成域特异键
            self.ds_value_proj.append(nn.Linear(in_channels, num_heads * head_dim * num_ds_tokens))  # 追加线性层生成域特异值

    def _reshape_heads(self, tensor: torch.Tensor) -> torch.Tensor:  # 辅助函数将(N,C,H,W)重排为注意力计算所需形状
        n, c, h, w = tensor.shape  # 解包张量形状
        tensor = tensor.view(n, self.num_heads, self.head_dim, h * w)  # 重塑为(N,头数,头维度,位置数)
        return tensor  # 返回重排后的张量

    def _compute_attention(self,
                           q: torch.Tensor,
                           k: torch.Tensor,
                           v: torch.Tensor,
                           ds_token_count: int) -> Tuple[torch.Tensor, torch.Tensor]:  # 计算注意力输出并返回注意力矩阵
        _, _, d, l = q.shape  # 解包查询张量形状以获取头维度与长度其余维度此处无需使用
        scale = 1.0 / math.sqrt(d)  # 计算缩放因子避免梯度爆炸保持数值稳定
        outputs: List[torch.Tensor] = []  # 初始化分块的注意力输出列表
        ds_ratios: List[torch.Tensor] = []  # 初始化分块的域特异比例列表
        for start in range(0, l, self.max_q_chunk):  # 按查询长度维度以max_q_chunk步长遍历
            end = min(start + self.max_q_chunk, l)  # 计算当前分块结束位置确保不越界
            q_chunk = q[..., start:end]  # 切片当前分块的查询张量形状为[N,H,D,chunk_L]
            attn_logits = torch.einsum('nhdl,nhdm->nhlm', q_chunk, k) * scale  # 计算当前分块的注意力打分避免一次性分配大矩阵
            attn_weights = F.softmax(attn_logits, dim=-1)  # 对当前分块沿键长度维度做softmax得到权重
            if self.attn_dropout > 0:  # 若需要应用dropout则在权重上施加
                attn_weights = F.dropout(attn_weights, p=self.attn_dropout, training=self.training)  # 使用框架训练模式控制随机性
            out_chunk = torch.einsum('nhlm,nhdm->nhdl', attn_weights, v)  # 将注意力权重与值张量相乘得到当前分块输出
            outputs.append(out_chunk)  # 将当前分块输出累积以便稍后拼接
            if ds_token_count > 0:  # 若启用了域特异令牌则计算对应比例
                ds_weights = attn_weights[..., -ds_token_count:]  # 获取当前分块中域特异令牌的权重
                total_weights = attn_weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)  # 计算总权重并截断避免除零
                ds_ratio_chunk = ds_weights.sum(dim=-1, keepdim=True) / total_weights  # 求得当前分块域特异权重比例
            else:
                ds_ratio_chunk = attn_weights.new_zeros(attn_weights.shape[:-1] + (1,))  # 若无域特异令牌则比例填零
            ds_ratios.append(ds_ratio_chunk)  # 将当前分块的域特异比例保存
        attn_output = torch.cat(outputs, dim=-1)  # 将所有分块输出按长度维度拼接恢复完整序列
        ds_ratio = torch.cat(ds_ratios, dim=-2)  # 将所有分块域特异比例按查询长度拼接与原逻辑保持一致
        return attn_output, ds_ratio  # 返回拼接后的注意力输出与域特异比例

    def forward(self,
                spatial_feats: Sequence[torch.Tensor],
                inv_feats: Sequence[torch.Tensor],
                ds_feats: Sequence[torch.Tensor]) -> Tuple[List[torch.Tensor], Dict[str, torch.Tensor]]:  # 前向传播融合特征
        assert len(spatial_feats) == len(self.levels), 'spatial feature count mismatches levels'  # 断言输入的空间特征数量与层级列表一致
        fused_feats: List[torch.Tensor] = []  # 初始化融合特征列表
        ds_ratios: List[torch.Tensor] = []  # 初始化域特异注意力比率列表
        for idx, (spatial, inv, ds) in enumerate(zip(spatial_feats, inv_feats, ds_feats)):  # 同步遍历三个特征列表
            q = self._reshape_heads(self.query_convs[idx](spatial))  # 对原始特征生成查询并重排维度
            k_spatial = self._reshape_heads(self.key_convs[idx](inv))  # 使用域不变特征生成键
            v_spatial = self._reshape_heads(self.value_convs[idx](inv))  # 使用域不变特征生成值
            k = k_spatial  # 初始化键集合为域不变键
            v = v_spatial  # 初始化值集合为域不变值
            ds_token_count = 0  # 初始化域特异令牌数量
            if self.use_ds_tokens and self.num_ds_tokens > 0:  # 判断是否启用域特异令牌
                pooled = ds.mean(dim=(-2, -1))  # 对域特异特征做全局平均得到通道描述
                key_tokens = self.ds_key_proj[idx](pooled)  # 通过线性层生成域特异键向量
                value_tokens = self.ds_value_proj[idx](pooled)  # 通过线性层生成域特异值向量
                key_tokens = key_tokens.view(-1, self.num_heads, self.head_dim, self.num_ds_tokens)  # 重塑域特异键以匹配多头形状
                value_tokens = value_tokens.view(-1, self.num_heads, self.head_dim, self.num_ds_tokens)  # 重塑域特异值以匹配多头形状
                gate = torch.sigmoid(self.ds_gate[idx])  # 通过Sigmoid获得域特异贡献门控系数
                value_tokens = value_tokens * gate  # 使用门控缩放域特异值抑制其主导性
                k = torch.cat([k_spatial, key_tokens], dim=-1)  # 将域特异键拼接到键集合中
                v = torch.cat([v_spatial, value_tokens], dim=-1)  # 将域特异值拼接到值集合中
                ds_token_count = self.num_ds_tokens  # 更新域特异令牌数量
            attn_output, ds_ratio = self._compute_attention(q, k, v, ds_token_count)  # 计算注意力输出以及域特异权重占比
            attn_output = attn_output.view(  # 将注意力输出调整回特征图形状
                spatial.shape[0], self.in_channels, spatial.shape[2], spatial.shape[3]  # 逐项传入批量与空间尺寸
            )  # 使用分行写法避免超出行宽限制
            fused = spatial + self.output_convs[idx](attn_output)  # 将注意力输出通过卷积并与原始特征残差相加
            fused_feats.append(fused)  # 保存融合后的特征
            ds_ratios.append(ds_ratio.mean(dim=(1, 2, 3)))  # 将域特异权重占比对头与空间维度求平均
        stats = {'ds_ratios': torch.stack(ds_ratios)}  # 将域特异占比堆叠成张量并存入统计字典
        return fused_feats, stats  # 返回融合特征列表以及统计信息
