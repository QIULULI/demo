from typing import Dict, List, Sequence  # 导入类型注解提升代码可读性

import torch  # 导入PyTorch主库执行张量运算
import torch.nn as nn  # 导入神经网络模块基类
import torch.nn.functional as F  # 导入常用函数模块以便使用归一化等操作

from mmdet.registry import MODELS  # 导入MMDetection注册表用于登记损失函数


@MODELS.register_module()  # 注册光谱分离损失类
class LossDecouple(nn.Module):  # 定义光谱分离损失模块
    def __init__(self,
                 idem_weight: float = 1.0,  # 设置幂等性损失权重默认1.0
                 orth_weight: float = 1.0,  # 设置正交性损失权重默认1.0
                 energy_weight: float = 1.0,  # 设置能量守恒损失权重默认1.0
                 eps: float = 1e-6  # 设置数值稳定用微小常数
                 ) -> None:  # 构造函数返回None
        super().__init__()  # 调用父类初始化
        self.idem_weight = idem_weight  # 保存幂等性损失权重
        self.orth_weight = orth_weight  # 保存正交性损失权重
        self.energy_weight = energy_weight  # 保存能量守恒损失权重
        self.eps = eps  # 保存数值稳定常数

    def forward(self,
                feats: Sequence[torch.Tensor],
                inv_feats: Sequence[torch.Tensor],
                ds_feats: Sequence[torch.Tensor],
                said_module: nn.Module) -> Dict[str, torch.Tensor]:  # 计算分离损失并返回字典
        idem_losses: List[torch.Tensor] = []  # 初始化幂等性损失列表
        orth_losses: List[torch.Tensor] = []  # 初始化正交性损失列表
        energy_losses: List[torch.Tensor] = []  # 初始化能量守恒损失列表
        with torch.enable_grad():  # 确保允许梯度用于两次通过
            idem_inv_feats, _ = said_module(inv_feats)  # 再次对低频特征应用SAID获取幂等性参考
        for feat, inv, ds, idem_inv in zip(feats, inv_feats, ds_feats, idem_inv_feats):  # 遍历层级计算各项损失
            idem_losses.append(F.mse_loss(idem_inv, inv))  # 计算幂等性损失鼓励重复应用不改变输出
            inv_flat = F.normalize(inv.view(inv.size(0), -1), dim=1, eps=self.eps)  # 将低频特征展平并L2归一化并使用eps稳定
            ds_flat = F.normalize(ds.view(ds.size(0), -1), dim=1, eps=self.eps)  # 将高频特征展平并L2归一化并使用eps稳定
            cosine = (inv_flat * ds_flat).sum(dim=1)  # 计算两者余弦相似度
            orth_losses.append((cosine.square()).mean())  # 将余弦平方作为正交性惩罚
            energy_total = feat.square().sum(dim=(1, 2, 3))  # 计算原始特征能量
            energy_inv = inv.square().sum(dim=(1, 2, 3))  # 计算低频特征能量
            energy_ds = ds.square().sum(dim=(1, 2, 3))  # 计算高频特征能量
            energy_residual = energy_inv + energy_ds - energy_total  # 计算能量残差
            energy_losses.append((energy_residual.square()).mean())  # 使用平方误差约束能量守恒
        loss_idem = self.idem_weight * torch.stack(idem_losses).mean()  # 聚合幂等性损失并施加权重
        loss_orth = self.orth_weight * torch.stack(orth_losses).mean()  # 聚合正交性损失并施加权重
        loss_energy = self.energy_weight * torch.stack(energy_losses).mean()  # 聚合能量损失并施加权重
        return {'loss_idem': loss_idem, 'loss_orth': loss_orth, 'loss_energy': loss_energy}  # 返回损失字典


@MODELS.register_module()  # 注册耦合损失类
class LossCouple(nn.Module):  # 定义耦合损失模块
    def __init__(self,
                 align_weight: float = 1.0,  # 设置特征对齐损失权重默认1.0
                 ds_weight: float = 1.0,  # 设置域特异抑制损失权重默认1.0
                 ds_margin: float = 0.2  # 设置域特异权重上限建议0.2
                 ) -> None:  # 构造函数返回None
        super().__init__()  # 调用父类初始化
        self.align_weight = align_weight  # 保存对齐损失权重
        self.ds_weight = ds_weight  # 保存域特异抑制权重
        self.ds_margin = ds_margin  # 保存域特异注意力阈值

    def forward(self,
                fused_feats: Sequence[torch.Tensor],
                teacher_inv_feats: Sequence[torch.Tensor],
                stats: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:  # 计算耦合损失并返回字典
        align_losses: List[torch.Tensor] = []  # 初始化特征对齐损失列表
        for fused, teacher_inv in zip(fused_feats, teacher_inv_feats):  # 遍历对应层级
            teacher_safe = teacher_inv.detach()  # 分离教师域不变特征的计算图防止梯度回流教师分支
            align_losses.append(F.mse_loss(fused, teacher_safe))  # 计算耦合后特征与域不变特征之间的MSE
        loss_align = self.align_weight * torch.stack(align_losses).mean()  # 聚合对齐损失并施加权重
        if 'ds_ratios' in stats:  # 若统计信息包含域特异占比
            ds_ratio = stats['ds_ratios']  # 取出域特异占比张量
            penalty = F.relu(ds_ratio - self.ds_margin)  # 超过阈值的部分产生惩罚
            loss_ds = self.ds_weight * penalty.mean()  # 平均惩罚并施加权重
        else:
            loss_ds = fused_feats[0].new_tensor(0.0)  # 若无统计信息则损失为零
        return {'loss_couple_align': loss_align, 'loss_couple_ds': loss_ds}  # 返回包含两项损失的字典
