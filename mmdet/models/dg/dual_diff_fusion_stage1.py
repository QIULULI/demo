
import copy  # 中文注释：引入copy模块用于安全复制配置与样本对象
from typing import Dict, List, Optional, Tuple, Union  # 中文注释：导入类型注解工具以增强代码可读性

import torch  # 中文注释：导入PyTorch主库以执行张量计算
import torch.nn.functional as F  # 中文注释：导入PyTorch函数式接口以便计算均方误差等损失
from torch import Tensor  # 中文注释：导入Tensor类型用于类型提示

from mmdet.models.detectors.base import BaseDetector  # 中文注释：导入基础检测器父类以便继承
from mmdet.models.detectors.Z_diffusion_detector import DiffusionDetector  # 中文注释：导入扩散检测器类型用于构建教师与学生分支
from mmdet.models.losses import KnowledgeDistillationKLDivLoss, L1Loss  # 中文注释：导入KL散度蒸馏损失与L1损失用于ROI蒸馏
from mmdet.models.utils import rename_loss_dict  # 中文注释：导入损失字典重命名工具以统一日志前缀
from mmdet.registry import MODELS  # 中文注释：导入模型注册表用于通过配置构建模块
from mmdet.structures import SampleList  # 中文注释：导入样本列表类型以明确接口约定
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig  # 中文注释：导入配置类型别名以保持与框架兼容


@MODELS.register_module()  # 中文注释：将当前检测器注册到MMDetection框架的模型注册表中
class DualDiffFusionStage1(BaseDetector):  # 中文注释：定义第一阶段红外-可见光扩散融合检测器

    def __init__(self,  # 中文注释：初始化函数负责解析配置并构建教师与学生检测器
                 teacher_ir: Union[DiffusionDetector, ConfigType],  # 中文注释：教师分支既可以传入实例也可以传入配置字典
                 student_rgb: Union[DiffusionDetector, ConfigType],  # 中文注释：学生分支同样支持实例或配置字典
                 train_cfg: OptConfigType = None,  # 中文注释：训练阶段附加配置用于设置蒸馏权重与调度参数
                 data_preprocessor: OptConfigType = None,  # 中文注释：数据预处理配置按照父类约定传递
                 init_cfg: OptMultiConfig = None) -> None:  # 中文注释：初始化权重的可选配置
        super().__init__(data_preprocessor=data_preprocessor, init_cfg=init_cfg)  # 中文注释：调用父类初始化确保基础组件正确注册
        self.teacher_ir = self._build_branch(teacher_ir, 'teacher_ir')  # 中文注释：构建教师分支或直接引用已有实例
        self.student_rgb = self._build_branch(student_rgb, 'student_rgb')  # 中文注释：构建学生分支或引用已有实例
        self.train_cfg = copy.deepcopy(train_cfg) if train_cfg is not None else dict()  # 中文注释：深拷贝训练配置防止外部副作用
        self.w_sup = float(self.train_cfg.get('w_sup', 1.0))  # 中文注释：读取学生监督损失权重默认值为1.0
        self.w_cross = float(self.train_cfg.get('w_cross', 1.0))  # 中文注释：读取交叉蒸馏损失权重默认值为1.0
        self.w_feat_kd = float(self.train_cfg.get('w_feat_kd', 0.0))  # 中文注释：读取特征蒸馏损失权重默认关闭
        self.enable_roi_kd = bool(self.train_cfg.get('enable_roi_kd', False))  # 中文注释：读取是否启用ROI蒸馏的布尔开关
        self.w_roi_kd = float(self.train_cfg.get('w_roi_kd', 1.0))  # 中文注释：读取ROI蒸馏损失权重默认值为1.0
        self.cross_warmup_iters = int(self.train_cfg.get('cross_warmup_iters', 0))  # 中文注释：读取交叉蒸馏预热迭代数默认0立即生效
        self.freeze_teacher = bool(self.train_cfg.get('freeze_teacher', True))  # 中文注释：读取是否冻结教师分支的布尔开关
        if self.freeze_teacher:  # 中文注释：当需要冻结教师时
            for param in self.teacher_ir.parameters():  # 中文注释：遍历教师模型的所有参数
                param.requires_grad_(False)  # 中文注释：显式禁止教师参数更新梯度
            self.teacher_ir.eval()  # 中文注释：冻结时将教师切换到评估模式以固定归一化统计量
        else:  # 中文注释：当教师需要联合训练时
            self.teacher_ir.train()  # 中文注释：允许教师保持训练模式参与梯度更新
        self._teacher_grad_check_passed = False  # 中文注释：初始化教师冻结断言缓存标记为未通过以便首次检测
        self.roi_cls_kd_criterion = KnowledgeDistillationKLDivLoss(  # 中文注释：构建ROI分类蒸馏使用的KL散度损失
            class_reduction='mean', reduction='mean', loss_weight=1.0)  # 中文注释：设置类别维度平均与批量平均的默认方式
        self.roi_reg_kd_criterion = L1Loss(reduction='mean', loss_weight=1.0)  # 中文注释：构建ROI回归蒸馏使用的L1损失
        self.local_iter = 0  # 中文注释：记录当前训练迭代计数以驱动交叉蒸馏预热

    @property
    def with_rpn(self) -> bool:  # 中文注释：判断学生检测器是否包含RPN头部
        return getattr(self.student_rgb, 'with_rpn', False)  # 中文注释：直接查询学生模型的with_rpn属性

    @property
    def with_roi_head(self) -> bool:  # 中文注释：判断学生检测器是否包含ROI头部
        return getattr(self.student_rgb, 'with_roi_head', False)  # 中文注释：直接查询学生模型的with_roi_head属性

    def extract_feat_student(self, batch_inputs: Tensor) -> Tuple[Tensor, ...]:  # 中文注释：封装学生分支的特征提取流程
        student_feats = self.student_rgb.extract_feat(batch_inputs)  # 中文注释：调用学生模型提取多尺度特征
        if isinstance(student_feats, (list, tuple)):  # 中文注释：当返回列表或元组时直接转换
            return tuple(student_feats)  # 中文注释：转换为元组以便后续统一处理
        return (student_feats,)  # 中文注释：单尺度情况时包装成单元素元组保持接口一致

    def extract_feat_teacher(self, batch_inputs: Tensor) -> Tuple[Tensor, ...]:  # 中文注释：封装教师分支的特征提取流程
        if self.freeze_teacher:  # 中文注释：若教师被冻结则使用无梯度上下文
            with torch.no_grad():  # 中文注释：关闭梯度避免教师参数更新
                teacher_feats = self.teacher_ir.extract_feat(batch_inputs)  # 中文注释：提取教师特征
        else:  # 中文注释：教师未冻结时直接执行前向
            teacher_feats = self.teacher_ir.extract_feat(batch_inputs)  # 中文注释：提取教师特征并允许梯度传播
        if isinstance(teacher_feats, (list, tuple)):  # 中文注释：当返回列表或元组时直接转换
            return tuple(teacher_feats)  # 中文注释：转换为元组保证统一接口
        return (teacher_feats,)  # 中文注释：单尺度情况时包装成元组

    def loss(self, batch_inputs: Tensor, batch_data_samples: SampleList) -> Dict[str, Tensor]:  # 中文注释：实现整体训练损失计算逻辑
        if self.freeze_teacher:  # 中文注释：仅当需要冻结教师时才执行断言校验
            if not getattr(self, '_teacher_grad_check_passed', False):  # 中文注释：通过缓存标记避免重复遍历参数
                offending_name: Optional[str] = None  # 中文注释：记录第一个未被冻结的参数名称
                for name, param in self.teacher_ir.named_parameters():  # 中文注释：遍历教师模型所有具名参数检查梯度状态
                    if param.requires_grad:  # 中文注释：一旦发现仍允许梯度的参数即刻记录
                        offending_name = name  # 中文注释：保存触发问题的参数名称便于排查
                        break  # 中文注释：找到问题后提前结束遍历提高效率
                if offending_name is not None:  # 中文注释：若检测到仍可训练的参数则立即抛出异常
                    raise AssertionError(  # 中文注释：抛出断言异常提示开发者检查冻结逻辑
                        '检测到freeze_teacher=True但教师参数仍保留梯度：'  # 中文注释：明确问题发生的场景
                        f'{offending_name}。请确认train_cfg.freeze_teacher为True，'  # 中文注释：提示检查配置项
                        '不要对teacher_ir参数调用requires_grad_(True)，并确保优化器未包含教师参数。')  # 中文注释：给出排查优化器与梯度设置的建议
                self._teacher_grad_check_passed = True  # 中文注释：若所有参数均已冻结则缓存结果避免后续重复检查
        student_feats = self.extract_feat_student(batch_inputs)  # 中文注释：首先提取学生的多尺度特征
        teacher_feats = self.extract_feat_teacher(batch_inputs)  # 中文注释：随后提取教师的多尺度特征
        assert len(student_feats) == len(teacher_feats), 'Student and teacher feature levels must match.'  # 中文注释：断言两者FPN层数一致
        losses: Dict[str, Tensor] = dict()  # 中文注释：初始化最终损失字典
        loss_total: Optional[Tensor] = None  # 中文注释：初始化总损失累加器
        stu_total: Optional[Tensor] = None  # 中文注释：初始化学生监督损失累加器
        cross_total: Optional[Tensor] = None  # 中文注释：初始化交叉蒸馏损失累加器

        def _accumulate(current: Optional[Tensor], value: Tensor) -> Tensor:  # 中文注释：定义内部函数用于累加张量
            return value if current is None else current + value  # 中文注释：当累加器为空时直接返回当前值否则执行加法

        stu_rpn_results: Optional[List] = None  # 中文注释：记录学生RPN生成的候选框供后续复用
        if self.w_sup > 0:  # 中文注释：仅当学生监督权重大于零时计算监督损失
            if self.with_rpn and hasattr(self.student_rgb, 'rpn_head'):  # 中文注释：当学生包含RPN头部时计算RPN损失
                proposal_cfg = self._get_proposal_cfg()  # 中文注释：读取RPN候选框配置
                rpn_losses, stu_rpn_results = self.student_rgb.rpn_head.loss_and_predict(  # 中文注释：基于学生特征计算RPN损失与候选框
                    student_feats, batch_data_samples, proposal_cfg=proposal_cfg)  # 中文注释：传入学生特征与真实标注
                if 'loss_cls' in rpn_losses:  # 中文注释：判断是否存在标准命名的分类损失
                    stu_rpn_loss_cls_raw = rpn_losses['loss_cls']  # 中文注释：读取分类损失张量
                elif 'loss_rpn_cls' in rpn_losses:  # 中文注释：兼容以loss_rpn_cls命名的分类损失
                    stu_rpn_loss_cls_raw = rpn_losses['loss_rpn_cls']  # 中文注释：读取分类损失张量
                else:  # 中文注释：缺失分类损失时抛出异常提醒配置问题
                    raise KeyError('RPN losses must contain loss_cls or loss_rpn_cls for student branch.')  # 中文注释：提示学生分支缺少分类损失
                if 'loss_bbox' in rpn_losses:  # 中文注释：判断是否存在标准命名的回归损失
                    stu_rpn_loss_bbox_raw = rpn_losses['loss_bbox']  # 中文注释：读取回归损失张量
                elif 'loss_rpn_bbox' in rpn_losses:  # 中文注释：兼容以loss_rpn_bbox命名的回归损失
                    stu_rpn_loss_bbox_raw = rpn_losses['loss_rpn_bbox']  # 中文注释：读取回归损失张量
                else:  # 中文注释：缺失回归损失时抛出异常提醒配置问题
                    raise KeyError('RPN losses must contain loss_bbox or loss_rpn_bbox for student branch.')  # 中文注释：提示学生分支缺少回归损失
                stu_rpn_loss_cls_weighted = stu_rpn_loss_cls_raw * self.w_sup  # 中文注释：将分类损失乘以学生监督权重
                stu_rpn_loss_bbox_weighted = stu_rpn_loss_bbox_raw * self.w_sup  # 中文注释：将回归损失乘以学生监督权重
                losses['stu_rpn_loss_cls'] = stu_rpn_loss_cls_weighted  # 中文注释：写入学生RPN分类损失并统一键名
                losses['stu_rpn_loss_bbox'] = stu_rpn_loss_bbox_weighted  # 中文注释：写入学生RPN回归损失并统一键名
                loss_total = _accumulate(loss_total, stu_rpn_loss_cls_weighted)  # 中文注释：将学生分类损失累加到总损失
                loss_total = _accumulate(loss_total, stu_rpn_loss_bbox_weighted)  # 中文注释：将学生回归损失累加到总损失
                stu_total = _accumulate(stu_total, stu_rpn_loss_cls_weighted)  # 中文注释：将学生分类损失累加到学生监督总和
                stu_total = _accumulate(stu_total, stu_rpn_loss_bbox_weighted)  # 中文注释：将学生回归损失累加到学生监督总和
            if self.with_roi_head and hasattr(self.student_rgb, 'roi_head'):  # 中文注释：当学生包含ROI头部时计算ROI损失
                roi_inputs = stu_rpn_results if stu_rpn_results is not None else self._prepare_roi_inputs(  # 中文注释：优先复用学生RPN候选框否则动态生成
                    student_feats, batch_data_samples)  # 中文注释：当缺少RPN结果时重新生成候选框
                roi_losses = self.student_rgb.roi_head.loss(student_feats, roi_inputs, batch_data_samples)  # 中文注释：基于学生特征计算ROI监督损失
                for key, value in rename_loss_dict('stu_', roi_losses).items():  # 中文注释：为ROI损失添加stu_前缀
                    weighted = value * self.w_sup  # 中文注释：乘以学生监督权重
                    losses[key] = weighted  # 中文注释：写入损失字典
                    loss_total = _accumulate(loss_total, weighted)  # 中文注释：累加到总损失
                    stu_total = _accumulate(stu_total, weighted)  # 中文注释：累加到学生监督损失总和
        if stu_total is not None:  # 中文注释：若存在学生监督损失则记录汇总指标
            losses['stu_loss_total'] = stu_total  # 中文注释：在日志中记录学生监督损失总和

        cross_weight = self.w_cross if self.local_iter >= self.cross_warmup_iters else 0.0  # 中文注释：根据预热迭代数决定交叉蒸馏权重
        cross_rpn_results: Optional[List] = None  # 中文注释：记录教师特征驱动的候选框供复用
        if cross_weight > 0:  # 中文注释：仅当交叉蒸馏权重大于零时执行
            if self.with_rpn and hasattr(self.student_rgb, 'rpn_head'):  # 中文注释：当学生包含RPN头部时计算交叉RPN损失
                proposal_cfg = self._get_proposal_cfg()  # 中文注释：读取候选框配置
                rpn_losses, cross_rpn_results = self.student_rgb.rpn_head.loss_and_predict(  # 中文注释：基于教师特征计算学生RPN损失
                    teacher_feats, batch_data_samples, proposal_cfg=proposal_cfg)  # 中文注释：传入教师特征与真实标注
                if 'loss_cls' in rpn_losses:  # 中文注释：判断是否存在标准命名的分类损失
                    cross_rpn_loss_cls_raw = rpn_losses['loss_cls']  # 中文注释：读取分类损失张量
                elif 'loss_rpn_cls' in rpn_losses:  # 中文注释：兼容以loss_rpn_cls命名的分类损失
                    cross_rpn_loss_cls_raw = rpn_losses['loss_rpn_cls']  # 中文注释：读取分类损失张量
                else:  # 中文注释：缺失分类损失时抛出异常提醒配置问题
                    raise KeyError('RPN losses must contain loss_cls or loss_rpn_cls for cross branch.')  # 中文注释：提示交叉分支缺少分类损失
                if 'loss_bbox' in rpn_losses:  # 中文注释：判断是否存在标准命名的回归损失
                    cross_rpn_loss_bbox_raw = rpn_losses['loss_bbox']  # 中文注释：读取回归损失张量
                elif 'loss_rpn_bbox' in rpn_losses:  # 中文注释：兼容以loss_rpn_bbox命名的回归损失
                    cross_rpn_loss_bbox_raw = rpn_losses['loss_rpn_bbox']  # 中文注释：读取回归损失张量
                else:  # 中文注释：缺失回归损失时抛出异常提醒配置问题
                    raise KeyError('RPN losses must contain loss_bbox or loss_rpn_bbox for cross branch.')  # 中文注释：提示交叉分支缺少回归损失
                cross_rpn_loss_cls_weighted = cross_rpn_loss_cls_raw * cross_weight  # 中文注释：将分类损失乘以交叉蒸馏权重
                cross_rpn_loss_bbox_weighted = cross_rpn_loss_bbox_raw * cross_weight  # 中文注释：将回归损失乘以交叉蒸馏权重
                losses['cross_rpn_loss_cls'] = cross_rpn_loss_cls_weighted  # 中文注释：写入交叉RPN分类损失并统一键名
                losses['cross_rpn_loss_bbox'] = cross_rpn_loss_bbox_weighted  # 中文注释：写入交叉RPN回归损失并统一键名
                loss_total = _accumulate(loss_total, cross_rpn_loss_cls_weighted)  # 中文注释：将交叉分类损失累加到总损失
                loss_total = _accumulate(loss_total, cross_rpn_loss_bbox_weighted)  # 中文注释：将交叉回归损失累加到总损失
                cross_total = _accumulate(cross_total, cross_rpn_loss_cls_weighted)  # 中文注释：将交叉分类损失累加到交叉总和
                cross_total = _accumulate(cross_total, cross_rpn_loss_bbox_weighted)  # 中文注释：将交叉回归损失累加到交叉总和
            if self.with_roi_head and hasattr(self.student_rgb, 'roi_head'):  # 中文注释：当学生包含ROI头部时计算交叉ROI损失
                roi_inputs = cross_rpn_results if cross_rpn_results is not None else self._prepare_roi_inputs(  # 中文注释：优先复用教师特征驱动的候选框
                    teacher_feats, batch_data_samples)  # 中文注释：当缺少候选框时重新生成
                roi_losses = self.student_rgb.roi_head.loss(teacher_feats, roi_inputs, batch_data_samples)  # 中文注释：基于教师特征计算学生ROI损失
                for key, value in rename_loss_dict('cross_', roi_losses).items():  # 中文注释：为交叉损失添加cross_前缀
                    weighted = value * cross_weight  # 中文注释：乘以交叉蒸馏权重
                    losses[key] = weighted  # 中文注释：写入损失字典
                    loss_total = _accumulate(loss_total, weighted)  # 中文注释：累加到总损失
                    cross_total = _accumulate(cross_total, weighted)  # 中文注释：累加到交叉蒸馏损失总和
        if cross_total is not None:  # 中文注释：若存在交叉蒸馏损失则记录汇总指标
            losses['cross_loss_total'] = cross_total  # 中文注释：记录交叉蒸馏损失总和

        if self.w_feat_kd > 0:  # 中文注释：当特征蒸馏权重大于零时计算特征KD损失
            mse_values: List[Tensor] = []  # 中文注释：创建列表存储各层均方误差
            for stu_feat, tea_feat in zip(student_feats, teacher_feats):  # 中文注释：逐层遍历学生与教师特征
                mse_values.append(F.mse_loss(stu_feat, tea_feat.detach()))  # 中文注释：计算当前层的均方误差并阻断教师梯度
            if mse_values:  # 中文注释：确保存在至少一层特征
                feat_kd_loss = sum(mse_values) / float(len(mse_values))  # 中文注释：将所有层的均方误差求平均
                weighted = feat_kd_loss * self.w_feat_kd  # 中文注释：乘以特征蒸馏权重
                losses['feat_kd_loss'] = weighted  # 中文注释：记录特征蒸馏损失
                loss_total = _accumulate(loss_total, weighted)  # 中文注释：累加到总损失

        if (self.enable_roi_kd and self.w_roi_kd > 0 and self.with_roi_head
                and hasattr(self.student_rgb, 'roi_head')
                and hasattr(self.student_rgb.roi_head, 'forward')):  # 中文注释：当启用ROI蒸馏且接口齐全时执行
            roi_inputs_for_kd = stu_rpn_results if stu_rpn_results is not None else self._prepare_roi_inputs(  # 中文注释：优先复用学生监督阶段的候选框
                student_feats, batch_data_samples)  # 中文注释：若未计算监督RPN则重新生成候选框
            student_roi_outputs = self.student_rgb.roi_head.forward(student_feats, roi_inputs_for_kd, batch_data_samples)  # 中文注释：获取学生ROI前向输出
            teacher_roi_outputs = self.student_rgb.roi_head.forward(teacher_feats, roi_inputs_for_kd, batch_data_samples)  # 中文注释：获取教师ROI前向输出
            if len(student_roi_outputs) >= 1 and len(teacher_roi_outputs) >= 1:  # 中文注释：存在分类输出时计算KL蒸馏
                stu_cls = student_roi_outputs[0]  # 中文注释：提取学生分类logits
                tea_cls = teacher_roi_outputs[0]  # 中文注释：提取教师分类logits
                if stu_cls.shape == tea_cls.shape:  # 中文注释：确保形状一致
                    cls_loss = self.roi_cls_kd_criterion(stu_cls, tea_cls.detach())  # 中文注释：计算KL散度蒸馏损失
                    weighted_cls = cls_loss * self.w_roi_kd  # 中文注释：乘以ROI蒸馏权重
                    losses['roi_cls_kd_loss'] = weighted_cls  # 中文注释：记录分类蒸馏损失
                    loss_total = _accumulate(loss_total, weighted_cls)  # 中文注释：累加到总损失
            if len(student_roi_outputs) >= 2 and len(teacher_roi_outputs) >= 2:  # 中文注释：存在回归输出时计算L1蒸馏
                stu_reg = student_roi_outputs[1]  # 中文注释：提取学生回归输出
                tea_reg = teacher_roi_outputs[1]  # 中文注释：提取教师回归输出
                if stu_reg.shape == tea_reg.shape:  # 中文注释：确保形状一致
                    reg_loss = self.roi_reg_kd_criterion(stu_reg, tea_reg.detach())  # 中文注释：计算L1蒸馏损失
                    weighted_reg = reg_loss * self.w_roi_kd  # 中文注释：乘以ROI蒸馏权重
                    losses['roi_reg_kd_loss'] = weighted_reg  # 中文注释：记录回归蒸馏损失
                    loss_total = _accumulate(loss_total, weighted_reg)  # 中文注释：累加到总损失

        if loss_total is None:  # 中文注释：若未累加任何损失则创建零张量占位
            loss_total = student_feats[0].sum() * 0  # 中文注释：使用学生特征创建零值张量保持梯度设备一致
        losses['loss_total'] = loss_total  # 中文注释：记录总损失供日志与反向传播使用
        self.local_iter += 1  # 中文注释：自增内部迭代计数以支持交叉蒸馏预热
        return losses  # 中文注释：返回完整的损失字典

    def predict(self, batch_inputs: Tensor, batch_data_samples: SampleList, rescale: bool = True) -> SampleList:  # 中文注释：推理接口直接委托学生模型
        return self.student_rgb.predict(batch_inputs, batch_data_samples, rescale=rescale)  # 中文注释：复用学生模型预测逻辑确保部署一致

    def _forward(self, batch_inputs: Tensor, batch_data_samples: SampleList) -> Tuple:  # 中文注释：定义前向推理接口以适配MMEngine导出
        return self.student_rgb._forward(batch_inputs, batch_data_samples)  # 中文注释：直接复用学生模型的前向实现

    def export_fused_teacher(self, path: str) -> None:  # 中文注释：导出融合后的教师权重供后续阶段使用
        torch.save(self.student_rgb.state_dict(), path)  # 中文注释：保存学生模型参数作为新的教师权重

    def _build_branch(self, module_or_cfg: Union[DiffusionDetector, ConfigType], name: str) -> DiffusionDetector:  # 中文注释：根据配置或实例构建检测分支
        if isinstance(module_or_cfg, DiffusionDetector):  # 中文注释：当传入扩散检测器实例时直接返回
            return module_or_cfg  # 中文注释：无需额外构建
        if isinstance(module_or_cfg, BaseDetector):  # 中文注释：允许传入其他检测器实例方便调试
            return module_or_cfg  # 中文注释：直接返回已有实例
        if isinstance(module_or_cfg, dict):  # 中文注释：当传入配置字典时通过注册表构建
            return MODELS.build(module_or_cfg)  # 中文注释：使用注册表实例化检测器
        raise TypeError(f'{name} must be DiffusionDetector or config dict, but got {type(module_or_cfg)!r}')  # 中文注释：类型不匹配时抛出异常

    def _prepare_roi_inputs(self, feats: Tuple[Tensor, ...], batch_data_samples: SampleList) -> List:  # 中文注释：根据特征与标注准备ROI阶段候选框
        if self.with_rpn and hasattr(self.student_rgb, 'rpn_head'):  # 中文注释：若学生包含RPN则使用给定特征生成候选框
            proposal_cfg = self._get_proposal_cfg()  # 中文注释：读取RPN候选框配置
            _, proposals = self.student_rgb.rpn_head.loss_and_predict(feats, batch_data_samples, proposal_cfg=proposal_cfg)  # 中文注释：调用RPN获取候选框并忽略损失
            return proposals  # 中文注释：返回候选框供ROI阶段使用
        return [getattr(sample, 'proposals') for sample in batch_data_samples]  # 中文注释：无RPN时回退到样本中预生成的候选框

    def _get_proposal_cfg(self):  # 中文注释：获取RPN候选框配置对象
        train_cfg = getattr(self.student_rgb, 'train_cfg', None)  # 中文注释：优先读取学生模型的训练配置
        if train_cfg is not None and hasattr(train_cfg, 'rpn_proposal'):  # 中文注释：训练配置存在rpn_proposal时优先使用
            return train_cfg.rpn_proposal  # 中文注释：返回训练态候选框配置
        test_cfg = getattr(self.student_rgb, 'test_cfg', None)  # 中文注释：否则尝试从测试配置中读取
        if test_cfg is not None and hasattr(test_cfg, 'rpn'):  # 中文注释：测试配置包含rpn字段时返回
            return test_cfg.rpn  # 中文注释：返回测试态候选框配置
        return None  # 中文注释：若均未配置则返回None表示使用默认逻辑


if __name__ == '__main__':  # 中文注释：提供最小化自检脚本方便快速验证逻辑
    class _ToyRPNHead(torch.nn.Module):  # 中文注释：定义简化版RPN头用于自检
        def loss_and_predict(self, feats, samples, proposal_cfg=None):  # 中文注释：实现最小接口返回固定损失与候选框
            loss_cls = torch.tensor(1.0, device=feats[0].device)  # 中文注释：构造恒定的RPN分类损失值
            loss_bbox = torch.tensor(0.5, device=feats[0].device)  # 中文注释：构造恒定的RPN回归损失值
            losses = {'loss_cls': loss_cls, 'loss_bbox': loss_bbox}  # 中文注释：封装同时包含分类与回归的RPN损失字典
            proposals = []  # 中文注释：创建候选框列表
            for _ in samples:  # 中文注释：遍历每个样本
                proposals.append(type('Proposal', (), {'bboxes': torch.zeros((1, 4), device=feats[0].device)})())  # 中文注释：为每个样本创建占位候选框对象
            return losses, proposals  # 中文注释：返回损失字典与候选框列表

    class _ToyROIHead(torch.nn.Module):  # 中文注释：定义简化版ROI头用于自检
        def loss(self, feats, proposals, samples):  # 中文注释：实现最小ROI监督损失接口
            return {'loss_roi': torch.tensor(2.0, device=feats[0].device)}  # 中文注释：返回恒定的ROI损失值

        def forward(self, feats, proposals, samples=None):  # 中文注释：实现ROI前向输出用于蒸馏
            cls_score = torch.ones((1, 2), device=feats[0].device)  # 中文注释：构造恒定的分类logits
            bbox_pred = torch.zeros((1, 4), device=feats[0].device)  # 中文注释：构造恒定的回归输出
            return (cls_score, bbox_pred)  # 中文注释：返回分类与回归输出元组

        @property
        def with_bbox(self):  # 中文注释：指示存在bbox分支
            return True  # 中文注释：返回True以符合主类假设

    class _ToyDetector(BaseDetector):  # 中文注释：定义简化版检测器用于构造教师与学生
        def __init__(self):  # 中文注释：初始化简化检测器
            super().__init__(data_preprocessor=None)  # 中文注释：调用父类初始化
            self.rpn_head = _ToyRPNHead()  # 中文注释：挂载简化RPN头
            self.roi_head = _ToyROIHead()  # 中文注释：挂载简化ROI头

        @property
        def with_rpn(self):  # 中文注释：指示包含RPN
            return True  # 中文注释：返回True

        @property
        def with_roi_head(self):  # 中文注释：指示包含ROI头
            return True  # 中文注释：返回True

        def extract_feat(self, batch_inputs: Tensor):  # 中文注释：实现特征提取逻辑
            feat1 = torch.ones((batch_inputs.size(0), 1, 1, 1), device=batch_inputs.device)  # 中文注释：构造第一层特征
            feat2 = 2 * torch.ones((batch_inputs.size(0), 1, 1, 1), device=batch_inputs.device)  # 中文注释：构造第二层特征
            return (feat1, feat2)  # 中文注释：返回由两层特征组成的元组

        def loss(self, batch_inputs: Tensor, batch_data_samples: SampleList):  # 中文注释：实现占位监督损失接口
            return {'loss_dummy': torch.tensor(0.0, device=batch_inputs.device)}  # 中文注释：返回零损失保持接口兼容

        def predict(self, batch_inputs: Tensor, batch_data_samples: SampleList, rescale: bool = True):  # 中文注释：实现占位预测接口
            return []  # 中文注释：返回空列表作为占位预测结果

        def _forward(self, batch_inputs: Tensor, batch_data_samples: SampleList):  # 中文注释：实现占位前向接口
            return tuple()  # 中文注释：返回空元组

    class _ToySample:  # 中文注释：定义简化数据样本结构
        def __init__(self):  # 中文注释：初始化样本
            self.gt_instances = type('gt', (), {'labels': torch.zeros(1, dtype=torch.long)})()  # 中文注释：构造带标签字段的占位实例
            self.proposals = type('prop', (), {'bboxes': torch.zeros((1, 4))})()  # 中文注释：构造占位候选框对象

    teacher = _ToyDetector()  # 中文注释：实例化教师模型
    student = _ToyDetector()  # 中文注释：实例化学生模型
    model = DualDiffFusionStage1(teacher, student, train_cfg=dict(  # 中文注释：构建融合模型并设置权重
        w_sup=2.0, w_cross=3.0, w_feat_kd=4.0, enable_roi_kd=True, w_roi_kd=5.0, cross_warmup_iters=0, freeze_teacher=True))  # 中文注释：设置示例配置确保全部分支生效
    dummy_inputs = torch.randn(1, 3, 4, 4)  # 中文注释：构造随机输入张量
    dummy_samples = [_ToySample()]  # 中文注释：构造单个简化样本列表
    losses = model.loss(dummy_inputs, dummy_samples)  # 中文注释：执行一次损失计算验证流程
    print('loss_keys', sorted(losses.keys()))  # 中文注释：打印损失键名验证命名规则
    required_rpn_keys = {'stu_rpn_loss_cls', 'stu_rpn_loss_bbox', 'cross_rpn_loss_cls', 'cross_rpn_loss_bbox'}  # 中文注释：定义必须存在的RPN键名集合
    print('rpn_keys_present', {key: (key in losses) for key in sorted(required_rpn_keys)})  # 中文注释：打印每个RPN键名是否存在
    part_keys = [key for key in losses.keys() if key not in ('loss_total', 'stu_loss_total', 'cross_loss_total')]  # 中文注释：过滤掉汇总项避免重复统计
    part_values = [losses[key] for key in part_keys]  # 中文注释：收集需要参与求和的损失值
    total_from_parts = torch.stack(part_values).sum() if part_values else torch.tensor(0.0, device=dummy_inputs.device)  # 中文注释：对有效损失进行求和
    print('consistency', torch.allclose(losses['loss_total'], total_from_parts))  # 中文注释：验证总损失等于各项损失之和
    for param in model.teacher_ir.parameters():  # 中文注释：遍历教师参数以准备制造梯度异常
        param.requires_grad_(True)  # 中文注释：临时开启教师梯度以触发断言
    model._teacher_grad_check_passed = False  # 中文注释：重置断言缓存标记确保重新执行校验
    try:  # 中文注释：使用try捕获预期的断言异常
        model.loss(dummy_inputs, dummy_samples)  # 中文注释：再次执行损失计算此时应触发断言
    except AssertionError as exc:  # 中文注释：捕获断言异常验证保护逻辑生效
        print('teacher_grad_check', str(exc))  # 中文注释：打印异常信息辅助人工确认提示内容
    else:  # 中文注释：若未抛出异常说明断言失效
        raise RuntimeError('期望在教师梯度开启时触发断言，但未发生。')  # 中文注释：主动抛出错误提醒维护者修复检测逻辑
    finally:  # 中文注释：无论是否触发异常都需恢复教师梯度状态
        for param in model.teacher_ir.parameters():  # 中文注释：遍历教师参数恢复冻结
            param.requires_grad_(False)  # 中文注释：重新禁用梯度保持原始设置
        model._teacher_grad_check_passed = False  # 中文注释：重置缓存标记以保证后续运行仍会执行检查
