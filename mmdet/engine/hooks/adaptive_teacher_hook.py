 # Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Optional, Tuple  # 引入类型提示所需的 Dict 与 Tuple，方便后续类型标注

import torch  # 导入基础 PyTorch 库以便访问张量类型与底层操作
import torch.nn as nn  # 导入 PyTorch 的神经网络模块以使用张量与参数相关操作
from mmengine.hooks import Hook  # 从 mmengine 中导入 Hook 基类以实现训练 Hook
from mmengine.logging import MMLogger  # 导入 MMLogger 以便记录警告或错误信息
from mmengine.model import is_model_wrapper  # 导入模型包装器判定函数以获取真实模型
from mmengine.runner import Runner, load_checkpoint  # 导入 Runner 类型与权重加载函数

from mmdet.registry import HOOKS  # 导入 HOOKS 注册器用于注册自定义 Hook


@HOOKS.register_module()
class AdaptiveTeacherHook(Hook):
    """Mean Teacher Hook.

    Mean Teacher is an efficient semi-supervised learning method in
    `Mean Teacher <https://arxiv.org/abs/1703.01780>`_.
    This method requires two models with exactly the same structure,
    as the student model and the teacher model, respectively.
    The student model updates the parameters through gradient descent,
    and the teacher model updates the parameters through
    exponential moving average of the student model.
    Compared with the student model, the teacher model
    is smoother and accumulates more knowledge.

    Args:
        momentum (float): The momentum used for updating teacher's parameter.
            Teacher's parameter are updated with the formula:
           `teacher = (1-momentum) * teacher + momentum * student`.
            Defaults to 0.0001.
        interval (int): Update teacher's parameter every interval iteration.
            Defaults to 1.
        skip_buffers (bool): Whether to skip the model buffers, such as
            batchnorm running stats (running_mean, running_var), it does not
            perform the ema operation. Default to True.
        strict_align (bool): Whether to enforce strict name/shape alignment
            between student and teacher parameters. Default to True. When
            using pretrained teacher checkpoints, ensure class numbers match
            the current task or reinitialize detection heads to avoid shape
            mismatches.
        auto_fill_missing_teacher (bool): Whether to automatically copy missing
            teacher tensors from the student before strict alignment check.
            Default to True to ease checkpoint migration while still reporting
            remaining alignment issues.

    Example::

        # 教师权重需与当前任务类别一致，否则需重新初始化头部
        custom_hooks = [
            dict(
                type='AdaptiveTeacherHook',
                momentum=0.0004,
                strict_align=True,
            )
        ]
    """

    def __init__(self,
                 momentum: float = 0.0004,  # 指定 EMA 的动量系数，默认值参考常用设置
                 interval: int = 1,  # 指定每隔多少个 iteration 进行一次教师模型更新
                 skip_buffer=True,  # 控制是否跳过缓冲区（如 BN 统计量），默认跳过
                 burn_up_iters=12000,  # 预热迭代数，保持原逻辑参数以兼容旧配置
                 strict_align: bool = True,  # 新增 strict_align 控制键与形状严格对齐，默认开启
                 auto_fill_missing_teacher: bool = True) -> None:  # 新增 auto_fill_missing_teacher 控制教师缺失权重自动补齐，默认开启
        assert 0 < momentum < 1  # 确保动量取值合法，避免异常行为
        self.momentum = momentum  # 保存动量系数供后续更新使用
        self.interval = interval  # 保存更新间隔参数
        self.skip_buffers = skip_buffer  # 保存是否跳过缓冲区的设置
        self.burn_up_iters = burn_up_iters  # 保存预热迭代数
        self.strict_align = strict_align  # 保存严格对齐开关，影响不匹配时的处理策略
        self.auto_fill_missing_teacher = auto_fill_missing_teacher  # 保存自动补齐开关，控制是否用学生权重填充教师缺失键

    def before_train(self, runner: Runner) -> None:
        """To check that teacher model and student model exist."""
        model = runner.model

        if is_model_wrapper(model):
            model = model.module
        if hasattr(model, 'model'):
            model = model.model

        assert hasattr(model, 'teacher')
        assert hasattr(model, 'student')

        # load student pretrained model
        if model.semi_train_cfg.get('student_pretrained'):
            load_checkpoint(model.student, model.semi_train_cfg.student_pretrained, map_location='cpu', strict=False)
            model.student.cuda()

        # only do it at initial stage
        if runner.iter == 0:
            self.momentum_update(model, 1)

    def after_train_iter(self,
                         runner: Runner,
                         batch_idx: int,
                         data_batch: Optional[dict] = None,
                         outputs: Optional[dict] = None) -> None:
        """Update teacher's parameter every self.interval iterations."""
        # if self.burn_up_iters > 0:
        #     model = runner.model
        #     if runner.iter < self.burn_up_iters:
        #         return
        #     if is_model_wrapper(model):
        #         model = model.module
        #     if hasattr(model, 'model'):
        #         model = model.model
        #     if runner.iter == self.burn_up_iters:
        #         self.momentum_update(model, 1)
        #         return
        #     if ((runner.iter - self.burn_up_iters) + 1) % self.interval != 0:
        #         return
        #     self.momentum_update(model, self.momentum)
        # else:
        if (runner.iter + 1) % self.interval != 0:
            return
        model = runner.model
        if is_model_wrapper(model):
            model = model.module
        if hasattr(model, 'model'):
            model = model.model
        self.momentum_update(model, self.momentum)

    def momentum_update(self, model: nn.Module, momentum: float) -> None:
        """Compute the moving average of the parameters using exponential
        moving average."""
        logger = MMLogger.get_current_instance()  # 获取全局日志实例，用于输出警告或错误信息

        def _collect_named_tensors(module: nn.Module, include_buffers: bool) -> Dict[str, nn.Parameter]:
            """收集模块中的参数与可选的缓冲区，返回名称到张量的映射。"""
            tensor_map: Dict[str, nn.Parameter] = {}  # 初始化名称与张量映射的字典
            for name, param in module.named_parameters():  # 遍历所有可训练参数
                tensor_map[name] = param  # 将参数写入映射，键为名称，值为张量
            if include_buffers:  # 当需要包含缓冲区时
                for name, buffer in module.named_buffers():  # 遍历所有缓冲区（如 BN 统计量）
                    tensor_map[name] = buffer  # 将缓冲区写入映射以参与对齐校验
            return tensor_map  # 返回收集好的映射

        student_tensors = _collect_named_tensors(model.student, not self.skip_buffers)  # 收集学生模型张量
        teacher_tensors = _collect_named_tensors(model.teacher, not self.skip_buffers)  # 收集教师模型张量

        def _get_tensor_from_teacher(name: str) -> Optional[nn.Parameter]:
            """根据名称分段访问教师模型，返回对应张量引用。"""
            obj = model.teacher  # 从教师模型本体开始逐级访问
            for part in name.split('.'):  # 逐段解析名称中的层级
                if isinstance(obj, nn.Module) and part in obj._parameters:  # 当当前对象是模块且包含同名参数
                    obj = obj._parameters[part]  # 直接取出参数张量
                    continue  # 继续处理下一段名称
                if isinstance(obj, nn.Module) and part in obj._buffers:  # 当当前对象是模块且包含同名缓冲区
                    obj = obj._buffers[part]  # 直接取出缓冲区张量
                    continue  # 继续处理下一段名称
                obj = getattr(obj, part, None)  # 否则尝试通过 getattr 获取子模块或属性
                if obj is None:  # 若获取失败直接返回 None
                    return None  # 终止查找并返回空
            return obj if isinstance(obj, (torch.Tensor, nn.Parameter)) else None  # 仅在返回张量或参数时有效

        filled_keys = []  # 初始化已补齐键的列表
        if self.auto_fill_missing_teacher:  # 当启用自动补齐教师缺失键时
            for name, student_tensor in student_tensors.items():  # 遍历学生模型的全部键与张量
                if name in teacher_tensors:  # 若教师已存在同名张量则跳过
                    continue  # 不做任何修改直接继续
                target_tensor = _get_tensor_from_teacher(name)  # 尝试在教师模型中根据名称找到对应张量引用
                if target_tensor is None:  # 若找不到对应张量
                    continue  # 无法补齐则跳过，保留缺失状态供后续严格校验
                target_tensor.data.copy_(student_tensor.data)  # 使用学生张量的数据覆盖教师张量的数据
                teacher_tensors[name] = target_tensor  # 将补齐后的张量记录到教师映射中以参与后续校验
                filled_keys.append(name)  # 记录已补齐的键名称便于日志输出
            if filled_keys:  # 若存在补齐操作
                logger.info(f"AdaptiveTeacherHook: 自动用学生权重补齐教师缺失键: {filled_keys}")  # 打印信息方便用户确认

        student_shapes: Dict[str, Tuple[int, ...]] = {
            name: tuple(tensor.shape) for name, tensor in student_tensors.items()
        }  # 构建学生模型名称到形状的映射
        teacher_shapes: Dict[str, Tuple[int, ...]] = {
            name: tuple(tensor.shape) for name, tensor in teacher_tensors.items()
        }  # 构建教师模型名称到形状的映射

        missing_in_teacher = [
            name for name in student_shapes.keys() if name not in teacher_shapes
        ]  # 统计教师缺少的键（包含补齐后的最新结果）
        missing_in_student = [
            name for name in teacher_shapes.keys() if name not in student_shapes
        ]  # 统计学生缺少的键
        shape_mismatch = []  # 初始化形状不匹配的列表
        for name in student_shapes.keys():  # 遍历学生模型的全部键
            if name in teacher_shapes and student_shapes[name] != teacher_shapes[name]:  # 当教师存在但形状不同
                shape_mismatch.append(
                    (name, student_shapes[name], teacher_shapes[name])
                )  # 记录不匹配的名称及对应形状

        has_alignment_issue = bool(missing_in_teacher or missing_in_student or shape_mismatch)  # 是否存在对齐问题
        if has_alignment_issue:  # 当检测到对齐问题时
            issue_lines = [
                'AdaptiveTeacherHook: 学生与教师参数未对齐，EMA 已跳过相关键。',  # 说明检测到的问题
                '常见成因：教师 checkpoint 类别数与当前任务不匹配、加载了不同数据集或迭代的权重、头部未重新初始化。',  # 提供常见原因提示
                '建议：使用与当前任务类别一致的教师权重，或在配置中重新初始化检测头部。',  # 提供解决建议
            ]  # 构建基础提示信息
            if filled_keys:  # 如果执行过自动补齐操作
                issue_lines.append(f'已自动补齐的键: {filled_keys}')  # 记录补齐的键以方便用户确认 checkpoint 质量
            if missing_in_teacher:  # 如果教师仍缺少键
                issue_lines.append(f'教师缺少的键: {missing_in_teacher}')  # 记录缺失键列表
            if missing_in_student:  # 如果学生缺少键
                issue_lines.append(f'学生缺少的键: {missing_in_student}')  # 记录缺失键列表
            if shape_mismatch:  # 如果存在形状不匹配
                mismatch_msg = [
                    f'{name}: 学生形状{stu_shape}, 教师形状{tea_shape}'
                    for name, stu_shape, tea_shape in shape_mismatch
                ]  # 逐项格式化形状差异
                issue_lines.append('形状不一致的键: ' + '; '.join(mismatch_msg))  # 汇总形状差异信息

            detail_msg = '\n'.join(issue_lines)  # 将所有提示信息拼接为多行字符串
            if self.strict_align:  # 当启用严格对齐时
                raise RuntimeError(detail_msg)  # 直接抛出异常以阻断训练并提示用户
            logger.warning(detail_msg)  # 在非严格模式下仅记录警告并继续

        valid_names = [
            name for name in student_tensors.keys()
            if name in teacher_tensors and student_shapes.get(name) == teacher_shapes.get(name)
        ]  # 仅保留键集合与形状一致的名称用于 EMA

        for name in valid_names:  # 遍历所有合法且匹配的键
            student_tensor = student_tensors[name]  # 取出学生模型对应张量
            teacher_tensor = teacher_tensors[name]  # 取出教师模型对应张量
            if not (student_tensor.dtype.is_floating_point and teacher_tensor.dtype.is_floating_point):  # 确保仅更新浮点类型
                continue  # 对非浮点张量直接跳过
            teacher_tensor.data.mul_(1 - momentum).add_(student_tensor.data, alpha=momentum)  # 按照 EMA 公式更新教师张量


# 小型自检：仅测试导入与核心逻辑调用
if __name__ == '__main__':  # 当脚本直接执行时运行快速自检
    dummy_hook = AdaptiveTeacherHook()  # 创建 Hook 实例
    print('AdaptiveTeacherHook 已加载，strict_align 默认开启。')  # 打印提示信息
