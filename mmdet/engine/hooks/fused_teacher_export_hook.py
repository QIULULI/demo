# -*- coding: utf-8 -*-  # 指定文件编码确保中文注释在不同环境下不会出现乱码
"""定期导出扩散教师融合权重的训练钩子实现。"""  # 顶部文档字符串概述钩子用途

from pathlib import Path  # 中文注释：用于构建并创建工作目录下的导出路径
from typing import Optional, Tuple  # 中文注释：为可选类型和元组类型注解提供支持

from mmengine.hooks import Hook  # 中文注释：导入MMEngine钩子基类以便继承自定义逻辑
from mmengine.model import is_model_wrapper  # 中文注释：用于在分布式训练场景下解封装模型
from mmengine.runner import Runner  # 中文注释：导入运行器类型以进行类型注解和静态检查

from mmdet.registry import HOOKS  # 中文注释：导入钩子注册表以便在配置文件中通过字符串引用


@HOOKS.register_module()  # 中文注释：将自定义钩子注册到MMDetection框架中供配置调用
class FusedTeacherExportHook(Hook):  # 中文注释：定义用于导出融合后教师权重的钩子类
    """定期调用 ``model.export_fused_teacher`` 保存学生融合权重。"""  # 中文注释：说明钩子的核心行为

    priority = 'LOW'  # 中文注释：设置默认优先级较低以便在常规梯度更新与日志记录之后执行导出

    def __init__(self, interval: int = 1000, by_epoch: bool = False, filename: str = 'student_rgb_fused.pth') -> None:  # 中文注释：构造函数接收导出间隔和文件名配置
        super().__init__()  # 中文注释：显式调用父类初始化以遵循基类约定
        self.interval = int(max(1, interval))  # 中文注释：将间隔转换为整数并至少为1防止无效配置
        self.by_epoch = bool(by_epoch)  # 中文注释：记录触发维度（按轮或按迭代）
        self.filename = filename  # 中文注释：保存导出文件名称供后续拼接完整路径
        self._last_export_signature: Optional[Tuple[str, int]] = None  # 中文注释：记录上一次导出的迭代或轮次标识用于去重
        self._manual_export_counter = 0  # 中文注释：当缺少迭代信息时使用计数器生成唯一标识

    def after_train_iter(self, runner: Runner, batch_idx: int, data_batch=None, outputs=None) -> None:  # 中文注释：在每次训练迭代结束后被调用
        if self.by_epoch:  # 中文注释：当配置为按轮导出时跳过迭代级检查
            return  # 中文注释：直接返回等待轮级回调触发导出
        if not self.every_n_train_iters(runner, self.interval):  # 中文注释：仅在达到设定的迭代间隔时触发导出
            return  # 中文注释：未到导出间隔时提前返回避免频繁写盘
        self._export_weights(runner)  # 中文注释：调用内部导出函数执行权重保存

    def after_train_epoch(self, runner: Runner) -> None:  # 中文注释：在每个训练轮结束后被调用
        if not self.by_epoch:  # 中文注释：当配置为按迭代导出时跳过轮级逻辑
            return  # 中文注释：直接返回保持按迭代导出模式
        if not self.every_n_epochs(runner, self.interval):  # 中文注释：仅在满足轮级间隔时触发导出
            return  # 中文注释：若未达到设定轮数则提前结束
        self._export_weights(runner)  # 中文注释：调用内部导出函数执行权重保存

    def after_train(self, runner: Runner) -> None:  # 中文注释：在整个训练流程结束后执行兜底导出
        self._export_weights(runner)  # 中文注释：调用内部导出逻辑确保最终权重被写盘

    def _build_export_signature(self, runner: Runner) -> Optional[Tuple[str, int]]:  # 中文注释：根据当前迭代或轮次构建导出标识
        if self.by_epoch:  # 中文注释：当以轮为单位判断导出间隔时
            epoch = getattr(runner, 'epoch', None)  # 中文注释：尝试读取运行器当前轮次
            if epoch is None:  # 中文注释：若不存在轮次信息则返回空标识
                return None  # 中文注释：缺少轮次时无法构建标识
            return ('epoch', int(epoch))  # 中文注释：返回包含轮次值的标识元组
        iteration = getattr(runner, 'iter', None)  # 中文注释：当按迭代判断时尝试读取当前迭代数
        if iteration is None:  # 中文注释：若运行器缺少迭代数则返回空标识
            return None  # 中文注释：缺少迭代信息时无法构建标识
        return ('iter', int(iteration))  # 中文注释：返回包含迭代值的标识元组

    def _export_weights(self, runner: Runner) -> None:  # 中文注释：封装导出逻辑便于复用
        signature = self._build_export_signature(runner)  # 中文注释：生成当前导出尝试的唯一标识
        if signature is not None and signature == self._last_export_signature:  # 中文注释：若与上一导出标识相同则跳过重复写盘
            return  # 中文注释：直接返回避免在相同迭代或轮次重复保存
        if signature is None:  # 中文注释：当无法获取迭代或轮次信息时
            self._manual_export_counter += 1  # 中文注释：递增手动计数器以生成唯一标识
            signature = ('manual', self._manual_export_counter)  # 中文注释：构建基于计数器的临时标识
        model = runner.model  # 中文注释：从运行器中获取当前被训练的模型实例
        if is_model_wrapper(model):  # 中文注释：若模型被分布式封装则需要获取其内部真实模块
            model = model.module  # 中文注释：解封装以访问实际的检测器对象
        export_callable = getattr(model, 'export_fused_teacher', None)  # 中文注释：尝试获取模型中的导出方法
        if export_callable is None:  # 中文注释：当模型不支持导出接口时无需继续执行
            return  # 中文注释：直接返回以避免抛出异常
        runner_rank = int(getattr(runner, 'rank', 0))  # 中文注释：读取运行器当前进程的rank并默认主进程为0
        runner_world_size = int(getattr(runner, 'world_size', 1))  # 中文注释：读取总进程数用于判断是否处于分布式环境
        runner_distributed = bool(getattr(runner, 'distributed', runner_world_size > 1))  # 中文注释：推断分布式标志用于兼容不同运行器实现
        if (runner_world_size > 1 or runner_distributed) and runner_rank != 0:  # 中文注释：在非主进程且确认为分布式训练时跳过写盘
            return  # 中文注释：直接返回确保只有rank 0执行导出操作
        work_dir: Optional[str] = getattr(runner, 'work_dir', None)  # 中文注释：读取运行器当前工作目录
        export_root = Path(work_dir) if work_dir is not None else Path('.')  # 中文注释：若未设置工作目录则退回当前目录
        export_root.mkdir(parents=True, exist_ok=True)  # 中文注释：确保导出目录存在以避免保存失败
        export_path = export_root / self.filename  # 中文注释：组合最终的权重文件路径
        export_path.parent.mkdir(parents=True, exist_ok=True)  # 中文注释：递归创建带层级的导出目录防止保存失败
        export_callable(str(export_path))  # 中文注释：调用模型导出函数并传入目标路径
        self._last_export_signature = signature  # 中文注释：保存当前导出标识供后续兜底逻辑判断
        logger = getattr(runner, 'logger', None)  # 中文注释：尝试获取日志记录器用于打印提示
        if logger is not None:  # 中文注释：若运行器配置了日志记录器则输出信息级日志
            logger.info(f'FusedTeacherExportHook 已导出融合教师权重到 {export_path}.')  # 中文注释：记录导出完成与路径详情


if __name__ == '__main__':  # 中文注释：提供最小化自检脚本便于快速验证钩子逻辑

    class _DummyModel:  # 中文注释：定义伪模型以模拟具有export_fused_teacher接口的检测器
        def __init__(self):  # 中文注释：初始化伪模型时记录导出调用次数
            self.called = 0  # 中文注释：记录导出被调用的次数

        def export_fused_teacher(self, path: str) -> None:  # 中文注释：模拟导出函数并写入空文件
            Path(path).write_text('dummy')  # 中文注释：向指定路径写入占位文本以模拟权重文件
            self.called += 1  # 中文注释：导出完成后累加调用计数

    class _DummyRunner:  # 中文注释：构造满足钩子接口需求的最小运行器占位对象
        def __init__(self, rank: int = 0, world_size: int = 1, distributed: bool = False, work_dir: str = './work_dirs/unit_test'):  # 中文注释：初始化伪Runner时支持配置分布式属性与导出目录
            self.model = _DummyModel()  # 中文注释：挂载伪模型
            self.work_dir = work_dir  # 中文注释：指定导出目录
            self.logger = None  # 中文注释：省略日志器以简化示例
            self.iter = 0  # 中文注释：初始化当前迭代计数供every_n_train_iters引用
            self.epoch = 0  # 中文注释：初始化当前轮次计数供every_n_epochs引用
            self.rank = rank  # 中文注释：记录当前进程rank以模拟分布式环境
            self.world_size = world_size  # 中文注释：记录总进程数辅助判定分布式状态
            self.distributed = distributed  # 中文注释：显式标注是否处于分布式模式

    runner = _DummyRunner(rank=0, world_size=2, distributed=True, work_dir='./work_dirs/unit_test_rank0')  # 中文注释：实例化rank 0的伪Runner模拟主进程
    hook = FusedTeacherExportHook(interval=2, by_epoch=False, filename='subdir/student_rgb_fused.pth')  # 中文注释：设置导出间隔并指定带子目录的文件名
    for idx in range(3):  # 中文注释：模拟三次训练迭代覆盖间隔与末次兜底导出
        runner.iter = idx + 1  # 中文注释：手动递增迭代计数以满足Hook的间隔判断
        hook.after_train_iter(runner, idx)  # 中文注释：手动触发迭代结束回调
    hook.after_train(runner)  # 中文注释：训练结束后调用兜底导出逻辑
    hook.after_train(runner)  # 中文注释：再次调用兜底导出验证重复保护逻辑
    export_file = Path(runner.work_dir) / 'subdir/student_rgb_fused.pth'  # 中文注释：拼接期望生成的嵌套路径文件
    assert export_file.exists()  # 中文注释：确认嵌套目录与权重文件均已正确生成
    assert runner.model.called == 2  # 中文注释：确认常规导出与兜底导出共执行两次且重复调用未增加次数
    export_file.unlink()  # 中文注释：清理生成的导出文件以避免影响后续测试

    runner_rank1 = _DummyRunner(rank=1, world_size=2, distributed=True, work_dir='./work_dirs/unit_test_rank1')  # 中文注释：实例化rank 1的伪Runner模拟从进程
    hook_rank1 = FusedTeacherExportHook(interval=1, by_epoch=False, filename='student_rgb_fused_rank1.pth')  # 中文注释：创建新的钩子并设置较短间隔便于触发导出逻辑
    for idx in range(2):  # 中文注释：模拟两次训练迭代验证从进程不会触发写盘
        runner_rank1.iter = idx + 1  # 中文注释：递增迭代计数以满足Hook的间隔判断
        hook_rank1.after_train_iter(runner_rank1, idx)  # 中文注释：调用迭代结束回调
    hook_rank1.after_train(runner_rank1)  # 中文注释：触发训练结束导出以验证从进程仍不写盘
    export_file_rank1 = Path(runner_rank1.work_dir) / 'student_rgb_fused_rank1.pth'  # 中文注释：拼接从进程预期的导出文件路径
    assert not export_file_rank1.exists()  # 中文注释：确认从进程未生成导出文件以符合rank限制
    assert runner_rank1.model.called == 0  # 中文注释：确认从进程模型未执行导出函数
    print('FusedTeacherExportHook 自检通过')  # 中文注释：输出自检通过提示
