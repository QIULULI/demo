# -*- coding: utf-8 -*-  # 指定文件编码确保中文注释在不同环境下不会出现乱码
"""定期导出扩散教师融合权重的训练钩子实现。"""  # 顶部文档字符串概述钩子用途

from pathlib import Path  # 中文注释：用于构建并创建工作目录下的导出路径
from typing import Optional  # 中文注释：为可选类型注解提供支持

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

    def _export_weights(self, runner: Runner) -> None:  # 中文注释：封装导出逻辑便于复用
        model = runner.model  # 中文注释：从运行器中获取当前被训练的模型实例
        if is_model_wrapper(model):  # 中文注释：若模型被分布式封装则需要获取其内部真实模块
            model = model.module  # 中文注释：解封装以访问实际的检测器对象
        export_callable = getattr(model, 'export_fused_teacher', None)  # 中文注释：尝试获取模型中的导出方法
        if export_callable is None:  # 中文注释：当模型不支持导出接口时无需继续执行
            return  # 中文注释：直接返回以避免抛出异常
        work_dir: Optional[str] = getattr(runner, 'work_dir', None)  # 中文注释：读取运行器当前工作目录
        export_root = Path(work_dir) if work_dir is not None else Path('.')  # 中文注释：若未设置工作目录则退回当前目录
        export_root.mkdir(parents=True, exist_ok=True)  # 中文注释：确保导出目录存在以避免保存失败
        export_path = export_root / self.filename  # 中文注释：组合最终的权重文件路径
        export_callable(str(export_path))  # 中文注释：调用模型导出函数并传入目标路径
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
        def __init__(self):  # 中文注释：初始化伪Runner时配置必要属性
            self.model = _DummyModel()  # 中文注释：挂载伪模型
            self.work_dir = './work_dirs/unit_test'  # 中文注释：指定导出目录
            self.logger = None  # 中文注释：省略日志器以简化示例
            self.iter = 0  # 中文注释：初始化当前迭代计数供every_n_train_iters引用
            self.epoch = 0  # 中文注释：初始化当前轮次计数供every_n_epochs引用

    runner = _DummyRunner()  # 中文注释：实例化伪Runner
    hook = FusedTeacherExportHook(interval=1, by_epoch=False)  # 中文注释：创建钩子对象设置每迭代导出
    for idx in range(2):  # 中文注释：模拟两次训练迭代
        runner.iter = idx + 1  # 中文注释：手动递增迭代计数以满足Hook的间隔判断
        hook.after_train_iter(runner, idx)  # 中文注释：手动触发迭代结束回调
    assert runner.model.called == 2  # 中文注释：确认导出函数被调用两次以验证逻辑正确
    print('FusedTeacherExportHook 自检通过')  # 中文注释：输出自检通过提示
