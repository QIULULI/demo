# -*- coding: utf-8 -*-  # 指定文件编码以确保中文注释能够被正确解析
"""导出学生权重到教师分支的训练后钩子实现。"""  # 提供模块级文档字符串简述功能用途

import copy  # 引入copy模块以便对学生权重执行深拷贝操作
from mmengine.hooks import Hook  # 导入Hook基类以继承并实现训练阶段回调
from mmengine.model import is_model_wrapper  # 导入工具函数以判断模型是否被分布式封装
from mmengine.runner import Runner  # 导入Runner类型用于类型注解和更清晰的接口说明

from mmdet.registry import HOOKS  # 导入注册表以便将钩子注册到MMDetection框架


@HOOKS.register_module()  # 使用注册装饰器将钩子类注册供配置文件调用
class StudentToTeacherExportHook(Hook):  # 定义导出学生权重到教师分支的钩子类
    """在训练结束前将学生分支权重拷贝至教师分支以保持兼容性。"""  # 类级文档字符串说明钩子意图

    priority = 'VERY_HIGH'  # 设置钩子优先级为极高以确保在保存checkpoint之前执行权重同步

    def after_train(self, runner: Runner) -> None:  # 重写after_train回调在训练完成时执行权重导出逻辑
        model = runner.model  # 从运行器中获取可能被封装的顶层模型实例
        if is_model_wrapper(model):  # 当模型被分布式或数据并行封装时需要取出真实模型
            model = model.module  # 解封装以获取底层实际模型对象
        if hasattr(model, 'model'):  # 当顶层包装器再次嵌套具体检测模型时继续向下取出
            model = model.model  # 提取真正包含师生分支的检测模型
        if not (hasattr(model, 'student') and hasattr(model, 'teacher')):  # 若模型缺少师生结构则无需同步直接返回
            return  # 直接结束钩子执行避免访问不存在的属性
        student_state = copy.deepcopy(model.student.state_dict())  # 深拷贝学生分支当前权重确保后续加载不会共享引用
        load_info = model.teacher.load_state_dict(student_state, strict=False)  # 将学生权重加载到教师分支并允许忽略少量不匹配键
        if hasattr(load_info, 'missing_keys') and (load_info.missing_keys or load_info.unexpected_keys):  # 若存在缺失或多余键则记录日志提示人工检查
            logger = getattr(runner, 'logger', None)  # 尝试从运行器获取日志记录器以输出告警信息
            if logger is not None:  # 当日志记录器存在时输出警告说明同步时的键差异
                logger.warning(f'StudentToTeacherExportHook 检测到未匹配的键：missing={load_info.missing_keys}, unexpected={load_info.unexpected_keys}')  # 使用警告级别记录未匹配键详情帮助排查问题
        logger = getattr(runner, 'logger', None)  # 再次尝试获取日志记录器以输出成功信息（若存在）
        if logger is not None:  # 若存在日志记录器则打印同步成功提示帮助确认钩子已执行
            logger.info('StudentToTeacherExportHook 已将学生分支权重同步至教师分支，确保旧版推理脚本兼容。')  # 输出信息级日志说明导出流程完成
