# -*- coding: utf-8 -*-  # 指定文件编码以确保中文注释能够被正确解析
"""导出学生权重到教师分支并同步落盘教师检查点的训练后钩子实现。"""  # 提供模块级文档字符串描述钩子职责

from typing import List  # 中文注释：用于类型提示候选教师键列表
import copy  # 引入copy模块以便对学生权重执行深拷贝操作
from pathlib import Path  # 中文注释：用于构造权重导出路径

import torch  # 中文注释：用于保存可训练教师的状态字典
from mmengine.hooks import Hook  # 导入Hook基类以继承并实现训练阶段回调
from mmengine.model import is_model_wrapper  # 导入工具函数以判断模型是否被分布式封装
from mmengine.runner import Runner  # 导入Runner类型用于类型注解和更清晰的接口说明

from mmdet.registry import HOOKS  # 导入注册表以便将钩子注册到MMDetection框架


@HOOKS.register_module()  # 使用注册装饰器将钩子类注册供配置文件调用
class StudentToTeacherExportHook(Hook):  # 定义导出学生权重到教师分支的钩子类
    """在训练结束前将学生分支权重拷贝至教师分支并导出指定教师权重。"""  # 类级文档字符串说明钩子意图

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
        if hasattr(model, 'trainable_diff_teacher_keys') and model.trainable_diff_teacher_keys:  # 中文注释：当模型存在可训练扩散教师时尝试导出目标教师
            teacher_repository = getattr(model, 'diff_teacher_bank', None) or getattr(model, 'diff_detectors', None)  # 中文注释：优先获取扩散教师资源库以便检索真实教师实例
            export_candidates: List[str] = []  # 中文注释：初始化候选教师键列表用于记录可能的导出对象
            explicit_candidate = getattr(model, 'export_target_diff_teacher_key', None)  # 中文注释：尝试读取显式配置的导出教师标识
            if explicit_candidate is not None:  # 中文注释：当存在显式配置时进入解析流程
                if isinstance(explicit_candidate, (list, tuple, set)):  # 中文注释：若配置为可迭代集合则逐个展开
                    export_candidates.extend(list(explicit_candidate))  # 中文注释：将集合中的所有候选键写入列表
                else:  # 中文注释：否则视为单个标识直接追加
                    export_candidates.append(explicit_candidate)  # 中文注释：追加显式指定的教师键
            if not export_candidates:  # 中文注释：若未通过显式配置获得目标教师则尝试读取备用字段
                fallback_candidate = getattr(model, 'target_diff_teacher_key', None)  # 中文注释：读取可能存在的目标教师字段以保持兼容
                if fallback_candidate is not None:  # 中文注释：当检测到备用字段时进入解析
                    if isinstance(fallback_candidate, (list, tuple, set)):  # 中文注释：若备用字段以集合形式提供则展开
                        export_candidates.extend(list(fallback_candidate))  # 中文注释：将所有备用候选写入列表
                    else:  # 中文注释：否则直接追加单一键值
                        export_candidates.append(fallback_candidate)  # 中文注释：记录备用目标教师键
            if not export_candidates:  # 中文注释：若仍未找到候选则默认使用全部可训练教师列表
                export_candidates = list(model.trainable_diff_teacher_keys)  # 中文注释：复制可训练教师键以免后续修改影响原列表
            target_teacher_key = None  # 中文注释：初始化最终选择的导出教师键
            if teacher_repository:  # 中文注释：仅在成功获取教师资源库时才进行匹配
                for candidate_key in export_candidates:  # 中文注释：遍历候选键列表查找存在的教师
                    if candidate_key in teacher_repository:  # 中文注释：当当前候选存在于教师资源库中时选定为目标
                        target_teacher_key = candidate_key  # 中文注释：记录当前命中的教师键供后续导出使用
                        break  # 中文注释：找到首个匹配后立即退出循环以避免多教师导出
                if target_teacher_key is None and model.trainable_diff_teacher_keys:  # 中文注释：若未命中候选但仍存在可训练教师则回退到列表首个元素
                    fallback_key = model.trainable_diff_teacher_keys[0]  # 中文注释：取出可训练教师列表中的首个键作为兜底选择
                    if fallback_key in teacher_repository:  # 中文注释：确认兜底教师确实存在于资源库中
                        target_teacher_key = fallback_key  # 中文注释：使用兜底键作为最终导出对象
            if target_teacher_key is not None and teacher_repository:  # 中文注释：当成功确定导出教师且教师资源库有效时执行落盘
                teacher_module = teacher_repository[target_teacher_key]  # 中文注释：根据目标键获取对应的教师模块实例
                teacher_state = {  # 中文注释：初始化教师权重字典并将参数迁移至CPU以减少保存体积
                    name: param.detach().cpu() for name, param in teacher_module.state_dict().items()  # 中文注释：遍历教师state_dict逐项拷贝张量到CPU
                }  # 中文注释：结束教师权重字典构造
                work_dir = getattr(runner, 'work_dir', None)  # 中文注释：读取当前运行目录以确定导出路径
                if work_dir is not None:  # 中文注释：仅在工作目录存在时执行保存操作
                    export_path = Path(work_dir) / 'Dual_Diffusion_Teacher.pth'  # 中文注释：构建教师权重文件的标准输出路径
                    torch.save({'state_dict': teacher_state}, export_path)  # 中文注释：使用常规字典格式写入教师权重以便后续加载
                    if logger is not None:  # 中文注释：若存在日志记录器则记录导出成功信息
                        logger.info('StudentToTeacherExportHook 已导出目标教师"%s"的权重至 %s.' % (target_teacher_key, export_path))  # 中文注释：通过格式化字符串输出导出教师键与保存路径
            else:  # 中文注释：当未能匹配到目标教师或资源库为空时输出告警
                if logger is not None:  # 中文注释：仅在日志记录器存在时输出警告
                    logger.warning('StudentToTeacherExportHook 未找到匹配的可训练教师用于导出，请检查配置。')  # 中文注释：提示用户检查可训练教师配置
