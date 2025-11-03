# Copyright (c) OpenMMLab. All rights reserved.
from .checkloss_hook import CheckInvalidLossHook
from .mean_teacher_hook import MeanTeacherHook
from .memory_profiler_hook import MemoryProfilerHook
from .num_class_check_hook import NumClassCheckHook
from .pipeline_switch_hook import PipelineSwitchHook
from .set_epoch_info_hook import SetEpochInfoHook
from .sync_norm_hook import SyncNormHook
from .utils import trigger_visualization_hook
from .visualization_hook import (DetVisualizationHook,
                                 GroundingVisualizationHook,
                                 TrackVisualizationHook)
from .yolox_mode_switch_hook import YOLOXModeSwitchHook
from .adaptive_teacher_hook import AdaptiveTeacherHook
from .student_to_teacher_export_hook import StudentToTeacherExportHook  # 引入训练结束后同步学生权重到教师分支的自定义钩子
__all__ = [
    'YOLOXModeSwitchHook', 'SyncNormHook', 'CheckInvalidLossHook',
    'SetEpochInfoHook', 'MemoryProfilerHook', 'DetVisualizationHook',
    'NumClassCheckHook', 'MeanTeacherHook', 'trigger_visualization_hook',
    'PipelineSwitchHook', 'TrackVisualizationHook',
    'GroundingVisualizationHook', 'AdaptiveTeacherHook',
    'StudentToTeacherExportHook'  # 将学生权重导出钩子加入导出列表供配置引用
]
