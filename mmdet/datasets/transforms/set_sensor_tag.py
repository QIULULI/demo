"""自定义传感器标记变换模块，确保管线中保留传感器元信息。"""  # 文件开头提供中文文档字符串描述用途
from typing import Any, Dict  # 导入类型注解以便明确输入输出结构

from mmcv.transforms.base import BaseTransform  # 从mmcv引入基础变换基类

from mmdet.registry import TRANSFORMS  # 导入变换注册器以注册自定义组件


@TRANSFORMS.register_module()  # 通过注册器装饰器声明该类可在配置中引用
class SetSensorTag(BaseTransform):  # 定义自定义变换类用于写入传感器标记
    """在结果字典及已有数据样本中补充传感器信息。"""  # 类级中文文档字符串说明功能

    def __init__(self, sensor: str) -> None:  # 初始化函数接收传感器名称字符串
        self.sensor = sensor  # 保存传感器标识以便在transform阶段写入

    def transform(self, results: Dict[str, Any]) -> Dict[str, Any]:  # 定义核心变换逻辑处理输入结果字典
        results['sensor'] = self.sensor  # 在结果字典中记录当前传感器类型
        data_samples = results.get('data_samples', None)  # 读取可能存在的数据样本对象以便同步元信息
        if data_samples is not None:  # 若数据样本存在则需要写入对应的传感器信息
            if isinstance(data_samples, dict):  # 若数据样本以字典形式存储多个分支数据
                for _branch, sample in data_samples.items():  # 遍历各分支的数据样本确保全部更新
                    if hasattr(sample, 'set_metainfo') and sample is not None:  # 确认对象支持设置元信息且非空
                        sample.set_metainfo({'sensor': self.sensor})  # 直接调用接口写入传感器元信息
            elif isinstance(data_samples, (list, tuple)):  # 若数据样本以序列形式存储多个对象
                for sample in data_samples:  # 遍历序列中的每个数据样本
                    if hasattr(sample, 'set_metainfo') and sample is not None:  # 确认对象支持设置元信息且非空
                        sample.set_metainfo({'sensor': self.sensor})  # 更新序列中每个数据样本的传感器元信息
            else:  # 若数据样本为单一数据对象
                if hasattr(data_samples, 'set_metainfo'):  # 确认对象具备设置元信息方法
                    data_samples.set_metainfo({'sensor': self.sensor})  # 更新单个数据样本的元信息
        return results  # 返回包含传感器标记的结果字典
