"""为多源采样器提供功能性测试并附带详细中文注释。"""  # 中文模块文档字符串说明
import importlib.util  # 导入importlib.util以按路径动态加载模块
import itertools  # 导入itertools以便对迭代器进行切片操作
import random  # 导入random以在缺失numpy时提供随机功能
import sys  # 导入sys以便向模块注册表注入模拟模块
import types  # 导入types以创建动态模块对象
from bisect import bisect_right  # 导入bisect_right用于根据累积大小反推数据源索引
from pathlib import Path  # 导入Path以便构造模块路径

import pytest  # 导入pytest以组织测试用例

try:  # 使用try导入numpy以兼容缺失依赖的环境
    import numpy as np  # 尝试导入numpy提供数值功能
except ModuleNotFoundError:  # 当numpy缺失时提供轻量级替身
    class _FakeArray(list):  # 定义可用于模拟numpy数组的轻量容器
        def sum(self):  # 提供sum方法用于求和
            return sum(self)  # 直接调用内建sum

        def __truediv__(self, value):  # 支持除法操作
            return _FakeArray([elem / value for elem in self])  # 返回逐元素相除的新数组

    class _FakeRandomModule:  # 定义简化的随机模块用于模拟np.random
        def __init__(self):  # 构造函数初始化内部随机数生成器
            self._rng = random.Random()  # 使用Python内建随机生成器

        def choice(self, choices, p=None):  # 实现choice方法以支持按概率采样
            sequence = list(choices)  # 将候选转换为列表以便索引访问
            if not sequence:  # 若候选为空则抛出异常
                raise ValueError('Choices cannot be empty.')  # 提示调用者候选不能为空
            if p is None:  # 若未提供概率则均匀选择
                return sequence[self._rng.randrange(len(sequence))]  # 使用均匀随机索引返回元素
            total = sum(p)  # 计算概率和以做归一化
            if total <= 0:  # 防止概率和非正导致异常
                raise ValueError('Sum of probabilities must be positive.')  # 抛出异常提示
            r = self._rng.random()  # 生成0到1之间的随机数
            cumulative = 0.0  # 初始化累积概率
            for value, prob in zip(sequence, p):  # 遍历候选及其概率
                cumulative += prob / total  # 累加归一化概率
                if r <= cumulative:  # 若随机值落在当前区间
                    return value  # 返回当前候选
            return sequence[-1]  # 若循环结束则返回最后一个候选

    def _fake_zeros(size, dtype=None):  # 定义zeros函数的轻量实现
        if isinstance(size, int):  # 仅支持整数长度
            return _FakeArray([0] * size)  # 返回指定长度的零数组容器
        raise TypeError('size must be an integer')  # 对非法输入抛出异常

    np = types.ModuleType('numpy')  # 创建名为numpy的模块替身
    np.int64 = int  # 使用内建int模拟int64
    np.float64 = float  # 使用内建float模拟float64
    np.zeros = _fake_zeros  # 将轻量实现绑定为zeros函数
    np.random = _FakeRandomModule()  # 绑定模拟的随机子模块
    sys.modules.setdefault('numpy', np)  # 将替身模块注册到sys.modules

mmengine_module = types.ModuleType('mmengine')  # 创建mmengine根模块的模拟对象
mmengine_dataset_module = types.ModuleType('mmengine.dataset')  # 创建mmengine.dataset模拟模块


class _BaseDataset:  # 定义基础数据集模拟类以满足类型引用
    pass  # 保持空实现即可满足采样器的类型检查需求


mmengine_dataset_module.BaseDataset = _BaseDataset  # 将模拟数据集类挂载到模块上
mmengine_dist_module = types.ModuleType('mmengine.dist')  # 创建mmengine.dist模拟模块


def _fake_get_dist_info():  # 定义模拟的分布式信息函数
    return 0, 1  # 返回单机默认的rank与world_size


def _fake_sync_random_seed():  # 定义模拟的同步随机种子函数
    return 0  # 返回固定种子以确保确定性


mmengine_dist_module.get_dist_info = _fake_get_dist_info  # 将模拟函数绑定到模块
mmengine_dist_module.sync_random_seed = _fake_sync_random_seed  # 将同步随机种子函数绑定到模块
mmengine_module.dataset = mmengine_dataset_module  # 将dataset子模块注册到mmengine根模块
mmengine_module.dist = mmengine_dist_module  # 将dist子模块注册到mmengine根模块
sys.modules.setdefault('mmengine', mmengine_module)  # 将模拟mmengine写入sys.modules避免导入错误
sys.modules.setdefault('mmengine.dataset', mmengine_dataset_module)  # 注册dataset子模块
sys.modules.setdefault('mmengine.dist', mmengine_dist_module)  # 注册dist子模块

mmdet_module = types.ModuleType('mmdet')  # 创建mmdet根模块模拟对象
mmdet_registry_module = types.ModuleType('mmdet.registry')  # 创建mmdet.registry模拟模块


class _FakeRegistry:  # 定义简易注册表以模拟DATA_SAMPLERS装饰器
    def register_module(self, *args, **kwargs):  # 提供register_module方法以兼容调用
        def _decorator(obj):  # 定义装饰器直接返回传入对象
            return obj  # 不做任何额外处理

        return _decorator  # 返回装饰器函数


mmdet_registry_module.DATA_SAMPLERS = _FakeRegistry()  # 将模拟注册表放入模块
mmdet_module.registry = mmdet_registry_module  # 将registry子模块挂载至根模块
sys.modules.setdefault('mmdet', mmdet_module)  # 注册mmdet根模块
sys.modules.setdefault('mmdet.registry', mmdet_registry_module)  # 注册mmdet.registry模块

torch_module = types.ModuleType('torch')  # 创建torch模块的模拟对象


class _FakeTensor(list):  # 定义简化的张量类型用于提供tolist方法
    def tolist(self):  # 实现tolist以兼容真实张量接口
        return list(self)  # 返回自身的列表拷贝


class _FakeGenerator:  # 定义简化的随机数生成器
    def __init__(self):  # 构造函数初始化随机数状态
        self._rng = random.Random()  # 使用Python随机生成器

    def manual_seed(self, seed):  # 提供manual_seed方法以设置种子
        self._rng.seed(seed)  # 将种子传递给内部随机生成器


def _fake_randperm(n, generator=None):  # 定义模拟的randperm函数
    sequence = list(range(n))  # 创建0到n-1的序列
    rng = generator._rng if generator is not None else random.Random()  # 选择适当的随机源
    rng.shuffle(sequence)  # 对序列就地打乱
    return _FakeTensor(sequence)  # 返回带有tolist方法的张量替身


def _fake_arange(n):  # 定义模拟的arange函数
    return _FakeTensor(range(n))  # 返回带有tolist方法的张量替身


torch_module.Generator = _FakeGenerator  # 将生成器类挂载到torch模拟模块
torch_module.randperm = _fake_randperm  # 将randperm函数挂载到torch模拟模块
torch_module.arange = _fake_arange  # 将arange函数挂载到torch模拟模块
sys.modules.setdefault('torch', torch_module)  # 注册torch模块以供导入

torch_utils_module = types.ModuleType('torch.utils')  # 创建torch.utils模拟模块
torch_utils_data_module = types.ModuleType('torch.utils.data')  # 创建torch.utils.data模拟模块


class _FakeSampler:  # 定义最小化的Sampler基类
    def __init__(self, data_source=None):  # 构造函数接受可选数据源
        self.data_source = data_source  # 保存传入的数据源


torch_utils_data_module.Sampler = _FakeSampler  # 在模拟模块中注册Sampler类
torch_utils_module.data = torch_utils_data_module  # 将data子模块挂载至torch.utils
sys.modules.setdefault('torch.utils', torch_utils_module)  # 注册torch.utils模块
sys.modules.setdefault('torch.utils.data', torch_utils_data_module)  # 注册torch.utils.data模块

MODULE_PATH = Path(__file__).resolve().parents[3] / 'mmdet/datasets/samplers/multi_source_sampler.py'  # 计算目标模块的文件路径
spec = importlib.util.spec_from_file_location('multi_source_sampler', MODULE_PATH)  # 基于路径创建模块描述
multi_source_sampler = importlib.util.module_from_spec(spec)  # 根据描述创建模块对象
assert spec.loader is not None  # 确保加载器存在以避免后续调用失败
spec.loader.exec_module(multi_source_sampler)  # 执行模块以便使用其中的类

GroupMultiSourceSampler = multi_source_sampler.GroupMultiSourceSampler  # 从动态加载的模块获取新采样器类
MultiSourceSampler = multi_source_sampler.MultiSourceSampler  # 获取基础多源采样器类


class DummyDataset:  # 定义简单数据集以提供宽高信息
    def __init__(self, shapes):  # 构造函数接收尺寸列表
        self.shapes = list(shapes)  # 将传入尺寸存储为列表以支持多次访问

    def __len__(self):  # 返回数据集中样本数量
        return len(self.shapes)  # 使用内部列表长度作为样本数

    def get_data_info(self, idx):  # 根据索引返回样本的宽高信息
        width, height = self.shapes[idx]  # 解包指定索引处的宽度和高度
        return dict(width=width, height=height)  # 返回符合采样器预期的字典


class DummyConcatDataset:  # 定义模拟的ConcatDataset以符合采样器接口
    def __init__(self, datasets):  # 构造函数接收子数据集列表
        self.datasets = list(datasets)  # 持有传入的子数据集实例
        self.cumulative_sizes = []  # 初始化累积大小列表
        total = 0  # 累加器用于累计样本数
        for dataset in self.datasets:  # 遍历每个子数据集
            total += len(dataset)  # 累加当前子数据集的样本数量
            self.cumulative_sizes.append(total)  # 将累加结果记录到累积大小列表

    def __len__(self):  # 返回整体数据集的样本总数
        return self.cumulative_sizes[-1] if self.cumulative_sizes else 0  # 若存在累积记录则返回最后一个值否则为零


class LegacyGroupMultiSourceSampler(MultiSourceSampler):  # 基于旧版逻辑的采样器实现
    def __init__(self, dataset, batch_size, source_ratio, shuffle=True, seed=None):  # 构造函数匹配旧版签名
        super().__init__(  # 调用基础多源采样器完成通用初始化
            dataset=dataset,  # 传递组合数据集
            batch_size=batch_size,  # 指定批次大小
            source_ratio=source_ratio,  # 指定源比例
            shuffle=shuffle,  # 指定是否打乱
            seed=seed)  # 指定随机种子
        self._get_source_group_info()  # 构建旧版所使用的分组统计信息
        self.group_source2inds = [{  # 初始化旧版依赖的分组到迭代器的映射
            source: self._indices_of_rank(  # 对每个数据源构建按rank切分的无限索引流
                self.group2size_per_source[source][group])  # 使用该分组在数据源中的样本数量作为上限
            for source in range(len(dataset.datasets))  # 遍历所有子数据源
        } for group in range(len(self.group_ratio))]  # 对每个分组生成上述映射

    def _get_source_group_info(self):  # 复刻旧版的分组统计逻辑
        self.group2size_per_source = [{0: 0, 1: 0}, {0: 0, 1: 0}]  # 初始化两个数据源的分组计数字典
        self.group2inds_per_source = [{0: [], 1: []}, {0: [], 1: []}]  # 初始化两个数据源的分组索引字典
        for source, dataset in enumerate(self.dataset.datasets):  # 遍历两个数据源
            for idx in range(len(dataset)):  # 遍历当前数据源中的每个样本索引
                data_info = dataset.get_data_info(idx)  # 获取样本宽高信息
                width, height = data_info['width'], data_info['height']  # 解包宽度与高度
                group = 0 if width < height else 1  # 按旧逻辑判断分组编号
                self.group2size_per_source[source][group] += 1  # 增加对应分组的样本计数
                self.group2inds_per_source[source][group].append(idx)  # 将样本索引加入对应分组列表
        self.group_sizes = np.zeros(2, dtype=np.int64)  # 创建长度为2的数组以统计整体分组样本数
        for group2size in self.group2size_per_source:  # 遍历每个数据源的分组计数
            for group, size in group2size.items():  # 遍历分组及其样本数
                self.group_sizes[group] += size  # 将样本数累加到全局分组统计
        total = int(sum(self.group_sizes))  # 计算总样本数
        if total == 0:  # 检查是否存在可采样样本
            raise ValueError('All groups are empty, unable to sample data.')  # 与旧逻辑一致抛出异常
        self.group_ratio = [count / total for count in self.group_sizes]  # 计算每个分组的采样概率

    def __iter__(self):  # 复刻旧版的迭代逻辑
        batch_buffer = []  # 初始化批次缓冲区
        while True:  # 构建无限迭代器
            group = np.random.choice(  # 按照分组概率选择当前批次的分组
                list(range(len(self.group_ratio))), p=self.group_ratio)  # 提供分组列表及其概率
            for source, num in enumerate(self.num_per_source):  # 遍历每个数据源及其需要的样本数
                batch_buffer_per_source = []  # 初始化当前源的缓冲区
                for idx in self.group_source2inds[group][source]:  # 遍历该分组对应数据源的索引迭代器
                    idx = self.group2inds_per_source[source][group][idx] + self.cumulative_sizes[source]  # 将局部索引转换为全局索引
                    batch_buffer_per_source.append(idx)  # 将转换后的索引加入缓冲区
                    if len(batch_buffer_per_source) == num:  # 如果达到所需数量则停止
                        batch_buffer += batch_buffer_per_source  # 将当前源的索引加入整体批次
                        break  # 结束当前源的采样
            yield from batch_buffer  # 依次输出批次中的索引
            batch_buffer = []  # 清空缓冲区以准备下一批


@pytest.fixture()  # 使用pytest夹具为np.random.choice提供可控返回值
def fixed_choice(monkeypatch):  # 接收monkeypatch以便动态修改函数
    def _patch(sequence):  # 定义内部函数接收预设返回序列
        iterator = iter(sequence)  # 将序列转为迭代器以按顺序返回值

        def _fake_choice(*args, **kwargs):  # 定义伪造的choice函数
            try:  # 尝试从迭代器获取下一个值
                return next(iterator)  # 若成功则返回该值
            except StopIteration:  # 若迭代器耗尽则回退到最后一个值
                return sequence[-1]  # 使用最后一个值保持确定性

        monkeypatch.setattr(np.random, 'choice', _fake_choice)  # 将np.random.choice替换为伪造版本

    return _patch  # 返回内部函数供测试调用


def test_group_multi_source_sampler_three_sources(fixed_choice):  # 测试三源采样比例是否满足设定
    ds0 = DummyDataset([(10, 20), (12, 24), (30, 10)])  # 构造第一份数据集包含竖图与横图
    ds1 = DummyDataset([(11, 22), (13, 26), (15, 30), (28, 14)])  # 构造第二份数据集同样提供多样尺寸
    ds2 = DummyDataset([(9, 18), (10, 30), (12, 48), (60, 20), (14, 28)])  # 构造第三份数据集以测试更多样本
    dataset = DummyConcatDataset([ds0, ds1, ds2])  # 使用三个子数据集构造ConcatDataset模拟对象

    sampler = GroupMultiSourceSampler(  # 创建待测采样器实例
        dataset=dataset,  # 指定组合数据集
        batch_size=6,  # 设置批次大小为6
        source_ratio=[1, 2, 3],  # 按照1:2:3的比例采样三个数据源
        shuffle=False,  # 关闭shuffle以便测试确定性
        seed=0)  # 固定随机种子便于复现

    fixed_choice([0, 0, 0])  # 将分组选择固定为竖图分组以稳定输出

    iterator = iter(sampler)  # 获取采样器的迭代器
    batch = [next(iterator) for _ in range(sampler.batch_size)]  # 收集一个批次的全局索引

    counts = [0] * len(dataset.datasets)  # 初始化计数器以统计每个源的命中次数
    for idx in batch:  # 遍历批次中的每个索引
        source = bisect_right(sampler.cumulative_sizes, idx) - 1  # 根据累积大小确定索引所属数据源
        counts[source] += 1  # 对应数据源的计数加一

    assert counts == sampler.num_per_source  # 验证每个数据源的采样数量符合设定比例


def test_two_source_sampling_matches_legacy_implementation(fixed_choice):  # 确认双源场景与旧实现一致
    ds0 = DummyDataset([(10, 20), (20, 10), (12, 24), (18, 36)])  # 构造第一份双源数据集
    ds1 = DummyDataset([(9, 18), (14, 28), (30, 10), (40, 20)])  # 构造第二份双源数据集
    dataset = DummyConcatDataset([ds0, ds1])  # 将两个数据集合并为ConcatDataset

    sampler = GroupMultiSourceSampler(  # 构造新版采样器实例
        dataset=dataset,  # 指定数据集
        batch_size=4,  # 设置批次大小
        source_ratio=[1, 1],  # 设置两个数据源等比例采样
        shuffle=False,  # 关闭shuffle以保证对比公平
        seed=0)  # 固定随机种子

    legacy_sampler = LegacyGroupMultiSourceSampler(  # 构造旧逻辑采样器用于对比
        dataset=dataset,  # 指定同一数据集
        batch_size=4,  # 保持批次大小一致
        source_ratio=[1, 1],  # 保持采样比例一致
        shuffle=False,  # 关闭shuffle保持一致性
        seed=0)  # 固定随机种子保持一致

    sample_count = sampler.batch_size * 3  # 计划比较连续三个批次的输出
    fixed_choice([0, 1, 0, 1, 0, 1])  # 为新版采样器预设分组选择序列
    new_indices = list(itertools.islice(iter(sampler), sample_count))  # 从新版采样器获取指定数量的索引
    fixed_choice([0, 1, 0, 1, 0, 1])  # 重置分组选择供旧版采样器复用相同序列
    old_indices = list(itertools.islice(iter(legacy_sampler), sample_count))  # 从旧版采样器获取对应数量的索引

    assert new_indices == old_indices  # 验证两个实现输出完全一致
