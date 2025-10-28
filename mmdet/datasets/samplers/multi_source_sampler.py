# Copyright (c) OpenMMLab. All rights reserved.
import itertools
from collections import defaultdict
from typing import Iterator, List, Optional, Sized, Union

import numpy as np
import torch
from mmengine.dataset import BaseDataset
from mmengine.dist import get_dist_info, sync_random_seed
from torch.utils.data import Sampler

from mmdet.registry import DATA_SAMPLERS


@DATA_SAMPLERS.register_module()
class MultiSourceSampler(Sampler):
    r"""Multi-Source Infinite Sampler.

    According to the sampling ratio, sample data from different
    datasets to form batches.

    Args:
        dataset (Sized): The dataset.
        batch_size (int): Size of mini-batch.
        source_ratio (list[int | float]): The sampling ratio of different
            source datasets in a mini-batch.
        shuffle (bool): Whether shuffle the dataset or not. Defaults to True.
        seed (int, optional): Random seed. If None, set a random seed.
            Defaults to None.

    Examples:
        >>> dataset_type = 'ConcatDataset'
        >>> sub_dataset_type = 'CocoDataset'
        >>> data_root = 'data/coco/'
        >>> sup_ann = '../coco_semi_annos/instances_train2017.1@10.json'
        >>> unsup_ann = '../coco_semi_annos/' \
        >>>             'instances_train2017.1@10-unlabeled.json'
        >>> dataset = dict(type=dataset_type,
        >>>     datasets=[
        >>>         dict(
        >>>             type=sub_dataset_type,
        >>>             data_root=data_root,
        >>>             ann_file=sup_ann,
        >>>             data_prefix=dict(img='train2017/'),
        >>>             filter_cfg=dict(filter_empty_gt=True, min_size=32),
        >>>             pipeline=sup_pipeline),
        >>>         dict(
        >>>             type=sub_dataset_type,
        >>>             data_root=data_root,
        >>>             ann_file=unsup_ann,
        >>>             data_prefix=dict(img='train2017/'),
        >>>             filter_cfg=dict(filter_empty_gt=True, min_size=32),
        >>>             pipeline=unsup_pipeline),
        >>>         ])
        >>>     train_dataloader = dict(
        >>>         batch_size=5,
        >>>         num_workers=5,
        >>>         persistent_workers=True,
        >>>         sampler=dict(type='MultiSourceSampler',
        >>>             batch_size=5, source_ratio=[1, 4]),
        >>>         batch_sampler=None,
        >>>         dataset=dataset)
    """

    def __init__(self,
                 dataset: Sized,
                 batch_size: int,
                 source_ratio: List[Union[int, float]],
                 shuffle: bool = True,
                 seed: Optional[int] = None) -> None:

        assert hasattr(dataset, 'cumulative_sizes'),\
            f'The dataset must be ConcatDataset, but get {dataset}'
        assert isinstance(batch_size, int) and batch_size > 0, \
            'batch_size must be a positive integer value, ' \
            f'but got batch_size={batch_size}'
        assert isinstance(source_ratio, list), \
            f'source_ratio must be a list, but got source_ratio={source_ratio}'
        assert len(source_ratio) == len(dataset.cumulative_sizes), \
            'The length of source_ratio must be equal to ' \
            f'the number of datasets, but got source_ratio={source_ratio}'

        rank, world_size = get_dist_info()
        self.rank = rank
        self.world_size = world_size

        self.dataset = dataset
        self.cumulative_sizes = [0] + dataset.cumulative_sizes
        self.batch_size = batch_size
        self.source_ratio = source_ratio

        self.num_per_source = [
            int(batch_size * sr / sum(source_ratio)) for sr in source_ratio
        ]
        self.num_per_source[0] = batch_size - sum(self.num_per_source[1:])

        assert sum(self.num_per_source) == batch_size, \
            'The sum of num_per_source must be equal to ' \
            f'batch_size, but get {self.num_per_source}'

        self.seed = sync_random_seed() if seed is None else seed
        self.shuffle = shuffle
        self.source2inds = {
            source: self._indices_of_rank(len(ds))
            for source, ds in enumerate(dataset.datasets)
        }

    def _infinite_indices(self, sample_size: int) -> Iterator[int]:
        """Infinitely yield a sequence of indices."""
        g = torch.Generator()
        g.manual_seed(self.seed)
        while True:
            if self.shuffle:
                yield from torch.randperm(sample_size, generator=g).tolist()
            else:
                yield from torch.arange(sample_size).tolist()

    def _indices_of_rank(self, sample_size: int) -> Iterator[int]:
        """Slice the infinite indices by rank."""
        yield from itertools.islice(
            self._infinite_indices(sample_size), self.rank, None,
            self.world_size)

    def __iter__(self) -> Iterator[int]:
        batch_buffer = []
        while True:
            for source, num in enumerate(self.num_per_source):
                batch_buffer_per_source = []
                for idx in self.source2inds[source]:
                    idx += self.cumulative_sizes[source]
                    batch_buffer_per_source.append(idx)
                    if len(batch_buffer_per_source) == num:
                        batch_buffer += batch_buffer_per_source
                        break
            yield from batch_buffer
            batch_buffer = []

    def __len__(self) -> int:
        return len(self.dataset)

    def set_epoch(self, epoch: int) -> None:
        """Not supported in `epoch-based runner."""
        pass


@DATA_SAMPLERS.register_module()
class GroupMultiSourceSampler(MultiSourceSampler):
    r"""Group Multi-Source Infinite Sampler.

    According to the sampling ratio, sample data from different
    datasets but the same group to form batches.

    Args:
        dataset (Sized): The dataset.
        batch_size (int): Size of mini-batch.
        source_ratio (list[int | float]): The sampling ratio of different
            source datasets in a mini-batch.
        shuffle (bool): Whether shuffle the dataset or not. Defaults to True.
        seed (int, optional): Random seed. If None, set a random seed.
            Defaults to None.
    """

    def __init__(self,
                 dataset: BaseDataset,
                 batch_size: int,
                 source_ratio: List[Union[int, float]],
                 shuffle: bool = True,
                 seed: Optional[int] = None) -> None:
        super().__init__(
            dataset=dataset,
            batch_size=batch_size,
            source_ratio=source_ratio,
            shuffle=shuffle,
            seed=seed)

        self._get_source_group_info()  # 调用内部方法以根据数据源收集分组统计信息
        num_sources = len(dataset.datasets)  # 记录数据源的数量以便后续根据源索引迭代
        self.group_source2inds = [{  # 为每一个分组构建一个字典存放对应数据源的索引迭代器
            source:  # 当前数据源的索引键
            self._indices_of_rank(  # 基于分布式环境切分无限索引流
                self.group2size_per_source[source].get(group, 0))  # 使用该分组在当前源中的样本数作为迭代器长度
            for source in range(num_sources)  # 遍历所有数据源以生成对应的迭代器
        } for group in range(len(self.group_sizes))]  # 对所有出现过的分组进行上述映射构建

    def _get_source_group_info(self) -> None:
        num_sources = len(self.dataset.datasets)  # 计算数据源数量以便初始化按源划分的结构
        self.group2size_per_source = [  # 按数据源维护的分组样本计数字典列表
            defaultdict(int) for _ in range(num_sources)  # 为每个数据源创建默认值为0的计数字典
        ]
        self.group2inds_per_source = [  # 按数据源维护的分组索引列表字典
            defaultdict(list) for _ in range(num_sources)  # 为每个数据源创建默认值为空列表的索引容器
        ]
        for source, dataset in enumerate(self.dataset.datasets):  # 遍历每个数据源及其数据集
            for idx in range(len(dataset)):  # 遍历当前数据源内的每一条样本索引
                data_info = dataset.get_data_info(idx)  # 读取样本信息以获取图像尺寸
                width, height = data_info['width'], data_info['height']  # 解包宽高信息
                group = 0 if width < height else 1  # 按照纵横比规则确定分组编号
                self.group2size_per_source[source][group] += 1  # 累加当前分组在该数据源内的样本数量
                self.group2inds_per_source[source][group].append(idx)  # 记录当前样本在该分组的原始索引

        if num_sources == 0:  # 当没有任何数据源时直接初始化空的分组统计并提前返回
            self.group_sizes = np.zeros(0, dtype=np.int64)  # 使用空数组表示没有分组样本数
            self.group_ratio = np.zeros(0, dtype=np.float64)  # 使用空数组表示没有分组概率
            return  # 无需继续统计

        max_group = 0  # 记录出现过的最大分组编号用于确定数组长度
        for group2size in self.group2size_per_source:  # 遍历每个数据源的分组计数
            if group2size:  # 当该数据源存在分组数据时
                max_group = max(max_group, max(group2size.keys()))  # 更新最大分组编号

        self.group_sizes = np.zeros(max_group + 1, dtype=np.int64)  # 根据最大分组编号初始化总计数数组
        for group2size in self.group2size_per_source:  # 再次遍历每个数据源的分组计数
            for group, size in group2size.items():  # 累加每个分组在该数据源中的样本数
                self.group_sizes[group] += size  # 写入分组总样本数
        total = int(self.group_sizes.sum())  # 计算所有分组样本数之和
        if total == 0:  # 若总体样本数为零表示无可采样数据
            raise ValueError('All groups are empty, unable to sample data.')  # 抛出异常提示
        self.group_ratio = self.group_sizes / total  # 将每个分组样本数归一化得到采样概率

    def __iter__(self) -> Iterator[int]:
        if len(self.group_ratio) == 0:  # 当分组概率为空时无法进行采样
            raise RuntimeError('No group information available for sampling.')  # 抛出异常提示调用方
        batch_buffer = []  # 初始化批次缓存列表
        while True:  # 构造无限迭代器以持续提供索引
            group = np.random.choice(  # 按照分组概率随机选择当前批次所属的分组
                list(range(len(self.group_ratio))), p=self.group_ratio)  # 提供所有分组编号以及对应概率
            for source, num in enumerate(self.num_per_source):  # 遍历每个数据源及其在批次中的采样数
                batch_buffer_per_source = []  # 初始化当前数据源的采样缓存
                group_indices = self.group2inds_per_source[source].get(  # 获取当前分组在该数据源中的全部样本索引列表
                    group, [])  # 若不存在该分组则返回空列表
                if not group_indices and num == 0:  # 当该数据源不需要采样且没有可用索引时直接跳过
                    continue  # 进入下一个数据源
                for idx in self.group_source2inds[group][source]:  # 通过预生成的无限索引流获取候选样本
                    if idx >= len(group_indices):  # 若索引超出该分组可用样本范围则停止
                        break  # 中断当前数据源的采样循环
                    idx = group_indices[idx] + self.cumulative_sizes[source]  # 将局部索引转换为全局ConcatDataset索引
                    batch_buffer_per_source.append(idx)  # 追加到当前数据源的批次缓存
                    if len(batch_buffer_per_source) == num:  # 当达到该数据源需要的样本数量时
                        batch_buffer += batch_buffer_per_source  # 合并到整体批次缓存
                        break  # 结束当前数据源的采样循环
            yield from batch_buffer  # 将完整批次的索引依次产出
            batch_buffer = []  # 清空批次缓存以便生成下一批
