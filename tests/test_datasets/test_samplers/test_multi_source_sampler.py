import importlib.util
import itertools
import random
import sys
import types
from bisect import bisect_right
from pathlib import Path

import pytest


try:  # pragma: no cover - import dependency if available
    import numpy as np  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - provide lightweight stub
    class _FakeArray(list):

        def sum(self):
            return sum(self)

        def __truediv__(self, value):
            return _FakeArray([elem / value for elem in self])

    class _FakeRandomModule:

        def __init__(self):
            self._rng = random.Random()

        def choice(self, choices, p=None):
            sequence = list(choices)
            if not sequence:
                raise ValueError('Choices cannot be empty.')
            if p is None:
                return sequence[self._rng.randrange(len(sequence))]
            total = sum(p)
            if total <= 0:
                raise ValueError('Sum of probabilities must be positive.')
            r = self._rng.random()
            cumulative = 0.0
            for choice, prob in zip(sequence, p):
                cumulative += prob / total
                if r <= cumulative:
                    return choice
            return sequence[-1]

    def _fake_zeros(size, dtype=None):
        if isinstance(size, int):
            return _FakeArray([0] * size)
        raise TypeError('size must be an integer')

    np = types.SimpleNamespace(
        int64=int,
        float64=float,
        zeros=_fake_zeros,
        random=_FakeRandomModule())
    sys.modules.setdefault('numpy', np)

try:  # pragma: no cover - import dependency if available
    import torch  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - provide lightweight stub
    class _FakeTensor(list):

        def tolist(self):
            return list(self)

    class _FakeGenerator:

        def __init__(self):
            self._rng = random.Random()

        def manual_seed(self, seed):
            self._rng.seed(seed)

    def _fake_randperm(n, generator=None):
        seq = list(range(n))
        rng = generator._rng if generator is not None else random.Random()
        rng.shuffle(seq)
        return _FakeTensor(seq)

    def _fake_arange(n):
        return _FakeTensor(list(range(n)))

    class _FakeSampler:

        def __init__(self, data_source=None):
            self.data_source = data_source

    torch = types.SimpleNamespace(  # type: ignore
        Generator=_FakeGenerator,
        randperm=_fake_randperm,
        arange=_fake_arange)

    torch_utils_data = types.SimpleNamespace(Sampler=_FakeSampler)
    torch_utils = types.SimpleNamespace(data=torch_utils_data)
    sys.modules.setdefault('torch', torch)
    sys.modules.setdefault('torch.utils', torch_utils)
    sys.modules.setdefault('torch.utils.data', torch_utils_data)

mmengine_module = types.ModuleType('mmengine')
mmengine_dataset_module = types.ModuleType('mmengine.dataset')


class _BaseDataset:  # pragma: no cover - minimal stub for type checking
    pass


mmengine_dataset_module.BaseDataset = _BaseDataset
mmengine_dist_module = types.ModuleType('mmengine.dist')


def _fake_get_dist_info():  # pragma: no cover - deterministic rank info
    return 0, 1


def _fake_sync_random_seed():  # pragma: no cover - deterministic seed
    return 0


mmengine_dist_module.get_dist_info = _fake_get_dist_info
mmengine_dist_module.sync_random_seed = _fake_sync_random_seed
mmengine_module.dataset = mmengine_dataset_module
mmengine_module.dist = mmengine_dist_module
sys.modules.setdefault('mmengine', mmengine_module)
sys.modules.setdefault('mmengine.dataset', mmengine_dataset_module)
sys.modules.setdefault('mmengine.dist', mmengine_dist_module)

mmdet_module = types.ModuleType('mmdet')
mmdet_registry_module = types.ModuleType('mmdet.registry')


class _FakeRegistry:  # pragma: no cover - decorator stub

    def register_module(self, *args, **kwargs):
        def _decorator(obj):
            return obj

        return _decorator


mmdet_registry_module.DATA_SAMPLERS = _FakeRegistry()
mmdet_module.registry = mmdet_registry_module
sys.modules.setdefault('mmdet', mmdet_module)
sys.modules.setdefault('mmdet.registry', mmdet_registry_module)

MODULE_PATH = Path(__file__).resolve().parents[3] / \
    'mmdet/datasets/samplers/multi_source_sampler.py'
spec = importlib.util.spec_from_file_location('multi_source_sampler', MODULE_PATH)
multi_source_sampler = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(multi_source_sampler)

GroupMultiSourceSampler = multi_source_sampler.GroupMultiSourceSampler
MultiSourceSampler = multi_source_sampler.MultiSourceSampler


class DummyDataset:

    def __init__(self, shapes):
        self.shapes = list(shapes)

    def __len__(self):
        return len(self.shapes)

    def __getitem__(self, idx):
        return self.shapes[idx]

    def get_data_info(self, idx):
        width, height = self.shapes[idx]
        return dict(width=width, height=height)


class LegacyGroupMultiSourceSampler(MultiSourceSampler):

    def __init__(self,
                 dataset,
                 batch_size,
                 source_ratio,
                 shuffle=True,
                 seed=None):
        super().__init__(
            dataset=dataset,
            batch_size=batch_size,
            source_ratio=source_ratio,
            shuffle=shuffle,
            seed=seed)
        self._get_source_group_info()
        self.group_source2inds = [{
            source:
            self._indices_of_rank(self.group2size_per_source[source][group])
            for source in range(len(dataset.datasets))
        } for group in range(len(self.group_ratio))]

    def _get_source_group_info(self):
        self.group2size_per_source = [{0: 0, 1: 0}, {0: 0, 1: 0}]
        self.group2inds_per_source = [{0: [], 1: []}, {0: [], 1: []}]
        for source, dataset in enumerate(self.dataset.datasets):
            for idx in range(len(dataset)):
                data_info = dataset.get_data_info(idx)
                width, height = data_info['width'], data_info['height']
                group = 0 if width < height else 1
                self.group2size_per_source[source][group] += 1
                self.group2inds_per_source[source][group].append(idx)

        self.group_sizes = np.zeros(2, dtype=np.int64)
        for group2size in self.group2size_per_source:
            for group, size in group2size.items():
                self.group_sizes[group] += size
        self.group_ratio = self.group_sizes / sum(self.group_sizes)

    def __iter__(self):
        batch_buffer = []
        while True:
            group = np.random.choice(
                list(range(len(self.group_ratio))), p=self.group_ratio)
            for source, num in enumerate(self.num_per_source):
                batch_buffer_per_source = []
                for idx in self.group_source2inds[group][source]:
                    idx = self.group2inds_per_source[source][group][
                        idx] + self.cumulative_sizes[source]
                    batch_buffer_per_source.append(idx)
                    if len(batch_buffer_per_source) == num:
                        batch_buffer += batch_buffer_per_source
                        break
            yield from batch_buffer
            batch_buffer = []


@pytest.fixture()
def fixed_choice(monkeypatch):
    def _patch(values):
        iterator = iter(values)

        def _fake_choice(*args, **kwargs):
            try:
                return next(iterator)
            except StopIteration:
                return values[-1]

        monkeypatch.setattr(np.random, 'choice', _fake_choice)

    return _patch


class DummyConcatDataset:

    def __init__(self, datasets):
        self.datasets = list(datasets)
        self.cumulative_sizes = []
        total = 0
        for dataset in self.datasets:
            total += len(dataset)
            self.cumulative_sizes.append(total)

    def __len__(self):
        return self.cumulative_sizes[-1] if self.cumulative_sizes else 0


def test_group_multi_source_sampler_three_sources(fixed_choice):
    ds0 = DummyDataset([(10, 20), (12, 24), (30, 10)])
    ds1 = DummyDataset([(11, 22), (13, 26), (15, 30), (28, 14)])
    ds2 = DummyDataset([(9, 18), (10, 30), (12, 48), (60, 20), (14, 28)])
    dataset = DummyConcatDataset([ds0, ds1, ds2])

    sampler = GroupMultiSourceSampler(
        dataset=dataset,
        batch_size=6,
        source_ratio=[1, 2, 3],
        shuffle=False,
        seed=0)

    fixed_choice([0, 0, 0])

    iterator = iter(sampler)
    batch = [next(iterator) for _ in range(sampler.batch_size)]

    counts = [0] * len(dataset.datasets)
    for idx in batch:
        source = bisect_right(sampler.cumulative_sizes, idx) - 1
        counts[source] += 1

    assert counts == sampler.num_per_source


def test_two_source_sampling_matches_legacy_implementation(fixed_choice):
    ds0 = DummyDataset([(10, 20), (20, 10), (12, 24), (18, 36)])
    ds1 = DummyDataset([(9, 18), (14, 28), (30, 10), (40, 20)])
    dataset = DummyConcatDataset([ds0, ds1])

    sampler = GroupMultiSourceSampler(
        dataset=dataset,
        batch_size=4,
        source_ratio=[1, 1],
        shuffle=False,
        seed=0)

    legacy_sampler = LegacyGroupMultiSourceSampler(
        dataset=dataset,
        batch_size=4,
        source_ratio=[1, 1],
        shuffle=False,
        seed=0)

    sample_count = sampler.batch_size * 3
    fixed_choice([0, 1, 0, 1, 0, 1])
    new_indices = list(itertools.islice(iter(sampler), sample_count))
    fixed_choice([0, 1, 0, 1, 0, 1])
    old_indices = list(itertools.islice(iter(legacy_sampler), sample_count))

    assert new_indices == old_indices
