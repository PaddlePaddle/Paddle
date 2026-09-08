# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
数据集和采样器测试 / Dataset and Sampler Tests

测试目标 / Test Target:
  paddle.io 数据集和数据加载器

覆盖的模块 / Covered Modules:
  - paddle.io.Dataset: 数据集基类
  - paddle.io.IterableDataset: 可迭代数据集
  - paddle.io.DataLoader: 数据加载器
  - paddle.io.Subset: 子集
  - paddle.io.random_split: 随机分割
  - paddle.io.BatchSampler: 批次采样器

作用 / Purpose:
  补充数据加载相关API的测试，提升覆盖率。
"""

import unittest

import numpy as np

import paddle
from paddle.io import (
    BatchSampler,
    DataLoader,
    Dataset,
    IterableDataset,
    SequenceSampler,
    Subset,
)

paddle.disable_static()


class SimpleDataset(Dataset):
    """简单数据集 / Simple dataset"""

    def __init__(self, size=100):
        super().__init__()
        self.data = np.random.randn(size, 4).astype('float32')
        self.labels = np.random.randint(0, 2, size).astype('int64')

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

    def __len__(self):
        return len(self.data)


class SimpleIterableDataset(IterableDataset):
    """简单可迭代数据集 / Simple iterable dataset"""

    def __init__(self, size=50):
        super().__init__()
        self.size = size
        self.data = np.random.randn(size, 4).astype('float32')

    def __iter__(self):
        for i in range(self.size):
            yield self.data[i]


class TestDataset(unittest.TestCase):
    """测试数据集 / Test Dataset"""

    def test_dataset_basic(self):
        """测试基本数据集 / Test basic dataset"""
        dataset = SimpleDataset(50)
        self.assertEqual(len(dataset), 50)
        item = dataset[0]
        self.assertEqual(len(item), 2)
        self.assertEqual(item[0].shape, (4,))

    def test_dataset_subset(self):
        """测试数据集子集 / Test dataset subset"""
        dataset = SimpleDataset(100)
        subset = Subset(dataset, list(range(20)))
        self.assertEqual(len(subset), 20)
        item = subset[0]
        self.assertEqual(item[0].shape, (4,))

    def test_iterable_dataset(self):
        """测试可迭代数据集 / Test iterable dataset"""
        dataset = SimpleIterableDataset(30)
        count = 0
        for item in dataset:
            count += 1
        self.assertEqual(count, 30)


class TestDataLoader(unittest.TestCase):
    """测试数据加载器 / Test DataLoader"""

    def test_dataloader_basic(self):
        """测试基本数据加载器 / Test basic DataLoader"""
        dataset = SimpleDataset(100)
        loader = DataLoader(dataset, batch_size=16, shuffle=False)
        batch = next(iter(loader))
        self.assertEqual(batch[0].shape, [16, 4])
        self.assertEqual(batch[1].shape, [16])

    def test_dataloader_shuffle(self):
        """测试随机打乱数据加载器 / Test shuffled DataLoader"""
        dataset = SimpleDataset(100)
        loader = DataLoader(dataset, batch_size=32, shuffle=True)
        batches = list(loader)
        self.assertGreater(len(batches), 0)

    def test_dataloader_drop_last(self):
        """测试丢弃最后不完整批次 / Test DataLoader drop_last"""
        dataset = SimpleDataset(
            90
        )  # 90 samples, batch_size=32: 2 full + 1 incomplete
        loader = DataLoader(dataset, batch_size=32, drop_last=True)
        batches = list(loader)
        self.assertEqual(len(batches), 2)
        for batch in batches:
            self.assertEqual(batch[0].shape[0], 32)

    def test_dataloader_num_workers(self):
        """测试多worker数据加载器 / Test DataLoader with num_workers"""
        dataset = SimpleDataset(50)
        loader = DataLoader(dataset, batch_size=16, num_workers=2)
        batches = list(loader)
        self.assertGreater(len(batches), 0)


class TestBatchSampler(unittest.TestCase):
    """测试批次采样器 / Test Batch Sampler"""

    def test_batch_sampler_basic(self):
        """测试基本批次采样器 / Test basic batch sampler"""
        sampler = SequenceSampler(SimpleDataset(100))
        batch_sampler = BatchSampler(sampler=sampler, batch_size=16)
        batches = list(batch_sampler)
        self.assertEqual(len(batches), 7)  # ceil(100/16)

    def test_batch_sampler_drop_last(self):
        """测试丢弃最后批次的采样器 / Test batch sampler with drop_last"""
        sampler = SequenceSampler(SimpleDataset(100))
        batch_sampler = BatchSampler(
            sampler=sampler, batch_size=16, drop_last=True
        )
        batches = list(batch_sampler)
        self.assertEqual(len(batches), 6)  # floor(100/16)
        for batch in batches:
            self.assertEqual(len(batch), 16)


class TestRandomSplit(unittest.TestCase):
    """测试随机分割 / Test random split"""

    def test_random_split_ratios(self):
        """测试按比例分割 / Test split by ratios"""
        from paddle.io import random_split

        dataset = SimpleDataset(100)
        train_set, val_set = random_split(dataset, [80, 20])
        self.assertEqual(len(train_set), 80)
        self.assertEqual(len(val_set), 20)

    def test_random_split_access(self):
        """测试分割后数据访问 / Test split data access"""
        from paddle.io import random_split

        dataset = SimpleDataset(50)
        split1, split2 = random_split(dataset, [30, 20])
        item = split1[0]
        self.assertEqual(item[0].shape, (4,))


if __name__ == '__main__':
    unittest.main()
