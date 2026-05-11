# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

# [AUTO-GENERATED] Unit test for paddle.io (Dataset, DataLoader, Sampler)
# 自动生成的单测，覆盖 paddle.io 模块中未覆盖的代码路径
# Target: cover uncovered lines in paddle/python/paddle/io/
# 目标：覆盖 DataLoader、Dataset、Sampler 的初始化和基本功能

"""
This test covers the following modules and code paths:
这个测试覆盖以下模块和代码路径：

1. TensorDataset - 单张量数据集初始化和索引 (returns tuple of scalars per index)
2. ComposeDataset - 组合数据集
3. SequenceSampler - 顺序采样器
4. RandomSampler - 随机采样器 (有/无放回)
5. BatchSampler - 批量采样器 (drop_last)
"""

import unittest

import paddle
from paddle.io import (
    BatchSampler,
    ComposeDataset,
    DataLoader,
    Dataset,
    RandomSampler,
    SequenceSampler,
    Subset,
    TensorDataset,
)


class SimpleDataset(Dataset):
    """Simple dataset for testing."""

    def __init__(self, data):
        self.data = data

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)


class TestTensorDataset(unittest.TestCase):
    """Test TensorDataset.
    TensorDataset in Paddle takes a list of tensors.
    __getitem__ returns a tuple of indexed rows (one per input tensor).
    """

    def setUp(self):
        paddle.disable_static()

    def test_tensor_dataset_basic(self):
        """Basic TensorDataset usage - returns tuple of indexed rows."""
        data1 = paddle.randn([10, 5])
        data2 = paddle.randn([10, 3])
        dataset = TensorDataset([data1, data2])
        self.assertEqual(len(dataset), 10)
        item = dataset[0]
        self.assertIsInstance(item, tuple)
        self.assertEqual(len(item), 2)
        # Each element is the 0th row of the corresponding input tensor
        self.assertEqual(item[0].shape, [5])
        self.assertEqual(item[1].shape, [3])

    def test_tensor_dataset_1d(self):
        """TensorDataset with 1D tensors."""
        data1 = paddle.randn([10])
        data2 = paddle.randn([10])
        dataset = TensorDataset([data1, data2])
        self.assertEqual(len(dataset), 10)
        item = dataset[0]
        self.assertIsInstance(item, tuple)
        self.assertEqual(len(item), 2)

    def test_tensor_dataset_iter(self):
        """Iterate over TensorDataset."""
        data = paddle.randn([5, 3])
        data2 = paddle.randn([5, 2])
        dataset = TensorDataset([data, data2])
        items = [dataset[i] for i in range(5)]
        self.assertEqual(len(items), 5)

    def test_tensor_dataset_varargs(self):
        """TensorDataset with variable arguments (*args)."""
        data1 = paddle.randn([10, 5])
        data2 = paddle.randn([10, 3])
        data3 = paddle.randn([10, 2])
        dataset = TensorDataset(data1, data2, data3)
        self.assertEqual(len(dataset), 10)
        item = dataset[0]
        self.assertIsInstance(item, tuple)
        self.assertEqual(len(item), 3)
        self.assertEqual(item[0].shape, [5])
        self.assertEqual(item[1].shape, [3])
        self.assertEqual(item[2].shape, [2])

    def test_tensor_dataset_varargs_single(self):
        """TensorDataset with single tensor as vararg."""
        data = paddle.randn([8, 4])
        dataset = TensorDataset(data)
        self.assertEqual(len(dataset), 8)
        item = dataset[0]
        self.assertIsInstance(item, tuple)
        self.assertEqual(len(item), 1)
        self.assertEqual(item[0].shape, [4])


class TestComposeDataset(unittest.TestCase):
    """Test ComposeDataset."""

    def setUp(self):
        paddle.disable_static()

    def test_compose_dataset_basic(self):
        """ComposeDataset combines multiple datasets."""
        ds1 = SimpleDataset(paddle.randn([10, 5]))
        ds2 = SimpleDataset(paddle.randn([10, 3]))
        composed = ComposeDataset([ds1, ds2])
        self.assertEqual(len(composed), 10)
        item = composed[0]
        self.assertEqual(len(item), 2)

    def test_compose_dataset_single(self):
        """ComposeDataset with single dataset."""
        ds = SimpleDataset(paddle.randn([5, 2]))
        composed = ComposeDataset([ds])
        self.assertEqual(len(composed), 5)


class TestSamplers(unittest.TestCase):
    """Test various samplers."""

    def setUp(self):
        paddle.disable_static()

    def test_sequence_sampler(self):
        """SequenceSampler yields indices in order."""
        sampler = SequenceSampler(data_source=list(range(10)))
        indices = list(sampler)
        self.assertEqual(indices, list(range(10)))

    def test_random_sampler_no_replacement(self):
        """RandomSampler without replacement."""
        sampler = RandomSampler(data_source=list(range(10)))
        indices = list(sampler)
        self.assertEqual(len(indices), 10)
        self.assertEqual(sorted(indices), list(range(10)))

    def test_random_sampler_with_replacement(self):
        """RandomSampler with replacement."""
        sampler = RandomSampler(
            data_source=list(range(5)), replacement=True, num_samples=20
        )
        indices = list(sampler)
        self.assertEqual(len(indices), 20)

    def test_batch_sampler_no_drop_last(self):
        """BatchSampler without drop_last."""
        sampler = SequenceSampler(data_source=list(range(10)))
        bs = BatchSampler(sampler=sampler, batch_size=3, drop_last=False)
        batches = list(bs)
        self.assertEqual(len(batches), 4)
        self.assertEqual(batches[0], [0, 1, 2])

    def test_batch_sampler_drop_last(self):
        """BatchSampler with drop_last."""
        sampler = SequenceSampler(data_source=list(range(10)))
        bs = BatchSampler(sampler=sampler, batch_size=3, drop_last=True)
        batches = list(bs)
        self.assertEqual(len(batches), 3)


class TestSubset(unittest.TestCase):
    """Test Subset."""

    def setUp(self):
        paddle.disable_static()

    def test_subset_basic(self):
        """Subset selects a subset of a dataset."""
        ds = SimpleDataset(paddle.randn([20, 5]))
        subset = Subset(ds, indices=[0, 2, 5, 10])
        self.assertEqual(len(subset), 4)
        item = subset[0]
        self.assertEqual(item.shape, [5])


class TestDataLoader(unittest.TestCase):
    """Test DataLoader with SimpleDataset."""

    def setUp(self):
        paddle.disable_static()

    def test_dataloader_basic(self):
        """DataLoader with SimpleDataset."""
        ds = SimpleDataset(paddle.randn([20, 5]))
        loader = DataLoader(ds, batch_size=4)
        batches = list(loader)
        self.assertEqual(len(batches), 5)
        # Each batch is a single stacked tensor
        self.assertIsInstance(batches[0], paddle.Tensor)
        self.assertEqual(batches[0].shape, [4, 5])

    def test_dataloader_batch_size_one(self):
        """DataLoader with batch_size=1."""
        ds = SimpleDataset(paddle.randn([5, 3]))
        loader = DataLoader(ds, batch_size=1)
        batches = list(loader)
        self.assertEqual(len(batches), 5)

    def test_dataloader_drop_last(self):
        """DataLoader with drop_last."""
        ds = SimpleDataset(paddle.randn([10, 3]))
        loader = DataLoader(ds, batch_size=4, drop_last=True)
        batches = list(loader)
        self.assertEqual(len(batches), 2)

    def test_dataloader_num_workers_zero(self):
        """DataLoader with num_workers=0."""
        ds = SimpleDataset(paddle.randn([8, 3]))
        loader = DataLoader(ds, batch_size=4, num_workers=0)
        batches = list(loader)
        self.assertEqual(len(batches), 2)

    def test_dataloader_return_list(self):
        """DataLoader with return_list=False returns numpy arrays."""
        ds = SimpleDataset(paddle.randn([8, 3]))
        loader = DataLoader(ds, batch_size=4, return_list=False)
        batches = list(loader)
        # With return_list=False, each batch is a list of numpy arrays
        self.assertEqual(len(batches), 2)


if __name__ == '__main__':
    unittest.main()
