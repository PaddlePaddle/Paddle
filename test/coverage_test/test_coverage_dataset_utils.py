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

# [AUTO-GENERATED] Unit test for paddle.io.dataloader.dataset
# 自动生成的单测，覆盖 paddle.io.dataloader.dataset 模块中未覆盖的代码
# Target: cover uncovered lines 323-336, 340-344, 389-410, 453-462, 588, 629-630, 677-710
#   in paddle/python/paddle/io/dataloader/dataset.py
# 目标：覆盖 dataset.py 中 TensorDataset 的创建和使用、to_list 辅助函数、
#   ComposeDataset 的初始化和错误检查、ChainDataset 的初始化和迭代、
#   random_split 的长度校验错误、_accumulate 空列表路径、
#   ConcatDataset 的创建和负索引

"""
This test covers the following modules and code paths:
这个测试覆盖以下模块和代码路径：

1. TensorDataset - creation, __getitem__, __len__ (lines 322-336)
   TensorDataset - 创建、索引和长度

2. to_list() helper function - all three branches: None, list/tuple, scalar (lines 340-344)
   to_list() 辅助函数 - 三个分支：None、列表/元组、标量

3. ComposeDataset - init validation and __getitem__ (lines 389-410)
   ComposeDataset - 初始化校验和索引取值

4. ChainDataset - init validation and __iter__ (lines 453-462)
   ChainDataset - 初始化校验和迭代

5. random_split - sum-of-lengths mismatch error (line 588)
   random_split - 长度总和不匹配错误

6. _accumulate - empty iterable early return (lines 629-630)
   _accumulate - 空可迭代对象的提前返回

7. ConcatDataset - cumsum, init, __len__, negative indexing (lines 677-710)
   ConcatDataset - 累加和、初始化、长度、负索引
"""

import unittest

import numpy as np

import paddle
from paddle.io import (
    ChainDataset,
    ComposeDataset,
    ConcatDataset,
    Dataset,
    IterableDataset,
    TensorDataset,
    random_split,
)
from paddle.io.dataloader.dataset import _accumulate


# Helper datasets
# 辅助数据集类
class SimpleMapDataset(Dataset):
    """A simple map-style dataset for testing.
    用于测试的简单映射式数据集。
    """

    def __init__(self, num_samples, return_type='tuple'):
        self.num_samples = num_samples
        self.return_type = return_type

    def __getitem__(self, idx):
        if self.return_type == 'tuple':
            return (
                np.array([idx], dtype='float32'),
                np.array([idx * 2], dtype='int64'),
            )
        elif self.return_type == 'scalar':
            return np.array([idx], dtype='float32')
        elif self.return_type == 'list':
            return [
                np.array([idx], dtype='float32'),
                np.array([idx * 2], dtype='int64'),
            ]

    def __len__(self):
        return self.num_samples


class SimpleIterableDataset(IterableDataset):
    """A simple iterable dataset for testing.
    用于测试的简单可迭代数据集。
    """

    def __init__(self, num_samples):
        self.num_samples = num_samples

    def __iter__(self):
        for i in range(self.num_samples):
            yield np.array([i], dtype='float32')


class TestTensorDataset(unittest.TestCase):
    """Test TensorDataset creation, __getitem__, and __len__.
    测试 TensorDataset 的创建、索引取值和长度。
    覆盖 paddle/python/paddle/io/dataloader/dataset.py 第 322-336 行。
    """

    def test_tensor_dataset_basic(self):
        """TensorDataset should store and index tensors correctly.
        TensorDataset 应正确存储和索引张量。
        """
        data = paddle.to_tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        labels = paddle.to_tensor([0, 1, 2])
        dataset = TensorDataset([data, labels])
        self.assertEqual(len(dataset), 3)

        item = dataset[0]
        self.assertEqual(len(item), 2)
        np.testing.assert_allclose(item[0].numpy(), [1.0, 2.0])
        np.testing.assert_allclose(item[1].numpy(), 0)

    def test_tensor_dataset_all_items(self):
        """TensorDataset should return correct items for all indices.
        TensorDataset 应对所有索引返回正确的元素。
        """
        data = paddle.arange(0, 15, dtype='float32').reshape([5, 3])
        labels = paddle.arange(0, 5, dtype='int64')
        dataset = TensorDataset([data, labels])

        for i in range(len(dataset)):
            item_data, item_label = dataset[i]
            np.testing.assert_allclose(item_data.numpy(), data[i].numpy())
            np.testing.assert_allclose(item_label.numpy(), labels[i].numpy())

    def test_tensor_dataset_single_tensor(self):
        """TensorDataset should work with a single tensor.
        TensorDataset 应支持仅一个张量的情况。
        """
        data = paddle.to_tensor([[1.0], [2.0], [3.0]])
        dataset = TensorDataset([data])
        self.assertEqual(len(dataset), 3)
        item = dataset[1]
        self.assertEqual(len(item), 1)
        np.testing.assert_allclose(item[0].numpy(), [2.0])

    def test_tensor_dataset_shape_mismatch(self):
        """TensorDataset should raise AssertionError for shape mismatch.
        当张量第一维大小不一致时应抛出 AssertionError。
        """
        data = paddle.to_tensor([[1.0], [2.0], [3.0]])
        labels = paddle.to_tensor([0, 1])
        with self.assertRaises(AssertionError):
            TensorDataset([data, labels])


class TestToListFunction(unittest.TestCase):
    """Test the to_list() helper function used by ComposeDataset.
    测试 ComposeDataset 使用的 to_list() 辅助函数。
    覆盖 paddle/python/paddle/io/dataloader/dataset.py 第 340-344 行。
    """

    def test_compose_with_tuple_return(self):
        """ComposeDataset with datasets returning tuples calls to_list with tuple.
        当数据集返回元组时，to_list 接收到元组并转换为列表。
        """
        ds1 = SimpleMapDataset(5, return_type='tuple')
        ds2 = SimpleMapDataset(5, return_type='tuple')
        dataset = ComposeDataset([ds1, ds2])
        item = dataset[0]
        # ds1 returns (data, label), ds2 returns (data, label) => 4 items
        self.assertEqual(len(item), 4)

    def test_compose_with_scalar_return(self):
        """ComposeDataset with datasets returning scalars calls to_list with scalar.
        当数据集返回标量时，to_list 接收到标量并包装为列表（覆盖第 344 行）。
        """
        ds1 = SimpleMapDataset(5, return_type='scalar')
        ds2 = SimpleMapDataset(5, return_type='scalar')
        dataset = ComposeDataset([ds1, ds2])
        item = dataset[0]
        # Each dataset returns a single value, wrapped in list => 2 items
        self.assertEqual(len(item), 2)

    def test_compose_with_list_return(self):
        """ComposeDataset with datasets returning lists calls to_list with list.
        当数据集返回列表时，to_list 接收到列表并转换（覆盖第 342-343 行）。
        """
        ds1 = SimpleMapDataset(5, return_type='list')
        ds2 = SimpleMapDataset(5, return_type='list')
        dataset = ComposeDataset([ds1, ds2])
        item = dataset[0]
        self.assertEqual(len(item), 4)


class TestComposeDatasetValidation(unittest.TestCase):
    """Test ComposeDataset initialization validation.
    测试 ComposeDataset 初始化校验。
    覆盖 paddle/python/paddle/io/dataloader/dataset.py 第 389-401 行。
    """

    def test_empty_datasets(self):
        """ComposeDataset should raise AssertionError for empty datasets.
        空数据集列表应抛出 AssertionError。
        """
        with self.assertRaises(AssertionError):
            ComposeDataset([])

    def test_non_dataset(self):
        """ComposeDataset should raise AssertionError for non-Dataset items.
        非 Dataset 对象应抛出 AssertionError。
        """
        with self.assertRaises(AssertionError):
            ComposeDataset(["not_a_dataset"])

    def test_iterable_dataset_rejected(self):
        """ComposeDataset should reject IterableDataset.
        应拒绝 IterableDataset。
        """
        with self.assertRaises(AssertionError):
            ComposeDataset([SimpleIterableDataset(10)])

    def test_length_mismatch(self):
        """ComposeDataset should raise AssertionError for length-mismatched datasets.
        长度不匹配的数据集应抛出 AssertionError。
        """
        with self.assertRaises(AssertionError):
            ComposeDataset([SimpleMapDataset(10), SimpleMapDataset(5)])

    def test_compose_len(self):
        """ComposeDataset.__len__ should delegate to first dataset.
        ComposeDataset 的长度应等于第一个子数据集的长度。
        """
        dataset = ComposeDataset([SimpleMapDataset(7), SimpleMapDataset(7)])
        self.assertEqual(len(dataset), 7)


class TestChainDatasetValidation(unittest.TestCase):
    """Test ChainDataset initialization and iteration.
    测试 ChainDataset 的初始化校验和迭代。
    覆盖 paddle/python/paddle/io/dataloader/dataset.py 第 453-462 行。
    """

    def test_empty_datasets(self):
        """ChainDataset should raise AssertionError for empty datasets.
        空数据集列表应抛出 AssertionError。
        """
        with self.assertRaises(AssertionError):
            ChainDataset([])

    def test_non_iterable_dataset(self):
        """ChainDataset should raise AssertionError for non-IterableDataset.
        非 IterableDataset 应抛出 AssertionError。
        """
        with self.assertRaises(AssertionError):
            ChainDataset([SimpleMapDataset(10)])

    def test_chain_iteration(self):
        """ChainDataset should iterate through all datasets sequentially.
        ChainDataset 应按顺序迭代所有数据集。
        """
        ds1 = SimpleIterableDataset(3)
        ds2 = SimpleIterableDataset(4)
        chain = ChainDataset([ds1, ds2])

        items = []
        for item in chain:
            items.append(item)
        self.assertEqual(len(items), 7)
        # First 3 items from ds1 (0,1,2), next 4 from ds2 (0,1,2,3)
        np.testing.assert_allclose(items[0], [0])
        np.testing.assert_allclose(items[2], [2])
        np.testing.assert_allclose(items[3], [0])
        np.testing.assert_allclose(items[6], [3])


class TestRandomSplitLengthError(unittest.TestCase):
    """Test random_split raises ValueError for mismatched lengths.
    测试 random_split 在长度不匹配时抛出 ValueError。
    覆盖 paddle/python/paddle/io/dataloader/dataset.py 第 587-590 行。
    """

    def test_lengths_sum_mismatch(self):
        """random_split should raise ValueError when sum(lengths) != len(dataset).
        当 lengths 之和不等于数据集长度时应抛出 ValueError。
        """
        dataset = SimpleMapDataset(10)
        with self.assertRaises(ValueError):
            random_split(dataset, [3, 3])

    def test_lengths_sum_too_large(self):
        """random_split should raise ValueError when sum exceeds dataset length.
        当 lengths 之和超过数据集长度时应抛出 ValueError。
        """
        dataset = SimpleMapDataset(5)
        with self.assertRaises(ValueError):
            random_split(dataset, [3, 5])

    def test_valid_random_split(self):
        """random_split should work when sum(lengths) == len(dataset).
        当 lengths 之和等于数据集长度时应正常工作。
        """
        dataset = SimpleMapDataset(10)
        splits = random_split(dataset, [3, 7])
        self.assertEqual(len(splits), 2)
        self.assertEqual(len(splits[0]), 3)
        self.assertEqual(len(splits[1]), 7)


class TestAccumulateEmpty(unittest.TestCase):
    """Test _accumulate with empty iterable.
    测试 _accumulate 处理空可迭代对象。
    覆盖 paddle/python/paddle/io/dataloader/dataset.py 第 629-630 行。
    """

    def test_accumulate_empty_list(self):
        """_accumulate with empty list should return empty.
        空列表应返回空结果。
        """
        result = list(_accumulate([]))
        self.assertEqual(result, [])

    def test_accumulate_normal(self):
        """_accumulate with normal list should return running totals.
        正常列表应返回累积和。
        """
        result = list(_accumulate([1, 2, 3, 4, 5]))
        self.assertEqual(result, [1, 3, 6, 10, 15])


class TestConcatDataset(unittest.TestCase):
    """Test ConcatDataset creation, negative indexing, and validation.
    测试 ConcatDataset 的创建、负索引和校验。
    覆盖 paddle/python/paddle/io/dataloader/dataset.py 第 677-710 行。
    """

    def test_concat_basic(self):
        """ConcatDataset should concatenate multiple datasets.
        ConcatDataset 应正确连接多个数据集。
        """
        ds1 = SimpleMapDataset(5)
        ds2 = SimpleMapDataset(3)
        concat = ConcatDataset([ds1, ds2])
        self.assertEqual(len(concat), 8)

    def test_concat_negative_index(self):
        """ConcatDataset should support negative indexing.
        ConcatDataset 应支持负索引。
        覆盖第 699-704 行。
        """
        ds1 = SimpleMapDataset(5, return_type='scalar')
        ds2 = SimpleMapDataset(3, return_type='scalar')
        concat = ConcatDataset([ds1, ds2])

        # Last item should be from ds2, index 2
        last_item = concat[-1]
        np.testing.assert_allclose(last_item, [2])

        # Second to last item
        second_last = concat[-2]
        np.testing.assert_allclose(second_last, [1])

    def test_concat_negative_index_out_of_bounds(self):
        """ConcatDataset should raise ValueError for out-of-bounds negative index.
        超出范围的负索引应抛出 ValueError。
        覆盖第 700-703 行。
        """
        ds1 = SimpleMapDataset(3)
        ds2 = SimpleMapDataset(2)
        concat = ConcatDataset([ds1, ds2])

        with self.assertRaises(ValueError):
            concat[-10]

    def test_concat_first_dataset_index(self):
        """ConcatDataset should correctly index into first dataset.
        ConcatDataset 应正确索引到第一个子数据集。
        覆盖第 706-707 行 (dataset_idx == 0 分支)。
        """
        ds1 = SimpleMapDataset(5, return_type='scalar')
        ds2 = SimpleMapDataset(3, return_type='scalar')
        concat = ConcatDataset([ds1, ds2])

        item = concat[2]
        np.testing.assert_allclose(item, [2])

    def test_concat_second_dataset_index(self):
        """ConcatDataset should correctly index into second dataset.
        ConcatDataset 应正确索引到第二个子数据集。
        覆盖第 708-709 行 (dataset_idx > 0 分支)。
        """
        ds1 = SimpleMapDataset(5, return_type='scalar')
        ds2 = SimpleMapDataset(3, return_type='scalar')
        concat = ConcatDataset([ds1, ds2])

        # Index 5 should be the first item in ds2
        item = concat[5]
        np.testing.assert_allclose(item, [0])

    def test_concat_empty_raises(self):
        """ConcatDataset should raise AssertionError for empty datasets.
        空数据集列表应抛出 AssertionError。
        """
        with self.assertRaises(AssertionError):
            ConcatDataset([])

    def test_concat_iterable_rejected(self):
        """ConcatDataset should reject IterableDataset.
        应拒绝 IterableDataset。
        """
        with self.assertRaises(AssertionError):
            ConcatDataset([SimpleIterableDataset(10)])

    def test_concat_cumsum(self):
        """ConcatDataset.cumsum should compute correct cumulative sizes.
        ConcatDataset.cumsum 应计算正确的累积大小。
        覆盖第 677-682 行。
        """
        ds1 = SimpleMapDataset(5)
        ds2 = SimpleMapDataset(3)
        ds3 = SimpleMapDataset(7)
        result = ConcatDataset.cumsum([ds1, ds2, ds3])
        self.assertEqual(result, [5, 8, 15])


if __name__ == '__main__':
    unittest.main()
