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

# [AUTO-GENERATED] Test file for paddle.io.dataloader.fetcher
# 覆盖模块: paddle/io/dataloader/fetcher.py
# 未覆盖行: 33,40,41,44,45,46,47,48,49,50,51,53,55,58,61,63,64,65,79
# Covered module: paddle/io/dataloader/fetcher.py
# Uncovered lines: 33,40,41,44,45,46,47,48,49,50,51,53,55,58,61,63,64,65,79

import unittest
from unittest.mock import MagicMock

import paddle
from paddle.io.dataloader.fetcher import (
    _DatasetFetcher,
    _IterableDatasetFetcher,
    _MapDatasetFetcher,
)


class TestDatasetFetcher(unittest.TestCase):
    """测试 _DatasetFetcher 基类
    Test _DatasetFetcher base class"""

    def test_fetch_not_implemented(self):
        """测试 _DatasetFetcher.fetch 未实现
        Test _DatasetFetcher.fetch not implemented"""
        dataset = MagicMock()
        fetcher = _DatasetFetcher(dataset, True, None, False)
        with self.assertRaises(NotImplementedError):
            fetcher.fetch([0, 1])

    def test_init_attributes(self):
        """测试 _DatasetFetcher 初始化属性
        Test _DatasetFetcher initialization attributes"""
        dataset = [1, 2, 3]
        fn = lambda x: x
        fetcher = _DatasetFetcher(dataset, True, fn, False)
        self.assertEqual(fetcher.dataset, [1, 2, 3])
        self.assertTrue(fetcher.auto_collate_batch)
        self.assertIs(fetcher.collate_fn, fn)
        self.assertFalse(fetcher.drop_last)


class TestIterableDatasetFetcher(unittest.TestCase):
    """测试 _IterableDatasetFetcher
    Test _IterableDatasetFetcher"""

    def _make_iterable(self, data):
        """创建一个可迭代数据集
        Create an iterable dataset"""

        class SimpleIterable(paddle.io.IterableDataset):
            def __init__(self, items):
                self.items = items

            def __iter__(self):
                return iter(self.items)

        return SimpleIterable(data)

    def test_fetch_auto_collate(self):
        """测试自动批处理的 fetch
        Test fetch with auto_collate_batch=True"""
        dataset = self._make_iterable([1, 2, 3, 4, 5])
        fetcher = _IterableDatasetFetcher(dataset, True, lambda x: x, False)
        result = fetcher.fetch([0, 1, 2])
        self.assertEqual(result, [1, 2, 3])

    def test_fetch_no_auto_collate(self):
        """测试非自动批处理的 fetch
        Test fetch with auto_collate_batch=False"""
        dataset = self._make_iterable([10, 20, 30])
        fetcher = _IterableDatasetFetcher(dataset, False, lambda x: x, False)
        result = fetcher.fetch([0])
        self.assertEqual(result, 10)

    def test_fetch_with_collate_fn(self):
        """测试带 collate_fn 的 fetch
        Test fetch with custom collate_fn"""
        dataset = self._make_iterable([1, 2, 3])
        fetcher = _IterableDatasetFetcher(
            dataset, True, lambda x: sum(x), False
        )
        result = fetcher.fetch([0, 1])
        self.assertEqual(result, 3)

    def test_fetch_drop_last(self):
        """测试 drop_last=True 时数据不足的情况
        Test fetch with drop_last=True when data is insufficient"""

        class LimitedIterable(paddle.io.IterableDataset):
            def __init__(self):
                pass

            def __iter__(self):
                yield 1
                yield 2

        dataset = LimitedIterable()
        fetcher = _IterableDatasetFetcher(
            dataset, True, lambda x: x, drop_last=True
        )
        # Fetch 2 items, get 2 - should be OK
        result = fetcher.fetch([0, 1])
        self.assertEqual(result, [1, 2])
        # Now try to fetch 3 items but only 0 left - should raise StopIteration
        with self.assertRaises(StopIteration):
            fetcher.fetch([0, 1, 2])

    def test_fetch_stop_iteration(self):
        """测试迭代器耗尽时的 StopIteration
        Test StopIteration when iterator is exhausted"""

        class EmptyIterable(paddle.io.IterableDataset):
            def __iter__(self):
                return iter([])

        dataset = EmptyIterable()
        fetcher = _IterableDatasetFetcher(dataset, True, lambda x: x, False)
        with self.assertRaises(StopIteration):
            fetcher.fetch([0])

    def test_fetch_done_event_set(self):
        """测试 done_event 被设置时返回 None
        Test fetch returns None when done_event is set"""
        dataset = self._make_iterable([1, 2, 3])
        fetcher = _IterableDatasetFetcher(dataset, True, lambda x: x, False)
        done_event = MagicMock()
        done_event.is_set.return_value = True
        result = fetcher.fetch([0, 1], done_event=done_event)
        self.assertIsNone(result)


class TestMapDatasetFetcher(unittest.TestCase):
    """测试 _MapDatasetFetcher
    Test _MapDatasetFetcher"""

    def _make_map_dataset(self, data):
        """创建一个 Map 风格数据集
        Create a map-style dataset"""

        class SimpleMapDataset(paddle.io.Dataset):
            def __init__(self, items):
                self.items = items

            def __getitem__(self, idx):
                return self.items[idx]

            def __len__(self):
                return len(self.items)

        return SimpleMapDataset(data)

    def test_fetch_auto_collate(self):
        """测试自动批处理的 fetch
        Test fetch with auto_collate_batch=True"""
        dataset = self._make_map_dataset([10, 20, 30, 40])
        fetcher = _MapDatasetFetcher(dataset, True, lambda x: x, False)
        result = fetcher.fetch([0, 1, 2])
        self.assertEqual(result, [10, 20, 30])

    def test_fetch_no_auto_collate(self):
        """测试非自动批处理的 fetch
        Test fetch with auto_collate_batch=False"""
        dataset = self._make_map_dataset([10, 20, 30])
        fetcher = _MapDatasetFetcher(dataset, False, lambda x: x, False)
        # With auto_collate_batch=False, batch_indices is used directly
        result = fetcher.fetch(1)
        self.assertEqual(result, 20)

    def test_fetch_with_collate_fn(self):
        """测试带 collate_fn 的 fetch
        Test fetch with custom collate_fn"""
        dataset = self._make_map_dataset([1, 2, 3])
        fetcher = _MapDatasetFetcher(dataset, True, lambda x: sum(x), False)
        result = fetcher.fetch([0, 1, 2])
        self.assertEqual(result, 6)

    def test_fetch_done_event_set(self):
        """测试 done_event 被设置时返回 None
        Test fetch returns None when done_event is set"""
        dataset = self._make_map_dataset([1, 2, 3])
        fetcher = _MapDatasetFetcher(dataset, True, lambda x: x, False)
        done_event = MagicMock()
        done_event.is_set.return_value = True
        result = fetcher.fetch([0, 1], done_event=done_event)
        self.assertIsNone(result)

    def test_fetch_no_done_event(self):
        """测试没有 done_event 时的正常 fetch
        Test normal fetch without done_event"""
        dataset = self._make_map_dataset([5, 10, 15])
        fetcher = _MapDatasetFetcher(dataset, True, lambda x: x, False)
        result = fetcher.fetch([0, 2])
        self.assertEqual(result, [5, 15])


if __name__ == '__main__':
    unittest.main()
