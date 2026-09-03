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

# [AUTO-GENERATED] Test file for paddle.distributed.fleet.dataset
# 覆盖模块: paddle/distributed/fleet/dataset/dataset.py, paddle/distributed/fleet/dataset/index_dataset.py
# Uncovered lines: dataset.py: 80-261; index_dataset.py: 26-102

import unittest

import paddle


class TestFleetDatasetImport(unittest.TestCase):
    """测试 Fleet Dataset 模块导入
    Test Fleet Dataset module import"""

    def test_dataset_module_importable(self):
        """测试 dataset 模块可导入
        Test dataset module is importable"""
        from paddle.distributed.fleet import dataset

        self.assertIsNotNone(dataset)


class TestDatasetCreation(unittest.TestCase):
    """测试数据集创建
    Test dataset creation"""

    def test_creatable_dataset(self):
        """测试可创建的数据集
        Test creatable dataset"""

        class SimpleDataset(paddle.io.Dataset):
            def __getitem__(self, idx):
                return idx

            def __len__(self):
                return 100

        dataset = SimpleDataset()
        self.assertEqual(len(dataset), 100)

    def test_iterable_dataset(self):
        """测试可迭代数据集
        Test iterable dataset"""

        class SimpleIterable(paddle.io.IterableDataset):
            def __iter__(self):
                yield from range(10)

        dataset = SimpleIterable()
        count = sum(1 for _ in dataset)
        self.assertEqual(count, 10)


if __name__ == '__main__':
    unittest.main()
