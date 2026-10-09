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

# [AUTO-GENERATED] Test file for paddle.io.dataloader.worker
# 覆盖模块: paddle/io/dataloader/worker.py
# Uncovered lines: worker: 41-395

import unittest

import paddle
from paddle.io.dataloader.worker import get_worker_info


class TestWorkerInfo(unittest.TestCase):
    """测试 WorkerInfo 类
    Test WorkerInfo class"""

    def test_worker_info_not_available(self):
        """测试非 worker 环境中 get_worker_info 返回 None
        Test get_worker_info returns None outside worker"""
        result = get_worker_info()
        self.assertIsNone(result)

    def test_dataloader_single_worker(self):
        """测试单 worker DataLoader
        Test single-worker DataLoader"""

        class SimpleDataset(paddle.io.Dataset):
            def __getitem__(self, idx):
                return idx

            def __len__(self):
                return 20

        dataset = SimpleDataset()
        loader = paddle.io.DataLoader(dataset, batch_size=4, num_workers=0)
        for batch in loader:
            self.assertEqual(batch.shape, [4])
            break

    def test_dataloader_multi_worker(self):
        """测试多 worker DataLoader
        Test multi-worker DataLoader"""

        class SimpleDataset(paddle.io.Dataset):
            def __getitem__(self, idx):
                return idx

            def __len__(self):
                return 20

        dataset = SimpleDataset()
        loader = paddle.io.DataLoader(dataset, batch_size=4, num_workers=2)
        count = 0
        for batch in loader:
            count += 1
        self.assertEqual(count, 5)


if __name__ == '__main__':
    unittest.main()
