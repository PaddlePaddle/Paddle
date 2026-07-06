#   Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

import unittest

import numpy as np

import paddle
from paddle.io import DataLoader, Dataset, IterableDataset


class RandomDataset(Dataset):
    def __init__(self, sample_num, feature_dim=10):
        self.sample_num = sample_num
        self.feature_dim = feature_dim
        self.data = np.random.rand(sample_num, feature_dim).astype('float32')
        self.label = np.random.randint(0, 2, size=(sample_num, 1)).astype(
            'int64'
        )

    def __getitem__(self, idx):
        return (
            paddle.to_tensor(self.data[idx]),
            paddle.to_tensor(self.label[idx]),
        )

    def __len__(self):
        return self.sample_num


class RandomIterableDataset(IterableDataset):
    def __init__(self, sample_num, feature_dim=10):
        self.sample_num = sample_num
        self.feature_dim = feature_dim

    def __iter__(self):
        for i in range(self.sample_num):
            yield paddle.rand([self.feature_dim])


def collate_batch(batch):
    data = paddle.stack([item[0] for item in batch])
    label = paddle.stack([item[1] for item in batch])
    return data, label


class TestDataLoaderWindowsMultiprocess(unittest.TestCase):
    def setUp(self):
        self.sample_num = 100
        self.batch_size = 8
        self.feature_dim = 10
        self.epoch_num = 3

    def run_simple_net(self, num_workers, use_shared_memory=False):
        paddle.seed(2026)
        np.random.seed(2026)

        dataset = RandomDataset(self.sample_num, self.feature_dim)
        loader = DataLoader(
            dataset,
            places=paddle.CPUPlace(),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=num_workers,
            use_shared_memory=use_shared_memory,
        )

        losses = []
        for epoch in range(self.epoch_num):
            for data, label in loader():
                # Simple linear layer
                pred = paddle.mean(data, axis=1, keepdim=True)
                loss = paddle.nn.functional.binary_cross_entropy_with_logits(
                    pred, paddle.cast(label, 'float32')
                )
                losses.append(loss.numpy())
        return np.mean(losses)

    def test_multiprocess_singleprocess_loss_close(self):
        """
        Test that multi-process (num_workers=2) and single-process (num_workers=0)
        produce similar loss values.
        """
        loss_single = self.run_simple_net(num_workers=0)
        loss_multi = self.run_simple_net(num_workers=2)
        diff = np.abs(loss_single - loss_multi) / np.abs(loss_single)
        self.assertLess(
            diff,
            1e-2,
            f"Loss difference too large: single={loss_single}, multi={loss_multi}, diff={diff}",
        )

    def test_multiprocess_with_shared_memory(self):
        """
        Test multi-process DataLoader with use_shared_memory=True.
        """
        loss_single = self.run_simple_net(num_workers=0)
        loss_multi = self.run_simple_net(num_workers=2, use_shared_memory=True)
        diff = np.abs(loss_single - loss_multi) / np.abs(loss_single)
        self.assertLess(
            diff,
            1e-2,
            f"Loss difference too large with shared memory: single={loss_single}, multi={loss_multi}, diff={diff}",
        )

    def test_multiprocess_more_workers(self):
        """
        Test with num_workers=4 to stress-test the worker pool.
        """
        loss_single = self.run_simple_net(num_workers=0)
        loss_multi = self.run_simple_net(num_workers=4)
        diff = np.abs(loss_single - loss_multi) / np.abs(loss_single)
        self.assertLess(
            diff,
            1e-2,
            f"Loss difference too large (4 workers): single={loss_single}, multi={loss_multi}, diff={diff}",
        )

    def test_multiprocess_persistent_workers(self):
        """
        Test with persistent_workers=True to ensure worker reuse works.
        """
        paddle.seed(2026)
        np.random.seed(2026)
        dataset = RandomDataset(self.sample_num, self.feature_dim)
        loader = DataLoader(
            dataset,
            places=paddle.CPUPlace(),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=2,
            use_shared_memory=False,
            persistent_workers=True,
        )
        losses = []
        for epoch in range(self.epoch_num):
            for data, label in loader():
                pred = paddle.mean(data, axis=1, keepdim=True)
                loss = paddle.nn.functional.binary_cross_entropy_with_logits(
                    pred, paddle.cast(label, 'float32')
                )
                losses.append(loss.numpy())
        loss_multi = np.mean(losses)

        loss_single = self.run_simple_net(num_workers=0)
        diff = np.abs(loss_single - loss_multi) / np.abs(loss_single)
        self.assertLess(
            diff,
            1e-2,
            f"Loss difference too large (persistent workers): single={loss_single}, multi={loss_multi}, diff={diff}",
        )

    def test_multiprocess_iterable_dataset(self):
        """
        Test multi-process DataLoader with IterableDataset.
        """
        dataset = RandomIterableDataset(50, self.feature_dim)
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=2,
        )
        count = 0
        for batch in loader():
            count += 1
        self.assertGreater(count, 0)


if __name__ == '__main__':
    unittest.main()
