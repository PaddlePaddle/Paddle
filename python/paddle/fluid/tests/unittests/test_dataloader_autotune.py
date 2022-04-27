#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import print_function
import unittest
import numpy as np

import paddle
import paddle.nn as nn
from paddle.io import Dataset, DataLoader, BatchSampler, SequenceSampler
from paddle.fluid.reader import set_autotune_config
import sys


class RandomDataset(Dataset):
    def __init__(self, num_samples):
        self.num_samples = num_samples

    def __getitem__(self, idx):
        image = np.random.random([10]).astype('float32')
        label = np.random.randint(0, 10 - 1, (1, )).astype('int64')
        return image, label

    def __len__(self):
        return self.num_samples


class SimpleNet(nn.Layer):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc = nn.Linear(10, 10)

    def forward(self, image):
        return self.fc(image)


class TestAutoTune(unittest.TestCase):
    def setUp(self):
        self.batch_size = 1
        self.dataset = RandomDataset(10)

    def test_dataloader_use_autotune(self):
        set_autotune_config(True, 1)
        loader = DataLoader(
            self.dataset, batch_size=self.batch_size, num_workers=0)

    def test_dataloader_disable_autotune(self):
        set_autotune_config(False)
        loader = DataLoader(
            self.dataset, batch_size=self.batch_size, num_workers=2)
        if (sys.platform == 'darwin' or sys.platform == 'win32'):
            self.assertEqual(loader.num_workers, 0)
        else:
            self.assertEqual(loader.num_workers, 2)

    def test_distributer_batch_sampler_autotune(self):
        set_autotune_config(True, 1)
        batch_sampler = paddle.io.DistributedBatchSampler(
            self.dataset, batch_size=self.batch_size)
        loader = DataLoader(
            self.dataset, batch_sampler=batch_sampler, num_workers=2)


if __name__ == '__main__':
    unittest.main()
