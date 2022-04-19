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
import tempfile

import logging
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.io import Dataset, DataLoader, BatchSampler, SequenceSampler
from paddle.fluid.reader import set_autotune
from paddle.fluid.log_helper import get_logger

#logger = get_logger(__name__, logging.DEBUG)
#log = logging.getLogger()
#handler = logging.StreamHandler()
#log.addHandler(handler)
#log.setLevel(logging.DEBUG)


class RandomDataset(Dataset):
    def __init__(self, num_samples):
        self.num_samples = num_samples

    def __getitem__(self, idx):
        image = np.random.random([100]).astype('float32')
        label = np.random.randint(0, 10 - 1, (1, )).astype('int64')
        return image, label

    def __len__(self):
        return self.num_samples


class SimpleNet(nn.Layer):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc = nn.Linear(100, 10)

    def forward(self, image, label=None):
        return self.fc(image)


class TestAutoTune(unittest.TestCase):
    def test_dataloader_use_autotune(self):
        def train():
            dataset = RandomDataset(10000)
            simple_net = SimpleNet()
            opt = paddle.optimizer.SGD(learning_rate=1e-3,
                                       parameters=simple_net.parameters())

            set_autotune(True)
            loader = DataLoader(
                dataset,
                batch_size=4,
                shuffle=True,
                drop_last=True,
                num_workers=2)
            for i, (image, label) in enumerate(loader()):
                out = simple_net(image)
                loss = F.cross_entropy(out, label)
                avg_loss = paddle.mean(loss)
                avg_loss.backward()
                opt.minimize(avg_loss)
                simple_net.clear_gradients()

        train()

    def test_dataloader_no_autotune(self):
        def train():
            dataset = RandomDataset(20 * 4)
            simple_net = SimpleNet()
            opt = paddle.optimizer.SGD(learning_rate=1e-3,
                                       parameters=simple_net.parameters())

            set_autotune(False)
            loader = DataLoader(
                dataset,
                batch_size=4,
                shuffle=True,
                drop_last=True,
                num_workers=2)
            self.assertEqual(loader.num_workers, 2)
            for i, (image, label) in enumerate(loader()):
                out = simple_net(image)
                loss = F.cross_entropy(out, label)
                avg_loss = paddle.mean(loss)
                avg_loss.backward()
                opt.minimize(avg_loss)
                simple_net.clear_gradients()

        train()

    def test_batchsampler_use_autotune(self):
        def train():
            num_samples_ = 1000
            num_classes_ = 10
            batch_size_ = 32
            shuffle_ = False
            drop_last_ = False
            dataset_ = RandomDataset(num_samples_)
            simple_net = SimpleNet()
            bs = BatchSampler(
                dataset=dataset_,
                batch_size=batch_size_,
                shuffle=shuffle_,
                drop_last=drop_last_)
            opt = paddle.optimizer.SGD(learning_rate=1e-3,
                                       parameters=simple_net.parameters())

            set_autotune(True)
            loader = DataLoader(dataset_, batch_sampler=bs, num_workers=2)
            for i, (image, label) in enumerate(loader()):
                out = simple_net(image)
                loss = F.cross_entropy(out, label)
                avg_loss = paddle.mean(loss)
                avg_loss.backward()
                opt.minimize(avg_loss)
                simple_net.clear_gradients()

        train()

    def test_sampler_use_autotune(self):
        def train():
            num_samples_ = 1000
            num_classes_ = 10
            batch_size_ = 32
            shuffle_ = False
            drop_last_ = False
            simple_net = SimpleNet()
            dataset_ = RandomDataset(num_samples_)
            sampler = SequenceSampler(dataset_)
            bs = BatchSampler(
                sampler=sampler, batch_size=batch_size_, drop_last=drop_last_)
            opt = paddle.optimizer.SGD(learning_rate=1e-3,
                                       parameters=simple_net.parameters())

            set_autotune(True)
            loader = DataLoader(dataset_, batch_sampler=bs, num_workers=2)
            for i, (image, label) in enumerate(loader()):
                out = simple_net(image)
                loss = F.cross_entropy(out, label)
                avg_loss = paddle.mean(loss)
                avg_loss.backward()
                opt.minimize(avg_loss)
                simple_net.clear_gradients()

        train()

    def test_distributer_batch_sampler_autotune(self):
        def train():
            num_samples_ = 1000
            num_classes_ = 10
            batch_size_ = 32
            shuffle_ = False
            drop_last_ = False
            simple_net = SimpleNet()
            dataset_ = RandomDataset(num_samples_)
            bs = paddle.io.DistributedBatchSampler(
                dataset_, batch_size=batch_size_)
            opt = paddle.optimizer.SGD(learning_rate=1e-3,
                                       parameters=simple_net.parameters())
            set_autotune(True)
            loader = DataLoader(dataset_, batch_sampler=bs, num_workers=2)

            for i, (image, label) in enumerate(loader()):
                out = simple_net(image)
                loss = F.cross_entropy(out, label)
                avg_loss = paddle.mean(loss)
                avg_loss.backward()
                opt.minimize(avg_loss)
                simple_net.clear_gradients()

        train()


if __name__ == '__main__':
    unittest.main()
