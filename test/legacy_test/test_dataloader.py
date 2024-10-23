# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
import paddle.nn.functional as F
from paddle import base
from paddle.io import DataLoader, Dataset

BATCH_NUM = 4
BATCH_SIZE = 8
EPOCH_NUM = 2

IMAGE_SIZE = 784
CLASS_NUM = 10


# define a random dataset
class RandomDataset(Dataset):
    def __init__(self, num_samples):
        self.num_samples = num_samples

    def __getitem__(self, idx):
        image = np.random.random([IMAGE_SIZE]).astype('float32')
        label = np.random.randint(0, CLASS_NUM - 1, (1,)).astype('int64')
        return image, label

    def __len__(self):
        return self.num_samples


dataset = RandomDataset(BATCH_NUM * BATCH_SIZE)


class TestDygraphDataLoader(unittest.TestCase):
    def setUp(self):
        self.batch_size = BATCH_SIZE
        self.batch_num = BATCH_NUM
        self.epoch_num = EPOCH_NUM

    def iter_loader_data(self, loader):
        for _ in range(self.epoch_num):
            for image, label in loader():
                relu = F.relu(image)
                self.assertEqual(image.shape, [self.batch_size, IMAGE_SIZE])
                self.assertEqual(label.shape, [self.batch_size, 1])
                self.assertEqual(relu.shape, [self.batch_size, IMAGE_SIZE])

    def test_single_process_loader_filedescriptor(self):
        with base.dygraph.guard():
            loader = DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=True,
                drop_last=True,
                use_shared_memory=True,
                num_workers=0,
            )
            self.iter_loader_data(loader)

    def test_multi_process_dataloader_filedescriptor(self):
        with base.dygraph.guard():
            loader = DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=True,
                drop_last=True,
                use_shared_memory=True,
                num_workers=2,
            )
            self.iter_loader_data(loader)

    def test_single_process_loader_filename(self):
        paddle.base.core.globals()[
            "FLAGS_dataloader_use_file_descriptor"
        ] = False
        with base.dygraph.guard():
            loader = DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=True,
                drop_last=True,
                use_shared_memory=True,
                num_workers=0,
            )
            self.iter_loader_data(loader)

    def test_multi_process_dataloader_filename(self):
        paddle.base.core.globals()[
            "FLAGS_dataloader_use_file_descriptor"
        ] = False
        with base.dygraph.guard():
            loader = DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=True,
                drop_last=True,
                use_shared_memory=True,
                num_workers=2,
            )
            self.iter_loader_data(loader)


if __name__ == '__main__':
    unittest.main()
