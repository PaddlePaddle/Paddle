# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import division

import unittest

import paddle.fluid as fluid
from paddle.io import BatchSampler, Dataset


class RandomDataset(Dataset):
    def __init__(self, sample_num, class_num):
        self.sample_num = sample_num
        self.class_num = class_num

    def __getitem__(self, idx):
        np.random.seed(idx)
        image = np.random.random([IMAGE_SIZE]).astype('float32')
        label = np.random.randint(0, CLASS_NUM - 1, (1, )).astype('int64')
        return image, label

    def __len__(self):
        return self.sample_num


class TestBatchSampler(unittest.TestCase):
    def setUp(self):
        self.num_samples = 1000
        self.num_classes = 10
        self.batch_size = 32
        self.shuffle = False
        self.drop_last = False

    def init_batch_sampler(self):
        dataset = RandomDataset(self.num_samples, self.num_classes)
        bs = BatchSampler(
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            drop_last=self.drop_last)
        return bs

    def test_main(self):
        bs = self.init_batch_sampler()
        # length check
        bs_len = (self.num_samples + int(not self.drop_last) \
                * (self.batch_size - 1)) // self.batch_size
        self.assertTrue(bs_len == len(bs))

        # output indices check
        if not self.shuffle:
            index = 0
            for indices in bs:
                for idx in indices:
                    self.assertTrue(index == idx)
                    index += 1


class TestBatchSamplerDropLast(TestBatchSampler):
    def setUp(self):
        self.num_samples = 1000
        self.num_classes = 10
        self.batch_size = 32
        self.shuffle = False
        self.drop_last = True


class TestBatchSamplerShuffle(TestBatchSampler):
    def setUp(self):
        self.num_samples = 1000
        self.num_classes = 10
        self.batch_size = 32
        self.shuffle = True
        self.drop_last = True


class TestBatchSamplerWithIndices(TestBatchSampler):
    def init_batch_sampler(self):
        bs = BatchSampler(
            indices=list(range(self.num_samples)),
            batch_size=self.batch_size,
            drop_last=self.drop_last)
        return bs


class TestBatchSamplerWithIndicesAndDataSource(unittest.TestCase):
    def setUp(self):
        self.num_samples = 1000
        self.num_classes = 10
        self.batch_size = 32
        self.shuffle = False
        self.drop_last = True

    def test_main(self):
        try:
            dataset = RandomDataset(self.num_samples, self.num_classes)
            bs = BatchSampler(
                dataset=dataset,
                indices=list(range(self.num_samples)),
                batch_size=self.batch_size,
                drop_last=self.drop_last)
            self.assertTrue(False)
        except AssertionError:
            pass


if __name__ == '__main__':
    unittest.main()
