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

import random
import unittest

import numpy as np

from paddle.io import (
    BatchSampler,
    Dataset,
    RandomSampler,
    Sampler,
    SequenceSampler,
    SubsetRandomSampler,
    WeightedRandomSampler,
)

IMAGE_SIZE = 32


class RandomDataset(Dataset):
    def __init__(self, sample_num, class_num):
        self.sample_num = sample_num
        self.class_num = class_num

    def __getitem__(self, idx):
        np.random.seed(idx)
        image = np.random.random([IMAGE_SIZE]).astype('float32')
        label = np.random.randint(0, self.class_num - 1, (1,)).astype('int64')
        return image, label

    def __len__(self):
        return self.sample_num


class TestSampler(unittest.TestCase):
    def test_main(self):
        dataset = RandomDataset(100, 10)
        sampler = Sampler(dataset)
        try:
            iter(sampler)
            self.assertTrue(False)
        except NotImplementedError:
            pass


class TestSequenceSampler(unittest.TestCase):
    def test_main(self):
        dataset = RandomDataset(100, 10)
        sampler = SequenceSampler(dataset)
        assert len(sampler) == 100

        for i, index in enumerate(iter(sampler)):
            assert i == index


class TestRandomSampler(unittest.TestCase):
    def test_main(self):
        dataset = RandomDataset(100, 10)
        sampler = RandomSampler(dataset)
        assert len(sampler) == 100

        rets = []
        for i in iter(sampler):
            rets.append(i)
        assert tuple(sorted(rets)) == tuple(range(0, 100))

    def test_with_num_samples(self):
        dataset = RandomDataset(100, 10)
        sampler = RandomSampler(dataset, num_samples=50, replacement=True)
        assert len(sampler) == 50

        rets = []
        for i in iter(sampler):
            rets.append(i)
            assert i >= 0 and i < 100

    def test_with_num_samples_and_without_replacement(self):
        dataset = RandomDataset(100, 10)
        sampler = RandomSampler(dataset, num_samples=80, replacement=False)
        assert len(sampler) == 80

        rets = []
        for i in iter(sampler):
            rets.append(i)
            assert i >= 0 and i < 100

    def test_with_generator(self):
        dataset = RandomDataset(100, 10)
        generator = iter(range(0, 60))
        sampler = RandomSampler(dataset, generator=generator)
        assert len(sampler) == 100

        rets = []
        for i in iter(sampler):
            rets.append(i)
        assert tuple(sorted(rets)) == tuple(range(0, 60))

    def test_with_generator_num_samples(self):
        dataset = RandomDataset(100, 10)
        generator = iter(range(0, 60))
        sampler = RandomSampler(
            dataset, generator=generator, num_samples=50, replacement=True
        )
        assert len(sampler) == 50

        rets = []
        for i in iter(sampler):
            rets.append(i)
        assert tuple(sorted(rets)) == tuple(range(0, 50))

    def test_with_num_samples_error(self):
        dataset = RandomDataset(100, 10)
        self.assertRaises(ValueError, RandomSampler, dataset, False, 120)


class TestSubsetRandomSampler(unittest.TestCase):
    def test_main(self):
        indices = list(range(100))
        random.shuffle(indices)
        indices = indices[:30]
        sampler = SubsetRandomSampler(indices)
        assert len(sampler) == len(indices)

        hints = {i: 0 for i in indices}
        for index in iter(sampler):
            hints[index] += 1
        for h in hints.values():
            assert h == 1

    def test_raise(self):
        try:
            sampler = SubsetRandomSampler([])
            self.assertTrue(False)
        except ValueError:
            self.assertTrue(True)


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
            drop_last=self.drop_last,
        )
        return bs

    def test_main(self):
        bs = self.init_batch_sampler()
        # length check
        bs_len = (
            self.num_samples + int(not self.drop_last) * (self.batch_size - 1)
        ) // self.batch_size
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


class TestBatchSamplerWithSampler(TestBatchSampler):
    def init_batch_sampler(self):
        dataset = RandomDataset(1000, 10)
        sampler = SequenceSampler(dataset)
        bs = BatchSampler(
            sampler=sampler,
            batch_size=self.batch_size,
            drop_last=self.drop_last,
        )
        return bs


class TestBatchSamplerWithSamplerDropLast(unittest.TestCase):
    def setUp(self):
        self.num_samples = 1000
        self.num_classes = 10
        self.batch_size = 32
        self.shuffle = False
        self.drop_last = True


class TestBatchSamplerWithSamplerShuffle(unittest.TestCase):
    def setUp(self):
        self.num_samples = 1000
        self.num_classes = 10
        self.batch_size = 32
        self.shuffle = True
        self.drop_last = True

    def test_main(self):
        try:
            dataset = RandomDataset(self.num_samples, self.num_classes)
            sampler = RandomSampler(dataset)
            bs = BatchSampler(
                sampler=sampler,
                shuffle=self.shuffle,
                batch_size=self.batch_size,
                drop_last=self.drop_last,
            )
            self.assertTrue(False)
        except AssertionError:
            pass


class TestBatchSamplerWithIterableSampler(TestBatchSampler):
    def init_batch_sampler(self):
        sampler = range(1000)
        bs = BatchSampler(
            sampler=sampler,
            batch_size=self.batch_size,
            drop_last=self.drop_last,
        )
        return bs


class TestBatchSamplerWithIterableSamplerDropLast(
    TestBatchSamplerWithIterableSampler
):
    def setUp(self):
        self.num_samples = 1000
        self.num_classes = 10
        self.batch_size = 32
        self.shuffle = False
        self.drop_last = True


class TestBatchSamplerWithIterableSamplerShuffle(
    TestBatchSamplerWithIterableSampler
):
    def setUp(self):
        self.num_samples = 1000
        self.num_classes = 10
        self.batch_size = 32
        self.shuffle = True
        self.drop_last = True


class TestWeightedRandomSampler(unittest.TestCase):
    def init_probs(self, total, pos):
        pos_probs = np.random.random((pos,)).astype('float32')
        probs = np.zeros((total,)).astype('float32')
        probs[:pos] = pos_probs
        np.random.shuffle(probs)
        return probs

    def test_replacement(self):
        probs = self.init_probs(20, 10)
        sampler = WeightedRandomSampler(probs, 30, True)
        assert len(sampler) == 30
        for idx in iter(sampler):
            assert probs[idx] > 0.0

    def test_no_replacement(self):
        probs = self.init_probs(20, 10)
        sampler = WeightedRandomSampler(probs, 10, False)
        assert len(sampler) == 10
        idxs = []
        for idx in iter(sampler):
            assert probs[idx] > 0.0
            idxs.append(idx)
        assert len(set(idxs)) == len(idxs)

    def test_assert(self):
        # all zeros
        probs = np.zeros((10,)).astype('float32')
        sampler = WeightedRandomSampler(probs, 10, True)
        try:
            for idx in iter(sampler):
                pass
            self.assertTrue(False)
        except AssertionError:
            self.assertTrue(True)

        # not enough pos
        probs = self.init_probs(10, 5)
        sampler = WeightedRandomSampler(probs, 10, False)
        try:
            for idx in iter(sampler):
                pass
            self.assertTrue(False)
        except AssertionError:
            self.assertTrue(True)

        # neg probs
        probs = -1.0 * np.ones((10,)).astype('float32')
        sampler = WeightedRandomSampler(probs, 10, True)
        try:
            for idx in iter(sampler):
                pass
            self.assertTrue(False)
        except AssertionError:
            self.assertTrue(True)

    def test_raise(self):
        # float num_samples
        probs = self.init_probs(10, 5)
        try:
            sampler = WeightedRandomSampler(probs, 2.3, True)
            self.assertTrue(False)
        except ValueError:
            self.assertTrue(True)

        # neg num_samples
        probs = self.init_probs(10, 5)
        try:
            sampler = WeightedRandomSampler(probs, -1, True)
            self.assertTrue(False)
        except ValueError:
            self.assertTrue(True)

        # no-bool replacement
        probs = self.init_probs(10, 5)
        try:
            sampler = WeightedRandomSampler(probs, 5, 5)
            self.assertTrue(False)
        except ValueError:
            self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()
