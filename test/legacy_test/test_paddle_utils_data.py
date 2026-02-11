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
import unittest

import paddle


class TestUtilsAttrError(unittest.TestCase):
    def test_error(self):
        with self.assertRaises(AttributeError):
            type(paddle.utils.nonexist)


class TestAlias(unittest.TestCase):
    def setUp(self):
        self.ioObject = paddle.io.Dataset
        self.utilsObject = paddle.utils.data.Dataset
        self.directObject = paddle.utils.data.dataset.Dataset

    def test_compatibility(self):
        self.assertTrue(type(self.ioObject) == type(self.utilsObject))
        self.assertTrue(type(self.ioObject) == type(self.directObject))


class TestChainDatasetAlias(TestAlias):
    def setUp(self):
        self.ioObject = paddle.io.ChainDataset
        self.utilsObject = paddle.utils.data.ChainDataset
        self.directObject = paddle.utils.data.dataset.ChainDataset


class TestConcatDatasetAlias(TestAlias):
    def setUp(self):
        self.ioObject = paddle.io.ConcatDataset
        self.utilsObject = paddle.utils.data.ConcatDataset
        self.directObject = paddle.utils.data.dataset.ConcatDataset


class TestIterableDatasetAlias(TestAlias):
    def setUp(self):
        self.ioObject = paddle.io.IterableDataset
        self.utilsObject = paddle.utils.data.IterableDataset
        self.directObject = paddle.utils.data.dataset.IterableDataset


class TestSamplerAlias(TestAlias):
    def setUp(self):
        self.ioObject = paddle.io.Sampler
        self.utilsObject = paddle.utils.data.Sampler
        self.directObject = paddle.utils.data.sampler.Sampler


class TestSequentialSamplerAlias(TestAlias):
    def setUp(self):
        self.ioObject = paddle.io.SequenceSampler
        self.utilsObject = paddle.utils.data.SequentialSampler
        self.directObject = paddle.utils.data.sampler.SequentialSampler


class TestSubsetAlias(TestAlias):
    def setUp(self):
        self.ioObject = paddle.io.Subset
        self.utilsObject = paddle.utils.data.Subset
        self.directObject = paddle.utils.data.dataset.Subset


class TestGetWorkerInfoAlias(TestAlias):
    def setUp(self):
        self.ioObject = paddle.io.get_worker_info
        self.utilsObject = paddle.utils.data.get_worker_info
        self.directObject = paddle.utils.data.dataloader.get_worker_info


class TestRandomSplitAlias(TestAlias):
    def setUp(self):
        self.ioObject = paddle.io.random_split
        self.utilsObject = paddle.utils.data.random_split
        self.directObject = paddle.utils.data.dataset.random_split


class TestDefaultCollateAlias(TestAlias):
    def setUp(self):
        self.ioObject = paddle.io.dataloader.collate.default_collate_fn
        self.utilsObject = paddle.utils.data.default_collate
        self.directObject = paddle.utils.data.dataloader.default_collate


if __name__ == "__main__":
    unittest.main()
