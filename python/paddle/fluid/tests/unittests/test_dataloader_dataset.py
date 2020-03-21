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

import os
import six
import time
import unittest
import numpy as np

import paddle.fluid as fluid
from paddle.fluid.io import Dataset, MnistDataset, BatchSampler, DataLoader
from paddle.fluid.dygraph.nn import Linear
from paddle.fluid.dygraph.base import to_variable


class TestDatasetAbstract(unittest.TestCase):
    def test_main(self):
        dataset = Dataset()
        try:
            d = dataset[0]
            self.assertTrue(False)
        except NotImplementedError:
            pass

        try:
            l = len(dataset)
            self.assertTrue(False)
        except NotImplementedError:
            pass


class TestMnistDataset(unittest.TestCase):
    def test_main(self):
        md = MnistDataset(mode='test')
        self.assertTrue(len(md) == 10000)

        for i in range(len(md)):
            image, label = md[i]
            self.assertTrue(image.shape[0] == 784)
            self.assertTrue(isinstance(label, int))


class TestMnistDatasetTrain(unittest.TestCase):
    def test_main(self):
        md = MnistDataset(mode='train')
        self.assertTrue(len(md) == 60000)

        for i in range(len(md)):
            image, label = md[i]
            self.assertTrue(image.shape[0] == 784)
            self.assertTrue(isinstance(label, int))


if __name__ == '__main__':
    unittest.main()
