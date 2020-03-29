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
import numpy as np

import paddle.fluid as fluid
from paddle.fluid.io import *


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


class TestMNISTTest(unittest.TestCase):
    def test_main(self):
        mnist = MNIST(mode='test')
        self.assertTrue(len(mnist) == 10000)

        for i in range(len(mnist)):
            image, label = mnist[i]
            self.assertTrue(image.shape[0] == 784)
            self.assertTrue(label.shape[0] == 1)
            self.assertTrue(0 <= int(label) <= 9)


class TestMNISTTrain(unittest.TestCase):
    def test_main(self):
        mnist = MNIST(mode='train')
        self.assertTrue(len(mnist) == 60000)

        for i in range(len(mnist)):
            image, label = mnist[i]
            self.assertTrue(image.shape[0] == 784)
            self.assertTrue(label.shape[0] == 1)
            self.assertTrue(0 <= int(label) <= 9)


class TestFlowersTrain(unittest.TestCase):
    def test_main(self):
        flowers = Flowers(mode='train')
        self.assertTrue(len(flowers) == 6149)

        # traversal whole dataset may cost a
        # long time, randomly check 1 sample
        idx = np.random.randint(0, 6149)
        image, label = flowers[idx]
        self.assertTrue(len(image.shape) == 3)
        self.assertTrue(image.shape[2] == 3)
        self.assertTrue(label.shape[0] == 1)


class TestFlowersValid(unittest.TestCase):
    def test_main(self):
        flowers = Flowers(mode='valid')
        self.assertTrue(len(flowers) == 1020)

        # traversal whole dataset may cost a
        # long time, randomly check 1 sample
        idx = np.random.randint(0, 1020)
        image, label = flowers[idx]
        self.assertTrue(len(image.shape) == 3)
        self.assertTrue(image.shape[2] == 3)
        self.assertTrue(label.shape[0] == 1)


class TestFlowersTest(unittest.TestCase):
    def test_main(self):
        flowers = Flowers(mode='test')
        self.assertTrue(len(flowers) == 1020)

        # traversal whole dataset may cost a
        # long time, randomly check 1 sample
        idx = np.random.randint(0, 1020)
        image, label = flowers[idx]
        self.assertTrue(len(image.shape) == 3)
        self.assertTrue(image.shape[2] == 3)
        self.assertTrue(label.shape[0] == 1)


if __name__ == '__main__':
    unittest.main()
