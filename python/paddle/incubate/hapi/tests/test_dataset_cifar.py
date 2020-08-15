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

import unittest
import os
import numpy as np
import tempfile
import shutil
import cv2

from paddle.incubate.hapi.datasets import *
from paddle.incubate.hapi.datasets.utils import _check_exists_and_download


class TestCifarTrain10(unittest.TestCase):
     def test_main(self):
         cifar = Cifar(mode='train10')
         self.assertTrue(len(cifar) == 50000)

          # traversal whole dataset may cost a
         # long time, randomly check 1 sample
         idx = np.random.randint(0, 50000)
         data, label = cifar[idx]
         self.assertTrue(len(data.shape) == 1)
         self.assertTrue(data.shape[0] == 3072)
         self.assertTrue(0 <= int(label) <= 9)


class TestCifarTest10(unittest.TestCase):
    def test_main(self):
        cifar = Cifar(mode='test10')
        self.assertTrue(len(cifar) == 10000)

        # traversal whole dataset may cost a
        # long time, randomly check 1 sample
        idx = np.random.randint(0, 10000)
        data, label = cifar[idx]
        self.assertTrue(len(data.shape) == 1)
        self.assertTrue(data.shape[0] == 3072)
        self.assertTrue(0 <= int(label) <= 9)


class TestCifarTrain100(unittest.TestCase):
    def test_main(self):
        cifar = Cifar(mode='train100')
        self.assertTrue(len(cifar) == 50000)

        # traversal whole dataset may cost a
        # long time, randomly check 1 sample
        idx = np.random.randint(0, 50000)
        data, label = cifar[idx]
        self.assertTrue(len(data.shape) == 1)
        self.assertTrue(data.shape[0] == 3072)
        self.assertTrue(0 <= int(label) <= 99)


class TestCifarTest100(unittest.TestCase):
    def test_main(self):
        cifar = Cifar(mode='test100')
        self.assertTrue(len(cifar) == 10000)

        # traversal whole dataset may cost a
        # long time, randomly check 1 sample
        idx = np.random.randint(0, 10000)
        data, label = cifar[idx]
        self.assertTrue(len(data.shape) == 1)
        self.assertTrue(data.shape[0] == 3072)
        self.assertTrue(0 <= int(label) <= 99)


if __name__ == '__main__':
    unittest.main()
