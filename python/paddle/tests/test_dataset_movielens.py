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
import numpy as np

from paddle.text.datasets import Movielens


class TestMovielensTrain(unittest.TestCase):
    def test_main(self):
        movielens = Movielens(mode='train')
        # movielens dataset random split train/test
        # not check dataset length here

        # traversal whole dataset may cost a
        # long time, randomly check 1 sample
        idx = np.random.randint(0, 900000)
        data = movielens[idx]
        self.assertTrue(len(data) == 8)
        for i, d in enumerate(data):
            self.assertTrue(len(d.shape) == 1)
            if i not in [5, 6]:
                self.assertTrue(d.shape[0] == 1)


class TestMovielensTest(unittest.TestCase):
    def test_main(self):
        movielens = Movielens(mode='test')
        # movielens dataset random split train/test
        # not check dataset length here

        # traversal whole dataset may cost a
        # long time, randomly check 1 sample
        idx = np.random.randint(0, 100000)
        data = movielens[idx]
        self.assertTrue(len(data) == 8)
        for i, d in enumerate(data):
            self.assertTrue(len(d.shape) == 1)
            if i not in [5, 6]:
                self.assertTrue(d.shape[0] == 1)


if __name__ == '__main__':
    unittest.main()
