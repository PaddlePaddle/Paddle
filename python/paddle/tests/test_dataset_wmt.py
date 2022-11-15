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

from paddle.text.datasets import WMT14, WMT16


class TestWMT14Train(unittest.TestCase):
    def test_main(self):
        wmt14 = WMT14(mode='train', dict_size=50)
        self.assertTrue(len(wmt14) == 191155)

        # traversal whole dataset may cost a
        # long time, randomly check 1 sample
        idx = np.random.randint(0, 191155)
        data = wmt14[idx]
        self.assertTrue(len(data) == 3)
        self.assertTrue(len(data[0].shape) == 1)
        self.assertTrue(len(data[1].shape) == 1)
        self.assertTrue(len(data[2].shape) == 1)


class TestWMT14Test(unittest.TestCase):
    def test_main(self):
        wmt14 = WMT14(mode='test', dict_size=50)
        self.assertTrue(len(wmt14) == 5957)

        # traversal whole dataset may cost a
        # long time, randomly check 1 sample
        idx = np.random.randint(0, 5957)
        data = wmt14[idx]
        self.assertTrue(len(data) == 3)
        self.assertTrue(len(data[0].shape) == 1)
        self.assertTrue(len(data[1].shape) == 1)
        self.assertTrue(len(data[2].shape) == 1)


class TestWMT14Gen(unittest.TestCase):
    def test_main(self):
        wmt14 = WMT14(mode='gen', dict_size=50)
        self.assertTrue(len(wmt14) == 3001)

        # traversal whole dataset may cost a
        # long time, randomly check 1 sample
        idx = np.random.randint(0, 3001)
        data = wmt14[idx]
        self.assertTrue(len(data) == 3)
        self.assertTrue(len(data[0].shape) == 1)
        self.assertTrue(len(data[1].shape) == 1)
        self.assertTrue(len(data[2].shape) == 1)


class TestWMT16Train(unittest.TestCase):
    def test_main(self):
        wmt16 = WMT16(
            mode='train', src_dict_size=50, trg_dict_size=50, lang='en'
        )
        self.assertTrue(len(wmt16) == 29000)

        # traversal whole dataset may cost a
        # long time, randomly check 1 sample
        idx = np.random.randint(0, 29000)
        data = wmt16[idx]
        self.assertTrue(len(data) == 3)
        self.assertTrue(len(data[0].shape) == 1)
        self.assertTrue(len(data[1].shape) == 1)
        self.assertTrue(len(data[2].shape) == 1)


class TestWMT16Test(unittest.TestCase):
    def test_main(self):
        wmt16 = WMT16(
            mode='test', src_dict_size=50, trg_dict_size=50, lang='en'
        )
        self.assertTrue(len(wmt16) == 1000)

        # traversal whole dataset may cost a
        # long time, randomly check 1 sample
        idx = np.random.randint(0, 1000)
        data = wmt16[idx]
        self.assertTrue(len(data) == 3)
        self.assertTrue(len(data[0].shape) == 1)
        self.assertTrue(len(data[1].shape) == 1)
        self.assertTrue(len(data[2].shape) == 1)


class TestWMT16Val(unittest.TestCase):
    def test_main(self):
        wmt16 = WMT16(mode='val', src_dict_size=50, trg_dict_size=50, lang='en')
        self.assertTrue(len(wmt16) == 1014)

        # traversal whole dataset may cost a
        # long time, randomly check 1 sample
        idx = np.random.randint(0, 1014)
        data = wmt16[idx]
        self.assertTrue(len(data) == 3)
        self.assertTrue(len(data[0].shape) == 1)
        self.assertTrue(len(data[1].shape) == 1)
        self.assertTrue(len(data[2].shape) == 1)


if __name__ == '__main__':
    unittest.main()
