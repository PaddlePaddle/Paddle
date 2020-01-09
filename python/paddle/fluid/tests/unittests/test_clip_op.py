#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
from op_test import OpTest


class TestClipOp(OpTest):
    def setUp(self):
        self.max_relative_error = 0.006
        self.initTestCase()
        input = np.random.random(self.shape).astype("float32")
        input[np.abs(input - self.min) < self.max_relative_error] = 0.5
        input[np.abs(input - self.max) < self.max_relative_error] = 0.5
        self.op_type = "clip"
        self.inputs = {'X': input, }
        self.attrs = {}
        self.attrs['min'] = self.min
        self.attrs['max'] = self.max
        self.outputs = {
            'Out': np.clip(self.inputs['X'], self.attrs['min'],
                           self.attrs['max'])
        }

    def test_check_output(self):
        self.check_output()

    def test_check_grad_normal(self):
        self.check_grad(['X'], 'Out')

    def initTestCase(self):
        self.shape = (10, 10)
        self.max = 0.7
        self.min = 0.1


class TestCase1(TestClipOp):
    def initTestCase(self):
        self.shape = (8, 16, 8)
        self.max = 0.7
        self.min = 0.0


class TestCase2(TestClipOp):
    def initTestCase(self):
        self.shape = (8, 16)
        self.max = 1.0
        self.min = 0.0


class TestCase3(TestClipOp):
    def initTestCase(self):
        self.shape = (4, 8, 16)
        self.max = 0.7
        self.min = 0.2


if __name__ == '__main__':
    unittest.main()
