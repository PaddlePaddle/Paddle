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


class TestPadOp(OpTest):
    def setUp(self):
        self.initTestCase()
        self.op_type = "pad_constant_like"
        self.inputs = {
            'X': np.random.random(self.x_shape).astype("float64"),
            'Y': np.random.random(self.y_shape).astype("float64")
        }
        self.attrs = {}
        self.attrs['pad_value'] = self.pad_value
        self.outputs = {
            'Out': np.pad(self.inputs['Y'],
                          self.paddings,
                          mode='constant',
                          constant_values=self.pad_value)
        }

    def test_check_output(self):
        self.check_output()

    def test_check_grad_normal(self):
        self.check_grad(['Y'], 'Out')

    def initTestCase(self):
        self.x_shape = (16, 40)
        self.y_shape = (3, 40)
        self.pad_value = 0.1
        self.paddings = [(0, 13), (0, 0)]


class TestCase1(TestPadOp):
    def initTestCase(self):
        self.x_shape = (4, 3, 4, 5)
        self.y_shape = (2, 3, 4, 5)
        self.paddings = [(0, 2), (0, 0), (0, 0), (0, 0)]
        self.pad_value = 0.5


class TestCase2(TestPadOp):
    def initTestCase(self):
        self.x_shape = (4, 3, 4, 10)
        self.y_shape = (2, 3, 2, 10)
        self.paddings = [(0, 2), (0, 0), (0, 2), (0, 0)]
        self.pad_value = 0.5


if __name__ == '__main__':
    unittest.main()
