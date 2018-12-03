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

import unittest
import numpy as np
from op_test import OpTest


class TestPad2dOp(OpTest):
    def setUp(self):
        self.pad_value = 0.0
        self.variable_paddings = False
        self.initTestCase()
        self.op_type = "pad2d"
        self.inputs = {'X': np.random.random(self.shape).astype("float32"), }
        self.attrs = {}
        if self.variable_paddings:
            self.attrs['paddings'] = []
            self.inputs['Paddings'] = np.array(self.paddings).flatten().astype(
                "int32")
        else:
            self.attrs['paddings'] = np.array(self.paddings).flatten()
        self.attrs['pad_value'] = self.pad_value
        self.attrs['mode'] = self.mode
        self.attrs['data_format'] = self.data_format
        if self.data_format == "NCHW":
            paddings = [(0, 0), (0, 0), (self.paddings[0], self.paddings[1]),
                        (self.paddings[2], self.paddings[3])]
        else:
            paddings = [(0, 0), (self.paddings[0], self.paddings[1]),
                        (self.paddings[2], self.paddings[3]), (0, 0)]
        if self.mode == "constant":
            out = np.pad(self.inputs['X'],
                         paddings,
                         mode=self.mode,
                         constant_values=self.pad_value)
        else:
            out = np.pad(self.inputs['X'], paddings, mode=self.mode)
        self.outputs = {'Out': out}

    def test_check_output(self):
        self.check_output()

    def test_check_grad_normal(self):
        self.check_grad(['X'], 'Out', max_relative_error=0.006)

    def initTestCase(self):
        self.shape = (2, 3, 4, 4)
        self.paddings = [0, 1, 2, 3]
        self.mode = "constant"
        self.data_format = "NCHW"
        self.pad_value = 0.0


class TestCase1(TestPad2dOp):
    def initTestCase(self):
        self.shape = (2, 3, 4, 4)
        self.paddings = [0, 1, 2, 3]
        self.mode = "reflect"
        self.data_format = "NCHW"


class TestCase2(TestPad2dOp):
    def initTestCase(self):
        self.shape = (2, 3, 4, 4)
        self.paddings = [0, 1, 2, 3]
        self.mode = "edge"
        self.data_format = "NCHW"


class TestCase3(TestPad2dOp):
    def initTestCase(self):
        self.shape = (2, 4, 4, 2)
        self.paddings = [0, 1, 2, 3]
        self.mode = "reflect"
        self.data_format = "NHWC"


class TestCase4(TestPad2dOp):
    def initTestCase(self):
        self.shape = (2, 4, 4, 2)
        self.paddings = [0, 1, 2, 3]
        self.mode = "edge"
        self.data_format = "NHWC"


class TestCase5(TestPad2dOp):
    def initTestCase(self):
        self.shape = (2, 4, 4, 2)
        self.paddings = [0, 1, 2, 3]
        self.mode = "constant"
        self.pad_value = 1.2
        self.data_format = "NHWC"


class TestCase6(TestPad2dOp):
    def initTestCase(self):
        self.shape = (2, 4, 4, 2)
        self.paddings = [0, 1, 2, 3]
        self.mode = "constant"
        self.pad_value = 1.2
        self.data_format = "NHWC"
        self.variable_paddings = True


class TestCase7(TestPad2dOp):
    def initTestCase(self):
        self.shape = (2, 3, 4, 4)
        self.paddings = [0, 1, 2, 3]
        self.mode = "reflect"
        self.data_format = "NCHW"
        self.variable_paddings = True


if __name__ == '__main__':
    unittest.main()
