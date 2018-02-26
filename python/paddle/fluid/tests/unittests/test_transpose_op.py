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


class TestTransposeOp(OpTest):
    def setUp(self):
        self.initTestCase()
        self.op_type = "transpose"
        self.inputs = {'X': np.random.random(self.shape).astype("float32")}
        self.attrs = {'axis': list(self.axis)}
        self.outputs = {'Out': self.inputs['X'].transpose(self.axis)}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X'], 'Out')

    def initTestCase(self):
        self.shape = (3, 4)
        self.axis = (1, 0)


class TestCase0(TestTransposeOp):
    def initTestCase(self):
        self.shape = (3, )
        self.axis = (0, )


class TestCase1(TestTransposeOp):
    def initTestCase(self):
        self.shape = (3, 4, 5)
        self.axis = (0, 2, 1)


class TestCase2(TestTransposeOp):
    def initTestCase(self):
        self.shape = (2, 3, 4, 5)
        self.axis = (0, 2, 3, 1)


class TestCase3(TestTransposeOp):
    def initTestCase(self):
        self.shape = (2, 3, 4, 5, 6)
        self.axis = (4, 2, 3, 1, 0)


class TestCase4(TestTransposeOp):
    def initTestCase(self):
        self.shape = (2, 3, 4, 5, 6, 1)
        self.axis = (4, 2, 3, 1, 0, 5)


if __name__ == '__main__':
    unittest.main()
