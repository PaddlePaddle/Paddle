# Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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


class BaseTestArgMinMaxOp(OpTest):
    def initTestCase(self):
        self.x = np.random.random((3, 4, 5)).astype('float32')
        self.axis = 0

    def setUp(self):
        self.initTestCase()
        self.inputs = {'X': self.x}
        self.attrs = {'axis': self.axis}
        self.outputs = {'Out': None}

    def test_check_output(self):
        self.check_output()


class TestArgMinOp(BaseTestArgMinMaxOp):
    def setUp(self):
        self.op_type = 'argmin'
        self.outputs = {'Out': np.argmin(self.x)}


class TestArgMaxOp(BaseTestArgMinMaxOp):
    def setUp(self):
        self.op_type = 'argmax'
        self.outputs = {'Out': np.argmax(self.x)}


class TestCase0ArgMin(TestArgMinOp):
    def setUp(self):
        self.x = np.random.random((3, 4, 5)).astype('float64')
        self.axis = -1


class TestCase1ArgMin(TestArgMinOp):
    def setUp(self):
        self.x = np.random.random((3, 4, 5)).astype('int64')
        self.axis = -2


class TestCase0ArgMax(TestArgMaxOp):
    def setUp(self):
        self.x = np.random.random((2, 3, 4, 5)).astype('float64')
        self.axis = 1


class TestCase1ArgMax(TestArgMaxOp):
    def setUp(self):
        self.x = np.random.random((7)).astype('int64')
        self.axis = 0


if __name__ == '__main__':
    unittest.main()
