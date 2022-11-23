#  Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest
import sys

sys.path.append("..")
import numpy as np
import paddle
import paddle.fluid as fluid
from paddle.fluid import Program, program_guard
from op_test import OpTest
from paddle.fluid.framework import _test_eager_guard

paddle.enable_static()


class TestClipOp(OpTest):

    def setUp(self):
        self.place = paddle.device.MLUPlace(0)
        self.__class__.use_mlu = True
        self.max_relative_error = 0.006
        self.python_api = paddle.clip

        self.inputs = {}
        self.initTestCase()

        self.op_type = "clip"
        self.attrs = {}
        self.attrs['min'] = self.min
        self.attrs['max'] = self.max
        if 'Min' in self.inputs:
            min_v = self.inputs['Min']
        else:
            min_v = self.attrs['min']

        if 'Max' in self.inputs:
            max_v = self.inputs['Max']
        else:
            max_v = self.attrs['max']

        input = np.random.random(self.shape).astype(self.dtype)
        input[np.abs(input - min_v) < self.max_relative_error] = 0.5
        input[np.abs(input - max_v) < self.max_relative_error] = 0.5
        self.inputs['X'] = input
        self.outputs = {'Out': np.clip(self.inputs['X'], min_v, max_v)}

    def test_check_output(self):
        self.check_output_with_place(self.place)

    def test_check_grad_normal(self):
        self.check_grad_with_place(self.place, ['X'], 'Out')

    def initTestCase(self):
        self.dtype = np.float32
        self.shape = (4, 10, 10)
        self.max = 0.8
        self.min = 0.3
        self.inputs['Max'] = np.array([0.8]).astype(self.dtype)
        self.inputs['Min'] = np.array([0.1]).astype(self.dtype)


class TestCase1(TestClipOp):

    def initTestCase(self):
        self.dtype = np.float32
        self.shape = (8, 16, 8)
        self.max = 0.7
        self.min = 0.0


class TestCase2(TestClipOp):

    def initTestCase(self):
        self.dtype = np.float32
        self.shape = (8, 16)
        self.max = 1.0
        self.min = 0.0


class TestCase3(TestClipOp):

    def initTestCase(self):
        self.dtype = np.float32
        self.shape = (4, 8, 16)
        self.max = 0.7
        self.min = 0.2


class TestCase4(TestClipOp):

    def initTestCase(self):
        self.dtype = np.float32
        self.shape = (4, 8, 8)
        self.max = 0.7
        self.min = 0.2
        self.inputs['Max'] = np.array([0.8]).astype(self.dtype)
        self.inputs['Min'] = np.array([0.3]).astype(self.dtype)


class TestCase5(TestClipOp):

    def initTestCase(self):
        self.dtype = np.float32
        self.shape = (4, 8, 16)
        self.max = 0.5
        self.min = 0.5


class TestCase6(TestClipOp):

    def initTestCase(self):
        self.dtype = np.float16
        self.shape = (4, 8, 8)
        self.max = 0.7
        self.min = 0.2
        self.inputs['Max'] = np.array([0.8]).astype(self.dtype)
        self.inputs['Min'] = np.array([0.3]).astype(self.dtype)


if __name__ == '__main__':
    unittest.main()
