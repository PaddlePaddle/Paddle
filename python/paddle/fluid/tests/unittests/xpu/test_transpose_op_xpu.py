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
import sys

sys.path.append("..")
from op_test_xpu import OpTest, XPUOpTest
import paddle
import paddle.fluid.core as core
import paddle.fluid as fluid
from paddle.fluid import compiler, Program, program_guard


class TestXPUTransposeOp(XPUOpTest):

    def setUp(self):
        self.init_op_type()
        self.initTestCase()
        self.use_xpu = True
        self.use_mkldnn = False
        self.inputs = {'X': np.random.random(self.shape).astype("float32")}
        self.attrs = {
            'axis': list(self.axis),
            'use_mkldnn': False,
            'use_xpu': True
        }
        self.outputs = {
            'XShape': np.random.random(self.shape).astype("float32"),
            'Out': self.inputs['X'].transpose(self.axis)
        }

    def init_op_type(self):
        self.op_type = "transpose2"
        self.use_mkldnn = False

    def test_check_output(self):
        if paddle.is_compiled_with_xpu():
            paddle.enable_static()
            place = paddle.XPUPlace(0)
            self.check_output_with_place(place=place, no_check_set=['XShape'])

    def test_check_grad(self):
        if paddle.is_compiled_with_xpu():
            paddle.enable_static()
            place = paddle.XPUPlace(0)
            self.check_grad_with_place(place, ['X'], 'Out')

    def initTestCase(self):
        self.shape = (3, 40)
        self.axis = (1, 0)


class TestCase0(TestXPUTransposeOp):

    def initTestCase(self):
        self.shape = (100, )
        self.axis = (0, )


class TestCase1(TestXPUTransposeOp):

    def initTestCase(self):
        self.shape = (3, 4, 10)
        self.axis = (0, 2, 1)


class TestCase2(TestXPUTransposeOp):

    def initTestCase(self):
        self.shape = (2, 3, 4, 5)
        self.axis = (0, 2, 3, 1)


class TestCase3(TestXPUTransposeOp):

    def initTestCase(self):
        self.shape = (2, 3, 4, 5, 6)
        self.axis = (4, 2, 3, 1, 0)


class TestCase4(TestXPUTransposeOp):

    def initTestCase(self):
        self.shape = (2, 3, 4, 5, 6, 1)
        self.axis = (4, 2, 3, 1, 0, 5)


class TestCase5(TestXPUTransposeOp):

    def initTestCase(self):
        self.shape = (2, 16, 96)
        self.axis = (0, 2, 1)


class TestCase6(TestXPUTransposeOp):

    def initTestCase(self):
        self.shape = (2, 10, 12, 16)
        self.axis = (3, 1, 2, 0)


class TestCase7(TestXPUTransposeOp):

    def initTestCase(self):
        self.shape = (2, 10, 2, 16)
        self.axis = (0, 1, 3, 2)


class TestCase8(TestXPUTransposeOp):

    def initTestCase(self):
        self.shape = (2, 3, 2, 3, 2, 4, 3, 3)
        self.axis = (0, 1, 3, 2, 4, 5, 6, 7)


class TestCase9(TestXPUTransposeOp):

    def initTestCase(self):
        self.shape = (2, 3, 2, 3, 2, 4, 3, 3)
        self.axis = (6, 1, 3, 5, 0, 2, 4, 7)


if __name__ == "__main__":
    unittest.main()
