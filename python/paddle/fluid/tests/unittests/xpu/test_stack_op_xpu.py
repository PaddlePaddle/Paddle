# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
import sys
sys.path.append("..")
import unittest
import numpy as np
import paddle.fluid.core as core
from op_test import OpTest, skip_check_grad_ci
from op_test_xpu import XPUOpTest
import paddle
import paddle.fluid as fluid
from paddle.fluid import Program, program_guard


@skip_check_grad_ci(reason="There is no grad kernel for stack_xpu op.")
class TestStackOpBase(XPUOpTest):
    def initDefaultParameters(self):
        self.num_inputs = 4
        self.input_dim = (5, 6, 7)
        self.axis = 0
        self.dtype = 'float32'

    def initParameters(self):
        pass

    def get_x_names(self):
        x_names = []
        for i in range(self.num_inputs):
            x_names.append('x{}'.format(i))
        return x_names

    def setUp(self):
        self.initDefaultParameters()
        self.initParameters()
        self.op_type = 'stack'
        self.x = []
        for i in range(self.num_inputs):
            self.x.append(
                np.random.random(size=self.input_dim).astype(self.dtype))

        tmp = []
        x_names = self.get_x_names()
        for i in range(self.num_inputs):
            tmp.append((x_names[i], self.x[i]))

        self.inputs = {'X': tmp}
        self.outputs = {'Y': np.stack(self.x, axis=self.axis)}
        self.attrs = {'axis': self.axis}

    def test_check_output(self):
        if paddle.is_compiled_with_xpu():
            paddle.enable_static()
            place = paddle.XPUPlace(0)
            self.check_output_with_place(place)

    def test_check_grad(self):
        if self.dtype == 'int64' or self.dtype == 'int32':
            pass
        else:
            if paddle.is_compiled_with_xpu():
                paddle.enable_static()
                place = paddle.XPUPlace(0)
                self.check_grad_with_place(place, self.get_x_names(), 'Y')


class TestStackOp1(TestStackOpBase):
    def initParameters(self):
        self.num_inputs = 16


class TestStackOp2(TestStackOpBase):
    def initParameters(self):
        self.num_inputs = 30


class TestStackOp3(TestStackOpBase):
    def initParameters(self):
        self.axis = -1

    def test_check_grad(self):
        pass


class TestStackOp4(TestStackOpBase):
    def initParameters(self):
        self.axis = -4

    def test_check_grad(self):
        pass


class TestStackOp5(TestStackOpBase):
    def initParameters(self):
        self.axis = 1


class TestStackOp6(TestStackOpBase):
    def initParameters(self):
        self.axis = 3


class TestStackOpint64(TestStackOpBase):
    def initDefaultParameters(self):
        self.num_inputs = 4
        self.input_dim = (5, 6, 7)
        self.axis = 0
        self.dtype = 'int64'

    def initParameters(self):
        self.num_inputs = 16


class TestStackOpint(TestStackOpBase):
    def initDefaultParameters(self):
        self.num_inputs = 4
        self.input_dim = (5, 6, 7)
        self.axis = 0
        self.dtype = 'int32'

    def initParameters(self):
        self.num_inputs = 16


if __name__ == '__main__':
    unittest.main()
