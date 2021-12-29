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
import sys
sys.path.append("..")
from op_test import OpTest
from op_test_xpu import XPUOpTest
import paddle
import paddle.fluid as fluid
from paddle.fluid import compiler, Program, program_guard

paddle.enable_static()


def huber_loss_forward(val, delta):
    abs_val = abs(val)
    if abs_val <= delta:
        return 0.5 * val * val
    else:
        return delta * (abs_val - 0.5 * delta)


class TestHuberLossOp(XPUOpTest):
    def setUp(self):
        self.set_xpu()
        self.op_type = 'huber_loss'
        self.place = paddle.XPUPlace(0)

        self.init_dtype()

        self.set_inputs()
        self.set_attrs()
        self.set_outputs()

    def set_inputs(self):
        shape = self.set_shape()
        x = np.random.uniform(0, 1., shape).astype(self.dtype)
        y = np.random.uniform(0, 1., shape).astype(self.dtype)
        self.inputs = {
            'X': OpTest.np_dtype_to_fluid_dtype(x),
            'Y': OpTest.np_dtype_to_fluid_dtype(y)
        }

    def set_attrs(self):
        self.attrs = {'delta': 0.5}

    def set_outputs(self):
        delta = self.attrs['delta']
        shape = self.set_shape()
        residual = self.inputs['Y'] - self.inputs['X']
        loss = np.vectorize(huber_loss_forward)(residual,
                                                delta).astype(self.dtype)
        self.outputs = {'Residual': residual, 'Out': loss.reshape(shape)}

    def set_shape(self):
        return (100, 1)

    def set_xpu(self):
        self.__class__.use_xpu = True

    def init_dtype(self):
        self.dtype = np.float32

    def test_check_output(self):
        self.check_output_with_place(self.place)

    def test_check_grad_normal(self):
        self.check_grad_with_place(self.place, ['X', 'Y'], 'Out')

    def test_check_grad_ingore_x(self):
        self.check_grad_with_place(
            self.place, ['Y'], 'Out', no_grad_set=set("residual"))

    def test_check_grad_ingore_y(self):
        self.check_grad_with_place(
            self.place, ['X'], 'Out', no_grad_set=set('residual'))


def TestHuberLossOp1(TestHuberLossOp):
    def set_shape(self):
        return (64)


def TestHuberLossOp2(TestHuberLossOp):
    def set_shape(self):
        return (6, 6)


def TestHuberLossOp3(TestHuberLossOp):
    def set_shape(self):
        return (6, 6, 1)


if __name__ == '__main__':
    unittest.main()
