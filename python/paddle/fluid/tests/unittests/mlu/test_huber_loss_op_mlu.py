#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
from op_test import OpTest
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


class TestHuberLossOp(OpTest):
    def setUp(self):
        self.op_type = 'huber_loss'
        self.set_mlu()
        self.python_out_sig = ["Out"]
        self.delta = 1.0
        self.init_input()
        shape = self.set_shape()
        residual = self.inputs['Y'] - self.inputs['X']
        loss = np.vectorize(huber_loss_forward)(residual, self.delta).astype(
            'float32'
        )
        self.attrs = {'delta': self.delta}
        self.outputs = {'Residual': residual, 'Out': loss.reshape(shape)}

    def init_input(self):
        shape = self.set_shape()
        self.inputs = {
            'X': np.random.uniform(0, 1.0, shape).astype('float32'),
            'Y': np.random.uniform(0, 1.0, shape).astype('float32'),
        }

    def set_mlu(self):
        self.__class__.use_mlu = True
        self.place = paddle.MLUPlace(0)

    def set_shape(self):
        return (100, 1)

    def test_check_output(self):
        self.check_output_with_place(self.place, atol=1e-3)

    def test_check_grad_normal(self):
        self.check_grad_with_place(self.place, ['X', 'Y'], 'Out')

    def test_check_grad_ingore_x(self):
        self.check_grad_with_place(
            self.place,
            ['Y'],
            'Out',
            max_relative_error=0.008,
            no_grad_set=set("residual"),
        )

    def test_check_grad_ingore_y(self):
        self.check_grad_with_place(
            self.place,
            ['X'],
            'Out',
            max_relative_error=0.008,
            no_grad_set=set('residual'),
        )


def TestHuberLossOp1(TestHuberLossOp):
    def set_shape(self):
        return 64


def TestHuberLossOp2(TestHuberLossOp):
    def set_shape(self):
        return (6, 6)


def TestHuberLossOp3(TestHuberLossOp):
    def set_shape(self):
        return (6, 6, 1)


if __name__ == '__main__':
    unittest.main()