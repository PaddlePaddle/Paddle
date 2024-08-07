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
from op_test import OpTest, convert_float_to_uint16

import paddle
from paddle.base import core


def huber_loss_forward(val, delta):
    abs_val = abs(val)
    if abs_val <= delta:
        return 0.5 * val * val
    else:
        return delta * (abs_val - 0.5 * delta)


def huber_loss_wrapper(x, y, delta):
    a = paddle._C_ops.huber_loss(x, y, delta)
    return a


class TestHuberLossOp(OpTest):
    def setUp(self):
        self.op_type = 'huber_loss'
        self.prim_op_type = "comp"
        self.python_out_sig = ["Out"]
        self.python_api = huber_loss_wrapper
        self.public_python_api = huber_loss_wrapper

        self.delta = 1.0
        self.init_dtype()
        self.init_input()
        shape = self.set_shape()
        residual = self.inputs['Y'] - self.inputs['X']
        loss = np.vectorize(huber_loss_forward)(residual, self.delta).astype(
            self.dtype
        )
        self.attrs = {'delta': self.delta}
        self.outputs = {'Residual': residual, 'Out': loss.reshape(shape)}

    def init_dtype(self):
        self.dtype = np.float32

    def init_input(self):
        shape = self.set_shape()
        self.inputs = {
            'X': np.random.uniform(0, 1.0, shape).astype(self.dtype),
            'Y': np.random.uniform(0, 1.0, shape).astype(self.dtype),
        }

    def set_shape(self):
        return (100, 1)

    def test_check_output(self):
        self.check_output(check_prim_pir=True)

    def test_check_grad_normal(self):
        self.check_grad(['X', 'Y'], 'Out')

    def test_check_grad_ignore_x(self):
        self.check_grad(['Y'], 'Out', no_grad_set=set("residual"))

    def test_check_grad_ignore_y(self):
        self.check_grad(['X'], 'Out', no_grad_set=set('residual'))


def TestHuberLossOp1(TestHuberLossOp):
    def set_shape(self):
        return 64


def TestHuberLossOp2(TestHuberLossOp):
    def set_shape(self):
        return (6, 6)


def TestHuberLossOp3(TestHuberLossOp):
    def set_shape(self):
        return (6, 6, 1)


class TestHuberLossFP16Op(TestHuberLossOp):
    def init_dtype(self):
        self.dtype = np.float16


@unittest.skipIf(
    not core.is_compiled_with_cuda()
    or not core.is_bfloat16_supported(core.CUDAPlace(0)),
    "core is not compiled with CUDA or not support bfloat16",
)
class TestHuberLossBF16Op(OpTest):
    def setUp(self):
        self.op_type = 'huber_loss'
        self.prim_op_type = "comp"
        self.python_out_sig = ["Out"]
        self.python_api = huber_loss_wrapper
        self.public_python_api = huber_loss_wrapper

        self.delta = 1.0
        self.init_dtype()
        self.init_input()
        shape = self.set_shape()
        residual = self.inputs['Y'] - self.inputs['X']
        loss = np.vectorize(huber_loss_forward)(residual, self.delta).astype(
            self.np_dtype
        )
        self.attrs = {'delta': self.delta}
        self.outputs = {'Residual': residual, 'Out': loss.reshape(shape)}

        self.place = core.CUDAPlace(0)
        self.inputs['X'] = convert_float_to_uint16(self.inputs['X'])
        self.inputs['Y'] = convert_float_to_uint16(self.inputs['Y'])
        self.outputs['Residual'] = convert_float_to_uint16(
            self.outputs['Residual']
        )
        self.outputs['Out'] = convert_float_to_uint16(self.outputs['Out'])

    def init_dtype(self):
        self.dtype = np.uint16
        self.np_dtype = np.float32

    def init_input(self):
        shape = self.set_shape()
        self.inputs = {
            'X': np.random.uniform(0, 1.0, shape).astype(self.np_dtype),
            'Y': np.random.uniform(0, 1.0, shape).astype(self.np_dtype),
        }

    def set_shape(self):
        return (100, 1)

    def test_check_output(self):
        self.check_output_with_place(self.place, check_prim_pir=True)

    def test_check_grad_normal(self):
        self.check_grad_with_place(self.place, ['X', 'Y'], 'Out')

    def test_check_grad_ignore_x(self):
        self.check_grad_with_place(
            self.place, ['Y'], 'Out', no_grad_set=set("residual")
        )

    def test_check_grad_ignore_y(self):
        self.check_grad_with_place(
            self.place, ['X'], 'Out', no_grad_set=set('residual')
        )


if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()
