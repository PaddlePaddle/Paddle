# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

import sys
import unittest
from os.path import dirname

import numpy as np

import paddle
from paddle import nn
from paddle.static import InputSpec

sys.path.append(dirname(dirname(__file__)))

import utils


class TransposeReshapeNet(nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        y_shape = paddle.shape(y)
        s0 = y_shape[0]
        s1 = y_shape[1]
        s2 = 4096
        y = paddle.transpose(x, [0, 2, 1, 3])
        out = paddle.reshape(y, [s0, s1, s2])

        return out


class TestTransposeReshape(unittest.TestCase):
    def setUp(self):
        paddle.seed(2024)
        self.prepare_data()

    def prepare_data(self):
        self.x = paddle.randn([4, 32, 128, 128], dtype="float16")
        self.y = paddle.randn([4, 128, 32, 128], dtype="float16")

    def check_jit_kernel_info(self, static_fn):
        utils.check_jit_kernel_number(static_fn, 1)
        utils.check_jit_kernel_structure(static_fn, {utils.JIT_KERNEL_NAME: 1})

    def eval(self, use_cinn):
        net = TransposeReshapeNet()
        input_spec = [
            InputSpec(shape=[None, 32, None, None], dtype="float16"),
            InputSpec(shape=[None, None, 32, 128], dtype="float16"),
        ]
        net = utils.apply_to_static(net, use_cinn, input_spec)
        net.eval()
        out = net(self.x, self.y)
        if use_cinn:
            self.check_jit_kernel_info(net.forward)
        return out

    def test_eval(self):
        dy_out = self.eval(use_cinn=False)
        if utils.unittest_use_cinn():
            cinn_out = self.eval(use_cinn=True)
            np.testing.assert_allclose(
                cinn_out.numpy(), dy_out.numpy(), atol=1e-6, rtol=1e-6
            )


class ReshapeTransposeNet(nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        y = paddle.reshape(x, [0, 0, 32, 128])
        out = paddle.transpose(y, [0, 2, 1, 3])

        return out


class TestReshapeTranspose(unittest.TestCase):
    def setUp(self):
        paddle.seed(2024)
        self.prepare_data()

    def prepare_data(self):
        self.x = paddle.randn([4, 16, 4096], dtype="float16")

    def check_jit_kernel_info(self, static_fn):
        utils.check_jit_kernel_number(static_fn, 1)
        utils.check_jit_kernel_structure(static_fn, {utils.JIT_KERNEL_NAME: 1})

    def eval(self, use_cinn):
        net = ReshapeTransposeNet()
        input_spec = [
            InputSpec(shape=[None, None, 4096], dtype="float16"),
        ]
        net = utils.apply_to_static(net, use_cinn, input_spec)
        net.eval()
        out = net(self.x)
        if use_cinn:
            self.check_jit_kernel_info(net.forward)
        return out

    def test_eval(self):
        dy_out = self.eval(use_cinn=False)
        if utils.unittest_use_cinn():
            cinn_out = self.eval(use_cinn=True)
            np.testing.assert_allclose(
                cinn_out.numpy(), dy_out.numpy(), atol=1e-6, rtol=1e-6
            )


if __name__ == '__main__':
    unittest.main()
