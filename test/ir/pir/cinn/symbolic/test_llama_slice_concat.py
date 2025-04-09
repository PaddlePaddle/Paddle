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


class SliceMultiConcatNet(nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x0 = paddle.shape(x)[0].reshape([1])
        x1 = paddle.full([1], 1, dtype="int64")
        out0 = paddle.concat([x0, x1])

        y = paddle.full([1], 1, dtype="int64")
        out1 = paddle.concat([x0, y])
        return out0, out1


class TestSliceMultiConcat(unittest.TestCase):
    def setUp(self):
        paddle.seed(2024)
        self.prepare_data()

    def prepare_data(self):
        self.shape = [64, 128]
        self.x = paddle.randint(0, 100, self.shape, dtype="int64")
        self.x.stop_gradient = False

    def check_jit_kernel_info(self, static_fn):
        utils.check_jit_kernel_number(static_fn, 1)
        utils.check_jit_kernel_structure(static_fn, {utils.JIT_KERNEL_NAME: 1})

    def eval(self, use_cinn):
        net = SliceMultiConcatNet()
        input_spec = [
            InputSpec(shape=[None, None], dtype="int64"),
        ]
        net = utils.apply_to_static(net, use_cinn, input_spec)
        net.eval()
        out = net(self.x)
        if use_cinn:
            self.check_jit_kernel_info(net.forward)
        return out

    def test_eval(self):
        dy_outs = self.eval(use_cinn=False)
        if utils.unittest_use_cinn():
            cinn_outs = self.eval(use_cinn=True)
            for dy_out, cinn_out in zip(dy_outs, cinn_outs):
                np.testing.assert_allclose(
                    cinn_out.numpy(), dy_out.numpy(), atol=1e-6, rtol=1e-6
                )


class SliceConcatNet(nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x0 = paddle.shape(x)[0].reshape([1])
        x1 = paddle.full([1], 1, dtype="int64")
        out = paddle.concat([x0, x1])
        return out


class TestSliceConcat(unittest.TestCase):
    def setUp(self):
        paddle.seed(2024)
        self.prepare_data()

    def prepare_data(self):
        self.x = paddle.randn([1, 32000], dtype="float16")
        self.x.stop_gradient = False

    def check_jit_kernel_info(self, static_fn):
        utils.check_jit_kernel_number(static_fn, 1)
        utils.check_jit_kernel_structure(static_fn, {utils.JIT_KERNEL_NAME: 1})

    def eval(self, use_cinn):
        net = SliceConcatNet()
        input_spec = [
            InputSpec(shape=[None, 32000], dtype="float16"),
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
