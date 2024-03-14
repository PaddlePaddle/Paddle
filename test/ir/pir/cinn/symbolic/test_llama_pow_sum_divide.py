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


class PowSumDivideNet(nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x, y, z, w):
        s0 = paddle.shape(y)
        s1 = paddle.shape(x)[1].reshape([1])

        shape = paddle.concat([s0, s1])
        out0 = paddle.reshape(z, shape).cast("float32")

        out1 = out0.pow(2)
        out2 = out1.sum(axis=2, keepdim=True)
        factor = paddle.full([1], 4096, dtype="float32")
        out3 = out2.divide(factor)
        out4 = out3 + 1e-6
        out5 = out4.pow(-0.5)
        out6 = out5.multiply(out0).cast("float16")
        out7 = out6.multiply(w)

        return out7


class TestPowSumDivide(unittest.TestCase):
    def setUp(self):
        paddle.seed(2024)
        self.prepare_data()

    def prepare_data(self):
        self.x = paddle.randn([64, 4096], dtype="float16")
        self.y = paddle.randint(0, 100, [64, 2], dtype="int64")
        self.z = paddle.randn([64, 8192], dtype="float16")
        self.w = paddle.randn([4096], dtype="float16")

    def check_jit_kernel_info(self, static_fn):
        utils.check_jit_kernel_number(static_fn, 1)
        utils.check_jit_kernel_structure(static_fn, {utils.JIT_KERNEL_NAME: 1})

    def eval(self, use_cinn):
        net = PowSumDivideNet()
        input_spec = [
            InputSpec(shape=[None, 4096], dtype="float16"),
            InputSpec(shape=[None, None], dtype="int64"),
            InputSpec(shape=[None, 4096], dtype="float16"),
            InputSpec(shape=[4096], dtype="float16"),
        ]
        net = utils.apply_to_static(net, use_cinn, input_spec)
        net.eval()
        out = net(self.x, self.y, self.z, self.w)
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
