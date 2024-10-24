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


class MultiAddNet(nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        shape = paddle.shape(x)
        mask = paddle.full(shape, 0, dtype="bool")

        x1 = paddle.full([1], 0, dtype="float64")
        x2 = paddle.full([1], -65504, dtype="float64")
        x3 = paddle.full([1], 0, dtype="float64")
        x4 = paddle.full([1], 0, dtype="float64")

        y = mask.cast("float64")
        z = x.cast("float64")

        s0 = x3 + x4
        s1 = s0 + y
        s2 = x1 + s1
        s3 = x2 + s1
        s4 = (z + s1).cast("bool")

        return s2, s3, s4


class TestMultiAdd(unittest.TestCase):
    def setUp(self):
        paddle.seed(2024)
        self.prepare_data()

    def prepare_data(self):
        self.x = paddle.randint(0, 1, [64, 1, 32, 128], dtype="int64").astype(
            "bool"
        )
        self.x.stop_gradient = False

    def check_jit_kernel_info(self, static_fn):
        utils.check_jit_kernel_number(static_fn, 1)
        utils.check_jit_kernel_structure(static_fn, {utils.JIT_KERNEL_NAME: 1})

    def eval(self, use_cinn):
        net = MultiAddNet()
        input_spec = [InputSpec(shape=[None, 1, None, None], dtype="bool")]
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


if __name__ == '__main__':
    unittest.main()
