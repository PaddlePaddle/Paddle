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


class LlamaRMSNorm(nn.Layer):
    def __init__(self):
        super().__init__()
        self.hidden_size = 4096
        # self.weight = paddle.create_parameter(
        #     shape=[self.hidden_size],
        #     dtype=paddle.float16,
        #     default_initializer=nn.initializer.Constant(0.2),
        # )
        self.variance_epsilon = 1e-6

    def forward(self, hidden_states, weight):
        hidden_states = hidden_states.astype("float32")
        variance = (hidden_states * hidden_states).sum(
            -1, keepdim=True
        ) / self.hidden_size
        hidden_states = (
            paddle.rsqrt(variance + self.variance_epsilon) * hidden_states
        )

        if weight.dtype in [paddle.float16, paddle.bfloat16]:
            hidden_states = paddle.cast(hidden_states, weight.dtype)
        return hidden_states * weight


class TestLlamaRMSNorm(unittest.TestCase):
    def setUp(self):
        paddle.seed(2022)
        self.prepare_data()

    def prepare_data(self):
        self.shape = [1, 17, 4096]
        self.hidden_states = paddle.randn(self.shape, dtype="float16")
        self.hidden_states.stop_gradient = True
        self.weight = paddle.randn([4096], dtype="float16")
        self.weight.stop_gradient = True

    def check_jit_kernel_info(self, static_fn):
        utils.check_jit_kernel_number(static_fn, 1)
        utils.check_jit_kernel_structure(static_fn, {utils.JIT_KERNEL_NAME: 1})

    def eval(self, use_cinn):
        net = LlamaRMSNorm()
        input_spec = [
            InputSpec(shape=[None, None, 4096], dtype='float16'),
            InputSpec(shape=[4096], dtype='float16'),
        ]

        net = utils.apply_to_static(net, use_cinn, input_spec)
        net.eval()
        out = net(self.hidden_states, self.weight)
        # if use_cinn:
        #     self.check_jit_kernel_info(net.forward)
        return out

    def test_eval(self):
        cinn_out = self.eval(use_cinn=True)
        dy_out = self.eval(use_cinn=False)
        np.testing.assert_allclose(
            cinn_out.numpy(), dy_out.numpy(), atol=1e-6, rtol=1e-6
        )


if __name__ == '__main__':
    unittest.main()
