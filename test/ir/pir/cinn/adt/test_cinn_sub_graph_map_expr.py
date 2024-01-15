# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import paddle
from paddle import nn


def apply_to_static(net, use_cinn):
    build_strategy = paddle.static.BuildStrategy()
    build_strategy.build_cinn_pass = use_cinn
    return paddle.jit.to_static(
        net, build_strategy=build_strategy, full_graph=True
    )


def exp_sub(x):
    y = paddle.exp(x)
    z = y - x
    return z


class CINNSubGraphNet(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.fn = exp_sub

    def forward(self, x):
        out = self.fn(x)
        return out


class TestCinnSubGraphBase(unittest.TestCase):
    """
    Test Pir API + @to_static + CINN.
    """

    def setUp(self):
        paddle.seed(2022)
        self.shape = [64, 128]
        self.axis = -1
        self.prepare_data()

    def prepare_data(self):
        self.x = paddle.randn(self.shape, dtype="float32")
        self.x.stop_gradient = False

    def train(self, use_cinn):
        paddle.seed(2022)
        net = CINNSubGraphNet()
        net = apply_to_static(net, use_cinn)
        net.eval()
        out = net(self.x)
        return out

    def test_forward(self):
        cinn_out = self.train(use_cinn=True)
        dy_out = self.train(use_cinn=False)
        np.testing.assert_allclose(cinn_out.numpy(), dy_out.numpy(), atol=1e-8)


class LlamaRMSNorm(nn.Layer):
    def __init__(self):
        super().__init__()
        self.hidden_size = 768
        self.weight = paddle.create_parameter(
            shape=[self.hidden_size],
            dtype=paddle.get_default_dtype(),
            default_initializer=nn.initializer.Constant(1.0),
        )
        self.variance_epsilon = 1e-6

    def forward(self, hidden_states):
        variance = hidden_states.sum(-1, keepdim=True)
        return variance * hidden_states


class TestLlamaRMSNorm(TestCinnSubGraphBase):
    def prepare_data(self):
        self.shape = [1, 2048, 768]
        self.hidden_states = paddle.randn(self.shape, dtype="float32")
        self.hidden_states.stop_gradient = False

    def eval(self, use_cinn):
        paddle.seed(2022)
        net = LlamaRMSNorm()
        net = apply_to_static(net, use_cinn)
        net.eval()
        out = net(self.hidden_states)
        return out

    def test_eval(self):
        cinn_out = self.eval(use_cinn=True)
        dy_out = self.eval(use_cinn=False)
        np.testing.assert_allclose(
            cinn_out.numpy(), dy_out.numpy(), atol=1e-4, rtol=1e-4
        )


if __name__ == '__main__':
    unittest.main()
