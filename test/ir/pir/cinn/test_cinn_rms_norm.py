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

import unittest

import numpy as np
import utils

import paddle


class CINNRMSNormSubGraph(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.variance_epsilon = 1e-6
        self.reduce_num = 4096

    def forward(self, hidden_states, weight):
        variance = hidden_states.pow(2).sum(-1, keepdim=True) / self.reduce_num
        hidden_states = (
            paddle.rsqrt(variance + self.variance_epsilon) * hidden_states
        )
        return hidden_states * weight


class PaddleRMSNormSubGraph(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.variance_epsilon = 1e-6

    def forward(self, hidden_states, weight):
        return paddle.incubate.nn.functional.fused_rms_norm(
            x=hidden_states,
            norm_weight=weight,
            norm_bias=None,
            epsilon=self.variance_epsilon,
            begin_norm_axis=2,
        )


class TestRMSNormSubGraph(unittest.TestCase):
    def setUp(self):
        paddle.seed(2022)
        self.prepare_data()

    def prepare_data(self):
        self.shape = [1, 13, 4096]
        self.x = paddle.uniform(self.shape, dtype="float32", min=-0.5, max=0.5)
        self.x.stop_gradient = False
        self.weight = paddle.ones(shape=[self.shape[-1]], dtype="float32")
        self.weight.stop_gradient = False

    def train(self, use_cinn):
        if use_cinn:
            net = CINNRMSNormSubGraph()
        else:
            net = PaddleRMSNormSubGraph()
        net.eval()
        net = utils.apply_to_static(net, use_cinn)
        for i in range(10000):
            out = net(self.x, self.weight)
        return out

    def test_train(self):
        cinn_out = self.train(use_cinn=True)

        dy_out = self.train(use_cinn=False)
        np.testing.assert_allclose(
            cinn_out.numpy(), dy_out.numpy(), atol=1e-6, rtol=1e-6
        )


if __name__ == '__main__':
    unittest.main()
