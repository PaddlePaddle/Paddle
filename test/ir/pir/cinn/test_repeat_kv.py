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
from test_cinn_sub_graph import TestCinnSubGraphBase, apply_to_static

import paddle
from paddle import nn


class RepeatKV(nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, hidden_states, n_rep):
        (
            batch,
            slen,
            num_key_value_heads,
            head_dim,
        ) = hidden_states.shape

        hidden_states = hidden_states.unsqueeze(-2).tile([1, 1, 1, n_rep, 1])
        return hidden_states.reshape(
            [batch, slen, num_key_value_heads * n_rep, head_dim]
        )


class TestRepeatKV(TestCinnSubGraphBase):
    def prepare_data(self):
        self.shape = [1, 2048, 8, 96]
        self.hidden_states = paddle.randn(self.shape, dtype="float32")
        self.hidden_states.stop_gradient = False

        self.n_rep = 4

    def eval(self, use_cinn):
        paddle.seed(2022)
        net = RepeatKV()
        net = apply_to_static(net, use_cinn)
        net.eval()
        out = net(self.hidden_states, self.n_rep)
        return out

    def test_eval(self):
        cinn_out = self.eval(use_cinn=True)
        dy_out = self.eval(use_cinn=False)
        np.testing.assert_allclose(
            cinn_out.numpy(), dy_out.numpy(), atol=1e-6, rtol=1e-6
        )


if __name__ == '__main__':
    unittest.main()
