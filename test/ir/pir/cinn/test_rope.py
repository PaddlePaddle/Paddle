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

from test_cinn_sub_graph import TestCinnSubGraphBase, apply_to_static

import paddle
from paddle import nn


class RotaryPosEmb(nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, q, k, cos, sin, position_ids):
        cos = cos.squeeze(axis=[0, 2])  # [seq_len, dim]
        sin = sin.squeeze(axis=[0, 2])  # [seq_len, dim]

        cos = cos[position_ids].unsqueeze(2)  # [bs, seq_len, 1, dim]
        sin = sin[position_ids].unsqueeze(2)  # [bs, seq_len, 1, dim]
        q_embed = (q * cos) + (self.rotate_half(q) * sin)
        k_embed = (k * cos) + (self.rotate_half(k) * sin)
        return q_embed, k_embed

    def rotate_half(self, x):
        """Rotates half the hidden dims of the input."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return paddle.concat([-x2, x1], axis=-1)  # shape is the same as x


class TestRotaryPosEmb(TestCinnSubGraphBase):
    def prepare_data(self):
        self.q = paddle.randn([1, 2048, 8, 96], dtype="float32")
        self.q.stop_gradient = False

        self.k = paddle.randn([1, 2048, 8, 96], dtype="float32")
        self.k.stop_gradient = False

        self.cos = paddle.randn([1, 2048, 1, 96], dtype="float32")
        self.cos.stop_gradient = False

        self.sin = paddle.randn([1, 2048, 1, 96], dtype="float32")
        self.sin.stop_gradient = False

        self.position_ids = paddle.arange(end=2048, dtype="int64").unsqueeze(0)
        self.position_ids.stop_gradient = False

    def eval(self, use_cinn):
        paddle.seed(2022)
        net = RotaryPosEmb()
        net.eval()
        if use_cinn:
            net = apply_to_static(net, use_cinn)

        out = net(self.q, self.k, self.cos, self.sin, self.position_ids)
        return out

    def test_eval(self):
        cinn_outs = self.eval(use_cinn=True)
        # dy_outs = self.eval(use_cinn=False)

        # TODO(phlrain): Need to check result
        # for cinn_out, dy_out in zip(cinn_outs, dy_outs):
        #     np.testing.assert_allclose(
        #         cinn_out.numpy(), dy_out.numpy(), atol=1e-8
        #     )


if __name__ == '__main__':
    unittest.main()
