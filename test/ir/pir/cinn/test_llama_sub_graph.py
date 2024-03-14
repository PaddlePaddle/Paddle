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
import utils
from test_cinn_sub_graph import TestCinnSubGraphBase

import paddle
from paddle import nn


class LlamaRMSNorm(nn.Layer):
    def __init__(self):
        super().__init__()
        self.hidden_size = 768
        self.weight = paddle.create_parameter(
            shape=[self.hidden_size],
            dtype="float32",
            default_initializer=nn.initializer.Constant(1.0),
        )
        self.variance_epsilon = 1e-6

    def forward(self, hidden_states):
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = (
            paddle.rsqrt(variance + self.variance_epsilon) * hidden_states
        )

        return hidden_states * self.weight


class TestLlamaRMSNorm(TestCinnSubGraphBase):
    def prepare_data(self):
        self.shape = [2, 2048, 768]
        self.hidden_states = paddle.randn(self.shape, dtype="float32")
        self.hidden_states.stop_gradient = False

    def eval(self, use_cinn):
        paddle.seed(2022)
        self.prepare_data()
        net = LlamaRMSNorm()
        net = utils.apply_to_static(net, use_cinn)

        net.eval()
        out = net(self.hidden_states)

        loss = out.sum()
        loss.backward()

        return out, net.weight.gradient(), self.hidden_states.gradient()

    def test_eval(self):
        cinn_out, cinn_dx, cinn_dh = self.eval(use_cinn=True)
        dy_out, dy_dx, dy_dh = self.eval(use_cinn=False)
        np.testing.assert_allclose(
            cinn_out.numpy(), dy_out.numpy(), atol=1e-5, rtol=1e-5
        )

        # np.testing.assert_allclose(cinn_dx, dy_dx, atol=1e-4)
        # np.testing.assert_allclose(cinn_dh, dy_dh, atol=1e-4)


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


# class TestRotaryPosEmb(TestCinnSubGraphBase):
#     def prepare_data(self):
#         self.q = paddle.randn([1, 2048, 8, 96], dtype="float32")
#         self.q.stop_gradient = False

#         self.k = paddle.randn([1, 2048, 8, 96], dtype="float32")
#         self.k.stop_gradient = False

#         self.cos = paddle.randn([1, 2048, 1, 96], dtype="float32")
#         self.cos.stop_gradient = False

#         self.sin = paddle.randn([1, 2048, 1, 96], dtype="float32")
#         self.sin.stop_gradient = False

#         self.position_ids = paddle.arange(end=2048, dtype="int64").unsqueeze(0)
#         self.position_ids.stop_gradient = False

#     def eval(self, use_cinn):
#         paddle.seed(2022)
#         self.prepare_data()
#         net = RotaryPosEmb()

#         net = utils.apply_to_static(net, use_cinn)
#         # net.eval()
#         out = net(self.q, self.k, self.cos, self.sin, self.position_ids)
#         loss = (out[0] + out[1]).sum()
#         loss.backward()
#         return out

#     def test_eval(self):
#         cinn_outs = self.eval(use_cinn=True)
#         dy_outs = self.eval(use_cinn=False)

#         # TODO(phlrain): Need to check result
#         for cinn_out, dy_out in zip(cinn_outs, dy_outs):
#             np.testing.assert_allclose(
#                 cinn_out.numpy(), dy_out.numpy(), atol=1e-8
#             )


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


# class TestRepeatKV(TestCinnSubGraphBase):
#     def prepare_data(self):
#         self.shape = [1, 2048, 8, 96]
#         self.hidden_states = paddle.randn(self.shape, dtype="float32")
#         self.hidden_states.stop_gradient = False
#         self.n_rep = 4

#     def check_jit_kernel_info(self, static_fn):
#         utils.check_jit_kernel_number(static_fn, 2)
#         # pd_op.tile is not fused into GroupOp
#         utils.check_jit_kernel_structure(static_fn, {'jit_kernel': 2})

#     def eval(self, use_cinn):
#         paddle.seed(2022)
#         net = RepeatKV()
#         net = utils.apply_to_static(net, use_cinn)
#         net.eval()
#         out = net(self.hidden_states, self.n_rep)
#         if use_cinn:
#             self.check_jit_kernel_info(net.forward)
#         return out

#     def test_eval(self):
#         cinn_out = self.eval(use_cinn=True)
#         dy_out = self.eval(use_cinn=False)
#         np.testing.assert_allclose(
#             cinn_out.numpy(), dy_out.numpy(), atol=1e-6, rtol=1e-6
#         )


if __name__ == '__main__':
    unittest.main()
