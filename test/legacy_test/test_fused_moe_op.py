# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
from op_test import OpTest

import paddle
import paddle.nn.functional as F
from paddle import nn
from paddle.incubate.nn.functional import fused_moe, swiglu
from paddle.nn.layer.common import Linear

paddle.seed(42)


class Expert(nn.Layer):
    def __init__(self, d_model, d_feedforward):
        super().__init__()
        self.fc1 = nn.Linear(
            d_model, d_feedforward
        )  # Swiglu expects twice the hidden_dim
        self.swiglu = swiglu
        self.fc2 = nn.Linear(d_feedforward // 2, d_model)

    def forward(self, x):
        x = self.fc1(x)
        x = swiglu(x)
        x = self.fc2(x)
        return x


class TestFusedMoEOp(OpTest):
    def setUp(self):
        self.config()
        self.rtol = 1e-3
        self.atol = 1e-3

        paddle.set_default_dtype(self.x_type)
        self.__class__.op_type = "fused_moe"
        # Since it's only used in inference.
        self.__class__.no_need_check_grad = True

        self.experts = nn.LayerList(
            [
                Expert(self.d_model, self.d_feedforward)
                for _ in range(self.num_expert)
            ]
        )

        self.bmm_w0 = paddle.to_tensor(
            np.array([expert.fc1.weight.numpy() for expert in self.experts]),
            dtype=paddle.float16,
        )
        self.bmm_b0 = paddle.to_tensor(
            np.array(
                [expert.fc1.bias.numpy() for expert in self.experts]
            ).reshape(self.num_expert, 1, -1),
            dtype=paddle.float16,
        )

        # d_model//2 for swiglu
        self.bmm_w1 = paddle.to_tensor(
            np.array([expert.fc2.weight.numpy() for expert in self.experts]),
            dtype=paddle.float16,
        )
        self.bmm_b1 = paddle.to_tensor(
            np.array(
                [expert.fc2.bias.numpy() for expert in self.experts]
            ).reshape(self.num_expert, 1, -1),
            dtype=paddle.float16,
        )

        self.tensor_x = paddle.to_tensor(
            np.random.randn(self.batch_size, self.seq_len, self.d_model)
            * 0.001,
            dtype=paddle.float16,
        )

        self.bmm_w0.stop_gradient = True
        self.bmm_b0.stop_gradient = True
        self.bmm_w1.stop_gradient = True
        self.bmm_b1.stop_gradient = True
        self.tensor_x.stop_gradient = True

        self.gate = Linear(self.d_model, self.num_expert)

        self.gate_weight = paddle.to_tensor(
            self.gate.weight.numpy(),
            dtype=paddle.float32,
        )
        self.gate_weight.stop_gradient = True

        paddle.set_default_dtype("float16")
        self.activation = swiglu

    def config(self):
        self.x_type = np.float16
        self.batch_size = 10
        self.seq_len = 128
        self.num_expert = 32
        self.d_model = 768
        self.d_feedforward = 3072
        self.top_k = 2

    def GetBaselineOut(self, tensor_x, gate_logits):
        def expert_choice_gating(logits, capacity, batch_idx, expert_idx):
            gates = F.softmax(logits, -1)
            indices1_s = paddle.topk(
                logits.transpose([0, 2, 1]), k=capacity, axis=-1
            )[1].cast("int32")
            seqlen_idx = indices1_s.reshape([-1])
            gather_idx = paddle.stack([batch_idx, seqlen_idx, expert_idx], -1)
            return expert_idx, gather_idx

        paddle.disable_static()
        capacity = 2
        batch_expert_idx = paddle.nonzero(
            paddle.ones(shape=[self.batch_size, self.num_expert, capacity])
        ).cast('int32')
        batch_idx = batch_expert_idx[:, 0]
        expert_idx = batch_expert_idx[:, 1]

        (expert_idx_flatten, gather_idx) = expert_choice_gating(
            gate_logits, capacity, batch_idx, expert_idx
        )

        outputs = paddle.zeros_like(tensor_x)

        batch_idx = gather_idx[:, :2]
        selected_token = tensor_x.gather_nd(batch_idx)

        batch_selected_token = selected_token.reshape(
            [self.batch_size, self.num_expert, -1, tensor_x.shape[-1]]
        )
        batch_selected_token = batch_selected_token.transpose(
            [1, 0, 2, 3]
        ).reshape([self.num_expert, -1, tensor_x.shape[-1]])

        output = paddle.bmm(batch_selected_token, self.bmm_w0) + self.bmm_b0
        output = self.activation(output)
        output = paddle.bmm(output, self.bmm_w1) + self.bmm_b1

        output = output.transpose([1, 0, 2]).reshape(
            [self.batch_size, -1, self.num_expert, tensor_x.shape[-1]]
        )
        output = output.transpose([0, 2, 1, 3])
        output = output.reshape([-1, tensor_x.shape[-1]])

        outputs = outputs.scatter_nd_add(batch_idx, output)
        return outputs

    def GetFusedMoeOut(self, tensor_x):
        paddle.disable_static()
        fused_out = fused_moe(
            tensor_x,
            self.gate_weight,
            self.bmm_w0,
            self.bmm_b0,
            self.bmm_w1,
            self.bmm_b1,
            "",
            2,
        )

        return fused_out

    def test_fused_moe_op(self):
        gate_logits = self.gate(self.tensor_x)
        ref_out = self.GetBaselineOut(self.tensor_x, gate_logits)
        fused_moe_out = self.GetFusedMoeOut(self.tensor_x)

        np.testing.assert_allclose(
            ref_out, fused_moe_out, rtol=self.rtol, atol=self.atol
        )

    def GetBaselineOut2(self, x):
        batch_size, seq_len, _ = x.shape
        gate_outputs = self.gate(x)
        topk_vals, topk_indices = paddle.topk(gate_outputs, self.top_k, axis=-1)

        expert_outputs = []
        for idx in range(self.num_expert):
            mask = (topk_indices == idx).astype(x.dtype)
            mask_combined = (mask.sum(axis=-1, keepdim=True) > 0).astype(
                mask.dtype
            )

            expert_output = self.experts[idx](x * mask_combined)
            expert_outputs.append(expert_output * mask_combined)

        combined_output = paddle.add_n(expert_outputs)
        return combined_output

    def test_fused_moe_op_new(self):
        ref_out = self.GetBaselineOut2(self.tensor_x)
        fused_moe_out = self.GetFusedMoeOut(self.tensor_x)

        np.testing.assert_allclose(
            ref_out, fused_moe_out, rtol=self.rtol, atol=self.atol
        )


class TestFusedMoEOpActGeluFp16(TestFusedMoEOp):
    def config(self):
        super().config()
        self.x_type = np.float16


if __name__ == "__main__":
    unittest.main()
