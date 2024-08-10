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
from op_test import OpTest
from test_sparse_attention_op import get_cuda_version

import paddle
import paddle.nn.functional as F
from paddle import nn
from paddle.incubate.nn.functional import moe, swiglu
from paddle.nn.layer.common import Linear

paddle.seed(42)


class Expert(nn.Layer):
    def __init__(self, d_model, d_feedforward):
        super().__init__()
        self.fc1 = nn.Linear(
            d_model, d_feedforward * 2
        )  # Swiglu expects twice the hidden_dim
        self.swiglu = swiglu
        self.fc2 = nn.Linear(d_feedforward, d_model)

    def forward(self, x):
        x = self.fc1(x)
        x = self.swiglu(x)
        x = self.fc2(x)
        return x


@unittest.skipIf(
    not paddle.is_compiled_with_cuda()
    or get_cuda_version() < 11030
    or paddle.device.cuda.get_device_capability()[0] < 8,
    "MoE requires CUDA >= 11.2 and CUDA_ARCH >= 8",
)
class TestMoEOp(OpTest):
    def setUp(self):
        self.config()
        self.rtol = 1e-2
        self.atol = 1e-2

        paddle.set_default_dtype(self.x_type)
        self.__class__.op_type = "moe"
        self.__class__.no_need_check_grad = True

        self.experts = nn.LayerList(
            [
                Expert(self.d_model, self.d_feedforward)
                for _ in range(self.num_expert)
            ]
        )

        self.bmm_w0 = paddle.to_tensor(
            np.array([expert.fc1.weight.numpy() for expert in self.experts]),
            dtype=self.x_type,
        )
        self.bmm_b0 = paddle.to_tensor(
            np.array(
                [expert.fc1.bias.numpy() for expert in self.experts]
            ).reshape(self.num_expert, 1, -1),
            dtype=self.x_type,
        )

        self.bmm_w1 = paddle.to_tensor(
            np.array([expert.fc2.weight.numpy() for expert in self.experts]),
            dtype=self.x_type,
        )
        self.bmm_b1 = paddle.to_tensor(
            np.array(
                [expert.fc2.bias.numpy() for expert in self.experts]
            ).reshape(self.num_expert, 1, -1),
            dtype=self.x_type,
        )

        self.tensor_x = paddle.to_tensor(
            np.random.randn(self.batch_size, self.seq_len, self.d_model) * 0.1,
            dtype=self.x_type,
        )

        self.bmm_w0.stop_gradient = True
        self.bmm_b0.stop_gradient = True
        self.bmm_w1.stop_gradient = True
        self.bmm_b1.stop_gradient = True
        self.tensor_x.stop_gradient = True

        self.gate = Linear(self.d_model, self.num_expert)
        self.gate_weight = self.gate.weight.cast(paddle.float32)
        self.gate_weight.stop_gradient = True

        paddle.set_default_dtype(self.x_type)
        self.activation = swiglu

    def config(self):
        self.x_type = np.float16
        self.batch_size = 10
        self.seq_len = 128
        self.num_expert = 32
        self.d_model = 768
        self.d_feedforward = 3072
        self.top_k = 2
        self.quant_method = "None"

    def GetMoEOut(self, tensor_x):
        paddle.disable_static(place=paddle.CUDAPlace(0))
        moe_out = moe(
            tensor_x,
            self.gate_weight,
            self.bmm_w0,
            self.bmm_b0,
            self.bmm_w1,
            self.bmm_b1,
            None if self.quant_method == "None" else self.scale0,
            None if self.quant_method == "None" else self.scale1,
            self.quant_method,
            2,
        )
        return moe_out

    def GetBaselineOut(self, hidden_states):
        paddle.disable_static(place=paddle.CUDAPlace(0))
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = paddle.reshape(hidden_states, [-1, hidden_dim])
        router_logits = self.gate(hidden_states).cast(paddle.float32)

        routing_weights = F.softmax(router_logits, axis=-1, dtype='float32')
        routing_weights, selected_experts = paddle.topk(
            routing_weights, self.top_k, axis=-1
        )
        routing_weights /= paddle.sum(routing_weights, axis=-1, keepdim=True)
        routing_weights = routing_weights.cast(np.float32)

        final_hidden_states = paddle.zeros_like(hidden_states)

        expert_mask = paddle.transpose(
            F.one_hot(selected_experts, num_classes=self.num_expert), [2, 1, 0]
        )

        for expert_idx in range(self.num_expert):
            expert_layer = self.experts[expert_idx]
            idx, top_x = paddle.where(expert_mask[expert_idx])

            current_state = paddle.index_select(
                hidden_states, top_x, axis=0
            ).reshape([-1, hidden_dim])
            current_hidden_states = (
                expert_layer(current_state) * routing_weights[top_x, idx]
            )
            paddle.index_add_(
                x=final_hidden_states,
                index=top_x.squeeze(),
                axis=0,
                value=current_hidden_states.to(hidden_states.dtype),
            )

        final_hidden_states = paddle.reshape(
            final_hidden_states, [batch_size, sequence_length, hidden_dim]
        )
        return final_hidden_states

    def test_moe_op(self):
        ref_out = self.GetBaselineOut(self.tensor_x).cast(np.float32)
        moe_out = self.GetMoEOut(self.tensor_x).cast(np.float32)
        np.testing.assert_allclose(
            ref_out, moe_out, rtol=self.rtol, atol=self.atol
        )


class TestMoEOpBf16(TestMoEOp):
    def config(self):
        super().config()
        self.x_type = paddle.bfloat16
        self.rtol = 1e-2
        self.atol = 1e-2


if __name__ == "__main__":
    unittest.main()
