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
from paddle.base.framework import default_main_program
from paddle.incubate.nn.functional import fused_ec_moe
from paddle.nn.layer.common import Linear

default_main_program().random_seed = 42


class TestFusedEcMoEOp(OpTest):
    def setUp(self):
        self.config()
        self.rtol = 1e-3
        self.atol = 1e-3

        paddle.set_default_dtype(self.x_type)
        self.__class__.op_type = "fused_ec_moe"
        # Since it's only used in inference.
        self.__class__.no_need_check_grad = True

        self.bmm_w0 = paddle.to_tensor(
            np.random.randn(self.num_expert, self.d_model, self.d_feedforward)
            * 0.001,
            dtype=paddle.float16,
        )
        self.bmm_b0 = paddle.to_tensor(
            np.random.randn(self.num_expert, 1, self.d_feedforward) * 0.001,
            dtype=paddle.float16,
        )
        self.bmm_w1 = paddle.to_tensor(
            np.random.randn(self.num_expert, self.d_feedforward, self.d_model)
            * 0.001,
            dtype=paddle.float16,
        )
        self.bmm_b1 = paddle.to_tensor(
            np.random.randn(self.num_expert, 1, self.d_model) * 0.001,
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

        paddle.set_default_dtype("float16")
        self.activation = getattr(F, self.act_method)

    def config(self):
        self.x_type = np.float16
        self.batch_size = 10
        self.seq_len = 128
        self.num_expert = 32
        self.d_model = 768
        self.d_feedforward = 3072
        self.act_method = 'gelu'

    def GetBaselineOut(self, tensor_x, gate_logits):
        def expert_choice_gating(logits, capacity, batch_idx, expert_idx):
            gates = F.softmax(logits, -1)
            indices1_s = paddle.topk(
                logits.transpose([0, 2, 1]), k=capacity, axis=-1
            )[1].cast("int32")
            seqlen_idx = indices1_s.reshape([-1])
            gather_idx = paddle.stack([batch_idx, seqlen_idx, expert_idx], -1)
            prob = paddle.gather_nd(gates, gather_idx)
            return prob, expert_idx, gather_idx, capacity

        paddle.disable_static()
        capacity = self.seq_len // 16
        batch_expert_idx = paddle.nonzero(
            paddle.ones(shape=[self.batch_size, self.num_expert, capacity])
        ).cast('int32')
        batch_idx = batch_expert_idx[:, 0]
        expert_idx = batch_expert_idx[:, 1]

        (
            expert_prob_flatten,
            expert_idx_flatten,
            gather_idx,
            cap,
        ) = expert_choice_gating(gate_logits, capacity, batch_idx, expert_idx)
        outputs = paddle.zeros_like(tensor_x)
        batch_prob = expert_prob_flatten.reshape(
            [self.batch_size, self.num_expert, -1, 1]
        )

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
        output = batch_prob * output
        output = output.reshape([-1, tensor_x.shape[-1]])

        outputs = outputs.scatter_nd_add(batch_idx, output)
        return outputs + tensor_x

    def GetFusedEcMoeOut(self, tensor_x, gate_logits):
        paddle.disable_static()
        fused_out = fused_ec_moe(
            tensor_x,
            gate_logits,
            self.bmm_w0,
            self.bmm_b0,
            self.bmm_w1,
            self.bmm_b1,
            self.act_method,
        )

        return fused_out

    def test_fused_ec_moe_op(self):
        gate_logits = self.gate(self.tensor_x)
        final_out_ref = self.GetBaselineOut(self.tensor_x, gate_logits)
        final_out = self.GetFusedEcMoeOut(self.tensor_x, gate_logits)

        np.testing.assert_allclose(
            final_out_ref, final_out, rtol=self.rtol, atol=self.atol
        )


class TestFusedEcMoEOpActGeluFp16(TestFusedEcMoEOp):
    def config(self):
        super().config()
        self.x_type = np.float16


class TestFusedEcMoEOpActReluFp16(TestFusedEcMoEOp):
    def config(self):
        super().config()
        self.x_type = np.float16
        self.act_method = "relu"


if __name__ == "__main__":
    unittest.main()
