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

import numpy as np

import paddle
import paddle.nn as nn
from paddle import tensor
import unittest
from op_test import OpTest, convert_float_to_uint16
from paddle import _C_ops
from paddle.fluid.framework import default_main_program

default_main_program().random_seed = 42


class TestFusedGateAttentionOp(OpTest):
    def setUp(self):
        self.config()
        self.generate_input_data()
        paddle.set_default_dtype(self.x_type)
        self.__class__.op_type = "fused_gate_attention"
        # use autograd to check grad in this unittest.
        self.__class__.no_need_check_grad = True

    def config(self):
        self.x_type = np.float32
        self.is_gating = False
        self.training = False
        self.batch_size = 1
        self.msa_length = 3
        self.res_length = 2
        self.q_dim = 6
        self.kv_dim = 2
        self.key_dim = 4
        self.num_heads = 2
        self.m_size = 4
        self.value_dim = self.key_dim
        self.out_dim = self.q_dim
        self.bias_attr = False

    def generate_input_data(self):
        np.random.seed(123)
        self.query = np.random.rand(self.batch_size, self.msa_length,
                                    self.res_length,
                                    self.q_dim).astype(self.x_type)
        self.q_weight = np.random.random(
            (self.q_dim, self.num_heads, self.key_dim)).astype(self.x_type)

        self.key = np.random.rand(self.batch_size, self.msa_length, self.m_size,
                                  self.kv_dim).astype(self.x_type)
        self.k_weight = np.random.random(
            (self.kv_dim, self.num_heads, self.key_dim)).astype(self.x_type)
        self.v_weight = np.random.random(
            (self.kv_dim, self.num_heads, self.value_dim)).astype(self.x_type)

        self.attn_mask = np.zeros(
            ((self.batch_size, self.msa_length, self.m_size)),
            dtype=self.x_type)
        attn_mask = paddle.to_tensor(self.attn_mask, stop_gradient=False)
        bias = 1e3 * (attn_mask)
        self.bias = paddle.unsqueeze(bias, axis=[2, 3])

        if self.bias_attr:
            self.nonbatched_bias = np.random.random(
                (self.batch_size, self.num_heads, self.res_length,
                 self.res_length)).astype(self.x_type)
        if self.is_gating:
            self.gating_w = np.random.random((
                self.q_dim, self.num_heads, self.value_dim)).astype(self.x_type)
            self.gating_b = np.random.random(
                (self.num_heads, self.value_dim)).astype(self.x_type)

        self.output_w = np.random.random(
            (self.num_heads, self.value_dim, self.out_dim)).astype(self.x_type)
        self.output_b = np.random.random((self.out_dim)).astype(self.x_type)

        self.dout = np.random.random(
            (self.batch_size, self.msa_length, self.res_length,
             self.q_dim)).astype(self.x_type)

    def GetBaselineOut(self):
        paddle.disable_static(place=paddle.CUDAPlace(0))
        c = self.key_dim**(-0.5)
        q_data = paddle.to_tensor(self.query, stop_gradient=False)
        q_weight = paddle.to_tensor(self.q_weight, stop_gradient=False)
        m_data = paddle.to_tensor(self.key, stop_gradient=False)
        k_weight = paddle.to_tensor(self.k_weight, stop_gradient=False)
        v_weight = paddle.to_tensor(self.v_weight, stop_gradient=False)

        output_b = paddle.to_tensor(self.output_b, stop_gradient=False)
        output_w = paddle.to_tensor(self.output_w, stop_gradient=False)
        src_mask = paddle.to_tensor(self.bias, stop_gradient=True)
        c = self.key_dim**(-0.5)
        q = paddle.einsum('nbqa,ahc->nbqhc', q_data, q_weight)
        k = paddle.einsum('nbka,ahc->nbkhc', m_data, k_weight)
        v = paddle.einsum('nbka,ahc->nbkhc', m_data, v_weight)
        logits = paddle.einsum(
            'nbqhc,nbkhc->nbhqk', q, k
        ) * c + src_mask  # [bs, msa_len, num_heads, res_len, c] * [bs, msa_len, num_heads, res_len, c] -> [bs, msa_len, num_heads, res_len, res_len]

        if self.bias_attr:
            nonbatched_bias = paddle.to_tensor(
                self.nonbatched_bias, stop_gradient=False)
            logits = logits + paddle.unsqueeze(nonbatched_bias, axis=1)

        weights = nn.functional.softmax(logits)
        weighted_avg = paddle.einsum(
            'nbhqk,nbkhc->nbqhc', weights, v
        )  #[bs, msa_len, num_heads, res_len, res_len] * [bs, msa_len, num_heads, res_len, c] -> [bs, msa_len, num_heads,res_len, c]

        if self.is_gating:
            gating_w = paddle.to_tensor(self.gating_w, stop_gradient=False)
            gating_b = paddle.to_tensor(self.gating_b, stop_gradient=False)
            gate_values_new = paddle.einsum('nbqc,chv->nbqhv', q_data,
                                            gating_w) + gating_b
            gate_values = nn.functional.sigmoid(gate_values_new)
            weighted_avg = weighted_avg * gate_values

        final_out = paddle.einsum(
            'nbqhc,hco->nbqo', weighted_avg, output_w
        ) + output_b  # [bs, msa_len, res_len, num_heads, c] * [num_heads, c, out_dim] -> [bs, msa_len, res_len, out_dim]
        paddle.autograd.backward(
            [final_out], [paddle.to_tensor(self.dout)], retain_graph=True)
        return final_out, q_data.grad

    def GetFusedGateAttentionOut(self):
        paddle.disable_static(place=paddle.CUDAPlace(0))
        q_data = paddle.to_tensor(self.query, stop_gradient=False)
        m_data = paddle.to_tensor(self.key, stop_gradient=False)
        q_weight = paddle.to_tensor(self.q_weight, stop_gradient=False)
        k_weight = paddle.to_tensor(self.k_weight, stop_gradient=False)
        v_weight = paddle.to_tensor(self.v_weight, stop_gradient=False)

        output_b = paddle.to_tensor(self.output_b, stop_gradient=False)
        output_w = paddle.to_tensor(self.output_w, stop_gradient=False)
        src_mask = paddle.to_tensor(self.bias, stop_gradient=True)

        if self.bias_attr:
            nonbatched_bias = paddle.to_tensor(
                self.nonbatched_bias, stop_gradient=False)
        else:
            nonbatched_bias = None
        if self.is_gating:
            gating_w = paddle.to_tensor(self.gating_w, stop_gradient=False)
            gating_b = paddle.to_tensor(self.gating_b, stop_gradient=False)
        else:
            gating_w = None
            gating_b = None
        _, _, _, _, _, _, _, final_out = _C_ops.fused_gate_attention(
            q_data, m_data, q_weight, k_weight, v_weight, None, nonbatched_bias,
            src_mask, gating_w, gating_b, output_w, output_b, 'is_gating',
            self.is_gating, 'is_merge', self.q_dim == self.kv_dim)

        paddle.autograd.backward(
            [final_out], [paddle.to_tensor(self.dout)], retain_graph=True)
        return final_out, q_data.grad

    def test_fused_gate_attention_op(self):
        final_out_ref, x_grad_ref = self.GetBaselineOut()
        final_out, x_grad = self.GetFusedGateAttentionOut()
        np.testing.assert_allclose(
            final_out_ref, final_out.numpy(), rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(
            x_grad_ref, x_grad.numpy(), rtol=1e-5, atol=1e-5)


class TestMergeQKVFusedGateAttentionOp(OpTest):
    def setUp(self):
        self.config()
        self.generate_input_data()
        paddle.set_default_dtype(self.x_type)
        self.__class__.op_type = "fused_gate_attention"
        # use autograd to check grad in this unittest.
        self.__class__.no_need_check_grad = True

    def config(self):
        self.x_type = np.float32
        self.is_gating = True
        self.training = False
        self.batch_size = 1
        self.msa_length = 3
        self.res_length = 2
        self.qkv_dim = 6
        self.num_heads = 2
        self.c = 4
        self.out_dim = self.qkv_dim
        self.bias_attr = True

    def generate_input_data(self):
        np.random.seed(123)
        self.query = np.random.rand(self.batch_size, self.msa_length,
                                    self.res_length,
                                    self.qkv_dim).astype(self.x_type)

        self.attn_mask = np.zeros(
            ((self.batch_size, self.msa_length, self.res_length)),
            dtype=self.x_type)
        attn_mask = paddle.to_tensor(self.attn_mask, stop_gradient=False)
        bias = 1e3 * (attn_mask)
        self.bias = paddle.unsqueeze(bias, axis=[2, 3])

        self.qkv_weight = np.random.random(
            (3, self.num_heads, self.c, self.qkv_dim)).astype(self.x_type)

        if self.bias_attr:
            self.nonbatched_bias = np.random.random(
                (self.batch_size, self.num_heads, self.res_length,
                 self.res_length)).astype(self.x_type)
        if self.is_gating:
            self.gating_w = np.random.random(
                (self.qkv_dim, self.num_heads, self.c)).astype(self.x_type)
            self.gating_b = np.random.random(
                (self.num_heads, self.c)).astype(self.x_type)

        self.output_w = np.random.random(
            (self.num_heads, self.c, self.out_dim)).astype(self.x_type)
        self.output_b = np.random.random((self.out_dim)).astype(self.x_type)

        self.dout = np.random.random(
            (self.batch_size, self.msa_length, self.res_length,
             self.qkv_dim)).astype(self.x_type)

    def GetBaselineOut(self):
        paddle.disable_static(place=paddle.CUDAPlace(0))
        c = self.c**(-0.5)
        tensor_query = paddle.to_tensor(self.query, stop_gradient=False)
        qkv_weight = paddle.to_tensor(self.qkv_weight, stop_gradient=False)
        output_b = paddle.to_tensor(self.output_b, stop_gradient=False)
        output_w = paddle.to_tensor(self.output_w, stop_gradient=False)
        src_mask = paddle.to_tensor(self.bias, stop_gradient=True)
        qkv_out_1 = paddle.einsum(
            'nbqa,khca->nbqkhc', tensor_query, qkv_weight
        )  # [bs, msa_len, res_len, qkv_dim] * [3, num_heads, c, qkv_dim] -trans_y=true-> [bs, msa_len, res_len, 3, num_heads, c]
        qkv_out = paddle.transpose(
            qkv_out_1,
            perm=[3, 0, 1, 4, 2, 5])  # [3, bs, msa_len, num_heads, res_len, c]
        q = paddle.squeeze(
            qkv_out[0:1, ::], axis=0)  # [bs, msa_len, num_heads, res_len, c]
        k = paddle.squeeze(
            qkv_out[1:2, ::], axis=0)  # [bs, msa_len, num_heads, res_len, c]
        v = paddle.squeeze(
            qkv_out[2:3, ::], axis=0)  # [bs, msa_len, num_heads, res_len, c]
        logits = paddle.einsum(
            'nbqhc,nbqkc->nbqhk', q, k
        ) * c + src_mask  # [bs, msa_len, num_heads, res_len, c] * [bs, msa_len, num_heads, res_len, c] -> [bs, msa_len, num_heads, res_len, res_len]

        if self.bias_attr:
            nonbatched_bias = paddle.to_tensor(
                self.nonbatched_bias, stop_gradient=False)
            logits = logits + paddle.unsqueeze(nonbatched_bias, axis=1)

        weights = nn.functional.softmax(logits)
        weighted_avg_1 = paddle.einsum(
            'nbqhk,nbqkc->nbqhc', weights, v
        )  #[bs, msa_len, num_heads, res_len, res_len] * [bs, msa_len, num_heads, res_len, c] -> [bs, msa_len, num_heads,res_len, c]

        weighted_avg = paddle.transpose(
            weighted_avg_1,
            perm=[0, 1, 3, 2, 4])  # [bs, msa_len, res_len, num_heads, c]
        if self.is_gating:
            gating_w = paddle.to_tensor(self.gating_w, stop_gradient=False)
            gating_b = paddle.to_tensor(self.gating_b, stop_gradient=False)
            gate_values_new = paddle.einsum('nbqc,chv->nbqhv', tensor_query,
                                            gating_w) + gating_b
            gate_values = nn.functional.sigmoid(gate_values_new)
            weighted_avg = weighted_avg * gate_values

        final_out = paddle.einsum(
            'nbqhc,hco->nbqo', weighted_avg, output_w
        ) + output_b  # [bs, msa_len, res_len, num_heads, c] * [num_heads, c, out_dim] -> [bs, msa_len, res_len, out_dim]
        paddle.autograd.backward(
            [final_out], [paddle.to_tensor(self.dout)], retain_graph=True)
        return final_out, tensor_query.grad

    def GetFusedGateAttentionOut(self):
        paddle.disable_static(place=paddle.CUDAPlace(0))
        x = paddle.to_tensor(self.query, stop_gradient=False)
        qkv_weight = paddle.to_tensor(self.qkv_weight, stop_gradient=False)
        output_b = paddle.to_tensor(self.output_b, stop_gradient=False)
        output_w = paddle.to_tensor(self.output_w, stop_gradient=False)
        src_mask = paddle.to_tensor(self.bias, stop_gradient=True)

        if self.bias_attr:
            nonbatched_bias = paddle.to_tensor(
                self.nonbatched_bias, stop_gradient=False)
        else:
            nonbatched_bias = None
        if self.is_gating:
            gating_w = paddle.to_tensor(self.gating_w, stop_gradient=False)
            gating_b = paddle.to_tensor(self.gating_b, stop_gradient=False)
        else:
            gating_w = None
            gating_b = None

        _, _, _, _, _, _, _, final_out = _C_ops.fused_gate_attention(
            x, x, None, None, None, qkv_weight, nonbatched_bias, src_mask,
            gating_w, gating_b, output_w, output_b, 'is_gating', self.is_gating,
            'is_merge', True)

        paddle.autograd.backward(
            [final_out], [paddle.to_tensor(self.dout)], retain_graph=True)
        return final_out, x.grad

    def test_fused_gate_attention_op(self):
        final_out_ref, x_grad_ref = self.GetBaselineOut()
        final_out, x_grad = self.GetFusedGateAttentionOut()
        np.testing.assert_allclose(
            final_out_ref, final_out.numpy(), rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(
            x_grad_ref, x_grad.numpy(), rtol=1e-5, atol=1e-5)


class TestMergeQKVFusedGateAttentionOpBiasIsNone(
        TestMergeQKVFusedGateAttentionOp):
    def config(self):
        super().config()
        self.bias_attr = False


class TestMergeQKVFusedGateAttentionGatingIsFalse(
        TestMergeQKVFusedGateAttentionOp):
    def config(self):
        super().config()
        self.is_gating = False


class TestMergeQKVFusedGateAttentionOpFp16(TestMergeQKVFusedGateAttentionOp):
    def config(self):
        super().config()
        self.x_type = np.float16

    def test_fused_gate_attention_op(self):
        final_out_ref, x_grad_ref = self.GetBaselineOut()
        final_out, x_grad = self.GetFusedGateAttentionOut()
        np.testing.assert_allclose(
            final_out_ref, final_out.numpy(), rtol=1e-5, atol=1e-1)
        np.testing.assert_allclose(
            x_grad_ref, x_grad.numpy(), rtol=1e-5, atol=1e-1)


class TestMergeQKVFusedGateAttentionOpBF16(TestMergeQKVFusedGateAttentionOp):
    def config(self):
        super().config()

    def generate_input_data(self):
        super().generate_input_data()
        self.query = convert_float_to_uint16(self.query)
        self.bias = convert_float_to_uint16(self.bias)
        self.dout = convert_float_to_uint16(self.dout)
        self.qkv_weight = convert_float_to_uint16(self.qkv_weight)
        if self.bias_attr:
            self.nonbatched_bias = convert_float_to_uint16(self.nonbatched_bias)
        if self.is_gating:
            self.gating_w = convert_float_to_uint16(self.gating_w)
            self.gating_b = convert_float_to_uint16(self.gating_b)
        self.output_w = convert_float_to_uint16(self.output_w)
        self.output_b = convert_float_to_uint16(self.output_b)

    def test_fused_gate_attention_op(self):
        final_out_ref, x_grad_ref = self.GetBaselineOut()
        final_out, x_grad = self.GetFusedGateAttentionOut()
        np.testing.assert_allclose(
            final_out_ref, final_out.numpy(), rtol=1e-3, atol=1e-1)
        np.testing.assert_allclose(
            x_grad_ref, x_grad.numpy(), rtol=1e-3, atol=1e-1)


if __name__ == "__main__":
    unittest.main()
