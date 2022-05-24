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
        self.__class__.op_type = "fused_gate_attention"
        # use autograd to check grad in this unittest.
        self.__class__.no_need_check_grad = True
        self.config()
        self.merge_qkv = self.q_dim == self.kv_dim
        self.generate_input_data()
        paddle.set_default_dtype(self.dtype)

    def config(self):
        self.dtype = np.float32
        self.has_gating = False
        self.batch_size = 1
        self.msa_len = 3
        self.res_len = 2
        self.q_dim = 6
        self.num_heads = 2
        self.key_dim = 4
        self.m_size = self.res_len
        self.kv_dim = self.q_dim
        self.out_dim = self.q_dim
        self.bias_attr = False

    def generate_input_data(self):
        def _random(shape):
            return np.random.random(shape).astype(self.dtype)

        np.random.seed(123)
        self.query = _random(
            (self.batch_size, self.msa_len, self.res_len, self.q_dim))
        if self.merge_qkv:
            self.key = None
            self.q_weight = None
            self.k_weight = None
            self.v_weight = None
            self.qkv_weight = _random(
                (3, self.num_heads, self.key_dim, self.q_dim))
        else:
            self.key = _random(
                (self.batch_size, self.msa_len, self.m_size, self.kv_dim))
            self.q_weight = _random((self.q_dim, self.num_heads, self.key_dim))
            self.k_weight = _random((self.kv_dim, self.num_heads, self.key_dim))
            self.v_weight = _random((self.kv_dim, self.num_heads, self.key_dim))
            self.qkv_weight = None

        self.attn_mask = _random(
            (self.batch_size, self.msa_len, 1, 1, self.m_size))

        if self.bias_attr:
            self.nonbatched_bias = _random(
                (self.batch_size, 1, self.num_heads, self.res_len, self.m_size))

        if self.has_gating:
            self.gating_w = _random((self.q_dim, self.num_heads, self.key_dim))
            self.gating_b = _random((self.num_heads, self.key_dim))

        self.output_w = _random((self.num_heads, self.key_dim, self.out_dim))
        self.output_b = _random((self.out_dim))

        self.dout = _random(
            (self.batch_size, self.msa_len, self.res_len, self.q_dim))

    def get_reference_out(self):
        paddle.disable_static(place=paddle.CUDAPlace(0))

        query = paddle.to_tensor(self.query, stop_gradient=False)
        if self.merge_qkv:
            key = None
            q_weight = None
            k_weight = None
            v_weight = None
            qkv_weight = paddle.to_tensor(self.qkv_weight, stop_gradient=False)
        else:
            key = paddle.to_tensor(self.key, stop_gradient=False)
            q_weight = paddle.to_tensor(self.q_weight, stop_gradient=False)
            k_weight = paddle.to_tensor(self.k_weight, stop_gradient=False)
            v_weight = paddle.to_tensor(self.v_weight, stop_gradient=False)
            qkv_weight = None

        src_mask = paddle.to_tensor(self.attn_mask, stop_gradient=True)

        c = self.key_dim**(-0.5)
        if self.merge_qkv:
            qkv_out = paddle.einsum('nbqa,khca->nbqkhc', query, qkv_weight)
            # [3, batch_size, msa_len, num_heads, res_len, key_dim]
            qkv_transpose_out = paddle.transpose(
                qkv_out, perm=[3, 0, 1, 4, 2, 5])
            # [batch_size, msa_len, num_heads, res_len, key_dim]
            q = paddle.squeeze(qkv_transpose_out[0:1, ::], axis=0) * c
            # [batch_size, msa_len, num_heads, res_len, key_dim]
            k = paddle.squeeze(qkv_transpose_out[1:2, ::], axis=0)
            # [batch_size, msa_len, num_heads, res_len, key_dim]
            v = paddle.squeeze(qkv_transpose_out[2:3, ::], axis=0)
        else:
            q = paddle.einsum('nbqa,ahc->nbqhc', query, q_weight) * c
            k = paddle.einsum('nbka,ahc->nbkhc', key, k_weight)
            v = paddle.einsum('nbka,ahc->nbkhc', key, v_weight)

        logits = paddle.einsum('nbqhc,nbkhc->nbhqk', q, k) + src_mask
        if self.bias_attr:
            nonbatched_bias = paddle.to_tensor(
                self.nonbatched_bias, stop_gradient=False)
            logits = logits + nonbatched_bias

        weights = nn.functional.softmax(logits)
        weighted_avg = paddle.einsum('nbhqk,nbkhc->nbqhc', weights, v)

        if self.has_gating:
            gating_w = paddle.to_tensor(self.gating_w, stop_gradient=False)
            gating_b = paddle.to_tensor(self.gating_b, stop_gradient=False)
            gate_values = paddle.einsum('nbqc,chv->nbqhv', query,
                                        gating_w) + gating_b
            gate_values = nn.functional.sigmoid(gate_values)
            weighted_avg = weighted_avg * gate_values

        output_b = paddle.to_tensor(self.output_b, stop_gradient=False)
        output_w = paddle.to_tensor(self.output_w, stop_gradient=False)

        out = paddle.einsum('nbqhc,hco->nbqo', weighted_avg,
                            output_w) + output_b
        paddle.autograd.backward(
            [out], [paddle.to_tensor(self.dout)], retain_graph=True)
        if self.merge_qkv:
            return out, query.grad, None
        else:
            return out, query.grad, key.grad

    def get_fused_gate_attention_out(self):
        paddle.disable_static(place=paddle.CUDAPlace(0))

        query = paddle.to_tensor(self.query, stop_gradient=False)
        if self.merge_qkv:
            key = None
            q_weight = None
            k_weight = None
            v_weight = None
            qkv_weight = paddle.to_tensor(self.qkv_weight, stop_gradient=False)
        else:
            key = paddle.to_tensor(self.key, stop_gradient=False)
            q_weight = paddle.to_tensor(self.q_weight, stop_gradient=False)
            k_weight = paddle.to_tensor(self.k_weight, stop_gradient=False)
            v_weight = paddle.to_tensor(self.v_weight, stop_gradient=False)
            qkv_weight = None

        src_mask = paddle.to_tensor(self.attn_mask, stop_gradient=True)

        if self.bias_attr:
            nonbatched_bias = paddle.to_tensor(
                self.nonbatched_bias, stop_gradient=False)
        else:
            nonbatched_bias = None
        if self.has_gating:
            gating_w = paddle.to_tensor(self.gating_w, stop_gradient=False)
            gating_b = paddle.to_tensor(self.gating_b, stop_gradient=False)
        else:
            gating_w = None
            gating_b = None

        output_w = paddle.to_tensor(self.output_w, stop_gradient=False)
        output_b = paddle.to_tensor(self.output_b, stop_gradient=False)

        query_out, key_out, value_out, qkv_out, qktv_out, softmax_out, gate_out, out = _C_ops.fused_gate_attention(
            query, key, q_weight, k_weight, v_weight, qkv_weight,
            nonbatched_bias, src_mask, gating_w, gating_b, output_w, output_b,
            'has_gating', self.has_gating, 'merge_qkv', self.merge_qkv)

        paddle.autograd.backward(
            [out], [paddle.to_tensor(self.dout)], retain_graph=True)
        if key is not None:
            return out, query.grad, key.grad
        else:
            return out, query.grad, None

    def test_fused_gate_attention_op(self):
        out_ref, query_grad_ref, key_grad_ref = self.get_reference_out()
        out, query_grad, key_grad = self.get_fused_gate_attention_out()
        np.testing.assert_allclose(out_ref, out.numpy(), rtol=1e-5, atol=1e-6)


#        np.testing.assert_allclose(
#            query_grad_ref, query_grad.numpy(), rtol=1e-5, atol=1e-5)

#class TestSeparatedQKVCase(TestFusedGateAttentionOp):
#    def config(self):
#        self.dtype = np.float32
#        self.has_gating = False
#        self.batch_size = 1
#        self.msa_len = 3
#        self.res_len = 2
#        self.q_dim = 6
#        self.num_heads = 2
#        self.key_dim = 4
#        self.m_size = 4
#        self.kv_dim = 2
#        self.out_dim = self.q_dim
#        self.bias_attr = False

#class TestMergeQKVNoBiasCase(TestMergeQKVCase):
#    def config(self):
#        super().config()
#        self.bias_attr = False
#
#
#class TestMergeQKVNoGatingCase(TestMergeQKVCase):
#    def config(self):
#        super().config()
#        self.has_gating = False
#
#
#class TestMergeQKVFp16Case(TestMergeQKVCase):
#    def config(self):
#        super().config()
#        self.dtype = np.float16
#
#    def test_fused_gate_attention_op(self):
#        out_ref, query_grad_ref, key_grad_ref = self.GetReferenceOut()
#        out, query_grad, key_grad = self.GetFusedGateAttentionOut()
#        np.testing.assert_allclose(
#            out_ref, out.numpy(), rtol=1e-5, atol=1e-1)
#        np.testing.assert_allclose(
#            query_grad_ref, query_grad.numpy(), rtol=1e-5, atol=1e-1)
#
#
#class TestMergeQKVBF16Case(TestMergeQKVCase):
#    def config(self):
#        super().config()
#
#    def generate_input_data(self):
#        super().generate_input_data()
#        self.query = convert_float_to_uint16(self.query)
#        self.attn_mask = convert_float_to_uint16(self.attn_mask)
#        self.dout = convert_float_to_uint16(self.dout)
#        self.qkv_weight = convert_float_to_uint16(self.qkv_weight)
#        if self.bias_attr:
#            self.nonbatched_bias = convert_float_to_uint16(self.nonbatched_bias)
#        if self.is_gating:
#            self.gating_w = convert_float_to_uint16(self.gating_w)
#            self.gating_b = convert_float_to_uint16(self.gating_b)
#        self.output_w = convert_float_to_uint16(self.output_w)
#        self.output_b = convert_float_to_uint16(self.output_b)
#
#    def test_fused_gate_attention_op(self):
#        final_out_ref, x_grad_ref = self.GetBaselineOut()
#        final_out, x_grad = self.GetFusedGateAttentionOut()
#        np.testing.assert_allclose(
#            final_out_ref, final_out.numpy(), rtol=1e-3, atol=1e-1)
#        np.testing.assert_allclose(
#            x_grad_ref, x_grad.numpy(), rtol=1e-3, atol=1e-1)

if __name__ == "__main__":
    unittest.main()
