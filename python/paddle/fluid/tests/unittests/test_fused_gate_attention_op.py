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
import paddle.fluid.core as core
import paddle.nn.functional as F
import paddle.incubate.nn.functional as incubate_f
from paddle.nn.layer.norm import LayerNorm
from paddle.nn.layer.common import Linear, Dropout
from paddle.nn.layer.transformer import _convert_attention_mask
from paddle import tensor
from paddle.fluid import layers
import unittest
from op_test import OpTest
from paddle.fluid.framework import default_main_program

default_main_program().random_seed = 42


class TestFusedAttentionOp(OpTest):
    def setUp(self):
        self.config()
        self.generate_input_data()
        paddle.set_default_dtype(self.x_type)
        self.__class__.op_type = "fused_gate_attention"
        # use autograd to check grad in this unittest.
        self.__class__.no_need_check_grad = True
        self.q_proj = Linear(
            self.embed_dim,
            self.embed_dim,
            self.weight_attr,
            bias_attr=self.bias_attr)
        self.k_proj = Linear(
            self.kdim,
            self.embed_dim,
            self.weight_attr,
            bias_attr=self.bias_attr)
        self.v_proj = Linear(
            self.vdim,
            self.embed_dim,
            self.weight_attr,
            bias_attr=self.bias_attr)
        self.out_proj = Linear(
            self.embed_dim,
            self.embed_dim,
            self.weight_attr,
            bias_attr=self.bias_attr)
        paddle.set_default_dtype(np.float32)
        # self.norm1 = LayerNorm(self.embed_dim)
        # self.norm2 = LayerNorm(self.embed_dim)
        paddle.set_default_dtype(self.x_type)
        # self.dropout = Dropout(self.dropout_prob, mode="upscale_in_train")

    def config(self):
        self.x_type = np.float32
        self.attn_mask_type = np.float64
        self.is_gating = True
        self.has_attn_mask = True
        # self.has_cache_kv = False
        self.training = True

        self.batch_size = 1
        self.msa_length = 132
        self.res_length = 256
        self.num_heads = 8
        self.head_dim = 32
        self.qkv_dim = 256
        # self.embed_dim = self.head_dim * self.num_heads
        # self.weight_attr = None
        # self.bias_attr = None
        # self.kdim, self.vdim = self.embed_dim, self.embed_dim
        # self.key_length, self.value_length = self.query_length, self.query_length

    def generate_input_data(self):
        self.query = np.random.rand(self.qkv_dim, self.num_heads,
                                    self.head_dim).astype(self.x_type)
        # out_seq_len = self.key_length
        if self.is_gating:
            self.gate_weight
            self.gate_bias

        if self.has_attn_mask:
            # [B, n_head, seq_len, out_seq_len]
            self.attn_mask = np.ones(
                (self.batch_size, self.num_heads, self.query_length,
                 out_seq_len),
                dtype=self.attn_mask_type)
            if self.attn_mask_type == np.int64:
                self.attn_mask = np.tril(self.attn_mask)
            elif self.attn_mask_type == np.float64:
                self.attn_mask = (np.tril(self.attn_mask) - 1.0) * 1e9
            else:
                raise ValueError(
                    "'attn_mask_type' should be 'int64' or 'float64'.")
        else:
            self.attn_mask = None
        self.key, self.value = self.query, self.query

        self.dout = np.random.random(
            (self.batch_size, self.msa_length, self.res_length,
             self.head_dim)).astype(self.x_type)

    def GetBaselineOut(self):
        paddle.disable_static(place=paddle.CUDAPlace(0))
        tensor_query = paddle.to_tensor(self.query, stop_gradient=False)

        if self.has_attn_mask:
            attn_mask = paddle.to_tensor(self.attn_mask, stop_gradient=False)
        else:
            attn_mask = None

        q = self.q_proj(ln1_out)
        q = tensor.reshape(x=q, shape=[0, 0, self.num_heads, self.head_dim])
        q_out = tensor.transpose(x=q, perm=[0, 2, 1, 3])
        k = self.k_proj(ln1_out)
        v = self.v_proj(ln1_out)
        k = tensor.reshape(x=k, shape=[0, 0, self.num_heads, self.head_dim])
        k_out = tensor.transpose(x=k, perm=[0, 2, 1, 3])
        v = tensor.reshape(x=v, shape=[0, 0, self.num_heads, self.head_dim])
        v_out = tensor.transpose(x=v, perm=[0, 2, 1, 3])

        # [B, n_head, seq_len, head_dim] * [B, n_head, out_seq_len, head_dim]
        # --> [B, n_head, seq_len, out_seq_len]
        qk_out = layers.matmul(
            x=q_out, y=k_out, transpose_y=True, alpha=self.head_dim**-0.5)

        attn_mask = _convert_attention_mask(attn_mask, qk_out.dtype)
        attn_mask_out = qk_out + attn_mask
        softmax_out = F.softmax(attn_mask_out)

        c = self.c**(-0.5)
        q = paddle.einsum('nbqa,ahc->nbqhc', q_data, self.query_w) * c
        k = paddle.einsum('nbka,ahc->nbkhc', m_data, self.key_w)
        v = paddle.einsum('nbka,ahc->nbkhc', m_data, self.value_w)
        logits = paddle.einsum('nbqhc,nbkhc->nbhqk', q, k) + bias

        if nonbatched_bias is not None:
            # wait if using async communication and dap, otherwise do nothing
            nonbatched_bias = all_gather_opp(
                nonbatched_bias, axis=2, sync=self.comm_sync)
            logits += paddle.unsqueeze(nonbatched_bias, axis=1)

        weights = nn.functional.softmax(logits)
        weighted_avg = paddle.einsum('nbhqk,nbkhc->nbqhc', weights, v)

        if self.gating:
            gate_values = paddle.einsum('nbqc,chv->nbqhv', q_data,
                                        self.gating_w) + self.gating_b
            gate_values = nn.functional.sigmoid(gate_values)
            weighted_avg *= gate_values

        output = paddle.einsum('nbqhc,hco->nbqo', weighted_avg,
                               self.output_w) + self.output_b
        # fmha_out = tensor.transpose(qktv_out, perm=[0, 2, 1, 3])
        # out_linear_in = tensor.reshape(
        #     x=fmha_out, shape=[0, 0, fmha_out.shape[2] * fmha_out.shape[3]])
        # out = self.out_proj(out_linear_in)
        paddle.autograd.backward(
            [final_out], [paddle.to_tensor(self.dout)], retain_graph=True)
        return final_out, tensor_query.grad

    def GetFusedGateAttentionOut(self):
        paddle.disable_static(place=paddle.CUDAPlace(0))
        q_proj_weight = paddle.to_tensor(
            self.q_proj.weight, stop_gradient=False)
        k_proj_weight = paddle.to_tensor(
            self.k_proj.weight, stop_gradient=False)
        v_proj_weight = paddle.to_tensor(
            self.v_proj.weight, stop_gradient=False)
        out_linear_weight = paddle.to_tensor(
            self.out_proj.weight, stop_gradient=False)

        if self.bias_attr is False:
            qkv_bias_tensor = None
            out_linear_bias = None
        else:
            q_proj_bias = paddle.to_tensor(
                self.q_proj.bias, stop_gradient=False)
            k_proj_bias = paddle.to_tensor(
                self.k_proj.bias, stop_gradient=False)
            v_proj_bias = paddle.to_tensor(
                self.v_proj.bias, stop_gradient=False)
            qkv_bias = np.concatenate(
                (q_proj_bias.numpy(), k_proj_bias.numpy(), v_proj_bias.numpy()))
            qkv_bias = qkv_bias.reshape((3, self.num_heads, self.head_dim))
            qkv_bias_tensor = paddle.to_tensor(qkv_bias, stop_gradient=False)
            out_linear_bias = paddle.to_tensor(
                self.out_proj.bias, stop_gradient=False)

        q_proj_weight = q_proj_weight.numpy().transpose((1, 0))
        k_proj_weight = k_proj_weight.numpy().transpose((1, 0))
        v_proj_weight = v_proj_weight.numpy().transpose((1, 0))
        qkv_weight = np.concatenate(
            (q_proj_weight, k_proj_weight, v_proj_weight))
        qkv_weight = qkv_weight.reshape(
            (3, self.num_heads, self.head_dim, self.embed_dim))

        x = paddle.to_tensor(self.query, stop_gradient=False)
        attn_mask = paddle.to_tensor(self.attn_mask, stop_gradient=False)
        qkv_weight_tensor = paddle.to_tensor(qkv_weight, stop_gradient=False)
        epsilon = 1e-05

        if attn_mask is not None:
            attn_mask = _convert_attention_mask(attn_mask, x.dtype)
            qkv_weight = paddle.stack(
                [self.query_w, self.key_w, self.value_w], axis=0)
            qkv_weight = paddle.transpose(qkv_weight, perm=[1, 0, 2, 3])
            output = F.fused_gate_attention(
                x=q_data,
                qkv_weight=qkv_weight,
                linear_weight=self.output_w,
                gate_weight=self.gating_w,
                linear_bias=self.output_b,
                gate_bias=self.gating_b,
                nonbatched_bias=nonbatched_bias,
                attn_mask=bias,
                is_gating=self.gating)
        final_out = incubate_f.fused_gate_attention(
            x, qkv_weight_tensor, out_linear_weight, out_linear_bias, attn_mask)

        paddle.autograd.backward(
            [final_out], [paddle.to_tensor(self.dout)], retain_graph=True)
        return final_out, x.grad

    def test_fused_attention_op(self):
        final_out_ref, x_grad_ref = self.GetBaselineOut()
        final_out, x_grad = self.GetFusedAttentionOut()
        np.testing.assert_allclose(
            final_out_ref, final_out.numpy(), rtol=1e-5, atol=1e-4)
        np.testing.assert_allclose(
            x_grad_ref, x_grad.numpy(), rtol=1e-5, atol=1e-4)


class TestFusedAttentionOpBiasIsNone(TestFusedAttentionOp):
    def config(self):
        super().config()
        self.bias_attr = False


class TestFusedAttentionOpPreLn(TestFusedAttentionOp):
    def config(self):
        super().config()
        self.is_gating = False


class TestFusedAttentionOpFp16(TestFusedAttentionOp):
    def config(self):
        super().config()
        self.x_type = np.float16

    def test_fused_attention_op(self):
        final_out_ref, x_grad_ref = self.GetBaselineOut()
        final_out, x_grad = self.GetFusedGateAttentionOut()
        np.testing.assert_allclose(
            final_out_ref, final_out.numpy(), rtol=1e-5, atol=1e-1)
        np.testing.assert_allclose(
            x_grad_ref, x_grad.numpy(), rtol=1e-5, atol=1e-1)


if __name__ == "__main__":
    unittest.main()
