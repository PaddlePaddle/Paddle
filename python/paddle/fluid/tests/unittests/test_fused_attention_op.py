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

import numpy as np
import paddle
import paddle.nn as nn
import paddle.fluid.core as core
import paddle.nn.functional as F
from paddle.nn.layer.fused_transformer import FusedMultiHeadAttention
from paddle.nn.layer.transformer import _convert_attention_mask
from paddle import tensor
from paddle.fluid import layers
from paddle.static import Program, program_guard
import unittest
from op_test import OpTest


def GetBaselineOut(pre_layer_norm, training, embed_dim, num_heads, head_dim,
                   query, attn_mask, pre_ln_scale, pre_ln_bias, ln_scale,
                   ln_bias, qkv_weight, qkv_bias, linear_weight, linear_bias,
                   attn_dropout_prob, dropout_prob):
    paddle.disable_static(place=paddle.CUDAPlace(0))
    tensor_query = paddle.to_tensor(query, stop_gradient=False)
    attn_mask = paddle.to_tensor(attn_mask, stop_gradient=False)
    residual = tensor_query
    pre_ln_scale = paddle.to_tensor(pre_ln_scale)
    pre_ln_bias = paddle.to_tensor(pre_ln_bias)
    ln_scale = paddle.to_tensor(ln_scale)
    ln_bias = paddle.to_tensor(ln_bias)
    linear_weight = paddle.to_tensor(linear_weight)
    linear_bias = paddle.to_tensor(linear_bias)

    # qkv_weight: [3, num_heads, self.head_dim, embed_dim]
    q_weight = qkv_weight[0:1, ::]
    k_weight = qkv_weight[1:2, ::]
    v_weight = qkv_weight[2:3, ::]
    q_weight = q_weight.reshape(num_heads * head_dim, embed_dim)
    k_weight = k_weight.reshape(num_heads * head_dim, embed_dim)
    v_weight = v_weight.reshape(num_heads * head_dim, embed_dim)
    q_weight = paddle.to_tensor(q_weight.transpose((1, 0)))
    k_weight = paddle.to_tensor(k_weight.transpose((1, 0)))
    v_weight = paddle.to_tensor(v_weight.transpose((1, 0)))
    # qkv_bias: [3, num_heads, self.head_dim]
    q_bias = qkv_bias[0:1, ::]
    q_bias = q_bias.reshape(num_heads * head_dim)
    k_bias = qkv_bias[1:2, ::]
    k_bias = k_bias.reshape(num_heads * head_dim)
    v_bias = qkv_bias[2:3, ::]
    v_bias = v_bias.reshape(num_heads * head_dim)
    q_bias = paddle.to_tensor(q_bias)
    k_bias = paddle.to_tensor(k_bias)
    v_bias = paddle.to_tensor(v_bias)

    for i in range(1):
        ln1_out = tensor_query
        if pre_layer_norm:
            ln1_out = F.layer_norm(tensor_query, embed_dim, pre_ln_scale,
                                   pre_ln_bias)

        q = F.linear(ln1_out, q_weight, q_bias)
        q = tensor.reshape(x=q, shape=[0, 0, num_heads, head_dim])
        q_out = tensor.transpose(x=q, perm=[0, 2, 1, 3])
        k = F.linear(ln1_out, k_weight, k_bias)
        v = F.linear(ln1_out, v_weight, v_bias)
        k = tensor.reshape(x=k, shape=[0, 0, num_heads, head_dim])
        k_out = tensor.transpose(x=k, perm=[0, 2, 1, 3])
        v = tensor.reshape(x=v, shape=[0, 0, num_heads, head_dim])
        v_out = tensor.transpose(x=v, perm=[0, 2, 1, 3])

        qk_out = layers.matmul(
            x=q_out, y=k_out, transpose_y=True, alpha=head_dim**-0.5)

        if attn_mask is not None:
            attn_mask = _convert_attention_mask(attn_mask, qk_out.dtype)
            attn_mask_out = qk_out + attn_mask
            softmax_out = F.softmax(attn_mask_out)
        else:
            softmax_out = F.softmax(qk_out)

        if attn_dropout_prob:
            dropout_out = F.dropout(
                softmax_out,
                attn_dropout_prob,
                training=training,
                mode="upscale_in_train")
            qktv_out = tensor.matmul(dropout_out, v_out)
        else:
            qktv_out = tensor.matmul(softmax_out, v_out)

        fmha_out = tensor.transpose(qktv_out, perm=[0, 2, 1, 3])
        linear_in = tensor.reshape(
            x=fmha_out, shape=[0, 0, fmha_out.shape[2] * fmha_out.shape[3]])
        out = F.linear(linear_in, linear_weight, linear_bias)

        residual_out = residual + F.dropout(
            out, dropout_prob, training=training, mode="upscale_in_train")
        final_out = F.layer_norm(residual_out, embed_dim, ln_scale, ln_bias)
    return final_out


class TestFusedAttentionOpFp32(OpTest):
    def setUp(self):
        self.config()
        self.common_config()
        self.generate_input_data()

    def config(self):
        self.x_type = np.float32
        self.pre_layer_norm = True
        self.batch_size = 8
        self.query_length = 128
        self.head_dim = 64
        self.num_heads = 16

    def common_config(self):
        self.__class__.op_type = "fused_attention"
        paddle.set_default_dtype(self.x_type)
        self.embed_dim = self.head_dim * self.num_heads
        self.kdim, self.vdim = self.embed_dim, self.embed_dim
        self.key_length, self.value_length = self.query_length, self.query_length
        self.attn_mask_type = np.float64
        self.training = True
        self.need_weight = False
        self.dropout_prob = 0.0
        self.attn_dropout_prob = 0.0
        self.weight_attr = None
        self.bias_attr = None

    def generate_input_data(self):
        self.query = np.random.rand(self.batch_size, self.query_length,
                                    self.embed_dim).astype(self.x_type)
        self.attn_mask = np.ones(
            (self.batch_size, self.num_heads, self.query_length,
             self.key_length),
            dtype=self.attn_mask_type)
        if self.attn_mask_type == np.int64:
            self.attn_mask = np.tril(self.attn_mask)
        elif self.attn_mask_type == np.float64:
            self.attn_mask = (np.tril(self.attn_mask) - 1.0) * 1e9
        else:
            raise ValueError("'attn_mask_type' should be 'int64' or 'float64'.")
        self.key, self.value = self.query, self.query

    def compute_result(self):
        paddle.disable_static(place=paddle.CUDAPlace(0))
        fused_attn = FusedMultiHeadAttention(
            self.embed_dim, self.num_heads, self.dropout_prob,
            self.attn_dropout_prob, self.kdim, self.vdim, self.pre_layer_norm,
            self.need_weight, self.weight_attr, self.bias_attr)
        out = fused_attn(
            paddle.to_tensor(self.query),
            paddle.to_tensor(self.query),
            paddle.to_tensor(self.query), paddle.to_tensor(self.attn_mask))
        ref_out = GetBaselineOut(self.pre_layer_norm, self.training,
                                 self.embed_dim, self.num_heads, self.head_dim,
                                 self.query, self.attn_mask,
                                 fused_attn.pre_ln_scale.numpy(),
                                 fused_attn.pre_ln_bias.numpy(),
                                 fused_attn.ln_scale.numpy(),
                                 fused_attn.ln_bias.numpy(),
                                 fused_attn.qkv_weight.numpy(),
                                 fused_attn.qkv_bias.numpy(),
                                 fused_attn.linear_weight.numpy(),
                                 fused_attn.linear_bias.numpy(),
                                 self.attn_dropout_prob, self.dropout_prob)
        return ref_out, out

    def test_fused_attention_op(self):
        ref_out, out = self.compute_result()
        self.assertTrue(np.allclose(ref_out, out, rtol=1e-5, atol=1e-5))


class TestFusedAttentionOpFp16(TestFusedAttentionOpFp32):
    def setUp(self):
        self.config()
        self.common_config()
        self.generate_input_data()

    def config(self):
        self.x_type = np.float16
        self.pre_layer_norm = True
        self.batch_size = 8
        self.query_length = 128
        self.head_dim = 64
        self.num_heads = 16

    def test_fused_attention_op(self):
        ref_out, out = self.compute_result()
        self.assertTrue(np.allclose(ref_out, out, rtol=1e-5, atol=1e-2))


if __name__ == "__main__":
    unittest.main()
