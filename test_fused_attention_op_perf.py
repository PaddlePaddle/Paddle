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
from paddle.nn.layer.norm import LayerNorm
from paddle.nn.layer.common import Linear, Dropout
from paddle.nn.layer.transformer import _convert_attention_mask
from paddle import tensor
from paddle.fluid import layers
import unittest
import time

x_type = np.float32

attn_mask_type = np.float64
pre_layer_norm = True
training = True

batch_size = 32
query_length = 128
head_dim = 64
num_heads = 16
embed_dim = head_dim * num_heads

dropout_prob = 0.0
attn_dropout_prob = 0.0
weight_attr = None
bias_attr = None
kdim, vdim = embed_dim, embed_dim
key_length, value_length = query_length, query_length

query = np.random.rand(batch_size, query_length, embed_dim).astype(x_type)
attn_mask = np.ones(
    (batch_size, num_heads, query_length, key_length), dtype=attn_mask_type)
if attn_mask_type == np.int64:
    attn_mask = np.tril(attn_mask)
elif attn_mask_type == np.float64:
    attn_mask = (np.tril(attn_mask) - 1.0) * 1e9
else:
    raise ValueError("'attn_mask_type' should be 'int64' or 'float64'.")
key, value = query, query

dout = np.random.random((batch_size, query_length, embed_dim)).astype(x_type)

paddle.set_default_dtype(x_type)
q_proj = Linear(embed_dim, embed_dim, weight_attr, bias_attr=bias_attr)
k_proj = Linear(kdim, embed_dim, weight_attr, bias_attr=bias_attr)
v_proj = Linear(vdim, embed_dim, weight_attr, bias_attr=bias_attr)
out_proj = Linear(embed_dim, embed_dim, weight_attr, bias_attr=bias_attr)
paddle.set_default_dtype(np.float32)
norm1 = LayerNorm(embed_dim)
norm2 = LayerNorm(embed_dim)
paddle.set_default_dtype(x_type)
dropout = Dropout(dropout_prob, mode="upscale_in_train")


def GetBaselineOut(tensor_query, tensor_attn_mask, residual):
    # paddle.disable_static(place=paddle.CUDAPlace(0))
    # tensor_query = paddle.to_tensor(query, stop_gradient=False)
    # tensor_attn_mask = paddle.to_tensor(attn_mask, stop_gradient=False)
    # residual = tensor_query

    # print("x_type is ", x_type)
    # paddle.device.cuda.synchronize(paddle.CUDAPlace(0))
    # time_start = time.time()

    # for i in range(10000):
    ln1_out = tensor_query
    if pre_layer_norm:
        ln1_out = norm1(tensor_query)

    q = q_proj(ln1_out)
    q = tensor.reshape(x=q, shape=[0, 0, num_heads, head_dim])
    q_out = tensor.transpose(x=q, perm=[0, 2, 1, 3])
    k = k_proj(ln1_out)
    v = v_proj(ln1_out)
    k = tensor.reshape(x=k, shape=[0, 0, num_heads, head_dim])
    k_out = tensor.transpose(x=k, perm=[0, 2, 1, 3])
    v = tensor.reshape(x=v, shape=[0, 0, num_heads, head_dim])
    v_out = tensor.transpose(x=v, perm=[0, 2, 1, 3])

    qk_out = layers.matmul(
        x=q_out, y=k_out, transpose_y=True, alpha=head_dim**-0.5)

    if tensor_attn_mask is not None:
        tensor_attn_mask = _convert_attention_mask(tensor_attn_mask,
                                                   qk_out.dtype)
        attn_mask_out = qk_out + tensor_attn_mask
        softmax_out = F.softmax(attn_mask_out)
    else:
        softmax_out = F.softmax(qk_out)

    if dropout_prob:
        dropout_out = F.dropout(
            softmax_out,
            dropout_prob,
            training=training,
            mode="upscale_in_train")
        qktv_out = tensor.matmul(dropout_out, v_out)
    else:
        qktv_out = tensor.matmul(softmax_out, v_out)

    fmha_out = tensor.transpose(qktv_out, perm=[0, 2, 1, 3])
    out_linear_in = tensor.reshape(
        x=fmha_out, shape=[0, 0, fmha_out.shape[2] * fmha_out.shape[3]])
    out = out_proj(out_linear_in)

    residual_out = residual + dropout(out)
    if not pre_layer_norm:
        final_out = norm1(residual_out)
    if pre_layer_norm:
        final_out = norm2(residual_out)
    # paddle.autograd.backward(
    #     [final_out], [paddle.to_tensor(dout)], retain_graph=True)

    # paddle.device.cuda.synchronize(paddle.CUDAPlace(0))
    # time_end = time.time()
    # all_time = time_end - time_start
    # print("baseline time is ", all_time)
    # #return final_out, tensor_query.grad
    return final_out


def TestBaselinePerf():
    paddle.disable_static(place=paddle.CUDAPlace(0))
    tensor_query = paddle.to_tensor(query, stop_gradient=False)
    tensor_attn_mask = paddle.to_tensor(attn_mask, stop_gradient=False)
    residual = tensor_query

    final_out = GetBaselineOut(tensor_query, tensor_attn_mask, residual)

    print("x_type is ", x_type)
    paddle.device.cuda.synchronize(paddle.CUDAPlace(0))
    time_start = time.time()
    for i in range(10000):
        final_out = GetBaselineOut(tensor_query, tensor_attn_mask, residual)
    paddle.device.cuda.synchronize(paddle.CUDAPlace(0))
    time_end = time.time()
    all_time = time_end - time_start
    print("baseline time is ", all_time)
    #return final_out, tensor_query.grad
    return final_out


def TestFusedAttentionPerf():
    paddle.disable_static(place=paddle.CUDAPlace(0))
    q_proj_weight = paddle.to_tensor(q_proj.weight, stop_gradient=False)
    q_proj_bias = paddle.to_tensor(q_proj.bias, stop_gradient=False)
    k_proj_weight = paddle.to_tensor(k_proj.weight, stop_gradient=False)
    k_proj_bias = paddle.to_tensor(k_proj.bias, stop_gradient=False)
    v_proj_weight = paddle.to_tensor(v_proj.weight, stop_gradient=False)
    v_proj_bias = paddle.to_tensor(v_proj.bias, stop_gradient=False)
    out_linear_weight = paddle.to_tensor(out_proj.weight, stop_gradient=False)
    out_linear_bias = paddle.to_tensor(out_proj.bias, stop_gradient=False)

    ln1_scale = paddle.to_tensor(norm1.weight, stop_gradient=False)
    ln1_bias = paddle.to_tensor(norm1.bias, stop_gradient=False)
    ln2_scale = paddle.to_tensor(norm2.weight, stop_gradient=False)
    ln2_bias = paddle.to_tensor(norm2.bias, stop_gradient=False)

    q_proj_weight = q_proj_weight.numpy().transpose((1, 0))
    k_proj_weight = k_proj_weight.numpy().transpose((1, 0))
    v_proj_weight = v_proj_weight.numpy().transpose((1, 0))
    qkv_weight = np.concatenate((q_proj_weight, k_proj_weight, v_proj_weight))
    qkv_weight = qkv_weight.reshape((3, num_heads, head_dim, embed_dim))

    qkv_bias = np.concatenate(
        (q_proj_bias.numpy(), k_proj_bias.numpy(), v_proj_bias.numpy()))
    qkv_bias = qkv_bias.reshape((3, num_heads, head_dim))

    x = paddle.to_tensor(query, stop_gradient=False)
    tensor_attn_mask = paddle.to_tensor(attn_mask, stop_gradient=False)
    qkv_weight_tensor = paddle.to_tensor(qkv_weight, stop_gradient=False)
    qkv_bias_tensor = paddle.to_tensor(qkv_bias, stop_gradient=False)
    epsilon = 1e-05
    ln2_epsilon = 1e-05

    if tensor_attn_mask is not None:
        tensor_attn_mask = _convert_attention_mask(tensor_attn_mask, x.dtype)

    final_out = F.fused_multihead_attention(
        x, qkv_weight_tensor, out_linear_weight, pre_layer_norm, ln1_scale,
        ln1_bias, ln2_scale, ln2_bias, epsilon, qkv_bias_tensor,
        out_linear_bias, tensor_attn_mask, dropout_prob, attn_dropout_prob,
        ln2_epsilon)

    paddle.device.cuda.synchronize(paddle.CUDAPlace(0))
    time_start = time.time()
    for i in range(10000):
        final_out = F.fused_multihead_attention(
            x, qkv_weight_tensor, out_linear_weight, pre_layer_norm, ln1_scale,
            ln1_bias, ln2_scale, ln2_bias, epsilon, qkv_bias_tensor,
            out_linear_bias, tensor_attn_mask, dropout_prob, attn_dropout_prob,
            ln2_epsilon)
        # paddle.autograd.backward(
        #     [final_out], [paddle.to_tensor(dout)], retain_graph=True)
    paddle.device.cuda.synchronize(paddle.CUDAPlace(0))
    time_end = time.time()
    all_time = time_end - time_start
    print("fused time is ", all_time)
    #return final_out, x.grad
    return final_out


def test_fused_attention_op():
    # final_out_ref, x_grad_ref = GetBaselineOut()
    # final_out, x_grad = GetFusedAttentionOut()
    #final_out_ref = TestBaselinePerf()
    final_out = TestFusedAttentionPerf()

    # np.testing.assert_allclose(
    #     final_out_ref, final_out.numpy(), rtol=1e-5, atol=1e-2)


test_fused_attention_op()
