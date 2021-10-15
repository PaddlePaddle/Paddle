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
from paddle import tensor
from paddle.fluid import layers
from paddle.nn.layer import fused_transformer
# import unittest
import time
#import paddle.fluid.profiler as profiler

place = paddle.CUDAPlace(0)

##
x_type = np.float32
pre_layer_norm = True
training = True

batch_size = 64
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

## 
query = np.random.rand(batch_size, query_length, embed_dim).astype(x_type)
key, value = query, query
seq_len_vec = np.full((batch_size, ), query_length, dtype=np.int32)
attn_low_windows = np.zeros((query_length, ), dtype=np.int32)
attn_high_windows = np.full((query_length, ), query_length, dtype=np.int32)
dout = np.random.random((batch_size, query_length, embed_dim)).astype(x_type)

## 
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

iters = 101


def print_time(desc, times):
    #times *= 1000
    print(desc, " total time = ", times, "s, avg time = ", times / (iters - 1),
          "s")
    print()


def GetBaselineOut():
    tensor_query = paddle.to_tensor(query, stop_gradient=False)
    residual = tensor_query
    dout_tensor = paddle.to_tensor(dout)

    paddle.device.cuda.synchronize(place)
    #t0 = time.time()
    for i in range(iters):
        if i == 1:
            t0 = time.time()
            #profiler.reset_profiler()
            core.nvprof_start()
            core.nvprof_enable_record_event()
            core.nvprof_nvtx_push(str(i))
        if i == 30:
            core.nvprof_nvtx_pop()
            core.nvprof_stop()
        if i > 1 and i < 30:
            core.nvprof_nvtx_pop()
            core.nvprof_nvtx_push(str(i))

        core.nvprof_nvtx_push("forward")

        ln1_out = tensor_query
        if pre_layer_norm:
            ln1_out = norm1(tensor_query)

        # get q, k, v
        q = q_proj(ln1_out)
        q = tensor.reshape(x=q, shape=[0, 0, num_heads, head_dim])
        q_out = tensor.transpose(x=q, perm=[0, 2, 1, 3])
        k = k_proj(ln1_out)
        v = v_proj(ln1_out)
        k = tensor.reshape(x=k, shape=[0, 0, num_heads, head_dim])
        k_out = tensor.transpose(x=k, perm=[0, 2, 1, 3])
        v = tensor.reshape(x=v, shape=[0, 0, num_heads, head_dim])
        v_out = tensor.transpose(x=v, perm=[0, 2, 1, 3])

        # q_out * k^t
        qk_out = layers.matmul(
            x=q_out, y=k_out, transpose_y=True, alpha=head_dim**-0.5)

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

        # combine heads
        fmha_out = tensor.transpose(qktv_out, perm=[0, 2, 1, 3])

        out_linear_in = tensor.reshape(
            x=fmha_out, shape=[0, 0, fmha_out.shape[2] * fmha_out.shape[3]])
        # project to output
        out = out_proj(out_linear_in)

        residual_out = residual + dropout(out)
        if not pre_layer_norm:
            final_out = norm1(residual_out)
        if pre_layer_norm:
            final_out = norm2(residual_out)
        core.nvprof_nvtx_pop()

        core.nvprof_nvtx_push("backward")
        paddle.autograd.backward([final_out], [dout_tensor], retain_graph=True)
        core.nvprof_nvtx_pop()
    paddle.device.cuda.synchronize(place)
    t1 = time.time()
    print_time("baseline dynamic: ", (t1 - t0))
    #return ln1_out, out, final_out, final_out.grad, out.grad, ln1_out.grad, tensor_query.grad
    return ln1_out, out, final_out


def GetFusedAttentionCuDNNFMHAOut():
    out_linear_bias = paddle.to_tensor(out_proj.bias, stop_gradient=False)

    ln1_scale = paddle.to_tensor(norm1.weight, stop_gradient=False)
    ln1_bias = paddle.to_tensor(norm1.bias, stop_gradient=False)
    ln2_scale = paddle.to_tensor(norm2.weight, stop_gradient=False)
    ln2_bias = paddle.to_tensor(norm2.bias, stop_gradient=False)

    weight = np.concatenate(
        (q_proj.weight, k_proj.weight, v_proj.weight, out_proj.weight))
    # print("weight shape: ")
    # print(weight.shape)

    x = paddle.to_tensor(query, stop_gradient=False)
    seq_len_tensor = paddle.to_tensor(seq_len_vec, stop_gradient=True)
    weight_tensor = paddle.to_tensor(weight, stop_gradient=False)
    epsilon = 1e-05
    ln2_epsilon = 1e-05
    dout_tensor = paddle.to_tensor(dout)

    paddle.device.cuda.synchronize(place)
    #profiler.start_profiler('All', 'OpDetail')
    #t0 = time.time()
    for i in range(iters):
        if i == 1:
            t0 = time.time()
            #profiler.reset_profiler()
            core.nvprof_start()
            core.nvprof_enable_record_event()
            core.nvprof_nvtx_push(str(i))
        if i == 30:
            core.nvprof_nvtx_pop()
            core.nvprof_stop()
        if i > 1 and i < 30:
            core.nvprof_nvtx_pop()
            core.nvprof_nvtx_push(str(i))

        core.nvprof_nvtx_push("forward")
        ln_out, out_linear_out, final_out = F.fused_multihead_attention_cudnn_impl(
            x, weight_tensor, seq_len_tensor, num_heads, pre_layer_norm,
            ln1_scale, ln1_bias, ln2_scale, ln2_bias, epsilon, out_linear_bias,
            dropout_prob, attn_dropout_prob, ln2_epsilon, attn_low_windows,
            attn_high_windows, seq_len_vec, seq_len_vec)
        core.nvprof_nvtx_pop()

        core.nvprof_nvtx_push("backward")
        paddle.autograd.backward([final_out], [dout_tensor], retain_graph=True)
        core.nvprof_nvtx_pop()
    paddle.device.cuda.synchronize(place)
    #profiler.stop_profiler('total', 'cudnn-fmha-profiler-res')

    t1 = time.time()
    print_time("fused dynamic: ", (t1 - t0))
    #return ln_out, out_linear_out, final_out, final_out.grad, out_linear_out.grad, ln_out.grad, x.grad
    return ln_out, out_linear_out, final_out


def test_fused_attention_cudnn_fmha_op():
    print(
        "self.batch_size, self.query_length, self.embed_dim, self.num_heads, self.head_dim = "
    )
    print(batch_size, query_length, embed_dim, num_heads, head_dim)
    #ln_out_ref, out_linear_out_ref, final_out_ref, final_grad_ref, out_linear_grad_ref, ln_grad_ref, x_grad_ref = GetBaselineOut()
    #ln_out, out_linear_out, final_out, final_grad, out_linear_grad, ln_grad, x_grad = GetFusedAttentionCuDNNFMHAOut()
    ln_out_ref, out_linear_out_ref, final_out_ref = GetBaselineOut()
    #ln_out, out_linear_out, final_out = GetFusedAttentionCuDNNFMHAOut()

    # np.testing.assert_allclose(
    #     ln_out_ref, ln_out.numpy(), rtol=1e-5, atol=1e-5)

    # np.testing.assert_allclose(
    #     out_linear_out_ref, out_linear_out.numpy(), rtol=1e-5, atol=1e-3)

    # np.testing.assert_allclose(
    #     final_out_ref, final_out.numpy(), rtol=1e-5, atol=1e-3)

    # np.testing.assert_allclose(
    #     final_grad_ref, final_grad.numpy(), rtol=1e-5, atol=1e-4)
    # np.testing.assert_allclose(
    #     out_linear_grad_ref, out_linear_grad.numpy(), rtol=1e-5, atol=1e-3)
    # np.testing.assert_allclose(
    #     ln_grad_ref, ln_grad.numpy(), rtol=1e-5, atol=1e-3)
    # np.testing.assert_allclose(
    #     x_grad_ref, x_grad.numpy(), rtol=1e-5, atol=1e-2)


test_fused_attention_cudnn_fmha_op()
