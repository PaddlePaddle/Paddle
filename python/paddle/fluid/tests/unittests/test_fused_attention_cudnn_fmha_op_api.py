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
#from paddle.nn.layer.fused_transformer import FusedMultiHeadAttention
from paddle.incubate.nn.layer.fused_transformer import FusedCudnnMultiHeadAttention
from paddle import tensor
from paddle.fluid import layers
from paddle.static import Program, program_guard
import unittest


def fc(x, weight):
    return np.matmul(x, weight)


def softmax(x):
    np.seterr(invalid='ignore')
    output = np.zeros(x.shape, dtype=np.float64)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            for k in range(x.shape[2]):
                x_curr = x[i, j, k, :]
                e_x = np.exp(x_curr - np.amax(x_curr))
                output[i, j, k, :] = e_x / np.sum(e_x)
    return output


def batch_matmul(x, y):
    assert x.shape[0] == y.shape[0]
    assert x.shape[1] == y.shape[1]
    retval = np.zeros(
        (x.shape[0], x.shape[1], x.shape[2], y.shape[3]), dtype=np.float64)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            retval[i, j, :, :] = np.matmul(x[i, j, :, :], y[i, j, :, :])
    return retval


def layer_norm(x, has_scale, has_bias, weight, bias, epsilon=1e-05):
    batch_size, src_len, d_model = x.shape
    x = x.reshape((batch_size * src_len, d_model))
    mu = np.mean(x, axis=1, keepdims=True)
    sigma_squar = np.sum(np.square(x - mu), axis=1) / d_model
    x1_up = (x - mu)
    x1_down_1 = sigma_squar + epsilon
    x1_down = np.sqrt(x1_down_1)
    x1_down = x1_down.reshape((x1_down.shape[0], 1))
    x1 = x1_up / x1_down
    x_scaled = x1
    if (has_scale):
        x_scaled = weight * x1
    x_scaled_bias = x_scaled
    if (has_bias):
        x_scaled_bias = x_scaled + bias
    x_scaled_bias = x_scaled_bias.reshape((batch_size, src_len, d_model))
    return x_scaled_bias

def compute_reference(pre_layer_norm, num_head, query, attn_mask, ln_scale, ln_bias, 
                      weight, has_bias):
    batch_size = query.shape[0]
    seq_len = query.shape[1]
    embed_dim = query.shape[2]
    head_dim = embed_dim//num_head

    # print(batch_size)
    # print(seq_len)
    # print(embed_dim)
    # print(head_dim)
    
    #[1, embed_dim, embed_dim]
    weight_stride = embed_dim * embed_dim
    bias_start = 4*weight_stride
    bias_stride = embed_dim
    q_weight = weight[0:weight_stride]
    k_weight = weight[weight_stride:2*weight_stride]
    v_weight = weight[2*weight_stride:3*weight_stride]
    out_linear_weight = weight[3*weight_stride:4*weight_stride]
    q_bias = weight[bias_start: bias_start+bias_stride]
    k_bias = weight[bias_start + bias_stride:bias_start + 2*bias_stride]
    v_bias = weight[(bias_start + 2*bias_stride):(bias_start + 3*bias_stride)]
    out_linear_bias = weight[(bias_start + 3*bias_stride):(bias_start + 4*bias_stride)]
    # print(weight.shape)
    # print(q_weight.shape)
    # print(k_weight.shape)
    # print(v_weight.shape)
    # print(out_linear_weight.shape)

    q_weight = q_weight.reshape(embed_dim, num_head*head_dim)
    k_weight = k_weight.reshape(embed_dim, num_head*head_dim)
    v_weight = v_weight.reshape(embed_dim, num_head*head_dim)
    out_linear_weight = out_linear_weight.reshape(embed_dim, embed_dim)

    q_bias = q_bias.reshape(embed_dim)
    k_bias = k_bias.reshape(embed_dim)
    v_bias = v_bias.reshape(embed_dim)
    out_linear_bias = out_linear_bias.reshape(embed_dim)

    if (pre_layer_norm):
        ln_out = layer_norm(query, True, has_bias, ln_scale, ln_bias)

    if (pre_layer_norm):
        ln_out = ln_out.reshape(batch_size * seq_len, embed_dim)
        # print(ln_out.shape)
        # print(q_weight.shape)
        q = fc(ln_out, q_weight)
        k = fc(ln_out, k_weight)
        v = fc(ln_out, v_weight)
        if has_bias:
            q_bias_out = q + q_bias
            k_bias_out = k + k_bias
            v_bias_out = v + v_bias
        else:
            q_bias_out = q
            k_bias_out = k
            v_bias_out = v
        ln_out = ln_out.reshape(batch_size, seq_len, embed_dim)
    else:
        query = query.reshape(batch_size * seq_len, embed_dim)
        q = fc(query, q_weight)
        k = fc(query, k_weight)
        v = fc(query, v_weight)
        if has_bias:
            q_bias_out = q + q_bias
            k_bias_out = k + k_bias
            v_bias_out = v + v_bias
        else:
            q_bias_out = q
            k_bias_out = k
            v_bias_out = v
        query = query.reshape(batch_size, seq_len, embed_dim)

    q = q_bias_out.reshape(batch_size, seq_len, num_head, head_dim)
    k = k_bias_out.reshape(batch_size, seq_len, num_head, head_dim)
    v = v_bias_out.reshape(batch_size, seq_len, num_head, head_dim)

    # [batch_size, num_head, seq_len, head_dim]
    q = q.transpose((0, 2, 1, 3))
    k = k.transpose((0, 2, 1, 3))
    v = v.transpose((0, 2, 1, 3))

    k = k.transpose([0, 1, 3, 2])  #[batch_size, num_head, head_dim, seq_len]
    qkt = batch_matmul(q, k / np.sqrt(head_dim, dtype=np.float64))

    # if attn_mask is not None:
    #     if attn_mask.dtype.name == 'int64':
    #         attn_mask = (attn_mask.astype(qkt.dtype) - 1.0) * 1e9
    #     else:
    #         attn_mask = attn_mask.astype(qkt.dtype)
    #     qkt += attn_mask

    # softmax
    softmax_out = softmax(qkt)
    attn_heads = batch_matmul(softmax_out, v)

    attn_heads = attn_heads.transpose(
        (0, 2, 1, 3))  # [batch_size, seq_len, num_head, head_dim]

    # out_linear
    out_linear_input = attn_heads.reshape(batch_size, seq_len,
                                          num_head * head_dim)
    out_linear_out = fc(out_linear_input, out_linear_weight)

    # bias add, dropout, residual add, layer_norm.
    if has_bias:
        out_linear_bias_out = out_linear_out + out_linear_bias
    else:
        out_linear_bias_out = out_linear_out

    out_linear_bias_dropout_out = out_linear_bias_out
    out_linear_bias_dropout_residual_out = query + out_linear_bias_dropout_out
    if pre_layer_norm:
        out_linear_bias_dropout_residual_ln_out = out_linear_bias_dropout_residual_out
    else: 
        out_linear_bias_dropout_residual_ln_out = layer_norm(
            out_linear_bias_dropout_residual_out, True, has_bias, ln_scale, ln_bias)
    #return ln_out, out_linear_out, out_linear_bias_dropout_residual_ln_out
    return out_linear_bias_dropout_residual_ln_out


class TestFusedCudnnAttentionAPI(unittest.TestCase):
    def setUp(self):
        self.setXType()
        self.setInputSize()
        self.setDropoutRate()
        self.setPreLn()
        self.setBiasAttr()
        self.config()
        paddle.set_default_dtype(self.x_type)
        self.generate_input_data()

    def setBiasAttr(self):
        self.bias_attr = None

    def setPreLn(self):
        self.pre_layer_norm = False

    def setXType(self):
        self.x_type = np.float32
        self.atol = 1e-2
    
    def setInputSize(self):
        self.batch_size = 3
        self.query_length = 2
        self.head_dim = 2
        self.num_heads = 2

    def setDropoutRate(self):
        self.dropout_prob = 0.0
        self.attn_dropout_prob = 0.0

    def config(self):
        self.attn_mask_type = np.float64
        self.training = True
        self.need_weight = False

        self.embed_dim = self.head_dim * self.num_heads

        self.weight_attr = None
        self.kdim, self.vdim = self.embed_dim, self.embed_dim
        self.key_length, self.value_length = self.query_length, self.query_length

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

        self.seq_len = np.full((self.batch_size, ), self.query_length, dtype=np.int32)
        self.attn_low_window = np.zeros((self.query_length, ), dtype=np.int32)
        self.attn_high_window = np.full((self.query_length, ), self.query_length, dtype=np.int32)
        

    def run_imperative(self, atol=1e-3):
        fused_attn = FusedCudnnMultiHeadAttention(
            self.embed_dim, self.num_heads, self.dropout_prob,
            self.attn_dropout_prob, self.kdim, self.vdim, self.pre_layer_norm,
            self.need_weight, self.weight_attr, self.bias_attr)

        if self.bias_attr is not False:
            has_bias = True
            fused_attn_ln_bias = fused_attn.ln_bias.numpy()
        else:
            has_bias = False
            fused_attn_ln_bias = None

        out = fused_attn(
            paddle.to_tensor(self.query),
            paddle.to_tensor(self.query),
            paddle.to_tensor(self.query), 
            paddle.to_tensor(self.attn_mask),
            paddle.to_tensor(self.seq_len), 
            paddle.to_tensor(self.attn_low_window, place=paddle.CPUPlace()),
            paddle.to_tensor(self.attn_high_window, place=paddle.CPUPlace()),
            paddle.to_tensor(self.seq_len, place=paddle.CPUPlace()))
            # paddle.to_tensor(self.attn_low_window),
            # paddle.to_tensor(self.attn_high_window),
            # paddle.to_tensor(self.seq_len))
        ref_out = compute_reference(self.pre_layer_norm, self.num_heads, self.query,
                                    self.attn_mask,
                                    fused_attn.ln_scale.numpy(),
                                    fused_attn_ln_bias,
                                    fused_attn.weight.numpy(), has_bias)
        # np.testing.assert_allclose(ref_ln, ln_out, rtol=1e-5, atol=1e-5)
        # np.testing.assert_allclose(ref_out_linear, linear_out, rtol=1e-5, atol=1e-3)
        np.testing.assert_allclose(ref_out, out, rtol=1e-5, atol=atol)


    def run_static(self, atol=1e-3):
        fused_attn = FusedCudnnMultiHeadAttention(
            self.embed_dim, self.num_heads, self.dropout_prob,
            self.attn_dropout_prob, self.kdim, self.vdim, self.pre_layer_norm,
            self.need_weight, self.weight_attr, self.bias_attr)

        x = paddle.static.data(
            name='X',
            shape=[self.batch_size, self.query_length, self.embed_dim],
            dtype=self.x_type)
        seq_len = paddle.static.data(
            name='SeqLen',
            shape=[
                self.batch_size
            ],
            dtype=np.int32)
        attn_low_window = paddle.static.data(
            name='AttnLowWin',
            shape=[
                self.query_length
            ],
            dtype=np.int32)
        attn_high_window = paddle.static.data(
            name='AttnHighWin',
            shape=[
                self.query_length
            ],
            dtype=np.int32)
        seq_len_host = paddle.static.data(
            name='SeqLenHost',
            shape=[
                self.batch_size
            ],
            dtype=np.int32)

        final_out = fused_attn(x, x, x, None, seq_len, attn_low_window, attn_high_window, seq_len_host)

        place = paddle.CUDAPlace(0)
        exe = paddle.static.Executor(place)
        exe.run(paddle.static.default_startup_program())

        compiled_prog = paddle.static.CompiledProgram(
            paddle.static.default_main_program())

        query_tensor = core.LoDTensor()
        query_tensor.set(self.query, core.CUDAPlace(0))
        attn_mask_tensor = core.LoDTensor()
        attn_mask_tensor.set(self.attn_mask, core.CUDAPlace(0))
        seq_len_tensor = core.LoDTensor()
        seq_len_tensor.set(self.seq_len, core.CUDAPlace(0))

        attn_low_window_cpu_tensor = core.LoDTensor()
        attn_low_window_cpu_tensor.set(self.attn_low_window, core.CPUPlace())
        attn_high_window_cpu_tensor = core.LoDTensor()
        attn_high_window_cpu_tensor.set(self.attn_high_window, core.CPUPlace())
        seq_len_cpu_tensor = core.LoDTensor()
        seq_len_cpu_tensor.set(self.seq_len, core.CPUPlace())

        # here, we use paralle executor, so the cpu tensor in feed list won't be transfered to device.
        # if use executor, the cpu tensor will be transfered to device.
        if self.bias_attr is not False:
            final_out, weight, ln_scale, ln_bias = exe.run(
                compiled_prog,
                feed=[{"X": self.query,
                    "SeqLen": self.seq_len,
                    "AttnLowWin": attn_low_window_cpu_tensor,
                    "AttnHighWin": attn_high_window_cpu_tensor,
                    "SeqLenHost": seq_len_cpu_tensor}],
                fetch_list=[
                    final_out, fused_attn.weight, 
                    fused_attn.ln_scale, fused_attn.ln_bias
                ])
        else: 
            ln_bias = None
            final_out, weight, ln_scale = exe.run(
                compiled_prog,
                feed=[{"X": self.query,
                    "SeqLen": self.seq_len,
                    "AttnLowWin": attn_low_window_cpu_tensor,
                    "AttnHighWin": attn_high_window_cpu_tensor,
                    "SeqLenHost": seq_len_cpu_tensor}],
                fetch_list=[
                    final_out, fused_attn.weight, 
                    fused_attn.ln_scale
                ])
        return final_out, weight, ln_scale, ln_bias


    def driver_static_api(self, atol=1e-3):
        if self.bias_attr is not False:
            has_bias = True
        else:
            has_bias = False

        paddle.enable_static()
        with paddle.static.program_guard(Program()):
            out, weight, ln_scale, ln_bias = self.run_static()
            #out = self.run_static()
        ref_out = compute_reference(self.pre_layer_norm, self.num_heads, self.query,
                                    self.attn_mask, ln_scale, ln_bias, weight, has_bias)

        # np.testing.assert_allclose(ref_ln, ln_out, rtol=1e-5, atol=1e-5)
        # np.testing.assert_allclose(ref_linear_out, linear_out, rtol=1e-5, atol=1e-3)
        np.testing.assert_allclose(ref_out, out, rtol=1e-5, atol=atol)


    def driver_dynamic_api(self, atol=1e-3):
        paddle.disable_static(place=paddle.CUDAPlace(0))
        self.run_imperative(atol)

    def test_driver(self):
        self.driver_dynamic_api(self.atol)
        self.driver_static_api(self.atol)


# class TestFusedCudnnAttentionAPIFp16(TestFusedCudnnAttentionAPI):
#     def setXType(self):
#         self.x_type = np.float16
#         self.atol = 1e-1

class TestFusedCudnnAttentionAPIPreLn(TestFusedCudnnAttentionAPI):
    def setPreLn(self):
        self.pre_layer_norm = True


class TestFusedCudnnAttentionAPIBiasIsNone(TestFusedCudnnAttentionAPI):
    def setBiasAttr(self):
        self.bias_attr = False
    

if __name__ == "__main__":
    unittest.main()
