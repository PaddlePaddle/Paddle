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
from paddle.incubate.nn.layer.fused_transformer import FusedMultiHeadAttention
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


def compute_reference(pre_layer_norm, query, attn_mask, ln_scale, ln_bias,
                      ln_2_scale, ln_2_bias, qkv_weight, qkv_bias,
                      out_linear_weight, out_linear_bias):
    batch_size = query.shape[0]
    seq_len = query.shape[1]
    embed_dim = query.shape[2]

    has_bias = True
    if ln_bias is None:
        has_bias = False

    if (pre_layer_norm):
        ln_out = layer_norm(query, True, has_bias, ln_scale, ln_bias)

    num_head = qkv_weight.shape[1]
    head_dim = qkv_weight.shape[2]
    # embed_dim, 3, num_heads, self.head_dim
    qkv_weight = qkv_weight.transpose((3, 0, 1, 2))
    qkv_weight = qkv_weight.reshape(qkv_weight.shape[0], qkv_weight.shape[1] *
                                    qkv_weight.shape[2] * qkv_weight.shape[3])

    if qkv_bias is not None:
        qkv_bias = qkv_bias.reshape(qkv_bias.shape[0] * qkv_bias.shape[1] *
                                    qkv_bias.shape[2])
    if (pre_layer_norm):
        ln_out = ln_out.reshape(batch_size * seq_len, embed_dim)
        qkv = fc(ln_out, qkv_weight)
        if qkv_bias is not None:
            qkv_bias_out = qkv + qkv_bias
        else:
            qkv_bias_out = qkv
        ln_out = ln_out.reshape(batch_size, seq_len, embed_dim)
    else:
        query = query.reshape(batch_size * seq_len, embed_dim)
        qkv = fc(query, qkv_weight)
        if qkv_bias is not None:
            qkv_bias_out = qkv + qkv_bias
        else:
            qkv_bias_out = qkv
        query = query.reshape(batch_size, seq_len, embed_dim)

    qkv_bias_out = qkv_bias_out.reshape(batch_size, seq_len, 3, num_head,
                                        head_dim)
    # q*k^t
    qkv_bias_out = qkv_bias_out.transpose(
        (2, 0, 1, 3, 4))  # 3, batch_size, seq_len, num_head, head_dim
    qkv_bias_out = qkv_bias_out.transpose(
        (0, 1, 3, 2, 4))  # 3, batch_size, num_head, seq_len, head_dim

    q = qkv_bias_out[0:1, ::]
    q = q.reshape(batch_size, num_head, seq_len, head_dim)
    k = qkv_bias_out[1:2, ::]  #[1, batch_size, num_head, seq_len, head_dim]
    k = k.reshape(batch_size, num_head, seq_len, head_dim)
    v = qkv_bias_out[2::]
    v = v.reshape(batch_size, num_head, seq_len, head_dim)

    k = k.transpose([0, 1, 3, 2])  #[batch_size, num_head, head_dim, seq_len]
    qkt = batch_matmul(q, k / np.sqrt(head_dim, dtype=np.float64))

    if attn_mask is not None:
        if attn_mask.dtype.name == 'int64':
            attn_mask = (attn_mask.astype(qkt.dtype) - 1.0) * 1e9
        else:
            attn_mask = attn_mask.astype(qkt.dtype)
        qkt += attn_mask

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
    if out_linear_bias is not None:
        out_linear_bias_out = out_linear_out + out_linear_bias
    else:
        out_linear_bias_out = out_linear_out
    out_linear_bias_dropout_out = out_linear_bias_out
    out_linear_bias_dropout_residual_out = query + out_linear_bias_dropout_out
    if not pre_layer_norm:
        out_linear_bias_dropout_residual_out = layer_norm(
            out_linear_bias_dropout_residual_out, True, has_bias, ln_2_scale,
            ln_2_bias)
    return out_linear_bias_dropout_residual_out


class TestFusedAttentionAPI(unittest.TestCase):
    def setUp(self):
        self.setXType()
        self.setPreLn()
        self.setAttnMask()
        self.setBiasAttr()
        self.config()
        self.generate_input_data()

    def setAttnMask(self):
        self.has_attn_mask = True

    def setBiasAttr(self):
        self.bias_attr = None

    def setPreLn(self):
        self.pre_layer_norm = False

    def setXType(self):
        self.x_type = np.float32

    def config(self):
        self.attn_mask_type = np.float64
        self.training = True
        self.need_weight = False

        self.batch_size = 1
        self.query_length = 2
        self.head_dim = 2
        self.num_heads = 2
        self.embed_dim = self.head_dim * self.num_heads

        self.dropout_prob = 0.0
        self.attn_dropout_prob = 0.0
        self.weight_attr = None

        self.kdim, self.vdim = self.embed_dim, self.embed_dim
        self.key_length, self.value_length = self.query_length, self.query_length

    def generate_input_data(self):
        self.query = np.random.rand(self.batch_size, self.query_length,
                                    self.embed_dim).astype(self.x_type)
        if self.has_attn_mask:
            self.attn_mask = np.ones(
                (self.batch_size, self.num_heads, self.query_length,
                 self.key_length),
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

    def run_imperative(self):
        if self.has_attn_mask:
            attn_mask_tensor = paddle.to_tensor(self.attn_mask)
        else:
            attn_mask_tensor = None
        fused_attn = FusedMultiHeadAttention(
            self.embed_dim, self.num_heads, self.dropout_prob,
            self.attn_dropout_prob, self.kdim, self.vdim, self.pre_layer_norm,
            self.need_weight, self.weight_attr, self.bias_attr)
        if self.bias_attr is not False:
            qkv_bias = np.random.random(fused_attn.qkv_bias.shape).astype(
                'float32')
            fused_attn.qkv_bias.set_value(paddle.to_tensor(qkv_bias))
        out = fused_attn(
            paddle.to_tensor(self.query),
            paddle.to_tensor(self.query),
            paddle.to_tensor(self.query), attn_mask_tensor)

        fused_attn_qkv_bias = None
        fused_attn_linear_bias = None
        fused_attn_pre_ln_bias = None
        fused_attn_ln_bias = None
        if self.bias_attr is not False:
            fused_attn_qkv_bias = fused_attn.qkv_bias.numpy()
            fused_attn_linear_bias = fused_attn.linear_bias.numpy()
            fused_attn_pre_ln_bias = fused_attn.pre_ln_bias.numpy()
            fused_attn_ln_bias = fused_attn.ln_bias.numpy()

        ref_out = compute_reference(
            self.pre_layer_norm, self.query, self.attn_mask,
            fused_attn.pre_ln_scale.numpy(), fused_attn_pre_ln_bias,
            fused_attn.ln_scale.numpy(), fused_attn_ln_bias,
            fused_attn.qkv_weight.numpy(), fused_attn_qkv_bias,
            fused_attn.linear_weight.numpy(), fused_attn_linear_bias)
        np.testing.assert_allclose(ref_out, out.numpy(), rtol=1e-5, atol=1e-4)

    def run_static(self):
        fused_attn = FusedMultiHeadAttention(
            self.embed_dim, self.num_heads, self.dropout_prob,
            self.attn_dropout_prob, self.kdim, self.vdim, self.pre_layer_norm,
            self.need_weight, self.weight_attr, self.bias_attr)

        x = paddle.static.data(
            name='X',
            shape=[self.batch_size, self.query_length, self.embed_dim],
            dtype=self.x_type)
        if self.has_attn_mask:
            attn_mask = paddle.static.data(
                name='SrcMask',
                shape=[
                    self.batch_size, self.num_heads, self.query_length,
                    self.key_length
                ],
                dtype=self.attn_mask_type)
            final_out = fused_attn(x, x, x, attn_mask)
        else:
            final_out = fused_attn(x, x, x)

        place = paddle.CUDAPlace(0)
        exe = paddle.static.Executor(place)
        exe.run(paddle.static.default_startup_program())

        qkv_bias = None
        linear_bias = None
        ln_bias = None
        ln_2_bias = None
        if self.has_attn_mask:
            if self.bias_attr is False:
                out, qkv_weight, out_linear_weight, ln_scale, ln_2_scale = exe.run(
                    paddle.static.default_main_program(),
                    feed={"X": self.query,
                          "SrcMask": self.attn_mask},
                    fetch_list=[
                        final_out, fused_attn.qkv_weight,
                        fused_attn.linear_weight, fused_attn.pre_ln_scale,
                        fused_attn.ln_scale
                    ])
            else:
                out, qkv_weight, qkv_bias, out_linear_weight, linear_bias, ln_scale, ln_bias, ln_2_scale, ln_2_bias = exe.run(
                    paddle.static.default_main_program(),
                    feed={"X": self.query,
                          "SrcMask": self.attn_mask},
                    fetch_list=[
                        final_out, fused_attn.qkv_weight, fused_attn.qkv_bias,
                        fused_attn.linear_weight, fused_attn.linear_bias,
                        fused_attn.pre_ln_scale, fused_attn.pre_ln_bias,
                        fused_attn.ln_scale, fused_attn.ln_bias
                    ])
        else:
            if self.bias_attr is False:
                out, qkv_weight, out_linear_weight, ln_scale, ln_2_scale = exe.run(
                    paddle.static.default_main_program(),
                    feed={"X": self.query, },
                    fetch_list=[
                        final_out, fused_attn.qkv_weight,
                        fused_attn.linear_weight, fused_attn.pre_ln_scale,
                        fused_attn.ln_scale
                    ])
            else:
                out, qkv_weight, qkv_bias, out_linear_weight, linear_bias, ln_scale, ln_bias, ln_2_scale, ln_2_bias = exe.run(
                    paddle.static.default_main_program(),
                    feed={"X": self.query, },
                    fetch_list=[
                        final_out, fused_attn.qkv_weight, fused_attn.qkv_bias,
                        fused_attn.linear_weight, fused_attn.linear_bias,
                        fused_attn.pre_ln_scale, fused_attn.pre_ln_bias,
                        fused_attn.ln_scale, fused_attn.ln_bias
                    ])
        return out, qkv_weight, qkv_bias, out_linear_weight, linear_bias, ln_scale, ln_bias, ln_2_scale, ln_2_bias

    def test_static_api(self):
        paddle.enable_static()
        with paddle.static.program_guard(Program()):
            out, qkv_weight, qkv_bias, linear_weight, linear_bias, ln_scale, ln_bias, ln_2_scale, ln_2_bias = self.run_static(
            )
        ref_out = compute_reference(self.pre_layer_norm, self.query,
                                    self.attn_mask, ln_scale, ln_bias,
                                    ln_2_scale, ln_2_bias, qkv_weight, qkv_bias,
                                    linear_weight, linear_bias)
        np.testing.assert_allclose(ref_out, out, rtol=1e-5, atol=1e-4)

    def test_dynamic_api(self):
        paddle.disable_static(place=paddle.CUDAPlace(0))
        self.run_imperative()


class TestFusedAttentionAPINoneAttnMask(TestFusedAttentionAPI):
    def setAttnMask(self):
        self.has_attn_mask = False

    def setPreLn(self):
        self.pre_layer_norm = True


class TestFusedAttentionAPIBiasIsNone(TestFusedAttentionAPI):
    def setBiasAttr(self):
        self.bias_attr = False


if __name__ == "__main__":
    unittest.main()
