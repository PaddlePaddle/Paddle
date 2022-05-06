# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
# Copyright (c) 2022 NVIDIA Corporation. All rights reserved.
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

from __future__ import print_function

import paddle
from paddle.fluid.data_feeder import convert_dtype
from paddle.nn import Linear
import numpy as np


def skip_unit_test():
    cudnn_version = paddle.device.get_cudnn_version()
    if ((cudnn_version >= 8300) and paddle.device.is_compiled_with_cuda()):
        return False
    return True


def get_dtype_str(dtype):
    if dtype == np.float16:
        return 'FP16'
    else:
        return 'FP32'


def compare(ref, res, atol, rtol):

    ref = ref.flatten()
    res = res.flatten()

    tmp_ref = ref.astype(np.float)
    tol = atol + rtol * abs(tmp_ref)

    diff = abs(res - ref)

    indices = np.transpose(np.where(diff > tol))
    if len(indices) == 0:
        return True
    return False


def split_weight_into_legacy_format(weight, embed_dim, with_bias):
    param_shape = (embed_dim, embed_dim)
    stride = embed_dim * embed_dim
    q_proj_weight = weight[:stride].reshape(param_shape)
    k_proj_weight = weight[stride:2 * stride].reshape(param_shape)
    v_proj_weight = weight[2 * stride:3 * stride].reshape(param_shape)
    out_proj_weight = weight[3 * stride:4 * stride].reshape(param_shape)
    if with_bias:
        bias_start = 4 * stride
        q_proj_bias = weight[bias_start:bias_start + embed_dim]
        k_proj_bias = weight[bias_start + embed_dim:bias_start + 2 * embed_dim]
        v_proj_bias = weight[bias_start + 2 * embed_dim:bias_start + 3 *
                             embed_dim]
        out_proj_bias = weight[bias_start + 3 * embed_dim:]
    else:
        q_proj_bias = np.zeros((embed_dim, ))
        k_proj_bias = np.zeros((embed_dim, ))
        v_proj_bias = np.zeros((embed_dim, ))
        out_proj_bias = np.zeros((embed_dim, ))

    return q_proj_weight, q_proj_bias, \
            k_proj_weight, k_proj_bias, \
            v_proj_weight, v_proj_bias, \
            out_proj_weight, out_proj_bias


def generate_weight(embed_dim, dtype, with_bias=True):
    weight = np.random.uniform(
        low=-0.03, high=0.03, size=(4 * embed_dim * embed_dim)).astype(dtype)
    if with_bias:
        weight = np.concatenate(
            (weight, np.random.uniform(
                low=-0.01, high=0.01, size=(4 * embed_dim, ))),
            dtype=dtype)
    q_proj_weight, q_proj_bias, \
    k_proj_weight, k_proj_bias, \
    v_proj_weight, v_proj_bias, \
    out_proj_weight, out_proj_bias = \
        split_weight_into_legacy_format(weight, embed_dim, with_bias)
    return weight,  q_proj_weight, q_proj_bias, \
           k_proj_weight, k_proj_bias, \
           v_proj_weight, v_proj_bias, \
           out_proj_weight, out_proj_bias


def generate_data(batch_size, max_seqlen, embed_dim, dtype, low=-0.5, high=0.5):
    assert batch_size > 0, "batch_size should be greater than 0"
    assert max_seqlen > 0, "max_seqlen should be greater than 0"
    query = np.random.uniform(
        low=low, high=high,
        size=(batch_size, max_seqlen, embed_dim)).astype(dtype)
    key = np.random.uniform(
        low=low, high=high,
        size=(batch_size, max_seqlen, embed_dim)).astype(dtype)
    value = np.random.uniform(
        low=low, high=high,
        size=(batch_size, max_seqlen, embed_dim)).astype(dtype)

    return query, key, value


def generate_varlen_data(seqlens, embed_dim, dtype, low=-0.5, high=0.5):
    assert len(seqlens) > 0, "batch size should be greater than 0"

    querys = [
        np.random.uniform(
            low=low, high=high, size=(1, seq_len, embed_dim)).astype(dtype)
        for seq_len in seqlens
    ]
    keys = [
        np.random.uniform(
            low=low, high=high, size=(1, seq_len, embed_dim)).astype(dtype)
        for seq_len in seqlens
    ]
    values = [
        np.random.uniform(
            low=low, high=high, size=(1, seq_len, embed_dim)).astype(dtype)
        for seq_len in seqlens
    ]

    query = np.concatenate(querys, axis=1)
    key = np.concatenate(keys, axis=1)
    value = np.concatenate(values, axis=1)

    return query, key, value


class MultiHeadAttentionRef(paddle.nn.Layer):
    def __init__(self,
                 embed_dim,
                 num_heads,
                 dropout=0.,
                 kdim=None,
                 vdim=None,
                 weight_attr=None,
                 bias_attr=None):
        super(MultiHeadAttentionRef, self).__init__()

        assert embed_dim > 0, ("Expected embed_dim to be greater than 0, "
                               "but recieved {}".format(embed_dim))
        assert num_heads > 0, ("Expected num_heads to be greater than 0, "
                               "but recieved {}".format(num_heads))

        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.num_heads = num_heads
        self.dropout = dropout

        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.q_proj = Linear(
            embed_dim, embed_dim, weight_attr, bias_attr=bias_attr)
        self.k_proj = Linear(
            self.kdim, embed_dim, weight_attr, bias_attr=bias_attr)
        self.v_proj = Linear(
            self.vdim, embed_dim, weight_attr, bias_attr=bias_attr)
        self.out_proj = Linear(
            embed_dim, embed_dim, weight_attr, bias_attr=bias_attr)

    def _convert_attention_mask(self, attn_mask, dtype):
        if attn_mask is not None and attn_mask.dtype != dtype:
            attn_mask_dtype = convert_dtype(attn_mask.dtype)
            if attn_mask_dtype == 'bool' or 'int' in attn_mask_dtype:
                attn_mask = (paddle.cast(attn_mask, dtype) - 1.0) * 1e9
            else:
                attn_mask = paddle.cast(attn_mask, dtype)
        return attn_mask

    def _prepare_qkv(self, query, key, value):
        q = self.q_proj(query)
        q = paddle.tensor.reshape(
            x=q, shape=[0, 0, self.num_heads, self.head_dim])
        q = paddle.tensor.transpose(x=q, perm=[0, 2, 1, 3])

        k, v = self.compute_kv(key, value)

        return (q, k, v)

    def compute_kv(self, key, value):
        k = self.k_proj(key)
        v = self.v_proj(value)
        k = paddle.tensor.reshape(
            x=k, shape=[0, 0, self.num_heads, self.head_dim])
        k = paddle.tensor.transpose(x=k, perm=[0, 2, 1, 3])
        v = paddle.tensor.reshape(
            x=v, shape=[0, 0, self.num_heads, self.head_dim])
        v = paddle.tensor.transpose(x=v, perm=[0, 2, 1, 3])
        return k, v

    def forward(self, query, key=None, value=None, attn_mask=None):
        key = query if key is None else key
        value = query if value is None else value

        q, k, v = self._prepare_qkv(query, key, value)

        product = paddle.matmul(
            x=q * (self.head_dim**-0.5), y=k, transpose_y=True)
        if attn_mask is not None:
            attn_mask = self._convert_attention_mask(attn_mask, product.dtype)
            product = product + attn_mask
        weights = paddle.nn.functional.softmax(product)
        if self.dropout:
            weights = F.dropout(
                weights,
                self.dropout,
                training=self.training,
                mode="upscale_in_train")

        out = paddle.tensor.matmul(weights, v)

        out = paddle.tensor.transpose(out, perm=[0, 2, 1, 3])
        out = paddle.tensor.reshape(
            x=out, shape=[0, 0, out.shape[2] * out.shape[3]])
        out = self.out_proj(out)

        return out
