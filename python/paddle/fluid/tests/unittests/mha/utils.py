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

import numpy as np


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
