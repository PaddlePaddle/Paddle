#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#   Copyright (c) 2021 NVIDIA Corporation. All rights reserved.
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

# TODO: define the classes of Transformer neural network
__all__ = []

import paddle
from paddle.fluid.layer_helper import LayerHelper
from ...fluid.data_feeder import check_variable_and_dtype, check_dtype

import numpy as np


def mha_data_prepare(qo_kv_seqlen, low_high_windows):
    helper = LayerHelper('mha_data_prepare', **locals())

    inputs = {
        'qo_kv_seqlen': qo_kv_seqlen,
        'low_high_windows': low_high_windows
    }

    qo_kv_seqlen_host = helper.create_variable_for_type_inference(
        'int32', stop_gradient=True)
    low_high_windows_host = helper.create_variable_for_type_inference(
        'int32', stop_gradient=True)

    outputs = {
        'qo_kv_seqlen_host': qo_kv_seqlen_host,
        'low_high_windows_host': low_high_windows_host
    }
    helper.append_op(type='mha_data_prepare', inputs=inputs, outputs=outputs)
    return qo_kv_seqlen_host, low_high_windows_host


def multi_head_attn(query,
                    key,
                    value,
                    weight,
                    meta_data,
                    seq_info,
                    is_training=True):

    # if in_dygraph_mode():
    #     pre_bias = _varbase_creator(dtype=x.dtype)
    #     _C_ops.matmul(x, weight, pre_bias, 'transpose_X', False, 'transpose_Y',
    #                   False, "alpha", 1)

    #     if bias is None:
    #         return pre_bias

    #     return _C_ops.elementwise_add(pre_bias, bias)
    # else:
    helper = LayerHelper('mha', **locals())
    dtype = query.dtype

    check_variable_and_dtype(query, 'query', ['float16', 'float32', 'float64'],
                             'mha')
    check_variable_and_dtype(key, 'key', ['float16', 'float32', 'float64'],
                             'mha')
    check_variable_and_dtype(value, 'value', ['float16', 'float32', 'float64'],
                             'mha')
    check_dtype(dtype, 'dtype', ['float16', 'float32', 'float64'], 'mha')

    inputs = {
        'query': query,
        'key': key,
        'value': value,
        'weight': weight,
        'qo_kv_seqlen': seq_info.qo_kv_seqlen
    }

    if seq_info.qo_kv_seqlen_host is not None:
        inputs['qo_kv_seqlen_host'] = seq_info.qo_kv_seqlen_host
    if seq_info.low_high_windows_host is not None:
        inputs['low_high_windows_host'] = seq_info.low_high_windows_host

    attrs = {
        'cache_key': weight.name,
        'pre_dropout_rate': meta_data.pre_dropout_rate,
        'post_dropout_rate': meta_data.post_dropout_rate,
        'num_heads': meta_data.num_heads,
        'softmax_scaler': meta_data.softmax_scaler,
        'embedding_size': meta_data.embed_dim,
        'query_proj_size': meta_data.proj_size,
        'key_proj_size': meta_data.proj_size,
        'value_proj_size': meta_data.proj_size,
        'output_proj_size': meta_data.embed_dim,
        'max_qo_seqlen': seq_info.max_seqlen,
        'max_kv_seqlen': seq_info.max_seqlen,
        'is_training': is_training,
        'enable_bias': meta_data.enable_bias
    }

    output = helper.create_variable_for_type_inference(dtype)
    helper.append_op(
        type='mha', inputs=inputs, outputs={'output': output}, attrs=attrs)
    return output
