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


def multi_head_attn(q, k, v, weight, meta_data, seq_data_info):

    # if in_dygraph_mode():
    #     pre_bias = _varbase_creator(dtype=x.dtype)
    #     _C_ops.matmul(x, weight, pre_bias, 'transpose_X', False, 'transpose_Y',
    #                   False, "alpha", 1)

    #     if bias is None:
    #         return pre_bias

    #     return _C_ops.elementwise_add(pre_bias, bias)
    # else:
    helper = LayerHelper('mha', **locals())
    dtype = q.dtype

    check_variable_and_dtype(q, 'q', ['float16', 'float32', 'float64'], 'mha')
    check_variable_and_dtype(k, 'k', ['float16', 'float32', 'float64'], 'mha')
    check_variable_and_dtype(v, 'v', ['float16', 'float32', 'float64'], 'mha')
    check_dtype(dtype, 'dtype', ['float16', 'float32', 'float64'], 'mha')

    inputs = {
        'Q': q,
        'K': k,
        'V': v,
        'W': weight,
        'QO_KV_Seqlen': seq_data_info.qo_kv_seqlen
    }

    if seq_data_info.qo_kv_seqlen_host is not None:
        inputs['QO_KV_Seqlen_host'] = seq_data_info.qo_kv_seqlen_host
    if seq_data_info.low_high_windows_host is not None:
        inputs['low_high_windows_host'] = seq_data_info.low_high_windows_host

    attrs = {
        'cache_key': weight.name,
        'attn_dropout_rate': meta_data.dropout_rate,
        'attn_heads': meta_data.nheads,
        'attn_sm_scaler': meta_data.sm_scaler,
        'attn_vec_size': meta_data.hidden_size,
        'attn_q_proj_size': meta_data.proj_size,
        'attn_k_proj_size': meta_data.proj_size,
        'attn_v_proj_size': meta_data.proj_size,
        'attn_o_proj_size': meta_data.hidden_size,
        'attn_max_qo_seq_len': seq_data_info.max_seqlen,
        'attn_max_kv_seq_len': seq_data_info.max_seqlen,
        'attn_beam_size': 1
    }

    output = helper.create_variable_for_type_inference(dtype)
    helper.append_op(
        type='mha', inputs=inputs, outputs={'O': output}, attrs=attrs)
    return output
