#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#   Copyright (c) 2022 NVIDIA Corporation. All rights reserved.
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


def mha_data_prepare(attn_mask):
    helper = LayerHelper('mha_data_prepare', **locals())

    inputs = {'attn_mask': attn_mask}

    qo_kv_seqlen = helper.create_variable_for_type_inference(
        'int32', stop_gradient=True)
    qo_kv_seqlen_host = helper.create_variable_for_type_inference(
        'int32', stop_gradient=True)
    low_high_windows_host = helper.create_variable_for_type_inference(
        'int32', stop_gradient=True)

    outputs = {
        'qo_kv_seqlen': qo_kv_seqlen,
        'qo_kv_seqlen_host': qo_kv_seqlen_host,
        'low_high_windows_host': low_high_windows_host
    }
    helper.append_op(type='mha_data_prepare', inputs=inputs, outputs=outputs)
    return qo_kv_seqlen, qo_kv_seqlen_host, low_high_windows_host


def multi_head_attn(query,
                    key,
                    value,
                    weight,
                    qo_kv_seqlen,
                    qo_kv_seqlen_host,
                    low_high_windows_host,
                    num_heads,
                    embedding_size,
                    proj_size,
                    max_qo_seqlen,
                    max_kv_seqlen,
                    enable_bias=True,
                    residual=None,
                    softmax_scaler=1.,
                    pre_dropout_rate=0.,
                    post_dropout_rate=0.,
                    is_training=True):

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
        'qo_kv_seqlen': qo_kv_seqlen,
        'qo_kv_seqlen_host': qo_kv_seqlen_host,
        'low_high_windows_host': low_high_windows_host
    }

    if residual is not None:
        inputs['residual'] = residual

    attrs = {
        'cache_key': weight.name,
        'pre_dropout_rate': pre_dropout_rate,
        'post_dropout_rate': post_dropout_rate,
        'num_heads': num_heads,
        'softmax_scaler': softmax_scaler,
        'embedding_size': embedding_size,
        'query_proj_size': proj_size,
        'key_proj_size': proj_size,
        'value_proj_size': proj_size,
        'output_proj_size': embedding_size,
        'max_qo_seqlen': max_qo_seqlen,
        'max_kv_seqlen': max_kv_seqlen,
        'is_training': is_training,
        'enable_bias': enable_bias
    }

    output = helper.create_variable_for_type_inference(dtype)
    helper.append_op(
        type='mha', inputs=inputs, outputs={'output': output}, attrs=attrs)
    return output
