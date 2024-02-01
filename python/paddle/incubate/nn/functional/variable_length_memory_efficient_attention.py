# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

# The following codes are from https://github.com/facebookresearch/xformers

# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import math

from paddle import _C_ops
from paddle.framework import LayerHelper, in_dynamic_or_pir_mode


def variable_length_memory_efficient_attention(
    query,
    key,
    value,
    seq_lens,
    kv_seq_lens,
    mask=None,
    scale=None,
    causal=False,
    pre_cache_length=0,
):
    """
    Cutlass Memory Efficient Variable Attention.
    This method requires SM_ARCH in sm70, sm75, sm80.

    Args:
        query (Tensor): The Query Tensor. Its shape is [batchsize, num_head, seq_len, head_size].
        key (Tensor): The Key Tensor. Its shape is [batchsize, num_head, seq_len, head_size].
        value (Tensor): The Value Tensor. Its shape is [batchsize, num_head, seq_len, head_size].
        seq_lens (Tensor): The sequence lengths of the sequences in the batch, used to index query. Its shape is [batchsize, 1].
        kv_seq_lens (Tensor): The sequence lengths of the sequences in the batch, used to index key and value. Its shape is [batchsize, 1].
        mask (Tensor): The Mask Tensor. Its shape is [batchsize, 1, query_seq_len, key_seq_len].
        scale (Float): The attention matrix's scale. Default is sqrt(1.0 / head_size).
        causal (Bool): Whether causal masking is used or not. Default is False.
        pre_cache_length (Int): The length of the pre-cache. Default is 0.
    Returns:
        Tensor: the output Tensor.

    Examples:
        .. code-block:: python

            >>> # doctest: +REQUIRES(env:GPU)
            >>> import math
            >>> import paddle
            >>> from paddle.incubate.nn.functional import variable_length_memory_efficient_attention
            >>> paddle.device.set_device('gpu')

            >>> batch = 1
            >>> num_head = 8
            >>> seq_len = 256
            >>> head_size = 32

            >>> dtype = paddle.float16

            >>> query = paddle.randn([batch, num_head, seq_len, head_size], dtype=dtype)
            >>> key = paddle.randn([batch, num_head, seq_len, head_size], dtype=dtype)
            >>> value = paddle.randn([batch, num_head, seq_len, head_size], dtype=dtype)
            >>> seq_lens = paddle.to_tensor([seq_len, ] * batch, dtype='int32')
            >>> mask = paddle.randn([batch, 1, seq_len, seq_len], dtype=dtype)

            >>> scale = float(1.0 / math.sqrt(head_size))
            >>> pre_cache_length = 0

            >>> def naive_attention_impl(query, key, value, mask, scale):
            ...     qk_res = paddle.matmul(query, key, transpose_y=True)
            ...     attention = qk_res * scale
            ...     attention = attention + mask
            ...     softmax_result = paddle.nn.functional.softmax(attention, -1)
            ...     result = paddle.matmul(softmax_result, value)
            ...     return result

            >>> out = naive_attention_impl(query, key, value, mask, scale)
            >>> # equals to: out = variable_length_memory_efficient_attention(query, key, value, seq_lens, seq_lens, mask, scale, pre_cache_length)

            >>> out.shape # [batch, num_head, seq_len, head_size]
            [1, 8, 256, 32]
    """
    if scale is None:
        head_size = query.shape[3]
        scale = float(1.0 / math.sqrt(head_size))

    if in_dynamic_or_pir_mode():
        return _C_ops.variable_length_memory_efficient_attention(
            query,
            key,
            value,
            seq_lens,
            kv_seq_lens,
            mask,
            scale,
            causal,
            pre_cache_length,
        )

    helper = LayerHelper(
        'variable_length_memory_efficient_attention', **locals()
    )
    out = helper.create_variable_for_type_inference(dtype=query.dtype)
    helper.append_op(
        type='variable_length_memory_efficient_attention',
        inputs={
            'query': query,
            'key': key,
            'value': value,
            'seq_lens': seq_lens,
            'kv_seq_lens': kv_seq_lens,
            "mask": mask,
        },
        attrs={
            "scale": scale,
            "causal": causal,
            "pre_cache_length": pre_cache_length,
        },
        outputs={'out': out},
    )
    return out
