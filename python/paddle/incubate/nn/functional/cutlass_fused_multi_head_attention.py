# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import math

from paddle import _C_ops
from paddle.fluid.framework import in_dygraph_mode
from paddle.fluid.layer_helper import LayerHelper


def cutlass_fused_multi_head_attention(
    query, key, value, mask=None, scale=None, causal=False
):
    """
    Cutlass Fused Multihead Attention.
    This method requires SM_ARCH in sm70, sm75, sm80.

    Args:
        query (Tensor): the Query Tensor. Its shape is [batchsize, seq_len, num_head, head_size].
        key (Tensor): the Key Tensor. Its shape is [batchsize, seq_len, num_head, head_size].
        value (Tensor): the Value Tensor. Its shape is [batchsize, seq_len, num_head, head_size].
        mask (Tensor): the Mask Tensor. Its shape is [batchsize, seq_len, num_head, seq_len]. And it can broadcast in each dims (which means you can set dimsize=1).
        scale (Float): the attention matrix's scale. Default is sqrt(1.0 / head_size).
        causal (Bool): whether causal masking is used or not. Default is False.
    Returns:
        Tensor: the output Tensor.

    Examples:
        .. code-block:: python

            # required: gpu
            import math
            import paddle
            from paddle.incubate.nn.functional import cutlass_fused_multi_head_attention

            batch = 1
            num_head = 8
            seq_len = 256
            head_size = 32

            dtype = paddle.float16

            query = paddle.randn([batch, seq_len, num_head, head_size], dtype=dtype)
            key = paddle.randn([batch, seq_len, num_head, head_size], dtype=dtype)
            value = paddle.randn([batch, seq_len, num_head, head_size], dtype=dtype)
            mask = paddle.randn([1, 1, 1, seq_len], dtype=dtype)

            scale = float(1.0 / math.sqrt(head_size))

            def naive_attention_impl(query, key, value, mask, scale):
                query = paddle.transpose(query, [0, 2, 1, 3])
                key = paddle.transpose(key, [0, 2, 1, 3])
                value = paddle.transpose(value, [0, 2, 1, 3])

                qk_res = paddle.matmul(query, key, transpose_y=True)
                attention = qk_res * scale
                attention = attention + mask
                softmax_result = paddle.nn.functional.softmax(attention, -1)
                result = paddle.matmul(softmax_result, value)
                result = paddle.transpose(result, [0, 2, 1, 3])
                return result

            out = naive_attention_impl(query, key, value, mask, scale)
            # equals to: out = cutlass_fused_multi_head_attention(query, key, value, mask, scale)

            print(out.shape) # [batch, seq_len, num_head, head_size]
    """
    if scale is None:
        head_size = query.shape[3]
        scale = float(1.0 / math.sqrt(head_size))

    if in_dygraph_mode():
        return _C_ops.cutlass_fused_multihead_attention(
            query, key, value, mask, scale, causal
        )

    helper = LayerHelper('cutlass_fused_multihead_attention', **locals())
    out = helper.create_variable_for_type_inference(dtype=query.dtype)
    seed_and_offset = helper.create_variable_for_type_inference(dtype='uint64')
    helper.append_op(
        type='cutlass_fused_multihead_attention',
        inputs={'Query': query, 'Key': key, 'Value': value, 'Mask': mask},
        attrs={"Scale": scale, "Causal": causal}, 
        outputs={'Out': out, "Seed_and_Offset": seed_and_offset}, 
    )

    return out, seed_and_offset
