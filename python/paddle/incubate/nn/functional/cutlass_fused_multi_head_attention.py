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
    query, key, value, mask=None, scale=0.0, causal=False, dropout_p=0.0
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
            # equals to: out = cutlass_fused_multi_head_attention(query, key, value, mask, scale, causal, dropout_p)

            print(out.shape) # [batch, seq_len, num_head, head_size]
    """
    if scale is None:
        head_size = query.shape[3]
        scale = float(1.0 / math.sqrt(head_size))

    if in_dygraph_mode():
        return _C_ops.cutlass_fused_multihead_attention(
            query, key, value, mask, scale, causal, dropout_p
        )

    helper = LayerHelper('cutlass_fused_multihead_attention', **locals())
    out = helper.create_variable_for_type_inference(dtype=query.dtype)
    # seed_and_offset = helper.create_variable_for_type_inference(dtype='uint64')
    seed_and_offset = helper.create_variable_for_type_inference(dtype='int32')
    helper.append_op(
        type='cutlass_fused_multihead_attention',
        inputs={'query': query, 'key': key, 'value': value, 'mask': mask},
        attrs = {"scale": scale, "causal": causal, "dropout_p": dropout_p}, 
        # inputs={'query': query, 'key': key, 'value': value, 'mask': mask, "scale": scale,
        #              "causal": causal, "dropout_p": dropout_p}, 
        outputs={'out': out, "seed_and_offset": seed_and_offset}, 
    )
    # helper.append_op(
    #     type='cutlass_fused_multihead_attention',
    #     inputs={'Query': query, 'Key': key, 'Value': value, 'Mask': mask},
    #     attrs={"Scale": scale, "Causal": causal, "Dropout_p": dropout_p}, 
    #     outputs={'Out': out, "Seed_and_Offset": seed_and_offset}, 
    # )

    return out, seed_and_offset

def cutlass_fused_multi_head_attention_grad(
    query, key, value, seed_and_offset, out, out_grad, scale=0.0, causal=False, dropout_p=0.0
):
    """
    Cutlass Fused Multihead Attention Grad.
    This method requires SM_ARCH in sm70, sm75, sm80.

    Args:
        query (Tensor): the Query Tensor. Its shape is [batchsize, seq_len, num_head, head_size].
        key (Tensor): the Key Tensor. Its shape is [batchsize, seq_len, num_head, head_size].
        value (Tensor): the Value Tensor. Its shape is [batchsize, seq_len, num_head, head_size].
        seed_and_offset (Tensor): the seed and offset of dropout random generation.
        out (Tensor): output tensor of forward.
        out_grad (Tensor): grad of output.
        causal (Bool): whether causal masking is used or not. Default is False.
        scale (Float): the attention matrix's scale. Default is sqrt(1.0 / head_size).
        dropout (Float): the probability of dropping out whole attn mask. Default is 0.0.
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
            seed_and_offset = paddle.randn([2], dtype=uint64_t)
            out = paddle.randn([batch, seq_len, num_head, head_size], dtype=dtype)
            out_grads = paddle.randn([batch, seq_len, num_head, head_size], dtype=dtype)

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
        return _C_ops.cutlass_fused_multihead_attention_grad(
            query, key, value, seed_and_offset, out, out_grad, scale, causal, dropout_p
        )

    helper = LayerHelper('cutlass_fused_multihead_attention_grad', **locals())
    query_grad = helper.create_variable_for_type_inference(dtype=query.dtype)
    key_grad = helper.create_variable_for_type_inference(dtype=query.dtype)
    value_grad = helper.create_variable_for_type_inference(dtype=query.dtype)
    helper.append_op(
        type='cutlass_fused_multihead_attention_grad',
        inputs={'query': query, 'key': key, 'value': value, 'seed_and_offset': seed_and_offset,
                'out': out, 'out_grad': out_grad},
        attrs={"scale": scale, "causal": causal, "dropout_p": dropout_p}, 
        # inputs={'query': query, 'key': key, 'value': value, 'seed_and_offset': seed_and_offset,
        #         'out': out, 'out_grad': out_grad, "scale": scale, "causal": causal, "dropout_p": dropout_p}, 
        outputs={'query_grad': query_grad, "key_grad": key_grad, "value_grad": value_grad}, 
    )

    return query_grad, key_grad, value_grad
