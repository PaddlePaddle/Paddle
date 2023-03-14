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

import paddle
from paddle import _C_ops
from paddle.fluid.framework import in_dygraph_mode
from paddle.fluid.layer_helper import LayerHelper

from .attn_bias import (
    BlockDiagonalCausalMask,
    BlockDiagonalCausalWithOffsetPaddedKeysMask,
    BlockDiagonalMask,
    LowerTriangularMask,
    LowerTriangularMaskWithTensorBias,
)

SUPPORTED_ATTN_BIAS_TYPES = {
    type(None),
    paddle.Tensor,
    LowerTriangularMask,
    LowerTriangularMaskWithTensorBias,
    BlockDiagonalMask,
    BlockDiagonalCausalMask,
    BlockDiagonalCausalWithOffsetPaddedKeysMask,
}


def _get_seqlen_info(attn_bias):
    if isinstance(
        attn_bias,
        (BlockDiagonalMask, BlockDiagonalCausalWithOffsetPaddedKeysMask),
    ):
        return (
            attn_bias.k_seqinfo.seqstart,
            attn_bias.q_seqinfo.seqstart,
            attn_bias.q_seqinfo.max_seqlen,
            attn_bias.k_seqinfo.max_seqlen,
        )
    else:
        return None, None, -1, -1


def _get_tensor_bias(attn_bias):
    if isinstance(attn_bias, paddle.Tensor):
        return attn_bias
    elif isinstance(attn_bias, LowerTriangularMaskWithTensorBias):
        return attn_bias._bias
    else:
        return None


def cutlass_fused_multi_head_attention(
    query,
    key,
    value,
    bias=None,
    cu_seqlens_q=None,
    cu_seqlens_k=None,
    seqstart_q=None,
    seqstart_k=None,
    causal_diagonal=None,
    seqlen_k=None,
    max_seqlen_q=0,
    max_seqlen_k=0,
    causal=False,
    dropout_p=0.0,
    scale=0.0,
    is_test=False,
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

    if in_dygraph_mode():
        return _C_ops.cutlass_fused_multihead_attention(
            query,
            key,
            value,
            bias,
            cu_seqlens_q,
            cu_seqlens_k,
            seqstart_q,
            seqstart_k,
            causal_diagonal,
            seqlen_k,
            max_seqlen_q,
            max_seqlen_k,
            causal,
            dropout_p,
            scale,
            is_test,
        )

    helper = LayerHelper('cutlass_fused_multihead_attention', **locals())
    output = helper.create_variable_for_type_inference(dtype=query.dtype)
    logsumexp = helper.create_variable_for_type_inference(dtype='float')
    seed_and_offset = helper.create_variable_for_type_inference(dtype='int32')
    helper.append_op(
        type='cutlass_fused_multihead_attention',
        inputs={
            'query': query,
            'key': key,
            'value': value,
            'bias': bias,
            "cu_seqlens_q": cu_seqlens_q,
            "cu_seqlens_k": cu_seqlens_k,
            "seqstart_q": seqstart_q,
            "seqstart_k": seqstart_k,
            "causal_diagonal": causal_diagonal,
            "seqlen_k": seqlen_k,
        },
        args={
            "max_seqlen_q": max_seqlen_q,
            "max_seqlen_k": max_seqlen_k,
            "causal": causal,
            "dropout_p": dropout_p,
            "scale": scale,
            "is_test": is_test,
        },
        outputs={
            'output': output,
            'logsumexp': logsumexp,
            "seed_and_offset": seed_and_offset,
        },
    )
    return output, logsumexp, seed_and_offset


def memory_efficient_attention(
    query, key, value, attn_bias, p=0.0, scale=None, training=True
):
    assert type(attn_bias) in SUPPORTED_ATTN_BIAS_TYPES
    causal = isinstance(
        attn_bias,
        (
            LowerTriangularMask,
            BlockDiagonalCausalMask,
            BlockDiagonalCausalWithOffsetPaddedKeysMask,
        ),
    )
    seqstart_k, seqstart_q, max_seqlen_q, max_seqlen_k = _get_seqlen_info(
        attn_bias
    )
    # NOTE: compute_logsumexp = training
    causal_diagonal = (
        attn_bias.causal_diagonal
        if isinstance(attn_bias, BlockDiagonalCausalWithOffsetPaddedKeysMask)
        else None
    )
    seqlen_k = (
        attn_bias.k_seqinfo.seqlen
        if isinstance(attn_bias, BlockDiagonalCausalWithOffsetPaddedKeysMask)
        else None
    )

    bias = _get_tensor_bias(attn_bias)

    output, logsumexp, seed_and_offset = cutlass_fused_multi_head_attention(
        query=query,
        key=key,
        value=value,
        bias=bias,
        cu_seqlens_q=seqstart_q,
        cu_seqlens_k=seqstart_k,
        seqstart_q=seqstart_q,
        seqstart_k=seqstart_k,
        causal_diagonal=causal_diagonal,
        seqlen_k=seqlen_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        causal=causal,
        dropout_p=p,
        scale=scale,
        is_test=not training,
    )

    return output, logsumexp, seed_and_offset
