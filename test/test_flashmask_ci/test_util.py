# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

import numpy as np
from einops import rearrange, repeat

import paddle


def construct_local_mask(
    seqlen_q,
    seqlen_k,
    window_size=(-1, -1),  # -1 means infinite window size
    sink_token_length=0,
    query_padding_mask=None,
    key_padding_mask=None,
    key_leftpad=None,
):
    row_idx = rearrange(paddle.arange(seqlen_q, dtype=paddle.int64), "s -> s 1")
    col_idx = paddle.arange(seqlen_k, dtype=paddle.int64)
    if key_leftpad is not None:
        key_leftpad = rearrange(key_leftpad, "b -> b 1 1 1")
        col_idx = repeat(col_idx, "s -> b 1 1 s", b=key_leftpad.shape[0])
        col_idx = paddle.where(
            col_idx >= key_leftpad, col_idx - key_leftpad, 2**32
        )
    sk = (
        seqlen_k
        if key_padding_mask is None
        else rearrange(key_padding_mask.sum(-1), "b -> b 1 1 1")
    )
    sq = (
        seqlen_q
        if query_padding_mask is None
        else rearrange(query_padding_mask.sum(-1), "b -> b 1 1 1")
    )
    if window_size[0] < 0:
        return col_idx > row_idx + sk - sq + window_size[1]
    else:
        sk = (
            paddle.full_like(col_idx, seqlen_k)
            if key_padding_mask is None
            else sk
        )
        return paddle.logical_or(
            col_idx > paddle.minimum(row_idx + sk - sq + window_size[1], sk),
            paddle.logical_and(
                col_idx < row_idx + sk - sq - window_size[0],
                col_idx >= sink_token_length,
            ),
        )


def attention_ref(
    q,
    k,
    v,
    query_padding_mask=None,
    key_padding_mask=None,
    key_leftpad=None,
    attn_bias=None,
    dropout_p=0.0,
    dropout_mask=None,
    causal=False,
    qv=None,
    q_descale=None,
    k_descale=None,
    v_descale=None,
    window_size=(-1, -1),  # -1 means infinite window size
    attention_chunk=0,
    sink_token_length=0,
    softcap=0.0,
    upcast=True,
    reorder_ops=False,
    intermediate_dtype=None,
):
    """
    Arguments:
        q: (batch_size, seqlen_q, nheads, head_dim)
        k: (batch_size, seqlen_k, nheads, head_dim)
        v: (batch_size, seqlen_k, nheads, head_dim_v)
        qv: (batch_size, seqlen_q, nheads, head_dim_v)
        query_padding_mask: (batch_size, seqlen_q)
        key_padding_mask: (batch_size, seqlen_k)
        attn_bias: broadcastable to (batch_size, nheads, seqlen_q, seqlen_k)
        dropout_p: float
        dropout_mask: (batch_size, nheads, seqlen_q, seqlen_k)
        causal: whether to apply causal masking
        upcast: whether to cast all inputs to fp32, do all computation in fp32, then cast
            output back to fp16/bf16.
        reorder_ops: whether to change the order of operations (scaling k instead of scaling k, etc.)
            without changing the math. This is to estimate the numerical error from operation
            reordering.
    Output:
        output: (batch_size, seqlen_q, nheads, head_dim_v)
        attention: (batch_size, nheads, seqlen_q, seqlen_k), softmax after dropout
    """
    if causal:
        window_size = (window_size[0], 0)
    dtype_og = q.dtype
    if upcast:
        q = paddle.cast(q, paddle.float32)
        k = paddle.cast(k, paddle.float32)
        v = paddle.cast(v, paddle.float32)
        if qv is not None:
            qv = paddle.cast(qv, paddle.float32)

    if q_descale is not None:
        raise AssertionError
        q_descale = repeat(
            q_descale, "b h -> b 1 (h g) 1", g=q.shape[2] // k.shape[2]
        )
        q = (q.cast(paddle.float32) * q_descale).cast(q.dtype)
        qv = (
            (qv.cast(paddle.float32) * q_descale).cast(qv.dtype)
            if qv is not None
            else None
        )

    if k_descale is not None:
        raise AssertionError
        k = (
            k.cast(paddle.float32) * rearrange(k_descale, "b h -> b 1 h 1")
        ).cast(k.dtype)

    if v_descale is not None:
        raise AssertionError
        v = (
            v.cast(paddle.float32) * rearrange(v_descale, "b h -> b 1 h 1")
        ).cast(v.dtype)

    seqlen_q, seqlen_k = q.shape[1], k.shape[1]

    # (batch_size, seqlen, nheads, head_dim) -> (batch_size, nheads, seqlen, head_dim)
    q = paddle.transpose(q, [0, 2, 1, 3])
    k = paddle.transpose(k, [0, 2, 1, 3])
    v = paddle.transpose(v, [0, 2, 1, 3])

    k = repeat(k, "b h s d -> b (h g) s d", g=q.shape[1] // k.shape[1])
    v = repeat(v, "b h s d -> b (h g) s d", g=q.shape[1] // v.shape[1])
    d = q.shape[-1]
    dv = v.shape[-1]
    softmax_scale = 1.0 / math.sqrt(d if qv is None else d + dv)

    if not reorder_ops:
        scores = paddle.matmul(q * softmax_scale, k, transpose_y=True)
    else:
        scores = paddle.matmul(q, k * softmax_scale, transpose_y=True)

    if qv is not None:
        raise AssertionError
        scores = scores + paddle.matmul(qv * softmax_scale, v, transpose_y=True)

    if softcap > 0:
        raise AssertionError
        scores = paddle.tanh(scores / softcap) * softcap

    if key_padding_mask is not None:
        raise AssertionError
        scores.masked_fill_(
            rearrange(~key_padding_mask, "b s -> b 1 1 s"), float("-inf")
        )
    local_mask = None

    if window_size[0] >= 0 or window_size[1] >= 0:
        local_mask = construct_local_mask(
            seqlen_q,
            seqlen_k,
            window_size,
            sink_token_length,
            query_padding_mask,
            key_padding_mask,
            key_leftpad=key_leftpad,
        )
    if attention_chunk > 0:
        raise AssertionError

    if local_mask is not None:
        scores.masked_fill_(local_mask, float("-inf"))
    if attn_bias is not None:
        scores = scores + attn_bias
        # when all values in a line of attn_bias are -inf, setting value in this line to a very small value
        # to prevent softmax giving nan output
        all_inf_mask = (attn_bias == -np.inf).all(axis=-1, keepdim=True)
        scores = paddle.where(
            all_inf_mask, paddle.full_like(scores, -1e9), scores
        )

    attention = paddle.nn.functional.softmax(scores, axis=-1).cast(v.dtype)

    if attn_bias is not None:
        # when all values in a line of attn_bias are -inf, we setting value in this line to a very small value
        # to prevent softmax giving nan output, however, after softmax, values in this line become 1/seqlen,
        # so setting them to 0 after softmax
        attention = paddle.where(
            all_inf_mask, paddle.zeros_like(attention), attention
        )

    # We want to mask here so that the attention matrix doesn't have any NaNs
    # Otherwise we'll get NaN in dV
    if query_padding_mask is not None:
        raise AssertionError
        attention = attention.masked_fill(
            rearrange(~query_padding_mask, "b s -> b 1 s 1"), 0.0
        )

    # Without this we might get NaN in dv
    if key_padding_mask is not None:
        raise AssertionError
        attention = attention.masked_fill(
            rearrange(~key_padding_mask, "b s -> b 1 1 s"), 0.0
        )
    # Some rows might be completely masked out so we fill them with zero instead of NaN
    if local_mask is not None:
        attention = attention.masked_fill(
            paddle.all(local_mask, axis=-1, keepdim=True), 0.0
        )
    dropout_scaling = 1.0 / (1 - dropout_p)
    # attention_drop = attention.masked_fill(~dropout_mask, 0.0) * dropout_scaling
    # output = paddle.matmul(attention_drop, v, transpose_y=True)
    if dropout_mask is not None:
        raise AssertionError
        attention_drop = attention.masked_fill(~dropout_mask, 0.0)
    else:
        attention_drop = attention
    if intermediate_dtype is not None:
        attention_drop = attention_drop.cast(intermediate_dtype).cast(
            attention_drop.dtype
        )
    output = paddle.matmul(attention_drop, v * dropout_scaling)
    output = paddle.transpose(output, [0, 2, 1, 3])
    if query_padding_mask is not None:
        output.masked_fill_(
            rearrange(~query_padding_mask, "b s -> b s 1 1"), 0.0
        )
    return output.cast(dtype=dtype_og), attention.cast(dtype=dtype_og)
