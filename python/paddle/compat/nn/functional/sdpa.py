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

from __future__ import annotations

from typing import TYPE_CHECKING

import paddle
import paddle.nn.functional as F

if TYPE_CHECKING:
    from paddle import Tensor


def _math_scaled_dot_product_attention(
    query,
    key,
    value,
    attn_mask,
    dropout_p,
    is_causal,
    scale,
    enable_gqa,
):
    output_dtype = query.dtype
    if (
        hasattr(query, "place")
        and query.place.is_cpu_place()
        and query.dtype in (paddle.float16, paddle.bfloat16)
    ):
        query = query.astype(paddle.float32)
        key = key.astype(paddle.float32)
        value = value.astype(paddle.float32)
        if attn_mask is not None and attn_mask.dtype != paddle.bool:
            attn_mask = attn_mask.astype(paddle.float32)

    if enable_gqa:
        query_heads = query.shape[-3]
        key_heads = key.shape[-3]
        value_heads = value.shape[-3]
        if (
            key_heads == 0
            or value_heads == 0
            or query_heads % key_heads != 0
            or query_heads % value_heads != 0
        ):
            raise ValueError(
                "The number of query heads must be divisible by the number "
                "of key/value heads when enable_gqa=True."
            )
        key_repeats = query_heads // key_heads
        value_repeats = query_heads // value_heads
        if key_repeats != 1:
            key = paddle.repeat_interleave(key, key_repeats, axis=-3)
        if value_repeats != 1:
            value = paddle.repeat_interleave(value, value_repeats, axis=-3)

    head_dim = query.shape[-1]
    scale_factor = head_dim**-0.5 if scale is None and head_dim != 0 else scale
    if scale_factor is None:
        scale_factor = 1.0
    scores = paddle.matmul(query, key, transpose_y=True) * scale_factor

    if is_causal:
        causal_mask = paddle.ones(
            [query.shape[-2], key.shape[-2]], dtype=paddle.bool
        ).tril()
        scores = paddle.where(
            causal_mask,
            scores,
            paddle.full_like(scores, -float("inf")),
        )
    if attn_mask is not None:
        if attn_mask.dtype == paddle.bool:
            scores = paddle.where(
                attn_mask,
                scores,
                paddle.full_like(scores, -float("inf")),
            )
        else:
            scores = paddle.where(
                attn_mask != -float("inf"),
                scores + attn_mask,
                paddle.full_like(scores, -float("inf")),
            )

    has_unmasked_score = paddle.any(
        scores != -float("inf"), axis=-1, keepdim=True
    )
    safe_scores = paddle.where(
        has_unmasked_score, scores, paddle.zeros_like(scores)
    )
    weights = F.softmax(safe_scores, axis=-1)
    weights = paddle.where(
        paddle.logical_and(has_unmasked_score, scores != -float("inf")),
        weights,
        paddle.zeros_like(weights),
    )
    if dropout_p > 0.0:
        weights = F.dropout(weights, p=dropout_p, training=True)
    out = paddle.matmul(weights, value)
    return out if out.dtype == output_dtype else out.astype(output_dtype)


def scaled_dot_product_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    attn_mask: Tensor | None = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: float | None = None,
    enable_gqa: bool = False,
) -> Tensor:
    r"""
    The equation is:

    .. math::

        result=softmax(\frac{ Q * K^T }{\sqrt{d}}) * V

    where : ``Q``, ``K``, and ``V`` represent the three input parameters of the attention module.
    The dimensions of the three parameters are the same.
    ``d`` represents the size of the last dimension of the three parameters.


    Warning:
        This API only verifies inputs with dtype float16 and bfloat16, other dtypes may fall back to math
        implementation, which is less optimized.

    Note:
        This API differs from :ref:`api_paddle_nn_functional_scaled_dot_product_attention` in that:
        The QKV layout of this API is [batch_size, num_heads, seq_len, head_dim] or [num_heads, seq_len, head_dim].

    Args:
        query(Tensor): The query tensor in the Attention module.
                        4-D tensor with shape:
                        [batch_size, num_heads, seq_len, head_dim].
                        3-D tensor with shape:
                        [num_heads, seq_len, head_dim].
                        The dtype can be float16 or bfloat16.
        key(Tensor): The key tensor in the Attention module.
                        4-D tensor with shape:
                        [batch_size, num_heads, seq_len, head_dim].
                        3-D tensor with shape:
                        [num_heads, seq_len, head_dim].
                        The dtype can be float16 or bfloat16.
        value(Tensor): The value tensor in the Attention module.
                        4-D tensor with shape:
                        [batch_size, num_heads, seq_len, head_dim].
                        3-D tensor with shape:
                        [num_heads, seq_len, head_dim].
                        The dtype can be float16 or bfloat16.
        attn_mask(Tensor, optional): The attention mask tensor. The shape should be broadcastable to
                        [batch_size, num_heads, seq_len_key, seq_len_query]. The dtype can be bool
                        or same type of query. The bool mask indicates the positions should take part
                        in attention. The non-bool mask will be added to attention score.

        is_causal(bool, optional): Whether enable causal mode. If True, the attention masking is a lower
                        triangular matrix when the mask is a square matrix. The attention masking has the
                        form of the upper left causal bias when the mask is a non-square matrix.
                        An error is thrown if both attn_mask and is_causal are set.
        scale(float, optional): The scaling factor used in the calculation of attention weights.
                        If None, scale = 1 / sqrt(head_dim).
        enable_gqa(bool, optional): Whether enable GQA mode. Default False.

    Returns:
        out(Tensor): The attention tensor.
                    4-D tensor with shape: [batch_size, num_heads, seq_len, head_dim].
                    3-D tensor with shape: [num_heads, seq_len, head_dim].
                    The dtype can be float16 or bfloat16.

    Examples:
        .. code-block:: pycon

            >>> # doctest: +SKIP('bfloat need V100 compile')
            >>> import paddle
            >>> q = paddle.rand((1, 2, 128, 16), dtype=paddle.bfloat16)
            >>> output = paddle.compat.nn.functional.scaled_dot_product_attention(q, q, q, None, 0.9, False)
            >>> print(output)
            >>> # doctest: -SKIP
    """
    if not 0.0 <= dropout_p <= 1.0:
        raise ValueError(
            f"dropout probability has to be between 0 and 1, but got {dropout_p}"
        )

    if is_causal and attn_mask is not None:
        raise RuntimeError(
            "Explicit attn_mask should not be set when is_causal=True"
        )

    use_math_fallback = (
        query.ndim not in (3, 4)
        or scale == 0
        or (
            hasattr(query, "place")
            and query.place.is_cpu_place()
            and (
                query.dtype in (paddle.float16, paddle.bfloat16)
                or attn_mask is not None
            )
        )
    )
    if use_math_fallback:
        return _math_scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask,
            dropout_p,
            is_causal,
            scale,
            enable_gqa,
        )

    query, key, value = (
        query.swapaxes(-3, -2),
        key.swapaxes(-3, -2),
        value.swapaxes(-3, -2),
    )
    out = F.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask,
        dropout_p,
        is_causal,
        True,  # training
        None,  # backend
        scale,
        enable_gqa,
        None,  # name
    )
    return out.swapaxes(-3, -2)
