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

    Note:
        This API differs from :ref:`api_paddle_nn_functional_scaled_dot_product_attention` in that:
            1. This API supports GQA(Generic Query Attention) mode.
            2. The QKV layout of this API is [batch_size, num_heads, seq_len, head_dim] or [num_heads, seq_len, head_dim].

    Warning:
        This API only supports inputs with dtype float16 and bfloat16.

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
        enable_gqa(bool, optional): Whether enable GQA mode.

    Returns:
        out(Tensor): The attention tensor.
                    4-D tensor with shape: [batch_size, num_heads, seq_len, head_dim].
                    3-D tensor with shape: [num_heads, seq_len, head_dim].
                    The dtype can be float16 or bfloat16.

    Examples:
        .. code-block:: python

            >>> # doctest: +SKIP('bfloat need V100 compile')
            >>> import paddle
            >>> q = paddle.rand((1, 2, 128, 16), dtype=paddle.bfloat16)
            >>> output = paddle.compat.nn.functional.scaled_dot_product_attention(q, q, q, None, 0.9, False)
            >>> print(output)
            >>> # doctest: -SKIP
    """
    if is_causal and attn_mask is not None:
        raise RuntimeError(
            "Explicit attn_mask should not be set when is_causal=True"
        )
    if len(query.shape) == 3:
        query = query.unsqueeze(0)
    if len(key.shape) == 3:
        key = key.unsqueeze(0)
    if len(value.shape) == 3:
        value = value.unsqueeze(0)
    if enable_gqa:
        k_heads, q_heads, v_heads = (
            key.shape[-3],
            query.shape[-3],
            value.shape[-3],
        )

        assert q_heads % k_heads == 0, (
            f"The number of groups in query({q_heads}) must be divisible by the number of groups in key({k_heads}) if GQA enabled."
        )
        assert k_heads == v_heads, (
            f"The number of groups in key({k_heads}) must be equal to the number of groups in value({v_heads}) if GQA enabled."
        )
        # repeat_interleave does not support float16 on GPU, so we manually expand the tensor
        if k_heads != q_heads:
            repeats = q_heads // k_heads
            key, value = key.unsqueeze(-3), value.unsqueeze(-3)
            key, value = (
                key.expand([-1, -1, repeats, -1, -1]),
                value.expand([-1, -1, repeats, -1, -1]),
            )
            key, value = (
                key.flatten(-4, -3).contiguous(),
                value.flatten(-4, -3).contiguous(),
            )

    if attn_mask is not None:
        batch_size = query.shape[0]
        seq_len_q = query.shape[2]
        num_heads = query.shape[1]
        seq_len_k = key.shape[2]

        target_shape = [batch_size, num_heads, seq_len_q, seq_len_k]
        attn_mask = attn_mask.expand(target_shape)
        if attn_mask.dtype == paddle.bool:
            neg_inf = float('-inf')
            zeros = paddle.zeros_like(attn_mask, dtype=query.dtype)
            masked_vals = paddle.full_like(
                attn_mask, fill_value=neg_inf, dtype=query.dtype
            )
            attn_mask = paddle.where(attn_mask, zeros, masked_vals)
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
        None,  # name
        None,  # backend
        scale,
    )
    return out.swapaxes(-3, -2)
