#   Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

from paddle import _legacy_C_ops
from paddle.framework import in_dynamic_mode

if TYPE_CHECKING:
    from paddle import Tensor


def fused_gate_attention(
    query: Tensor,
    key: Tensor | None = None,
    query_weight: Tensor | None = None,
    key_weight: Tensor | None = None,
    value_weight: Tensor | None = None,
    qkv_weight: Tensor | None = None,
    gate_linear_weight: Tensor | None = None,
    gate_linear_bias: Tensor | None = None,
    out_linear_weight: Tensor | None = None,
    out_linear_bias: Tensor | None = None,
    nonbatched_bias: Tensor | None = None,
    attn_mask: Tensor | None = None,
    has_gating: bool = True,
    merge_qkv: bool = True,
    use_flash_attn: bool = False,
) -> Tensor:
    r"""
    Attention maps queries and a set of key-value pairs to outputs, and
    Gate Attention performs multiple parallel attention to jointly attending
    to information from different representation subspaces. This API only
    support self_attention. The pseudo code is as follows:

    .. code-block:: text

        c = c ** (-0.5)
        q = paddle.einsum('nbqa,ahc->nbqhc', q_data, query_w) * c
        k = paddle.einsum('nbka,ahc->nbkhc', m_data, key_w)
        v = paddle.einsum('nbka,ahc->nbkhc', m_data, value_w)
        logits = paddle.einsum('nbqhc,nbkhc->nbhqk', q, k) + bias

        if nonbatched_bias is not None:
            logits += paddle.unsqueeze(nonbatched_bias, axis=1)

        weights = paddle.nn.functional.softmax(logits)
        weighted_avg = paddle.einsum('nbhqk,nbkhc->nbqhc', weights, v)

        if has_gating:
            gate_values = paddle.einsum('nbqc,chv->nbqhv', q_data, gating_w) + gating_b
            gate_values = paddle.nn.functional.sigmoid(gate_values)
            weighted_avg *= gate_values

        output = paddle.einsum('nbqhc,hco->nbqo', weighted_avg, output_w) + output_b


    Args:
        query (Tensor): The input query tensor. The shape is [batch_size, msa_len, res_len, q_dim].
        key (Tensor, optional): The input key tensor, which can be set when
            merge_qkv is False. The shape is [batch_size, msa_len, m_size, kv_dim]. Default None.
        query_weight (Tensor, optional): The weight of query linear, which should be set when input
            key is not None. The shape is [q_dim, num_heads, head_dim]. Default None.
        key_weight (Tensor, optional): The weight of key linear, which should be set when input key
            is not None. The shape is [kv_dim, num_heads, head_dim]. Default None.
        value_weight (Tensor, optional): The weight of value linear, which should be set when input
            key is not None. The shape is [kv_dim, num_heads, head_dim]. Default None.
        qkv_weight (Tensor, optional): The weight of qkv linear, which should be set when merge_qkv
            is True. The shape is [3, num_heads, head_dim, q_dim]. Default None.
        gate_linear_weight (Tensor, optional): The weight of gating linear, which should be set when
            has_gating is True. The shape is [q_dim, num_heads, head_dim]. Default None.
        gate_linear_bias (Tensor, optional): The bias of gating linear, which should be set when
            has_gating is True. The shape is [num_heads, head_dim]. Default None.
        out_linear_weight (Tensor, optional): The weight of output linear. The shape is [num_heads, head_dim, q_dim]. Default None.
        out_linear_bias (Tensor): The bias of output linear, the shape is [q_dim]. Default None.
        nonbatched_bias (Tensor, optional): The extra bias. The shape is [batch_size, 1, num_heads, res_len, m_size]. Default None.
        attn_mask (Tensor, optional):  The attention mask. The shape is [batch_size, msa_len, 1, 1, res_len]. Default None.
        has_gating (bool, optional): Whether has the gating linear. Default True.
        merge_qkv (bool, optional): Whether has the gating linear. Default True.
        use_flash_attn (bool, optional): Whether use flash-attention to speedup. Default False.

    Returns:
        Tensor: The output Tensor, the data type and shape is same as `query`.

    Examples:

        .. code-block:: python

            >>> # doctest: +REQUIRES(env:GPU)
            >>> import paddle
            >>> import paddle.incubate.nn.functional as F

            >>> # batch_size = 2
            >>> # msa_len = 4
            >>> # res_len = 2
            >>> # q_dim = 4
            >>> # num_heads = 8
            >>> # head_dim = 4
            >>> # m_size = res_len (when merge_qkv is True)

            >>> # query: [batch_size, msa_len, res_len, q_dim]
            >>> query = paddle.rand(shape=[2, 4, 2, 4], dtype="float32")

            >>> # qkv_weight:  [3, n_heads, head_dim, q_dim]
            >>> qkv_weight = paddle.rand(shape=[3, 8, 4, 4], dtype="float32")

            >>> # nonbatched_bias: [batch_size, 1, num_heads, res_len, m_size]
            >>> nonbatched_bias = paddle.rand(shape=[2, 1, 8, 2, 2], dtype="float32")

            >>> # attn_mask: [batch_size, msa_len, 1, 1, m_size]
            >>> attn_mask = paddle.rand(shape=[2, 4, 1, 1, 2], dtype="float32")

            >>> # gate_linear_weight: [q_dim, num_heads, head_dim]
            >>> gate_linear_weight = paddle.rand(shape=[4, 8, 4], dtype="float32")
            >>> # gate_bias: [num_heads, head_dim]
            >>> gate_linear_bias = paddle.rand(shape=[8, 4], dtype="float32")

            >>> # out_linear_weight: [num_heads, head_dim, q_dim]
            >>> out_linear_weight = paddle.rand(shape=[8, 4, 4], dtype="float32")
            >>> # out_linear_bias: [q_dim]
            >>> out_linear_bias = paddle.rand(shape=[4], dtype="float32")

            >>> # output: [batch_size, msa_len, res_len, q_dim]
            >>> output = F.fused_gate_attention(
            ...     query=query,
            ...     qkv_weight=qkv_weight,
            ...     gate_linear_weight=gate_linear_weight,
            ...     gate_linear_bias=gate_linear_bias,
            ...     out_linear_weight=out_linear_weight,
            ...     out_linear_bias=out_linear_bias,
            ...     nonbatched_bias=nonbatched_bias,
            ...     attn_mask=attn_mask,
            ...     has_gating=True,
            ...     merge_qkv=True)
            >>> print(output.shape)
            [2, 4, 2, 4]

    """
    if in_dynamic_mode():
        _, _, _, _, _, _, _, _, out = _legacy_C_ops.fused_gate_attention(
            query,
            key,
            query_weight,
            key_weight,
            value_weight,
            qkv_weight,
            nonbatched_bias,
            attn_mask,
            gate_linear_weight,
            gate_linear_bias,
            out_linear_weight,
            out_linear_bias,
            'has_gating',
            has_gating,
            'merge_qkv',
            merge_qkv,
            "use_flash_attn",
            use_flash_attn,
        )
        return out
