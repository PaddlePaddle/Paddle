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

from paddle import _legacy_C_ops
from paddle.fluid.framework import _non_static_mode


def fused_gate_attention(
    query,
    key=None,
    query_weight=None,
    key_weight=None,
    value_weight=None,
    qkv_weight=None,
    gate_linear_weight=None,
    gate_linear_bias=None,
    out_linear_weight=None,
    out_linear_bias=None,
    nonbatched_bias=None,
    attn_mask=None,
    has_gating=True,
    merge_qkv=True,
    use_flash_attn=False,
):
    r"""
    Attention mapps queries and a set of key-value pairs to outputs, and
    Gate Attention performs multiple parallel attention to jointly attending
    to information from different representation subspaces. This API only
    support self_attention. The pseudo code is as follows:

    .. code-block:: python

        c = self.c ** (-0.5)
        q = paddle.einsum('nbqa,ahc->nbqhc', q_data, self.query_w) * c
        k = paddle.einsum('nbka,ahc->nbkhc', m_data, self.key_w)
        v = paddle.einsum('nbka,ahc->nbkhc', m_data, self.value_w)
        logits = paddle.einsum('nbqhc,nbkhc->nbhqk', q, k) + bias

        if nonbatched_bias is not None:
            logits += paddle.unsqueeze(nonbatched_bias, axis=1)

        weights = nn.functional.softmax(logits)
        weighted_avg = paddle.einsum('nbhqk,nbkhc->nbqhc', weights, v)

        if self.gating:
            gate_values = paddle.einsum('nbqc,chv->nbqhv', q_data,
                                        self.gating_w) + self.gating_b
            gate_values = nn.functional.sigmoid(gate_values)
            weighted_avg *= gate_values

        output = paddle.einsum('nbqhc,hco->nbqo', weighted_avg,
                               self.output_w) + self.output_b

    Args:
        query (Tensor): The input tensor of fused_gate_head_attention. The shape is
            [batch_size, seq_len_m, seq_len_r, q_dim].
        qkv_weight (Tensor): The qkv weight tensor. The shape is [num_head, c, q_dim].
        gate_linear_weight (Tensor, optional): The gate weight tensor. The shape is [qkv_dim, num_head, c]
        gate_linear_bias (Tensor, optional): The gate bias tensor. The shape is [num_head, c]. Default None.
        out_linear_weight (Tensor, optional): The linear weight tensor. The shape is [num_head, c, out_dim].
        out_linear_bias (Tensor): The bias of linear. The shape is [out_dim]. Default None.
        nonbatched_bias (Tensor, optional): The extra bias. The shape is [batch_size, 1, num_head, seq_len_r, m_size]. Default None.
        attn_mask (Tensor):  The mask tensor. The shape is [batch_size, seq_len_m, 1, 1, seq_len_r]. Default None.
        has_gating (bool, optional): whether has the gating linear. Default True.

    Returns:
        Tensor: The output Tensor, the data type and shape is same as `query`.

    Examples:

        .. code-block:: python

            # required: gpu
            import paddle
            import paddle.incubate.nn.functional as F

            # query: [batch_size, seq_len_m, seq_len_r, qkv_dim]
            query = paddle.rand(shape=[2, 3, 2, 4], dtype="float32")
            # qkv_weight:  [3，n_head, c, qkv_dim]
            qkv_weight = paddle.rand(shape=[3, 8, 4, 4], dtype="float32")

            # out_linear_weight: [num_head, c, out_dim]
            out_linear_weight = paddle.rand(shape=[8, 4, 4], dtype="float32")
            # out_linear_bias: [out_dim]
            out_linear_bias = paddle.rand(shape=[4], dtype="float32")

            # attn_mask: [batch_size, seq_len_m, seq_len_r]
            attn_mask = paddle.rand(shape=[1, 3, 2], dtype="float32")
            attn_mask = paddle.unsqueeze(attn_mask, axis=[2, 3])

            # gate_linear_weight: [qkv_dim, num_head, c]
            gate_linear_weight = paddle.rand(shape=[4, 8, 4], dtype="float32")
            # gate_bias: [num_head, c]
            gate_linear_bias = paddle.rand(shape=[8, 4], dtype="float32")
            # nonbatched_bias: [batch_size, num_head, seq_len_r, seq_len_r]
            nonbatched_bias = paddle.rand(shape=[2, 8, 2, 2], dtype="float32")
            nonbatched_bias = paddle.unsqueeze(nonbatched_bias, axis=1)

            # output: [batch_size, seq_len_m, seq_len_r, qkv_dim]
            output = F.fused_gate_attention(
                    query=query,
                    qkv_weight=qkv_weight,
                    gate_linear_weight=gate_linear_weight,
                    gate_linear_bias=gate_linear_bias,
                    out_linear_weight=out_linear_weight,
                    out_linear_bias=out_linear_bias,
                    nonbatched_bias=nonbatched_bias,
                    attn_mask=attn_mask,
                    has_gating=True,
                    merge_qkv=True)
            # [2, 3, 2, 4]
            print(output.shape)
    """
    if _non_static_mode():
        _, _, _, _, _, _, _, out = _legacy_C_ops.fused_gate_attention(
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
