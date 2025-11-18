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
from paddle import nn
from paddle.nn.initializer import XavierNormal, XavierUniform

if TYPE_CHECKING:
    from paddle import Tensor
    from paddle._typing import DTypeLike, PlaceLike


class MultiheadAttention(nn.Layer):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        add_bias_kv: bool = False,
        add_zero_attn: bool = False,
        kdim: int | None = None,
        vdim: int | None = None,
        batch_first: bool = False,
        device: PlaceLike | None = None,
        dtype: DTypeLike | None = None,
    ) -> None:
        if dtype:
            super().__init__(dtype=dtype)
        else:
            super().__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = (
            self.kdim == embed_dim and self.vdim == embed_dim
        )
        self.num_heads = num_heads
        self.dropout = dropout
        self.batch_first = batch_first
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim

        if self._qkv_same_embed_dim:
            self.in_proj_weight = self.create_parameter(
                shape=[3 * embed_dim, embed_dim],
                dtype=self._dtype,
                is_bias=False,
                device=device,
                default_initializer=XavierUniform(),
            )
            self.q_proj_weight = None
            self.k_proj_weight = None
            self.v_proj_weight = None
        else:
            self.q_proj_weight = self.create_parameter(
                shape=[embed_dim, embed_dim],
                dtype=self._dtype,
                is_bias=False,
                device=device,
                default_initializer=XavierUniform(),
            )
            self.k_proj_weight = self.create_parameter(
                shape=[embed_dim, self.kdim],
                dtype=self._dtype,
                is_bias=False,
                device=device,
                default_initializer=XavierUniform(),
            )
            self.v_proj_weight = self.create_parameter(
                shape=[embed_dim, self.vdim],
                dtype=self._dtype,
                is_bias=False,
                device=device,
                default_initializer=XavierUniform(),
            )
            self.in_proj_weight = None

        if bias:
            if self._qkv_same_embed_dim:
                self.in_proj_bias = self.create_parameter(
                    shape=[3 * embed_dim],
                    dtype=self._dtype,
                    is_bias=True,
                    device=device,
                )
                self.q_proj_bias = None
                self.k_proj_bias = None
                self.v_proj_bias = None
            else:
                self.in_proj_bias = None
                self.q_proj_bias = self.create_parameter(
                    shape=[embed_dim],
                    dtype=self._dtype,
                    is_bias=True,
                    device=device,
                )
                self.k_proj_bias = self.create_parameter(
                    shape=[embed_dim],
                    dtype=self._dtype,
                    is_bias=True,
                    device=device,
                )
                self.v_proj_bias = self.create_parameter(
                    shape=[embed_dim],
                    dtype=self._dtype,
                    is_bias=True,
                    device=device,
                )
        else:
            self.in_proj_bias = None
            self.q_proj_bias = None
            self.k_proj_bias = None
            self.v_proj_bias = None

        self.out_proj = paddle.compat.nn.Linear(
            embed_dim, embed_dim, bias=bias, dtype=self._dtype
        )

        self.add_bias_kv = add_bias_kv
        self.add_zero_attn = add_zero_attn

        if add_bias_kv:
            self.bias_k = self.create_parameter(
                shape=[1, 1, embed_dim],
                dtype=self._dtype,
                is_bias=True,
                device=device,
                default_initializer=XavierNormal(),
            )
            self.bias_v = self.create_parameter(
                shape=[1, 1, embed_dim],
                dtype=self._dtype,
                is_bias=True,
                device=device,
                default_initializer=XavierNormal(),
            )
        else:
            self.bias_k = self.bias_v = None

    def _convert_bool_mask_to_float(
        self, mask: paddle.Tensor, dtype: DTypeLike
    ) -> paddle.Tensor:
        """
        Convert boolean mask to float mask. True -> -inf, False -> 0.0

        Args:
            mask (paddle.Tensor): boolean mask
            dtype (DTypeLike): float dtype

        Returns:
            paddle.Tensor: float mask
        """
        assert mask.dtype == paddle.bool, (
            f"mask must be boolean, but got {mask.dtype}"
        )
        filler = paddle.to_tensor(paddle.finfo(dtype).min, dtype=dtype)
        return paddle.where(mask, filler, paddle.zeros_like(mask, dtype=dtype))

    def _combine_masks(
        self, mask1: paddle.Tensor, mask2: paddle.Tensor, dtype: DTypeLike
    ) -> paddle.Tensor:
        """
        Safely combine two masks, mask can be bool or float.

        If both mask are bool, this function equals to
        paddle.logical_or(mask1, mask2) and return boolean mask.

        Otherwise, the boolean mask will be converted to float and combined with
        the float mask using addition.

        Args:
            mask1 (paddle.Tensor): mask1
            mask2 (paddle.Tensor): mask2

        Returns:
            paddle.Tensor: combined mask
        """
        if mask1.dtype == paddle.bool and mask2.dtype == paddle.bool:
            return mask1 | mask2

        if mask1.dtype == paddle.bool:
            mask1 = self._convert_bool_mask_to_float(mask1, dtype=dtype)
        if mask2.dtype == paddle.bool:
            mask2 = self._convert_bool_mask_to_float(mask2, dtype=dtype)

        return mask1 + mask2

    def _pad_mask(self, mask: Tensor, pad_amt: int = 1) -> Tensor:
        shape = mask.shape
        pad_shape = [*shape[:-1], pad_amt]

        if mask.dtype == paddle.bool:
            pad_tensor = paddle.zeros(pad_shape, dtype=paddle.bool)
        else:
            pad_tensor = paddle.zeros(pad_shape, dtype=mask.dtype)
        return paddle.concat([mask, pad_tensor], axis=-1)

    def forward(
        self,
        query: paddle.Tensor,
        key: paddle.Tensor,
        value: paddle.Tensor,
        key_padding_mask: paddle.Tensor | None = None,
        need_weights: bool = True,
        attn_mask: paddle.Tensor | None = None,
        average_attn_weights: bool = True,
        is_causal: bool = False,
    ) -> tuple[paddle.Tensor, paddle.Tensor | None]:
        is_batched = query.dim() == 3
        if not is_batched:
            query = query.unsqueeze(1)
            key = key.unsqueeze(1)
            value = value.unsqueeze(1)

        if not self.batch_first:
            query = query.transpose([1, 0, 2])
            key = key.transpose([1, 0, 2])
            value = value.transpose([1, 0, 2])

        batch_size, target_seq_len, _ = query.shape
        source_seq_len = key.shape[1]
        if self._qkv_same_embed_dim:
            if id(query) == id(key) and id(key) == id(value):
                qkv = F.linear(query, self.in_proj_weight.T, self.in_proj_bias)
                q, k, v = qkv.split(3, axis=-1)
            else:
                q_w, k_w, v_w = self.in_proj_weight.chunk(3, axis=0)
                q_b, k_b, v_b = (
                    self.in_proj_bias.chunk(3, axis=0)
                    if self.in_proj_bias is not None
                    else (None,) * 3
                )
                q = F.linear(query, q_w.T, q_b)
                k = F.linear(
                    key,
                    k_w.T,
                    k_b,
                )
                v = F.linear(
                    value,
                    v_w.T,
                    v_b,
                )
        else:
            q = F.linear(query, self.q_proj_weight.T, self.q_proj_bias)
            k = F.linear(key, self.k_proj_weight.T, self.k_proj_bias)
            v = F.linear(value, self.v_proj_weight.T, self.v_proj_bias)

        src_len_before_bias = key.shape[1]
        if self.add_bias_kv:
            k = paddle.concat(
                [k, self.bias_k.expand([batch_size, -1, -1])], axis=1
            )
            v = paddle.concat(
                [v, self.bias_v.expand([batch_size, -1, -1])], axis=1
            )

            if attn_mask is not None:
                attn_mask = self._pad_mask(attn_mask)
            if key_padding_mask is not None:
                key_padding_mask = self._pad_mask(key_padding_mask)
            source_seq_len += 1

        q = q.reshape(
            [batch_size, target_seq_len, self.num_heads, self.head_dim]
        ).transpose([0, 2, 1, 3])
        k = k.reshape(
            [batch_size, source_seq_len, self.num_heads, self.head_dim]
        ).transpose([0, 2, 1, 3])
        v = v.reshape(
            [batch_size, source_seq_len, self.num_heads, self.head_dim]
        ).transpose([0, 2, 1, 3])

        if self.add_zero_attn:
            zeros = paddle.zeros(
                [batch_size, self.num_heads, 1, self.head_dim], dtype=k.dtype
            )
            k = paddle.concat([k, zeros], axis=2)
            v = paddle.concat([v, zeros], axis=2)

            if attn_mask is not None:
                attn_mask = self._pad_mask(attn_mask)
            if key_padding_mask is not None:
                key_padding_mask = self._pad_mask(key_padding_mask)

            source_seq_len += 1

        can_use_sdpa = not need_weights and q.dtype in [
            paddle.float16,
            paddle.bfloat16,
        ]

        should_auto_gen_causal = is_causal
        if not can_use_sdpa:
            should_auto_gen_causal = False

        final_mask = None
        if should_auto_gen_causal:
            raw_causal_mask = paddle.triu(
                paddle.ones(
                    [target_seq_len, src_len_before_bias], dtype=paddle.bool
                ),
                diagonal=1,
            )

            if self.add_bias_kv:
                final_mask = self._pad_mask(raw_causal_mask, pad_amt=1)
            elif self.add_zero_attn:
                final_mask = self._pad_mask(raw_causal_mask, pad_amt=1)
            else:
                final_mask = raw_causal_mask

        if attn_mask is not None:
            if attn_mask.dim() == 2:
                final_mask = attn_mask
            elif attn_mask.dim() == 3:
                final_mask = attn_mask.reshape(
                    [batch_size, self.num_heads, target_seq_len, source_seq_len]
                )
            else:
                raise ValueError(f"attn_mask dim error: {attn_mask.dim()}")

        if key_padding_mask is not None:
            kp_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            final_mask = (
                kp_mask
                if final_mask is None
                else self._combine_masks(final_mask, kp_mask, dtype=q.dtype)
            )

        attn_weights = None

        sdpa_is_causal = is_causal if final_mask is None else False
        if can_use_sdpa:
            attn_output = F.scaled_dot_product_attention(
                q.transpose([0, 2, 1, 3]),
                k.transpose([0, 2, 1, 3]),
                v.transpose([0, 2, 1, 3]),
                attn_mask=final_mask,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=sdpa_is_causal,
                training=self.training,
            )
            attn_output = attn_output.reshape(
                [batch_size, target_seq_len, self.embed_dim]
            )
        else:
            scores = paddle.matmul(q, k, transpose_y=True)
            scores = scores / (self.head_dim**0.5)

            if final_mask is not None:
                if final_mask.dtype == paddle.bool:
                    final_mask = self._convert_bool_mask_to_float(
                        final_mask, scores.dtype
                    )
                scores = scores + final_mask

            weights = F.softmax(scores, axis=-1)
            weights = F.dropout(weights, self.dropout, training=self.training)
            attn_weights = weights

            ctx = paddle.matmul(weights, v)

            attn_output = ctx.transpose([0, 2, 1, 3]).reshape(
                [batch_size, target_seq_len, self.embed_dim]
            )

        attn_output = self.out_proj(attn_output)

        if not self.batch_first:
            attn_output = attn_output.transpose([1, 0, 2])

        if need_weights:
            if average_attn_weights:
                attn_weights = attn_weights.mean(axis=1)
        else:
            attn_weights = None

        if not is_batched:
            attn_output = attn_output.squeeze(1)
            if attn_weights is not None:
                if average_attn_weights:
                    attn_weights = attn_weights.squeeze(0)
                else:
                    attn_weights = attn_weights.squeeze(0)

        return attn_output, attn_weights
