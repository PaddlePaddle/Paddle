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
    r"""
    Allows the model to jointly attend to information from different representation subspaces.

    Multi-Head Attention is defined as:

    .. math::
        \text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1,\dots,\text{head}_h)W^O

    where :math:`\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)`.

    Please refer to `Attention Is All You Need <https://arxiv.org/pdf/1706.03762.pdf>`_
    for more details.

    .. note::
        This layer will use the optimized implementation
        :func:`paddle.nn.functional.scaled_dot_product_attention` if no need to return the attention weights.

    Parameters:
        embed_dim (int): Total dimension of the model.
        num_heads (int): The number of heads in multi-head attention.
        dropout (float, optional): The dropout probability used on attention
            weights to drop some attention targets. 0 for no dropout. Default 0.0.
        bias (bool, optional): If specified, adds bias to input / output projection layers.
            Default: True.
        add_bias_kv (bool, optional): If specified, adds bias to the key and value sequences
            at axis=0. Default: False.
        add_zero_attn (bool, optional): If specified, adds a new batch of zeros to the
            key and value sequences at axis=1. Default: False.
        kdim (int, optional): Total number of features for keys. If None, assumed equal to
            `embed_dim`. Default: None.
        vdim (int, optional): Total number of features for values. If None, assumed equal to
            `embed_dim`. Default: None.
        batch_first (bool, optional): If True, then the input and output tensors are provided
            as [batch, seq, feature]. Default: False.
        device (PlaceLike|None, optional): The device to initialize parameters on. Default: None.
        dtype (DTypeLike|None, optional): The data type of the parameters. Default: None.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> from paddle.compat import nn

            >>> # Example with batch_first=True
            >>> embed_dim, num_heads = 128, 8
            >>> multihead_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

            >>> # query: [batch_size, target_seq_len, embed_dim]
            >>> query = paddle.randn([32, 10, embed_dim])
            >>> # key, value: [batch_size, source_seq_len, embed_dim]
            >>> key = paddle.randn([32, 20, embed_dim])
            >>> value = paddle.randn([32, 20, embed_dim])

            >>> attn_output, attn_output_weights = multihead_attn(query, key, value)
            >>> print(attn_output.shape)
            paddle.Size([32, 10, 128])
    """

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

        self.in_proj_bias = None
        self.q_proj_bias = None
        self.k_proj_bias = None
        self.v_proj_bias = None

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
            if bias:
                self.in_proj_bias = self.create_parameter(
                    shape=[3 * embed_dim],
                    dtype=self._dtype,
                    is_bias=True,
                    device=device,
                )

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

        pad_tensor = paddle.zeros(pad_shape, dtype=mask.dtype)
        return paddle.concat([mask, pad_tensor], axis=-1)

    def _project_qkv(
        self, query: Tensor, key: Tensor, value: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        # in: [batch, seq_len, embed]
        # out: [batch, seq_len, embed]
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
                k = F.linear(key, k_w.T, k_b)
                v = F.linear(value, v_w.T, v_b)
        else:
            q = F.linear(query, self.q_proj_weight.T, self.q_proj_bias)
            k = F.linear(key, self.k_proj_weight.T, self.k_proj_bias)
            v = F.linear(value, self.v_proj_weight.T, self.v_proj_bias)
        return q, k, v

    def _prepare_qkv_heads(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        batch_size: int,
        target_seq_len: int,
    ) -> tuple[Tensor, Tensor, Tensor]:
        # in: [batch, seq_len, num_head * dim]
        # out: [batch, num_head, seq_len, dim]
        if self.add_bias_kv:
            k = paddle.concat(
                [k, self.bias_k.expand([batch_size, -1, -1])], axis=1
            )
            v = paddle.concat(
                [v, self.bias_v.expand([batch_size, -1, -1])], axis=1
            )

        q = q.reshape(
            [batch_size, target_seq_len, self.num_heads, self.head_dim]
        ).transpose([0, 2, 1, 3])

        current_src_len = k.shape[1]
        k = k.reshape(
            [batch_size, current_src_len, self.num_heads, self.head_dim]
        ).transpose([0, 2, 1, 3])
        v = v.reshape(
            [batch_size, current_src_len, self.num_heads, self.head_dim]
        ).transpose([0, 2, 1, 3])

        if self.add_zero_attn:
            zeros = paddle.zeros(
                [batch_size, self.num_heads, 1, self.head_dim], dtype=k.dtype
            )
            k = paddle.concat([k, zeros], axis=2)
            v = paddle.concat([v, zeros], axis=2)

        return q, k, v

    def _prepare_attn_mask(
        self,
        attn_mask: Tensor | None,
        key_padding_mask: Tensor | None,
        target_seq_len: int,
        src_len_before_bias: int,
        dtype: DTypeLike,
        batch_size: int,
        is_causal: bool,
        need_weights: bool,
    ) -> Tensor | None:
        # Do not generate attn_mask if is_causal is True and add_bias_kv is False
        # and add_zero_attn is False. In such case, we pass attn_mask as None to
        # select efficient implementation backend of sdpa.
        if (
            is_causal
            and not self.add_bias_kv
            and not self.add_zero_attn
            and key_padding_mask is None
            and not need_weights
        ):
            return None

        if attn_mask is None and not is_causal and key_padding_mask is None:
            return None

        if attn_mask is None:
            if is_causal:
                attn_mask = paddle.triu(
                    paddle.ones(
                        [target_seq_len, src_len_before_bias], dtype=paddle.bool
                    ),
                    diagonal=1,
                )
            else:
                attn_mask = paddle.zeros(
                    [target_seq_len, src_len_before_bias], dtype=dtype
                )

        pad_count = int(self.add_zero_attn + self.add_bias_kv)

        if pad_count > 0:
            attn_mask = self._pad_mask(attn_mask, pad_amt=pad_count)
            if key_padding_mask is not None:
                key_padding_mask = self._pad_mask(
                    key_padding_mask, pad_amt=pad_count
                )

        if attn_mask.dim() == 2:
            attn_mask = attn_mask.expand(
                [batch_size * self.num_heads, *attn_mask.shape]
            )
        if attn_mask.dim() == 3:
            attn_mask = attn_mask.reshape(
                [batch_size, self.num_heads, target_seq_len, -1]
            )

        if key_padding_mask is not None:
            # [N, len_k+pad_count] -> [N, 1, 1, len_k+pad_count]
            key_padding_mask = key_padding_mask.unsqueeze(axis=[1, 2])
            key_padding_mask = key_padding_mask.repeat(
                [1, *attn_mask.shape[1:3], 1]
            )
            attn_mask = self._combine_masks(attn_mask, key_padding_mask, dtype)

        if attn_mask.dtype != dtype:
            if attn_mask.dtype == paddle.bool:
                attn_mask = self._convert_bool_mask_to_float(attn_mask, dtype)
            else:
                attn_mask = attn_mask.astype(dtype)

        return attn_mask

    def _attention_core(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        final_mask: Tensor | None,
        need_weights: bool,
        is_causal: bool,
    ) -> tuple[Tensor, Tensor | None]:
        # in: [batch, num_head, seq_len, head_dim]
        # out: [batch, num_head, seq_len, head_dim]
        batch_size, _, target_seq_len, _ = q.shape
        is_causal = is_causal and final_mask is None

        if not need_weights:
            attn_output = (
                paddle.compat.nn.functional.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    attn_mask=final_mask,
                    dropout_p=self.dropout if self.training else 0.0,
                    is_causal=is_causal,
                )
            )
            attn_output = attn_output.transpose([0, 2, 1, 3])
            attn_output = attn_output.reshape(
                [batch_size, target_seq_len, self.embed_dim]
            )
            return attn_output, None
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

            ctx = paddle.matmul(weights, v)
            attn_output = ctx.transpose([0, 2, 1, 3]).reshape(
                [batch_size, target_seq_len, self.embed_dim]
            )
            return attn_output, weights if need_weights else None

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
        r"""
        Forward pass of the MultiheadAttention layer.

        .. note::
            If ``need_weights`` is ``False``, this api will fallback to native math implementation,
            otherwise it will call ``paddle.compat.nn.functional.scaled_dot_product_attention`` to
            compute the attention score.

            To achieve better performance, explicitly set ``need_weights=False``,
            and set ``is_causal=True`` if the attn_mask is the causal mask.

        Parameters:
            query (Tensor): The query embeddings. Shape depends on `batch_first`.
                If `batch_first` is False, shape is `[target_seq_len, batch_size, embed_dim]`.
                If `batch_first` is True, shape is `[batch_size, target_seq_len, embed_dim]`.
            key (Tensor): The key embeddings. Shape depends on `batch_first`.
                If `batch_first` is False, shape is `[source_seq_len, batch_size, kdim]`.
                If `batch_first` is True, shape is `[batch_size, source_seq_len, kdim]`.
            value (Tensor): The value embeddings. Shape depends on `batch_first`.
                If `batch_first` is False, shape is `[source_seq_len, batch_size, vdim]`.
                If `batch_first` is True, shape is `[batch_size, source_seq_len, vdim]`.
            key_padding_mask (Tensor, optional): If specified, a mask indicating which
                elements within `key` to ignore for the purpose of attention (i.e. treat as "padding").
                Can be a boolean mask (True indicates padding) or a float mask.
                Shape is `[batch_size, source_seq_len]`. Default: None.
            need_weights (bool, optional): Indicate whether to return the attention
                weights. Default: True.
            attn_mask (Tensor, optional): 2D or 3D mask that prevents attention to certain positions.
                A 2D mask will be broadcasted for all batches while a 3D mask allows different masks
                for the entries in the batch. Shape is `[target_seq_len, source_seq_len]` or
                `[batch_size * num_heads, target_seq_len, source_seq_len]`. Default: None.
            average_attn_weights (bool, optional): If True, indicates that the returned
                `attn_weights` should be averaged across heads. Default: True.
            is_causal (bool, optional): If True, implies that a causal mask is applied to
                the attention implementation. If attn_mask is None and is_causal is True,
                a causal mask is automatically created and used in the attention computation.
                Default: False.

        Returns:
            tuple[Tensor, Tensor|None]:
                - **attn_output** (Tensor): The output of the attention mechanism.
                  Shape matches `query` (based on `batch_first`).
                - **attn_output_weights** (Tensor|None): The attention weights. Returns None if
                  `need_weights` is False. Shape is `[batch_size, target_seq_len, source_seq_len]`
                  if `average_attn_weights` is True.
                  If `average_attn_weights` is False, shape is
                  `[batch_size, num_heads, target_seq_len, source_seq_len]`.
        """
        is_batched = query.dim() == 3
        if not is_batched:
            query = query.unsqueeze(0 if self.batch_first else 1)
            key = key.unsqueeze(0 if self.batch_first else 1)
            value = value.unsqueeze(0 if self.batch_first else 1)
            if key_padding_mask is not None and key_padding_mask.dim() != 2:
                key_padding_mask = key_padding_mask.unsqueeze(0)

        if not self.batch_first:
            query = query.transpose([1, 0, 2])
            key = key.transpose([1, 0, 2])
            value = value.transpose([1, 0, 2])

        batch_size, target_seq_len, _ = query.shape
        src_len_before_bias = key.shape[1]
        if key_padding_mask is not None:
            assert key_padding_mask.shape == (batch_size, src_len_before_bias)

        q, k, v = self._project_qkv(query, key, value)

        q, k, v = self._prepare_qkv_heads(q, k, v, batch_size, target_seq_len)

        final_mask = self._prepare_attn_mask(
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            target_seq_len=target_seq_len,
            src_len_before_bias=src_len_before_bias,
            dtype=q.dtype,
            batch_size=batch_size,
            is_causal=is_causal,
            need_weights=need_weights,
        )

        attn_output, attn_weights = self._attention_core(
            q, k, v, final_mask, need_weights, is_causal
        )

        attn_output = self.out_proj(attn_output)

        if not self.batch_first:
            attn_output = attn_output.transpose([1, 0, 2])

        if need_weights and attn_weights is not None:
            if average_attn_weights:
                attn_weights = attn_weights.mean(axis=1)

        if not is_batched:
            attn_output = attn_output.squeeze(0 if self.batch_first else 1)
            if attn_weights is not None:
                attn_weights = attn_weights.squeeze(0)

        return attn_output, attn_weights
