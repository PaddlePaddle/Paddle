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

import os

import paddle
from paddle import _C_ops, in_dynamic_mode
from paddle.fluid.layer_helper import LayerHelper

g_use_flash_attn_v1 = (
    os.getenv('FLAGS_flash_attn_version', 'v2').strip().lower() == 'v1'
)


@paddle.utils.print_utils.print_args
def flash_attention(
    query,
    key,
    value,
    dropout=0.0,
    causal=False,
    return_softmax=False,
    *,
    fixed_seed_offset=None,
    rng_name="",
    training=True,
    name=None,
):
    r"""
    The equation is:

    .. math::

        result=softmax(\frac{ Q * K^T }{\sqrt{d}}) * V

    where : ``Q``, ``K``, and ``V`` represent the three input parameters of the attention module.
    The dimensions of the three parameters are the same.
    ``d`` represents the size of the last dimension of the three parameters.

    Warning:
        This API is only support inputs with dtype float16 and bfloat16.

    Args:
        query(Tensor): The query tensor in the Attention module.
                        4-D tensor with shape:
                        [batch_size, seq_len, num_heads, head_dim].
                        The dtype can be float61 or bfloat16.
        key(Tensor): The key tensor in the Attention module.
                        4-D tensor with shape:
                        [batch_size, seq_len, num_heads, head_dim].
                        The dtype can be float61 or bfloat16.
        value(Tensor): The value tensor in the Attention module.
                        4-D tensor with shape:
                        [batch_size, seq_len, num_heads, head_dim].
                        The dtype can be float61 or bfloat16.
        dropout(float): The dropout ratio.
        causal(bool): Whether enable causal mode.
        return_softmax(bool): Whether to return softmax.
        fixed_seed_offset(Tensor, optional): With fixed seed, offset for dropout mask.
        training(bool): Whether it is in the training phase.
        rng_name(str): The name to select Generator.
        name(str, optional): The default value is None. Normally there is no need for user
                        to set this property. For more information, please refer to
                        :ref:`api_guide_Name`.

    Returns:
        out(Tensor): The attention tensor.
                    4-D tensor with shape: [batch_size, seq_len, num_heads, head_dim].
                    The dtype can be float16 or bfloat16.
        softmax(Tensor): The softmax tensor. None if return_softmax is False.

    Examples:
        .. code-block:: python

            # required: skiptest
            import paddle

            q = paddle.rand((1, 128, 2, 16), dtype=paddle.float16)

            output = paddle.nn.functional.flash_attention(q, q, q, 0.9, False, False)
            print(output)
    """
    if in_dynamic_mode():
        if g_use_flash_attn_v1:
            (result_attention, result_softmax, _, _) = _C_ops.flash_attn_v1(
                query,
                key,
                value,
                dropout,
                causal,
                return_softmax,
                not training,
            )

        else:
            (result_attention, result_softmax, _, _) = _C_ops.flash_attn(
                query,
                key,
                value,
                fixed_seed_offset,
                None,
                dropout,
                causal,
                return_softmax,
                not training,
                rng_name,
            )
        return result_attention, result_softmax if return_softmax else None

    helper = LayerHelper('flash_attn', **locals())
    dtype = helper.input_dtype(input_param_name='q')
    out = helper.create_variable_for_type_inference(dtype)
    softmax = helper.create_variable_for_type_inference(dtype)
    softmax_lse = helper.create_variable_for_type_inference(paddle.float32)
    seed_offset = helper.create_variable_for_type_inference(paddle.int64)
    inputs = {
        'q': query,
        'k': key,
        'v': value,
        'fixed_seed_offset': fixed_seed_offset,
    }
    outputs = {
        'out': out,
        'softmax': softmax,
        'softmax_lse': softmax_lse,
        'seed_offset': seed_offset,
    }
    helper.append_op(
        type='flash_attn',
        inputs=inputs,
        outputs=outputs,
        attrs={
            'dropout': dropout,
            'causal': causal,
            'return_softmax': return_softmax,
            'is_test': not training,
            'rng_name': rng_name,
        },
    )
    return out, softmax if return_softmax else None


def flash_attn_unpadded(
    query,
    key,
    value,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    scale,
    dropout=0.0,
    causal=False,
    return_softmax=False,
    fixed_seed_offset=None,
    rng_name="",
    training=True,
    name=None,
):
    r"""
    The equation is:

    .. math::

        result=softmax(\frac{ Q * K^T }{\sqrt{d}}) * V

    where : ``Q``, ``K``, and ``V`` represent the three input parameters of the attention module.
    The dimensions of the three parameters are the same.
    ``d`` represents the size of the last dimension of the three parameters.

    Warning:
        This API is only support inputs with dtype float16 and bfloat16.

    Args:
        query(Tensor): The query tensor in the Attention module.
                        3-D tensor with shape:
                        [total_seq_len, num_heads, head_dim].
                        The dtype can be float61 or bfloat16.
        key(Tensor): The key tensor in the Attention module.
                        3-D tensor with shape:
                        [total_seq_len, num_heads, head_dim].
                        The dtype can be float61 or bfloat16.
        value(Tensor): The value tensor in the Attention module.
                        3-D tensor with shape:
                        [total_seq_len, num_heads, head_dim].
                        The dtype can be float61 or bfloat16.
        cu_seqlens_q(Tensor): The cumulative sequence lengths of the sequences in the batch,
                        used to index query.
        cu_seqlens_k(Tensor): The cumulative sequence lengths of the sequences in the batch,
                        used to index key and value.
        max_seqlen_q(int): Maximum sequence length of query in the batch.
        max_seqlen_k(int): Maximum sequence length of key/value in the batch.
        scale(float): The scaling of QK^T before applying softmax.
        dropout(float): The dropout ratio.
        causal(bool): Whether enable causal mode.
        return_softmax(bool): Whether to return softmax.
        fixed_seed_offset(Tensor, optional): With fixed seed, offset for dropout mask.
        rng_name(str): The name to select Generator.
        training(bool): Whether it is in the training phase.
        name(str, optional): The default value is None. Normally there is no need for user
                        to set this property. For more information, please refer to
                        :ref:`api_guide_Name`.

    Returns:
        out(Tensor): The attention tensor.
                    4-D tensor with shape: [batch_size, seq_len, num_heads, head_dim].
                    The dtype can be float16 or bfloat16.
        softmax(Tensor): The softmax tensor. None if return_softmax is False.

    Examples:
        .. code-block:: python

            # required: skiptest
            import paddle

            q = paddle.rand((1, 128, 2, 16), dtype=paddle.float16)

            output = paddle.nn.functional.flash_attn_unpadded(q, q, q, 0.9, False, False)
            print(output)
    """
    if in_dynamic_mode():
        if g_use_flash_attn_v1:
            (result_attention, result_softmax,) = _C_ops.flash_attn_unpadded(
                query,
                key,
                value,
                cu_seqlens_q,
                cu_seqlens_k,
                max_seqlen_q,
                max_seqlen_k,
                scale,
                dropout,
                causal,
                return_softmax,
                not training,
            )
        else:
            (result_attention, result_softmax,) = _C_ops.flash_attn_unpadded(
                query,
                key,
                value,
                cu_seqlens_q,
                cu_seqlens_k,
                fixed_seed_offset,
                None,
                max_seqlen_q,
                max_seqlen_k,
                scale,
                dropout,
                causal,
                return_softmax,
                not training,
                rng_name,
            )
        return result_attention, result_softmax if return_softmax else None

    helper = LayerHelper('flash_attn_unpadded', **locals())
    dtype = helper.input_dtype(input_param_name='q')
    out = helper.create_variable_for_type_inference(dtype)
    softmax = helper.create_variable_for_type_inference(dtype)
    softmax_lse = helper.create_variable_for_type_inference(paddle.float32)
    seed_offset = helper.create_variable_for_type_inference(paddle.int64)
    inputs = {
        'q': query,
        'k': key,
        'v': value,
        'cu_seqlens_q': cu_seqlens_q,
        'cu_seqlens_k': cu_seqlens_k,
        'fixed_seed_offset': fixed_seed_offset,
    }
    outputs = {
        'out': out,
        'softmax': softmax,
        'softmax_lse': softmax_lse,
        'seed_offset': seed_offset,
    }
    helper.append_op(
        type='flash_attn_unpadded',
        inputs=inputs,
        outputs=outputs,
        attrs={
            'max_seqlen_q': max_seqlen_q,
            'max_seqlen_k': max_seqlen_k,
            'scale': scale,
            'dropout': dropout,
            'causal': causal,
            'return_softmax': return_softmax,
            'is_test': not training,
            'rng_name': rng_name,
        },
    )
    return out, softmax if return_softmax else None


def flash_attention_with_mask(
    query,
    key,
    value,
    attn_mask=None,
    dropout=0.0,
    causal=False,
    training=True,
    name=None,
):
    r"""
    The equation is:

    .. math::

        result=softmax(\frac{ Q * K^T }{\sqrt{d}}) * V

    where : ``Q``, ``K``, and ``V`` represent the three input parameters of the attention module.
    The dimensions of the three parameters are the same.
    ``d`` represents the size of the last dimension of the three parameters.

    Warning:
        This API only supports inputs with dtype float16 and bfloat16.

    Args:
        query(Tensor): The query tensor in the Attention module.
                        4-D tensor with shape:
                        [batch_size, seq_len, num_heads, head_dim].
                        The dtype can be float61 or bfloat16.
        key(Tensor): The key tensor in the Attention module.
                        4-D tensor with shape:
                        [batch_size, seq_len, num_heads, head_dim].
                        The dtype can be float61 or bfloat16.
        value(Tensor): The value tensor in the Attention module.
                        4-D tensor with shape:
                        [batch_size, seq_len, num_heads, head_dim].
                        The dtype can be float61 or bfloat16.
        attn_mask(Tensor,optional): A float mask of the same type as query,
                        key, value that is added to the attention score.
        dropout(float): The dropout ratio.
        causal(bool): Whether enable causal mode.
        training(bool): Whether it is in the training phase.
        name(str, optional): The default value is None. Normally there is no need for user
                        to set this property. For more information, please refer to
                        :ref:`api_guide_Name`.

    Returns:
        out(Tensor): The attention tensor.
                    4-D tensor with shape: [batch_size, seq_len, num_heads, head_dim].
                    The dtype can be float16 or bfloat16.

    Examples:
        .. code-block:: python

            >>> # doctest: +SKIP()
            >>> import paddle
            >>> q = paddle.rand((1, 128, 2, 16), dtype=paddle.bfloat16)
            >>> output = paddle.nn.functional.scaled_dot_product_attention(q, q, q, None, 0.9, False)
            >>> print(output)
            >>> # doctest: -SKIP
    """
    if attn_mask is None:
        out, _ = flash_attention(query, key, value, dropout, causal)
    else:
        fixed_seed_offset = None
        return_softmax = False
        rng_name = ""
        out, _, _, _ = _C_ops.flash_attn(
            query,
            key,
            value,
            fixed_seed_offset,
            attn_mask,
            dropout,
            causal,
            return_softmax,
            not training,
            rng_name,
        )
    return out


def flash_attention_with_sparse_mask(
    query,
    key,
    value,
    attn_mask_start_row_indices,
    attn_mask_start_row=0,
    dropout=0.0,
    causal=False,
    return_softmax=False,
    return_softmax_lse=False,
    return_seed_offset=False,
    training=True,
    name=None,
):
    r"""
    The equation is:

    .. math::

        result=softmax(\frac{ Q * K^T }{\sqrt{d}}) * V

    where : ``Q``, ``K``, and ``V`` represent the three input parameters of the attention module.
    The dimensions of the three parameters are the same.
    ``d`` represents the size of the last dimension of the three parameters.

    Warning:
        This API only supports inputs with dtype float16 and bfloat16.

    Args:
        query(Tensor): The query tensor in the Attention module.
                        4-D tensor with shape:
                        [batch_size, seq_len, num_heads, head_dim].
                        The dtype can be float61 or bfloat16.
        key(Tensor): The key tensor in the Attention module.
                        4-D tensor with shape:
                        [batch_size, seq_len, num_heads, head_dim].
                        The dtype can be float61 or bfloat16.
        value(Tensor): The value tensor in the Attention module.
                        4-D tensor with shape:
                        [batch_size, seq_len, num_heads, head_dim].
                        The dtype can be float61 or bfloat16.
        attn_mask_start_row_indices(Tensor): A sparse attention mask
                        indices tensor, the shape is [batch_size, num_head, seq_len],
                        The value of each element indicates the row index where the
                        mask starts in score matrix. The dtype must be int32.
        attn_mask_start_row(int,optional): When `attn_mask_start_row_indices` is passed
                        in and the minimum row number is known to be greater than 0,
                        it can set `attn_mask_start_row` for performance improvement.
                        The default value is 0.
        dropout(float): The dropout ratio.
        causal(bool): Whether enable causal mode.
        training(bool): Whether it is in the training phase.
        name(str, optional): The default value is None. Normally there is no need for user
                        to set this property. For more information, please refer to
                        :ref:`api_guide_Name`.

    Returns:
        out(Tensor): The attention tensor.
                    4-D tensor with shape: [batch_size, seq_len, num_heads, head_dim].
                    The dtype can be float16 or bfloat16.

    Examples:
        .. code-block:: python
            >>> # doctest: +SKIP('bfloat need V100 compile')
            >>> import paddle
            >>> import numpy as np
            >>> def generate_start_rows(bz, num_head, rows, cols, start_row):
            >>>     assert rows == cols, f"rows {rows} must be equal to cols {cols}."
            >>>     start_rows_list = []
            >>>     for bz_idx in range(bz):
            >>>         for head_idx in range(num_head):
            >>>             start_rows = np.array([rows+1] * cols)
            >>>             mask_pos = np.random.choice(cols-1, cols - start_row, replace=False)
            >>>             index = np.arange(start_row, rows)
            >>>             mask_pos = np.concatenate([mask_pos[mask_pos < index - 1], mask_pos[mask_pos >= index - 1]])
            >>>             start_rows[mask_pos] = index
            >>>             start_rows_list.append(start_rows)
            >>>     start_rows_arr = np.array(start_rows_list).reshape([bz, num_head, rows])
            >>>     return start_rows_arr
            >>> q = paddle.rand((1, 128, 2, 16), dtype=paddle.bfloat16)
            >>> attn_mask_start_row = 48
            >>> start_row_indices = generate_start_rows(1, 2, 128, 128, attn_mask_start_row)
            >>> attn_mask_start_row_indices = paddle.to_tensor(start_row_indices, dtype=paddle.int32)
            >>> out = paddle.nn.functional.flash_attention.flash_attention_with_sparse_mask(
            >>>     q, q, q,
            >>>     attn_mask_start_row_indices=attn_mask_start_row_indices,
            >>>     attn_mask_start_row=attn_mask_start_row,
            >>>     dropout_p=0.9,
            >>>     is_causal=True,
            >>> )
            >>> print(output)
            >>> # doctest: -SKIP
    """

    assert (
        attn_mask_start_row_indices is not None
    ), f"attn_mask_start_row_indices must be not None, but got {attn_mask_start_row_indices}"
    assert (
        causal is True
    ), f"causal must be True when attn_mask_start_row_indices is not None, but got {causal}"
    assert (
        attn_mask_start_row_indices.dtype == paddle.int32
    ), f"attn_mask_start_row_indices.dtype must be paddle.int32, but got {attn_mask_start_row_indices.dtype}"
    assert isinstance(
        attn_mask_start_row, int
    ), f"attn_mask_start_row must be int, but got {type(attn_mask_start_row)}"
    assert (
        attn_mask_start_row >= 0
    ), f"Should set attn_mask_start_row >=0 when attn_mask_start_row_indices is not None, but got {attn_mask_start_row}"

    fixed_seed_offset = None
    return_softmax = False
    rng_name = ""

    (
        out,
        result_softmax,
        result_softmax_lse,
        result_seed_offset,
    ) = _C_ops.flash_attn_with_sparse_mask(
        query,
        key,
        value,
        attn_mask_start_row_indices,
        fixed_seed_offset,
        dropout,
        causal,
        attn_mask_start_row,
        return_softmax,
        not training,
        rng_name,
    )
    outputs = [out]
    if return_softmax:
        outputs += [result_softmax]
    if return_softmax_lse:
        outputs += [result_softmax_lse]
    if return_seed_offset:
        outputs += [result_seed_offset]
    if len(outputs) == 1:
        return outputs[0]
    else:
        return outputs
