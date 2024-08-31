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

from typing import TYPE_CHECKING, Literal, overload

import paddle
import paddle.nn.functional as F
from paddle import _C_ops, in_dynamic_mode
from paddle.base.framework import in_dynamic_or_pir_mode
from paddle.base.layer_helper import LayerHelper
from paddle.base.wrapped_decorator import signature_safe_contextmanager

g_enable_math = None
g_enable_flash = None
g_enable_mem_efficient = None

if TYPE_CHECKING:
    from collections.abc import Generator

    from paddle import Tensor


@signature_safe_contextmanager
def sdp_kernel(
    enable_math: bool = False,
    enable_flash: bool = True,
    enable_mem_efficient: bool = True,
) -> Generator[None, None, None]:
    r"""
    With the sdp_kernel context manager, different algorithm implementations can
    be selected for scaled_dot_product_attention.
    """
    global g_enable_math, g_enable_flash, g_enable_mem_efficient
    original_enable_math = g_enable_math
    original_enable_flash = g_enable_math
    original_enable_mem_efficient = g_enable_mem_efficient

    g_enable_math = enable_math
    g_enable_flash = enable_flash
    g_enable_mem_efficient = enable_mem_efficient
    try:
        yield
    finally:
        g_enable_math = original_enable_math
        g_enable_flash = original_enable_flash
        g_enable_mem_efficient = original_enable_mem_efficient


# special for XPU device
def get_triangle_upper_mask(x: Tensor) -> Tensor:
    mask = paddle.full_like(x, -1e4)
    mask.stop_gradient = True
    mask = paddle.triu(mask, diagonal=1)
    mask.stop_gradient = True
    return mask


@overload
def _math_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    dropout_rate: float = ...,
    causal: bool = ...,
    return_softmax: Literal[False] = ...,
    training: bool = ...,
) -> tuple[Tensor, None]: ...


@overload
def _math_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    dropout_rate: float = ...,
    causal: bool = ...,
    return_softmax: Literal[True] = ...,
    training: bool = ...,
) -> tuple[Tensor, Tensor]: ...


@overload
def _math_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    dropout_rate: float = ...,
    causal: bool = ...,
    return_softmax: bool = ...,
    training: bool = ...,
) -> tuple[Tensor, Tensor | None]: ...


def _math_attention(
    query,
    key,
    value,
    dropout_rate=0.0,
    causal=False,
    return_softmax=False,
    training=True,
):
    r"""
    This is a basic implementation of scaled dot product attention composed of
    combinations of fundamental components.
    """
    head_dim = query.shape[-1]
    query = paddle.transpose(query, [0, 2, 1, 3])
    key = paddle.transpose(key, [0, 2, 1, 3])
    value = paddle.transpose(value, [0, 2, 1, 3])
    product = paddle.matmul(x=query * (head_dim**-0.5), y=key, transpose_y=True)

    if not causal:
        weights = F.softmax(product)
    else:
        # special for XPU device
        place = paddle.get_device()
        if "xpu" in place:
            # softmax_mask_fuse_upper_triangle is not supported on XPU, use plain implementation
            mask = get_triangle_upper_mask(product)
            product = product + mask
            weights = F.softmax(product)
        else:
            weights = paddle.incubate.softmax_mask_fuse_upper_triangle(product)
    if dropout_rate > 0.0:
        weights = F.dropout(
            weights, dropout_rate, training=training, mode="upscale_in_train"
        )

    out = paddle.matmul(weights, value)
    out = paddle.transpose(out, [0, 2, 1, 3])
    return out, weights if return_softmax else None


def _select_sdp_cuda(head_dim: int) -> str:
    if head_dim <= 256:
        return "flash_attn"
    else:
        return "mem_efficient"


def _select_sdp(head_dim: int) -> str:
    r"""
    There are currently three different implementation options available for
    scaled dot product attention, and the chosen approach depends on whether it
    is determined by the sdp_kernel configuration or specified through input values.
    """
    place = paddle.get_device()

    if "xpu" in place:
        return "flash_attn"

    # not use sdp_kernel
    if g_enable_flash is None:
        if "gpu" not in place:
            return "math"
        else:
            return _select_sdp_cuda(head_dim)

    if (
        g_enable_math is False
        and g_enable_flash is False
        and g_enable_mem_efficient is False
    ):
        raise AssertionError(
            "No available backend for scaled_dot_product_attention was found."
        )

    if g_enable_math is True:
        if g_enable_flash is False and g_enable_mem_efficient is False:
            return "math"
        if "gpu" not in place:
            return "math"
    if g_enable_flash is True and g_enable_mem_efficient is True:
        return _select_sdp_cuda(head_dim)
    if g_enable_flash is True:
        return "flash_attn"
    return "mem_efficient"


@overload
def flash_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    dropout: float = ...,
    causal: bool = ...,
    return_softmax: Literal[False] = ...,
    *,
    fixed_seed_offset: Tensor | None = ...,
    rng_name: str = ...,
    training: bool = ...,
    name: str | None = ...,
) -> tuple[Tensor, None]: ...


@overload
def flash_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    dropout: float = ...,
    causal: bool = ...,
    return_softmax: Literal[True] = ...,
    *,
    fixed_seed_offset: Tensor | None = ...,
    rng_name: str = ...,
    training: bool = ...,
    name: str | None = ...,
) -> tuple[Tensor, Tensor]: ...


@overload
def flash_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    dropout: float = ...,
    causal: bool = ...,
    return_softmax: bool = ...,
    *,
    fixed_seed_offset: Tensor | None = ...,
    rng_name: str = ...,
    training: bool = ...,
    name: str | None = ...,
) -> tuple[Tensor, Tensor | None]: ...


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
        fixed_seed_offset(Tensor|None, optional): With fixed seed, offset for dropout mask.
        training(bool): Whether it is in the training phase.
        rng_name(str): The name to select Generator.
        name(str|None, optional): The default value is None. Normally there is no need for user
                        to set this property. For more information, please refer to
                        :ref:`api_guide_Name`.

    Returns:
        out(Tensor): The attention tensor.
                    4-D tensor with shape: [batch_size, seq_len, num_heads, head_dim].
                    The dtype can be float16 or bfloat16.
        softmax(Tensor): The softmax tensor. None if return_softmax is False.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> paddle.seed(2023)
            >>> q = paddle.rand((1, 128, 2, 16))

            >>> output = paddle.nn.functional.flash_attention.flash_attention(q, q, q, 0.9, False, False)
            >>> print(output)
            (Tensor(shape=[1, 128, 2, 16], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[[[0.34992966, 0.34456208, 0.45826620, ..., 0.39883569,
                0.42132431, 0.39157745],
               [0.76687670, 0.65837246, 0.69117945, ..., 0.82817286,
                0.76690865, 0.71485823]],
              ...,
              [[0.71662450, 0.57275224, 0.57053083, ..., 0.48108247,
                0.53336465, 0.54540104],
               [0.59137970, 0.51350880, 0.50449550, ..., 0.38860250,
                0.40526697, 0.60541755]]]]), None)

    """
    head_dim = query.shape[3]
    sdp_func_name = _select_sdp(head_dim)

    if sdp_func_name == "flash_attn":
        if in_dynamic_or_pir_mode():
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
    else:
        if sdp_func_name == "mem_efficient":
            from paddle.incubate.nn.memory_efficient_attention import (
                memory_efficient_attention,
            )

            output = memory_efficient_attention(
                query,
                key,
                value,
                attn_bias=None,
                p=dropout,
                scale=None,
                training=training,
            )
            return output, None
        else:
            return _math_attention(
                query,
                key,
                value,
                dropout_rate=dropout,
                causal=causal,
                return_softmax=return_softmax,
                training=training,
            )


@overload
def flash_attn_qkvpacked(
    qkv: Tensor,
    dropout: float = ...,
    causal: bool = ...,
    return_softmax: Literal[False] = ...,
    *,
    fixed_seed_offset: Tensor | None = ...,
    rng_name: str = ...,
    training: bool = ...,
    name: str | None = ...,
) -> tuple[Tensor, None]: ...


@overload
def flash_attn_qkvpacked(
    qkv: Tensor,
    dropout: float = ...,
    causal: bool = ...,
    return_softmax: Literal[True] = ...,
    *,
    fixed_seed_offset: Tensor | None = ...,
    rng_name: str = ...,
    training: bool = ...,
    name: str | None = ...,
) -> tuple[Tensor, Tensor]: ...


@overload
def flash_attn_qkvpacked(
    qkv: Tensor,
    dropout: float = ...,
    causal: bool = ...,
    return_softmax: bool = ...,
    *,
    fixed_seed_offset: Tensor | None = ...,
    rng_name: str = ...,
    training: bool = ...,
    name: str | None = ...,
) -> tuple[Tensor, Tensor | None]: ...


def flash_attn_qkvpacked(
    qkv,
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
        This API only supports inputs with dtype float16 and bfloat16.
        Don't call this API if flash_attn is not supported.

    Args:
        qkv(Tensor): The query/key/value packed tensor in the Attention module.
                        5-D tensor with shape:
                        [batchsize, seqlen , num_heads/num_heads_k + 2, num_heads_k, head_dim].
                        The dtype can be float16 or bfloat16.
        dropout(float): The dropout ratio.
        causal(bool): Whether enable causal mode.
        return_softmax(bool): Whether to return softmax.
        fixed_seed_offset(Tensor|None, optional): With fixed seed, offset for dropout mask.
        training(bool): Whether it is in the training phase.
        rng_name(str): The name to select Generator.
        name(str|None, optional): The default value is None. Normally there is no need for user
                        to set this property. For more information, please refer to
                        :ref:`api_guide_Name`.

    Returns:
        - out(Tensor). The attention tensor. 4-D tensor with shape: [batch_size, seq_len, num_heads, head_dim]. The dtype can be float16 or bfloat16.
        - softmax(Tensor). The softmax tensor. None if return_softmax is False.

    Examples:
        .. code-block:: python

            >>> # doctest: +SKIP('flash_attn need A100 compile')
            >>> import paddle

            >>> paddle.seed(2023)
            >>> q = paddle.rand((1, 128, 2, 16))
            >>> qkv = paddle.stack([q, q, q], axis=2)
            >>> output = paddle.nn.functional.flash_attn_qkvpacked(qkv, 0.9, False, False)
            >>> print(output)
            (Tensor(shape=[1, 128, 2, 16], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[[[0.34992966, 0.34456208, 0.45826620, ..., 0.39883569,
                0.42132431, 0.39157745],
               [0.76687670, 0.65837246, 0.69117945, ..., 0.82817286,
                0.76690865, 0.71485823]],
              ...,
              [[0.71662450, 0.57275224, 0.57053083, ..., 0.48108247,
                0.53336465, 0.54540104],
               [0.59137970, 0.51350880, 0.50449550, ..., 0.38860250,
                0.40526697, 0.60541755]]]]), None)
            >>> # doctest: -SKIP

    """
    head_dim = qkv.shape[-1]
    sdp_func_name = _select_sdp(head_dim)

    if sdp_func_name == "flash_attn":
        if in_dynamic_or_pir_mode():
            (
                result_attention,
                result_softmax,
                _,
                _,
            ) = _C_ops.flash_attn_qkvpacked(
                qkv,
                fixed_seed_offset,
                None,
                dropout,
                causal,
                return_softmax,
                not training,
                rng_name,
            )
            return result_attention, result_softmax if return_softmax else None

        helper = LayerHelper('flash_attn_qkvpacked', **locals())
        dtype = helper.input_dtype(input_param_name='qkv')
        out = helper.create_variable_for_type_inference(dtype)
        softmax = helper.create_variable_for_type_inference(dtype)
        softmax_lse = helper.create_variable_for_type_inference(paddle.float32)
        seed_offset = helper.create_variable_for_type_inference(paddle.int64)
        inputs = {
            'qkv': qkv,
            'fixed_seed_offset': fixed_seed_offset,
        }
        outputs = {
            'out': out,
            'softmax': softmax,
            'softmax_lse': softmax_lse,
            'seed_offset': seed_offset,
        }
        helper.append_op(
            type='flash_attn_qkvpacked',
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
    else:
        # don't call qkvpacked if not using flash_attn
        query = qkv[:, :, :-2].reshape([0, 0, -1, qkv.shape[-1]])
        key = qkv[:, :, -2]
        value = qkv[:, :, -1]
        if sdp_func_name == "mem_efficient":
            from paddle.incubate.nn.memory_efficient_attention import (
                memory_efficient_attention,
            )

            output = memory_efficient_attention(
                query,
                key,
                value,
                attn_bias=None,
                p=dropout,
                scale=None,
                training=training,
            )
            return output, None
        else:
            return _math_attention(
                query,
                key,
                value,
                dropout_rate=dropout,
                causal=causal,
                return_softmax=return_softmax,
                training=training,
            )


@overload
def flash_attn_unpadded(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    cu_seqlens_q: Tensor,
    cu_seqlens_k: Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    scale: float,
    dropout: float = ...,
    causal: bool = ...,
    return_softmax: Literal[False] = ...,
    fixed_seed_offset: Tensor | None = ...,
    rng_name: str = ...,
    training: bool = ...,
    name: str | None = ...,
) -> tuple[Tensor, None]: ...


@overload
def flash_attn_unpadded(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    cu_seqlens_q: Tensor,
    cu_seqlens_k: Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    scale: float,
    dropout: float = ...,
    causal: bool = ...,
    return_softmax: Literal[True] = ...,
    fixed_seed_offset: Tensor | None = ...,
    rng_name: str = ...,
    training: bool = ...,
    name: str | None = ...,
) -> tuple[Tensor, Tensor]: ...


@overload
def flash_attn_unpadded(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    cu_seqlens_q: Tensor,
    cu_seqlens_k: Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    scale: float,
    dropout: float = ...,
    causal: bool = ...,
    return_softmax: bool = ...,
    fixed_seed_offset: Tensor | None = ...,
    rng_name: str = ...,
    training: bool = ...,
    name: str | None = ...,
) -> tuple[Tensor, Tensor | None]: ...


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
    rng_name='',
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
        dropout(float, optional): The dropout ratio.
        causal(bool, optional): Whether enable causal mode.
        return_softmax(bool, optional): Whether to return softmax.
        fixed_seed_offset(Tensor|None, optional): With fixed seed, offset for dropout mask.
        rng_name(str, optional): The name to select Generator.
        training(bool, optional): Whether it is in the training phase.
        name(str|None, optional): The default value is None. Normally there is no need for user
                        to set this property. For more information, please refer to
                        :ref:`api_guide_Name`.

    Returns:
        out(Tensor): The attention tensor.
                    3-D tensor with shape: [total_seq_len, num_heads, head_dim].
                    The dtype can be float16 or bfloat16.
        softmax(Tensor): The softmax tensor. None if return_softmax is False.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> paddle.seed(2023)
            >>> q = paddle.rand((2, 128, 8, 16), dtype='float16')
            >>> cu = paddle.arange(0, 384, 128, dtype='int32')
            >>> qq = paddle.reshape(q, [256, 8, 16])
            >>> output = paddle.nn.functional.flash_attention.flash_attn_unpadded(qq, qq, qq, cu, cu, 128, 128, 0.25, 0.0, False, False)

    """
    if in_dynamic_mode():
        (
            result_attention,
            result_softmax,
        ) = _C_ops.flash_attn_unpadded(
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


@overload
def flash_attn_varlen_qkvpacked(
    qkv: Tensor,
    cu_seqlens_q: Tensor,
    cu_seqlens_k: Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    scale: float,
    dropout: float = ...,
    causal: bool = ...,
    return_softmax: Literal[False] = ...,
    fixed_seed_offset: Tensor | None = ...,
    rng_name: str = ...,
    varlen_padded: bool = ...,
    training: bool = ...,
    name: str | None = ...,
) -> tuple[Tensor, None]: ...


@overload
def flash_attn_varlen_qkvpacked(
    qkv: Tensor,
    cu_seqlens_q: Tensor,
    cu_seqlens_k: Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    scale: float,
    dropout: float = ...,
    causal: bool = ...,
    return_softmax: Literal[True] = ...,
    fixed_seed_offset: Tensor | None = ...,
    rng_name: str = ...,
    varlen_padded: bool = ...,
    training: bool = ...,
    name: str | None = ...,
) -> tuple[Tensor, Tensor]: ...


@overload
def flash_attn_varlen_qkvpacked(
    qkv: Tensor,
    cu_seqlens_q: Tensor,
    cu_seqlens_k: Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    scale: float,
    dropout: float = ...,
    causal: bool = ...,
    return_softmax: bool = ...,
    fixed_seed_offset: Tensor | None = ...,
    rng_name: str = ...,
    varlen_padded: bool = ...,
    training: bool = ...,
    name: str | None = ...,
) -> tuple[Tensor, Tensor | None]: ...


def flash_attn_varlen_qkvpacked(
    qkv,
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
    varlen_padded=True,
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
        qkv(Tensor): The padded query/key/value packed tensor in the Attention module. The padding part won't be computed
                        4-D tensor with shape:
                        [total_seq_len, num_heads/num_heads_k + 2, num_heads_k, head_dim].
                        The dtype can be float16 or bfloat16.
        cu_seqlens_q(Tensor): The cumulative sequence lengths of the sequences in the batch,
                        used to index query.
        cu_seqlens_k(Tensor): The cumulative sequence lengths of the sequences in the batch,
                        used to index key and value.
        max_seqlen_q(int): Maximum sequence length of query in the batch. Note it's the padding length, not the max actual seqlen
        max_seqlen_k(int): Maximum sequence length of key/value in the batch.
        scale(float): The scaling of QK^T before applying softmax.
        dropout(float, optional): The dropout ratio.
        causal(bool, optional): Whether enable causal mode.
        return_softmax(bool, optional): Whether to return softmax.
        fixed_seed_offset(Tensor|None, optional): With fixed seed, offset for dropout mask.
        rng_name(str, optional): The name to select Generator.
        training(bool, optional): Whether it is in the training phase.
        name(str|None, optional): The default value is None. Normally there is no need for user
                        to set this property. For more information, please refer to
                        :ref:`api_guide_Name`.

    Returns:
        - out(Tensor). The attention tensor. The tensor is padded by zeros. 3-D tensor with shape: [total_seq_len, num_heads, head_dim]. The dtype can be float16 or bfloat16.
        - softmax(Tensor). The softmax tensor. None if return_softmax is False.

    Examples:
        .. code-block:: python

            >>> # doctest: +SKIP('flash_attn need A100 compile')
            >>> import paddle
            >>> paddle.seed(2023)
            >>> q = paddle.rand((2, 128, 8, 16), dtype='float16')
            >>> cu = paddle.arange(0, 384, 128, dtype='int32')
            >>> qq = paddle.reshape(q, [256, 8, 16])
            >>> qkv = paddle.stack([qq, qq, qq], axis=2)
            >>> output = paddle.nn.functional.flash_attn_varlen_qkvpacked(qkv, cu, cu, 128, 128, 0.25, 0.0, False, False)
            >>> # doctest: -SKIP

    """
    if in_dynamic_mode():
        (
            result_attention,
            result_softmax,
        ) = _C_ops.flash_attn_varlen_qkvpacked(
            qkv,
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
            varlen_padded,
        )
        return result_attention, result_softmax if return_softmax else None

    helper = LayerHelper('flash_attn_varlen_qkvpacked', **locals())
    dtype = helper.input_dtype(input_param_name='qkv')
    out = helper.create_variable_for_type_inference(dtype)
    softmax = helper.create_variable_for_type_inference(dtype)
    softmax_lse = helper.create_variable_for_type_inference(paddle.float32)
    seed_offset = helper.create_variable_for_type_inference(paddle.int64)
    inputs = {
        'qkv': qkv,
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
        type='flash_attn_varlen_qkvpacked',
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


def scaled_dot_product_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    attn_mask: Tensor | None = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    training: bool = True,
    name: str | None = None,
) -> Tensor:
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
                        The dtype can be float16 or bfloat16.
        key(Tensor): The key tensor in the Attention module.
                        4-D tensor with shape:
                        [batch_size, seq_len, num_heads, head_dim].
                        The dtype can be float16 or bfloat16.
        value(Tensor): The value tensor in the Attention module.
                        4-D tensor with shape:
                        [batch_size, seq_len, num_heads, head_dim].
                        The dtype can be float16 or bfloat16.
        attn_mask(Tensor, optional): A float mask of the same type as query,
                        key, value that is added to the attention score.
        dropout_p(float, optional): The dropout ratio.
        is_causal(bool, optional): Whether enable causal mode.
        training(bool, optional): Whether it is in the training phase.
        name(str|None, optional): The default value is None. Normally there is no need for user
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
            >>> q = paddle.rand((1, 128, 2, 16), dtype=paddle.bfloat16)
            >>> output = paddle.nn.functional.scaled_dot_product_attention(q, q, q, None, 0.9, False)
            >>> print(output)
            >>> # doctest: -SKIP
    """

    if attn_mask is None:
        # downgraded to ordinary flash attention implementation
        out, _ = flash_attention(query, key, value, dropout_p, is_causal)
        return out
    else:
        if in_dynamic_or_pir_mode():
            fixed_seed_offset = None
            return_softmax = False
            rng_name = ""
            out, _, _, _ = _C_ops.flash_attn(
                query,
                key,
                value,
                fixed_seed_offset,
                attn_mask,
                dropout_p,
                is_causal,
                return_softmax,
                not training,
                rng_name,
            )
            return out
        else:
            helper = LayerHelper('flash_attn', **locals())
            dtype = helper.input_dtype(input_param_name='q')
            out = helper.create_variable_for_type_inference(dtype)
            softmax = helper.create_variable_for_type_inference(dtype)
            softmax_lse = helper.create_variable_for_type_inference(
                paddle.float32
            )
            seed_offset = helper.create_variable_for_type_inference(
                paddle.int64
            )
            inputs = {
                'q': query,
                'k': key,
                'v': value,
                'attn_mask': attn_mask,
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
                    'dropout': dropout_p,
                    'causal': is_causal,
                    'return_softmax': False,
                    'is_test': not training,
                    'rng_name': '',
                },
            )
            return out


def flashmask_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    startend_row_indices: Tensor | None = None,
    *,
    dropout: float = 0.0,
    causal: bool = False,
    window_size: int | tuple | None = None,
    return_softmax_lse: bool = False,
    return_seed_offset: bool = False,
    fixed_seed_offset: Tensor | None = None,
    rng_name: str = "",
    training: bool = True,
    name: str | None = None,
):
    r"""
    Implements FlashAttention with a sparse mask representation.

    The equation is:

    .. math::

        result = softmax(\frac{Q \cdot K^T}{\sqrt{d}} + M) \cdot V

    where ``Q``, ``K``, and ``V`` are the input tensors of the attention module.
    They share the same dimensions, and ``d`` represents the size of the last dimension.
    ``M`` is the dense mask.

    The figure below shows examples of various masks, with the Score matrix depicted. Gray areas indicate elements that are masked. The numbers above represent the values of `startend_row_indices`. A single row of numbers indicates that the shape of `startend_row_indices` is `[batch_size, num_heads, seq_len, 1]`. Two rows of numbers indicate that the shape of `startend_row_indices` is `[batch_size, num_heads, seq_len, 2]`. Four rows of numbers indicate that the shape of `startend_row_indices` is `[batch_size, num_heads, seq_len, 4]`.

    .. image:: https:/githubraw.cdn.bcebos.com/PaddlePaddle/docs/develop/docs/images/FlashMask1.png
        :width: 900
        :alt: pipeline
        :align: center

    In Figure (a), where `causal=True`, the values of `startend_row_indices` are as follows:

    .. code-block:: python

        >>> # doctest: +SKIP('Only example')
        >>> print(startend_row_indices)
        Tensor(shape=[1, 1, 10, 1], dtype=int32, place=Place(gpu:0), stop_gradient=True,
            [[[[5 ],
                [5 ],
                [5 ],
                [5 ],
                [5 ],
                [5 ],
                [5 ],
                [5],
                [5],
                [5]]]])
        >>> # doctest: -SKIP


    In Figure (b), where `causal=True`, the values of `startend_row_indices` are as follows:

    .. code-block:: python

        >>> # doctest: +SKIP('Only example')
        >>> print(startend_row_indices)
        Tensor(shape=[1, 1, 10, 1], dtype=int32, place=Place(gpu:0), stop_gradient=True,
            [[[[4 ],
                [4 ],
                [4 ],
                [4 ],
                [7 ],
                [7 ],
                [7 ],
                [10],
                [10],
                [10]]]])
        >>> # doctest: -SKIP

    .. image:: https:/githubraw.cdn.bcebos.com/PaddlePaddle/docs/develop/docs/images/FlashMask2.png
        :width: 900
        :alt: pipeline
        :align: center

    In Figure (c), where `causal=True`, the values of `startend_row_indices` are as follows:

    .. code-block:: python

        >>> # doctest: +SKIP('Only example')
        >>> print(startend_row_indices)
        Tensor(shape=[1, 1, 10, 1], dtype=int32, place=Place(gpu:0), stop_gradient=True,
            [[[[10 ],
                [10 ],
                [10 ],
                [10 ],
                [7 ],
                [7 ],
                [7 ],
                [10],
                [10],
                [10]]]])
        >>> # doctest: -SKIP

    In Figure (d), where `causal=True`, the values of `startend_row_indices` are as follows:

    .. code-block:: python

        >>> # doctest: +SKIP('Only example')
        >>> print(startend_row_indices)
        Tensor(shape=[1, 1, 10, 1], dtype=int32, place=Place(gpu:0), stop_gradient=True,
            [[[[10 ],
                [4 ],
                [5 ],
                [6 ],
                [7 ],
                [8 ],
                [9 ],
                [10],
                [10],
                [10]]]])
        >>> # doctest: -SKIP

    .. image:: https:/githubraw.cdn.bcebos.com/PaddlePaddle/docs/develop/docs/images/FlashMask3.png
        :width: 900
        :alt: pipeline
        :align: center

    In Figure (e), where `causal=True`, the values of `startend_row_indices` are as follows:

    .. code-block:: python

        >>> # doctest: +SKIP('Only example')
        >>> print(startend_row_indices)
        Tensor(shape=[1, 1, 10, 2], dtype=int32, place=Place(gpu:0), stop_gradient=True,
            [[[[4 , 7 ],
                [4 , 7 ],
                [4 , 7 ],
                [4 , 7 ],
                [10, 10],
                [10, 10],
                [10, 10],
                [10, 10],
                [10, 10],
                [10, 10]]]])
        >>> # doctest: -SKIP

    In Figure (f), where `causal=False`, the values of `startend_row_indices` are as follows:

    .. code-block:: python

        >>> # doctest: +SKIP('Only example')
        >>> print(startend_row_indices)
        Tensor(shape=[1, 1, 10, 2], dtype=int32, place=Place(gpu:0), stop_gradient=True,
            [[[[4 , 0 ],
                [4 , 0 ],
                [4 , 0 ],
                [4 , 0 ],
                [7, 4],
                [7, 4],
                [7, 4],
                [10, 7],
                [10, 7],
                [10, 7]]]])
        >>> # doctest: -SKIP

    .. image:: https:/githubraw.cdn.bcebos.com/PaddlePaddle/docs/develop/docs/images/FlashMask4.png
        :width: 900
        :alt: pipeline
        :align: center

    In Figure (g), where `causal=False`, the values of `startend_row_indices` are as follows:

    .. code-block:: python

        >>> # doctest: +SKIP('Only example')
        >>> print(startend_row_indices)
        Tensor(shape=[1, 1, 10, 4], dtype=int32, place=Place(gpu:0), stop_gradient=True,
            [[[[10, 10, 0 , 0 ],
                [10, 10, 0 , 0 ],
                [10, 10, 0 , 0 ],
                [3 , 10, 0 , 0 ],
                [4 , 10, 3 , 4 ],
                [5 , 10, 3 , 5 ],
                [6 , 10, 3 , 6 ],
                [7 , 10, 3 , 7 ],
                [8 , 10, 3 , 8 ],
                [9 , 10, 3 , 9 ]]]])
        >>> # doctest: -SKIP

    In Figure (h), where `causal=True`, the values of `startend_row_indices` are as follows:

    .. code-block:: python

        >>> # doctest: +SKIP('Only example')
        >>> print(startend_row_indices)
        Tensor(shape=[1, 1, 10, 1], dtype=int32, place=Place(gpu:0), stop_gradient=True,
            [[[[10 ],
                [4 ],
                [8 ],
                [6 ],
                [10 ],
                [7 ],
                [10 ],
                [9],
                [10],
                [10]]]])
        >>> # doctest: -SKIP

    Warning:
        This API only supports inputs with dtype float16 and bfloat16.

    Args
    ::::::::::::
        - **query** (Tensor) - The query tensor in the attention module.
                        A 4-D tensor with shape [batch_size, seq_len, num_heads, head_dim].
                        The dtype can be float16 or bfloat16.
        - **key** (Tensor) - The key tensor in the attention module.
                        A 4-D tensor with shape [batch_size, seq_len, num_heads, head_dim].
                        The dtype can be float16 or bfloat16.
        - **value** (Tensor) - The value tensor in the attention module.
                        A 4-D tensor with shape [batch_size, seq_len, num_heads, head_dim].
                        The dtype can be float16 or bfloat16.
        - **startend_row_indices** (Tensor)
            - A sparse attention mask indices tensor.
                A 4-D tensor with shape [batch_size, num_heads, seq_len, {1, 2, 4}].
                The dtype must be int32. num_heads can be 1 or the same as key's num_heads. When num_heads is 1, it will be broadcast to match key's num_heads.
                Depending on the value of the causal parameter, startend_row_indices can take different shapes and meanings, with the values in startend_row_indices being denoted as r1, r2, r3, r4 sequentially.
            - When `causal=True` and the shape is [batch_size, num_heads, seq_len, 1],
                indicating unidirectional attention. The value represents the starting row index of the left
                lower triangular mask in the dense mask. The value r1 in startend_row_indices indicates that elements in the lower left triangle of the Score matrix starting from the r1-th row downwards (inclusive) will be masked.
            - When `causal=True` and the shape is [batch_size, num_heads, seq_len, 2],
                indicating unidirectional attention. The values represent the starting and ending row indices of
                the left lower triangular mask in the dense mask. The values r1, r2 in startend_row_indices indicate that elements in the lower left triangle of the Score matrix starting from the r1-th row downwards (inclusive) but above the r2-th row (exclusive) will be masked.
            - When `causal=False` and the shape is [batch_size, num_heads, seq_len, 2],
                indicating bidirectional attention. The values represent the starting row index of the left
                lower triangular mask and the ending row index of the right upper triangular mask in the dense mask. The values r1, r2 in startend_row_indices indicate that elements in the lower left triangle of the Score matrix starting from the r1-th row downwards (inclusive) will be masked, and elements in the upper right triangle starting from the r2-th row upwards (exclusive) will be masked.
            - When `causal=False` and the shape is [batch_size, num_heads, seq_len, 4] ,
                indicating bidirectional attention. The values represent the start and end row indices of the
                left lower triangular mask and the start and end row indices of the right upper triangular mask in the dense mask. The values r1, r2, r3, r4 in startend_row_indices indicate that elements in the lower left triangle of the Score matrix starting from the r1-th row downwards (inclusive) but above the r2-th row (exclusive) will be masked, and elements in the upper right triangle starting from the r3-th row downwards (inclusive) but above the r4-th row (exclusive) will be masked.
        - **dropout** (float) - The dropout ratio. Default is 0.0.
        - **causal** (bool) - Whether to enable causal mode. Default is False.
        - **window_size** (int|tuple, optional) - Indicates the window size of sliding window local attention.
                        If causal mode is enabled, Query at position i will only attend to keys between [i - window_size, i] or [i - window_size[0], i].
                        If causal mode is disabled, Query at position i will only attend to keys between [i - window_size, i + window_size] or [i - window_size[0], i + window_size[1]].
        - **return_softmax_lse** (bool) - Whether to return the log-sum-exp of the softmax. Default is False.
        - **return_seed_offset** (bool) - Whether to return the random seed offset. Default is False.
        - **fixed_seed_offset** (Tensor, optional): With fixed seed, offset for dropout mask.
        - **rng_name** (str) - The name to select Generator.
        - **training** (bool) - Whether the module is in training mode. Default is True.
        - **name** (str, optional) - Name of the operation. Default is None. Normally, users do not need to set this property.
                                For more information, refer to :ref:`api_guide_Name` .

    Returns
    ::::::::::::
        Tensor: The computed attention result with the same shape as the input `value`.

    Examples:
        .. code-block:: python

            >>> # doctest: +SKIP('flash_attn need A100 compile')
            >>> import paddle

            >>> paddle.seed(2023)
            >>> q = paddle.rand((1, 128, 2, 32),dtype="float16")
            >>> startend_row_indices = paddle.randint(0, 128, (1, 2, 128, 1), dtype="int32")
            >>> output = paddle.nn.functional.flashmask_attention(q, q, q, startend_row_indices, causal=True)
            >>> print(output)
            Tensor(shape=[1, 128, 2, 32], dtype=float16, place=Place(gpu:0), stop_gradient=True,
           [[[[0.81201172, 0.99609375, 0.51074219, ..., 0.80126953,
               0.07232666, 0.83496094],
              [0.34838867, 0.44970703, 0.56103516, ..., 0.68164062,
               0.10986328, 0.07733154]],

             [[0.68603516, 0.85253906, 0.51074219, ..., 0.72119141,
               0.37426758, 0.44531250],
              [0.20300293, 0.79833984, 0.81738281, ..., 0.87890625,
               0.68994141, 0.58496094]],

             [[0.39990234, 0.57080078, 0.40942383, ..., 0.87158203,
               0.14978027, 0.77343750],
              [0.18750000, 0.79443359, 0.76904297, ..., 0.86865234,
               0.76171875, 0.61035156]],

             ...,

             [[0.29321289, 0.67675781, 0.47143555, ..., 0.36621094,
               0.61035156, 0.35668945],
              [0.45825195, 0.21228027, 0.72949219, ..., 0.77246094,
               0.41723633, 0.41870117]],

             [[0.76660156, 0.55322266, 0.73876953, ..., 0.26416016,
               0.63769531, 0.55810547],
              [0.69677734, 0.59863281, 0.77783203, ..., 0.64599609,
               0.36059570, 0.42919922]],

             [[0.31030273, 0.91064453, 0.71826172, ..., 0.29125977,
               0.34423828, 0.60986328],
              [0.73583984, 0.84619141, 0.96728516, ..., 0.61816406,
               0.07440186, 0.55224609]]]])
            >>> # doctest: -SKIP

    """

    if window_size is not None:
        if isinstance(window_size, int):
            window_size = (window_size, window_size)
        sq = query.shape[1]
        bsz = query.shape[0]
        assert (
            startend_row_indices is None
        ), "can't use window_size with startend_row_indices"
        if causal:
            startend_row_indices = paddle.arange(
                window_size[0] + 1, sq + window_size[0] + 1, dtype="int32"
            ).reshape((1, 1, sq, 1))
            startend_row_indices = paddle.clip(
                startend_row_indices, max=sq
            ).repeat_interleave(bsz, 0)

        else:
            startend_row_indices = paddle.empty((1, 1, sq, 2), dtype="int32")
            startend_row_indices[0, 0, :, 0] = paddle.arange(
                window_size[0] + 1, sq + window_size[0] + 1, dtype="int32"
            )
            startend_row_indices[0, 0, :, 1] = paddle.arange(
                -window_size[1], sq - window_size[1], dtype="int32"
            )
            startend_row_indices = paddle.clip(
                startend_row_indices, min=0, max=sq
            ).repeat_interleave(bsz, 0)

    if startend_row_indices is None:
        (
            out,
            result_softmax,
            result_softmax_lse,
            result_seed_offset,
        ) = _C_ops.flash_attn(
            query,
            key,
            value,
            fixed_seed_offset,
            None,
            dropout,
            causal,
            False,
            not training,
            rng_name,
        )

    else:
        assert (
            startend_row_indices.dtype == paddle.int32
        ), f"startend_row_indices.dtype must be paddle.int32, but got {startend_row_indices.dtype}"
        assert (
            len(startend_row_indices.shape) == 4
        ), f"startend_row_indices rank must be 4,but got {startend_row_indices.shape}"

        assert (
            startend_row_indices.shape[0] == key.shape[0]
        ), f"startend_row_indices.shape[0] must be equal to batch_size, but got {startend_row_indices.shape[0]} and {key.shape[0]}"

        assert (
            startend_row_indices.shape[2] == key.shape[1]
        ), f"startend_row_indices.shape[2] must be equal to seqlen_k, but got {startend_row_indices.shape[2]} and {key.shape[2]}"
        assert startend_row_indices.shape[1] in [
            1,
            key.shape[2],
        ], "startend_row_indices head_num must be equal to 1(broadcast) or hean_num_k."

        if causal:
            if startend_row_indices.shape[-1] == 1:
                has_end = False
            elif startend_row_indices.shape[-1] == 2:
                has_end = True
            else:
                raise ValueError(
                    f"Invalid shape of startend_row_indices, when causal is True, the last dimension should be either 1 or 2 but got {startend_row_indices.shape[-1]}"
                )
        else:
            if startend_row_indices.shape[-1] == 2:
                has_end = False
            elif startend_row_indices.shape[-1] == 4:
                has_end = True
                raise NotImplementedError(
                    "ending row index is not implemented yet."
                )
            else:
                raise ValueError(
                    f"Invalid shape of startend_row_indices, when causal is False, the last dimension should be either 2 or 4 but got {startend_row_indices.shape[-1]}"
                )

        (
            out,
            result_softmax,
            result_softmax_lse,
            result_seed_offset,
        ) = _C_ops.flashmask_attention(
            query,
            key,
            value,
            startend_row_indices,
            fixed_seed_offset,
            dropout,
            causal,
            False,
            not training,
            rng_name,
        )

    outputs = [out]
    if return_softmax_lse:
        outputs += [result_softmax_lse]
    if return_seed_offset:
        outputs += [result_seed_offset]
    if len(outputs) == 1:
        return outputs[0]
    else:
        return outputs


def calc_reduced_attention_scores(
    query: paddle.Tensor, key: paddle.Tensor, softmax_lse: paddle.Tensor
) -> paddle.Tensor:
    r"""
    The equation is:

    .. math::

        result=reduce\_sum(softmax(\frac{ Q * K^T }{\sqrt{d}}), dim=-2)

    Warning:
        This API only supports inputs with dtype float16 and bfloat16.

    Args:
        query(Tensor): The query tensor in the Attention module.
                        4-D tensor with shape:
                        [batch_size, seqlen_q, num_heads, head_dim].
                        The dtype can be float16 or bfloat16.
        key(Tensor): The key tensor in the Attention module.
                        4-D tensor with shape:
                        [batch_size, seqlen_k, num_heads, head_dim].
                        The dtype can be float16 or bfloat16.
        softmax_lse(Tensor): The logsumexp of each row returned by _C_ops.flash_attn().
                        3-D tensor with shape:
                        [batch_size, num_heads, seqlen_q_rounded], where seqlen_q_rounded = ceil(seqlen_q/128).
                        The dtype is float32.
    Returns:
        reduced_attention_scores(Tensor), The reduce sum of attention scores across seqlen_q.
        4-D tensor with shape: [batch_size, num_heads, 1, seqlen_k]. The dtype is float32.
    Examples:
        .. code-block:: python

            >>> # doctest: +SKIP('reduce_attn_scores need A100 compile')
            >>> import paddle
            >>> import numpy as np
            >>> import paddle._C_ops as _C_ops
            >>> from paddle.nn.functional.flash_attention import (
            >>>     calc_reduced_attention_scores
            >>> )
            >>> np.random.seed(2024)
            >>> q_shape = (5,1024,16,128)
            >>> k_shape = (5,2048,16,128)
            >>> dtype = 'float16'
            >>> query = np.random.random(q_shape)
            >>> key = np.random.random(k_shape)
            >>> q = paddle.to_tensor(
            >>>     query, place=place, dtype=dtype, stop_gradient=True
            >>> )
            >>> k = paddle.to_tensor(
            >>>     key, place=place, dtype=dtype, stop_gradient=True
            >>> )
            >>> _, _, softmax_lse, _ = _C_ops.flash_attn(
            >>>     q,
            >>>     k,
            >>>     k,
            >>>     (None,), #fixed_seed_offset
            >>>     None, #attn_mask
            >>>     0.0, #dropout
            >>>     False, #causal
            >>>     False, #return_softmax
            >>>     False, #is_test
            >>>     "" #rng_name
            >>> )
            >>> reduced_attn_scores = calc_reduced_attention_scores(
            >>>     q,
            >>>     k,
            >>>     softmax_lse,
            >>> )
            >>> # doctest: -SKIP
    """
    assert (
        query.stop_gradient and key.stop_gradient
    ), 'calc_reduced_attention_scores() is for inference only.'

    if in_dynamic_or_pir_mode():
        reduced_scores = _C_ops.calc_reduced_attn_scores(
            query, key, softmax_lse
        )
        return reduced_scores

    helper = LayerHelper('calc_reduced_attn_scores', **locals())
    reduced_scores = helper.create_variable_for_type_inference(paddle.float32)
    softmax = helper.create_variable_for_type_inference(paddle.float32)
    inputs = {
        'q': query,
        'k': key,
        'softmax_lse': softmax_lse,
    }
    outputs = {
        'reduced_scores': reduced_scores,
    }
    helper.append_op(
        type='calc_reduced_attn_scores',
        inputs=inputs,
        outputs=outputs,
    )
    return reduced_scores
