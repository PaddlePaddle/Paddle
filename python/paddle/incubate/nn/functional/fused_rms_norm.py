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

from __future__ import annotations

from typing import TYPE_CHECKING, overload

import paddle
from paddle import _C_ops
from paddle.framework import LayerHelper, in_dynamic_or_pir_mode

if TYPE_CHECKING:
    from paddle import Tensor


@overload
def fused_layer_norm(
    x: Tensor,
    norm_weight: Tensor,
    norm_bias: Tensor,
    epsilon: float,
    begin_norm_axis: int,
    bias: Tensor | None = ...,
    residual: None = ...,
    quant_scale: float = ...,
    quant_round_type: float = ...,
    quant_max_bound: float = ...,
    quant_min_bound: float = ...,
) -> Tensor: ...


@overload
def fused_layer_norm(
    x: Tensor,
    norm_weight: Tensor,
    norm_bias: Tensor,
    epsilon: float,
    begin_norm_axis: int,
    bias: Tensor | None = ...,
    residual: Tensor = ...,
    quant_scale: float = ...,
    quant_round_type: float = ...,
    quant_max_bound: float = ...,
    quant_min_bound: float = ...,
) -> tuple[Tensor, Tensor]: ...


def fused_rms_norm(
    x,
    norm_weight,
    norm_bias,
    epsilon,
    begin_norm_axis,
    bias=None,
    residual=None,
    quant_scale=-1,
    quant_round_type=0,
    quant_max_bound=0,
    quant_min_bound=0,
):
    r"""
    Apply Fused RMSNorm kernel. Also support RMSNorm(bias + residual + x) fused pattern.

    Args:
        x (Tensor): the input Tensor..
        norm_weight (Tensor): the weight Tensor to affine output.
        norm_bias (Tensor): the bias Tensor to affine output.
        epsilon (float): a small float number to avoid divide 0.
        begin_norm_axis (int): the begin axis to normalize.
        bias (optional|Tensor): the previous layers's bias to fused.
        residual (optional|Tensor): the residual input to fused.
        quant_scale (float): the quant scale.
        quant_round_type (float): the quant round type.
        quant_max_bound (float): the quant max bound to clip.
        quant_min_bound (float): the quant min bound to clip.


    Returns:
        Tensor: the output Tensor.

    Examples:
        .. code-block:: python

            >>> # doctest: +REQUIRES(env:GPU)
            >>> import paddle
            >>> paddle.device.set_device('gpu')

            >>> paddle_x = paddle.cast(paddle.randn(shape=[32, 256]), dtype=paddle.float16)
            >>> paddle_weight = paddle.cast(paddle.randn(shape=[256]), dtype=paddle.float16)
            >>> paddle_bias = paddle.cast(paddle.randn(shape=[256]), dtype=paddle.float16)
            >>> epsilon = 1e-6
            >>> paddle_rmsnorm = paddle.incubate.nn.functional.fused_rms_norm(paddle_x, paddle_weight, paddle_bias, epsilon, 1)
    """
    if in_dynamic_or_pir_mode():
        return _C_ops.rms_norm(
            x,
            bias,
            residual,
            norm_weight,
            norm_bias,
            epsilon,
            begin_norm_axis,
            quant_scale,
            quant_round_type,
            quant_max_bound,
            quant_min_bound,
        )
    # static mode
    helper = LayerHelper('rms_norm', **locals())
    out = None
    if quant_scale <= 0:
        out = helper.create_variable_for_type_inference(dtype=x.dtype)
    else:
        out = helper.create_variable_for_type_inference(dtype=paddle.int8)
    outputs_dict = {}
    outputs_dict['out'] = out

    residual_out = helper.create_variable_for_type_inference(dtype=x.dtype)
    outputs_dict['residual_out'] = residual_out

    inv_var = helper.create_variable_for_type_inference(dtype=paddle.float32)
    outputs_dict['inv_var'] = inv_var

    inputs = {'x': x, 'norm_weight': norm_weight}
    if norm_bias is not None:
        inputs['norm_bias'] = norm_bias
    if residual is not None:
        inputs['residual'] = residual
    if bias is not None:
        inputs['bias'] = bias

    helper.append_op(
        type='rms_norm',
        inputs=inputs,
        attrs={
            "epsilon": epsilon,
            "begin_norm_axis": begin_norm_axis,
            "quant_scale": quant_scale,
            "quant_round_type": quant_round_type,
            "quant_max_bound": quant_max_bound,
            "quant_min_bound": quant_min_bound,
        },
        outputs=outputs_dict,
    )
    return (out, residual_out, outputs_dict['inv_var'])
