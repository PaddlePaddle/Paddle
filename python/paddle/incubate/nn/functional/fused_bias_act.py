# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

if TYPE_CHECKING:
    from paddle import Tensor

from paddle import _C_ops
from paddle.framework import LayerHelper, in_dynamic_or_pir_mode


def fused_bias_act(
    x: Tensor,
    bias: Tensor | None = None,
    dequant_scales: Tensor | None = None,
    shift: Tensor | None = None,
    smooth: Tensor | None = None,
    act_method: str = "gelu",
    compute_dtype: str = "default",
    quant_scale: float = -1,
    quant_round_type: int = 0,
    quant_max_bound: float = 0,
    quant_min_bound: float = 0,
) -> Tensor:
    """
    Applies fused_bias_act kernel

    Args:
        x (Tensor): the input Tensor.
        bias (Tensor, optional): the input bias Tensor. If it is None, no bias addition would be performed. Otherwise, the bias will be added before activation function. Default: None.
        dequant_scales (Tensor, optional): the dequantization scale tensor, If it is None, no dequantization will be performed. Default: None.
        shift (Tensor, optional): the shift tensor, used to shift the input tensor before activation function. If None, no translation will be performed. Default: None.
        smooth (Tensor, optional): the smooth tensor, used to smooth the input tensor before activation function. If None, no smoothing processing will be performed. Default: None.
        act_method (Str, optional): the activation method, specify the activation function to be used. Default: gelu.
        compute_dtype (Str, optional): a compute dtype, is used to represent the input data type. Default is "default", which means compute dtype is determined by input dtype.
        quant_scale (Float, optional): the quant scale. Default: -1.
        quant_round_type (Int, optional): the quant round type, if 0 is set, value will be rounding to nearest ties to even. If 1 is set, value will be rounding to nearest ties away from zero. Default: 0.
        quant_max_bound (Float, optional): the max bound of float type to int type. Default: 0.
        quant_min_bound (Float, optional): the min bound of float type to int type. Default: 0.


    Returns:
        Tensor: the output Tensor.

    Examples:
        .. code-block:: python

            >>> # doctest: +REQUIRES(env:GPU)
            >>> import paddle
            >>> from paddle.incubate.nn.functional import fused_bias_act

            >>> paddle.set_device('gpu')
            >>> x = paddle.randn([3, 5])
            >>> bias = paddle.randn([5])
            >>> out = fused_bias_act(x, bias)
            >>> print(out.shape)
            [3, 5]
    """
    if in_dynamic_or_pir_mode():

        return _C_ops.fused_bias_act(
            x,
            bias,
            dequant_scales,
            shift,
            smooth,
            act_method,
            compute_dtype,
            quant_scale,
            quant_round_type,
            quant_max_bound,
            quant_min_bound,
        )
    helper = LayerHelper("fused_bias_act")
    if x.dtype == "int32":
        if compute_dtype == "bf16":
            dtype = "uint16"
        elif compute_dtype == "fp16":
            dtype = "float16"
        elif compute_dtype == "fp32":
            dtype = "float32"
        out = helper.create_variable_for_type_inference(dtype=dtype)
    else:
        out = helper.create_variable_for_type_inference(dtype=x.dtype)

    inputs = {}
    inputs["x"] = x
    if bias is not None:
        inputs["bias"] = bias
    if dequant_scales is not None:
        inputs["dequant_scales"] = dequant_scales

    if shift is not None:
        inputs["shift"] = shift

    if smooth is not None:
        inputs["smooth"] = smooth

    attrs = {
        "act_method": act_method,
        "compute_dtype": compute_dtype,
        "quant_scale": quant_scale,
        "quant_round_type": quant_round_type,
        "quant_max_bound": quant_max_bound,
        "quant_min_bound": quant_min_bound,
    }

    helper.append_op(
        type="fused_bias_act",
        inputs=inputs,
        outputs={"out": out},
        attrs=attrs,
    )
    return out
