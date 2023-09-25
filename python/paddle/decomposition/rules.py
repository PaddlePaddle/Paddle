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

from paddle import _pir_ops

from .primitives import *  # noqa: F403
from .register import register_decomp


@register_decomp('pd_op.mean')
def mean(x, axis, keepdim):
    """define composite rule of op mean"""
    x_shape = x.shape
    if axis in (None, []):
        axis = tuple(range(0, len(x_shape)))
    axes = (axis,) if isinstance(axis, int) else axis
    sum_x = sum(x, axis=axes, keepdim=keepdim)
    value_to_fill = 1
    for axis in axes:
        value_to_fill *= x_shape[axis]
    norm = fill_constant(
        shape=[],
        value=value_to_fill,
        dtype=sum_x.dtype,
    )
    res = sum_x / norm
    return res


@register_decomp('pd_op.gelu')
def gelu_composite(x, approximate):
    """define composite rule of op gelu"""
    M_SQRT1_2 = (
        0.70710678118654752440  # /* 1/sqrt(2) */ copy from gelu-kernel.cc
    )
    M_2_SQRTPI = 1.12837916709551257390  # /* 2/sqrt(pi) */
    full_shape = x.shape if len(x.shape) == 0 else [1]
    one = ones(full_shape, x.dtype)
    half = full(full_shape, 0.5, x.dtype)
    # Todo(cz): after symbol overload, add and multiply will be replaced by "+" and "*"
    if approximate:
        # gelu(x) = 0.5 * x * (1 + tanh(sqrt(2 / \pi) * (x + 0.044715 * x^{3})))
        kAlpha = full(full_shape, M_2_SQRTPI * M_SQRT1_2, x.dtype)
        GELU_CONSTANT = full(full_shape, 0.044715, x.dtype)
        tanh_out = tanh(kAlpha * (x + GELU_CONSTANT * x * x * x))
        out = x * half * (one + tanh_out)
        return out

    else:
        # gelu(x) = 0.5 * x *  (1 + erf(x / sqrt(2)))

        cdf = half * (one + _pir_ops.erf(x * full(x.shape, M_SQRT1_2, x.dtype)))
        out = x * cdf
        return out


@register_decomp('pd_op.rsqrt')
def rsqrt_composite(x):
    """define composite rule of op rsqrt."""
    # rsqrt(x) = x^(-0.5)
    is_amp = False
    from paddle.base.data_feeder import convert_dtype

    dtype = convert_dtype(x.dtype)
    if dtype in ["float16", "uint16"]:
        is_amp = True
        x = cast(x, "float32")
    y = full(x.shape if len(x.shape) == 0 else [1], -0.5, x.dtype)
    res = pow(x, y)
    return res if not is_amp else cast(res, dtype)


@register_decomp('pd_op.pow')
def pow_composite(x, y):
    """
    define composite rule of op pow
    res = x^y
    """
    is_amp = False
    from paddle.base.data_feeder import convert_dtype

    dtype = convert_dtype(x.dtype)
    if dtype in ["float16", "uint16"]:
        is_amp = True
        x = cast(x, "float32")

    if isinstance(y, (int, float)):
        y = full(x.shape if len(x.shape) == 0 else [1], y, x.dtype)
    res = pow(x, y)
    if is_amp:
        res = cast(res, dtype)
    return res


@register_decomp('pd_op.layer_norm')
def layernorm_composite(x, scale, bias, epsilon, begin_norm_axis):
    """
    define composite rule of op layer_norm
    out = (x - mean(x)) / sqrt(var + epsilon))
    var = mean((x-mean(x))^2)
    """
    is_amp = False
    from paddle.base.data_feeder import convert_dtype

    dtype = convert_dtype(x.dtype)
    if dtype in ["float16", "uint16"]:
        is_amp = True
        x = cast(x, "float32")
        scale = cast(scale, "float32") if scale else scale
        bias = cast(bias, "float32") if bias else bias

    axis = tuple(range(begin_norm_axis, len(x.shape)))
    mean_ = mean(x, axis=axis, keepdim=True)
    difference = x - mean_
    var_tmp1 = difference * difference
    variance = mean(var_tmp1, axis=axis, keepdim=True)
    var_tmp3 = variance + epsilon
    rsqrt_var = rsqrt(var_tmp3)
    out = difference * rsqrt_var

    if scale is not None:
        if x.shape[begin_norm_axis:] != scale.shape:
            scale = reshape(scale, x.shape[begin_norm_axis:])
        out = out * scale
    if bias is not None:
        if x.shape[begin_norm_axis:] != bias.shape:
            bias = reshape(bias, x.shape[begin_norm_axis:])
        out = out + bias

    mean_ = reshape(mean_, [-1])
    variance = reshape(variance, [-1])
    if is_amp:
        out = cast(out, dtype)
    return out, mean_, variance


@register_decomp('pd_op.add_n')
def sum_composite(x):
    ans = x[0]
    for xi in x[1:]:
        ans = xi + ans
    return ans
