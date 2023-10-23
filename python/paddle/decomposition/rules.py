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
def gelu(x, approximate):
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


@register_decomp('pd_op.sqrt')
def sqrt(x):
    """
    define composite rule of op sqrt
    res = pow(x, 0.5)
    """
    is_amp = False
    from paddle.base.data_feeder import convert_dtype

    dtype = convert_dtype(x.dtype)
    if dtype in ["float16", "uint16"]:
        is_amp = True
        x = cast(x, "float32")

    y = full(x.shape if len(x.shape) == 0 else [1], 0.5, x.dtype)
    res = pow_composite(x, y)
    return res if not is_amp else cast(res, dtype)


@register_decomp('pd_op.rsqrt')
def rsqrt(x):
    """define composite rule of op rsqrt."""
    # rsqrt(x) = x^(-0.5)
    is_amp = False
    from paddle.base.data_feeder import convert_dtype

    dtype = convert_dtype(x.dtype)
    if dtype in ["float16", "uint16"]:
        is_amp = True
        x = cast(x, "float32")
    y = full(x.shape if len(x.shape) == 0 else [1], -0.5, x.dtype)
    res = pow_composite(x, y)
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
def layernorm(x, scale, bias, epsilon, begin_norm_axis):
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


@register_decomp('pd_op.dropout')
def dropout(x, seed_tensor, p, is_test, mode, seed, fix_seed):
    """define composite rule of op dropout.
    upscale_in_train:
        train: out = input * mask / ( 1.0 - p )
        inference: out = input
    downscale_in_infer
        train: out = input * mask
        inference: out = input * (1.0 - p)
    """
    from paddle import assign
    from paddle.base import core
    from paddle.base.data_feeder import convert_dtype

    fix_seed = True if fix_seed is None else fix_seed
    seed = seed if fix_seed else 0
    upscale_in_train = mode == "upscale_in_train"

    x_dtype = convert_dtype(x.dtype)
    mask = bernoulli(shape=x.shape, dtype=x_dtype, p=p, seed=seed)

    uint8_type = convert_dtype(core.VarDesc.VarType.UINT8)
    if upscale_in_train:
        if not is_test:
            # Process p=1.0 for avoid devide zero error (x*mask/(1.0-p))
            if p == 1.0:
                return fill_constant(
                    shape=x.shape, value=0.0, dtype=x.dtype
                ) * x, zeros(x.shape, uint8_type)
            else:
                return x * mask / fill_constant(
                    shape=x.shape, value=(1.0 - p), dtype=x.dtype
                ), cast(mask, uint8_type)
        else:
            return assign(x), cast(mask, uint8_type)
    else:
        if not is_test:
            return x * mask, cast(mask, uint8_type)
        else:
            return x * fill_constant(
                shape=x.shape, value=(1.0 - p), dtype=x.dtype
            ), cast(mask, uint8_type)


def bernoulli(shape, dtype, p, seed=0):
    from paddle.base.data_feeder import convert_dtype

    # TODO(jiabin) Fix uniform doesn't support float16 error in CINN
    new_dtype = (
        "float32" if convert_dtype(dtype) in ["float16", "uint16"] else dtype
    )
    return cast(
        greater_equal(
            uniform(shape, new_dtype, min=0.0, max=1.0, seed=seed),
            fill_constant(shape if len(shape) == 0 else [1], new_dtype, p),
        ),
        dtype,
    )


@register_decomp('pd_op.add_n')
def add_n(x):
    ans = x[0]
    for xi in x[1:]:
        ans = xi + ans
    return ans


@register_decomp('pd_op.silu')
def silu(x):
    """
    define composite rule of op silu
    res = x / (1 + exp(-x))
    """
    is_amp = False
    from paddle.base.data_feeder import convert_dtype

    dtype = convert_dtype(x.dtype)
    if dtype in ["float16", "uint16"]:
        is_amp = True
        x = cast(x, "float32")

    sum_temp = exp(-x) + 1
    res = x / sum_temp
    return res if not is_amp else cast(res, dtype)


@register_decomp('pd_op.softmax')
def softmax(x, axis):
    """define composite rule of op softmax"""
    is_amp = False
    from paddle.base.data_feeder import convert_dtype

    # Softmax need fp32 compute since it has sum op in
    dtype = convert_dtype(x.dtype)
    if dtype in ["float16", "uint16"]:
        is_amp = True
        x = cast(x, "float32")
    if not x.shape:
        # do not return 1, to ensure gradients
        res = exp(x - x)
        if is_amp:
            res = cast(res, "float16")
        return res
    max_temp = max(x, axis, keepdim=True)
    max_temp.stop_gradient = True
    molecular = exp(x - max_temp)
    denominator = sum(molecular, axis=axis, keepdim=True)
    res = divide(molecular, denominator)
    if is_amp:
        res = cast(res, dtype)
    return res


@register_decomp('pd_op.full_like')
def full_like(x, fill_value, dtype, place=None):
    """define composite rule of op full_like."""
    """op name: full_like  op type name: fill_any_like."""
    """arg place is not used, add it here to keep same as python api."""
    fill_value = fill_value.get_defining_op().attrs()["value"]
    val = full(x.shape, fill_value, dtype)
    return val


@register_decomp('pd_op.stack')
def stack(x, axis):
    """
    define composite rule of op stack
    unsqueeze each dimension of the input (use reshape), and then concat
    """
    x_shape = x[0].shape
    if axis < 0:
        axis += len(x_shape) + 1
    out_shape = x_shape[:axis] + [1] + x_shape[axis:]
    out = concat([reshape(item, out_shape) for item in x], axis)
    return out


@register_decomp('pd_op.squeeze')
def squeeze(x, axis):
    """define composite rule of squeeze"""
    """
    canonicalize dim within range 0 to rank and
    determine new shape after squeeze op
    if axis not specified, remove all dims equal to 1
    otherwise, remove dims equal to 1 in axis
    axis can only be list, not int
    """
    axis = axis.get_defining_op().attrs()["value"]
    rank = len(x.shape)
    if rank == 0:
        return [assign(x), None]
    if len(axis) == 0:
        dims = set(range(rank))
    else:
        dims = {ax % rank for ax in axis}
    new_shape = []
    for d, s in enumerate(x.shape):
        if not (s == 1 and (d in dims)):
            new_shape.append(s)
    out = reshape(x, new_shape)
    return [out, None]


@register_decomp('pd_op.unsqueeze')
def unsqueeze(x, axis):
    """define composite rule of op unsqueeze"""
    """using reshape to implement unsqueeze op"""
    axis = axis.get_defining_op().attrs()["value"]
    x_shape = list(x.shape)
    axis_list = list(axis)
    for i in axis_list:
        if i < 0:
            i += len(x_shape) + 1
        x_shape = (
            x_shape[:i]
            + [
                1,
            ]
            + x_shape[i:]
        )
    out = reshape(x, x_shape)
    return [out, None]
