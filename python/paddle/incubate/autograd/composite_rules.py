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

# This file contains composite rules of nonbasic operations. There are some notes:
# 1. When define composite rule of some op, you can only use primitive ops defined in primitives.py.
# 2. The name and args of target op must be corresponding with standard description of op in
#    ops.yaml or dygraph_ops.yaml.

import functools
import operator

from paddle.base import core

from .primitives import *  # noqa: F403
from .primreg import REGISTER_COMPOSITE, lookup_composite


def _composite(op, *args):
    _lowerrule = lookup_composite(op.type)
    return _lowerrule(op, *args)


@REGISTER_COMPOSITE('softmax')
def softmax_composite(x, axis):
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


@REGISTER_COMPOSITE('batch_norm')
def composite_batchnorm(
    x,
    run_mean,
    run_var,
    scale,
    bias,
    is_test,
    momentum,
    epsilon,
    data_layout,
    use_global_stats,
    trainable_statistics,
):
    """
    define composite rule of op batch_norm
    As the same with op kernel, the position of saved variance indeed return inverse std.
    """

    is_amp = False
    from paddle.base.data_feeder import convert_dtype

    dtype = convert_dtype(x.dtype)
    if dtype in ["float16", "uint16"]:
        is_amp = True
        x = cast(x, "float32")
        scale = cast(scale, "float32") if scale else scale
        bias = cast(bias, "float32") if bias else bias

    feature_axis = (
        1 if data_layout in ('NC', 'NCL', 'NCHW', 'NCHWD') else len(x.shape) - 1
    )

    use_run_stat = (is_test and (not trainable_statistics)) or use_global_stats
    reduce_axes = tuple(i for i in range(len(x.shape)) if i != feature_axis)
    stats_shape = tuple(
        1 if i in reduce_axes else s for i, s in enumerate(x.shape)
    )

    half = full([1], -0.5, x.dtype)

    if not use_run_stat:
        batch_mean = mean(x, reduce_axes)
        temp = mean(x * x, reduce_axes)
        batch_var = temp - batch_mean * batch_mean
        inv_std = pow((batch_var + epsilon), half)
        if data_layout == "NHWC":
            x_hat = (x - batch_mean) * inv_std
        else:
            x_hat = (x - reshape(batch_mean, stats_shape)) * reshape(
                inv_std, stats_shape
            )

        run_mean = momentum * run_mean + (1 - momentum) * batch_mean
        run_var = momentum * run_var + (1 - momentum) * batch_var
    else:
        batch_mean = zeros(run_mean.shape, run_mean.dtype)
        batch_var = zeros(run_var.shape, run_var.dtype)
        inv_std = pow((batch_var + epsilon), half)
        if data_layout == "NHWC":
            x_hat = (x - run_mean) * pow((run_var + epsilon), half)
        else:
            x_hat = (x - reshape(run_mean, stats_shape)) * pow(
                (reshape(run_var, stats_shape) + epsilon), half
            )
    if data_layout == "NHWC":
        y = scale * x_hat + bias
    else:
        y = reshape(scale, stats_shape) * x_hat + reshape(bias, stats_shape)
    if is_amp:
        y = cast(y, dtype)

    # add op assign to detach tensor in void unsafe change outside the rule.
    batch_mean_ = assign(batch_mean)
    inv_std_ = assign(inv_std)
    run_mean_ = assign(run_mean)
    run_var_ = assign(run_var)

    # reserve_space is not needed in composite rule, but still return None to keep same as phi op definition.
    reserve_space = None
    if not use_run_stat:
        return y, run_mean_, run_var_, batch_mean_, inv_std_, reserve_space
    else:
        return y, run_mean_, run_var_, None, None, reserve_space


@REGISTER_COMPOSITE('layer_norm')
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

    # keep the mean and variance shape as input x before begin_norm_axis
    mean_ = reshape(mean_, x.shape[:begin_norm_axis])
    variance = reshape(variance, x.shape[:begin_norm_axis])
    if is_amp:
        out = cast(out, dtype)
    return out, mean_, variance


@REGISTER_COMPOSITE('instance_norm')
def instancenorm_composite(x, scale, bias, epsilon):
    """
    define composite rule of op instance_norm
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

    n, c, h, w = x.shape
    axis = tuple(range(2, len(x.shape)))
    mean_ = mean(x, axis=axis, keepdim=True)
    difference = x - mean_
    var_tmp1 = difference * difference
    variance = mean(var_tmp1, axis=axis, keepdim=True)
    var_tmp3 = variance + epsilon
    sqrt_var = pow(var_tmp3, full([1], 0.5, dtype=var_tmp3.dtype))
    out = difference / sqrt_var

    if scale is not None:
        scale_tile = reshape(scale, [1, c, 1, 1])
        out = out * scale_tile
    if bias is not None:
        bias_tile = reshape(bias, [1, c, 1, 1])
        out = out + bias_tile

    mean_ = reshape(mean_, [-1])
    saved_variance = 1 / sqrt_var
    saved_variance = reshape(saved_variance, [-1])

    if is_amp:
        out = cast(out, dtype)

    return out, mean_, saved_variance


@REGISTER_COMPOSITE('gelu')
def gelu_composite(x, approximate):
    """define composite rule of op gelu"""
    M_SQRT1_2 = (
        0.70710678118654752440  # /* 1/sqrt(2) */ copy from gelu-kernel.cc
    )
    M_2_SQRTPI = 1.12837916709551257390  # /* 2/sqrt(pi) */
    full_shape = x.shape if len(x.shape) == 0 else [1]
    one = ones(full_shape, x.dtype)
    half = full(full_shape, 0.5, x.dtype)
    if approximate:
        # gelu(x) = 0.5 * x * (1 + tanh(sqrt(2 / \pi) * (x + 0.044715 * x^{3})))
        kAlpha = full(full_shape, M_2_SQRTPI * M_SQRT1_2, x.dtype)
        GELU_CONSTANT = full(full_shape, 0.044715, x.dtype)
        tanh_out = tanh(kAlpha * (x + GELU_CONSTANT * x * x * x))
        out = x * half * (one + tanh_out)
        return out

    else:
        # gelu(x) = 0.5 * x *  (1 + erf(x / sqrt(2)))
        cdf = half * (one + erf(x * full(x.shape, M_SQRT1_2, x.dtype)))
        out = x * cdf
        return out


@REGISTER_COMPOSITE('reduce_mean')
def mean_composite(x, axis, keepdim):
    """define composite rule of op mean"""
    is_amp = False
    from paddle.base.data_feeder import convert_dtype

    dtype = convert_dtype(x.dtype)
    if dtype in ["float16", "uint16"]:
        is_amp = True
        x = cast(x, "float32")

    if axis in (None, []):
        axis = tuple(range(0, len(x.shape)))
    axes = (axis,) if isinstance(axis, int) else axis
    sum_x = sum(x, axis=axes, keepdim=keepdim)
    ele_nums_list = [x.shape[axis] for axis in axes]
    if ele_nums_list == []:
        value_to_fill = 1
    else:
        value_to_fill = functools.reduce(operator.mul, ele_nums_list)
    norm = fill_constant(
        shape=[],
        value=value_to_fill,
        dtype=sum_x.dtype,
    )
    res = divide(sum_x, norm)
    if is_amp:
        res = cast(res, dtype)
    return res


@REGISTER_COMPOSITE('expand_v2')
def expand_v2_composite(x, shape):
    """
    define composite rule of op expand_v2, expand_v2->expand
    repeat_times = shape / x.shape
    out = tile(x, repeat_times = repeat_times)
    """
    shape_in = x.shape
    dim_out = len(shape)
    dim_in = len(shape_in)
    assert dim_in <= dim_out and dim_out >= 0
    repeat_times = []
    for i in range(dim_out):
        offset = dim_out - i
        dim = dim_in - offset
        size_in = shape_in[dim] if dim >= 0 else 1
        size_out = shape[i]
        if size_out == -1:
            assert dim >= 0
            repeat = 1
        else:
            assert size_out % size_in == 0
            repeat = int(size_out / size_in)
        repeat_times.append(repeat)
    if dim_in < dim_out:
        shape_in_expand = []
        for i in range(dim_out - dim_in):
            shape_in_expand.append(1)
        shape_in_expand.extend(shape_in)
        x_reshape = reshape(x, shape_in_expand)
        return tile(x_reshape, repeat_times=repeat_times)
    return tile(x, repeat_times=repeat_times)


@REGISTER_COMPOSITE('expand_as_v2')
def expand_as_v2_composite(x, y, target_shape):
    """
    define composite rule of op expand_as_v2, expand_as_v2->expand_as
    repeat_times = target_shape / x.shape
    out = tile(x, repeat_times = repeat_times)
    """
    shape_in = x.shape
    if y is not None:
        target_shape = y.shape
    assert target_shape is not None
    dim_out = len(target_shape)
    dim_in = len(shape_in)
    assert dim_in <= dim_out and dim_out >= 0
    repeat_times = []
    for i in range(dim_out):
        offset = dim_out - i
        dim = dim_in - offset
        size_in = shape_in[dim] if dim >= 0 else 1
        size_out = target_shape[i]
        if size_out == -1:
            assert dim >= 0
            repeat = 1
        else:
            assert size_out % size_in == 0
            repeat = int(size_out / size_in)
        repeat_times.append(repeat)
    if dim_in < dim_out:
        shape_in_expand = []
        for i in range(dim_out - dim_in):
            shape_in_expand.append(1)
        shape_in_expand.extend(shape_in)
        x_reshape = reshape(x, shape_in_expand)
        return tile(x_reshape, repeat_times=repeat_times)
    return tile(x, repeat_times=repeat_times)


@REGISTER_COMPOSITE('stack')
def stack_composite(x, axis):
    """
    define composite rule of op stack
    unsqueeze each dimension of the input (use reshape), and then concat
    """
    x_shape = x[0].shape
    if axis < 0:
        axis += len(x_shape) + 1
    out_shape = x_shape[:axis] + (1,) + x_shape[axis:]
    out = concat([reshape(item, out_shape) for item in x], axis)
    return out


@REGISTER_COMPOSITE('flatten_contiguous_range')
def flatten_contiguous_range_composite(x, start_axis, stop_axis):
    """
    define composite rule of op flatten, flatten_contiguous_range -> flatten.

    xshape is the dim with 0 added to the front of x, keep the shape information of x to calculate the grad.
    CINN doesn't need xshape for backward pass, return none instead of xshape.
    shape_out is the parameter of reshape, get from start_axis and stop_axis.
    out = reshape(x, shape=shape_out), xshape
    """
    shape_in = x.shape
    start_dim = start_axis if len(shape_in) != 0 else 0
    end_dim = stop_axis if len(shape_in) != 0 else 0
    assert start_dim <= end_dim
    if len(shape_in) == 0:
        return reshape(x, shape=[1]), None
    if start_dim == end_dim:
        return reshape(x, shape=shape_in), None
    slice_numel = 1
    for i in range(start_dim, end_dim + 1):
        slice_numel *= shape_in[i]
    shape_out = []
    for i in range(start_dim):
        shape_out.append(shape_in[i])
    shape_out.append(slice_numel)
    for i in range(end_dim + 1, len(shape_in)):
        shape_out.append(shape_in[i])
    return reshape(x, shape=shape_out), None


@REGISTER_COMPOSITE('dropout')
def dropout_composite(x, seed_tensor, p, is_test, mode, seed, fix_seed):
    """define composite rule of op dropout.
    upscale_in_train:
        train: out = input * mask / ( 1.0 - p )
        inference: out = input
    downscale_in_infer
        train: out = input * mask
        inference: out = input * (1.0 - p)
    """
    fix_seed = True if fix_seed is None else fix_seed
    seed = seed if fix_seed else 0
    upscale_in_train = mode == "upscale_in_train"

    mask = bernoulli(shape=x.shape, dtype=x.dtype, p=p, seed=seed)

    if upscale_in_train:
        if not is_test:
            # Process p=1.0 for avoid divide zero error (x*mask/(1.0-p))
            if p == 1.0:
                return 0.0 * x, zeros(x.shape, core.VarDesc.VarType.UINT8)
            else:
                return x * mask / (1.0 - p), cast(
                    mask, core.VarDesc.VarType.UINT8
                )
        else:
            return assign(x), cast(mask, core.VarDesc.VarType.UINT8)
    else:
        if not is_test:
            return x * mask, cast(mask, core.VarDesc.VarType.UINT8)
        else:
            return x * (1.0 - p), cast(mask, core.VarDesc.VarType.UINT8)


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


@REGISTER_COMPOSITE('hard_swish')
def hard_swish_composite(x):
    """define composite rule of op hard_swish.
    offset=3, threshold=6, scale=6
    out = minimum(
        maximum(x + offset, 0), threshold
    ) * x / scale
    """
    threshold = 6.0
    scale = 6.0
    offset = 3.0
    full_shape = x.shape if len(x.shape) == 0 else [1]
    res = (
        minimum(
            maximum(
                x + full(full_shape, offset, dtype=x.dtype),
                full(full_shape, 0.0, dtype=x.dtype),
            ),
            full(full_shape, threshold, dtype=x.dtype),
        )
        * x
        / full(full_shape, scale, dtype=x.dtype)
    )
    return res


@REGISTER_COMPOSITE('index_select')
def index_select_composite(x, index, axis):
    """define composite rule of op index_select."""
    if axis < 0:
        axis = len(x.shape) + axis
    res = gather(x, index, axis=axis)
    return res


@REGISTER_COMPOSITE('sigmoid')
def sigmoid_composite(x):
    """
    define composite rule of op sigmoid
    res = 1 / (1 + exp(-x))
    """
    is_amp = False
    from paddle.base.data_feeder import convert_dtype

    dtype = convert_dtype(x.dtype)
    if dtype in ["float16", "uint16"]:
        is_amp = True
        x = cast(x, "float32")

    sum_temp = 1 + exp(-x)
    res = 1 / sum_temp
    return res if not is_amp else cast(res, dtype)


@REGISTER_COMPOSITE('silu')
def silu_composite(x):
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

    sum_temp = 1 + exp(-x)
    res = x / sum_temp
    return res if not is_amp else cast(res, dtype)


@REGISTER_COMPOSITE('meshgrid')
def meshgrid_composite(inputs):
    """
    define composite rule of op meshgrid
    If the input has N tensors of size S_0, ... S_n-1, then the output will also have N tensors, where
    each tensor is of shape (S_0, ..., S_n-1).
    E.g. a1 is Tensor [1,2,3]
         b1 is Tensor [4,5]
         r1, r2 = paddle.meshgrid([a1, b1])
         r1 is Tensor [[1,1], [2,2], [3,3]]
         r2 is Tensor [[4,5], [4,5], [4,5]]
    """
    size = len(inputs)
    shape = [1] * size
    for i in range(size):
        dim = inputs[i].dim()
        assert dim == 0 or dim == 1
        if dim == 1:
            shape[i] = inputs[i].shape[0]
    outputs = []
    for i in range(size):
        view_shape = [1] * size
        view_shape[i] = shape[i]
        outputs.append(inputs[i].reshape(view_shape).broadcast_to(shape))
    return outputs


@REGISTER_COMPOSITE('fill_any_like')
def fill_any_like(x, fill_value, dtype, place=None):
    """define composite rule of op full_like."""
    """op name: full_like  op type name: fill_any_like."""
    """arg place is not used, add it here to keep same as python api."""
    val = full(x.shape, fill_value, dtype)
    return val


@REGISTER_COMPOSITE('squeeze2')
def squeeze2_composite(x, axis):
    """define composite rule of squeeze"""
    """
    canonicalize dim within range 0 to rank and
    determine new shape after squeeze op
    if axis not specified, remove all dims equal to 1
    otherwise, remove dims equal to 1 in axis
    axis can only be list, not int
    """
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


@REGISTER_COMPOSITE('sqrt')
def sqrt_composite(x):
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
    res = pow(x, y)
    return res if not is_amp else cast(res, dtype)


@REGISTER_COMPOSITE('pow')
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


@REGISTER_COMPOSITE('relu')
def relu_composite(x):
    """define composite rule of op relu."""
    # relu(x) = max(x, 0)
    if len(x.shape) == 0:
        return maximum(x, full(x.shape, 0.0, x.dtype))
    else:
        return maximum(x, full([1], 0.0, x.dtype))


@REGISTER_COMPOSITE('unsqueeze2')
def unsqueeze_composite(x, axis):
    """define composite rule of op unsqueeze"""
    """using reshape to implement unsqueeze op"""
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


@REGISTER_COMPOSITE('group_norm')
def group_norm_composite(x, scale, bias, epsilon, groups, data_layout):
    """
    define composite rule of op group_norm.
    x = ((x - mean) / sqrt(var + epsilon)) * scale + bias
    mean and var are computed from groups
    """
    # original GroupNorm op cannot support NHWC format
    assert data_layout == 'NCHW'
    N, C, H, W = x.shape

    is_amp = False
    from paddle.base.data_feeder import convert_dtype

    dtype = convert_dtype(x.dtype)
    # when inputs are float16 or bfloat16, convert to float32 in computing
    if dtype in ["float16", "uint16"]:
        is_amp = True
        x = cast(x, "float32")
        scale = cast(scale, "float32")
        bias = cast(bias, "float32")

    x = reshape(x, (N * groups, -1))
    mean_ = mean(x, axis=1, keepdim=True)
    var_ = mean(x * x, axis=1, keepdim=True) - mean_ * mean_
    var_ = maximum(var_, zeros_like(var_))
    var_inv = 1 / sqrt(var_ + epsilon)
    out = (x - mean_) * var_inv
    out = reshape(out, (N, C, H, W))
    if scale is not None:
        out = out * reshape(scale, (-1, 1, 1))
    if bias is not None:
        out = out + reshape(bias, (-1, 1, 1))
    ret_mean_ = reshape(mean_, (N, groups))
    ret_var_ = reshape(var_, (N, groups))
    # return output in float16 or bfloat16, mean and var in float32
    if is_amp:
        out = cast(out, dtype)
    return out, ret_mean_, ret_var_


@REGISTER_COMPOSITE('sum')
def sum_composite(x):
    ans = 0
    for xi in x:
        ans += xi
    return ans


@REGISTER_COMPOSITE('leaky_relu')
def leaky_relu_composite(x, negative_slope):
    """define composite rule of op leaky_relu."""
    if negative_slope < 1.0:
        return maximum(x, negative_slope * x)
    else:
        return minimum(x, negative_slope * x)
