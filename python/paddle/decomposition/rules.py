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


from .primitives import *  # noqa: F403
from .register import register_decomp


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
