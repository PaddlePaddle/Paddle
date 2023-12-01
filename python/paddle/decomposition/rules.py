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


@register_decomp('pd_op.add_n')
def add_n(x):
    ans = x[0]
    for xi in x[1:]:
        ans = xi + ans
    return ans


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
