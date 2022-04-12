# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

from .primreg import REGISTER_JVP, REGISTER_TRANSPOSE
from .primreg import lookup_fn, lookup_jvp, lookup_transpose
from .primops import (neg, add, sub, mul, div, sqrt, tanh, reshape, broadcast,
                      transpose, split, concat, reduce, matmul, slice_select,
                      slice_assign, gather, scatter_add, fill_const)


def _jvp(op, *args):
    _jvprule = lookup_jvp(op.type)
    return _jvprule(op, *args)

def _transpose(op, dot_checker, *args):
    _transposerule = lookup_transpose(op.type)
    return _transposerule(op, dot_checker, *args)


def get_input_vars(op):
    return tuple(map(op.block.var, op.input_arg_names))


def get_output_vars(op):
    return tuple(map(op.block.var, op.output_arg_names))


def linear_jvp(op, *args, **kwargs):
    fn = lookup_fn(op.type)
    out_dot = fn(*args, **kwargs)
    return out_dot

## Register linearize rules

@REGISTER_JVP('add_p')
def add_jvp(op, x_dot, y_dot):
    return linear_jvp(op, x_dot, y_dot)


@REGISTER_JVP('sub_p')
def sub_jvp(op, x_dot, y_dot): 
    return linear_jvp(op, x_dot, y_dot)


@REGISTER_JVP('mul_p')
def mul_jvp(op, x_dot, y_dot):
    assert op.type == 'mul_p' 
    x, y = get_input_vars(op)
    t1, t2 = mul(x_dot, y), mul(x, y_dot)
    z_dot = add(t1, t2)
    return z_dot


@REGISTER_JVP('div_p')
def div_jvp(op, x_dot, y_dot):
    x, y = get_input_vars(op)
    t1, t2 = div(x_dot, y), div(mul(x, y_dot), mul(y, y))
    z_dot = sub(t1, t2)
    return z_dot


@REGISTER_JVP('sqrt_p')
def sqrt_jvp(op, x_dot):
    x, = get_input_vars(op)
    c2 = fill_const(value=2.0, shape=x.shape, dtype=x.dtype)
    y_dot = div(x_dot, mul(c2, sqrt(x)))
    return y_dot


@REGISTER_JVP('tanh_p')
def tanh_jvp(op, x_dot):
    y, = get_output_vars(op)
    c1 = fill_const(value=1.0, shape=y.shape, dtype=y.dtype)
    y_dot = mul(x_dot, sub(c1, mul(y, y)))
    return y_dot


@REGISTER_JVP('reshape_p')
def reshape_jvp(op, x_dot):
    shape = op.attr('shape')
    return linear_jvp(op, x_dot, shape=shape)


@REGISTER_JVP('broadcast_p')
def broadcast_jvp(op, x_dot):
    shape = op.attr('shape')
    return linear_jvp(op, x_dot, shape=shape)


@REGISTER_JVP('transpose_p')
def transpose_jvp(op, x_dot):
    axis = op.attr('axis')
    return linear_jvp(op, x_dot, axis=axis)


@REGISTER_JVP('split_p')
def split_jvp(op, x_dot):
    num_or_sections = op.attr('num_or_sections')
    axis = op.attr('axis')
    return linear_jvp(op, x_dot, num_or_sections=num_or_sections, axis=axis)


@REGISTER_JVP('concat_p')
def concat_jvp(op, xs_dot):
    axis = op.attr('axis')
    return linear_jvp(op, xs_dot, axis=axis)


@REGISTER_JVP('reduce_p')
def reduce_jvp(op, x_dot):
    axis = op.attr('axis')
    keepdim = op.attr('keepdim')
    return linear_jvp(op, x_dot, axis=axis, keepdim=keepdim)


@REGISTER_JVP('matmul_p')
def matmul_jvp(op, x_dot, y_dot):
    x, y = get_input_vars(op)
    t1 = matmul(x, y_dot)
    t2 = matmul(x_dot, y)
    z_dot = add(t1, t2)
    return z_dot


@REGISTER_JVP('slice_select_p')
def slice_select_jvp(op, x_dot):
    axis = op.attr('axis')
    starts = op.attr('starts')
    ends = op.attr('ends')
    strides = op.attr('strides')
    return linear_jvp(op, x_dot, axis=axis, starts=starts, ends=ends,
                      strides=strides)


@REGISTER_JVP('slice_assign_p')
def slice_assign_jvp(op, x_dot, y_dot):
    axis = op.attr('axis')
    starts = op.attr('starts')
    ends = op.attr('ends')
    strides = op.attr('strides')
    return linear_jvp(op, x_dot, y_dot, axis=axis, starts=starts, ends=ends,
                      strides=strides)


@REGISTER_JVP('gather_p')
def gather_jvp(op, x_dot):
    _, indextensor = get_input_vars(op)
    axis = op.attr('axis')
    return linear_jvp(op, x_dot, indextensor, axis=axis)


@REGISTER_JVP('scatter_add_p')
def scatter_add_jvp(op, x_dot, y_dot):
    _, _, indextensor = get_input_vars(op)
    axis = op.attr('axis')
    return linear_jvp(op, x_dot, y_dot, indextensor, axis=axis)


## Register transpose rules

@REGISTER_TRANSPOSE('add_p')
def add_transpose(op, check_dot, z_bar):
    x, y = get_input_vars(op)
    assert check_dot(x) and check_dot(y)
    return z_bar, z_bar


def sub_transpose(op, check_dot, z_bar):
    x, y = get_input_vars(op)
    assert check_dot(x) and check_dot(y)
    return z_bar, neg(z_bar)


@REGISTER_TRANSPOSE('mul_p')
def mul_transpose(op, check_dot, z_bar):
    x, y = get_input_vars(op)
    assert check_dot(x) ^ check_dot(y)
    if x.is_dot:
        return mul(z_bar, y), None
    else:
        return None, mul(x, z_bar)


@REGISTER_TRANSPOSE('div_p')
def div_transpose(op, check_dot, z_bar):
    x, y = get_input_vars(op)
    assert check_dot(x) and not check_dot(y)
    return div(z_bar, y), None


@REGISTER_TRANSPOSE('reshape_p')
def reshape_transpose(op, check_dot, y_bar):
    x, = get_input_vars(op)
    assert check_dot(x)
    return reshape(y_bar, shape=x.shape)


@REGISTER_TRANSPOSE('broadcast_p')
def broadcast_transpose(op, check_dot, y_bar):
    x, = get_input_vars(op)
    assert check_dot(x)
    bat = len(y_bar.shape) - len(x.shape)
    axis = list(range(bat))
    keepdim = [(bat + i) for i, s in enumerate(x.shape) if s == 1]
    axis += keepdim
    return reduce(y_bar, axis=axis, keepdim=keepdim)


@REGISTER_TRANSPOSE('transpose_p')
def transpose_transpose(op, check_dot, y_bar):
    x, = get_input_vars(op)
    assert check_dot(x)
    axis = op.attr('axis')
    reordered = sorted((k, i) for i, k in enumerate(axis))
    axis = [i for k, i in reordered]
    return transpose(y_bar, axis=axis)


@REGISTER_TRANSPOSE('split_p')
def split_transpose(op, check_dot,  ys_bar):
    x, = get_input_vars(op)
    assert check_dot(x)
    return concat(ys_bar, axis=op.attr('axis'))


@REGISTER_TRANSPOSE('concat_p')
def concat_transpose(op, check_dot, y_bar):
    xs = get_input_vars(op)
    for x in xs:
        assert check_dot(x)
    axis = op.attr('axis')
    sections = [x.shape[axis] for x in xs]
    return split(y_bar, num_or_sections=sections, axis=axis)


@REGISTER_TRANSPOSE('reduce_p')
def reduce_transpose(op, check_dot, y_bar):
    x, = get_input_vars(op)
    assert check_dot(x)
    shape = x.shape
    for i in op.attr('axis'):
        shape[i] = 1
    t = reshape(y_bar, shape=shape)
    return broadcast(t, shape=x.shape)


@REGISTER_TRANSPOSE('matmul_p')
def matmul_transpose(op, check_dot, z_bar):
    x, y = get_input_vars(op)
    assert check_dot(x) ^ check_dot(y)
    if x.is_dot:
        return matmul(z_bar, transpose(y)), None
    else:
        return None, matmul(transpose(x), z_bar)


@REGISTER_TRANSPOSE('slice_select_p')
def slice_select_transpose(op, check_dot, y_bar):
    x, = get_input_vars(op)
    assert check_dot(x)
    zeros = fill_const(value=0.0, shape=x.shape, dtype=x.dtype)
    axis = op.attr('axis')
    starts = op.attr('starts')
    ends = op.attr('ends')
    strides = op.attr('strides')
    return slice_assign(zeros, y_bar, axis=axis, starts=starts, ends=ends,
                        strides=strides)


@REGISTER_TRANSPOSE('slice_assign_p')
def slice_assign_transpose(op, check_dot, z_bar):
    x, y = get_input_vars(op)
    assert check_dot(x) and check_dot(y)
    zeros = fill_const(value=0.0, shape=y.shape, dtype=y.dtype)
    axis = op.attr('axis')
    starts = op.attr('starts')
    ends = op.attr('ends')
    strides = op.attr('strides')
    x_bar = slice_assign(z_bar, zeros, axis=axis, starts=starts, ends=ends,
                        strides=strides)
    y_bar = slice_select(z_bar, axis=axis, starts=starts, ends=ends,
                        strides=strides)
    return x_bar, y_bar


@REGISTER_TRANSPOSE('gather_p')
def gather_transpose(op, check_dot, y_bar):
    x, indextensor = get_input_vars(op)
    assert check_dot(x)
    axis = op.attr('axis')
    return scatter_add(y_bar, indextensor, axis=axis)


@REGISTER_TRANSPOSE('scatter_add_p')
def scatter_add_transpose(op, check_dot, y_bar):
    x, indextensor = get_input_vars(op)
    assert check_dot(x)
    axis = op.attr('axis')
    return gather(y_bar, indextensor, axis=axis)
