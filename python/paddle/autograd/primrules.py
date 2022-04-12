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

import paddle
from .primreg import REGISTER_ORIG2PRIM, REGISTER_PRIM2ORIG, REGISTER_JVP, REGISTER_TRANSPOSE
from .primreg import lookup_fn, lookup_orig2prim, lookup_prim2orig, lookup_jvp, lookup_transpose
from .primops import (neg, add, sub, mul, div, sqrt, tanh, reshape, broadcast,
                      transpose, split, concat, reduce, matmul, slice_select,
                      slice_assign, gather, scatter_add, fill_const,
                      strided_slice_grad)


def _orig2prim(op):
    _lowerrule = lookup_orig2prim(op.type)
    return _lowerrule(op)


def _prim2orig(op):
    _lowerrule = lookup_prim2orig(op.type)
    return _lowerrule(op)


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


## Register orig2prim lower rules
"""
These original ops are fully supported:

elementwise_add
tanh
reshape2
fill_zeros_like
sum
index_select
elementwise_sub
scale
assign

These original ops are partially supported:

matmul_v2
concat
slice
p_norm
"""


@REGISTER_ORIG2PRIM('matmul_v2')
def matmul_v2_orig2prim(op):
    def trans(shape):
        last_shape = shape[-1]
        shape[-1] = shape[-2]
        shape[-2] = last_shape
        return shape

    x, y = get_input_vars(op)
    assert len(x.shape) < 4 and len(
        y.shape) < 4, 'Do not support multi batchsize dimensions currently.'
    if op.attr('trans_x'):
        x = transpose(x, shape=trans(x.shape))
    if op.attr('trans_y'):
        y = transpose(y, shape=trans(y.shape))
    return matmul(x, y)


@REGISTER_ORIG2PRIM('elementwise_add')
def elementwise_add_orig2prim(op):
    x, y = get_input_vars(op)
    if x.shape != y.shape:
        y = broadcast(y, shape=x.shape)
    if op.attr('Scale_x'):
        tmp = fill_constant(
            shape=x.shape, dtype=x.dtype, value=op.attr('Scale_x'))
        x = mul(x, tmp)
    if op.attr('Scale_y'):
        tmp = fill_constant(
            shape=y.shape, dtype=y.dtype, value=op.attr('Scale_y'))
        y = mul(y, tmp)
    z = add(x, y)
    if op.attr('Scale_out'):
        tmp = fill_constant(
            shape=z.shape, dtype=z.dtype, value=op.attr('Scale_out'))
        z = mul(z, tmp)
    return z


@REGISTER_ORIG2PRIM('tanh')
def tanh_orig2prim(op):
    x = get_input_vars(op)
    return tanh(x)


## NOTE(lml): The second output of reshape2 Xshape, can't be described by prim ops, use paddle.shape() interface instead.
@REGISTER_ORIG2PRIM('reshape2')
def reshape2_orig2prim(op):
    _, _, x = get_input_vars(op)
    y, _ = get_output_vars(op)
    return reshape(x, shape=y.shape), paddle.shape(x)


@REGISTER_ORIG2PRIM('concat')
def concat_orig2prim(op):
    axis_t, *xs = get_input_vars(op)
    assert axis_t is None, 'Can not lower concat into prim ops with axistensor.'
    return concat(xs, axis=op.attr('axis'))


@REGISTER_ORIG2PRIM('slice')
def slice_orig2prim(op):
    ends_t, ends_tl, x, starts_t, starts_tl = get_input_vars(op)

    assert starts_t is None, 'Can not lower concat into prim ops with startstensor.'
    assert ends_t is None, 'Can not lower concat into prim ops with endstensor.'
    assert starts_tl is None, 'Can not lower concat into prim ops with startstensorlist.'
    assert ends_tl is None, 'Can not lower concat into prim ops with endstensorlist.'
    starts = op.attr('starts')
    ends = op.attr('ends')
    strides = [1 for _ in starts]
    axis = op.attr('axes')
    y = slice_select(x, starts=starts, ends=ends, strides=strides, axis=axis)
    if op.attr('decrease_axis') is not None:
        y = reshape(y, shape=get_output_vars(op).shape)
    return y


@REGISTER_ORIG2PRIM('fill_zeros_like')
def fill_zeros_like_orig2prim(op):
    x, = get_input_vars(op)
    return fill_constant(x, shape=x.shape, value=0.0)


@REGISTER_ORIG2PRIM('sum')
def sum_orig2prim(op):
    x0, *x_other = get_input_vars(op)
    for x in x_other:
        x0 = add(x0, x)
    return x0


@REGISTER_ORIG2PRIM('p_norm')
def p_norm_orig2prim(op):
    def num_el(shape):
        n = 1
        for s in shape:
            n = n * s
        return n

    assert op.attr(
        'porder') - 2.0 < 1e-5, 'Only support lower l2 norm currently'
    assert op.attr(
        'asvector'), 'Only support lower pnorm when asvector=True currently'
    x, = get_input_vars(op)
    if len(x.shape) > 1:
        x = reshape(x, shape=[num_el(x.shape)])
    return sqrt(reduce(mul(x, x), axis=0))


@REGISTER_ORIG2PRIM('index_select')
def index_select_orig2prim(op):
    index_t, x = get_input_vars(op)
    return gather(x, indextensor=index_t, axis=op.attr('dim'))


@REGISTER_ORIG2PRIM('elementwise_sub')
def elementwise_sub_orig2prim(op):
    x, y = get_input_vars(op)
    if x.shape != y.shape:
        y = broadcast(y, shape=x.shape)
    if op.attr('Scale_x'):
        tmp = fill_constant(
            shape=x.shape, dtype=x.dtype, value=op.attr('Scale_x'))
        x = mul(x, tmp)
    if op.attr('Scale_y'):
        tmp = fill_constant(
            shape=y.shape, dtype=y.dtype, value=op.attr('Scale_y'))
        y = mul(y, tmp)
    z = sub(x, y)
    if op.attr('Scale_out'):
        tmp = fill_constant(
            shape=z.shape, dtype=z.dtype, value=op.attr('Scale_out'))
        z = mul(z, tmp)
    return z


@REGISTER_ORIG2PRIM('scale')
def scale_orig2prim(op):
    x = get_input_vars(op)
    scale_t = fill_constant(
        shape=x.shape, dtype=x.dtype, value=op.attr('scale'))
    bias_t = fill_constant(shape=x.shape, dtype=x.dtype, value=op.attr('bias'))
    if op.attr('bias_after_scale'):
        return add(mul(x, scale_t), bias_t)
    else:
        return mul(add(x, bias_t), scale_t)


@REGISTER_ORIG2PRIM('assign')
def assign_orig2prim(op):
    x = get_input_vars(op)
    zero_t = fill_constant(shape=x.shape, dtype=x.dtype, value=0.0)
    return add(x, zero_t)


## Register orig2prim lower rules


@REGISTER_PRIM2ORIG('add_p')
def add_prim2orig(op):
    x, y = get_input_vars(op)
    return paddle.add(x, y)


@REGISTER_PRIM2ORIG('sub_p')
def sub_prim2orig(op):
    x, y = get_input_vars(op)
    return paddle.sub(x, y)


@REGISTER_PRIM2ORIG('mul_p')
def mul_prim2orig(op):
    x, y = get_input_vars(op)
    return paddle.mul(x, y)


@REGISTER_PRIM2ORIG('div_p')
def div_prim2orig(op):
    x, y = get_input_vars(op)
    return paddle.div(x, y)


@REGISTER_PRIM2ORIG('sqrt_p')
def sqrt_prim2orig(op):
    x, = get_input_vars(op)
    return paddle.sqrt(x)


@REGISTER_PRIM2ORIG('tanh_p')
def tanh_prim2orig(op):
    x, = get_input_vars(op)
    return paddle.tanh(x)


@REGISTER_PRIM2ORIG('reshape_p')
def reshape_prim2orig(op):
    x, = get_input_vars(op)
    y, _ = paddle.reshape(x, shape=op.attr('shape'))
    return y


@REGISTER_PRIM2ORIG('broadcast_p')
def broadcast_prim2orig(op):
    x, = get_input_vars(op)
    return paddle.broadcast_to(x, shape=op.attr('shape'))


@REGISTER_PRIM2ORIG('transpose_p')
def transpose_prim2orig(op):
    x, = get_input_vars(op)
    return paddle.transpose(x, perm=op.attr('axis'))


@REGISTER_PRIM2ORIG('split_p')
def split_prim2orig(op):
    x, = get_input_vars(op)
    num_or_sections = op.attr('num_or_sections')
    if len(num_or_sections) == 1:
        num_or_sections = num_or_sections[0]
    return paddle.split(
        x, num_or_sections=num_or_sections, axis=op.attr('axis'))


@REGISTER_PRIM2ORIG('concat_p')
def concat_prim2orig(op):
    xs = get_input_vars(op)
    return paddle.concat(xs, axis=op.attr('axis'))


@REGISTER_PRIM2ORIG('reduce_p')
def reduce_prim2orig(op):
    xs = get_input_vars(op)
    return paddle.sum(xs, axis=op.attr('axis'), keepdim=op.attr('keepdim'))


@REGISTER_PRIM2ORIG('matmul_p')
def matmul_prim2orig(op):
    x, y = get_input_vars(op)
    return paddle.matmul(x, y)


@REGISTER_PRIM2ORIG('slice_select_p')
def slice_select_prim2orig(op):
    x, = get_input_vars(op)
    return paddle.strided_slice(
        x,
        axes=op.attr('axis'),
        starts=op.attr('starts'),
        ends=op.attr('ends'),
        strides=op.attr('strides'))


def strided_slice_grad(y_grad, x, axis, starts, ends, strides, x_grad=None):
    attrs = {'axes': axis, 'starts': starts, 'ends': ends, 'strides': strides}
    helper = LayerHelper('strided_slice_grad', **locals())
    if out is None:
        out = helper.create_variable_for_type_inference(dtype=x.dtype)
    return x_grad


@REGISTER_PRIM2ORIG('slice_assign_p')
def slice_assign_prim2orig(op):
    x, y = get_input_vars(op)
    return paddle.add(x,
                      strided_slice_grad(
                          y,
                          x,
                          axis=op.attr('axis'),
                          starts=op.attr('starts'),
                          ends=op.attr('ends'),
                          strides=op.attr(strides)))


@REGISTER_PRIM2ORIG('gather_p')
def gather_prim2orig(op):
    index_t, x = get_input_vars(op)
    return paddle.gather(x, index_t, axis=op.attr('axis'))


@REGISTER_PRIM2ORIG('scatter_add_p')
def scatter_add_prim2orig(op):
    index_t, x, y = get_input_vars(op)
    return paddle.put_along_axis(
        x, index_t, y, axis=op.attr('axis'), reduce='add')


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
    return linear_jvp(
        op, x_dot, axis=axis, starts=starts, ends=ends, strides=strides)


@REGISTER_JVP('slice_assign_p')
def slice_assign_jvp(op, x_dot, y_dot):
    axis = op.attr('axis')
    starts = op.attr('starts')
    ends = op.attr('ends')
    strides = op.attr('strides')
    return linear_jvp(
        op, x_dot, y_dot, axis=axis, starts=starts, ends=ends, strides=strides)


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
def split_transpose(op, check_dot, ys_bar):
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
    return slice_assign(
        zeros, y_bar, axis=axis, starts=starts, ends=ends, strides=strides)


@REGISTER_TRANSPOSE('slice_assign_p')
def slice_assign_transpose(op, check_dot, z_bar):
    x, y = get_input_vars(op)
    assert check_dot(x) and check_dot(y)
    zeros = fill_const(value=0.0, shape=y.shape, dtype=y.dtype)
    axis = op.attr('axis')
    starts = op.attr('starts')
    ends = op.attr('ends')
    strides = op.attr('strides')
    x_bar = slice_assign(
        z_bar, zeros, axis=axis, starts=starts, ends=ends, strides=strides)
    y_bar = slice_select(
        z_bar, axis=axis, starts=starts, ends=ends, strides=strides)
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
