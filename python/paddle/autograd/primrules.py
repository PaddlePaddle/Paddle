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
from .primreg import (lookup_fn, lookup_orig2prim, lookup_prim2orig, lookup_jvp,
                      lookup_transpose, op_position_inputs, op_position_output)
from .primops import (neg, add, sub, mul, div, sqrt, tanh, reshape, broadcast,
                      transpose, split, concat, reduce, matmul, slice_select,
                      slice_assign, gather, scatter_add, fill_const, set_value)

INT_DTYPE_2_STRING = {
    int(paddle.bool): 'bool',
    int(paddle.float16): 'float16',
    int(paddle.bfloat16): 'uint16',
    int(paddle.float32): 'float32',
    int(paddle.float64): 'float64',
    int(paddle.int8): 'int8',
    int(paddle.int16): 'int16',
    int(paddle.int32): 'int32',
    int(paddle.int64): 'int64',
    int(paddle.uint8): 'uint8',
    int(paddle.complex64): 'complex64',
    int(paddle.complex128): 'complex128',
}


def _orig2prim(op, *args):
    _lowerrule = lookup_orig2prim(op.type)
    return _lowerrule(op, *args)


def _prim2orig(op, *args):
    _lowerrule = lookup_prim2orig(op.type)
    return _lowerrule(op, *args)


def _jvp(op, *args):
    _jvprule = lookup_jvp(op.type)
    return _jvprule(op, *args)


def _transpose(op, dot_checker, *args):
    _transposerule = lookup_transpose(op.type)
    return _transposerule(op, dot_checker, *args)


def get_var_block(block, names):
    assert isinstance(names, list)
    if len(names) == 0:
        return None
    elif len(names) == 1:
        return block.var(names[0])
    else:
        return [block.var(name) for name in names]


def get_input_vars(op):
    return tuple(map(op.block.var, op.input_arg_names))


def get_output_vars(op):
    return tuple(map(op.block.var, op.output_arg_names))


def get_input_var_list(op):
    if op.input_names is None:
        return []
    else:
        return [get_var_block(op.block, op.input(n)) for n in op.input_names]


def get_output_var_list(op):
    if op.output_names is None:
        return []
    else:
        return [get_var_block(op.block, op.output(n)) for n in op.output_names]


def linear_jvp(op, *args, **kwargs):
    fn = lookup_fn(op.type)
    out_dot = fn(*args, **kwargs)
    return out_dot


## Register orig2prim lower rules
"""
These original ops are fully supported:

elementwise_add
tanh
fill_zeros_like
sum
index_select
elementwise_sub
scale
assign

These original ops are partially supported:

matmul_v2
reshape2
concat
slice
p_norm
"""


@REGISTER_ORIG2PRIM('matmul_v2')
def matmul_v2_orig2prim(op, x, y):
    def trans(shape):
        ret = [i for i in range(len(shape))]
        ret[-1], ret[-2] = ret[-2], ret[-1]
        return ret

    assert len(x.shape) < 4 and len(
        y.shape) < 4, 'Do not support multi batchsize dimensions currently.'

    if len(x.shape) == 1:
        x = broadcast(x, shape=[1, x.shape[0]])
    if len(y.shape) == 1:
        y = broadcast(y, shape=[y.shape[0], 1])
    if op.attr('trans_x'):
        x = transpose(x, axis=trans(x.shape))
    if op.attr('trans_y'):
        y = transpose(y, axis=trans(y.shape))
    return matmul(x, y)


@REGISTER_ORIG2PRIM('elementwise_add')
def elementwise_add_orig2prim(op, x, y):
    if x.shape != y.shape:
        y = broadcast(y, shape=x.shape)
    if op.attr('Scale_x') - 1.0 > 1e-5:
        tmp = fill_const(shape=x.shape, dtype=x.dtype, value=op.attr('Scale_x'))
        x = mul(x, tmp)
    if op.attr('Scale_y') - 1.0 > 1e-5:
        tmp = fill_const(shape=y.shape, dtype=y.dtype, value=op.attr('Scale_y'))
        y = mul(y, tmp)
    z = add(x, y)
    if op.attr('Scale_out') - 1.0 > 1e-5:
        tmp = fill_const(
            shape=z.shape, dtype=z.dtype, value=op.attr('Scale_out'))
        z = mul(z, tmp)
    return z


@REGISTER_ORIG2PRIM('tanh')
def tanh_orig2prim(op, x):
    return tanh(x)


## NOTE(lml): The second output of reshape2 Xshape, which is only used in reshape2_grad, is meanlingless in new autograd mechanism, thus we use a zero tensor instead.
@REGISTER_ORIG2PRIM('reshape2')
def reshape2_orig2prim(op, shape_t, shape_tl, x):
    assert shape_t is None, 'Can not lower reshape2 into prim ops with shapetensor.'
    assert shape_tl is None, 'Can not lower reshape2 into prim ops with shapetensorlist.'
    y, xshape = get_output_vars(op)
    return reshape(
        x, shape=y.shape), fill_const(
            shape=xshape.shape, dtype=xshape.dtype, value=0.0)


@REGISTER_ORIG2PRIM('concat')
def concat_orig2prim(op, axis_t, xs):
    assert axis_t is None, 'Can not lower concat into prim ops with axistensor.'
    return concat(xs, axis=op.attr('axis'))


@REGISTER_ORIG2PRIM('slice')
def slice_orig2prim(op, ends_t, ends_tl, x, starts_t, starts_tl):
    assert starts_t is None, 'Can not lower concat into prim ops with startstensor.'
    assert ends_t is None, 'Can not lower concat into prim ops with endstensor.'
    assert starts_tl is None, 'Can not lower concat into prim ops with startstensorlist.'
    assert ends_tl is None, 'Can not lower concat into prim ops with endstensorlist.'
    starts = op.attr('starts')
    ends = op.attr('ends')
    strides = [1 for _ in starts]
    axis = op.attr('axes')
    y = slice_select(x, starts=starts, ends=ends, strides=strides, axis=axis)
    # op.attr('decrease_axis') is p[]
    if op.attr('decrease_axis'):
        # get_output_vars return tuple
        y = reshape(y, shape=get_output_vars(op)[0].shape)
    return y


@REGISTER_ORIG2PRIM('fill_zeros_like')
def fill_zeros_like_orig2prim(op, x):
    return fill_const(value=0.0, shape=x.shape, dtype=x.dtype)


@REGISTER_ORIG2PRIM('sum')
def sum_orig2prim(op, xs):
    x0 = xs[0]
    for x in xs[1:]:
        x0 = add(x0, x)
    return x0


@REGISTER_ORIG2PRIM('p_norm')
def p_norm_orig2prim(op, x):
    def num_el(shape):
        n = 1
        for s in shape:
            n = n * s
        return n

    assert op.attr(
        'porder') - 2.0 < 1e-5, 'Only support lower l2 norm currently'
    assert op.attr(
        'asvector'), 'Only support lower pnorm when asvector=True currently'
    if len(x.shape) > 1:
        x = reshape(x, shape=[num_el(x.shape)])
    return sqrt(reduce(mul(x, x), axis=[0]))


@REGISTER_ORIG2PRIM('index_select')
def index_select_orig2prim(op, index_t, x):
    return gather(x, indextensor=index_t, axis=op.attr('dim'))


@REGISTER_ORIG2PRIM('elementwise_sub')
def elementwise_sub_orig2prim(op, x, y):
    if x.shape != y.shape:
        y = broadcast(y, shape=x.shape)
    if op.attr('Scale_x'):
        tmp = fill_const(shape=x.shape, dtype=x.dtype, value=op.attr('Scale_x'))
        x = mul(x, tmp)
    if op.attr('Scale_y'):
        tmp = fill_const(shape=y.shape, dtype=y.dtype, value=op.attr('Scale_y'))
        y = mul(y, tmp)
    z = sub(x, y)
    if op.attr('Scale_out'):
        tmp = fill_const(
            shape=z.shape, dtype=z.dtype, value=op.attr('Scale_out'))
        z = mul(z, tmp)
    return z


@REGISTER_ORIG2PRIM('scale')
def scale_orig2prim(op, x):
    scale_t = fill_const(shape=x.shape, dtype=x.dtype, value=op.attr('scale'))
    bias_t = fill_const(shape=x.shape, dtype=x.dtype, value=op.attr('bias'))
    if op.attr('bias_after_scale'):
        return add(mul(x, scale_t), bias_t)
    else:
        return mul(add(x, bias_t), scale_t)


@REGISTER_ORIG2PRIM('assign')
def assign_orig2prim(op, x):
    zero_t = fill_const(shape=x.shape, dtype=x.dtype, value=0.0)
    return add(x, zero_t)


## Register prim2orig lower rules


@REGISTER_PRIM2ORIG('add_p')
def add_prim2orig(op, x, y):
    return paddle.add(x, y)


@REGISTER_PRIM2ORIG('sub_p')
def sub_prim2orig(op, x, y):
    return paddle.subtract(x, y)


@REGISTER_PRIM2ORIG('mul_p')
def mul_prim2orig(op, x, y):
    return paddle.multiply(x, y)


@REGISTER_PRIM2ORIG('div_p')
def div_prim2orig(op, x, y):
    return paddle.divide(x, y)


@REGISTER_PRIM2ORIG('sqrt_p')
def sqrt_prim2orig(op, x):
    return paddle.sqrt(x)


@REGISTER_PRIM2ORIG('tanh_p')
def tanh_prim2orig(op, x):
    return paddle.tanh(x)


@REGISTER_PRIM2ORIG('reshape_p')
def reshape_prim2orig(op, x):
    return paddle.reshape(x, shape=op.attr('shape'))


@REGISTER_PRIM2ORIG('broadcast_p')
def broadcast_prim2orig(op, x):
    return paddle.broadcast_to(x, shape=op.attr('shape'))


@REGISTER_PRIM2ORIG('transpose_p')
def transpose_prim2orig(op, x):
    return paddle.transpose(x, perm=op.attr('axis'))


@REGISTER_PRIM2ORIG('split_p')
def split_prim2orig(op, x):
    num_or_sections = op.attr('num_or_sections')
    if len(num_or_sections) == 1:
        num_or_sections = num_or_sections[0]
    return paddle.split(
        x, num_or_sections=num_or_sections, axis=op.attr('axis'))


@REGISTER_PRIM2ORIG('concat_p')
def concat_prim2orig(op, xs):
    return paddle.concat(xs, axis=op.attr('axis'))


@REGISTER_PRIM2ORIG('reduce_p')
def reduce_prim2orig(op, x):
    return paddle.sum(x, axis=op.attr('axis'), keepdim=op.attr('keepdim'))


@REGISTER_PRIM2ORIG('matmul_p')
def matmul_prim2orig(op, x, y):
    return paddle.matmul(x, y)


@REGISTER_PRIM2ORIG('slice_select_p')
def slice_select_prim2orig(op, x):
    return paddle.strided_slice(
        x,
        axes=op.attr('axis'),
        starts=op.attr('starts'),
        ends=op.attr('ends'),
        strides=op.attr('strides'))


@REGISTER_PRIM2ORIG('slice_assign_p')
def slice_assign_prim2orig(op, x, y):
    x_copy = add(x, fill_const(0.0, x.shape, x.dtype))
    return set_value(
        x_copy,
        y,
        axis=op.attr('axis'),
        starts=op.attr('starts'),
        ends=op.attr('ends'),
        strides=op.attr('strides'),
        out=x_copy)


@REGISTER_PRIM2ORIG('gather_p')
def gather_prim2orig(op, index_t, x):
    return paddle.gather(x, index_t, axis=op.attr('axis'))


@REGISTER_PRIM2ORIG('scatter_add_p')
def scatter_add_prim2orig(op, index_t, x, y):
    # assert op.attr('axis') == 0
    # using scatter_nd_add
    return paddle.put_along_axis(
        x, index_t, y, axis=op.attr('axis'), reduce='add')


@REGISTER_PRIM2ORIG('fill_constant_p')
def fill_constant_prim2orig(op):
    return paddle.full(
        shape=op.attr('shape'),
        fill_value=op.attr('value'),
        dtype=INT_DTYPE_2_STRING[op.attr('dtype')])


## Register linearize rules
@REGISTER_JVP('add_p')
def add_jvp(op, x_dot, y_dot):
    if x_dot is None:
        return y_dot
    elif y_dot is None:
        return x_dot
    else:
        return linear_jvp(op, x_dot, y_dot)


@REGISTER_JVP('sub_p')
def sub_jvp(op, x_dot, y_dot):
    if x_dot is None:
        return neg(y_dot)
    elif y_dot is None:
        return x_dot
    else:
        return linear_jvp(op, x_dot, y_dot)


@REGISTER_JVP('mul_p')
def mul_jvp(op, x_dot, y_dot):
    if x_dot is None and y_dot is None:
        return None
    x, y = op_position_inputs(op)
    if x_dot is None:
        return mul(x, y_dot)
    elif y_dot is None:
        return mul(x_dot, y)
    else:
        t1, t2 = mul(x_dot, y), mul(x, y_dot)
        z_dot = add(t1, t2)
        return z_dot


@REGISTER_JVP('div_p')
def div_jvp(op, x_dot, y_dot):
    if x_dot is None and y_dot is None:
        return None
    x, y = op_position_inputs(op)
    if y_dot is None:
        return div(x_dot, y)
    elif x_dot is None:
        return div(mul(x, y_dot), mul(y, y))
    else:
        t1 = div(x_dot, y)
        t2 = div(mul(x, y_dot), mul(y, y))
        return sub(t1, t2)


@REGISTER_JVP('sqrt_p')
def sqrt_jvp(op, x_dot):
    if x_dot is None:
        return None
    y = op_position_output(op)
    c2 = fill_const(value=2.0, shape=y.shape, dtype=y.dtype)
    y_dot = div(x_dot, mul(c2, y))
    return y_dot


@REGISTER_JVP('tanh_p')
def tanh_jvp(op, x_dot):
    if x_dot is None:
        return None
    y = op_position_output(op)
    c1 = fill_const(value=1.0, shape=y.shape, dtype=y.dtype)
    y_dot = mul(x_dot, sub(c1, mul(y, y)))
    return y_dot


@REGISTER_JVP('reshape_p')
def reshape_jvp(op, x_dot):
    if x_dot is None:
        return None
    shape = op.attr('shape')
    return linear_jvp(op, x_dot, shape=shape)


@REGISTER_JVP('broadcast_p')
def broadcast_jvp(op, x_dot):
    if x_dot is None:
        return None
    shape = op.attr('shape')
    return linear_jvp(op, x_dot, shape=shape)


@REGISTER_JVP('transpose_p')
def transpose_jvp(op, x_dot):
    if x_dot is None:
        return None
    axis = op.attr('axis')
    return linear_jvp(op, x_dot, axis=axis)


@REGISTER_JVP('split_p')
def split_jvp(op, x_dot):
    if x_dot is None:
        return None
    num_or_sections = op.attr('num_or_sections')
    axis = op.attr('axis')
    return linear_jvp(op, x_dot, num_or_sections=num_or_sections, axis=axis)


@REGISTER_JVP('concat_p')
def concat_jvp(op, xs_dot):
    if xs_dot is None:
        return None
    axis = op.attr('axis')
    return linear_jvp(op, xs_dot, axis=axis)


@REGISTER_JVP('reduce_p')
def reduce_jvp(op, x_dot):
    if x_dot is None:
        return None
    axis = op.attr('axis')
    keepdim = op.attr('keepdim')
    return linear_jvp(op, x_dot, axis=axis, keepdim=keepdim)


@REGISTER_JVP('matmul_p')
def matmul_jvp(op, x_dot, y_dot):
    if x_dot is None and y_dot is None:
        return None
    x, y = op_position_inputs(op)
    if x_dot is None:
        return matmul(x, y_dot)
    elif y_dot is None:
        return matmul(x_dot, y)
    else:
        t1 = matmul(x, y_dot)
        t2 = matmul(x_dot, y)
        return add(t1, t2)


@REGISTER_JVP('slice_select_p')
def slice_select_jvp(op, x_dot):
    if x_dot is None:
        return None
    axis = op.attr('axis')
    starts = op.attr('starts')
    ends = op.attr('ends')
    strides = op.attr('strides')
    return linear_jvp(
        op, x_dot, axis=axis, starts=starts, ends=ends, strides=strides)


@REGISTER_JVP('slice_assign_p')
def slice_assign_jvp(op, x_dot, y_dot):
    if x_dot is None:
        assert y_dot is None
        return None
    else:
        assert y_dot is not None
    axis = op.attr('axis')
    starts = op.attr('starts')
    ends = op.attr('ends')
    strides = op.attr('strides')
    return linear_jvp(
        op, x_dot, y_dot, axis=axis, starts=starts, ends=ends, strides=strides)


@REGISTER_JVP('gather_p')
def gather_jvp(op, x_dot, indextensor):
    if x_dot is None:
        return None
    _, indextensor = op_position_inputs(op)
    axis = op.attr('axis')
    return linear_jvp(op, x_dot, indextensor, axis=axis)


@REGISTER_JVP('scatter_add_p')
def scatter_add_jvp(op, x_dot, y_dot):
    if x_dot is None:
        return None
    _, _, indextensor = op_position_inputs(op)
    axis = op.attr('axis')
    return linear_jvp(op, x_dot, y_dot, indextensor, axis=axis)


## Register transpose rules


@REGISTER_TRANSPOSE('add_p')
def add_transpose(op, check_dot, z_bar):
    x, y = op_position_inputs(op)
    assert check_dot(x) or check_dot(y)
    x_bar = z_bar if check_dot(x) else None
    y_bar = z_bar if check_dot(y) else None
    return x_bar, y_bar


@REGISTER_TRANSPOSE('sub_p')
def sub_transpose(op, check_dot, z_bar):
    x, y = op_position_inputs(op)
    assert check_dot(x) or check_dot(y)
    x_bar = z_bar if check_dot(x) else None
    y_bar = neg(z_bar) if check_dot(y) else None
    return x_bar, y_bar


@REGISTER_TRANSPOSE('mul_p')
def mul_transpose(op, check_dot, z_bar):
    x, y = op_position_inputs(op)
    assert check_dot(x) ^ check_dot(y)
    if check_dot(x):
        return mul(z_bar, y), None
    else:
        return None, mul(x, z_bar)


@REGISTER_TRANSPOSE('div_p')
def div_transpose(op, check_dot, z_bar):
    x, y = op_position_inputs(op)
    assert not check_dot(y)
    x_bar = div(z_bar, y) if check_dot(x) else None
    return x_bar, None


@REGISTER_TRANSPOSE('reshape_p')
def reshape_transpose(op, check_dot, y_bar):
    x, = op_position_inputs(op)
    assert check_dot(x)
    return reshape(y_bar, shape=x.shape)


@REGISTER_TRANSPOSE('broadcast_p')
def broadcast_transpose(op, check_dot, y_bar):
    x, = op_position_inputs(op)
    assert check_dot(x)
    bat = len(y_bar.shape) - len(x.shape)
    axis = list(range(bat))
    keepdim = [(bat + i) for i, s in enumerate(x.shape) if s == 1]
    axis += keepdim
    # TODO: Change it. keepdim boolean
    out = reduce(y_bar, axis=axis, keepdim=False)
    return reshape(out, x.shape)


@REGISTER_TRANSPOSE('transpose_p')
def transpose_transpose(op, check_dot, y_bar):
    x, = op_position_inputs(op)
    assert check_dot(x)
    axis = op.attr('axis')
    reordered = sorted((k, i) for i, k in enumerate(axis))
    axis = [i for k, i in reordered]
    return transpose(y_bar, axis=axis)


@REGISTER_TRANSPOSE('split_p')
def split_transpose(op, check_dot, ys_bar):
    x, = op_position_inputs(op)
    assert check_dot(x)
    return concat(ys_bar, axis=op.attr('axis'))


@REGISTER_TRANSPOSE('concat_p')
def concat_transpose(op, check_dot, y_bar):
    xs, = op_position_inputs(op)
    for x in xs:
        assert check_dot(x)
    axis = op.attr('axis')
    sections = [x.shape[axis] for x in xs]
    return split(y_bar, num_or_sections=sections, axis=axis)


@REGISTER_TRANSPOSE('reduce_p')
def reduce_transpose(op, check_dot, y_bar):
    x, = op_position_inputs(op)
    assert check_dot(x)
    axes = op.attr('axis')
    shape = tuple(1 if i in axes else size for i, size in enumerate(x.shape))
    t = reshape(y_bar, shape=shape)
    return broadcast(t, shape=x.shape)


@REGISTER_TRANSPOSE('matmul_p')
def matmul_transpose(op, check_dot, z_bar):
    x, y = op_position_inputs(op)
    assert check_dot(x) ^ check_dot(y)
    # TODO: replace it. this is hacky
    axis = [1, 0] if len(x.shape) == 2 else [0, 2, 1]
    if check_dot(x):
        return matmul(z_bar, transpose(y, axis=axis)), None
    else:
        return None, matmul(transpose(x, axis=axis), z_bar)


@REGISTER_TRANSPOSE('slice_select_p')
def slice_select_transpose(op, check_dot, y_bar):
    x, = op_position_inputs(op)
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
    x, y = op_position_inputs(op)
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
    x, indextensor = op_position_inputs(op)
    assert check_dot(x)
    axis = op.attr('axis')
    zeros = fill_const(0.0, x.shape, x.dtype)
    x_bar = scatter_add(zeros, y_bar, indextensor, axis=axis)
    indextensor_bar = None
    return x_bar, indextensor_bar


@REGISTER_TRANSPOSE('scatter_add_p')
def scatter_add_transpose(op, check_dot, z_bar):
    indextensor, x, y = get_input_vars(op)
    assert check_dot(x) and check_dot(y)
    axis = op.attr('axis')
    zeros = fill_const(value=0.0, shape=y.shape, dtype=y.dtype)
    x_bar = scatter_add(z_bar, zeros, indextensor, axis=axis)
    y_bar = gather(z_bar, indextensor, axis=axis)
    indextensor_bar = None   
    return x_bar, y_bar, indextensor_bar
