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

from .primops import (add, sub, mul, div, sqrt, tanh, reshape, broadcast,
                      transpose, split, concat, reduce, matmul, slice_select,
                      slice_assign, gather, scatter_add, fill_const)

import functools


class Registry(object):
    """ A general registry object. """

    def __init__(self, name):
        self.name = name
        self.tab = {}

    def register(self, name, value):
        assert name not in self.tab
        self.tab[name] = value
    
    def lookup(self, name):
        assert name in self.tab, f'No registry entry is found with name: {name}'
        return self.tab[name]

_primop_jvp = Registry('primop_jvps')
_primop_transpose = Registry('primop_vjps')


def REGISTER_JVP(op_type):
    """Decorator for registering the JVP function for a primitive op.
    
    Usage:

    .. code-block:: python

        @RegisterJVP('add')
        def add_jvp(op, x_dot, y_dot):
            return primops.add(x_dot, y_dot)
    
    """
    assert isinstance(op_type, str)
    def wrapper(f):
        def _jvp(op, *args, **kwargs): 
            assert op.type == op_type
            return f(op, *args, **kwargs)
        _primop_jvp.register(op_type, _jvp)

    return wrapper


def REGISTER_TRANSPOSE(op_type):
    """Decorator for registering the VJP function for a primitive op.
    
    Usage:

    .. code-block:: python

        @RegisterJVP('add')
        def add_transpose(op, z_bar):
            return z_bar, z_bar
    
    """
    assert isinstance(op_type, str)
    def wrapper(f):
        def _transpose(op, *args, **kwargs): 
            assert op.type == op_type
            return f(op, *args, **kwargs)
        _primop_transpose.register(op_type, _transpose)

    return wrapper


def get_input_vars(op):
    return tuple(map(op.block.var, op.input_arg_names))


def get_output_vars(op):
    return tuple(map(op.block.var, op.output_arg_names))


def linear_jvp(op, *args):
    out_dot = op(*args, **op.all_attrs())
    return out_dot


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
    z_dot = t1 - t2
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
    return linear_jvp(op, x_dot)


@REGISTER_JVP('broadcast_p')
def broadcast_jvp(op, x_dot):
    return linear_jvp(op, x_dot)


@REGISTER_JVP('transpose_p')
def transpose_jvp(op, x_dot):
    return linear_jvp(op, x_dot)


@REGISTER_JVP('split_p')
def split_jvp(op, x_dot):
    return linear_jvp(op, x_dot)


@REGISTER_JVP('concat_p')
def concat_jvp(op, xs_dot):
    return linear_jvp(op, xs_dot)


@REGISTER_JVP('reduce_p')
def reduce_jvp(op, x_dot):
    return linear_jvp(op, x_dot)


@REGISTER_JVP('matmul_p')
def matmul_jvp(op, x_dot, y_dot):
    x, y = get_input_vars(op)
    t1 = matmul(x, y_dot)
    t2 = matmul(x_dot, y)
    z_dot = add(t1, t2)
    return z_dot


@REGISTER_JVP('slice_select_p')
def slice_select_jvp(op, x_dot):
    return linear_jvp(op, x_dot)


@REGISTER_JVP('slice_assign_p')
def slice_assign_jvp(op, x_dot, y_dot):
    return linear_jvp(op, x_dot, y_dot)


@REGISTER_JVP('gather_p')
def gather_jvp(op, x_dot):
    _, indextensor = get_input_vars(op)
    return linear_jvp(op, x_dot, indextensor)


@REGISTER_JVP('scatter_p')
def scatter_add_jvp(op, x_dot, y_dot):
    _, _, indextensor = get_input_vars(op)
    return linear_jvp(op, x_dot, y_dot, indextensor)


