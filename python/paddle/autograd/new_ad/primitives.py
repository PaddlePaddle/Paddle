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

from typing import Any
from .runner import get_current_runner


# primitives
class Primitive(object):
    """ Primitive OP.
  
    In instance of `Primitive` identifies a primitive and provides
    interfaces for using the primitive.

    """

    def __init__(self, optype) -> None:
        self.optype = optype

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        runner = get_current_runner()
        runner.run_op(self, *args, **kwargs)


RESHAPE = Primitive('reshape_p')
BCAST = Primitive('broadcast_p')
REDUCE = Primitive('reduce_p')
TRANSPOSE = Primitive('transpose_p')
SPLIT = Primitive('split_p')
CONCAT = Primitive('concat_p')
SLISELECT = Primitive('slice_select_p')
SLIASSIGN = Primitive('slice_assign_p')
INDSELECT = Primitive('index_select_p')
INDASSIGN = Primitive('index_assign_p')
ADD = Primitive('add_p')
SUB = Primitive('sub_p')
MUL = Primitive('mul_p')
DIV = Primitive('div_p')
SQRT = Primitive('sqrt_p')
TANH = Primitive('tanh_p')
MATMUL = Primitive('matmul_p')
FILL = Primitive('fill_constant_p')

# jvp and transpose rules on primitives
jvpmakers = {}
transposemakers = {}


def add_jvpmaker(x, y):
    return lambda tx, ty: ADD(tx, ty)


def sub_jvpmaker(x, y):
    return lambda tx, ty: SUB(tx, ty)


def mul_jvpmaker(x, y):
    return lambda tx, ty: ADD(MUL(x, ty), MUL(tx, y))


jvpmakers[ADD] = add_jvpmaker
jvpmakers[SUB] = sub_jvpmaker
jvpmakers[MUL] = mul_jvpmaker


def add_transposemaker(x, y):
    assert x.is_tangent and y.is_tangent
    return lambda t: t, lambda t: t


def sub_transposemaker(x, y):
    assert x.is_tangent and y.is_tangent
    return lambda t: t, lambda t: NEG(t)


def mul_transposemaker(x, y):
    assert x.is_tangent ^ y.is_tangent
    if x.is_tangent:
        return lambda t: (MUL(t, y), None)
    else:
        return lambda t: (None, MUL(x, t))


transposemakers[ADD] = add_transposemaker
transposemakers[SUB] = sub_transposemaker
transposemakers[MUL] = mul_transposemaker


# Changes on original operator and variable 
# TODO(add convert2primitive method to class Operator)
class Operator(object):
    def convert2primitive(self) -> None:
        runner = get_current_runner()
        runner.run_op(self)


# TODO(add is_tangent content to class Variable)
class Variable(object):
    def __init__(self, is_tangent=False):
        self.is_tangent = is_tangent


# rules to transform original operator to primitives
primitivemakers = {}


def add_maker(x, y):
    if x.shape == y.shape:
        return lambda _x, _y: ADD(_x, _y)
    else:
        return lambda _x, _y: ADD(_x, BCAST(_y, x.shape))


primitivemakers['elementwise_add'] = add_maker
