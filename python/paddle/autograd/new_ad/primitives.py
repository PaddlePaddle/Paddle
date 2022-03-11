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

    def __init__(self, optype, nins, nouts) -> None:
        self.optype = optype
        self.nins = nins
        self.nouts = nouts

    def __call__(self, ins, outs, attrs) -> Any:
        runner = get_current_runner()
        runner.run_op(self, ins, outs, attrs)


RESHAPE = Primitive('reshape_p', 1, 1)
BCAST = Primitive('broadcast_p', 1, 1)
REDUCE = Primitive('reduce_p', 1, 1)
TRANSPOSE = Primitive('transpose_p', 1, 1)
SPLIT = Primitive('split_p', 1, None)
CONCAT = Primitive('concat_p', None, 1)
SLISELECT = Primitive('slice_select_p', 1, 1)
SLIASSIGN = Primitive('slice_assign_p', 2, 1)
INDSELECT = Primitive('index_select_p', None, 1)
INDASSIGN = Primitive('index_assign_p', None, 1)
ADD = Primitive('add_p', 2, 1)
SUB = Primitive('sub_p', 2, 1)
MUL = Primitive('mul_p', 2, 1)
DIV = Primitive('div_p', 2, 1)
SQRT = Primitive('sqrt_p', 1, 1)
TANH = Primitive('tanh_p', 1, 1)
MATMUL = Primitive('matmul_p', 2, 1)
FILL = Primitive('fill_constant_p', None, 1)

# jvp and transpose rules on primitives
jvpmakers = {}
transposemakers = {}


def add_jvpmaker(x, y, z):
    return lambda tx, ty: ADD(tx, ty, make_var(True, z))


def sub_jvpmaker(x, y, z):
    return lambda tx, ty: SUB(tx, ty, make_var(True, z))


def mul_jvpmaker(x, y, z):
    return lambda tx, ty: ADD(MUL(x, ty, make_var(True, z)), MUL(tx, y, make_var(True, z)), make_var(True, z))


jvpmakers[ADD] = add_jvpmaker
jvpmakers[SUB] = sub_jvpmaker
jvpmakers[MUL] = mul_jvpmaker


def add_transposemaker(x, y):
    assert is_tangent(x) and is_tangent(y)
    return lambda t: t, lambda t: t


def sub_transposemaker(x, y):
    assert is_tangent(x) and is_tangent(y)
    return lambda t: t, lambda t: NEG(t)


def mul_transposemaker(x, y):
    assert is_tangent(x) ^ is_tangent(y)
    if is_tangent(x):
        return lambda t: (MUL(t, y), None)
    else:
        return lambda t: (None, MUL(x, t))


transposemakers[ADD] = add_transposemaker
transposemakers[SUB] = sub_transposemaker
transposemakers[MUL] = mul_transposemaker

# rules to transform original operator to primitives
primitivemakers = {}


# TODO(lml): how to support concat
def add_maker(x, y, z):
    if x.shape == y.shape:
        return lambda _x, _y: ADD(_x, _y, z.name())
    else:
        return lambda _x, _y: ADD(_x, BCAST(_y, make_var(), shape=x.shape))


primitivemakers['elementwise_add'] = add_maker
