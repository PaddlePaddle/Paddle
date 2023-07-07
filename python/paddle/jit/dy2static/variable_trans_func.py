#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
from paddle.fluid.framework import Variable
from paddle.utils import gast, is_sequence, map_structure

from .utils import UndefinedVar, create_undefined_variable

__all__ = []


def create_undefined_var(name):
    func_code = f"{name} = _jst.UndefinedVar('{name}')"
    return gast.parse(func_code).body[0]


def create_fill_constant_node(name, value=0):
    func_code = f"{name} = paddle.full(shape=[1], "
    if isinstance(value, bool):
        func_code += "dtype='bool', fill_value={}, name='{}')".format(
            value, name
        )
        return gast.parse(func_code).body[0]
    if isinstance(value, float):
        func_code += "dtype='float64', fill_value={}, name='{}')".format(
            value, name
        )
        return gast.parse(func_code).body[0]

    if isinstance(value, int):
        func_code += "dtype='int64', fill_value={}, name='{}')".format(
            value, name
        )
        return gast.parse(func_code).body[0]


def to_static_variable(x):
    '''
    Translate a Python Tensor to PaddlePaddle static graph Tensor
    '''
    if isinstance(x, bool):
        return paddle.full(shape=[], dtype='bool', fill_value=x)
    if isinstance(x, float):
        return paddle.full(shape=[], dtype='float64', fill_value=x)
    if isinstance(x, int):
        return paddle.full(shape=[], dtype='int64', fill_value=x)
    if isinstance(x, UndefinedVar) or x is None:
        """
        for early return case, we need a variable to represent None, current we use data_layer_not_check.
        """
        return create_undefined_variable()
    if is_sequence(x):
        return map_structure(to_static_variable, x)
    return x


def create_bool_as_type(x, value=True):
    '''
    Create a bool variable, which type is the same as x.
    '''
    if isinstance(x, Variable):
        return paddle.full(shape=[1], fill_value=value, dtype="bool")
    else:
        return value


def create_bool_node(name, value):
    '''
    Create a assign stmt for name = value .
    '''
    assert isinstance(value, bool)
    node = f"{name} = {value}"
    return gast.parse(node).body[0]
