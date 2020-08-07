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

from __future__ import print_function

import six
import gast

from paddle.fluid import core
from paddle.fluid.framework import Variable
from paddle.fluid.layers import fill_constant
from paddle.fluid.layers.control_flow import array_write, array_read
from paddle.fluid.layer_helper import LayerHelper

__all__ = [
    'to_static_variable_gast_node', 'create_static_variable_gast_node',
    'data_layer_not_check'
]


def data_layer_not_check(name, shape, dtype='float32', lod_level=0):
    """
    This function creates a variable on the global block. Unlike
    `paddle.fluid.data` , the created variable doesn't check the dtype and the
    shape of feed data because dygraph input data can be variable-length.
    This API is used in translating dygraph into static graph.

     Note: 
        The default :code:`stop_gradient` attribute of the Variable created by
        this API is true, which means the gradient won't be passed backward
        through the data Varaible. Set :code:`var.stop_gradient = False` If
        user would like to pass backward gradient.

    Args:
       name (str): The name/alias of the variable, see :ref:`api_guide_Name`
           for more details.
       shape (list|tuple): List|Tuple of integers declaring the shape. You can
           set "None" at a dimension to indicate the dimension can be of any
           size. For example, it is useful to set changeable batch size as "None" 
       dtype (np.dtype|VarType|str, optional): The type of the data. Supported
           dtype: bool, float16, float32, float64, int8, int16, int32, int64,
           uint8. Default: float32
       lod_level (int, optional): The LoD level of the LoDTensor. Usually users
           don't have to set this value. For more details about when and how to
           use LoD level, see :ref:`user_guide_lod_tensor` . Default: 0

    Returns:
        Variable: The global variable that gives access to the data.
    """
    helper = LayerHelper('data', **locals())
    shape = list(shape)
    for i in six.moves.range(len(shape)):
        if shape[i] is None:
            shape[i] = -1

    return helper.create_global_variable(
        name=name,
        shape=shape,
        dtype=dtype,
        type=core.VarDesc.VarType.LOD_TENSOR,
        stop_gradient=True,
        lod_level=lod_level,
        is_data=True,
        need_check_feed=False)


def to_static_variable_gast_node(name):
    func_code = "{} = fluid.dygraph.dygraph_to_static.variable_trans_func\
        .to_static_variable({})".format(name, name)
    return gast.parse(func_code).body[0]


def create_static_variable_gast_node(name):
    func_code = "{} = fluid.dygraph.dygraph_to_static.variable_trans_func\
        .data_layer_not_check(name='{}', shape=[-1], dtype='float32')".format(
        name, name)
    return gast.parse(func_code).body[0]


def create_fill_constant_node(name, value):
    func_code = "{} = fluid.layers.fill_constant(shape=[1], ".format(name)
    if isinstance(value, bool):
        func_code += "dtype='bool', value={})".format(value)
        return gast.parse(func_code).body[0]
    if isinstance(value, float):
        func_code += "dtype='float64', value={})".format(value)
        return gast.parse(func_code).body[0]

    if six.PY2:
        if isinstance(value, int):
            func_code += "dtype='int32', value={})".format(value)
            return gast.parse(func_code).body[0]
        if isinstance(value, long):
            func_code += "dtype='int64', value={})".format(value)
            return gast.parse(func_code).body[0]
    else:
        if isinstance(value, int):
            func_code += "dtype='int64', value={})".format(value)
            return gast.parse(func_code).body[0]


def to_static_variable(x):
    '''
    Translate a Python variable to PaddlePaddle static graph variable
    '''
    if isinstance(x, bool):
        return fill_constant(shape=[1], dtype='bool', value=x)
    if isinstance(x, float):
        return fill_constant(shape=[1], dtype='float64', value=x)

    if isinstance(x, six.integer_types):
        return fill_constant(shape=[1], dtype='int64', value=x)

    return x


class DynamicList(object):
    '''
    A class casts python list to LoDTensorArray dynamically at run time.
    '''

    def __init__(self, iterable):
        self.private_type = self._get_same_type(iterable)
        # self.private_array is LoDTensorArray if all types are LoDTensor
        # else python list
        if self.private_type == Varaible:
            self.private_array = core.LoDTensorArray()
            count = 0
            for i in iterable:
                array_i = fill_constant(shape=[1], dtype='int64', value=i)
                array_write(args[i], array_i, self.private_array)
        else:
            self.private_type = list(iterable)

    def __len__(self):
        return len(self.private_array)

    def __getitem__(self, i):
        return self.private_array[i]

    def __setitem__(self, i, elem):
        if self.private_type == Variable and not isinstance(elem, Variable):
            # Transfer from LoDTensorArray to python list
            self.private_array = self.private_array._move_to_list()
            self.private_type = self._get_same_type(args)
        self.private_array[i] = elem

    def append(self, elem):
        if self.private_type == Variable and not isinstance(elem, Variable):
            # Transfer from LoDTensorArray to python list
            self.private_array = self.private_array._move_to_list()
            self.private_array.append(elem)
            self.private_type = self._get_same_type(args)
            return
        self.private_array.append(elem)

    def pop(self):
        ret = self.private_array.pop()
        if len(self.private_array) == 0:
            self.private_type = Variable
            self.private_array = core.LoDTensorArray()
        return ret

    def extend(self, iterable):
        if self.private_type == Variable:
            for i in iterable:
                self.private_array.append(i)
            return
        self.private_array.extend(iterable)

    def insert(self, i, x):
        if self.private_type == Variable:
            raise NotImplementedError(
                "LoDTensorArray hasn't implemented 'insert' method.")
        self.private_array.insert(x)

    def remove(self, x):
        if self.private_type == Variable:
            raise NotImplementedError(
                "LoDTensorArray hasn't implemented 'remove' method.")
        self.private_array.remove(x)

    def clear(self):
        self.private_type = Variable
        self.private_array = core.LoDTensorArray()

    def index(self, x, start=None, end=None):
        if self.private_type == Variable:
            raise NotImplementedError(
                "LoDTensorArray hasn't implemented 'index' method.")
        return self.private_array.index(x, start, end)

    def count(self, x):
        if self.private_type == Variable:
            raise NotImplementedError(
                "LoDTensorArray hasn't implemented 'count' method.")
        return self.private_array.count(x)

    def sort(self, key=None, reverse=None):
        if self.private_type == Variable:
            raise NotImplementedError(
                "LoDTensorArray hasn't implemented 'sort' method.")
        return self.private_array.sort(key, reverse)

    def reverse(self):
        if self.private_type == Variable:
            raise NotImplementedError(
                "LoDTensorArray hasn't implemented 'reverse' method.")
        return self.private_array.reverse()

    def copy(self):
        return self.private_array[:]

    def _get_same_type(self, iterable):
        """
        Returns the type if all iterable have same type. Returns Variable if
        iterable is empty. Returns None if iterable contains different types.
        """
        len_args = len(iterable)
        if len_args == 0:
            return Variable
        ret = "Undefined"
        for i in iterable:
            if ret == "Undefined":
                ret = type(i)
            elif not isinstance(i, ret):
                return None
        return ret
