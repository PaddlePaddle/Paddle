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

import numpy as np
import warnings
import six
import os
import inspect
from ..fluid.layer_helper import LayerHelper
from ..fluid.param_attr import ParamAttr
from ..fluid.framework import Variable, OpProtoHolder, in_dygraph_mode, convert_np_dtype_to_dtype_
from ..fluid import core
from ..fluid.data_feeder import convert_dtype, check_variable_and_dtype, check_type, check_dtype
from ..fluid.layers import utils

# TODO: define functions of linear algebra   
__all__ = [
    #           'matmul', 
    #           'dot',
    #           'einsum',
    #           'morm',
    'transpose',
    #           'dist',
    #           't',
    #           'cross',
    #           'cholesky',
    #           'tensordot',
    'bmm'
]


def transpose(x, perm, name=None):
    """
    Permute the data dimensions of `input` according to `perm`.

    The `i`-th dimension  of the returned tensor will correspond to the
    perm[i]-th dimension of `input`.

    Args:
        x (Variable): The input Tensor. It is a N-D Tensor of data types float32, float64, int32.
        perm (list): Permute the input according to the data of perm.
        name (str): The name of this layer. It is optional.

    Returns:
        Variable: A transposed n-D Tensor, with data type being float32, float64, int32, int64.

    For Example:

        .. code-block:: text

         x = [[[ 1  2  3  4] [ 5  6  7  8] [ 9 10 11 12]]
             [[13 14 15 16] [17 18 19 20] [21 22 23 24]]]
         shape(x) =  [2,3,4]

         # Example 1
         perm0 = [1,0,2]
         y_perm0 = [[[ 1  2  3  4] [13 14 15 16]]
                   [[ 5  6  7  8]  [17 18 19 20]]
                   [[ 9 10 11 12]  [21 22 23 24]]]
         shape(y_perm0) = [3,2,4]

         # Example 2
         perm1 = [2,1,0]
         y_perm1 = [[[ 1 13] [ 5 17] [ 9 21]]
                   [[ 2 14] [ 6 18] [10 22]]
                   [[ 3 15]  [ 7 19]  [11 23]]
                   [[ 4 16]  [ 8 20]  [12 24]]]
         shape(y_perm1) = [4,3,2]

    Examples:

        .. code-block:: python

            # use append_batch_size=False to avoid prepending extra
            # batch size in shape
            import paddle.fluid as fluid
            import paddle.tensor as tensor
            x = fluid.layers.data(name='x', shape=[2, 3, 4],
                            dtype='float32', append_batch_size=False)
            x_transposed = tensor.transpose(x, perm=[1, 0, 2])
            print x_transposed.shape
            #(3L, 2L, 4L)

    """
    if in_dygraph_mode():
        attrs = {'axis': perm}
        inputs = {'X': [x]}
        outs = core.ops.transpose2(inputs, attrs)
        return outs['Out'][0]

    check_variable_and_dtype(
        x, 'x', ['float16', 'float32', 'float64', 'int32', 'int64'],
        'transpose')
    check_type(perm, 'perm', list, 'transpose')

    if len(perm) != len(x.shape):
        raise ValueError(
            "Input(perm) is the permutation of dimensions of Input(x), "
            "its length should be equal to dimensions of Input(x), "
            "but received dimension of Input(x) is %s, "
            "the length of Input(perm) is %s." % (len(x.shape), len(perm)))
    for idx, dim in enumerate(perm):
        if dim >= len(x.shape):
            raise ValueError(
                "Each element in Input(perm) should be less than Input(x)'s dimension, "
                "but %d-th element in Input(perm) is %d which exceeds Input(x)'s "
                "dimension %d." % (idx, perm[idx], len(x.shape)))

    helper = LayerHelper('transpose', **locals())
    out = helper.create_variable_for_type_inference(x.dtype)
    x_shape = helper.create_variable_for_type_inference(x.dtype)
    helper.append_op(
        type='transpose2',
        inputs={'X': [x]},
        outputs={'Out': [out],
                 'XShape': [x_shape]},
        attrs={'axis': perm})
    return out


def bmm(x, y, name=None):
    def __check_input(x, y):
        var_names = {'x': x, 'y': y}
        for name, val in var_names.items():
            check_type_and_dtype(val, name, Variable,
                                 ['float16', 'float32', 'float64'], 'matmul')
        x_shape = list(x.shape)
        y_shape = list(y.shape)
        if len(x_shape) != 3 or len(y_shape) != 3:
            raise ValueError("Both of the two matrix should be 3 dimensions,"
                             "But received X shape: %s, Y shape: %s." %
                             (len(x_shape), len(y_shape)))

        # check the inner 2 dimensions
        if x_shape[-1] != y_shape[-2]:
            raise ValueError(
                (x_shape[-1] == -1) or (y_shape[-2] == -1),
                "After performing an optional transpose, Input X's width should be "
                "equal to Y's width for multiplication prerequisites."
                "But received X's shape: %s, Y's shape: %s\n" %
                (x_shape, y_shape))

        for i, dim_x in enumerate(x_shape[:-2]):
            # don't check neg shape
            if dim_x < 0 or y_shape[i] < 0:
                continue
            if dim_x != y_shape[i]:
                raise ValueError(
                    "When the matrix is larger than 2 dimensions, the higher "
                    "dimensional values of the two matrices need to be equal. "
                    "But received x_shape[%d] != y_shape[%d]. X's shape: %s, "
                    "Y's shape: %s.\n" % (i, i, x_shape, y_shape))

    __check_input(x, y)

    helper = LayerHelper('bmm', **locals())
    out = helper.create_variable_for_type_inference(dtype=x.dtype)
    helper.append_op(
        type='bmm', inputs={'X': x,
                            'Y': y}, outputs={'Out': out}, attrs=attrs)
    return out
