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
    #            'matmul', 
    #            'dot',
    #            'einsum',
    #            'morm',
    #            'transpose',
    #            'dist',
    't'
]

#            'cross',
#            'cholesky',
#            'tensordot'


def t(input, name=None):
    """
    Expects input to be <= 2-D tensor and transposes dimensions 0 and 1.

    0-D and 1-D tensors are returned as it is and 2-D tensor can be seen as 
    a short-hand function for transpose(input, 0, 1).

    Args:
        input (Variable): The input Tensor. It is a N-D (N<=2) Tensor of data types float32, float64, int32.
        name(str, optional): The default value is None.  Normally there is no need for 
            user to set this property.  For more information, please refer to :ref:`api_guide_Name`

    Returns:
        Variable: A transposed n-D Tensor, with data type being float32, float64, int32, int64.

    For Example:

        .. code-block:: text

        # Example 1 (1-D tensor)
         x = tensor([0.79])
         paddle.t(x) = tensor([0.79])

         # Example 2 (1-D tensor)
         x = tensor([0.79, 0.84, 0.32])
         paddle.t(x) = tensor([0.79, 0.84, 0.32])
        
         # Example 3 (2-D tensor)
         x = tensor([0.79, 0.84, 0.32],
                    [0.64, 0.14, 0.57])
         paddle.t(x) = tensor([0.79, 0.64],
                                     [0.84, 0.14],
                                     [0.32, 0.57])

    Examples:

        .. code-block:: python

            import paddle
            import paddle.fluid as fluid
            x = fluid.data(name='x', shape=[2, 3],
                            dtype='float32')
            x_transposed = paddle.t(x)
            print x_transposed.shape
            #(3L, 2L)

    """
    if len(input.shape) > 2:
        raise ValueError(
            "Input(input) only support N-D (N<=2) tensor, but received "
            "length of Input(input) is %s. Perhaps you can use paddle."
            "tensor.transpose() instead." % len(input.shape))
    if in_dygraph_mode():
        if len(input.shape) == 1:
            return input
        # 2-D tensor
        if input.shape[0] == 1 or input.shape[1] == 1:
            return input
        perm = [1, 0]
        out, _ = core.ops.transpose2(input, 'axis', perm)
        return out

    check_variable_and_dtype(
        input, 'input', ['float16', 'float32', 'float64', 'int32', 'int64'],
        'transpose')

    helper = LayerHelper('t', **locals())
    out = helper.create_variable_for_type_inference(input.dtype)
    input_shape = helper.create_variable_for_type_inference(input.dtype)
    if len(input.shape) == 1:
        out = input
    elif input.shape[0] == 1 or input.shape[1] == 1:
        out = input
    else:
        helper.append_op(
            type='transpose2',
            inputs={'X': [input]},
            outputs={'Out': [out],
                     'XShape': [input_shape]},
            attrs={'axis': [1, 0]})
    return out
