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
from ..fluid.framework import Variable, OpProtoHolder, in_dygraph_mode, convert_np_dtype_to_dtype_, _varbase_creator, device_guard
from ..fluid import core
from ..fluid.data_feeder import convert_dtype, check_variable_and_dtype, check_type, check_dtype
from ..fluid.layers import utils
from ..fluid.initializer import Constant
from ..fluid.layers import fill_constant

# TODO: define functions to get create a tensor  
__all__ = [
    #            'create_tensor',
    #            'create_lod_tensor', 
    #            'create_random_int_lodtensor',
    #            'crop_tensor', 
    #            'diag', 'eye', 
    #            'fill_constant', 
    #            'get_tensor_from_selected_rows', 
    #            'linspace', 
    #            'ones', 
    #            'ones_like', 
    #            'range', 
    #            'zeros', 
    #            'zeros_like', 
    'arange',
    #            'eye',
    'full',
    #            'linspace',
    #            'full_like',
    #            'triu',
    #            'tril',
    #            'meshgrid'
]


def full(shape,
         fill_value,
         out=None,
         dtype=None,
         device=None,
         stop_gradient=True,
         name=None):
    """
    This function return a Tensor with the `fill_value` which size is same as `shape`
    
    Args:
        shape(list|tuple|Variable): Shape of the Tensor to be created.
                The data type is ``int32`` or ``int64`` . If ``shape`` is a list or tuple,
                the elements of it should be integers or Tensors with shape [1].
                If ``shape`` is an Variable, it should be an 1-D Tensor .
        value(float): The constant value used to initialize the Tensor to be created.
        out(Variable, optional): Optional output which can be any created 
            Variable that meets the requirements to store the result of operation.
            if out is None, a new Varibale will be create to store the result.
        dtype(np.dtype|core.VarDesc.VarType|str, optional): Data type of the output tensor
            which can be float16, float32, float64, int32, int64, if dytpe is `None`, the data
            type of created tensor is `float32`
        device(str, optional): This parameter specifies that the Tensor is created 
            on the GPU or CPU.
        stop_gradient(bool, optional): Indicating if we stop gradient from current(out) Variable,
            default value is True.
        name(str, optional): The default value is None.  Normally there is no need for user to set this
            property.  For more information, please refer to :ref:`api_guide_Name`.
    
    Examples:
        .. code-block:: python

          import paddle.tensor as tensor
          import paddle.fluid as fluid

          data1 = tensor.full(shape=[2,1], full_value=0, dtype='int64') # data1=[[0],[0]]
          data2 = tensor.full(shape=[2,1], full_value=5, dtype='int64', device='gpu') # data2=[[5],[5]]

          # attr shape is a list which contains Variable Tensor.
          positive_2 = fluid.layers.fill_constant([1], "int32", 2)
          data3 = tensor.full(shape=[1, positive_2], dtype='float32', full_value=1.5) # data3=[1.5, 1.5]

          # attr shape is an Variable Tensor.
          shape = fluid.layers.fill_constant([1,2], "int32", 2) # shape=[2,2]
          data4 = tensor.full(shape=shape, dtype='bool', full_value=True) # data4=[[True,True],[True,True]]
    """

    helper = LayerHelper("full", **locals())

    if dtype is None:
        dtype = 'float32'

    check_dtype(dtype, 'create data type',
                ['bool', 'float16', 'float32', 'float64', 'int32', 'int64'],
                'full')
    check_type(shape, 'shape', (Variable, list, tuple), 'full')

    if out is None:
        out = helper.create_variable_for_type_inference(dtype=dtype)

    out.stop_gradient = stop_gradient

    with device_guard(device):
        out = fill_constant(shape=shape, dtype=dtype, value=fill_value, out=out)
    return out


def arange(start, end, step, dtype):
    """
    Return evenly spaced values within a given interval.

    Values are generated within the half-open interval [start, stop) (in other words,
    the interval including start but excluding stop).

    Parameters:
        start(float32 | float64 | int32 | int64 | Variable): Start of interval. The interval includes this value.
            when start is Variable, it is a 1-D Tensor with shape [1].
        end(float32 | float64 | int32 | int64 | Variable): End of interval. The interval does not include this
                                 value, except in some cases where step is not an integer
                                 and floating point round-off affects the length of out. When end is Variable,
                                 it is a 1-D Tensor with shape [1].
        step(float32 | float64 | int32 | int64 | Variable): Spacing between values. For any output out, this is the
                                  distance between two adjacent values, out[i+1] - out[i].
        dtype(str|core.VarDesc.VarType): the data type of the output tensor, can be float32, float64, int32, int64.

    Returns: a 1-D Tensor which is evenly spaced values within a given interval. Its data type is set by dtype.
    
    Return type: Variable

    examples:

        .. code-block:: python

             import paddle
             data = paddle.arange(0, 10, 2, 'int32')

    """
    helper = LayerHelper("range", **locals())

    check_dtype(dtype, 'create data type',
                ['float32', 'float64', 'int32', 'int64'], 'range')

    dtype = convert_dtype(dtype)
    if not isinstance(start, Variable):
        start = fill_constant([1], dtype, start)
    elif convert_dtype(start.dtype) != dtype:
        # make sure that start, end, step has the same dtype as
        # `dtype`
        start = cast(x=start, dtype=dtype)

    if not isinstance(end, Variable):
        end = fill_constant([1], dtype, end)
    elif convert_dtype(end.dtype) != dtype:
        end = cast(x=end, dtype=dtype)

    return out


def arange(start, end, step, dtype):
    """
    Return evenly spaced values within a given interval.

    Values are generated within the half-open interval [start, stop) (in other words,
    the interval including start but excluding stop).

    Parameters:
        start(float32 | float64 | int32 | int64 | Variable): Start of interval. The interval includes this value.
            when start is Variable, it is a 1-D Tensor with shape [1].
        end(float32 | float64 | int32 | int64 | Variable): End of interval. The interval does not include this
                                 value, except in some cases where step is not an integer
                                 and floating point round-off affects the length of out. When end is Variable,
                                 it is a 1-D Tensor with shape [1].
        step(float32 | float64 | int32 | int64 | Variable): Spacing between values. For any output out, this is the
                                  distance between two adjacent values, out[i+1] - out[i].
        dtype(str|core.VarDesc.VarType): the data type of the output tensor, can be float32, float64, int32, int64.

    Returns: a 1-D Tensor which is evenly spaced values within a given interval. Its data type is set by dtype.
    
    Return type: Variable

    examples:

        .. code-block:: python

             import paddle
             data = paddle.arange(0, 10, 2, 'int32')

    """
    helper = LayerHelper("range", **locals())

    check_dtype(dtype, 'create data type',
                ['float32', 'float64', 'int32', 'int64'], 'range')

    dtype = convert_dtype(dtype)
    if not isinstance(start, Variable):
        start = fill_constant([1], dtype, start)
    elif convert_dtype(start.dtype) != dtype:
        # make sure that start, end, step has the same dtype as
        # `dtype`
        start = cast(x=start, dtype=dtype)

    if not isinstance(end, Variable):
        end = fill_constant([1], dtype, end)
    elif convert_dtype(end.dtype) != dtype:
        end = cast(x=end, dtype=dtype)

    if not isinstance(step, Variable):
        step = fill_constant([1], dtype, step)
    elif convert_dtype(step.dtype) != dtype:
        step = cast(x=step, dtype=dtype)

    out = helper.create_variable_for_type_inference(dtype=start.dtype)

    helper.append_op(
        type='range',
        inputs={'Start': start,
                'End': end,
                'Step': step},
        outputs={'Out': [out]})
    out.stop_gradient = True
    return out
