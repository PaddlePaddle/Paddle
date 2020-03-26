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
    #            'arrange',
    'eye'
]

#            'full',
#            'linspace',
#            'full_like',
#            'triu',
#            'tril',
#            'meshgrid'


def eye(num_rows,
        num_columns=None,
        out=None,
        dtype='float32',
        stop_gradient=True,
        name=None):
    """
    **eye**

    This function constructs an identity tensor, or a batch of tensor.

    Args:
        num_rows(int): the number of rows in each batch tensor.
        num_columns(int, optional): the number of columns in each batch tensor.
                          If None, default: num_rows.
        out(Variable, optional): Optional output which can be any created 
            Variable that meets the requirements to store the result of operation.
            if out is None, a new Varibale will be create to store the result.
        dtype(string, optional): The data type of the returned tensor.
                       It should be int32, int64, float16, float32, float64.
        stop_gradient(bool, optional): Whether stop calculating gradients. Default:True.
        name(str, optional): The default value is None.  Normally there is no need for 
            user to set this property.  For more information, please refer to :ref:`api_guide_Name`

    Returns:
        Variable: An identity Tensor or LoDTensor of shape [num_rows, num_columns].

    Examples:
        .. code-block:: python

          import paddle
          data = paddle.tensor.eye(3, dtype='int32')
          # [[1, 0, 0]
          #  [0, 1, 0]
          #  [0, 0, 1]]

          data = paddle.tensor.eye(2, 3, dtype='int32')
          # [[1, 0, 0]
          #  [0, 1, 0]]

    """

    helper = LayerHelper("eye", **locals())
    if not isinstance(num_rows, int) or num_rows < 0:
        raise TypeError("num_rows should be a non-negative int")
    if num_columns is not None:
        if not isinstance(num_columns, int) or num_columns < 0:
            raise TypeError("num_columns should be a non-negative int")
    else:
        num_columns = num_rows
    if out is None:
        out = helper.create_variable_for_type_inference(dtype=dtype)
    c_dtype = convert_np_dtype_to_dtype_(dtype)
    helper.append_op(
        type='eye',
        inputs={},
        outputs={'Out': [out]},
        attrs={
            'num_rows': num_rows,
            'num_columns': num_columns,
            'dtype': c_dtype
        },
        stop_gradient=True)
    out.stop_gradient = stop_gradient
    return out
