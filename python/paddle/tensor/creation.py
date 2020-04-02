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

# TODO: define functions to get create a tensor  

from __future__ import print_function
from ..fluid.framework import Variable
from ..fluid.initializer import Constant
from ..fluid.layer_helper import LayerHelper
from ..fluid.data_feeder import check_variable_and_dtype, check_type, check_dtype, convert_dtype
from ..fluid.framework import convert_np_dtype_to_dtype_, in_dygraph_mode, _varbase_creator, device_guard
from ..fluid.layers import fill_constant

__all__ = [
    'create_tensor',
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
    'eye',
    'full',
    #            'linspace',
    #            'full_like',
    #            'triu',
    #            'tril',
    #            'meshgrid'
]


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
          data = paddle.eye(3, dtype='int32')
          # [[1, 0, 0]
          #  [0, 1, 0]
          #  [0, 0, 1]]

          data = paddle.eye(2, 3, dtype='int32')
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
