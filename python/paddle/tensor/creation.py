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
from ..fluid.framework import Variable
from ..fluid.initializer import Constant
from ..fluid.layers import core
from ..fluid.layer_helper import LayerHelper
from ..fluid.data_feeder import check_variable_and_dtype, check_type, check_dtype, convert_dtype
from ..fluid.framework import convert_np_dtype_to_dtype_, in_dygraph_mode, _varbase_creator, device_guard, OpProtoHolder
from ..fluid.layers import fill_constant
from paddle.common_ops_import import *

# TODO: define functions to get create a tensor  
from ..fluid.layers import crop_tensor  #DEFINE_ALIAS
from ..fluid.layers import diag  #DEFINE_ALIAS
from ..fluid.layers import eye  #DEFINE_ALIAS
from ..fluid.layers import fill_constant  #DEFINE_ALIAS

from ..fluid.layers import create_tensor  #DEFINE_ALIAS

__all__ = [
    'create_tensor',
    #       'create_lod_tensor',
    #       'create_random_int_lodtensor',
    'crop_tensor',
    'diag',
    'eye',
    'fill_constant',
    #       'get_tensor_from_selected_rows',
    'linspace',
    'ones',
    'ones_like',
    'zeros',
    'zeros_like',
    'arange',
    'eye',
    'full',
    'full_like',
    'triu',
    'tril',
    'meshgrid'
]


def full_like(input,
              fill_value,
              out=None,
              dtype=None,
              device=None,
              stop_gradient=True,
              name=None):
    """
	:alias_main: paddle.full_like
	:alias: paddle.full_like,paddle.tensor.full_like,paddle.tensor.creation.full_like

    **full_like**
    This function creates a tensor filled with `fill_value` which has identical shape and dtype 
    with `input`.

    Args:
        input(Variable): The input tensor which specifies shape and data type. The data type can be bool, float16, float32, float64, int32, int64.
        fill_value(bool|float|int): The value to fill the tensor with. Default value is 0. Note: this value shouldn't exceed the range of the output data type.
        out(Variable, optional): Optional output which can be any created Variable that meets the requirements to store the result of operation. If out is None, a new Varibale will be create to store the result. Default value is None.
        dtype(np.dtype|core.VarDesc.VarType|str, optional): The data type of output. The default value is None, which means the output data type is the same as input.
        device (string, optional): Which device to run the operator. The :attr:`device` must be None, 'cpu', 'gpu'. If :attr:`device` is None, it will be the device that the user set in the paddle program. Default value is None.
        stop_gradient(bool, optional): Indicating if we stop gradient from current(out) Variable. Default value is True.
        name(str, optional): The default value is None. Normally there is no need for user to set this property. For more information, please refer to :ref:`api_guide_Name`
    
    Returns:
        out(Variable): The Tensor variable storing the output.
    
    Examples:
        .. code-block:: python

          import paddle
          import paddle.fluid as fluid
          import numpy as np
          input = fluid.data(name='input', dtype='float32', shape=[2, 3])
          output = paddle.full_like(input, 2.0)
          exe = fluid.Executor(fluid.CPUPlace())
          exe.run(fluid.default_startup_program())
          img=np.array([[1, 2, 3], [4, 5, 6]]).astype(np.float32)
          res = exe.run(fluid.default_main_program(), feed={'input':img}, fetch_list=[output])
          print(res) # [array([[2., 2., 2.], [2., 2., 2.]], dtype=float32)]
    """
    helper = LayerHelper("full_like", **locals())

    var_dtype = None
    if dtype is None:
        var_dtype = input.dtype
    else:
        check_dtype(
            dtype, 'dtype',
            ['bool', 'float16', 'float32', 'float64', 'int32', 'int64'],
            'full_like')
        var_dtype = convert_np_dtype_to_dtype_(dtype)

    if out is None:
        out = helper.create_variable_for_type_inference(dtype=dtype)

    helper.append_op(
        type='fill_any_like',
        inputs={'X': [input]},
        attrs={'value': fill_value,
               "dtype": var_dtype},
        outputs={'Out': [out]})
    out.stop_gradient = stop_gradient

    return out


def linspace(start, stop, num, dtype, out=None, device=None, name=None):
    """
	:alias_main: paddle.linspace
	:alias: paddle.linspace,paddle.tensor.linspace,paddle.tensor.creation.linspace

    This OP return fixed number of evenly spaced values within a given interval.
    
    **NOTICE**: The output of this OP has no gradient.

    Args:
        start(float|Variable): The input :attr:`start` is start variable of range. It is a float scalar, \
            or a tensor of shape [1] with input data type float32, float64.
        stop(float|Variable): The input :attr:`stop` is start variable of range. It is a float scalar, \
            or a tensor of shape [1] with input data type float32, float64.
        num(int|Variable): The input :attr:`num` is given num of the sequence. It is an int scalar, \
            or a tensor of shape [1] with type int32.
        dtype(string): The data type of output tensor, it could be 'float32' and 'float64'.
        out (Variable, optional): Optional output which can be any created 
            Variable that meets the requirements to store the result of operation.
            if out is None, a new Varibale will be create to store the result. Default: None.
        device (string, optional): Which device to run the operator. The :attr:`device` must be
            None, 'cpu', 'gpu'. If :attr:`device` is None, it will be choose the device that the user set in 
            the paddle program. Default: None.
        name(str, optional): Normally there is no need for user to set this property. 
            For more information, please refer to :ref:`api_guide_Name`.Default: None.

    Returns:
        Variable, the output data type will be float32, float64.: The 1-D tensor with fixed number of evenly spaced values, \
        the data shape of this tensor is :math:`[num]` . If the :attr:`num` is set 1, the output tensor just has \
        the value with input :attr:`start`. 

    Examples:
        .. code-block:: python

             import paddle
             data = paddle.linspace(0, 10, 5, dtype='float32') # [0.0,  2.5,  5.0,  7.5, 10.0]
             data = paddle.linspace(0, 10, 1, dtype='float32') # [0.0]

    """
    helper = LayerHelper("linspace", **locals())

    if not isinstance(start, Variable):
        start = fill_constant([1], dtype, start)
    if not isinstance(stop, Variable):
        stop = fill_constant([1], dtype, stop)
    if not isinstance(num, Variable):
        num = fill_constant([1], 'int32', num)

    if out is None:
        out = helper.create_variable_for_type_inference(dtype=start.dtype)
    else:
        check_dtype(
            out.dtype, out.name,
            convert_dtype(start.dtype), 'linspace',
            "The out data type '%s' in linspace must be the same with '%s' seted by parameter 'dtype'."
            % (out.dtype, dtype))
        if name:
            warning.warn(
                "The output Variable name of the paddle.tensor.linspace operation can only be given by parameter out or name.\
                When parameter out and name are set at the same time, out has a higher priority than name. \
                Finally, the output Variable name is same as the out name %s." %
                out.name,
                category=UserWarning,
                stacklevel=2)

    if device is not None:
        if device not in ['cpu', 'gpu']:
            raise ValueError(
                "The value of 'device' in linspace operation must be cpu or gpu, but received %s."
                % (device))
        else:
            with device_guard(device):
                helper.append_op(
                    type='linspace',
                    inputs={'Start': start,
                            'Stop': stop,
                            'Num': num},
                    outputs={'Out': [out]})
    else:
        helper.append_op(
            type='linspace',
            inputs={'Start': start,
                    'Stop': stop,
                    'Num': num},
            outputs={'Out': [out]})

    return out


def ones(shape, dtype=None, out=None, device=None):
    """
	:alias_main: paddle.ones
	:alias: paddle.ones,paddle.tensor.ones,paddle.tensor.creation.ones

    The OP creates a tensor of specified :attr:`shape` and :attr:`dtype`, and fills it with 1.

    Args:
        shape(tuple|list): Shape of output tensor.
        dtype(np.dtype|core.VarDesc.VarType|str): Data type of output tensor, it supports
            bool, float16, float32, float64, int32 and int64.
        out(Variable, optional): Optional output which can be any created 
            Variable that meets the requirements to store the result of operation.
            if out is None, a new Varibale will be create to store the result.
        device(str, optional): Which device to run the operator. The :attr:`device` must be
            None,'cpu', 'gpu'. If :attr:`device` is None, it will be choose the device that the user set in 
            the paddle program. Default value is False.

    Returns:
        Variable: A tensor of data type :attr:`dtype` with shape :attr:`shape` and all elements set to 1.

    Examples:
        .. code-block:: python

          import paddle
          data = paddle.ones(shape=[3, 2], dtype='float32') # [[1., 1.], [1., 1.], [1., 1.]]
          data = paddle.ones(shape=[2, 2], dtype='float32', device='cpu') # [[1., 1.], [1., 1.]]
    """
    check_dtype(dtype, 'create data type',
                ['bool', 'float16', 'float32', 'float64', 'int32', 'int64'],
                'zeros')

    if device is not None:
        if device not in ['cpu', 'gpu']:
            raise ValueError(
                "The value of 'device' in zeros_op must be cpu or gpu, but received %s."
                % (device))
        with fluid.device_guard(device):
            return fill_constant(value=1.0, shape=shape, dtype=dtype, out=out)
    return fill_constant(value=1.0, shape=shape, dtype=dtype, out=out)


def ones_like(input, dtype=None, device=None, name=None):
    """
	:alias_main: paddle.ones_like
	:alias: paddle.ones_like,paddle.tensor.ones_like,paddle.tensor.creation.ones_like

    This function creates a ones tensor which has identical shape and dtype 
    with `input`.

    Args:
        input(Variable): The input tensor which specifies shape and dtype.The dtype of input can be 
            float32, float64, int32, int64.
        dtype(np.dtype|core.VarDesc.VarType|str, optional): The data type can be set bool, float32, float64, int32, int64. 
            The default value is None, the dtype is the same as input.
        device(str, optional): Which device to run the operator. The :attr:`device` must be
            None, 'cpu', 'gpu'. If :attr:`device` is None, it will be choose the device that the user set in 
            the paddle program. Default value is None.
        name(str, optional): The name of output variable, normally there is no need for user to set this this property. 
            Default value is None, the framework set the name of output variable.  
    Returns:
        out(Variable): The tensor variable storing the output.

    Examples:
        .. code-block:: python

          import paddle
          import paddle.fluid as fluid

          x = fluid.data(name='x', dtype='float32', shape=[3])
          data = paddle.ones_like(x) # data=[1.0, 1.0, 1.0]
          data1 = paddle.ones_like(input=x, device="gpu") data1=[1.0, 1.0. 1.0]

    """

    helper = LayerHelper("zeros_like", **locals())

    attrs = {"value": 1.0}
    var_dtype = None
    if dtype is not None:
        check_dtype(
            dtype, 'create data type',
            ['bool', 'float16', 'float32', 'float64', 'int32', 'int64'],
            'zeros_like')
        var_dtype = convert_np_dtype_to_dtype_(dtype)
        attrs["dtype"] = var_dtype
    else:
        var_dtype = input.dtype

    out = helper.create_variable_for_type_inference(dtype=var_dtype)

    if device is not None:
        if device not in ['cpu', 'gpu']:
            raise ValueError(
                "The value of 'device' in zeros_op must be cpu or gpu, but received %s."
                % (device))
        with fluid.device_guard(device):
            helper.append_op(
                type='fill_any_like',
                inputs={'X': [input]},
                attrs=attrs,
                outputs={'Out': [out]})
            return out
    helper.append_op(
        type='fill_any_like',
        inputs={'X': [input]},
        attrs=attrs,
        outputs={'Out': [out]})
    out.stop_gradient = True
    return out


def zeros(shape, dtype, out=None, device=None):
    """
	:alias_main: paddle.zeros
	:alias: paddle.zeros,paddle.tensor.zeros,paddle.tensor.creation.zeros

    The OP creates a tensor of specified :attr:`shape` and :attr:`dtype`, and fills it with 0.

    Args:
        shape(tuple|list): Shape of output tensor.
        dtype(np.dtype|core.VarDesc.VarType|str): Data type of output tensor, it supports
            bool, float16, float32, float64, int32 and int64.
        out(Variable, optional): Optional output which can be any created 
            Variable that meets the requirements to store the result of operation.
            if out is None, a new Varibale will be create to store the result.
        device(str, optional): Which device to run the operator. The :attr:`device` must be
            None,'cpu', 'gpu'. If :attr:`device` is None, it will be choose the device that the user set in 
            the paddle program. Default value is False.

    Returns:
        Variable: A tensor of data type :attr:`dtype` with shape :attr:`shape` and all elements set to 0.

    Examples:
        .. code-block:: python

          import paddle
          data = paddle.zeros(shape=[3, 2], dtype='float32') # [[0., 0.], [0., 0.], [0., 0.]]
          data = paddle.zeros(shape=[2, 2], dtype='float32', device='cpu') # [[0., 0.], [0., 0.]]
    """
    check_dtype(dtype, 'create data type',
                ['bool', 'float16', 'float32', 'float64', 'int32', 'int64'],
                'zeros')
    if device is not None:
        if device not in ['cpu', 'gpu']:
            raise ValueError(
                "The value of 'device' in zeros_op must be cpu or gpu, but received %s."
                % (device))
        with fluid.device_guard(device):
            return fill_constant(value=0.0, shape=shape, dtype=dtype, out=out)

    return fill_constant(value=0.0, shape=shape, dtype=dtype, out=out)


def zeros_like(input, dtype=None, device=None, name=None):
    """
	:alias_main: paddle.zeros_like
	:alias: paddle.zeros_like,paddle.tensor.zeros_like,paddle.tensor.creation.zeros_like

    This function creates a zeros tensor which has identical shape and dtype 
    with `input`.

    Args:
        input(Variable): The input tensor which specifies shape and dtype.The dtype of input can be 
            bool, float32, float64, int32, int64.
        dtype(np.dtype|core.VarDesc.VarType|str, optional): The data type can be set bool, float32, float64, int32, int64. 
            The default value is None, the dtype is the same as input.
        device(str, optional): Which device to run the operator. The :attr:`device` must be
            None, 'cpu', 'gpu'. If :attr:`device` is None, it will be choose the device that the user set in 
            the paddle program. Default value is None.
        name(str, optional): The name of output variable, normally there is no need for user to set this this property. 
            Default value is None, the framework set the name of output variable.  

    Returns:
        out(Variable): The tensor variable storing the output.

    Examples:
        .. code-block:: python

          import paddle
          import paddle.fluid as fluid

          x = fluid.data(name='x', dtype='float32', shape=[3])
          data = paddle.ones_like(x) # data=[1.0, 1.0, 1.0]
          data1 = paddle.ones_like(input=x, device="gpu") #data1=[1.0, 1.0. 1.0]

    """

    helper = LayerHelper("zeros_like", **locals())

    attrs = {"value": 0.0}
    var_dtype = None
    if dtype is not None:
        check_dtype(dtype, 'create data type',
                    ['bool', 'float32', 'float64', 'int32', 'int64'],
                    'zeros_like')
        var_dtype = convert_np_dtype_to_dtype_(dtype)
        attrs["dtype"] = var_dtype
    else:
        var_dtype = input.dtype

    out = helper.create_variable_for_type_inference(dtype=var_dtype)

    if device is not None:
        if device not in ['cpu', 'gpu']:
            raise ValueError(
                "The value of 'device' in zeros_op must be cpu or gpu, but received %s."
                % (device))
        with fluid.device_guard(device):
            helper.append_op(
                type='fill_any_like',
                inputs={'X': [input]},
                attrs=attrs,
                outputs={'Out': [out]})
            return out
    helper.append_op(
        type='fill_any_like',
        inputs={'X': [input]},
        attrs=attrs,
        outputs={'Out': [out]})
    out.stop_gradient = True
    return out


def eye(num_rows,
        num_columns=None,
        out=None,
        dtype='float32',
        stop_gradient=True,
        name=None):
    """
    **eye**
    This function constructs an identity tensor.

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
	:alias_main: paddle.full
	:alias: paddle.full,paddle.tensor.full,paddle.tensor.creation.full

    This Op return a Tensor with the `fill_value` which size is same as `shape`
    
    Args:
        shape(list|tuple|Variable): Shape of the Tensor to be created.
                The data type is ``int32`` or ``int64`` . If ``shape`` is a list or tuple,
                the elements of it should be integers or Tensors with shape [1].
                If ``shape`` is an Variable, it should be an 1-D Tensor .
        fill_value(bool|float16|float32|float64|int32|int64|Variable): The constant value
            used to initialize the Tensor to be created. If fill_value is an Variable, it must be an 1-D Tensor.
        out(Variable, optional): Optional output which can be any created 
            Variable that meets the requirements to store the result of operation.
            if out is None, a new Varibale will be create to store the result.
        dtype(np.dtype|core.VarDesc.VarType|str, optional): Data type of the output tensor
            which can be float16, float32, float64, int32, int64, if dytpe is `None`, the data
            type of created tensor is `float32`
        device(str, optional): On which device to run this Op. The :attr:`device` must be
            None, 'cpu' or 'gpu'. If :attr:`device` is None, the device that the user set in 
            the paddle program will be chosen. Default value is None.
        stop_gradient(bool, optional): Indicating if we stop gradient from current(out) Variable,
            default value is True.
        name(str, optional): The default value is None.  Normally there is no need for user to set this
            property.  For more information, please refer to :ref:`api_guide_Name`.
    
    Returns:
        Variable: Tensor which is created according to shape and dtype.

    Raises:
        TypeError: The `dtype` must be one of None, bool, float16, float32, float64, int32 and int64.
        TypeError: The `out` must be a Variable.
        TypeError: The `shape` must be one of Variable, list tuple.
    
    Examples:
        .. code-block:: python

          import paddle
          import paddle.fluid as fluid

          data1 = paddle.full(shape=[2,1], fill_value=0, dtype='int64') # data1=[[0],[0]]
          data2 = paddle.full(shape=[2,1], fill_value=5, dtype='int64', device='gpu') # data2=[[5],[5]]

          # attr shape is a list which contains Variable Tensor.
          positive_2 = fluid.layers.fill_constant([1], "int32", 2)
          data3 = paddle.full(shape=[1, positive_2], dtype='float32', fill_value=1.5) # data3=[1.5, 1.5]

          # attr shape is an Variable Tensor.
          shape = fluid.layers.fill_constant([1,2], "int32", 2) # shape=[2,2]
          data4 = paddle.full(shape=shape, dtype='bool', fill_value=True) # data4=[[True,True],[True,True]]
          
          # attr value is an Variable Tensor.
          val = fluid.layers.fill_constant([1], "float32", 2.0) # val=[2.0]
          data5 = paddle.full(shape=[2,1], fill_value=val, dtype='float32') #data5=[[2.0],[2.0]]
    """

    helper = LayerHelper("full", **locals())

    if dtype is None:
        dtype = 'float32'

    check_dtype(dtype, 'create data type',
                ['bool', 'float16', 'float32', 'float64', 'int32', 'int64'],
                'full')
    check_type(shape, 'shape', (Variable, list, tuple), 'full')
    if out is not None:
        check_type(shape, 'out', (Variable), 'full')

    if out is None:
        out = helper.create_variable_for_type_inference(dtype=dtype)

    out.stop_gradient = stop_gradient

    with device_guard(device):
        out = fill_constant(shape=shape, dtype=dtype, value=fill_value, out=out)
    return out


def arange(start, end, step=1, dtype=None, name=None):
    """
	:alias_main: paddle.arange
	:alias: paddle.arange,paddle.tensor.arange,paddle.tensor.creation.arange

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
             # expected out put: [0, 2, 4, 6, 8]
             data = paddle.arange(0, 10, 2, 'int32')

         #dygraph mode
             import paddle
             import paddle.fluid as fluid
             with fluid.dygraph.guard():
                 x = paddle.arange(0, 6, 2) 
                 # x: [0, 2, 4]
                 # x dtype: float32
             
    """
    helper = LayerHelper("range", **locals())

    if dtype is None:
        dtype = 'float32'

    check_dtype(dtype, 'create data type',
                ['float32', 'float64', 'int32', 'int64'], 'range')

    dtype = convert_dtype(dtype)
    if not isinstance(start, Variable):
        start = fill_constant([1], dtype, start)

    if not isinstance(end, Variable):
        end = fill_constant([1], dtype, end)

    if not isinstance(step, Variable):
        step = fill_constant([1], dtype, step)

    out = helper.create_variable_for_type_inference(dtype=start.dtype)

    helper.append_op(
        type='range',
        inputs={'Start': start,
                'End': end,
                'Step': step},
        outputs={'Out': [out]})
    out.stop_gradient = True
    return out


def _tril_triu_op(helper):
    """Base op of tril_op and triu_op
    """
    op_type = helper.layer_type
    x = helper.kwargs.get('input', None)

    assert x is not None, 'x cannot be None in {}'.format(op_type)
    check_variable_and_dtype(x, 'x', ['float32', 'float64', 'int32', 'int64'],
                             op_type)
    if len(x.shape) < 2:
        raise ValueError("input shape in {} must be at least 2-D".format(
            op_type))
    diagonal = helper.kwargs.get('diagonal', 0)
    if not isinstance(diagonal, (int, )):
        raise TypeError("diagonal in {} must be a python Int".format(op_type))
    name = helper.kwargs.get('name', None)

    if name is None:
        out = helper.create_variable_for_type_inference(dtype=x.dtype)
    else:
        out = helper.create_variable(
            name=name, dtype=x.dtype, persistable=False)

    helper.append_op(
        type="tril_triu",
        inputs={"X": x},
        attrs={
            "diagonal": diagonal,
            "lower": True if op_type == 'tril' else False,
        },
        outputs={"Out": out}, )

    return out


def tril(input, diagonal=0, name=None):
    """
	:alias_main: paddle.tril
	:alias: paddle.tril,paddle.tensor.tril,paddle.tensor.creation.tril

    This op returns the lower triangular part of a matrix (2-D tensor) or batch
    of matrices :attr:`input`, the other elements of the result tensor are set 
    to 0. The lower triangular part of the matrix is defined as the elements 
    on and below the diagonal.

    Args:
        input (Variable): The input variable which is a Tensor.
            Support data types: ``float64``, ``float32``, ``int32``, ``int64``.
        diagonal (int, optional): The diagonal to consider, default value is 0.
            If :attr:`diagonal` = 0, all elements on and below the main diagonal are
            retained. A positive value includes just as many diagonals above the main
            diagonal, and similarly a negative value excludes just as many diagonals below
            the main diagonal. The main diagonal are the set of indices
            :math:`\{(i, i)\}` for :math:`i \in [0, \min\{d_{1}, d_{2}\} - 1]` where
            :math:`d_{1}, d_{2}` are the dimensions of the matrix.
        name (str, optional): The default value is None. Normally there is no need for
            user to set this property. For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Variable: Tensor, results of lower triangular operation by the specified diagonal of input tensor,
        it's data type is the same as input's Tensor.

    Raises:
        TypeError: diagonal is not a int type.
        ValueError: dimension of :attr:`input` is less than 2.

    Examples:
        .. code-block:: python

            import numpy as np
            import paddle.tensor as tensor
            import paddle.fluid as fluid

            data = np.arange(1, 13, dtype="int64").reshape(3,-1)
            # array([[ 1,  2,  3,  4],
            #        [ 5,  6,  7,  8],
            #        [ 9, 10, 11, 12]])
            x = fluid.data(shape=(-1, 4), dtype='int64', name='x')
            exe = fluid.Executor(fluid.CPUPlace())

            # example 1, default diagonal
            tril = tensor.tril(x)
            tril_out, = exe.run(fluid.default_main_program(), feed={"x": data},
                fetch_list=[tril], return_numpy=True)
            # array([[ 1,  0,  0,  0],
            #        [ 5,  6,  0,  0],
            #        [ 9, 10, 11,  0]])

            # example 2, positive diagonal value
            tril = tensor.tril(x, diagonal=2)
            tril_out, = exe.run(fluid.default_main_program(), feed={"x": data},
                fetch_list=[tril], return_numpy=True)
            # array([[ 1,  2,  3,  0], 
            #        [ 5,  6,  7,  8],
            #        [ 9, 10, 11, 12]])

            # example 3, negative diagonal value
            tril = tensor.tril(x, diagonal=-1)
            tril_out, = exe.run(fluid.default_main_program(), feed={"x": data},
                fetch_list=[tril], return_numpy=True)
            # array([[ 0,  0,  0,  0],
            #        [ 5,  0,  0,  0],
            #        [ 9, 10,  0,  0]])

    """
    if in_dygraph_mode():
        op = getattr(core.ops, 'tril_triu')
        return op(input, 'diagonal', diagonal, "lower", True)

    return _tril_triu_op(LayerHelper('tril', **locals()))


def triu(input, diagonal=0, name=None):
    """
	:alias_main: paddle.triu
	:alias: paddle.triu,paddle.tensor.triu,paddle.tensor.creation.triu

    This op returns the upper triangular part of a matrix (2-D tensor) or batch of matrices
    :attr:`input`, the other elements of the result tensor are set to 0.
    The upper triangular part of the matrix is defined as the elements on and
    above the diagonal.

    Args:
        input (Variable): The input variable which is a Tensor.
            Support data types: ``float64``, ``float32``, ``int32``, ``int64``.
        diagonal (int, optional): The diagonal to consider, default value is 0.
            If :attr:`diagonal` = 0, all elements on and above the main diagonal are
            retained. A positive value excludes just as many diagonals above the main
            diagonal, and similarly a negative value includes just as many diagonals below
            the main diagonal. The main diagonal are the set of indices
            :math:`\{(i, i)\}` for :math:`i \in [0, \min\{d_{1}, d_{2}\} - 1]` where
            :math:`d_{1}, d_{2}` are the dimensions of the matrix.
        name (str, optional): The default value is None. Normally there is no need for
            user to set this property. For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Variable: Tensor, results of upper triangular operation by the specified diagonal of input tensor,
        it's data type is the same as input's Tensor.

    Raises:
        TypeError: diagonal is not a int type.
        ValueError: dimension of :attr:`input` is less than 2.

    Examples:
        .. code-block:: python

            import numpy as np
            import paddle.fluid as fluid
            import paddle.tensor as tensor

            data = np.arange(1, 13, dtype="int64").reshape(3,-1)
            # array([[ 1,  2,  3,  4],
            #        [ 5,  6,  7,  8],
            #        [ 9, 10, 11, 12]])
            x = fluid.data(shape=(-1, 4), dtype='int64', name='x')
            exe = fluid.Executor(fluid.CPUPlace())

            # example 1, default diagonal
            triu = tensor.triu(x)
            triu_out, = exe.run(fluid.default_main_program(), feed={"x": data},
                fetch_list=[triu], return_numpy=True)
            # array([[ 1,  2,  3,  4],
            #        [ 0,  6,  7,  8],
            #        [ 0,  0, 11, 12]])

            # example 2, positive diagonal value
            triu = tensor.triu(x, diagonal=2)
            triu_out, = exe.run(fluid.default_main_program(), feed={"x": data},
                fetch_list=[triu], return_numpy=True)
            # array([[0, 0, 3, 4],
            #        [0, 0, 0, 8],
            #        [0, 0, 0, 0]])

            # example 3, negative diagonal value
            triu = tensor.triu(x, diagonal=-1)
            triu_out, = exe.run(fluid.default_main_program(), feed={"x": data},
                fetch_list=[triu], return_numpy=True)
            # array([[ 1,  2,  3,  4],
            #        [ 5,  6,  7,  8],
            #        [ 0, 10, 11, 12]])

    """
    if in_dygraph_mode():
        op = getattr(core.ops, 'tril_triu')
        return op(input, 'diagonal', diagonal, "lower", False)

    return _tril_triu_op(LayerHelper('triu', **locals()))


def meshgrid(input, name=None):
    """
	:alias_main: paddle.meshgrid
	:alias: paddle.meshgrid,paddle.tensor.meshgrid,paddle.tensor.creation.meshgrid

    This op takes a list of N tensors as input, each of which is 1-dimensional 
    vector, and creates N-dimensional grids.
    
    Args:
        input(Variable) : tensors (list of tensor): the shapes of input k tensors are (N1,), 
            (N2,),..., (Nk,). Support data types: ``float64``, ``float32``, ``int32``, ``int64``.
        name (str, optional): The default value is None. Normally there is no need for
            user to set this property. For more information, please refer to :ref:`api_guide_Name`.
 
    Returns:
         Variable: k tensors. The shape of each tensor is (N1, N2, ..., Nk)

    Examples:
      .. code-block:: python

          import paddle
          import paddle.fluid as fluid
          import numpy as np

          x = fluid.data(name='x', shape=[100], dtype='int32')
          y = fluid.data(name='y', shape=[200], dtype='int32')

          input_1 = np.random.randint(0, 100, [100, ]).astype('int32')
          input_2 = np.random.randint(0, 100, [200, ]).astype('int32')

          exe = fluid.Executor(place=fluid.CPUPlace())
          grid_x, grid_y = paddle.tensor.meshgrid([x, y])
          res_1, res_2 = exe.run(fluid.default_main_program(),
                                 feed={'x': input_1,
                                       'y': input_2},
                                 fetch_list=[grid_x, grid_y])
     
          #the shape of res_1 is (100, 200)
          #the shape of res_2 is (100, 200)

      .. code-block:: python

          #example 2: in dygraph mode

          import paddle
          import paddle.fluid as fluid
          import numpy as np

          input_3 = np.random.randint(0, 100, [100, ]).astype('int32')
          input_4 = np.random.randint(0, 100, [200, ]).astype('int32')
          with fluid.dygraph.guard():
              tensor_3 = fluid.dygraph.to_variable(input_3)
              tensor_4 = fluid.dygraph.to_variable(input_4)
              grid_x, grid_y = paddle.tensor.meshgrid([tensor_3, tensor_4])

          #the shape of grid_x is (100, 200)
          #the shape of grid_y is (100, 200)

    """

    if in_dygraph_mode():
        num = len(input)
        out = core.ops.meshgrid(input, num)
        return out

    helper = LayerHelper('meshgrid', **locals())

    if not isinstance(input, list):
        raise TypeError("The type of input in meshgrid should be list.")

    for id, input_ in enumerate(input):
        check_dtype(input_.dtype, 'create data type',
                    ['float16', 'float32', 'float64', 'int32', 'int64'],
                    'meshgrid')

    num = len(input)
    out = [
        helper.create_variable_for_type_inference(dtype=input[i].dtype)
        for i in range(num)
    ]
    helper.append_op(type='meshgrid', inputs={'X': input}, outputs={'Out': out})

    return out
