#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
import math
from paddle.common_ops_import import fill_constant
from ..fluid.layers import utils
from ..static import Variable, device_guard
from ..framework import _current_expected_place, _get_paddle_place
from ..framework import dygraph_only
from ..framework import core
from ..framework import in_dygraph_mode, _non_static_mode
from ..framework import LayerHelper
from ..fluid.data_feeder import check_variable_and_dtype, check_type, check_dtype, convert_dtype
from ..framework import convert_np_dtype_to_dtype_, _varbase_creator, OpProtoHolder
# TODO: define functions to get create a tensor  
import paddle
from paddle import _C_ops
from ..fluid.framework import _in_legacy_dygraph, _in_eager_without_dygraph_check
import warnings

__all__ = []


def _complex_to_real_dtype(dtype):
    if dtype == core.VarDesc.VarType.COMPLEX64:
        return core.VarDesc.VarType.FP32
    elif dtype == core.VarDesc.VarType.COMPLEX128:
        return core.VarDesc.VarType.FP64
    else:
        return dtype


def _real_to_complex_dtype(dtype):
    if dtype == core.VarDesc.VarType.FP32:
        return core.VarDesc.VarType.COMPLEX64
    elif dtype == core.VarDesc.VarType.FP64:
        return core.VarDesc.VarType.COMPLEX128
    else:
        return dtype


def linspace(start, stop, num, dtype=None, name=None):
    r"""
    This OP return fixed number of evenly spaced values within a given interval.

    Args:
        start(int|float|Tensor): The input :attr:`start` is start variable of range. It is a scalar, \
            or a Tensor of shape [1] with input data type int32, int64, float32 or float64.
        stop(int|float|Tensor): The input :attr:`stop` is start variable of range. It is a scalar, \
            or a Tensor of shape [1] with input data type int32, int64, float32 or float64.
        num(int|Tensor): The input :attr:`num` is given num of the sequence. It is an int scalar, \
            or a Tensor of shape [1] with data type int32.
        dtype(np.dtype|str, optional): The data type of output tensor, it could be
            int32, int64, float32 and float64. Default: if None, the data type is float32.
        name(str, optional): Normally there is no need for user to set this property. 
            For more information, please refer to :ref:`api_guide_Name`.Default: None.

    Returns:
        Tensor: the output data type will be float32, float64. The 1-D tensor with fixed number of evenly spaced values, \
        the data shape of this tensor is :math:`[num]` . If the :attr:`num` is set 1, the output tensor just has \
        the value with input :attr:`start`. 

    Examples:
        .. code-block:: python

             import paddle
             data = paddle.linspace(0, 10, 5, 'float32') # [0.0,  2.5,  5.0,  7.5, 10.0]
             data = paddle.linspace(0, 10, 1, 'float32') # [0.0]

    """
    if dtype is None:
        dtype = 'float32'
    tensor_num = num
    tensor_start = start
    tensor_stop = stop
    if not isinstance(num, Variable):
        check_type(num, 'num', (int), 'linspace')
    if not isinstance(dtype, core.VarDesc.VarType):
        dtype = convert_np_dtype_to_dtype_(dtype)
    if not isinstance(start, Variable):
        with device_guard("cpu"):
            tensor_start = fill_constant([1], dtype, start)
    if not isinstance(stop, Variable):
        with device_guard("cpu"):
            tensor_stop = fill_constant([1], dtype, stop)
    if not isinstance(num, Variable):
        with device_guard("cpu"):
            tensor_num = fill_constant([1], 'int32', num)
    if _non_static_mode():
        return _C_ops.linspace(tensor_start, tensor_stop, tensor_num, 'dtype',
                               dtype)

    helper = LayerHelper("linspace", **locals())

    start_dtype = convert_dtype(tensor_start.dtype)
    stop_dtype = convert_dtype(tensor_stop.dtype)
    out_dtype = convert_dtype(dtype)
    if isinstance(start, Variable):
        check_dtype(start.dtype, 'start',
                    ['float32', 'float64', 'int32', 'int64'], 'linspace')
    else:
        check_type(start, 'start', (int, float), 'linspace')

    if isinstance(stop, Variable):
        check_dtype(stop.dtype, 'stop',
                    ['float32', 'float64', 'int32', 'int64'], 'linspace')
    else:
        check_type(stop, 'stop', (int, float), 'linspace')
    if isinstance(num, Variable):
        check_dtype(num.dtype, 'num', ['int32'], 'linspace')
    check_dtype(dtype, 'dtype', ['int32', 'int64', 'float32', 'float64'],
                'linspace')
    if ((stop_dtype == "float64" or start_dtype == "float64") and
            out_dtype in ["float32", "int32"]) or ((stop_dtype == "int64" or
                                                    start_dtype == "int64") and
                                                   out_dtype == "int32"):
        raise ValueError(
            "The dtype of start/stop is {}/{} but the attr(dtype) of linspace is {}, "
            "which may cause data type overflows. Please reset attr(dtype) of linspace."
            .format(start_dtype, stop_dtype, dtype))

    out = helper.create_variable_for_type_inference(dtype=dtype)

    helper.append_op(
        type='linspace',
        inputs={'Start': tensor_start,
                'Stop': tensor_stop,
                'Num': tensor_num},
        attrs={'dtype': dtype},
        outputs={'Out': [out]})
    if isinstance(num, int):
        out.desc.set_shape((num, ))
    return out


def logspace(start, stop, num, base=10.0, dtype=None, name=None):
    r"""
    Return fixed number of logarithmical-evenly spaced values within the interval \
    :math:`[base^{start}, base^{stop}]`.
    
    Notes:
        This API does not compute the gradient.
    
    Args:
        start(int|float|Tensor): The input :attr:`start` is exponent of first entry in \
            the sequence. It is a scalar, or a Tensor of shape [1] with input data \
            type int32, int64, float32 or float64.
        stop(int|float|Tensor): The input :attr:`stop` is exponent of last entry in the \
            sequence. It is a scalar, or a Tensor of shape [1] with input data \
            type int32, int64, float32 or float64.
        num(int|Tensor): The input :attr:`num` is given number of items in the sequence. \
            It is an int scalar, or a Tensor of shape [1] with data type int32.
        base(int|float|Tensor): The input :attr:`base` is base of the logarithm function. \
            It is a scalar, or a Tensor of shape [1] with input data type int32, int64, \
            float32 or float64.
        dtype(np.dtype|str, optional): The data type of output tensor, it could be \
            int32, int64, float32 or float64. Default: if None, the data type is float32. \
        name(str, optional): Normally there is no need for user to set this property. \
            For more information, please refer to :ref:`api_guide_Name`. Default: None.

    Returns:
        Tensor: The output data type will be float32, float64. The 1-D tensor with \
        fixed number of logarithmical-evenly spaced values, the data shape of this \
        tensor is :math:`[num]`. If the :attr:`num` is set 1, the output tensor \
        just has the value with exponential of :attr:`start` with base :attr:`base`. 

    Examples:
        .. code-block:: python
            :name: logspace-example

            import paddle
            data = paddle.logspace(0, 10, 5, 2, 'float32')
            # [1.          , 5.65685415  , 32.         , 181.01933289, 1024.       ]
            data = paddle.logspace(0, 10, 1, 2, 'float32')
            # [1.]
    """
    if dtype is None:
        dtype = 'float32'
    tensor_num = num
    tensor_start = start
    tensor_stop = stop
    tensor_base = base
    if not isinstance(num, Variable):
        check_type(num, 'num', (int), 'logspace')
    if not isinstance(dtype, core.VarDesc.VarType):
        dtype = convert_np_dtype_to_dtype_(dtype)
    if not isinstance(start, Variable):
        with device_guard("cpu"):
            tensor_start = fill_constant([1], dtype, start)
    if not isinstance(stop, Variable):
        with device_guard("cpu"):
            tensor_stop = fill_constant([1], dtype, stop)
    if not isinstance(num, Variable):
        with device_guard("cpu"):
            tensor_num = fill_constant([1], 'int32', num)
    if not isinstance(base, Variable):
        with device_guard("cpu"):
            tensor_base = fill_constant([1], dtype, base)
    if _non_static_mode():
        return _C_ops.logspace(tensor_start, tensor_stop, tensor_num,
                               tensor_base, 'dtype', dtype)

    helper = LayerHelper("logspace", **locals())

    start_dtype = convert_dtype(tensor_start.dtype)
    stop_dtype = convert_dtype(tensor_stop.dtype)
    base_dtype = convert_dtype(tensor_base.dtype)
    out_dtype = convert_dtype(dtype)
    if isinstance(start, Variable):
        check_dtype(start.dtype, 'start',
                    ['float32', 'float64', 'int32', 'int64'], 'logspace')
    else:
        check_type(start, 'start', (int, float), 'logspace')

    if isinstance(stop, Variable):
        check_dtype(stop.dtype, 'stop',
                    ['float32', 'float64', 'int32', 'int64'], 'logspace')
    else:
        check_type(stop, 'stop', (int, float), 'logspace')

    if isinstance(num, Variable):
        check_dtype(num.dtype, 'num', ['int32'], 'logspace')

    if isinstance(base, Variable):
        check_dtype(base.dtype, 'base',
                    ['float32', 'float64', 'int32', 'int64'], 'logspace')
    else:
        check_type(base, 'base', (int, float), 'logspace')

    check_dtype(dtype, 'dtype', ['int32', 'int64', 'float32', 'float64'],
                'logspace')
    if ((stop_dtype == "float64" or start_dtype == "float64"
                                 or base_dtype == "float64")
                                 and out_dtype in ["float32", "int32"]) or \
       ((stop_dtype == "int64" or start_dtype == "int64"
                               or base_dtype == "int64")
                               and out_dtype == "int32"):
        raise ValueError(
            "The dtype of start/stop/base is {}/{}/{} but the attr(dtype) of logspace is {}, "
            "which may cause data type overflows. Please reset attr(dtype) of logspace."
            .format(start_dtype, stop_dtype, base_dtype, dtype))

    out = helper.create_variable_for_type_inference(dtype=dtype)

    helper.append_op(
        type='logspace',
        inputs={
            'Start': tensor_start,
            'Stop': tensor_stop,
            'Num': tensor_num,
            'Base': tensor_base
        },
        attrs={'dtype': dtype},
        outputs={'Out': [out]})
    if isinstance(num, int):
        out.desc.set_shape((num, ))
    return out


@dygraph_only
def to_tensor(data, dtype=None, place=None, stop_gradient=True):
    r"""
    Constructs a ``paddle.Tensor`` from ``data`` , 
    which can be scalar, tuple, list, numpy\.ndarray, paddle\.Tensor.

    If the ``data`` is already a Tensor, copy will be performed and return a new tensor.
    If you only want to change stop_gradient property, please call ``Tensor.stop_gradient = stop_gradient`` directly.

    Args:
        data(scalar|tuple|list|ndarray|Tensor): Initial data for the tensor.
            Can be a scalar, list, tuple, numpy\.ndarray, paddle\.Tensor.
        dtype(str|np.dtype, optional): The desired data type of returned tensor. Can be 'bool' , 'float16' , 
            'float32' , 'float64' , 'int8' , 'int16' , 'int32' , 'int64' , 'uint8',
            'complex64' , 'complex128'. Default: None, infers dtype from ``data`` 
            except for python float number which gets dtype from ``get_default_type`` .
        place(CPUPlace|CUDAPinnedPlace|CUDAPlace|str, optional): The place to allocate Tensor. Can be  
            CPUPlace, CUDAPinnedPlace, CUDAPlace. Default: None, means global place. If ``place`` is 
            string, It can be ``cpu``, ``gpu:x`` and ``gpu_pinned``, where ``x`` is the index of the GPUs. 
        stop_gradient(bool, optional): Whether to block the gradient propagation of Autograd. Default: True.

    Returns:
        Tensor: A Tensor constructed from ``data`` .

    Raises:
        TypeError: If the data type of ``data`` is not scalar, list, tuple, np.ndarray, paddle.Tensor
        ValueError: If ``data`` is tuple|list, it can't contain nested tuple|list with different lengths , such as: [[1, 2], [3, 4, 5]]
        TypeError: If ``dtype`` is not bool, float16, float32, float64, int8, int16, int32, int64, uint8, complex64, complex128
        ValueError: If ``place`` is not paddle.CPUPlace, paddle.CUDAPinnedPlace, paddle.CUDAPlace or specified pattern string. 

    Examples:

    .. code-block:: python

        import paddle
                
        type(paddle.to_tensor(1))
        # <class 'paddle.Tensor'>

        paddle.to_tensor(1)
        # Tensor(shape=[1], dtype=int64, place=CPUPlace, stop_gradient=True,
        #        [1])

        x = paddle.to_tensor(1, stop_gradient=False)
        print(x)
        # Tensor(shape=[1], dtype=int64, place=CPUPlace, stop_gradient=False,
        #        [1])

        paddle.to_tensor(x)  # A new tensor will be created with default stop_gradient=True
        # Tensor(shape=[1], dtype=int64, place=CPUPlace, stop_gradient=True,
        #        [1])        

        paddle.to_tensor([[0.1, 0.2], [0.3, 0.4]], place=paddle.CPUPlace(), stop_gradient=False)
        # Tensor(shape=[2, 2], dtype=float32, place=CPUPlace, stop_gradient=False,
        #        [[0.10000000, 0.20000000],
        #         [0.30000001, 0.40000001]])

        type(paddle.to_tensor([[1+1j, 2], [3+2j, 4]], dtype='complex64'))
        # <class 'paddle.Tensor'>

        paddle.to_tensor([[1+1j, 2], [3+2j, 4]], dtype='complex64')
        # Tensor(shape=[2, 2], dtype=complex64, place=CPUPlace, stop_gradient=True,
        #        [[(1+1j), (2+0j)],
        #         [(3+2j), (4+0j)]])
    """
    place = _get_paddle_place(place)
    if place is None:
        place = _current_expected_place()
    elif not isinstance(place, (core.Place, core.CPUPlace, core.CUDAPinnedPlace,
                                core.CUDAPlace, core.NPUPlace, core.XPUPlace,
                                core.MLUPlace, core.CustomPlace)):
        raise ValueError(
            "'place' must be any of paddle.Place, paddle.CPUPlace, paddle.CUDAPinnedPlace, paddle.CUDAPlace, paddle.NPUPlace, paddle.XPUPlace, paddle.MLUPlace, paddle.CustomPlace"
        )

    if not isinstance(data, np.ndarray):

        def _handle_dtype(data, dtype):
            if dtype:
                if convert_dtype(dtype) != convert_dtype(data.dtype):
                    return data.astype(convert_dtype(dtype))
            return data

        if np.isscalar(data) and not isinstance(data, str):
            data = np.array([data])
        elif isinstance(data, (list, tuple)):
            data = np.array(data)
            if data.dtype == np.object:
                raise ValueError(
                    "\n\tFaild to convert input data to a regular ndarray :\n\t - Usually "
                    "this means the input data contains nested lists with different lengths. "
                )
        elif isinstance(data, paddle.Tensor) and not in_dygraph_mode():
            data = data._copy_to(place, False)
            data = _handle_dtype(data, dtype)
            data.stop_gradient = stop_gradient
            return data
        elif isinstance(data, core.eager.Tensor) and in_dygraph_mode():
            data = data._copy_to(place, False)
            data = _handle_dtype(data, dtype)
            data.stop_gradient = stop_gradient
            return data
        elif isinstance(data, (core.LoDTensor, core.Tensor)):
            # should't expose it to users, just for internal use.
            # convert core.Tensor/core.LoDTensor to VarBase first
            # Currenly, there is no copy when places are same
            if in_dygraph_mode():
                data = core.eager.Tensor(data)
            else:
                data = paddle.Tensor(data)
            if not data.place._equals(place):
                data = data._copy_to(place, False)
            data = _handle_dtype(data, dtype)
            data.stop_gradient = stop_gradient
            return data
        else:
            raise TypeError(
                "Can't constructs a 'paddle.Tensor' with data type {}, data type must be scalar|list|tuple|np.ndarray|paddle.Tensor".
                format(type(data)))
        if not dtype:
            if data.dtype in [
                    'float16', 'float32', 'float64', 'complex64', 'complex128'
            ]:
                default_type = paddle.get_default_dtype()
                if np.iscomplexobj(data):
                    default_type = 'complex64' if default_type in [
                        'float16', 'float32'
                    ] else 'complex128'
                data = data.astype(default_type)
            # Windows default type is 'int32', while Linux/Mac is 'int64'. Unify they.
            if data.dtype in ['int32']:
                default_type = "int64"
                data = data.astype(default_type)

    if dtype and convert_dtype(dtype) != data.dtype:
        data = data.astype(convert_dtype(dtype))

    if _in_eager_without_dygraph_check() and isinstance(data, np.ndarray):
        return core.eager.Tensor(
            value=data,
            place=place,
            persistable=False,
            zero_copy=False,
            name=None,
            stop_gradient=stop_gradient)
    else:
        return paddle.Tensor(
            value=data,
            place=place,
            persistable=False,
            zero_copy=False,
            stop_gradient=stop_gradient)


def full_like(x, fill_value, dtype=None, name=None):
    """

    This function creates a tensor filled with ``fill_value`` which has identical shape of ``x`` and ``dtype``.
    If the ``dtype`` is None, the data type of Tensor is same with ``x``.

    Args:
        x(Tensor): The input tensor which specifies shape and data type. The data type can be bool, float16, float32, float64, int32, int64.
        fill_value(bool|float|int): The value to fill the tensor with. Note: this value shouldn't exceed the range of the output data type.
        dtype(np.dtype|str, optional): The data type of output. The data type can be one
            of bool, float16, float32, float64, int32, int64. The default value is None, which means the output 
            data type is the same as input.
        name(str, optional): The default value is None. Normally there is no need for user to set this property. For more information, please refer to :ref:`api_guide_Name`
    
    Returns:
        Tensor: Tensor which is created according to ``x``, ``fill_value`` and ``dtype``.
    
    Examples:
        .. code-block:: python

          import paddle
          import numpy as np
          
          input = paddle.full(shape=[2, 3], fill_value=0.0, dtype='float32', name='input')
          output = paddle.full_like(input, 2.0)
          # [[2. 2. 2.]
          #  [2. 2. 2.]]
    """

    if dtype is None:
        dtype = x.dtype
    else:
        if not isinstance(dtype, core.VarDesc.VarType):
            dtype = convert_np_dtype_to_dtype_(dtype)

    if in_dygraph_mode():
        return _C_ops.final_state_full_like(x, fill_value, dtype, x.place)

    if _in_legacy_dygraph():
        return _C_ops.fill_any_like(x, 'value', fill_value, 'dtype', dtype)

    helper = LayerHelper("full_like", **locals())
    check_variable_and_dtype(
        x, 'x',
        ['bool', 'float16', 'float32', 'float64', 'int16', 'int32', 'int64'],
        'full_like')
    check_dtype(
        dtype, 'dtype',
        ['bool', 'float16', 'float32', 'float64', 'int16', 'int32', 'int64'],
        'full_like/zeros_like/ones_like')
    out = helper.create_variable_for_type_inference(dtype=dtype)

    helper.append_op(
        type='fill_any_like',
        inputs={'X': [x]},
        attrs={'value': fill_value,
               "dtype": dtype},
        outputs={'Out': [out]})
    out.stop_gradient = True
    return out


def ones(shape, dtype=None, name=None):
    """

    The OP creates a tensor of specified :attr:`shape` and :attr:`dtype`, and fills it with 1.

    Args:
        shape(tuple|list|Tensor): Shape of the Tensor to be created, the data type of shape is int32 or int64.
        dtype(np.dtype|str, optional): Data type of output Tensor, it supports
            bool, float16, float32, float64, int32 and int64. Default: if None, the data type is 'float32'.
        name(str, optional): The default value is None. Normally there is no need for user to set this property. For more information, please refer to :ref:`api_guide_Name`
    
    Returns:
        Tensor: A tensor of data type :attr:`dtype` with shape :attr:`shape` and all elements set to 1.

    Examples:
        .. code-block:: python

          import paddle 
          
          # default dtype for ones OP
          data1 = paddle.ones(shape=[3, 2]) 
          # [[1. 1.]
          #  [1. 1.]
          #  [1. 1.]]
          
          data2 = paddle.ones(shape=[2, 2], dtype='int32') 
          # [[1 1]
          #  [1 1]]
          
          # shape is a Tensor
          shape = paddle.full(shape=[2], dtype='int32', fill_value=2)
          data3 = paddle.ones(shape=shape, dtype='int32') 
          # [[1 1]
          #  [1 1]]
    """
    if dtype is None:
        dtype = 'float32'
    return fill_constant(value=1.0, shape=shape, dtype=dtype, name=name)


def ones_like(x, dtype=None, name=None):
    """
    This OP returns a Tensor filled with the value 1, with the same shape and
    data type (use ``dtype`` if ``dtype`` is not None) as ``x``.

    Args:
        x(Tensor): The input tensor which specifies shape and dtype. The
            dtype of ``x`` can be bool, float16, float32, float64, int32, int64.
        dtype(str|np.dtype, optional): The data type of the
            output tensor. Supported data types: bool, float16, float32, float64,
            int32, int64. If ``dtype`` is None, the data type is the same as ``x``.
            Default is None.
        name(str, optional): The default value is None. Normally there is no
            need for user to set this property. For more information, please
            refer to :ref:`api_guide_Name`.

    Returns:
        Tensor: A Tensor filled with the value 1, with the same shape and
        data type (use ``dtype`` if ``dtype`` is not None) as ``x``.

    Raise:
        TypeError: If ``dtype`` is not None and is not bool, float16, float32,
        float64, int32 or int64.

    Examples:
        .. code-block:: python

            import paddle

            x = paddle.to_tensor([1,2,3])
            out1 = paddle.ones_like(x) # [1., 1., 1.]
            out2 = paddle.ones_like(x, dtype='int32') # [1, 1, 1]

    """
    return full_like(x=x, fill_value=1, dtype=dtype, name=name)


def zeros(shape, dtype=None, name=None):
    """
    The OP creates a tensor of specified :attr:`shape` and :attr:`dtype`, and fills it with 0.

    Args:
        shape(tuple|list|Tensor): Shape of the Tensor to be created, the data type of ``shape`` is int32 or int64.
        dtype(np.dtype|str, optional): Data type of output Tensor, it supports
            bool, float16, float32, float64, int32 and int64. Default: if None, the date type is float32.
        name(str, optional): The default value is None.  Normally there is no need for user to set this
            property.  For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor: A tensor of data type :attr:`dtype` with shape :attr:`shape` and all elements set to 0.

    Examples:
        .. code-block:: python

          import paddle
          
          data = paddle.zeros(shape=[3, 2], dtype='float32') 
          # [[0. 0.]
          #  [0. 0.]
          #  [0. 0.]]
          data = paddle.zeros(shape=[2, 2]) 
          # [[0. 0.]
          #  [0. 0.]]
          
          # shape is a Tensor
          shape = paddle.full(shape=[2], dtype='int32', fill_value=2)
          data3 = paddle.zeros(shape=shape, dtype='int32') 
          # [[0 0]
          #  [0 0]]
    """
    if dtype is None:
        dtype = 'float32'
    return fill_constant(value=0.0, shape=shape, dtype=dtype, name=name)


def zeros_like(x, dtype=None, name=None):
    """
    This OP returns a Tensor filled with the value 0, with the same shape and
    data type (use ``dtype`` if ``dtype`` is not None) as ``x``.

    Args:
        x(Tensor): The input tensor which specifies shape and dtype. The
            dtype of ``x`` can be bool, float16, float32, float64, int32, int64.
        dtype(str|np.dtype, optional): The data type of the
            output tensor. Supported data types: bool, float16, float32, float64,
            int32, int64. If ``dtype`` is None, the data type is the same as ``x``.
            Default is None.
        name(str, optional): The default value is None. Normally there is no
            need for user to set this property. For more information, please
            refer to :ref:`api_guide_Name`.

    Returns:
        Tensor: A Tensor filled with the value 0, with the same shape and
        data type (use ``dtype`` if ``dtype`` is not None) as ``x``.

    Raise:
        TypeError: If ``dtype`` is not None and is not bool, float16, float32,
        float64, int32 or int64.

    Examples:
        .. code-block:: python

            import paddle

            x = paddle.to_tensor([1, 2, 3])
            out1 = paddle.zeros_like(x) # [0., 0., 0.]
            out2 = paddle.zeros_like(x, dtype='int32') # [0, 0, 0]

    """
    return full_like(x=x, fill_value=0, dtype=dtype, name=name)


def eye(num_rows, num_columns=None, dtype=None, name=None):
    """
    
    This function constructs 2-D Tensor with ones on the diagonal and zeros elsewhere.

    Args:
        num_rows(int): the number of rows in each batch Tensor.
        num_columns(int, optional): the number of columns in each batch Tensor.
            If None, default: num_rows.
        dtype(np.dtype|str, optional): The data type of the returned Tensor.
            It should be int32, int64, float16, float32, float64. Default: if None, the data type
            is float32.
        name(str, optional): The default value is None.  Normally there is no need for 
            user to set this property.  For more information, please refer to :ref:`api_guide_Name`

    Returns:
        Tensor: An identity Tensor or LoDTensor of shape [num_rows, num_columns].

    Examples:
        .. code-block:: python
          
          import paddle

          data = paddle.eye(3, dtype='int32')
          # [[1 0 0]
          #  [0 1 0]
          #  [0 0 1]]
          data = paddle.eye(2, 3, dtype='int32')
          # [[1 0 0]
          #  [0 1 0]]
    """

    if dtype is None:
        dtype = 'float32'
    if num_columns is None:
        num_columns = num_rows

    if not isinstance(dtype, core.VarDesc.VarType):
        dtype = convert_np_dtype_to_dtype_(dtype)
    if num_columns is not None:
        if not isinstance(num_columns, int) or num_columns < 0:
            raise TypeError("num_columns should be a non-negative int")
    else:
        num_columns = num_rows

    if _non_static_mode():
        out = _C_ops.eye('dtype', dtype, 'num_rows', num_rows, 'num_columns',
                         num_columns)

    else:
        helper = LayerHelper("eye", **locals())
        check_dtype(dtype, 'dtype',
                    ['float16', 'float32', 'float64', 'int32', 'int64'], 'eye')
        if not isinstance(num_rows, int) or num_rows < 0:
            raise TypeError("num_rows should be a non-negative int")
        out = helper.create_variable_for_type_inference(dtype=dtype)
        helper.append_op(
            type='eye',
            inputs={},
            outputs={'Out': [out]},
            attrs={
                'num_rows': num_rows,
                'num_columns': num_columns,
                'dtype': dtype
            },
            stop_gradient=True)

    out.stop_gradient = True
    return out


def full(shape, fill_value, dtype=None, name=None):
    """

    This Op return a Tensor with the ``fill_value`` which size is same as ``shape``.
    
    Args:
        shape(list|tuple|Tensor): Shape of the Tensor to be created.
                The data type is ``int32`` or ``int64`` . If ``shape`` is a list or tuple,
                the elements of it should be integers or Tensors with shape [1].
                If ``shape`` is an Tensor, it should be an 1-D Tensor .
        fill_value(bool|float|int|Tensor): The constant value
            used to initialize the Tensor to be created. If ``fill_value`` is an Tensor, it must be an 1-D Tensor.
        dtype(np.dtype|str, optional): Data type of the output Tensor
            which can be float16, float32, float64, int32, int64, if dytpe is `None`, the data
            type of created Tensor is `float32`
        name(str, optional): The default value is None.  Normally there is no need for user to set this
            property.  For more information, please refer to :ref:`api_guide_Name`.
    
    Returns:
        Tensor: Tensor which is created according to ``shape``, ``fill_value`` and ``dtype``.

    Examples:
        .. code-block:: python

          import paddle

          data1 = paddle.full(shape=[2,1], fill_value=0, dtype='int64') 
          #[[0]
          # [0]]

          # attr shape is a list which contains Tensor.
          positive_2 = paddle.full([1], 2, "int32")
          data3 = paddle.full(shape=[1, positive_2], dtype='float32', fill_value=1.5)
          # [[1.5 1.5]]

          # attr shape is a Tensor.
          shape = paddle.full([2], 2, "int32")
          data4 = paddle.full(shape=shape, dtype='bool', fill_value=True) 
          # [[True True] 
          #  [True True]]
          
          # attr fill_value is a Tensor.
          val = paddle.full([1], 2.0, "float32")
          data5 = paddle.full(shape=[2,1], fill_value=val, dtype='float32')
          # [[2.0] 
          #  [2.0]]
    """

    if dtype is None:
        dtype = 'float32'

    return fill_constant(shape=shape, dtype=dtype, value=fill_value, name=name)


def arange(start=0, end=None, step=1, dtype=None, name=None):
    """
    This OP returns a 1-D Tensor with spaced values within a given interval.

    Values are generated into the half-open interval [``start``, ``end``) with
    the ``step``. (the interval including ``start`` but excluding ``end``).

    If ``dtype`` is float32 or float64, we advise adding a small epsilon to
    ``end`` to avoid floating point rounding errors when comparing against ``end``.

    Parameters:
        start(float|int|Tensor): Start of interval. The interval includes this
            value. If ``end`` is None, the half-open interval is [0, ``start``).
            If ``start`` is a Tensor, it is a 1-D Tensor with shape [1], with
            data type int32, int64, float32, float64. Default is 0.
        end(float|int|Tensor, optional): End of interval. The interval does not
            include this value. If ``end`` is a Tensor, it is a 1-D Tensor with
            shape [1], with data type int32, int64, float32, float64. If ``end``
            is None, the half-open interval is [0, ``start``). Default is None.
        step(float|int|Tensor, optional): Spacing between values. For any out,
            it is the istance between two adjacent values, out[i+1] - out[i].
            If ``step`` is a Tensor, it is a 1-D Tensor with shape [1], with
            data type int32, int64, float32, float64. Default is 1.
        dtype(str|np.dtype, optional): The data type of the
            output tensor. Supported data types: int32, int64, float32, float64.
            If ``dytpe`` is None, the data type is float32. Default is None.
        name(str, optional): The default value is None. Normally there is no
            need for user to set this property. For more information, please
            refer to :ref:`api_guide_Name`.

    Returns: 
        Tensor: A 1-D Tensor with values from the interval [``start``, ``end``)
        taken with common difference ``step`` beginning from ``start``. Its
        data type is set by ``dtype``.

    Raises:
        TypeError: If ``dtype`` is not int32, int64, float32, float64.

    Examples:
        .. code-block:: python

            import paddle

            out1 = paddle.arange(5)
            # [0, 1, 2, 3, 4]

            out2 = paddle.arange(3, 9, 2.0)
            # [3, 5, 7]

            # use 4.999 instead of 5.0 to avoid floating point rounding errors
            out3 = paddle.arange(4.999, dtype='float32')
            # [0., 1., 2., 3., 4.]

            start_var = paddle.to_tensor([3])
            out4 = paddle.arange(start_var, 7)
            # [3, 4, 5, 6]
             
    """
    if dtype is None:
        dtype = 'int64'
    if end is None:
        end = start
        start = 0

    if not isinstance(dtype, core.VarDesc.VarType):
        dtype = convert_np_dtype_to_dtype_(dtype)

    if not isinstance(start, Variable):
        with device_guard("cpu"):
            start = fill_constant([1], dtype, start, force_cpu=True)
    elif start.dtype != dtype:
        start = paddle.cast(start, dtype)

    if not isinstance(end, Variable):
        with device_guard("cpu"):
            end = fill_constant([1], dtype, end, force_cpu=True)
    elif end.dtype != dtype:
        end = paddle.cast(end, dtype)

    if not isinstance(step, Variable):
        with device_guard("cpu"):
            step = fill_constant([1], dtype, step, force_cpu=True)
    elif step.dtype != dtype:
        step = paddle.cast(step, dtype)

    if in_dygraph_mode():
        return _C_ops.final_state_arange(start, end, step, dtype,
                                         _current_expected_place())

    if _in_legacy_dygraph():
        out = _C_ops.range(start, end, step)
        out.stop_gradient = True
        return out

    out_shape = None
    if not isinstance(start, Variable) and not isinstance(
            end, Variable) and not isinstance(step, Variable):
        out_shape = [int(math.ceil((end - start) / step))]

    check_dtype(dtype, 'dtype', ['float32', 'float64', 'int32', 'int64'],
                'range/arange')
    helper = LayerHelper('range', **locals())
    out = helper.create_variable_for_type_inference(dtype, shape=out_shape)
    helper.append_op(
        type='range',
        inputs={'Start': start,
                'End': end,
                'Step': step},
        outputs={'Out': out})
    out.stop_gradient = True
    return out


def _tril_triu_op(helper):
    """Base op of tril_op and triu_op
    """
    op_type = helper.layer_type
    x = helper.kwargs.get('x', None)

    assert x is not None, 'x cannot be None in {}'.format(op_type)
    check_variable_and_dtype(
        x, 'x', ['float16', 'float32', 'float64', 'int32', 'int64'], op_type)
    if len(x.shape) < 2:
        raise ValueError("x shape in {} must be at least 2-D".format(op_type))
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


def tril(x, diagonal=0, name=None):
    r"""
    This op returns the lower triangular part of a matrix (2-D tensor) or batch
    of matrices :attr:`x`, the other elements of the result tensor are set 
    to 0. The lower triangular part of the matrix is defined as the elements 
    on and below the diagonal.

    Args:
        x (Tensor): The input x which is a Tensor.
            Support data types: ``bool``, ``float64``, ``float32``, ``int32``, ``int64``.
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
        Tensor: Results of lower triangular operation by the specified diagonal of input tensor x,
        it's data type is the same as x's Tensor.

    Raises:
        TypeError: diagonal is not a int type.
        ValueError: dimension of :attr:`x` is less than 2.

    Examples:
        .. code-block:: python

            import numpy as np
            import paddle

            data = np.arange(1, 13, dtype="int64").reshape(3,-1)
            # array([[ 1,  2,  3,  4],
            #        [ 5,  6,  7,  8],
            #        [ 9, 10, 11, 12]])


            x = paddle.to_tensor(data)
            
            tril1 = paddle.tensor.tril(x)
            # array([[ 1,  0,  0,  0],
            #        [ 5,  6,  0,  0],
            #        [ 9, 10, 11,  0]])

            # example 2, positive diagonal value
            tril2 = paddle.tensor.tril(x, diagonal=2)
            # array([[ 1,  2,  3,  0], 
            #        [ 5,  6,  7,  8],
            #        [ 9, 10, 11, 12]])

            # example 3, negative diagonal value
            tril3 = paddle.tensor.tril(x, diagonal=-1)
            # array([[ 0,  0,  0,  0],
            #        [ 5,  0,  0,  0],
            #        [ 9, 10,  0,  0]])

    """
    if in_dygraph_mode():
        return _C_ops.final_state_tril_triu(x, diagonal, True)

    if _in_legacy_dygraph():
        op = getattr(_C_ops, 'tril_triu')
        return op(x, 'diagonal', diagonal, "lower", True)

    return _tril_triu_op(LayerHelper('tril', **locals()))


def triu(x, diagonal=0, name=None):
    r"""
    This op returns the upper triangular part of a matrix (2-D tensor) or batch of matrices
    :attr:`x`, the other elements of the result tensor are set to 0.
    The upper triangular part of the matrix is defined as the elements on and
    above the diagonal.

    Args:
        x (Tensor): The input x which is a Tensor.
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
        Tensor: Results of upper triangular operation by the specified diagonal of input tensor x,
        it's data type is the same as x's Tensor.

    Raises:
        TypeError: diagonal is not a int type.
        ValueError: dimension of :attr:`x` is less than 2.

    Examples:
        .. code-block:: python

            import numpy as np
            import paddle

            data = np.arange(1, 13, dtype="int64").reshape(3,-1)
            # array([[ 1,  2,  3,  4],
            #        [ 5,  6,  7,  8],
            #        [ 9, 10, 11, 12]])


            # example 1, default diagonal
            x = paddle.to_tensor(data)
            triu1 = paddle.tensor.triu(x)
            # array([[ 1,  2,  3,  4],
            #        [ 0,  6,  7,  8],
            #        [ 0,  0, 11, 12]])

            # example 2, positive diagonal value
            triu2 = paddle.tensor.triu(x, diagonal=2)
            # array([[0, 0, 3, 4],
            #        [0, 0, 0, 8],
            #        [0, 0, 0, 0]])

            # example 3, negative diagonal value
            triu3 = paddle.tensor.triu(x, diagonal=-1)
            # array([[ 1,  2,  3,  4],
            #        [ 5,  6,  7,  8],
            #        [ 0, 10, 11, 12]])

    """
    if in_dygraph_mode():
        return _C_ops.final_state_tril_triu(x, diagonal, False)

    if _in_legacy_dygraph():
        op = getattr(_C_ops, 'tril_triu')
        return op(x, 'diagonal', diagonal, "lower", False)

    return _tril_triu_op(LayerHelper('triu', **locals()))


def meshgrid(*args, **kwargs):
    """
    This op takes a list of N tensors as input *args, each of which is 1-dimensional 
    vector, and creates N-dimensional grids.
    
    Args:
        *args(Tensor|list of Tensor) : tensors (tuple(list) of tensor): the shapes of input k tensors are (N1,), 
            (N2,),..., (Nk,). Support data types: ``float64``, ``float32``, ``int32``, ``int64``.
        **kwargs (optional): Currently, we only accept name in **kwargs 
            The default value is None. Normally there is no need for
            user to set this property. For more information, please refer to :ref:`api_guide_Name`.
 
    Returns:
         Tensor: k tensors. The shape of each tensor is (N1, N2, ..., Nk)

    Examples:
      .. code-block:: python

          import paddle

          x = paddle.randint(low=0, high=100, shape=[100])
          y = paddle.randint(low=0, high=100, shape=[200])

          grid_x, grid_y = paddle.meshgrid(x, y)

          print(grid_x.shape)
          print(grid_y.shape)

          #the shape of res_1 is (100, 200)
          #the shape of res_2 is (100, 200)

    """

    if len(args) == 1 and isinstance(args[0], (list, tuple)):
        args = args[0]
    if _in_legacy_dygraph():
        num = len(args)
        out = _C_ops.meshgrid(list(args), num)
        return out
    if in_dygraph_mode():
        return _C_ops.final_state_meshgrid(list(args))

    name = kwargs.get("name", None)
    helper = LayerHelper('meshgrid', **locals())

    if not isinstance(args, (list, tuple)):
        raise TypeError("The type of input args in meshgrid should be list.")

    for id, input_ in enumerate(args):
        check_dtype(input_.dtype, 'create data type',
                    ['float16', 'float32', 'float64', 'int32', 'int64'],
                    'meshgrid')

    num = len(args)
    out = [
        helper.create_variable_for_type_inference(dtype=args[i].dtype)
        for i in range(num)
    ]
    helper.append_op(
        type='meshgrid', inputs={'X': list(args)}, outputs={'Out': out})

    return out


def diagflat(x, offset=0, name=None):
    """
    If ``x`` is a vector (1-D tensor), a 2-D square tensor with the elements of ``x`` as the diagonal is returned.

    If ``x`` is a tensor (more than 1-D), a 2-D square tensor with the elements of flattened ``x`` as the diagonal is returned.

    The argument ``offset`` controls the diagonal offset.


    If ``offset`` = 0, it is the main diagonal.

    If ``offset`` > 0, it is superdiagonal.

    If ``offset`` < 0, it is subdiagonal.

    Args:
        x (Tensor): The input tensor. It can be any shape. Its data type should be float32, float64, int32, int64.
        offset (int, optional): The diagonal offset. A positive value represents superdiagonal, 0 represents the main diagonal, and a negative value represents subdiagonal. Default: 0 (main diagonal).
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor, a square matrix. The output data type is the same as input data type.

    Examples:
        .. code-block:: python

          import paddle

          x = paddle.to_tensor([1, 2, 3])
          y = paddle.diagflat(x)
          print(y.numpy())
          # [[1 0 0]
          #  [0 2 0]
          #  [0 0 3]]

          y = paddle.diagflat(x, offset=1)
          print(y.numpy())
          # [[0 1 0 0]
          #  [0 0 2 0]
          #  [0 0 0 3]
          #  [0 0 0 0]]

          y = paddle.diagflat(x, offset=-1)
          print(y.numpy())
          # [[0 0 0 0]
          #  [1 0 0 0]
          #  [0 2 0 0]
          #  [0 0 3 0]]
        
        .. code-block:: python

          import paddle

          x = paddle.to_tensor([[1, 2], [3, 4]])
          y = paddle.diagflat(x)
          print(y.numpy())
          # [[1 0 0 0]
          #  [0 2 0 0]
          #  [0 0 3 0]
          #  [0 0 0 4]]

          y = paddle.diagflat(x, offset=1)
          print(y.numpy())
          # [[0 1 0 0 0]
          #  [0 0 2 0 0]
          #  [0 0 0 3 0]
          #  [0 0 0 0 4]
          #  [0 0 0 0 0]]

          y = paddle.diagflat(x, offset=-1)
          print(y.numpy())
          # [[0 0 0 0 0]
          #  [1 0 0 0 0]
          #  [0 2 0 0 0]
          #  [0 0 3 0 0]
          #  [0 0 0 4 0]]
    """
    padding_value = 0
    if paddle.in_dynamic_mode():
        if len(x.shape) == 1:
            return _C_ops.diag_v2(x, "offset", offset, "padding_value",
                                  padding_value)
        else:
            y, _ = _C_ops.flatten_contiguous_range(x, "start_axis", 0,
                                                   "stop_axis", -1)
            return _C_ops.diag_v2(y, "offset", offset, "padding_value",
                                  padding_value)

    check_type(x, 'x', (Variable), 'diagflat')
    check_dtype(x.dtype, 'x', ['float32', 'float64', 'int32', 'int64'],
                'diagflat')
    check_type(offset, 'offset', (int), 'diagflat')

    helper = LayerHelper("diagflat", **locals())
    out1 = helper.create_variable_for_type_inference(dtype=x.dtype)
    out1_shape = helper.create_variable_for_type_inference(x.dtype)
    out2 = helper.create_variable_for_type_inference(dtype=x.dtype)

    if len(x.shape) == 1:
        helper.append_op(
            type='diag_v2',
            inputs={'X': x},
            outputs={'Out': out2},
            attrs={'offset': offset,
                   'padding_value': padding_value})
    else:
        helper.append_op(
            type='flatten_contiguous_range',
            inputs={'X': x},
            outputs={'Out': out1,
                     'XShape': out1_shape},
            attrs={'start_axis': 0,
                   'stop_axis': -1})
        out1.stop_gradient = True

        helper.append_op(
            type='diag_v2',
            inputs={'X': out1},
            outputs={'Out': out2},
            attrs={'offset': offset,
                   'padding_value': padding_value})
    out2.stop_gradient = True
    return out2


def diag(x, offset=0, padding_value=0, name=None):
    """
    If ``x`` is a vector (1-D tensor), a 2-D square tensor with the elements of ``x`` as the diagonal is returned.

    If ``x`` is a matrix (2-D tensor), a 1-D tensor with the diagonal elements of ``x`` is returned.

    The argument ``offset`` controls the diagonal offset:

    If ``offset`` = 0, it is the main diagonal.

    If ``offset`` > 0, it is superdiagonal.

    If ``offset`` < 0, it is subdiagonal.

    Args:
        x (Tensor): The input tensor. Its shape is either 1-D or 2-D. Its data type should be float32, float64, int32, int64.
        offset (int, optional): The diagonal offset. A positive value represents superdiagonal, 0 represents the main diagonal, and a negative value represents subdiagonal.
        padding_value (int|float, optional): Use this value to fill the area outside the specified diagonal band. Only takes effect when the input is a 1-D Tensor. The default value is 0.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor, a square matrix or a vector. The output data type is the same as input data type.

    Examples:
        .. code-block:: python

          import paddle

          paddle.disable_static()
          x = paddle.to_tensor([1, 2, 3])
          y = paddle.diag(x)
          print(y.numpy())
          # [[1 0 0]
          #  [0 2 0]
          #  [0 0 3]]

          y = paddle.diag(x, offset=1)
          print(y.numpy())
          # [[0 1 0 0]
          #  [0 0 2 0]
          #  [0 0 0 3]
          #  [0 0 0 0]]

          y = paddle.diag(x, padding_value=6)
          print(y.numpy())
          # [[1 6 6]
          #  [6 2 6]
          #  [6 6 3]]

        .. code-block:: python

          import paddle

          paddle.disable_static()
          x = paddle.to_tensor([[1, 2, 3], [4, 5, 6]])
          y = paddle.diag(x)
          print(y.numpy())
          # [1 5]

          y = paddle.diag(x, offset=1)
          print(y.numpy())
          # [2 6]

          y = paddle.diag(x, offset=-1)
          print(y.numpy())
          # [4]
    """
    if in_dygraph_mode():
        return _C_ops.final_state_diag(x, offset, padding_value)
    else:
        if _in_legacy_dygraph():
            return _C_ops.diag_v2(x, "offset", offset, "padding_value",
                                  padding_value)
        else:
            check_type(x, 'x', (Variable), 'diag_v2')
            check_dtype(x.dtype, 'x', ['float32', 'float64', 'int32', 'int64'],
                        'diag_v2')
            check_type(offset, 'offset', (int), 'diag_v2')
            check_type(padding_value, 'padding_value', (int, float), 'diag_v2')
            if len(x.shape) != 1 and len(x.shape) != 2:
                raise ValueError(
                    "The dimension of input x must be either 1 or 2, but received {}".
                    format(len(x.shape)))

            helper = LayerHelper("diag_v2", **locals())

            out = helper.create_variable_for_type_inference(dtype=x.dtype)

            helper.append_op(
                type='diag_v2',
                inputs={'X': x},
                outputs={'Out': out},
                attrs={'offset': offset,
                       'padding_value': padding_value})

            out.stop_gradient = True
            return out


def empty(shape, dtype=None, name=None):
    """
    This Op returns a Tensor with uninitialized data which size is same as ``shape``.
    
    Args:
        shape(list|tuple|Tensor): Shape of the Tensor to be created.
                The data type of dimension of shape is ``int32`` or ``int64`` . If ``shape`` is a list or tuple,
                the elements of it should be integers or Tensors with shape [1].
                If ``shape`` is an Tensor, it should be an 1-D Tensor.
        dtype(np.dtype|str, optional): Data type of the output Tensor
            which can be bool, float16, float32, float64, int32, int64, if dytpe is `None`, the data
            type of created Tensor use global default dtype (see ``get_default_dtype``
            for details).
        name(str, optional): The default value is None. Normally there is no need for user to set this
            property. For more information, please refer to :ref:`api_guide_Name`.
    
    Returns:
        Tensor: Tensor which is created according to ``shape`` and ``dtype``, and is uninitialized.

    Examples:
        .. code-block:: python

          import paddle
          import numpy as np

          paddle.set_device("cpu")  # and use cpu device

          # example 1: argument ``shape`` is a list which doesn't contain Tensor.
          data1 = paddle.empty(shape=[2,3], dtype='float32')
          #[[4.3612203e+27 1.8176809e+31 1.3555911e-19]     # uninitialized
          # [1.1699684e-19 1.3563156e-19 3.6408321e-11]]    # uninitialized

          # example 2: argument ``shape`` is a Tensor, the data type must be int64 or int32.
          shape_data = np.array([2, 3]).astype('int32')
          shape = paddle.to_tensor(shape_data)
          data2 = paddle.empty(shape=shape, dtype='float32')
          #[[1.7192326e-37 4.8125365e-38 1.9866003e-36]     # uninitialized
          # [1.3284029e-40 7.1117408e-37 2.5353012e+30]]    # uninitialized

          # example 3: argument ``shape`` is a list which contains Tensor.
          dim2_data = np.array([3]).astype('int32')
          dim2 = paddle.to_tensor(dim2_data)
          data3 = paddle.empty(shape=[2, dim2], dtype='float32')
          #[[1.1024214e+24 7.0379409e+22 6.5737699e-34]     # uninitialized
          # [7.5563101e+31 7.7130405e+31 2.8020654e+20]]    # uninitialized
    """

    if dtype is None:
        dtype = paddle.get_default_dtype()

    dtype = convert_dtype(dtype)

    if paddle.in_dynamic_mode():
        shape = utils.convert_shape_to_list(shape)
        out = _C_ops.empty('shape', shape, 'dtype',
                           convert_np_dtype_to_dtype_(dtype))
        out.stop_gradient = True
        return out

    helper = LayerHelper("empty", **locals())
    inputs = {}

    check_dtype(dtype, 'dtype',
                ['bool', 'float16', 'float32', 'float64', 'int32', 'int64'],
                'empty')
    check_type(shape, 'shape', (Variable, list, tuple), 'empty')

    if isinstance(shape, Variable):
        check_dtype(shape.dtype, 'shape', ['int32', 'int64'], 'empty')

    attrs = {}
    utils.get_shape_tensor_inputs(
        inputs=inputs, attrs=attrs, shape=shape, op_type='empty')

    out = helper.create_variable_for_type_inference(dtype=dtype)
    attrs['dtype'] = convert_np_dtype_to_dtype_(dtype)
    helper.append_op(
        type='empty',
        inputs=inputs,
        outputs={'Out': [out]},
        attrs=attrs,
        stop_gradient=True)
    out.stop_gradient = True
    return out


def empty_like(x, dtype=None, name=None):
    """
    This Op returns a Tensor with uninitialized data which has identical shape of ``x`` and ``dtype``.
    If the ``dtype`` is None, the data type of Tensor is same with ``x``.
    
    Args:
        x(Tensor): The input tensor which specifies shape and data type. The data type can be bool, float16, float32, float64, int32, int64.
        dtype(np.dtype|str, optional): The data type of output. The data type can be one
            of bool, float16, float32, float64, int32, int64. The default value is None, which means the output 
            data type is the same as input.
        name(str, optional): The default value is None. Normally there is no need for user to set this
            property. For more information, please refer to :ref:`api_guide_Name`.
    
    Returns:
        Tensor: Tensor which is created according to ``x`` and ``dtype``, and is uninitialized.

    Examples:
        .. code-block:: python

          import paddle
          import numpy as np

          paddle.set_device("cpu")  # and use cpu device

          x = paddle.randn([2, 3], 'float32')
          output = paddle.empty_like(x)
          #[[1.8491974e+20 1.8037303e+28 1.7443726e+28]     # uninitialized
          # [4.9640171e+28 3.0186127e+32 5.6715899e-11]]    # uninitialized
    """

    if dtype is None:
        dtype = x.dtype
    dtype = convert_dtype(dtype)

    if paddle.in_dynamic_mode():
        out = _C_ops.empty('shape', x.shape, 'dtype',
                           convert_np_dtype_to_dtype_(dtype))
        out.stop_gradient = True
        return out

    helper = LayerHelper("empty_like", **locals())
    check_variable_and_dtype(
        x, 'x', ['bool', 'float16', 'float32', 'float64', 'int32', 'int64'],
        'empty_like')
    check_dtype(dtype, 'dtype',
                ['bool', 'float16', 'float32', 'float64', 'int32', 'int64'],
                'empty_like')
    out = helper.create_variable_for_type_inference(dtype=dtype)

    inputs = {}
    attrs = {}
    attrs['dtype'] = convert_np_dtype_to_dtype_(dtype)
    shape = paddle.shape(x)
    utils.get_shape_tensor_inputs(
        inputs=inputs, attrs=attrs, shape=shape, op_type='empty_like')

    helper.append_op(
        type='empty',
        inputs=inputs,
        outputs={'Out': [out]},
        attrs=attrs,
        stop_gradient=True)
    out.stop_gradient = True
    return out


def assign(x, output=None):
    """

    The OP copies the :attr:`x` to the :attr:`output`.
 
    Parameters:
        x (Tensor|np.ndarray|list|tuple|scalar): A tensor, numpy ndarray, tuple/list of scalar,
            or scalar. Its data type supports float16, float32, float64, int32, int64, and bool.
            Note: the float64 data will be converted to float32 because of current platform protobuf
            data limitation.
        output (Tensor, optional): A tensor. If :attr:`output` is None, a new tensor will
            be created as :attr:`output`. Default: None.
 
    Returns:
        Tensor: A tensor with the same shape, data type and value as :attr:`x`.
 
    Examples:
        .. code-block:: python
 
          import paddle
          import numpy as np
          data = paddle.full(shape=[3, 2], fill_value=2.5, dtype='float64') # [[2.5, 2.5], [2.5, 2.5], [2.5, 2.5]]
          array = np.array([[1, 1],
                            [3, 4],
                            [1, 3]]).astype(np.int64)
          result1 = paddle.zeros(shape=[3, 3], dtype='float32')
          paddle.assign(array, result1) # result1 = [[1, 1], [3 4], [1, 3]]
          result2 = paddle.assign(data)  # result2 = [[2.5, 2.5], [2.5, 2.5], [2.5, 2.5]]
          result3 = paddle.assign(np.array([[2.5, 2.5], [2.5, 2.5], [2.5, 2.5]], dtype='float32')) # result3 = [[2.5, 2.5], [2.5, 2.5], [2.5, 2.5]]
    """
    input = x
    helper = LayerHelper('assign', **locals())
    check_type(input, 'input', (Variable, np.ndarray, list, tuple, float, int,
                                bool), 'assign')
    is_inplace = True if output is not None else False

    if np.isscalar(input) and not isinstance(input, str):
        input = np.array([input])
    elif isinstance(input, (list, tuple)):
        input = np.array(input)
    # NOTE(Aurelius84): Why we judge core.VarBase?
    # In case of @to_static, a VarBase can be as input of `assign`,
    # but _non_static_mode()==False under @to_static, which means
    # isinstance(VarBase, Variable) == False. It will cause return None
    # after this api.
    if isinstance(input, (Variable, core.VarBase)):
        if _non_static_mode():
            if output is None:
                if _in_legacy_dygraph():
                    output = core.VarBase()
                else:
                    output = core.eager.Tensor()
            _C_ops.assign(input, output)
        else:
            check_dtype(input.dtype, 'input', [
                'float16', 'uint16', 'float32', 'float64', 'int32', 'int64',
                'uint8', 'bool'
            ], 'assign', '(When the type of input in assign is Variable.)')
            if output is None:
                output = helper.create_variable_for_type_inference(
                    dtype=input.dtype)
            helper.append_op(
                type='assign', inputs={'X': [input]},
                outputs={'Out': [output]})
    elif isinstance(input, np.ndarray):
        # Not support [var, var, ...] currently.
        if len(input.shape) > 0 and any(isinstance(x, Variable) for x in input):
            raise TypeError(
                "Required type(input) numpy.ndarray, but found `list(Variable)` in input."
            )
        dtype = convert_np_dtype_to_dtype_(input.dtype)
        if dtype == core.VarDesc.VarType.FP64:
            # Setting FP64 numpy data is not supported in Paddle, so we
            # use FP32 here
            warnings.warn(
                "paddle.assign doesn't support float64 input now due "
                "to current platform protobuf data limitation, we convert "
                "it to float32")
            dtype = core.VarDesc.VarType.FP32
        if dtype == core.VarDesc.VarType.BOOL:
            value_name = "bool_values"
            values = [int(v) for v in input.flat]
        elif dtype == core.VarDesc.VarType.FP32:
            value_name = "fp32_values"
            values = [float(v) for v in input.flat]
        elif dtype == core.VarDesc.VarType.INT32:
            value_name = "int32_values"
            values = [int(v) for v in input.flat]
        elif dtype == core.VarDesc.VarType.INT64:
            value_name = "int64_values"
            values = [int(v) for v in input.flat]
        else:
            raise TypeError(
                "When the type of 'input' in assign is numpy.ndarray, "
                "the data type of 'input' must be bool, float32, int32 or int64, but "
                "received %s." % convert_dtype(dtype))
        if input.size > 1024 * 1024:
            raise ValueError("The size of input is too big. Please consider "
                             "saving it to file and 'load_op' to load it")
        if output is None:
            output = helper.create_variable_for_type_inference(
                dtype=input.dtype)
        helper.append_op(
            type='assign_value',
            outputs={'Out': [output]},
            attrs={
                'dtype': dtype,
                'shape': list(input.shape),
                value_name: values
            })

    if is_inplace and _non_static_mode():
        output._bump_inplace_version()

    return output


def clone(x, name=None):
    """
    Returns a copy of input Tensor. It will always have a Tensor copy. 
    
    In addition, This function is derivable, so gradients will flow back from the output to input.

    Parameters:
        x (Tensor): The input Tensor.
        name(str, optional): The default value is None. Normally there is no need for user to set this
            property. For more information, please refer to :ref:`api_guide_Name`.

    Returns: A Tensor copied from ``input`` .

    Examples:
        .. code-block:: python

            import paddle

            x = paddle.ones([2])
            x.stop_gradient = False
            clone_x = paddle.clone(x)

            y = clone_x**3
            y.backward()
            print(clone_x.grad)          # [3]
            print(x.grad)                # [3]
    """
    return x.clone()


#NOTE(zhiqiu): not public 
def _memcpy(input, place=None, output=None):
    """

    The OP copies the :attr:`input` to the :attr:`output`.
    NOTE: currently, only support CUDAPlace <-> CUDAPinnedPlace or NPUPlace <-> CPUPlace.

    Parameters:
        input (Tensor): A tensor. Its data type supports float16, float32, float64, int32, int64, and bool.
        device (Place): Target place for the output.
        output (Tensor, optional): A tensor. If :attr:`output` is None, a new tensor will
            be created as :attr:`output`. Default: None.

    Returns:
        Tensor: A tensor with the same shape, data type and value as :attr:`input`.

    Examples:
        .. code-block:: python

          import paddle
          import numpy as np
          data = paddle.full(shape=[3, 2], fill_value=2.5, dtype='float64') # [[2.5, 2.5], [2.5, 2.5], [2.5, 2.5]]
          result = paddle._memcpy(data, place=paddle.CPUPlace())  # result2 = [[2.5, 2.5], [2.5, 2.5], [2.5, 2.5]]
    """
    helper = LayerHelper('memcpy', **locals())
    check_type(input, 'input', (Variable), 'memcpy')

    if isinstance(input, (Variable, core.VarBase)):
        check_dtype(input.dtype, 'input', [
            'float16', 'uint16', 'float32', 'float64', 'int32', 'int64',
            'uint8', 'bool'
        ], 'memcpy', '(When the type of input in memcpy is Variable.)')
    if output is None:
        output = helper.create_variable_for_type_inference(dtype=input.dtype)

    dst_place_type = -1
    if place is None:
        dst_place_type = -1
    else:
        p = core.Place()
        p.set_place(place)
        if p.is_cpu_place():
            dst_place_type = 0
        elif p.is_gpu_place():
            dst_place_type = 1
        elif p.is_cuda_pinned_place():
            dst_place_type = 2
        elif p.is_xpu_place():
            dst_place_type = 3
        elif p.is_npu_place():
            dst_place_type = 4

    attrs = {'dst_place_type': dst_place_type}
    helper.append_op(
        type='memcpy',
        inputs={'X': [input]},
        outputs={'Out': [output]},
        attrs=attrs)
    return output


def complex(real, imag, name=None):
    """Return a compelx tensor given the real and image component.

    Args:
        real (Tensor): The real component. The data type should be 'float32' or 'float64'.
        imag (Tensor): The image component. The data type should be the same as ``real``.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor: The output tensor. The data type is 'complex64' or 'complex128', with the same precision as ``real`` and ``imag``.

    **Note**:
        ``paddle.complex`` supports broadcasting. If you want know more about broadcasting, please refer to :ref:`user_guide_broadcasting` .

    Examples:
        .. code-block:: python

            import paddle
            x = paddle.arange(2, dtype=paddle.float32).unsqueeze(-1)
            y = paddle.arange(3, dtype=paddle.float32)
            z = paddle.complex(x, y)
            print(z.numpy())

            # [[0.+0.j 0.+1.j 0.+2.j]
            #  [1.+0.j 1.+1.j 1.+2.j]]
    """
    if paddle.in_dynamic_mode():
        return paddle._C_ops.complex(real, imag)

    check_variable_and_dtype(real, 'real', ['float32', 'float64'], 'complex')
    check_variable_and_dtype(imag, 'imag', ['float32', 'float64'], 'complex')

    op_type = "complex"
    helper = LayerHelper(op_type, **locals())
    inputs = {"X": real, "Y": imag}
    out = helper.create_variable_for_type_inference(
        dtype=_real_to_complex_dtype(real.dtype))
    outputs = {"Out": out}
    attrs = {}
    helper.append_op(type=op_type, inputs=inputs, attrs=attrs, outputs=outputs)
    return out
