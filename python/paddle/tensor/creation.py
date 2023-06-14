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

# TODO: define functions to get create a tensor

import math
import re
import warnings

import numpy as np

import paddle
from paddle import _C_ops

from ..fluid.data_feeder import (
    check_dtype,
    check_type,
    check_variable_and_dtype,
    convert_dtype,
    convert_float_to_uint16,
)
from ..fluid.framework import Variable, device_guard
from ..fluid.param_attr import ParamAttr
from ..framework import (
    LayerHelper,
    _current_expected_place,
    _get_paddle_place,
    convert_np_dtype_to_dtype_,
    core,
    in_dynamic_mode,
)

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


def create_global_var(
    shape, value, dtype, persistable=False, force_cpu=False, name=None
):
    """
    This function creates a new tensor variable with value in the global block(block 0).

    Args:
        shape (list[int]|tuple[int]): Shape of the variable
        value (float): The value of the variable. The new created
                      variable will be filled with it.
        dtype (str): Data type of the variable
        persistable (bool, optional): If this variable is persistable.
                           Default: False
        force_cpu (bool, optional): Force this variable to be on CPU.
                         Default: False
        name (str, optional): For detailed information, please refer to
           :ref:`api_guide_Name` . Usually name is no need to set and None by default.

    Returns:
        Variable: The created Variable

    Examples:
        .. code-block:: python

            import paddle
            paddle.enable_static()
            var = paddle.static.create_global_var(shape=[2,3], value=1.0, dtype='float32',
                                           persistable=True, force_cpu=True, name='new_var')
    """
    check_type(shape, 'shape', (list, tuple, np.ndarray), 'create_global_var')
    for item in shape:
        check_type(
            item,
            'item of shape',
            (
                int,
                np.uint8,
                np.int8,
                np.int16,
                np.int32,
                np.int64,
            ),
            'create_global_var',
        )

    check_dtype(
        dtype,
        'dtype',
        [
            'bool',
            'float16',
            'float32',
            'float64',
            'int8',
            'int16',
            'int32',
            'int64',
            'uint8',
            'uint16',
        ],
        'create_global_var',
    )

    helper = LayerHelper("global_var", **locals())
    var = helper.create_global_variable(
        dtype=dtype,
        shape=shape,
        persistable=persistable,
        name=name,
        stop_gradient=True,
    )
    helper.set_variable_initializer(
        var,
        initializer=paddle.nn.initializer.ConstantInitializer(
            value=float(value), force_cpu=force_cpu
        ),
    )

    return var


def create_parameter(
    shape, dtype, name=None, attr=None, is_bias=False, default_initializer=None
):
    """
    This function creates a parameter. The parameter is a learnable variable, which can have
    gradient, and can be optimized.

    Note:
        This is a very low-level API. This API is useful when you create operator by your self, instead of using layers.

    Args:
        shape (list of int): Shape of the parameter
        dtype (str): Data type of the parameter
        name (str, optional): For detailed information, please refer to
           :ref:`api_guide_Name` . Usually name is no need to set and None by default.
        attr (ParamAttr, optional): Attributes of the parameter
        is_bias (bool, optional): This can affect which default initializer is chosen
                       when default_initializer is None. If is_bias,
                       initializer.Constant(0.0) will be used. Otherwise,
                       Xavier() will be used.
        default_initializer (Initializer, optional): Initializer for the parameter

    Returns:
        The created parameter.

    Examples:
        .. code-block:: python

            import paddle
            paddle.enable_static()
            W = paddle.create_parameter(shape=[784, 200], dtype='float32')
    """
    check_type(shape, 'shape', (list, tuple, np.ndarray), 'create_parameter')
    for item in shape:
        check_type(
            item,
            'item of shape',
            (
                int,
                np.uint8,
                np.int8,
                np.int16,
                np.int32,
                np.int64,
            ),
            'create_parameter',
        )

    check_dtype(
        dtype,
        'dtype',
        [
            'bool',
            'float16',
            'uint16',
            'float32',
            'float64',
            'int8',
            'int16',
            'int32',
            'int64',
            'uint8',
        ],
        'create_parameter',
    )
    check_type(attr, 'attr', (type(None), ParamAttr), 'create_parameter')
    check_type(
        default_initializer,
        'default_initializer',
        (type(None), paddle.nn.initializer.Initializer),
        'create_parameter',
    )

    helper = LayerHelper("create_parameter", **locals())
    if attr is None:
        attr = ParamAttr(name=name)
    return helper.create_parameter(
        attr, shape, convert_dtype(dtype), is_bias, default_initializer
    )


def create_tensor(dtype, name=None, persistable=False):
    """
    Create a variable, which will hold a Tensor with data type dtype.

    Args:
        dtype(string|numpy.dtype): the data type of Tensor to be created, the
            data type is bool, float16, float32, float64, int8, int16, int32 and int64.
        name(string, optional): The default value is None.  Normally there is no need for
            user to set this property.  For more information, please refer to :ref:`api_guide_Name`
        persistable(bool): Set the persistable flag of the create tensor.
            default value is False.

    Returns:
        Variable: The tensor to be created according to dtype.

    Examples:
        .. code-block:: python

          import paddle
          tensor = paddle.tensor.create_tensor(dtype='float32')
    """
    check_dtype(
        dtype,
        'dtype',
        [
            'bool',
            'float16',
            'float32',
            'float64',
            'int8',
            'int32',
            'int32',
            'int64',
        ],
        'create_tensor',
    )
    helper = LayerHelper("create_tensor", **locals())
    return helper.create_variable(
        name=helper.name, dtype=dtype, persistable=persistable
    )


def linspace(start, stop, num, dtype=None, name=None):
    r"""
    Return fixed number of evenly spaced values within a given interval. Note: no gradient calculation is performed.

    Args:
        start(int|float|Tensor): The input :attr:`start` is start of range. It is a int, float, \
            or a 0-D Tensor with data type int32, int64, float32 or float64.
        stop(int|float|Tensor): The input :attr:`stop` is end of range. It is a int, float, \
            or a 0-D Tensor with data type int32, int64, float32 or float64.
        num(int|Tensor): The input :attr:`num` is given num of the sequence. It is an int, \
            or a 0-D Tensor with data type int32.
        dtype(np.dtype|str, optional): The data type of output tensor, it could be
            int32, int64, float32 and float64. Default: if None, the data type is float32.
        name(str, optional): For details, please refer to :ref:`api_guide_Name`. Generally, no setting is required. Default: None.

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
            tensor_start = fill_constant([1], dtype, start, force_cpu=True)
    if not isinstance(stop, Variable):
        with device_guard("cpu"):
            tensor_stop = fill_constant([1], dtype, stop, force_cpu=True)
    if not isinstance(num, Variable):
        with device_guard("cpu"):
            tensor_num = fill_constant([1], 'int32', num, force_cpu=True)
    if in_dynamic_mode():
        return _C_ops.linspace(
            tensor_start,
            tensor_stop,
            tensor_num,
            dtype,
            _current_expected_place(),
        )
    else:
        helper = LayerHelper("linspace", **locals())

        start_dtype = convert_dtype(tensor_start.dtype)
        stop_dtype = convert_dtype(tensor_stop.dtype)
        out_dtype = convert_dtype(dtype)
        if isinstance(start, Variable):
            check_dtype(
                start.dtype,
                'start',
                ['float16', 'uint16', 'float32', 'float64', 'int32', 'int64'],
                'linspace',
            )
        else:
            check_type(start, 'start', (int, float), 'linspace')

        if isinstance(stop, Variable):
            check_dtype(
                stop.dtype,
                'stop',
                ['float16', 'uint16', 'float32', 'float64', 'int32', 'int64'],
                'linspace',
            )
        else:
            check_type(stop, 'stop', (int, float), 'linspace')
        if isinstance(num, Variable):
            check_dtype(num.dtype, 'num', ['int32'], 'linspace')
        check_dtype(
            dtype,
            'dtype',
            ['float16', 'uint16', 'float32', 'float64', 'int32', 'int64'],
            'linspace',
        )
        if (
            (stop_dtype == "float64" or start_dtype == "float64")
            and out_dtype in ["float32", "int32"]
        ) or (
            (stop_dtype == "int64" or start_dtype == "int64")
            and out_dtype == "int32"
        ):
            raise ValueError(
                "The dtype of start/stop is {}/{} but the attr(dtype) of linspace is {}, "
                "which may cause data type overflows. Please reset attr(dtype) of linspace.".format(
                    start_dtype, stop_dtype, dtype
                )
            )

        out = helper.create_variable_for_type_inference(dtype=dtype)

        helper.append_op(
            type='linspace',
            inputs={
                'Start': tensor_start,
                'Stop': tensor_stop,
                'Num': tensor_num,
            },
            attrs={'dtype': dtype},
            outputs={'Out': [out]},
        )
        if isinstance(num, int):
            out.desc.set_shape((num,))
        return out


def logspace(start, stop, num, base=10.0, dtype=None, name=None):
    r"""
    Return fixed number of logarithmical-evenly spaced values within the interval \
    :math:`[base^{start}, base^{stop}]`.

    Notes:
        This API does not compute the gradient.

    Args:
        start(int|float|Tensor): The input :attr:`start` is exponent of first entry in \
            the sequence. It is a scalar, or a 0-D Tensor of shape [] with input data \
            type int32, int64, float32 or float64.
        stop(int|float|Tensor): The input :attr:`stop` is exponent of last entry in the \
            sequence. It is a scalar, or a 0-D Tensor of shape [] with input data \
            type int32, int64, float32 or float64.
        num(int|Tensor): The input :attr:`num` is given number of items in the sequence. \
            It is an int scalar, or a 0-D Tensor of shape [] with data type int32.
        base(int|float|Tensor): The input :attr:`base` is base of the logarithm function. \
            It is a scalar, or a 0-D Tensor of shape [] with input data type int32, int64, \
            float32 or float64.
        dtype(np.dtype|str, optional): The data type of output tensor, it could be \
            int32, int64, float32 or float64. Default: if None, the data type is float32. \
        name(str, optional): For details, please refer to :ref:`api_guide_Name`. Generally, no setting is required. Default: None.

    Returns:
        Tensor: The output data type will be float32, float64. The 1-D tensor with \
        fixed number of logarithmical-evenly spaced values, the data shape of this \
        tensor is :math:`[num]`. If the :attr:`num` is set 1, the output tensor \
        just has the value with exponential of :attr:`start` with base :attr:`base`.

    Examples:
        .. code-block:: python

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
    if in_dynamic_mode():
        return _C_ops.logspace(
            tensor_start,
            tensor_stop,
            tensor_num,
            tensor_base,
            dtype,
            _current_expected_place(),
        )
    else:
        helper = LayerHelper("logspace", **locals())

        start_dtype = convert_dtype(tensor_start.dtype)
        stop_dtype = convert_dtype(tensor_stop.dtype)
        base_dtype = convert_dtype(tensor_base.dtype)
        out_dtype = convert_dtype(dtype)
        if isinstance(start, Variable):
            check_dtype(
                start.dtype,
                'start',
                ['float32', 'float64', 'int32', 'int64'],
                'logspace',
            )
        else:
            check_type(start, 'start', (int, float), 'logspace')

        if isinstance(stop, Variable):
            check_dtype(
                stop.dtype,
                'stop',
                ['float32', 'float64', 'int32', 'int64'],
                'logspace',
            )
        else:
            check_type(stop, 'stop', (int, float), 'logspace')

        if isinstance(num, Variable):
            check_dtype(num.dtype, 'num', ['int32'], 'logspace')

        if isinstance(base, Variable):
            check_dtype(
                base.dtype,
                'base',
                ['float32', 'float64', 'int32', 'int64'],
                'logspace',
            )
        else:
            check_type(base, 'base', (int, float), 'logspace')

        check_dtype(
            dtype, 'dtype', ['int32', 'int64', 'float32', 'float64'], 'logspace'
        )
        if (
            (
                stop_dtype == "float64"
                or start_dtype == "float64"
                or base_dtype == "float64"
            )
            and out_dtype in ["float32", "int32"]
        ) or (
            (
                stop_dtype == "int64"
                or start_dtype == "int64"
                or base_dtype == "int64"
            )
            and out_dtype == "int32"
        ):
            raise ValueError(
                "The dtype of start/stop/base is {}/{}/{} but the attr(dtype) of logspace is {}, "
                "which may cause data type overflows. Please reset attr(dtype) of logspace.".format(
                    start_dtype, stop_dtype, base_dtype, dtype
                )
            )

        out = helper.create_variable_for_type_inference(dtype=dtype)

        helper.append_op(
            type='logspace',
            inputs={
                'Start': tensor_start,
                'Stop': tensor_stop,
                'Num': tensor_num,
                'Base': tensor_base,
            },
            attrs={'dtype': dtype},
            outputs={'Out': [out]},
        )
        if isinstance(num, int):
            out.desc.set_shape((num,))
        return out


def _to_tensor_non_static(data, dtype=None, place=None, stop_gradient=True):
    def _handle_tensor_dtype(tensor, dtype):
        if dtype:
            if convert_dtype(dtype) != convert_dtype(tensor.dtype):
                return tensor.astype(convert_dtype(dtype))
        return tensor

    def _handle_np_dtype(ndarray, dtype):
        if dtype:
            if convert_dtype(dtype) != convert_dtype(ndarray.dtype):
                # should not ndarray.astype('uint16') directly, data bits is wrong
                if convert_dtype(dtype) in ['uint16']:
                    return convert_float_to_uint16(ndarray.astype('float32'))
                else:
                    return ndarray.astype(convert_dtype(dtype))

        return ndarray

    if isinstance(data, np.number):  # Special case for numpy scalars
        data = np.array(data)

    if not isinstance(data, np.ndarray):

        if np.isscalar(data) and not isinstance(data, str):
            data = np.array(data)
        elif isinstance(data, (list, tuple)):
            data = np.array(data)
            if data.dtype == np.object_:
                raise ValueError(
                    "\n\tFaild to convert input data to a regular ndarray :\n\t - Usually "
                    "this means the input data contains nested lists with different lengths. "
                )
        elif isinstance(data, paddle.Tensor) and not in_dynamic_mode():
            data = data._copy_to(place, False)
            data = _handle_tensor_dtype(data, dtype)
            data.stop_gradient = stop_gradient
            return data
        elif isinstance(data, core.eager.Tensor) and in_dynamic_mode():
            data = data._copy_to(place, False)
            data = _handle_tensor_dtype(data, dtype)
            data.stop_gradient = stop_gradient
            return data
        elif isinstance(data, (core.LoDTensor, core.Tensor)):
            # should't expose it to users, just for internal use.
            # convert core.Tensor/core.LoDTensor to Tensor first
            # Currenly, there is no copy when places are same
            if in_dynamic_mode():
                data = core.eager.Tensor(data)
            else:
                data = paddle.Tensor(data)
            if not data.place._equals(place):
                data = data._copy_to(place, False)
            data = _handle_tensor_dtype(data, dtype)
            data.stop_gradient = stop_gradient
            return data
        else:
            raise TypeError(
                "Can't constructs a 'paddle.Tensor' with data type {}, data type must be scalar|list|tuple|np.ndarray|paddle.Tensor".format(
                    type(data)
                )
            )
        if not dtype:
            if data.dtype in [
                'float16',
                'float32',
                'float64',
                'complex64',
                'complex128',
            ]:
                default_type = paddle.get_default_dtype()
                if np.iscomplexobj(data):
                    default_type = (
                        'complex64'
                        if default_type in ['float16', 'float32']
                        else 'complex128'
                    )
                data = _handle_np_dtype(data, default_type)
            # Windows default type is 'int32', while Linux/Mac is 'int64'. Unify they.
            if data.dtype in ['int32']:
                data = data.astype("int64")

    if dtype:
        data = _handle_np_dtype(data, dtype)

    if isinstance(data, np.ndarray):
        return core.eager.Tensor(
            value=data,
            place=place,
            persistable=False,
            zero_copy=False,
            name=None,
            stop_gradient=stop_gradient,
        )
    else:
        return paddle.Tensor(
            value=data,
            place=place,
            persistable=False,
            zero_copy=False,
            stop_gradient=stop_gradient,
        )


def _to_tensor_static(data, dtype=None, stop_gradient=None):

    if isinstance(data, Variable):
        output = data
        if dtype is not None and dtype != data.dtype:
            output = paddle.cast(output, dtype)
    else:
        if isinstance(data, np.number):  # Special case for numpy scalars
            data = np.array(data)

        if not isinstance(data, np.ndarray):
            if np.isscalar(data) and not isinstance(data, str):
                data = np.array(data)
            elif isinstance(data, (list, tuple)):
                try:
                    '''
                    In numpy version >= 1.24.0, case like:
                        np.array([Variable, 1, 2])
                    is not supported, it will raise error (numpy returns an numpy array with dtype='object' in version <= 1.23.5)

                    Thus, process nested structure in except block
                    '''
                    data = np.array(data)

                    # for numpy version <= 1.23.5
                    if data.dtype == 'object':
                        raise RuntimeError("Numpy get dtype `object`.")

                except:
                    to_stack_list = [None] * len(data)
                    for idx, d in enumerate(data):
                        to_stack_list[idx] = _to_tensor_static(
                            d, dtype, stop_gradient
                        )
                    data = paddle.stack(to_stack_list)
                    data = paddle.squeeze(data, -1)

            else:
                raise RuntimeError(
                    f"Do not support transform type `{type(data)}` to tensor"
                )

            # fix numpy default dtype
            if data.dtype in ['float16', 'float32', 'float64']:
                data = data.astype(paddle.get_default_dtype())

        if dtype:
            target_dtype = dtype
        elif hasattr(data, 'dtype') and data.dtype != 'object':
            target_dtype = data.dtype
        else:
            target_dtype = paddle.get_default_dtype()
        target_dtype = convert_dtype(target_dtype)

        output = assign(data)

        if convert_dtype(output.dtype) != target_dtype:
            output = paddle.cast(output, target_dtype)

    output.stop_gradient = stop_gradient

    return output


def to_tensor(data, dtype=None, place=None, stop_gradient=True):
    r"""
    Constructs a ``paddle.Tensor`` from ``data`` ,
    which can be scalar, tuple, list, numpy\.ndarray, paddle\.Tensor.

    If the ``data`` is already a Tensor, copy will be performed and return a new tensor.
    If you only want to change stop_gradient property, please call ``Tensor.stop_gradient = stop_gradient`` directly.

    .. code-block:: text

        We use the dtype conversion rules following this:
                Keep dtype
        np.number ───────────► paddle.Tensor
                                (0-D Tensor)
                    default_dtype
        Python Number ───────────────► paddle.Tensor
                                        (0-D Tensor)
                    Keep dtype
        np.ndarray ───────────► paddle.Tensor

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

    Examples:

    .. code-block:: python

        import paddle

        type(paddle.to_tensor(1))
        # <class 'paddle.Tensor'>

        paddle.to_tensor(1)
        # Tensor(shape=[], dtype=int64, place=CPUPlace, stop_gradient=True,
        #        1)

        x = paddle.to_tensor(1, stop_gradient=False)
        # Tensor(shape=[], dtype=int64, place=CPUPlace, stop_gradient=False,
        #        1)

        paddle.to_tensor(x)  # A new tensor will be created with default stop_gradient=True
        # Tensor(shape=[], dtype=int64, place=CPUPlace, stop_gradient=True,
        #        1)

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

    if in_dynamic_mode():
        return _to_tensor_non_static(data, dtype, place, stop_gradient)

    # call assign for static graph
    else:
        re_exp = re.compile(r'[(](.+?)[)]', re.S)
        place_str = re.findall(re_exp, str(place))[0]

        with paddle.static.device_guard(place_str):
            return _to_tensor_static(data, dtype, stop_gradient)


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
        name(str, optional): For details, please refer to :ref:`api_guide_Name`. Generally, no setting is required. Default: None.

    Returns:
        Tensor: Tensor which is created according to ``x``, ``fill_value`` and ``dtype``.

    Examples:
        .. code-block:: python

          import paddle

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
    if in_dynamic_mode():
        return _C_ops.full_like(x, fill_value, dtype, x.place)
    else:
        helper = LayerHelper("full_like", **locals())
        check_variable_and_dtype(
            x,
            'x',
            [
                'bool',
                'float16',
                'float32',
                'float64',
                'int16',
                'int32',
                'int64',
                'uint16',
            ],
            'full_like',
        )
        check_dtype(
            dtype,
            'dtype',
            [
                'bool',
                'float16',
                'float32',
                'float64',
                'int16',
                'int32',
                'int64',
                'uint16',
            ],
            'full_like/zeros_like/ones_like',
        )
        out = helper.create_variable_for_type_inference(dtype=dtype)

        helper.append_op(
            type='fill_any_like',
            inputs={'X': [x]},
            attrs={'value': fill_value, "dtype": dtype},
            outputs={'Out': [out]},
        )
        out.stop_gradient = True
        return out


def fill_constant(shape, dtype, value, force_cpu=False, out=None, name=None):
    if in_dynamic_mode():
        place = _current_expected_place()
        if force_cpu:
            place = core.CPUPlace()
        if isinstance(shape, (list, tuple)):
            shape = paddle.utils.convert_shape_to_list(shape)

        if not isinstance(dtype, core.VarDesc.VarType):
            dtype = convert_np_dtype_to_dtype_(dtype)

        if out is None:
            out = _C_ops.full(shape, float(value), dtype, place)
            out.stop_gradient = True
            return out

        if out is not None:
            # final state mode is support out is not None.
            _C_ops.full_(out, shape, float(value), dtype, place)
            out.stop_gradient = True
            return out
    else:
        attrs = {'force_cpu': force_cpu}
        dtype = convert_dtype(dtype)
        if not isinstance(value, Variable):
            if dtype in ['int8', 'uint8', 'int16', 'int32', 'int64']:
                attrs['str_value'] = str(int(value))
                attrs['value'] = int(value)
            else:
                attrs['str_value'] = str(float(value))
                attrs['value'] = float(value)

        helper = LayerHelper("fill_constant", **locals())
        inputs = {}
        if isinstance(value, Variable):
            if convert_dtype(value.dtype) != dtype:
                value = paddle.cast(value, dtype)
            inputs['ValueTensor'] = value

        paddle.utils.check_shape(shape)
        check_dtype(
            dtype,
            'dtype',
            [
                'bool',
                'float16',
                'float32',
                'float64',
                'int8',
                'uint8',
                'int16',
                'int32',
                'int64',
                'complex64',
                'complex128',
                'uint16',
            ],
            'fill_constant',
        )
        check_type(shape, 'shape', (Variable, list, tuple), 'fill_constant')

        if out is not None:
            check_variable_and_dtype(
                out, 'out', [convert_dtype(dtype)], 'fill_constant'
            )

        helper = LayerHelper("fill_constant", **locals())
        paddle.utils.get_shape_tensor_inputs(
            inputs=inputs, attrs=attrs, shape=shape, op_type='fill_constant'
        )

        if out is None:
            out = helper.create_variable_for_type_inference(dtype=dtype)
        attrs['dtype'] = out.dtype
        helper.append_op(
            type='fill_constant',
            inputs=inputs,
            outputs={'Out': [out]},
            attrs=attrs,
            stop_gradient=True,
        )
        out.stop_gradient = True
        return out


def ones(shape, dtype=None, name=None):
    """
    Create a Tensor of specified :attr:`shape` and :attr:`dtype` and fill it with 1.

    Args:
        shape (tuple|list|Tensor): Shape of the Tensor to be created. The data type is ``int32`` or ``int64`` .
            If ``shape`` is a list or tuple, the elements of it should be integers or 0-D Tensor with shape [].
            If ``shape`` is an Tensor, it should be an 1-D Tensor which represents a list.
        dtype (np.dtype|str, optional): Data type of output Tensor, it should be one of
            bool, float16, float32, float64, int32 and int64. If it is set to None, the data type will be float32.
        name (str, optional): For details, please refer to :ref:`api_guide_Name`. Generally, no setting is required. Default: None.

    Returns:
        Tensor: A Tensor of data type :attr:`dtype` with shape :attr:`shape` and all elements are 1.

    Examples:
        .. code-block:: python

            import paddle

            # shape is a list/tuple
            data1 = paddle.ones(shape=[3, 2])
            # [[1. 1.]
            #  [1. 1.]
            #  [1. 1.]]

            # shape is a Tensor
            shape = paddle.to_tensor([3, 2])
            data2 = paddle.ones(shape=shape)
            # [[1. 1.]
            #  [1. 1.]
            #  [1. 1.]]

            # shape is a Tensor List
            shape = [paddle.to_tensor(3), paddle.to_tensor(2)]
            data3 = paddle.ones(shape=shape)
            # [[1. 1.]
            #  [1. 1.]
            #  [1. 1.]]
    """
    if dtype is None:
        dtype = core.VarDesc.VarType.FP32
    return fill_constant(value=1.0, shape=shape, dtype=dtype, name=name)


def ones_like(x, dtype=None, name=None):
    """
    Returns a Tensor filled with the value 1, with the same shape and
    data type (use ``dtype`` if ``dtype`` is not None) as ``x``.

    Args:
        x(Tensor): The input tensor which specifies shape and dtype. The
            dtype of ``x`` can be bool, float16, float32, float64, int32, int64.
        dtype(str|np.dtype, optional): The data type of the
            output tensor. Supported data types: bool, float16, float32, float64,
            int32, int64. If ``dtype`` is None, the data type is the same as ``x``.
            Default is None.
        name(str, optional): For details, please refer to :ref:`api_guide_Name`. Generally, no setting is required. Default: None.

    Returns:
        Tensor: A Tensor filled with the value 1, with the same shape and
        data type (use ``dtype`` if ``dtype`` is not None) as ``x``.

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
    Creates a tensor of specified :attr:`shape` and :attr:`dtype`, and fills it with 0.

    Args:
        shape (tuple|list|Tensor): Shape of the Tensor to be created. The data type is ``int32`` or ``int64`` .
            If ``shape`` is a list or tuple, each element of it should be integer or 0-D Tensor with shape [].
            If ``shape`` is an Tensor, it should be an 1-D Tensor which represents a list.
        dtype(np.dtype|str, optional): Data type of output Tensor, it supports
            bool, float16, float32, float64, int32 and int64. Default: if None, the date type is float32.
        name(str, optional): The default value is None.  Normally there is no need for user to set this
            property.  For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor: A tensor of data type :attr:`dtype` with shape :attr:`shape` and all elements set to 0.

    Examples:
        .. code-block:: python

            import paddle

            # shape is a list/tuple
            data1 = paddle.zeros(shape=[3, 2])
            # [[0. 0.]
            #  [0. 0.]
            #  [0. 0.]]

            # shape is a Tensor
            shape = paddle.to_tensor([3, 2])
            data2 = paddle.zeros(shape=shape)
            # [[0. 0.]
            #  [0. 0.]
            #  [0. 0.]]

            # shape is a Tensor List
            shape = [paddle.to_tensor(3), paddle.to_tensor(2)]
            data3 = paddle.zeros(shape=shape)
            # [[0. 0.]
            #  [0. 0.]
            #  [0. 0.]]
    """
    if dtype is None:
        dtype = 'float32'
    return fill_constant(value=0.0, shape=shape, dtype=dtype, name=name)


def zeros_like(x, dtype=None, name=None):
    """
    Returns a Tensor filled with the value 0, with the same shape and
    data type (use ``dtype`` if ``dtype`` is not None) as ``x``.

    Args:
        x(Tensor): The input tensor which specifies shape and dtype. The
            dtype of ``x`` can be bool, float16, float32, float64, int32, int64.
        dtype(str|np.dtype, optional): The data type of the
            output tensor. Supported data types: bool, float16, float32, float64,
            int32, int64. If ``dtype`` is None, the data type is the same as ``x``.
            Default is None.
        name (str, optional): For details, please refer to :ref:`api_guide_Name`. Generally, no setting is required. Default: None.

    Returns:
        Tensor: A Tensor filled with the value 0, with the same shape and
        data type (use ``dtype`` if ``dtype`` is not None) as ``x``.


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
        name(str, optional): For details, please refer to :ref:`api_guide_Name`. Generally, no setting is required. Default: None.

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

    def _check_attr(attr, message):
        if isinstance(attr, ((Variable, core.eager.Tensor))):
            assert len(attr.shape) == 1 and attr.shape[0] in [1, -1]
        elif not isinstance(attr, int) or attr < 0:
            raise TypeError(f"{message} should be a non-negative int.")

    _check_attr(num_rows, "num_rows")

    if dtype is None:
        dtype = core.VarDesc.VarType.FP32
    elif not isinstance(dtype, core.VarDesc.VarType):
        dtype = convert_np_dtype_to_dtype_(dtype)
    if num_columns is not None:
        _check_attr(num_columns, "num_columns")
    else:
        num_columns = num_rows

    if in_dynamic_mode():
        out = _C_ops.eye(
            num_rows, num_columns, dtype, _current_expected_place()
        )
    else:
        helper = LayerHelper("eye", **locals())
        check_dtype(
            dtype,
            'dtype',
            ['float16', 'float32', 'float64', 'int32', 'int64'],
            'eye',
        )
        out = helper.create_variable_for_type_inference(dtype=dtype)
        helper.append_op(
            type='eye',
            inputs={},
            outputs={'Out': [out]},
            attrs={
                'num_rows': num_rows,
                'num_columns': num_columns,
                'dtype': dtype,
            },
            stop_gradient=True,
        )

    out.stop_gradient = True
    return out


def full(shape, fill_value, dtype=None, name=None):
    """

    Return a Tensor with the ``fill_value`` which size is same as ``shape``.

    Args:
        shape (tuple|list|Tensor): Shape of the Tensor to be created. The data type is ``int32`` or ``int64`` .
            If ``shape`` is a list or tuple, each element of it should be integer or 0-D Tensor with shape [].
            If ``shape`` is an Tensor, it should be an 1-D Tensor which represents a list.
        fill_value(bool|float|int|Tensor): The constant value used to initialize the Tensor to be created.
            If ``fill_value`` is an Tensor, it shoule be an 0-D Tensor which represents a scalar.
        dtype(np.dtype|str, optional): Data type of the output Tensor
            which can be float16, float32, float64, int32, int64, if dytpe is `None`, the data
            type of created Tensor is `float32`.
        name (str, optional): For details, please refer to :ref:`api_guide_Name`. Generally, no setting is required. Default: None.

    Returns:
        Tensor: Tensor which is created according to ``shape``, ``fill_value`` and ``dtype``.

    Examples:
        .. code-block:: python

            import paddle

            # shape is a list/tuple
            data1 = paddle.full(shape=[3, 2], fill_value=1.)
            # [[1. 1.]
            #  [1. 1.]
            #  [1. 1.]]

            # shape is a Tensor
            shape = paddle.to_tensor([3, 2])
            data2 = paddle.full(shape=shape, fill_value=2.)
            # [[2. 2.]
            #  [2. 2.]
            #  [2. 2.]]

            # shape is a Tensor List
            shape = [paddle.to_tensor(3), paddle.to_tensor(2)]
            data3 = paddle.full(shape=shape, fill_value=3.)
            # [[3. 3.]
            #  [3. 3.]
            #  [3. 3.]]

            # fill_value is a Tensor.
            val = paddle.full([], 2.0, "float32")
            data5 = paddle.full(shape=[3, 2], fill_value=val)
            # [[2. 2.]
            #  [2. 2.]
            #  [2. 2.]]
    """

    if dtype is None:
        dtype = 'float32'

    return fill_constant(shape=shape, dtype=dtype, value=fill_value, name=name)


def arange(start=0, end=None, step=1, dtype=None, name=None):
    """
    Returns a 1-D Tensor with spaced values within a given interval.

    Values are generated into the half-open interval [``start``, ``end``) with
    the ``step``. (the interval including ``start`` but excluding ``end``).

    If ``dtype`` is float32 or float64, we advise adding a small epsilon to
    ``end`` to avoid floating point rounding errors when comparing against ``end``.

    Parameters:
        start(float|int|Tensor): Start of interval. The interval includes this
            value. If ``end`` is None, the half-open interval is [0, ``start``).
            If ``start`` is a Tensor, it is a 0-D Tensor which represents a scalar
            and data type is int32, int64, float32, float64. Default is 0.
        end(float|int|Tensor, optional): End of interval. The interval does not
            include this value. If ``end`` is a Tensor, it is a 0-D Tensor which
            represents a scalar and data type is int32, int64, float32, float64.
            If ``end`` is None, the half-open interval is [0, ``start``).
            Default is None.
        step(float|int|Tensor, optional): Spacing between values. For any out,
            it is the istance between two adjacent values, out[i+1] - out[i].
            If ``step`` is a Tensor, it is a 0-D Tensor which represents a scalar
            and data type is int32, int64, float32, float64. . Default is 1.
        dtype(str|np.dtype, optional): The data type of the
            output tensor. Supported data types: int32, int64, float32, float64.
            If ``dytpe`` is None, the data type is float32. Default is None.
        name (str, optional): For details, please refer to :ref:`api_guide_Name`. Generally, no setting is required. Default: None.

    Returns:
        Tensor: A 1-D Tensor with values from the interval [``start``, ``end``)
        taken with common difference ``step`` beginning from ``start``. Its
        data type is set by ``dtype``.

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

            start_var = paddle.to_tensor(3)
            out4 = paddle.arange(start_var, 7)
            # [3, 4, 5, 6]

    """
    if end is None:
        end = start
        start = 0

    if dtype is None:
        for val in [start, end, step]:
            if isinstance(val, Variable):
                if not val.is_integer():
                    dtype = paddle.get_default_dtype()
                    break
                else:
                    dtype = 'int64'
            else:
                if not isinstance(val, np.integer) and not isinstance(val, int):
                    dtype = paddle.get_default_dtype()
                    break
                else:
                    dtype = 'int64'

    out_shape = None
    if not in_dynamic_mode() and (
        not isinstance(start, Variable)
        and not isinstance(end, Variable)
        and not isinstance(step, Variable)
    ):
        out_shape = [int(math.ceil((end - start) / step))]

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

    if in_dynamic_mode():
        return _C_ops.arange(start, end, step, dtype, _current_expected_place())
    else:
        check_dtype(
            dtype,
            'dtype',
            ['float32', 'float64', 'int32', 'int64', 'float16', 'uint16'],
            'range/arange',
        )
        helper = LayerHelper('range', **locals())
        out = helper.create_variable_for_type_inference(dtype, shape=out_shape)
        helper.append_op(
            type='range',
            inputs={'Start': start, 'End': end, 'Step': step},
            outputs={'Out': out},
        )
        out.stop_gradient = True
        if out_shape is not None:
            out.desc.set_shape(out_shape)
        return out


def _tril_triu_op(helper):
    """Base op of tril_op and triu_op"""
    op_type = helper.layer_type
    x = helper.kwargs.get('x', None)

    assert x is not None, f'x cannot be None in {op_type}'
    check_variable_and_dtype(
        x,
        'x',
        ['float16', 'uint16', 'float32', 'float64', 'int32', 'int64', 'bool'],
        op_type,
    )
    if len(x.shape) < 2:
        raise ValueError(f"x shape in {op_type} must be at least 2-D")
    diagonal = helper.kwargs.get('diagonal', 0)
    if not isinstance(diagonal, (int,)):
        raise TypeError(f"diagonal in {op_type} must be a python Int")
    name = helper.kwargs.get('name', None)

    if name is None:
        out = helper.create_variable_for_type_inference(dtype=x.dtype)
    else:
        out = helper.create_variable(
            name=name, dtype=x.dtype, persistable=False
        )

    helper.append_op(
        type="tril_triu",
        inputs={"X": x},
        attrs={
            "diagonal": diagonal,
            "lower": True if op_type == 'tril' else False,
        },
        outputs={"Out": out},
    )

    return out


def tril(x, diagonal=0, name=None):
    r"""
    Returns the lower triangular part of a matrix (2-D tensor) or batch
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
        name (str, optional): For details, please refer to :ref:`api_guide_Name`. Generally, no setting is required. Default: None.

    Returns:
        Tensor: Results of lower triangular operation by the specified diagonal of input tensor x,
        it's data type is the same as x's Tensor.

    Examples:
        .. code-block:: python

            import paddle

            data = paddle.arange(1, 13, dtype="int64").reshape([3,-1])
            # Tensor(shape=[3, 4], dtype=int64, place=Place(cpu), stop_gradient=True,
            #        [[1 , 2 , 3 , 4 ],
            #         [5 , 6 , 7 , 8 ],
            #         [9 , 10, 11, 12]])

            tril1 = paddle.tril(data)
            # Tensor(shape=[3, 4], dtype=int64, place=Place(cpu), stop_gradient=True,
            #        [[1 , 0 , 0 , 0 ],
            #         [5 , 6 , 0 , 0 ],
            #         [9 , 10, 11, 0 ]])

            # example 2, positive diagonal value
            tril2 = paddle.tril(data, diagonal=2)
            # Tensor(shape=[3, 4], dtype=int64, place=Place(cpu), stop_gradient=True,
            #        [[1 , 2 , 3 , 0 ],
            #         [5 , 6 , 7 , 8 ],
            #         [9 , 10, 11, 12]])

            # example 3, negative diagonal value
            tril3 = paddle.tril(data, diagonal=-1)
            # Tensor(shape=[3, 4], dtype=int64, place=Place(cpu), stop_gradient=True,
            #        [[0 , 0 , 0 , 0 ],
            #         [5 , 0 , 0 , 0 ],
            #         [9 , 10, 0 , 0 ]])
    """
    if in_dynamic_mode():
        return _C_ops.tril(x, diagonal)
    else:
        return _tril_triu_op(LayerHelper('tril', **locals()))


def triu(x, diagonal=0, name=None):
    r"""
    Return the upper triangular part of a matrix (2-D tensor) or batch of matrices
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
        name (str, optional): For details, please refer to :ref:`api_guide_Name`. Generally, no setting is required. Default: None.

    Returns:
        Tensor: Results of upper triangular operation by the specified diagonal of input tensor x,
        it's data type is the same as x's Tensor.

    Examples:
        .. code-block:: python

            import paddle

            x = paddle.arange(1, 13, dtype="int64").reshape([3,-1])
            # Tensor(shape=[3, 4], dtype=int64, place=Place(cpu), stop_gradient=True,
            #        [[1 , 2 , 3 , 4 ],
            #         [5 , 6 , 7 , 8 ],
            #         [9 , 10, 11, 12]])

            # example 1, default diagonal
            triu1 = paddle.tensor.triu(x)
            # Tensor(shape=[3, 4], dtype=int64, place=Place(cpu), stop_gradient=True,
            #        [[1 , 2 , 3 , 4 ],
            #         [0 , 6 , 7 , 8 ],
            #         [0 , 0 , 11, 12]])

            # example 2, positive diagonal value
            triu2 = paddle.tensor.triu(x, diagonal=2)
            # Tensor(shape=[3, 4], dtype=int64, place=Place(cpu), stop_gradient=True,
            #        [[0, 0, 3, 4],
            #         [0, 0, 0, 8],
            #         [0, 0, 0, 0]])

            # example 3, negative diagonal value
            triu3 = paddle.tensor.triu(x, diagonal=-1)
            # Tensor(shape=[3, 4], dtype=int64, place=Place(cpu), stop_gradient=True,
            #        [[1 , 2 , 3 , 4 ],
            #         [5 , 6 , 7 , 8 ],
            #         [0 , 10, 11, 12]])

    """
    if in_dynamic_mode():
        return _C_ops.triu(x, diagonal)
    else:
        return _tril_triu_op(LayerHelper('triu', **locals()))


def meshgrid(*args, **kwargs):
    """

    Takes a list of N tensors as input :attr:`*args`, each of which is 1-dimensional vector, and creates N-dimensional grids.

    Args:
        *args(Tensor|list of Tensor) : tensors (tuple(list) of tensor): the shapes of input k tensors are (N1,),
            (N2,),..., (Nk,). Support data types: ``float64``, ``float16``, ``float32``, ``int32``, ``int64``.
        **kwargs (optional): Currently, only accept name in **kwargs
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
    if in_dynamic_mode():
        return _C_ops.meshgrid(list(args))
    else:
        name = kwargs.get("name", None)
        helper = LayerHelper('meshgrid', **locals())

        if not isinstance(args, (list, tuple)):
            raise TypeError(
                "The type of input args in meshgrid should be list."
            )

        for id, input_ in enumerate(args):
            check_dtype(
                input_.dtype,
                'create data type',
                ['uint16', 'float16', 'float32', 'float64', 'int32', 'int64'],
                'meshgrid',
            )

        num = len(args)
        out = [
            helper.create_variable_for_type_inference(dtype=args[i].dtype)
            for i in range(num)
        ]
        helper.append_op(
            type='meshgrid', inputs={'X': list(args)}, outputs={'Out': out}
        )

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
        x (Tensor): The input tensor. It can be any shape. Its data type should be float16, float32, float64, int32, int64.
        offset (int, optional): The diagonal offset. A positive value represents superdiagonal, 0 represents the main diagonal, and a negative value represents subdiagonal. Default: 0 (main diagonal).
        name (str, optional): For details, please refer to :ref:`api_guide_Name`. Generally, no setting is required. Default: None.

    Returns:
        Tensor, a square matrix. The output data type is the same as input data type.

    Examples:
        .. code-block:: python
            :name: code-example-1

            import paddle

            x = paddle.to_tensor([1, 2, 3])
            y = paddle.diagflat(x)
            print(y)
            # Tensor(shape=[3, 3], dtype=int64, place=Place(cpu), stop_gradient=True,
            #        [[1, 0, 0],
            #         [0, 2, 0],
            #         [0, 0, 3]])

            y = paddle.diagflat(x, offset=1)
            print(y)
            # Tensor(shape=[4, 4], dtype=int64, place=Place(cpu), stop_gradient=True,
            #        [[0, 1, 0, 0],
            #         [0, 0, 2, 0],
            #         [0, 0, 0, 3],
            #         [0, 0, 0, 0]])

            y = paddle.diagflat(x, offset=-1)
            print(y)
            # Tensor(shape=[4, 4], dtype=int64, place=Place(cpu), stop_gradient=True,
            #        [[0, 0, 0, 0],
            #         [1, 0, 0, 0],
            #         [0, 2, 0, 0],
            #         [0, 0, 3, 0]])

        .. code-block:: python
            :name: code-example-2

            import paddle

            x = paddle.to_tensor([[1, 2], [3, 4]])
            y = paddle.diagflat(x)
            print(y)
            # Tensor(shape=[4, 4], dtype=int64, place=Place(cpu), stop_gradient=True,
            #        [[1, 0, 0, 0],
            #         [0, 2, 0, 0],
            #         [0, 0, 3, 0],
            #         [0, 0, 0, 4]])

            y = paddle.diagflat(x, offset=1)
            print(y)
            # Tensor(shape=[5, 5], dtype=int64, place=Place(cpu), stop_gradient=True,
            #        [[0, 1, 0, 0, 0],
            #         [0, 0, 2, 0, 0],
            #         [0, 0, 0, 3, 0],
            #         [0, 0, 0, 0, 4],
            #         [0, 0, 0, 0, 0]])

            y = paddle.diagflat(x, offset=-1)
            print(y)
            # Tensor(shape=[5, 5], dtype=int64, place=Place(cpu), stop_gradient=True,
            #        [[0, 0, 0, 0, 0],
            #         [1, 0, 0, 0, 0],
            #         [0, 2, 0, 0, 0],
            #         [0, 0, 3, 0, 0],
            #         [0, 0, 0, 4, 0]])
    """
    if in_dynamic_mode():
        if len(x.shape) <= 1:
            return _C_ops.diag(x, offset, 0)
        else:
            y = _C_ops.flatten(x, 0, -1)
            return _C_ops.diag(y, offset, 0)
    else:
        padding_value = 0
        check_type(x, 'x', (Variable), 'diagflat')
        check_dtype(
            x.dtype,
            'x',
            ['float16', 'float32', 'float64', 'int32', 'int64'],
            'diagflat',
        )
        check_type(offset, 'offset', (int), 'diagflat')

        helper = LayerHelper("diagflat", **locals())
        out1 = helper.create_variable_for_type_inference(dtype=x.dtype)
        out1_shape = helper.create_variable_for_type_inference(x.dtype)
        out2 = helper.create_variable_for_type_inference(dtype=x.dtype)

        if len(x.shape) <= 1:
            helper.append_op(
                type='diag_v2',
                inputs={'X': x},
                outputs={'Out': out2},
                attrs={'offset': offset, 'padding_value': padding_value},
            )
        else:
            helper.append_op(
                type='flatten_contiguous_range',
                inputs={'X': x},
                outputs={'Out': out1, 'XShape': out1_shape},
                attrs={'start_axis': 0, 'stop_axis': -1},
            )
            out1.stop_gradient = True

            helper.append_op(
                type='diag_v2',
                inputs={'X': out1},
                outputs={'Out': out2},
                attrs={'offset': offset, 'padding_value': padding_value},
            )
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
        x (Tensor): The input tensor. Its shape is either 1-D or 2-D. Its data type should be float16, float32, float64, int32, int64.
        offset (int, optional): The diagonal offset. A positive value represents superdiagonal, 0 represents the main diagonal, and a negative value represents subdiagonal.
        padding_value (int|float, optional): Use this value to fill the area outside the specified diagonal band. Only takes effect when the input is a 1-D Tensor. The default value is 0.
        name (str, optional): For details, please refer to :ref:`api_guide_Name`. Generally, no setting is required. Default: None.

    Returns:
        Tensor, a square matrix or a vector. The output data type is the same as input data type.

    Examples:
        .. code-block:: python
            :name: code-example-1

            import paddle

            paddle.disable_static()
            x = paddle.to_tensor([1, 2, 3])
            y = paddle.diag(x)
            print(y)
            # Tensor(shape=[3, 3], dtype=int64, place=Place(cpu), stop_gradient=True,
            #        [[1, 0, 0],
            #         [0, 2, 0],
            #         [0, 0, 3]])

            y = paddle.diag(x, offset=1)
            print(y)
            # Tensor(shape=[4, 4], dtype=int64, place=Place(cpu), stop_gradient=True,
            #        [[0, 1, 0, 0],
            #         [0, 0, 2, 0],
            #         [0, 0, 0, 3],
            #         [0, 0, 0, 0]])

            y = paddle.diag(x, padding_value=6)
            print(y)
            # Tensor(shape=[3, 3], dtype=int64, place=Place(cpu), stop_gradient=True,
            #        [[1, 6, 6],
            #         [6, 2, 6],
            #         [6, 6, 3]])

        .. code-block:: python
            :name: code-example-2

            import paddle

            paddle.disable_static()
            x = paddle.to_tensor([[1, 2, 3], [4, 5, 6]])
            y = paddle.diag(x)
            print(y)
            # Tensor(shape=[2], dtype=int64, place=Place(cpu), stop_gradient=True,
            #        [1, 5])

            y = paddle.diag(x, offset=1)
            print(y)
            # Tensor(shape=[2], dtype=int64, place=Place(cpu), stop_gradient=True,
            #        [2, 6])

            y = paddle.diag(x, offset=-1)
            print(y)
            # Tensor(shape=[1], dtype=int64, place=Place(cpu), stop_gradient=True,
            #        [4])
    """
    if in_dynamic_mode():
        return _C_ops.diag(x, offset, padding_value)
    else:
        check_type(x, 'x', (Variable), 'diag_v2')
        check_dtype(
            x.dtype,
            'x',
            ['float16', 'uint16', 'float32', 'float64', 'int32', 'int64'],
            'diag_v2',
        )
        check_type(offset, 'offset', (int), 'diag_v2')
        check_type(padding_value, 'padding_value', (int, float), 'diag_v2')
        if len(x.shape) != 1 and len(x.shape) != 2:
            raise ValueError(
                "The dimension of input x must be either 1 or 2, but received {}".format(
                    len(x.shape)
                )
            )

        helper = LayerHelper("diag_v2", **locals())

        out = helper.create_variable_for_type_inference(dtype=x.dtype)

        helper.append_op(
            type='diag_v2',
            inputs={'X': x},
            outputs={'Out': out},
            attrs={'offset': offset, 'padding_value': padding_value},
        )

        out.stop_gradient = True
        return out


def empty(shape, dtype=None, name=None):
    """
    Returns a Tensor with uninitialized data which size is same as ``shape``.

    Args:
        shape (tuple|list|Tensor): Shape of the Tensor to be created. The data type is ``int32`` or ``int64`` .
            If ``shape`` is a list or tuple, each element of it should be integer or 0-D Tensor with shape [].
            If ``shape`` is an Tensor, it should be an 1-D Tensor which represents a list.
        dtype(np.dtype|str, optional): Data type of the output Tensor
            which can be bool, float16, float32, float64, int32, int64, complex64, complex128 if dytpe is `None`, the data
            type of created Tensor use global default dtype (see ``get_default_dtype``
            for details).
        name(str, optional): For details, please refer to :ref:`api_guide_Name`. Generally, no setting is required. Default: None.

    Returns:
        Tensor: Tensor which is created according to ``shape`` and ``dtype``, and is uninitialized.

    Examples:
        .. code-block:: python

            import paddle

            # shape is a list/tuple
            data1 = paddle.empty(shape=[3, 2])
            # [[1. 1.]
            #  [1. 1.]
            #  [1. 1.]]

            # shape is a Tensor
            shape = paddle.to_tensor([3, 2])
            data2 = paddle.empty(shape=shape)
            # [[1. 1.]
            #  [1. 1.]
            #  [1. 1.]]

            # shape is a Tensor List
            shape = [paddle.to_tensor(3), paddle.to_tensor(2)]
            data3 = paddle.empty(shape=shape)
            # [[1. 1.]
            #  [1. 1.]
            #  [1. 1.]]
    """

    if dtype is None:
        dtype = paddle.get_default_dtype()

    dtype = convert_dtype(dtype)

    if in_dynamic_mode():
        shape = paddle.utils.convert_shape_to_list(shape)
        out = _C_ops.empty(
            shape, convert_np_dtype_to_dtype_(dtype), _current_expected_place()
        )
        out.stop_gradient = True
        return out
    else:
        helper = LayerHelper("empty", **locals())
        inputs = {}

        check_dtype(
            dtype,
            'dtype',
            [
                'bool',
                'float16',
                'float32',
                'float64',
                'int32',
                'int64',
                'complex64',
                'complex128',
            ],
            'empty',
        )
        check_type(shape, 'shape', (Variable, list, tuple), 'empty')

        if isinstance(shape, Variable):
            check_dtype(shape.dtype, 'shape', ['int32', 'int64'], 'empty')

        attrs = {}
        paddle.utils.get_shape_tensor_inputs(
            inputs=inputs, attrs=attrs, shape=shape, op_type='empty'
        )

        out = helper.create_variable_for_type_inference(dtype=dtype)
        attrs['dtype'] = convert_np_dtype_to_dtype_(dtype)
        helper.append_op(
            type='empty',
            inputs=inputs,
            outputs={'Out': [out]},
            attrs=attrs,
            stop_gradient=True,
        )
        out.stop_gradient = True
        return out


def empty_like(x, dtype=None, name=None):
    """
    Returns a Tensor with uninitialized data which has identical shape of ``x`` and ``dtype``.
    If the ``dtype`` is None, the data type of Tensor is same with ``x``.

    Args:
        x(Tensor): The input tensor which specifies shape and data type. The data type can be bool, float16, float32, float64, int32, int64.
        dtype(np.dtype|str, optional): The data type of output. The data type can be one
            of bool, float16, float32, float64, int32, int64. The default value is None, which means the output
            data type is the same as input.
        name(str, optional): For details, please refer to :ref:`api_guide_Name`. Generally, no setting is required. Default: None.

    Returns:
        Tensor: Tensor which is created according to ``x`` and ``dtype``, and is uninitialized.

    Examples:
        .. code-block:: python

          import paddle

          paddle.set_device("cpu")  # and use cpu device

          x = paddle.randn([2, 3], 'float32')
          output = paddle.empty_like(x)
          #[[1.8491974e+20 1.8037303e+28 1.7443726e+28]     # uninitialized
          # [4.9640171e+28 3.0186127e+32 5.6715899e-11]]    # uninitialized
    """

    if dtype is None:
        dtype = x.dtype
    dtype = convert_dtype(dtype)

    if in_dynamic_mode():
        out = _C_ops.empty(
            x.shape,
            convert_np_dtype_to_dtype_(dtype),
            _current_expected_place(),
        )
        out.stop_gradient = True
        return out
    else:
        helper = LayerHelper("empty_like", **locals())
        check_variable_and_dtype(
            x,
            'x',
            [
                'bool',
                'float16',
                'float32',
                'float64',
                'int32',
                'int64',
                'uint16',
            ],
            'empty_like',
        )
        check_dtype(
            dtype,
            'dtype',
            [
                'bool',
                'float16',
                'float32',
                'float64',
                'int32',
                'int64',
                'uint16',
            ],
            'empty_like',
        )
        out = helper.create_variable_for_type_inference(dtype=dtype)

        inputs = {}
        attrs = {}
        attrs['dtype'] = convert_np_dtype_to_dtype_(dtype)
        shape = paddle.shape(x)
        paddle.utils.get_shape_tensor_inputs(
            inputs=inputs, attrs=attrs, shape=shape, op_type='empty_like'
        )

        helper.append_op(
            type='empty',
            inputs=inputs,
            outputs={'Out': [out]},
            attrs=attrs,
            stop_gradient=True,
        )
        out.stop_gradient = True
        return out


def assign(x, output=None):
    """

    Copy value of the :attr:`x` to the :attr:`output`.

    Parameters:
        x (Tensor|np.ndarray|list|tuple|scalar): A Tensor, numpy ndarray, tuple/list of scalar,
            or scalar. Its data type can be float16, float32, float64, int32, int64 or bool. Note: the float64 data will be converted to float32 because of current platform protobuf
            data limitation.
        output (Tensor, optional): A Tensor. If :attr:`output` is None, a new Tensor will be created as :attr:`output`. Default: None.

    Returns:
        Tensor: A Tensor with the same shape, data type and value as :attr:`x`.

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
    # speed up
    if x is output and isinstance(x, Variable):
        return x

    input = x
    helper = LayerHelper('assign', **locals())
    check_type(
        input,
        'input',
        (Variable, np.ndarray, list, tuple, float, int, bool),
        'assign',
    )

    if np.isscalar(input) and not isinstance(input, str):
        input = np.array([input])
    elif isinstance(input, (list, tuple)):
        input = np.array(input)
    # NOTE(Aurelius84): Why we judge core.Tensor?
    # In case of @to_static, a Tensor can be as input of `assign`,
    # but in_dynamic_mode()==False under @to_static, which means
    # isinstance(Tensor, Variable) == False. It will cause return None
    # after this api.
    if isinstance(input, (Variable, core.eager.Tensor)):
        if in_dynamic_mode():
            if output is None:
                output = _C_ops.assign(input)
            else:
                _C_ops.assign_out_(input, output)
        else:
            check_dtype(
                input.dtype,
                'input',
                [
                    'float16',
                    'uint16',
                    'float32',
                    'float64',
                    'int32',
                    'int64',
                    'uint8',
                    'int8',
                    'bool',
                ],
                'assign',
                '(When the type of input in assign is Variable.)',
            )
            if output is None:
                output = helper.create_variable_for_type_inference(
                    dtype=input.dtype
                )
            helper.append_op(
                type='assign', inputs={'X': [input]}, outputs={'Out': [output]}
            )
    elif isinstance(input, np.ndarray):
        # We now support the form of [var, VAR...] if the Var.shape=[1,]
        if len(input.shape) > 0 and any(isinstance(x, Variable) for x in input):
            # We only deal with the case where the list is nested one level, convert all scalars into variables, and then use stack to process. It is necessary to ensure the consistency of types.
            if not all(
                [
                    x.shape == (1,)
                    for x in input
                    if isinstance(x, (Variable, core.eager.Tensor))
                ]
            ):
                raise TypeError(
                    "Unsupport paddle.assign([Variable, Variable...]) with non-scalar variable."
                )

            def convert_scalar(x):
                if not isinstance(x, (Variable, core.eager.Tensor)):
                    return assign(x)
                return x

            to_stack_list = list(map(convert_scalar, input))
            ret = paddle.stack(to_stack_list)
            ret = paddle.squeeze(ret, -1)
            return ret

        if input.dtype == 'object':
            """may be this form [[Var], [Var], [3], [4]], we reject them."""
            raise TypeError(
                "The type of received input == `object`, it is not supported to convert to tensor, such as [[Var], [Var], [3], [4]]"
            )

        dtype = convert_np_dtype_to_dtype_(input.dtype)
        if dtype == core.VarDesc.VarType.FP64:
            # Setting FP64 numpy data is not supported in Paddle, so we
            # use FP32 here
            warnings.warn(
                "paddle.assign doesn't support float64 input now due "
                "to current platform protobuf data limitation, we convert "
                "it to float32"
            )
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
                "received %s." % convert_dtype(dtype)
            )
        if input.size > 1024 * 1024:
            raise ValueError(
                "The size of input is too big. Please consider "
                "saving it to file and 'load_op' to load it"
            )
        if in_dynamic_mode():
            if output is None:
                output = zeros(list(input.shape), dtype)
            _C_ops.assign_value_(
                output,
                list(input.shape),
                dtype,
                values,
                _current_expected_place(),
            )
        else:
            if output is None:
                output = helper.create_variable_for_type_inference(
                    dtype=input.dtype
                )
            helper.append_op(
                type='assign_value',
                outputs={'Out': [output]},
                attrs={
                    'dtype': dtype,
                    'shape': list(input.shape),
                    value_name: values,
                },
            )

    return output


def clone(x, name=None):
    """
    Returns a copy of input Tensor. It will always have a Tensor copy.

    In addition, This function is derivable, so gradients will flow back from the output to input.

    Parameters:
        x (Tensor): The input Tensor.
        name(str, optional): For details, please refer to :ref:`api_guide_Name`. Generally, no setting is required. Default: None.

    Returns:
        Tensor, A Tensor copied from ``input``.

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


# NOTE(zhiqiu): not public
def _memcpy(input, place=None, output=None):
    """

    The OP copies the :attr:`input` to the :attr:`output`.
    NOTE: currently, only support CUDAPlace <-> CUDAPinnedPlace.

    Parameters:
        input (Tensor): A tensor. Its data type supports float16, float32, float64, int32, int64, and bool.
        device (Place): Target place for the output.
        output (Tensor, optional): A tensor. If :attr:`output` is None, a new tensor will
            be created as :attr:`output`. Default: None.

    Returns:
        Tensor, A tensor with the same shape, data type and value as :attr:`input`.

    Examples:
        .. code-block:: python

          import paddle

          data = paddle.full(shape=[3, 2], fill_value=2.5, dtype='float64') # [[2.5, 2.5], [2.5, 2.5], [2.5, 2.5]]
          result = paddle._memcpy(data, place=paddle.CPUPlace())  # result2 = [[2.5, 2.5], [2.5, 2.5], [2.5, 2.5]]
    """
    helper = LayerHelper('memcpy', **locals())
    check_type(input, 'input', (Variable), 'memcpy')

    if isinstance(input, (Variable, core.eager.Tensor)):
        check_dtype(
            input.dtype,
            'input',
            [
                'float16',
                'uint16',
                'float32',
                'float64',
                'int32',
                'int64',
                'uint8',
                'int8',
                'bool',
            ],
            'memcpy',
            '(When the type of input in memcpy is Variable.)',
        )
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

    attrs = {'dst_place_type': dst_place_type}
    helper.append_op(
        type='memcpy',
        inputs={'X': [input]},
        outputs={'Out': [output]},
        attrs=attrs,
    )
    return output


def complex(real, imag, name=None):
    """Return a compelx tensor given the real and image component.

    Args:
        real (Tensor): The real component. The data type should be 'float32' or 'float64'.
        imag (Tensor): The image component. The data type should be the same as ``real``.
        name (str, optional): For details, please refer to :ref:`api_guide_Name`. Generally, no setting is required. Default: None.

    Returns:
        Tensor: The output tensor. The data type is 'complex64' or 'complex128', with the same precision as ``real`` and ``imag``.

    Note:
        ``paddle.complex`` supports broadcasting. If you want know more about broadcasting, please refer to `Introduction to Tensor`_ .

        .. _Introduction to Tensor: ../../guides/beginner/tensor_en.html#chapter5-broadcasting-of-tensor

    Examples:
        .. code-block:: python

            import paddle
            x = paddle.arange(2, dtype=paddle.float32).unsqueeze(-1)
            y = paddle.arange(3, dtype=paddle.float32)
            z = paddle.complex(x, y)
            print(z)
            # Tensor(shape=[2, 3], dtype=complex64, place=Place(cpu), stop_gradient=True,
            #        [[0j    , 1j    , 2j    ],
            #         [(1+0j), (1+1j), (1+2j)]])
    """
    if in_dynamic_mode():
        return _C_ops.complex(real, imag)
    else:
        check_variable_and_dtype(
            real, 'real', ['float32', 'float64'], 'complex'
        )
        check_variable_and_dtype(
            imag, 'imag', ['float32', 'float64'], 'complex'
        )

        op_type = "complex"
        helper = LayerHelper(op_type, **locals())
        inputs = {"X": real, "Y": imag}
        out = helper.create_variable_for_type_inference(
            dtype=_real_to_complex_dtype(real.dtype)
        )
        outputs = {"Out": out}
        attrs = {}
        helper.append_op(
            type=op_type, inputs=inputs, attrs=attrs, outputs=outputs
        )
        return out


def tril_indices(row, col, offset=0, dtype='int64'):
    """
    Return the indices of the lower triangular part of the 2-D matrix
    whose row and col is knowed.Indices are ordered based on row and then columns.
    The lower triangular part of the matrix is defined as the elements on
    and below the diagonal.

    Args:
        row (int): The input x which is a int number describe the number of row of the matrix.
        col (int): The input x which is a int number describe the number of col of the matrix.
        offset (int, optional): The offset to consider, default value is 0.

            - If offset = 0, all elements on and below the main diagonal are retained.
            - If offset > 0, include just as many diagonals above the main diagonal.
            - If offset < 0, excludes just as many diagonals below the main diagonal.

        dtype (int, optional): the data type of the output tensor, can be int32, int64.

    Returns:
        Tensor: Results of the indices of lower triangular part of a row * col matrix,
        where the first row contains row coordinates of and the second row contains column coordinates.

    Examples:
        .. code-block:: python

            import paddle

            # example 1, default offset value
            data1 = paddle.tril_indices(4,4,0)
            print(data1)
            # [[0, 1, 1, 2, 2, 2, 3, 3, 3, 3],
            #  [0, 0, 1, 0, 1, 2, 0, 1, 2, 3]]

            # example 2, positive offset value
            data2 = paddle.tril_indices(4,4,2)
            print(data2)
            # [[0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3],
            #  [0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3]]

            # example 3, negative offset value
            data3 = paddle.tril_indices(4,4,-1)
            print(data3)
            # [[ 1, 2, 2, 3, 3, 3],
            #  [ 0, 0, 1, 0, 1, 2]]
    """
    if not isinstance(dtype, core.VarDesc.VarType):
        dtype = convert_np_dtype_to_dtype_(dtype)

    if in_dynamic_mode():
        if col is None:
            col = row
        out = _C_ops.tril_indices(
            row, col, offset, dtype, _current_expected_place()
        )
        return out
    else:
        if not isinstance(row, int) or row < 0:
            raise TypeError("row should be a non-negative int")

        if col is not None:
            if not isinstance(col, int) or col < 0:
                raise TypeError("col should be a non-negative int")
        else:
            col = row

        if not isinstance(offset, int):
            raise TypeError("offset should be a  int")

        helper = LayerHelper("tril_indices", **locals())

        out = helper.create_variable_for_type_inference(dtype=dtype)

        helper.append_op(
            type='tril_indices',
            inputs={},
            outputs={'out': [out]},
            attrs={'rows': row, 'cols': col, 'offset': offset, 'dtype': dtype},
        )
    return out


def triu_indices(row, col=None, offset=0, dtype='int64'):
    """
    Return the indices of the upper triangular part of the 2-D matrix
    whose row and col is known. Indices are ordered based on row and then columns.
    The upper triangular part of the matrix is defined as the elements on
    and above the diagonal.

    Args:
        row (int): The input x which is a int number describe the number of row of the matrix.
        col (int, optional): The input x which is a int number describe the number of col of the matrix.
            default value for col is None, then it will be set equal to row, indicting a square matix.
        offset (int, optional): The offset to consider, default value is 0.

            - If offset = 0, all elements on and above the main diagonal are retained.
            - If offset > 0, include just as few diagonals above the main diagonal.
            - If offset < 0, excludes just as few diagonals below the main diagonal.

        dtype (str|np.dtype|paddle.dtype, optional): the data type of the output tensor,
            can be int32, int64, default value is int64.
    Returns:
        Tensor: Results of the indices of upper triangular part of a row * col matrix,
        where the first row contains row coordinates of and the second row contains column coordinates.

    Examples:
        .. code-block:: python

            import paddle
            # example 1, default offset value
            data1 = paddle.triu_indices(4,4,0)
            print(data1)
            # [[0, 0, 0, 0, 1, 1, 1, 2, 2, 3],
            #  [0, 1, 2, 3, 1, 2, 3, 2, 3, 3]]
            # example 2, positive offset value
            data2 = paddle.triu_indices(4,4,2)
            print(data2)
            # [[0, 0, 1],
            #  [2, 3, 3]]
            # example 3, negative offset value
            data3 = paddle.triu_indices(4,4,-1)
            print(data3)
            # [[0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 3, 3],
            #  [0, 1, 2, 3, 0, 1, 2, 3, 1, 2, 3, 2, 3]]
    """
    if not isinstance(dtype, core.VarDesc.VarType):
        dtype = convert_np_dtype_to_dtype_(dtype)

    if in_dynamic_mode():
        if col is None:
            col = row
        out = _C_ops.triu_indices(
            row, col, offset, dtype, _current_expected_place()
        )
        return out
    else:
        if not isinstance(row, int) or row < 0:
            raise TypeError("row should be a non-negative int")

        if col is not None:
            if not isinstance(col, int) or col < 0:
                raise TypeError("col should be a non-negative int")
        else:
            col = row

        if not isinstance(offset, int):
            raise TypeError("offset should be a int")

        helper = LayerHelper("triu_indices", **locals())

        out = helper.create_variable_for_type_inference(dtype=dtype)

        helper.append_op(
            type='triu_indices',
            inputs={},
            outputs={'out': [out]},
            attrs={'row': row, 'col': col, 'offset': offset, 'dtype': dtype},
        )
    return out


def polar(abs, angle, name=None):
    """Return a Cartesian coordinates corresponding to the polar coordinates compelx tensor given the ``abs`` and ``angle`` component.

    Args:
        abs (Tensor): The abs component. The data type should be 'float32' or 'float64'.
        angle (Tensor): The anglee component. The data type should be the same as ``abs``.
        name (str, optional): For details, please refer to :ref:`api_guide_Name`. Generally, no setting is required. Default: None.

    Returns:
        Tensor: The output tensor. The data type is 'complex64' or 'complex128', with the same precision as ``abs`` and ``angle``.

    Note:
        ``paddle.polar`` supports broadcasting. If you want know more about broadcasting, please refer to `Introduction to Tensor`_ .

        .. _Introduction to Tensor: ../../guides/beginner/tensor_en.html#chapter5-broadcasting-of-tensor

    Examples:
        .. code-block:: python

            import paddle
            import numpy as np

            abs = paddle.to_tensor([1, 2], dtype=paddle.float64)
            angle = paddle.to_tensor([np.pi / 2, 5 * np.pi / 4], dtype=paddle.float64)
            out = paddle.polar(abs, angle)
            print(out)
            # Tensor(shape=[2], dtype=complex128, place=Place(cpu), stop_gradient=True,
            #       [ (6.123233995736766e-17+1j) ,
            #       (-1.4142135623730954-1.414213562373095j)])
    """
    check_variable_and_dtype(abs, 'abs', ['float32', 'float64'], 'paddle.polar')
    check_variable_and_dtype(
        angle, 'angle', ['float32', 'float64'], 'paddle.polar'
    )

    return paddle.complex(abs * paddle.cos(angle), abs * paddle.sin(angle))
