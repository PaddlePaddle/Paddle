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

from ..fluid.framework import Variable
from ..fluid.framework import unique_name
from ..fluid.framework import _current_expected_place
from ..fluid.framework import dygraph_only
from ..fluid.initializer import Constant
from ..fluid.layers import core
from ..fluid.layer_helper import LayerHelper
from ..fluid.data_feeder import check_variable_and_dtype, check_type, check_dtype, convert_dtype
from ..fluid.framework import convert_np_dtype_to_dtype_, in_dygraph_mode, _varbase_creator, device_guard, OpProtoHolder
from ..fluid.layers import fill_constant
from paddle.common_ops_import import *

# TODO: define functions to get create a tensor  
from ..fluid.layers import crop_tensor  #DEFINE_ALIAS
from ..fluid.layers import fill_constant  #DEFINE_ALIAS
from ..fluid.layers import linspace  #DEFINE_ALIAS
import paddle

__all__ = [
    'to_tensor',
    'crop_tensor',
    'diag',
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


@dygraph_only
def to_tensor(data, dtype=None, place=None, stop_gradient=True):
    """
    Constructs a ``paddle.Tensor`` or ``paddle.ComplexTensor`` from ``data`` , 
    which can be scalar, tuple, list, numpy\.ndarray, paddle\.Tensor, paddle\.ComplexTensor.

    If the ``data`` is already a tensor, and ``dtype`` or ``place`` does't change, no copy 
    will be performed and return origin tensor, otherwise a new tensor will be constructed
    and returned. Similarly, if the data is an numpy\.ndarray of with the same ``dtype`` 
    and the current place is cpu, no copy will be performed.

    The ``ComplexTensor`` is a unique type of paddle. If x is ``ComplexTensor``, then 
    ``x.real`` is the real part, and ``x.imag`` is the imaginary part.

    Args:
        data(scalar|tuple|list|ndarray|Tensor|ComplexTensor): Initial data for the tensor.
            Can be a scalar, list, tuple, numpy\.ndarray, paddle\.Tensor, paddle\.ComplexTensor.
        dtype(str|np.dtype, optional): The desired data type of returned tensor. Can be 'bool' , 'float16' , 
            'float32' , 'float64' , 'int8' , 'int16' , 'int32' , 'int64' , 'uint8'. And
            'complex64' , 'complex128' only for ComplexTensor. Default: None, infers dtype from ``data`` 
            except for python float number which gets dtype from ``get_default_type`` .
        place(CPUPlace|CUDAPinnedPlace|CUDAPlace, optional): The place to allocate Tensor. Can be  
            CPUPlace, CUDAPinnedPlace, CUDAPlace. Default: None, means global place.
        stop_gradient(bool, optional): Whether to block the gradient propagation of Autograd. Default: True.

    Returns:
        Tensor: A Tensor or ComplexTensor constructed from ``data`` .

    Raises:
        TypeError: If the data type of ``data`` is not scalar, list, tuple, numpy.ndarray, paddle.Tensor, paddle.ComplexTensor
        ValueError: If ``data`` is tuple|list, it can't contain nested tuple|list with different lengths , such as: [[1, 2], [3, 4, 5]]
        TypeError: If ``dtype`` is not bool, float16, float32, float64, int8, int16, int32, int64, uint8, complex64, complex128
        ValueError: If ``place`` is not paddle.CPUPlace, paddle.CUDAPinnedPlace, paddle.CUDAPlace

    Examples:

    .. code-block:: python

        import paddle
        import numpy as np
        paddle.disable_static()
                
        type(paddle.to_tensor(1))
        # <class 'paddle.Tensor'>

        paddle.to_tensor(1)
        # Tensor: generated_tensor_0
        # - place: CUDAPlace(0)   # allocate on global default place CPU:0
        # - shape: [1]
        # - layout: NCHW
        # - dtype: int64_t
        # - data: [1]

        x = paddle.to_tensor(1)
        paddle.to_tensor(x, dtype='int32', place=paddle.CPUPlace()) # A new tensor will be constructed due to different dtype or place
        # Tensor: generated_tensor_01
        # - place: CPUPlace
        # - shape: [1]
        # - layout: NCHW
        # - dtype: int
        # - data: [1]

        paddle.to_tensor((1.1, 2.2), place=paddle.CUDAPinnedPlace())
        # Tensor: generated_tensor_1
        #   - place: CUDAPinnedPlace
        #   - shape: [2]
        #   - layout: NCHW
        #   - dtype: double
        #   - data: [1.1 2.2]

        paddle.to_tensor([[0.1, 0.2], [0.3, 0.4]], place=paddle.CUDAPlace(0), stop_gradient=False)
        # Tensor: generated_tensor_2
        #   - place: CUDAPlace(0)
        #   - shape: [2, 2]
        #   - layout: NCHW
        #   - dtype: double
        #   - data: [0.1 0.2 0.3 0.4]

        type(paddle.to_tensor([[1+1j, 2], [3+2j, 4]]), dtype='complex64')
        # <class 'paddle.ComplexTensor'>

        paddle.to_tensor([[1+1j, 2], [3+2j, 4]], dtype='complex64')
        # ComplexTensor[real]: generated_tensor_0.real
        #   - place: CUDAPlace(0)
        #   - shape: [2, 2]
        #   - layout: NCHW
        #   - dtype: float
        #   - data: [1 2 3 4]
        # ComplexTensor[imag]: generated_tensor_0.imag
        #   - place: CUDAPlace(0)
        #   - shape: [2, 2]
        #   - layout: NCHW
        #   - dtype: float
        #   - data: [1 0 2 0]
    """

    if place is None:
        place = _current_expected_place()
    elif not isinstance(place,
                        (core.CPUPlace, core.CUDAPinnedPlace, core.CUDAPlace)):
        raise ValueError(
            "'place' must be any of paddle.Place, paddle.CUDAPinnedPlace, paddle.CUDAPlace"
        )

    #Todo(zhouwei): Support allocate tensor on any other specified card
    if isinstance(place, core.CUDAPlace) and isinstance(
            _current_expected_place(), core.CUDAPlace) and place._get_device_id(
            ) != _current_expected_place()._get_device_id():
        place = _current_expected_place()

    if not isinstance(data, np.ndarray):
        if np.isscalar(data) and not isinstance(data, str):
            data = np.array([data])
        elif isinstance(data, (list, tuple)):
            data = np.array(data)
            if data.dtype == np.object:
                raise ValueError(
                    "\n\tFaild to convert input data to a regular ndarray :\n\t - Usually "
                    "this means the input data contains nested lists with different lengths. "
                )
        elif isinstance(data, paddle.Tensor):
            data.stop_gradient = stop_gradient
            if not data.place._equals(place):
                data = data._copy_to(place, False)
            if dtype:
                if convert_dtype(dtype) != convert_dtype(data.dtype):
                    return data.astype(convert_dtype(dtype))
            return data
        elif isinstance(data, paddle.ComplexTensor):
            return data
        else:
            raise TypeError(
                "Can't constructs a 'paddle.Tensor' with data type {}, data type must be scalar|list|tuple|numpy.ndarray|paddle.Tensor|paddle.ComplexTensor".
                format(type(data)))
        if not dtype and data.dtype in [
                'float16', 'float32', 'float64', 'complex64', 'complex128'
        ]:
            default_type = paddle.get_default_dtype()
            if np.iscomplexobj(data):
                default_type = 'complex64' if default_type in [
                    'float16', 'float32'
                ] else 'complex128'
            data = data.astype(default_type)

    if dtype and convert_dtype(dtype) != data.dtype:
        data = data.astype(dtype)

    if not np.iscomplexobj(data):
        if dtype and convert_dtype(dtype) != data.dtype:
            data = data.astype(dtype)
        return paddle.Tensor(
            value=data,
            place=place,
            persistable=False,
            zero_copy=True,
            stop_gradient=stop_gradient)
    else:
        name = unique_name.generate('generated_tensor')
        real_tensor = paddle.Tensor(
            value=data.real,
            place=place,
            zero_copy=True,
            name=name + ".real",
            stop_gradient=stop_gradient)
        imag_tensor = paddle.Tensor(
            value=data.imag,
            place=place,
            zero_copy=True,
            name=name + ".imag",
            stop_gradient=stop_gradient)
        return paddle.ComplexTensor(real_tensor, imag_tensor)


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
          
          paddle.disable_static()  # Now we are in imperative mode 
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
        return core.ops.fill_any_like(x, 'value', fill_value, 'dtype', dtype)

    helper = LayerHelper("full_like", **locals())
    check_variable_and_dtype(
        x, 'x', ['bool', 'float16', 'float32', 'float64', 'int32', 'int64'],
        'full_like')
    check_dtype(dtype, 'dtype',
                ['bool', 'float16', 'float32', 'float64', 'int32', 'int64'],
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
          paddle.disable_static()
          
          # default dtype for ones OP
          data1 = paddle.ones(shape=[3, 2]) 
          # [[1. 1.]
          #  [1. 1.]
          #  [1. 1.]]
          
          data2 = paddle.ones(shape=[2, 2], dtype='int32') 
          # [[1 1]
          #  [1 1]]
          
          # shape is a Tensor
          shape = paddle.fill_constant(shape=[2], dtype='int32', value=2)
          data3 = paddle.ones(shape=shape, dtype='int32') 
          # [[1 1]
          #  [1 1]]
    """
    if dtype is None:
        dtype = 'float32'
    return fill_constant(value=1.0, shape=shape, dtype=dtype, name=name)


def ones_like(x, dtype=None, name=None):
    """
	:alias_main: paddle.ones_like
	:alias: paddle.tensor.ones_like, paddle.tensor.creation.ones_like

    This OP returns a Tensor filled with the value 1, with the same shape and
    data type (use ``dtype`` if ``dtype`` is not None) as ``x``.

    Args:
        x(Tensor): The input tensor which specifies shape and dtype. The
            dtype of ``x`` can be bool, float16, float32, float64, int32, int64.
        dtype(str|np.dtype|core.VarDesc.VarType, optional): The data type of the
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

            paddle.disable_static()

            x = paddle.to_tensor([1,2,3])
            out1 = paddle.zeros_like(x) # [1., 1., 1.]
            out2 = paddle.zeros_like(x, dtype='int32') # [1, 1, 1]

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
          
          paddle.disable_static()  # Now we are in imperative mode
          data = paddle.zeros(shape=[3, 2], dtype='float32') 
          # [[0. 0.]
          #  [0. 0.]
          #  [0. 0.]]
          data = paddle.zeros(shape=[2, 2]) 
          # [[0. 0.]
          #  [0. 0.]]
          
          # shape is a Tensor
          shape = paddle.fill_constant(shape=[2], dtype='int32', value=2)
          data3 = paddle.zeros(shape=shape, dtype='int32') 
          # [[0 0]
          #  [0 0]]
    """
    if dtype is None:
        dtype = 'float32'
    return fill_constant(value=0.0, shape=shape, dtype=dtype, name=name)


def zeros_like(x, dtype=None, name=None):
    """
	:alias_main: paddle.zeros_like
	:alias: paddle.tensor.zeros_like, paddle.tensor.creation.zeros_like

    This OP returns a Tensor filled with the value 0, with the same shape and
    data type (use ``dtype`` if ``dtype`` is not None) as ``x``.

    Args:
        x(Tensor): The input tensor which specifies shape and dtype. The
            dtype of ``x`` can be bool, float16, float32, float64, int32, int64.
        dtype(str|np.dtype|core.VarDesc.VarType, optional): The data type of the
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

            paddle.disable_static()

            x = paddle.to_tensor([1,2,3])
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

          paddle.disable_static()  # Now we are in imperative mode
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
    return paddle.fluid.layers.eye(num_rows=num_rows,
                                   num_columns=num_columns,
                                   batch_shape=None,
                                   dtype=dtype,
                                   name=name)


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

          paddle.disable_static()  # Now we are in imperative mode
          data1 = paddle.full(shape=[2,1], fill_value=0, dtype='int64') 
          #[[0]
          # [0]]

          # attr shape is a list which contains Tensor.
          positive_2 = paddle.fill_constant([1], "int32", 2)
          data3 = paddle.full(shape=[1, positive_2], dtype='float32', fill_value=1.5)
          # [[1.5 1.5]]

          # attr shape is a Tensor.
          shape = paddle.fill_constant([2], "int32", 2)
          data4 = paddle.full(shape=shape, dtype='bool', fill_value=True) 
          # [[True True] 
          #  [True True]]
          
          # attr fill_value is a Tensor.
          val = paddle.fill_constant([1], "float32", 2.0)
          data5 = paddle.full(shape=[2,1], fill_value=val, dtype='float32')
          # [[2.0] 
          #  [2.0]]
    """

    if dtype is None:
        dtype = 'float32'

    return fill_constant(shape=shape, dtype=dtype, value=fill_value, name=name)


def arange(start=0, end=None, step=1, dtype=None, name=None):
    """
	:alias_main: paddle.arange
	:alias: paddle.tensor.arange, paddle.tensor.creation.arange

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
        dtype(str|np.dtype|core.VarDesc.VarType, optional): The data type of the
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

    examples:

        .. code-block:: python

        import paddle

        paddle.disable_static()

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

    return paddle.fluid.layers.range(start, end, step, dtype, name)


def _tril_triu_op(helper):
    """Base op of tril_op and triu_op
    """
    op_type = helper.layer_type
    x = helper.kwargs.get('x', None)

    assert x is not None, 'x cannot be None in {}'.format(op_type)
    check_variable_and_dtype(x, 'x', ['float32', 'float64', 'int32', 'int64'],
                             op_type)
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
    """
	:alias_main: paddle.tril
	:alias: paddle.tril,paddle.tensor.tril,paddle.tensor.creation.tril

    This op returns the lower triangular part of a matrix (2-D tensor) or batch
    of matrices :attr:`x`, the other elements of the result tensor are set 
    to 0. The lower triangular part of the matrix is defined as the elements 
    on and below the diagonal.

    Args:
        x (Variable): The input variable x which is a Tensor.
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
        Variable: Tensor, results of lower triangular operation by the specified diagonal of input tensor x,
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

            paddle.disable_static()

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
        op = getattr(core.ops, 'tril_triu')
        return op(x, 'diagonal', diagonal, "lower", True)

    return _tril_triu_op(LayerHelper('tril', **locals()))


def triu(x, diagonal=0, name=None):
    """
	:alias_main: paddle.triu
	:alias: paddle.triu,paddle.tensor.triu,paddle.tensor.creation.triu

    This op returns the upper triangular part of a matrix (2-D tensor) or batch of matrices
    :attr:`x`, the other elements of the result tensor are set to 0.
    The upper triangular part of the matrix is defined as the elements on and
    above the diagonal.

    Args:
        x (Variable): The input variable x which is a Tensor.
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
        Variable: Tensor, results of upper triangular operation by the specified diagonal of input tensor x,
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

            paddle.disable_static()

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
        op = getattr(core.ops, 'tril_triu')
        return op(x, 'diagonal', diagonal, "lower", False)

    return _tril_triu_op(LayerHelper('triu', **locals()))


def meshgrid(*args, **kwargs):
    """
	:alias_main: paddle.meshgrid
	:alias: paddle.meshgrid,paddle.tensor.meshgrid,paddle.tensor.creation.meshgrid

    This op takes a list of N tensors as input *args, each of which is 1-dimensional 
    vector, and creates N-dimensional grids.
    
    Args:
        *args(Variable|list of Variable) : tensors (tuple(list) of tensor): the shapes of input k tensors are (N1,), 
            (N2,),..., (Nk,). Support data types: ``float64``, ``float32``, ``int32``, ``int64``.
        **kwargs (optional): Currently, we only accept name in **kwargs 
            The default value is None. Normally there is no need for
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
          grid_x, grid_y = paddle.tensor.meshgrid(x, y)
          res_1, res_2 = exe.run(fluid.default_main_program(),
                                 feed={'x': input_1,
                                       'y': input_2},
                                 fetch_list=[grid_x, grid_y])
     
          #the shape of res_1 is (100, 200)
          #the shape of res_2 is (100, 200)

      .. code-block:: python

          #example 2: in dygraph mode

          import paddle
          import numpy as np
          
          paddle.disable_static()

          input_3 = np.random.randint(0, 100, [100, ]).astype('int32')
          input_4 = np.random.randint(0, 100, [200, ]).astype('int32')
          tensor_3 = paddle.to_tensor(input_3)
          tensor_4 = paddle.to_tensor(input_4)
          grid_x, grid_y = paddle.tensor.meshgrid(tensor_3, tensor_4)

          #the shape of grid_x is (100, 200)
          #the shape of grid_y is (100, 200)

    """

    if len(args) == 1 and isinstance(args[0], (list, tuple)):
        args = args[0]
    if in_dygraph_mode():
        num = len(args)
        out = core.ops.meshgrid(list(args), num)
        return out

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


def diag(x, offset=0, padding_value=0, name=None):
    """
    If ``x`` is a vector (1-D tensor), a 2-D square tensor whth the elements of ``x`` as the diagonal is returned.

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
        return core.ops.diag_v2(x, "offset", offset, "padding_value",
                                padding_value)

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
