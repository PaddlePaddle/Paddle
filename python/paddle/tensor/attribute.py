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

# TODO: define functions to get tensor attributes

import numpy as np

import paddle
from paddle import _C_ops

from ..base.data_feeder import check_type, check_variable_and_dtype
from ..base.framework import in_dynamic_or_pir_mode, use_pir_api
from ..common_ops_import import Variable
from ..framework import LayerHelper, core
from .creation import _complex_to_real_dtype, assign

__all__ = []


def rank(input):
    """

    Returns the number of dimensions for a tensor, which is a 0-D int32 Tensor.

    Args:
        input (Tensor): The input Tensor with shape of :math:`[N_1, N_2, ..., N_k]`, the data type is arbitrary.

    Returns:
        Tensor, the output data type is int32.: The 0-D tensor with the dimensions of the input Tensor.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> input = paddle.rand((3, 100, 100))
            >>> rank = paddle.rank(input)
            >>> print(rank.numpy())
            3
    """
    check_type(input, 'input', (Variable, paddle.pir.Value), 'input')
    ndims = len(input.shape)
    out = assign(np.array(ndims, 'int32'))

    return out


def shape(input):
    """
    Get the shape of the input.

    .. code-block:: text

        Case1:
            Given N-D Tensor:
                input = [ [1, 2, 3, 4], [5, 6, 7, 8] ]

            Then:
                input.shape = [2, 4]

        Case2:
            Given SelectedRows:
                input.rows = [0, 4, 19]
                input.height = 20
                input.value = [ [1, 2], [3, 4], [5, 6] ]  # inner tensor
            Then:
                input.shape = [3, 2]

    Args:
        input (Variable): The input can be N-D Tensor or SelectedRows with data type bool, bfloat16, float16, float32, float64, int32, int64.
                          If input variable is type of SelectedRows, returns the shape of it's inner tensor.

    Returns:
        Variable (Tensor): The shape of the input variable.

    Examples:
        .. code-block:: python

            >>> import numpy as np
            >>> import paddle
            >>> paddle.enable_static()

            >>> inputs = paddle.static.data(name="x", shape=[3, 100, 100], dtype="float32")
            >>> output = paddle.shape(inputs)

            >>> exe = paddle.static.Executor(paddle.CPUPlace())
            >>> exe.run(paddle.static.default_startup_program())

            >>> img = np.ones((3, 100, 100)).astype(np.float32)

            >>> res = exe.run(paddle.static.default_main_program(), feed={'x':img}, fetch_list=[output])
            >>> print(res)
            [array([  3, 100, 100], dtype=int32)]
    """
    if in_dynamic_or_pir_mode():
        out = _C_ops.shape(input)
        out.stop_gradient = True
        return out
    else:
        check_variable_and_dtype(
            input,
            'input',
            [
                'bool',
                'uint16',
                'float16',
                'float32',
                'float64',
                'int32',
                'int64',
                'complex64',
                'complex128',
                'uint16',
            ],
            'shape',
        )
        helper = LayerHelper('shape', **locals())
        out = helper.create_variable_for_type_inference(dtype='int32')
        helper.append_op(
            type='shape',
            inputs={'Input': input},
            outputs={'Out': out},
            stop_gradient=True,
        )
        out.stop_gradient = True
        return out


def is_complex(x):
    """Return whether x is a tensor of complex data type(complex64 or complex128).

    Args:
        x (Tensor): The input tensor.

    Returns:
        bool: True if the data type of the input is complex data type, otherwise false.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> x = paddle.to_tensor([1 + 2j, 3 + 4j])
            >>> print(paddle.is_complex(x))
            True

            >>> x = paddle.to_tensor([1.1, 1.2])
            >>> print(paddle.is_complex(x))
            False

            >>> x = paddle.to_tensor([1, 2, 3])
            >>> print(paddle.is_complex(x))
            False
    """
    if not isinstance(
        x, (paddle.Tensor, paddle.static.Variable, paddle.pir.Value)
    ):
        raise TypeError(f"Expected Tensor, but received type of x: {type(x)}")
    dtype = x.dtype
    is_complex_dtype = (
        dtype == core.VarDesc.VarType.COMPLEX64
        or dtype == core.VarDesc.VarType.COMPLEX128
        or dtype == core.DataType.COMPLEX64
        or dtype == core.DataType.COMPLEX128
    )
    return is_complex_dtype


def is_floating_point(x):
    """
    Returns whether the dtype of `x` is one of paddle.float64, paddle.float32, paddle.float16, and paddle.bfloat16.

    Args:
        x (Tensor): The input tensor.

    Returns:
        bool: True if the dtype of `x` is floating type, otherwise false.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> x = paddle.arange(1., 5., dtype='float32')
            >>> y = paddle.arange(1, 5, dtype='int32')
            >>> print(paddle.is_floating_point(x))
            True
            >>> print(paddle.is_floating_point(y))
            False
    """
    if not isinstance(
        x, (paddle.Tensor, paddle.static.Variable, paddle.pir.Value)
    ):
        raise TypeError(f"Expected Tensor, but received type of x: {type(x)}")
    dtype = x.dtype
    is_fp_dtype = (
        dtype == core.VarDesc.VarType.FP32
        or dtype == core.VarDesc.VarType.FP64
        or dtype == core.VarDesc.VarType.FP16
        or dtype == core.VarDesc.VarType.BF16
        or dtype == core.DataType.FLOAT32
        or dtype == core.DataType.FLOAT64
        or dtype == core.DataType.FLOAT16
        or dtype == core.DataType.BFLOAT16
    )
    return is_fp_dtype


def is_integer(x):
    """Return whether x is a tensor of integral data type.

    Args:
        x (Tensor): The input tensor.

    Returns:
        bool: True if the data type of the input is integer data type, otherwise false.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> x = paddle.to_tensor([1 + 2j, 3 + 4j])
            >>> print(paddle.is_integer(x))
            False

            >>> x = paddle.to_tensor([1.1, 1.2])
            >>> print(paddle.is_integer(x))
            False

            >>> x = paddle.to_tensor([1, 2, 3])
            >>> print(paddle.is_integer(x))
            True
    """
    if not isinstance(
        x, (paddle.Tensor, paddle.static.Variable, paddle.pir.Value)
    ):
        raise TypeError(f"Expected Tensor, but received type of x: {type(x)}")
    dtype = x.dtype

    is_int_dtype = False
    if not use_pir_api():
        is_int_dtype = (
            dtype == core.VarDesc.VarType.UINT8
            or dtype == core.VarDesc.VarType.INT8
            or dtype == core.VarDesc.VarType.INT16
            or dtype == core.VarDesc.VarType.INT32
            or dtype == core.VarDesc.VarType.INT64
        )
    else:
        is_int_dtype = (
            dtype == core.DataType.UINT8
            or dtype == core.DataType.INT8
            or dtype == core.DataType.INT16
            or dtype == core.DataType.INT32
            or dtype == core.DataType.INT64
        )

    return is_int_dtype


def real(x, name=None):
    """
    Returns a new Tensor containing real values of the input Tensor.

    Args:
        x (Tensor): the input Tensor, its data type could be complex64 or complex128.
        name (str, optional): The default value is None. Normally there is no need for
            user to set this property. For more information, please refer to :ref:`api_guide_Name` .

    Returns:
        Tensor: a Tensor containing real values of the input Tensor.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> x = paddle.to_tensor(
            ...     [[1 + 6j, 2 + 5j, 3 + 4j], [4 + 3j, 5 + 2j, 6 + 1j]])
            >>> print(x)
            Tensor(shape=[2, 3], dtype=complex64, place=Place(cpu), stop_gradient=True,
            [[(1+6j), (2+5j), (3+4j)],
             [(4+3j), (5+2j), (6+1j)]])

            >>> real_res = paddle.real(x)
            >>> print(real_res)
            Tensor(shape=[2, 3], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[1., 2., 3.],
             [4., 5., 6.]])

            >>> real_t = x.real()
            >>> print(real_t)
            Tensor(shape=[2, 3], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[1., 2., 3.],
             [4., 5., 6.]])
    """
    if in_dynamic_or_pir_mode():
        return _C_ops.real(x)
    else:
        check_variable_and_dtype(x, 'x', ['complex64', 'complex128'], 'real')
        helper = LayerHelper('real', **locals())
        out = helper.create_variable_for_type_inference(
            dtype=_complex_to_real_dtype(helper.input_dtype())
        )
        helper.append_op(type='real', inputs={'X': x}, outputs={'Out': out})
        return out


def imag(x, name=None):
    """
    Returns a new tensor containing imaginary values of input tensor.

    Args:
        x (Tensor): the input tensor, its data type could be complex64 or complex128.
        name (str, optional): The default value is None. Normally there is no need for
            user to set this property. For more information, please refer to :ref:`api_guide_Name` .

    Returns:
        Tensor: a tensor containing imaginary values of the input tensor.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> x = paddle.to_tensor(
            ...     [[1 + 6j, 2 + 5j, 3 + 4j], [4 + 3j, 5 + 2j, 6 + 1j]])
            >>> print(x)
            Tensor(shape=[2, 3], dtype=complex64, place=Place(cpu), stop_gradient=True,
            [[(1+6j), (2+5j), (3+4j)],
             [(4+3j), (5+2j), (6+1j)]])

            >>> imag_res = paddle.imag(x)
            >>> print(imag_res)
            Tensor(shape=[2, 3], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[6., 5., 4.],
             [3., 2., 1.]])

            >>> imag_t = x.imag()
            >>> print(imag_t)
            Tensor(shape=[2, 3], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[6., 5., 4.],
             [3., 2., 1.]])
    """
    if in_dynamic_or_pir_mode():
        return _C_ops.imag(x)
    else:
        check_variable_and_dtype(x, 'x', ['complex64', 'complex128'], 'imag')
        helper = LayerHelper('imag', **locals())
        out = helper.create_variable_for_type_inference(
            dtype=_complex_to_real_dtype(helper.input_dtype())
        )
        helper.append_op(type='imag', inputs={'X': x}, outputs={'Out': out})
        return out
