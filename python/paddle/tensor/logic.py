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

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from typing_extensions import TypeGuard

import paddle
from paddle import _C_ops
from paddle.tensor.creation import full
from paddle.tensor.math import broadcast_shape
from paddle.utils.inplace_utils import inplace_apis_in_dygraph_only

from ..base.data_feeder import check_type, check_variable_and_dtype
from ..common_ops_import import Variable
from ..framework import (
    LayerHelper,
    in_dynamic_mode,
    in_dynamic_or_pir_mode,
    in_pir_mode,
)

if TYPE_CHECKING:
    from paddle import Tensor

__all__ = []


def _logical_op(
    op_name: str,
    x: Tensor,
    y: Tensor | None,
    out: Tensor | None = None,
    name: str | None = None,
    binary_op: bool = True,
) -> Tensor:
    if in_dynamic_mode():
        op = getattr(_C_ops, op_name)
        if binary_op:
            return op(x, y)
        else:
            return op(x)
    else:
        check_variable_and_dtype(
            x,
            "x",
            [
                "bool",
                "int8",
                "int16",
                "int32",
                "int64",
                "float16",
                "float32",
                "float64",
                "uint16",
                "complex64",
                "complex128",
            ],
            op_name,
        )
        if y is not None:
            check_variable_and_dtype(
                y,
                "y",
                [
                    "bool",
                    "int8",
                    "int16",
                    "int32",
                    "int64",
                    "float16",
                    "float32",
                    "float64",
                    "uint16",
                    "complex64",
                    "complex128",
                ],
                op_name,
            )
        if out is not None:
            check_type(out, "out", Variable, op_name)

        helper = LayerHelper(op_name, **locals())

        if out is None:
            out = helper.create_variable_for_type_inference(dtype=x.dtype)

        if binary_op:
            helper.append_op(
                type=op_name, inputs={"X": x, "Y": y}, outputs={"Out": out}
            )
        else:
            helper.append_op(
                type=op_name, inputs={"X": x}, outputs={"Out": out}
            )

        return out


def logical_and(
    x: Tensor, y: Tensor, out: Tensor | None = None, name: str | None = None
) -> Tensor:
    r"""

    Compute element-wise logical AND on ``x`` and ``y``, and return ``out``. ``out`` is N-dim boolean ``Tensor``.
    Each element of ``out`` is calculated by

    .. math::

        out = x \&\& y

    Note:
        ``paddle.logical_and`` supports broadcasting. If you want know more about broadcasting, please refer to `Introduction to Tensor`_ .

        .. _Introduction to Tensor: ../../guides/beginner/tensor_en.html#chapter5-broadcasting-of-tensor

    Args:
        x (Tensor): the input tensor, it's data type should be one of bool, int8, int16, in32, in64, bfloat16, float16, float32, float64, complex64, complex128.
        y (Tensor): the input tensor, it's data type should be one of bool, int8, int16, in32, in64, bfloat16, float16, float32, float64, complex64, complex128.
        out(Tensor|None, optional): The ``Tensor`` that specifies the output of the operator, which can be any ``Tensor`` that has been created in the program. The default value is None, and a new ``Tensor`` will be created to save the output.
        name (str|None, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        N-D Tensor. A location into which the result is stored. It's dimension equals with ``x``.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> x = paddle.to_tensor([True])
            >>> y = paddle.to_tensor([True, False, True, False])
            >>> res = paddle.logical_and(x, y)
            >>> print(res)
            Tensor(shape=[4], dtype=bool, place=Place(cpu), stop_gradient=True,
            [True , False, True , False])

    """
    if in_dynamic_or_pir_mode():
        return _C_ops.logical_and(x, y)

    return _logical_op(
        op_name="logical_and", x=x, y=y, name=name, out=out, binary_op=True
    )


@inplace_apis_in_dygraph_only
def logical_and_(x: Tensor, y: Tensor, name: str | None = None) -> Tensor:
    r"""
    Inplace version of ``logical_and`` API, the output Tensor will be inplaced with input ``x``.
    Please refer to :ref:`api_paddle_logical_and`.
    """
    out_shape = broadcast_shape(x.shape, y.shape)
    if out_shape != x.shape:
        raise ValueError(
            f"The shape of broadcast output {out_shape} is different from that of inplace tensor {x.shape} in the Inplace operation."
        )
    if in_dynamic_mode():
        return _C_ops.logical_and_(x, y)


def logical_or(
    x: Tensor, y: Tensor, out: Tensor | None = None, name: str | None = None
) -> Tensor:
    """

    ``logical_or`` operator computes element-wise logical OR on ``x`` and ``y``, and returns ``out``. ``out`` is N-dim boolean ``Tensor``.
    Each element of ``out`` is calculated by

    .. math::

        out = x || y

    Note:
        ``paddle.logical_or`` supports broadcasting. If you want know more about broadcasting, please refer to `Introduction to Tensor`_ .

        .. _Introduction to Tensor: ../../guides/beginner/tensor_en.html#chapter5-broadcasting-of-tensor

    Args:
        x (Tensor): the input tensor, it's data type should be one of bool, int8, int16, in32, in64, bfloat16, float16, float32, float64, complex64, complex128.
        y (Tensor): the input tensor, it's data type should be one of bool, int8, int16, in32, in64, bfloat16, float16, float32, float64, complex64, complex128.
        out(Tensor|None, optional): The ``Variable`` that specifies the output of the operator, which can be any ``Tensor`` that has been created in the program. The default value is None, and a new ``Tensor`` will be created to save the output.
        name (str|None, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        N-D Tensor. A location into which the result is stored. It's dimension equals with ``x``.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> x = paddle.to_tensor([True, False], dtype="bool").reshape([2, 1])
            >>> y = paddle.to_tensor([True, False, True, False], dtype="bool").reshape([2, 2])
            >>> res = paddle.logical_or(x, y)
            >>> print(res)
            Tensor(shape=[2, 2], dtype=bool, place=Place(cpu), stop_gradient=True,
            [[True , True ],
             [True , False]])
    """
    if in_dynamic_or_pir_mode():
        return _C_ops.logical_or(x, y)
    return _logical_op(
        op_name="logical_or", x=x, y=y, name=name, out=out, binary_op=True
    )


@inplace_apis_in_dygraph_only
def logical_or_(x: Tensor, y: Tensor, name: str | None = None) -> Tensor:
    r"""
    Inplace version of ``logical_or`` API, the output Tensor will be inplaced with input ``x``.
    Please refer to :ref:`api_paddle_logical_or`.
    """
    out_shape = broadcast_shape(x.shape, y.shape)
    if out_shape != x.shape:
        raise ValueError(
            f"The shape of broadcast output {out_shape} is different from that of inplace tensor {x.shape} in the Inplace operation."
        )
    if in_dynamic_mode():
        return _C_ops.logical_or_(x, y)


def logical_xor(
    x: Tensor, y: Tensor, out: Tensor | None = None, name: str | None = None
) -> Tensor:
    r"""

    ``logical_xor`` operator computes element-wise logical XOR on ``x`` and ``y``, and returns ``out``. ``out`` is N-dim boolean ``Tensor``.
    Each element of ``out`` is calculated by

    .. math::

        out = (x || y) \&\& !(x \&\& y)

    Note:
        ``paddle.logical_xor`` supports broadcasting. If you want know more about broadcasting, please refer to `Introduction to Tensor`_ .

        .. _Introduction to Tensor: ../../guides/beginner/tensor_en.html#chapter5-broadcasting-of-tensor

    Args:
        x (Tensor): the input tensor, it's data type should be one of bool, int8, int16, int32, int64, bfloat16, float16, float32, float64, complex64, complex128.
        y (Tensor): the input tensor, it's data type should be one of bool, int8, int16, int32, int64, bfloat16, float16, float32, float64, complex64, complex128.
        out(Tensor|None, optional): The ``Tensor`` that specifies the output of the operator, which can be any ``Tensor`` that has been created in the program. The default value is None, and a new ``Tensor`` will be created to save the output.
        name (str|None, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        N-D Tensor. A location into which the result is stored. It's dimension equals with ``x``.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> x = paddle.to_tensor([True, False], dtype="bool").reshape([2, 1])
            >>> y = paddle.to_tensor([True, False, True, False], dtype="bool").reshape([2, 2])
            >>> res = paddle.logical_xor(x, y)
            >>> print(res)
            Tensor(shape=[2, 2], dtype=bool, place=Place(cpu), stop_gradient=True,
            [[False, True ],
             [True , False]])
    """
    if in_dynamic_or_pir_mode():
        return _C_ops.logical_xor(x, y)

    return _logical_op(
        op_name="logical_xor", x=x, y=y, name=name, out=out, binary_op=True
    )


@inplace_apis_in_dygraph_only
def logical_xor_(x: Tensor, y: Tensor, name: str | None = None) -> Tensor:
    r"""
    Inplace version of ``logical_xor`` API, the output Tensor will be inplaced with input ``x``.
    Please refer to :ref:`api_paddle_logical_xor`.
    """
    out_shape = broadcast_shape(x.shape, y.shape)
    if out_shape != x.shape:
        raise ValueError(
            f"The shape of broadcast output {out_shape} is different from that of inplace tensor {x.shape} in the Inplace operation."
        )
    if in_dynamic_mode():
        return _C_ops.logical_xor_(x, y)


def logical_not(
    x: Tensor, out: Tensor | None = None, name: str | None = None
) -> Tensor:
    """

    ``logical_not`` operator computes element-wise logical NOT on ``x``, and returns ``out``. ``out`` is N-dim boolean ``Variable``.
    Each element of ``out`` is calculated by

    .. math::

        out = !x

    Note:
        ``paddle.logical_not`` supports broadcasting. If you want know more about broadcasting, please refer to `Introduction to Tensor`_ .

        .. _Introduction to Tensor: ../../guides/beginner/tensor_en.html#chapter5-broadcasting-of-tensor

    Args:

        x(Tensor):  Operand of logical_not operator. Must be a Tensor of type bool, int8, int16, in32, in64, bfloat16, float16, float32, or float64, complex64, complex128.
        out(Tensor|None): The ``Tensor`` that specifies the output of the operator, which can be any ``Tensor`` that has been created in the program. The default value is None, and a new ``Tensor` will be created to save the output.
        name(str|None, optional): The default value is None. Normally there is no need for users to set this property. For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        N-D Tensor. A location into which the result is stored. It's dimension equals with ``x``.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> x = paddle.to_tensor([True, False, True, False])
            >>> res = paddle.logical_not(x)
            >>> print(res)
            Tensor(shape=[4], dtype=bool, place=Place(cpu), stop_gradient=True,
            [False, True , False, True ])
    """
    if in_dynamic_or_pir_mode():
        return _C_ops.logical_not(x)
    return _logical_op(
        op_name="logical_not", x=x, y=None, name=name, out=out, binary_op=False
    )


@inplace_apis_in_dygraph_only
def logical_not_(x: Tensor, name: str | None = None) -> Tensor:
    r"""
    Inplace version of ``logical_not`` API, the output Tensor will be inplaced with input ``x``.
    Please refer to :ref:`api_paddle_logical_not`.
    """
    if in_dynamic_mode():
        return _C_ops.logical_not_(x)


def is_empty(x: Tensor, name: str | None = None) -> Tensor:
    """

    Test whether a Tensor is empty.

    Args:
        x (Tensor): The Tensor to be tested.
        name (str|None, optional): The default value is ``None`` . Normally users don't have to set this parameter. For more information, please refer to :ref:`api_guide_Name` .

    Returns:
        Tensor: A bool scalar Tensor. True if 'x' is an empty Tensor.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> input = paddle.rand(shape=[4, 32, 32], dtype='float32')
            >>> res = paddle.is_empty(x=input)
            >>> print(res)
            Tensor(shape=[], dtype=bool, place=Place(cpu), stop_gradient=True,
            False)

    """
    if in_dynamic_mode():
        return _C_ops.is_empty(x)

    check_variable_and_dtype(
        x, 'x', ['float32', 'float64', 'int32', 'int64'], 'is_empty'
    )
    check_type(name, "name", (str, type(None)), "is_empty")
    if in_pir_mode():
        return _C_ops.is_empty(x)
    else:
        helper = LayerHelper("is_empty", **locals())
        cond = helper.create_variable_for_type_inference(dtype='bool')
        cond.stop_gradient = True
        helper.append_op(
            type='is_empty', inputs={'X': [x]}, outputs={'Out': [cond]}
        )
        return cond


def equal_all(x: Tensor, y: Tensor, name: str | None = None) -> Tensor:
    """
    Returns the truth value of :math:`x == y`. True if two inputs have the same elements, False otherwise.

    Note:
        The output has no gradient.

    Args:
        x(Tensor): Tensor, data type is bool, float32, float64, int32, int64.
        y(Tensor): Tensor, data type is bool, float32, float64, int32, int64.
        name(str|None, optional): The default value is None.  Normally there is no need for
            user to set this property.  For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor: output Tensor, data type is bool, value is [False] or [True].

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> x = paddle.to_tensor([1, 2, 3])
            >>> y = paddle.to_tensor([1, 2, 3])
            >>> z = paddle.to_tensor([1, 4, 3])
            >>> result1 = paddle.equal_all(x, y)
            >>> print(result1)
            Tensor(shape=[], dtype=bool, place=Place(cpu), stop_gradient=True,
            True)
            >>> result2 = paddle.equal_all(x, z)
            >>> print(result2)
            Tensor(shape=[], dtype=bool, place=Place(cpu), stop_gradient=True,
            False)
    """
    if in_dynamic_or_pir_mode():
        return _C_ops.equal_all(x, y)
    else:
        helper = LayerHelper("equal_all", **locals())
        out = helper.create_variable_for_type_inference(dtype='bool')
        helper.append_op(
            type='equal_all',
            inputs={'X': [x], 'Y': [y]},
            outputs={'Out': [out]},
        )
        return out


def allclose(
    x: Tensor,
    y: Tensor,
    rtol: float = 1e-05,
    atol: float = 1e-08,
    equal_nan: bool = False,
    name: str | None = None,
) -> Tensor:
    r"""
    Check if all :math:`x` and :math:`y` satisfy the condition:

    .. math::
        \left| x - y \right| \leq atol + rtol \times \left| y \right|

    elementwise, for all elements of :math:`x` and :math:`y`. This is analogous to :math:`numpy.allclose`, namely that it returns :math:`True` if
    two tensors are elementwise equal within a tolerance.

    Args:
        x (Tensor): The input tensor, it's data type should be float16, float32, float64.
        y (Tensor): The input tensor, it's data type should be float16, float32, float64.
        rtol (float, optional): The relative tolerance. Default: :math:`1e-5` .
        atol (float, optional): The absolute tolerance. Default: :math:`1e-8` .
        equal_nan (bool, optional): ${equal_nan_comment}. Default: False.
        name (str|None, optional): Name for the operation. For more information, please
            refer to :ref:`api_guide_Name`. Default: None.

    Returns:
        Tensor: The output tensor, it's data type is bool.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> x = paddle.to_tensor([10000., 1e-07])
            >>> y = paddle.to_tensor([10000.1, 1e-08])
            >>> result1 = paddle.allclose(x, y, rtol=1e-05, atol=1e-08, equal_nan=False, name="ignore_nan")
            >>> print(result1)
            Tensor(shape=[], dtype=bool, place=Place(cpu), stop_gradient=True,
            False)
            >>> result2 = paddle.allclose(x, y, rtol=1e-05, atol=1e-08, equal_nan=True, name="equal_nan")
            >>> print(result2)
            Tensor(shape=[], dtype=bool, place=Place(cpu), stop_gradient=True,
            False)
            >>> x = paddle.to_tensor([1.0, float('nan')])
            >>> y = paddle.to_tensor([1.0, float('nan')])
            >>> result1 = paddle.allclose(x, y, rtol=1e-05, atol=1e-08, equal_nan=False, name="ignore_nan")
            >>> print(result1)
            Tensor(shape=[], dtype=bool, place=Place(cpu), stop_gradient=True,
            False)
            >>> result2 = paddle.allclose(x, y, rtol=1e-05, atol=1e-08, equal_nan=True, name="equal_nan")
            >>> print(result2)
            Tensor(shape=[], dtype=bool, place=Place(cpu), stop_gradient=True,
            True)
    """

    if in_dynamic_mode():
        return _C_ops.allclose(x, y, rtol, atol, equal_nan)
    elif in_pir_mode():
        check_variable_and_dtype(
            x,
            "input",
            ['bool', 'int32', 'int64', 'float16', 'float32', 'float64'],
            'allclose',
        )
        check_variable_and_dtype(
            y,
            "input",
            ['bool', 'int32', 'int64', 'float16', 'float32', 'float64'],
            'allclose',
        )
        if not isinstance(rtol, (float, paddle.pir.Value)):
            raise TypeError(
                f"Type of input rtol must be float, but received type {type(rtol)}"
            )
        if not isinstance(atol, (float, paddle.pir.Value)):
            raise TypeError(
                f"Type of input atol must be float, but received type {type(atol)}"
            )
        check_type(equal_nan, 'equal_nan', bool, 'allclose')
        return _C_ops.allclose(x, y, rtol, atol, equal_nan)
    else:
        check_variable_and_dtype(
            x,
            "input",
            ['bool', 'int32', 'int64', 'float16', 'float32', 'float64'],
            'allclose',
        )
        check_variable_and_dtype(
            y,
            "input",
            ['bool', 'int32', 'int64', 'float16', 'float32', 'float64'],
            'allclose',
        )
        check_type(rtol, 'rtol', float, 'allclose')
        check_type(atol, 'atol', float, 'allclose')
        check_type(equal_nan, 'equal_nan', bool, 'allclose')

        helper = LayerHelper("allclose", **locals())
        out = helper.create_variable_for_type_inference(dtype='bool')

        inputs = {'Input': x, 'Other': y}
        outputs = {'Out': out}
        attrs = {'rtol': str(rtol), 'atol': str(atol), 'equal_nan': equal_nan}
        helper.append_op(
            type='allclose', inputs=inputs, outputs=outputs, attrs=attrs
        )

        return out


def equal(x: Tensor, y: Tensor, name: str | None = None) -> Tensor:
    """

    This layer returns the truth value of :math:`x == y` elementwise.

    Note:
        The output has no gradient.

    Args:
        x (Tensor): Tensor, data type is bool, float16, float32, float64, uint8, int8, int16, int32, int64, complex64, complex128.
        y (Tensor): Tensor, data type is bool, float16, float32, float64, uint8, int8, int16, int32, int64, complex64, complex128.
        name (str|None, optional): The default value is None. Normally there is no need for
            user to set this property.  For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor: output Tensor, it's shape is the same as the input's Tensor,
        and the data type is bool. The result of this op is stop_gradient.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> x = paddle.to_tensor([1, 2, 3])
            >>> y = paddle.to_tensor([1, 3, 2])
            >>> result1 = paddle.equal(x, y)
            >>> print(result1)
            Tensor(shape=[3], dtype=bool, place=Place(cpu), stop_gradient=True,
            [True , False, False])
    """
    if not isinstance(
        y, (int, bool, float, Variable, complex, paddle.pir.Value)
    ):
        raise TypeError(
            f"Type of input args must be float, bool, complex, int or Tensor, but received type {type(y)}"
        )
    if not isinstance(y, (Variable, paddle.pir.Value, complex)):
        y = full(shape=[], dtype=x.dtype, fill_value=y)

    if isinstance(y, complex):
        # full not support for complex yet
        y = paddle.to_tensor(y)

    if in_dynamic_or_pir_mode():
        return _C_ops.equal(x, y)
    else:
        check_variable_and_dtype(
            x,
            "x",
            [
                "bool",
                "float16",
                "float32",
                "float64",
                "uint8",
                "int8",
                "int16",
                "int32",
                "int64",
                "uint16",
                "complex64",
                "complex128",
            ],
            "equal",
        )
        check_variable_and_dtype(
            y,
            "y",
            [
                "bool",
                "float16",
                "float32",
                "float64",
                "uint8",
                "int8",
                "int16",
                "int32",
                "int64",
                "uint16",
                "complex64",
                "complex128",
            ],
            "equal",
        )
        helper = LayerHelper("equal", **locals())
        out = helper.create_variable_for_type_inference(dtype='bool')
        out.stop_gradient = True
        helper.append_op(
            type='equal',
            inputs={'X': [x], 'Y': [y]},
            outputs={'Out': [out]},
        )
        return out


@inplace_apis_in_dygraph_only
def equal_(x: Tensor, y: Tensor, name: str | None = None) -> Tensor:
    r"""
    Inplace version of ``equal`` API, the output Tensor will be inplaced with input ``x``.
    Please refer to :ref:`api_paddle_equal`.
    """
    out_shape = broadcast_shape(x.shape, y.shape)
    if out_shape != x.shape:
        raise ValueError(
            f"The shape of broadcast output {out_shape} is different from that of inplace tensor {x.shape} in the Inplace operation."
        )
    if in_dynamic_or_pir_mode():
        return _C_ops.equal_(x, y)


def greater_equal(x: Tensor, y: Tensor, name: str | None = None) -> Tensor:
    """
    Returns the truth value of :math:`x >= y` elementwise, which is equivalent function to the overloaded operator `>=`.

    Note:
        The output has no gradient.

    Args:
        x (Tensor): First input to compare which is N-D tensor. The input data type should be bool, bfloat16, float16, float32, float64, uint8, int8, int16, int32, int64.
        y (Tensor): Second input to compare which is N-D tensor. The input data type should be bool, bfloat16, float16, float32, float64, uint8, int8, int16, int32, int64.
        name (str|None, optional): The default value is None.  Normally there is no need for
            user to set this property.  For more information, please refer to :ref:`api_guide_Name`.
    Returns:
        Tensor: The output shape is same as input :attr:`x`. The output data type is bool.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> x = paddle.to_tensor([1, 2, 3])
            >>> y = paddle.to_tensor([1, 3, 2])
            >>> result1 = paddle.greater_equal(x, y)
            >>> print(result1)
            Tensor(shape=[3], dtype=bool, place=Place(cpu), stop_gradient=True,
            [True , False, True ])
    """
    if in_dynamic_or_pir_mode():
        return _C_ops.greater_equal(x, y)
    else:
        check_variable_and_dtype(
            x,
            "x",
            [
                "bool",
                "float16",
                "float32",
                "float64",
                "uint8",
                "int8",
                "int16",
                "int32",
                "int64",
                "uint16",
            ],
            "greater_equal",
        )
        check_variable_and_dtype(
            y,
            "y",
            [
                "bool",
                "float16",
                "float32",
                "float64",
                "uint8",
                "int8",
                "int16",
                "int32",
                "int64",
                "uint16",
            ],
            "greater_equal",
        )
        helper = LayerHelper("greater_equal", **locals())
        out = helper.create_variable_for_type_inference(dtype='bool')
        out.stop_gradient = True
        helper.append_op(
            type='greater_equal',
            inputs={'X': [x], 'Y': [y]},
            outputs={'Out': [out]},
        )
        return out


@inplace_apis_in_dygraph_only
def greater_equal_(x: Tensor, y: Tensor, name: str | None = None) -> Tensor:
    r"""
    Inplace version of ``greater_equal`` API, the output Tensor will be inplaced with input ``x``.
    Please refer to :ref:`api_paddle_greater_equal`.
    """
    out_shape = broadcast_shape(x.shape, y.shape)
    if out_shape != x.shape:
        raise ValueError(
            f"The shape of broadcast output {out_shape} is different from that of inplace tensor {x.shape} in the Inplace operation."
        )
    if in_dynamic_mode():
        return _C_ops.greater_equal_(x, y)


def greater_than(x: Tensor, y: Tensor, name: str | None = None) -> Tensor:
    """
    Returns the truth value of :math:`x > y` elementwise, which is equivalent function to the overloaded operator `>`.

    Note:
        The output has no gradient.

    Args:
        x (Tensor): First input to compare which is N-D tensor. The input data type should be bool, bfloat16, float16, float32, float64, uint8, int8, int16, int32, int64.
        y (Tensor): Second input to compare which is N-D tensor. The input data type should be bool, bfloat16, float16, float32, float64, uint8, int8, int16, int32, int64.
        name (str|None, optional): The default value is None.  Normally there is no need for
            user to set this property.  For more information, please refer to :ref:`api_guide_Name`.
    Returns:
        Tensor: The output shape is same as input :attr:`x`. The output data type is bool.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> x = paddle.to_tensor([1, 2, 3])
            >>> y = paddle.to_tensor([1, 3, 2])
            >>> result1 = paddle.greater_than(x, y)
            >>> print(result1)
            Tensor(shape=[3], dtype=bool, place=Place(cpu), stop_gradient=True,
            [False, False, True ])
    """
    if in_dynamic_or_pir_mode():
        return _C_ops.greater_than(x, y)
    else:
        check_variable_and_dtype(
            x,
            "x",
            [
                "bool",
                "float16",
                "float32",
                "float64",
                "uint8",
                "int8",
                "int16",
                "int32",
                "int64",
                "uint16",
            ],
            "greater_than",
        )
        check_variable_and_dtype(
            y,
            "y",
            [
                "bool",
                "float16",
                "float32",
                "float64",
                "uint8",
                "int8",
                "int16",
                "int32",
                "int64",
                "uint16",
            ],
            "greater_than",
        )
        helper = LayerHelper("greater_than", **locals())
        out = helper.create_variable_for_type_inference(dtype='bool')
        out.stop_gradient = True
        helper.append_op(
            type='greater_than',
            inputs={'X': [x], 'Y': [y]},
            outputs={'Out': [out]},
        )
        return out


@inplace_apis_in_dygraph_only
def greater_than_(x: Tensor, y: Tensor, name: str | None = None) -> Tensor:
    r"""
    Inplace version of ``greater_than`` API, the output Tensor will be inplaced with input ``x``.
    Please refer to :ref:`api_paddle_greater_than`.
    """
    out_shape = broadcast_shape(x.shape, y.shape)
    if out_shape != x.shape:
        raise ValueError(
            f"The shape of broadcast output {out_shape} is different from that of inplace tensor {x.shape} in the Inplace operation."
        )
    if in_dynamic_mode():
        return _C_ops.greater_than_(x, y)


def less_equal(x: Tensor, y: Tensor, name: str | None = None) -> Tensor:
    """
    Returns the truth value of :math:`x <= y` elementwise, which is equivalent function to the overloaded operator `<=`.

    Note:
        The output has no gradient.

    Args:
        x (Tensor): First input to compare which is N-D tensor. The input data type should be bool, bfloat16, float16, float32, float64, uint8, int8, int16, int32, int64.
        y (Tensor): Second input to compare which is N-D tensor. The input data type should be bool, bfloat16, float16, float32, float64, uint8, int8, int16, int32, int64.
        name (str|None, optional): The default value is None.  Normally there is no need for
            user to set this property.  For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor: The output shape is same as input :attr:`x`. The output data type is bool.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> x = paddle.to_tensor([1, 2, 3])
            >>> y = paddle.to_tensor([1, 3, 2])
            >>> result1 = paddle.less_equal(x, y)
            >>> print(result1)
            Tensor(shape=[3], dtype=bool, place=Place(cpu), stop_gradient=True,
            [True , True , False])
    """
    if in_dynamic_or_pir_mode():
        return _C_ops.less_equal(x, y)
    else:
        check_variable_and_dtype(
            x,
            "x",
            [
                "bool",
                "float16",
                "float32",
                "float64",
                "uint8",
                "int8",
                "int16",
                "int32",
                "int64",
                "uint16",
            ],
            "less_equal",
        )
        check_variable_and_dtype(
            y,
            "y",
            [
                "bool",
                "float16",
                "float32",
                "float64",
                "uint8",
                "int8",
                "int16",
                "int32",
                "int64",
                "uint16",
            ],
            "less_equal",
        )
        helper = LayerHelper("less_equal", **locals())
        out = helper.create_variable_for_type_inference(dtype='bool')
        out.stop_gradient = True
        helper.append_op(
            type='less_equal',
            inputs={'X': [x], 'Y': [y]},
            outputs={'Out': [out]},
        )
        return out


@inplace_apis_in_dygraph_only
def less_equal_(x: Tensor, y: Tensor, name: str | None = None) -> Tensor:
    r"""
    Inplace version of ``less_equal`` API, the output Tensor will be inplaced with input ``x``.
    Please refer to :ref:`api_paddle_less_equal`.
    """
    out_shape = broadcast_shape(x.shape, y.shape)
    if out_shape != x.shape:
        raise ValueError(
            f"The shape of broadcast output {out_shape} is different from that of inplace tensor {x.shape} in the Inplace operation."
        )
    if in_dynamic_mode():
        return _C_ops.less_equal_(x, y)


def less_than(x: Tensor, y: Tensor, name: str | None = None) -> Tensor:
    """
    Returns the truth value of :math:`x < y` elementwise, which is equivalent function to the overloaded operator `<`.

    Note:
        The output has no gradient.

    Args:
        x (Tensor): First input to compare which is N-D tensor. The input data type should be bool, bfloat16, float16, float32, float64, uint8, int8, int16, int32, int64.
        y (Tensor): Second input to compare which is N-D tensor. The input data type should be bool, bfloat16, float16, float32, float64, uint8, int8, int16, int32, int64.
        name (str|None, optional): The default value is None.  Normally there is no need for
            user to set this property.  For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor: The output shape is same as input :attr:`x`. The output data type is bool.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> x = paddle.to_tensor([1, 2, 3])
            >>> y = paddle.to_tensor([1, 3, 2])
            >>> result1 = paddle.less_than(x, y)
            >>> print(result1)
            Tensor(shape=[3], dtype=bool, place=Place(cpu), stop_gradient=True,
            [False, True , False])
    """
    if in_dynamic_or_pir_mode():
        return _C_ops.less_than(x, y)
    else:
        check_variable_and_dtype(
            x,
            "x",
            [
                "bool",
                "float16",
                "float32",
                "float64",
                "uint8",
                "int8",
                "int16",
                "int32",
                "int64",
                "uint16",
            ],
            "less_than",
        )
        check_variable_and_dtype(
            y,
            "y",
            [
                "bool",
                "float16",
                "float32",
                "float64",
                "uint8",
                "int8",
                "int16",
                "int32",
                "int64",
                "uint16",
            ],
            "less_than",
        )
        helper = LayerHelper("less_than", **locals())
        out = helper.create_variable_for_type_inference(dtype='bool')
        out.stop_gradient = True

        helper.append_op(
            type='less_than',
            inputs={'X': [x], 'Y': [y]},
            outputs={'Out': [out]},
        )
        return out


@inplace_apis_in_dygraph_only
def less_than_(x: Tensor, y: Tensor, name: str | None = None) -> Tensor:
    r"""
    Inplace version of ``less_than`` API, the output Tensor will be inplaced with input ``x``.
    Please refer to :ref:`api_paddle_less_than`.
    """
    out_shape = broadcast_shape(x.shape, y.shape)
    if out_shape != x.shape:
        raise ValueError(
            f"The shape of broadcast output {out_shape} is different from that of inplace tensor {x.shape} in the Inplace operation."
        )
    if in_dynamic_mode():
        return _C_ops.less_than_(x, y)


def less(x: Tensor, y: Tensor, name: str | None = None) -> Tensor:
    """
    Returns the truth value of :math:`x < y` elementwise, which is equivalent function to the overloaded operator `<`.

    Note:
        The output has no gradient.

    Args:
        x (Tensor): First input to compare which is N-D tensor. The input data type should be bool, bfloat16, float16, float32, float64, uint8, int8, int16, int32, int64.
        y (Tensor): Second input to compare which is N-D tensor. The input data type should be bool, bfloat16, float16, float32, float64, uint8, int8, int16, int32, int64.
        name (str|None, optional): The default value is None.  Normally there is no need for
            user to set this property.  For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor: The output shape is same as input :attr:`x`. The output data type is bool.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> x = paddle.to_tensor([1, 2, 3])
            >>> y = paddle.to_tensor([1, 3, 2])
            >>> result1 = paddle.less(x, y)
            >>> print(result1)
            Tensor(shape=[3], dtype=bool, place=Place(cpu), stop_gradient=True,
            [False, True , False])
    """

    # Directly call less_than API
    return less_than(x, y, name)


@inplace_apis_in_dygraph_only
def less_(x: Tensor, y: Tensor, name: str | None = None) -> Tensor:
    r"""
    Inplace version of ``less_`` API, the output Tensor will be inplaced with input ``x``.
    Please refer to :ref:`api_paddle_less`.
    """

    # Directly call less_than_ API
    return less_than_(x, y, name)


def not_equal(x: Tensor, y: Tensor, name: str | None = None) -> Tensor:
    """
    Returns the truth value of :math:`x != y` elementwise, which is equivalent function to the overloaded operator `!=`.

    Note:
        The output has no gradient.

    Args:
        x (Tensor): First input to compare which is N-D tensor. The input data type should be bool, bfloat16, float16, float32, float64, uint8, int8, int16, int32, int64, complex64, complex128.
        y (Tensor): Second input to compare which is N-D tensor. The input data type should be bool, bfloat16, float16, float32, float64, uint8, int8, int16, int32, int64, complex64, complex128.
        name (str|None, optional): The default value is None.  Normally there is no need for
            user to set this property.  For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor: The output shape is same as input :attr:`x`. The output data type is bool.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> x = paddle.to_tensor([1, 2, 3])
            >>> y = paddle.to_tensor([1, 3, 2])
            >>> result1 = paddle.not_equal(x, y)
            >>> print(result1)
            Tensor(shape=[3], dtype=bool, place=Place(cpu), stop_gradient=True,
            [False, True , True ])
    """
    if in_dynamic_or_pir_mode():
        return _C_ops.not_equal(x, y)
    else:
        check_variable_and_dtype(
            x,
            "x",
            [
                "bool",
                "float16",
                "float32",
                "float64",
                "uint8",
                "int8",
                "int16",
                "int32",
                "int64",
                "uint16",
                "complex64",
                "complex128",
            ],
            "not_equal",
        )
        check_variable_and_dtype(
            y,
            "y",
            [
                "bool",
                "float16",
                "float32",
                "float64",
                "uint8",
                "int8",
                "int16",
                "int32",
                "int64",
                "uint16",
                "complex64",
                "complex128",
            ],
            "not_equal",
        )
        helper = LayerHelper("not_equal", **locals())
        out = helper.create_variable_for_type_inference(dtype='bool')
        out.stop_gradient = True

        helper.append_op(
            type='not_equal',
            inputs={'X': [x], 'Y': [y]},
            outputs={'Out': [out]},
        )
        return out


@inplace_apis_in_dygraph_only
def not_equal_(x: Tensor, y: Tensor, name: str | None = None) -> Tensor:
    r"""
    Inplace version of ``not_equal`` API, the output Tensor will be inplaced with input ``x``.
    Please refer to :ref:`api_paddle_not_equal`.
    """
    out_shape = broadcast_shape(x.shape, y.shape)
    if out_shape != x.shape:
        raise ValueError(
            f"The shape of broadcast output {out_shape} is different from that of inplace tensor {x.shape} in the Inplace operation."
        )
    if in_dynamic_mode():
        return _C_ops.not_equal_(x, y)


def is_tensor(x: Any) -> TypeGuard[Tensor]:
    """

    Tests whether input object is a paddle.Tensor.

    Args:
        x (object): Object to test.

    Returns:
        A boolean value. True if ``x`` is a paddle.Tensor, otherwise False.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> input1 = paddle.rand(shape=[2, 3, 5], dtype='float32')
            >>> check = paddle.is_tensor(input1)
            >>> print(check)
            True

            >>> input3 = [1, 4]
            >>> check = paddle.is_tensor(input3)
            >>> print(check)
            False

    """
    if in_dynamic_or_pir_mode():
        return isinstance(x, (paddle.Tensor, paddle.pir.Value))
    else:
        return isinstance(x, Variable)


def _bitwise_op(
    op_name: str,
    x: Tensor,
    y: Tensor | None,
    out: Tensor | None = None,
    name: str | None = None,
    binary_op: bool = True,
) -> Tensor:
    if in_dynamic_mode():
        op = getattr(_C_ops, op_name)
        if binary_op:
            return op(x, y)
        else:
            return op(x)
    else:
        check_variable_and_dtype(
            x,
            "x",
            ["bool", "uint8", "int8", "int16", "int32", "int64"],
            op_name,
        )
        if y is not None:
            check_variable_and_dtype(
                y,
                "y",
                ["bool", "uint8", "int8", "int16", "int32", "int64"],
                op_name,
            )
        if out is not None:
            check_type(out, "out", Variable, op_name)

        helper = LayerHelper(op_name, **locals())
        if binary_op:
            assert x.dtype == y.dtype

        if out is None:
            out = helper.create_variable_for_type_inference(dtype=x.dtype)

        if binary_op:
            helper.append_op(
                type=op_name, inputs={"X": x, "Y": y}, outputs={"Out": out}
            )
        else:
            helper.append_op(
                type=op_name, inputs={"X": x}, outputs={"Out": out}
            )

        return out


def bitwise_and(
    x: Tensor, y: Tensor, out: Tensor | None = None, name: str | None = None
) -> Tensor:
    r"""

    Apply ``bitwise_and`` on Tensor ``X`` and ``Y`` .

    .. math::
        Out = X \& Y

    Note:
        ``paddle.bitwise_and`` supports broadcasting. If you want know more about broadcasting, please refer to please refer to `Introduction to Tensor`_ .

        .. _Introduction to Tensor: ../../guides/beginner/tensor_en.html#chapter5-broadcasting-of-tensor

    Args:
        x (Tensor): Input Tensor of ``bitwise_and`` . It is a N-D Tensor of bool, uint8, int8, int16, int32, int64.
        y (Tensor): Input Tensor of ``bitwise_and`` . It is a N-D Tensor of bool, uint8, int8, int16, int32, int64.
        out (Tensor|None, optional): Result of ``bitwise_and`` . It is a N-D Tensor with the same data type of input Tensor. Default: None.
        name (str|None, optional): The default value is None.  Normally there is no need for
            user to set this property.  For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor: Result of ``bitwise_and`` . It is a N-D Tensor with the same data type of input Tensor.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> x = paddle.to_tensor([-5, -1, 1])
            >>> y = paddle.to_tensor([4,  2, -3])
            >>> res = paddle.bitwise_and(x, y)
            >>> print(res)
            Tensor(shape=[3], dtype=int64, place=Place(cpu), stop_gradient=True,
            [0, 2, 1])
    """
    if in_dynamic_or_pir_mode() and out is None:
        return _C_ops.bitwise_and(x, y)
    return _bitwise_op(
        op_name="bitwise_and", x=x, y=y, name=name, out=out, binary_op=True
    )


def __rand__(x: Tensor, y: int | bool):
    if isinstance(y, (int, bool)):
        y_tensor = paddle.to_tensor(y, dtype=x.dtype)
        return bitwise_and(y_tensor, x, None, None)
    else:
        raise TypeError(
            f"unsupported operand type(s) for |: '{type(y).__name__}' and 'Tensor'"
        )


@inplace_apis_in_dygraph_only
def bitwise_and_(x: Tensor, y: Tensor, name: str | None = None) -> Tensor:
    r"""
    Inplace version of ``bitwise_and`` API, the output Tensor will be inplaced with input ``x``.
    Please refer to :ref:`api_paddle_bitwise_and`.
    """
    out_shape = broadcast_shape(x.shape, y.shape)
    if out_shape != x.shape:
        raise ValueError(
            f"The shape of broadcast output {out_shape} is different from that of inplace tensor {x.shape} in the Inplace operation."
        )
    if in_dynamic_or_pir_mode():
        return _C_ops.bitwise_and_(x, y)


def bitwise_or(
    x: Tensor, y: Tensor, out: Tensor | None = None, name: str | None = None
) -> Tensor:
    r"""

    Apply ``bitwise_or`` on Tensor ``X`` and ``Y`` .

    .. math::
        Out = X | Y

    Note:
        ``paddle.bitwise_or`` supports broadcasting. If you want know more about broadcasting, please refer to please refer to `Introduction to Tensor`_ .

        .. _Introduction to Tensor: ../../guides/beginner/tensor_en.html#chapter5-broadcasting-of-tensor

    Args:
        x (Tensor): Input Tensor of ``bitwise_or`` . It is a N-D Tensor of bool, uint8, int8, int16, int32, int64.
        y (Tensor): Input Tensor of ``bitwise_or`` . It is a N-D Tensor of bool, uint8, int8, int16, int32, int64.
        out (Tensor|None, optional): Result of ``bitwise_or`` . It is a N-D Tensor with the same data type of input Tensor. Default: None.
        name (str|None, optional): The default value is None.  Normally there is no need for
            user to set this property.  For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor: Result of ``bitwise_or`` . It is a N-D Tensor with the same data type of input Tensor.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> x = paddle.to_tensor([-5, -1, 1])
            >>> y = paddle.to_tensor([4,  2, -3])
            >>> res = paddle.bitwise_or(x, y)
            >>> print(res)
            Tensor(shape=[3], dtype=int64, place=Place(cpu), stop_gradient=True,
            [-1, -1, -3])
    """
    if in_dynamic_or_pir_mode() and out is None:
        return _C_ops.bitwise_or(x, y)

    return _bitwise_op(
        op_name="bitwise_or", x=x, y=y, name=name, out=out, binary_op=True
    )


def __ror__(
    x: Tensor,
    y: int | bool,
    out: Tensor | None = None,
    name: str | None = None,
) -> Tensor:
    if isinstance(y, (int, bool)):
        y = paddle.to_tensor(y, dtype=x.dtype)
        return bitwise_or(y, x, out=out, name=name)
    else:
        raise TypeError(
            f"unsupported operand type(s) for |: '{type(y).__name__}' and 'Tensor'"
        )


@inplace_apis_in_dygraph_only
def bitwise_or_(x: Tensor, y: Tensor, name: str | None = None) -> Tensor:
    r"""
    Inplace version of ``bitwise_or`` API, the output Tensor will be inplaced with input ``x``.
    Please refer to :ref:`api_paddle_bitwise_or`.
    """
    out_shape = broadcast_shape(x.shape, y.shape)
    if out_shape != x.shape:
        raise ValueError(
            f"The shape of broadcast output {out_shape} is different from that of inplace tensor {x.shape} in the Inplace operation."
        )
    if in_dynamic_mode():
        return _C_ops.bitwise_or_(x, y)


def bitwise_xor(
    x: Tensor, y: Tensor, out: Tensor | None = None, name: str | None = None
) -> Tensor:
    r"""

    Apply ``bitwise_xor`` on Tensor ``X`` and ``Y`` .

    .. math::
        Out = X ^\wedge Y

    Note:
        ``paddle.bitwise_xor`` supports broadcasting. If you want know more about broadcasting, please refer to please refer to `Introduction to Tensor`_ .

        .. _Introduction to Tensor: ../../guides/beginner/tensor_en.html#chapter5-broadcasting-of-tensor

    Args:
        x (Tensor): Input Tensor of ``bitwise_xor`` . It is a N-D Tensor of bool, uint8, int8, int16, int32, int64.
        y (Tensor): Input Tensor of ``bitwise_xor`` . It is a N-D Tensor of bool, uint8, int8, int16, int32, int64.
        out (Tensor|None, optional): Result of ``bitwise_xor`` . It is a N-D Tensor with the same data type of input Tensor. Default: None.
        name (str|None, optional): The default value is None.  Normally there is no need for
            user to set this property.  For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor: Result of ``bitwise_xor`` . It is a N-D Tensor with the same data type of input Tensor.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> x = paddle.to_tensor([-5, -1, 1])
            >>> y = paddle.to_tensor([4,  2, -3])
            >>> res = paddle.bitwise_xor(x, y)
            >>> print(res)
            Tensor(shape=[3], dtype=int64, place=Place(cpu), stop_gradient=True,
            [-1, -3, -4])
    """
    if in_dynamic_or_pir_mode() and out is None:
        return _C_ops.bitwise_xor(x, y)
    return _bitwise_op(
        op_name="bitwise_xor", x=x, y=y, name=name, out=out, binary_op=True
    )


def __rxor__(
    x: Tensor,
    y: int | bool,
    out: Tensor | None = None,
    name: str | None = None,
) -> Tensor:
    if isinstance(y, (int, bool)):
        y = paddle.to_tensor(y, dtype=x.dtype)
        return bitwise_xor(y, x, out=out, name=name)
    else:
        raise TypeError(
            f"unsupported operand type(s) for |: '{type(y).__name__}' and 'Tensor'"
        )


@inplace_apis_in_dygraph_only
def bitwise_xor_(x: Tensor, y: Tensor, name: str | None = None) -> Tensor:
    r"""
    Inplace version of ``bitwise_xor`` API, the output Tensor will be inplaced with input ``x``.
    Please refer to :ref:`api_paddle_bitwise_xor`.
    """
    out_shape = broadcast_shape(x.shape, y.shape)
    if out_shape != x.shape:
        raise ValueError(
            f"The shape of broadcast output {out_shape} is different from that of inplace tensor {x.shape} in the Inplace operation."
        )
    if in_dynamic_mode():
        return _C_ops.bitwise_xor_(x, y)


def bitwise_not(
    x: Tensor, out: Tensor | None = None, name: str | None = None
) -> Tensor:
    r"""

    Apply ``bitwise_not`` on Tensor ``X``.

    .. math::
        Out = \sim X

    Note:
        ``paddle.bitwise_not`` supports broadcasting. If you want know more about broadcasting, please refer to please refer to `Introduction to Tensor`_ .

        .. _Introduction to Tensor: ../../guides/beginner/tensor_en.html#chapter5-broadcasting-of-tensor

    Args:
        x (Tensor): Input Tensor of ``bitwise_not`` . It is a N-D Tensor of bool, uint8, int8, int16, int32, int64.
        out (Tensor|None, optional): Result of ``bitwise_not`` . It is a N-D Tensor with the same data type of input Tensor. Default: None.
        name (str|None, optional): The default value is None.  Normally there is no need for
            user to set this property.  For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor: Result of ``bitwise_not`` . It is a N-D Tensor with the same data type of input Tensor.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> x = paddle.to_tensor([-5, -1, 1])
            >>> res = paddle.bitwise_not(x)
            >>> print(res)
            Tensor(shape=[3], dtype=int64, place=Place(cpu), stop_gradient=True,
            [ 4,  0, -2])
    """
    if in_dynamic_or_pir_mode() and out is None:
        return _C_ops.bitwise_not(x)

    return _bitwise_op(
        op_name="bitwise_not", x=x, y=None, name=name, out=out, binary_op=False
    )


@inplace_apis_in_dygraph_only
def bitwise_not_(x: Tensor, name: str | None = None) -> Tensor:
    r"""
    Inplace version of ``bitwise_not`` API, the output Tensor will be inplaced with input ``x``.
    Please refer to :ref:`api_paddle_bitwise_not`.
    """
    if in_dynamic_mode():
        return _C_ops.bitwise_not_(x)


def bitwise_invert(
    x: Tensor, out: Tensor | None = None, name: str | None = None
) -> Tensor:
    r"""
    Apply ``bitwise_not`` (bitwise inversion) on Tensor ``x``.

    This is an alias to the ``paddle.bitwise_not`` function.

    .. math::
        Out = \sim X

    Note:
        ``paddle.bitwise_invert`` is functionally equivalent to ``paddle.bitwise_not``.

    Args:
        x (Tensor): Input Tensor of ``bitwise_invert``. It is a N-D Tensor of bool, uint8, int8, int16, int32, int64.
        out (Tensor|None, optional): Result of ``bitwise_invert``. It is a N-D Tensor with the same data type as the input Tensor. Default: None.
        name (str|None, optional): The default value is None. This property is typically not set by the user.
            For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor: Result of ``bitwise_invert``. It is a N-D Tensor with the same data type as the input Tensor.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> x = paddle.to_tensor([-5, -1, 1])
            >>> res = x.bitwise_invert()
            >>> print(res)
            Tensor(shape=[3], dtype=int64, place=Place(cpu), stop_gradient=True,
            [ 4,  0, -2])

    """
    # Directly call bitwise_not for the implementation
    return bitwise_not(x, out=out, name=name)


@inplace_apis_in_dygraph_only
def bitwise_invert_(x: Tensor, name: str | None = None) -> Tensor:
    r"""
    Inplace version of ``bitwise_invert`` API, the output Tensor will be inplaced with input ``x``.
    Please refer to :ref:`api_paddle_bitwise_invert_`.
    """
    # Directly call bitwise_not_ for the implementation
    return bitwise_not_(x, name=name)


def isclose(
    x: Tensor,
    y: Tensor,
    rtol: float = 1e-05,
    atol: float = 1e-08,
    equal_nan: bool = False,
    name: str | None = None,
) -> Tensor:
    r"""
    Check if all :math:`x` and :math:`y` satisfy the condition:

    .. math::

        \left| x - y \right| \leq atol + rtol \times \left| y \right|

    elementwise, for all elements of :math:`x` and :math:`y`. The behaviour of this
    operator is analogous to :math:`numpy.isclose`, namely that it returns :math:`True` if
    two tensors are elementwise equal within a tolerance.

    Args:
        x(Tensor): The input tensor, it's data type should be float16, float32, float64, complex64, complex128.
        y(Tensor): The input tensor, it's data type should be float16, float32, float64, complex64, complex128.
        rtol(float, optional): The relative tolerance. Default: :math:`1e-5` .
        atol(float, optional): The absolute tolerance. Default: :math:`1e-8` .
        equal_nan(bool, optional): If :math:`True` , then two :math:`NaNs` will be compared as equal. Default: :math:`False` .
        name (str|None, optional): Name for the operation. For more information, please
            refer to :ref:`api_guide_Name`. Default: None.

    Returns:
        Tensor: The output tensor, it's data type is bool.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> x = paddle.to_tensor([10000., 1e-07])
            >>> y = paddle.to_tensor([10000.1, 1e-08])
            >>> result1 = paddle.isclose(x, y, rtol=1e-05, atol=1e-08,
            ...                          equal_nan=False, name="ignore_nan")
            >>> print(result1)
            Tensor(shape=[2], dtype=bool, place=Place(cpu), stop_gradient=True,
            [True , False])
            >>> result2 = paddle.isclose(x, y, rtol=1e-05, atol=1e-08,
            ...                          equal_nan=True, name="equal_nan")
            >>> print(result2)
            Tensor(shape=[2], dtype=bool, place=Place(cpu), stop_gradient=True,
            [True , False])
            >>> x = paddle.to_tensor([1.0, float('nan')])
            >>> y = paddle.to_tensor([1.0, float('nan')])
            >>> result1 = paddle.isclose(x, y, rtol=1e-05, atol=1e-08,
            ...                          equal_nan=False, name="ignore_nan")
            >>> print(result1)
            Tensor(shape=[2], dtype=bool, place=Place(cpu), stop_gradient=True,
            [True , False])
            >>> result2 = paddle.isclose(x, y, rtol=1e-05, atol=1e-08,
            ...                          equal_nan=True, name="equal_nan")
            >>> print(result2)
            Tensor(shape=[2], dtype=bool, place=Place(cpu), stop_gradient=True,
            [True, True])
    """

    if in_dynamic_or_pir_mode():
        if in_pir_mode():
            check_variable_and_dtype(
                x,
                "input",
                ['float16', 'float32', 'float64', 'complex64', 'complex128'],
                'isclose',
            )
            check_variable_and_dtype(
                y,
                "input",
                ['float16', 'float32', 'float64', 'complex64', 'complex128'],
                'isclose',
            )
            if isinstance(rtol, paddle.pir.Value):
                check_variable_and_dtype(
                    rtol,
                    "input",
                    ['float64'],
                    'isclose',
                )
            else:
                check_type(rtol, 'rtol', float, 'isclose')
            if isinstance(atol, paddle.pir.Value):
                check_variable_and_dtype(
                    atol,
                    "input",
                    ['float64'],
                    'isclose',
                )
            else:
                check_type(atol, 'atol', float, 'isclose')
            check_type(equal_nan, 'equal_nan', bool, 'isclose')
        return _C_ops.isclose(x, y, rtol, atol, equal_nan)
    else:
        check_variable_and_dtype(
            x,
            "input",
            ['float16', 'float32', 'float64', 'complex64', 'complex128'],
            'isclose',
        )
        check_variable_and_dtype(
            y,
            "input",
            ['float16', 'float32', 'float64', 'complex64', 'complex128'],
            'isclose',
        )
        check_type(rtol, 'rtol', float, 'isclose')
        check_type(atol, 'atol', float, 'isclose')
        check_type(equal_nan, 'equal_nan', bool, 'isclose')

        helper = LayerHelper("isclose", **locals())
        out = helper.create_variable_for_type_inference(dtype='bool')

        inputs = {'Input': x, 'Other': y}
        outputs = {'Out': out}
        attrs = {'rtol': str(rtol), 'atol': str(atol), 'equal_nan': equal_nan}
        helper.append_op(
            type='isclose', inputs=inputs, outputs=outputs, attrs=attrs
        )
        return out
