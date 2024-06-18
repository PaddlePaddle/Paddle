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

# TODO: define logic functions of a tensor

import paddle

from ..base.data_feeder import check_type, check_variable_and_dtype
from ..common_ops_import import Variable

Tensor = paddle.base.framework.core.eager.Tensor

from paddle import _C_ops
from paddle.tensor.creation import full
from paddle.tensor.math import broadcast_shape
from paddle.utils.inplace_utils import inplace_apis_in_dygraph_only

from ..framework import (
    LayerHelper,
    in_dynamic_mode,
    in_dynamic_or_pir_mode,
    in_pir_mode,
)

__all__ = []


def _logical_op(op_name, x, y, out=None, name=None, binary_op=True):
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


def logical_and(x, y, out=None, name=None):
    r"""

    Compute element-wise logical AND on ``x`` and ``y``, and return ``out``. ``out`` is N-dim boolean ``Tensor``.
    Each element of ``out`` is calculated by

    .. math::

        out = x \&\& y

    Note:
        ``paddle.logical_and`` supports broadcasting. If you want know more about broadcasting, please refer to `Introduction to Tensor`_ .

        .. _Introduction to Tensor: ../../guides/beginner/tensor_en.html#chapter5-broadcasting-of-tensor

    Args:
        x (Tensor): the input tensor, it's data type should be one of bool, int8, int16, in32, in64, float16, float32, float64, complex64, complex128.
        y (Tensor): the input tensor, it's data type should be one of bool, int8, int16, in32, in64, float16, float32, float64, complex64, complex128.
        out(Tensor, optional): The ``Tensor`` that specifies the output of the operator, which can be any ``Tensor`` that has been created in the program. The default value is None, and a new ``Tensor`` will be created to save the output.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

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
def logical_and_(x, y, name=None):
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


def logical_or(x, y, out=None, name=None):
    """

    ``logical_or`` operator computes element-wise logical OR on ``x`` and ``y``, and returns ``out``. ``out`` is N-dim boolean ``Tensor``.
    Each element of ``out`` is calculated by

    .. math::

        out = x || y

    Note:
        ``paddle.logical_or`` supports broadcasting. If you want know more about broadcasting, please refer to `Introduction to Tensor`_ .

        .. _Introduction to Tensor: ../../guides/beginner/tensor_en.html#chapter5-broadcasting-of-tensor

    Args:
        x (Tensor): the input tensor, it's data type should be one of bool, int8, int16, in32, in64, float16, float32, float64, complex64, complex128.
        y (Tensor): the input tensor, it's data type should be one of bool, int8, int16, in32, in64, float16, float32, float64, complex64, complex128.
        out(Tensor): The ``Variable`` that specifies the output of the operator, which can be any ``Tensor`` that has been created in the program. The default value is None, and a new ``Tensor`` will be created to save the output.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

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
def logical_or_(x, y, name=None):
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


def logical_xor(x, y, out=None, name=None):
    r"""

    ``logical_xor`` operator computes element-wise logical XOR on ``x`` and ``y``, and returns ``out``. ``out`` is N-dim boolean ``Tensor``.
    Each element of ``out`` is calculated by

    .. math::

        out = (x || y) \&\& !(x \&\& y)

    Note:
        ``paddle.logical_xor`` supports broadcasting. If you want know more about broadcasting, please refer to `Introduction to Tensor`_ .

        .. _Introduction to Tensor: ../../guides/beginner/tensor_en.html#chapter5-broadcasting-of-tensor

    Args:
        x (Tensor): the input tensor, it's data type should be one of bool, int8, int16, int32, int64, float16, float32, float64, complex64, complex128.
        y (Tensor): the input tensor, it's data type should be one of bool, int8, int16, int32, int64, float16, float32, float64, complex64, complex128.
        out(Tensor): The ``Tensor`` that specifies the output of the operator, which can be any ``Tensor`` that has been created in the program. The default value is None, and a new ``Tensor`` will be created to save the output.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

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
def logical_xor_(x, y, name=None):
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


def logical_not(x, out=None, name=None):
    """

    ``logical_not`` operator computes element-wise logical NOT on ``x``, and returns ``out``. ``out`` is N-dim boolean ``Variable``.
    Each element of ``out`` is calculated by

    .. math::

        out = !x

    Note:
        ``paddle.logical_not`` supports broadcasting. If you want know more about broadcasting, please refer to `Introduction to Tensor`_ .

        .. _Introduction to Tensor: ../../guides/beginner/tensor_en.html#chapter5-broadcasting-of-tensor

    Args:

        x(Tensor):  Operand of logical_not operator. Must be a Tensor of type bool, int8, int16, in32, in64, float16, float32, or float64, complex64, complex128.
        out(Tensor): The ``Tensor`` that specifies the output of the operator, which can be any ``Tensor`` that has been created in the program. The default value is None, and a new ``Tensor` will be created to save the output.
        name(str|None): The default value is None. Normally there is no need for users to set this property. For more information, please refer to :ref:`api_guide_Name`.

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
def logical_not_(x, name=None):
    r"""
    Inplace version of ``logical_not`` API, the output Tensor will be inplaced with input ``x``.
    Please refer to :ref:`api_paddle_logical_not`.
    """
    if in_dynamic_mode():
        return _C_ops.logical_not_(x)


def is_empty(x, name=None):
    """

    Test whether a Tensor is empty.

    Args:
        x (Tensor): The Tensor to be tested.
        name (str, optional): The default value is ``None`` . Normally users don't have to set this parameter. For more information, please refer to :ref:`api_guide_Name` .

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


def equal_all(x, y, name=None):
    """
    Returns the truth value of :math:`x == y`. True if two inputs have the same elements, False otherwise.

    Note:
        The output has no gradient.

    Args:
        x(Tensor): Tensor, data type is bool, float32, float64, int32, int64.
        y(Tensor): Tensor, data type is bool, float32, float64, int32, int64.
        name(str, optional): The default value is None.  Normally there is no need for
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


def allclose(x, y, rtol=1e-05, atol=1e-08, equal_nan=False, name=None):
    r"""
    Check if all :math:`x` and :math:`y` satisfy the condition:

    .. math::
        \left| x - y \right| \leq atol + rtol \times \left| y \right|

    elementwise, for all elements of :math:`x` and :math:`y`. This is analogous to :math:`numpy.allclose`, namely that it returns :math:`True` if
    two tensors are elementwise equal within a tolerance.

    Args:
        x (Tensor): The input tensor, it's data type should be float16, float32, float64.
        y (Tensor): The input tensor, it's data type should be float16, float32, float64.
        rtol (rtoltype, optional): The relative tolerance. Default: :math:`1e-5` .
        atol (atoltype, optional): The absolute tolerance. Default: :math:`1e-8` .
        equal_nan (bool, optional): Whether to compare nan as equal. Default: False.
        name (str, optional): Name for the operation. For more information, please
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

    if in_dynamic_or_pir_mode():
        return _C_ops.allclose(x, y, rtol, atol, equal_nan)
    else:
        check_variable_and_dtype(
            x, "input", ['float16', 'float32', 'float64'], 'allclose'
        )
        check_variable_and_dtype(
            y, "input", ['float16', 'float32', 'float64'], 'allclose'
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


def equal(x, y, name=None):
    """

    This layer returns the truth value of :math:`x == y` elementwise.

    Note:
        The output has no gradient.

    Args:
        x (Tensor): Tensor, data type is bool, float16, float32, float64, uint8, int8, int16, int32, int64.
        y (Tensor): Tensor, data type is bool, float16, float32, float64, uint8, int8, int16, int32, int64.
        name (str, optional): The default value is None. Normally there is no need for
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
    if not isinstance(y, (int, bool, float, Variable, paddle.pir.Value)):
        raise TypeError(
            f"Type of input args must be float, bool, int or Tensor, but received type {type(y)}"
        )
    if not isinstance(y, (Variable, paddle.pir.Value)):
        y = full(shape=[], dtype=x.dtype, fill_value=y)

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
def equal_(x, y, name=None):
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


def greater_equal(x, y, name=None):
    """
    Returns the truth value of :math:`x >= y` elementwise, which is equivalent function to the overloaded operator `>=`.

    Note:
        The output has no gradient.

    Args:
        x (Tensor): First input to compare which is N-D tensor. The input data type should be bool, float16, float32, float64, uint8, int8, int16, int32, int64.
        y (Tensor): Second input to compare which is N-D tensor. The input data type should be bool, float16, float32, float64, uint8, int8, int16, int32, int64.
        name (str, optional): The default value is None.  Normally there is no need for
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
def greater_equal_(x, y, name=None):
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


def greater_than(x, y, name=None):
    """
    Returns the truth value of :math:`x > y` elementwise, which is equivalent function to the overloaded operator `>`.

    Note:
        The output has no gradient.

    Args:
        x (Tensor): First input to compare which is N-D tensor. The input data type should be bool, float16, float32, float64, uint8, int8, int16, int32, int64.
        y (Tensor): Second input to compare which is N-D tensor. The input data type should be bool, float16, float32, float64, uint8, int8, int16, int32, int64.
        name (str, optional): The default value is None.  Normally there is no need for
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
def greater_than_(x, y, name=None):
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


def less_equal(x, y, name=None):
    """
    Returns the truth value of :math:`x <= y` elementwise, which is equivalent function to the overloaded operator `<=`.

    Note:
        The output has no gradient.

    Args:
        x (Tensor): First input to compare which is N-D tensor. The input data type should be bool, float16, float32, float64, uint8, int8, int16, int32, int64.
        y (Tensor): Second input to compare which is N-D tensor. The input data type should be bool, float16, float32, float64, uint8, int8, int16, int32, int64.
        name (str, optional): The default value is None.  Normally there is no need for
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
def less_equal_(x, y, name=None):
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


def less_than(x, y, name=None):
    """
    Returns the truth value of :math:`x < y` elementwise, which is equivalent function to the overloaded operator `<`.

    Note:
        The output has no gradient.

    Args:
        x (Tensor): First input to compare which is N-D tensor. The input data type should be bool, float16, float32, float64, uint8, int8, int16, int32, int64.
        y (Tensor): Second input to compare which is N-D tensor. The input data type should be bool, float16, float32, float64, uint8, int8, int16, int32, int64.
        name (str, optional): The default value is None.  Normally there is no need for
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
def less_than_(x, y, name=None):
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


def not_equal(x, y, name=None):
    """
    Returns the truth value of :math:`x != y` elementwise, which is equivalent function to the overloaded operator `!=`.

    Note:
        The output has no gradient.

    Args:
        x (Tensor): First input to compare which is N-D tensor. The input data type should be bool, float32, float64, uint8, int8, int16, int32, int64.
        y (Tensor): Second input to compare which is N-D tensor. The input data type should be bool, float32, float64, uint8, int8, int16, int32, int64.
        name (str, optional): The default value is None.  Normally there is no need for
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
def not_equal_(x, y, name=None):
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


def is_tensor(x):
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
        return isinstance(
            x, (Tensor, paddle.base.core.eager.Tensor, paddle.pir.Value)
        )
    else:
        return isinstance(x, Variable)


def _bitwise_op(op_name, x, y, out=None, name=None, binary_op=True):
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


def bitwise_and(x, y, out=None, name=None):
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
        out (Tensor, optional): Result of ``bitwise_and`` . It is a N-D Tensor with the same data type of input Tensor. Default: None.
        name (str, optional): The default value is None.  Normally there is no need for
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


@inplace_apis_in_dygraph_only
def bitwise_and_(x, y, name=None):
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


def bitwise_or(x, y, out=None, name=None):
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
        out (Tensor, optional): Result of ``bitwise_or`` . It is a N-D Tensor with the same data type of input Tensor. Default: None.
        name (str, optional): The default value is None.  Normally there is no need for
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


@inplace_apis_in_dygraph_only
def bitwise_or_(x, y, name=None):
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


def bitwise_xor(x, y, out=None, name=None):
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
        out (Tensor, optional): Result of ``bitwise_xor`` . It is a N-D Tensor with the same data type of input Tensor. Default: None.
        name (str, optional): The default value is None.  Normally there is no need for
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


@inplace_apis_in_dygraph_only
def bitwise_xor_(x, y, name=None):
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


def bitwise_not(x, out=None, name=None):
    r"""

    Apply ``bitwise_not`` on Tensor ``X``.

    .. math::
        Out = \sim X

    Note:
        ``paddle.bitwise_not`` supports broadcasting. If you want know more about broadcasting, please refer to please refer to `Introduction to Tensor`_ .

        .. _Introduction to Tensor: ../../guides/beginner/tensor_en.html#chapter5-broadcasting-of-tensor

    Args:
        x (Tensor): Input Tensor of ``bitwise_not`` . It is a N-D Tensor of bool, uint8, int8, int16, int32, int64.
        out (Tensor, optional): Result of ``bitwise_not`` . It is a N-D Tensor with the same data type of input Tensor. Default: None.
        name (str, optional): The default value is None.  Normally there is no need for
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
def bitwise_not_(x, name=None):
    r"""
    Inplace version of ``bitwise_not`` API, the output Tensor will be inplaced with input ``x``.
    Please refer to :ref:`api_paddle_bitwise_not`.
    """
    if in_dynamic_mode():
        return _C_ops.bitwise_not_(x)


def isclose(x, y, rtol=1e-05, atol=1e-08, equal_nan=False, name=None):
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
        rtol(rtoltype, optional): The relative tolerance. Default: :math:`1e-5` .
        atol(atoltype, optional): The absolute tolerance. Default: :math:`1e-8` .
        equal_nan(equalnantype, optional): If :math:`True` , then two :math:`NaNs` will be compared as equal. Default: :math:`False` .
        name (str, optional): Name for the operation. For more information, please
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
