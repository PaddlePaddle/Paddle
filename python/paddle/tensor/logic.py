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
from paddle._C_ops import (  # noqa: F401
    allclose,
    bitwise_and,
    bitwise_and_,
    bitwise_not,
    bitwise_not_,
    bitwise_or,
    bitwise_or_,
    bitwise_xor,
    bitwise_xor_,
    greater_than,
    isclose,
    logical_and,
    logical_not,
    logical_or,
    logical_xor,
)
from paddle.tensor.creation import full
from paddle.tensor.math import broadcast_shape
from paddle.utils.decorator_utils import (
    param_one_alias,
    param_two_alias,
)
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
        .. code-block:: pycon

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
        .. code-block:: pycon

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


@param_two_alias(["x", "input"], ["y", "other"])
def equal(
    x: Tensor, y: Tensor, name: str | None = None, *, out: Tensor | None = None
) -> Tensor:
    """

    This layer returns the truth value of :math:`x == y` elementwise.

    Note:
        The output has no gradient.

    Args:
        x (Tensor): Tensor, data type is bool, float16, float32, float64, uint8, int8, int16, int32, int64, complex64, complex128.
            alias: ``input``
        y (Tensor): Tensor, data type is bool, float16, float32, float64, uint8, int8, int16, int32, int64, complex64, complex128.
            alias: ``other``
        name (str|None, optional): The default value is None. Normally there is no need for
            user to set this property.  For more information, please refer to :ref:`api_guide_Name`.
        out (Tensor, optional): Output tensor. If provided, the result will be stored in this tensor.

    Returns:
        Tensor: output Tensor, it's shape is the same as the input's Tensor,
        and the data type is bool. The result of this op is stop_gradient.

    Examples:
        .. code-block:: pycon

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
        return _C_ops.equal(x, y, out=out)
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


# Current op mechanism does not support `Tensor.op1(other)` if op1 is an alias for op2 and op2 has been sunk to C++ layer.
# Since greater_than has been sunk, `gt` is added here to avoid the alias issue.
# TODO(LittleHeroZZZX): Please remove this and use alias instead once the issue described above is fixed. @DanielSun11
@param_two_alias(["x", "input"], ["y", "other"])
def gt(
    x: Tensor, y: Tensor, name: str | None = None, *, out: Tensor | None = None
) -> Tensor:
    """
    Returns the truth value of :math:`x > y` elementwise, which is equivalent function to the overloaded operator `>`.

    Note:
        The output has no gradient.

    Args:
        x (Tensor): First input to compare which is N-D tensor. The input data type should be bool, bfloat16, float16, float32, float64, uint8, int8, int16, int32, int64, complex64, complex128.
            Alias: ``input``.
        y (Tensor): Second input to compare which is N-D tensor. The input data type should be bool, bfloat16, float16, float32, float64, uint8, int8, int16, int32, int64, complex64, complex128.
            Alias: ``other``.
        name (str|None, optional): The default value is None.  Normally there is no need for
            user to set this property.  For more information, please refer to :ref:`api_guide_Name`.
        out (Tensor, optional): The output tensor. If provided, the result will be stored in this tensor.

    Returns:
        Tensor: The output shape is same as input :attr:`x`. The output data type is bool.

    Examples:
        .. code-block:: pycon

            >>> import paddle

            >>> x = paddle.to_tensor([1, 2, 3])
            >>> y = paddle.to_tensor([1, 3, 2])
            >>> result1 = paddle.gt(x, y)
            >>> print(result1)
            Tensor(shape=[3], dtype=bool, place=Place(cpu), stop_gradient=True,
            [False, False, True ])
    """
    if in_dynamic_or_pir_mode():
        return _C_ops.greater_than(x, y, out=out)
    else:
        raise NotImplementedError(
            "paddle.gt does not support legacy static mode."
        )


@param_two_alias(["x", "input"], ["y", "other"])
def greater_equal(
    x: Tensor, y: Tensor, name: str | None = None, *, out: Tensor | None = None
) -> Tensor:
    """
    Returns the truth value of :math:`x >= y` elementwise, which is equivalent function to the overloaded operator `>=`.

    Note:
        The output has no gradient.

    Args:
        x (Tensor): First input to compare which is N-D tensor. The input data type should be bool, bfloat16, float16, float32, float64, uint8, int8, int16, int32, int64, complex64, complex128.
            Alias: ``input``.
        y (Tensor): Second input to compare which is N-D tensor. The input data type should be bool, bfloat16, float16, float32, float64, uint8, int8, int16, int32, int64, complex64, complex128.
            Alias: ``other``.
        name (str|None, optional): The default value is None.  Normally there is no need for
            user to set this property.  For more information, please refer to :ref:`api_guide_Name`.
        out (Tensor, optional): The output tensor. If set, the result will be stored in this tensor. Default is None.
    Returns:
        Tensor: The output shape is same as input :attr:`x`. The output data type is bool.

    Examples:
        .. code-block:: pycon

            >>> import paddle

            >>> x = paddle.to_tensor([1, 2, 3])
            >>> y = paddle.to_tensor([1, 3, 2])
            >>> result1 = paddle.greater_equal(x, y)
            >>> print(result1)
            Tensor(shape=[3], dtype=bool, place=Place(cpu), stop_gradient=True,
            [True , False, True ])
    """
    if in_dynamic_or_pir_mode():
        return _C_ops.greater_equal(x, y, out=out)
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
                "complex64",
                "complex128",
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


@param_two_alias(["x", "input"], ["y", "other"])
def less_equal(
    x: Tensor, y: Tensor, name: str | None = None, *, out: Tensor | None = None
) -> Tensor:
    """
    Returns the truth value of :math:`x <= y` elementwise, which is equivalent function to the overloaded operator `<=`.

    Note:
        The output has no gradient.

    Args:
        x (Tensor): First input to compare which is N-D tensor. The input data type should be bool, bfloat16, float16, float32, float64, uint8, int8, int16, int32, int64, complex64, complex128.
            Alias: ``input``.
        y (Tensor): Second input to compare which is N-D tensor. The input data type should be bool, bfloat16, float16, float32, float64, uint8, int8, int16, int32, int64, complex64, complex128.
            Alias: ``other``.
        name (str|None, optional): The default value is None.  Normally there is no need for
            user to set this property.  For more information, please refer to :ref:`api_guide_Name`.
        out (Tensor, optional): The output tensor. If set, the result will be stored in this tensor. Default is None.

    Returns:
        Tensor: The output shape is same as input :attr:`x`. The output data type is bool.

    Examples:
        .. code-block:: pycon

            >>> import paddle

            >>> x = paddle.to_tensor([1, 2, 3])
            >>> y = paddle.to_tensor([1, 3, 2])
            >>> result1 = paddle.less_equal(x, y)
            >>> print(result1)
            Tensor(shape=[3], dtype=bool, place=Place(cpu), stop_gradient=True,
            [True , True , False])
    """
    if in_dynamic_or_pir_mode():
        return _C_ops.less_equal(x, y, out=out)
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
                "complex64",
                "complex128",
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


@param_two_alias(["x", "input"], ["y", "other"])
def less_than(
    x: Tensor, y: Tensor, name: str | None = None, *, out: Tensor | None = None
) -> Tensor:
    """
    Returns the truth value of :math:`x < y` elementwise, which is equivalent function to the overloaded operator `<`.

    Note:
        The output has no gradient.

    Args:
        x (Tensor): First input to compare which is N-D tensor. The input data type should be bool, bfloat16, float16, float32, float64, uint8, int8, int16, int32, int64, complex64, complex128.
            Alias: ``input``
        y (Tensor): Second input to compare which is N-D tensor. The input data type should be bool, bfloat16, float16, float32, float64, uint8, int8, int16, int32, int64, complex64, complex128.
            Alias: ``other``
        name (str|None, optional): The default value is None.  Normally there is no need for
            user to set this property.  For more information, please refer to :ref:`api_guide_Name`.
        out (Tensor, optional): The output tensor. If set, the result will be stored in this tensor. Default is None.

    Returns:
        Tensor: The output shape is same as input :attr:`x`. The output data type is bool.

    Examples:
        .. code-block:: pycon

            >>> import paddle

            >>> x = paddle.to_tensor([1, 2, 3])
            >>> y = paddle.to_tensor([1, 3, 2])
            >>> result1 = paddle.less_than(x, y)
            >>> print(result1)
            Tensor(shape=[3], dtype=bool, place=Place(cpu), stop_gradient=True,
            [False, True , False])
    """
    if in_dynamic_or_pir_mode():
        return _C_ops.less_than(x, y, out=out)
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
                "complex64",
                "complex128",
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


@inplace_apis_in_dygraph_only
def less_(x: Tensor, y: Tensor, name: str | None = None) -> Tensor:
    r"""
    Inplace version of ``less_`` API, the output Tensor will be inplaced with input ``x``.
    Please refer to :ref:`api_paddle_less`.
    """

    # Directly call less_than_ API
    return less_than_(x, y, name)


@param_two_alias(["x", "input"], ["y", "other"])
def not_equal(
    x: Tensor, y: Tensor, name: str | None = None, *, out: Tensor | None = None
) -> Tensor:
    """
    Returns the truth value of :math:`x != y` elementwise, which is equivalent function to the overloaded operator `!=`.

    Note:
        The output has no gradient.

    Args:
        x (Tensor): First input to compare which is N-D tensor. The input data type should be bool, bfloat16, float16, float32, float64, uint8, int8, int16, int32, int64, complex64, complex128.
            Alias: ``input``.
        y (Tensor): Second input to compare which is N-D tensor. The input data type should be bool, bfloat16, float16, float32, float64, uint8, int8, int16, int32, int64, complex64, complex128.
            Alias: ``other``.
        name (str|None, optional): The default value is None.  Normally there is no need for
            user to set this property.  For more information, please refer to :ref:`api_guide_Name`.
        out (Tensor, optional): The output tensor. If set, the result will be stored in this tensor. Default is None.

    Returns:
        Tensor: The output shape is same as input :attr:`x`. The output data type is bool.

    Examples:
        .. code-block:: pycon

            >>> import paddle

            >>> x = paddle.to_tensor([1, 2, 3])
            >>> y = paddle.to_tensor([1, 3, 2])
            >>> result1 = paddle.not_equal(x, y)
            >>> print(result1)
            Tensor(shape=[3], dtype=bool, place=Place(cpu), stop_gradient=True,
            [False, True , True ])
    """
    if in_dynamic_or_pir_mode():
        return _C_ops.not_equal(x, y, out=out)
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


@param_one_alias(["x", "obj"])
def is_tensor(x: Any) -> TypeGuard[Tensor]:
    """

    Tests whether input object is a paddle.Tensor.

    .. note::
        Alias Support: The parameter name ``obj`` can be used as an alias for ``x``.
        For example, ``is_tensor(obj=tensor_x)`` is equivalent to ``is_tensor(x=tensor_x)``.

    Args:
        x (object): Object to test. alias: ``obj``.

    Returns:
        A boolean value. True if ``x`` is a paddle.Tensor, otherwise False.

    Examples:
        .. code-block:: pycon

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


def __rand__(x: Tensor, y: int | bool):
    if isinstance(y, (int, bool)):
        y_tensor = paddle.to_tensor(y, dtype=x.dtype)
        return bitwise_and(y_tensor, x)
    else:
        raise TypeError(
            f"unsupported operand type(s) for |: '{type(y).__name__}' and 'Tensor'"
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
        .. code-block:: pycon

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
