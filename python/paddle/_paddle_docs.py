# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

import inspect
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Literal

    import paddle
    from paddle import Tensor
    from paddle._typing import DataLayout2D, DTypeLike

from .base.dygraph.generated_tensor_methods_patch import _all_method_op_map


def add_doc_and_signature(func_def):
    """
    Decorator for documentation-only shell functions.

    The decorated function has no implementation (body is ``...``); its sole purpose
    is to carry an API signature and a docstring, like a ``.pyi`` stub. At import time
    this decorator copies that signature / docstring onto the real paddle objects
    (``paddle.*`` / ``paddle.Tensor.*`` / ``paddle.nn.functional.*``) wherever they exist.
    The shell body is never executed.
    """

    func_name = func_def.__name__
    docstr = inspect.getdoc(func_def)
    signature = inspect.signature(func_def)
    for _, generated_name, generated_func in _all_method_op_map:
        if generated_name == func_name:
            generated_func.__doc__ = docstr
            generated_func.__signature__ = signature
            break
    return func_def


@add_doc_and_signature
def acos(
    x: Tensor,
    name: str | None = None,
    *,
    out: Tensor | None = None,
) -> Tensor:
    r"""
    Acos Activation Operator.

    .. math::
        out = cos^{-1}(x)

    Args:
        x (Tensor): Input of Acos operator, an N-D Tensor, with data type float32, float64, float16, bfloat16,
            uint8, int8, int16, int32, int64, complex64 or complex128. Alias: ``input``.
        name (str|None, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Keyword Args:
        out (Tensor|optional): The output tensor. Default: None.

    Returns:
        Tensor. Output of Acos operator, a Tensor with shape same as input
            (integer types are autocasted into float32).

    Examples:
        .. code-block:: pycon

            >>> import paddle

            >>> x = paddle.to_tensor([-0.4, -0.2, 0.1, 0.3])
            >>> out = paddle.acos(x)
            >>> print(out)
            Tensor(shape=[4], dtype=float32, place=Place(cpu), stop_gradient=True,
            [1.98231316, 1.77215421, 1.47062886, 1.26610363])
    """
    ...


@add_doc_and_signature
def acosh(
    x: Tensor,
    name: str | None = None,
    *,
    out: Tensor | None = None,
) -> Tensor:
    r"""
    Acosh Activation Operator.

    .. math::
       out = acosh(x)

    Args:
        x (Tensor): Input of Acosh operator, an N-D Tensor, with data type float32, float64, float16, bfloat16,
            uint8, int8, int16, int32, int64, complex64 or complex128. Alias: ``input``.
        name (str|None, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Keyword Args:
        out (Tensor|optional): The output tensor. Default: None.

    Returns:
        Tensor. Output of Acosh operator, a Tensor with shape same as input
            (integer types are autocasted into float32).

    Examples:
        .. code-block:: pycon

            >>> import paddle

            >>> x = paddle.to_tensor([1.0, 3.0, 4.0, 5.0])
            >>> out = paddle.acosh(x)
            >>> print(out)
            Tensor(shape=[4], dtype=float32, place=Place(cpu), stop_gradient=True,
            [0.        , 1.76274717, 2.06343699, 2.29243159])
    """
    ...


@add_doc_and_signature
def sinh(
    x: Tensor,
    name: str | None = None,
    *,
    out: Tensor | None = None,
) -> Tensor:
    r"""
    Sinh Activation Operator.

    .. math::
       out = sinh(x)

    Args:
        x (Tensor): Input of Sinh operator, an N-D Tensor, with data type float32, float64, float16, bfloat16,
            uint8, int8, int16, int32, int64, complex64 or complex128. Alias: ``input``.
        name (str|None, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Keyword Args:
        out (Tensor|optional): The output tensor. Default: None.

    Returns:
        Tensor. Output of Sinh operator, a Tensor with shape same as input
            (integer types are autocasted into float32).

    Examples:
        .. code-block:: pycon

            >>> import paddle

            >>> x = paddle.to_tensor([-0.4, -0.2, 0.1, 0.3])
            >>> out = paddle.sinh(x)
            >>> print(out)
            Tensor(shape=[4], dtype=float32, place=Place(cpu), stop_gradient=True,
            [-0.41075233, -0.20133601,  0.10016675,  0.30452031])
    """
    ...


@add_doc_and_signature
def amin(
    x: Tensor,
    axis: int | Sequence[int] | None = None,
    keepdim: bool = False,
    name: str | None = None,
    *,
    out: Tensor | None = None,
) -> Tensor:
    r"""
    Computes the minimum of tensor elements over the given axis

    Note:
        The difference between min and amin is: If there are multiple minimum elements,
        amin evenly distributes gradient between these equal values,
        while min propagates gradient to all of them.

    Args:
        x (Tensor): A tensor, the data type is float32, float64, int32, int64,
            the dimension is no more than 4.
        axis (int|list|tuple|None, optional): The axis along which the minimum is computed.
            If :attr:`None`, compute the minimum over all elements of
            `x` and return a Tensor with a single element,
            otherwise must be in the range :math:`[-x.ndim, x.ndim)`.
            If :math:`axis[i] < 0`, the axis to reduce is :math:`x.ndim + axis[i]`.
        keepdim (bool, optional): Whether to reserve the reduced dimension in the
            output Tensor. The result tensor will have one fewer dimension
            than the `x` unless :attr:`keepdim` is true, default
            value is False.
        out (Tensor|None, optional): Output tensor. If provided in dynamic graph, the result will
            be written to this tensor and also returned. The returned tensor and `out` share memory
            and autograd meta. Default: None.
        name (str|None, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor, results of minimum on the specified axis of input tensor,
        it's data type is the same as input's Tensor.
    Keyword args:
        out(Tensor, optional): The output tensor.
    Examples:
        .. code-block:: pycon

            >>> import paddle
            >>> # data_x is a Tensor with shape [2, 4] with multiple minimum elements
            >>> # the axis is a int element

            >>> x = paddle.to_tensor([[0.2, 0.1, 0.1, 0.1], [0.1, 0.1, 0.6, 0.7]], dtype='float64', stop_gradient=False)
            >>> # There are 5 minimum elements:
            >>> # 1) amin evenly distributes gradient between these equal values,
            >>> #    thus the corresponding gradients are 1/5=0.2;
            >>> # 2) while min propagates gradient to all of them,
            >>> #    thus the corresponding gradient are 1.
            >>> result1 = paddle.amin(x)
            >>> result1.backward()
            >>> result1
            Tensor(shape=[], dtype=float64, place=Place(cpu), stop_gradient=False,
            0.10000000)
            >>> x.grad
            Tensor(shape=[2, 4], dtype=float64, place=Place(cpu), stop_gradient=False,
            [[0.        , 0.20000000, 0.20000000, 0.20000000],
             [0.20000000, 0.20000000, 0.        , 0.        ]])

            >>> x.clear_grad()
            >>> result1_min = paddle.min(x)
            >>> result1_min.backward()
            >>> result1_min
            Tensor(shape=[], dtype=float64, place=Place(cpu), stop_gradient=False,
            0.10000000)


            >>> x.clear_grad()
            >>> result2 = paddle.amin(x, axis=0)
            >>> result2.backward()
            >>> result2
            Tensor(shape=[4], dtype=float64, place=Place(cpu), stop_gradient=False,
            [0.10000000, 0.10000000, 0.10000000, 0.10000000])
            >>> x.grad
            Tensor(shape=[2, 4], dtype=float64, place=Place(cpu), stop_gradient=False,
            [[0.        , 0.50000000, 1.        , 1.        ],
             [1.        , 0.50000000, 0.        , 0.        ]])

            >>> x.clear_grad()
            >>> result3 = paddle.amin(x, axis=-1)
            >>> result3.backward()
            >>> result3
            Tensor(shape=[2], dtype=float64, place=Place(cpu), stop_gradient=False,
            [0.10000000, 0.10000000])
            >>> x.grad
            Tensor(shape=[2, 4], dtype=float64, place=Place(cpu), stop_gradient=False,
            [[0.        , 0.33333333, 0.33333333, 0.33333333],
             [0.50000000, 0.50000000, 0.        , 0.        ]])

            >>> x.clear_grad()
            >>> result4 = paddle.amin(x, axis=1, keepdim=True)
            >>> result4.backward()
            >>> result4
            Tensor(shape=[2, 1], dtype=float64, place=Place(cpu), stop_gradient=False,
            [[0.10000000],
             [0.10000000]])
            >>> x.grad
            Tensor(shape=[2, 4], dtype=float64, place=Place(cpu), stop_gradient=False,
            [[0.        , 0.33333333, 0.33333333, 0.33333333],
             [0.50000000, 0.50000000, 0.        , 0.        ]])

            >>> # data_y is a Tensor with shape [2, 2, 2]
            >>> # the axis is list
            >>> y = paddle.to_tensor([[[0.2, 0.1], [0.1, 0.1]], [[0.1, 0.1], [0.6, 0.7]]], dtype='float64', stop_gradient=False)
            >>> result5 = paddle.amin(y, axis=[1, 2])
            >>> result5.backward()
            >>> result5
            Tensor(shape=[2], dtype=float64, place=Place(cpu), stop_gradient=False,
            [0.10000000, 0.10000000])
            >>> y.grad
            Tensor(shape=[2, 2, 2], dtype=float64, place=Place(cpu), stop_gradient=False,
            [[[0.        , 0.33333333],
              [0.33333333, 0.33333333]],
             [[0.50000000, 0.50000000],
              [0.        , 0.        ]]])

            >>> y.clear_grad()
            >>> result6 = paddle.amin(y, axis=[0, 1])
            >>> result6.backward()
            >>> result6
            Tensor(shape=[2], dtype=float64, place=Place(cpu), stop_gradient=False,
            [0.10000000, 0.10000000])
            >>> y.grad
            Tensor(shape=[2, 2, 2], dtype=float64, place=Place(cpu), stop_gradient=False,
            [[[0.        , 0.33333333],
              [0.50000000, 0.33333333]],
             [[0.50000000, 0.33333333],
              [0.        , 0.        ]]])
    """
    ...


@add_doc_and_signature
def aminmax(
    x: Tensor,
    axis: int | Sequence[int] | None = None,
    keepdim: bool = False,
    *,
    out: tuple[Tensor, Tensor] | None = None,
) -> tuple[Tensor, Tensor]:
    r"""
    Computes both the minimum and maximum of tensor elements over the given axis.

    Note:
        Like amin and amax, if there are multiple minimum/maximum elements,
        aminmax evenly distributes gradient between these equal values.

    Args:
        x (Tensor): A tensor, the data type is float32, float64, int32, int64.
            Alias: ``input``.
        axis (int|list|tuple|None, optional): The axis along which the minimum and maximum
            are computed. If :attr:`None`, compute over all elements of
            `x` and return Tensors with a single element,
            otherwise must be in the range :math:`[-x.ndim, x.ndim)`.
            If :math:`axis[i] < 0`, the axis to reduce is :math:`x.ndim + axis[i]`.
            Alias: ``dim``.
        keepdim (bool, optional): Whether to reserve the reduced dimension in the
            output Tensors. The result tensors will have one fewer dimension
            than the `x` unless :attr:`keepdim` is true, default
            value is False.

    Keyword args:
        out(tuple(Tensor, Tensor), optional): The output tensors.

    Returns:
        tuple(Tensor, Tensor), the minimum and maximum results on the specified axis
        of input tensor, the data type is the same as `x`.

    Examples:
        .. code-block:: pycon

            >>> import paddle
            >>> x = paddle.to_tensor([[0.1, 0.9, 0.9, 0.9], [0.9, 0.9, 0.6, 0.7]], dtype='float64', stop_gradient=False)
            >>> # min_val, max_val = paddle.aminmax(x)  # doctest to be enabled after API is merged
    """
    ...


@add_doc_and_signature
def amax(
    x: Tensor,
    axis: int | Sequence[int] | None = None,
    keepdim: bool = False,
    name: str | None = None,
    *,
    out: Tensor | None = None,
) -> Tensor:
    r"""
    Computes the maximum of tensor elements over the given axis.

    Note:
        The difference between max and amax is: If there are multiple maximum elements,
        amax evenly distributes gradient between these equal values,
        while max propagates gradient to all of them.

    Args:
        x (Tensor): A tensor, the data type is float32, float64, int32, int64,
            the dimension is no more than 4.
        axis (int|list|tuple|None, optional): The axis along which the maximum is computed.
            If :attr:`None`, compute the maximum over all elements of
            `x` and return a Tensor with a single element,
            otherwise must be in the range :math:`[-x.ndim(x), x.ndim(x))`.
            If :math:`axis[i] < 0`, the axis to reduce is :math:`x.ndim + axis[i]`.
        keepdim (bool, optional): Whether to reserve the reduced dimension in the
            output Tensor. The result tensor will have one fewer dimension
            than the `x` unless :attr:`keepdim` is true, default
            value is False.
        out (Tensor|None, optional): Output tensor. If provided in dynamic graph, the result will
            be written to this tensor and also returned. The returned tensor and `out` share memory
            and autograd meta. Default: None.
        name (str|None, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.
    Keyword args:
        out(Tensor, optional): The output tensor.
    Returns:
        Tensor, results of maximum on the specified axis of input tensor,
        it's data type is the same as `x`.

    Examples:
        .. code-block:: pycon

            >>> import paddle
            >>> # data_x is a Tensor with shape [2, 4] with multiple maximum elements
            >>> # the axis is a int element

            >>> x = paddle.to_tensor([[0.1, 0.9, 0.9, 0.9], [0.9, 0.9, 0.6, 0.7]], dtype='float64', stop_gradient=False)
            >>> # There are 5 maximum elements:
            >>> # 1) amax evenly distributes gradient between these equal values,
            >>> #    thus the corresponding gradients are 1/5=0.2;
            >>> # 2) while max propagates gradient to all of them,
            >>> #    thus the corresponding gradient are 1.
            >>> result1 = paddle.amax(x)
            >>> result1.backward()
            >>> result1
            Tensor(shape=[], dtype=float64, place=Place(cpu), stop_gradient=False,
            0.90000000)
            >>> x.grad
            Tensor(shape=[2, 4], dtype=float64, place=Place(cpu), stop_gradient=False,
            [[0.        , 0.20000000, 0.20000000, 0.20000000],
             [0.20000000, 0.20000000, 0.        , 0.        ]])

            >>> x.clear_grad()
            >>> result1_max = paddle.max(x)
            >>> result1_max.backward()
            >>> result1_max
            Tensor(shape=[], dtype=float64, place=Place(cpu), stop_gradient=False,
            0.90000000)


            >>> x.clear_grad()
            >>> result2 = paddle.amax(x, axis=0)
            >>> result2.backward()
            >>> result2
            Tensor(shape=[4], dtype=float64, place=Place(cpu), stop_gradient=False,
            [0.90000000, 0.90000000, 0.90000000, 0.90000000])
            >>> x.grad
            Tensor(shape=[2, 4], dtype=float64, place=Place(cpu), stop_gradient=False,
            [[0.        , 0.50000000, 1.        , 1.        ],
             [1.        , 0.50000000, 0.        , 0.        ]])

            >>> x.clear_grad()
            >>> result3 = paddle.amax(x, axis=-1)
            >>> result3.backward()
            >>> result3
            Tensor(shape=[2], dtype=float64, place=Place(cpu), stop_gradient=False,
            [0.90000000, 0.90000000])
            >>> x.grad
            Tensor(shape=[2, 4], dtype=float64, place=Place(cpu), stop_gradient=False,
            [[0.        , 0.33333333, 0.33333333, 0.33333333],
             [0.50000000, 0.50000000, 0.        , 0.        ]])

            >>> x.clear_grad()
            >>> result4 = paddle.amax(x, axis=1, keepdim=True)
            >>> result4.backward()
            >>> result4
            Tensor(shape=[2, 1], dtype=float64, place=Place(cpu), stop_gradient=False,
            [[0.90000000],
             [0.90000000]])
            >>> x.grad
            Tensor(shape=[2, 4], dtype=float64, place=Place(cpu), stop_gradient=False,
            [[0.        , 0.33333333, 0.33333333, 0.33333333],
             [0.50000000, 0.50000000, 0.        , 0.        ]])

            >>> # data_y is a Tensor with shape [2, 2, 2]
            >>> # the axis is list
            >>> y = paddle.to_tensor([[[0.1, 0.9], [0.9, 0.9]], [[0.9, 0.9], [0.6, 0.7]]], dtype='float64', stop_gradient=False)
            >>> result5 = paddle.amax(y, axis=[1, 2])
            >>> result5.backward()
            >>> result5
            Tensor(shape=[2], dtype=float64, place=Place(cpu), stop_gradient=False,
            [0.90000000, 0.90000000])
            >>> y.grad
            Tensor(shape=[2, 2, 2], dtype=float64, place=Place(cpu), stop_gradient=False,
            [[[0.        , 0.33333333],
              [0.33333333, 0.33333333]],
             [[0.50000000, 0.50000000],
              [0.        , 0.        ]]])

            >>> y.clear_grad()
            >>> result6 = paddle.amax(y, axis=[0, 1])
            >>> result6.backward()
            >>> result6
            Tensor(shape=[2], dtype=float64, place=Place(cpu), stop_gradient=False,
            [0.90000000, 0.90000000])
            >>> y.grad
            Tensor(shape=[2, 2, 2], dtype=float64, place=Place(cpu), stop_gradient=False,
            [[[0.        , 0.33333333],
              [0.50000000, 0.33333333]],
             [[0.50000000, 0.33333333],
              [0.        , 0.        ]]])
    """
    ...


@add_doc_and_signature
def all(
    x: Tensor,
    axis: int | Sequence[int] | None = None,
    keepdim: bool = False,
    name: str | None = None,
    *,
    out: Tensor | None = None,
) -> Tensor:
    r"""
    Computes the ``logical and`` of tensor elements over the given dimension.

    Args:
        x (Tensor): An N-D Tensor, the input data type should be 'bool', 'float32', 'float64', 'int32', 'int64', 'complex64', 'complex128'.
        axis (int|list|tuple|None, optional): The dimensions along which the ``logical and`` is compute. If
            :attr:`None`, and all elements of :attr:`x` and return a
            Tensor with a single element, otherwise must be in the
            range :math:`[-rank(x), rank(x))`. If :math:`axis[i] < 0`,
            the dimension to reduce is :math:`rank + axis[i]`.
        keepdim (bool, optional): Whether to reserve the reduced dimension in the
            output Tensor. The result Tensor will have one fewer dimension
            than the :attr:`x` unless :attr:`keepdim` is true, default
            value is False.
        name (str|None, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Keyword Args:
        out (Tensor|optional): The output tensor.

    Returns:
        Tensor: Results the ``logical and`` on the specified axis of input Tensor `x`,  it's data type is bool.

    Examples:
        .. code-block:: pycon

            >>> import paddle
            >>> # x is a bool Tensor with following elements:
            >>> #    [[True, False]
            >>> #     [True, True]]
            >>> x = paddle.to_tensor([[1, 0], [1, 1]], dtype='int32')
            >>> x
            Tensor(shape=[2, 2], dtype=int32, place=Place(cpu), stop_gradient=True,
            [[1, 0],
             [1, 1]])
            >>> x = paddle.cast(x, 'bool')

            >>> # out1 should be False
            >>> out1 = paddle.all(x)
            >>> out1
            Tensor(shape=[], dtype=bool, place=Place(cpu), stop_gradient=True,
            False)

            >>> # out2 should be [True, False]
            >>> out2 = paddle.all(x, axis=0)
            >>> out2
            Tensor(shape=[2], dtype=bool, place=Place(cpu), stop_gradient=True,
            [True , False])

            >>> # keepdim=False, out3 should be [False, True], out.shape should be (2,)
            >>> out3 = paddle.all(x, axis=-1)
            >>> out3
            Tensor(shape=[2], dtype=bool, place=Place(cpu), stop_gradient=True,
            [False, True ])

            >>> # keepdim=True, out4 should be [[False], [True]], out.shape should be (2, 1)
            >>> out4 = paddle.all(x, axis=1, keepdim=True)
            >>> out4
            Tensor(shape=[2, 1], dtype=bool, place=Place(cpu), stop_gradient=True,
            [[False],
             [True ]])
    """
    ...


@add_doc_and_signature
def argmax(
    x: Tensor,
    axis: int | None = None,
    keepdim: bool = False,
    dtype: DTypeLike = "int64",
    name: str | None = None,
) -> Tensor:
    r"""
    Computes the indices of the max elements of the input tensor's
    element along the provided axis.

    Args:
        x (Tensor): An input N-D Tensor with type float16, float32, float64, int16,
            int32, int64, uint8.
        axis (int|None, optional): Axis to compute indices along. The effective range
            is [-R, R), where R is x.ndim. when axis < 0, it works the same way
            as axis + R. Default is None, the input `x` will be into the flatten tensor, and selecting the min value index.
        keepdim (bool, optional): Whether to keep the given axis in output. If it is True, the dimensions will be same as input x and with size one in the axis. Otherwise the output dimensions is one fewer than x since the axis is squeezed. Default is False.
        dtype (str|np.dtype, optional): Data type of the output tensor which can
                    be int32, int64. The default value is ``int64`` , and it will
                    return the int64 indices.
        name (str|None, optional): For details, please refer to :ref:`api_guide_Name`. Generally, no setting is required. Default: None.

    Returns:
        Tensor, return the tensor of int32 if set :attr:`dtype` is int32, otherwise return the tensor of int64.

    Examples:
        .. code-block:: pycon

            >>> import paddle

            >>> x = paddle.to_tensor([[5, 8, 9, 5], [0, 0, 1, 7], [6, 9, 2, 4]])
            >>> out1 = paddle.argmax(x)
            >>> print(out1.numpy())
            2
            >>> out2 = paddle.argmax(x, axis=0)
            >>> print(out2.numpy())
            [2 2 0 1]
            >>> out3 = paddle.argmax(x, axis=-1)
            >>> print(out3.numpy())
            [2 3 1]
            >>> out4 = paddle.argmax(x, axis=0, keepdim=True)
            >>> print(out4.numpy())
            [[2 2 0 1]]
    """
    ...


@add_doc_and_signature
def argmin(
    x: Tensor,
    axis: int | None = None,
    keepdim: bool = False,
    dtype: DTypeLike = "int64",
    name: str | None = None,
) -> Tensor:
    r"""
    Computes the indices of the min elements of the input tensor's
    element along the provided axis.

    Args:
        x (Tensor): An input N-D Tensor with type float16, float32, float64, int16,
            int32, int64, uint8.
        axis (int|None, optional): Axis to compute indices along. The effective range
            is [-R, R), where R is x.ndim. when axis < 0, it works the same way
            as axis + R. Default is None, the input `x` will be into the flatten tensor, and selecting the min value index.
        keepdim (bool, optional): Whether to keep the given axis in output. If it is True, the dimensions will be same as input x and with size one in the axis. Otherwise the output dimensions is one fewer than x since the axis is squeezed. Default is False.
        dtype (str|np.dtype, optional): Data type of the output tensor which can
                    be int32, int64. The default value is 'int64', and it will
                    return the int64 indices.
        name (str|None, optional): For details, please refer to :ref:`api_guide_Name`. Generally, no setting is required. Default: None.

    Returns:
        Tensor, return the tensor of `int32` if set :attr:`dtype` is `int32`, otherwise return the tensor of `int64`.

    Examples:
        .. code-block:: pycon

            >>> import paddle

            >>> x = paddle.to_tensor([[5, 8, 9, 5], [0, 0, 1, 7], [6, 9, 2, 4]])
            >>> out1 = paddle.argmin(x)
            >>> print(out1.numpy())
            4
            >>> out2 = paddle.argmin(x, axis=0)
            >>> print(out2.numpy())
            [1 1 1 2]
            >>> out3 = paddle.argmin(x, axis=-1)
            >>> print(out3.numpy())
            [0 0 2]
            >>> out4 = paddle.argmin(x, axis=0, keepdim=True)
            >>> print(out4.numpy())
            [[1 1 1 2]]
    """
    ...


@add_doc_and_signature
def atan(
    x: Tensor,
    name: str | None = None,
    *,
    out: Tensor | None = None,
) -> Tensor:
    r"""
    Arctangent Operator.

    .. math::
       out = tan^{-1}(x)

    Args:
        x (Tensor): Input of Atan operator, an N-D Tensor, with data type float32, float64, float16, bfloat16,
            uint8, int8, int16, int32, int64, complex64 or complex128. Alias: ``input``.
        name (str|None, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Keyword Args:
        out (Tensor|optional): The output tensor. Default: None.

    Returns:
        Tensor. Same shape and dtype as input x
            (integer types are autocasted into float32).

    Examples:
        .. code-block:: pycon

            >>> import paddle

            >>> x = paddle.to_tensor([-0.4, -0.2, 0.1, 0.3])
            >>> out = paddle.atan(x)
            >>> print(out)
            Tensor(shape=[4], dtype=float32, place=Place(cpu), stop_gradient=True,
            [-0.38050640, -0.19739556,  0.09966865,  0.29145682])
    """
    ...


@add_doc_and_signature
def atanh(
    x: Tensor,
    name: str | None = None,
    *,
    out: Tensor | None = None,
) -> Tensor:
    r"""
    Atanh Activation Operator.

    .. math::
       out = atanh(x)

    Args:
        x (Tensor): Input of Atan operator, an N-D Tensor, with data type float32, float64, float16, bfloat16,
            uint8, int8, int16, int32, int64, complex64 or complex128. Alias: ``input``.
        name (str|None, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Keyword Args:
        out (Tensor|optional): The output tensor. Default: None.

    Returns:
        Tensor. Output of Atanh operator, a Tensor with shape same as input
            (integer types are autocasted into float32).

    Examples:
        .. code-block:: pycon

            >>> import paddle

            >>> x = paddle.to_tensor([-0.4, -0.2, 0.1, 0.3])
            >>> out = paddle.atanh(x)
            >>> print(out)
            Tensor(shape=[4], dtype=float32, place=Place(cpu), stop_gradient=True,
            [-0.42364895, -0.20273255,  0.10033534,  0.30951962])
    """
    ...


@add_doc_and_signature
def atan2(
    x: Tensor,
    y: Tensor,
    name: str | None = None,
    *,
    out: Tensor | None = None,
) -> Tensor:
    r"""Element-wise arctangent of x/y with consideration of the quadrant.

    Equation:
        .. math::

            atan2(x,y)=\left\{\begin{matrix}
            & tan^{-1}(\frac{x}{y}) & y > 0 \\
            & tan^{-1}(\frac{x}{y}) + \pi & x>=0, y < 0 \\
            & tan^{-1}(\frac{x}{y}) - \pi & x<0, y < 0 \\
            & +\frac{\pi}{2} & x>0, y = 0 \\
            & -\frac{\pi}{2} & x<0, y = 0 \\
            &\text{undefined} & x=0, y = 0
            \end{matrix}\right.

    Args:
        x (Tensor): An N-D Tensor, the data type is int32, int64, float16, float32, float64.
            Alias: ``input``.
        y (Tensor): An N-D Tensor, must have the same type as `x`.
            Alias: ``other``.
        name (str|None, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.
        out (Tensor|None, optional): The output Tensor. If set, the result will be stored in this Tensor. Default is None.

    Returns:
        out (Tensor): An N-D Tensor, the shape and data type is the same with input (The output data type is float64 when the input data type is int).

    Examples:
        .. code-block:: pycon

            >>> import paddle

            >>> x = paddle.to_tensor([-1, +1, +1, -1]).astype('float32')
            >>> x
            Tensor(shape=[4], dtype=float32, place=Place(cpu), stop_gradient=True,
            [-1,  1,  1, -1])

            >>> y = paddle.to_tensor([-1, -1, +1, +1]).astype('float32')
            >>> y
            Tensor(shape=[4], dtype=float32, place=Place(cpu), stop_gradient=True,
            [-1,  -1,  1, 1])

            >>> out = paddle.atan2(x, y)
            >>> out
            Tensor(shape=[4], dtype=float32, place=Place(cpu), stop_gradient=True,
            [-2.35619450,  2.35619450,  0.78539819, -0.78539819])

    """
    ...


@add_doc_and_signature
def log2(
    x: Tensor,
    name: str | None = None,
    *,
    out: Tensor | None = None,
) -> Tensor:
    r"""
    Calculates the log to the base 2 of the given input tensor, element-wise.

    .. math::

        Out = \log_2x

    Args:
        x (Tensor): Input tensor must be one of the following types: int32, int64, float16, bfloat16, float32, float64, complex64, complex128.
        name (str|None, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.
        out (Tensor, optional): The output Tensor. If set, the result will be stored in this Tensor. Default: None.

    Returns:
        Tensor: The log to the base 2 of the input Tensor computed element-wise.

    Examples:

        .. code-block:: pycon

            >>> import paddle

            >>> # example 1: x is a float
            >>> x_i = paddle.to_tensor([[1.0], [2.0]])
            >>> res = paddle.log2(x_i)
            >>> res
            Tensor(shape=[2, 1], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[0.],
             [1.]])

            >>> # example 2: x is float32
            >>> x_i = paddle.full(shape=[1], fill_value=2, dtype='float32')
            >>> paddle.to_tensor(x_i)
            >>> res = paddle.log2(x_i)
            >>> res
            Tensor(shape=[1], dtype=float32, place=Place(cpu), stop_gradient=True,
            [1.])

            >>> # example 3: x is float64
            >>> x_i = paddle.full(shape=[1], fill_value=2, dtype='float64')
            >>> paddle.to_tensor(x_i)
            >>> res = paddle.log2(x_i)
            >>> res
            Tensor(shape=[1], dtype=float64, place=Place(cpu), stop_gradient=True,
            [1.])
    """
    ...


@add_doc_and_signature
def log10(
    x: Tensor,
    name: str | None = None,
    *,
    out: Tensor | None = None,
) -> Tensor:
    r"""
    Calculates the log to the base 10 of the given input tensor, element-wise.

    .. math::

        Out = \log_{10}x

    Args:
        x (Tensor): Input tensor must be one of the following types: int32, int64, float16, bfloat16, float32, float64, complex64, complex128.
        name (str|None, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.
        out (Tensor, optional): The output Tensor. If set, the result will be stored in this Tensor. Default: None.

    Returns:
        Tensor: The log to the base 10 of the input Tensor computed element-wise.

    Examples:

        .. code-block:: pycon

            >>> import paddle

            >>> # example 1: x is a float
            >>> x_i = paddle.to_tensor([[1.0], [10.0]])
            >>> res = paddle.log10(x_i)
            >>> res
            Tensor(shape=[2, 1], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[0.],
             [1.]])

            >>> # example 2: x is float32
            >>> x_i = paddle.full(shape=[1], fill_value=10, dtype='float32')
            >>> paddle.to_tensor(x_i)
            >>> res = paddle.log10(x_i)
            >>> res
            Tensor(shape=[1], dtype=float32, place=Place(cpu), stop_gradient=True,
            [1.])

            >>> # example 3: x is float64
            >>> x_i = paddle.full(shape=[1], fill_value=10, dtype='float64')
            >>> paddle.to_tensor(x_i)
            >>> res = paddle.log10(x_i)
            >>> res
            Tensor(shape=[1], dtype=float64, place=Place(cpu), stop_gradient=True,
            [1.])
    """
    ...


@add_doc_and_signature
def asinh(
    x: Tensor,
    name: str | None = None,
    *,
    out: Tensor | None = None,
) -> Tensor:
    r"""
    Asinh Activation Operator.

    .. math::
       out = asinh(x)

    Args:
        x (Tensor): Input of Asinh operator, an N-D Tensor, with data type float32, float64, float16, bfloat16,
            uint8, int8, int16, int32, int64, complex64 or complex128. Alias: ``input``.
        name (str|None, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Keyword Args:
        out (Tensor|optional): The output tensor. Default: None.

    Returns:
        Tensor. Output of Asinh operator, a Tensor with shape same as input
            (integer types are autocasted into float32).

    Examples:
        .. code-block:: pycon

            >>> import paddle

            >>> x = paddle.to_tensor([-0.4, -0.2, 0.1, 0.3])
            >>> out = paddle.asinh(x)
            >>> print(out)
            Tensor(shape=[4], dtype=float32, place=Place(cpu), stop_gradient=True,
            [-0.39003533, -0.19869010,  0.09983408,  0.29567307])
    """
    ...


@add_doc_and_signature
def reciprocal(
    x: Tensor,
    name: str | None = None,
    *,
    out: Tensor | None = None,
) -> Tensor:
    r"""
    Reciprocal Activation Operator.

    .. math::
        out = \\frac{1}{x}

    Args:
        x (Tensor): Input of Reciprocal operator, an N-D Tensor, with data type float32, float64, float16, bfloat16,
            uint8, int8, int16, int32, int64. Alias: ``input``.
        name (str|None, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Keyword Args:
        out (Tensor|optional): The output tensor. Default: None.

    Returns:
        Tensor. Output of Reciprocal operator, a Tensor with shape same as input
            (integer types are autocasted into float32).

    Examples:
        .. code-block:: pycon

            >>> import paddle

            >>> x = paddle.to_tensor([-0.4, -0.2, 0.1, 0.3])
            >>> out = paddle.reciprocal(x)
            >>> print(out)
            Tensor(shape=[4], dtype=float32, place=Place(cpu), stop_gradient=True,
            [-2.50000000, -5.        ,  10.       ,  3.33333325])
    """
    ...


@add_doc_and_signature
def square(
    x: Tensor,
    name: str | None = None,
    *,
    out: Tensor | None = None,
) -> Tensor:
    r"""
    Square each elements of the inputs.

    .. math::
       out = x^2

    Args:
        x (Tensor): Input of Square operator, an N-D Tensor, with data type int32, int64, float32, float64, float16, complex64 or complex128. Alias: ``input``.
        name (str|None, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Keyword Args:
        out (Tensor|optional): The output tensor. Default: None.

    Returns:
        Tensor. Output of Square operator, a Tensor with shape same as input.

    Examples:
        .. code-block:: pycon

            >>> import paddle

            >>> x = paddle.to_tensor([-0.4, -0.2, 0.1, 0.3])
            >>> out = paddle.square(x)
            >>> print(out)
            Tensor(shape=[4], dtype=float32, place=Place(cpu), stop_gradient=True,
            [0.16000001, 0.04000000, 0.01000000, 0.09000000])
    """
    ...


@add_doc_and_signature
def tan(
    x: Tensor,
    name: str | None = None,
    *,
    out: Tensor | None = None,
) -> Tensor:
    r"""
    Tangent Operator. Computes tangent of x element-wise.

    Input range is `(k*pi-pi/2, k*pi+pi/2)` and output range is `(-inf, inf)`.

    .. math::
       out = tan(x)

    Args:
        x (Tensor): Input of Tan operator, an N-D Tensor, with data type float32, float64, float16,
            bfloat16, uint8, int8, int16, int32, int64, complex64 or complex128. Alias: ``input``.
        name (str|None, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Keyword Args:
        out (Tensor|optional): The output tensor. Default: None.

    Returns:
        Tensor. Output of Tan operator, a Tensor with shape same as input
            (integer types are autocasted into float32).

    Examples:
        .. code-block:: pycon

            >>> import paddle

            >>> x = paddle.to_tensor([-0.4, -0.2, 0.1, 0.3])
            >>> out = paddle.tan(x)
            >>> print(out)
            Tensor(shape=[4], dtype=float32, place=Place(cpu), stop_gradient=True,
            [-0.42279324, -0.20271003,  0.10033467,  0.30933627])
    """
    ...


@add_doc_and_signature
def log1p(
    x: Tensor,
    name: str | None = None,
    *,
    out: Tensor | None = None,
) -> Tensor:
    r"""
    Calculates the natural log of the given input tensor plus 1, element-wise.

    .. math::

        Out = \ln(x+1)

    Args:
        x (Tensor): Input Tensor. Must be one of the following types: int32, int64, float16, bfloat16, float32, float64, complex64, complex128.
        name (str|None, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.
        out (Tensor, optional): The output Tensor. If set, the result will be stored in this Tensor. Default: None.

    Returns:
        Tensor: The natural log of the input Tensor plus 1 computed element-wise.

    Examples:

        .. code-block:: pycon

            >>> import paddle

            >>> data = paddle.to_tensor([[0], [1]], dtype='float32')
            >>> res = paddle.log1p(data)
            >>> res
            Tensor(shape=[2, 1], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[0.        ],
             [0.69314718]])
    """
    ...


@add_doc_and_signature
def matmul(
    x: Tensor,
    y: Tensor,
    transpose_x: bool = False,
    transpose_y: bool = False,
    name: str | None = None,
    *,
    out: Tensor | None = None,
) -> Tensor:
    r"""
    Applies matrix multiplication to two tensors. `matmul` follows
    the complete broadcast rules,
    and its behavior is consistent with `np.matmul`.

    Currently, the input tensors' number of dimensions can be any, `matmul` can be used to
    achieve the `dot`, `matmul` and `batchmatmul`.

    The actual behavior depends on the shapes of :math:`x`, :math:`y` and the
    flag values of :attr:`transpose_x`, :attr:`transpose_y`. Specifically:

    - If a transpose flag is specified, the last two dimensions of the tensor
      are transposed. If the tensor is ndim-1 of shape, the transpose is invalid. If the tensor
      is ndim-1 of shape :math:`[D]`, then for :math:`x` it is treated as :math:`[1, D]`, whereas
      for :math:`y` it is the opposite: It is treated as :math:`[D, 1]`.

    The multiplication behavior depends on the dimensions of `x` and `y`. Specifically:

    - If both tensors are 1-dimensional, the dot product result is obtained.

    - If both tensors are 2-dimensional, the matrix-matrix product is obtained.

    - If the `x` is 1-dimensional and the `y` is 2-dimensional,
      a `1` is prepended to its dimension in order to conduct the matrix multiply.
      After the matrix multiply, the prepended dimension is removed.

    - If the `x` is 2-dimensional and `y` is 1-dimensional,
      the matrix-vector product is obtained.

    - If both arguments are at least 1-dimensional and at least one argument
      is N-dimensional (where N > 2), then a batched matrix multiply is obtained.
      If the first argument is 1-dimensional, a 1 is prepended to its dimension
      in order to conduct the batched matrix multiply and removed after.
      If the second argument is 1-dimensional, a 1 is appended to its
      dimension for the purpose of the batched matrix multiple and removed after.
      The non-matrix (exclude the last two dimensions) dimensions are
      broadcasted according the broadcast rule.
      For example, if input is a (j, 1, n, m) tensor and the other is a (k, m, p) tensor,
      out will be a (j, k, n, p) tensor.

    Args:
        x (Tensor): The input tensor which is a Tensor.
        y (Tensor): The input tensor which is a Tensor.
        transpose_x (bool, optional): Whether to transpose :math:`x` before multiplication. Default is False.
        transpose_y (bool, optional): Whether to transpose :math:`y` before multiplication. Default is False.
        name (str|None, optional): If set None, the layer will be named automatically. For more information, please refer to :ref:`api_guide_Name`. Default is None.
        out (Tensor, optional): The output tensor. If set, the result will be stored in this tensor. Default is None.

    Returns:
        Tensor: The output Tensor.

    Examples:

        .. code-block:: pycon

            >>> import paddle

            >>> # vector * vector
            >>> x = paddle.rand([10])
            >>> y = paddle.rand([10])
            >>> z = paddle.matmul(x, y)
            >>> print(z.shape)
            paddle.Size([])

            >>> # matrix * vector
            >>> x = paddle.rand([10, 5])
            >>> y = paddle.rand([5])
            >>> z = paddle.matmul(x, y)
            >>> print(z.shape)
            paddle.Size([10])

            >>> # batched matrix * broadcasted vector
            >>> x = paddle.rand([10, 5, 2])
            >>> y = paddle.rand([2])
            >>> z = paddle.matmul(x, y)
            >>> print(z.shape)
            paddle.Size([10, 5])

            >>> # batched matrix * batched matrix
            >>> x = paddle.rand([10, 5, 2])
            >>> y = paddle.rand([10, 2, 5])
            >>> z = paddle.matmul(x, y)
            >>> print(z.shape)
            paddle.Size([10, 5, 5])

            >>> # batched matrix * broadcasted matrix
            >>> x = paddle.rand([10, 1, 5, 2])
            >>> y = paddle.rand([1, 3, 2, 5])
            >>> z = paddle.matmul(x, y)
            >>> print(z.shape)
            paddle.Size([10, 3, 5, 5])
    """
    ...


@add_doc_and_signature
def multiply(
    x: Tensor, y: Tensor, name: str | None = None, *, out: Tensor | None = None
) -> Tensor:
    r"""
    multiply two tensors element-wise. The equation is:

    .. math::
        out = x * y

    Note:
        Supported shape of :attr:`x` and :attr:`y` for this operator:
        1. `x.shape` == `y.shape`.
        2. `x.shape` could be the continuous subsequence of `y.shape`.
        ``paddle.multiply`` supports broadcasting. If you would like to know more about broadcasting, please refer to `Introduction to Tensor`_ .

        .. _Introduction to Tensor: ../../guides/beginner/tensor_en.html#chapter5-broadcasting-of-tensor

    Args:
        x (Tensor): the input tensor, its data type should be one of bfloat16, float16, float32, float64, int32, int64, bool, complex64, complex128.
        y (Tensor): the input tensor, its data type should be one of bfloat16, float16, float32, float64, int32, int64, bool, complex64, complex128.
        name (str|None, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        N-D Tensor. A location into which the result is stored. If :attr:`x`, :attr:`y` have different shapes and are "broadcastable", the resulting tensor shape is the shape of :attr:`x` and :attr:`y` after broadcasting. If :attr:`x`, :attr:`y` have the same shape, its shape is the same as :attr:`x` and :attr:`y`.

    Examples:

        .. code-block:: pycon

            >>> import paddle

            >>> x = paddle.to_tensor([[1, 2], [3, 4]])
            >>> y = paddle.to_tensor([[5, 6], [7, 8]])
            >>> res = paddle.multiply(x, y)
            >>> print(res)
            Tensor(shape=[2, 2], dtype=int64, place=Place(cpu), stop_gradient=True,
            [[5 , 12],
             [21, 32]])
            >>> x = paddle.to_tensor([[[1, 2, 3], [1, 2, 3]]])
            >>> y = paddle.to_tensor([2])
            >>> res = paddle.multiply(x, y)
            >>> print(res)
            Tensor(shape=[1, 2, 3], dtype=int64, place=Place(cpu), stop_gradient=True,
            [[[2, 4, 6],
              [2, 4, 6]]])

    """
    ...


@add_doc_and_signature
def logsumexp(
    x: Tensor,
    axis: int | Sequence[int] | None = None,
    keepdim: bool = False,
    name: str | None = None,
    *,
    out: Tensor | None = None,
) -> Tensor:
    r"""
    Calculates the log of the sum of exponential of ``x`` along ``axis`` .

    .. math::
       logsumexp(x) = \log\sum exp(x)

    Args:
        x (Tensor): The input Tensor with data type bfloat16, float16, float32,
            float64, uint8, int8, int16, int32, int64, which have no more than
            4 dimensions.
        axis (int|list|tuple|None, optional): The axis along which to perform
            logsumexp calculations. ``axis`` should be int, list(int) or
            tuple(int). If ``axis`` is a list/tuple of dimension(s), logsumexp
            is calculated along all element(s) of ``axis`` . ``axis`` or
            element(s) of ``axis`` should be in range [-D, D), where D is the
            dimensions of ``x`` . If ``axis`` or element(s) of ``axis`` is
            less than 0, it works the same way as :math:`axis + D` . If
            ``axis`` is None, logsumexp is calculated along all elements of
            ``x``. Default is None.
        keepdim (bool, optional): Whether to reserve the reduced dimension(s)
            in the output Tensor. If ``keep_dim`` is True, the dimensions of
            the output Tensor is the same as ``x`` except in the reduced
            dimensions(it is of size 1 in this case). Otherwise, the shape of
            the output Tensor is squeezed in ``axis`` . Default is False.
        name (str|None, optional): Name for the operation (optional, default is None).
            For more information, please refer to :ref:`api_guide_Name`.
    Keyword Args:
        out (Tensor|optional): The output tensor.
    Returns:
        Tensor, results of logsumexp along ``axis`` of ``x``, with the same data
        type as ``x`` (integer types are autocasted into float32).

    Examples:

        .. code-block:: pycon

            >>> import paddle

            >>> x = paddle.to_tensor([[-1.5, 0.0, 2.0], [3.0, 1.2, -2.4]])
            >>> out1 = paddle.logsumexp(x)
            >>> out1
            Tensor(shape=[], dtype=float32, place=Place(cpu), stop_gradient=True,
            3.46912265)
            >>> out2 = paddle.logsumexp(x, 1)
            >>> out2
            Tensor(shape=[2], dtype=float32, place=Place(cpu), stop_gradient=True,
            [2.15317822, 3.15684605])

    """
    ...


@add_doc_and_signature
def softplus(
    x: Tensor,
    beta: float = 1,
    threshold: float = 20,
    name: str | None = None,
) -> Tensor:
    r"""
    softplus activation

    .. math::
        softplus(x)=\begin{cases}
                \frac{1}{\beta} * \\log(1 + e^{\beta * x}),&x\\leqslant\frac{\varepsilon}{\beta};\\
                x,&x>\frac{\varepsilon}{\beta}.
            \\end{cases}

    Parameters:
        x (Tensor): The input Tensor with data type float32, float64, complex64, complex128.
        beta (float, optional): The value of :math:`\beta` for softplus. Default is 1
        threshold (float, optional): The value of :math:`\varepsilon` for softplus. Default is 20
        name (str|None, optional): For details, please refer to :ref:`api_guide_Name`. Generally, no setting is required. Default: None.

    Returns:
        A Tensor with the same data type and shape as ``x`` .

    Examples:
        .. code-block:: pycon

            >>> import paddle
            >>> import paddle.nn.functional as F

            >>> x = paddle.to_tensor([-0.4, -0.2, 0.1, 0.3], dtype='float32')
            >>> out = F.softplus(x)
            >>> print(out)
            Tensor(shape=[4], dtype=float32, place=Place(cpu), stop_gradient=True,
            [0.51301527, 0.59813893, 0.74439669, 0.85435522])
    """
    ...


@add_doc_and_signature
def i0(
    x: Tensor,
    name: str | None = None,
    *,
    out: Tensor | None = None,
) -> Tensor:
    r"""
    The function used to calculate modified bessel function of order 0.

    Equation:
        ..  math::

            I_0(x) = \\sum^{\\infty}_{k=0}\frac{(x^2/4)^k}{(k!)^2}

    Args:
        x (Tensor): The input tensor, it's data type should be float32, float64,
            uint8, int8, int16, int32, int64.
        name (str|None, optional): For details, please refer to :ref:`api_guide_Name`. Generally, no setting is required. Default: None.

    Returns:
        - out (Tensor), A Tensor. the value of the modified bessel function of order 0 at x
            (integer types are autocasted into float32).

    Examples:
        .. code-block:: pycon

            >>> import paddle

            >>> x = paddle.to_tensor([0, 1, 2, 3, 4], dtype="float32")
            >>> paddle.i0(x)
            Tensor(shape=[5], dtype=float32, place=Place(cpu), stop_gradient=True,
            [0.99999994 , 1.26606596 , 2.27958512 , 4.88079262 , 11.30192089])
    """
    ...


@add_doc_and_signature
def i0e(
    x: Tensor,
    name: str | None = None,
    *,
    out: Tensor | None = None,
) -> Tensor:
    r"""
    The function used to calculate exponentially scaled modified Bessel function of order 0.

    Equation:
        ..  math::

            I_0(x) = \\sum^{\\infty}_{k=0}\frac{(x^2/4)^k}{(k!)^2} \\
            I_{0e}(x) = e^{-|x|}I_0(x)

    Args:
        x (Tensor): The input tensor, it's data type should be float32, float64,
            uint8, int8, int16, int32, int64.
        name (str|None, optional): For details, please refer to :ref:`api_guide_Name`. Generally, no setting is required. Default: None.

    Returns:
        - out (Tensor), A Tensor. the value of the exponentially scaled modified Bessel function of order 0 at x
            (integer types are autocasted into float32).

    Examples:
        .. code-block:: pycon

            >>> import paddle

            >>> x = paddle.to_tensor([0, 1, 2, 3, 4], dtype="float32")
            >>> print(paddle.i0e(x))
            Tensor(shape=[5], dtype=float32, place=Place(cpu), stop_gradient=True,
            [0.99999994, 0.46575963, 0.30850831, 0.24300036, 0.20700191])
    """
    ...


@add_doc_and_signature
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
        \\left| x - y \right| \\leq atol + rtol \times \\left| y \right|

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
        .. code-block:: pycon

            >>> import paddle
            >>> x = paddle.to_tensor([10000.0, 1e-07])
            >>> y = paddle.to_tensor([10000.1, 1e-08])
            >>> result1 = paddle.isclose(x, y, rtol=1e-05, atol=1e-08, equal_nan=False, name="ignore_nan")
            >>> print(result1)
            Tensor(shape=[2], dtype=bool, place=Place(cpu), stop_gradient=True,
            [True , False])
            >>> result2 = paddle.isclose(x, y, rtol=1e-05, atol=1e-08, equal_nan=True, name="equal_nan")
            >>> print(result2)
            Tensor(shape=[2], dtype=bool, place=Place(cpu), stop_gradient=True,
            [True , False])
            >>> x = paddle.to_tensor([1.0, float('nan')])
            >>> y = paddle.to_tensor([1.0, float('nan')])
            >>> result1 = paddle.isclose(x, y, rtol=1e-05, atol=1e-08, equal_nan=False, name="ignore_nan")
            >>> print(result1)
            Tensor(shape=[2], dtype=bool, place=Place(cpu), stop_gradient=True,
            [True , False])
            >>> result2 = paddle.isclose(x, y, rtol=1e-05, atol=1e-08, equal_nan=True, name="equal_nan")
            >>> print(result2)
            Tensor(shape=[2], dtype=bool, place=Place(cpu), stop_gradient=True,
            [True, True])
    """
    ...


# zhengsheng
@add_doc_and_signature
def isfinite(
    x: Tensor,
    name: str | None = None,
) -> Tensor:
    r"""
    Return whether every element of input tensor is finite number or not.

    Args:
        x (Tensor): The input tensor, it's data type should be float16, float32, float64, int32, int64, complex64, complex128.
            alias: ``input``.
        name (str|None, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        `Tensor`, the bool result which shows every element of `x` whether it is finite number or not.

    Examples:
        .. code-block:: pycon

            >>> import paddle
            >>> x = paddle.to_tensor([float('-inf'), -2, 3.6, float('inf'), 0, float('-nan'), float('nan')])
            >>> out = paddle.isfinite(x)
            >>> out
            Tensor(shape=[7], dtype=bool, place=Place(cpu), stop_gradient=True,
            [False, True , True , False, True , False, False])
    """
    ...


@add_doc_and_signature
def isinf(
    x: Tensor,
    name: str | None = None,
) -> Tensor:
    r"""
    Return whether every element of input tensor is `+/-INF` or not.

    Args:
        x (Tensor): The input tensor, it's data type should be float16, float32, float64, uint8, int8, int16, int32, int64, complex64, complex128.
            alias: ``input``.
        name (str|None, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        `Tensor`, the bool result which shows every element of `x` whether it is `+/-INF` or not.

    Examples:
        .. code-block:: pycon

            >>> import paddle
            >>> x = paddle.to_tensor([float('-inf'), -2, 3.6, float('inf'), 0, float('-nan'), float('nan')])
            >>> out = paddle.isinf(x)
            >>> out
            Tensor(shape=[7], dtype=bool, place=Place(cpu), stop_gradient=True,
            [True , False, False, True , False, False, False])
    """
    ...


@add_doc_and_signature
def isnan(
    x: Tensor,
    name: str | None = None,
) -> Tensor:
    r"""
    Return whether every element of input tensor is `NaN` or not.

    Args:
        x (Tensor): The input tensor, it's data type should be float16, float32, float64, int32, int64, complex64, complex128.
            alias: ``input``.
        name (str|None, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        `Tensor`, the bool result which shows every element of `x` whether it is `NaN` or not.

    Examples:
        .. code-block:: pycon

            >>> import paddle

            >>> x = paddle.to_tensor([float('-inf'), -2, 3.6, float('inf'), 0, float('-nan'), float('nan')])
            >>> out = paddle.isnan(x)
            >>> out
            Tensor(shape=[7], dtype=bool, place=Place(cpu), stop_gradient=True,
            [False, False, False, False, False, True , True ])
    """
    ...


@add_doc_and_signature
def roll(
    x: Tensor,
    shifts: int | Sequence[int],
    axis: int | Sequence[int] | None = None,
    name: str | None = None,
) -> Tensor:
    r"""
    Roll the `x` tensor along the given axis(axes). With specific 'shifts', Elements that
    roll beyond the last position are re-introduced at the first according to 'shifts'.
    If a axis is not specified,
    the tensor will be flattened before rolling and then restored to the original shape.

    Args:
        x (Tensor): The x tensor as input. alias: ``input``.
        shifts (int|list|tuple): The number of places by which the elements
                           of the `x` tensor are shifted.
        axis (int|list|tuple, optional): axis(axes) along which to roll. Default: None
            alias: ``dim``.
        name(str|None, optional): The default value is None.  Normally there is no need for user to set this property.
                For more information, please refer to :ref:`api_guide_Name` .
    The image below shows a 2D tensor `[[1,2,3],[4,5,6],[7,8,9]]` being transformed into tensors with
    different shapes through the roll operation.
    .. image:: https://githubraw.cdn.bcebos.com/PaddlePaddle/docs/develop/docs/images/api_legend/roll.png
        :width: 700
        :align: center
        :alt: legend of roll API

    Returns:
        Tensor, A Tensor with same data type as `x`.

    Examples:
        .. code-block:: pycon

            >>> import paddle
            >>> x = paddle.to_tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
            >>> out_z1 = paddle.roll(x, shifts=1)
            >>> print(out_z1.numpy())
            [[9. 1. 2.]
             [3. 4. 5.]
             [6. 7. 8.]]
            >>> out_z2 = paddle.roll(x, shifts=1, axis=0)
            >>> print(out_z2.numpy())
            [[7. 8. 9.]
             [1. 2. 3.]
             [4. 5. 6.]]
            >>> out_z3 = paddle.roll(x, shifts=1, axis=1)
            >>> print(out_z3.numpy())
            [[3. 1. 2.]
             [6. 4. 5.]
             [9. 7. 8.]]
    """
    ...


@add_doc_and_signature
def ceil(
    x: Tensor, name: str | None = None, *, out: Tensor | None = None
) -> Tensor:
    r"""
    Ceil Operator. Computes ceil of x element-wise.

    .. math::
        out = \\left \\lceil x \\right \\rceil

    Args:
        x (Tensor): Input of Ceil operator, an N-D Tensor, with data type float32, float64, float16, bfloat16,
            uint8, int8, int16, int32, int64. Alias: ``input``.
        name (str|None, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Keyword Args:
        out (Tensor|optional): The output tensor. Default: None.

    Returns:
        Tensor. Output of Ceil operator, a Tensor with shape same as input
            (integer types are autocasted into float32).

    Examples:
        .. code-block:: pycon

            >>> import paddle

            >>> x = paddle.to_tensor([-0.4, -0.2, 0.1, 0.3])
            >>> out = paddle.ceil(x)
            >>> print(out)
            Tensor(shape=[4], dtype=float32, place=Place(cpu), stop_gradient=True,
            [-0., -0., 1. , 1. ])
    """
    ...


@add_doc_and_signature
def sum(
    x: Tensor,
    axis: int | Sequence[int] | None = None,
    dtype: DTypeLike | None = None,
    keepdim: bool = False,
    name: str | None = None,
    *,
    out: Tensor | None = None,
) -> Tensor:
    r"""
    Computes the sum of tensor elements over the given dimension.

    This API has two signatures:

    1. ``paddle.sum(x, axis=None, dtype=None, keepdim=False, name=None, *, out=None)`` (Paddle-style)
    2. ``paddle.sum(input, dim=None, keepdim=False, dtype=None, *, out=None)`` (PyTorch-style)

    Args:
        x (Tensor): An N-D Tensor, the data type is bool, bfloat16, float16, float32, float64,
            uint8, int8, int16, int32, int64, complex64, complex128.
            alias: ``input``.
        axis (int|list|tuple|None, optional): The dimensions along which the sum is performed. If
            :attr:`None`, sum all elements of :attr:`x` and return a
            Tensor with a single element, otherwise must be in the
            range :math:`[-rank(x), rank(x))`. If :math:`axis[i] < 0`,
            the dimension to reduce is :math:`rank + axis[i]`.
            alias: ``dim``.
        dtype (str|paddle.dtype|np.dtype, optional): The dtype of output Tensor. The default value is None, the dtype
            of output is the same as input Tensor `x`.
        keepdim (bool, optional): Whether to reserve the reduced dimension in the
            output Tensor. The result Tensor will have one fewer dimension
            than the :attr:`x` unless :attr:`keepdim` is true, default
            value is False.
        name (str|None, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.
        out (Tensor|None, optional): The output tensor. Default: None.

    Returns:
        Tensor: Results of summation operation on the specified axis of input Tensor `x`,
        if `x.dtype='bool'`, `x.dtype='int32'`, it's data type is `'int64'`,
        otherwise it's data type is the same as `x`.

    Examples:
        .. code-block:: pycon

            >>> import paddle

            >>> # x is a Tensor with following elements:
            >>> #    [[0.2, 0.3, 0.5, 0.9]
            >>> #     [0.1, 0.2, 0.6, 0.7]]
            >>> # Each example is followed by the corresponding output tensor.
            >>> x = paddle.to_tensor([[0.2, 0.3, 0.5, 0.9], [0.1, 0.2, 0.6, 0.7]])
            >>> out1 = paddle.sum(x)
            >>> out1
            Tensor(shape=[], dtype=float32, place=Place(cpu), stop_gradient=True,
            3.50000000)
            >>> out2 = paddle.sum(x, axis=0)
            >>> out2
            Tensor(shape=[4], dtype=float32, place=Place(cpu), stop_gradient=True,
            [0.30000001, 0.50000000, 1.10000002, 1.59999990])
            >>> out3 = paddle.sum(x, axis=-1)
            >>> out3
            Tensor(shape=[2], dtype=float32, place=Place(cpu), stop_gradient=True,
            [1.89999998, 1.60000002])
            >>> out4 = paddle.sum(x, axis=1, keepdim=True)
            >>> out4
            Tensor(shape=[2, 1], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[1.89999998],
             [1.60000002]])

            >>> # y is a Tensor with shape [2, 2, 2] and elements as below:
            >>> #      [[[1, 2], [3, 4]],
            >>> #      [[5, 6], [7, 8]]]
            >>> # Each example is followed by the corresponding output tensor.
            >>> y = paddle.to_tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
            >>> out5 = paddle.sum(y, axis=[1, 2])
            >>> out5
            Tensor(shape=[2], dtype=int64, place=Place(cpu), stop_gradient=True,
            [10, 26])
            >>> out6 = paddle.sum(y, axis=[0, 1])
            >>> out6
            Tensor(shape=[2], dtype=int64, place=Place(cpu), stop_gradient=True,
            [16, 20])

            >>> # x is a Tensor with following elements:
            >>> #    [[True, True, True, True]
            >>> #     [False, False, False, False]]
            >>> # Each example is followed by the corresponding output tensor.
            >>> x = paddle.to_tensor([[True, True, True, True], [False, False, False, False]])
            >>> out7 = paddle.sum(x)
            >>> out7
            Tensor(shape=[], dtype=int64, place=Place(cpu), stop_gradient=True,
            4)
            >>> out8 = paddle.sum(x, axis=0)
            >>> out8
            Tensor(shape=[4], dtype=int64, place=Place(cpu), stop_gradient=True,
            [1, 1, 1, 1])
            >>> out9 = paddle.sum(x, axis=1)
            >>> out9
            Tensor(shape=[2], dtype=int64, place=Place(cpu), stop_gradient=True,
            [4, 0])
    """
    ...


@add_doc_and_signature
def index_put(
    x: Tensor,
    indices: Sequence[Tensor],
    value: Tensor,
    accumulate: bool = False,
    name: str | None = None,
) -> Tensor:
    r"""
    Puts values from the tensor values into the tensor x using the indices specified in indices (which is a tuple of Tensors).
    The expression paddle.index_put_(x, indices, values) is equivalent to tensor[indices] = values. Returns x.
    If accumulate is True, the elements in values are added to x. If accumulate is False, the behavior is undefined if indices contain duplicate elements.

    Args:
        x (Tensor) : The Source Tensor. Supported data types are int32, int64, float16, float32, float64, bool.
        indices (list[Tensor]|tuple[Tensor]): The tuple of Tensor containing the indices to index.
            The data type of ``tensor in indices`` must be int32, int64 or bool.
        value (Tensor): The tensor used to be assigned to x.
        accumulate (bool, optional): Whether the elements in values are added to x. Default: False.
        name(str|None, optional): For details, please refer to :ref:`api_guide_Name`. Generally, no setting is required. Default: None.

    Returns:
        Tensor, same dimension and dtype with x.

    Examples:
        .. code-block:: pycon

            >>> import paddle

            >>> x = paddle.zeros([3, 3])
            >>> value = paddle.ones([3])
            >>> ix1 = paddle.to_tensor([0, 1, 2])
            >>> ix2 = paddle.to_tensor([1, 2, 1])
            >>> indices = (ix1, ix2)

            >>> out = paddle.index_put(x, indices, value)
            >>> print(x)
            Tensor(shape=[3, 3], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[0., 0., 0.],
             [0., 0., 0.],
             [0., 0., 0.]])
            >>> print(out)
            Tensor(shape=[3, 3], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[0., 1., 0.],
             [0., 0., 1.],
             [0., 1., 0.]])
    """
    ...


@add_doc_and_signature
def index_put_(
    x: Tensor,
    indices: Sequence[Tensor],
    value: Tensor,
    accumulate: bool = False,
    name: str | None = None,
) -> Tensor:
    r"""
    Inplace version of ``index_put`` API, the output Tensor will be inplaced with input ``x``.
    Please refer to :ref:`api_paddle_index_put`.
    """
    ...


# liuyi
@add_doc_and_signature
def any(
    x: Tensor,
    axis: int | Sequence[int] | None = None,
    keepdim: bool = False,
    name: str | None = None,
    *,
    out: Tensor | None = None,
) -> Tensor:
    r"""
    Computes the ``logical or`` of tensor elements over the given dimension, and return the result.

    Args:
        x (Tensor): An N-D Tensor, the input data type should be 'bool', 'float32', 'float64', 'int32', 'int64', 'complex64', 'complex128'.
            alias: ``input``.
        axis (int|list|tuple|None, optional): The dimensions along which the ``logical or`` is compute. If
            :attr:`None`, and all elements of :attr:`x` and return a
            Tensor with a single element, otherwise must be in the
            range :math:`[-rank(x), rank(x))`. If :math:`axis[i] < 0`,
            the dimension to reduce is :math:`rank + axis[i]`.
            alias: ``dim``.
        keepdim (bool, optional): Whether to reserve the reduced dimension in the
            output Tensor. The result Tensor will have one fewer dimension
            than the :attr:`x` unless :attr:`keepdim` is true, default
            value is False.
        name (str|None, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.
        out (Tensor|None, optional): The output tensor. Default: None.

    Returns:
        Tensor: Results the ``logical or`` on the specified axis of input Tensor `x`,  it's data type is bool.

    Examples:
        .. code-block:: pycon

            >>> import paddle

            >>> x = paddle.to_tensor([[1, 0], [1, 1]], dtype='int32')
            >>> x = paddle.assign(x)
            >>> x
            Tensor(shape=[2, 2], dtype=int32, place=Place(cpu), stop_gradient=True,
            [[1, 0],
             [1, 1]])
            >>> x = paddle.cast(x, 'bool')
            >>> # x is a bool Tensor with following elements:
            >>> #    [[True, False]
            >>> #     [True, True]]

            >>> # out1 should be True
            >>> out1 = paddle.any(x)
            >>> out1
            Tensor(shape=[], dtype=bool, place=Place(cpu), stop_gradient=True,
            True)

            >>> # out2 should be [True, True]
            >>> out2 = paddle.any(x, axis=0)
            >>> out2
            Tensor(shape=[2], dtype=bool, place=Place(cpu), stop_gradient=True,
            [True, True])

            >>> # keepdim=False, out3 should be [True, True], out.shape should be (2,)
            >>> out3 = paddle.any(x, axis=-1)
            >>> out3
            Tensor(shape=[2], dtype=bool, place=Place(cpu), stop_gradient=True,
            [True, True])

            >>> # keepdim=True, result should be [[True], [True]], out.shape should be (2,1)
            >>> out4 = paddle.any(x, axis=1, keepdim=True)
            >>> out4
            Tensor(shape=[2, 1], dtype=bool, place=Place(cpu), stop_gradient=True,
            [[True],
             [True]])

    """
    ...


@add_doc_and_signature
def expand_as(
    x: Tensor,
    y: Tensor,
    name: str | None = None,
) -> Tensor:
    r"""

    Expand the input tensor ``x`` to the same shape as the input tensor ``y``.

    Both the number of dimensions of ``x`` and ``y`` must be less than or equal to 6, and the number of dimensions of ``y`` must be greater than or equal to that of ``x``. The dimension to expand must have a value of 0.

    The following diagram illustrates how a one-dimensional tensor is transformed into a tensor with a shape of [2,3] through the expand_as operation. The target tensor has a shape of [2,3], and through expand_as, the one-dimensional tensor is expanded into a tensor with a shape of [2,3].

    .. image:: https://raw.githubusercontent.com/PaddlePaddle/docs/develop/docs/images/api_legend/expand_as.png
        :width: 800
        :alt: expand_as API
        :align: center

    Args:
        x (Tensor): The input tensor, its data type is bool, float32, float64, int32 or int64.
        y (Tensor): The input tensor that gives the shape to expand to.
        name (str|None, optional): The default value is None. Normally there is no need for user to set this property. For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        N-D Tensor, A Tensor with the same shape as ``y``. The data type is the same as ``x``.

    Examples:
        .. code-block:: pycon

            >>> import paddle

            >>> data_x = paddle.to_tensor([1, 2, 3], 'int32')
            >>> data_y = paddle.to_tensor([[1, 2, 3], [4, 5, 6]], 'int32')
            >>> out = paddle.expand_as(data_x, data_y)
            >>> print(out)
            Tensor(shape=[2, 3], dtype=int32, place=Place(cpu), stop_gradient=True,
            [[1, 2, 3],
             [1, 2, 3]])
    """
    ...


# shenwei
@add_doc_and_signature
def grid_sample(
    x: Tensor,
    grid: Tensor,
    mode: str = 'bilinear',
    padding_mode: Literal["zeros", "reflection", "border"] = 'zeros',
    align_corners: bool = True,
    name: str | None = None,
) -> Tensor:
    r"""
    Sample input X by using bilinear interpolation or
    nearest interpolation based on flow field grid, which is usually
    generated by :code:`affine_grid` . When the input X is 4-D Tensor,
    the grid of shape [N, H, W, 2] is the concatenation of (x, y)
    coordinates with shape [N, H, W] each, where x is indexing the 4th
    dimension (in width dimension) of input data x and y is indexing
    the 3rd dimension (in height dimension), finally results is the
    bilinear interpolation or nearest value of 4 nearest corner
    points. The output tensor shape will be [N, C, H, W]. When the input X
    is 5-D Tensor, the grid of shape [N, D, H, W, 3] is the concatenation
    of (x, y, z) coordinates with shape [N, D, H, W] each, where x is
    indexing the 5th dimension (in width dimension) of input data x, y is
    indexing the 4th dimension (in height dimension) and z is indexing the
    3rd dimension (in depth dimension) finally results is the bilinear
    interpolation or nearest value of 8 nearest corner points. The output
    tensor shape will be [N, C, D, H, W].


    Step 1:

    Get (x, y) grid coordinates and scale to [0, H-1/W-1].

    .. code-block:: text

        grid_x = 0.5 * (grid[:, :, :, 0] + 1) * (W - 1)
        grid_y = 0.5 * (grid[:, :, :, 1] + 1) * (H - 1)

    Step 2:

    Indices input data X with grid (x, y) in each [H, W] area, and bilinear
    interpolate point value by 4 nearest points or nearest interpolate point value
    by nearest point.

    .. code-block:: text

        wn ------- y_n ------- en
        |           |           |
        |          d_n          |
        |           |           |
        x_w --d_w-- grid--d_e-- x_e
        |           |           |
        |          d_s          |
        |           |           |
        ws ------- y_s ------- wn

        For bilinear interpolation:
        x_w = floor(x)              // west side x coord
        x_e = x_w + 1               // east side x coord
        y_n = floor(y)              // north side y coord
        y_s = y_s + 1               // south side y coord
        d_w = grid_x - x_w          // distance to west side
        d_e = x_e - grid_x          // distance to east side
        d_n = grid_y - y_n          // distance to north side
        d_s = y_s - grid_y          // distance to south side
        wn = X[:, :, y_n, x_w]      // north-west point value
        en = X[:, :, y_n, x_e]      // north-east point value
        ws = X[:, :, y_s, x_w]      // south-east point value
        es = X[:, :, y_s, x_w]      // north-east point value

        output = wn * d_e * d_s + en * d_w * d_s
                + ws * d_e * d_n + es * d_w * d_n

    Args:
        x(Tensor): The input tensor, which is a 4-D tensor with shape
                     [N, C, H, W] or a 5-D tensor with shape [N, C, D, H, W],
                     N is the batch size, C is the channel number,
                     D, H and W is the feature depth, height and width.
                     The data type is float32 or float64.
                    alias: ``input``.
        grid(Tensor): Input grid tensor, which is a 4-D tensor with shape [N, grid_H,
                        grid_W, 2] or a 5-D tensor with shape [N, grid_D, grid_H,
                        grid_W, 3]. The data type is float32 or float64.
        mode(str, optional): The interpolation method which can be 'bilinear' or 'nearest'.
                         Default: 'bilinear'.
        padding_mode(str, optional) The padding method used when source index
                   is out of input images. It can be 'zeros', 'reflection' and 'border'.
                   Default: zeros.
        align_corners(bool, optional): If `align_corners` is true, it will projects
                   -1 and 1 to the centers of the corner pixels. Otherwise, it will
                   projects -1 and 1 to the image edges.
        name(str|None, optional): For detailed information, please refer
                             to :ref:`api_guide_Name`. Usually name is no need to set and
                             None by default.

    Returns:

        Tensor, The shape of output is [N, C, grid_H, grid_W] or [N, C, grid_D, grid_H, grid_W] in which `grid_D` is the depth of grid,
                `grid_H` is the height of grid and `grid_W` is the width of grid. The data type is same as input tensor.

    Examples:

        .. code-block:: pycon

            >>> import paddle
            >>> import paddle.nn.functional as F

            >>> # x shape=[1, 1, 3, 3]
            >>> x = paddle.to_tensor([[[[-0.6, 0.8, -0.5], [-0.5, 0.2, 1.2], [1.4, 0.3, -0.2]]]], dtype='float64')
            >>> # grid.shape = [1, 3, 4, 2]
            >>> grid = paddle.to_tensor(
            ...     [
            ...         [
            ...             [[0.2, 0.3], [-0.4, -0.3], [-0.9, 0.3], [-0.9, -0.6]],
            ...             [[0.4, 0.1], [0.9, -0.8], [0.4, 0.5], [0.5, -0.2]],
            ...             [[0.1, -0.8], [-0.3, -1.0], [0.7, 0.4], [0.2, 0.8]],
            ...         ]
            ...     ],
            ...     dtype='float64',
            ... )
            >>> y_t = F.grid_sample(x, grid, mode='bilinear', padding_mode='border', align_corners=True)
            >>> print(y_t)
            Tensor(shape=[1, 1, 3, 4], dtype=float64, place=Place(cpu), stop_gradient=True,
            [[[[ 0.34000000,  0.01600000,  0.08600000, -0.44800000],
               [ 0.55000000, -0.07600000,  0.35000000,  0.59000000],
               [ 0.59600000,  0.38000000,  0.52000000,  0.24000000]]]])
    """
    ...


@add_doc_and_signature
def pixel_shuffle(
    x: Tensor,
    upscale_factor: int,
    data_format: DataLayout2D = 'NCHW',
    name: str | None = None,
) -> Tensor:
    r"""
    This API implements pixel shuffle operation.
    See more details in :ref:`PixelShuffle <api_paddle_nn_PixelShuffle>` .

    Parameters:
        x (Tensor): 4-D tensor, the data type should be float32 or float64.
            alias: ``input``.
        upscale_factor (int): factor to increase spatial resolution.
        data_format (str, optional): The data format of the input and output data. An optional string from: ``"NCHW"``, ``"NHWC"``. When it is ``"NCHW"``, the data is stored in the order of: [batch_size, input_channels, input_height, input_width]. Default: ``"NCHW"``.
        name (str|None, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Out (Tensor): Reshaped tensor according to the new dimension.

    Examples:
        .. code-block:: pycon

            >>> import paddle
            >>> import paddle.nn.functional as F

            >>> x = paddle.randn(shape=[2, 9, 4, 4])
            >>> out_var = F.pixel_shuffle(x, 3)
            >>> print(out_var.shape)
            paddle.Size([2, 1, 12, 12])
    """
    ...


@add_doc_and_signature
def gelu(
    x: Tensor,
    approximate: Literal["tanh", "none"] | bool = False,
    name: str | None = None,
) -> Tensor:
    r"""
    gelu activation.

    The activation function of Gelu is calculated element by element. More information refers to :ref: `Gaussian Error Linear Units`.

    The approximate parameter must be True, False, "tanh", or "none".

    If approximate is True or "tanh":

    .. math::

        gelu(x) = 0.5 * x * (1 + tanh(\\sqrt{\\frac{2}{\\pi}} * (x + 0.044715x^{3})))

    else:

    .. math::

        gelu(x) = 0.5 * x * (1 + erf(\\frac{x}{\\sqrt{2}}))

    Parameters:
        x (Tensor): The input Tensor with data type float32, float64.
            alias: ``input``.
        approximate (str|bool, optional): Whether to enable approximation. Default is False.
        name (str|None, optional): For details, please refer to :ref:`api_guide_Name`. Generally, no setting is required. Default: None.

    Returns:
        A Tensor with the same data type and shape as ``x`` .

    Examples:
        .. code-block:: pycon

            >>> import paddle
            >>> import paddle.nn.functional as F

            >>> x = paddle.to_tensor([[-1, 0.5], [1, 1.5]])
            >>> out1 = F.gelu(x)
            >>> print(out1)
            Tensor(shape=[2, 2], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[-0.15865529,  0.34573123],
             [ 0.84134471,  1.39978933]])
            >>> out2 = F.gelu(x, True)
            >>> print(out2)
            Tensor(shape=[2, 2], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[-0.15880796,  0.34571400],
             [ 0.84119201,  1.39957154]])
            >>> out3 = F.gelu(x, "none")
            >>> print(out3)
            Tensor(shape=[2, 2], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[-0.15865529,  0.34573123],
             [ 0.84134471,  1.39978933]])
            >>> out4 = F.gelu(x, "tanh")
            >>> print(out4)
            Tensor(shape=[2, 2], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[-0.15880796,  0.34571400],
             [ 0.84119201,  1.39957154]])
    """
    ...


@add_doc_and_signature
def sigmoid(
    x: Tensor,
    name: str | None = None,
    *,
    out: Tensor | None = None,
) -> Tensor:
    r"""
    Sigmoid Activation.

    .. math::
       out = \\frac{1}{1 + e^{-x}}

    Args:
        x (Tensor): Input of Sigmoid operator, an N-D Tensor, with data type bfloat16, float16, float32, float64,
            uint8, int8, int16, int32, int64, complex64 or complex128. Alias: ``input``.
        name (str|None, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Keyword Args:
        out (Tensor|optional): The output tensor. Default: None.

    Returns:
        Tensor. Output of Sigmoid operator, a Tensor with shape same as input
            (integer types are autocasted into float32).

    Examples:
        .. code-block:: pycon

            >>> import paddle
            >>> import paddle.nn.functional as F

            >>> x = paddle.to_tensor([-0.4, -0.2, 0.1, 0.3])
            >>> out = F.sigmoid(x)
            >>> print(out)
            Tensor(shape=[4], dtype=float32, place=Place(cpu), stop_gradient=True,
            [0.40131235, 0.45016602, 0.52497917, 0.57444251])
    """
    ...


# zhouxin
@add_doc_and_signature
def greater_than(
    x: Tensor,
    y: Tensor,
    name: str | None = None,
    *,
    out: Tensor | None = None,
) -> Tensor:
    r"""
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
            >>> result1 = paddle.greater_than(x, y)
            >>> print(result1)
            Tensor(shape=[3], dtype=bool, place=Place(cpu), stop_gradient=True,
            [False, False, True ])
    """
    ...


@add_doc_and_signature
def sin(
    x: Tensor,
    name: str | None = None,
    *,
    out: Tensor | None = None,
) -> Tensor:
    r"""
    Sine Activation Operator.

    .. math::
       out = sin(x)

    Args:
        x (Tensor): Input of Sin operator, an N-D Tensor, with data type float32, float64, float16, bfloat16,
            uint8, int8, int16, int32, int64, complex64 or complex128. Alias: ``input``.
        name (str|None, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Keyword Args:
        out (Tensor|optional): The output tensor. Default: None.

    Returns:
        Tensor. Output of Sin operator, a Tensor with shape same as input
            (integer types are autocasted into float32).

    Examples:
        .. code-block:: pycon

            >>> import paddle

            >>> x = paddle.to_tensor([-0.4, -0.2, 0.1, 0.3])
            >>> out = paddle.sin(x)
            >>> print(out)
            Tensor(shape=[4], dtype=float32, place=Place(cpu), stop_gradient=True,
            [-0.38941833, -0.19866933,  0.09983342,  0.29552022])
    """
    ...


@add_doc_and_signature
def sign(
    x: Tensor,
    name: str | None = None,
    *,
    out: Tensor | None = None,
) -> Tensor:
    r"""
    Returns sign of every element in `x`: For real numbers, 1 for positive, -1 for negative and 0 for zero. For complex numbers, the return value is a complex number with unit magnitude. If a complex number element is zero, the result is 0+0j.

    Args:
        x (Tensor): The input tensor. The data type can be uint8, int8, int16, int32, int64, bfloat16, float16, float32, float64, complex64 or complex128. Alias: ``input``.
        name (str|None, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.
        out (Tensor|None, optional): The output tensor. If set, the result will be stored in this tensor. Default is None.

    Returns:
        Tensor: The output sign tensor with identical shape and data type to the input :attr:`x`.

    Examples:
        .. code-block:: pycon

            >>> import paddle

            >>> x = paddle.to_tensor([3.0, 0.0, -2.0, 1.7], dtype='float32')
            >>> out = paddle.sign(x=x)
            >>> out
            Tensor(shape=[4], dtype=float32, place=Place(cpu), stop_gradient=True,
            [ 1.,  0., -1.,  1.])
    """
    ...


@add_doc_and_signature
def lgamma(
    x: Tensor,
    name: str | None = None,
    *,
    out: Tensor | None = None,
) -> Tensor:
    r"""
    Calculates the lgamma of the given input tensor, element-wise.

    This operator performs elementwise lgamma for input $X$.
    :math:`out = log\Gamma(x)`

    Args:
        x (Tensor): Input Tensor. Must be one of the following types: bfloat16, float16, float32, float64,
            uint8, int8, int16, int32, int64. Alias: ``input``.
        name (str|None, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Keyword args:
        out(Tensor, optional): The output tensor.

    Returns:
        Tensor, the lgamma of the input Tensor, the shape and data type is the same with input
            (integer types are autocasted into float32).

    Examples:
        .. code-block:: pycon

            >>> import paddle

            >>> x = paddle.to_tensor([-0.4, -0.2, 0.1, 0.3])
            >>> out = paddle.lgamma(x)
            >>> out
            Tensor(shape=[4], dtype=float32, place=Place(cpu), stop_gradient=True,
            [1.31452465, 1.76149750, 2.25271273, 1.09579802])
    """
    ...


@add_doc_and_signature
def log(
    x: Tensor,
    name: str | None = None,
    *,
    out: Tensor | None = None,
) -> Tensor:
    r"""
    Calculates the natural log of the given input Tensor, element-wise.

    .. math::

        Out = \ln(x)

    Args:
        x (Tensor): Input Tensor. Must be one of the following types: int32, int64, float16, bfloat16, float32, float64, complex64, complex128. Alias: ``input``.
        name (str|None): The default value is None. Normally there is no need for user to set this property. For more information, please refer to :ref:`api_guide_Name`
        out (Tensor, optional): The output Tensor. If set, the result will be stored in this tensor. Default is None.


    Returns:
        Tensor: The natural log of the input Tensor computed element-wise.

    Examples:

        .. code-block:: pycon

            >>> import paddle

            >>> x = [[2, 3, 4], [7, 8, 9]]
            >>> x = paddle.to_tensor(x, dtype='float32')
            >>> print(paddle.log(x))
            Tensor(shape=[2, 3], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[0.69314718, 1.09861231, 1.38629436],
             [1.94591010, 2.07944155, 2.19722462]])
    """
    ...


@add_doc_and_signature
def rsqrt(
    x: Tensor,
    name: str | None = None,
    *,
    out: Tensor | None = None,
) -> Tensor:
    r"""
    Rsqrt Activation Operator.

    Please make sure input is legal in case of numeric errors.

    .. math::
       out = \\frac{1}{\\sqrt{x}}

    Args:
        x (Tensor): Input of Rsqrt operator, an N-D Tensor, with data type float32, float64, float16, bfloat16,
            uint8, int8, int16, int32, int64. Alias: ``input``.
        name (str|None, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Keyword Args:
        out (Tensor|optional): The output tensor. Default: None.

    Returns:
        Tensor. Output of Rsqrt operator, a Tensor with shape same as input
            (integer types are autocasted into float32).

    Examples:
        .. code-block:: pycon

            >>> import paddle

            >>> x = paddle.to_tensor([0.1, 0.2, 0.3, 0.4])
            >>> out = paddle.rsqrt(x)
            >>> print(out)
            Tensor(shape=[4], dtype=float32, place=Place(cpu), stop_gradient=True,
            [3.16227770, 2.23606801, 1.82574177, 1.58113885])
    """
    ...


@add_doc_and_signature
def cos(
    x: Tensor,
    name: str | None = None,
    *,
    out: Tensor | None = None,
) -> Tensor:
    r"""
    Cosine Operator. Computes cosine of x element-wise.

    Input range is `(-inf, inf)` and output range is `[-1,1]`.

    .. math::
       out = cos(x)

    Args:
        x (Tensor): Input of Cos operator, an N-D Tensor, with data type float32, float64, float16, bfloat16,
            uint8, int8, int16, int32, int64, complex64, complex128. Alias: ``input``.
        name (str|None, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Keyword Args:
        out (Tensor|optional): The output tensor. Default: None.

    Returns:
        Tensor. Output of Cos operator, a Tensor with shape same as input
            (integer types are autocasted into float32).

    Examples:
        .. code-block:: pycon

            >>> import paddle

            >>> x = paddle.to_tensor([-0.4, -0.2, 0.1, 0.3])
            >>> out = paddle.cos(x)
            >>> print(out)
            Tensor(shape=[4], dtype=float32, place=Place(cpu), stop_gradient=True,
            [0.92106098, 0.98006660, 0.99500418, 0.95533651])
    """
    ...


@add_doc_and_signature
def cosh(
    x: Tensor,
    name: str | None = None,
    *,
    out: Tensor | None = None,
) -> Tensor:
    r"""
    Cosh Activation Operator.

    Input range `(-inf, inf)`, output range `(1, inf)`.

    .. math::
       out = \\frac{exp(x)+exp(-x)}{2}

    Args:
        x (Tensor): Input of Cosh operator, an N-D Tensor, with data type float32, float64, float16, bfloat16,
            uint8, int8, int16, int32, int64, complex64 or complex128. Alias: ``input``.
        name (str|None, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Keyword Args:
        out (Tensor|optional): The output tensor. Default: None.

    Returns:
        Tensor. Output of Cosh operator, a Tensor with shape same as input
            (integer types are autocasted into float32).

    Examples:
        .. code-block:: pycon

            >>> import paddle

            >>> x = paddle.to_tensor([-0.4, -0.2, 0.1, 0.3])
            >>> out = paddle.cosh(x)
            >>> print(out)
            Tensor(shape=[4], dtype=float32, place=Place(cpu), stop_gradient=True,
            [1.08107233, 1.02006674, 1.00500417, 1.04533851])
    """
    ...


@add_doc_and_signature
def floor(
    x: Tensor,
    name: str | None = None,
    *,
    out: Tensor | None = None,
) -> Tensor:
    r"""
    Floor Activation Operator. Computes floor of x element-wise.

    .. math::
        out = \\lfloor x \\rfloor

    Args:
        x (Tensor): Input of Floor operator, an N-D Tensor, with data type float32, float64, float16, bfloat16,
            uint8, int8, int16, int32, int64. Alias: ``input``.
        name (str|None, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Keyword Args:
        out (Tensor|optional): The output tensor. Default: None.

    Returns:
        Tensor. Output of Floor operator, a Tensor with shape same as input
            (integer types are autocasted into float32).

    Examples:
        .. code-block:: pycon

            >>> import paddle

            >>> x = paddle.to_tensor([-0.4, -0.2, 0.1, 0.3])
            >>> out = paddle.floor(x)
            >>> print(out)
            Tensor(shape=[4], dtype=float32, place=Place(cpu), stop_gradient=True,
            [-1., -1.,  0.,  0.])
    """
    ...


# hehongyu
@add_doc_and_signature
def maximum(
    x: Tensor,
    y: Tensor,
    name: str | None = None,
    *,
    out: Tensor | None = None,
) -> Tensor:
    r"""
    Compare two tensors and returns a new tensor containing the element-wise maxima. The equation is:

    .. math::
        out = max(x, y)

    Note:
        ``paddle.maximum`` supports broadcasting. If you want know more about broadcasting, please refer to  `Introduction to Tensor`_ .

        .. _Introduction to Tensor: ../../guides/beginner/tensor_en.html#chapter5-broadcasting-of-tensor

    Args:
        x (Tensor): the input tensor, it's data type should be bfloat16, float16, float32, float64, int32, int64.
        y (Tensor): the input tensor, it's data type should be bfloat16, float16, float32, float64, int32, int64.
        name (str|None, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.
        out(Tensor, optional): The output tensor.

    Returns:
        N-D Tensor. A location into which the result is stored. If x, y have different shapes and are "broadcastable", the resulting tensor shape is the shape of x and y after broadcasting. If x, y have the same shape,  its shape is the same as x and y.

    Examples:

        .. code-block:: pycon

            >>> import paddle

            >>> x = paddle.to_tensor([[1, 2], [7, 8]])
            >>> y = paddle.to_tensor([[3, 4], [5, 6]])
            >>> res = paddle.maximum(x, y)
            >>> print(res)
            Tensor(shape=[2, 2], dtype=int64, place=Place(cpu), stop_gradient=True,
            [[3, 4],
             [7, 8]])

            >>> x = paddle.to_tensor([[1, 2, 3], [1, 2, 3]])
            >>> y = paddle.to_tensor([3, 0, 4])
            >>> res = paddle.maximum(x, y)
            >>> print(res)
            Tensor(shape=[2, 3], dtype=int64, place=Place(cpu), stop_gradient=True,
            [[3, 2, 4],
             [3, 2, 4]])

            >>> x = paddle.to_tensor([2, 3, 5], dtype='float32')
            >>> y = paddle.to_tensor([1, float("nan"), float("nan")], dtype='float32')
            >>> res = paddle.maximum(x, y)
            >>> print(res)
            Tensor(shape=[3], dtype=float32, place=Place(cpu), stop_gradient=True,
            [2. , nan, nan])

            >>> x = paddle.to_tensor([5, 3, float("inf")], dtype='float32')
            >>> y = paddle.to_tensor([1, -float("inf"), 5], dtype='float32')
            >>> res = paddle.maximum(x, y)
            >>> print(res)
            Tensor(shape=[3], dtype=float32, place=Place(cpu), stop_gradient=True,
            [5.  , 3.  , inf.])
    """
    ...


@add_doc_and_signature
def minimum(
    x: Tensor,
    y: Tensor,
    name: str | None = None,
    *,
    out: Tensor | None = None,
) -> Tensor:
    r"""
    Compare two tensors and return a new tensor containing the element-wise minima. The equation is:

    .. math::
        out = min(x, y)

    Note:
        ``paddle.minimum`` supports broadcasting. If you want know more about broadcasting, please refer to `Introduction to Tensor`_ .

        .. _Introduction to Tensor: ../../guides/beginner/tensor_en.html#chapter5-broadcasting-of-tensor

    Args:
        x (Tensor): the input tensor, it's data type should be bfloat16, float16, float32, float64, int32, int64.
        y (Tensor): the input tensor, it's data type should be bfloat16, float16, float32, float64, int32, int64.
        name (str|None, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.
        out(Tensor, optional): The output tensor.

    Returns:
        Tensor. If x, y have different shapes and are "broadcastable", the resulting tensor shape is the shape of x and y after broadcasting. If x, y have the same shape,  its shape is the same as x and y.

    Examples:

        .. code-block:: pycon

            >>> import paddle

            >>> x = paddle.to_tensor([[1, 2], [7, 8]])
            >>> y = paddle.to_tensor([[3, 4], [5, 6]])
            >>> res = paddle.minimum(x, y)
            >>> print(res)
            Tensor(shape=[2, 2], dtype=int64, place=Place(cpu), stop_gradient=True,
            [[1, 2],
             [5, 6]])

            >>> x = paddle.to_tensor([[[1, 2, 3], [1, 2, 3]]])
            >>> y = paddle.to_tensor([3, 0, 4])
            >>> res = paddle.minimum(x, y)
            >>> print(res)
            Tensor(shape=[1, 2, 3], dtype=int64, place=Place(cpu), stop_gradient=True,
            [[[1, 0, 3],
              [1, 0, 3]]])

            >>> x = paddle.to_tensor([2, 3, 5], dtype='float32')
            >>> y = paddle.to_tensor([1, float("nan"), float("nan")], dtype='float32')
            >>> res = paddle.minimum(x, y)
            >>> print(res)
            Tensor(shape=[3], dtype=float32, place=Place(cpu), stop_gradient=True,
            [1. , nan, nan])

            >>> x = paddle.to_tensor([5, 3, float("inf")], dtype='float64')
            >>> y = paddle.to_tensor([1, -float("inf"), 5], dtype='float64')
            >>> res = paddle.minimum(x, y)
            >>> print(res)
            Tensor(shape=[3], dtype=float64, place=Place(cpu), stop_gradient=True,
            [ 1.  , -inf.,  5.  ])
    """
    ...


@add_doc_and_signature
def sqrt(
    x: Tensor,
    name: str | None = None,
    *,
    out: Tensor | None = None,
) -> Tensor:
    r"""
    Sqrt Activation Operator.

    .. math::
       out=\\sqrt{x}=x^{1/2}

    Args:
        x (Tensor): Input of Sqrt operator, an N-D Tensor, with data type float32, float64, float16, bfloat16
            uint8, int8, int16, int32, int64. Alias: ``input``.
        name (str|None, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Keyword Args:
        out (Tensor|optional): The output tensor. Default: None.

    Returns:
        Tensor. Output of Sqrt operator, a Tensor with shape same as input
            (integer types are autocasted into float32).

    Examples:
        .. code-block:: pycon

            >>> import paddle

            >>> x = paddle.to_tensor([0.1, 0.2, 0.3, 0.4])
            >>> out = paddle.sqrt(x)
            >>> print(out)
            Tensor(shape=[4], dtype=float32, place=Place(cpu), stop_gradient=True,
            [0.31622776, 0.44721359, 0.54772258, 0.63245553])
    """
    ...


# lousiyu


# zhengshijie
@add_doc_and_signature
def tril(
    x: Tensor,
    diagonal: int = 0,
    name: str | None = None,
    *,
    out: Tensor | None = None,
) -> Tensor:
    r"""
    Returns the lower triangular part of a matrix (2-D tensor) or batch
    of matrices :attr:`x`, the other elements of the result tensor are set
    to 0. The lower triangular part of the matrix is defined as the elements
    on and below the diagonal.

    Args:
        x (Tensor): The input x which is a Tensor.
            Support data types: ``bool``, ``float64``, ``float32``, ``int32``, ``int64``, ``complex64``, ``complex128``.
        diagonal (int, optional): The diagonal to consider, default value is 0.
            If :attr:`diagonal` = 0, all elements on and below the main diagonal are
            retained. A positive value includes just as many diagonals above the main
            diagonal, and similarly a negative value excludes just as many diagonals below
            the main diagonal. The main diagonal are the set of indices
            :math:`\{(i, i)\}` for :math:`i \in [0, \min\{d_{1}, d_{2}\} - 1]` where
            :math:`d_{1}, d_{2}` are the dimensions of the matrix.
        name(str|None, optional): For details, please refer to :ref:`api_guide_Name`. Generally, no setting is required. Default: None.
        out(Tensor, optional): The output tensor.

    Returns:
        Tensor: Results of lower triangular operation by the specified diagonal of input tensor x,
        it's data type is the same as x's Tensor.

    Examples:
        .. code-block:: pycon

            >>> import paddle

            >>> data = paddle.arange(1, 13, dtype="int64").reshape([3, -1])
            >>> print(data)
            Tensor(shape=[3, 4], dtype=int64, place=Place(cpu), stop_gradient=True,
            [[1 , 2 , 3 , 4 ],
             [5 , 6 , 7 , 8 ],
             [9 , 10, 11, 12]])

            >>> tril1 = paddle.tril(data)
            >>> print(tril1)
            Tensor(shape=[3, 4], dtype=int64, place=Place(cpu), stop_gradient=True,
            [[1 , 0 , 0 , 0 ],
             [5 , 6 , 0 , 0 ],
             [9 , 10, 11, 0 ]])

            >>> # example 2, positive diagonal value
            >>> tril2 = paddle.tril(data, diagonal=2)
            >>> print(tril2)
            Tensor(shape=[3, 4], dtype=int64, place=Place(cpu), stop_gradient=True,
            [[1 , 2 , 3 , 0 ],
             [5 , 6 , 7 , 8 ],
             [9 , 10, 11, 12]])

            >>> # example 3, negative diagonal value
            >>> tril3 = paddle.tril(data, diagonal=-1)
            >>> print(tril3)
            Tensor(shape=[3, 4], dtype=int64, place=Place(cpu), stop_gradient=True,
            [[0 , 0 , 0 , 0 ],
             [5 , 0 , 0 , 0 ],
             [9 , 10, 0 , 0 ]])
    """
    ...


@add_doc_and_signature
def triu(
    x: Tensor,
    diagonal: int = 0,
    name: str | None = None,
    *,
    out: Tensor | None = None,
) -> Tensor:
    r"""
    Return the upper triangular part of a matrix (2-D tensor) or batch of matrices
    :attr:`x`, the other elements of the result tensor are set to 0.
    The upper triangular part of the matrix is defined as the elements on and
    above the diagonal.

    Args:
        x (Tensor): The input x which is a Tensor.
            Support data types: ``float64``, ``float32``, ``int32``, ``int64``, ``complex64``, ``complex128``.
        diagonal (int, optional): The diagonal to consider, default value is 0.
            If :attr:`diagonal` = 0, all elements on and above the main diagonal are
            retained. A positive value excludes just as many diagonals above the main
            diagonal, and similarly a negative value includes just as many diagonals below
            the main diagonal. The main diagonal are the set of indices
            :math:`\{(i, i)\}` for :math:`i \in [0, \min\{d_{1}, d_{2}\} - 1]` where
            :math:`d_{1}, d_{2}` are the dimensions of the matrix.
        name(str|None, optional): For details, please refer to :ref:`api_guide_Name`. Generally, no setting is required. Default: None.
        out(Tensor, optional): The output tensor.

    Returns:
        Tensor: Results of upper triangular operation by the specified diagonal of input tensor x,
        it's data type is the same as x's Tensor.

    Examples:
        .. code-block:: pycon

            >>> import paddle

            >>> x = paddle.arange(1, 13, dtype="int64").reshape([3, -1])
            >>> print(x)
            Tensor(shape=[3, 4], dtype=int64, place=Place(cpu), stop_gradient=True,
            [[1 , 2 , 3 , 4 ],
             [5 , 6 , 7 , 8 ],
             [9 , 10, 11, 12]])

            >>> # example 1, default diagonal
            >>> triu1 = paddle.tensor.triu(x)
            >>> print(triu1)
            Tensor(shape=[3, 4], dtype=int64, place=Place(cpu), stop_gradient=True,
            [[1 , 2 , 3 , 4 ],
             [0 , 6 , 7 , 8 ],
             [0 , 0 , 11, 12]])

            >>> # example 2, positive diagonal value
            >>> triu2 = paddle.tensor.triu(x, diagonal=2)
            >>> print(triu2)
            Tensor(shape=[3, 4], dtype=int64, place=Place(cpu), stop_gradient=True,
            [[0, 0, 3, 4],
             [0, 0, 0, 8],
             [0, 0, 0, 0]])

            >>> # example 3, negative diagonal value
            >>> triu3 = paddle.tensor.triu(x, diagonal=-1)
            >>> print(triu3)
            Tensor(shape=[3, 4], dtype=int64, place=Place(cpu), stop_gradient=True,
            [[1 , 2 , 3 , 4 ],
             [5 , 6 , 7 , 8 ],
             [0 , 10, 11, 12]])
    """
    ...


@add_doc_and_signature
def bmm(
    x: Tensor,
    y: Tensor,
    name: str | None = None,
    *,
    out: Tensor | None = None,
) -> Tensor:
    r"""
    Applies batched matrix multiplication to two tensors.

    Both of the two input tensors must be three-dimensional and share the same batch size.

    If x is a (b, m, k) tensor, y is a (b, k, n) tensor, the output will be a (b, m, n) tensor.

    Args:
        x (Tensor): The input Tensor.
        y (Tensor): The input Tensor.
        name (str|None): A name for this layer(optional). If set None, the layer
            will be named automatically. Default: None.
        out(Tensor, optional): The output tensor.

    Returns:
        Tensor: The product Tensor.

    Examples:
        .. code-block:: pycon

            >>> import paddle

            >>> # In imperative mode:
            >>> # size x: (2, 2, 3) and y: (2, 3, 2)
            >>> x = paddle.to_tensor([[[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]], [[3.0, 3.0, 3.0], [4.0, 4.0, 4.0]]])
            >>> y = paddle.to_tensor([[[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]], [[4.0, 4.0], [5.0, 5.0], [6.0, 6.0]]])
            >>> out = paddle.bmm(x, y)
            >>> print(out)
            Tensor(shape=[2, 2, 2], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[[6. , 6. ],
              [12., 12.]],
             [[45., 45.],
              [60., 60.]]])
    """
    ...


# lihaoyang
@add_doc_and_signature
def logical_and(
    x: Tensor,
    y: Tensor,
    name: str | None = None,
    *,
    out: Tensor | None = None,
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
        x (Tensor): the input tensor, it's data type should be one of bool, int8, int16, int32, int64, bfloat16, float16, float32, float64, complex64, complex128.
            Alias: ``input``.
        y (Tensor): the input tensor, it's data type should be one of bool, int8, int16, int32, int64, bfloat16, float16, float32, float64, complex64, complex128.
            Alias: ``other``.
        out(Tensor|None, optional): The ``Tensor`` that specifies the output of the operator, which can be any ``Tensor`` that has been created in the program. The default value is None, and a new ``Tensor`` will be created to save the output.
        name (str|None, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        N-D Tensor. A location into which the result is stored. It's dimension equals with ``x``.

    Examples:
        .. code-block:: pycon

            >>> import paddle

            >>> x = paddle.to_tensor([True])
            >>> y = paddle.to_tensor([True, False, True, False])
            >>> res = paddle.logical_and(x, y)
            >>> print(res)
            Tensor(shape=[4], dtype=bool, place=Place(cpu), stop_gradient=True,
            [True , False, True , False])
    """
    ...


@add_doc_and_signature
def logical_or(
    x: Tensor,
    y: Tensor,
    name: str | None = None,
    *,
    out: Tensor | None = None,
) -> Tensor:
    r"""
    ``logical_or`` operator computes element-wise logical OR on ``x`` and ``y``, and returns ``out``. ``out`` is N-dim boolean ``Tensor``.
    Each element of ``out`` is calculated by

    .. math::

        out = x || y

    Note:
        ``paddle.logical_or`` supports broadcasting. If you want know more about broadcasting, please refer to `Introduction to Tensor`_ .

        .. _Introduction to Tensor: ../../guides/beginner/tensor_en.html#chapter5-broadcasting-of-tensor

    Args:
        x (Tensor): the input tensor, it's data type should be one of bool, int8, int16, int32, int64, bfloat16, float16, float32, float64, complex64, complex128.
            Alias: ``input``.
        y (Tensor): the input tensor, it's data type should be one of bool, int8, int16, int32, int64, bfloat16, float16, float32, float64, complex64, complex128.
            Alias: ``other``.
        out(Tensor|None, optional): The ``Variable`` that specifies the output of the operator, which can be any ``Tensor`` that has been created in the program. The default value is None, and a new ``Tensor`` will be created to save the output.
        name (str|None, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        N-D Tensor. A location into which the result is stored. It's dimension equals with ``x``.

    Examples:
        .. code-block:: pycon

            >>> import paddle

            >>> x = paddle.to_tensor([True, False], dtype="bool").reshape([2, 1])
            >>> y = paddle.to_tensor([True, False, True, False], dtype="bool").reshape([2, 2])
            >>> res = paddle.logical_or(x, y)
            >>> print(res)
            Tensor(shape=[2, 2], dtype=bool, place=Place(cpu), stop_gradient=True,
            [[True , True ],
             [True , False]])
    """
    ...


@add_doc_and_signature
def logical_not(
    x: Tensor,
    name: str | None = None,
    *,
    out: Tensor | None = None,
) -> Tensor:
    r"""
    ``logical_not`` operator computes element-wise logical NOT on ``x``, and returns ``out``. ``out`` is N-dim boolean ``Variable``.
    Each element of ``out`` is calculated by

    .. math::

        out = !x

    Note:
        ``paddle.logical_not`` supports broadcasting. If you want know more about broadcasting, please refer to `Introduction to Tensor`_ .

        .. _Introduction to Tensor: ../../guides/beginner/tensor_en.html#chapter5-broadcasting-of-tensor

    Args:
        x(Tensor):  Operand of logical_not operator. Must be a Tensor of type bool, int8, int16, int32, int64, bfloat16, float16, float32, or float64, complex64, complex128.
            Alias: ``input``.
        out(Tensor|None): The ``Tensor`` that specifies the output of the operator, which can be any ``Tensor`` that has been created in the program. The default value is None, and a new ``Tensor` will be created to save the output.
        name(str|None, optional): The default value is None. Normally there is no need for users to set this property. For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        N-D Tensor. A location into which the result is stored. It's dimension equals with ``x``.

    Examples:
        .. code-block:: pycon

            >>> import paddle

            >>> x = paddle.to_tensor([True, False, True, False])
            >>> res = paddle.logical_not(x)
            >>> print(res)
            Tensor(shape=[4], dtype=bool, place=Place(cpu), stop_gradient=True,
            [False, True , False, True ])
    """
    ...


@add_doc_and_signature
def logical_xor(
    x: Tensor,
    y: Tensor,
    name: str | None = None,
    *,
    out: Tensor | None = None,
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
            Alias: ``input``.
        y (Tensor): the input tensor, it's data type should be one of bool, int8, int16, int32, int64, bfloat16, float16, float32, float64, complex64, complex128.
            Alias: ``other``.
        out(Tensor|None, optional): The ``Tensor`` that specifies the output of the operator, which can be any ``Tensor`` that has been created in the program. The default value is None, and a new ``Tensor`` will be created to save the output.
        name (str|None, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        N-D Tensor. A location into which the result is stored. It's dimension equals with ``x``.

    Examples:
        .. code-block:: pycon

            >>> import paddle

            >>> x = paddle.to_tensor([True, False], dtype="bool").reshape([2, 1])
            >>> y = paddle.to_tensor([True, False, True, False], dtype="bool").reshape([2, 2])
            >>> res = paddle.logical_xor(x, y)
            >>> print(res)
            Tensor(shape=[2, 2], dtype=bool, place=Place(cpu), stop_gradient=True,
            [[False, True ],
             [True , False]])
    """
    ...


@add_doc_and_signature
def dot(
    x: Tensor,
    y: Tensor,
    name: str | None = None,
    *,
    out: Tensor | None = None,
) -> Tensor:
    r"""
    This operator calculates inner product for vectors.

    Note:
       Support 1-d and 2-d Tensor. When it is 2d, the first dimension of this matrix
       is the batch dimension, which means that the vectors of multiple batches are dotted.

    Parameters:
        x (Tensor): 1-D or 2-D ``Tensor``. Its dtype should be ``float32``, ``float64``, ``int32``, ``int64``, ``complex64``, ``complex128``
            Alias: ``input``.
        y (Tensor): 1-D or 2-D ``Tensor``. Its dtype should be ``float32``, ``float64``, ``int32``, ``int64``, ``complex64``, ``complex128``
            Alias: ``other``.
        name (str|None, optional): Name of the output. Default is None. It's used to print debug info for developers. Details: :ref:`api_guide_Name`

    Keyword args:
        out (Tensor|None, optional): The output tensor.

    Returns:
        Tensor: the calculated result Tensor.

    Examples:

        .. code-block:: pycon

            >>> import paddle

            >>> # 1-D Tensor * 1-D Tensor
            >>> x = paddle.to_tensor([1, 2, 3])
            >>> y = paddle.to_tensor([4, 5, 6])
            >>> z = paddle.dot(x, y)
            >>> print(z)
            Tensor(shape=[], dtype=int64, place=Place(cpu), stop_gradient=True,
            32)

            >>> # 2-D Tensor * 2-D Tensor
            >>> x = paddle.to_tensor([[1, 2, 3], [2, 4, 6]])
            >>> y = paddle.to_tensor([[4, 5, 6], [4, 5, 6]])
            >>> z = paddle.dot(x, y)
            >>> print(z)
            Tensor(shape=[2], dtype=int64, place=Place(cpu), stop_gradient=True,
            [32, 64])
    """
    ...


@add_doc_and_signature
def tanh(
    x: Tensor,
    name: str | None = None,
    *,
    out: Tensor | None = None,
) -> Tensor:
    r"""

    Tanh Activation Operator.

    .. math::
        out = \frac{e^{x} - e^{-x}}{e^{x} + e^{-x}}

    Args:
        x (Tensor): Input of Tanh operator, an N-D Tensor, with data type bfloat16, float32, float64,
            float16, uint8, int8, int16, int32, int64. Alias: ``input``.
        name (str|None, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.
        out (Tensor|None, optional): The output tensor. Default: None.

    Returns:
        Output of Tanh operator, a Tensor with same data type and shape as input
            (integer types are autocasted into float32).

    Examples:
        .. code-block:: pycon

            >>> import paddle

            >>> x = paddle.to_tensor([-0.4, -0.2, 0.1, 0.3])
            >>> out = paddle.tanh(x)
            >>> out
            Tensor(shape=[4], dtype=float32, place=Place(cpu), stop_gradient=True,
            [-0.37994900, -0.19737528,  0.09966799,  0.29131261])
    """
    ...


@add_doc_and_signature
def exp(
    x: Tensor,
    name: str | None = None,
    *,
    out: Tensor | None = None,
) -> Tensor:
    r"""

    Computes exp of x element-wise with a natural number `e` as the base.

    .. math::
        out = e^x

    Args:
        x (Tensor): Input of Exp operator, an N-D Tensor, with data type int32, int64, bfloat16, float16, float32, float64, complex64 or complex128.
            Alias: ``input``.
        name (str|None, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Keyword Args:
        out (Tensor|optional): The output tensor. Default: None.

    Returns:
        Tensor. Output of Exp operator, a Tensor with shape same as input.

    Examples:
        .. code-block:: pycon

            >>> import paddle

            >>> x = paddle.to_tensor([-0.4, -0.2, 0.1, 0.3])
            >>> out = paddle.exp(x)
            >>> print(out)
            Tensor(shape=[4], dtype=float32, place=Place(cpu), stop_gradient=True,
            [0.67032003, 0.81873077, 1.10517097, 1.34985888])
    """
    ...


@add_doc_and_signature
def expm1(
    x: Tensor,
    name: str | None = None,
    *,
    out: Tensor | None = None,
) -> Tensor:
    r"""

    Expm1 Operator. Computes expm1 of x element-wise with a natural number :math:`e` as the base.

    .. math::
        out = e^x - 1

    Args:
        x (Tensor): Input of Expm1 operator, an N-D Tensor, with data type int32, int64, bfloat16, float16, float32, float64, complex64 or complex128.
            Alias: ``input``.
        name (str|None, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Keyword Args:
        out (Tensor|optional): The output tensor. Default: None.

    Returns:
        Tensor. Output of Expm1 operator, a Tensor with shape same as input.

    Examples:
        .. code-block:: pycon

            >>> import paddle

            >>> x = paddle.to_tensor([-0.4, -0.2, 0.1, 0.3])
            >>> out = paddle.expm1(x)
            >>> print(out)
            Tensor(shape=[4], dtype=float32, place=Place(cpu), stop_gradient=True,
            [-0.32967997, -0.18126924,  0.10517092,  0.34985882])
    """
    ...


@add_doc_and_signature
def diag(
    x: Tensor,
    offset: int = 0,
    padding_value: int = 0,
    name: str | None = None,
    *,
    out: Tensor | None = None,
) -> Tensor:
    r"""
    If ``x`` is a vector (1-D tensor), a 2-D square tensor with the elements of ``x`` as the diagonal is returned.

    If ``x`` is a matrix (2-D tensor), a 1-D tensor with the diagonal elements of ``x`` is returned.

    The argument ``offset`` controls the diagonal offset:

    If ``offset`` = 0, it is the main diagonal.

    If ``offset`` > 0, it is superdiagonal.

    If ``offset`` < 0, it is subdiagonal.

    Args:
        x (Tensor): The input tensor. Its shape is either 1-D or 2-D. Its data type should be float16, float32, float64, int32, int64, complex64, complex128.
            Alias: ``input``.
        offset (int, optional): The diagonal offset. A positive value represents superdiagonal, 0 represents the main diagonal, and a negative value represents subdiagonal. Default: 0.
            Alias: ``diagonal``.
        padding_value (int|float, optional): Use this value to fill the area outside the specified diagonal band. Only takes effect when the input is a 1-D Tensor. Default: 0.
        name(str|None, optional): For details, please refer to :ref:`api_guide_Name`. Generally, no setting is required. Default: None.
        out (Tensor, optional): The output Tensor. If set, the result will be stored in this Tensor. Default: None.

    Returns:
        Tensor, a square matrix or a vector. The output data type is the same as input data type.

    Examples:
        .. code-block:: pycon
            :name: diag-example-1

            >>> import paddle

            >>> paddle.disable_static()
            >>> x = paddle.to_tensor([1, 2, 3])
            >>> y = paddle.diag(x)
            >>> print(y)
            Tensor(shape=[3, 3], dtype=int64, place=Place(cpu), stop_gradient=True,
            [[1, 0, 0],
             [0, 2, 0],
             [0, 0, 3]])

            >>> y = paddle.diag(x, offset=1)
            >>> print(y)
            Tensor(shape=[4, 4], dtype=int64, place=Place(cpu), stop_gradient=True,
            [[0, 1, 0, 0],
             [0, 0, 2, 0],
             [0, 0, 0, 3],
             [0, 0, 0, 0]])

            >>> y = paddle.diag(x, padding_value=6)
            >>> print(y)
            Tensor(shape=[3, 3], dtype=int64, place=Place(cpu), stop_gradient=True,
            [[1, 6, 6],
             [6, 2, 6],
             [6, 6, 3]])

            >>> y = paddle.diag(input=x, diagonal=2)  # type: ignore[call-arg]
            >>> print(y)
            Tensor(shape=[5, 5], dtype=int64, place=Place(cpu), stop_gradient=True,
            [[0, 0, 1, 0, 0],
             [0, 0, 0, 2, 0],
             [0, 0, 0, 0, 3],
             [0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0]])

            >>> y = paddle.diag(x=x, diagonal=0, padding_value=-1)  # type: ignore[call-arg]
            >>> print(y)
            Tensor(shape=[3, 3], dtype=int64, place=Place(cpu), stop_gradient=True,
            [[ 1, -1, -1],
             [-1,  2, -1],
             [-1, -1,  3]])

        .. code-block:: pycon
            :name: diag-example-2

            >>> import paddle

            >>> paddle.disable_static()
            >>> x = paddle.to_tensor([[1, 2, 3], [4, 5, 6]])
            >>> y = paddle.diag(x)
            >>> print(y)
            Tensor(shape=[2], dtype=int64, place=Place(cpu), stop_gradient=True,
            [1, 5])

            >>> y = paddle.diag(x, offset=1)
            >>> print(y)
            Tensor(shape=[2], dtype=int64, place=Place(cpu), stop_gradient=True,
            [2, 6])

            >>> y = paddle.diag(x, offset=-1)
            >>> print(y)
            Tensor(shape=[1], dtype=int64, place=Place(cpu), stop_gradient=True,
            [4])
    """
    ...


@add_doc_and_signature
def diagonal(
    x: Tensor,
    offset: int = 0,
    axis1: int = 0,
    axis2: int = 1,
    name: str | None = None,
) -> Tensor:
    r"""

    Computes the diagonals of the input tensor x.

    If ``x`` is 2D, returns the diagonal.
    If ``x`` has larger dimensions, diagonals be taken from the 2D planes specified by axis1 and axis2.
    By default, the 2D planes formed by the first and second axis of the input tensor x.

    The argument ``offset`` determines where diagonals are taken from input tensor x:

    - If offset = 0, it is the main diagonal.
    - If offset > 0, it is above the main diagonal.
    - If offset < 0, it is below the main diagonal.

    Args:
        x (Tensor): The input tensor x. Must be at least 2-dimensional. The input data type should be bool, int32,
            int64, bfloat16, float16, float32, float64. Alias: ``input``.
        offset (int, optional): Which diagonals in input tensor x will be taken. Default: 0 (main diagonals).
        axis1 (int, optional): The first axis with respect to take diagonal. Default: 0. Alias: ``dim1``.
        axis2 (int, optional): The second axis with respect to take diagonal. Default: 1. Alias: ``dim2``.
        name (str|None, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor: a partial view of input tensor in specify two dimensions, the output data type is the same as input data type.

    Examples:
        .. code-block:: pycon

            >>> import paddle

            >>> paddle.seed(2023)
            >>> x = paddle.rand([2, 2, 3], 'float32')
            >>> print(x)
            Tensor(shape=[2, 2, 3], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[[0.86583614, 0.52014720, 0.25960937],
              [0.90525323, 0.42400089, 0.40641287]],
             [[0.97020894, 0.74437362, 0.51785129],
              [0.73292869, 0.97786582, 0.04315904]]])

            >>> out1 = paddle.diagonal(x)
            >>> print(out1)
            Tensor(shape=[3, 2], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[0.86583614, 0.73292869],
             [0.52014720, 0.97786582],
             [0.25960937, 0.04315904]])

            >>> out2 = paddle.diagonal(x, offset=0, axis1=2, axis2=1)
            >>> print(out2)
            Tensor(shape=[2, 2], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[0.86583614, 0.42400089],
             [0.97020894, 0.97786582]])

            >>> out3 = paddle.diagonal(x, offset=1, axis1=0, axis2=1)
            >>> print(out3)
            Tensor(shape=[3, 1], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[0.90525323],
             [0.42400089],
             [0.40641287]])

            >>> out4 = paddle.diagonal(x, offset=0, axis1=1, axis2=2)
            >>> print(out4)
            Tensor(shape=[2, 2], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[0.86583614, 0.42400089],
             [0.97020894, 0.97786582]])
    """
    ...


@add_doc_and_signature
def round(
    x: Tensor,
    decimals: int = 0,
    name: str | None = None,
    *,
    out: Tensor | None = None,
) -> Tensor:
    r"""

    Round the values in the input to the nearest integer value.

    .. code-block:: text

        input:
          x.shape = [4]
          x.data = [1.2, -0.9, 3.4, 0.9]

        output:
          out.shape = [4]
          out.data = [1., -1., 3., 1.]

    Args:
        x (Tensor): Input of Round operator, an N-D Tensor, with data type bfloat16, int32, int64, float32, float64, float16, complex64 or complex128.
            Alias: ``input``.
        decimals(int): Rounded decimal place (default: 0).
        name (str|None, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.
        out (Tensor|None, optional): The output tensor. Default: None.

    Returns:
        Tensor. Output of Round operator, a Tensor with shape same as input.

    Examples:
        .. code-block:: pycon

            >>> import paddle

            >>> x = paddle.to_tensor([-0.5, -0.2, 0.6, 1.5])
            >>> out = paddle.round(x)
            >>> print(out)
            Tensor(shape=[4], dtype=float32, place=Place(cpu), stop_gradient=True,
            [-0., -0.,  1.,  2.])
    """
    ...


@add_doc_and_signature
def round_(
    x: Tensor,
    decimals: int = 0,
    name: str | None = None,
) -> Tensor:
    r"""

    Inplace version of ``round`` API, output Tensor will be inplaced with input ``x``.
    Please refer to :ref:`api_paddle_round`.

    Args:
        x (Tensor): Input of Round operator, an N-D Tensor, with data type bfloat16, int32, int64, float32, float64, float16, complex64 or complex128. Alias: ``input``.
        decimals(int): Rounded decimal place (default: 0).
        name (str|None, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor. Output of Round operator, same as input ``x``.

    Examples:
        .. code-block:: pycon

            >>> import paddle

            >>> x = paddle.to_tensor([-0.5, -0.2, 0.6, 1.5])
            >>> x.round_()
            >>> print(x)
            Tensor(shape=[4], dtype=float32, place=Place(cpu), stop_gradient=True,
            [-0., -0.,  1.,  2.])
    """
    ...


@add_doc_and_signature
def abs(
    x: Tensor,
    name: str | None = None,
    *,
    out: Tensor | None = None,
) -> Tensor:
    r"""
    Perform elementwise abs for input `x`.

    .. math::

        out = |x|

    Args:
        x (Tensor): The input Tensor with data type int32, int64, float16, float32, float64, complex64 and complex128.
            Alias: ``input``.
        name (str|None, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.
        out (Tensor|None, optional): The output tensor. Default: None.

    Returns:
        Tensor.A Tensor with the same data type and shape as :math:`x`.

    Examples:
        .. code-block:: pycon

            >>> import paddle

            >>> x = paddle.to_tensor([-0.4, -0.2, 0.1, 0.3])
            >>> out = paddle.abs(x)
            >>> print(out)
            Tensor(shape=[4], dtype=float32, place=Place(cpu), stop_gradient=True,
            [0.40000001, 0.20000000, 0.10000000, 0.30000001])
    """
    ...


@add_doc_and_signature
def abs_(
    x: Tensor,
    name: str | None = None,
) -> Tensor:
    r"""
    Inplace version of ``abs`` API, the output Tensor will be inplaced with input ``x``.
    """
    ...


@add_doc_and_signature
def nextafter(
    x: Tensor,
    y: Tensor,
    name: str | None = None,
    *,
    out: Tensor | None = None,
) -> Tensor:
    r"""
    Return the next floating-point value after input towards other, elementwise.
    The shapes of input and other must be broadcastable.

    Args:
        x (Tensor): An N-D Tensor, the data type is float32, float64.
        y (Tensor): An N-D Tensor, the data type is float32, float64.
        name(str, optional):Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.
        out (Tensor, optional): The output Tensor. If set, the result will be stored in this Tensor. Default: None.

    Returns:
        out (Tensor): An N-D Tensor, the shape and data type is the same with input.

    Examples:
        .. code-block:: pycon

            >>> import paddle
            >>> out = paddle.nextafter(paddle.to_tensor([1.0, 2.0]), paddle.to_tensor([2.0, 1.0]))
            >>> out
            Tensor(shape=[2], dtype=float32, place=Place(cpu), stop_gradient=True,
            [1.00000012, 1.99999988])
    """
    ...


@add_doc_and_signature
def angle(
    x: Tensor,
    name: str | None = None,
    *,
    out: Tensor | None = None,
) -> Tensor:
    r"""
    Element-wise angle of complex numbers. For non-negative real numbers, the angle is 0 while
    for negative real numbers, the angle is :math:`\pi`, and NaNs are propagated.

    Equation:
        .. math::

            angle(x)=arctan2(x.imag, x.real)

    Args:
        x (Tensor): An N-D Tensor, the data type is complex64, complex128, or float32, float64 .
        name (str|None, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.
        out (Tensor, optional): The output Tensor. If set, the result will be stored in this Tensor. Default: None.

    Returns:
        Tensor: An N-D Tensor of real data type with the same precision as that of x's data type.

    Examples:
        .. code-block:: pycon

            >>> import paddle

            >>> x = paddle.to_tensor([-2, -1, 0, 1]).unsqueeze(-1).astype('float32')
            >>> y = paddle.to_tensor([-2, -1, 0, 1]).astype('float32')
            >>> z = x + 1j * y
            >>> z
            Tensor(shape=[4, 4], dtype=complex64, place=Place(cpu), stop_gradient=True,
            [[(-2.00000000-2.00000000j), (-2.00000000-1.00000000j),
              (-2.00000000+0.00000000j), (-2.00000000+1.00000000j)],
             [(-1.00000000-2.00000000j), (-1.00000000-1.00000000j),
              (-1.00000000+0.00000000j), (-1.00000000+1.00000000j)],
             [(0.00000000-2.00000000j) , (0.00000000-1.00000000j) ,
               (0.00000000+0.00000000j),  (0.00000000+1.00000000j)],
             [ (1.00000000-2.00000000j),  (1.00000000-1.00000000j),
               (1.00000000+0.00000000j),  (1.00000000+1.00000000j)]])

            >>> theta = paddle.angle(z)
            >>> theta
            Tensor(shape=[4, 4], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[-2.35619450, -2.67794514,  3.14159274,  2.67794514],
             [-2.03444386, -2.35619450,  3.14159274,  2.35619450],
             [-1.57079637, -1.57079637,  0.        ,  1.57079637],
             [-1.10714877, -0.78539819,  0.        ,  0.78539819]])
    """
    ...


@add_doc_and_signature
def real(
    x: Tensor,
    name: str | None = None,
    *,
    out: Tensor | None = None,
) -> Tensor:
    r"""
    Returns a new Tensor containing real values of the input Tensor.

    Args:
        x (Tensor): the input Tensor, its data type could be complex64 or complex128. Alias: ``input``.
        name (str|None, optional): The default value is None. Normally there is no need for
            user to set this property. For more information, please refer to :ref:`api_guide_Name` .

    Keyword args:
        out(Tensor, optional): The output tensor.

    Returns:
        Tensor: a Tensor containing real values of the input Tensor.

    Examples:
        .. code-block:: pycon

            >>> import paddle

            >>> x = paddle.to_tensor(
            ...     [
            ...         [1 + 6j, 2 + 5j, 3 + 4j],
            ...         [4 + 3j, 5 + 2j, 6 + 1j],
            ...     ]
            ... )
            >>> print(x)
            Tensor(shape=[2, 3], dtype=complex64, place=Place(cpu), stop_gradient=True,
             [[(1.00000000+6.00000000j), (2.00000000+5.00000000j),
               (3.00000000+4.00000000j)],
              [(4.00000000+3.00000000j), (5.00000000+2.00000000j),
               (6.00000000+1.00000000j)]])

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
    ...


@add_doc_and_signature
def imag(
    x: Tensor,
    name: str | None = None,
    *,
    out: Tensor | None = None,
) -> Tensor:
    r"""
    Returns a new tensor containing imaginary values of input tensor.

    Args:
        x (Tensor): the input tensor, its data type could be complex64 or complex128.
        name (str|None, optional): The default value is None. Normally there is no need for
            user to set this property. For more information, please refer to :ref:`api_guide_Name` .

    Keyword args:
        out(Tensor, optional): The output tensor.

    Returns:
        Tensor: a tensor containing imaginary values of the input tensor.

    Examples:
        .. code-block:: pycon

            >>> import paddle

            >>> x = paddle.to_tensor(
            ...     [
            ...         [1 + 6j, 2 + 5j, 3 + 4j],
            ...         [4 + 3j, 5 + 2j, 6 + 1j],
            ...     ]
            ... )
            >>> print(x)
            Tensor(shape=[2, 3], dtype=complex64, place=Place(cpu), stop_gradient=True,
            [[(1.00000000+6.00000000j), (2.00000000+5.00000000j), (3.00000000+4.00000000j)],
             [(4.00000000+3.00000000j), (5.00000000+2.00000000j), (6.00000000+1.00000000j)]])

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
    ...


@add_doc_and_signature
def heaviside(
    x: Tensor,
    y: Tensor,
    name: str | None = None,
    *,
    out: Tensor | None = None,
) -> Tensor:
    r"""
    Computes the Heaviside step function determined by corresponding element in y for each element in x. The equation is

    .. math::
        heaviside(x, y)=
            \left\{
                \begin{array}{lcl}
                0,& &\text{if} \ x < 0, \\
                y,& &\text{if} \ x = 0, \\
                1,& &\text{if} \ x > 0.
                \end{array}
            \right.

    Note:
        ``paddle.heaviside`` supports broadcasting. If you want know more about broadcasting, please refer to `Introduction to Tensor`_ .

        .. _Introduction to Tensor: ../../guides/beginner/tensor_en.html#chapter5-broadcasting-of-tensor

    Args:
        x (Tensor): The input tensor of Heaviside step function, it's data type should be bfloat16, float16, float32, float64, int32 or int64. Alias: ``input``.
        y (Tensor): The tensor that determines a Heaviside step function, it's data type should be bfloat16, float16, float32, float64, int32 or int64. Alias: ``values``.
        name (str|None, optional): Name for the operation (optional, default is None). Normally there is no need for user to set this property. For more information, please refer to :ref:`api_guide_Name`.
        out (Tensor, optional): The output Tensor. If set, the result will be stored in this Tensor. Default: None.

    Returns:
        N-D Tensor. A location into which the result is stored. If x and y have different shapes and are broadcastable, the resulting tensor shape is the shape of x and y after broadcasting. If x, y have the same shape, its shape is the same as x and y.

    Examples:
        .. code-block:: pycon

            >>> import paddle
            >>> x = paddle.to_tensor([-0.5, 0, 0.5])
            >>> y = paddle.to_tensor([0.1])
            >>> paddle.heaviside(x, y)
            Tensor(shape=[3], dtype=float32, place=Place(cpu), stop_gradient=True,
            [0.        , 0.10000000, 1.        ])
            >>> x = paddle.to_tensor([[-0.5, 0, 0.5], [-0.5, 0.5, 0]])
            >>> y = paddle.to_tensor([0.1, 0.2, 0.3])
            >>> paddle.heaviside(x, y)
            Tensor(shape=[2, 3], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[0.        , 0.20000000, 1.        ],
             [0.        , 1.        , 0.30000001]])
    """
    ...


@add_doc_and_signature
def asin(x: Tensor, name: str | None = None) -> Tensor:
    r"""
    Arcsine Operator.

    .. math::
        out = sin^{-1}(x)

    Args:
        x (Tensor): Input of Asin operator, an N-D Tensor, with data type float32, float64, float16, bfloat16,
            uint8, int8, int16, int32, int64, complex64 or complex128. Alias: ``input``.
        name (str|None, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor. Same shape and data type as input (integer types are autocasted into float32)

    Examples:
        .. code-block:: pycon

            >>> import paddle

            >>> x = paddle.to_tensor([-0.4, -0.2, 0.1, 0.3])
            >>> out = paddle.asin(x)
            >>> print(out)
            Tensor(shape=[4], dtype=float32, place=Place(cpu), stop_gradient=True,
            [-0.41151685, -0.20135793,  0.10016742,  0.30469266])
    """
    ...


@add_doc_and_signature
def inverse(
    x: Tensor,
    name: str | None = None,
    *,
    out: Tensor | None = None,
) -> Tensor:
    r"""
    Takes the inverse of the square matrix. A square matrix is a matrix with
    the same number of rows and columns. The input can be a square matrix
    (2-D Tensor) or batches of square matrices.

    Args:
        x (Tensor): The input tensor. The last two
            dimensions should be equal. When the number of dimensions is
            greater than 2, it is treated as batches of square matrix. The data
            type can be float32, float64, complex64, complex128.
            Alias: ``input``.
        name (str|None, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.
        out (Tensor, optional): The output Tensor. If set, the result will be stored in this Tensor. Default: None.

    Returns:
        Tensor: A Tensor holds the inverse of x. The shape and data type
                        is the same as x.

    Examples:
        .. code-block:: pycon

            >>> import paddle

            >>> mat = paddle.to_tensor([[2, 0], [0, 2]], dtype='float32')
            >>> inv = paddle.inverse(mat)
            >>> print(inv)
            Tensor(shape=[2, 2], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[0.50000000, 0.        ],
             [0.        , 0.50000000]])

    """
    ...


@add_doc_and_signature
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
            Alias: ``input``.
        y (Tensor): The input tensor, it's data type should be float16, float32, float64.
            Alias: ``other``.
        rtol (float, optional): The relative tolerance. Default: :math:`1e-5` .
        atol (float, optional): The absolute tolerance. Default: :math:`1e-8` .
        equal_nan (bool, optional): ${equal_nan_comment}. Default: False.
        name (str|None, optional): Name for the operation. For more information, please
            refer to :ref:`api_guide_Name`. Default: None.

    Returns:
        Tensor: The output tensor, it's data type is bool.

    Examples:
        .. code-block:: pycon

            >>> import paddle

            >>> x = paddle.to_tensor([10000.0, 1e-07])
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
    ...


@add_doc_and_signature
def fmax(
    x: Tensor,
    y: Tensor,
    name: str | None = None,
    *,
    out: Tensor | None = None,
) -> Tensor:
    r"""
    Compares elements at corresponding positions of two tensors and returns a new tensor containing maximum value of element.
    If one of them is a nan value, the other value is directly returned, if both are nan values, then the first nan value is returned.
    The equation is:

    .. math::
        out = fmax(x, y)

    Note:
        ``paddle.fmax`` supports broadcasting. If you want know more about broadcasting, please refer to `Introduction to Tensor`_ .

        .. _Introduction to Tensor: ../../guides/beginner/tensor_en.html#chapter5-broadcasting-of-tensor

    Args:
        x (Tensor): the input tensor, it's data type should be bfloat16, float16, float32, float64, int32, int64. Alias: ``input``.
        y (Tensor): the input tensor, it's data type should be bfloat16, float16, float32, float64, int32, int64. Alias: ``other``.
        name (str|None, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        N-D Tensor. A location into which the result is stored. If x, y have different shapes and are "broadcastable", the resulting tensor shape is the shape of x and y after broadcasting. If x, y have the same shape,  its shape is the same as x and y.

    Examples:

        .. code-block:: pycon

            >>> import paddle

            >>> x = paddle.to_tensor([[1, 2], [7, 8]])
            >>> y = paddle.to_tensor([[3, 4], [5, 6]])
            >>> res = paddle.fmax(x, y)
            >>> print(res)
            Tensor(shape=[2, 2], dtype=int64, place=Place(cpu), stop_gradient=True,
            [[3, 4],
             [7, 8]])

            >>> x = paddle.to_tensor([[1, 2, 3], [1, 2, 3]])
            >>> y = paddle.to_tensor([3, 0, 4])
            >>> res = paddle.fmax(x, y)
            >>> print(res)
            Tensor(shape=[2, 3], dtype=int64, place=Place(cpu), stop_gradient=True,
            [[3, 2, 4],
             [3, 2, 4]])

            >>> x = paddle.to_tensor([2, 3, 5], dtype='float32')
            >>> y = paddle.to_tensor([1, float("nan"), float("nan")], dtype='float32')
            >>> res = paddle.fmax(x, y)
            >>> print(res)
            Tensor(shape=[3], dtype=float32, place=Place(cpu), stop_gradient=True,
            [2., 3., 5.])

            >>> x = paddle.to_tensor([5, 3, float("inf")], dtype='float32')
            >>> y = paddle.to_tensor([1, -float("inf"), 5], dtype='float32')
            >>> res = paddle.fmax(x, y)
            >>> print(res)
            Tensor(shape=[3], dtype=float32, place=Place(cpu), stop_gradient=True,
            [5.  , 3.  , inf.])
    """
    ...


@add_doc_and_signature
def fmin(
    x: Tensor,
    y: Tensor,
    name: str | None = None,
    *,
    out: Tensor | None = None,
) -> Tensor:
    r"""
    Compares elements at corresponding positions of two tensors and returns a new tensor containing minimum value of element.
    If one of them is a nan value, the other value is directly returned, if both are nan values, then the first nan value is returned.
    The equation is:

    .. math::
        out = fmin(x, y)

    Note:
        ``paddle.fmin`` supports broadcasting. If you want know more about broadcasting, please refer to `Introduction to Tensor`_ .

        .. _Introduction to Tensor: ../../guides/beginner/tensor_en.html#chapter5-broadcasting-of-tensor

    Args:
        x (Tensor): the input tensor, it's data type should be bfloat16, float16, float32, float64, int32, int64.
        y (Tensor): the input tensor, it's data type should be bfloat16, float16, float32, float64, int32, int64.
        name (str|None, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        N-D Tensor. A location into which the result is stored. If x, y have different shapes and are "broadcastable", the resulting tensor shape is the shape of x and y after broadcasting. If x, y have the same shape,  its shape is the same as x and y.

    Examples:

        .. code-block:: pycon

            >>> import paddle

            >>> x = paddle.to_tensor([[1, 2], [7, 8]])
            >>> y = paddle.to_tensor([[3, 4], [5, 6]])
            >>> res = paddle.fmin(x, y)
            >>> print(res)
            Tensor(shape=[2, 2], dtype=int64, place=Place(cpu), stop_gradient=True,
            [[1, 2],
             [5, 6]])

            >>> x = paddle.to_tensor([[[1, 2, 3], [1, 2, 3]]])
            >>> y = paddle.to_tensor([3, 0, 4])
            >>> res = paddle.fmin(x, y)
            >>> print(res)
            Tensor(shape=[1, 2, 3], dtype=int64, place=Place(cpu), stop_gradient=True,
            [[[1, 0, 3],
              [1, 0, 3]]])

            >>> x = paddle.to_tensor([2, 3, 5], dtype='float32')
            >>> y = paddle.to_tensor([1, float("nan"), float("nan")], dtype='float32')
            >>> res = paddle.fmin(x, y)
            >>> print(res)
            Tensor(shape=[3], dtype=float32, place=Place(cpu), stop_gradient=True,
            [1., 3., 5.])

            >>> x = paddle.to_tensor([5, 3, float("inf")], dtype='float64')
            >>> y = paddle.to_tensor([1, -float("inf"), 5], dtype='float64')
            >>> res = paddle.fmin(x, y)
            >>> print(res)
            Tensor(shape=[3], dtype=float64, place=Place(cpu), stop_gradient=True,
            [ 1.  , -inf.,  5.  ])
    """
    ...


@add_doc_and_signature
def bincount(
    x: Tensor,
    weights: Tensor | None = None,
    minlength: int = 0,
    name: str | None = None,
) -> Tensor:
    r"""
    Computes frequency of each value in the input tensor.

    Args:
        x (Tensor): A Tensor with non-negative integer. Should be 1-D tensor.
        weights (Tensor, optional): Weight for each value in the input tensor. Should have the same shape as input. Default is None.
        minlength (int, optional): Minimum number of bins. Should be non-negative integer. Default is 0.
        name (str|None, optional): Normally there is no need for user to set this property.
            For more information, please refer to :ref:`api_guide_Name`. Default is None.

    Returns:
        Tensor: The tensor of frequency.

    Examples:
        .. code-block:: pycon

            >>> import paddle
            >>> x = paddle.to_tensor([1, 2, 1, 4, 5])
            >>> result1 = paddle.bincount(x)
            >>> print(result1)
            Tensor(shape=[6], dtype=int64, place=Place(cpu), stop_gradient=True,
            [0, 2, 1, 0, 1, 1])

            >>> w = paddle.to_tensor([2.1, 0.4, 0.1, 0.5, 0.5])
            >>> result2 = paddle.bincount(x, weights=w)
            >>> print(result2)
            Tensor(shape=[6], dtype=float32, place=Place(cpu), stop_gradient=True,
            [0.        , 2.19999981, 0.40000001, 0.        , 0.50000000, 0.50000000])
    """
    ...


@add_doc_and_signature
def bitwise_and(
    x: Tensor,
    y: Tensor,
    name: str | None = None,
    *,
    out: Tensor | None = None,
) -> Tensor:
    r"""
    Apply ``bitwise_and`` on Tensor ``X`` and ``Y``.

    .. math::
        Out = X \\& Y

    Note:
        ``paddle.bitwise_and`` supports broadcasting. If you want know more about broadcasting, please refer to `Introduction to Tensor`_ .

        .. _Introduction to Tensor: ../../guides/beginner/tensor_en.html#chapter5-broadcasting-of-tensor

    Args:
        x (Tensor): Input Tensor of ``bitwise_and``. It is a N-D Tensor of bool, uint8, int8, int16, int32, int64. Alias: ``input``.
        y (Tensor): Input Tensor of ``bitwise_and``. It is a N-D Tensor of bool, uint8, int8, int16, int32, int64. Alias: ``other``.
        name (str|None, optional): The default value is None. Normally there is no need for
            user to set this property. For more information, please refer to :ref:`api_guide_Name`.
    Keyword args:
        out (Tensor, optional): The output Tensor. If set, the result will be stored in this Tensor. Default: None.

    Returns:
        Tensor: Result of ``bitwise_and``. It is a N-D Tensor with the same data type of input Tensor.

    Examples:
        .. code-block:: pycon

            >>> import paddle
            >>> x = paddle.to_tensor([-5, -1, 1])
            >>> y = paddle.to_tensor([4, 2, -3])
            >>> res = paddle.bitwise_and(x, y)
            >>> print(res)
            Tensor(shape=[3], dtype=int64, place=Place(cpu), stop_gradient=True,
            [0, 2, 1])
    """
    ...


@add_doc_and_signature
def bitwise_and_(
    x: Tensor,
    y: Tensor,
    name: str | None = None,
) -> Tensor:
    r"""
    Inplace version of ``bitwise_and`` API, the output Tensor will be inplaced with input ``x``.
    Please refer to :ref:`api_paddle_bitwise_and`.
    """
    ...


@add_doc_and_signature
def bitwise_or(
    x: Tensor,
    y: Tensor,
    name: str | None = None,
    *,
    out: Tensor | None = None,
) -> Tensor:
    r"""
    Apply ``bitwise_or`` on Tensor ``X`` and ``Y``.

    .. math::
        Out = X | Y

    Note:
        ``paddle.bitwise_or`` supports broadcasting. If you want know more about broadcasting, please refer to `Introduction to Tensor`_ .

        .. _Introduction to Tensor: ../../guides/beginner/tensor_en.html#chapter5-broadcasting-of-tensor

    Args:
        x (Tensor): Input Tensor of ``bitwise_or``. It is a N-D Tensor of bool, uint8, int8, int16, int32, int64.
            Alias: ``input``.
        y (Tensor): Input Tensor of ``bitwise_or``. It is a N-D Tensor of bool, uint8, int8, int16, int32, int64.
            Alias: ``other``.
        name (str|None, optional): The default value is None. Normally there is no need for
            user to set this property. For more information, please refer to :ref:`api_guide_Name`.
    Keyword args:
        out (Tensor, optional): The output Tensor. If set, the result will be stored in this Tensor. Default: None.

    Returns:
        Tensor: Result of ``bitwise_or``. It is a N-D Tensor with the same data type of input Tensor.

    Examples:
        .. code-block:: pycon

            >>> import paddle
            >>> x = paddle.to_tensor([-5, -1, 1])
            >>> y = paddle.to_tensor([4, 2, -3])
            >>> res = paddle.bitwise_or(x, y)
            >>> print(res)
            Tensor(shape=[3], dtype=int64, place=Place(cpu), stop_gradient=True,
            [-1, -1, -3])
    """
    ...


@add_doc_and_signature
def bitwise_or_(
    x: Tensor,
    y: Tensor,
    name: str | None = None,
) -> Tensor:
    r"""
    Inplace version of ``bitwise_or`` API, the output Tensor will be inplaced with input ``x``.
    Please refer to :ref:`api_paddle_bitwise_or`.
    """
    ...


@add_doc_and_signature
def bitwise_xor(
    x: Tensor,
    y: Tensor,
    name: str | None = None,
    *,
    out: Tensor | None = None,
) -> Tensor:
    r"""
    Apply ``bitwise_xor`` on Tensor ``X`` and ``Y``.

    .. math::
        Out = X ^\\wedge Y

    Note:
        ``paddle.bitwise_xor`` supports broadcasting. If you want know more about broadcasting, please refer to `Introduction to Tensor`_ .

        .. _Introduction to Tensor: ../../guides/beginner/tensor_en.html#chapter5-broadcasting-of-tensor

    Args:
        x (Tensor): Input Tensor of ``bitwise_xor``. It is a N-D Tensor of bool, uint8, int8, int16, int32, int64. Alias: ``input``.
        y (Tensor): Input Tensor of ``bitwise_xor``. It is a N-D Tensor of bool, uint8, int8, int16, int32, int64. Alias: ``other``.
        name (str|None, optional): The default value is None. Normally there is no need for
            user to set this property. For more information, please refer to :ref:`api_guide_Name`.
    Keyword args:
        out (Tensor, optional): The output Tensor. If set, the result will be stored in this Tensor. Default: None.

    Returns:
        Tensor: Result of ``bitwise_xor``. It is a N-D Tensor with the same data type of input Tensor.

    Examples:
        .. code-block:: pycon

            >>> import paddle
            >>> x = paddle.to_tensor([-5, -1, 1])
            >>> y = paddle.to_tensor([4, 2, -3])
            >>> res = paddle.bitwise_xor(x, y)
            >>> print(res)
            Tensor(shape=[3], dtype=int64, place=Place(cpu), stop_gradient=True,
            [-1, -3, -4])
    """
    ...


@add_doc_and_signature
def bitwise_xor_(
    x: Tensor,
    y: Tensor,
    name: str | None = None,
) -> Tensor:
    r"""
    Inplace version of ``bitwise_xor`` API, the output Tensor will be inplaced with input ``x``.
    Please refer to :ref:`api_paddle_bitwise_xor`.
    """
    ...


@add_doc_and_signature
def bitwise_not(
    x: Tensor,
    name: str | None = None,
    *,
    out: Tensor | None = None,
) -> Tensor:
    r"""
    Apply ``bitwise_not`` on Tensor ``X``.

    .. math::
        Out = \\sim X

    Note:
        ``paddle.bitwise_not`` supports broadcasting. If you want know more about broadcasting, please refer to `Introduction to Tensor`_ .

        .. _Introduction to Tensor: ../../guides/beginner/tensor_en.html#chapter5-broadcasting-of-tensor

    Args:
        x (Tensor): Input Tensor of ``bitwise_not``. It is a N-D Tensor of bool, uint8, int8, int16, int32, int64. Alias: ``input``.
        name (str|None, optional): The default value is None. Normally there is no need for
            user to set this property. For more information, please refer to :ref:`api_guide_Name`.
    Keyword args:
        out (Tensor, optional): The output Tensor. If set, the result will be stored in this Tensor. Default: None.

    Returns:
        Tensor: Result of ``bitwise_not``. It is a N-D Tensor with the same data type of input Tensor.

    Examples:
        .. code-block:: pycon

            >>> import paddle
            >>> x = paddle.to_tensor([-5, -1, 1])
            >>> res = paddle.bitwise_not(x)
            >>> print(res)
            Tensor(shape=[3], dtype=int64, place=Place(cpu), stop_gradient=True,
            [ 4,  0, -2])
    """
    ...


@add_doc_and_signature
def bitwise_not_(
    x: Tensor,
    name: str | None = None,
) -> Tensor:
    r"""
    Inplace version of ``bitwise_not`` API, the output Tensor will be inplaced with input ``x``.
    Please refer to :ref:`api_paddle_bitwise_not`.
    """
    ...


@add_doc_and_signature
def bitwise_left_shift(
    x: Tensor,
    y: Tensor,
    is_arithmetic: bool = True,
    name: str | None = None,
    *,
    out: Tensor | None = None,
) -> Tensor:
    r"""
    Apply ``bitwise_left_shift`` on Tensor ``X`` and ``Y`` .

    .. math::

        Out = X \ll Y

    .. note::

        ``paddle.bitwise_left_shift`` supports broadcasting. If you want know more about broadcasting, please refer to please refer to `Introduction to Tensor`_ .

    .. _Introduction to Tensor: ../../guides/beginner/tensor_en.html#chapter5-broadcasting-of-tensor

    Args:
        x (Tensor): Input Tensor of ``bitwise_left_shift`` . It is a N-D Tensor of uint8, int8, int16, int32, int64. Alias: ``input``.
        y (Tensor): Input Tensor of ``bitwise_left_shift`` . It is a N-D Tensor of uint8, int8, int16, int32, int64. Alias: ``other``.
        is_arithmetic (bool, optional): A boolean indicating whether to choose arithmetic shift, if False, means logic shift. Default True.
        name (str|None, optional): The default value is None. Normally there is no need for user to set this property. For more information, please refer to :ref:`api_guide_Name`.
    Keyword args:
        out (Tensor, optional): The output Tensor. If set, the result will be stored in this Tensor. Default: None.

    Returns:
        Tensor: Result of ``bitwise_left_shift`` . It is a N-D Tensor with the same data type of input Tensor.

    Examples:
        .. code-block:: pycon
            :name: bitwise_left_shift_example1

            >>> import paddle
            >>> x = paddle.to_tensor([[1, 2, 4, 8], [16, 17, 32, 65]])
            >>> y = paddle.to_tensor(
            ...     [
            ...         [
            ...             1,
            ...             2,
            ...             3,
            ...             4,
            ...         ],
            ...         [2, 3, 2, 1],
            ...     ]
            ... )
            >>> paddle.bitwise_left_shift(x, y, is_arithmetic=True)
            Tensor(shape=[2, 4], dtype=int64, place=Place(gpu:0), stop_gradient=True,
                   [[2  , 8  , 32 , 128],
                    [64 , 136, 128, 130]])

        .. code-block:: pycon
            :name: bitwise_left_shift_example2

            >>> import paddle
            >>> x = paddle.to_tensor([[1, 2, 4, 8], [16, 17, 32, 65]])
            >>> y = paddle.to_tensor(
            ...     [
            ...         [
            ...             1,
            ...             2,
            ...             3,
            ...             4,
            ...         ],
            ...         [2, 3, 2, 1],
            ...     ]
            ... )
            >>> paddle.bitwise_left_shift(x, y, is_arithmetic=False)
            Tensor(shape=[2, 4], dtype=int64, place=Place(gpu:0), stop_gradient=True,
                [[2  , 8  , 32 , 128],
                    [64 , 136, 128, 130]])
    """
    ...


@add_doc_and_signature
def bitwise_left_shift_(
    x: Tensor,
    y: Tensor,
    is_arithmetic: bool = True,
    name: str | None = None,
) -> Tensor:
    r"""
    Inplace version of ``bitwise_left_shift`` API, the output Tensor will be inplaced with input ``x``.
    Please refer to :ref:`api_paddle_bitwise_left_shift`.
    """
    ...


@add_doc_and_signature
def bitwise_right_shift(
    x: Tensor,
    y: Tensor,
    is_arithmetic: bool = True,
    name: str | None = None,
    *,
    out: Tensor | None = None,
) -> Tensor:
    r"""
    Apply ``bitwise_right_shift`` on Tensor ``X`` and ``Y`` .

    .. math::

        Out = X \gg Y

    .. note::

        ``paddle.bitwise_right_shift`` supports broadcasting. If you want know more about broadcasting, please refer to please refer to `Introduction to Tensor`_ .

    .. _Introduction to Tensor: ../../guides/beginner/tensor_en.html#chapter5-broadcasting-of-tensor

    Args:
        x (Tensor): Input Tensor of ``bitwise_right_shift`` . It is a N-D Tensor of uint8, int8, int16, int32, int64. Alias: ``input``.
        y (Tensor): Input Tensor of ``bitwise_right_shift`` . It is a N-D Tensor of uint8, int8, int16, int32, int64. Alias: ``other``.
        is_arithmetic (bool, optional): A boolean indicating whether to choose arithmetic shift, if False, means logic shift. Default True.
        name (str|None, optional): The default value is None. Normally there is no need for user to set this property. For more information, please refer to :ref:`api_guide_Name`.
    Keyword args:
        out (Tensor, optional): The output Tensor. If set, the result will be stored in this Tensor. Default: None.

    Returns:
        Tensor: Result of ``bitwise_right_shift`` . It is a N-D Tensor with the same data type of input Tensor.

    Examples:
        .. code-block:: pycon
            :name: bitwise_right_shift_example1

            >>> import paddle
            >>> x = paddle.to_tensor([[10, 20, 40, 80], [16, 17, 32, 65]])
            >>> y = paddle.to_tensor(
            ...     [
            ...         [
            ...             1,
            ...             2,
            ...             3,
            ...             4,
            ...         ],
            ...         [2, 3, 2, 1],
            ...     ]
            ... )
            >>> paddle.bitwise_right_shift(x, y, is_arithmetic=True)
            Tensor(shape=[2, 4], dtype=int64, place=Place(gpu:0), stop_gradient=True,
                   [[5 , 5 , 5 , 5 ],
                    [4 , 2 , 8 , 32]])

        .. code-block:: pycon
            :name: bitwise_right_shift_example2

            >>> import paddle
            >>> x = paddle.to_tensor([[-10, -20, -40, -80], [-16, -17, -32, -65]], dtype=paddle.int8)
            >>> y = paddle.to_tensor(
            ...     [
            ...         [
            ...             1,
            ...             2,
            ...             3,
            ...             4,
            ...         ],
            ...         [2, 3, 2, 1],
            ...     ],
            ...     dtype=paddle.int8,
            ... )
            >>> paddle.bitwise_right_shift(x, y, is_arithmetic=False)
            Tensor(shape=[2, 4], dtype=int8, place=Place(gpu:0), stop_gradient=True,
                [[123, 59 , 27 , 11 ],
                    [60 , 29 , 56 , 95 ]])
    """
    ...


@add_doc_and_signature
def bitwise_right_shift_(
    x: Tensor,
    y: Tensor,
    is_arithmetic: bool = True,
    name: str | None = None,
) -> Tensor:
    r"""
    Inplace version of ``bitwise_right_shift`` API, the output Tensor will be inplaced with input ``x``.
    Please refer to :ref:`api_paddle_bitwise_right_shift`.
    """
    ...


@add_doc_and_signature
def conj(x: Tensor, name: str | None = None) -> Tensor:
    r"""
    This function computes the conjugate of the Tensor elementwisely.

    Args:
        x (Tensor): The input Tensor which hold the complex numbers.
            Optional data types are: bfloat16, float16, complex64, complex128, float32, float64, int32 or int64.
        name (str|None, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        out (Tensor): The conjugate of input. The shape and data type is the same with input. If the elements of tensor is real type such as float32, float64, int32 or int64, the out is the same with input.

    Examples:
        .. code-block:: pycon

            >>> import paddle

            >>> data = paddle.to_tensor([[1 + 1j, 2 + 2j, 3 + 3j], [4 + 4j, 5 + 5j, 6 + 6j]])
            >>> data
            Tensor(shape=[2, 3], dtype=complex64, place=Place(cpu), stop_gradient=True,
            [[(1+1j), (2+2j), (3+3j)],
             [(4+4j), (5+5j), (6+6j)]])

            >>> conj_data = paddle.conj(data)
            >>> conj_data
            Tensor(shape=[2, 3], dtype=complex64, place=Place(cpu), stop_gradient=True,
            [[(1-1j), (2-2j), (3-3j)],
             [(4-4j), (5-5j), (6-6j)]])
    """
    ...


@add_doc_and_signature
def i1(
    x: Tensor, name: str | None = None, *, out: Tensor | None = None
) -> Tensor:
    r"""
    The function is used to calculate modified bessel function of order 1.

    Args:
        x (Tensor): The input tensor, it's data type should be float32, float64,
            uint8, int8, int16, int32, int64.
        name (str|None, optional): For details, please refer to :ref:`api_guide_Name`. Generally, no setting is required. Default: None.
    Keyword args:
        out (Tensor), A Tensor. the value of the modified bessel function of order 1 at x
            (integer types are autocasted into float32).

    Returns:
        Tensor: The value of modified bessel function of order 1.

    Examples:
        .. code-block:: pycon

            >>> import paddle

            >>> x = paddle.to_tensor([0, 1, 2, 3, 4], dtype="float32")
            >>> print(paddle.i1(x))
            Tensor(shape=[5], dtype=float32, place=Place(cpu), stop_gradient=True,
            [0.        , 0.56515908, 1.59063685, 3.95337057, 9.75946712])
    """
    ...


@add_doc_and_signature
def i1e(
    x: Tensor, name: str | None = None, *, out: Tensor | None = None
) -> Tensor:
    r"""
    The function is used to calculate exponentially scaled modified Bessel function of order 1.

    Args:
        x (Tensor): The input tensor, it's data type should be float32, float64,
            uint8, int8, int16, int32, int64.
        name (str|None, optional): For details, please refer to :ref:`api_guide_Name`. Generally, no setting is required. Default: None.
    Keyword args:
        out (Tensor), A Tensor. the value of the exponentially scaled modified Bessel function of order 1 at x
            (integer types are autocasted into float32).

    Returns:
        Tensor: The value of exponentially scaled modified Bessel function of order 1.

    Examples:
        .. code-block:: pycon

            >>> import paddle

            >>> x = paddle.to_tensor([0, 1, 2, 3, 4], dtype="float32")
            >>> print(paddle.i1e(x))
            Tensor(shape=[5], dtype=float32, place=Place(cpu), stop_gradient=True,
            [0.        , 0.20791042, 0.21526928, 0.19682673, 0.17875087])
    """
    ...


@add_doc_and_signature
def addmm(
    input: Tensor,
    x: Tensor,
    y: Tensor,
    beta: float = 1.0,
    alpha: float = 1.0,
    name: str | None = None,
    *,
    out: Tensor | None = None,
) -> Tensor:
    r"""
    Perform matrix multiplication for input $x$ and $y$.
    $input$ is added to the final result.
    The equation is:

    ..  math::
        Out = alpha * x * y + beta * input

    $Input$, $x$ and $y$ can carry the LoD (Level of Details) information, or not. But the output only shares the LoD information with input $input$.

    Args:
        input (Tensor): The input Tensor to be added to the final result.
        x (Tensor): The first input Tensor for matrix multiplication. Alias: ``mat1``.
        y (Tensor): The second input Tensor for matrix multiplication. Alias: ``mat2``.
        beta (float, optional): Coefficient of $input$, default is 1.
        alpha (float, optional): Coefficient of $x*y$, default is 1.
        name (str|None, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.
    Keyword args:
        out (Tensor, optional): The output Tensor. If set, the result will be stored in this Tensor. Default: None.

    Returns:
        Tensor: The output Tensor of addmm.

    Examples:
        .. code-block:: pycon

            >>> import paddle

            >>> x = paddle.ones([2, 2])
            >>> y = paddle.ones([2, 2])
            >>> input = paddle.ones([2, 2])

            >>> out = paddle.addmm(input=input, x=x, y=y, beta=0.5, alpha=5.0)

            >>> print(out)
            Tensor(shape=[2, 2], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[10.50000000, 10.50000000],
             [10.50000000, 10.50000000]])
    """
    ...


@add_doc_and_signature
def addmm_(
    input: Tensor,
    x: Tensor,
    y: Tensor,
    beta: float = 1.0,
    alpha: float = 1.0,
    name: str | None = None,
) -> Tensor:
    r"""
    Inplace version of ``addmm`` API, the output Tensor will be inplaced with input ``input``.
    Please refer to :ref:`api_paddle_addmm`.
    """
    ...


@add_doc_and_signature
def baddbmm(
    input: Tensor,
    x: Tensor,
    y: Tensor,
    beta: float = 1.0,
    alpha: float = 1.0,
    out_dtype: paddle.dtype | None = None,
    name: str | None = None,
    *,
    out: Tensor | None = None,
) -> Tensor:
    r"""
    Perform batch matrix multiplication for input :math:`x` and :math:`y`.
    :math:`input` is added to the final result.
    The equation is:
    .. math::
        out = \beta \times input + \alpha \times x \times y
    where :math:`\beta` and :math:`\alpha` are scaling factors.
    Args:
        input (Tensor): The input tensor to be added to the final result. It should be a 2-D or 3-D tensor.
            Data type should be float16, float32, float64, uint16.
        x (Tensor): The first batch of matrices to be multiplied. It should be a 3-D tensor with shape [b, n, p].
            Data type should be float16, float32, float64, uint16.
            Alias: ``batch1``.
        y (Tensor): The second batch of matrices to be multiplied. It should be a 3-D tensor with shape [b, p, m].
            Data type should be float16, float32, float64, uint16.
            Alias: ``batch2``.
        beta (float, optional): The scaling factor for input. Default: 1.0.
        alpha (float, optional): The scaling factor for x @ y. Default: 1.0.
        out_dtype (paddle.dtype|None, optional): The desired data type of the returned tensor. If None, the output tensor will have the same data type as input. Default: None.
        name (str|None, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.
    Keyword args:
        out (Tensor, optional): The output Tensor. If set, the result will be stored in this Tensor. Default: None.
    Returns:
        Tensor: The output tensor should be a 3-D tensor with shape [b, n, m].
    Examples:
        .. code-block:: pycon

            >>> import paddle

            >>> x = paddle.ones([2, 2, 2])
            >>> y = paddle.ones([2, 2, 2])
            >>> input = paddle.ones([2, 2, 2])

            >>> out = paddle.baddbmm(input=input, x=x, y=y, beta=0.5, alpha=5.0)
            >>> out
            Tensor(shape=[2, 2, 2], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[[10.50000000, 10.50000000],
              [10.50000000, 10.50000000]],
             [[10.50000000, 10.50000000],
              [10.50000000, 10.50000000]]])
    """
    ...


@add_doc_and_signature
def baddbmm_(
    input: Tensor,
    x: Tensor,
    y: Tensor,
    beta: float = 1.0,
    alpha: float = 1.0,
    out_dtype: paddle.dtype | None = None,
    name: str | None = None,
) -> Tensor:
    r"""
    Inplace version of ``baddbmm`` API, the output Tensor will be inplaced with input ``input``.
    Please refer to :ref:`api_paddle_baddbmm`.
    """
    ...


@add_doc_and_signature
def cross(
    x: Tensor,
    y: Tensor,
    axis: int = 9,
    name: str | None = None,
) -> Tensor:
    r"""
    Computes the cross product between two tensors along an axis.

    Inputs must have the same shape, and the length of their axes should be equal to 3.
    If `axis` is not given, it defaults to the first axis found with the length 3.

    Args:
        x (Tensor): The first input tensor, the data type is float16, float32, float64, int32, int64, complex64, complex128. Alias: ``input``.
        y (Tensor): The second input tensor, the data type is float16, float32, float64, int32, int64, complex64, complex128. Alias: ``other``.
        axis (int, optional): The axis along which to compute the cross product. It defaults to be 9 which indicates using the first axis found with the length 3. Alias: ``dim``.
        name (str|None, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor. A Tensor with same data type as `x`.

    Examples:
        .. code-block:: pycon

            >>> import paddle

            >>> x = paddle.to_tensor(
            ...     [
            ...         [1.0, 1.0, 1.0],
            ...         [2.0, 2.0, 2.0],
            ...         [3.0, 3.0, 3.0],
            ...     ]
            ... )
            >>> y = paddle.to_tensor(
            ...     [
            ...         [1.0, 1.0, 1.0],
            ...         [1.0, 1.0, 1.0],
            ...         [1.0, 1.0, 1.0],
            ...     ]
            ... )
            >>> z1 = paddle.cross(x, y)
            >>> print(z1)
            Tensor(shape=[3, 3], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[-1., -1., -1.],
             [ 2.,  2.,  2.],
             [-1., -1., -1.]])

            >>> z2 = paddle.cross(x, y, axis=1)
            >>> print(z2)
            Tensor(shape=[3, 3], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[0., 0., 0.],
             [0., 0., 0.],
             [0., 0., 0.]])
    """
    ...


@add_doc_and_signature
def dist(x: Tensor, y: Tensor, p: float = 2, name: str | None = None) -> Tensor:
    r"""
    Returns the p-norm of (x - y). It is not a norm in a strict sense, only as a measure
    of distance. The shapes of x and y must be broadcastable. The definition is as follows, for
    details, please refer to the `Introduction to Tensor <../../guides/beginner/tensor_en.html#chapter5-broadcasting-of-tensor>`_:

    - Each input has at least one dimension.
    - Match the two input dimensions from back to front, the dimension sizes must either be equal, one of them is 1, or one of them does not exist.

    Where, z = x - y, the shapes of x and y are broadcastable, then the shape of z can be
    obtained as follows:

    1. If the number of dimensions of x and y are not equal, prepend 1 to the dimensions of the
    tensor with fewer dimensions.

    For example, The shape of x is [8, 1, 6, 1], the shape of y is [7, 1, 5], prepend 1 to the
    dimension of y.

    x (4-D Tensor):  8 x 1 x 6 x 1

    y (4-D Tensor):  1 x 7 x 1 x 5

    2. Determine the size of each dimension of the output z: choose the maximum value from the
    two input dimensions.

    z (4-D Tensor):  8 x 7 x 6 x 5

    If the number of dimensions of the two inputs are the same, the size of the output can be
    directly determined in step 2. When p takes different values, the norm formula is as follows:

    When p = 0, defining $0^0=0$, the zero-norm of z is simply the number of non-zero elements of z.

    .. math::

        ||z||_{0}=\lim_{p \\rightarrow 0}\sum_{i=1}^{m}|z_i|^{p}

    When p = inf, the inf-norm of z is the maximum element of the absolute value of z.

    .. math::

        ||z||_\infty=\max_i |z_i|

    When p = -inf, the negative-inf-norm of z is the minimum element of the absolute value of z.

    .. math::

        ||z||_{-\infty}=\min_i |z_i|

    Otherwise, the p-norm of z follows the formula,

    .. math::

        ||z||_{p}=(\sum_{i=1}^{m}|z_i|^p)^{\\frac{1}{p}}

    Args:
        x (Tensor): 1-D to 6-D Tensor, its data type is bfloat16, float16, float32 or float64. Alias: ``input``.
        y (Tensor): 1-D to 6-D Tensor, its data type is bfloat16, float16, float32 or float64. Alias: ``other``.
        p (float, optional): The norm to be computed, its data type is float32 or float64. Default: 2.
        name (str|None, optional): The default value is `None`. Normally there is no need for
            user to set this property. For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor: Tensor that is the p-norm of (x - y).

    Examples:
        .. code-block:: pycon

            >>> import paddle

            >>> x = paddle.to_tensor([[3, 3], [3, 3]], dtype="float32")
            >>> y = paddle.to_tensor([[3, 3], [3, 1]], dtype="float32")
            >>> out = paddle.dist(x, y, 0)
            >>> print(out)
            Tensor(shape=[], dtype=float32, place=Place(cpu), stop_gradient=True,
            1.)

            >>> out = paddle.dist(x, y, 2)
            >>> print(out)
            Tensor(shape=[], dtype=float32, place=Place(cpu), stop_gradient=True,
            2.)

            >>> out = paddle.dist(x, y, float("inf"))
            >>> print(out)
            Tensor(shape=[], dtype=float32, place=Place(cpu), stop_gradient=True,
            2.)

            >>> out = paddle.dist(x, y, float("-inf"))
            >>> print(out)
            Tensor(shape=[], dtype=float32, place=Place(cpu), stop_gradient=True,
            0.)
    """
    ...


@add_doc_and_signature
def flip(
    x: Tensor, axis: Sequence[int] | int, name: str | None = None
) -> Tensor:
    r"""
    Reverse the order of a n-D tensor along given axis in axis.

    The image below illustrates how ``flip`` works.

    .. image:: https://githubraw.cdn.bcebos.com/PaddlePaddle/docs/develop/docs/images/api_legend/flip.png
        :width: 500
        :alt: legend of flip API
        :align: center

    Args:
        x (Tensor): A Tensor with shape :math:`[N_1, N_2,..., N_k]` . The data type of the input Tensor x
            should be float32, float64, int32, int64, bool. Alias: ``input``.
        axis (list|tuple|int): The axis(axes) to flip on. Negative indices for indexing from the end are accepted. Alias: ``dims``.
        name (str|None, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor, Tensor or DenseTensor calculated by flip layer. The data type is same with input x.

    Examples:
        .. code-block:: pycon

            >>> # doctest: +SKIP("This has diff in xdoctest env")
            >>> import paddle

            >>> image_shape = (3, 2, 2)
            >>> img = paddle.arange(image_shape[0] * image_shape[1] * image_shape[2]).reshape(image_shape)
            >>> tmp = paddle.flip(img, [0, 1])
            >>> print(tmp)
            Tensor(shape=[3, 2, 2], dtype=int64, place=Place(cpu), stop_gradient=True,
            [[[10, 11],
              [8 , 9 ]],
             [[6 , 7 ],
              [4 , 5 ]],
             [[2 , 3 ],
              [0 , 1 ]]])

            >>> out = paddle.flip(tmp, -1)
            >>> print(out)
            Tensor(shape=[3, 2, 2], dtype=int64, place=Place(cpu), stop_gradient=True,
            [[[11, 10],
              [9 , 8 ]],
             [[7 , 6 ],
              [5 , 4 ]],
             [[3 , 2 ],
              [1 , 0 ]]])
    """
    ...


@add_doc_and_signature
def renorm(x: Tensor, p: float, axis: int, max_norm: float) -> Tensor:
    r"""
    This operator is used to calculate the p-norm along the axis,
    suppose the input-shape on axis dimension has the value of T, then
    the tensor is split into T parts, the p-norm should be calculated for each
    part, if the p-norm for part i is larger than max-norm, then each element
    in part i should be re-normalized at the same scale so that part-i' p-norm equals
    max-norm exactly, otherwise part-i stays unchanged.

    Args:
        x (Tensor): The input Tensor. Alias: ``input``.
        p (float): The power of the norm operation.
        axis (int): the dimension to slice the tensor. Alias: ``dim``.
        max_norm (float): the maximal norm limit. Alias: ``maxnorm``.

    Returns:
        Tensor: the renorm Tensor.

    Examples:
        .. code-block:: pycon

            >>> import paddle
            >>> input = [
            ...     [[2.0, 2.0, -2.0], [3.0, 0.3, 3.0]],
            ...     [[2.0, -8.0, 2.0], [3.1, 3.7, 3.0]],
            ... ]
            >>> x = paddle.to_tensor(input, dtype='float32')
            >>> y = paddle.renorm(x, 1.0, 2, 2.05)
            >>> print(y)
            Tensor(shape=[2, 2, 3], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[[ 0.40594056,  0.29285714, -0.41000000],
              [ 0.60891086,  0.04392857,  0.61500001]],
             [[ 0.40594056, -1.17142856,  0.41000000],
              [ 0.62920785,  0.54178572,  0.61500001]]])
    """
    ...


@add_doc_and_signature
def renorm_(x: Tensor, p: float, axis: int, max_norm: float) -> Tensor:
    r"""
    Inplace version of ``renorm`` API, the output Tensor will be inplaced with input ``x``.
    Please refer to :ref:`api_paddle_renorm`.
    """
    ...


@add_doc_and_signature
def poisson(x: Tensor, name: str | None = None) -> Tensor:
    r"""
    Returns a tensor filled with random number from a Poisson Distribution.

    .. math::

        out_i \sim Poisson (lambda = x_i)

    Args:
        x (Tensor): A tensor with rate parameter of poisson Distribution. The data type
            should be bfloat16, float16, float32, float64. Alias: ``input``.
        name (str|None, optional): The default value is None. Normally there is no
            need for user to set this property. For more information, please
            refer to :ref:`api_guide_Name`.

    Returns:
        Tensor: A Tensor filled with random number with the same shape and dtype as ``x``.

    Examples:
        .. code-block:: pycon

            >>> import paddle
            >>> paddle.set_device('cpu')
            >>> paddle.seed(100)

            >>> x = paddle.uniform([2, 3], min=1.0, max=5.0)
            >>> out = paddle.poisson(x)
            >>> print(out)
            Tensor(shape=[2, 3], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[2., 5., 0.],
             [5., 1., 3.]])
    """
    ...


@add_doc_and_signature
def kthvalue(
    x: Tensor,
    k: int,
    axis: int | None = None,
    keepdim: bool = False,
    name: str | None = None,
    *,
    out: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    r"""
    Find values and indices of the k-th smallest at the axis.

    Args:
        x (Tensor): A N-D Tensor with type float16, float32, float64, int32, int64. Alias: ``input``.
        k (int): The k for the k-th smallest number to look for along the axis.
        axis (int, optional): Axis to compute indices along. The effective range
            is [-R, R), where R is x.ndim. when axis < 0, it works the same way
            as axis + R. The default is None. And if the axis is None, it will computed as -1 by default. Alias: ``dim``.
        keepdim (bool, optional): Whether to keep the given axis in output. If it is True, the dimensions will be same as input x and with size one in the axis. Otherwise the output dimensions is one fewer than x since the axis is squeezed. Default is False.
        name (str, optional): For details, please refer to :ref:`api_guide_Name`. Generally, no setting is required. Default: None.

    Keyword args:
        out(Tensor, optional): The output tensor.

    Returns:
        tuple(Tensor), return the values and indices. The value data type is the same as the input `x`. The indices data type is int64.

    Examples:
        .. code-block:: pycon

            >>> import paddle

            >>> x = paddle.randn((2, 3, 2))
            >>> print(x)
            >>> # doctest: +SKIP('Different environments yield different output.')
            Tensor(shape=[2, 3, 2], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[[ 0.11855337, -0.30557564],
              [-0.09968963,  0.41220093],
              [ 1.24004936,  1.50014710]],
             [[ 0.08612321, -0.92485696],
              [-0.09276631,  1.15149164],
              [-1.46587241,  1.22873247]]])
            >>> # doctest: -SKIP
            >>> y = paddle.kthvalue(x, 2, 1)
            >>> print(y)
            >>> # doctest: +SKIP('Different environments yield different output.')
            (Tensor(shape=[2, 2], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[ 0.11855337,  0.41220093],
             [-0.09276631,  1.15149164]]), Tensor(shape=[2, 2], dtype=int64, place=Place(cpu), stop_gradient=True,
            [[0, 1],
             [1, 1]]))
            >>> # doctest: -SKIP
    """
    ...


@add_doc_and_signature
def kron(
    x: Tensor,
    y: Tensor,
    name: str | None = None,
    *,
    out: Tensor | None = None,
) -> Tensor:
    r"""
    Compute the Kronecker product of two tensors, a
    composite tensor made of blocks of the second tensor scaled by the
    first.
    Assume that the rank of the two tensors, $X$ and $Y$
    are the same, if necessary prepending the smallest with ones. If the
    shape of $X$ is [$r_0$, $r_1$, ..., $r_N$] and the shape of $Y$ is
    [$s_0$, $s_1$, ..., $s_N$], then the shape of the output tensor is
    [$r_{0}s_{0}$, $r_{1}s_{1}$, ..., $r_{N}s_{N}$]. The elements are
    products of elements from $X$ and $Y$.
    The equation is:
    $$
    output[k_{0}, k_{1}, ..., k_{N}] = X[i_{0}, i_{1}, ..., i_{N}] *
    Y[j_{0}, j_{1}, ..., j_{N}]
    $$
    where
    $$
    k_{t} = i_{t} * s_{t} + j_{t}, t = 0, 1, ..., N
    $$

    Args:
        x (Tensor): the first operand of kron op, data type: bfloat16, float16, float32, float64, int32 or int64. Alias: ``input``.
        y (Tensor): the second operand of kron op, data type: bfloat16, float16, float32, float64, int32 or int64. Its data type should be the same with x. Alias: ``other``.
        name (str|None, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Keyword args:
        out(Tensor, optional): The output tensor.

    Returns:
        Tensor: The output of kron, data type: bfloat16, float16, float32, float64, int32 or int64. Its data is the same with x.

    Examples:
        .. code-block:: pycon

            >>> import paddle
            >>> x = paddle.to_tensor([[1, 2], [3, 4]], dtype='int64')
            >>> y = paddle.to_tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype='int64')
            >>> out = paddle.kron(x, y)
            >>> out
            Tensor(shape=[6, 6], dtype=int64, place=Place(cpu), stop_gradient=True,
            [[1 , 2 , 3 , 2 , 4 , 6 ],
             [4 , 5 , 6 , 8 , 10, 12],
             [7 , 8 , 9 , 14, 16, 18],
             [3 , 6 , 9 , 4 , 8 , 12],
             [12, 15, 18, 16, 20, 24],
             [21, 24, 27, 28, 32, 36]])
    """
    ...


@add_doc_and_signature
def mv(
    x: Tensor,
    vec: Tensor,
    name: str | None = None,
    *,
    out: Tensor | None = None,
) -> Tensor:
    r"""
    Performs a matrix-vector product of the matrix x and the vector vec.

    Args:
        x (Tensor): A tensor with shape :math:`[M, N]` , The data type of the input Tensor x
            should be one of float32, float64. Alias: ``input``.
        vec (Tensor): A tensor with shape :math:`[N]` , The data type of the input Tensor x
            should be one of float32, float64.
        name (str|None, optional): Normally there is no need for user to set this property.
            For more information, please refer to :ref:`api_guide_Name`. Default is None.

    Keyword args:
        out(Tensor, optional): The output tensor.

    Returns:
        Tensor: The tensor which is produced by x and vec.

    Examples:
        .. code-block:: pycon

            >>> # x: [M, N], vec: [N]
            >>> # paddle.mv(x, vec)  # out: [M]

            >>> import paddle

            >>> x = paddle.to_tensor([[2, 1, 3], [3, 0, 1]]).astype("float64")
            >>> vec = paddle.to_tensor([3, 5, 1]).astype("float64")
            >>> out = paddle.mv(x, vec)
            >>> print(out)
            Tensor(shape=[2], dtype=float64, place=Place(cpu), stop_gradient=True,
            [14., 10.])
    """
    ...


@add_doc_and_signature
def remainder_(
    x: Tensor,
    y: Tensor,
    name: str | None = None,
) -> Tensor:
    r"""
    Inplace version of ``remainder`` API, the output Tensor will be inplaced with input ``x``.
    Please refer to :ref:`api_paddle_remainder`.

    Args:
        x (Tensor): the input tensor, it's data type should be bfloat16, float16, float32, float64, int32, int64. Alias: ``input``.
        y (Tensor): the input tensor, it's data type should be bfloat16, float16, float32, float64, int32, int64. Alias: ``other``.
        name (str|None, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor: The output tensor with the same shape and dtype as x.
    """
    ...


@add_doc_and_signature
def mod_(
    x: Tensor,
    y: Tensor,
    name: str | None = None,
) -> Tensor:
    r"""
    Inplace version of ``mod`` API, the output Tensor will be inplaced with input ``x``.
    Please refer to :ref:`api_paddle_mod`.
    """
    ...


@add_doc_and_signature
def floor_mod_(
    x: Tensor,
    y: Tensor,
    name: str | None = None,
) -> Tensor:
    r"""
    Inplace version of ``floor_mod`` API, the output Tensor will be inplaced with input ``x``.
    Please refer to :ref:`api_paddle_floor_mod_`.
    """
    ...


@add_doc_and_signature
def pow_(
    x: Tensor,
    y: float,
    name: str | None = None,
) -> Tensor:
    r"""
    Inplace version of ``pow`` API, the output Tensor will be inplaced with input ``x``.
    Please refer to :ref:`api_paddle_pow`.

    Args:
        x (Tensor): The input Tensor. It can be any dimension. The data type should be bfloat16, float16, float32, float64, int32 or int64.
            alias: ``input``.
        y (float|int): The exponent value. The data type should be float or int.
            alias: ``exponent``.
        name (str|None, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor: The output tensor with the same shape and dtype as x.
    """
    ...


@add_doc_and_signature
def floor_divide_(
    x: Tensor,
    y: Tensor | int,
    name: str | None = None,
) -> Tensor:
    r"""
    Inplace version of ``floor_divide`` API, the output Tensor will be inplaced with input ``x``.
    Please refer to :ref:`api_paddle_floor_divide`.

    Args:
        x (Tensor): The input tensor, it's data type should be bfloat16, float16, float32, float64, int32, int64.
            alias: ``input``.
        y (Tensor|int): The input tensor or scalar. If y is a tensor, its shape should be broadcastable with x.
            alias: ``other``.
        name (str|None, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor: The output tensor with the same shape and dtype as x.
    """
    ...


@add_doc_and_signature
def erf(
    x: Tensor,
    name: str | None = None,
    *,
    out: Tensor | None = None,
) -> Tensor:
    r"""
    The error function.

    For more details, see `Error function <https://en.wikipedia.org/wiki/Error_function>`_.

    .. math::

        out = \frac{2}{\sqrt{\pi}} \int_{0}^{x}e^{- \eta^{2}}d\eta

    Args:
        x (Tensor): The input tensor, it's data type should be float32, float64, uint8, int8, int16, int32, int64. Alias: ``input``.
        name (str|None, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Keyword Arguments:
        out (Tensor, optional): The output tensor that has the computed result.

    Returns:
        Tensor. The output of Erf, dtype: float32 or float64 (integer types are autocasted into float32), shape: same as input.

    Examples:

        .. code-block:: pycon

            >>> import paddle

            >>> x = paddle.to_tensor([-0.4, -0.2, 0.1, 0.3])
            >>> out = paddle.erf(x)
            >>> print(out)
            Tensor(shape=[4], dtype=float32, place=Place(cpu), stop_gradient=True,
            [-0.42839241, -0.22270259,  0.11246292,  0.32862678])
    """
    ...


@add_doc_and_signature
def erf_(
    x: Tensor,
    name: str | None = None,
) -> Tensor:
    r"""
    Inplace version of ``erf`` API, the output Tensor will be inplaced with input ``x``.
    """
    ...


@add_doc_and_signature
def exp_(
    x: Tensor,
    name: str | None = None,
) -> Tensor:
    r"""
    Inplace version of ``exp`` API, the output Tensor will be inplaced with input ``x``.
    """
    ...


@add_doc_and_signature
def sqrt_(
    x: Tensor,
    name: str | None = None,
) -> Tensor:
    r"""
    Inplace version of ``sqrt`` API, the output Tensor will be inplaced with input ``x``.
    """
    ...


@add_doc_and_signature
def rsqrt_(
    x: Tensor,
    name: str | None = None,
) -> Tensor:
    r"""
    Inplace version of ``rsqrt`` API, the output Tensor will be inplaced with input ``x``.
    """
    ...


@add_doc_and_signature
def ceil_(
    x: Tensor,
    name: str | None = None,
) -> Tensor:
    r"""
    Inplace version of ``ceil`` API, the output Tensor will be inplaced with input ``x``.
    """
    ...


@add_doc_and_signature
def floor_(
    x: Tensor,
    name: str | None = None,
) -> Tensor:
    r"""
    Inplace version of ``floor`` API, the output Tensor will be inplaced with input ``x``.
    """
    ...


@add_doc_and_signature
def reciprocal_(
    x: Tensor,
    name: str | None = None,
) -> Tensor:
    r"""
    Inplace version of ``reciprocal`` API, the output Tensor will be inplaced with input ``x``.
    """
    ...


@add_doc_and_signature
def sigmoid_(
    x: Tensor,
    name: str | None = None,
) -> Tensor:
    r"""
    Inplace version of ``sigmoid`` API, the output Tensor will be inplaced with input ``x``.
    """
    ...


@add_doc_and_signature
def sin_(
    x: Tensor,
    name: str | None = None,
) -> Tensor:
    r"""
    Inplace version of ``sin`` API, the output Tensor will be inplaced with input ``x``.
    """
    ...


@add_doc_and_signature
def sinh_(
    x: Tensor,
    name: str | None = None,
) -> Tensor:
    r"""
    Inplace version of ``sinh`` API, the output Tensor will be inplaced with input ``x``.
    """
    ...


@add_doc_and_signature
def asin_(
    x: Tensor,
    name: str | None = None,
) -> Tensor:
    r"""
    Inplace version of ``asin`` API, the output Tensor will be inplaced with input ``x``.
    """
    ...


@add_doc_and_signature
def asinh_(
    x: Tensor,
    name: str | None = None,
) -> Tensor:
    r"""
    Inplace version of ``asinh`` API, the output Tensor will be inplaced with input ``x``.
    """
    ...


@add_doc_and_signature
def cos_(
    x: Tensor,
    name: str | None = None,
) -> Tensor:
    r"""
    Inplace version of ``cos`` API, the output Tensor will be inplaced with input ``x``.
    """
    ...


@add_doc_and_signature
def cosh_(
    x: Tensor,
    name: str | None = None,
) -> Tensor:
    r"""
    Inplace version of ``cosh`` API, the output Tensor will be inplaced with input ``x``.
    """
    ...


@add_doc_and_signature
def acos_(
    x: Tensor,
    name: str | None = None,
) -> Tensor:
    r"""
    Inplace version of ``acos`` API, the output Tensor will be inplaced with input ``x``.
    """
    ...


@add_doc_and_signature
def acosh_(
    x: Tensor,
    name: str | None = None,
) -> Tensor:
    r"""
    Inplace version of ``acosh`` API, the output Tensor will be inplaced with input ``x``.
    """
    ...


@add_doc_and_signature
def tan_(
    x: Tensor,
    name: str | None = None,
) -> Tensor:
    r"""
    Inplace version of ``tan`` API, the output Tensor will be inplaced with input ``x``.
    """
    ...


@add_doc_and_signature
def atan_(
    x: Tensor,
    name: str | None = None,
) -> Tensor:
    r"""
    Inplace version of ``atan`` API, the output Tensor will be inplaced with input ``x``.
    """
    ...


@add_doc_and_signature
def atanh_(
    x: Tensor,
    name: str | None = None,
) -> Tensor:
    r"""
    Inplace version of ``atanh`` API, the output Tensor will be inplaced with input ``x``.
    """
    ...


@add_doc_and_signature
def expm1_(
    x: Tensor,
    name: str | None = None,
) -> Tensor:
    r"""
    Inplace version of ``expm1`` API, the output Tensor will be inplaced with input ``x``.
    """
    ...


@add_doc_and_signature
def square_(
    x: Tensor,
    name: str | None = None,
) -> Tensor:
    r"""
    Inplace version of ``square`` API, the output Tensor will be inplaced with input ``x``.
    """
    ...
