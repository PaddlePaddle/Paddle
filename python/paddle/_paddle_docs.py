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

import inspect

import paddle

from .base.dygraph.generated_tensor_methods_patch import (
    funcs_map,
    methods_map,
    nn_funcs_map,
)

# Add docstr for some C++ functions in paddle
_add_docstr = paddle.base.core.eager._add_docstr
_code_template = R"""
from __future__ import annotations

{}:
    ...

"""


def _parse_function_signature(func_name: str, code: str) -> inspect.Signature:
    code = _code_template.format(code.strip())
    code_obj = compile(code, "<string>", "exec")
    globals = {}
    eval(code_obj, globals)
    return inspect.signature(globals[func_name])


# sundong
def add_doc_and_signature(func_name: str, docstr: str, func_def: str) -> None:
    """
    Add docstr for function (paddle.*) and method (paddle.Tensor.*) if method exists
    """
    python_api_sig = _parse_function_signature(func_name, func_def)
    for module in [paddle, paddle.Tensor]:
        if hasattr(module, func_name):
            func = getattr(module, func_name)
            if inspect.isfunction(func):
                func.__doc__ = docstr
            elif inspect.ismethod(func):
                func.__self__.__doc__ = docstr
            elif inspect.isbuiltin(func):
                _add_docstr(func, docstr)
    methods_dict = dict(methods_map)
    funcs_dict = dict(funcs_map)
    nn_funcs_dict = dict(nn_funcs_map)
    all_funcs_dict = methods_dict | funcs_dict | nn_funcs_dict
    if func_name in all_funcs_dict.keys():
        tensor_func = all_funcs_dict[func_name]
        tensor_func.__signature__ = python_api_sig


add_doc_and_signature(
    "acos",
    r"""
    Acos Activation Operator.

    .. math::
        out = cos^{-1}(x)

    Args:
        x (Tensor): Input of Acos operator, an N-D Tensor, with data type float32, float64, float16, bfloat16,
            uint8, int8, int16, int32, int64, complex64 or complex128.
        name (str|None, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor. Output of Acos operator, a Tensor with shape same as input
            (integer types are autocasted into float32).

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> x = paddle.to_tensor([-0.4, -0.2, 0.1, 0.3])
            >>> out = paddle.acos(x)
            >>> print(out)
            Tensor(shape=[4], dtype=float32, place=Place(cpu), stop_gradient=True,
            [1.98231316, 1.77215421, 1.47062886, 1.26610363])
""",
    """
def acos(
    x: Tensor,
    name: str | None = None,
    *,
    out: Tensor | None = None,
    ) -> Tensor
""",
)

add_doc_and_signature(
    "acosh",
    r"""
Acosh Activation Operator.

    .. math::
       out = acosh(x)

    Args:
        x (Tensor): Input of Acosh operator, an N-D Tensor, with data type float32, float64, float16, bfloat16,
            uint8, int8, int16, int32, int64, complex64 or complex128.
        name (str|None, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor. Output of Acosh operator, a Tensor with shape same as input
            (integer types are autocasted into float32).

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> x = paddle.to_tensor([1., 3., 4., 5.])
            >>> out = paddle.acosh(x)
            >>> print(out)
            Tensor(shape=[4], dtype=float32, place=Place(cpu), stop_gradient=True,
            [0.        , 1.76274717, 2.06343699, 2.29243159])
""",
    """
def acosh(
    x: Tensor,
    name: str | None = None,
    *,
    out: Tensor | None = None,
    ) -> Tensor
""",
)

add_doc_and_signature(
    "sinh",
    r"""
    Sinh Activation Operator.

    .. math::
       out = sinh(x)

    Args:
        x (Tensor): Input of Sinh operator, an N-D Tensor, with data type float32, float64, float16, bfloat16,
            uint8, int8, int16, int32, int64, complex64 or complex128.
        name (str|None, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor. Output of Sinh operator, a Tensor with shape same as input
            (integer types are autocasted into float32).

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> x = paddle.to_tensor([-0.4, -0.2, 0.1, 0.3])
            >>> out = paddle.sinh(x)
            >>> print(out)
            Tensor(shape=[4], dtype=float32, place=Place(cpu), stop_gradient=True,
            [-0.41075233, -0.20133601,  0.10016675,  0.30452031])
    """,
    """
def sinh(
    x: Tensor,
    name: str | None = None,
    *,
    out: Tensor | None = None,
    ) -> Tensor
""",
)

add_doc_and_signature(
    "amin",
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
        .. code-block:: python

            >>> import paddle
            >>> # data_x is a Tensor with shape [2, 4] with multiple minimum elements
            >>> # the axis is a int element

            >>> x = paddle.to_tensor([[0.2, 0.1, 0.1, 0.1],
            ...                         [0.1, 0.1, 0.6, 0.7]],
            ...                         dtype='float64', stop_gradient=False)
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
            >>> y = paddle.to_tensor([[[0.2, 0.1], [0.1, 0.1]],
            ...                       [[0.1, 0.1], [0.6, 0.7]]],
            ...                       dtype='float64', stop_gradient=False)
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
""",
    """
def amin(
    x: Tensor,
    axis: int | Sequence[int] | None = None,
    keepdim: bool = False,
    name: str | None = None,
    *,
    out: Tensor | None = None,
) -> Tensor
""",
)

add_doc_and_signature(
    "amax",
    """
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
        .. code-block:: python

            >>> import paddle
            >>> # data_x is a Tensor with shape [2, 4] with multiple maximum elements
            >>> # the axis is a int element

            >>> x = paddle.to_tensor([[0.1, 0.9, 0.9, 0.9],
            ...                         [0.9, 0.9, 0.6, 0.7]],
            ...                         dtype='float64', stop_gradient=False)
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
            >>> y = paddle.to_tensor([[[0.1, 0.9], [0.9, 0.9]],
            ...                         [[0.9, 0.9], [0.6, 0.7]]],
            ...                         dtype='float64', stop_gradient=False)
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
""",
    """
def amax(
    x: Tensor,
    axis: int | Sequence[int] | None = None,
    keepdim: bool = False,
    name: str | None = None,
    *,
    out: Tensor | None = None,
) -> Tensor
""",
)

add_doc_and_signature(
    "all",
    """
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
        .. code-block:: python

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
""",
    """
def all(
    x: Tensor,
    axis: int | Sequence[int] | None = None,
    keepdim: bool = False,
    name: str | None = None,
    *,
    out: Tensor | None = None,
) -> Tensor
""",
)
add_doc_and_signature(
    "argmax",
    """
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
        .. code-block:: python

            >>> import paddle

            >>> x = paddle.to_tensor([[5,8,9,5],
            ...                       [0,0,1,7],
            ...                       [6,9,2,4]])
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
    """,
    """
    def argmax(
    x: Tensor,
    axis: int | None = None,
    keepdim: bool = False,
    dtype: DTypeLike = "int64",
    name: str | None = None,
) -> Tensor
    """,
)
add_doc_and_signature(
    "argmin",
    """
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
        .. code-block:: python

            >>> import paddle

            >>> x =  paddle.to_tensor([[5,8,9,5],
            ...                        [0,0,1,7],
            ...                        [6,9,2,4]])
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
    """,
    """
    def argmin(
    x: Tensor,
    axis: int | None = None,
    keepdim: bool = False,
    dtype: DTypeLike = "int64",
    name: str | None = None,
) -> Tensor
    """,
)

add_doc_and_signature(
    "atanh",
    r"""
    Atanh Activation Operator.

    .. math::
       out = atanh(x)

    Args:
        x (Tensor): Input of Atan operator, an N-D Tensor, with data type float32, float64, float16, bfloat16,
            uint8, int8, int16, int32, int64, complex64 or complex128.
        name (str|None, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor. Output of Atanh operator, a Tensor with shape same as input
            (integer types are autocasted into float32).

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> x = paddle.to_tensor([-0.4, -0.2, 0.1, 0.3])
            >>> out = paddle.atanh(x)
            >>> print(out)
            Tensor(shape=[4], dtype=float32, place=Place(cpu), stop_gradient=True,
            [-0.42364895, -0.20273255,  0.10033534,  0.30951962])
""",
    """
def atanh(
    x: Tensor,
    name: str | None = None,
    *,
    out: Tensor | None = None,
    ) -> Tensor
""",
)

add_doc_and_signature(
    "log2",
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

        .. code-block:: python

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
    """,
    "def log2(x: Tensor, name: str | None = None, * , out: Tensor | None = None) -> Tensor",
)
add_doc_and_signature(
    "log10",
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

        .. code-block:: python

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
    """,
    "def log10(x: Tensor, name: str | None = None, * , out: Tensor | None = None) -> Tensor",
)
add_doc_and_signature(
    "log1p",
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

        .. code-block:: python

            >>> import paddle

            >>> data = paddle.to_tensor([[0], [1]], dtype='float32')
            >>> res = paddle.log1p(data)
            >>> res
            Tensor(shape=[2, 1], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[0.        ],
             [0.69314718]])
    """,
    "def log1p(x: Tensor, name: str | None = None, * , out: Tensor | None = None) -> Tensor",
)
add_doc_and_signature(
    "matmul",
    """
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
    """,
    """    def matmul(
    x: Tensor,
    y: Tensor,
    transpose_x: bool = False,
    transpose_y: bool = False,
    name: str | None = None,
    *,
    out: Tensor | None = None,
) -> Tensor""",
)
add_doc_and_signature(
    "multiply",
    """
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

        .. code-block:: python

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

    """,
    """def multiply(x: Tensor,
                    y: Tensor,
                    name: str | None = None,
                    *,
                    out: Tensor | None = None) -> Tensor""",
)
add_doc_and_signature(
    "logsumexp",
    r"""
    Calculates the log of the sum of exponentials of ``x`` along ``axis`` .

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

    .. code-block:: python

        >>> import paddle

        >>> x = paddle.to_tensor([[-1.5, 0., 2.], [3., 1.2, -2.4]])
        >>> out1 = paddle.logsumexp(x)
        >>> out1
        Tensor(shape=[], dtype=float32, place=Place(cpu), stop_gradient=True,
        3.46912265)
        >>> out2 = paddle.logsumexp(x, 1)
        >>> out2
        Tensor(shape=[2], dtype=float32, place=Place(cpu), stop_gradient=True,
        [2.15317822, 3.15684605])

    """,
    """
def logsumexp(
    x: Tensor,
    axis: int | Sequence[int] | None = None,
    keepdim: bool = False,
    name: str | None = None,
    *,
    out: Tensor | None = None,
) -> Tensor
    """,
)
add_doc_and_signature(
    "softplus",
    """
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
        .. code-block:: python

            >>> import paddle
            >>> import paddle.nn.functional as F

            >>> x = paddle.to_tensor([-0.4, -0.2, 0.1, 0.3], dtype='float32')
            >>> out = F.softplus(x)
            >>> print(out)
            Tensor(shape=[4], dtype=float32, place=Place(cpu), stop_gradient=True,
            [0.51301527, 0.59813893, 0.74439669, 0.85435522])
    """,
    """
    def softplus(
    x: Tensor, beta: float = 1, threshold: float = 20, name: str | None = None
) -> Tensor
""",
)
add_doc_and_signature(
    "isclose",
    """
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
     """,
    """
def isclose(
     x: Tensor,
     y: Tensor,
     rtol: float = 1e-05,
     atol: float = 1e-08,
     equal_nan: bool = False,
     name: str | None = None,
 ) -> Tensor
""",
)


# zhengsheng
add_doc_and_signature(
    "isfinite",
    """
    Return whether every element of input tensor is finite number or not.

    .. note::
        Alias Support: The parameter name ``input`` can be used as an alias for ``x``.
        For example, ``isfinite(input=tensor_x)`` is equivalent to ``isfinite(x=tensor_x)``.

    Args:
        x (Tensor): The input tensor, it's data type should be float16, float32, float64, int32, int64, complex64, complex128.
            alias: ``input``.
        name (str|None, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        `Tensor`, the bool result which shows every element of `x` whether it is finite number or not.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> x = paddle.to_tensor([float('-inf'), -2, 3.6, float('inf'), 0, float('-nan'), float('nan')])
            >>> out = paddle.isfinite(x)
            >>> out
            Tensor(shape=[7], dtype=bool, place=Place(cpu), stop_gradient=True,
            [False, True , True , False, True , False, False])
    """,
    """
def isfinite(
    x: Tensor,
    name: str | None = None,
) -> Tensor
""",
)

add_doc_and_signature(
    "isinf",
    """
    Return whether every element of input tensor is `+/-INF` or not.

    .. note::
        Alias Support: The parameter name ``input`` can be used as an alias for ``x``.
        For example, ``isinf(input=tensor_x)`` is equivalent to ``isinf(x=tensor_x)``.

    Args:
        x (Tensor): The input tensor, it's data type should be float16, float32, float64, uint8, int8, int16, int32, int64, complex64, complex128.
            alias: ``input``.
        name (str|None, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        `Tensor`, the bool result which shows every element of `x` whether it is `+/-INF` or not.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> x = paddle.to_tensor([float('-inf'), -2, 3.6, float('inf'), 0, float('-nan'), float('nan')])
            >>> out = paddle.isinf(x)
            >>> out
            Tensor(shape=[7], dtype=bool, place=Place(cpu), stop_gradient=True,
            [True , False, False, True , False, False, False])
    """,
    """
def isinf(
    x: Tensor,
    name: str | None = None,
) -> Tensor
""",
)

add_doc_and_signature(
    "isnan",
    """
    Return whether every element of input tensor is `NaN` or not.

    .. note::
        Alias Support: The parameter name ``input`` can be used as an alias for ``x``.
        For example, ``isnan(input=tensor_x)`` is equivalent to ``isnan(x=tensor_x)``.

    Args:
        x (Tensor): The input tensor, it's data type should be float16, float32, float64, int32, int64, complex64, complex128.
            alias: ``input``.
        name (str|None, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        `Tensor`, the bool result which shows every element of `x` whether it is `NaN` or not.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> x = paddle.to_tensor([float('-inf'), -2, 3.6, float('inf'), 0, float('-nan'), float('nan')])
            >>> out = paddle.isnan(x)
            >>> out
            Tensor(shape=[7], dtype=bool, place=Place(cpu), stop_gradient=True,
            [False, False, False, False, False, True , True ])
    """,
    """
def isnan(
    x: Tensor,
    name: str | None = None,
) -> Tensor
""",
)

add_doc_and_signature(
    "roll",
    """
    Roll the `x` tensor along the given axis(axes). With specific 'shifts', Elements that
    roll beyond the last position are re-introduced at the first according to 'shifts'.
    If a axis is not specified,
    the tensor will be flattened before rolling and then restored to the original shape.

    .. note::
        Alias Support: The parameter name ``input`` can be used as an alias for ``x``, and the parameter name ``dim`` can be used as an alias for ``axis``.
        For example, ``roll(input=tensor_x, dim=1)`` is equivalent to ``roll(x=tensor_x, axis=1)``.

    Args:
        x (Tensor): The x tensor as input.
            alias: ``input``.
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
        .. code-block:: python

            >>> import paddle
            >>> x = paddle.to_tensor([[1.0, 2.0, 3.0],
            ...                       [4.0, 5.0, 6.0],
            ...                       [7.0, 8.0, 9.0]])
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
    """,
    """
def roll(
    x: Tensor,
    shifts: int | Sequence[int],
    axis: int | Sequence[int] | None = None,
    name: str | None = None,
) -> Tensor
""",
)

add_doc_and_signature(
    "ceil",
    """
    Ceil Operator. Computes ceil of x element-wise.

    .. math::
        out = \\left \\lceil x \\right \\rceil

    Args:
        x (Tensor): Input of Ceil operator, an N-D Tensor, with data type float32, float64, float16, bfloat16,
            uint8, int8, int16, int32, int64.
            alias: ``input``.
        name (str|None, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.
        out (Tensor|None, optional): The output tensor. Default: None.

    Returns:
        Tensor. Output of Ceil operator, a Tensor with shape same as input
            (integer types are autocasted into float32).

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> x = paddle.to_tensor([-0.4, -0.2, 0.1, 0.3])
            >>> out = paddle.ceil(x)
            >>> print(out)
            Tensor(shape=[4], dtype=float32, place=Place(cpu), stop_gradient=True,
            [-0., -0., 1. , 1. ])
    """,
    """
def ceil(
    x: Tensor,
    name: str | None = None,
    *,
    out: Tensor | None = None
) -> Tensor
""",
)

add_doc_and_signature(
    "sum",
    """
    Computes the sum of tensor elements over the given dimension.
     .. note::
        Parameter order support: When passing positional parameters, it is possible to support swapping the positional order of dtype and axis.
        For example, ``sum(x, axis, keepdim, dtype)`` is equivalent to ``sum(x, axis, dtype, keepdim)``.
        Alias Support: The parameter name ``input`` can be used as an alias for ``x`` and the parameter name ``dim`` can be used as an alias for ``axis``.
        For example, ``sum(input=tensor_x, dim=1)`` is equivalent to ``sum(x=tensor_x, axis=1)``.

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
        .. code-block:: python

            >>> import paddle

            >>> # x is a Tensor with following elements:
            >>> #    [[0.2, 0.3, 0.5, 0.9]
            >>> #     [0.1, 0.2, 0.6, 0.7]]
            >>> # Each example is followed by the corresponding output tensor.
            >>> x = paddle.to_tensor([[0.2, 0.3, 0.5, 0.9],
            ...                       [0.1, 0.2, 0.6, 0.7]])
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
            >>> y = paddle.to_tensor([[[1, 2], [3, 4]],
            ...                       [[5, 6], [7, 8]]])
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
            >>> x = paddle.to_tensor([[True, True, True, True],
            ...                       [False, False, False, False]])
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
    """,
    """
def sum(
    x: Tensor,
    axis: int | Sequence[int] | None = None,
    dtype: DTypeLike | None = None,
    keepdim: bool = False,
    name: str | None = None,
    *,
    out: Tensor | None = None,
) -> Tensor
""",
)

add_doc_and_signature(
    "index_put",
    """
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
        .. code-block:: python

            >>> import paddle

            >>> x = paddle.zeros([3, 3])
            >>> value = paddle.ones([3])
            >>> ix1 = paddle.to_tensor([0,1,2])
            >>> ix2 = paddle.to_tensor([1,2,1])
            >>> indices=(ix1,ix2)

            >>> out = paddle.index_put(x,indices,value)
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
    """,
    """
def index_put(
    x: Tensor,
    indices: Sequence[Tensor],
    value: Tensor,
    accumulate: bool = False,
    name: str | None = None,
) -> Tensor
""",
)

add_doc_and_signature(
    "index_put_",
    """
    Inplace version of ``index_put`` API, the output Tensor will be inplaced with input ``x``.
    Please refer to :ref:`api_paddle_index_put`.
    """,
    """
def index_put_(
    x: Tensor,
    indices: Sequence[Tensor],
    value: Tensor,
    accumulate: bool = False,
    name: str | None = None,
) -> Tensor
""",
)

# liuyi
add_doc_and_signature(
    "any",
    """
    Computes the ``logical or`` of tensor elements over the given dimension, and return the result.

    .. note::
        Alias Support: The parameter name ``input`` can be used as an alias for ``x``, and the parameter name ``dim`` can be used as an alias for ``axis``.
        For example, ``any(input=tensor_x, dim=1)`` is equivalent to ``any(x=tensor_x, axis=1)``.

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
        .. code-block:: python

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

    """,
    """
    def any(
        x: Tensor,
        axis: int | Sequence[int] | None = None,
        keepdim: bool = False,
        name: str | None = None,
        *,
        out: Tensor | None = None
    ) -> Tensor
    """,
)
add_doc_and_signature(
    "expand_as",
    """

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
        .. code-block:: python

            >>> import paddle

            >>> data_x = paddle.to_tensor([1, 2, 3], 'int32')
            >>> data_y = paddle.to_tensor([[1, 2, 3], [4, 5, 6]], 'int32')
            >>> out = paddle.expand_as(data_x, data_y)
            >>> print(out)
            Tensor(shape=[2, 3], dtype=int32, place=Place(cpu), stop_gradient=True,
            [[1, 2, 3],
             [1, 2, 3]])
    """,
    """
    def expand_as(x: Tensor, y: Tensor, name: str | None = None) -> Tensor
    """,
)

# shenwei
add_doc_and_signature(
    "grid_sample",
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

    .. note::
        Alias Support: The parameter name ``input`` can be used as an alias for ``x``.

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

        .. code-block:: python

            >>> import paddle
            >>> import paddle.nn.functional as F

            >>> # x shape=[1, 1, 3, 3]
            >>> x = paddle.to_tensor([[[[-0.6,  0.8, -0.5],
            ...                         [-0.5,  0.2,  1.2],
            ...                         [ 1.4,  0.3, -0.2]]]], dtype='float64')
            >>> # grid.shape = [1, 3, 4, 2]
            >>> grid = paddle.to_tensor([[[[ 0.2,  0.3],
            ...                            [-0.4, -0.3],
            ...                            [-0.9,  0.3],
            ...                            [-0.9, -0.6]],
            ...                           [[ 0.4,  0.1],
            ...                            [ 0.9, -0.8],
            ...                            [ 0.4,  0.5],
            ...                            [ 0.5, -0.2]],
            ...                           [[ 0.1, -0.8],
            ...                            [-0.3, -1. ],
            ...                            [ 0.7,  0.4],
            ...                            [ 0.2,  0.8]]]], dtype='float64')
            >>> y_t = F.grid_sample(
            ...     x,
            ...     grid,
            ...     mode='bilinear',
            ...     padding_mode='border',
            ...     align_corners=True
            ... )
            >>> print(y_t)
            Tensor(shape=[1, 1, 3, 4], dtype=float64, place=Place(cpu), stop_gradient=True,
            [[[[ 0.34000000,  0.01600000,  0.08600000, -0.44800000],
               [ 0.55000000, -0.07600000,  0.35000000,  0.59000000],
               [ 0.59600000,  0.38000000,  0.52000000,  0.24000000]]]])
    """,
    """
    def grid_sample(
        x: Tensor,
        grid: Tensor,
        mode: str = 'bilinear',
        padding_mode: Literal["zeros", "reflection", "border"] = 'zeros',
        align_corners: bool = True,
        name: str | None = None,
    ) -> Tensor
    """,
)

add_doc_and_signature(
    "gelu",
    """
    gelu activation.

    The activation function of Gelu is calculated element by element. More information refers to :ref: `Gaussian Error Linear Units`.

    approximate parameter must be True, False, "tanh", "none".

    if approximate is True or "tanh"

    .. math::

        gelu(x) = 0.5 * x * (1 + tanh(\\sqrt{\frac{2}{\\pi}} * (x + 0.044715x^{3})))

    else

    .. math::

        gelu(x) = 0.5 * x * (1 + erf(\frac{x}{\\sqrt{2}}))

     .. note::
        Alias Support: The parameter name ``input`` can be used as an alias for ``x``.
        For example, ``gelu(input=tensor_x)`` is equivalent to ``gelu(x=tensor_x)``.

    Parameters:
        x (Tensor): The input Tensor with data type float32, float64.
            alias: ``input``.
        approximate (str|bool, optional): Whether to enable approximation. Default is False.
        name (str|None, optional): For details, please refer to :ref:`api_guide_Name`. Generally, no setting is required. Default: None.

    Returns:
        A Tensor with the same data type and shape as ``x`` .

    Examples:
        .. code-block:: python

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
    """,
    """
    def gelu(
        x: Tensor,
        approximate: Literal["tanh", "none"] | bool = False,
        name: str | None = None,
    ) -> Tensor
    """,
)

add_doc_and_signature(
    "sigmoid",
    r"""
    Sigmoid Activation.

    .. math::
       out = \\frac{1}{1 + e^{-x}}

    .. note::
        Alias Support: The parameter name ``input`` can be used as an alias for ``x``.
        For example, ``sigmoid(input=tensor_x)`` is equivalent to ``sigmoid(x=tensor_x)``.

    Args:
        x (Tensor): Input of Sigmoid operator, an N-D Tensor, with data type bfloat16, float16, float32, float64,
            uint8, int8, int16, int32, int64, complex64 or complex128.
        name (str|None, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.
    Keyword Args:
        out (Tensor|optional): The output tensor.

    Returns:
        Tensor. Output of Sigmoid operator, a Tensor with shape same as input
            (integer types are autocasted into float32).

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> import paddle.nn.functional as F

            >>> x = paddle.to_tensor([-0.4, -0.2, 0.1, 0.3])
            >>> out = F.sigmoid(x)
            >>> print(out)
            Tensor(shape=[4], dtype=float32, place=Place(cpu), stop_gradient=True,
            [0.40131235, 0.45016602, 0.52497917, 0.57444251])
    """,
    """
    def sigmoid(
        x: paddle.Tensor,
        name: str | None = None,
        *,
        out: Tensor | None = None,
    ) -> paddle.Tensor
    """,
)

# zhouxin
add_doc_and_signature(
    "greater_than",
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
        .. code-block:: python

            >>> import paddle

            >>> x = paddle.to_tensor([1, 2, 3])
            >>> y = paddle.to_tensor([1, 3, 2])
            >>> result1 = paddle.greater_than(x, y)
            >>> print(result1)
            Tensor(shape=[3], dtype=bool, place=Place(cpu), stop_gradient=True,
            [False, False, True ])
    """,
    """
    def greater_than(
    x: Tensor, y: Tensor, name: str | None = None, *, out: Tensor | None = None
    ) -> Tensor
    """,
)

add_doc_and_signature(
    "sin",
    """
    Sine Activation Operator.

    .. math::
       out = sin(x)

    Args:
        x (Tensor): Input of Sin operator, an N-D Tensor, with data type float32, float64, float16, bfloat16,
            uint8, int8, int16, int32, int64, complex64 or complex128. Alias: ``input``.
        name (str|None, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.
        out (Tensor, optional): The output tensor. If set, the result will be stored in this tensor. Default is None.

    Returns:
        Tensor. Output of Sin operator, a Tensor with shape same as input
            (integer types are autocasted into float32).

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> x = paddle.to_tensor([-0.4, -0.2, 0.1, 0.3])
            >>> out = paddle.sin(x)
            >>> print(out)
            Tensor(shape=[4], dtype=float32, place=Place(cpu), stop_gradient=True,
            [-0.38941833, -0.19866933,  0.09983342,  0.29552022])
    """,
    """
def sin(
    x: Tensor, name: str | None = None, *, out: Tensor | None = None
) -> Tensor
    """,
)

add_doc_and_signature(
    "sign",
    """
    Returns sign of every element in `x`: For real numbers, 1 for positive, -1 for negative and 0 for zero. For complex numbers, the return value is a complex number with unit magnitude. If a complex number element is zero, the result is 0+0j.

    Args:
        x (Tensor): The input tensor. The data type can be uint8, int8, int16, int32, int64, bfloat16, float16, float32, float64, complex64 or complex128. Alias: ``input``.
        name (str|None, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.
        out (Tensor|None, optional): The output tensor. If set, the result will be stored in this tensor. Default is None.

    Returns:
        Tensor: The output sign tensor with identical shape and data type to the input :attr:`x`.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> x = paddle.to_tensor([3.0, 0.0, -2.0, 1.7], dtype='float32')
            >>> out = paddle.sign(x=x)
            >>> out
            Tensor(shape=[4], dtype=float32, place=Place(cpu), stop_gradient=True,
            [ 1.,  0., -1.,  1.])
    """,
    """
def sign(
    x: Tensor, name: str | None = None, *, out: Tensor | None = None
) -> Tensor
    """,
)

add_doc_and_signature(
    "log",
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

        .. code-block:: python

            >>> import paddle

            >>> x = [[2, 3, 4], [7, 8, 9]]
            >>> x = paddle.to_tensor(x, dtype='float32')
            >>> print(paddle.log(x))
            Tensor(shape=[2, 3], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[0.69314718, 1.09861231, 1.38629436],
             [1.94591010, 2.07944155, 2.19722462]])
    """,
    """
def log(
    x: Tensor, name: str | None = None, *, out: Tensor | None = None
) -> Tensor
    """,
)

add_doc_and_signature(
    "rsqrt",
    """
    Rsqrt Activation Operator.

    Please make sure input is legal in case of numeric errors.

    .. math::
       out = \\frac{1}{\\sqrt{x}}

    Args:
        x (Tensor): Input of Rsqrt operator, an N-D Tensor, with data type float32, float64, float16, bfloat16,
            uint8, int8, int16, int32, int64. Alias: ``input``.
        name (str|None, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.
        out (Tensor, optional): The output tensor. If set, the result will be stored in this tensor. Default is None.

    Returns:
        Tensor. Output of Rsqrt operator, a Tensor with shape same as input
            (integer types are autocasted into float32).

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> x = paddle.to_tensor([0.1, 0.2, 0.3, 0.4])
            >>> out = paddle.rsqrt(x)
            >>> print(out)
            Tensor(shape=[4], dtype=float32, place=Place(cpu), stop_gradient=True,
            [3.16227770, 2.23606801, 1.82574177, 1.58113885])
    """,
    """
def rsqrt(
    x: Tensor, name: str | None = None, *, out: Tensor | None = None
) -> Tensor
    """,
)

add_doc_and_signature(
    "cos",
    """
    Cosine Operator. Computes cosine of x element-wise.

    Input range is `(-inf, inf)` and output range is `[-1,1]`.

    .. math::
       out = cos(x)

    Args:
        x (Tensor): Input of Cos operator, an N-D Tensor, with data type float32, float64, float16, bfloat16,
            uint8, int8, int16, int32, int64, complex64, complex128. Alias: ``input``.
        name (str|None, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.
        out (Tensor, optional): The output tensor. If set, the result will be stored in this tensor. Default is None.

    Returns:
        Tensor. Output of Cos operator, a Tensor with shape same as input
            (integer types are autocasted into float32).

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> x = paddle.to_tensor([-0.4, -0.2, 0.1, 0.3])
            >>> out = paddle.cos(x)
            >>> print(out)
            Tensor(shape=[4], dtype=float32, place=Place(cpu), stop_gradient=True,
            [0.92106098, 0.98006660, 0.99500418, 0.95533651])
    """,
    """
def cos(
    x: Tensor, name: str | None = None, *, out: Tensor | None = None
) -> Tensor
    """,
)

add_doc_and_signature(
    "cosh",
    """
    Cosh Activation Operator.

    Input range `(-inf, inf)`, output range `(1, inf)`.

    .. math::
       out = \\frac{exp(x)+exp(-x)}{2}

    Args:
        x (Tensor): Input of Cosh operator, an N-D Tensor, with data type float32, float64, float16, bfloat16,
            uint8, int8, int16, int32, int64, complex64 or complex128.
        name (str|None, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.
        out (Tensor, optional): The output Tensor. If set, the result will be stored in this Tensor. Default: None.

    Returns:
        Tensor. Output of Cosh operator, a Tensor with shape same as input
            (integer types are autocasted into float32).

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> x = paddle.to_tensor([-0.4, -0.2, 0.1, 0.3])
            >>> out = paddle.cosh(x)
            >>> print(out)
            Tensor(shape=[4], dtype=float32, place=Place(cpu), stop_gradient=True,
            [1.08107233, 1.02006674, 1.00500417, 1.04533851])
    """,
    "def cosh(x: Tensor, name: str | None = None, *, out: Tensor | None = None) -> Tensor",
)

add_doc_and_signature(
    "floor",
    """
    Floor Activation Operator. Computes floor of x element-wise.

    .. math::
        out = \\lfloor x \\rfloor

    Args:
        x (Tensor): Input of Floor operator, an N-D Tensor, with data type float32, float64, float16, bfloat16,
            uint8, int8, int16, int32, int64. Alias: ``input``.
        name (str|None, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.
        out (Tensor, optional): The output tensor. If set, the result will be stored in this tensor. Default is None.

    Returns:
        Tensor. Output of Floor operator, a Tensor with shape same as input
            (integer types are autocasted into float32).

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> x = paddle.to_tensor([-0.4, -0.2, 0.1, 0.3])
            >>> out = paddle.floor(x)
            >>> print(out)
            Tensor(shape=[4], dtype=float32, place=Place(cpu), stop_gradient=True,
            [-1., -1.,  0.,  0.])
    """,
    """
def floor(
    x: Tensor, name: str | None = None, *, out: Tensor | None = None
) -> Tensor
    """,
)
# hehongyu
add_doc_and_signature(
    "maximum",
    """
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

        .. code-block:: python

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
    """,
    """
    def maximum(
        x: Tensor,
        y: Tensor,
        name: str | None = None,
        *,
        out: Tensor | None = None,
    ) -> Tensor
    """,
)

add_doc_and_signature(
    "minimum",
    """
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

        .. code-block:: python

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
    """,
    """
    def minimum(
        x: Tensor,
        y: Tensor,
        name: str | None = None,
        *,
        out: Tensor | None = None,
    ) -> Tensor
    """,
)

add_doc_and_signature(
    "sqrt",
    """
    Sqrt Activation Operator.

    .. math::
       out=\\sqrt{x}=x^{1/2}

    Args:
        x (Tensor): Input of Sqrt operator, an N-D Tensor, with data type float32, float64, float16, bfloat16
            uint8, int8, int16, int32, int64.
        name (str|None, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor. Output of Sqrt operator, a Tensor with shape same as input
            (integer types are autocasted into float32).

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> x = paddle.to_tensor([0.1, 0.2, 0.3, 0.4])
            >>> out = paddle.sqrt(x)
            >>> print(out)
            Tensor(shape=[4], dtype=float32, place=Place(cpu), stop_gradient=True,
            [0.31622776, 0.44721359, 0.54772258, 0.63245553])
    """,
    """
def sqrt(
    x: Tensor,
    name: str | None = None,
    *,
    out: Tensor | None = None,
) -> Tensor
    """,
)

# lousiyu

# zhengshijie
add_doc_and_signature(
    "tril",
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
        .. code-block:: python

            >>> import paddle

            >>> data = paddle.arange(1, 13, dtype="int64").reshape([3,-1])
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
    """,
    """
def tril(
    x: Tensor,
    diagonal: int = 0,
    name: str | None = None,
    *,
    out: Tensor | None = None,
) -> Tensor
""",
)


add_doc_and_signature(
    "triu",
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
        .. code-block:: python

            >>> import paddle

            >>> x = paddle.arange(1, 13, dtype="int64").reshape([3,-1])
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

    """,
    """
def triu(
    x: Tensor,
    diagonal: int = 0,
    name: str | None = None,
    *,
    out: Tensor | None = None,
) -> Tensor
""",
)

add_doc_and_signature(
    "bmm",
    """
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
        .. code-block:: python

            >>> import paddle

            >>> # In imperative mode:
            >>> # size x: (2, 2, 3) and y: (2, 3, 2)
            >>> x = paddle.to_tensor([[[1.0, 1.0, 1.0],
            ...                     [2.0, 2.0, 2.0]],
            ...                     [[3.0, 3.0, 3.0],
            ...                     [4.0, 4.0, 4.0]]])
            >>> y = paddle.to_tensor([[[1.0, 1.0],[2.0, 2.0],[3.0, 3.0]],
            ...                     [[4.0, 4.0],[5.0, 5.0],[6.0, 6.0]]])
            >>> out = paddle.bmm(x, y)
            >>> print(out)
            Tensor(shape=[2, 2, 2], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[[6. , 6. ],
              [12., 12.]],
             [[45., 45.],
              [60., 60.]]])

    """,
    """
def bmm(
    x: Tensor,
    y: Tensor,
    name: str | None = None,
    *,
    out: Tensor | None = None,
) -> Tensor
""",
)


# lihaoyang
add_doc_and_signature(
    "logical_and",
    r"""
    Compute element-wise logical AND on ``x`` and ``y``, and return ``out``. ``out`` is N-dim boolean ``Tensor``.
    Each element of ``out`` is calculated by

    .. math::

        out = x \&\& y

    Note:
        ``paddle.logical_and`` supports broadcasting. If you want know more about broadcasting, please refer to `Introduction to Tensor`_ .

        .. _Introduction to Tensor: ../../guides/beginner/tensor_en.html#chapter5-broadcasting-of-tensor

    .. note::
        Alias Support:
        1. The parameter name ``input`` can be used as an alias for ``x``.
        2. The parameter name ``other`` can be used as an alias for ``y``.

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
        .. code-block:: python

            >>> import paddle

            >>> x = paddle.to_tensor([True])
            >>> y = paddle.to_tensor([True, False, True, False])
            >>> res = paddle.logical_and(x, y)
            >>> print(res)
            Tensor(shape=[4], dtype=bool, place=Place(cpu), stop_gradient=True,
            [True , False, True , False])
""",
    """
def logical_and(
    x: Tensor, y: Tensor, name: str | None = None, *, out: Tensor | None = None
) -> Tensor
""",
)

add_doc_and_signature(
    "logical_or",
    """
    ``logical_or`` operator computes element-wise logical OR on ``x`` and ``y``, and returns ``out``. ``out`` is N-dim boolean ``Tensor``.
    Each element of ``out`` is calculated by

    .. math::

        out = x || y

    Note:
        ``paddle.logical_or`` supports broadcasting. If you want know more about broadcasting, please refer to `Introduction to Tensor`_ .

        .. _Introduction to Tensor: ../../guides/beginner/tensor_en.html#chapter5-broadcasting-of-tensor

    .. note::
        Alias Support:
        1. The parameter name ``input`` can be used as an alias for ``x``.
        2. The parameter name ``other`` can be used as an alias for ``y``.

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
        .. code-block:: python

            >>> import paddle

            >>> x = paddle.to_tensor([True, False], dtype="bool").reshape([2, 1])
            >>> y = paddle.to_tensor([True, False, True, False], dtype="bool").reshape([2, 2])
            >>> res = paddle.logical_or(x, y)
            >>> print(res)
            Tensor(shape=[2, 2], dtype=bool, place=Place(cpu), stop_gradient=True,
            [[True , True ],
             [True , False]])
""",
    """
def logical_or(
    x: Tensor, y: Tensor, name: str | None = None, *, out: Tensor | None = None
) -> Tensor
""",
)

add_doc_and_signature(
    "logical_not",
    """
    ``logical_not`` operator computes element-wise logical NOT on ``x``, and returns ``out``. ``out`` is N-dim boolean ``Variable``.
    Each element of ``out`` is calculated by

    .. math::

        out = !x

    Note:
        ``paddle.logical_not`` supports broadcasting. If you want know more about broadcasting, please refer to `Introduction to Tensor`_ .

        .. _Introduction to Tensor: ../../guides/beginner/tensor_en.html#chapter5-broadcasting-of-tensor

    .. note::
        Alias Support:
        1. The parameter name ``input`` can be used as an alias for ``x``.

    Args:
        x(Tensor):  Operand of logical_not operator. Must be a Tensor of type bool, int8, int16, int32, int64, bfloat16, float16, float32, or float64, complex64, complex128.
            Alias: ``input``.
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
""",
    """
def logical_not(
    x: Tensor, name: str | None = None, *, out: Tensor | None = None
) -> Tensor
""",
)

add_doc_and_signature(
    "logical_xor",
    r"""
    ``logical_xor`` operator computes element-wise logical XOR on ``x`` and ``y``, and returns ``out``. ``out`` is N-dim boolean ``Tensor``.
    Each element of ``out`` is calculated by

    .. math::

        out = (x || y) \&\& !(x \&\& y)

    Note:
        ``paddle.logical_xor`` supports broadcasting. If you want know more about broadcasting, please refer to `Introduction to Tensor`_ .

        .. _Introduction to Tensor: ../../guides/beginner/tensor_en.html#chapter5-broadcasting-of-tensor

    .. note::
        Alias Support:
        1. The parameter name ``input`` can be used as an alias for ``x``.
        2. The parameter name ``other`` can be used as an alias for ``y``.

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
        .. code-block:: python

            >>> import paddle

            >>> x = paddle.to_tensor([True, False], dtype="bool").reshape([2, 1])
            >>> y = paddle.to_tensor([True, False, True, False], dtype="bool").reshape([2, 2])
            >>> res = paddle.logical_xor(x, y)
            >>> print(res)
            Tensor(shape=[2, 2], dtype=bool, place=Place(cpu), stop_gradient=True,
            [[False, True ],
             [True , False]])
""",
    """
def logical_xor(
    x: Tensor, y: Tensor, name: str | None = None, *, out: Tensor | None = None
) -> Tensor
""",
)

add_doc_and_signature(
    "dot",
    """
    This operator calculates inner product for vectors.

    Note:
       Support 1-d and 2-d Tensor. When it is 2d, the first dimension of this matrix
       is the batch dimension, which means that the vectors of multiple batches are dotted.

    .. note::
        Alias Support:
        1. The parameter name ``input`` can be used as an alias for ``x``.
        2. The parameter name ``other`` can be used as an alias for ``y``.

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

        .. code-block:: python

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
""",
    """
def dot(
    x: Tensor, y: Tensor, name: str | None = None, *, out: Tensor | None = None
) -> Tensor
""",
)

add_doc_and_signature(
    "tanh",
    r"""

    Tanh Activation Operator.

    .. math::
        out = \frac{e^{x} - e^{-x}}{e^{x} + e^{-x}}

    .. note::
        Alias Support:
        1. The parameter name ``input`` can be used as an alias for ``x``.

    Args:
        x (Tensor): Input of Tanh operator, an N-D Tensor, with data type bfloat16, float32, float64,
            float16, uint8, int8, int16, int32, int64. Alias: ``input``.
        name (str|None, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.
        out (Tensor|None, optional): The output tensor. Default: None.

    Returns:
        Output of Tanh operator, a Tensor with same data type and shape as input
            (integer types are autocasted into float32).

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> x = paddle.to_tensor([-0.4, -0.2, 0.1, 0.3])
            >>> out = paddle.tanh(x)
            >>> out
            Tensor(shape=[4], dtype=float32, place=Place(cpu), stop_gradient=True,
            [-0.37994900, -0.19737528,  0.09966799,  0.29131261])
""",
    """
def tanh(
    x: Tensor, name: str | None = None, *, out: Tensor | None = None
) -> Tensor
""",
)

add_doc_and_signature(
    "exp",
    """

    Computes exp of x element-wise with a natural number `e` as the base.

    .. math::
        out = e^x

    .. note::
        Alias Support:
        1. The parameter name ``input`` can be used as an alias for ``x``.

    Args:
        x (Tensor): Input of Exp operator, an N-D Tensor, with data type int32, int64, bfloat16, float16, float32, float64, complex64 or complex128.
            Alias: ``input``.
        name (str|None, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.
        out (Tensor|None, optional): The output tensor. Default: None.

    Returns:
        Tensor. Output of Exp operator, a Tensor with shape same as input.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> x = paddle.to_tensor([-0.4, -0.2, 0.1, 0.3])
            >>> out = paddle.exp(x)
            >>> print(out)
            Tensor(shape=[4], dtype=float32, place=Place(cpu), stop_gradient=True,
            [0.67032003, 0.81873077, 1.10517097, 1.34985888])
""",
    """
def exp(
    x: Tensor, name: str | None = None, *, out: Tensor | None = None
) -> Tensor
""",
)

add_doc_and_signature(
    "expm1",
    """

    Expm1 Operator. Computes expm1 of x element-wise with a natural number :math:`e` as the base.

    .. math::
        out = e^x - 1

    .. note::
        Alias Support:
        1. The parameter name ``input`` can be used as an alias for ``x``.

    Args:
        x (Tensor): Input of Expm1 operator, an N-D Tensor, with data type int32, int64, bfloat16, float16, float32, float64, complex64 or complex128.
            Alias: ``input``.
        name (str|None, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.
        out (Tensor|None, optional): The output tensor. Default: None.

    Returns:
        Tensor. Output of Expm1 operator, a Tensor with shape same as input.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> x = paddle.to_tensor([-0.4, -0.2, 0.1, 0.3])
            >>> out = paddle.expm1(x)
            >>> print(out)
            Tensor(shape=[4], dtype=float32, place=Place(cpu), stop_gradient=True,
            [-0.32967997, -0.18126924,  0.10517092,  0.34985882])
""",
    """
def expm1(
    x: Tensor, name: str | None = None, *, out: Tensor | None = None
) -> Tensor
""",
)

add_doc_and_signature(
    "diagonal",
    """

    Computes the diagonals of the input tensor x.

    If ``x`` is 2D, returns the diagonal.
    If ``x`` has larger dimensions, diagonals be taken from the 2D planes specified by axis1 and axis2.
    By default, the 2D planes formed by the first and second axis of the input tensor x.

    The argument ``offset`` determines where diagonals are taken from input tensor x:

    - If offset = 0, it is the main diagonal.
    - If offset > 0, it is above the main diagonal.
    - If offset < 0, it is below the main diagonal.

    .. note::
        Alias Support:
        1. The parameter name ``input`` can be used as an alias for ``x``.
        2. The parameter name ``dim1`` can be used as an alias for ``axis1``.
        3. The parameter name ``dim2`` can be used as an alias for ``axis2``.

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
        .. code-block:: python

            >>> import paddle

            >>> paddle.seed(2023)
            >>> x = paddle.rand([2, 2, 3],'float32')
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
""",
    """
def diagonal(
    x: Tensor,
    offset: int = 0,
    axis1: int = 0,
    axis2: int = 1,
    name: str | None = None,
) -> Tensor
""",
)

add_doc_and_signature(
    "round",
    """

    Round the values in the input to the nearest integer value.

    .. code-block:: text

        input:
          x.shape = [4]
          x.data = [1.2, -0.9, 3.4, 0.9]

        output:
          out.shape = [4]
          out.data = [1., -1., 3., 1.]

    .. note::
        Alias Support:
        1. The parameter name ``input`` can be used as an alias for ``x``.

    Args:
        x (Tensor): Input of Round operator, an N-D Tensor, with data type bfloat16, int32, int64, float32, float64, float16, complex64 or complex128.
            Alias: ``input``.
        decimals(int): Rounded decimal place (default: 0).
        name (str|None, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.
        out (Tensor|None, optional): The output tensor. Default: None.

    Returns:
        Tensor. Output of Round operator, a Tensor with shape same as input.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> x = paddle.to_tensor([-0.5, -0.2, 0.6, 1.5])
            >>> out = paddle.round(x)
            >>> print(out)
            Tensor(shape=[4], dtype=float32, place=Place(cpu), stop_gradient=True,
            [-0., -0.,  1.,  2.])
""",
    """
def round(
    x: Tensor, decimals: int = 0, name: str | None = None, *, out: Tensor | None = None,
) -> Tensor
""",
)

add_doc_and_signature(
    "abs",
    """
    Perform elementwise abs for input `x`.

    .. math::

        out = |x|

    .. note::
        Alias Support:
        1. The parameter name ``input`` can be used as an alias for ``x``.

    Args:
        x (Tensor): The input Tensor with data type int32, int64, float16, float32, float64, complex64 and complex128.
            Alias: ``input``.
        name (str|None, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.
        out (Tensor|None, optional): The output tensor. Default: None.

    Returns:
        Tensor.A Tensor with the same data type and shape as :math:`x`.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> x = paddle.to_tensor([-0.4, -0.2, 0.1, 0.3])
            >>> out = paddle.abs(x)
            >>> print(out)
            Tensor(shape=[4], dtype=float32, place=Place(cpu), stop_gradient=True,
            [0.40000001, 0.20000000, 0.10000000, 0.30000001])
""",
    """
def abs(
    x: Tensor, name: str | None = None, *, out: Tensor | None = None
) -> Tensor
""",
)
# lubingxin

# chenhuangrun

# zhanrongrun

# other
add_doc_and_signature(
    "asin",
    f"""
    Arcsine Operator.

    .. math::
        out = sin^{-1}(x)

    Args:
        x (Tensor): Input of Asin operator, an N-D Tensor, with data type float32, float64, float16, bfloat16,
            uint8, int8, int16, int32, int64, complex64 or complex128.
        name (str|None, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor. Same shape and data type as input (integer types are autocasted into float32)

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> x = paddle.to_tensor([-0.4, -0.2, 0.1, 0.3])
            >>> out = paddle.asin(x)
            >>> print(out)
            Tensor(shape=[4], dtype=float32, place=Place(cpu), stop_gradient=True,
            [-0.41151685, -0.20135793,  0.10016742,  0.30469266])
    """,
    """
def asin(
    x: Tensor,
    name: str | None = None
) -> Tensor
""",
)
