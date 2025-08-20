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

from .base.dygraph.generated_tensor_methods_patch import methods_map

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
    if func_name in methods_dict.keys():
        tensor_func = methods_dict[func_name]
        tensor_func.__signature__ = python_api_sig


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
        name (str|None, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor, results of minimum on the specified axis of input tensor,
        it's data type is the same as input's Tensor.

    Examples:
        .. code-block:: python
            >>> # type: ignore
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
        name (str|None, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor, results of maximum on the specified axis of input tensor,
        it's data type is the same as `x`.

    Examples:
        .. code-block:: python
            >>> # type: ignore
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
            >>> # type: ignore
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
            >>> # type: ignore
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
            >>> # type: ignore
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

# liuyi

# shenwei

# zhouxin

# hehongyu

# lousiyu

# zhengshijie

# lihaoyang

# lubingxin

# chenhuangrun

# zhanrongrun

# other
