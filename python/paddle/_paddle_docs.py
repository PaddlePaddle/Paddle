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

# Add docstr for some C++ functions in paddle
_add_docstr = paddle.base.core.eager._add_docstr


def add_doc_all(method: str, docstr: str) -> None:
    """
    Add docstr for function (paddle.*) and method (paddle.Tensor.*) if method exists
    """
    for module in [paddle, paddle.Tensor]:
        if hasattr(module, method):
            func = getattr(module, method)
            if inspect.isfunction(func):
                func.__doc__ = docstr
            elif inspect.ismethod(func):
                func.__self__.__doc__ = docstr
            elif inspect.isbuiltin(func):
                _add_docstr(func, docstr)


__all__ = ['add_doc_all']
add_doc_all(
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
)

add_doc_all(
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
)

add_doc_all(
    "sqrt",
    """
    Sqrt Activation Operator.
    .. math::
       out=\\sqrt{x}=x^{1/2}
    Args:
        x (Tensor): Input of Sqrt operator, an N-D Tensor, with data type float32, float64, float16, bfloat16
            uint8, int8, int16, int32, int64.
        out (Tensor|None, optional): Output tensor. If provided in dynamic graph, the result will
            be written to this tensor and also returned. The returned tensor and `out` share memory
            and autograd meta. Default: None.
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
)

add_doc_all(
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
        out (Tensor|None, optional): Output tensor. If provided in dynamic graph, the result will
            be written to this tensor and also returned. The returned tensor and `out` share memory
            and autograd meta. Default: None.
        name (str|None, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.
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
)

add_doc_all(
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
        out (Tensor|None, optional): Output tensor. If provided in dynamic graph, the result will
            be written to this tensor and also returned. The returned tensor and `out` share memory
            and autograd meta. Default: None.
        name (str|None, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.
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
)
