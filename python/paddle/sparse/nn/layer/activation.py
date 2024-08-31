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

from __future__ import annotations

from typing import TYPE_CHECKING

from paddle.nn import Layer

from .. import functional as F

if TYPE_CHECKING:
    from paddle import Tensor

__all__ = []


class ReLU(Layer):
    """

    Sparse ReLU Activation, requiring x to be a SparseCooTensor or SparseCsrTensor.

    .. math::

        ReLU(x) = max(x, 0)

    Parameters:
        name (str|None, optional): Name for the operation (optional, default is None).
            For more information, please refer to :ref:`api_guide_Name`.

    Shape:
        - input: Sparse Tensor with any shape.
        - output: Sparse Tensor with the same shape as input.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> dense_x = paddle.to_tensor([-2., 0., 1.])
            >>> sparse_x = dense_x.to_sparse_coo(1)
            >>> relu = paddle.sparse.nn.ReLU()
            >>> out = relu(sparse_x)
            >>> print(out)
            Tensor(shape=[3], dtype=paddle.float32, place=Place(cpu), stop_gradient=True,
                   indices=[[0, 2]],
                   values=[0., 1.])
    """

    def __init__(self, name: str | None = None) -> None:
        super().__init__()
        self._name = name

    def forward(self, x: Tensor) -> Tensor:
        return F.relu(x, self._name)

    def extra_repr(self) -> str:
        name_str = f'name={self._name}' if self._name else ''
        return name_str


class Softmax(Layer):
    r"""

    Sparse Softmax Activation, requiring x to be a SparseCooTensor or SparseCsrTensor.

    Note:
        Only support axis=-1 for SparseCsrTensor, which is faster when read data
        by row (axis=-1).

    Transform x to dense matrix, and :math:`i` is row index, :math:`j` is column index.
    If axis=-1, We have:

    .. math::

        softmax_ij = \frac{\exp(x_ij - max_j(x_ij))}{\sum_j(exp(x_ij - max_j(x_ij))}

    Parameters:
        axis (int, optional): The axis along which to perform softmax calculations. Only support -1 for SparseCsrTensor.
        name (str|None, optional): Name for the operation (optional, default is None).
            For more information, please refer to :ref:`api_guide_Name`.

    Shape:
        - input: SparseCooTensor / SparseCsrTensor with any shape.
        - output: Sparse Tensor with the same shape as input.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> paddle.seed(2022)

            >>> mask = paddle.rand((3, 4)) < 0.7
            >>> x = paddle.rand((3, 4)) * mask.astype('float32')
            >>> print(x)
            Tensor(shape=[3, 4], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[0.88156885, 0.14463395, 0.17831714, 0.43818203],
             [0.07617740, 0.75576496, 0.        , 0.61921930],
             [0.        , 0.        , 0.42460245, 0.03001321]])

            >>> csr = x.to_sparse_csr()
            >>> print(csr)
            Tensor(shape=[3, 4], dtype=paddle.float32, place=Place(cpu), stop_gradient=True,
                   crows=[0, 4, 7, 9],
                   cols=[0, 1, 2, 3, 0, 1, 3, 2, 3],
                   values=[0.88156885, 0.14463395, 0.17831714, 0.43818203, 0.07617740,
                           0.75576496, 0.61921930, 0.42460245, 0.03001321])

            >>> softmax = paddle.sparse.nn.Softmax()
            >>> out = softmax(csr)
            >>> print(out)
            Tensor(shape=[3, 4], dtype=paddle.float32, place=Place(cpu), stop_gradient=True,
                   crows=[0, 4, 7, 9],
                   cols=[0, 1, 2, 3, 0, 1, 3, 2, 3],
                   values=[0.38234913, 0.18298410, 0.18925257, 0.24541418, 0.21302439,
                           0.42031071, 0.36666498, 0.59738696, 0.40261301])

            >>> coo = x.to_sparse_coo(sparse_dim=2)
            >>> print(coo)
            Tensor(shape=[3, 4], dtype=paddle.float32, place=Place(cpu), stop_gradient=True,
                   indices=[[0, 0, 0, 0, 1, 1, 1, 2, 2],
                            [0, 1, 2, 3, 0, 1, 3, 2, 3]],
                   values=[0.88156885, 0.14463395, 0.17831714, 0.43818203, 0.07617740,
                           0.75576496, 0.61921930, 0.42460245, 0.03001321])

            >>> out = softmax(coo)
            >>> print(out)
            Tensor(shape=[3, 4], dtype=paddle.float32, place=Place(cpu), stop_gradient=True,
                   indices=[[0, 0, 0, 0, 1, 1, 1, 2, 2],
                            [0, 1, 2, 3, 0, 1, 3, 2, 3]],
                   values=[0.38234913, 0.18298411, 0.18925257, 0.24541420, 0.21302438,
                           0.42031071, 0.36666498, 0.59738696, 0.40261301])
    """

    def __init__(self, axis: int = -1, name: str | None = None) -> None:
        super().__init__()
        self._axis = axis
        self._name = name

    def forward(self, x: Tensor) -> Tensor:
        return F.softmax(x, self._axis, self._name)

    def extra_repr(self) -> str:
        name_str = f'name={self._name}' if self._name else ''
        return name_str


class ReLU6(Layer):
    """

    Sparse ReLU6 Activation, requiring x to be a SparseCooTensor or SparseCsrTensor.

    .. math::

        ReLU6(x) = min(max(0,x), 6)

    Parameters:
        name (str|None, optional): Name for the operation (optional, default is None).
            For more information, please refer to :ref:`api_guide_Name`.

    Shape:
        - input: Sparse Tensor with any shape.
        - output: Sparse Tensor with the same shape as input.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> dense_x = paddle.to_tensor([-2., 0., 8.])
            >>> sparse_x = dense_x.to_sparse_coo(1)
            >>> relu6 = paddle.sparse.nn.ReLU6()
            >>> out = relu6(sparse_x)
    """

    def __init__(self, name: str | None = None) -> None:
        super().__init__()
        self._name = name

    def forward(self, x: Tensor) -> Tensor:
        return F.relu6(x, self._name)

    def extra_repr(self) -> str:
        name_str = f'name={self._name}' if self._name else ''
        return name_str


class LeakyReLU(Layer):
    r"""

    Sparse Leaky ReLU Activation, requiring x to be a SparseCooTensor or SparseCsrTensor.

    .. math::

        LeakyReLU(x)=
            \left\{
                \begin{array}{rcl}
                    x, & & if \ x >= 0 \\
                    negative\_slope * x, & & otherwise \\
                \end{array}
            \right.

    Parameters:
        negative_slope (float, optional): Slope of the activation function at
            :math:`x < 0` . Default is 0.01.
        name (str|None, optional): Name for the operation (optional, default is None).
            For more information, please refer to :ref:`api_guide_Name`.

    Shape:
        - input: Sparse Tensor with any shape.
        - output: Sparse Tensor with the same shape as input.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> dense_x = paddle.to_tensor([-2., 0., 5.])
            >>> sparse_x = dense_x.to_sparse_coo(1)
            >>> leaky_relu = paddle.sparse.nn.LeakyReLU(0.5)
            >>> out = leaky_relu(sparse_x)

    """

    def __init__(
        self, negative_slope: float = 0.01, name: str | None = None
    ) -> None:
        super().__init__()
        self._negative_slope = negative_slope
        self._name = name

    def forward(self, x: Tensor) -> Tensor:
        return F.leaky_relu(x, self._negative_slope, self._name)

    def extra_repr(self) -> str:
        name_str = f'name={self._name}' if self._name else ''
        return name_str
