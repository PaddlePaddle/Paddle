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

from paddle import _C_ops
from paddle.base.framework import in_dynamic_or_pir_mode

if TYPE_CHECKING:
    from paddle import Tensor

__all__ = []


def addmm(
    input: Tensor,
    x: Tensor,
    y: Tensor,
    beta: float = 1.0,
    alpha: float = 1.0,
    name: str | None = None,
) -> Tensor:
    """
    Note:
        This API is only supported from ``CUDA 11.0`` .

    Applies matrix multiplication for `x` and `y` , `input` is added to
    the final result. The equation is:

    ..  math::

        out = alpha * x * y + beta * input

    The supported input/output Tensor layout are as follows:

    Note:
        input[SparseCsrTensor] + x[SparseCsrTensor] @ y[SparseCsrTensor] -> out[SparseCsrTensor]
        input[DenseTensor] + x[SparseCsrTensor] @ y[DenseTensor] -> out[DenseTensor]
        input[SparseCooTensor] + x[SparseCooTensor] @ y[SparseCooTensor] -> out[SparseCooTensor]
        input[DenseTensor] + x[SparseCooTensor] @ y[DenseTensor] -> out[DenseTensor]

    It supports backward propagation.

    Dimensions `input` , `x` , `y` must be same and >= 2D. Automatic broadcasting of Tensor is not supported.

    Args:
        input (SparseTensor|DenseTensor): The input tensor. Shape is [*, M, N]. The data type can be float32 or float64.
        x (SparseTensor): The input SparseTensor. Shape is [*, M, K]. The data type can be float32 or float64.
        y (SparseTensor|DenseTensor): The input tensor. Shape is [*, K, N]. The data type can be float32 or float64.
        beta (float, optional): Coefficient of `input` . Default: 1.0
        alpha (float, optional): Coefficient of `x * y` . Default: 1.0
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        SparseTensor|DenseTensor: Tensor type, date type and shape is the same with `input` .

    Examples:

        .. code-block:: python

            >>> # doctest: +REQUIRES(env:GPU)
            >>> import paddle
            >>> paddle.device.set_device('gpu')

            >>> # dense + csr @ dense -> dense
            >>> input = paddle.rand([3, 2])
            >>> crows = [0, 1, 2, 3]
            >>> cols = [1, 2, 0]
            >>> values = [1., 2., 3.]
            >>> x = paddle.sparse.sparse_csr_tensor(crows, cols, values, [3, 3])
            >>> y = paddle.rand([3, 2])
            >>> out = paddle.sparse.addmm(input, x, y, 3.0, 2.0)

            >>> # dense + coo @ dense -> dense
            >>> input = paddle.rand([3, 2])
            >>> indices = [[0, 1, 2], [1, 2, 0]]
            >>> values = [1., 2., 3.]
            >>> x = paddle.sparse.sparse_coo_tensor(indices, values, [3, 3])
            >>> y = paddle.rand([3, 2])
            >>> out = paddle.sparse.addmm(input, x, y, 3.0, 2.0)

    """
    assert (
        in_dynamic_or_pir_mode()
    ), "Currently, Sparse API only support dynamic mode or pir mode."
    return _C_ops.sparse_addmm(input, x, y, beta, alpha)
