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

from paddle import _C_ops
from paddle.fluid.framework import dygraph_only

__all__ = []


@dygraph_only
def tanh(x, name=None):
    """
    sparse tanh activation, requiring x to be a sparse coo or sparse csr tensor.

    .. math::

        out = tanh(x)

    Parameters:
        x (Tensor): The input Sparse Tensor with data type float32, float64.
        name (str, optional): Name for the operation (optional, default is None).
            For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        A Sparse Tensor with the same data type and shape as ``x`` .

    Examples:
        .. code-block:: python

            import paddle
            from paddle.fluid.framework import _test_eager_guard

            with _test_eager_guard():
                dense_x = paddle.to_tensor([-2, 0, 1], dtype='float32')
                sparse_x = dense_x.to_sparse_coo(1)
                out = paddle.incubate.sparse.tanh(sparse_x)
    """
    return _C_ops.sparse_tanh(x)


@dygraph_only
def sqrt(x, name=None):
    """
    Calculate square root of x, requiring x to be a sparse coo or sparse csr tensor.

    .. math::

        out = sqrt(x)

    Parameters:
        x (Tensor): The input Sparse Tensor with data type float32, float64.
        name (str, optional): Name for the operation (optional, default is None).
            For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        A Sparse Tensor with the same data type and shape as ``x`` .

    Examples:
        .. code-block:: python

            import paddle
            from paddle.fluid.framework import _test_eager_guard

            with _test_eager_guard():
                dense_x = paddle.to_tensor([4, 0, 1], dtype='float32')
                sparse_x = dense_x.to_sparse_coo(1)
                out = paddle.incubate.sparse.sqrt(sparse_x)
    """
    return _C_ops.sparse_sqrt(x)


@dygraph_only
def sin(x, name=None):
    """
    Calculate sin of x, requiring x to be a sparse coo or sparse csr tensor.

    .. math::

        out = sin(x)

    Parameters:
        x (Tensor): The input Sparse Tensor with data type float32, float64.
        name (str, optional): Name for the operation (optional, default is None).
            For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        A Sparse Tensor with the same data type and shape as ``x`` .

    Examples:
        .. code-block:: python

            import paddle
            from paddle.fluid.framework import _test_eager_guard

            with _test_eager_guard():
                dense_x = paddle.to_tensor([-2, 0, 3], dtype='float32')
                sparse_x = dense_x.to_sparse_coo(1)
                out = paddle.incubate.sparse.sin(sparse_x)
    """
    return _C_ops.sparse_sin(x)
