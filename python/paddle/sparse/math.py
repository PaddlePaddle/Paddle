# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
"""
math functions
"""
from __future__ import print_function
import numpy as np

from paddle import _C_ops, in_dynamic_mode
from ..framework import core, dygraph_only
from ..tensor import to_tensor
from ..tensor import max
from ..fluid.data_feeder import check_variable_and_dtype, check_type, check_dtype, convert_dtype


def add(x, y, name=None):
    """
    Examples:

    ..  code-block:: python

        import paddle
        x = paddle.to_tensor([[0, -1, 0, 2], [0, 0, -3, 0], [4, 5, 0, 0]], 'float32')
        y = paddle.to_tensor([[0, 0, 0, -2], [0, 2, -3, 0], [2, 3, 4, 8]], 'float64')
        sparse_x = dense_x.to_sparse_csr()
        sparse_y = dense_y.to_sparse_csr()
        sparse_z = paddle.add(sparse_x, sparse_y)
        print(sparse_z.to_dense())

        # [[ 0., -1.,  0.,  0.],
        #  [ 0.,  2., -6.,  0.],
        #  [ 6.,  8.,  4.,  8.]]

    """

    assert in_dynamic_mode(), "Currently, Sparse API only support dynamic mode"
    assert x.is_sparse_csr(
    ), "Currently, sparse.add only support the input of SparseCsrTensor"

    return _C_ops.final_state_sparse_elementwise_add(x, y)
    # return x.elementwise_add(y)


def subtract(x, y, name=None):
    """
    Substract two tensors element-wise. The equation is:

    .. math::
        out = x - y

    **Note**:
    ``paddle.subtract`` supports broadcasting. If you want know more about broadcasting, please refer to :ref:`user_guide_broadcasting` .

    Args:
        x (Tensor): the input tensor, it's data type should be float32, float64, int32, int64.
        y (Tensor): the input tensor, it's data type should be float32, float64, int32, int64.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        N-D Tensor. A location into which the result is stored. If x, y have different shapes and are "broadcastable", the resulting tensor shape is the shape of x and y after broadcasting. If x, y have the same shape,  its shape is the same as x and y.

    Examples:

        .. code-block:: python

            import numpy as np
            import paddle

            x = paddle.to_tensor([[1, 2], [7, 8]])
            y = paddle.to_tensor([[5, 6], [3, 4]])
            res = paddle.subtract(x, y)
            print(res)
            #       [[-4, -4],
            #        [4, 4]]

            x = paddle.to_tensor([[[1, 2, 3], [1, 2, 3]]])
            y = paddle.to_tensor([1, 0, 4])
            res = paddle.subtract(x, y)
            print(res)
            #       [[[ 0,  2, -1],
            #         [ 0,  2, -1]]]

            x = paddle.to_tensor([2, np.nan, 5], dtype='float32')
            y = paddle.to_tensor([1, 4, np.nan], dtype='float32')
            res = paddle.subtract(x, y)
            print(res)
            #       [ 1., nan, nan]

            x = paddle.to_tensor([5, np.inf, -np.inf], dtype='float64')
            y = paddle.to_tensor([1, 4, 5], dtype='float64')
            res = paddle.subtract(x, y)
            print(res)
            #       [   4.,  inf., -inf.]

    """
    assert in_dynamic_mode(), "Currently, Sparse API only support dynamic mode"
    assert x.is_sparse_csr(
    ), "Currently, sparse.add only support the input of SparseCsrTensor"

    return _C_ops.final_state_sparse_elementwise_sub(x, y)


def multiply(x, y, name=None):
    """
    multiply two tensors element-wise. The equation is:

    .. math::
        out = x * y

    **Note**:
    ``paddle.multiply`` supports broadcasting. If you would like to know more about broadcasting, please refer to :ref:`user_guide_broadcasting` .

    Args:
        x (Tensor): the input tensor, its data type should be one of float32, float64, int32, int64, bool.
        y (Tensor): the input tensor, its data type should be one of float32, float64, int32, int64, bool.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        N-D Tensor. A location into which the result is stored. If x, y have different shapes and are "broadcastable", the resulting tensor shape is the shape of x and y after broadcasting. If x, y have the same shape,  its shape is the same as x and y.

    Examples:

        ..  code-block:: python

            import paddle

            x = paddle.to_tensor([[1, 2], [3, 4]])
            y = paddle.to_tensor([[5, 6], [7, 8]])
            res = paddle.multiply(x, y)
            print(res) # [[5, 12], [21, 32]]

            x = paddle.to_tensor([[[1, 2, 3], [1, 2, 3]]])
            y = paddle.to_tensor([2])
            res = paddle.multiply(x, y)
            print(res) # [[[2, 4, 6], [2, 4, 6]]]

    """

    assert in_dynamic_mode(), "Currently, Sparse API only support dynamic mode"
    assert x.is_sparse_csr(
    ), "Currently, sparse.add only support the input of SparseCsrTensor"

    return _C_ops.final_state_sparse_elementwise_mul(x, y)


def divide(x, y, name=None):
    """
    Divide two tensors element-wise. The equation is:

    .. math::
        out = x / y

    **Note**:
    ``paddle.divide`` supports broadcasting. If you want know more about broadcasting, please refer to :ref:`user_guide_broadcasting` .

    Args:
        x (Tensor): the input tensor, it's data type should be float32, float64, int32, int64.
        y (Tensor): the input tensor, it's data type should be float32, float64, int32, int64.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        N-D Tensor. A location into which the result is stored. If x, y have different shapes and are "broadcastable", the resulting tensor shape is the shape of x and y after broadcasting. If x, y have the same shape,  its shape is the same as x and y.

    Examples:

        ..  code-block:: python

            import paddle

            x = paddle.to_tensor([2, 3, 4], dtype='float64')
            y = paddle.to_tensor([1, 5, 2], dtype='float64')
            z = paddle.divide(x, y)
            print(z)  # [2., 0.6, 2.]

    """
    assert in_dynamic_mode(), "Currently, Sparse API only support dynamic mode"
    assert x.is_sparse_csr(
    ), "Currently, sparse.add only support the input of SparseCsrTensor"

    return _C_ops.final_state_sparse_elementwise_div(x, y)