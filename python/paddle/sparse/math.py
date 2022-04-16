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

from paddle import _C_ops, in_dynamic_mode


def add(x, y, name=None):
    """
    Add two sparse tensors element-wise. The equation is:

    .. math::
        out = x + y

    Args:
        x (Tensor): the input tensor, it's data type should be float32, float64.
        y (Tensor): the input tensor, it's data type should be float32, float64.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor: the result tensor.

    Examples:

    ..  code-block:: python

        from paddle.fluid.framework import _test_eager_guard

        with _test_eager_guard():
            x = paddle.to_tensor([[0, -1, 0, 2], [0, 0, -3, 0], [4, 5, 0, 0]], 'float32')
            y = paddle.to_tensor([[0, 0, 0, -2], [0, 2, -3, 0], [2, 3, 4, 8]], 'float32')
            sparse_x = x.to_sparse_csr()
            sparse_y = y.to_sparse_csr()
            sparse_z = paddle.sparse.add(sparse_x, sparse_y)
            print(sparse_z.to_dense())

        # [[ 0., -1.,  0.,  0.],
        # [ 0.,  2., -6.,  0.],
        # [ 6.,  8.,  4.,  8.]]

    """

    assert in_dynamic_mode(), "Currently, Sparse API only support dynamic mode"
    # assert x.is_sparse_csr(
    # ), "Currently, sparse.add only support the input of SparseCsrTensor"
    if x.is_sparse_coo() or y.is_sparse_coo():
        if x.is_sparse_coo():
            _x = x.to_sparse_csr()
        else:
            _x = x
        if y.is_sparse_coo():
            _y = y.to_sparse_csr()
        else:
            _y = y
        return _C_ops.final_state_sparse_elementwise_add(_x, _y).to_sparse_coo(x.dim())

    return _C_ops.final_state_sparse_elementwise_add(x, y)


def subtract(x, y, name=None):
    """
    Subtract two sparse tensors element-wise. The equation is:

    .. math::
        out = x - y

    Args:
        x (Tensor): the input tensor, it's data type should be float32, float64.
        y (Tensor): the input tensor, it's data type should be float32, float64.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor: the result tensor.

    Examples:

    ..  code-block:: python

        from paddle.fluid.framework import _test_eager_guard

        with _test_eager_guard():
            x = paddle.to_tensor([[0, -1, 0, 2], [0, 0, -3, 0], [4, 5, 0, 0]], 'float32')
            y = paddle.to_tensor([[0, 0, 0, -2], [0, 2, -3, 0], [2, 3, 4, 8]], 'float32')
            sparse_x = x.to_sparse_csr()
            sparse_y = y.to_sparse_csr()
            sparse_z = paddle.sparse.subtract(sparse_x, sparse_y)
            print(sparse_z.to_dense())

        # [[ 0., -1.,  0.,  4.],
        # [ 0., -2.,  0.,  0.],
        # [ 2.,  2., -4., -8.]]

    """

    assert in_dynamic_mode(), "Currently, Sparse API only support dynamic mode"
    if x.is_sparse_coo() or y.is_sparse_coo():
        if x.is_sparse_coo():
            _x = x.to_sparse_csr()
        else:
            _x = x
        if y.is_sparse_coo():
            _y = y.to_sparse_csr()
        else:
            _y = y
        return _C_ops.final_state_sparse_elementwise_sub(_x, _y).to_sparse_coo(x.dim())

    return _C_ops.final_state_sparse_elementwise_sub(x, y)


def multiply(x, y, name=None):
    """
    Multiply two sparse tensors element-wise. The equation is:

    .. math::
        out = x * y

    Args:
        x (Tensor): the input tensor, it's data type should be float32, float64.
        y (Tensor): the input tensor, it's data type should be float32, float64.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor: the result tensor.

    Examples:

    ..  code-block:: python

        from paddle.fluid.framework import _test_eager_guard

        with _test_eager_guard():
            x = paddle.to_tensor([[0, -1, 0, 2], [0, 0, -3, 0], [4, 5, 0, 0]], 'float32')
            y = paddle.to_tensor([[0, 0, 0, -2], [0, 2, -3, 0], [2, 3, 4, 8]], 'float32')
            sparse_x = x.to_sparse_csr()
            sparse_y = y.to_sparse_csr()
            sparse_z = paddle.sparse.multiply(sparse_x, sparse_y)
            print(sparse_z.to_dense())

        # [[ 0.,  0.,  0., -4.],
        # [ 0.,  0.,  9.,  0.],
        # [ 8., 15.,  0.,  0.]]

    """

    assert in_dynamic_mode(), "Currently, Sparse API only support dynamic mode"
    if x.is_sparse_coo() or y.is_sparse_coo():
        if x.is_sparse_coo():
            _x = x.to_sparse_csr()
        else:
            _x = x
        if y.is_sparse_coo():
            _y = y.to_sparse_csr()
        else:
            _y = y
        return _C_ops.final_state_sparse_elementwise_mul(_x, _y).to_sparse_coo(x.dim())

    return _C_ops.final_state_sparse_elementwise_mul(x, y)


def divide(x, y, name=None):
    """
    Divide two sparse tensors element-wise. The equation is:

    .. math::
        out = x / y

    Args:
        x (Tensor): the input tensor, it's data type should be float32, float64.
        y (Tensor): the input tensor, it's data type should be float32, float64.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor: the result tensor.

    Examples:

    ..  code-block:: python

        from paddle.fluid.framework import _test_eager_guard

        with _test_eager_guard():
            x = paddle.to_tensor([[0, -1, 0, 2], [0, 0, -3, 0], [4, 5, 0, 0]], 'float32')
            y = paddle.to_tensor([[0, 0, 0, -2], [0, 2, -3, 0], [2, 3, 4, 8]], 'float32')
            sparse_x = x.to_sparse_csr()
            sparse_y = y.to_sparse_csr()
            sparse_z = paddle.sparse.divide(sparse_x, sparse_y)
            print(sparse_z.to_dense())

        # [[ nan      , -inf.     ,  nan      , -1.       ],
        # [ nan      ,  0.       ,  1.       ,  nan      ],
        # [ 2.       , 1.66666663,  0.       ,  0.       ]]

    """

    assert in_dynamic_mode(), "Currently, Sparse API only support dynamic mode"
    if x.is_sparse_coo() or y.is_sparse_coo():
        if x.is_sparse_coo():
            _x = x.to_sparse_csr()
        else:
            _x = x
        if y.is_sparse_coo():
            _y = y.to_sparse_csr()
        else:
            _y = y
        return _C_ops.final_state_sparse_elementwise_div(_x, _y).to_sparse_coo(x.dim())

    return _C_ops.final_state_sparse_elementwise_div(x, y)
