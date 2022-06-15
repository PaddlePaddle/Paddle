# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
sparse math functions
"""
from __future__ import print_function

from paddle import _C_ops, in_dynamic_mode, device, int32, int64
from paddle.tensor import cast
from paddle.incubate.sparse import sparse_csr_tensor


def _cast_coo(x, dtype, name=None):
    indices = x.indices()
    values = cast(x.values(), dtype)
    return _C_ops.final_state_sparse_create_sparse_coo_tensor(
        values, indices, x.shape)


def _cast_csr(x, dtype, name=None):
    crows = x.crows()
    cols = x.cols()
    values = cast(x.values(), dtype)
    return sparse_csr_tensor(crows, cols, values, x.shape)


def _cast(x, dtype, name=None):
    if x.is_sparse_coo():
        return _cast_coo(x, dtype, name)
    return _cast_csr(x, dtype, name)


def add(x, y, name=None):
    """
    Add two sparse tensors element-wise. Input x and y's shape should be identical and have same sparse
    type（SparseCooTensor or SparseCsrTensor）.If input is SparseCooTensor, x and y's sparse_dim should be identical.
    The equation is:

    .. math::
        out = x + y

    Args:
        x (Tensor): the input tensor, it's data type should be float32, float64, int32, int64.
        y (Tensor): the input tensor, it's data type should be float32, float64, int32, int64.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor: the result tensor.

    Examples:

    ..  code-block:: python

        import paddle
        from paddle.fluid.framework import _test_eager_guard

        paddle.device.set_device("cpu")

        with _test_eager_guard():
            x = paddle.to_tensor([[0, -1, 0, 2], [0, 0, -3, 0], [4, 5, 0, 0]], 'float32')
            y = paddle.to_tensor([[0, 0, 0, -2], [0, 2, -3, 0], [2, 3, 4, 8]], 'float32')
            sparse_x = x.to_sparse_csr()
            sparse_y = y.to_sparse_csr()
            sparse_z = paddle.incubate.sparse.add(sparse_x, sparse_y)
            print(sparse_z.to_dense())

        # [[ 0., -1.,  0.,  0.],
        # [ 0.,  2., -6.,  0.],
        # [ 6.,  8.,  4.,  8.]]

    """
    assert device.get_device(
    ) == "cpu", "Currently, Sparse add only support CPU device."
    assert in_dynamic_mode(), "Currently, Sparse API only support dynamic mode"
    assert x.is_sparse_csr() == y.is_sparse_csr(
    ), f"Expect sparse tensor type to be same"
    if x.is_sparse_coo() or x.is_sparse_csr():
        return _C_ops.final_state_sparse_add(x, y)
    else:
        raise ValueError(
            "Currently, sparse.add only support the input of SparseCooTensor or SparseCsrTensor"
        )


def subtract(x, y, name=None):
    """
    Subtract two sparse tensors element-wise. Input x and y's shape should be identical and have same sparse
    type（SparseCooTensor or SparseCsrTensor）.If input is SparseCooTensor, x and y's sparse_dim should be identical.
    The equation is:

    .. math::
        out = x - y

    Args:
        x (Tensor): the input tensor, it's data type should be float32, float64, int32, int64.
        y (Tensor): the input tensor, it's data type should be float32, float64, int32, int64.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor: the result tensor.

    Examples:

    ..  code-block:: python

        import paddle
        from paddle.fluid.framework import _test_eager_guard

        paddle.device.set_device("cpu")

        with _test_eager_guard():
            x = paddle.to_tensor([[0, -1, 0, 2], [0, 0, -3, 0], [4, 5, 0, 0]], 'float32')
            y = paddle.to_tensor([[0, 0, 0, -2], [0, 2, -3, 0], [2, 3, 4, 8]], 'float32')
            sparse_x = x.to_sparse_csr()
            sparse_y = y.to_sparse_csr()
            sparse_z = paddle.incubate.sparse.subtract(sparse_x, sparse_y)
            print(sparse_z.to_dense())

        # [[ 0., -1.,  0.,  4.],
        # [ 0., -2.,  0.,  0.],
        # [ 2.,  2., -4., -8.]]

    """
    assert device.get_device(
    ) == "cpu", "Currently, Sparse subtract only support CPU device."
    assert in_dynamic_mode(), "Currently, Sparse API only support dynamic mode"
    assert x.is_sparse_csr() == y.is_sparse_csr(
    ), f"Expect sparse tensor type to be same"
    if x.is_sparse_coo() or x.is_sparse_csr():
        return _C_ops.final_state_sparse_subtract(x, y)
    else:
        raise ValueError(
            "Currently, sparse.subtract only support the input of SparseCooTensor or SparseCsrTensor"
        )


def multiply(x, y, name=None):
    """
    Multiply two sparse tensors element-wise. Input x and y's shape should be identical and have same sparse
    type（SparseCooTensor or SparseCsrTensor）.If input is SparseCooTensor, x and y's sparse_dim should be identical.
    The equation is:

    .. math::
        out = x * y

    Args:
        x (Tensor): the input tensor, it's data type should be float32, float64, int32, int64.
        y (Tensor): the input tensor, it's data type should be float32, float64, int32, int64.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor: the result tensor.

    Examples:

    ..  code-block:: python

        import paddle
        from paddle.fluid.framework import _test_eager_guard

        paddle.device.set_device("cpu")

        with _test_eager_guard():
            x = paddle.to_tensor([[0, -1, 0, 2], [0, 0, -3, 0], [4, 5, 0, 0]], 'float32')
            y = paddle.to_tensor([[0, 0, 0, -2], [0, 2, -3, 0], [2, 3, 4, 8]], 'float32')
            sparse_x = x.to_sparse_csr()
            sparse_y = y.to_sparse_csr()
            sparse_z = paddle.incubate.sparse.multiply(sparse_x, sparse_y)
            print(sparse_z.to_dense())

        # [[ 0.,  0.,  0., -4.],
        # [ 0.,  0.,  9.,  0.],
        # [ 8., 15.,  0.,  0.]]

    """
    assert device.get_device(
    ) == "cpu", "Currently, Sparse multiply only support CPU device."
    assert in_dynamic_mode(), "Currently, Sparse API only support dynamic mode"
    assert x.is_sparse_csr() == y.is_sparse_csr(
    ), f"Expect sparse tensor type to be same"
    if x.is_sparse_coo() or x.is_sparse_csr():
        return _C_ops.final_state_sparse_multiply(x, y)
    else:
        raise ValueError(
            "Currently, sparse.multiply only support the input of SparseCooTensor or SparseCsrTensor"
        )


def divide(x, y, name=None):
    """
    Divide two sparse tensors element-wise. Input x and y's shape should be identical and have same sparse
    type（SparseCooTensor or SparseCsrTensor）.If input is SparseCooTensor, x and y's sparse_dim should be identical.
    The equation is:

    .. math::
        out = x / y

    Args:
        x (Tensor): the input tensor, it's data type should be float32, float64, int32, int64.
        y (Tensor): the input tensor, it's data type should be float32, float64, int32, int64.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor: the result tensor.

    Examples:

    ..  code-block:: python

        import paddle
        from paddle.fluid.framework import _test_eager_guard

        paddle.device.set_device("cpu")

        with _test_eager_guard():
            x = paddle.to_tensor([[0, -1, 0, 2], [0, 0, -3, 0], [4, 5, 0, 0]], 'float32')
            y = paddle.to_tensor([[0, 0, 0, -2], [0, 2, -3, 0], [2, 3, 4, 8]], 'float32')
            sparse_x = x.to_sparse_csr()
            sparse_y = y.to_sparse_csr()
            sparse_z = paddle.incubate.sparse.divide(sparse_x, sparse_y)
            print(sparse_z.to_dense())

        # [[ nan      , -inf.     ,  nan      , -1.       ],
        # [ nan      ,  0.       ,  1.       ,  nan      ],
        # [ 2.       , 1.66666663,  0.       ,  0.       ]]

    """
    assert device.get_device(
    ) == "cpu", "Currently, Sparse divide only support CPU device."
    assert in_dynamic_mode(), "Currently, Sparse API only support dynamic mode"
    assert x.is_sparse_csr() == y.is_sparse_csr(
    ), f"Expect sparse tensor type to be same"

    if x.dtype in [int32, int64]:
        if x.is_sparse_coo() or x.is_sparse_csr():
            cx = _cast(x, 'float32')
            cy = _cast(y, 'float32')
            return _C_ops.final_state_sparse_divide(cx, cy)
        else:
            raise ValueError(
                "Currently, sparse.divide only support the input of SparseCooTensor or SparseCsrTensor"
            )
    else:
        if x.is_sparse_coo() or x.is_sparse_csr():
            return _C_ops.final_state_sparse_divide(x, y)
        else:
            raise ValueError(
                "Currently, sparse.divide only support the input of SparseCooTensor or SparseCsrTensor"
            )
