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
from paddle.base.framework import (
    core,
    dygraph_only,
    in_dygraph_mode,
    in_dynamic_or_pir_mode,
    in_pir_mode,
)
from paddle.base.layer_helper import LayerHelper

if TYPE_CHECKING:
    from paddle import Tensor

__all__ = []

_int_dtype_ = [
    core.VarDesc.VarType.UINT8,
    core.VarDesc.VarType.INT8,
    core.VarDesc.VarType.INT16,
    core.VarDesc.VarType.INT32,
    core.VarDesc.VarType.INT64,
    core.VarDesc.VarType.BOOL,
    core.DataType.UINT8,
    core.DataType.INT8,
    core.DataType.INT16,
    core.DataType.INT32,
    core.DataType.INT64,
    core.DataType.BOOL,
]

_pir_int_dtype_ = {
    core.DataType.UINT8: 1,
    core.DataType.INT8: 1,
    core.DataType.INT16: 2,
    core.DataType.INT32: 4,
    core.DataType.INT64: 8,
    core.DataType.BOOL: 1,
}


def matmul(x: Tensor, y: Tensor, name: str | None = None) -> Tensor:
    """
    Note:
        This API is only supported from ``CUDA 11.0`` .

    Applies matrix multiplication of two Tensors.

    The supported input/output Tensor type are as follows:

    Note:
        x[SparseCsrTensor] @ y[SparseCsrTensor] -> out[SparseCsrTensor]
        x[SparseCsrTensor] @ y[DenseTensor] -> out[DenseTensor]
        x[SparseCooTensor] @ y[SparseCooTensor] -> out[SparseCooTensor]
        x[SparseCooTensor] @ y[DenseTensor] -> out[DenseTensor]

    It supports backward propagation.

    Dimensions `x` and `y` must be >= 2D. Automatic broadcasting of Tensor is not supported.
    the shape of `x` should be `[*, M, K]` , and the shape of `y` should be `[*, K, N]` , where `*`
    is zero or more batch dimensions.

    Args:
        x (SparseTensor): The input tensor. It can be SparseCooTensor/SparseCsrTensor. The data type can be float32 or float64.
        y (SparseTensor|DenseTensor): The input tensor. It can be SparseCooTensor/SparseCsrTensor/DenseTensor. The data type can be float32 or float64.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        SparseTensor|DenseTensor: Determined by `x` and `y` .

    Examples:

        .. code-block:: python

            >>> # doctest: +REQUIRES(env:GPU)
            >>> import paddle
            >>> paddle.device.set_device('gpu')

            >>> # csr @ dense -> dense
            >>> crows = [0, 1, 2, 3]
            >>> cols = [1, 2, 0]
            >>> values = [1., 2., 3.]
            >>> csr = paddle.sparse.sparse_csr_tensor(crows, cols, values, [3, 3])
            >>> print(csr)
            Tensor(shape=[3, 3], dtype=paddle.float32, place=Place(gpu:0), stop_gradient=True,
                   crows=[0, 1, 2, 3],
                   cols=[1, 2, 0],
                   values=[1., 2., 3.])

            >>> dense = paddle.ones([3, 2])
            >>> out = paddle.sparse.matmul(csr, dense)
            >>> print(out)
            Tensor(shape=[3, 2], dtype=float32, place=Place(gpu:0), stop_gradient=True,
                   [[1., 1.],
                    [2., 2.],
                    [3., 3.]])

            >>> # coo @ dense -> dense
            >>> indices = [[0, 1, 2], [1, 2, 0]]
            >>> values = [1., 2., 3.]
            >>> coo = paddle.sparse.sparse_coo_tensor(indices, values, [3, 3])
            >>> print(coo)
            Tensor(shape=[3, 3], dtype=paddle.float32, place=Place(gpu:0), stop_gradient=True,
                   indices=[[0, 1, 2],
                            [1, 2, 0]],
                   values=[1., 2., 3.])

            >>> dense = paddle.ones([3, 2])
            >>> out = paddle.sparse.matmul(coo, dense)
            >>> print(out)
            Tensor(shape=[3, 2], dtype=float32, place=Place(gpu:0), stop_gradient=True,
                   [[1., 1.],
                    [2., 2.],
                    [3., 3.]])
    """
    assert (
        in_dynamic_or_pir_mode()
    ), "Currently, Sparse API only support dynamic mode or pir mode."
    return _C_ops.sparse_matmul(x, y)


def masked_matmul(
    x: Tensor, y: Tensor, mask: Tensor, name: str | None = None
) -> Tensor:
    """
    Note:
        This API is only supported from ``CUDA 11.3`` .

    Applies matrix multiplication of two Dense Tensors.

    The supported input/output Tensor layout are as follows:

    Note:
        x[DenseTensor] @ y[DenseTensor] * mask[SparseCooTensor] -> out[SparseCooTensor]
        x[DenseTensor] @ y[DenseTensor] * mask[SparseCsrTensor] -> out[SparseCsrTensor]

    It supports backward propagation.

    Dimensions `x` and `y` must be  >= 2D. Automatic broadcasting of Tensor is not supported.
    the shape of `x` should be `[*, M, K]` , and the shape of `y` should be `[*, K, N]` , and the shape of `mask` should be `[*, M, N]` ,
    where `*` is zero or more batch dimensions.

    Args:
        x (DenseTensor): The input tensor. It is DenseTensor. The data type can be float32 or float64.
        y (DenseTensor): The input tensor. It is DenseTensor. The data type can be float32 or float64.
        mask (SparseTensor): The mask tensor, which can be SparseCooTensor/SparseCsrTensor. It specify sparse coordinates. The data type can be float32 or float64.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        SparseTensor: SparseCooTensor or SparseCsrTensor, which is same with `mask` .

    Examples:

        .. code-block:: python

            >>> # doctest: +REQUIRES(env:GPU)
            >>> import paddle
            >>> paddle.device.set_device('gpu')
            >>> paddle.seed(100)

            >>> # dense @ dense * csr_mask -> csr
            >>> crows = [0, 2, 3, 5]
            >>> cols = [1, 3, 2, 0, 1]
            >>> values = [1., 2., 3., 4., 5.]
            >>> dense_shape = [3, 4]
            >>> mask = paddle.sparse.sparse_csr_tensor(crows, cols, values, dense_shape)
            >>> print(mask)
            Tensor(shape=[3, 4], dtype=paddle.float32, place=Place(gpu:0), stop_gradient=True,
                  crows=[0, 2, 3, 5],
                  cols=[1, 3, 2, 0, 1],
                  values=[1., 2., 3., 4., 5.])

            >>> x = paddle.rand([3, 5])
            >>> y = paddle.rand([5, 4])

            >>> out = paddle.sparse.masked_matmul(x, y, mask)
            >>> print(out)
            Tensor(shape=[3, 4], dtype=paddle.float32, place=Place(gpu:0), stop_gradient=True,
                   crows=[0, 2, 3, 5],
                   cols=[1, 3, 2, 0, 1],
                   values=[0.98986477, 0.97800624, 1.14591956, 0.68561077, 0.94714981])

    """
    assert (
        in_dynamic_or_pir_mode()
    ), "Currently, Sparse API only support dynamic mode or pir mode."
    return _C_ops.sparse_masked_matmul(x, y, mask)


def mv(x: Tensor, vec: Tensor, name: str | None = None) -> Tensor:
    """
    Note:
        This API is only supported from ``CUDA 11.0`` .

    Applies matrix-vector product of Sparse Matrix 'x' and Dense vector 'vec' .

    The supported input/output Tensor layout are as follows:

    Note:
        x[SparseCsrTensor] @ vec[DenseTensor] -> out[DenseTensor]
        x[SparseCooTensor] @ vec[DenseTensor] -> out[DenseTensor]

    It supports backward propagation.

    The shape of `x` should be `[M, N]` , and the shape of `vec` should be `[N]` ,
    and the shape of `out` will be `[M]` .

    Args:
        x (SparseTensor): The input 2D tensor. It must be SparseCooTensor/SparseCsrTensor. The data type can be float32 or float64.
        vec (DenseTensor): The input 1D tensor. It must be DenseTensor vector. The data type can be float32 or float64.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        DenseTensor: 1D DenseTensor whose dtype is same with input.

    Examples:

        .. code-block:: python

            >>> # doctest: +REQUIRES(env:GPU)
            >>> import paddle
            >>> paddle.device.set_device('gpu')
            >>> paddle.seed(100)

            >>> # csr @ dense -> dense
            >>> crows = [0, 2, 3, 5]
            >>> cols = [1, 3, 2, 0, 1]
            >>> values = [1., 2., 3., 4., 5.]
            >>> dense_shape = [3, 4]
            >>> csr = paddle.sparse.sparse_csr_tensor(crows, cols, values, dense_shape)
            >>> print(csr)
            Tensor(shape=[3, 4], dtype=paddle.float32, place=Place(gpu:0), stop_gradient=True,
                   crows=[0, 2, 3, 5],
                   cols=[1, 3, 2, 0, 1],
                   values=[1., 2., 3., 4., 5.])
            >>> vec = paddle.randn([4])

            >>> out = paddle.sparse.mv(csr, vec)
            >>> print(out)
            Tensor(shape=[3], dtype=float32, place=Place(gpu:0), stop_gradient=True,
                   [-3.85499096, -2.42975140, -1.75087738])

    """
    assert (
        in_dynamic_or_pir_mode()
    ), "Currently, Sparse API only support dynamic mode or pir mode."
    return _C_ops.sparse_mv(x, vec)


def add(x: Tensor, y: Tensor, name: str | None = None) -> Tensor:
    """
    Add two sparse tensors element-wise. Input x and y's shape should be identical and have same sparse
    type（SparseCooTensor or SparseCsrTensor）.If input is SparseCooTensor, x and y's sparse_dim should be identical.
    The equation is:

    .. math::
        out = x + y

    Args:
        x (Tensor): the input tensor, it's data type should be float32, float64, int32, int64, complex64, complex128.
        y (Tensor): the input tensor, it's data type should be float32, float64, int32, int64, complex64, complex128.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor: the result tensor.

    Examples:

    ..  code-block:: python

        >>> import paddle

        >>> paddle.device.set_device("cpu")

        >>> x = paddle.to_tensor([[0, -1, 0, 2], [0, 0, -3, 0], [4, 5, 0, 0]], 'float32')
        >>> y = paddle.to_tensor([[0, 0, 0, -2], [0, 2, -3, 0], [2, 3, 4, 8]], 'float32')
        >>> sparse_x = x.to_sparse_csr()
        >>> sparse_y = y.to_sparse_csr()
        >>> sparse_z = paddle.sparse.add(sparse_x, sparse_y)
        >>> print(sparse_z.to_dense())
        Tensor(shape=[3, 4], dtype=float32, place=Place(cpu), stop_gradient=True,
               [[ 0., -1.,  0.,  0.],
                [ 0.,  2., -6.,  0.],
                [ 6.,  8.,  4.,  8.]])

    """

    if in_dynamic_or_pir_mode():
        return _C_ops.sparse_add(x, y)
    else:
        op_type = 'sparse_add'
        inputs = {'x': x, 'y': y}
        helper = LayerHelper(op_type)
        out = helper.create_sparse_variable_for_type_inference(x.dtype)
        helper.append_op(
            type=op_type, inputs=inputs, outputs={'out': out}, attrs={}
        )
        return out


def subtract(x: Tensor, y: Tensor, name: str | None = None) -> Tensor:
    """
    Subtract two sparse tensors element-wise. Input x and y's shape should be identical and have same sparse
    type（SparseCooTensor or SparseCsrTensor）.If input is SparseCooTensor, x and y's sparse_dim should be identical.
    The equation is:

    .. math::
        out = x - y

    Args:
        x (Tensor): the input tensor, it's data type should be float32, float64, int32, int64, complex64, complex128.
        y (Tensor): the input tensor, it's data type should be float32, float64, int32, int64, complex64, complex128.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor: the result tensor.

    Examples:

        ..  code-block:: python

            >>> import paddle

            >>> paddle.device.set_device("cpu")

            >>> x = paddle.to_tensor([[0, -1, 0, 2], [0, 0, -3, 0], [4, 5, 0, 0]], 'float32')
            >>> y = paddle.to_tensor([[0, 0, 0, -2], [0, 2, -3, 0], [2, 3, 4, 8]], 'float32')
            >>> sparse_x = x.to_sparse_csr()
            >>> sparse_y = y.to_sparse_csr()
            >>> sparse_z = paddle.sparse.subtract(sparse_x, sparse_y)
            >>> print(sparse_z.to_dense())
            Tensor(shape=[3, 4], dtype=float32, place=Place(cpu), stop_gradient=True,
                   [[ 0., -1.,  0.,  4.],
                    [ 0., -2.,  0.,  0.],
                    [ 2.,  2., -4., -8.]])

    """

    if in_dygraph_mode():
        return _C_ops.sparse_subtract(x, y)
    elif in_pir_mode():
        return _C_ops.sparse_subtract(x, y)
    else:
        raise RuntimeError(
            "We currently only support dynamic graph mode or the new IR mode."
        )


def multiply(x: Tensor, y: Tensor, name: str | None = None) -> Tensor:
    """
    Multiply two sparse tensors element-wise. Input x and y's shape should be identical and have same sparse
    type（SparseCooTensor or SparseCsrTensor）.If input is SparseCooTensor, x and y's sparse_dim should be identical.
    The equation is:

    .. math::
        out = x * y

    Args:
        x (Tensor): the input tensor, it's data type should be float32, float64, int32, int64, complex64, complex128.
        y (Tensor): the input tensor, it's data type should be float32, float64, int32, int64, complex64, complex128.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor: the result tensor.

    Examples:

        ..  code-block:: python

            >>> import paddle

            >>> paddle.device.set_device("cpu")

            >>> x = paddle.to_tensor([[0, -1, 0, 2], [0, 0, -3, 0], [4, 5, 0, 0]], 'float32')
            >>> y = paddle.to_tensor([[0, 0, 0, -2], [0, 2, -3, 0], [2, 3, 4, 8]], 'float32')
            >>> sparse_x = x.to_sparse_csr()
            >>> sparse_y = y.to_sparse_csr()
            >>> sparse_z = paddle.sparse.multiply(sparse_x, sparse_y)
            >>> print(sparse_z.to_dense())
            Tensor(shape=[3, 4], dtype=float32, place=Place(cpu), stop_gradient=True,
                   [[ 0., -0.,  0., -4.],
                    [ 0.,  0.,  9.,  0.],
                    [ 8., 15.,  0.,  0.]])

    """

    if isinstance(y, (int, float)):
        return _C_ops.sparse_scale(x, float(y), 0.0, True)
    else:
        if in_dygraph_mode():
            return _C_ops.sparse_multiply(x, y)
        elif in_pir_mode():
            return _C_ops.sparse_multiply(x, y)
        else:
            raise RuntimeError(
                "We currently only support dynamic graph mode or the new IR mode."
            )


def divide(x: Tensor, y: Tensor, name: str | None = None) -> Tensor:
    """
    Divide two sparse tensors element-wise. Input x and y's shape should be identical and have same sparse
    type（SparseCooTensor or SparseCsrTensor）.If input is SparseCooTensor, x and y's sparse_dim should be identical.
    The equation is:

    .. math::
        out = x / y

    Args:
        x (Tensor): the input tensor, it's data type should be float32, float64, int32, int64, complex64, complex128.
        y (Tensor): the input tensor, it's data type should be float32, float64, int32, int64, complex64, complex128.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor: the result tensor.

    Examples:

        ..  code-block:: python

            >>> import paddle

            >>> paddle.device.set_device("cpu")

            >>> x = paddle.to_tensor([[0, -1, 0, 2], [0, 0, -3, 0], [4, 5, 0, 0]], 'float32')
            >>> y = paddle.to_tensor([[0, 0, 0, -2], [0, 2, -3, 0], [2, 3, 4, 8]], 'float32')
            >>> sparse_x = x.to_sparse_csr()
            >>> sparse_y = y.to_sparse_csr()
            >>> sparse_z = paddle.sparse.divide(sparse_x, sparse_y)
            >>> print(sparse_z.to_dense())
            Tensor(shape=[3, 4], dtype=float32, place=Place(cpu), stop_gradient=True,
                   [[ nan      , -inf.     ,  nan      , -1.       ],
                    [ nan      ,  0.       ,  1.       ,  nan      ],
                    [ 2.       , 1.66666663,  0.       ,  0.       ]])

    """

    if isinstance(y, (int, float)):
        return _C_ops.sparse_divide_scalar(x, float(y))
    else:
        if in_dygraph_mode():
            return _C_ops.sparse_divide(x, y)
        elif in_pir_mode():
            return _C_ops.sparse_divide(x, y)
        else:
            raise RuntimeError(
                "We currently only support dynamic graph mode or the new IR mode."
            )


def is_same_shape(x: Tensor, y: Tensor) -> bool:
    """
    Return the results of shape comparison between two Tensors, check whether x.shape equal to y.shape.
    Any two type Tensor among DenseTensor/SparseCooTensor/SparseCsrTensor are supported.

    Args:
        x (Tensor): The input tensor. It can be DenseTensor/SparseCooTensor/SparseCsrTensor.
        y (Tensor): The input tensor. It can be DenseTensor/SparseCooTensor/SparseCsrTensor.

    Returns:
        bool: True for same shape and False for different shape.

    Examples:

        .. code-block:: python

            >>> import paddle

            >>> x = paddle.rand([2, 3, 8])
            >>> y = paddle.rand([2, 3, 8])
            >>> y = y.to_sparse_csr()
            >>> z = paddle.rand([2, 5])

            >>> paddle.sparse.is_same_shape(x, y)
            True
            >>> paddle.sparse.is_same_shape(x, z)
            False

    """
    assert (
        in_dynamic_or_pir_mode()
    ), "Currently, Sparse API only support dynamic mode or pir mode."
    return x.is_same_shape(y)


@dygraph_only
def mask_as(x: Tensor, mask: Tensor, name: str | None = None) -> Tensor:
    r"""
    Filter the input dense tensor `x` using the `indices` of the sparse matrix `mask`,
    which in turn generates a sparse matrix of the corresponding format.
    The input `x` and `mask` must have the same shape, and the sparse tensor returned has the same indices as `mask`
    even `zero` values exist in the corresponding indices.

    Args:
        x (Tensor): The input tensor. It should be a DenseTensor.
            The data type can be float32, float64, int32, int64, complex64, complex128, int8, int16, float16.
        mask (Tensor): The input tensor. It can be SparseCooTensor or SparseCsrTensor.
            It should be 2D or 3D when the mask is SparseCsrTensor.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor: A sparse tensor.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> paddle.set_device('cpu')

            >>> # csr sparse tensor
            >>> crows = [0, 2, 3, 5]
            >>> cols = [1, 3, 2, 0, 1]
            >>> values = [1., 2., 3., 4., 5.]
            >>> dense_shape = [3, 4]
            >>> csr = paddle.sparse.sparse_csr_tensor(crows, cols, values, dense_shape)
            >>> paddle.seed(2024)
            >>> x = paddle.rand(dense_shape).astype(csr.dtype)
            >>> out = paddle.sparse.mask_as(x, csr)
            >>> print(out)
            Tensor(shape=[3, 4], dtype=paddle.float32, place=Place(cpu), stop_gradient=True,
            crows=[0, 2, 3, 5],
            cols=[1, 3, 2, 0, 1],
            values=[0.23659813, 0.08467803, 0.64152628, 0.66596609, 0.90394485])

            >>> # coo sparse tensor
            >>> indices = [[0, 1, 2], [1, 2, 0]]
            >>> values = [1.0, 2.0, 3.0]
            >>> dense_shape = [3, 3]
            >>> coo = paddle.sparse.sparse_coo_tensor(indices, values, dense_shape)
            >>> paddle.seed(2024)
            >>> x = paddle.rand(dense_shape).astype(coo.dtype)
            >>> out = paddle.sparse.mask_as(x, coo)
            >>> print(out)
            Tensor(shape=[3, 3], dtype=paddle.float32, place=Place(cpu), stop_gradient=True,
            indices=[[0, 1, 2],
                     [1, 2, 0]],
            values=[0.23659813, 0.40340215, 0.64152628])

    """
    return _C_ops.sparse_mask_as(x, mask)
