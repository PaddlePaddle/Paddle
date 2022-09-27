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

from paddle import _C_ops, _legacy_C_ops
from paddle.fluid.framework import dygraph_only, core
from paddle import in_dynamic_mode
from paddle.fluid.layer_helper import LayerHelper
from .unary import cast

__all__ = []

_int_dtype_ = [
    core.VarDesc.VarType.UINT8,
    core.VarDesc.VarType.INT8,
    core.VarDesc.VarType.INT16,
    core.VarDesc.VarType.INT32,
    core.VarDesc.VarType.INT64,
    core.VarDesc.VarType.BOOL,
]


@dygraph_only
def matmul(x, y, name=None):
    """
    Note:
        This API is only supported from ``CUDA 11.0`` .

    Applies matrix multiplication of two Tensors.

    The supported input/output Tensor layout are as follows:

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
        x (Tensor): The input tensor. It can be SparseCooTensor/SparseCsrTensor. The data type can be float32 or float64.
        y (Tensor): The input tensor. It can be SparseCooTensor/SparseCsrTensor/DenseTensor. The data type can be float32 or float64.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor: Its layout is determined by that of `x` and `y` .

    Examples:

        .. code-block:: python

            import paddle

            # csr @ dense -> dense
            crows = [0, 1, 2, 3]
            cols = [1, 2, 0]
            values = [1., 2., 3.]
            csr = paddle.incubate.sparse.sparse_csr_tensor(crows, cols, values, [3, 3])
            # Tensor(shape=[3, 3], dtype=paddle.float32, place=Place(gpu:0), stop_gradient=True,
            #        crows=[0, 1, 2, 3],
            #        cols=[1, 2, 0],
            #        values=[1., 2., 3.])
            dense = paddle.ones([3, 2])
            out = paddle.incubate.sparse.matmul(csr, dense)
            # Tensor(shape=[3, 2], dtype=float32, place=Place(gpu:0), stop_gradient=True,
            #        [[1., 1.],
            #         [2., 2.],
            #         [3., 3.]])

            # coo @ dense -> dense
            indices = [[0, 1, 2], [1, 2, 0]]
            values = [1., 2., 3.]
            coo = paddle.incubate.sparse.sparse_coo_tensor(indices, values, [3, 3])
            # Tensor(shape=[3, 3], dtype=paddle.float32, place=Place(gpu:0), stop_gradient=True,
            #        indices=[[0, 1, 2],
            #                 [1, 2, 0]],
            #        values=[1., 2., 3.])
            dense = paddle.ones([3, 2])
            out = paddle.incubate.sparse.matmul(coo, dense)
            # Tensor(shape=[3, 2], dtype=float32, place=Place(gpu:0), stop_gradient=True,
            #        [[1., 1.],
            #         [2., 2.],
            #         [3., 3.]])
    """
    return _C_ops.sparse_matmul(x, y)


@dygraph_only
def masked_matmul(x, y, mask, name=None):
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
        x (Tensor): The input tensor. It is DenseTensor. The data type can be float32 or float64.
        y (Tensor): The input tensor. It is DenseTensor. The data type can be float32 or float64.
        mask (Tensor): The mask tensor, which can be SparseCooTensor/SparseCsrTensor. It specify sparse coordinates. The data type can be float32 or float64.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor: SparseCoo or SparseCsr, which is determined by that of `mask` .

    Examples:

        .. code-block:: python

            import paddle
            paddle.seed(100)

            # dense @ dense * csr_mask -> csr
            crows = [0, 2, 3, 5]
            cols = [1, 3, 2, 0, 1]
            values = [1., 2., 3., 4., 5.]
            dense_shape = [3, 4]
            mask = paddle.incubate.sparse.sparse_csr_tensor(crows, cols, values, dense_shape)
            # Tensor(shape=[3, 4], dtype=paddle.float32, place=Place(gpu:0), stop_gradient=True,
            #       crows=[0, 2, 3, 5],
            #       cols=[1, 3, 2, 0, 1],
            #       values=[1., 2., 3., 4., 5.])

            x = paddle.rand([3, 5])
            y = paddle.rand([5, 4])

            out = paddle.incubate.sparse.masked_matmul(x, y, mask)
            # Tensor(shape=[3, 4], dtype=paddle.float32, place=Place(gpu:0), stop_gradient=True,
            #        crows=[0, 2, 3, 5],
            #        cols=[1, 3, 2, 0, 1],
            #        values=[0.98986477, 0.97800624, 1.14591956, 0.68561077, 0.94714981])

    """
    return _C_ops.sparse_masked_matmul(x, y, mask)


@dygraph_only
def mv(x, vec, name=None):
    """
    Note:
        This API is only supported from ``CUDA 11.0`` .

    Applies matrix-vector product of Sparse Matrix 'x' and Dense vector 'vec' .

    The supported input/output Tensor layout are as follows:

    Note:
        x[SparseCsrTensor] @ y[DenseTensor] -> out[SparseCsrTensor]
        x[SparseCooTensor] @ y[DenseTensor] -> out[SparseCooTensor]

    It supports backward propagation.

    The shape of `x` should be `[M, N]` , and the shape of `y` should be `[N]` ,
    and the shape of `out` will be `[M]` .

    Args:
        x (Tensor): The input 2D tensor. It must be SparseCooTensor/SparseCsrTensor. The data type can be float32 or float64.
        y (Tensor): The input 1D tensor. It must be DenseTensor vector. The data type can be float32 or float64.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor: 1D Tensor.

    Examples:

        .. code-block:: python

            import paddle
            from paddle.fluid.framework import _test_eager_guard
            paddle.seed(100)

            # csr @ dense -> dense
            with _test_eager_guard():
                crows = [0, 2, 3, 5]
                cols = [1, 3, 2, 0, 1]
                values = [1., 2., 3., 4., 5.]
                dense_shape = [3, 4]
                csr = paddle.incubate.sparse.sparse_csr_tensor(crows, cols, values, dense_shape)
                # Tensor(shape=[3, 4], dtype=paddle.float32, place=Place(gpu:0), stop_gradient=True,
                #        crows=[0, 2, 3, 5],
                #        cols=[1, 3, 2, 0, 1],
                #        values=[1., 2., 3., 4., 5.])
                vec = paddle.randn([4])

                out = paddle.incubate.sparse.mv(csr, vec)
                # Tensor(shape=[3], dtype=float32, place=Place(gpu:0), stop_gradient=True,
                #        [-3.85499096, -2.42975140, -1.75087738])

    """
    return _C_ops.sparse_mv(x, vec)


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
    if y.dtype != x.dtype:
        y = cast(y, None, x.dtype)

    if in_dynamic_mode():
        return _C_ops.sparse_add(x, y)
    else:
        op_type = 'sparse_add'
        inputs = {'x': x, 'y': y}
        helper = LayerHelper(op_type)
        out = helper.create_sparse_variable_for_type_inference(x.dtype)
        helper.append_op(type=op_type,
                         inputs=inputs,
                         outputs={'out': out},
                         attrs={})
        return out


@dygraph_only
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
    if y.dtype != x.dtype:
        y = _C_ops.sparse_cast(y, None, x.dtype)
    return _C_ops.sparse_subtract(x, y)


@dygraph_only
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
    if isinstance(y, (int, float)):
        return _C_ops.sparse_scale(x, float(y), 0.0, True)
    else:
        if y.dtype != x.dtype:
            y = _C_ops.sparse_cast(y, None, x.dtype)
        return _C_ops.sparse_multiply(x, y)


@dygraph_only
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
    if x.dtype in _int_dtype_:
        x = _C_ops.sparse_cast(x, None, core.VarDesc.VarType.FP32)

    if isinstance(y, (int, float)):
        return _C_ops.sparse_divide_scalar(x, float(y))
    else:
        if y.dtype != x.dtype:
            y = _C_ops.sparse_cast(y, None, x.dtype)
        return _C_ops.sparse_divide(x, y)


@dygraph_only
def is_same_shape(x, y):
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

            import paddle

            x = paddle.rand([2, 3, 8])
            y = paddle.rand([2, 3, 8])
            y = y.to_sparse_csr()
            z = paddle.rand([2, 5])

            paddle.incubate.sparse.is_same_shape(x, y)
            # True
            paddle.incubate.sparse.is_same_shape(x, z)
            # False

    """
    return x.is_same_shape(y)
