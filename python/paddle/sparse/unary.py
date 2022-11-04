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

import numpy as np

from paddle import _C_ops, _legacy_C_ops
from paddle.fluid.framework import (
    dygraph_only,
    core,
    convert_np_dtype_to_dtype_,
)

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
def sin(x, name=None):
    """
    Calculate elementwise sin of SparseTensor, requiring x to be a SparseCooTensor or SparseCsrTensor.

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

            dense_x = paddle.to_tensor([-2., 0., 1.])
            sparse_x = dense_x.to_sparse_coo(1)
            out = paddle.sparse.sin(sparse_x)

    """
    return _C_ops.sparse_sin(x)


@dygraph_only
def tan(x, name=None):
    """
    Calculate elementwise tan of SparseTensor, requiring x to be a SparseCooTensor or SparseCsrTensor.

    .. math::

        out = tan(x)

    Parameters:
        x (Tensor): The input Sparse Tensor with data type float32, float64.
        name (str, optional): Name for the operation (optional, default is None).
            For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        A Sparse Tensor with the same data type and shape as ``x`` .

    Examples:
        .. code-block:: python

            import paddle

            dense_x = paddle.to_tensor([-2., 0., 1.])
            sparse_x = dense_x.to_sparse_coo(1)
            out = paddle.sparse.tan(sparse_x)

    """
    return _C_ops.sparse_tan(x)


@dygraph_only
def asin(x, name=None):
    """
    Calculate elementwise asin of SparseTensor, requiring x to be a SparseCooTensor or SparseCsrTensor.

    .. math::

        out = asin(x)

    Parameters:
        x (Tensor): The input Sparse Tensor with data type float32, float64.
        name (str, optional): Name for the operation (optional, default is None).
            For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        A Sparse Tensor with the same data type and shape as ``x`` .

    Examples:
        .. code-block:: python

            import paddle

            dense_x = paddle.to_tensor([-2., 0., 1.])
            sparse_x = dense_x.to_sparse_coo(1)
            out = paddle.sparse.asin(sparse_x)

    """
    return _C_ops.sparse_asin(x)


@dygraph_only
def transpose(x, perm, name=None):
    """
    Changes the perm order of ``x`` without changing its data, requiring x to be a SparseCooTensor or SparseCsrTensor.

    .. math::

        out = transpose(x, perm)

    Parameters:
        x (Tensor): The input Sparse Tensor with data type float32, float64.
        perm (list|tuple): Permute the input according to the data of perm.
        name (str, optional): Name for the operation (optional, default is None).
            For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        A transposed Sparse Tensor with the same data type as ``x``.

    Examples:
        .. code-block:: python

            import paddle

            dense_x = paddle.to_tensor([[-2., 0.], [1., 2.]])
            sparse_x = dense_x.to_sparse_coo(1)
            out = paddle.sparse.transpose(sparse_x, [1, 0])

    """
    return _C_ops.sparse_transpose(x, perm)


@dygraph_only
def atan(x, name=None):
    """
    Calculate elementwise atan of SparseTensor, requiring x to be a SparseCooTensor or SparseCsrTensor.

    .. math::

        out = atan(x)

    Parameters:
        x (Tensor): The input Sparse Tensor with data type float32, float64.
        name (str, optional): Name for the operation (optional, default is None).
            For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        A Sparse Tensor with the same data type and shape as ``x`` .

    Examples:
        .. code-block:: python

            import paddle

            dense_x = paddle.to_tensor([-2., 0., 1.])
            sparse_x = dense_x.to_sparse_coo(1)
            out = paddle.sparse.atan(sparse_x)

    """
    return _C_ops.sparse_atan(x)


@dygraph_only
def sinh(x, name=None):
    """
    Calculate elementwise sinh of SparseTensor, requiring x to be a SparseCooTensor or SparseCsrTensor.

    .. math::

        out = sinh(x)

    Parameters:
        x (Tensor): The input Sparse Tensor with data type float32, float64.
        name (str, optional): Name for the operation (optional, default is None).
            For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        A Sparse Tensor with the same data type and shape as ``x`` .

    Examples:
        .. code-block:: python

            import paddle

            dense_x = paddle.to_tensor([-2., 0., 1.])
            sparse_x = dense_x.to_sparse_coo(1)
            out = paddle.sparse.sinh(sparse_x)

    """
    return _C_ops.sparse_sinh(x)


@dygraph_only
def asinh(x, name=None):
    """
    Calculate elementwise asinh of SparseTensor, requiring x to be a SparseCooTensor or SparseCsrTensor.

    .. math::

        out = asinh(x)

    Parameters:
        x (Tensor): The input Sparse Tensor with data type float32, float64.
        name (str, optional): Name for the operation (optional, default is None).
            For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        A Sparse Tensor with the same data type and shape as ``x`` .

    Examples:
        .. code-block:: python

            import paddle

            dense_x = paddle.to_tensor([-2., 0., 1.])
            sparse_x = dense_x.to_sparse_coo(1)
            out = paddle.sparse.asinh(sparse_x)

    """
    return _C_ops.sparse_asinh(x)


@dygraph_only
def atanh(x, name=None):
    """
    Calculate elementwise atanh of SparseTensor, requiring x to be a SparseCooTensor or SparseCsrTensor.

    .. math::

        out = atanh(x)

    Parameters:
        x (Tensor): The input Sparse Tensor with data type float32, float64.
        name (str, optional): Name for the operation (optional, default is None).
            For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        A Sparse Tensor with the same data type and shape as ``x`` .

    Examples:
        .. code-block:: python

            import paddle

            dense_x = paddle.to_tensor([-2., 0., 1.])
            sparse_x = dense_x.to_sparse_coo(1)
            out = paddle.sparse.atanh(sparse_x)

    """
    return _C_ops.sparse_atanh(x)


@dygraph_only
def tanh(x, name=None):
    """
    Calculate elementwise tanh of SparseTensor, requiring x to be a SparseCooTensor or SparseCsrTensor.

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

            dense_x = paddle.to_tensor([-2., 0., 1.])
            sparse_x = dense_x.to_sparse_coo(1)
            out = paddle.sparse.tanh(sparse_x)

    """
    return _C_ops.sparse_tanh(x)


@dygraph_only
def square(x, name=None):
    """
    Calculate elementwise square of SparseTensor, requiring x to be a SparseCooTensor or SparseCsrTensor.

    .. math::

        out = square(x)

    Parameters:
        x (Tensor): The input Sparse Tensor with data type float32, float64.
        name (str, optional): Name for the operation (optional, default is None).
            For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        A Sparse Tensor with the same data type and shape as ``x`` .

    Examples:
        .. code-block:: python

            import paddle

            dense_x = paddle.to_tensor([-2., 0., 1.])
            sparse_x = dense_x.to_sparse_coo(1)
            out = paddle.sparse.square(sparse_x)

    """
    return _C_ops.sparse_square(x)


@dygraph_only
def sqrt(x, name=None):
    """
    Calculate elementwise sqrt of SparseTensor, requiring x to be a SparseCooTensor or SparseCsrTensor.

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

            dense_x = paddle.to_tensor([-2., 0., 1.])
            sparse_x = dense_x.to_sparse_coo(1)
            out = paddle.sparse.sqrt(sparse_x)

    """
    return _C_ops.sparse_sqrt(x)


@dygraph_only
def log1p(x, name=None):
    """
    Calculate the natural log of (1+x), requiring x to be a SparseCooTensor or SparseCsrTensor.

    .. math::

        out = ln(1+x)

    Parameters:
        x (Tensor): The input Sparse Tensor with data type float32, float64.
        name (str, optional): Name for the operation (optional, default is None).
            For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        A Sparse Tensor with the same data type and shape as ``x`` .

    Examples:
        .. code-block:: python

            import paddle

            dense_x = paddle.to_tensor([-2, 0, 1], dtype='float32')
            sparse_x = dense_x.to_sparse_coo(1)
            out = paddle.sparse.log1p(sparse_x)

    """
    return _C_ops.sparse_log1p(x)


@dygraph_only
def cast(x, index_dtype=None, value_dtype=None, name=None):
    """
    cast non-zero-index of SparseTensor to `index_dtype`, non-zero-element of SparseTensor to
    `value_dtype` , requiring x to be a SparseCooTensor or SparseCsrTensor.

    Parameters:
        x (Tensor): The input Sparse Tensor with data type float32, float64.
        index_dtype (np.dtype|str, optional): Data type of the index of SparseCooTensor,
            or crows/cols of SparseCsrTensor. Can be uint8, int8, int16, int32, int64.
        value_dtype (np.dtype|str, optional): Data type of the value of SparseCooTensor,
            SparseCsrTensor. Can be bool, float16, float32, float64, int8, int32, int64, uint8.
        name (str, optional): Name for the operation (optional, default is None).
            For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        A Sparse Tensor with the same data type and shape as ``x`` .

    Examples:
        .. code-block:: python

            import paddle

            dense_x = paddle.to_tensor([-2, 0, 1])
            sparse_x = dense_x.to_sparse_coo(1)
            out = paddle.sparse.cast(sparse_x, 'int32', 'float64')

    """
    if index_dtype and not isinstance(index_dtype, core.VarDesc.VarType):
        index_dtype = convert_np_dtype_to_dtype_(index_dtype)
    if value_dtype and not isinstance(value_dtype, core.VarDesc.VarType):
        value_dtype = convert_np_dtype_to_dtype_(value_dtype)
    return _C_ops.sparse_cast(x, index_dtype, value_dtype)


@dygraph_only
def pow(x, factor, name=None):
    """
    Calculate elementwise pow of x, requiring x to be a SparseCooTensor or SparseCsrTensor.

    .. math::

        out = x^{factor}

    Parameters:
        x (Tensor): The input Sparse Tensor with data type float32, float64.
        factor (float|int): factor of pow.
        name (str, optional): Name for the operation (optional, default is None).
            For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        A Sparse Tensor with the same data type and shape as ``x`` .

    Examples:
        .. code-block:: python

            import paddle

            dense_x = paddle.to_tensor([-2, 0, 3], dtype='float32')
            sparse_x = dense_x.to_sparse_coo(1)
            out = paddle.sparse.pow(sparse_x, 2)

    """
    return _C_ops.sparse_pow(x, float(factor))


@dygraph_only
def neg(x, name=None):
    """
    Calculate elementwise negative of x, requiring x to be a SparseCooTensor or SparseCsrTensor.

    .. math::

        out = -x

    Parameters:
        x (Tensor): The input Sparse Tensor with data type float32, float64.
        name (str, optional): Name for the operation (optional, default is None).
            For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        A Sparse Tensor with the same data type and shape as ``x`` .

    Examples:
        .. code-block:: python

            import paddle

            dense_x = paddle.to_tensor([-2, 0, 3], dtype='float32')
            sparse_x = dense_x.to_sparse_coo(1)
            out = paddle.sparse.neg(sparse_x)

    """
    return _C_ops.sparse_scale(x, -1.0, 0.0, True)


@dygraph_only
def abs(x, name=None):
    """
    Calculate elementwise absolute value of x, requiring x to be a SparseCooTensor or SparseCsrTensor.

    .. math::

        out = |x|

    Parameters:
        x (Tensor): The input Sparse Tensor with data type float32, float64.
        name (str, optional): Name for the operation (optional, default is None).
            For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        A Sparse Tensor with the same data type and shape as ``x`` .

    Examples:
        .. code-block:: python

            import paddle

            dense_x = paddle.to_tensor([-2, 0, 3], dtype='float32')
            sparse_x = dense_x.to_sparse_coo(1)
            out = paddle.sparse.abs(sparse_x)

    """
    return _C_ops.sparse_abs(x)


@dygraph_only
def coalesce(x, name=None):
    r"""
    the coalesced operator include sorted and merge, after coalesced, the indices of x is sorted and unique.

    Parameters:
        x (Tensor): the input SparseCooTensor.
        name (str, optional): Name for the operation (optional, default is None).
            For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor: return the SparseCooTensor after coalesced.

    Examples:
        .. code-block:: python

            import paddle

            indices = [[0, 0, 1], [1, 1, 2]]
            values = [1.0, 2.0, 3.0]
            sp_x = paddle.sparse.sparse_coo_tensor(indices, values)
            sp_x = paddle.sparse.coalesce(sp_x)
            print(sp_x.indices())
            #[[0, 1], [1, 2]]
            print(sp_x.values())
            #[3.0, 3.0]
    """
    return _C_ops.sparse_coalesce(x)


@dygraph_only
def rad2deg(x, name=None):
    """
    Convert each of the elements of input x from angles in radians to degrees,
    requiring x to be a SparseCooTensor or SparseCsrTensor.

    .. math::

        rad2deg(x) = 180/ \pi * x

    Parameters:
        x (Tensor): The input Sparse Tensor with data type float32, float64, int32, int64.
        name (str, optional): Name for the operation (optional, default is None).
            For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        A Sparse Tensor with the same data type and shape as ``x`` .

    Examples:
        .. code-block:: python

            import paddle

            dense_x = paddle.to_tensor([3.142, 0., -3.142])
            sparse_x = dense_x.to_sparse_coo(1)
            out = paddle.sparse.rad2deg(sparse_x)

    """
    if x.dtype in _int_dtype_:
        x = _C_ops.sparse_cast(x, None, core.VarDesc.VarType.FP32)
    return _C_ops.sparse_scale(x, 180.0 / np.pi, 0.0, True)


@dygraph_only
def deg2rad(x, name=None):
    """
    Convert each of the elements of input x from degrees to angles in radians,
    requiring x to be a SparseCooTensor or SparseCsrTensor.

    .. math::

        deg2rad(x) = \pi * x / 180

    Parameters:
        x (Tensor): The input Sparse Tensor with data type float32, float64, int32, int64.
        name (str, optional): Name for the operation (optional, default is None).
            For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        A Sparse Tensor with the same data type and shape as ``x`` .

    Examples:
        .. code-block:: python

            import paddle

            dense_x = paddle.to_tensor([-180, 0, 180])
            sparse_x = dense_x.to_sparse_coo(1)
            out = paddle.sparse.deg2rad(sparse_x)

    """
    if x.dtype in _int_dtype_:
        x = _C_ops.sparse_cast(x, None, core.VarDesc.VarType.FP32)
    return _C_ops.sparse_scale(x, np.pi / 180.0, 0.0, True)


@dygraph_only
def expm1(x, name=None):
    """
    Calculate elementwise `exp(x)-1` , requiring x to be a SparseCooTensor or SparseCsrTensor.

    .. math::

        out = exp(x) - 1

    Parameters:
        x (Tensor): The input Sparse Tensor with data type float32, float64.
        name (str, optional): Name for the operation (optional, default is None).
            For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        A Sparse Tensor with the same data type and shape as ``x`` .

    Examples:
        .. code-block:: python

            import paddle

            dense_x = paddle.to_tensor([-2., 0., 1.])
            sparse_x = dense_x.to_sparse_coo(1)
            out = paddle.sparse.expm1(sparse_x)
    """
    return _C_ops.sparse_expm1(x)


@dygraph_only
def reshape(x, shape, name=None):
    """
    Changes the shape of ``x`` without changing its value, requiring x to be a SparseCooTensor or SparseCsrTensor.
    Currently this function can only reshape the sparse dims of ``x`` , but ``shape`` argument must be specified
    as the shape of the reshaped tensor.

    Note that if x is a SparseCsrTensor, then len(shape) must be 2 or 3.

    There are some tricks when specifying the target shape.

        - 1. -1 means the value of this dimension is inferred from the total element number of x and remaining dimensions. Thus one and only one dimension can be set -1.

        - 2. 0 means the actual dimension value is going to be copied from the corresponding dimension of x. The indices of 0 in the target shape can not exceed the rank of x.

    Here are some examples to explain it.

        - 1. Given a 3-D tensor x with a shape [2, 4, 6], and the target shape is [6, 8], the reshape operator will transform x into a 2-D tensor with shape [6, 8] and leaving x's data unchanged.

        - 2. Given a 3-D tensor x with a shape [2, 4, 6], and the target shape is [2, 3, -1, 2], the reshape operator will transform x into a 4-D tensor with shape [2, 3, 4, 2] and leaving x's data unchanged. In this case, one dimension of the target shape is set to -1, the value of this dimension is inferred from the total element number of x and remaining dimensions.

        - 3. Given a 3-D tensor x with a shape [2, 4, 6], and the target shape is [-1, 0, 3, 2], the reshape operator will transform x into a 4-D tensor with shape [2, 4, 3, 2] and leaving x's data unchanged. In this case, besides -1, 0 means the actual dimension value is going to be copied from the corresponding dimension of x.

    Args:
        x (Tensor): The input sparse tensor with data type ``float32``, ``float64``, ``int32``, ``int64`` or ``bool``.
        shape (list|tuple): Define the target shape. At most one dimension of the target shape can be -1.
                        The data type is ``int32``.
        name (str, optional): Name for the operation (optional, default is None).
            For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor: A reshaped Tensor with the same data type as ``x``.

    Examples:
        .. code-block:: python

            import paddle

            x_shape = [6, 2, 3]
            new_shape = [1, 0, 2, -1, 3]
            format = "coo"

            dense_x = paddle.randint(-100, 100, x_shape) * paddle.randint(0, 2, x_shape)

            if format == "coo":
                sp_x = dense_x.to_sparse_coo(len(x_shape))
            else:
                sp_x = dense_x.to_sparse_csr()
            sp_out = paddle.sparse.reshape(sp_x, new_shape)

            print(sp_out)
            # the shape of sp_out is [1, 2, 2, 3, 3]

    """
    return _C_ops.sparse_reshape(x, shape)
