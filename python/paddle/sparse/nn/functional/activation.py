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

__all__ = []

from paddle import _C_ops, in_dynamic_mode
from paddle.base.framework import dygraph_only
from paddle.base.layer_helper import LayerHelper


def relu(x, name=None):
    """
    sparse relu activation, requiring x to be a SparseCooTensor or SparseCsrTensor.

    .. math::

        out = max(x, 0)

    Parameters:
        x (Tensor): The input Sparse Tensor with data type float32, float64.
        name (str, optional): Name for the operation (optional, default is None).
            For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        A Sparse Tensor with the same data type and shape as ``x`` .

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> dense_x = paddle.to_tensor([-2., 0., 1.])
            >>> sparse_x = dense_x.to_sparse_coo(1)
            >>> out = paddle.sparse.nn.functional.relu(sparse_x)
            >>> print(out)
            Tensor(shape=[3], dtype=paddle.float32, place=Place(cpu), stop_gradient=True,
                   indices=[[0, 2]],
                   values=[0., 1.])
    """
    if in_dynamic_mode():
        return _C_ops.sparse_relu(x)
    else:
        op_type = 'sparse_relu'
        helper = LayerHelper(op_type)
        out = helper.create_sparse_variable_for_type_inference(x.dtype)
        helper.append_op(
            type=op_type, inputs={'x': x}, outputs={'out': out}, attrs={}
        )
        return out


def softmax(x, axis=-1, name=None):
    r"""
    sparse softmax activation, requiring x to be a SparseCooTensor or SparseCsrTensor.

    Note:
        Only support axis=-1 for SparseCsrTensor, which is faster when read data
        by row (axis=-1).

    From the point of view of dense matrix, for each row :math:`i` and each column :math:`j`
    in the matrix, we have:

    .. math::

        softmax_ij = \frac{\exp(x_ij - max_j(x_ij))}{\sum_j(exp(x_ij - max_j(x_ij))}

    Parameters:
        x (Tensor): The input tensor. It can be SparseCooTensor/SparseCsrTensor. The data type can be float32 or float64.
        axis (int, optional): The axis along which to perform softmax calculations. Only support -1 for SparseCsrTensor.
        name (str, optional): Name for the operation (optional, default is None).
            For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor: SparseCoo or SparseCsr, whose layout is the same with `x` .

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> paddle.seed(100)

            >>> mask = paddle.rand((3, 4)) < 0.5
            >>> x = paddle.rand((3, 4)) * mask
            >>> print(x)
            Tensor(shape=[3, 4], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[0.        , 0.95717543, 0.43864486, 0.        ],
             [0.84765935, 0.45680618, 0.39412445, 0.        ],
             [0.59444654, 0.        , 0.78364515, 0.        ]])

            >>> csr = x.to_sparse_csr()
            >>> print(csr)
            Tensor(shape=[3, 4], dtype=paddle.float32, place=Place(cpu), stop_gradient=True,
                   crows=[0, 2, 5, 7],
                   cols=[1, 2, 0, 1, 2, 0, 2],
                   values=[0.95717543, 0.43864486, 0.84765935, 0.45680618, 0.39412445,
                           0.59444654, 0.78364515])

            >>> out = paddle.sparse.nn.functional.softmax(csr)
            >>> print(out)
            Tensor(shape=[3, 4], dtype=paddle.float32, place=Place(cpu), stop_gradient=True,
                   crows=[0, 2, 5, 7],
                   cols=[1, 2, 0, 1, 2, 0, 2],
                   values=[0.62680405, 0.37319586, 0.43255258, 0.29261294, 0.27483448,
                           0.45284089, 0.54715902])

            >>> coo = x.to_sparse_coo(sparse_dim=2)
            >>> print(coo)
            Tensor(shape=[3, 4], dtype=paddle.float32, place=Place(cpu), stop_gradient=True,
                   indices=[[0, 0, 1, 1, 1, 2, 2],
                            [1, 2, 0, 1, 2, 0, 2]],
                   values=[0.95717543, 0.43864486, 0.84765935, 0.45680618, 0.39412445,
                           0.59444654, 0.78364515])

            >>> out = paddle.sparse.nn.functional.softmax(coo)
            >>> print(out)
            Tensor(shape=[3, 4], dtype=paddle.float32, place=Place(cpu), stop_gradient=True,
                   indices=[[0, 0, 1, 1, 1, 2, 2],
                            [1, 2, 0, 1, 2, 0, 2]],
                   values=[0.62680405, 0.37319589, 0.43255258, 0.29261294, 0.27483445,
                           0.45284092, 0.54715902])
    """
    if in_dynamic_mode():
        return _C_ops.sparse_softmax(x, axis)
    else:
        op_type = 'sparse_softmax'
        helper = LayerHelper(op_type)
        out = helper.create_sparse_variable_for_type_inference(x.dtype)
        helper.append_op(
            type=op_type,
            inputs={'x': x},
            outputs={'out': out},
            attrs={'axis': axis},
        )
        return out


@dygraph_only
def relu6(x, name=None):
    """
    sparse relu6 activation, requiring x to be a SparseCooTensor or SparseCsrTensor.

    .. math::

        relu6(x) = min(max(0, x), 6)

    Parameters:
        x (Tensor): The input Sparse Tensor with data type float32, float64.
        name (str, optional): Name for the operation (optional, default is None).
            For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        A Sparse Tensor with the same data type and shape as ``x`` .

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> dense_x = paddle.to_tensor([-2., 0., 8.])
            >>> sparse_x = dense_x.to_sparse_coo(1)
            >>> out = paddle.sparse.nn.functional.relu6(sparse_x)
    """
    return _C_ops.sparse_relu6(x)


@dygraph_only
def leaky_relu(x, negative_slope=0.01, name=None):
    r"""
    sparse leaky_relu activation, requiring x to be a SparseCooTensor or SparseCsrTensor.

    .. math::
        leaky\_relu(x)=
        \left\{
            \begin{array}{rcl}
                x, & & if \ x >= 0 \\
                negative\_slope * x, & & otherwise \\
            \end{array}
        \right.

    Parameters:
        x (Tensor): The input Sparse Tensor with data type float32, float64.
        negative_slope (float, optional): Slope of the activation function at
            :math:`x < 0` . Default is 0.01.
        name (str, optional): Name for the operation (optional, default is None).
            For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        A Sparse Tensor with the same data type and shape as ``x`` .

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> dense_x = paddle.to_tensor([-2., 0., 5.])
            >>> sparse_x = dense_x.to_sparse_coo(1)
            >>> out = paddle.sparse.nn.functional.leaky_relu(sparse_x, 0.5)
    """
    return _C_ops.sparse_leaky_relu(x, negative_slope)
