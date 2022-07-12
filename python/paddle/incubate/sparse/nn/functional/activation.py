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

from paddle import _C_ops
from paddle.fluid.framework import dygraph_only


@dygraph_only
def relu(x, name=None):
    """
    sparse relu activation, requiring x to be a sparse coo or sparse csr tensor.

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

            import paddle
            from paddle.fluid.framework import _test_eager_guard

            with _test_eager_guard():
                dense_x = paddle.to_tensor([-2, 0, 1], dtype='float32')
                sparse_x = dense_x.to_sparse_coo(1)
                out = paddle.incubate.sparse.nn.functional.relu(sparse_x) 
    """
    return _C_ops.final_state_sparse_relu(x)


@dygraph_only
def softmax(x, axis=-1, name=None):
    """
    sparse softmax activation, x must be SparseCsrTensor or SparseCooTensor.

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

            import paddle
            import numpy as np
            from paddle.fluid.framework import _test_eager_guard

            paddle.seed(100)

            with _test_eager_guard():
                mask = np.random.rand(3, 4) < 0.5
                np_x = np.random.rand(3, 4) * mask
                # [[0.         0.         0.96823406 0.19722934]
                #  [0.94373937 0.         0.02060066 0.71456372]
                #  [0.         0.         0.         0.98275049]]

                csr = paddle.to_tensor(np_x).to_sparse_csr()
                # Tensor(shape=[3, 4], dtype=paddle.float64, place=Place(gpu:0), stop_gradient=True, 
                #        crows=[0, 2, 5, 6], 
                #        cols=[2, 3, 0, 2, 3, 3], 
                #        values=[0.96823406, 0.19722934, 0.94373937, 0.02060066, 0.71456372,
                #                0.98275049])

                out = paddle.incubate.sparse.nn.functional.softmax(csr)
                # Tensor(shape=[3, 4], dtype=paddle.float64, place=Place(gpu:0), stop_gradient=True, 
                #        crows=[0, 2, 5, 6], 
                #        cols=[2, 3, 0, 2, 3, 3], 
                #        values=[0.68373820, 0.31626180, 0.45610887, 0.18119845, 0.36269269,
                #                1.        ])
    
    """
    return _C_ops.final_state_sparse_softmax(x, axis)
