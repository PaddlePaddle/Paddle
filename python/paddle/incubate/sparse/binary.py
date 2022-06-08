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

from paddle.common_ops_import import dygraph_only
from paddle import _C_ops

__all__ = []


@dygraph_only
def mm(x, y):
    """
    Warning:    
        This API is only used from ``CUDA 11.2`` and linux.

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

    Returns:
        Tensor: Its layout is determined by that of `x` and `y` .

    Examples:

        .. code-block:: python

            import numpy as np
            import paddle
            from paddle.fluid.framework import _test_eager_guard
            
            paddle.seed(100)

            with _test_eager_guard():
                # csr @ dense -> dense
                crows = [0, 2, 3, 5]
                cols = [1, 3, 2, 0, 1]
                values = [1., 2., 3., 4., 5.]
                dense_shape = [3, 4]
                csr = paddle.incubate.sparse.sparse_csr_tensor(crows, cols, values, dense_shape)
                dense = paddle.randn([4, 3])
                
                out = paddle.incubate.sparse.mm(csr, dense)
                # Tensor(shape=[3, 3], dtype=float32, place=Place(gpu:0), stop_gradient=True,
                #        [[-1.94294846 , -3.33990622 ,  0.62359387 ],
                #         [-4.12815523 ,  3.46535444 , -3.27413893 ],
                #         [-0.15209436 , -19.23207283, -3.35593438 ]])

    """
    return _C_ops.final_state_sparse_mm(x, y)


@dygraph_only
def mm_mask_as(x, y, mask):
    """
    Warning:    
        This API is only used from ``CUDA 11.2`` and linux.

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

    Returns:
        Tensor: SparseCoo or SparseCsr, which is determined by that of `mask` .

    Examples:

        .. code-block:: python

            import numpy as np
            import paddle
            from paddle.fluid.framework import _test_eager_guard
            
            paddle.seed(100)

            with _test_eager_guard():
                # dense @ dense * csr_mask -> csr
                crows = [0, 2, 3, 5]
                cols = [1, 3, 2, 0, 1]
                values = [1., 2., 3., 4., 5.]
                dense_shape = [3, 4]
                mask = paddle.incubate.sparse.sparse_csr_tensor(crows, cols, values, dense_shape)
                
                x = paddle.rand([3, 5])
                y = paddle.rand([5, 4])

                out = paddle.incubate.sparse.mm_mask_as(x, y, mask)
                # Tensor(shape=[3, 4], dtype=paddle.float32, place=Place(gpu:0), stop_gradient=True, 
                #        crows=[0, 2, 3, 5], 
                #        cols=[1, 3, 2, 0, 1], 
                #        values=[0.98986477, 0.97800624, 1.14591956, 0.68561077, 0.94714981])

    """
    return _C_ops.final_state_sparse_mm_mask_as(x, y, mask)
