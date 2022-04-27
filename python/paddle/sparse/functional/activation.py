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
                dense_x = paddle.to_tensor([-2, 0, 1])
                sparse_x = dense_x.to_sparse_coo(1)
                out = paddle.sparse.functional.relu(sparse_x) 
    """

    assert in_dynamic_mode(), "Currently, Sparse API only support dynamic mode"

    if x.is_sparse_coo():
        return _C_ops.final_state_sparse_coo_relu(x)
    elif x.is_sparse_csr():
        return _C_ops.final_state_sparse_csr_relu(x)
    else:
        raise ValueError(
            "Currently, sparse.relu only support the input of SparseCooTensor or SparseCsrTensor"
        )


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
                dense_x = paddle.to_tensor([-2, 0, 1])
                sparse_x = dense_x.to_sparse_coo(1)
                out = paddle.sparse.tanh(sparse_x)
    """

    assert in_dynamic_mode(), "Currently, Sparse API only support dynamic mode"

    if x.is_sparse_coo():
        return _C_ops.final_state_sparse_coo_tanh(x)
    elif x.is_sparse_csr():
        return _C_ops.final_state_sparse_csr_tanh(x)
    else:
        raise ValueError(
            "Currently, sparse.tanh only support the input of SparseCooTensor or SparseCsrTensor"
        )
