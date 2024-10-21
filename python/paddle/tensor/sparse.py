#   Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

from paddle.base.framework import (
    dygraph_only,
)

if TYPE_CHECKING:
    from paddle import Tensor

__all__ = []


@dygraph_only
def is_coalesced(x: Tensor, name: str | None = None) -> Tensor:
    r"""
    Check whether the Tensor is a coalesced SparseCooTensor. If not it will return False.
    Any Tensor type among DenseTensor/SparseCooTensor/SparseCsrTensor are supported.

    Args:
        x (Tensor): The input tensor. It can be DenseTensor/SparseCooTensor/SparseCsrTensor.
        name (str|None, optional): Name for the operation (optional, default is None).
            For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        bool: True if the Tensor is a coalesced SparseCooTensor, and False otherwise.

    Examples:

        .. code-block:: python

            >>> import paddle

            >>> indices = [[0, 0, 1], [1, 1, 2]]
            >>> values = [1.0, 2.0, 3.0]
            >>> x = paddle.sparse.sparse_coo_tensor(indices, values)
            >>> x.is_coalesced()
            False
            >>> x = x.coalesce()
            >>> x.is_coalesced()
            True

            >>> indices = [[0, 1, 1], [1, 1, 2]]
            >>> values = [1.0, 2.0, 3.0]
            >>> x = paddle.sparse.sparse_coo_tensor(indices, values)
            >>> x.is_coalesced()
            True

            >>> x = paddle.to_tensor([[1., 2., 3.]])
            >>> x.is_coalesced()
            False

            >>> x = x.to_sparse_csr()
            >>> x.is_coalesced()
            False
    """
    if not x.is_sparse_coo():
        return False
    nnz = x.indices().shape[1]
    x_coalesced = x.coalesce()
    coalesced_nnz = x_coalesced.indices().shape[1]
    return nnz == coalesced_nnz
