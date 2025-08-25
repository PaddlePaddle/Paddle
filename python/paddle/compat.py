# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved
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

import paddle

if TYPE_CHECKING:
    from paddle import Tensor

from .tensor.compat import (
    Unfold,
    sort,
    split,
)

__all__ = ['split', 'sort', 'Unfold', 'median', 'nanmedian']


def median(
    input: Tensor,
    dim: int | None = None,
    keepdim: bool = False,
    *,
    out: tuple[Tensor, Tensor] | Tensor | None = None,
) -> tuple[Tensor, Tensor] | Tensor:
    """
    Returns the median of the values in input.

    Args:
        input (Tensor): The input tensor.
        dim (int|None, optional): The dimension to reduce. If None, computes the median over all elements. Default is None.
        keepdim (bool, optional): Whether the output tensor has dim retained or not. Default is False.
        out (Tensor|tuple[Tensor, Tensor], optional): If provided, the result will be written into this tensor.
            For global median (dim=None), out must be a single tensor.
            For median along a dimension (dim specified, including dim=-1), out must be a tuple of two tensors (values, indices).

    Returns:
        Tensor|tuple[Tensor, Tensor]: The median values. If dim is None, returns a single tensor. If dim is specified (including dim=-1), returns a tuple of (values, indices).

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> x = paddle.to_tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
            >>> result = paddle.compat.median(x)
            >>> print(result)
            Tensor(shape=[], dtype=int64, place=Place(cpu), stop_gradient=True, 5)

            >>> values, indices = paddle.compat.median(x, dim=1)
            >>> print(values)
            Tensor(shape=[3], dtype=int64, place=Place(cpu), stop_gradient=True, [2, 5, 8])
            >>> print(indices)
            Tensor(shape=[3], dtype=int64, place=Place(cpu), stop_gradient=True, [1, 1, 1])

            >>> # Using out parameter
            >>> out_values = paddle.zeros([3], dtype='int64')
            >>> out_indices = paddle.zeros([3], dtype='int64')
            >>> paddle.compat.median(x, dim=1, out=(out_values, out_indices))
            >>> print(out_values)
            Tensor(shape=[3], dtype=int64, place=Place(cpu), stop_gradient=True, [2, 5, 8])
    """
    if dim is None:
        result = paddle.median(input, axis=dim, keepdim=keepdim, mode='min')
        if out is not None:
            if isinstance(out, tuple):
                raise ValueError(
                    "For global median (dim=None), out must be a single tensor"
                )
            paddle.assign(result, out)
            return out
        return result
    else:
        result, indices = paddle.median(
            input, axis=dim, keepdim=keepdim, mode='min'
        )
        if out is not None:
            if isinstance(out, tuple) and len(out) == 2:
                out_values, out_indices = out
                if out_values is not None:
                    paddle.assign(result, out_values)
                if out_indices is not None:
                    paddle.assign(indices, out_indices)
                return out_values, out_indices
            else:
                raise ValueError(
                    "For median with dim specified, out must be a tuple of two tensors"
                )
        return result, indices


def nanmedian(
    input: Tensor,
    dim: int | None = None,
    keepdim: bool = False,
    *,
    out: tuple[Tensor, Tensor] | Tensor | None = None,
) -> tuple[Tensor, Tensor] | Tensor:
    """
    Returns the median of the values in input, ignoring NaN values.

    Args:
        input (Tensor): The input tensor.
        dim (int|None, optional): The dimension to reduce. If None, computes the nanmedian over all elements. Default is None.
        keepdim (bool, optional): Whether the output tensor has dim retained or not. Default is False.
        out (Tensor|tuple[Tensor, Tensor], optional): If provided, the result will be written into this tensor.
            For global nanmedian (dim=None), out must be a single tensor.
            For nanmedian along a dimension (dim specified, including dim=-1), out must be a tuple of two tensors (values, indices).

    Returns:
        Tensor|tuple[Tensor, Tensor]: The median values, ignoring NaN. If dim is None, returns a single tensor. If dim is specified (including dim=-1), returns a tuple of (values, indices).

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> import numpy as np

            >>> x = paddle.to_tensor([[1, float('nan'), 3], [4, 5, 6], [float('nan'), 8, 9]], dtype='float32')
            >>> result = paddle.compat.nanmedian(x)
            >>> print(result)
            Tensor(shape=[], dtype=float32, place=Place(cpu), stop_gradient=True, 5.0)

            >>> values, indices = paddle.compat.nanmedian(x, dim=1)
            >>> print(values)
            Tensor(shape=[3], dtype=float32, place=Place(cpu), stop_gradient=True, [2.0, 5.0, 8.5])
            >>> print(indices)
            Tensor(shape=[3], dtype=int64, place=Place(cpu), stop_gradient=True, [1, 1, 1])

            >>> # Using out parameter
            >>> out_values = paddle.zeros([3], dtype='float32')
            >>> out_indices = paddle.zeros([3], dtype='int64')
            >>> paddle.compat.nanmedian(x, dim=1, out=(out_values, out_indices))
            >>> print(out_values)
            Tensor(shape=[3], dtype=float32, place=Place(cpu), stop_gradient=True, [2.0, 5.0, 8.5])
    """
    if dim is None:
        result = paddle.nanmedian(input, axis=dim, keepdim=keepdim, mode='min')
        if out is not None:
            if isinstance(out, tuple):
                raise ValueError(
                    "For global nanmedian (dim=None), out must be a single tensor"
                )
            paddle.assign(result, out)
            return out
        return result
    else:
        result, indices = paddle.nanmedian(
            input, axis=dim, keepdim=keepdim, mode='min'
        )
        if out is not None:
            if isinstance(out, tuple) and len(out) == 2:
                out_values, out_indices = out
                if out_values is not None:
                    paddle.assign(result, out_values)
                if out_indices is not None:
                    paddle.assign(indices, out_indices)
                return out_values, out_indices
            else:
                raise ValueError(
                    "For nanmedian with dim specified, out must be a tuple of two tensors"
                )
        return result, indices
