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

from paddle.utils.decorator_utils import ForbidKeywordsDecorator

from .tensor.compat import (
    MinMaxRetType,
    Unfold,
    _check_out_status,
    max,
    min,
    sort,
    split,
)

__all__ = ['split', 'sort', 'Unfold', 'median', 'nanmedian', 'max', 'min']


class MedianRetType(MinMaxRetType):
    pass


@ForbidKeywordsDecorator(
    illegal_keys={"x", "axis"},
    func_name="paddle.compat.median",
    correct_name="paddle.median",
)
def median(
    input: Tensor,
    dim: int | None = None,
    keepdim: bool = False,
    *,
    out: tuple[Tensor, Tensor] | Tensor | None = None,
) -> Tensor | MedianRetType:
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
        Tensor|MedianRetType: If dim is None, returns a single tensor. If dim is specified (including dim=-1),
        returns a named tuple MedianRetType(values: Tensor, indices: Tensor).

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> x = paddle.to_tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
            >>> result = paddle.compat.median(x)
            >>> print(result)
            Tensor(shape=[], dtype=int64, place=Place(cpu), stop_gradient=True, 5)

            >>> ret = paddle.compat.median(x, dim=1)
            >>> print(ret.values)
            Tensor(shape=[3], dtype=int64, place=Place(cpu), stop_gradient=True, [2, 5, 8])
            >>> print(ret.indices)
            Tensor(shape=[3], dtype=int64, place=Place(cpu), stop_gradient=True, [1, 1, 1])

            >>> # Using out parameter
            >>> out_values = paddle.zeros([3], dtype='int64')
            >>> out_indices = paddle.zeros([3], dtype='int64')
            >>> paddle.compat.median(x, dim=1, out=(out_values, out_indices))
            >>> print(out_values)
            Tensor(shape=[3], dtype=int64, place=Place(cpu), stop_gradient=True, [2, 5, 8])
    """
    if dim is None:
        _check_out_status(out, False)
        result = paddle.median(input, axis=dim, keepdim=keepdim, mode='min')
        if out is not None:
            paddle.assign(result, out)
            return out
        return result
    else:
        _check_out_status(out, True)
        values, indices = paddle.median(
            input, axis=dim, keepdim=keepdim, mode='min'
        )
        if out is not None:
            paddle.assign(values, out[0])
            paddle.assign(indices, out[1])
            return MedianRetType(values=out[0], indices=out[1])
        return MedianRetType(values=values, indices=indices)


@ForbidKeywordsDecorator(
    illegal_keys={"x", "axis"},
    func_name="paddle.compat.nanmedian",
    correct_name="paddle.nanmedian",
)
def nanmedian(
    input: Tensor,
    dim: int | None = None,
    keepdim: bool = False,
    *,
    out: tuple[Tensor, Tensor] | Tensor | None = None,
) -> Tensor | MedianRetType:
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
        Tensor|MedianRetType: The median values, ignoring NaN. If dim is None, returns a single tensor. If dim is specified (including dim=-1),
        returns a named tuple MedianRetType(values: Tensor, indices: Tensor).

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> import numpy as np

            >>> x = paddle.to_tensor([[1, float('nan'), 3], [4, 5, 6], [float('nan'), 8, 9]], dtype='float32')
            >>> result = paddle.compat.nanmedian(x)
            >>> print(result)
            Tensor(shape=[], dtype=float32, place=Place(cpu), stop_gradient=True, 5.0)

            >>> ret = paddle.compat.nanmedian(x, dim=1)
            >>> print(ret.values)
            Tensor(shape=[3], dtype=float32, place=Place(cpu), stop_gradient=True, [1.0, 5.0, 8.0])
            >>> print(ret.indices)
            Tensor(shape=[3], dtype=int64, place=Place(cpu), stop_gradient=True, [0, 1, 1])

            >>> # Using out parameter
            >>> out_values = paddle.zeros([3], dtype='float32')
            >>> out_indices = paddle.zeros([3], dtype='int64')
            >>> paddle.compat.nanmedian(x, dim=1, out=(out_values, out_indices))
            >>> print(out_values)
            Tensor(shape=[3], dtype=float32, place=Place(cpu), stop_gradient=True, [1.0, 5.0, 8.0])
    """
    if dim is None:
        _check_out_status(out, False)
        result = paddle.nanmedian(input, axis=dim, keepdim=keepdim, mode='min')
        if out is not None:
            paddle.assign(result, out)
            return out
        return result
    else:
        _check_out_status(out, True)
        values, indices = paddle.nanmedian(
            input, axis=dim, keepdim=keepdim, mode='min'
        )
        # This conversion is needed because PyTorch returns index 0 for all-nan rows,
        # while PaddlePaddle returns index -1 for all-nan rows
        indices = paddle.maximum(indices, paddle.zeros_like(indices))

        if out is not None:
            paddle.assign(values, out[0])
            paddle.assign(indices, out[1])
            return MedianRetType(values=out[0], indices=out[1])
        return MedianRetType(values=values, indices=indices)
