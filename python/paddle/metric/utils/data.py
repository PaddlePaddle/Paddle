# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

"""Data utilities for paddle.metric."""

from typing import TYPE_CHECKING, Any, Callable

import paddle

if TYPE_CHECKING:
    from collections.abc import Sequence

METRIC_EPS = 1e-06


def apply_to_collection(
    data: Any,
    dtype: Any,
    function: Callable,
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Recursively apply a function to all elements of a collection matching a given dtype.

    Args:
        data: the collection to apply the function to
        dtype: the dtype to match. Can be a type or tuple of types.
        function: the function to apply
        *args: positional arguments to pass to the function
        **kwargs: keyword arguments to pass to the function

    Returns:
        the transformed collection
    """
    if isinstance(data, dtype):
        return function(data, *args, **kwargs)

    if isinstance(data, dict):
        return {
            key: apply_to_collection(item, dtype, function, *args, **kwargs)
            for key, item in data.items()
        }

    if isinstance(data, (tuple, list)):
        out = [
            apply_to_collection(item, dtype, function, *args, **kwargs)
            for item in data
        ]
        return data.__class__(out)

    return data


def dim_zero_cat(x: paddle.Tensor | list[paddle.Tensor]) -> paddle.Tensor:
    """Concatenation along the zero dimension."""
    if isinstance(x, paddle.Tensor):
        return x
    x = [y.unsqueeze(0) if y.numel() == 1 and y.ndim == 0 else y for y in x]
    if not x:
        raise ValueError("No samples to concatenate")
    return paddle.concat(x, axis=0)


def dim_zero_sum(x: paddle.Tensor) -> paddle.Tensor:
    """Summation along the zero dimension."""
    return x.sum(axis=0)


def dim_zero_mean(x: paddle.Tensor) -> paddle.Tensor:
    """Average along the zero dimension."""
    return x.mean(axis=0)


def dim_zero_max(x: paddle.Tensor) -> paddle.Tensor:
    """Max along the zero dimension."""
    return x.max(axis=0)


def dim_zero_min(x: paddle.Tensor) -> paddle.Tensor:
    """Min along the zero dimension."""
    return x.min(axis=0)


def _flatten(x: Sequence) -> list:
    """Flatten list of list into single list."""
    return [item for sublist in x for item in sublist]


def _flatten_dict(x: dict) -> tuple[dict, bool]:
    """Flatten dict of dicts into single dict, checking for duplicate keys."""
    new_dict: dict = {}
    duplicates = False
    for key, value in x.items():
        if isinstance(value, dict):
            for k, v in value.items():
                if k in new_dict:
                    duplicates = True
                new_dict[k] = v
        else:
            if key in new_dict:
                duplicates = True
            new_dict[key] = value
    return new_dict, duplicates


def to_onehot(
    label_tensor: paddle.Tensor, num_classes: int | None = None
) -> paddle.Tensor:
    """Convert a dense label tensor to one-hot format.

    Args:
        label_tensor: dense label tensor, with shape [N, d1, d2, ...]
        num_classes: number of classes C

    Returns:
        A sparse label tensor with shape [N, C, d1, d2, ...]
    """
    if num_classes is None:
        num_classes = int(label_tensor.max().detach().item() + 1)
    tensor_onehot = paddle.zeros(
        [label_tensor.shape[0], num_classes, *label_tensor.shape[1:]],
        dtype=label_tensor.dtype,
    )
    index = label_tensor.cast("int64").unsqueeze(1).expand_as(tensor_onehot)
    return tensor_onehot.put_along_axis(index, 1.0, axis=1)


def _top_k_with_half_precision_support(
    x: paddle.Tensor, k: int = 1, axis: int = 1
) -> paddle.Tensor:
    """topk does not support half precision on CPU in some cases."""
    if x.dtype == paddle.float16 and not paddle.device.is_compiled_with_cuda():
        idx = paddle.argsort(x, axis=axis, stable=True).flip(axis=axis)
        return idx.slice([axis], [0], [k])
    return paddle.topk(x, k=k, axis=axis).indices


def select_topk(
    prob_tensor: paddle.Tensor, topk: int = 1, axis: int = 1
) -> paddle.Tensor:
    """Convert a probability tensor to binary by selecting top-k highest entries.

    Args:
        prob_tensor: dense tensor of shape ``[..., C, ...]``
        topk: number of the highest entries to turn into 1s
        axis: dimension on which to compare entries

    Returns:
        A binary tensor of the same shape as the input tensor of type ``int32``
    """
    topk_tensor = paddle.zeros_like(prob_tensor, dtype=paddle.int32)
    if topk == 1:
        indices = prob_tensor.argmax(axis=axis, keepdim=True)
        topk_tensor = topk_tensor.put_along_axis(indices, 1, axis=axis)
    else:
        indices = _top_k_with_half_precision_support(
            prob_tensor, k=topk, axis=axis
        )
        topk_tensor = topk_tensor.put_along_axis(indices, 1, axis=axis)
    return topk_tensor.cast("int32")


def to_categorical(x: paddle.Tensor, argmax_dim: int = 1) -> paddle.Tensor:
    """Convert a tensor of probabilities to a dense label tensor.

    Args:
        x: probabilities to get the categorical label [N, d1, d2, ...]
        argmax_dim: dimension to apply

    Return:
        A tensor with categorical labels [N, d2, ...]
    """
    return paddle.argmax(x, axis=argmax_dim)


def _squeeze_scalar_element_tensor(x: paddle.Tensor) -> paddle.Tensor:
    return x.squeeze() if x.numel() == 1 else x


def _squeeze_if_scalar(data: Any) -> Any:
    return apply_to_collection(
        data, paddle.Tensor, _squeeze_scalar_element_tensor
    )


def _bincount(x: paddle.Tensor, minlength: int | None = None) -> paddle.Tensor:
    """Implement bincount for paddle tensors.

    Args:
        x: tensor to count (must contain non-negative integers)
        minlength: minimum length of the output tensor

    Returns:
        Number of occurrences for each unique element in x
    """
    if minlength is None:
        minlength = int(paddle.max(x).item()) + 1
    return paddle.bincount(x, minlength=minlength)


def _cumsum(
    x: paddle.Tensor,
    axis: int | None = 0,
    dtype: paddle.dtype | None = None,
) -> paddle.Tensor:
    """Implement cumulative summation."""
    return paddle.cumsum(x, axis=axis, dtype=dtype)


def _flexible_bincount(x: paddle.Tensor) -> paddle.Tensor:
    """Similar to `_bincount`, but works also with tensors that do not contain continuous values."""
    x = x - x.min()
    unique_x = paddle.unique(x)
    output = _bincount(x, minlength=int(paddle.max(unique_x).item()) + 1)
    return output[unique_x]


def allclose(tensor1: paddle.Tensor, tensor2: paddle.Tensor) -> bool:
    """Wrap paddle.allclose to be robust towards dtype difference."""
    if tensor1.dtype != tensor2.dtype:
        tensor2 = tensor2.cast(tensor1.dtype)
    return bool(paddle.allclose(tensor1, tensor2).item())


def interp(
    x: paddle.Tensor, xp: paddle.Tensor, fp: paddle.Tensor
) -> paddle.Tensor:
    """One-dimensional linear interpolation for monotonically increasing sample points.

    Args:
        x: x-coordinates where to evaluate the interpolated values
        xp: x-coordinates of the data points, must be increasing
        fp: y-coordinates of the data points, same length as xp
    """
    m = (fp[1:] - fp[:-1]) / (xp[1:] - xp[:-1])
    b = fp[:-1] - m * xp[:-1]
    indices = (
        paddle.sum(
            paddle.greater_equal(x.unsqueeze(1), xp.unsqueeze(0)), axis=1
        )
        - 1
    )
    indices = paddle.clip(indices, 0, len(m) - 1)
    return m[indices] * x + b[indices]
