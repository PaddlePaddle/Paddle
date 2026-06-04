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

"""Distributed utilities for paddle.metric."""

from typing import Any

from typing_extensions import Literal

import paddle


def reduce(
    x: paddle.Tensor,
    reduction: Literal["elementwise_mean", "sum", "none"] | None,
) -> paddle.Tensor:
    """Reduces a given tensor by a given reduction method.

    Args:
        x: the tensor to reduce
        reduction: reduction method ('elementwise_mean', 'none', 'sum')

    Returns:
        reduced Tensor
    """
    if reduction == "elementwise_mean":
        return paddle.mean(x)
    if reduction == "none" or reduction is None:
        return x
    if reduction == "sum":
        return paddle.sum(x)
    raise ValueError("Reduction parameter unknown.")


def class_reduce(
    num: paddle.Tensor,
    denom: paddle.Tensor,
    weights: paddle.Tensor,
    class_reduction: Literal["micro", "macro", "weighted", "none"]
    | None = "none",
) -> paddle.Tensor:
    """Reduce classification metrics of the form ``num / denom * weights``.

    Args:
        num: numerator tensor
        denom: denominator tensor
        weights: weights for each class
        class_reduction: reduction method for multiclass problems

    Raises:
        ValueError: If ``class_reduction`` is not a valid option.
    """
    valid_reduction = ("micro", "macro", "weighted", "none", None)
    fraction = (
        paddle.sum(num) / paddle.sum(denom)
        if class_reduction == "micro"
        else num / denom
    )
    # Replace NaN with 0
    fraction = paddle.where(
        paddle.isnan(fraction), paddle.zeros_like(fraction), fraction
    )
    if class_reduction == "micro":
        return fraction
    if class_reduction == "macro":
        return paddle.mean(fraction)
    if class_reduction == "weighted":
        return paddle.sum(
            fraction * (weights.cast("float32") / paddle.sum(weights))
        )
    if class_reduction == "none" or class_reduction is None:
        return fraction
    raise ValueError(
        f"Reduction parameter {class_reduction} unknown. Choose between: {valid_reduction}"
    )


def _simple_gather_all_tensors(
    result: paddle.Tensor, group: Any, world_size: int
) -> list[paddle.Tensor]:
    with paddle.no_grad():
        gathered_result = [paddle.zeros_like(result) for _ in range(world_size)]
        paddle.distributed.all_gather(
            tensor_list=gathered_result, tensor=result, group=group
        )
    gathered_result[paddle.distributed.get_rank(group)] = result
    return gathered_result


def gather_all_tensors(
    result: paddle.Tensor, group: Any | None = None
) -> list[paddle.Tensor]:
    """Gather all tensors from several ddp processes onto a list that is broadcast to all processes.

    Works on tensors that have the same number of dimensions, but where each dimension may differ.
    Tensors are padded, gathered and then trimmed to secure equal workload for all processes.

    Args:
        result: the value to sync
        group: the process group to gather results from. Defaults to all processes (world)

    Return:
        list with size equal to the process group
    """
    if group is None:
        group = paddle.distributed.new_group()
    result = result.contiguous()
    world_size = paddle.distributed.get_world_size(group)
    paddle.distributed.barrier(group=group)
    if result.ndim == 0:
        return _simple_gather_all_tensors(result, group, world_size)
    local_size = paddle.to_tensor(result.shape, dtype="int64")
    local_sizes = [paddle.zeros_like(local_size) for _ in range(world_size)]
    paddle.distributed.all_gather(
        tensor_list=local_sizes, tensor=local_size, group=group
    )
    max_size = paddle.stack(local_sizes).max(axis=0)
    all_sizes_equal = all(all(ls == max_size) for ls in local_sizes)
    if all_sizes_equal:
        return _simple_gather_all_tensors(result, group, world_size)
    with paddle.no_grad():
        pad_dims = []
        pad_by = (max_size - local_size).detach().cpu()
        for val in reversed(pad_by):
            pad_dims.append(0)
            pad_dims.append(int(val.item()))
        result_padded = paddle.nn.functional.pad(result, pad_dims)
        gathered_result = [
            paddle.zeros_like(result_padded) for _ in range(world_size)
        ]
        paddle.distributed.all_gather(
            tensor_list=gathered_result, tensor=result_padded, group=group
        )
        for idx, item_size in enumerate(local_sizes):
            slice_param = [slice(int(dim_size)) for dim_size in item_size]
            gathered_result[idx] = gathered_result[idx][tuple(slice_param)]
    gathered_result[paddle.distributed.get_rank(group)] = result
    return gathered_result
