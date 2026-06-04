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

"""Check utilities for paddle.metric."""

import os
from collections.abc import Mapping, Sequence
from functools import partial
from time import perf_counter
from typing import Any, no_type_check
from unittest.mock import Mock

import paddle

_DOCTEST_DOWNLOAD_TIMEOUT = int(os.environ.get("DOCTEST_DOWNLOAD_TIMEOUT", 120))
_SKIP_SLOW_DOCTEST = bool(os.environ.get("SKIP_SLOW_DOCTEST", 0))


def _check_for_empty_tensors(
    preds: paddle.Tensor, target: paddle.Tensor
) -> bool:
    return preds.numel() == target.numel() == 0


def _check_same_shape(preds: paddle.Tensor, target: paddle.Tensor) -> None:
    """Check that predictions and target have the same shape, else raise error."""
    if preds.shape != target.shape:
        raise RuntimeError(
            f"Predictions and targets are expected to have the same shape, but got {preds.shape} and {target.shape}."
        )


def _check_retrieval_functional_inputs(
    preds: paddle.Tensor,
    target: paddle.Tensor,
    allow_non_binary_target: bool = False,
) -> tuple[paddle.Tensor, paddle.Tensor]:
    """Check ``preds`` and ``target`` tensors are of the same shape and of the correct data type."""
    if preds.shape != target.shape:
        raise ValueError("`preds` and `target` must be of the same shape")
    if preds.numel() == 0 or preds.ndim == 0:
        raise ValueError(
            "`preds` and `target` must be non-empty and non-scalar tensors"
        )
    return _check_retrieval_target_and_prediction_types(
        preds, target, allow_non_binary_target=allow_non_binary_target
    )


def _check_retrieval_inputs(
    indexes: paddle.Tensor,
    preds: paddle.Tensor,
    target: paddle.Tensor,
    allow_non_binary_target: bool = False,
    ignore_index: int | None = None,
) -> tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor]:
    """Check ``indexes``, ``preds`` and ``target`` tensors are of the same shape and of the correct data type."""
    if indexes.shape != preds.shape or preds.shape != target.shape:
        raise ValueError(
            "`indexes`, `preds` and `target` must be of the same shape"
        )
    if indexes.dtype != paddle.int64:
        raise ValueError("`indexes` must be a tensor of long integers")
    if ignore_index is not None:
        valid_positions = target != ignore_index
        indexes, preds, target = (
            indexes[valid_positions],
            preds[valid_positions],
            target[valid_positions],
        )
    if indexes.numel() == 0 or indexes.ndim == 0:
        raise ValueError(
            "`indexes`, `preds` and `target` must be non-empty and non-scalar tensors"
        )
    preds, target = _check_retrieval_target_and_prediction_types(
        preds, target, allow_non_binary_target=allow_non_binary_target
    )
    return indexes.cast("int64").flatten(), preds, target


def _check_retrieval_target_and_prediction_types(
    preds: paddle.Tensor,
    target: paddle.Tensor,
    allow_non_binary_target: bool = False,
) -> tuple[paddle.Tensor, paddle.Tensor]:
    """Check ``preds`` and ``target`` tensors are of the same shape and of the correct data type."""
    if (
        target.dtype not in (paddle.bool, paddle.int64, paddle.int32)
        and not target.is_floating_point()
    ):
        raise ValueError(
            "`target` must be a tensor of booleans, integers or floats"
        )
    if not preds.is_floating_point():
        raise ValueError("`preds` must be a tensor of floats")
    if not allow_non_binary_target and (target.max() > 1 or target.min() < 0):
        raise ValueError("`target` must contain `binary` values")
    target = (
        target.cast("float32")
        if target.is_floating_point()
        else target.cast("int64")
    )
    preds = preds.cast("float32")
    return preds.flatten(), target.flatten()


def _allclose_recursive(res1: Any, res2: Any, atol: float = 1e-06) -> bool:
    """Recursively asserting that two results are within a certain tolerance."""
    if isinstance(res1, paddle.Tensor):
        return bool(paddle.allclose(res1, res2, atol=atol).item())
    if isinstance(res1, str):
        return res1 == res2
    if isinstance(res1, Sequence):
        return all(_allclose_recursive(r1, r2) for r1, r2 in zip(res1, res2))
    if isinstance(res1, Mapping):
        return all(_allclose_recursive(res1[k], res2[k]) for k in res1)
    return res1 == res2


def is_overridden(method_name: str, instance: object, parent: object) -> bool:
    """Check if a method has been overridden by an instance compared to its parent class."""
    instance_attr = getattr(instance, method_name, None)
    if instance_attr is None:
        return False
    if hasattr(instance_attr, "__wrapped__"):
        instance_attr = instance_attr.__wrapped__
    if isinstance(instance_attr, Mock):
        instance_attr = instance_attr._mock_wraps
    elif isinstance(instance_attr, partial):
        instance_attr = instance_attr.func
    if instance_attr is None:
        return False
    parent_attr = getattr(parent, method_name, None)
    if parent_attr is None:
        raise ValueError("The parent should define the method")
    return instance_attr.__code__ != parent_attr.__code__


@no_type_check
def check_forward_full_state_property(
    metric_class: Any,
    init_args: dict[str, Any] | None = None,
    input_args: dict[str, Any] | None = None,
    num_update_to_compare: Sequence[int] = [10, 100, 1000],
    reps: int = 5,
) -> None:
    """Check if the ``full_state_update`` property can safely be set to ``False``.

    Args:
        metric_class: metric class object that should be checked
        init_args: dict containing arguments for initializing the metric class
        input_args: dict containing arguments to pass to ``forward``
        num_update_to_compare: list of integers for how many steps to compare over
        reps: number of repetitions of speedup test
    """
    init_args = init_args or {}
    input_args = input_args or {}

    class FullState(metric_class):
        full_state_update = True

    class PartState(metric_class):
        full_state_update = False

    fullstate = FullState(**init_args)
    partstate = PartState(**init_args)
    equal = True
    try:
        for _ in range(num_update_to_compare[0]):
            equal = equal & _allclose_recursive(
                fullstate(**input_args), partstate(**input_args)
            )
    except RuntimeError:
        equal = False
    res1 = fullstate.compute()
    try:
        res2 = partstate.compute()
    except RuntimeError:
        equal = False
    equal = equal & _allclose_recursive(res1, res2)
    if not equal:
        print("Recommended setting `full_state_update=True`")
        return
    res = paddle.zeros([2, len(num_update_to_compare), reps])
    for i, metric in enumerate([fullstate, partstate]):
        for j, t in enumerate(num_update_to_compare):
            for r in range(reps):
                start = perf_counter()
                for _ in range(t):
                    _ = metric(**input_args)
                end = perf_counter()
                res[i, j, r] = end - start
                metric.reset()
    mean = res.mean(axis=-1)
    std = res.std(axis=-1)
    for t in range(len(num_update_to_compare)):
        print(
            f"Full state for {num_update_to_compare[t]} steps took: {mean[0, t]:.3f}+-{std[0, t]:.3f}"
        )
        print(
            f"Partial state for {num_update_to_compare[t]} steps took: {mean[1, t]:.3f}+-{std[1, t]:.3f}"
        )
    faster = (mean[1, -1] < mean[0, -1]).item()
    print(f"Recommended setting `full_state_update={not faster}`")
