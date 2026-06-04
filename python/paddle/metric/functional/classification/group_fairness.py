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

from typing_extensions import Literal

import paddle
from paddle.metric.functional.classification.stat_scores import (
    _binary_stat_scores_arg_validation,
    _binary_stat_scores_format,
    _binary_stat_scores_tensor_validation,
    _binary_stat_scores_update,
)
from paddle.metric.utils import rank_zero_warn
from paddle.metric.utils.compute import _safe_divide
from paddle.metric.utils.data import _flexible_bincount


def _groups_validation(groups: paddle.Tensor, num_groups: int) -> None:
    """Validate groups tensor.

    - The largest number in the tensor should not be larger than the number of groups. The group identifiers should
    be ``0, 1, ..., (num_groups - 1)``.
    - The group tensor should be dtype long.

    """
    if paddle.max(groups) > num_groups:
        raise ValueError(
            f"The largest number in the groups tensor is {paddle.max(groups)}, which is larger than the specified",
            f"number of groups {num_groups}. The group identifiers should be ``0, 1, ..., (num_groups - 1)``.",
        )
    if groups.dtype != paddle.long:
        raise ValueError(
            f"Expected dtype of argument groups to be long, not {groups.dtype}."
        )


def _groups_format(groups: paddle.Tensor) -> paddle.Tensor:
    """Reshape groups to correspond to preds and target."""
    return groups.reshape(groups.shape[0], -1)


def _binary_groups_stat_scores(
    preds: paddle.Tensor,
    target: paddle.Tensor,
    groups: paddle.Tensor,
    num_groups: int,
    threshold: float = 0.5,
    ignore_index: int | None = None,
    validate_args: bool = True,
) -> list[tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.Tensor]]:
    """Compute the true/false positives and true/false negatives rates for binary classification by group.

    Related to `Type I and Type II errors`_.

    """
    if validate_args:
        _binary_stat_scores_arg_validation(threshold, "global", ignore_index)
        _binary_stat_scores_tensor_validation(
            preds, target, "global", ignore_index
        )
        _groups_validation(groups, num_groups)
    preds, target = _binary_stat_scores_format(
        preds, target, threshold, ignore_index
    )
    groups = _groups_format(groups)
    indexes, indices = paddle.sort(groups.squeeze(1))
    preds = preds[indices]
    target = target[indices]
    split_sizes = _flexible_bincount(indexes).detach().cpu().tolist()
    group_preds = list(paddle.split(preds, split_sizes, axis=0))
    group_target = list(paddle.split(target, split_sizes, axis=0))
    return [
        _binary_stat_scores_update(group_p, group_t)
        for group_p, group_t in zip(group_preds, group_target)
    ]


def _groups_reduce(
    group_stats: list[
        tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.Tensor]
    ],
) -> dict[str, paddle.Tensor]:
    """Compute rates for all the group statistics."""
    return {
        f"group_{group}": (paddle.stack(stats) / paddle.stack(stats).sum())
        for group, stats in enumerate(group_stats)
    }


def _groups_stat_transform(
    group_stats: list[
        tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.Tensor]
    ],
) -> dict[str, paddle.Tensor]:
    """Transform group statistics by creating a tensor for each statistic."""
    return {
        "tp": paddle.stack([stat[0] for stat in group_stats]),
        "fp": paddle.stack([stat[1] for stat in group_stats]),
        "tn": paddle.stack([stat[2] for stat in group_stats]),
        "fn": paddle.stack([stat[3] for stat in group_stats]),
    }


def binary_groups_stat_rates(
    preds: paddle.Tensor,
    target: paddle.Tensor,
    groups: paddle.Tensor,
    num_groups: int,
    threshold: float = 0.5,
    ignore_index: int | None = None,
    validate_args: bool = True,
) -> dict[str, paddle.Tensor]:
    """Compute the true/false positives and true/false negatives rates for binary classification by group.

    Related to `Type I and Type II errors`_.

    Accepts the following input tensors:

    - ``preds`` (int or float tensor): ``(N, ...)``. If preds is a floating point tensor with values outside
      [0,1] range we consider the input to be logits and will auto apply sigmoid per element. Additionally,
      we convert to int tensor with thresholding using the value in ``threshold``.
    - ``target`` (int tensor): ``(N, ...)``.
    - ``groups`` (int tensor): ``(N, ...)``. The group identifiers should be ``0, 1, ..., (num_groups - 1)``.

    The additional dimensions are flatted along the batch dimension.

    Args:
        preds: Tensor with predictions.
        target: Tensor with true labels.
        groups: Tensor with group identifiers. The group identifiers should be ``0, 1, ..., (num_groups - 1)``.
        num_groups: The number of groups.
        threshold: Threshold for transforming probability to binary {0,1} predictions.
        ignore_index:
            Specifies a target value that is ignored and does not contribute to the metric calculation
        validate_args: bool indicating if input arguments and tensors should be validated for correctness.
            Set to ``False`` for faster computations.

    Returns:
        The metric returns a dict with a group identifier as key and a tensor with the tp, fp, tn and fn rates as value.

    Example (preds is int tensor):
        >>> from paddle.metric.functional.classification import binary_groups_stat_rates
        >>> target = paddle.to_tensor([0, 1, 0, 1, 0, 1])
        >>> preds = paddle.to_tensor([0, 1, 0, 1, 0, 1])
        >>> groups = paddle.to_tensor([0, 1, 0, 1, 0, 1])
        >>> binary_groups_stat_rates(preds, target, groups, 2)
        {'group_0': tensor([0., 0., 1., 0.]), 'group_1': tensor([1., 0., 0., 0.])}

    Example (preds is float tensor):
        >>> from paddle.metric.functional.classification import binary_groups_stat_rates
        >>> target = paddle.to_tensor([0, 1, 0, 1, 0, 1])
        >>> preds = paddle.to_tensor([0.11, 0.84, 0.22, 0.73, 0.33, 0.92])
        >>> groups = paddle.to_tensor([0, 1, 0, 1, 0, 1])
        >>> binary_groups_stat_rates(preds, target, groups, 2)
        {'group_0': tensor([0., 0., 1., 0.]), 'group_1': tensor([1., 0., 0., 0.])}

    """
    group_stats = _binary_groups_stat_scores(
        preds,
        target,
        groups,
        num_groups,
        threshold,
        ignore_index,
        validate_args,
    )
    return _groups_reduce(group_stats)


def _compute_binary_demographic_parity(
    tp: paddle.Tensor, fp: paddle.Tensor, tn: paddle.Tensor, fn: paddle.Tensor
) -> dict[str, paddle.Tensor]:
    """Compute demographic parity based on the binary stats."""
    pos_rates = _safe_divide(tp + fp, tp + fp + tn + fn)
    min_pos_rate_id = paddle.argmin(pos_rates)
    max_pos_rate_id = paddle.argmax(pos_rates)
    return {
        f"DP_{min_pos_rate_id}_{max_pos_rate_id}": _safe_divide(
            pos_rates[min_pos_rate_id], pos_rates[max_pos_rate_id]
        )
    }


def demographic_parity(
    preds: paddle.Tensor,
    groups: paddle.Tensor,
    threshold: float = 0.5,
    ignore_index: int | None = None,
    validate_args: bool = True,
) -> dict[str, paddle.Tensor]:
    """`Demographic parity`_ compares the positivity rates between all groups.

    If more than two groups are present, the disparity between the lowest and highest group is reported. The lowest
    positivity rate is divided by the highest, so a lower value means more discrimination against the numerator.
    In the results this is also indicated as the key of dict is DP_{identifier_low_group}_{identifier_high_group}.

    .. math::
        \\text{DP} = \\dfrac{\\min_a PR_a}{\\max_a PR_a}.

    where :math:`\\text{PR}` represents the positivity rate for group :math:`\\text{a}`.

    Accepts the following input tensors:

    - ``preds`` (int or float tensor): ``(N, ...)``. If preds is a floating point tensor with values outside
      [0,1] range we consider the input to be logits and will auto apply sigmoid per element. Additionally,
      we convert to int tensor with thresholding using the value in ``threshold``.
    - ``groups`` (int tensor): ``(N, ...)``. The group identifiers should be ``0, 1, ..., (num_groups - 1)``.
    - ``target`` (int tensor): ``(N, ...)``.

    The additional dimensions are flatted along the batch dimension.

    Args:
        preds: Tensor with predictions.
        groups: Tensor with group identifiers. The group identifiers should be ``0, 1, ..., (num_groups - 1)``.
        threshold: Threshold for transforming probability to binary {0,1} predictions.
        ignore_index:
            Specifies a target value that is ignored and does not contribute to the metric calculation
        validate_args: bool indicating if input arguments and tensors should be validated for correctness.
            Set to ``False`` for faster computations.

    Returns:
        The metric returns a dict where the key identifies the group with the lowest and highest positivity rates
        as follows: DP_{identifier_low_group}_{identifier_high_group}. The value is a tensor with the DP rate.

    Example (preds is int tensor):
        >>> from paddle.metric.functional.classification import demographic_parity
        >>> preds = paddle.to_tensor([0, 1, 0, 1, 0, 1])
        >>> groups = paddle.to_tensor([0, 1, 0, 1, 0, 1])
        >>> demographic_parity(preds, groups)
        {'DP_0_1': tensor(0.)}

    Example (preds is float tensor):
        >>> from paddle.metric.functional.classification import demographic_parity
        >>> preds = paddle.to_tensor([0.11, 0.84, 0.22, 0.73, 0.33, 0.92])
        >>> groups = paddle.to_tensor([0, 1, 0, 1, 0, 1])
        >>> demographic_parity(preds, groups)
        {'DP_0_1': tensor(0.)}

    """
    num_groups = paddle.unique(groups).shape[0]
    target = paddle.zeros(preds.shape)
    group_stats = _binary_groups_stat_scores(
        preds,
        target,
        groups,
        num_groups,
        threshold,
        ignore_index,
        validate_args,
    )
    transformed_group_stats = _groups_stat_transform(group_stats)
    return _compute_binary_demographic_parity(**transformed_group_stats)


def _compute_binary_equal_opportunity(
    tp: paddle.Tensor, fp: paddle.Tensor, tn: paddle.Tensor, fn: paddle.Tensor
) -> dict[str, paddle.Tensor]:
    """Compute equal opportunity based on the binary stats."""
    true_pos_rates = _safe_divide(tp, tp + fn)
    min_pos_rate_id = paddle.argmin(true_pos_rates)
    max_pos_rate_id = paddle.argmax(true_pos_rates)
    return {
        f"EO_{min_pos_rate_id}_{max_pos_rate_id}": _safe_divide(
            true_pos_rates[min_pos_rate_id], true_pos_rates[max_pos_rate_id]
        )
    }


def equal_opportunity(
    preds: paddle.Tensor,
    target: paddle.Tensor,
    groups: paddle.Tensor,
    threshold: float = 0.5,
    ignore_index: int | None = None,
    validate_args: bool = True,
) -> dict[str, paddle.Tensor]:
    """`Equal opportunity`_ compares the true positive rates between all groups.

    If more than two groups are present, the disparity between the lowest and highest group is reported. The lowest
    true positive rate is divided by the highest, so a lower value means more discrimination against the numerator.
    In the results this is also indicated as the key of dict is EO_{identifier_low_group}_{identifier_high_group}.

    .. math::
        \\text{DP} = \\dfrac{\\min_a TPR_a}{\\max_a TPR_a}.

    where :math:`\\text{TPR}` represents the true positives rate for group :math:`\\text{a}`.

    Accepts the following input tensors:

    - ``preds`` (int or float tensor): ``(N, ...)``. If preds is a floating point tensor with values outside
      [0,1] range we consider the input to be logits and will auto apply sigmoid per element. Additionally,
      we convert to int tensor with thresholding using the value in ``threshold``.
    - ``target`` (int tensor): ``(N, ...)``.
    - ``groups`` (int tensor): ``(N, ...)``. The group identifiers should be ``0, 1, ..., (num_groups - 1)``.

    The additional dimensions are flatted along the batch dimension.

    Args:
        preds: Tensor with predictions.
        target: Tensor with true labels.
        groups: Tensor with group identifiers. The group identifiers should be ``0, 1, ..., (num_groups - 1)``.
        threshold: Threshold for transforming probability to binary {0,1} predictions.
        ignore_index:
            Specifies a target value that is ignored and does not contribute to the metric calculation
        validate_args: bool indicating if input arguments and tensors should be validated for correctness.
            Set to ``False`` for faster computations.

    Returns:
        The metric returns a dict where the key identifies the group with the lowest and highest true positives rates
        as follows: EO_{identifier_low_group}_{identifier_high_group}. The value is a tensor with the EO rate.

    Example (preds is int tensor):
        >>> from paddle.metric.functional.classification import equal_opportunity
        >>> target = paddle.to_tensor([0, 1, 0, 1, 0, 1])
        >>> preds = paddle.to_tensor([0, 1, 0, 1, 0, 1])
        >>> groups = paddle.to_tensor([0, 1, 0, 1, 0, 1])
        >>> equal_opportunity(preds, target, groups)
        {'EO_0_1': tensor(0.)}

    Example (preds is float tensor):
        >>> from paddle.metric.functional.classification import equal_opportunity
        >>> target = paddle.to_tensor([0, 1, 0, 1, 0, 1])
        >>> preds = paddle.to_tensor([0.11, 0.84, 0.22, 0.73, 0.33, 0.92])
        >>> groups = paddle.to_tensor([0, 1, 0, 1, 0, 1])
        >>> equal_opportunity(preds, target, groups)
        {'EO_0_1': tensor(0.)}

    """
    num_groups = paddle.unique(groups).shape[0]
    group_stats = _binary_groups_stat_scores(
        preds,
        target,
        groups,
        num_groups,
        threshold,
        ignore_index,
        validate_args,
    )
    transformed_group_stats = _groups_stat_transform(group_stats)
    return _compute_binary_equal_opportunity(**transformed_group_stats)


def binary_fairness(
    preds: paddle.Tensor,
    target: paddle.Tensor,
    groups: paddle.Tensor,
    task: Literal["demographic_parity", "equal_opportunity", "all"] = "all",
    threshold: float = 0.5,
    ignore_index: int | None = None,
    validate_args: bool = True,
) -> dict[str, paddle.Tensor]:
    """Compute either `Demographic parity`_ and `Equal opportunity`_ ratio for binary classification problems.

    This is done by setting the ``task`` argument to either ``'demographic_parity'``, ``'equal_opportunity'``
    or ``all``. See the documentation of
    :func:`~paddle.metric.functional.classification.demographic_parity`
    and :func:`~paddle.metric.functional.classification.equal_opportunity` for the specific details of
    each argument influence and examples.

    Args:
        preds: Tensor with predictions.
        target: Tensor with true labels (not required for demographic_parity).
        groups: Tensor with group identifiers. The group identifiers should be ``0, 1, ..., (num_groups - 1)``.
        task: The task to compute. Can be either ``demographic_parity`` or ``equal_opportunity`` or ``all``.
        threshold: Threshold for transforming probability to binary {0,1} predictions.
        ignore_index:
            Specifies a target value that is ignored and does not contribute to the metric calculation
        validate_args: bool indicating if input arguments and tensors should be validated for correctness.
            Set to ``False`` for faster computations.

    """
    if task not in ["demographic_parity", "equal_opportunity", "all"]:
        raise ValueError(
            f"Expected argument `task` to either be ``demographic_parity``,``equal_opportunity`` or ``all`` but got {task}."
        )
    if task == "demographic_parity":
        if target is not None:
            rank_zero_warn(
                "The task demographic_parity does not require a target.",
                UserWarning,
            )
        target = paddle.zeros(preds.shape)
    num_groups = paddle.unique(groups).shape[0]
    group_stats = _binary_groups_stat_scores(
        preds,
        target,
        groups,
        num_groups,
        threshold,
        ignore_index,
        validate_args,
    )
    transformed_group_stats = _groups_stat_transform(group_stats)
    if task == "demographic_parity":
        return _compute_binary_demographic_parity(**transformed_group_stats)
    if task == "equal_opportunity":
        return _compute_binary_equal_opportunity(**transformed_group_stats)
    if task == "all":
        return {
            **_compute_binary_demographic_parity(**transformed_group_stats),
            **_compute_binary_equal_opportunity(**transformed_group_stats),
        }
    return None
