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

from typing import TYPE_CHECKING, Any

from typing_extensions import Literal

import paddle
from paddle import Tensor
from paddle.metric.functional.classification.group_fairness import (
    _binary_groups_stat_scores,
    _compute_binary_demographic_parity,
    _compute_binary_equal_opportunity,
)
from paddle.metric.functional.classification.stat_scores import (
    _binary_stat_scores_arg_validation,
)
from paddle.metric.metric import Metric
from paddle.metric.utils import rank_zero_warn
from paddle.metric.utils.imports import _MATPLOTLIB_AVAILABLE

if TYPE_CHECKING:
    from collections.abc import Sequence

    from paddle.metric.utils.plot import _AX_TYPE, _PLOT_OUT_TYPE

if not _MATPLOTLIB_AVAILABLE:
    __doctest_skip__ = ["BinaryFairness.plot"]


class _AbstractGroupStatScores(Metric):
    """Create and update states for computing group stats tp, fp, tn and fn."""

    tp: Tensor
    fp: Tensor
    tn: Tensor
    fn: Tensor

    def _create_states(self, num_groups: int) -> None:
        default = lambda: paddle.zeros(num_groups, dtype=paddle.long)
        self.declare("tp", default(), dist_reduce_fn="sum")
        self.declare("fp", default(), dist_reduce_fn="sum")
        self.declare("tn", default(), dist_reduce_fn="sum")
        self.declare("fn", default(), dist_reduce_fn="sum")

    def _update_states(
        self,
        group_stats: list[
            tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.Tensor]
        ],
    ) -> None:
        for group, stats in enumerate(group_stats):
            tp, fp, tn, fn = stats
            self.tp[group] += tp
            self.fp[group] += fp
            self.tn[group] += tn
            self.fn[group] += fn


class BinaryGroupStatRates(_AbstractGroupStatScores):
    """Computes the true/false positives and true/false negatives rates for binary classification by group.

    Related to `Type I and Type II errors`_.

    Accepts the following input tensors:

    - ``preds`` (int or float tensor): ``(N, ...)``. If preds is a floating point tensor with values outside
      [0,1] range we consider the input to be logits and will auto apply sigmoid per element. Additionally,
      we convert to int tensor with thresholding using the value in ``threshold``.
    - ``target`` (int tensor): ``(N, ...)``.
    - ``groups`` (int tensor): ``(N, ...)``. The group identifiers should be ``0, 1, ..., (num_groups - 1)``.

    The additional dimensions are flatted along the batch dimension.

    Args:
        num_groups: The number of groups.
        threshold: Threshold for transforming probability to binary {0,1} predictions.
        ignore_index: Specifies a target value that is ignored and does not contribute to the metric calculation
        validate_args: bool indicating if input arguments and tensors should be validated for correctness.
            Set to ``False`` for faster computations.
        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Returns:
        The metric returns a dict with a group identifier as key and a tensor with the tp, fp, tn and fn rates as value.

    Example (preds is int tensor):
        >>> from paddle.metric.classification import BinaryGroupStatRates
        >>> target = paddle.to_tensor([0, 1, 0, 1, 0, 1])
        >>> preds = paddle.to_tensor([0, 1, 0, 1, 0, 1])
        >>> groups = paddle.to_tensor([0, 1, 0, 1, 0, 1])
        >>> metric = BinaryGroupStatRates(num_groups=2)
        >>> metric(preds, target, groups)
        {'group_0': tensor([0., 0., 1., 0.]), 'group_1': tensor([1., 0., 0., 0.])}

    Example (preds is float tensor):
        >>> from paddle.metric.classification import BinaryGroupStatRates
        >>> target = paddle.to_tensor([0, 1, 0, 1, 0, 1])
        >>> preds = paddle.to_tensor([0.11, 0.84, 0.22, 0.73, 0.33, 0.92])
        >>> groups = paddle.to_tensor([0, 1, 0, 1, 0, 1])
        >>> metric = BinaryGroupStatRates(num_groups=2)
        >>> metric(preds, target, groups)
        {'group_0': tensor([0., 0., 1., 0.]), 'group_1': tensor([1., 0., 0., 0.])}

    """

    is_differentiable: bool = False
    higher_is_better: bool = False
    full_state_update: bool = False
    plot_lower_bound: float = 0.0
    plot_upper_bound: float = 1.0

    def __init__(
        self,
        num_groups: int,
        threshold: float = 0.5,
        ignore_index: int | None = None,
        validate_args: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        if validate_args:
            _binary_stat_scores_arg_validation(
                threshold, "global", ignore_index
            )
        if not isinstance(num_groups, int) and num_groups < 2:
            raise ValueError(
                f"Expected argument `num_groups` to be an int larger than 1, but got {num_groups}"
            )
        self.num_groups = num_groups
        self.threshold = threshold
        self.ignore_index = ignore_index
        self.validate_args = validate_args
        self._create_states(self.num_groups)

    def update(
        self, preds: paddle.Tensor, target: paddle.Tensor, groups: paddle.Tensor
    ) -> None:
        """Update state with predictions, target and group identifiers.

        Args:
            preds: Tensor with predictions.
            target: Tensor with true labels.
            groups: Tensor with group identifiers. The group identifiers should be ``0, 1, ..., (num_groups - 1)``.

        """
        group_stats = _binary_groups_stat_scores(
            preds,
            target,
            groups,
            self.num_groups,
            self.threshold,
            self.ignore_index,
            self.validate_args,
        )
        self._update_states(group_stats)

    def compute(self) -> dict[str, paddle.Tensor]:
        """Compute tp, fp, tn and fn rates based on inputs passed in to ``update`` previously."""
        results = paddle.stack((self.tp, self.fp, self.tn, self.fn), axis=1)
        return {
            f"group_{i}": (group / group.sum())
            for i, group in enumerate(results)
        }


class BinaryFairness(_AbstractGroupStatScores):
    """Computes `Demographic parity`_ and `Equal opportunity`_ ratio for binary classification problems.

    Accepts the following input tensors:

    - ``preds`` (int or float tensor): ``(N, ...)``. If preds is a floating point tensor with values outside
      [0,1] range we consider the input to be logits and will auto apply sigmoid per element. Additionally,
      we convert to int tensor with thresholding using the value in ``threshold``.
    - ``groups`` (int tensor): ``(N, ...)``. The group identifiers should be ``0, 1, ..., (num_groups - 1)``.
    - ``target`` (int tensor): ``(N, ...)``.

    The additional dimensions are flatted along the batch dimension.

    This class computes the ratio between positivity rates and true positives rates for different groups.
    If more than two groups are present, the disparity between the lowest and highest group is reported.
    A disparity between positivity rates indicates a potential violation of demographic parity, and between
    true positive rates indicates a potential violation of equal opportunity.

    The lowest rate is divided by the highest, so a lower value means more discrimination against the numerator.
    In the results this is also indicated as the key of dict is {metric}_{identifier_low_group}_{identifier_high_group}.

    Args:
        num_groups: The number of groups.
        task: The task to compute. Can be either ``demographic_parity`` or ``equal_opportunity`` or ``all``.
        threshold: Threshold for transforming probability to binary {0,1} predictions.
        ignore_index: Specifies a target value that is ignored and does not contribute to the metric calculation
        validate_args: bool indicating if input arguments and tensors should be validated for correctness.
            Set to ``False`` for faster computations.
        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Returns:
        The metric returns a dict where the key identifies the metric and groups with the lowest and highest true
        positives rates as follows: {metric}__{identifier_low_group}_{identifier_high_group}.
        The value is a tensor with the disparity rate.

    Example (preds is int tensor):
        >>> from paddle.metric.classification import BinaryFairness
        >>> target = paddle.to_tensor([0, 1, 0, 1, 0, 1])
        >>> preds = paddle.to_tensor([0, 1, 0, 1, 0, 1])
        >>> groups = paddle.to_tensor([0, 1, 0, 1, 0, 1])
        >>> metric = BinaryFairness(2)
        >>> metric(preds, target, groups)
        {'DP_0_1': tensor(0.), 'EO_0_1': tensor(0.)}

    Example (preds is float tensor):
        >>> from paddle.metric.classification import BinaryFairness
        >>> target = paddle.to_tensor([0, 1, 0, 1, 0, 1])
        >>> preds = paddle.to_tensor([0.11, 0.84, 0.22, 0.73, 0.33, 0.92])
        >>> groups = paddle.to_tensor([0, 1, 0, 1, 0, 1])
        >>> metric = BinaryFairness(2)
        >>> metric(preds, target, groups)
        {'DP_0_1': tensor(0.), 'EO_0_1': tensor(0.)}

    """

    is_differentiable: bool = False
    higher_is_better: bool = False
    full_state_update: bool = False
    plot_lower_bound: float = 0.0
    plot_upper_bound: float = 1.0

    def __init__(
        self,
        num_groups: int,
        task: Literal["demographic_parity", "equal_opportunity", "all"] = "all",
        threshold: float = 0.5,
        ignore_index: int | None = None,
        validate_args: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        if task not in ["demographic_parity", "equal_opportunity", "all"]:
            raise ValueError(
                f"Expected argument `task` to either be ``demographic_parity``,``equal_opportunity`` or ``all`` but got {task}."
            )
        if validate_args:
            _binary_stat_scores_arg_validation(
                threshold, "global", ignore_index
            )
        if not isinstance(num_groups, int) and num_groups < 2:
            raise ValueError(
                f"Expected argument `num_groups` to be an int larger than 1, but got {num_groups}"
            )
        self.num_groups = num_groups
        self.task = task
        self.threshold = threshold
        self.ignore_index = ignore_index
        self.validate_args = validate_args
        self._create_states(self.num_groups)

    def update(
        self, preds: paddle.Tensor, target: paddle.Tensor, groups: paddle.Tensor
    ) -> None:
        """Update state with predictions, groups, and target.

        Args:
            preds: Tensor with predictions.
            target: Tensor with true labels.
            groups: Tensor with group identifiers. The group identifiers should be ``0, 1, ..., (num_groups - 1)``.

        """
        if self.task == "demographic_parity":
            if target is not None:
                rank_zero_warn(
                    "The task demographic_parity does not require a target.",
                    UserWarning,
                )
            target = paddle.zeros(preds.shape)
        group_stats = _binary_groups_stat_scores(
            preds,
            target,
            groups,
            self.num_groups,
            self.threshold,
            self.ignore_index,
            self.validate_args,
        )
        self._update_states(group_stats)

    def compute(self) -> dict[str, paddle.Tensor]:
        """Compute fairness criteria based on inputs passed in to ``update`` previously."""
        if self.task == "demographic_parity":
            return _compute_binary_demographic_parity(
                self.tp, self.fp, self.tn, self.fn
            )
        if self.task == "equal_opportunity":
            return _compute_binary_equal_opportunity(
                self.tp, self.fp, self.tn, self.fn
            )
        if self.task == "all":
            return {
                **_compute_binary_demographic_parity(
                    self.tp, self.fp, self.tn, self.fn
                ),
                **_compute_binary_equal_opportunity(
                    self.tp, self.fp, self.tn, self.fn
                ),
            }
        return None

    def plot(
        self,
        val: paddle.Tensor | Sequence[paddle.Tensor] | None = None,
        ax: _AX_TYPE | None = None,
    ) -> _PLOT_OUT_TYPE:
        """Plot a single or multiple values from the metric.

        Args:
            val: Either a single result from calling `metric.forward` or `metric.compute` or a list of these results.
                If no value is provided, will automatically call `metric.compute` and plot that result.
            ax: An matplotlib axis object. If provided will add plot to that axis

        Returns:
            Figure object and Axes object

        Raises:
            ModuleNotFoundError:
                If `matplotlib` is not installed

        .. plot::
            :scale: 75

            >>> from paddle import ones, rand, randint
            >>> # Example plotting a single value
            >>> from paddle.metric.classification import BinaryFairness
            >>> metric = BinaryFairness(2)
            >>> metric.update(rand(50), randint(2, (50,)), ones(50).long())
            >>> fig_, ax_ = metric.plot()

        .. plot::
            :scale: 75

            >>> from paddle import ones, rand, randint
            >>> # Example plotting multiple values
            >>> from paddle.metric.classification import BinaryFairness
            >>> metric = BinaryFairness(2)
            >>> values = []
            >>> for _ in range(10):
            ...     values.append(metric(rand(50), randint(2, (50,)), ones(50).long()))
            >>> fig_, ax_ = metric.plot(values)

        """
        return self._plot(val, ax)
