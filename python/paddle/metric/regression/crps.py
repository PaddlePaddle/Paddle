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

import paddle
from paddle import Tensor
from paddle.metric.functional.regression.crps import _crps_update
from paddle.metric.metric import Metric
from paddle.metric.utils.imports import _MATPLOTLIB_AVAILABLE

if TYPE_CHECKING:
    from collections.abc import Sequence

    from paddle.metric.utils.plot import _AX_TYPE, _PLOT_OUT_TYPE

if not _MATPLOTLIB_AVAILABLE:
    __doctest_skip__ = ["ContinuousRankedProbabilityScore.plot"]


class ContinuousRankedProbabilityScore(Metric):
    """Computes continuous ranked probability score.

    .. math::
        CRPS(F, y) = \\int_{-\\infty}^{\\infty} (F(x) - 1_{x \\geq y})^2 dx

    where :math:`F` is the predicted cumulative distribution function and :math:`y` is the true target. The metric is
    usually used to evaluate probabilistic regression models, such as forecasting models. A lower CRPS indicates a
    better forecast, meaning that forecasted probabilities are closer to the true observed values. CRPS can also be
    seen as a generalization of the brier score for non binary classification problems.

    As input to ``forward`` and ``update`` the metric accepts the following input:

    - ``preds`` (:class:`~paddle.Tensor`): Predicted float tensor with shape ``(N,d)``
    - ``target`` (:class:`~paddle.Tensor`): Ground truth float tensor with shape ``(N,d)``

    As output of ``forward`` and ``compute`` the metric returns the following output:

    - ``cosine_similarity`` (:class:`~paddle.Tensor`): A float tensor with the cosine similarity

    Args:
        reduction: how to reduce over the batch dimension using 'sum', 'mean' or 'none' (taking the individual scores)
        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Example:
        >>> from paddle import randn
        >>> from paddle.metric.regression import ContinuousRankedProbabilityScore
        >>> preds = randn(10, 5)
        >>> target = randn(10)
        >>> crps = ContinuousRankedProbabilityScore()
        >>> crps(preds, target)
        tensor(0.7731)

    """

    is_differentiable: bool = False
    higher_is_better: bool = False
    full_state_update: bool = False
    plot_lower_bound: float = 0.0
    score: Tensor
    total: Tensor

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.declare("score", default=paddle.zeros(1), dist_reduce_fn="sum")
        self.declare("total", default=paddle.zeros(1), dist_reduce_fn="sum")

    def update(self, preds: paddle.Tensor, target: paddle.Tensor) -> None:
        """Update state with predictions and targets.

        Args:
            preds: Predictions from model
            target: Ground truth values

        """
        batch_size, diff, ensemble_sum = _crps_update(preds, target)
        self.score += paddle.sum(diff - ensemble_sum)
        self.total += batch_size

    def compute(self) -> paddle.Tensor:
        """Compute the continuous ranked probability score over state."""
        return self.score / self.total

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
            Figure and Axes object

        Raises:
            ModuleNotFoundError:
                If `matplotlib` is not installed

        .. plot::
            :scale: 75

            >>> from paddle import randn
            >>> # Example plotting a single value
            >>> from paddle.metric.regression import ContinuousRankedProbabilityScore
            >>> metric = ContinuousRankedProbabilityScore()
            >>> metric.update(randn(10, 5), randn(10))
            >>> fig_, ax_ = metric.plot()

        .. plot::
            :scale: 75

            >>> from paddle import randn
            >>> # Example plotting multiple values
            >>> from paddle.metric.regression import ContinuousRankedProbabilityScore
            >>> metric = ContinuousRankedProbabilityScore()
            >>> values = []
            >>> for _ in range(10):
            ...     values.append(metric(randn(10, 5), randn(10)))
            >>> fig, ax = metric.plot(values)

        """
        return self._plot(val, ax)
