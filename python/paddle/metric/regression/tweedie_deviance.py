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
from paddle.metric.functional.regression.tweedie_deviance import (
    _tweedie_deviance_score_compute,
    _tweedie_deviance_score_update,
)
from paddle.metric.metric import Metric
from paddle.metric.utils.imports import _MATPLOTLIB_AVAILABLE

if TYPE_CHECKING:
    from collections.abc import Sequence

    from paddle.metric.utils.plot import _AX_TYPE, _PLOT_OUT_TYPE

if not _MATPLOTLIB_AVAILABLE:
    __doctest_skip__ = ["TweedieDevianceScore.plot"]


class TweedieDevianceScore(Metric):
    """Compute the `Tweedie Deviance Score`_.

    .. math::
        deviance\\_score(\\hat{y},y) =
        \\begin{cases}
        (\\hat{y} - y)^2, & \\text{for }p=0\\\\
        2 * (y * log(\\frac{y}{\\hat{y}}) + \\hat{y} - y),  & \\text{for }p=1\\\\
        2 * (log(\\frac{\\hat{y}}{y}) + \\frac{y}{\\hat{y}} - 1),  & \\text{for }p=2\\\\
        2 * (\\frac{(max(y,0))^{2 - p}}{(1 - p)(2 - p)} - \\frac{y(\\hat{y})^{1 - p}}{1 - p} + \\frac{(
            \\hat{y})^{2 - p}}{2 - p}), & \\text{otherwise}
        \\end{cases}

    where :math:`y` is a tensor of targets values, :math:`\\hat{y}` is a tensor of predictions, and
    :math:`p` is the `power`.

    As input to ``forward`` and ``update`` the metric accepts the following input:

    - ``preds`` (:class:`~paddle.Tensor`): Predicted float tensor with shape ``(N,...)``
    - ``target`` (:class:`~paddle.Tensor`): Ground truth float tensor with shape ``(N,...)``

    As output of ``forward`` and ``compute`` the metric returns the following output:

    - ``deviance_score`` (:class:`~paddle.Tensor`): A tensor with the deviance score

    Args:
        power:

            - power < 0 : Extreme stable distribution. (Requires: preds > 0.)
            - power = 0 : Normal distribution. (Requires: targets and preds can be any real numbers.)
            - power = 1 : Poisson distribution. (Requires: targets >= 0 and y_pred > 0.)
            - 1 < p < 2 : Compound Poisson distribution. (Requires: targets >= 0 and preds > 0.)
            - power = 2 : Gamma distribution. (Requires: targets > 0 and preds > 0.)
            - power = 3 : Inverse Gaussian distribution. (Requires: targets > 0 and preds > 0.)
            - otherwise : Positive stable distribution. (Requires: targets > 0 and preds > 0.)

        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Example:
        >>> from paddle.metric.regression import TweedieDevianceScore
        >>> targets = paddle.to_tensor([1.0, 2.0, 3.0, 4.0])
        >>> preds = paddle.to_tensor([4.0, 3.0, 2.0, 1.0])
        >>> deviance_score = TweedieDevianceScore(power=2)
        >>> deviance_score(preds, targets)
        tensor(1.2083)

    """

    is_differentiable: bool = True
    higher_is_better = None
    full_state_update: bool = False
    plot_lower_bound: float = 0.0
    sum_deviance_score: Tensor
    num_observations: Tensor

    def __init__(self, power: float = 0.0, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if 0 < power < 1:
            raise ValueError(
                f"Deviance Score is not defined for power={power}."
            )
        self.power: float = power
        self.declare(
            "sum_deviance_score", paddle.tensor(0.0), dist_reduce_fn="sum"
        )
        self.declare("num_observations", paddle.tensor(0), dist_reduce_fn="sum")

    def update(self, preds: paddle.Tensor, targets: paddle.Tensor) -> None:
        """Update metric states with predictions and targets."""
        sum_deviance_score, num_observations = _tweedie_deviance_score_update(
            preds, targets, self.power
        )
        self.sum_deviance_score += sum_deviance_score
        self.num_observations += num_observations

    def compute(self) -> paddle.Tensor:
        """Compute metric."""
        return _tweedie_deviance_score_compute(
            self.sum_deviance_score, self.num_observations
        )

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
            >>> from paddle.metric.regression import TweedieDevianceScore
            >>> metric = TweedieDevianceScore()
            >>> metric.update(
            ...     randn(
            ...         10,
            ...     ),
            ...     randn(
            ...         10,
            ...     ),
            ... )
            >>> fig_, ax_ = metric.plot()

        .. plot::
            :scale: 75

            >>> from paddle import randn
            >>> # Example plotting multiple values
            >>> from paddle.metric.regression import TweedieDevianceScore
            >>> metric = TweedieDevianceScore()
            >>> values = []
            >>> for _ in range(10):
            ...     values.append(
            ...         metric(
            ...             randn(
            ...                 10,
            ...             ),
            ...             randn(
            ...                 10,
            ...             ),
            ...         )
            ...     )
            >>> fig, ax = metric.plot(values)

        """
        return self._plot(val, ax)
