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
from paddle.metric.functional.regression.r2 import (
    _r2_score_compute,
    _r2_score_update,
)
from paddle.metric.metric import Metric
from paddle.metric.utils.imports import _MATPLOTLIB_AVAILABLE

if TYPE_CHECKING:
    from collections.abc import Sequence

    from paddle.metric.utils.plot import _AX_TYPE, _PLOT_OUT_TYPE

if not _MATPLOTLIB_AVAILABLE:
    __doctest_skip__ = ["R2Score.plot"]


class R2Score(Metric):
    """Compute r2 score also known as `R2 Score_Coefficient Determination`_.

    .. math:: R^2 = 1 - \\frac{SS_{res}}{SS_{tot}}

    where :math:`SS_{res}=\\sum_i (y_i - f(x_i))^2` is the sum of residual squares, and
    :math:`SS_{tot}=\\sum_i (y_i - \\bar{y})^2` is total sum of squares. Can also calculate
    adjusted r2 score given by

    .. math:: R^2_{adj} = 1 - \\frac{(1-R^2)(n-1)}{n-k-1}

    where the parameter :math:`k` (the number of independent regressors) should be provided as the `adjusted` argument.
    The score is only proper defined when :math:`SS_{tot}\\neq 0`, which can happen for near constant targets. In this
    case a score of 0 is returned. By definition the score is bounded between :math:`-inf` and 1.0, with 1.0 indicating
    perfect prediction, 0 indicating constant prediction and negative values indicating worse than constant prediction.

    As input to ``forward`` and ``update`` the metric accepts the following input:

    - ``preds`` (:class:`~paddle.Tensor`): Predictions from model in float tensor with shape ``(N,)``
      or ``(N, M)`` (multioutput)
    - ``target`` (:class:`~paddle.Tensor`): Ground truth values in float tensor with shape ``(N,)``
      or ``(N, M)`` (multioutput)

    As output of ``forward`` and ``compute`` the metric returns the following output:

    - ``r2score`` (:class:`~paddle.Tensor`): A tensor with the r2 score(s)

    In the case of multioutput, as default the variances will be uniformly averaged over the additional dimensions.
    Please see argument ``multioutput`` for changing this behavior.

    Args:
        num_outputs: Number of outputs in multioutput setting
        adjusted: number of independent regressors for calculating adjusted r2 score.
        multioutput: Defines aggregation in the case of multiple output scores. Can be one of the following strings:

            * ``'raw_values'`` returns full set of scores
            * ``'uniform_average'`` scores are uniformly averaged
            * ``'variance_weighted'`` scores are weighted by their individual variances
        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    .. warning::
        Argument ``num_outputs`` in ``R2Score`` has been deprecated because it is no longer necessary and will be
        removed in v1.6.0 of TorchMetrics. The number of outputs is now automatically inferred from the shape
        of the input tensors.

    Raises:
        ValueError:
            If ``adjusted`` parameter is not an integer larger or equal to 0.
        ValueError:
            If ``multioutput`` is not one of ``"raw_values"``, ``"uniform_average"`` or ``"variance_weighted"``.

    Example (single output):
        >>> from paddle import tensor
        >>> from paddle.metric.regression import R2Score
        >>> target = tensor([3, -0.5, 2, 7])
        >>> preds = tensor([2.5, 0.0, 2, 8])
        >>> r2score = R2Score()
        >>> r2score(preds, target)
        tensor(0.9486)

    Example (multioutput):
        >>> from paddle import tensor
        >>> from paddle.metric.regression import R2Score
        >>> target = tensor([[0.5, 1], [-1, 1], [7, -6]])
        >>> preds = tensor([[0, 2], [-1, 2], [8, -5]])
        >>> r2score = R2Score(multioutput='raw_values')
        >>> r2score(preds, target)
        tensor([0.9654, 0.9082])

    """

    is_differentiable: bool = True
    higher_is_better: bool = True
    full_state_update: bool = False
    plot_upper_bound: float = 1.0
    sum_squared_error: Tensor
    sum_error: Tensor
    residual: Tensor
    total: Tensor

    def __init__(
        self,
        adjusted: int = 0,
        multioutput: str = "uniform_average",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if adjusted < 0 or not isinstance(adjusted, int):
            raise ValueError(
                "`adjusted` parameter should be an integer larger or equal to 0."
            )
        self.adjusted = adjusted
        allowed_multioutput = (
            "raw_values",
            "uniform_average",
            "variance_weighted",
        )
        if multioutput not in allowed_multioutput:
            raise ValueError(
                f"Invalid input to argument `multioutput`. Choose one of the following: {allowed_multioutput}"
            )
        self.multioutput = multioutput
        self.declare(
            "sum_squared_error",
            default=paddle.tensor(0.0),
            dist_reduce_fn="sum",
        )
        self.declare(
            "sum_error", default=paddle.tensor(0.0), dist_reduce_fn="sum"
        )
        self.declare(
            "residual", default=paddle.tensor(0.0), dist_reduce_fn="sum"
        )
        self.declare("total", default=paddle.tensor(0), dist_reduce_fn="sum")

    def update(self, preds: paddle.Tensor, target: paddle.Tensor) -> None:
        """Update state with predictions and targets."""
        sum_squared_error, sum_error, residual, total = _r2_score_update(
            preds, target
        )
        self.sum_squared_error = self.sum_squared_error + sum_squared_error
        self.sum_error = self.sum_error + sum_error
        self.residual = self.residual + residual
        self.total = self.total + total

    def compute(self) -> paddle.Tensor:
        """Compute r2 score over the metric states."""
        return _r2_score_compute(
            self.sum_squared_error,
            self.sum_error,
            self.residual,
            self.total,
            self.adjusted,
            self.multioutput,
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
            >>> from paddle.metric.regression import R2Score
            >>> metric = R2Score()
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
            >>> from paddle.metric.regression import R2Score
            >>> metric = R2Score()
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
