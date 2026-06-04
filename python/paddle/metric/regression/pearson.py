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
from paddle.metric.functional.regression.pearson import (
    _pearson_corrcoef_compute,
    _pearson_corrcoef_update,
)
from paddle.metric.metric import Metric
from paddle.metric.utils.imports import _MATPLOTLIB_AVAILABLE

if TYPE_CHECKING:
    from collections.abc import Sequence

    from paddle.metric.utils.plot import _AX_TYPE, _PLOT_OUT_TYPE

if not _MATPLOTLIB_AVAILABLE:
    __doctest_skip__ = ["PearsonCorrCoef.plot"]


def _final_aggregation(
    means_x: paddle.Tensor,
    means_y: paddle.Tensor,
    maxs_abs_x: paddle.Tensor,
    maxs_abs_y: paddle.Tensor,
    vars_x: paddle.Tensor,
    vars_y: paddle.Tensor,
    corrs_xy: paddle.Tensor,
    nbs: paddle.Tensor,
    eps: float = 1e-10,
) -> tuple[
    paddle.Tensor,
    paddle.Tensor,
    paddle.Tensor,
    paddle.Tensor,
    paddle.Tensor,
    paddle.Tensor,
    paddle.Tensor,
    paddle.Tensor,
]:
    """Aggregate the statistics from multiple devices.

    Formula taken from here: `Parallel algorithm for calculating variance
    <https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm>`_

    We use `eps` to avoid division by zero when `n1` and `n2` are both zero. Generally, the value of `eps` should not
    matter, as if `n1` and `n2` are both zero, all the states will also be zero.

    """
    if len(means_x) == 1:
        return (
            means_x[0],
            means_y[0],
            maxs_abs_x[0],
            maxs_abs_y[0],
            vars_x[0],
            vars_y[0],
            corrs_xy[0],
            nbs[0],
        )
    mx1 = means_x[0]
    my1 = means_y[0]
    max1 = maxs_abs_x[0]
    may1 = maxs_abs_y[0]
    vx1 = vars_x[0]
    vy1 = vars_y[0]
    cxy1 = corrs_xy[0]
    n1 = nbs[0]
    for i in range(1, len(means_x)):
        mx2 = means_x[i]
        my2 = means_y[i]
        max2 = maxs_abs_x[i]
        may2 = maxs_abs_y[i]
        vx2 = vars_x[i]
        vy2 = vars_y[i]
        cxy2 = corrs_xy[i]
        n2 = nbs[i]
        nb = paddle.where(paddle.logical_or(n1, n2), n1 + n2, eps)
        mean_x = (n1 * mx1 + n2 * mx2) / nb
        mean_y = (n1 * my1 + n2 * my2) / nb
        n12_b = n1 * n2 / nb
        delta_x = mx2 - mx1
        delta_y = my2 - my1
        var_x = vx1 + vx2 + n12_b * delta_x**2
        var_y = vy1 + vy2 + n12_b * delta_y**2
        corr_xy = cxy1 + cxy2 + n12_b * delta_x * delta_y
        max_abs_dev_x = paddle.maximum(max1, max2)
        max_abs_dev_y = paddle.maximum(may1, may2)
        mx1 = mean_x
        my1 = mean_y
        max1 = max_abs_dev_x
        may1 = max_abs_dev_y
        vx1 = var_x
        vy1 = var_y
        cxy1 = corr_xy
        n1 = nb
    return (
        mean_x,
        mean_y,
        max_abs_dev_x,
        max_abs_dev_y,
        var_x,
        var_y,
        corr_xy,
        nb,
    )


class PearsonCorrCoef(Metric):
    """Compute `Pearson Correlation Coefficient`_.

    .. math::
        P_{corr}(x,y) = \\frac{cov(x,y)}{\\sigma_x \\sigma_y}

    Where :math:`y` is a tensor of target values, and :math:`x` is a tensor of predictions.

    As input to ``forward`` and ``update`` the metric accepts the following input:

    - ``preds`` (:class:`~paddle.Tensor`): either single output float tensor with shape ``(N,)``
      or multioutput float tensor of shape ``(N,d)``
    - ``target`` (:class:`~paddle.Tensor`): either single output tensor with shape ``(N,)``
      or multioutput tensor of shape ``(N,d)``

    As output of ``forward`` and ``compute`` the metric returns the following output:

    - ``pearson`` (:class:`~paddle.Tensor`): A tensor with the Pearson Correlation Coefficient

    Args:
        num_outputs: Number of outputs in multioutput setting
        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Example (single output regression):
        >>> from paddle.metric.regression import PearsonCorrCoef
        >>> target = paddle.to_tensor([3, -0.5, 2, 7])
        >>> preds = paddle.to_tensor([2.5, 0.0, 2, 8])
        >>> pearson = PearsonCorrCoef()
        >>> pearson(preds, target)
        tensor(0.9849)

    Example (multi output regression):
        >>> from paddle.metric.regression import PearsonCorrCoef
        >>> target = paddle.to_tensor([[3, -0.5], [2, 7]])
        >>> preds = paddle.to_tensor([[2.5, 0.0], [2, 8]])
        >>> pearson = PearsonCorrCoef(num_outputs=2)
        >>> pearson(preds, target)
        tensor([1., 1.])

    """

    is_differentiable: bool = True
    higher_is_better: bool | None = None
    full_state_update: bool = True
    plot_lower_bound: float = -1.0
    plot_upper_bound: float = 1.0
    preds: list[paddle.Tensor]
    target: list[paddle.Tensor]
    mean_x: Tensor
    mean_y: Tensor
    max_abs_dev_x: Tensor
    max_abs_dev_y: Tensor
    var_x: Tensor
    var_y: Tensor
    corr_xy: Tensor
    n_total: Tensor

    def __init__(self, num_outputs: int = 1, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if not isinstance(num_outputs, int) and num_outputs < 1:
            raise ValueError(
                "Expected argument `num_outputs` to be an int larger than 0, but got {num_outputs}"
            )
        self.num_outputs = num_outputs
        self.declare(
            "mean_x",
            default=paddle.zeros(self.num_outputs),
            dist_reduce_fn=None,
        )
        self.declare(
            "mean_y",
            default=paddle.zeros(self.num_outputs),
            dist_reduce_fn=None,
        )
        self.declare(
            "max_abs_dev_x",
            default=paddle.zeros(self.num_outputs),
            dist_reduce_fn=None,
        )
        self.declare(
            "max_abs_dev_y",
            default=paddle.zeros(self.num_outputs),
            dist_reduce_fn=None,
        )
        self.declare(
            "var_x", default=paddle.zeros(self.num_outputs), dist_reduce_fn=None
        )
        self.declare(
            "var_y", default=paddle.zeros(self.num_outputs), dist_reduce_fn=None
        )
        self.declare(
            "corr_xy",
            default=paddle.zeros(self.num_outputs),
            dist_reduce_fn=None,
        )
        self.declare(
            "n_total",
            default=paddle.zeros(self.num_outputs),
            dist_reduce_fn=None,
        )

    def update(self, preds: paddle.Tensor, target: paddle.Tensor) -> None:
        """Update state with predictions and targets."""
        (
            self.mean_x,
            self.mean_y,
            self.max_abs_dev_x,
            self.max_abs_dev_y,
            self.var_x,
            self.var_y,
            self.corr_xy,
            self.n_total,
        ) = _pearson_corrcoef_update(
            preds=preds,
            target=target,
            mean_x=self.mean_x,
            mean_y=self.mean_y,
            max_abs_dev_x=self.max_abs_dev_x,
            max_abs_dev_y=self.max_abs_dev_y,
            var_x=self.var_x,
            var_y=self.var_y,
            corr_xy=self.corr_xy,
            num_prior=self.n_total,
            num_outputs=self.num_outputs,
        )

    def compute(self) -> paddle.Tensor:
        """Compute pearson correlation coefficient over state."""
        if (
            self.num_outputs == 1
            and self.mean_x.size > 1
            or self.num_outputs > 1
            and self.mean_x.ndim > 1
        ):
            (
                _,
                _,
                max_abs_dev_x,
                max_abs_dev_y,
                var_x,
                var_y,
                corr_xy,
                n_total,
            ) = _final_aggregation(
                means_x=self.mean_x,
                means_y=self.mean_y,
                maxs_abs_x=self.max_abs_dev_x,
                maxs_abs_y=self.max_abs_dev_y,
                vars_x=self.var_x,
                vars_y=self.var_y,
                corrs_xy=self.corr_xy,
                nbs=self.n_total,
            )
        else:
            max_abs_dev_x = self.max_abs_dev_x
            max_abs_dev_y = self.max_abs_dev_y
            var_x = self.var_x
            var_y = self.var_y
            corr_xy = self.corr_xy
            n_total = self.n_total
        return _pearson_corrcoef_compute(
            max_abs_dev_x, max_abs_dev_y, var_x, var_y, corr_xy, n_total
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
            >>> from paddle.metric.regression import PearsonCorrCoef
            >>> metric = PearsonCorrCoef()
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
            >>> from paddle.metric.regression import PearsonCorrCoef
            >>> metric = PearsonCorrCoef()
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
