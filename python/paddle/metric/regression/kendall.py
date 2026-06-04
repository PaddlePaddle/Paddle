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

from paddle.metric.functional.regression.kendall import (
    _kendall_corrcoef_compute,
    _kendall_corrcoef_update,
    _MetricVariant,
    _TestAlternative,
)
from paddle.metric.metric import Metric
from paddle.metric.utils.data import dim_zero_cat
from paddle.metric.utils.imports import _MATPLOTLIB_AVAILABLE

if TYPE_CHECKING:
    from collections.abc import Sequence

    import paddle
    from paddle.metric.utils.plot import _AX_TYPE, _PLOT_OUT_TYPE

if not _MATPLOTLIB_AVAILABLE:
    __doctest_skip__ = ["KendallRankCorrCoef.plot"]


class KendallRankCorrCoef(Metric):
    """Compute `Kendall Rank Correlation Coefficient`_.

    .. math::
        tau_a = \\frac{C - D}{C + D}

    where :math:`C` represents concordant pairs, :math:`D` stands for discordant pairs.

    .. math::
        tau_b = \\frac{C - D}{\\sqrt{(C + D + T_{preds}) * (C + D + T_{target})}}

    where :math:`C` represents concordant pairs, :math:`D` stands for discordant pairs and :math:`T` represents
    a total number of ties.

    .. math::
        tau_c = 2 * \\frac{C - D}{n^2 * \\frac{m - 1}{m}}

    where :math:`C` represents concordant pairs, :math:`D` stands for discordant pairs, :math:`n` is a total number
    of observations and :math:`m` is a ``min`` of unique values in ``preds`` and ``target`` sequence.

    Definitions according to Definition according to `The Treatment of Ties in Ranking Problems`_.

    As input to ``forward`` and ``update`` the metric accepts the following input:

    - ``preds`` (:class:`~paddle.Tensor`): Sequence of data in float tensor of either shape ``(N,)`` or ``(N,d)``
    - ``target`` (:class:`~paddle.Tensor`): Sequence of data in float tensor of either shape ``(N,)`` or ``(N,d)``

    As output of ``forward`` and ``compute`` the metric returns the following output:

    - ``kendall`` (:class:`~paddle.Tensor`): A tensor with the correlation tau statistic,
      and if it is not None, the p-value of corresponding statistical test.

    Args:
        variant: Indication of which variant of Kendall's tau to be used
        t_test: Indication whether to run t-test
        alternative: Alternative hypothesis for t-test. Possible values:
            - 'two-sided': the rank correlation is nonzero
            - 'less': the rank correlation is negative (less than zero)
            - 'greater':  the rank correlation is positive (greater than zero)
        num_outputs: Number of outputs in multioutput setting
        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Raises:
        ValueError: If ``t_test`` is not of a type bool
        ValueError: If ``t_test=True`` and ``alternative=None``

    Example (single output regression):
        >>> from paddle import tensor
        >>> from paddle.metric.regression import KendallRankCorrCoef
        >>> preds = tensor([2.5, 0.0, 2, 8])
        >>> target = tensor([3, -0.5, 2, 1])
        >>> kendall = KendallRankCorrCoef()
        >>> kendall(preds, target)
        tensor(0.3333)

    Example (multi output regression):
        >>> from paddle.metric.regression import KendallRankCorrCoef
        >>> preds = tensor([[2.5, 0.0], [2, 8]])
        >>> target = tensor([[3, -0.5], [2, 1]])
        >>> kendall = KendallRankCorrCoef(num_outputs=2)
        >>> kendall(preds, target)
        tensor([1., 1.])

    Example (single output regression with t-test):
        >>> from paddle.metric.regression import KendallRankCorrCoef
        >>> preds = tensor([2.5, 0.0, 2, 8])
        >>> target = tensor([3, -0.5, 2, 1])
        >>> kendall = KendallRankCorrCoef(t_test=True, alternative='two-sided')
        >>> kendall(preds, target)
        (tensor(0.3333), tensor(0.4969))

    Example (multi output regression with t-test):
        >>> from paddle.metric.regression import KendallRankCorrCoef
        >>> preds = tensor([[2.5, 0.0], [2, 8]])
        >>> target = tensor([[3, -0.5], [2, 1]])
        >>> kendall = KendallRankCorrCoef(t_test=True, alternative='two-sided', num_outputs=2)
        >>> kendall(preds, target)
        (tensor([1., 1.]), tensor([nan, nan]))

    """

    is_differentiable = False
    higher_is_better = None
    full_state_update = True
    plot_lower_bound: float = 0.0
    plot_upper_bound: float = 1.0
    preds: list[paddle.Tensor]
    target: list[paddle.Tensor]

    def __init__(
        self,
        variant: Literal["a", "b", "c"] = "b",
        t_test: bool = False,
        alternative: Literal["two-sided", "less", "greater"]
        | None = "two-sided",
        num_outputs: int = 1,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if not isinstance(t_test, bool):
            raise ValueError(
                f"Argument `t_test` is expected to be of a type `bool`, but got {type(t_test)}."
            )
        if t_test and alternative is None:
            raise ValueError(
                "Argument `alternative` is required if `t_test=True` but got `None`."
            )
        self.variant = _MetricVariant.from_str(str(variant))
        self.alternative = (
            _TestAlternative.from_str(str(alternative)) if t_test else None
        )
        self.num_outputs = num_outputs
        self.declare("preds", [], dist_reduce_fn="cat")
        self.declare("target", [], dist_reduce_fn="cat")

    def update(self, preds: paddle.Tensor, target: paddle.Tensor) -> None:
        """Update variables required to compute Kendall rank correlation coefficient."""
        self.preds, self.target = _kendall_corrcoef_update(
            preds, target, self.preds, self.target, num_outputs=self.num_outputs
        )

    def compute(
        self,
    ) -> paddle.Tensor | tuple[paddle.Tensor, paddle.Tensor]:
        """Compute Kendall rank correlation coefficient, and optionally p-value of corresponding statistical test."""
        preds = dim_zero_cat(self.preds)
        target = dim_zero_cat(self.target)
        tau, p_value = _kendall_corrcoef_compute(
            preds, target, self.variant, self.alternative
        )
        if p_value is not None:
            return tau, p_value
        return tau

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
            >>> from paddle.metric.regression import KendallRankCorrCoef
            >>> metric = KendallRankCorrCoef()
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
            >>> from paddle.metric.regression import KendallRankCorrCoef
            >>> metric = KendallRankCorrCoef()
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
