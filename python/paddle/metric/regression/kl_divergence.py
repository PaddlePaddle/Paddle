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

from typing import TYPE_CHECKING, Any, cast

from typing_extensions import Literal

import paddle
from paddle import Tensor
from paddle.metric.functional.regression.kl_divergence import (
    _kld_compute,
    _kld_update,
)
from paddle.metric.metric import Metric
from paddle.metric.utils.data import dim_zero_cat
from paddle.metric.utils.imports import _MATPLOTLIB_AVAILABLE

if TYPE_CHECKING:
    from collections.abc import Sequence

    from paddle.metric.utils.plot import _AX_TYPE, _PLOT_OUT_TYPE

if not _MATPLOTLIB_AVAILABLE:
    __doctest_skip__ = ["KLDivergence.plot"]


class KLDivergence(Metric):
    """Compute the `KL divergence`_.

    .. math::
        D_{KL}(P||Q) = \\sum_{x\\in\\mathcal{X}} P(x) \\log\\frac{P(x)}{Q{x}}

    Where :math:`P` and :math:`Q` are probability distributions where :math:`P` usually represents a distribution
    over data and :math:`Q` is often a prior or approximation of :math:`P`. It should be noted that the KL divergence
    is a non-symmetrical metric i.e. :math:`D_{KL}(P||Q) \\neq D_{KL}(Q||P)`.

    As input to ``forward`` and ``update`` the metric accepts the following input:

    - ``p`` (:class:`~paddle.Tensor`): a data distribution with shape ``(N, d)``
    - ``q`` (:class:`~paddle.Tensor`): prior or approximate distribution with shape ``(N, d)``

    As output of ``forward`` and ``compute`` the metric returns the following output:

    - ``kl_divergence`` (:class:`~paddle.Tensor`): A tensor with the KL divergence

    Args:
        log_prob: bool indicating if input is log-probabilities or probabilities. If given as probabilities,
            will normalize to make sure the distributes sum to 1.
        reduction:
            Determines how to reduce over the ``N``/batch dimension:

            - ``'mean'`` [default]: Averages score across samples
            - ``'sum'``: Sum score across samples
            - ``'none'`` or ``None``: Returns score per sample

        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Raises:
        TypeError:
            If ``log_prob`` is not an ``bool``.
        ValueError:
            If ``reduction`` is not one of ``'mean'``, ``'sum'``, ``'none'`` or ``None``.

    .. attention::
        Half precision is only support on GPU for this metric.

    Example:
        >>> from paddle import tensor
        >>> from paddle.metric.regression import KLDivergence
        >>> p = tensor([[0.36, 0.48, 0.16]])
        >>> q = tensor([[1 / 3, 1 / 3, 1 / 3]])
        >>> kl_divergence = KLDivergence()
        >>> kl_divergence(p, q)
        tensor(0.0853)

    """

    is_differentiable: bool = True
    higher_is_better: bool = False
    full_state_update: bool = False
    plot_lower_bound: float = 0.0
    measures: paddle.Tensor | list[paddle.Tensor]
    total: Tensor

    def __init__(
        self,
        log_prob: bool = False,
        reduction: Literal["mean", "sum", "none"] | None = "mean",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if not isinstance(log_prob, bool):
            raise TypeError(
                f"Expected argument `log_prob` to be bool but got {log_prob}"
            )
        self.log_prob = log_prob
        allowed_reduction = ["mean", "sum", "none", None]
        if reduction not in allowed_reduction:
            raise ValueError(
                f"Expected argument `reduction` to be one of {allowed_reduction} but got {reduction}"
            )
        self.reduction = reduction
        if self.reduction in ["mean", "sum"]:
            self.declare("measures", paddle.tensor(0.0), dist_reduce_fn="sum")
        else:
            self.declare("measures", [], dist_reduce_fn="cat")
        self.declare("total", paddle.tensor(0), dist_reduce_fn="sum")

    def update(self, p: paddle.Tensor, q: paddle.Tensor) -> None:
        """Update metric states with predictions and targets."""
        measures, total = _kld_update(p, q, self.log_prob)
        if self.reduction is None or self.reduction == "none":
            cast("list[paddle.Tensor]", self.measures).append(measures)
        else:
            self.measures = cast("Tensor", self.measures) + measures.sum()
            self.total += total

    def compute(self) -> paddle.Tensor:
        """Compute metric."""
        measures: Tensor = (
            dim_zero_cat(cast("list[paddle.Tensor]", self.measures))
            if self.reduction in ["none", None]
            else cast("Tensor", self.measures)
        )
        return _kld_compute(measures, self.total, self.reduction)

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
            >>> from paddle.metric.regression import KLDivergence
            >>> metric = KLDivergence()
            >>> metric.update(randn(10, 3).softmax(dim=-1), randn(10, 3).softmax(dim=-1))
            >>> fig_, ax_ = metric.plot()

        .. plot::
            :scale: 75

            >>> from paddle import randn
            >>> # Example plotting multiple values
            >>> from paddle.metric.regression import KLDivergence
            >>> metric = KLDivergence()
            >>> values = []
            >>> for _ in range(10):
            ...     values.append(metric(randn(10, 3).softmax(dim=-1), randn(10, 3).softmax(dim=-1)))
            >>> fig, ax = metric.plot(values)

        """
        return self._plot(val, ax)
