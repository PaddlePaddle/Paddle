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

from paddle.metric.functional.regression.cosine_similarity import (
    _cosine_similarity_compute,
    _cosine_similarity_update,
)
from paddle.metric.metric import Metric
from paddle.metric.utils.data import dim_zero_cat
from paddle.metric.utils.imports import _MATPLOTLIB_AVAILABLE

if TYPE_CHECKING:
    from collections.abc import Sequence

    import paddle
    from paddle.metric.utils.plot import _AX_TYPE, _PLOT_OUT_TYPE

if not _MATPLOTLIB_AVAILABLE:
    __doctest_skip__ = ["CosineSimilarity.plot"]


class CosineSimilarity(Metric):
    """Compute the `Cosine Similarity`_.

    .. math::
        cos_{sim}(x,y) = \\frac{x \\cdot y}{||x|| \\cdot ||y||} =
        \\frac{\\sum_{i=1}^n x_i y_i}{\\sqrt{\\sum_{i=1}^n x_i^2}\\sqrt{\\sum_{i=1}^n y_i^2}}

    where :math:`y` is a tensor of target values, and :math:`x` is a tensor of predictions.

    As input to ``forward`` and ``update`` the metric accepts the following input:

    - ``preds`` (:class:`~paddle.Tensor`): Predicted float tensor with shape ``(N,d)``
    - ``target`` (:class:`~paddle.Tensor`): Ground truth float tensor with shape ``(N,d)``

    As output of ``forward`` and ``compute`` the metric returns the following output:

    - ``cosine_similarity`` (:class:`~paddle.Tensor`): A float tensor with the cosine similarity

    Args:
        reduction: how to reduce over the batch dimension using 'sum', 'mean' or 'none' (taking the individual scores)
        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Example:
        >>> from paddle import tensor
        >>> from paddle.metric.regression import CosineSimilarity
        >>> target = tensor([[0, 1], [1, 1]])
        >>> preds = tensor([[0, 1], [0, 1]])
        >>> cosine_similarity = CosineSimilarity(reduction='mean')
        >>> cosine_similarity(preds, target)
        tensor(0.8536)

    """

    is_differentiable: bool = True
    higher_is_better: bool = True
    full_state_update: bool = False
    plot_lower_bound: float = 0.0
    plot_upper_bound: float = 1.0
    preds: list[paddle.Tensor]
    target: list[paddle.Tensor]

    def __init__(
        self,
        reduction: Literal["mean", "sum", "none"] | None = "sum",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        allowed_reduction = "sum", "mean", "none", None
        if reduction not in allowed_reduction:
            raise ValueError(
                f"Expected argument `reduction` to be one of {allowed_reduction} but got {reduction}"
            )
        self.reduction = reduction
        self.declare("preds", [], dist_reduce_fn="cat")
        self.declare("target", [], dist_reduce_fn="cat")

    def update(self, preds: paddle.Tensor, target: paddle.Tensor) -> None:
        """Update metric states with predictions and targets."""
        preds, target = _cosine_similarity_update(preds, target)
        self.preds.append(preds)
        self.target.append(target)

    def compute(self) -> paddle.Tensor:
        """Compute metric."""
        preds = dim_zero_cat(self.preds)
        target = dim_zero_cat(self.target)
        return _cosine_similarity_compute(preds, target, self.reduction)

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
            >>> from paddle.metric.regression import CosineSimilarity
            >>> metric = CosineSimilarity()
            >>> metric.update(randn(10, 2), randn(10, 2))
            >>> fig_, ax_ = metric.plot()

        .. plot::
            :scale: 75

            >>> from paddle import randn
            >>> # Example plotting multiple values
            >>> from paddle.metric.regression import CosineSimilarity
            >>> metric = CosineSimilarity()
            >>> values = []
            >>> for _ in range(10):
            ...     values.append(metric(randn(10, 2), randn(10, 2)))
            >>> fig, ax = metric.plot(values)

        """
        return self._plot(val, ax)
