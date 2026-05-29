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
from paddle.metric.metric import Metric
from paddle.metric.utils.imports import _MATPLOTLIB_AVAILABLE
from paddle.metric.wrappers.abstract import WrapperMetric

if TYPE_CHECKING:
    from collections.abc import Sequence

    from paddle.metric.utils.plot import _AX_TYPE, _PLOT_OUT_TYPE

if not _MATPLOTLIB_AVAILABLE:
    __doctest_skip__ = ["MinMaxMetric.plot"]


class MinMaxMetric(WrapperMetric):
    """Wrapper metric that tracks both the minimum and maximum of a scalar/tensor across an experiment.

    The min/max value will be updated each time ``.compute`` is called.

    Args:
        base_metric:
            The metric of which you want to keep track of its maximum and minimum values.
        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Raises:
        ValueError
            If ``base_metric` argument is not a subclasses instance of ``paddle.metric.Metric``

    Example::
        >>> import paddle
        >>> from paddle.metric.wrappers import MinMaxMetric
        >>> from paddle.metric.classification import BinaryAccuracy
        >>> from pprint import pprint
        >>> base_metric = BinaryAccuracy()
        >>> minmax_metric = MinMaxMetric(base_metric)
        >>> preds_1 = paddle.Tensor([[0.1, 0.9], [0.2, 0.8]])
        >>> preds_2 = paddle.Tensor([[0.9, 0.1], [0.2, 0.8]])
        >>> labels = paddle.Tensor([[0, 1], [0, 1]]).long()
        >>> pprint(minmax_metric(preds_1, labels))
        {'max': tensor(1.), 'min': tensor(1.), 'raw': tensor(1.)}
        >>> pprint(minmax_metric.compute())
        {'max': tensor(1.), 'min': tensor(1.), 'raw': tensor(1.)}
        >>> minmax_metric.update(preds_2, labels)
        >>> pprint(minmax_metric.compute())
        {'max': tensor(1.), 'min': tensor(0.7500), 'raw': tensor(0.7500)}

    """

    full_state_update: bool | None = True
    min_val: Tensor
    max_val: Tensor

    def __init__(self, base_metric: Metric, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if not isinstance(base_metric, Metric):
            raise ValueError(
                f"Expected base metric to be an instance of `paddle.metric.Metric` but received {base_metric}"
            )
        self._base_metric = base_metric
        self.min_val = paddle.tensor(float("inf"))
        self.max_val = paddle.tensor(float("-inf"))

    def update(self, *args: Any, **kwargs: Any) -> None:
        """Update the underlying metric."""
        self._base_metric.update(*args, **kwargs)

    def compute(self) -> dict[str, paddle.Tensor]:
        """Compute the underlying metric as well as max and min values for this metric.

        Returns a dictionary that consists of the computed value (``raw``), as well as the minimum (``min``) and maximum
        (``max``) values.

        """
        val = self._base_metric.compute()
        if not self._is_suitable_val(val):
            raise RuntimeError(
                f"Returned value from base metric should be a float or scalar tensor, but got {val}."
            )
        self.max_val = max(self.max_val.to(val.place), val)
        self.min_val = min(self.min_val.to(val.place), val)
        return {"raw": val, "max": self.max_val, "min": self.min_val}

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Use the original forward method of the base metric class."""
        return super(WrapperMetric, self).forward(*args, **kwargs)

    def reset(self) -> None:
        """Set ``max_val`` and ``min_val`` to the initialization bounds and resets the base metric."""
        super().reset()
        self._base_metric.reset()

    @staticmethod
    def _is_suitable_val(val: float | paddle.Tensor) -> bool:
        """Check whether min/max is a scalar value."""
        if isinstance(val, (int, float)):
            return True
        if isinstance(val, paddle.Tensor):
            return val.size == 1
        return False

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

            >>> # Example plotting a single value
            >>> import paddle
            >>> from paddle.metric.wrappers import MinMaxMetric
            >>> from paddle.metric.classification import BinaryAccuracy
            >>> metric = MinMaxMetric(BinaryAccuracy())
            >>> metric.update(paddle.randint(2, (20,)), paddle.randint(2, (20,)))
            >>> fig_, ax_ = metric.plot()

        .. plot::
            :scale: 75

            >>> # Example plotting multiple values
            >>> import paddle
            >>> from paddle.metric.wrappers import MinMaxMetric
            >>> from paddle.metric.classification import BinaryAccuracy
            >>> metric = MinMaxMetric(BinaryAccuracy())
            >>> values = []
            >>> for _ in range(3):
            ...     values.append(metric(paddle.randint(2, (20,)), paddle.randint(2, (20,))))
            >>> fig_, ax_ = metric.plot(values)

        """
        return self._plot(val, ax)
