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

from copy import deepcopy
from typing import TYPE_CHECKING, Any, cast

import paddle
from paddle import Tensor
from paddle.metric.collections import MetricCollection
from paddle.metric.metric import Metric
from paddle.metric.utils.imports import _MATPLOTLIB_AVAILABLE
from paddle.metric.utils.plot import (
    _AX_TYPE,
    _PLOT_OUT_TYPE,
    plot_single_or_multi_val,
)
from paddle.metric.utils.prints import rank_zero_warn
from paddle.metric.wrappers import ClasswiseWrapper

if TYPE_CHECKING:
    from collections.abc import Sequence

if not _MATPLOTLIB_AVAILABLE:
    __doctest_skip__ = ["MetricTracker.plot"]


class MetricTracker(paddle.nn.LayerList):
    """A wrapper class that can help keeping track of a metric or metric collection over time.

    The wrapper implements the standard ``.update()``, ``.compute()``, ``.reset()`` methods that just
    calls corresponding method of the currently tracked metric. However, the following additional methods are
    provided:

        -``MetricTracker.n_steps``: number of metrics being tracked
        -``MetricTracker.increment()``: initialize a new metric for being tracked
        -``MetricTracker.compute_all()``: get the metric value for all steps
        -``MetricTracker.best_metric()``: returns the best value

    Out of the box, this wrapper class fully supports that the base metric being tracked is a single `Metric`, a
    `MetricCollection` or another `MetricWrapper` wrapped around a metric. However, multiple layers of nesting, such
    as using a `Metric` inside a `MetricWrapper` inside a `MetricCollection` is not fully supported, especially the
    `.best_metric` method that cannot auto compute the best metric and index for such nested structures.

    Args:
        metric: instance of a ``paddle.metric.Metric`` or ``paddle.metric.MetricCollection``
            to keep track of at each timestep.
        maximize: either single bool or list of bool indicating if higher metric values are
            better (``True``) or lower is better (``False``).

    Example (single metric):
        >>> from paddle import randint
        >>> from paddle.metric.wrappers import MetricTracker
        >>> from paddle.metric.classification import MulticlassAccuracy
        >>> tracker = MetricTracker(MulticlassAccuracy(num_classes=10, average='micro'))
        >>> for epoch in range(5):
        ...     tracker.increment()
        ...     for batch_idx in range(5):
        ...         tracker.update(randint(10, (100,)), randint(10, (100,)))
        ...     print(f"current acc={tracker.compute()}")
        current acc=0.1120000034570694
        current acc=0.08799999952316284
        current acc=0.12600000202655792
        current acc=0.07999999821186066
        current acc=0.10199999809265137
        >>> best_acc, which_epoch = tracker.best_metric(return_step=True)
        >>> best_acc  # doctest: +ELLIPSIS
        0.1260...
        >>> which_epoch
        2
        >>> tracker.compute_all()
        tensor([0.1120, 0.0880, 0.1260, 0.0800, 0.1020])

    Example (multiple metrics using MetricCollection):
        >>> from paddle import randn
        >>> from paddle.metric.wrappers import MetricTracker
        >>> from paddle.metric import MetricCollection
        >>> from paddle.metric.regression import MeanSquaredError, ExplainedVariance
        >>> tracker = MetricTracker(MetricCollection([MeanSquaredError(), ExplainedVariance()]), maximize=[False])
        >>> for epoch in range(5):
        ...     tracker.increment()
        ...     for batch_idx in range(5):
        ...         tracker.update(randn(100), randn(100))
        ...     print(f"current stats={tracker.compute()}")  # doctest: +NORMALIZE_WHITESPACE
        current stats={'MeanSquaredError': tensor(2.3292), 'ExplainedVariance': tensor(-0.9516)}
        current stats={'MeanSquaredError': tensor(2.1370), 'ExplainedVariance': tensor(-1.0775)}
        current stats={'MeanSquaredError': tensor(2.1695), 'ExplainedVariance': tensor(-0.9945)}
        current stats={'MeanSquaredError': tensor(2.1072), 'ExplainedVariance': tensor(-1.1878)}
        current stats={'MeanSquaredError': tensor(2.0562), 'ExplainedVariance': tensor(-1.0754)}
        >>> from pprint import pprint
        >>> best_res, which_epoch = tracker.best_metric(return_step=True)
        >>> pprint(best_res)  # doctest: +ELLIPSIS
        {'ExplainedVariance': -0.951...,
         'MeanSquaredError': 2.056...}
        >>> which_epoch
        {'MeanSquaredError': 4, 'ExplainedVariance': 0}
        >>> pprint(tracker.compute_all())
        {'ExplainedVariance': tensor([-0.9516, -1.0775, -0.9945, -1.1878, -1.0754]),
         'MeanSquaredError': tensor([2.3292, 2.1370, 2.1695, 2.1072, 2.0562])}

    """

    maximize: bool | list[bool]
    _base_metric: Metric | MetricCollection

    def __init__(
        self,
        metric: Metric | MetricCollection,
        maximize: bool | list[bool] | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(metric, (Metric, MetricCollection)):
            raise TypeError(
                f"Metric arg need to be an instance of a paddle.metric `Metric` or `MetricCollection` but got {metric}"
            )
        self._base_metric = metric
        if maximize is None:
            if isinstance(metric, Metric):
                if getattr(metric, "higher_is_better", None) is None:
                    raise AttributeError(
                        f"The metric '{metric.__class__.__name__}' does not have a 'higher_is_better' attribute. Please provide the `maximize` argument explicitly."
                    )
                self.maximize = metric.higher_is_better
            elif isinstance(metric, MetricCollection):
                self.maximize = []
                for name, m in metric.items():
                    if getattr(m, "higher_is_better", None) is None:
                        raise AttributeError(
                            f"The metric '{name}' in the MetricCollection does not have a 'higher_is_better' attribute. Please provide the `maximize` argument explicitly."
                        )
                    if isinstance(m, ClasswiseWrapper) and isinstance(
                        m.metric.num_classes, int
                    ):
                        m_higher_is_better = [
                            m.higher_is_better
                            for _ in range(int(m.metric.num_classes))
                        ]
                    else:
                        m_higher_is_better = [m.higher_is_better]
                    self.maximize.extend(m_higher_is_better)
        else:
            if not isinstance(maximize, (bool, list)):
                raise ValueError(
                    "Argument `maximize` should either be a single bool or list of bool"
                )
            if isinstance(maximize, list) and not all(
                isinstance(m, bool) for m in maximize
            ):
                raise ValueError(
                    "Argument `maximize` is list but not type of bool."
                )
            if (
                isinstance(maximize, list)
                and isinstance(metric, MetricCollection)
                and len(maximize) != len(metric)
            ):
                raise ValueError(
                    "The len of argument `maximize` should match the length of the metric collection"
                )
            if isinstance(metric, Metric) and not isinstance(maximize, bool):
                raise ValueError(
                    "Argument `maximize` should be a single bool when `metric` is a single Metric"
                )
            self.maximize = maximize
        self._increment_called = False

    @property
    def n_steps(self) -> int:
        """Returns the number of times the tracker has been incremented."""
        return len(self) - 1

    def increment(self) -> None:
        """Create a new instance of the input metric that will be updated next."""
        self._increment_called = True
        self.append(deepcopy(self._base_metric))

    def forward(self, *args: Any, **kwargs: Any) -> None:
        """Call forward of the current metric being tracked."""
        self._check_for_increment("forward")
        if not isinstance(self[-1], (Metric, MetricCollection)):
            raise TypeError(
                f"Expected the last item to be a Metric or MetricCollection, but got {type(self[-1])}."
            )
        return self[-1](*args, **kwargs)

    def update(self, *args: Any, **kwargs: Any) -> None:
        """Update the current metric being tracked."""
        self._check_for_increment("update")
        if not isinstance(self[-1], (Metric, MetricCollection)):
            raise TypeError(
                f"Expected the last item to be a Metric or MetricCollection, but got {type(self[-1])}."
            )
        self[-1].update(*args, **kwargs)

    def compute(self) -> Any:
        """Call compute of the current metric being tracked."""
        self._check_for_increment("compute")
        if not isinstance(self[-1], (Metric, MetricCollection)):
            raise TypeError(
                f"Expected the last item to be a Metric or MetricCollection, but got {type(self[-1])}."
            )
        return self[-1].compute()

    def compute_all(self) -> Any:
        """Compute the metric value for all tracked metrics.

        Return:
            By default will try stacking the results from all increments into a single tensor if the tracked base
            object is a single metric. If a metric collection is provided a dict of stacked tensors will be returned.
            If the stacking process fails a list of the computed results will be returned.

        Raises:
            ValueError:
                If `self.increment` have not been called before this method is called.

        """
        self._check_for_increment("compute_all")
        res: list[Any] = []
        for i, metric in enumerate(self):
            if i == 0:
                continue
            if not isinstance(metric, (Metric, MetricCollection)):
                raise TypeError(
                    f"Expected the item to be a Metric or MetricCollection, but got {type(metric)}."
                )
            res.append(metric.compute())
        try:
            if isinstance(res[0], dict):
                keys = res[0].keys()
                return {
                    k: paddle.stack([cast("Tensor", r[k]) for r in res], axis=0)
                    for k in keys
                }
            if isinstance(res[0], list):
                return paddle.stack(
                    [
                        paddle.stack(cast("list[paddle.Tensor]", r), axis=0)
                        for r in res
                    ],
                    axis=0,
                )
            return paddle.stack(cast("list[paddle.Tensor]", res), axis=0)
        except TypeError:
            return res
        return res

    def reset(self) -> None:
        """Reset the current metric being tracked."""
        if not isinstance(self[-1], (Metric, MetricCollection)):
            raise TypeError(
                f"Expected the last item to be a Metric or MetricCollection, but got {type(self[-1])}."
            )
        self[-1].reset()

    def reset_all(self) -> None:
        """Reset all metrics being tracked."""
        for metric in self:
            if not isinstance(metric, (Metric, MetricCollection)):
                raise TypeError(
                    f"Expected all metrics to be Metric or MetricCollection, but got {type(metric)}."
                )
            metric.reset()

    def best_metric(
        self, return_step: bool = False
    ) -> (
        None
        | float
        | paddle.Tensor
        | tuple[int | float | paddle.Tensor, int | float | paddle.Tensor]
        | tuple[None, None]
        | dict[str, float | None]
        | tuple[dict[str, float | None], dict[str, int | None]]
    ):
        """Return the highest metric out of all tracked.

        Args:
            return_step: If ``True`` will also return the step with the highest metric value.

        Returns:
            Either a single value or a tuple, depends on the value of ``return_step`` and the object being tracked.

            - If a single metric is being tracked and ``return_step=False`` then a single tensor will be returned
            - If a single metric is being tracked and ``return_step=True`` then a 2-element tuple will be returned,
              where the first value is optimal value and second value is the corresponding optimal step
            - If a metric collection is being tracked and ``return_step=False`` then a single dict will be returned,
              where keys correspond to the different values of the collection and the values are the optimal metric
              value
            - If a metric collection is being bracked and ``return_step=True`` then a 2-element tuple will be returned
              where each is a dict, with keys corresponding to the different values of th collection and the values
              of the first dict being the optimal values and the values of the second dict being the optimal step

            In addition the value in all cases may be ``None`` if the underlying metric does have a proper defined way
            of being optimal or in the case where a nested structure of metrics are being tracked.

        """
        res = self.compute_all()
        if isinstance(res, list):
            rank_zero_warn(
                "Encountered nested structure. You are probably using a metric collection inside a metric collection, or a metric wrapper inside a metric collection, which is not supported by `.best_metric()` method. Returning `None` instead."
            )
            if return_step:
                return None, None
            return None
        if isinstance(self._base_metric, Metric) and not isinstance(
            self._base_metric, ClasswiseWrapper
        ):
            fn = paddle.max if self.maximize else paddle.min
            try:
                value, idx = fn(res, 0)
                if return_step:
                    return value.item(), idx.item()
                return value.item()
            except (ValueError, RuntimeError) as error:
                rank_zero_warn(
                    f"Encountered the following error when trying to get the best metric: {error}this is probably due to the 'best' not being defined for this metric.Returning `None` instead.",
                    UserWarning,
                )
                if return_step:
                    return None, None
                return None
        else:
            maximize = (
                self.maximize
                if isinstance(self.maximize, list)
                else len(res) * [self.maximize]
            )
            value, idx = {}, {}
            for i, (k, v) in enumerate(res.items()):
                try:
                    fn = paddle.max if maximize[i] else paddle.min
                    out = fn(v, 0)
                    value[k], idx[k] = out[0].item(), out[1].item()
                except (ValueError, RuntimeError) as error:
                    rank_zero_warn(
                        f"Encountered the following error when trying to get the best metric for metric {k}:{error} this is probably due to the 'best' not being defined for this metric.Returning `None` instead.",
                        UserWarning,
                    )
                    value[k], idx[k] = None, None
            if return_step:
                return value, idx
            return value

    def _check_for_increment(self, method: str) -> None:
        """Check that a metric that can be updated/used for computations has been initialized."""
        if not self._increment_called:
            raise ValueError(
                f"`{method}` cannot be called before `.increment()` has been called."
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

            >>> # Example plotting a single value
            >>> import paddle
            >>> from paddle.metric.wrappers import MetricTracker
            >>> from paddle.metric.classification import BinaryAccuracy
            >>> tracker = MetricTracker(BinaryAccuracy(), maximize=True)
            >>> for epoch in range(5):
            ...     tracker.increment()
            ...     for batch_idx in range(5):
            ...         tracker.update(paddle.randint(2, (10,)), paddle.randint(2, (10,)))
            >>> fig_, ax_ = tracker.plot()  # plot all epochs

        """
        val = val if val is not None else self.compute_all()
        fig, ax = plot_single_or_multi_val(
            val, ax=ax, name=self.__class__.__name__
        )
        return fig, ax
