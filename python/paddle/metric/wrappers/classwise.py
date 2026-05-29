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

from paddle.metric.metric import Metric
from paddle.metric.utils.imports import _MATPLOTLIB_AVAILABLE
from paddle.metric.wrappers.abstract import WrapperMetric

if TYPE_CHECKING:
    from collections.abc import Sequence

    import paddle
    from paddle.metric.utils.plot import _AX_TYPE, _PLOT_OUT_TYPE

if not _MATPLOTLIB_AVAILABLE:
    __doctest_skip__ = ["ClasswiseWrapper.plot"]


class ClasswiseWrapper(WrapperMetric):
    """Wrapper metric for altering the output of classification metrics.

    This metric works together with classification metrics that returns multiple values (one value per class) such that
    label information can be automatically included in the output.

    Args:
        metric: base metric that should be wrapped. It is assumed that the metric outputs a single
            tensor that is split along the first dimension.
        labels: list of strings indicating the different classes.
        prefix: string that is prepended to the metric names.
        postfix: string that is appended to the metric names.

    Example::
        Basic example where the output of a metric is unwrapped into a dictionary with the class index as keys:

        >>> from paddle import randint, randn
        >>> from paddle.metric.wrappers import ClasswiseWrapper
        >>> from paddle.metric.classification import MulticlassAccuracy
        >>> metric = ClasswiseWrapper(MulticlassAccuracy(num_classes=3, average=None))
        >>> preds = randn(10, 3).softmax(dim=-1)
        >>> target = randint(3, (10,))
        >>> metric(preds, target)  # doctest: +NORMALIZE_WHITESPACE
        {'multiclassaccuracy_0': tensor(0.5000),
         'multiclassaccuracy_1': tensor(0.7500),
         'multiclassaccuracy_2': tensor(0.)}

    Example::
        Using custom name via prefix and postfix:

        >>> from paddle import randint, randn
        >>> from paddle.metric.wrappers import ClasswiseWrapper
        >>> from paddle.metric.classification import MulticlassAccuracy
        >>> metric_pre = ClasswiseWrapper(MulticlassAccuracy(num_classes=3, average=None), prefix="acc-")
        >>> metric_post = ClasswiseWrapper(MulticlassAccuracy(num_classes=3, average=None), postfix="-acc")
        >>> preds = randn(10, 3).softmax(dim=-1)
        >>> target = randint(3, (10,))
        >>> metric_pre(preds, target)  # doctest: +NORMALIZE_WHITESPACE
        {'acc-0': tensor(0.3333), 'acc-1': tensor(0.6667), 'acc-2': tensor(0.)}
        >>> metric_post(preds, target)  # doctest: +NORMALIZE_WHITESPACE
        {'0-acc': tensor(0.3333), '1-acc': tensor(0.6667), '2-acc': tensor(0.)}

    Example::
        Providing labels as a list of strings:

        >>> from paddle import randint, randn
        >>> from paddle.metric.wrappers import ClasswiseWrapper
        >>> from paddle.metric.classification import MulticlassAccuracy
        >>> metric = ClasswiseWrapper(
        ...    MulticlassAccuracy(num_classes=3, average=None),
        ...    labels=["horse", "fish", "dog"]
        ... )
        >>> preds = randn(10, 3).softmax(dim=-1)
        >>> target = randint(3, (10,))
        >>> metric(preds, target)  # doctest: +NORMALIZE_WHITESPACE
        {'multiclassaccuracy_horse': tensor(0.),
         'multiclassaccuracy_fish': tensor(0.3333),
         'multiclassaccuracy_dog': tensor(0.4000)}

    Example::
        Classwise can also be used in combination with :class:`~paddle.metric.MetricCollection`. In this case, everything
        will be flattened into a single dictionary:

        >>> from paddle import randint, randn
        >>> from paddle.metric import MetricCollection
        >>> from paddle.metric.wrappers import ClasswiseWrapper
        >>> from paddle.metric.classification import MulticlassAccuracy, MulticlassRecall
        >>> labels = ["horse", "fish", "dog"]
        >>> metric = MetricCollection(
        ...     {'multiclassaccuracy': ClasswiseWrapper(MulticlassAccuracy(num_classes=3, average=None), labels),
        ...     'multiclassrecall': ClasswiseWrapper(MulticlassRecall(num_classes=3, average=None), labels)}
        ... )
        >>> preds = randn(10, 3).softmax(dim=-1)
        >>> target = randint(3, (10,))
        >>> metric(preds, target)  # doctest: +NORMALIZE_WHITESPACE
        {'multiclassaccuracy_horse': tensor(0.6667),
         'multiclassaccuracy_fish': tensor(0.3333),
         'multiclassaccuracy_dog': tensor(0.5000),
         'multiclassrecall_horse': tensor(0.6667),
         'multiclassrecall_fish': tensor(0.3333),
         'multiclassrecall_dog': tensor(0.5000)}

    """

    metric: Metric
    labels: list[str] | None

    def __init__(
        self,
        metric: Metric,
        labels: list[str] | None = None,
        prefix: str | None = None,
        postfix: str | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(metric, Metric):
            raise ValueError(
                f"Expected argument `metric` to be an instance of `paddle.metric.Metric` but got {metric}"
            )
        self.metric = metric
        if labels is not None and not (
            isinstance(labels, list)
            and all(isinstance(lab, str) for lab in labels)
        ):
            raise ValueError(
                f"Expected argument `labels` to either be `None` or a list of strings but got {labels}"
            )
        self.labels = labels
        if prefix is not None and not isinstance(prefix, str):
            raise ValueError(
                f"Expected argument `prefix` to either be `None` or a string but got {prefix}"
            )
        self._prefix = prefix
        if postfix is not None and not isinstance(postfix, str):
            raise ValueError(
                f"Expected argument `postfix` to either be `None` or a string but got {postfix}"
            )
        self._postfix = postfix
        self._update_count = 1

    @property
    def higher_is_better(self) -> bool | None:
        """Return if the metric is higher the better."""
        return self.metric.higher_is_better

    def _filter_kwargs(self, **kwargs: Any) -> dict[str, Any]:
        """Filter kwargs for the metric."""
        return self.metric._filter_kwargs(**kwargs)

    def _convert_output(self, x: paddle.Tensor) -> dict[str, Any]:
        """Convert output to dictionary with labels as keys."""
        if not self._prefix and not self._postfix:
            prefix = f"{self.metric.__class__.__name__.lower()}_"
            postfix = ""
        else:
            prefix = self._prefix or ""
            postfix = self._postfix or ""
        if self.labels is None:
            return {f"{prefix}{i}{postfix}": val for i, val in enumerate(x)}
        return {
            f"{prefix}{lab}{postfix}": val for lab, val in zip(self.labels, x)
        }

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Calculate on batch and accumulate to global state."""
        return self._convert_output(self.metric(*args, **kwargs))

    def update(self, *args: Any, **kwargs: Any) -> None:
        """Update state."""
        self.metric.update(*args, **kwargs)

    def compute(self) -> dict[str, paddle.Tensor]:
        """Compute metric."""
        return self._convert_output(self.metric.compute())

    def reset(self) -> None:
        """Reset metric."""
        self.metric.reset()

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
            >>> from paddle.metric.wrappers import ClasswiseWrapper
            >>> from paddle.metric.classification import MulticlassAccuracy
            >>> metric = ClasswiseWrapper(MulticlassAccuracy(num_classes=3, average=None))
            >>> metric.update(paddle.randint(3, (20,)), paddle.randint(3, (20,)))
            >>> fig_, ax_ = metric.plot()

        .. plot::
            :scale: 75

            >>> # Example plotting multiple values
            >>> import paddle
            >>> from paddle.metric.wrappers import ClasswiseWrapper
            >>> from paddle.metric.classification import MulticlassAccuracy
            >>> metric = ClasswiseWrapper(MulticlassAccuracy(num_classes=3, average=None))
            >>> values = []
            >>> for _ in range(3):
            ...     values.append(metric(paddle.randint(3, (20,)), paddle.randint(3, (20,))))
            >>> fig_, ax_ = metric.plot(values)

        """
        return self._plot(val, ax)

    def __getattr__(self, name: str) -> paddle.Tensor | paddle.nn.Layer:
        """Get attribute from classwise wrapper."""
        if (
            name == "metric"
            or name in self.__dict__
            and name not in self.metric.__dict__
        ):
            return super().__getattr__(name)
        return getattr(self.metric, name)

    def __setattr__(self, name: str, value: Any) -> None:
        """Set attribute to classwise wrapper."""
        if hasattr(self, "metric") and name in self.metric._defaults:
            setattr(self.metric, name, value)
        else:
            super().__setattr__(name, value)
            if name == "metric":
                self._defaults = self.metric._defaults
                self._persistent = self.metric._persistent
                self._reductions = self.metric._reductions
