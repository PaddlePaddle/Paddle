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

from collections.abc import Iterable, Sequence
from copy import deepcopy
from typing import Any

import paddle
from paddle.metric.collections import MetricCollection
from paddle.metric.metric import Metric
from paddle.metric.utils.imports import _MATPLOTLIB_AVAILABLE
from paddle.metric.utils.plot import _AX_TYPE, _PLOT_OUT_TYPE
from paddle.metric.wrappers.abstract import WrapperMetric

if not _MATPLOTLIB_AVAILABLE:
    __doctest_skip__ = ["MultitaskWrapper.plot"]


class MultitaskWrapper(WrapperMetric):
    """Wrapper class for computing different metrics on different tasks in the context of multitask learning.

    In multitask learning the different tasks requires different metrics to be evaluated. This wrapper allows
    for easy evaluation in such cases by supporting multiple predictions and targets through a dictionary.
    Note that only metrics where the signature of `update` follows the standard `preds, target` is supported.

    Args:
        task_metrics:
            Dictionary associating each task to a Metric or a MetricCollection. The keys of the dictionary represent the
            names of the tasks, and the values represent the metrics to use for each task.
        prefix:
            A string to append in front of the metric keys. If not provided, will default to an empty string.
        postfix:
            A string to append after the keys of the output dict. If not provided, will default to an empty string.

    .. tip::
        The use prefix and postfix allows for easily creating task wrappers for training, validation and test.
        The arguments are only changing the output keys of the computed metrics and not the input keys. This means
        that a ``MultitaskWrapper`` initialized as ``MultitaskWrapper({"task": Metric()}, prefix="train_")`` will
        still expect the input to be a dictionary with the key "task", but the output will be a dictionary with the key
        "train_task".

    Raises:
        TypeError:
            If argument `task_metrics` is not an dictionary
        TypeError:
            If not all values in the `task_metrics` dictionary is instances of `Metric` or `MetricCollection`
        ValueError:
            If `prefix` is not a string
        ValueError:
            If `postfix` is not a string

    Example (with a single metric per class):
         >>> import paddle
         >>> from paddle.metric.wrappers import MultitaskWrapper
         >>> from paddle.metric.regression import MeanSquaredError
         >>> from paddle.metric.classification import BinaryAccuracy
         >>>
         >>> classification_target = paddle.to_tensor([0, 1, 0])
         >>> regression_target = paddle.to_tensor([2.5, 5.0, 4.0])
         >>> targets = {"Classification": classification_target, "Regression": regression_target}
         >>>
         >>> classification_preds = paddle.to_tensor([0, 0, 1])
         >>> regression_preds = paddle.to_tensor([3.0, 5.0, 2.5])
         >>> preds = {"Classification": classification_preds, "Regression": regression_preds}
         >>>
         >>> metrics = MultitaskWrapper({"Classification": BinaryAccuracy(), "Regression": MeanSquaredError()})
         >>> metrics.update(preds, targets)
         >>> metrics.compute()
         {'Classification': tensor(0.3333), 'Regression': tensor(0.8333)}

    Example (with several metrics per task):
         >>> import paddle
         >>> from paddle.metric import MetricCollection
         >>> from paddle.metric.wrappers import MultitaskWrapper
         >>> from paddle.metric.regression import MeanSquaredError, MeanAbsoluteError
         >>> from paddle.metric.classification import BinaryAccuracy, BinaryF1Score
         >>>
         >>> classification_target = paddle.to_tensor([0, 1, 0])
         >>> regression_target = paddle.to_tensor([2.5, 5.0, 4.0])
         >>> targets = {"Classification": classification_target, "Regression": regression_target}
         >>>
         >>> classification_preds = paddle.to_tensor([0, 0, 1])
         >>> regression_preds = paddle.to_tensor([3.0, 5.0, 2.5])
         >>> preds = {"Classification": classification_preds, "Regression": regression_preds}
         >>>
         >>> metrics = MultitaskWrapper(
         ...     {
         ...         "Classification": MetricCollection(BinaryAccuracy(), BinaryF1Score()),
         ...         "Regression": MetricCollection(MeanSquaredError(), MeanAbsoluteError()),
         ...     }
         ... )
         >>> metrics.update(preds, targets)
         >>> metrics.compute()
         {'Classification': {'BinaryAccuracy': tensor(0.3333), 'BinaryF1Score': tensor(0.)},
          'Regression': {'MeanSquaredError': tensor(0.8333), 'MeanAbsoluteError': tensor(0.6667)}}

    Example (with a prefix and postfix):
        >>> import paddle
        >>> from paddle.metric.wrappers import MultitaskWrapper
        >>> from paddle.metric.regression import MeanSquaredError
        >>> from paddle.metric.classification import BinaryAccuracy
        >>>
        >>> classification_target = paddle.to_tensor([0, 1, 0])
        >>> regression_target = paddle.to_tensor([2.5, 5.0, 4.0])
        >>> targets = {"Classification": classification_target, "Regression": regression_target}
        >>> classification_preds = paddle.to_tensor([0, 0, 1])
        >>> regression_preds = paddle.to_tensor([3.0, 5.0, 2.5])
        >>> preds = {"Classification": classification_preds, "Regression": regression_preds}
        >>>
        >>> metrics = MultitaskWrapper({"Classification": BinaryAccuracy(), "Regression": MeanSquaredError()}, prefix="train_")
        >>> metrics.update(preds, targets)
        >>> metrics.compute()
        {'train_Classification': tensor(0.3333), 'train_Regression': tensor(0.8333)}

    """

    is_differentiable: bool = False

    def __init__(
        self,
        task_metrics: dict[str, Metric | MetricCollection],
        prefix: str | None = None,
        postfix: str | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(task_metrics, dict):
            raise TypeError(
                f"Expected argument `task_metrics` to be a dict. Found task_metrics = {task_metrics}"
            )
        for metric in task_metrics.values():
            if not isinstance(metric, (Metric, MetricCollection)):
                raise TypeError(
                    f"Expected each task's metric to be a Metric or a MetricCollection. Found a metric of type {type(metric)}"
                )
        self.task_metrics = paddle.nn.LayerDict(task_metrics)
        if prefix is not None and not isinstance(prefix, str):
            raise ValueError(
                f"Expected argument `prefix` to either be `None` or a string but got {prefix}"
            )
        self._prefix = prefix or ""
        if postfix is not None and not isinstance(postfix, str):
            raise ValueError(
                f"Expected argument `postfix` to either be `None` or a string but got {postfix}"
            )
        self._postfix = postfix or ""

    def items(
        self, flatten: bool = True
    ) -> Iterable[tuple[str, paddle.nn.Layer]]:
        """Iterate over task and task metrics.

        Args:
            flatten: If True, will iterate over all sub-metrics in the case of a MetricCollection.
                If False, will iterate over the task names and the corresponding metrics.

        """
        for task_name, metric in self.task_metrics.items():
            if flatten and isinstance(metric, MetricCollection):
                for sub_metric_name, sub_metric in metric.items():
                    yield (
                        f"{self._prefix}{task_name}_{sub_metric_name}{self._postfix}",
                        sub_metric,
                    )
            else:
                yield f"{self._prefix}{task_name}{self._postfix}", metric

    def keys(self, flatten: bool = True) -> Iterable[str]:
        """Iterate over task names.

        Args:
            flatten: If True, will iterate over all sub-metrics in the case of a MetricCollection.
                If False, will iterate over the task names and the corresponding metrics.

        """
        for task_name, metric in self.task_metrics.items():
            if flatten and isinstance(metric, MetricCollection):
                for sub_metric_name in metric:
                    yield f"{self._prefix}{task_name}_{sub_metric_name}{self._postfix}"
            else:
                yield f"{self._prefix}{task_name}{self._postfix}"

    def values(self, flatten: bool = True) -> Iterable[paddle.nn.Layer]:
        """Iterate over task metrics.

        Args:
            flatten: If True, will iterate over all sub-metrics in the case of a MetricCollection.
                If False, will iterate over the task names and the corresponding metrics.

        """
        for metric in self.task_metrics.values():
            if flatten and isinstance(metric, MetricCollection):
                yield from metric.values()
            else:
                yield metric

    def update(
        self, task_preds: dict[str, Any], task_targets: dict[str, Any]
    ) -> None:
        """Update each task's metric with its corresponding pred and target.

        Args:
            task_preds: Dictionary associating each task to a Tensor of pred.
            task_targets: Dictionary associating each task to a Tensor of target.

        """
        if (
            not self.task_metrics.keys()
            == task_preds.keys()
            == task_targets.keys()
        ):
            raise ValueError(
                f"Expected arguments `task_preds` and `task_targets` to have the same keys as the wrapped `task_metrics`. Found task_preds.keys() = {task_preds.keys()}, task_targets.keys() = {task_targets.keys()} and self.task_metrics.keys() = {self.task_metrics.keys()}"
            )
        for task_name, metric in self.task_metrics.items():
            pred = task_preds[task_name]
            target = task_targets[task_name]
            if not isinstance(metric, (Metric, MetricCollection)):
                raise TypeError(
                    f"Expected each task's metric to be a Metric or a MetricCollection. Found a metric of type {type(metric)}"
                )
            metric.update(pred, target)

    def _convert_output(self, output: dict[str, Any]) -> dict[str, Any]:
        """Convert the output of the underlying metrics to a dictionary with the task names as keys."""
        return {
            f"{self._prefix}{task_name}{self._postfix}": task_output
            for task_name, task_output in output.items()
        }

    def compute(self) -> dict[str, Any]:
        """Compute metrics for all tasks."""
        output: dict[str, Any] = {}
        for task_name, metric in self.task_metrics.items():
            if not isinstance(metric, (Metric, MetricCollection)):
                raise TypeError(
                    f"Expected each task's metric to be a Metric or a MetricCollection. Found a metric of type {type(metric)}"
                )
            output[task_name] = metric.compute()
        return self._convert_output(output)

    def forward(
        self,
        task_preds: dict[str, paddle.Tensor],
        task_targets: dict[str, paddle.Tensor],
    ) -> dict[str, Any]:
        """Call underlying forward methods for all tasks and return the result as a dictionary."""
        return self._convert_output(
            {
                task_name: metric(
                    task_preds[task_name], task_targets[task_name]
                )
                for task_name, metric in self.task_metrics.items()
            }
        )

    def reset(self) -> None:
        """Reset all underlying metrics."""
        for metric in self.task_metrics.values():
            if not isinstance(metric, (Metric, MetricCollection)):
                raise TypeError(
                    f"Expected each task's metric to be a Metric or a MetricCollection. Found a metric of type {type(metric)}"
                )
            metric.reset()
        super().reset()

    @staticmethod
    def _check_arg(arg: str | None, name: str) -> str | None:
        if arg is None or isinstance(arg, str):
            return arg
        raise ValueError(
            f"Expected input `{name}` to be a string, but got {type(arg)}"
        )

    def clone(
        self, prefix: str | None = None, postfix: str | None = None
    ) -> MultitaskWrapper:
        """Make a copy of the metric.

        Args:
            prefix: a string to append in front of the metric keys
            postfix: a string to append after the keys of the output dict.

        """
        multitask_copy = deepcopy(self)
        multitask_copy._prefix = self._check_arg(prefix, "prefix") or ""
        multitask_copy._postfix = self._check_arg(postfix, "prefix") or ""
        return multitask_copy

    def plot(
        self,
        val: dict | Sequence[dict] | None = None,
        axes: Sequence[_AX_TYPE] | None = None,
    ) -> Sequence[_PLOT_OUT_TYPE]:
        """Plot a single or multiple values from the metric.

        All tasks' results are plotted on individual axes.

        Args:
            val: Either a single result from calling `metric.forward` or `metric.compute` or a list of these results.
                If no value is provided, will automatically call `metric.compute` and plot that result.
            axes: Sequence of matplotlib axis objects. If provided, will add the plots to the provided axis objects.
                If not provided, will create them.

        Returns:
            Sequence of tuples with Figure and Axes object for each task.

        .. plot::
            :scale: 75

            >>> # Example plotting a single value
            >>> import paddle
            >>> from paddle.metric.wrappers import MultitaskWrapper
            >>> from paddle.metric.regression import MeanSquaredError
            >>> from paddle.metric.classification import BinaryAccuracy
            >>>
            >>> classification_target = paddle.to_tensor([0, 1, 0])
            >>> regression_target = paddle.to_tensor([2.5, 5.0, 4.0])
            >>> targets = {"Classification": classification_target, "Regression": regression_target}
            >>>
            >>> classification_preds = paddle.to_tensor([0, 0, 1])
            >>> regression_preds = paddle.to_tensor([3.0, 5.0, 2.5])
            >>> preds = {"Classification": classification_preds, "Regression": regression_preds}
            >>>
            >>> metrics = MultitaskWrapper({"Classification": BinaryAccuracy(), "Regression": MeanSquaredError()})
            >>> metrics.update(preds, targets)
            >>> value = metrics.compute()
            >>> fig_, ax_ = metrics.plot(value)

        .. plot::
            :scale: 75

            >>> # Example plotting multiple values
            >>> import paddle
            >>> from paddle.metric.wrappers import MultitaskWrapper
            >>> from paddle.metric.regression import MeanSquaredError
            >>> from paddle.metric.classification import BinaryAccuracy
            >>>
            >>> classification_target = paddle.to_tensor([0, 1, 0])
            >>> regression_target = paddle.to_tensor([2.5, 5.0, 4.0])
            >>> targets = {"Classification": classification_target, "Regression": regression_target}
            >>>
            >>> classification_preds = paddle.to_tensor([0, 0, 1])
            >>> regression_preds = paddle.to_tensor([3.0, 5.0, 2.5])
            >>> preds = {"Classification": classification_preds, "Regression": regression_preds}
            >>>
            >>> metrics = MultitaskWrapper({"Classification": BinaryAccuracy(), "Regression": MeanSquaredError()})
            >>> values = []
            >>> for _ in range(10):
            ...     values.append(metrics(preds, targets))
            >>> fig_, ax_ = metrics.plot(values)

        """
        if axes is not None:
            if not isinstance(axes, Sequence):
                raise TypeError(
                    f"Expected argument `axes` to be a Sequence. Found type(axes) = {type(axes)}"
                )
            if not all(isinstance(ax, _AX_TYPE) for ax in axes):
                raise TypeError(
                    "Expected each ax in argument `axes` to be a matplotlib axis object"
                )
            if len(axes) != len(self.task_metrics):
                raise ValueError(
                    f"Expected argument `axes` to be a Sequence of the same length as the number of tasks.Found len(axes) = {len(axes)} and {len(self.task_metrics)} tasks"
                )
        val = val if val is not None else self.compute()
        fig_axs = []
        for i, (task_name, task_metric) in enumerate(self.task_metrics.items()):
            ax = axes[i] if axes is not None else None
            if not isinstance(task_metric, (Metric, MetricCollection)):
                raise TypeError(
                    f"Expected each task's metric to be a Metric or a MetricCollection. Found a metric of type {type(task_metric)}"
                )
            if isinstance(val, dict):
                f, a = task_metric.plot(val[task_name], ax=ax)
            elif isinstance(val, Sequence):
                f, a = task_metric.plot([v[task_name] for v in val], ax=ax)
            else:
                raise TypeError(
                    f"Expected argument `val` to be None or of type Dict or Sequence[Dict]. Found type(val)= {type(val)}"
                )
            fig_axs.append((f, a))
        return fig_axs
