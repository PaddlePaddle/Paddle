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
    __doctest_skip__ = ["Running.plot"]


class Running(WrapperMetric):
    """Running wrapper for metrics.

    Using this wrapper allows for calculating metrics over a running window of values, instead of the whole history of
    values. This is beneficial when you want to get a better estimate of the metric during training and don't want to
    wait for the whole training to finish to get epoch level estimates.

    The running window is defined by the `window` argument. The window is a fixed size and this wrapper will store a
    duplicate of the underlying metric state for each value in the window. Thus memory usage will increase linearly
    with window size. Use accordingly. Also note that the running only works with metrics that have the
    `full_state_update` set to `False`.

    Importantly, the wrapper does not alter the value of the `forward` method of the underlying metric. Thus, forward
    will still return the value on the current batch. To get the running value call `compute` instead.

    Args:
        base_metric: The metric to wrap.
        window: The size of the running window.

    Example (single metric):
        >>> from paddle import tensor
        >>> from paddle.metric.wrappers import Running
        >>> from paddle.metric.aggregation import SumMetric
        >>> metric = Running(SumMetric(), window=3)
        >>> for i in range(6):
        ...     current_val = metric(tensor([i]))
        ...     running_val = metric.compute()
        ...     total_val = tensor(sum(list(range(i + 1))))  # value we would get from `compute` without running
        ...     print(f"{current_val=}, {running_val=}, {total_val=}")
        current_val=tensor(0.), running_val=tensor(0.), total_val=tensor(0)
        current_val=tensor(1.), running_val=tensor(1.), total_val=tensor(1)
        current_val=tensor(2.), running_val=tensor(3.), total_val=tensor(3)
        current_val=tensor(3.), running_val=tensor(6.), total_val=tensor(6)
        current_val=tensor(4.), running_val=tensor(9.), total_val=tensor(10)
        current_val=tensor(5.), running_val=tensor(12.), total_val=tensor(15)

    Example (metric collection):
        >>> from paddle import tensor
        >>> from paddle.metric.wrappers import Running
        >>> from paddle.metric import MetricCollection
        >>> from paddle.metric.aggregation import SumMetric, MeanMetric
        >>> # note that running is input to collection, not the other way
        >>> metric = MetricCollection({"sum": Running(SumMetric(), 3), "mean": Running(MeanMetric(), 3)})
        >>> for i in range(6):
        ...     current_val = metric(tensor([i]))
        ...     running_val = metric.compute()
        ...     print(f"{current_val=}, {running_val=}")
        current_val={'mean': tensor(0.), 'sum': tensor(0.)}, running_val={'mean': tensor(0.), 'sum': tensor(0.)}
        current_val={'mean': tensor(1.), 'sum': tensor(1.)}, running_val={'mean': tensor(0.5000), 'sum': tensor(1.)}
        current_val={'mean': tensor(2.), 'sum': tensor(2.)}, running_val={'mean': tensor(1.), 'sum': tensor(3.)}
        current_val={'mean': tensor(3.), 'sum': tensor(3.)}, running_val={'mean': tensor(2.), 'sum': tensor(6.)}
        current_val={'mean': tensor(4.), 'sum': tensor(4.)}, running_val={'mean': tensor(3.), 'sum': tensor(9.)}
        current_val={'mean': tensor(5.), 'sum': tensor(5.)}, running_val={'mean': tensor(4.), 'sum': tensor(12.)}

    """

    def __init__(self, base_metric: Metric, window: int = 5) -> None:
        super().__init__()
        if not isinstance(base_metric, Metric):
            raise ValueError(
                f"Expected argument `metric` to be an instance of `paddle.metric.Metric` but got {base_metric}"
            )
        if not (isinstance(window, int) and window > 0):
            raise ValueError(
                f"Expected argument `window` to be a positive integer but got {window}"
            )
        self.base_metric = base_metric
        self.window = window
        if base_metric.full_state_update is not False:
            raise ValueError(
                f"Expected attribute `full_state_update` set to `False` but got {base_metric.full_state_update}"
            )
        self._num_vals_seen = 0
        for key in base_metric._defaults:
            for i in range(window):
                self.declare(
                    name=key + f"_{i}",
                    default=base_metric._defaults[key],
                    dist_reduce_fn=base_metric._reductions[key],
                )

    def update(self, *args: Any, **kwargs: Any) -> None:
        """Update the underlying metric and save state afterwards."""
        val = self._num_vals_seen % self.window
        self.base_metric.update(*args, **kwargs)
        for key in self.base_metric._defaults:
            setattr(self, key + f"_{val}", getattr(self.base_metric, key))
        self.base_metric.reset()
        self._num_vals_seen += 1

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Forward input to the underlying metric and save state afterwards."""
        val = self._num_vals_seen % self.window
        res = self.base_metric.forward(*args, **kwargs)
        for key in self.base_metric._defaults:
            setattr(self, key + f"_{val}", getattr(self.base_metric, key))
        self.base_metric.reset()
        self._num_vals_seen += 1
        self._computed = None
        return res

    def compute(self) -> Any:
        """Compute the metric over the running window."""
        for i in range(self.window):
            self.base_metric._reduce_states(
                {
                    key: getattr(self, key + f"_{i}")
                    for key in self.base_metric._defaults
                }
            )
        self.base_metric._update_count = self._num_vals_seen
        val = self.base_metric.compute()
        self.base_metric.reset()
        return val

    def reset(self) -> None:
        """Reset metric."""
        super().reset()
        self._num_vals_seen = 0

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
            >>> from paddle.metric.wrappers import Running
            >>> from paddle.metric.aggregation import SumMetric
            >>> metric = Running(SumMetric(), 2)
            >>> metric.update(paddle.randn(20, 2))
            >>> fig_, ax_ = metric.plot()

        .. plot::
            :scale: 75

            >>> # Example plotting multiple values
            >>> import paddle
            >>> from paddle.metric.wrappers import Running
            >>> from paddle.metric.aggregation import SumMetric
            >>> metric = Running(SumMetric(), 2)
            >>> values = []
            >>> for _ in range(3):
            ...     values.append(metric(paddle.randn(20, 2)))
            >>> fig_, ax_ = metric.plot(values)

        """
        return self._plot(val, ax)
