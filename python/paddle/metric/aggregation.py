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

"""Aggregation metrics for summarizing streams of values."""

import warnings
from typing import Any, Callable

from typing_extensions import Literal

import paddle
from paddle.metric.metric import Metric


def _dim_zero_cat(x: paddle.Tensor | list[paddle.Tensor]) -> paddle.Tensor:
    """Concatenate tensors along dimension zero."""
    if isinstance(x, paddle.Tensor):
        return x
    if not x:
        raise ValueError("No samples to concatenate")
    x = [y.unsqueeze(0) if y.ndim == 0 else y for y in x]
    return paddle.concat(x, axis=0)


class BaseAggregator(Metric):
    """Base class for aggregation metrics.

    Args:
        fn: string specifying the reduction function
        default_value: default tensor value to use for the metric state
        nan_strategy: 'error', 'warn', 'ignore', 'disable', or a float
        state_name: name of the metric state
        kwargs: Additional keyword arguments
    """

    is_differentiable = None
    higher_is_better = None
    full_state_update: bool = False

    def __init__(
        self,
        fn: Callable | str,
        default_value: paddle.Tensor | list,
        nan_strategy: Literal["error", "warn", "ignore", "disable"]
        | float = "error",
        state_name: str = "value",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        allowed_nan_strategy = ("error", "warn", "ignore", "disable")
        if nan_strategy not in allowed_nan_strategy and not isinstance(
            nan_strategy, float
        ):
            raise ValueError(
                f"Arg `nan_strategy` should either be a float or one of {allowed_nan_strategy} but got {nan_strategy}."
            )
        self.nan_strategy = nan_strategy
        self.declare(state_name, default=default_value, dist_reduce_fn=fn)
        self.state_name = state_name

    def _cast_and_nan_check_input(
        self,
        x: float | paddle.Tensor,
        weight: float | paddle.Tensor | None = None,
    ) -> tuple[paddle.Tensor, paddle.Tensor]:
        """Convert input ``x`` to a tensor and check for Nans."""
        if not isinstance(x, paddle.Tensor):
            x = paddle.to_tensor(x, dtype=paddle.get_default_dtype())
        if weight is not None and not isinstance(weight, paddle.Tensor):
            weight = paddle.to_tensor(weight, dtype=paddle.get_default_dtype())

        if self.nan_strategy != "disable":
            nans = paddle.isnan(x)
            if weight is not None:
                nans_weight = paddle.isnan(weight)
            else:
                nans_weight = paddle.zeros_like(nans, dtype=paddle.bool)
                weight = paddle.ones_like(x, dtype=paddle.get_default_dtype())

            if nans.any() or nans_weight.any():
                if self.nan_strategy == "error":
                    raise RuntimeError("Encountered `nan` values in tensor")
                if self.nan_strategy in ("ignore", "warn"):
                    if self.nan_strategy == "warn":
                        warnings.warn(
                            "Encountered `nan` values in tensor. Will be removed.",
                            UserWarning,
                        )
                    mask = ~(nans | nans_weight)
                    x = x[mask]
                    weight = weight[mask]
                else:
                    if not isinstance(self.nan_strategy, float):
                        raise ValueError(
                            f"`nan_strategy` shall be float but you pass {self.nan_strategy}"
                        )
                    x[nans | nans_weight] = self.nan_strategy
                    weight[nans | nans_weight] = 1.0
        else:
            weight = paddle.ones_like(x, dtype=paddle.get_default_dtype())

        return x.astype(paddle.get_default_dtype()), weight.astype(
            paddle.get_default_dtype()
        )

    def update(self, value: float | paddle.Tensor) -> None:
        """Overwrite in child class."""
        pass

    def compute(self) -> paddle.Tensor:
        """Compute the aggregated value."""
        return getattr(self, self.state_name)


class MaxMetric(BaseAggregator):
    """Aggregate a stream of value into their maximum value.

    Args:
        nan_strategy: 'error', 'warn', 'ignore', 'disable', or a float
        kwargs: Additional keyword arguments
    """

    full_state_update: bool = True

    def __init__(
        self,
        nan_strategy: Literal["error", "warn", "ignore", "disable"]
        | float = "warn",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            "max",
            paddle.to_tensor(-float("inf"), dtype=paddle.get_default_dtype()),
            nan_strategy,
            state_name="max_value",
            **kwargs,
        )

    def update(self, value: float | paddle.Tensor) -> None:
        """Update state with data."""
        value, _ = self._cast_and_nan_check_input(value)
        if value.size != 0:
            self.max_value = paddle.maximum(self.max_value, value.max())

    def plot(
        self,
        val: paddle.Tensor | list[paddle.Tensor] | None = None,
        ax: Any = None,
    ) -> tuple[Any, Any]:
        """Plot a single or multiple values from the metric."""
        import matplotlib.pyplot as plt

        if val is None:
            val = self.compute()
        fig, ax = plt.subplots() if ax is None else (None, ax)
        if isinstance(val, paddle.Tensor) and val.ndim == 0:
            ax.bar(0, val.item(), width=0.5, color="skyblue", edgecolor="black")
            ax.set_xticks([0])
            ax.set_xticklabels(["Max"])
            ax.set_ylabel("Value")
            ax.set_title(self.name if self.name else "Maximum Value")
            ax.text(
                0,
                val.item() + 0.02,
                f"{val.item():.3f}",
                ha="center",
                va="bottom",
            )
        else:
            values = [
                v.item() if isinstance(v, paddle.Tensor) else v for v in val
            ]
            ax.plot(
                values, marker="o", linestyle="-", linewidth=2, markersize=6
            )
            ax.set_xlabel("Step")
            ax.set_ylabel("Maximum Value")
            ax.grid(True)
            ax.set_title(self.name if self.name else "Maximum Value Over Time")
        return fig, ax


class MinMetric(BaseAggregator):
    """Aggregate a stream of value into their minimum value.

    Args:
        nan_strategy: 'error', 'warn', 'ignore', 'disable', or a float
        kwargs: Additional keyword arguments
    """

    full_state_update: bool = True

    def __init__(
        self,
        nan_strategy: Literal["error", "warn", "ignore", "disable"]
        | float = "warn",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            "min",
            paddle.to_tensor(float("inf"), dtype=paddle.get_default_dtype()),
            nan_strategy,
            state_name="min_value",
            **kwargs,
        )

    def update(self, value: float | paddle.Tensor) -> None:
        """Update state with data."""
        value, _ = self._cast_and_nan_check_input(value)
        if value.size != 0:
            self.min_value = paddle.minimum(self.min_value, value.min())

    def plot(
        self,
        val: paddle.Tensor | list[paddle.Tensor] | None = None,
        ax: Any = None,
    ) -> tuple[Any, Any]:
        """Plot a single or multiple values from the metric."""
        import matplotlib.pyplot as plt

        if val is None:
            val = self.compute()
        fig, ax = plt.subplots() if ax is None else (None, ax)
        if isinstance(val, paddle.Tensor) and val.ndim == 0:
            ax.bar(0, val.item(), width=0.5, color="skyblue", edgecolor="black")
            ax.set_xticks([0])
            ax.set_xticklabels(["Min"])
            ax.set_ylabel("Value")
            ax.set_title(self.name if self.name else "Minimum Value")
            ax.text(
                0,
                val.item() + 0.02,
                f"{val.item():.3f}",
                ha="center",
                va="bottom",
            )
        else:
            values = [
                v.item() if isinstance(v, paddle.Tensor) else v for v in val
            ]
            ax.plot(
                values, marker="o", linestyle="-", linewidth=2, markersize=6
            )
            ax.set_xlabel("Step")
            ax.set_ylabel("Minimum Value")
            ax.grid(True)
            ax.set_title(self.name if self.name else "Minimum Value Over Time")
        return fig, ax


class SumMetric(BaseAggregator):
    """Aggregate a stream of value into their sum.

    Args:
        nan_strategy: 'error', 'warn', 'ignore', 'disable', or a float
        kwargs: Additional keyword arguments
    """

    def __init__(
        self,
        nan_strategy: Literal["error", "warn", "ignore", "disable"]
        | float = "warn",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            "sum",
            paddle.to_tensor(0.0, dtype=paddle.get_default_dtype()),
            nan_strategy,
            state_name="sum_value",
            **kwargs,
        )

    def update(self, value: float | paddle.Tensor) -> None:
        """Update state with data."""
        value, _ = self._cast_and_nan_check_input(value)
        if value.size != 0:
            self.sum_value += value.sum()

    def plot(
        self,
        val: paddle.Tensor | list[paddle.Tensor] | None = None,
        ax: Any = None,
    ) -> tuple[Any, Any]:
        """Plot a single or multiple values from the metric."""
        import matplotlib.pyplot as plt

        if val is None:
            val = self.compute()
        fig, ax = plt.subplots() if ax is None else (None, ax)
        if isinstance(val, paddle.Tensor) and val.ndim == 0:
            ax.bar(0, val.item(), width=0.5, color="skyblue", edgecolor="black")
            ax.set_xticks([0])
            ax.set_xticklabels(["Sum"])
            ax.set_ylabel("Value")
            ax.set_title(self.name if self.name else "Sum Value")
            ax.text(
                0,
                val.item() + 0.02,
                f"{val.item():.3f}",
                ha="center",
                va="bottom",
            )
        else:
            values = [
                v.item() if isinstance(v, paddle.Tensor) else v for v in val
            ]
            ax.plot(
                values, marker="o", linestyle="-", linewidth=2, markersize=6
            )
            ax.set_xlabel("Step")
            ax.set_ylabel("Sum Value")
            ax.grid(True)
            ax.set_title(self.name if self.name else "Sum Value Over Time")
        return fig, ax


class CatMetric(BaseAggregator):
    """Concatenate a stream of values.

    Args:
        nan_strategy: 'error', 'warn', 'ignore', 'disable', or a float
        kwargs: Additional keyword arguments
    """

    def __init__(
        self,
        nan_strategy: Literal["error", "warn", "ignore", "disable"]
        | float = "warn",
        **kwargs: Any,
    ) -> None:
        super().__init__("cat", [], nan_strategy, **kwargs)

    def update(self, value: float | paddle.Tensor) -> None:
        """Update state with data."""
        value, _ = self._cast_and_nan_check_input(value)
        if value.size != 0:
            self.value.append(value)

    def compute(self) -> paddle.Tensor:
        """Compute the aggregated value."""
        if isinstance(self.value, list) and self.value:
            return _dim_zero_cat(self.value)
        return self.value

    def plot(
        self,
        val: paddle.Tensor | list[paddle.Tensor] | None = None,
        ax: Any = None,
    ) -> tuple[Any, Any]:
        """Plot a single or multiple values from the metric."""
        import matplotlib.pyplot as plt

        if val is None:
            val = self.compute()
        fig, ax = plt.subplots() if ax is None else (None, ax)
        if isinstance(val, paddle.Tensor) and val.ndim == 1:
            ax.hist(
                val.numpy(),
                bins=20,
                color="skyblue",
                edgecolor="black",
                alpha=0.7,
            )
            ax.set_xlabel("Value")
            ax.set_ylabel("Frequency")
            ax.set_title(self.name if self.name else "Distribution of Values")
            ax.grid(True, alpha=0.3)
        else:
            raise NotImplementedError(
                "Plotting multiple values for CatMetric is not supported"
            )
        return fig, ax


class MeanMetric(BaseAggregator):
    """Aggregate a stream of value into their mean value.

    Args:
        nan_strategy: 'error', 'warn', 'ignore', 'disable', or a float
        kwargs: Additional keyword arguments
    """

    def __init__(
        self,
        nan_strategy: Literal["error", "warn", "ignore", "disable"]
        | float = "warn",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            "sum",
            paddle.to_tensor(0.0, dtype=paddle.get_default_dtype()),
            nan_strategy,
            state_name="mean_value",
            **kwargs,
        )
        self.declare(
            "weight",
            default=paddle.to_tensor(0.0, dtype=paddle.get_default_dtype()),
            dist_reduce_fn="sum",
        )

    def update(
        self,
        value: float | paddle.Tensor,
        weight: float | paddle.Tensor | None = None,
    ) -> None:
        """Update state with data."""
        if not isinstance(value, paddle.Tensor):
            value = paddle.to_tensor(value, dtype=paddle.get_default_dtype())
        if weight is None:
            weight = paddle.ones_like(value, dtype=paddle.get_default_dtype())
        elif not isinstance(weight, paddle.Tensor):
            weight = paddle.to_tensor(weight, dtype=paddle.get_default_dtype())
        weight = paddle.broadcast_to(weight, value.shape)
        value, weight = self._cast_and_nan_check_input(value, weight)
        if value.size == 0:
            return
        self.mean_value += (value * weight).sum()
        self.weight += weight.sum()

    def compute(self) -> paddle.Tensor:
        """Compute the aggregated value."""
        return self.mean_value / self.weight

    def plot(
        self,
        val: paddle.Tensor | list[paddle.Tensor] | None = None,
        ax: Any = None,
    ) -> tuple[Any, Any]:
        """Plot a single or multiple values from the metric."""
        import matplotlib.pyplot as plt

        if val is None:
            val = self.compute()
        fig, ax = plt.subplots() if ax is None else (None, ax)
        if isinstance(val, paddle.Tensor) and val.ndim == 0:
            ax.bar(0, val.item(), width=0.5, color="skyblue", edgecolor="black")
            ax.set_xticks([0])
            ax.set_xticklabels(["Mean"])
            ax.set_ylabel("Value")
            ax.set_title(self.name if self.name else "Mean Value")
            ax.text(
                0,
                val.item() + 0.02,
                f"{val.item():.3f}",
                ha="center",
                va="bottom",
            )
        else:
            values = [
                v.item() if isinstance(v, paddle.Tensor) else v for v in val
            ]
            ax.plot(
                values, marker="o", linestyle="-", linewidth=2, markersize=6
            )
            ax.set_xlabel("Step")
            ax.set_ylabel("Mean Value")
            ax.grid(True)
            ax.set_title(self.name if self.name else "Mean Value Over Time")
        return fig, ax


class RunningMean(Metric):
    """Aggregate a stream of value into their mean over a running window.

    Args:
        window: The size of the running window.
        nan_strategy: 'error', 'warn', 'ignore', 'disable', or a float
        kwargs: Additional keyword arguments
    """

    def __init__(
        self,
        window: int = 5,
        nan_strategy: Literal["error", "warn", "ignore", "disable"]
        | float = "warn",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.window = window
        self.nan_strategy = nan_strategy
        self.declare("buffer", default=[], dist_reduce_fn="cat")

    def update(self, value: float | paddle.Tensor) -> None:
        """Update state with data."""
        if not isinstance(value, paddle.Tensor):
            value = paddle.to_tensor(value, dtype=paddle.get_default_dtype())
        self.buffer.append(value)
        if len(self.buffer) > self.window:
            self.buffer = self.buffer[-self.window :]

    def compute(self) -> paddle.Tensor:
        """Compute the running mean."""
        if not self.buffer:
            return paddle.to_tensor(0.0)

        values = []
        for v in self.buffer:
            if isinstance(v, paddle.Tensor):
                v_flat = v.flatten()
                if self.nan_strategy == "disable":
                    values.append(v_flat)
                elif self.nan_strategy == "error":
                    if paddle.isnan(v_flat).any():
                        raise RuntimeError("Encountered `nan` values in tensor")
                    values.append(v_flat)
                elif self.nan_strategy == "warn":
                    if paddle.isnan(v_flat).any():
                        warnings.warn(
                            "Encountered `nan` values in tensor. Will be removed.",
                            UserWarning,
                        )
                    v_clean = v_flat[~paddle.isnan(v_flat)]
                    if v_clean.size > 0:
                        values.append(v_clean)
                elif self.nan_strategy == "ignore":
                    v_clean = v_flat[~paddle.isnan(v_flat)]
                    if v_clean.size > 0:
                        values.append(v_clean)
                else:
                    v_imputed = paddle.where(
                        paddle.isnan(v_flat), self.nan_strategy, v_flat
                    )
                    values.append(v_imputed)
            else:
                values.append(paddle.to_tensor(v))

        if not values:
            return paddle.to_tensor(0.0)

        all_values = paddle.concat([v.flatten() for v in values])
        return all_values.mean()

    def plot(
        self,
        val: paddle.Tensor | list[paddle.Tensor] | None = None,
        ax: Any = None,
    ) -> tuple[Any, Any]:
        """Plot a single or multiple values from the metric."""
        import matplotlib.pyplot as plt

        if val is None:
            val = self.compute()
        fig, ax = plt.subplots() if ax is None else (None, ax)
        if isinstance(val, paddle.Tensor) and val.ndim == 0:
            ax.bar(0, val.item(), width=0.5, color="skyblue", edgecolor="black")
            ax.set_xticks([0])
            ax.set_xticklabels(["Running Mean"])
            ax.set_ylabel("Value")
            ax.set_title(
                self.name
                if self.name
                else f"Running Mean (window={self.window})"
            )
            ax.text(
                0,
                val.item() + 0.02,
                f"{val.item():.3f}",
                ha="center",
                va="bottom",
            )
        else:
            values = [
                v.item() if isinstance(v, paddle.Tensor) else v for v in val
            ]
            ax.plot(
                values, marker="o", linestyle="-", linewidth=2, markersize=6
            )
            ax.set_xlabel("Step")
            ax.set_ylabel("Running Mean")
            ax.grid(True)
            ax.set_title(
                self.name
                if self.name
                else f"Running Mean Over Time (window={self.window})"
            )
        return fig, ax


class RunningSum(Metric):
    """Aggregate a stream of value into their sum over a running window.

    Args:
        window: The size of the running window.
        nan_strategy: 'error', 'warn', 'ignore', 'disable', or a float
        kwargs: Additional keyword arguments
    """

    def __init__(
        self,
        window: int = 5,
        nan_strategy: Literal["error", "warn", "ignore", "disable"]
        | float = "warn",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.window = window
        self.nan_strategy = nan_strategy
        self.declare("buffer", default=[], dist_reduce_fn="cat")

    def update(self, value: float | paddle.Tensor) -> None:
        """Update state with data."""
        if not isinstance(value, paddle.Tensor):
            value = paddle.to_tensor(value, dtype=paddle.get_default_dtype())
        self.buffer.append(value)
        if len(self.buffer) > self.window:
            self.buffer = self.buffer[-self.window :]

    def compute(self) -> paddle.Tensor:
        """Compute the running sum."""
        if not self.buffer:
            return paddle.to_tensor(0.0)

        values = []
        for v in self.buffer:
            if isinstance(v, paddle.Tensor):
                v_flat = v.flatten()
                if self.nan_strategy == "disable":
                    values.append(v_flat)
                elif self.nan_strategy == "error":
                    if paddle.isnan(v_flat).any():
                        raise RuntimeError("Encountered `nan` values in tensor")
                    values.append(v_flat)
                elif self.nan_strategy == "warn":
                    if paddle.isnan(v_flat).any():
                        warnings.warn(
                            "Encountered `nan` values in tensor. Will be removed.",
                            UserWarning,
                        )
                    v_clean = v_flat[~paddle.isnan(v_flat)]
                    if v_clean.size > 0:
                        values.append(v_clean)
                elif self.nan_strategy == "ignore":
                    v_clean = v_flat[~paddle.isnan(v_flat)]
                    if v_clean.size > 0:
                        values.append(v_clean)
                else:
                    v_imputed = paddle.where(
                        paddle.isnan(v_flat), self.nan_strategy, v_flat
                    )
                    values.append(v_imputed)
            else:
                values.append(paddle.to_tensor(v))

        if not values:
            return paddle.to_tensor(0.0)

        all_values = paddle.concat([v.flatten() for v in values])
        return all_values.sum()

    def plot(
        self,
        val: paddle.Tensor | list[paddle.Tensor] | None = None,
        ax: Any = None,
    ) -> tuple[Any, Any]:
        """Plot a single or multiple values from the metric."""
        import matplotlib.pyplot as plt

        if val is None:
            val = self.compute()
        fig, ax = plt.subplots() if ax is None else (None, ax)
        if isinstance(val, paddle.Tensor) and val.ndim == 0:
            ax.bar(0, val.item(), width=0.5, color="skyblue", edgecolor="black")
            ax.set_xticks([0])
            ax.set_xticklabels(["Running Sum"])
            ax.set_ylabel("Value")
            ax.set_title(
                self.name
                if self.name
                else f"Running Sum (window={self.window})"
            )
            ax.text(
                0,
                val.item() + 0.02,
                f"{val.item():.3f}",
                ha="center",
                va="bottom",
            )
        else:
            values = [
                v.item() if isinstance(v, paddle.Tensor) else v for v in val
            ]
            ax.plot(
                values, marker="o", linestyle="-", linewidth=2, markersize=6
            )
            ax.set_xlabel("Step")
            ax.set_ylabel("Running Sum")
            ax.grid(True)
            ax.set_title(
                self.name
                if self.name
                else f"Running Sum Over Time (window={self.window})"
            )
        return fig, ax
