# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
from __future__ import annotations

#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Metric Collection for grouping multiple metrics together."""

import warnings
from collections import OrderedDict
from collections.abc import Hashable, Iterable, Iterator, Sequence
from copy import deepcopy
from typing import Any

from typing_extensions import Literal

import paddle
from paddle.metric.metric import Metric


def _flatten(x: Sequence) -> list:
    """Flatten list of list into single list."""
    return [item for sublist in x for item in sublist]


def _flatten_dict(x: dict) -> tuple[dict, bool]:
    """Flatten dict of dicts into single dict and checking for duplicates in keys along the way."""
    new_dict = {}
    duplicates = False
    for key, value in x.items():
        if isinstance(value, dict):
            for k, v in value.items():
                if k in new_dict:
                    duplicates = True
                new_dict[k] = v
        else:
            if key in new_dict:
                duplicates = True
            new_dict[key] = value
    return new_dict, duplicates


def allclose(tensor1: paddle.Tensor, tensor2: paddle.Tensor) -> bool:
    """Check if two tensors are close."""
    if tensor1.dtype != tensor2.dtype:
        tensor2 = tensor2.astype(tensor1.dtype)
    return paddle.allclose(x=tensor1, y=tensor2).item()


def _remove_prefix(string: str, prefix: str) -> str:
    """Remove prefix from string if present."""
    return string.removeprefix(prefix)


def _remove_suffix(string: str, suffix: str) -> str:
    """Remove suffix from string if present."""
    return string.removesuffix(suffix)


def plot_single_or_multi_val(
    val: paddle.Tensor | Sequence[paddle.Tensor] | dict[str, paddle.Tensor] | Sequence[dict[str, paddle.Tensor]],
    ax: Any = None,
    higher_is_better: bool | None = None,
    lower_bound: float | None = None,
    upper_bound: float | None = None,
    legend_name: str | None = None,
    name: str | None = None,
) -> tuple[Any, Any]:
    """Plot a single metric value or multiple values."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots() if ax is None else (None, ax)
    ax.get_xaxis().set_visible(False)

    if isinstance(val, paddle.Tensor):
        if val.size == 1:
            ax.plot([val.item()], marker="o", markersize=10)
        else:
            for i, v in enumerate(val):
                label = f"{legend_name} {i}" if legend_name else f"{i}"
                ax.plot(i, v.item(), marker="o", markersize=10, linestyle="None", label=label)
    elif isinstance(val, dict):
        for i, (k, v) in enumerate(val.items()):
            if v.size != 1:
                ax.plot(v.numpy(), marker="o", markersize=10, linestyle="-", label=k)
                ax.get_xaxis().set_visible(True)
                ax.set_xlabel("Step")
                ax.set_xticks(range(len(v)))
            else:
                ax.plot(i, v.item(), marker="o", markersize=10, label=k)
    elif isinstance(val, Sequence):
        n_steps = len(val)
        if isinstance(val[0], dict):
            val_dict = {k: paddle.stack([val[i][k] for i in range(n_steps)]) for k in val[0]}
            for k, v in val_dict.items():
                ax.plot(v.numpy(), marker="o", markersize=10, linestyle="-", label=k)
        else:
            val_tensor = paddle.stack(val, 0)
            multi_series = val_tensor.ndim != 1
            val_tensor = val_tensor.T if multi_series else val_tensor.unsqueeze(0)
            for i, v in enumerate(val_tensor):
                label = (f"{legend_name} {i}" if legend_name else f"{i}") if multi_series else ""
                ax.plot(v.numpy(), marker="o", markersize=10, linestyle="-", label=label)
        ax.get_xaxis().set_visible(True)
        ax.set_xlabel("Step")
        ax.set_xticks(range(n_steps))
    else:
        raise ValueError("Got unknown format for argument `val`.")

    handles, labels = ax.get_legend_handles_labels()
    if handles and labels:
        ax.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=3, fancybox=True, shadow=True)

    ylim = ax.get_ylim()
    factor = 0.1 * (upper_bound - lower_bound) if (lower_bound is not None and upper_bound is not None) else 0.1 * (ylim[1] - ylim[0])
    ax.set_ylim(
        bottom=lower_bound - factor if lower_bound is not None else ylim[0] - factor,
        top=upper_bound + factor if upper_bound is not None else ylim[1] + factor,
    )
    ax.grid(True)
    ax.set_ylabel(name if name is not None else None)

    xlim = ax.get_xlim()
    xfactor = 0.1 * (xlim[1] - xlim[0])
    y_lines = [v for v in [lower_bound, upper_bound] if v is not None]
    if y_lines:
        ax.hlines(y_lines, xlim[0], xlim[1], linestyles="dashed", colors="k")

    if higher_is_better is not None:
        if lower_bound is not None and not higher_is_better:
            ax.set_xlim(xlim[0] - xfactor, xlim[1])
            ax.text(xlim[0], lower_bound, s="Optimal \n value", horizontalalignment="center", verticalalignment="center")
        if upper_bound is not None and higher_is_better:
            ax.set_xlim(xlim[0] - xfactor, xlim[1])
            ax.text(xlim[0], upper_bound, s="Optimal \n value", horizontalalignment="center", verticalalignment="center")

    return fig, ax


class MetricCollection(paddle.nn.LayerDict):
    """MetricCollection class chains metrics with the same call pattern into one single class.

    Args:
        metrics: list/tuple, dict, or Metric/MetricCollection instances
        prefix: string to prepend to output dict keys
        postfix: string to append to output dict keys
        compute_groups: enable compute group optimization (default True)
    """

    def __init__(
        self,
        metrics: Metric | MetricCollection | Sequence[Metric | MetricCollection] | dict[str, Metric | MetricCollection],
        *additional_metrics: Metric,
        prefix: str | None = None,
        postfix: str | None = None,
        compute_groups: bool | list[list[str]] = True,
    ) -> None:
        super().__init__()
        self.prefix = self._check_arg(prefix, "prefix")
        self.postfix = self._check_arg(postfix, "postfix")
        self._enable_compute_groups = compute_groups
        self._groups_checked: bool = False
        self._state_is_copy: bool = False
        self._groups: dict[int, list[str]] = {}
        self._from_collection: bool = False
        self.add_metrics(metrics, *additional_metrics)

    @property
    def metric_state(self) -> dict[str, dict[str, Any]]:
        """Get the current state of the metric."""
        return {k: m.metric_state for k, m in self.items(keep_base=False, copy_state=False)}

    def forward(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """Call forward for each metric sequentially."""
        return self._compute_and_reduce("forward", *args, **kwargs)

    def update(self, *args: Any, **kwargs: Any) -> None:
        """Call update for each metric sequentially."""
        if self._groups_checked:
            for k in self.keys(keep_base=True):
                mi = getattr(self, str(k))
                mi._computed = None
            for cg in self._groups.values():
                m0 = getattr(self, cg[0])
                m0.update(*args, **m0._filter_kwargs(**kwargs))
            self._state_is_copy = False
            self._compute_groups_create_state_ref()
        else:
            for m in self.values(copy_state=False):
                m.update(*args, **m._filter_kwargs(**kwargs))
            if self._enable_compute_groups:
                self._merge_compute_groups()
                self._state_is_copy = False
                self._compute_groups_create_state_ref()
                self._groups_checked = True

    def _merge_compute_groups(self) -> None:
        """Iterate over metrics, merge compute groups with matching states."""
        num_groups = len(self._groups)
        while True:
            for cg_idx1, cg_members1 in deepcopy(self._groups).items():
                for cg_idx2, cg_members2 in deepcopy(self._groups).items():
                    if cg_idx1 == cg_idx2:
                        continue
                    metric1 = getattr(self, cg_members1[0])
                    metric2 = getattr(self, cg_members2[0])
                    if self._equal_metric_states(metric1, metric2):
                        self._groups[cg_idx1].extend(self._groups.pop(cg_idx2))
                        break
                if len(self._groups) != num_groups:
                    break
            if len(self._groups) == num_groups:
                break
            num_groups = len(self._groups)
        temp = deepcopy(self._groups)
        self._groups = {}
        for idx, values in enumerate(temp.values()):
            self._groups[idx] = values

    @staticmethod
    def _equal_metric_states(metric1: Metric, metric2: Metric) -> bool:
        """Check if the metric state of two metrics are the same."""
        if len(metric1._defaults) == 0 or len(metric2._defaults) == 0:
            return False
        if metric1._defaults.keys() != metric2._defaults.keys():
            return False
        for key in metric1._defaults:
            state1 = getattr(metric1, key)
            state2 = getattr(metric2, key)
            if type(state1) != type(state2):
                return False
            if isinstance(state1, paddle.Tensor) and isinstance(state2, paddle.Tensor):
                if not (state1.shape == state2.shape and allclose(state1, state2)):
                    return False
            if isinstance(state1, list) and isinstance(state2, list):
                if not all(s1.shape == s2.shape and allclose(s1, s2) for s1, s2 in zip(state1, state2)):
                    return False
        return True

    def _compute_groups_create_state_ref(self, copy: bool = False) -> None:
        """Create reference between metrics in the same compute group."""
        if not self._state_is_copy:
            for cg in self._groups.values():
                m0 = getattr(self, cg[0])
                for i in range(1, len(cg)):
                    mi = getattr(self, cg[i])
                    for state in m0._defaults:
                        m0_state = getattr(m0, state)
                        setattr(mi, state, deepcopy(m0_state) if copy else m0_state)
                    mi._update_count = deepcopy(m0._update_count) if copy else m0._update_count
        self._state_is_copy = copy

    def compute(self) -> dict[str, Any]:
        """Compute the result for each metric in the collection."""
        return self._compute_and_reduce("compute")

    def _compute_and_reduce(self, method_name: Literal["compute", "forward"], *args: Any, **kwargs: Any) -> dict[str, Any]:
        """Compute result from collection and reduce into a single dictionary."""
        result = {}
        for k, m in self.items(keep_base=True, copy_state=False):
            if method_name == "compute":
                res = m.compute()
            elif method_name == "forward":
                res = m(*args, **m._filter_kwargs(**kwargs))
            else:
                raise ValueError(f"method_name should be either 'compute' or 'forward', but got {method_name}")
            result[k] = res
        _, duplicates = _flatten_dict(result)
        flattened_results = {}
        for k, m in self.items(keep_base=True, copy_state=False):
            res = result[k]
            if isinstance(res, dict):
                for key, v in res.items():
                    if duplicates:
                        stripped_k = k.replace(getattr(m, "prefix", ""), "")
                        stripped_k = stripped_k.replace(getattr(m, "postfix", ""), "")
                        key = f"{stripped_k}_{key}"
                    if getattr(m, "_from_collection", False) and m.prefix is not None:
                        key = f"{m.prefix}{key}"
                    if getattr(m, "_from_collection", False) and m.postfix is not None:
                        key = f"{key}{m.postfix}"
                    flattened_results[key] = v
            else:
                flattened_results[k] = res
        return {self._set_name(k): v for k, v in flattened_results.items()}

    def reset(self) -> None:
        """Call reset for each metric sequentially."""
        for m in self.values(copy_state=False):
            m.reset()
        if self._enable_compute_groups and self._groups_checked:
            self._compute_groups_create_state_ref()

    def clone(self, prefix: str | None = None, postfix: str | None = None) -> MetricCollection:
        """Make a copy of the metric collection."""
        mc = deepcopy(self)
        if prefix:
            mc.prefix = self._check_arg(prefix, "prefix")
        if postfix:
            mc.postfix = self._check_arg(postfix, "postfix")
        return mc

    def persistent(self, mode: bool = True) -> None:
        """Change if metric states should be saved to its state_dict after initialization."""
        for m in self.values(copy_state=False):
            m.persistent(mode)

    def add_metrics(
        self,
        metrics: Metric | MetricCollection | Sequence[Metric | MetricCollection] | dict[str, Metric | MetricCollection],
        *additional_metrics: Metric,
    ) -> None:
        """Add new metrics to Metric Collection."""
        if isinstance(metrics, Metric):
            metrics = [metrics]
        if isinstance(metrics, Sequence):
            metrics = list(metrics)
            remain: list = []
            for m in additional_metrics:
                sel = metrics if isinstance(m, Metric) else remain
                sel.append(m)
            if remain:
                warnings.warn(f"You have passes extra arguments {remain} which are not `Metric` so they will be ignored.")
        elif additional_metrics:
            raise ValueError(
                f"You have passes extra arguments {additional_metrics} which are not compatible with first passed dictionary {metrics} so they will be ignored."
            )
        if isinstance(metrics, dict):
            for name in sorted(metrics.keys()):
                metric = metrics[name]
                if not isinstance(metric, (Metric, MetricCollection)):
                    raise ValueError(f"Value {metric} belonging to key {name} is not an instance of `Metric` or `MetricCollection`")
                if isinstance(metric, Metric):
                    self[name] = metric
                else:
                    for k, v in metric.items(keep_base=False):
                        v.postfix = metric.postfix
                        v.prefix = metric.prefix
                        v._from_collection = True
                        self[f"{name}_{k}"] = v
        elif isinstance(metrics, Sequence):
            for metric in metrics:
                if not isinstance(metric, (Metric, MetricCollection)):
                    raise ValueError(f"Input {metric} to `MetricCollection` is not a instance of `Metric` or `MetricCollection`")
                if isinstance(metric, Metric):
                    name = metric.__class__.__name__
                    if name in self:
                        raise ValueError(f"Encountered two metrics both named {name}")
                    self[name] = metric
                else:
                    for k, v in metric.items(keep_base=False):
                        v.postfix = metric.postfix
                        v.prefix = metric.prefix
                        v._from_collection = True
                        self[k] = v
        elif isinstance(metrics, MetricCollection):
            for name, metric in metrics.items(keep_base=False):
                if name in self:
                    raise ValueError(f"Metric with name '{name}' already exists in the collection.")
                self[name] = metric
        else:
            raise ValueError(
                f"Unknown input to MetricCollection. Expected, `Metric`, `MetricCollection` or `dict`/`sequence` of the previous, but got {metrics}"
            )
        self._groups_checked = False
        if self._enable_compute_groups:
            self._init_compute_groups()
        else:
            self._groups = {}

    def _init_compute_groups(self) -> None:
        """Initialize compute groups."""
        if isinstance(self._enable_compute_groups, list):
            self._groups = dict(enumerate(self._enable_compute_groups))
            for v in self._groups.values():
                for metric in v:
                    if metric not in self:
                        raise ValueError(
                            f"Input {metric} in `compute_groups` argument does not match a metric in the collection."
                        )
            already_in_group = _flatten(self._groups.values())
            counter = len(self._groups)
            for k in self.keys(keep_base=True):
                if k not in already_in_group:
                    self._groups[counter] = [k]
                    counter += 1
            self._groups_checked = True
        else:
            self._groups = {i: [str(k)] for i, k in enumerate(self.keys(keep_base=True))}

    @property
    def compute_groups(self) -> dict[int, list[str]]:
        """Return a dict with the current compute groups in the collection."""
        return self._groups

    def _set_name(self, base: str) -> str:
        """Adjust name of metric with both prefix and postfix."""
        name = base if self.prefix is None else self.prefix + base
        return name if self.postfix is None else name + self.postfix

    def _to_renamed_dict(self) -> dict[str, Metric]:
        """Return a dict with renamed keys."""
        dict_modules = OrderedDict()
        for k, v in self._modules.items():
            dict_modules[self._set_name(k)] = v
        return dict_modules

    def __iter__(self) -> Iterator[Hashable]:
        """Return an iterator over the keys of the MetricDict."""
        return iter(self.keys())

    def keys(self, keep_base: bool = False) -> Iterable[Hashable]:
        """Return an iterable of the LayerDict key."""
        if keep_base:
            return self._modules.keys()
        return self._to_renamed_dict().keys()

    def items(self, keep_base: bool = False, copy_state: bool = True) -> Iterable[tuple[str, Metric]]:
        """Return an iterable of the LayerDict key/value pairs."""
        self._compute_groups_create_state_ref(copy_state)
        if keep_base:
            return self._modules.items()
        return self._to_renamed_dict().items()

    def values(self, copy_state: bool = True) -> Iterable[Metric]:
        """Return an iterable of the LayerDict values."""
        self._compute_groups_create_state_ref(copy_state)
        return self._modules.values()

    def __getitem__(self, key: str, copy_state: bool = True) -> Metric:
        """Retrieve a single metric from the collection."""
        self._compute_groups_create_state_ref(copy_state)
        if self.prefix:
            key = _remove_prefix(key, self.prefix)
        if self.postfix:
            key = _remove_suffix(key, self.postfix)
        return self._modules[key]

    @staticmethod
    def _check_arg(arg: str | None, name: str) -> str | None:
        """Check if argument is a string or None."""
        if arg is None or isinstance(arg, str):
            return arg
        raise ValueError(f"Expected input `{name}` to be a string, but got {type(arg)}")

    def __repr__(self) -> str:
        """Return the representation of the metric collection."""
        repr_str = super().__repr__()[:-2]
        if self.prefix:
            repr_str += f",\n  prefix={self.prefix}{',' if self.postfix else ''}"
        if self.postfix:
            repr_str += f"{',' if not self.prefix else ''}\n  postfix={self.postfix}"
        return repr_str + "\n)"

    def set_dtype(self, dst_type: str | paddle.dtype) -> MetricCollection:
        """Transfer all metric state to specific dtype."""
        for m in self.values(copy_state=False):
            m.set_dtype(dst_type)
        return self

    def plot(
        self,
        val: dict | Sequence[dict] | None = None,
        ax: Any | Sequence[Any] | None = None,
        together: bool = False,
    ) -> Sequence[tuple[Any, Any]] | tuple[Any, Any]:
        """Plot a single or multiple values from the metric."""
        import matplotlib.pyplot as plt

        if not isinstance(together, bool):
            raise ValueError(f"Expected argument `together` to be a boolean, but got {type(together)}")
        if ax is not None:
            if together and not isinstance(ax, plt.Axes):
                raise ValueError(f"Expected argument `ax` to be a matplotlib axis object, but got {type(ax)} when `together=True`")
            if not together and not (isinstance(ax, Sequence) and all(isinstance(a, plt.Axes) for a in ax) and len(ax) == len(self)):
                raise ValueError(f"Expected argument `ax` to be a sequence of matplotlib axis objects, but got {type(ax)} when `together=False`")
        val = val or self.compute()
        if together:
            return plot_single_or_multi_val(val, ax=ax)
        fig_axs = []
        for i, (k, m) in enumerate(self.items(keep_base=False, copy_state=False)):
            if isinstance(val, dict):
                f, a = m.plot(val[k], ax=ax[i] if ax is not None else ax)
            elif isinstance(val, Sequence):
                f, a = m.plot([v[k] for v in val], ax=ax[i] if ax is not None else ax)
            fig_axs.append((f, a))
        return fig_axs
