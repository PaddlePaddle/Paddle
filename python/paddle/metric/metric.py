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

"""Paddle metric base classes: Metric, CompositionalMetric.

Merged from the user's paddle-native implementation and paddle.metric upstream features.
"""

from __future__ import annotations

import functools
import inspect
import warnings
from abc import ABC, abstractmethod
from collections.abc import Generator, Sequence
from contextlib import contextmanager
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Callable, ClassVar

import paddle
import paddle.distributed as dist
from paddle import nn

if TYPE_CHECKING:
    import builtins

    from paddle import Tensor
    from paddle.base.core import ProcessGroup

__all__ = ["Metric", "CompositionalMetric"]


# ============ Utility Functions ============


def _dim_zero_sum(x: Tensor) -> Tensor:
    return x.sum(axis=0)


def _dim_zero_mean(x: Tensor) -> Tensor:
    return x.mean(axis=0)


def _dim_zero_min(x: Tensor) -> Tensor:
    return x.min(axis=0)


def _dim_zero_max(x: Tensor) -> Tensor:
    return x.max(axis=0)


def _dim_zero_cat(x: Tensor | list[Tensor]) -> Tensor:
    if isinstance(x, (list, tuple)):
        return paddle.concat(x, axis=0)
    return x


def _flatten(x: list) -> list:
    return [item for sublist in x for item in sublist]


def _squeeze_if_scalar(x: Any) -> Any:
    if isinstance(x, paddle.Tensor) and x.numel() == 1:
        return x.squeeze()
    return x


def _apply_to_collection(
    data: Any,
    dtype: Any,
    func: Callable,
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Recursively apply a function to all elements of a collection matching dtype."""
    if isinstance(data, dtype):
        return func(data, *args, **kwargs)
    if isinstance(data, (list, tuple)):
        return type(data)(
            _apply_to_collection(d, dtype, func, *args, **kwargs) for d in data
        )
    if isinstance(data, dict):
        return {
            k: _apply_to_collection(v, dtype, func, *args, **kwargs)
            for k, v in data.items()
        }
    return data


def _gather_all_tensors(
    tensor: Tensor, group: ProcessGroup | None = None
) -> list[Tensor]:
    if not dist.is_initialized():
        return [tensor]
    group = group or dist.get_world_group()
    world_size = dist.get_world_size(group)
    if world_size == 1:
        return [tensor]
    tensor_list = [paddle.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(tensor_list, tensor, group=group)
    return tensor_list


def _distributed_available() -> bool:
    return dist.is_available() and dist.is_initialized()


def _neg(x: paddle.Tensor) -> paddle.Tensor:
    return -paddle.abs(x)


# ============ Metric Base Class ============


class Metric(ABC, nn.Layer):
    """Base class for all metrics.

    Inherits from :class:`paddle.nn.Layer` and provides:

    1. State management through :meth:`declare` (alias :meth:`declare`).
    2. Automatic device synchronization via :class:`paddle.nn.Layer`.
    3. Distributed synchronization across processes.
    4. Serialization support via ``state_dict`` / ``set_state_dict``.
    5. Operator overloading for metric composition.

    Args:
        name: Name of the metric. If None, uses class name.
        dist_reduce_fn: Default distributed reduction function for all states.
            Can be ``'sum'``, ``'mean'``, ``'cat'``, ``'min'``, ``'max'``, or a callable.
        sync_on_compute: Whether to synchronize states when :meth:`compute` is called.
        dist_sync_on_step: Whether to synchronize on :meth:`forward`.
        process_group: Process group for distributed synchronization.
        compute_with_cache: Whether to cache the result of :meth:`compute`.
        compute_on_cpu: Whether to move list states to CPU to save GPU memory.
        dist_sync_fn: Custom function for distributed synchronization.
        distributed_available_fn: Function to check if distributed is available.
    """

    __jit_ignored_attributes__: ClassVar[list[str]] = ["device"]
    __jit_unused_properties__: ClassVar[list[str]] = [
        "is_differentiable",
        "higher_is_better",
        "plot_lower_bound",
        "plot_upper_bound",
        "plot_legend_name",
        "metric_state",
        "_update_called",
    ]
    is_differentiable: bool | None = None
    higher_is_better: bool | None = None
    full_state_update: bool | None = None
    plot_lower_bound: float | None = None
    plot_upper_bound: float | None = None
    plot_legend_name: str | None = None

    def __init__(
        self,
        name: str | None = None,
        dist_reduce_fn: str | Callable = "sum",
        sync_on_compute: bool = True,
        dist_sync_on_step: bool = False,
        process_group: ProcessGroup | None = None,
        compute_with_cache: bool = True,
        compute_on_cpu: bool = False,
        dist_sync_fn: Callable | None = None,
        distributed_available_fn: Callable | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        # Backward-compat alias: dist_reduce_fx -> dist_reduce_fn
        if "dist_reduce_fx" in kwargs:
            dist_reduce_fn = kwargs.pop("dist_reduce_fx")
        self._name = name or self.__class__.__name__
        self._device = paddle.CPUPlace()
        self._dtype = paddle.get_default_dtype()

        # Configuration
        self._default_dist_reduce_fn = dist_reduce_fn
        self._sync_on_compute = sync_on_compute
        self._dist_sync_on_step = dist_sync_on_step
        self._process_group = process_group
        self._compute_with_cache = compute_with_cache
        self._compute_on_cpu = compute_on_cpu
        self._dist_sync_fn = dist_sync_fn
        self._distributed_available_fn = (
            distributed_available_fn or _distributed_available
        )

        # State tracking
        self._defaults: dict[str, list | paddle.Tensor] = {}
        self._reductions: dict[str, str | Callable[..., Any] | None] = {}
        self._persistent: dict[str, bool] = {}

        # Internal state
        self._update_count = 0
        self._computed = None
        self._forward_cache = None
        self._is_synced = False
        self._cache: dict[str, list[paddle.Tensor] | paddle.Tensor] | None = (
            None
        )
        self._to_sync = self._sync_on_compute
        self._should_unsync = True
        self._enable_grad = False
        self._dtype_convert = False

        # Signature for kwargs filtering
        self._update_signature = inspect.signature(self.update)

        # Wrap core methods
        self.update: Callable = self._wrap_update(self.update)
        self.compute: Callable = self._wrap_compute(self.compute)

        # Backward-compat alias: add_state -> declare
        self.add_state = self.declare

    # ============ Properties ============

    @property
    def name(self) -> str:
        return self._name

    @property
    def names(self) -> list[str]:
        """Expanded list of names for this metric.

        For single-value metrics this is ``[self.name]``.  Multi-value
        metrics (e.g. ROC, PrecisionRecallCurve) override this to return
        one suffixed name per value, e.g. ``['ROC_0', 'ROC_1', 'ROC_2']``.
        """
        return [self._name]

    @property
    def update_called(self) -> bool:
        return self._update_count > 0

    @property
    def update_count(self) -> int:
        return self._update_count

    @property
    def metric_state(self) -> dict[str, list[paddle.Tensor] | paddle.Tensor]:
        return {attr: getattr(self, attr) for attr in self._defaults}

    @property
    def device(self) -> Any:
        return self._device

    @property
    def dtype(self) -> Any:
        return self._dtype

    @property
    def _update_called(self) -> bool:
        return self.update_called

    # ============ State Management ============

    def declare(
        self,
        name: str,
        default: list | paddle.Tensor,
        dist_reduce_fn: str | Callable | None = None,
        persistent: bool = False,
    ) -> None:
        """Add a metric state variable.

        Metric states behave like buffers/parameters of :class:`~paddle.nn.Layer`:
        they are transferred when ``.to()`` is called and can be synchronized across
        distributed processes.

        Args:
            name: The name of the state variable (accessible as ``self.name``).
            default: Default value — a :class:`~paddle.Tensor` or an empty list.
            dist_reduce_fn: Reduction function for distributed sync.
                ``"sum"``, ``"mean"``, ``"cat"``, ``"min"``, ``"max"``, a callable, or None.
            persistent: Whether the state is saved in ``state_dict``.
        """
        # In static graph mode ``paddle.zeros()`` returns a ``pir.Value``
        # instead of a ``Tensor``.  Accept it gracefully so that metric
        # objects can be *constructed* under ``enable_static()``; actual
        # computation is skipped later via ``_supported_metrics``.
        _is_valid = isinstance(default, (paddle.Tensor, list)) or (
            not paddle.framework.in_dynamic_mode()
            and default is not None
            and not isinstance(default, dict)
        )
        if not _is_valid or (isinstance(default, list) and default):
            raise ValueError(
                "state variable must be a tensor or any empty list (where you can append tensors)"
            )
        if dist_reduce_fn == "sum":
            dist_reduce_fn = _dim_zero_sum
        elif dist_reduce_fn == "mean":
            dist_reduce_fn = _dim_zero_mean
        elif dist_reduce_fn == "max":
            dist_reduce_fn = _dim_zero_max
        elif dist_reduce_fn == "min":
            dist_reduce_fn = _dim_zero_min
        elif dist_reduce_fn == "cat":
            dist_reduce_fn = _dim_zero_cat
        elif dist_reduce_fn is not None and not callable(dist_reduce_fn):
            raise ValueError(
                "`dist_reduce_fn` must be callable or one of ['mean', 'sum', 'cat', 'min', 'max', None]"
            )
        if isinstance(default, paddle.Tensor):
            default = default.contiguous()
            self.register_buffer(name, default, persistable=persistent)
        elif isinstance(default, list):
            setattr(self, name, default)
        else:
            # Static-graph ``pir.Value`` — store as plain attribute; no buffer
            # registration (not a real Tensor) and no deepcopy (Value is not
            # copyable).
            setattr(self, name, default)
            self._defaults[name] = default
            self._persistent[name] = persistent
            self._reductions[name] = dist_reduce_fn
            return
        self._defaults[name] = deepcopy(default)
        self._persistent[name] = persistent
        self._reductions[name] = dist_reduce_fn

    # ============ Core Lifecycle ============

    @abstractmethod
    def update(self, *args: Any, **kwargs: Any) -> None:
        """Override this method to update the state variables of your metric class."""

    @abstractmethod
    def compute(self) -> Any:
        """Override this method to compute the final metric value."""

    def reset(self) -> None:
        """Reset metric state variables to their default value."""
        self._update_count = 0
        self._forward_cache = None
        self._computed = None
        for attr, default in self._defaults.items():
            current_val = getattr(self, attr)
            if isinstance(default, paddle.Tensor):
                device = (
                    current_val.place
                    if hasattr(current_val, "place")
                    else paddle.CPUPlace()
                )
                setattr(self, attr, default.clone().to(device))
            elif isinstance(default, list):
                getattr(self, attr).clear()
        self._cache = None
        self._is_synced = False

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Aggregate and evaluate batch input directly.

        Updates the global metric state and returns the metric value for the current batch.
        """
        if self._is_synced:
            raise RuntimeError(
                "The Metric shouldn't be synced when performing ``forward``. "
                "HINT: Did you forget to call ``unsync``?"
            )
        if (
            self.full_state_update
            or self.full_state_update is None
            or self._dist_sync_on_step
        ):
            self._forward_cache = self._forward_full_state_update(
                *args, **kwargs
            )
        else:
            self._forward_cache = self._forward_reduce_state_update(
                *args, **kwargs
            )
        return self._forward_cache

    def _forward_full_state_update(self, *args: Any, **kwargs: Any) -> Any:
        """Forward using two calls to update (safe, works for all metrics)."""
        self.update(*args, **kwargs)
        _update_count = self._update_count
        self._to_sync = self._dist_sync_on_step
        self._should_unsync = False
        _temp_compute_on_cpu = self._compute_on_cpu
        self._compute_on_cpu = False
        cache = self._copy_state_dict()
        self._enable_grad = True
        self.reset()
        self.update(*args, **kwargs)
        batch_val = self.compute()
        for attr, val in cache.items():
            setattr(self, attr, val)
        self._update_count = _update_count
        self._is_synced = False
        self._should_unsync = True
        self._to_sync = self._sync_on_compute
        self._computed = None
        self._enable_grad = False
        self._compute_on_cpu = _temp_compute_on_cpu
        if self._compute_on_cpu:
            self._move_list_states_to_cpu()
        return batch_val

    def _forward_reduce_state_update(self, *args: Any, **kwargs: Any) -> Any:
        """Forward using single call to update (fast, for reducible metrics)."""
        global_state = self._copy_state_dict()
        _update_count = self._update_count
        self.reset()
        self._to_sync = self._dist_sync_on_step
        self._should_unsync = False
        _temp_compute_on_cpu = self._compute_on_cpu
        self._compute_on_cpu = False
        self._enable_grad = True
        self.update(*args, **kwargs)
        batch_val = self.compute()
        self._update_count = _update_count + 1
        with paddle.no_grad():
            self._reduce_states(global_state)
        self._is_synced = False
        self._should_unsync = True
        self._to_sync = self._sync_on_compute
        self._computed = None
        self._enable_grad = False
        self._compute_on_cpu = _temp_compute_on_cpu
        if self._compute_on_cpu:
            self._move_list_states_to_cpu()
        return batch_val

    def merge_state(self, incoming_state: dict[str, Any] | Metric) -> None:
        """Merge incoming metric state to the current state."""
        if not isinstance(incoming_state, (dict, Metric)):
            raise ValueError(
                f"Expected incoming state to be a dict or Metric instance, got {type(incoming_state)}"
            )
        if (
            self.full_state_update
            or self.full_state_update is None
            or self._dist_sync_on_step
        ):
            raise RuntimeError(
                "``merge_state`` is not supported for metrics with ``full_state_update=True`` or "
                "``dist_sync_on_step=True``. Please overwrite the merge_state method in the metric class."
            )
        if isinstance(incoming_state, Metric):
            if not isinstance(incoming_state, self.__class__):
                raise ValueError(
                    f"Expected incoming state to be an instance of {self.__class__.__name__} but got "
                    f"{type(incoming_state)}"
                )
            incoming_state = incoming_state.metric_state
        self._reduce_states(incoming_state)

    def _reduce_states(self, incoming_state: dict[str, Any]) -> None:
        for attr in self._defaults:
            local_state = getattr(self, attr)
            if attr not in incoming_state:
                raise ValueError(
                    f"Expected state variable {attr} to be present in incoming state {incoming_state}"
                )
            global_state = incoming_state[attr]
            reduce_fn = self._reductions[attr]
            if reduce_fn == _dim_zero_sum:
                reduced = global_state + local_state
            elif reduce_fn == _dim_zero_mean:
                reduced = (
                    (self._update_count - 1) * global_state + local_state
                ).astype(paddle.get_default_dtype()) / self._update_count
            elif reduce_fn == _dim_zero_max:
                reduced = paddle.maximum(global_state, local_state)
            elif reduce_fn == _dim_zero_min:
                reduced = paddle.minimum(global_state, local_state)
            elif reduce_fn == _dim_zero_cat:
                if isinstance(global_state, paddle.Tensor):
                    reduced = paddle.concat([global_state, local_state])
                else:
                    reduced = global_state + local_state
            elif reduce_fn is None and isinstance(global_state, paddle.Tensor):
                reduced = paddle.stack([global_state, local_state])
            elif reduce_fn is None and isinstance(global_state, list):
                reduced = _flatten([global_state, local_state])
            elif reduce_fn and callable(reduce_fn):
                reduced = reduce_fn(paddle.stack([global_state, local_state]))
            else:
                raise TypeError(f"Unsupported reduce_fn: {reduce_fn}")
            setattr(self, attr, reduced)

    # ============ Distributed Synchronization ============

    def sync(
        self,
        dist_sync_fn: Callable | None = None,
        process_group: Any | None = None,
        should_sync: bool = True,
        distributed_available: Callable | None = None,
    ) -> None:
        if self._is_synced and should_sync:
            raise RuntimeError("The Metric has already been synced.")
        dist_available_fn = (
            distributed_available or self._distributed_available_fn
        )
        if not should_sync or not dist_available_fn():
            return
        sync_fn = dist_sync_fn or self._dist_sync_fn or _gather_all_tensors
        group = process_group or self._process_group
        self._cache = self._copy_state_dict()

        for name in self._defaults:
            state = getattr(self, name)
            if state is None:
                continue
            if isinstance(state, paddle.Tensor):
                gathered = sync_fn(state, group=group)
                reduce_fn = self._reductions[name]
                if reduce_fn is not None:
                    reduced = reduce_fn(paddle.stack(gathered))
                else:
                    reduced = paddle.stack(gathered)
                setattr(self, name, reduced)
            elif isinstance(state, list):
                all_elements = []
                for elem in state:
                    if isinstance(elem, paddle.Tensor):
                        elem_gathered = sync_fn(elem, group=group)
                        all_elements.extend(elem_gathered)
                    else:
                        all_elements.append(elem)
                reduce_fn = self._reductions[name]
                if reduce_fn is not None:
                    if all_elements and isinstance(
                        all_elements[0], paddle.Tensor
                    ):
                        reduced = reduce_fn(paddle.stack(all_elements))
                    else:
                        reduced = all_elements
                else:
                    reduced = all_elements
                setattr(self, name, reduced)
        self._is_synced = True

    def unsync(self, should_unsync: bool = True) -> None:
        if not should_unsync:
            return
        if not self._is_synced:
            raise RuntimeError("The Metric has already been un-synced.")
        if self._cache is None:
            raise RuntimeError(
                "The internal cache should exist to unsync the Metric."
            )
        for attr, val in self._cache.items():
            setattr(self, attr, val)
        self._is_synced = False
        self._cache = None

    @contextmanager
    def sync_context(
        self,
        dist_sync_fn: Callable | None = None,
        process_group: Any | None = None,
        should_sync: bool = True,
        should_unsync: bool = True,
        distributed_available: Callable | None = None,
    ) -> Generator:
        self.sync(
            dist_sync_fn=dist_sync_fn,
            process_group=process_group,
            should_sync=should_sync,
            distributed_available=distributed_available,
        )
        yield
        self.unsync(should_unsync=self._is_synced and should_unsync)

    # ============ Serialization ============

    def state_dict(
        self,
        destination: dict[str, Any] | None = None,
        prefix: str = "",
        keep_vars: bool = False,
    ) -> dict[str, Any]:
        destination = super().state_dict(
            destination=destination, prefix=prefix, keep_vars=keep_vars
        )
        for key in self._defaults:
            if not self._persistent.get(key, False):
                continue
            current_val = getattr(self, key)
            if key in self._buffers:
                continue
            if not keep_vars:
                if isinstance(current_val, paddle.Tensor):
                    current_val = current_val.detach()
                elif isinstance(current_val, list):
                    current_val = [
                        (v.detach() if isinstance(v, paddle.Tensor) else v)
                        for v in current_val
                    ]
            destination[prefix + key] = deepcopy(current_val)
        return destination

    def set_state_dict(
        self,
        state_dict: dict[str, Any],
        use_structured_name: bool = True,
    ) -> None:
        # Load buffer states
        for name in self._defaults:
            key = name if use_structured_name else name
            if key in state_dict:
                if (
                    isinstance(state_dict[key], paddle.Tensor)
                    and name in self._buffers
                ):
                    getattr(self, name).set_value(state_dict[key])
                else:
                    setattr(self, name, state_dict[key])

    def _load_from_state_dict(
        self,
        state_dict: dict,
        prefix: str,
        local_metadata: dict,
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        for key in self._defaults:
            name = prefix + key
            if name in state_dict:
                setattr(self, key, state_dict.pop(name))

    def _copy_state_dict(self) -> dict[str, paddle.Tensor | list[Any]]:
        cache: dict[str, paddle.Tensor | list[Any]] = {}
        for attr in self._defaults:
            current_value = getattr(self, attr)
            if isinstance(current_value, paddle.Tensor):
                cache[attr] = current_value.detach().clone()
            else:
                cache[attr] = [
                    (
                        _.detach().clone()
                        if isinstance(_, paddle.Tensor)
                        else deepcopy(_)
                    )
                    for _ in current_value
                ]
        return cache

    def persistent(self, mode: bool = False) -> None:
        """Change if metric states should be saved to state_dict."""
        for key in self._persistent:
            self._persistent[key] = mode
            if key in self._buffers:
                if mode:
                    self._non_persistable_buffer_names_set.discard(key)
                else:
                    self._non_persistable_buffer_names_set.add(key)

    def clone(self) -> Metric:
        return deepcopy(self)

    # ============ Pickle Support ============

    def __getstate__(self) -> dict[str, Any]:
        return {
            k: v
            for k, v in self.__dict__.items()
            if k not in ["update", "compute", "_update_signature", "_device"]
        }

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._device = paddle.CPUPlace()
        self._update_signature = inspect.signature(self.update)
        self.update: Callable = self._wrap_update(self.update)
        self.compute: Callable = self._wrap_compute(self.compute)

    # ============ Hash ============

    def __hash__(self) -> int:
        hash_vals = [self.__class__.__name__, id(self)]
        for key in self._defaults:
            val = getattr(self, key)
            if hasattr(val, "__iter__") and not isinstance(val, paddle.Tensor):
                hash_vals.extend(val)
            else:
                hash_vals.append(val)
        return hash(tuple(hash_vals))

    # ============ Attribute Protection ============

    def __setattr__(self, name: str, value: Any) -> None:
        _protected = {
            "higher_is_better",
            "is_differentiable",
            "full_state_update",
            "plot_lower_bound",
            "plot_upper_bound",
            "plot_legend_name",
        }
        if name in _protected:
            for cls in type(self).__mro__:
                if name in cls.__dict__:
                    raise RuntimeError(f"Can't set const `{name}`.")
        super().__setattr__(name, value)

    # ============ Dtype Management ============

    def type(self, dst_type: str | paddle.dtype) -> Metric:
        return self

    def float(self) -> Metric:
        return self

    def double(self) -> Metric:
        return self

    def half(self) -> Metric:
        return self

    def set_dtype(self, dst_type: str | paddle.dtype) -> Metric:
        """Transfer all metric state to specific dtype."""
        self._dtype_convert = True
        out = super().astype(dst_type)
        out._dtype_convert = False
        return out

    def _apply(
        self, fn: Callable, exclude_state: Sequence[str] = ""
    ) -> nn.Layer:
        """Overwrite _apply to also move metric states to the correct device."""
        this = super()._apply(fn)
        fs = str(fn)
        cond = any(
            f in fs
            for f in [
                "Layer.type",
                "Layer.half",
                "Layer.float",
                "Layer.double",
                "Layer.bfloat16",
            ]
        )
        if not self._dtype_convert and cond:
            return this
        for key, value in this._defaults.items():
            if key in exclude_state:
                continue
            if isinstance(value, paddle.Tensor):
                this._defaults[key] = fn(value)
            elif isinstance(value, Sequence):
                this._defaults[key] = [fn(v) for v in value]
            current_val = getattr(this, key)
            if isinstance(current_val, paddle.Tensor):
                setattr(this, key, fn(current_val))
            elif isinstance(current_val, Sequence):
                setattr(this, key, [fn(cur_v) for cur_v in current_val])
        if this._computed is not None:
            this._computed = _apply_to_collection(
                this._computed, paddle.Tensor, fn
            )
        if this._forward_cache is not None:
            this._forward_cache = _apply_to_collection(
                this._forward_cache, paddle.Tensor, fn
            )
        return this

    # ============ Method Wrappers ============

    def _wrap_update(self, update: Callable) -> Callable:
        @functools.wraps(update)
        def wrapper(*args: Any, **kwargs: Any) -> None:
            self._computed = None
            self._update_count += 1
            with paddle.set_grad_enabled(self._enable_grad):
                update(*args, **kwargs)
            if self._compute_on_cpu:
                self._move_list_states_to_cpu()

        return wrapper

    def _wrap_compute(self, compute: Callable) -> Callable:
        @functools.wraps(compute)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if not self.update_called:
                warnings.warn(
                    f"The ``compute`` method of metric {self.__class__.__name__} was called before the "
                    f"``update`` method which may lead to errors.",
                    UserWarning,
                )
            if self._computed is not None and self._compute_with_cache:
                return self._computed
            with self.sync_context(
                dist_sync_fn=self._dist_sync_fn,
                process_group=self._process_group,
                should_sync=self._to_sync,
                should_unsync=self._should_unsync,
                distributed_available=self._distributed_available_fn,
            ):
                value = _squeeze_if_scalar(compute(*args, **kwargs))
                value = _apply_to_collection(
                    value, paddle.Tensor, lambda x: x.clone()
                )
            if self._compute_with_cache:
                self._computed = value
            return value

        return wrapper

    def _move_list_states_to_cpu(self) -> None:
        for key in self._defaults:
            current_val = getattr(self, key)
            if isinstance(current_val, Sequence):
                setattr(
                    self,
                    key,
                    [
                        cur_v.to("cpu")
                        if isinstance(cur_v, paddle.Tensor)
                        else cur_v
                        for cur_v in current_val
                    ],
                )

    def _filter_kwargs(self, **kwargs: Any) -> dict[str, Any]:
        _params = (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        )
        _sign_params = self._update_signature.parameters
        filtered_kwargs = {
            k: v
            for k, v in kwargs.items()
            if k in _sign_params and _sign_params[k].kind not in _params
        }
        exists_var_keyword = any(
            v.kind == inspect.Parameter.VAR_KEYWORD
            for v in _sign_params.values()
        )
        if not filtered_kwargs and not exists_var_keyword:
            return {}
        if exists_var_keyword:
            return kwargs
        return filtered_kwargs

    # ============ Plotting ============

    def plot(
        self,
        val: paddle.Tensor
        | Sequence[paddle.Tensor]
        | dict[str, paddle.Tensor]
        | None = None,
        ax: Any | None = None,
    ) -> Any:
        """Plot the metric value. Override in subclasses for custom plotting."""
        raise NotImplementedError

    def _plot(
        self,
        val: paddle.Tensor
        | Sequence[paddle.Tensor]
        | dict[str, paddle.Tensor]
        | Sequence[dict[str, paddle.Tensor]]
        | None = None,
        ax: Any | None = None,
    ) -> Any:
        from paddle.metric.utils.plot import plot_single_or_multi_val

        val = val if val is not None else self.compute()
        fig, ax = plot_single_or_multi_val(
            val,
            ax=ax,
            higher_is_better=self.higher_is_better,
            name=self.__class__.__name__,
            lower_bound=self.plot_lower_bound,
            upper_bound=self.plot_upper_bound,
            legend_name=self.plot_legend_name,
        )
        return fig, ax

    # ============ Composition Operators ============

    def __add__(
        self, other: Metric | builtins.float | paddle.Tensor
    ) -> CompositionalMetric:
        return CompositionalMetric(paddle.add, self, other)

    def __and__(
        self, other: Metric | builtins.float | paddle.Tensor
    ) -> CompositionalMetric:
        return CompositionalMetric(paddle.bitwise_and, self, other)

    def __eq__(
        self, other: Metric | builtins.float | paddle.Tensor
    ) -> CompositionalMetric:
        return CompositionalMetric(paddle.equal, self, other)

    def __floordiv__(
        self, other: Metric | builtins.float | paddle.Tensor
    ) -> CompositionalMetric:
        return CompositionalMetric(paddle.floor_divide, self, other)

    def __ge__(
        self, other: Metric | builtins.float | paddle.Tensor
    ) -> CompositionalMetric:
        return CompositionalMetric(paddle.greater_equal, self, other)

    def __gt__(
        self, other: Metric | builtins.float | paddle.Tensor
    ) -> CompositionalMetric:
        return CompositionalMetric(paddle.greater_than, self, other)

    def __le__(
        self, other: Metric | builtins.float | paddle.Tensor
    ) -> CompositionalMetric:
        return CompositionalMetric(paddle.less_equal, self, other)

    def __lt__(
        self, other: Metric | builtins.float | paddle.Tensor
    ) -> CompositionalMetric:
        return CompositionalMetric(paddle.less_than, self, other)

    def __matmul__(
        self, other: Metric | builtins.float | paddle.Tensor
    ) -> CompositionalMetric:
        return CompositionalMetric(paddle.matmul, self, other)

    def __mod__(
        self, other: Metric | builtins.float | paddle.Tensor
    ) -> CompositionalMetric:
        return CompositionalMetric(paddle.mod, self, other)

    def __mul__(
        self, other: Metric | builtins.float | paddle.Tensor
    ) -> CompositionalMetric:
        return CompositionalMetric(paddle.multiply, self, other)

    def __ne__(
        self, other: Metric | builtins.float | paddle.Tensor
    ) -> CompositionalMetric:
        return CompositionalMetric(paddle.not_equal, self, other)

    def __or__(
        self, other: Metric | builtins.float | paddle.Tensor
    ) -> CompositionalMetric:
        return CompositionalMetric(paddle.bitwise_or, self, other)

    def __pow__(
        self, other: Metric | builtins.float | paddle.Tensor
    ) -> CompositionalMetric:
        return CompositionalMetric(paddle.pow, self, other)

    def __sub__(
        self, other: Metric | builtins.float | paddle.Tensor
    ) -> CompositionalMetric:
        return CompositionalMetric(paddle.subtract, self, other)

    def __truediv__(
        self, other: Metric | builtins.float | paddle.Tensor
    ) -> CompositionalMetric:
        return CompositionalMetric(paddle.divide, self, other)

    def __xor__(
        self, other: Metric | builtins.float | paddle.Tensor
    ) -> CompositionalMetric:
        return CompositionalMetric(paddle.bitwise_xor, self, other)

    # Reverse operators
    def __radd__(
        self, other: Metric | builtins.float | paddle.Tensor
    ) -> CompositionalMetric:
        return CompositionalMetric(paddle.add, other, self)

    def __rand__(
        self, other: Metric | builtins.float | paddle.Tensor
    ) -> CompositionalMetric:
        return CompositionalMetric(paddle.bitwise_and, other, self)

    def __rfloordiv__(
        self, other: Metric | builtins.float | paddle.Tensor
    ) -> CompositionalMetric:
        return CompositionalMetric(paddle.floor_divide, other, self)

    def __rmatmul__(
        self, other: Metric | builtins.float | paddle.Tensor
    ) -> CompositionalMetric:
        return CompositionalMetric(paddle.matmul, other, self)

    def __rmod__(
        self, other: Metric | builtins.float | paddle.Tensor
    ) -> CompositionalMetric:
        return CompositionalMetric(paddle.mod, other, self)

    def __rmul__(
        self, other: Metric | builtins.float | paddle.Tensor
    ) -> CompositionalMetric:
        return CompositionalMetric(paddle.multiply, other, self)

    def __ror__(
        self, other: Metric | builtins.float | paddle.Tensor
    ) -> CompositionalMetric:
        return CompositionalMetric(paddle.bitwise_or, other, self)

    def __rpow__(
        self, other: Metric | builtins.float | paddle.Tensor
    ) -> CompositionalMetric:
        return CompositionalMetric(paddle.pow, other, self)

    def __rsub__(
        self, other: Metric | builtins.float | paddle.Tensor
    ) -> CompositionalMetric:
        return CompositionalMetric(paddle.subtract, other, self)

    def __rtruediv__(
        self, other: Metric | builtins.float | paddle.Tensor
    ) -> CompositionalMetric:
        return CompositionalMetric(paddle.divide, other, self)

    def __rxor__(
        self, other: Metric | builtins.float | paddle.Tensor
    ) -> CompositionalMetric:
        return CompositionalMetric(paddle.bitwise_xor, other, self)

    # Unary operators
    def __abs__(self) -> CompositionalMetric:
        return CompositionalMetric(paddle.abs, self, None)

    def __inv__(self) -> CompositionalMetric:
        return CompositionalMetric(paddle.bitwise_not, self, None)

    def __invert__(self) -> CompositionalMetric:
        return self.__inv__()

    def __neg__(self) -> CompositionalMetric:
        return CompositionalMetric(_neg, self, None)

    def __pos__(self) -> CompositionalMetric:
        return CompositionalMetric(paddle.abs, self, None)

    def __getitem__(self, idx: int) -> CompositionalMetric:
        return CompositionalMetric(lambda x: x[idx], self, None)

    def __getnewargs__(self) -> tuple:
        return tuple(Metric.__str__(self))

    __iter__ = None


# ============ CompositionalMetric ============


class CompositionalMetric(Metric):
    """Composition of two metrics with a specific operator."""

    def __init__(
        self,
        operator: Callable,
        metric_a: Metric | float | paddle.Tensor,
        metric_b: Metric | float | paddle.Tensor | None,
    ) -> None:
        super().__init__()
        self.op = operator
        if isinstance(metric_a, paddle.Tensor):
            self.register_buffer("metric_a", metric_a, persistent=False)
        else:
            self.metric_a = metric_a
        if isinstance(metric_b, paddle.Tensor):
            self.register_buffer("metric_b", metric_b, persistent=False)
        else:
            self.metric_b = metric_b

    def _sync_dist(
        self,
        dist_sync_fn: Callable | None = None,
        process_group: Any | None = None,
    ) -> None:
        pass  # Syncing done in child metrics

    def update(self, *args: Any, **kwargs: Any) -> None:
        if isinstance(self.metric_a, Metric):
            self.metric_a.update(
                *args, **self.metric_a._filter_kwargs(**kwargs)
            )
        if isinstance(self.metric_b, Metric):
            self.metric_b.update(
                *args, **self.metric_b._filter_kwargs(**kwargs)
            )

    def compute(self) -> Any:
        val_a = (
            self.metric_a.compute()
            if isinstance(self.metric_a, Metric)
            else self.metric_a
        )
        val_b = (
            self.metric_b.compute()
            if isinstance(self.metric_b, Metric)
            else self.metric_b
        )
        # Ensure operands are tensors (Paddle ops don't accept raw floats)
        if val_a is not None and not isinstance(val_a, paddle.Tensor):
            val_a = paddle.to_tensor(val_a)
        if val_b is not None and not isinstance(val_b, paddle.Tensor):
            val_b = paddle.to_tensor(val_b)
        if val_b is None:
            return self.op(val_a)
        return self.op(val_a, val_b)

    @paddle.jit.not_to_static
    def forward(self, *args: Any, **kwargs: Any) -> Any:
        val_a = (
            self.metric_a(*args, **self.metric_a._filter_kwargs(**kwargs))
            if isinstance(self.metric_a, Metric)
            else self.metric_a
        )
        val_b = (
            self.metric_b(*args, **self.metric_b._filter_kwargs(**kwargs))
            if isinstance(self.metric_b, Metric)
            else self.metric_b
        )
        if val_a is None:
            self._forward_cache = None
            return self._forward_cache
        if val_b is None:
            if isinstance(self.metric_b, Metric):
                self._forward_cache = None
                return self._forward_cache
            if not isinstance(val_a, paddle.Tensor):
                val_a = paddle.to_tensor(val_a)
            self._forward_cache = self.op(val_a)
            return self._forward_cache
        if not isinstance(val_a, paddle.Tensor):
            val_a = paddle.to_tensor(val_a)
        if not isinstance(val_b, paddle.Tensor):
            val_b = paddle.to_tensor(val_b)
        self._forward_cache = self.op(val_a, val_b)
        return self._forward_cache

    def reset(self) -> None:
        if isinstance(self.metric_a, Metric):
            self.metric_a.reset()
        if isinstance(self.metric_b, Metric):
            self.metric_b.reset()

    def persistent(self, mode: bool = False) -> None:
        if isinstance(self.metric_a, Metric):
            self.metric_a.persistent(mode=mode)
        if isinstance(self.metric_b, Metric):
            self.metric_b.persistent(mode=mode)

    def __repr__(self) -> str:
        _op_metrics = f"""(
  {self.op.__name__}(
    {self.metric_a!r},
    {self.metric_b!r}
  )
)"""
        return self.__class__.__name__ + _op_metrics

    def _wrap_compute(self, compute: Callable) -> Callable:
        return compute
