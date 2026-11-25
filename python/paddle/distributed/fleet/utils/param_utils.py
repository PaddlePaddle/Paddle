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
"""Copy parameter attributes, including ones attached from outside.

The sharding optimizers create a placeholder slice parameter for each
parameter (``_create_slice_param``) and used to copy a fixed list of attribute
names onto it (``DEFAULT_COPY_ATTRS`` below). Code outside the optimizer often
attaches more attributes to a parameter with ``setattr`` (``color``,
``split_axis``, ...). :func:`copy_param_attrs` copies both:

1. every name in ``DEFAULT_COPY_ATTRS`` that the source has;
2. every attribute attached from outside.

Other attributes a parameter has on its own are not copied:

* Python-level attributes that ``EagerParamBase.__init__`` creates and that are
  not in ``DEFAULT_COPY_ATTRS`` (``regularizer``, ``_init_func``,
  ``_init_op_creator``), even if outside code later reassigned them;
* C-level tensor properties (``name``, ``grad``, ``stop_gradient``,
  ``trainable``, ...), which live on the pybind type, not in ``__dict__``.

Some attributes refer to the source parameter's own gradient, storage or
runtime state; sharing them would make the destination read or write the
source's data. They are never copied:

* names in ``DEFAULT_EXCLUDE_ATTRS``;
* values tied to the source object: methods bound to it (``__self__ is src``,
  e.g. ``get_main_grad``) and weak references (e.g. ``comm_buffer_ref``).
"""

from __future__ import annotations

import functools
import weakref
from typing import TYPE_CHECKING, Any

import paddle
from paddle.base import unique_name
from paddle.base.framework import EagerParamBase

if TYPE_CHECKING:
    from collections.abc import Iterable

__all__ = ["copy_param_attrs"]

# The names the sharding optimizers copied before external attributes were
# supported. Each is copied whenever ``hasattr(src, name)`` is true, so
# class-level or property-backed values on parameter subclasses are copied too.
DEFAULT_COPY_ATTRS = (
    "is_distributed",
    "optimize_attr",
    "do_model_average",
    "need_clip",
    "no_sync",
    "is_firstly_shared",
)

# Attributes Paddle's distributed code attaches that belong to the source
# parameter only.
DEFAULT_EXCLUDE_ATTRS = frozenset(
    {
        # Gradients and things derived from them.
        "grad",
        "main_grad",
        "_main_grad_addr",  # main_grad.data_ptr() of the source
        "get_main_grad",  # method bound to the source
        "_fused_kernel_fp32_grad",
        "master_weight",  # fp32 copy of the source's full value
        # References to the buffers that hold the source's data.
        "comm_buffer_ref",
        "fw_storage",
        "bw_storage",
        "_fusion_buffer",
        "_param_offsets",
        # Per-parameter runtime state.
        "_fsdp_grad_hooked",
        "use_count",
        "status",
        "unslice",
        # Markers that route a parameter to a sharding path. The slice param
        # is already sharded, so the inner optimizer must not see them.
        "_need_shard",  # mix_precision_utils.py -> FullyShardOptimizer
        "_need_shard_auto",  # optimizer.py -> fully_shard.shard_accumulators
    }
)

_WEAKREF_PROXY_TYPES = (weakref.ProxyType, weakref.CallableProxyType)


@functools.lru_cache(maxsize=1)
def _builtin_attr_names() -> frozenset[str]:
    """Names ``EagerParamBase.__init__`` puts into an instance's ``__dict__``.

    ``EagerParamBase.__init__`` has two paths, ``__init_by_shape__`` and
    ``__init_by_tensor__`` (which also sets ``_is_param``), so one probe is
    built through each and the union of their names is taken. This keeps the
    list in sync with the constructor. The probes are built under
    ``unique_name.guard()`` because the constructor calls
    ``unique_name.generate`` even when a name is passed; without the guard,
    names of parameters created later would shift.
    """
    with unique_name.guard():
        by_shape = EagerParamBase(shape=[1], dtype=paddle.float32)
        by_tensor = EagerParamBase(paddle.zeros([1], dtype=paddle.float32))
    return frozenset(vars(by_shape)) | frozenset(vars(by_tensor))


def _is_tied_to(value: Any, src: Any) -> bool:
    """Whether ``value`` is a weak reference or a method bound to ``src``."""
    # Check proxies by exact type before anything else: attribute access on a
    # proxy (including the __class__ lookup isinstance does) is forwarded to
    # its referent and raises ReferenceError once the referent is dead.
    if type(value) in _WEAKREF_PROXY_TYPES or isinstance(value, weakref.ref):
        return True
    return getattr(value, "__self__", None) is src


def get_external_attr_names(
    param: Any, exclude: Iterable[str] = DEFAULT_EXCLUDE_ATTRS
) -> list[str]:
    """Return the sorted names of attributes attached to ``param`` from outside.

    An attribute counts as attached from outside when it is in
    ``vars(param)`` but a freshly constructed ``EagerParamBase`` does not have
    it. Names in ``exclude`` are dropped.
    """
    try:
        names = vars(param).keys()
    except TypeError:  # no __dict__, so nothing can have been attached
        return []
    builtin = _builtin_attr_names()
    exclude = frozenset(exclude)
    return sorted(n for n in names if n not in builtin and n not in exclude)


def copy_param_attrs(
    src: Any,
    dst: Any,
    copy_attrs: Iterable[str] = DEFAULT_COPY_ATTRS,
    exclude: Iterable[str] = DEFAULT_EXCLUDE_ATTRS,
) -> list[str]:
    """Shallow-copy parameter attributes from ``src`` to ``dst``.

    Copies, in this order:

    1. each name in ``copy_attrs`` that ``src`` has;
    2. each attribute attached to ``src`` from outside
       (see :func:`get_external_attr_names`).

    Each value is shared by reference
    (``setattr(dst, name, getattr(src, name))``). Existing values on ``dst``
    with the same name are overwritten.

    Never copied, even when listed in ``copy_attrs``: names in ``exclude``,
    and values that are weak references or methods bound to ``src``.

    Args:
        src: Source parameter.
        dst: Destination parameter.
        copy_attrs: Names to copy whenever ``src`` has them.
        exclude: Names that must never be copied. Extend the default with
            ``DEFAULT_EXCLUDE_ATTRS | {"other_name"}``.

    Returns:
        The list of copied attribute names, in copy order.
    """
    exclude = frozenset(exclude)
    candidates = [n for n in copy_attrs if n not in exclude and hasattr(src, n)]
    candidates += [
        n for n in get_external_attr_names(src, exclude) if n not in candidates
    ]
    copied = []
    for name in candidates:
        value = getattr(src, name)
        if _is_tied_to(value, src):
            continue
        setattr(dst, name, value)
        copied.append(name)
    return copied
