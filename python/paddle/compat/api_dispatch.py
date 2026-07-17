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

"""``enable_compat(level=2)`` API dispatch: route ``paddle.*`` (and
``paddle.Tensor`` methods) to the torch-aligned ``paddle.compat.*`` for
external callers while paddle-internal callers keep native semantics.

The enable/disable/level lifecycle lives in ``paddle.compat.proxy``; this
module only installs/removes the dispatchers and holds the dispatch state.
"""

from __future__ import annotations

import contextvars
import importlib
import inspect
import pkgutil
import sys
from functools import wraps
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import types
    from collections.abc import Generator


class _MissingType:
    """Sentinel marking a paddle attribute that did not exist before aliasing."""


_MISSING = _MissingType()

# (live module, attr name) -> original value (or ``_MISSING``) for every
# ``paddle.*`` symbol aliased to its ``paddle.compat.*`` counterpart, so the
# paddle namespace can be restored exactly on ``disable_compat()``.
_PADDLE_NAMESPACE_SAVED: dict[tuple[types.ModuleType, str], Any] = {}

# Dispatch state: _COMPAT_ENABLED is the process-wide on/off; _COMPAT_SUSPENDED is
# a per-thread/async-task override so a running compat API uses native internals.
_COMPAT_ENABLED = False
_COMPAT_SUSPENDED: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "paddle_compat_suspended", default=False
)


def is_compat_api_enabled() -> bool:
    """Whether ``paddle.*`` currently dispatches to the torch-aligned compat APIs
    in this thread / async task."""
    return _COMPAT_ENABLED and not _COMPAT_SUSPENDED.get()


def _is_paddle_namespace_aliased() -> bool:
    """Whether level-2 aliases are currently installed process-wide."""
    return bool(_PADDLE_NAMESPACE_SAVED)


class compat_api_guard:
    """Suspend/restore compat dispatch for the current thread / async task.
    The level-2 dispatcher suspends aliases while a compat implementation runs,
    so its internal ``paddle.*`` calls hit native. Context manager and decorator;
    the decorator preserves ``__signature__``."""

    def __init__(self, enable: bool = True) -> None:
        self._enable = enable
        self._token = None

    def __enter__(self) -> None:
        self._token = _COMPAT_SUSPENDED.set(not self._enable)

    def __exit__(self, *exc: object) -> bool:
        _COMPAT_SUSPENDED.reset(self._token)
        self._token = None
        return False

    def __call__(self, func: Any) -> Any:
        enable = self._enable

        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # A fresh guard per call keeps re-entrant / concurrent use correct.
            with compat_api_guard(enable):
                return func(*args, **kwargs)

        # Every decorated callable is a pure-Python compat API, so its
        # signature is always introspectable.
        wrapper.__signature__ = inspect.signature(func)
        return wrapper


def _caller_is_paddle_internal() -> bool:
    """True when the caller of the dispatched ``paddle.*`` API is a ``paddle``
    module. Paddle's own internals call these APIs with native kwargs/defaults, so
    they (and the compat impls) get native; only external callers see compat."""
    # _getframe(1) = our caller (the dispatch site); its f_back = who called paddle.X.
    frame = sys._getframe(1).f_back
    if frame is None:
        return False
    name = frame.f_globals.get("__name__") or ""
    return name == "paddle" or name.startswith("paddle.")


def dispatch_compat_api(compat_fn: Any) -> Any:
    """Wrap a native ``paddle`` callable to route external callers to
    ``compat_fn`` while compat is enabled; paddle-internal callers and the
    disabled state get the native callable. Installed only under
    ``enable_compat(level=2)``; ``disable_compat`` restores the originals,
    so the default hot path is untouched."""

    def decorator(native_fn: Any) -> Any:
        getframe = sys._getframe
        suspended = _COMPAT_SUSPENDED.get

        @wraps(native_fn)
        def dispatcher(*args: Any, **kwargs: Any) -> Any:
            if _COMPAT_ENABLED and not suspended():
                caller = getframe().f_back
                name = (
                    "" if caller is None else caller.f_globals.get("__name__")
                ) or ""
                if name != "paddle" and not name.startswith("paddle."):
                    with compat_api_guard(enable=False):
                        return compat_fn(*args, **kwargs)
            return native_fn(*args, **kwargs)

        dispatcher.__compat_fn__ = compat_fn
        dispatcher.__native_fn__ = native_fn
        dispatcher.__signature__ = inspect.signature(compat_fn)
        return dispatcher

    return decorator


def _iter_compat_modules() -> Generator[types.ModuleType, None, None]:
    """Yield every ``paddle.compat`` (sub)module, root package included.

    ``pkgutil.walk_packages`` skips the starting package, so the root
    ``paddle.compat`` (holding the top-level functions) is yielded explicitly.
    """
    import paddle.compat

    yield paddle.compat
    for module_info in pkgutil.walk_packages(
        paddle.compat.__path__,
        paddle.compat.__name__ + ".",
    ):
        yield importlib.import_module(module_info.name)


def _make_caller_aware_class_proxy(native_cls: type, compat_cls: type) -> type:
    """Caller-aware ``paddle.X`` class: instantiates ``compat_cls`` for external
    callers, ``native_cls`` for paddle-internal ones; ``isinstance``/``issubclass``
    accept either. Subclasses ``compat_cls`` so a user subclass derived under
    level=2 keeps the torch-style (compat) constructor/methods."""

    class _CompatAwareMeta(type(compat_cls)):
        def __call__(cls, *args: Any, **kwargs: Any) -> Any:
            if cls is proxy:
                if is_compat_api_enabled() and not _caller_is_paddle_internal():
                    return compat_cls(*args, **kwargs)
                return native_cls(*args, **kwargs)
            return super().__call__(*args, **kwargs)

        def __instancecheck__(cls, instance: Any) -> bool:
            return isinstance(instance, (native_cls, compat_cls))

        def __subclasscheck__(cls, subclass: Any) -> bool:
            return issubclass(subclass, native_cls) or issubclass(
                subclass, compat_cls
            )

    proxy = _CompatAwareMeta(
        native_cls.__name__,
        (compat_cls,),
        {
            "__module__": native_cls.__module__,
            "__compat_cls__": compat_cls,
            "__native_cls__": native_cls,
        },
    )
    proxy.__signature__ = inspect.signature(compat_cls)
    return proxy


def _patch_tensor_methods() -> None:
    """Route ``paddle.Tensor.<m>`` to the compat function for the root compat APIs
    that torch also exposes as Tensor methods (max/min/sort/split/unique/...), so
    ``x.max(dim=1)`` works torch-style for external callers (native for internal).
    The dispatcher is patched directly like any paddle Tensor method: the
    descriptor protocol forwards the tensor as the first positional argument,
    which is exactly the compat function's ``input`` parameter.
    """
    import paddle
    import paddle.compat as compat_root

    for attr_name in getattr(compat_root, "__all__", ()):
        if attr_name.startswith("_"):
            continue
        native_method = getattr(paddle.Tensor, attr_name, _MISSING)
        if native_method is _MISSING or not callable(native_method):
            continue
        compat_fn = getattr(compat_root, attr_name)
        _PADDLE_NAMESPACE_SAVED[(paddle.Tensor, attr_name)] = native_method
        setattr(
            paddle.Tensor,
            attr_name,
            dispatch_compat_api(compat_fn)(native_method),
        )


def _apply_paddle_namespace_aliases() -> None:
    """Install caller-aware dispatchers/proxies for every public ``paddle.compat.*``
    symbol onto ``paddle.*`` (functions -> dispatcher, classes -> proxy, torch-only
    symbols -> direct alias) plus the Tensor methods. Idempotent; ``enable_compat``
    calls this only for ``level=2``."""
    if _PADDLE_NAMESPACE_SAVED:
        return

    global _COMPAT_ENABLED
    COMPAT_PREFIX = "paddle.compat"
    PADDLE_PREFIX = "paddle"
    PUBLIC_ATTR_DECLARATION = "__all__"

    for compat_module in _iter_compat_modules():
        if not hasattr(compat_module, PUBLIC_ATTR_DECLARATION):
            continue
        # paddle.compat -> paddle ; paddle.compat.nn.functional -> paddle.nn.functional
        target_name = compat_module.__name__.replace(
            COMPAT_PREFIX, PADDLE_PREFIX, 1
        )
        try:
            target_module = importlib.import_module(target_name)
        except ModuleNotFoundError:
            # compat-only subpackage with no paddle counterpart: nothing to alias.
            continue
        for attr_name in getattr(compat_module, PUBLIC_ATTR_DECLARATION):
            if attr_name.startswith("_"):
                continue
            compat_attr = getattr(compat_module, attr_name)
            current = getattr(target_module, attr_name, _MISSING)
            if current is compat_attr:
                # Already the same object: skip so restore has nothing to undo.
                continue
            _PADDLE_NAMESPACE_SAVED[(target_module, attr_name)] = current
            if current is _MISSING:
                # torch-only symbol: alias directly.
                setattr(target_module, attr_name, compat_attr)
            elif isinstance(compat_attr, type):
                # existing class: caller-aware proxy.
                setattr(
                    target_module,
                    attr_name,
                    _make_caller_aware_class_proxy(current, compat_attr),
                )
            else:
                # existing function: caller-aware dispatcher.
                setattr(
                    target_module,
                    attr_name,
                    dispatch_compat_api(compat_attr)(current),
                )
    _patch_tensor_methods()
    _COMPAT_ENABLED = True


def _restore_paddle_namespace_aliases() -> None:
    """Undo :func:`_apply_paddle_namespace_aliases`, restoring the paddle namespace."""
    global _COMPAT_ENABLED
    for (target_module, attr_name), original in _PADDLE_NAMESPACE_SAVED.items():
        if original is _MISSING:
            if hasattr(target_module, attr_name):
                delattr(target_module, attr_name)
        else:
            setattr(target_module, attr_name, original)
    _PADDLE_NAMESPACE_SAVED.clear()
    _COMPAT_ENABLED = False
