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

import importlib
import inspect
import pkgutil
import sys
from functools import wraps
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import types
    from collections.abc import Generator


# (live module, attr name) -> original value for every patched ``paddle.*``
# symbol, so ``disable_compat()`` can restore the paddle namespace.
_PADDLE_NAMESPACE_SAVED: dict[tuple[types.ModuleType, str], Any] = {}


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
        @wraps(native_fn)
        def dispatcher(*args: Any, **kwargs: Any) -> Any:
            if (
                len(_PADDLE_NAMESPACE_SAVED) > 0
                and not _caller_is_paddle_internal()
            ):
                return compat_fn(*args, **kwargs)
            return native_fn(*args, **kwargs)

        dispatcher.__compat_fn__ = compat_fn
        dispatcher.__native_fn__ = native_fn
        dispatcher.__signature__ = inspect.signature(compat_fn)
        return dispatcher

    return decorator


def _iter_compat_modules() -> Generator[types.ModuleType, None, None]:
    """Yield ``paddle.compat`` modules that declare ``__all__``.

    ``pkgutil.walk_packages`` skips the starting package, so the root
    ``paddle.compat`` (holding the top-level functions) is yielded explicitly.
    """
    import paddle.compat

    if hasattr(paddle.compat, "__all__"):
        yield paddle.compat
    for module_info in pkgutil.walk_packages(
        paddle.compat.__path__,
        paddle.compat.__name__ + ".",
    ):
        compat_module = importlib.import_module(module_info.name)
        if not hasattr(compat_module, "__all__"):
            continue
        yield compat_module


def _make_caller_aware_class_proxy(native_cls: type, compat_cls: type) -> type:
    """Create a class proxy that selects compat only for external callers."""

    class _CompatAwareMeta(type(compat_cls)):
        def __call__(cls, *args: Any, **kwargs: Any) -> Any:
            if cls is proxy:
                if (
                    len(_PADDLE_NAMESPACE_SAVED) > 0
                    and not _caller_is_paddle_internal()
                ):
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
        native_method = getattr(paddle.Tensor, attr_name, None)
        if native_method is None:
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
    symbol that has a ``paddle.*`` counterpart, plus the Tensor methods."""
    if _PADDLE_NAMESPACE_SAVED:
        return

    for compat_module in _iter_compat_modules():
        target_name = compat_module.__name__.replace(
            "paddle.compat", "paddle", 1
        )
        try:
            target_module = importlib.import_module(target_name)
        except ModuleNotFoundError:
            continue
        for attr_name in compat_module.__all__:
            compat_attr = getattr(compat_module, attr_name)
            current = getattr(target_module, attr_name, None)
            if current is None or current is compat_attr:
                continue
            _PADDLE_NAMESPACE_SAVED[(target_module, attr_name)] = current
            if isinstance(compat_attr, type):
                setattr(
                    target_module,
                    attr_name,
                    _make_caller_aware_class_proxy(current, compat_attr),
                )
            else:
                setattr(
                    target_module,
                    attr_name,
                    dispatch_compat_api(compat_attr)(current),
                )
    _patch_tensor_methods()


def _restore_paddle_namespace_aliases() -> None:
    """Undo :func:`_apply_paddle_namespace_aliases`, restoring the paddle namespace."""
    for (target_module, attr_name), original in _PADDLE_NAMESPACE_SAVED.items():
        setattr(target_module, attr_name, original)
    _PADDLE_NAMESPACE_SAVED.clear()
