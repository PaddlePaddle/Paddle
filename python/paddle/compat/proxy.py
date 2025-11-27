# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

import importlib
import importlib.abc
import importlib.util
import inspect
import pkgutil
import sys
import types
import warnings
from contextlib import contextmanager
from functools import cache
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterable


def warning_about_fake_interface(name: str):
    warnings.warn(
        f"The interface '{name}' is a fake implementation for torch compatibility. "
        "It does not have the actual functionality of PyTorch. "
        "Please refer to the PaddlePaddle documentation for equivalent functionality.",
        category=UserWarning,
        stacklevel=2,
    )


def create_fake_class(name, attrs: dict[str, Any]):
    """Create a fake class with the given name and attributes."""
    new_fn = lambda *args, **kwargs: warning_about_fake_interface(name)
    attrs["__init__"] = new_fn
    return type(name, (), attrs)


def create_fake_function(name):
    """Create a fake function with the given name and implementation."""
    fn = lambda *args, **kwargs: warning_about_fake_interface(name)
    fn.__name__ = name
    return fn


class OverriddenAttribute:
    def get_value(self):
        raise NotImplementedError


class LazyImportOverriddenAttribute(OverriddenAttribute):
    def __init__(self, full_name: str):
        self._full_name = full_name

    def get_value(self):
        parts = self._full_name.split(".")
        root_module = importlib.import_module(parts[0])
        result = root_module
        for part in parts[1:]:
            result = getattr(result, part)
        return result


class RawOverriddenAttribute(OverriddenAttribute):
    def __init__(self, value: Any):
        self._value = value

    def get_value(self):
        return self._value


class ProxyModule(types.ModuleType):
    def __init__(
        self,
        original_module: types.ModuleType,
        proxy_name: str,
        overrides: dict[str, OverriddenAttribute],
    ):
        super().__init__(proxy_name)
        self._original_module = original_module
        self._proxy_name = proxy_name
        self._overrides = overrides

    def __getattr__(self, name: str) -> Any:
        if name in self._overrides:
            return self._overrides[name].get_value()
        return getattr(self._original_module, name)


GLOBAL_OVERRIDES: dict[str, OverriddenAttribute] = {
    "torch.relu": LazyImportOverriddenAttribute("paddle.nn.functional.relu"),
}

TORCH_PROXY_BLOCKED_MODULES = {
    "tvm_ffi",
    "transformers",
}

MAGIC_DISABLED_MODULE_ATTR: str = "__disable_torch_proxy__"
MAGIC_ENABLED_MODULE_ATTR: str = "__enable_torch_proxy__"


def _extend_torch_proxy_overrides(
    overrides: dict[str, OverriddenAttribute],
) -> None:
    GLOBAL_OVERRIDES.update(overrides)


@cache
def _register_compat_override():
    import paddle.compat

    PADDLE_PREFIX = "paddle.compat"
    TORCH_PREFIX = "torch"
    PUBLIC_ATTR_DECLARATION = "__all__"

    compat_overrides = {}
    for module_info in pkgutil.walk_packages(
        paddle.compat.__path__,
        paddle.compat.__name__ + ".",
    ):
        module = importlib.import_module(module_info.name)
        if hasattr(module, PUBLIC_ATTR_DECLARATION):
            public_attrs = getattr(module, PUBLIC_ATTR_DECLARATION)
            torch_module_name = module_info.name.replace(
                PADDLE_PREFIX, TORCH_PREFIX, 1
            )
            for attr_name in public_attrs:
                if attr_name.startswith("_"):
                    continue
                paddle_attr = getattr(module, attr_name)
                torch_attr_name = f"{torch_module_name}.{attr_name}"
                compat_overrides[torch_attr_name] = RawOverriddenAttribute(
                    paddle_attr
                )
    _extend_torch_proxy_overrides(compat_overrides)


def _is_specific_module_or_its_submodule(name: str, module: str) -> bool:
    return name == module or name.startswith(f"{module}.")


def _is_torch_module(name: str) -> bool:
    return _is_specific_module_or_its_submodule(name, "torch")


def _is_torch_proxy_local_enabled_module(name: str, scope: set[str]) -> bool:
    for enabled_module in scope:
        if _is_specific_module_or_its_submodule(name, enabled_module):
            return True
    return False


def _is_torch_proxy_blocked_module(name: str) -> bool:
    for blocked_module in TORCH_PROXY_BLOCKED_MODULES:
        if _is_specific_module_or_its_submodule(name, blocked_module):
            return True
    return False


def _is_called_by_module_with_specific_dunder_attr(dunder_attr: str) -> bool:
    stack = inspect.stack()
    for frame_info in stack[1:]:
        if frame_info.frame.f_globals.get(dunder_attr):
            return True
    return False


def _is_called_by_torch_proxy_blocked_module():
    return _is_called_by_module_with_specific_dunder_attr(
        MAGIC_DISABLED_MODULE_ATTR
    )


def _is_called_by_torch_proxy_local_enabled_module():
    return _is_called_by_module_with_specific_dunder_attr(
        MAGIC_ENABLED_MODULE_ATTR
    )


class TorchProxyMetaFinder:
    """
    PyTorch compatibility layer for PaddlePaddle.

    This class provides a way to `import torch` but actually loads PaddlePaddle.

    Inspired by the setuptools _distutils_hack.
    """

    _local_enabled_scope: set[str]
    _globally_enabled: bool

    def __init__(self, scope: set[str] | None = None):
        self._set_scope(scope)

    def _set_scope(self, scope: set[str] | None):
        self._local_enabled_scope = scope or set()
        self._globally_enabled = scope is None

    def find_spec(self, fullname, path, target=None):
        if _is_torch_proxy_blocked_module(fullname):
            return self._find_spec_for_torch_proxy_blocked_module(fullname)

        if _is_torch_proxy_local_enabled_module(
            fullname, self._local_enabled_scope
        ):
            return self._find_spec_for_torch_proxy_local_enabled_module(
                fullname
            )

        if not _is_torch_module(fullname):
            return None

        if _is_called_by_torch_proxy_blocked_module():
            return None

        if (
            not self._globally_enabled
            and not _is_called_by_torch_proxy_local_enabled_module()
        ):
            return None

        return self._find_spec_for_torch_module(fullname)

    def _find_spec_for_specific_module(
        self,
        fullname: str,
        enable_proxy_when_exec_module: bool,
        patched_dunder_attr: str,
    ):
        # Return a special loader that imports the blocked module without torch proxy
        with use_torch_proxy_guard(enable=False):
            spec = importlib.util.find_spec(fullname)
            if spec is None:
                return None
            original_loader = spec.loader
            if original_loader is None:
                return None

            class TorchBlockedModuleLoader(importlib.abc.Loader):
                def create_module(self, spec):
                    mod = original_loader.create_module(spec)
                    if mod is None:
                        # If original loader returns None, create default module
                        # and ensure it has necessary attributes from spec
                        mod = types.ModuleType(spec.name)
                        mod.__spec__ = spec
                        mod.__loader__ = self
                        if spec.origin is not None:
                            mod.__file__ = spec.origin
                        if spec.submodule_search_locations is not None:
                            mod.__path__ = list(spec.submodule_search_locations)
                    return mod

                def exec_module(self, module):
                    # Import the real module with torch proxy disabled
                    with use_torch_proxy_guard(
                        enable=enable_proxy_when_exec_module, silent=True
                    ):
                        original_loader.exec_module(module)
                    # Mark module as torch proxy disabled/local enabled
                    module.__dict__[patched_dunder_attr] = True

        spec.loader = TorchBlockedModuleLoader()
        return spec

    def _find_spec_for_torch_proxy_local_enabled_module(self, fullname: str):
        return self._find_spec_for_specific_module(
            fullname,
            enable_proxy_when_exec_module=True,
            patched_dunder_attr=MAGIC_ENABLED_MODULE_ATTR,
        )

    def _find_spec_for_torch_proxy_blocked_module(self, fullname: str):
        return self._find_spec_for_specific_module(
            fullname,
            enable_proxy_when_exec_module=False,
            patched_dunder_attr=MAGIC_DISABLED_MODULE_ATTR,
        )

    def _find_spec_for_torch_module(self, fullname: str):
        # Map the requested torch fullname to the corresponding paddle fullname.
        module_name = fullname.replace("torch", "paddle", 1)
        source_module = importlib.import_module(module_name)
        overrides = {
            k.removeprefix(f"{fullname}."): v
            for k, v in GLOBAL_OVERRIDES.items()
            if k.startswith(f"{fullname}.")
        }

        is_pkg = hasattr(source_module, "__path__")

        class TorchProxyLoader(importlib.abc.Loader):
            def __init__(self, source, target_name):
                self._source = source
                self._target_name = target_name

            def create_module(self, spec):
                # Create a new module object that will act as the "torch..." module.
                mod = ProxyModule(self._source, self._target_name, overrides)
                # Preserve file/path information for tooling/debugging.
                mod.__file__ = getattr(self._source, "__file__", None)
                if is_pkg:
                    # package must expose __path__ so import machinery can find submodules
                    mod.__path__ = list(getattr(self._source, "__path__", []))
                    mod.__package__ = self._target_name
                else:
                    mod.__package__ = self._target_name.rpartition('.')[0]
                return mod

            def exec_module(self, module):
                # Populate the new module with attributes from the source paddle module.
                # Skip a few special attributes that should reflect the new module name.
                for k, v in self._source.__dict__.items():
                    if k in ("__name__", "__package__", "__path__", "__spec__"):
                        continue
                    if k in overrides:
                        continue
                    if isinstance(v, types.ModuleType):
                        v = ProxyModule(
                            v,
                            f"{self._target_name}.{k}",
                            {
                                kk.removeprefix(f"{k}."): vv
                                for kk, vv in overrides.items()
                                if kk.startswith(f"{k}.")
                            },
                        )
                    module.__dict__[k] = v

        # Use fullname for the spec name and mark as package when appropriate so that
        # statements like `import torch.nn.functional` work correctly.
        return importlib.util.spec_from_loader(
            fullname,
            TorchProxyLoader(source_module, fullname),
            is_package=is_pkg,
            origin=getattr(source_module, "__file__", None),
        )


TORCH_PROXY_FINDER = TorchProxyMetaFinder()


def _clear_torch_modules():
    for name in list(sys.modules):
        if _is_torch_module(name):
            del sys.modules[name]


def _modify_scope_of_torch_proxy(
    scope: set[str] | None,
    *,
    silent: bool = False,
) -> None:
    def _warn_or_not(msg: str):
        if silent:
            return
        warnings.warn(msg)

    if TORCH_PROXY_FINDER not in sys.meta_path:
        TORCH_PROXY_FINDER._set_scope(scope)
        return

    if TORCH_PROXY_FINDER._globally_enabled:
        if scope is not None:
            _warn_or_not(
                "PyTorch already enabled globally, scope modification ignored."
            )
        TORCH_PROXY_FINDER._set_scope(scope)
        return
    if scope is None:
        _warn_or_not(
            "Enabling PyTorch proxy globally, previous scope will be ignored."
        )
        TORCH_PROXY_FINDER._globally_enabled = True
        return
    if scope != TORCH_PROXY_FINDER._local_enabled_scope:
        _warn_or_not(
            f"Extending PyTorch proxy scope, previous scope: {TORCH_PROXY_FINDER._local_enabled_scope}, new scope: {scope}."
        )
    TORCH_PROXY_FINDER._local_enabled_scope |= scope


def enable_torch_proxy(
    *,
    scope: set[str] | None = None,
    silent: bool = False,
) -> None:
    """
    Enable the PyTorch proxy by adding the TorchProxyMetaFinder to sys.meta_path.
    This allows importing 'torch' modules that are actually proxies to PaddlePaddle.

    Example:
        .. code-block:: python

            >>> import paddle
            >>> paddle.compat.enable_torch_proxy()  # Enable torch proxy globally
            >>> import torch  # This will import paddle as torch
            >>> assert torch.sin is paddle.sin
    """
    _register_compat_override()
    _clear_torch_modules()
    _modify_scope_of_torch_proxy(scope, silent=silent)
    sys.meta_path.insert(0, TORCH_PROXY_FINDER)


def disable_torch_proxy() -> None:
    """
    Disable the PyTorch proxy by removing the TorchProxyMetaFinder from sys.meta_path.
    This prevents 'torch' imports from being proxied to PaddlePaddle.

    Example:
        .. code-block:: python

            >>> import paddle
            >>> paddle.compat.enable_torch_proxy()  # Enable torch proxy globally
            >>> import torch  # This will import paddle as torch
            >>> assert torch.sin is paddle.sin
            >>> paddle.compat.disable_torch_proxy()  # Disable torch proxy
            >>> try:
            ...     import torch  # This will raise ModuleNotFoundError
            ... except ModuleNotFoundError:
            ...     print("PyTorch proxy is disabled.")
    """
    if TORCH_PROXY_FINDER in sys.meta_path:
        sys.meta_path.remove(TORCH_PROXY_FINDER)
        _clear_torch_modules()
        return
    warnings.warn("torch proxy is not installed.")


@contextmanager
def use_torch_proxy_guard(
    *,
    enable: bool = True,
    scope: set[str] | None = None,
    silent: bool = False,
):
    """
    Context manager to temporarily enable or disable the PyTorch proxy.

    When `enable` is True (default), the PyTorch proxy is enabled for the duration
    of the context and restored to its previous state afterwards. When `enable`
    is False, the PyTorch proxy is disabled for the duration of the context and
    restored afterwards.

    Args:
        enable (bool, optional): Whether to enable or disable the PyTorch proxy
            within the context. Defaults to True.

    Example:
        .. code-block:: python

            >>> import paddle

            >>> with paddle.compat.use_torch_proxy_guard():
            ...     # code that requires the Torch proxy to be enabled
            ...     import torch
            ...     assert torch.sin is paddle.sin
            ...     # Temporarily disable the Torch proxy
            ...     with paddle.compat.use_torch_proxy_guard(enable=False):
            ...         try:
            ...             import torch
            ...         except ModuleNotFoundError:
            ...             print("Torch proxy is disabled within this block.")
            ...     # Torch proxy is re-enabled here
            ...     import torch
            ...     assert torch.sin is paddle.sin
    """
    already_has_torch_proxy = TORCH_PROXY_FINDER in sys.meta_path
    original_local_enabled_scope = TORCH_PROXY_FINDER._local_enabled_scope
    original_globally_enabled = TORCH_PROXY_FINDER._globally_enabled
    if enable == already_has_torch_proxy and (
        (original_globally_enabled and scope is None)
        or (original_local_enabled_scope == (scope or set()))
    ):
        yield
        return
    if enable:
        enable_torch_proxy(scope=scope, silent=silent)
        try:
            yield
        finally:
            TORCH_PROXY_FINDER._local_enabled_scope = (
                original_local_enabled_scope
            )
            TORCH_PROXY_FINDER._globally_enabled = original_globally_enabled
            disable_torch_proxy()
    else:
        disable_torch_proxy()
        try:
            yield
        finally:
            enable_torch_proxy(scope=None, silent=True)
            TORCH_PROXY_FINDER._local_enabled_scope = (
                original_local_enabled_scope
            )
            TORCH_PROXY_FINDER._globally_enabled = original_globally_enabled


def extend_torch_proxy_blocked_modules(modules: Iterable[str]):
    """Add modules to the PyTorch proxy blocked list.

    Modules in the blocked list will not use PyTorch proxy when imported,
    and their functions will not trigger PyTorch proxy when called.

    Args:
        modules(Iterable[str]): An iterable of module names to block from PyTorch proxy.

    Example:
        .. code-block:: python

            >>> import paddle
            >>> paddle.compat.enable_torch_proxy()  # Enable torch proxy globally
            >>> # Add 'my_custom_module' to the blocked list
            >>> paddle.compat.extend_torch_proxy_blocked_modules(['my_custom_module'])
            >>> import my_custom_module  # This import will not use torch proxy
    """
    TORCH_PROXY_BLOCKED_MODULES.update(modules)
