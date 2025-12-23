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
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from collections.abc import Generator, Iterable

    from typing_extensions import TypeAlias

    _ScopeType: TypeAlias = str | Iterable[str] | None


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
    frame = inspect.currentframe()
    while frame is not None:
        if frame.f_globals.get(dunder_attr):
            return True
        frame = frame.f_back
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
            if fullname in TORCH_MODULES_CACHE:
                return self._find_spec_for_cached_torch_module(fullname)
            return None

        if (
            not self._globally_enabled
            and not _is_called_by_torch_proxy_local_enabled_module()
        ):
            if fullname in TORCH_MODULES_CACHE:
                return self._find_spec_for_cached_torch_module(fullname)
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

            class SpecificModuleLoader(importlib.abc.Loader):
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

        spec.loader = SpecificModuleLoader()
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

    def _find_spec_for_cached_torch_module(self, fullname: str):
        module = TORCH_MODULES_CACHE[fullname]

        # Return cached module before enable proxy
        class CachedTorchModuleLoader(importlib.abc.Loader):
            def create_module(self, spec):
                return module

            def exec_module(self, module):
                pass

        # Always treat cached modules as packages to allow submodules to be loaded.
        # This is necessary because some modules (e.g. torch._C) are not packages
        # but have submodules (e.g. torch._C._dynamo) attached to them.
        spec = importlib.util.spec_from_loader(
            fullname,
            CachedTorchModuleLoader(),
            origin=getattr(module, "__file__", None),
            is_package=True,
        )
        spec.submodule_search_locations = list(getattr(module, "__path__", []))
        return spec

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
TORCH_MODULES_CACHE: dict[str, types.ModuleType] = {}


def _clear_torch_proxy_modules():
    for name, module in list(sys.modules.items()):
        if _is_torch_module(name) and isinstance(module, ProxyModule):
            del sys.modules[name]


def _swap_torch_modules_to_cache():
    for name, module in list(sys.modules.items()):
        if _is_torch_module(name):
            if not isinstance(module, ProxyModule):
                TORCH_MODULES_CACHE[name] = sys.modules[name]
            del sys.modules[name]


def _copy_torch_modules_from_cache():
    for name in list(TORCH_MODULES_CACHE):
        assert _is_torch_module(name), f"`{name}` is not a PyTorch module"
        sys.modules[name] = TORCH_MODULES_CACHE[name]


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
            "Enabling PyTorch compat globally, previous scope will be ignored."
        )
        TORCH_PROXY_FINDER._globally_enabled = True
        return
    if scope != TORCH_PROXY_FINDER._local_enabled_scope:
        _warn_or_not(
            f"Extending PyTorch compat scope, previous scope: {TORCH_PROXY_FINDER._local_enabled_scope}, new scope: {scope}."
        )
    TORCH_PROXY_FINDER._local_enabled_scope |= scope


def _parse_scope(scope: str | Iterable[str] | None) -> set[str] | None:
    if scope is None:
        return None
    if isinstance(scope, str):
        return {scope}
    return set(scope)


def enable_torch_proxy(
    *,
    scope: _ScopeType = None,
    blocked_modules: _ScopeType = None,
    backend: Literal["torch"] = "torch",
    silent: bool = False,
) -> None:
    """
    Enable the PyTorch compat by adding the TorchProxyMetaFinder to sys.meta_path.
    This allows importing 'torch' modules that are actually proxies to PaddlePaddle.

    Args:
        scope (str or Iterable[str], optional): Specific module or modules to enable
            PyTorch compat for. If None, enables PyTorch compat globally. Defaults to None.
        blocked_modules (str or Iterable[str], optional): Specific module or modules to
            exclude from PyTorch compat. Defaults to None.
        backend (str, optional): The backend to enable compat for. Currently only
            "torch" is supported. Defaults to "torch".
        silent (bool, optional): If True, suppresses warnings about scope changes.
            Defaults to False.

    Example:
        .. code-block:: pycon
            :name: enable-compat-in-global-scope

            >>> import paddle
            >>> paddle.enable_compat()  # Enable torch compat globally
            >>> import torch  # type: ignore[import-not-found] # This will import paddle as torch
            >>> assert torch.sin is paddle.sin
            >>> paddle.disable_compat()  # Disable torch compat

        .. code-block:: pycon
            :name: enable-compat-in-specific-scope

            >>> import paddle
            >>> paddle.enable_compat(scope={"triton"})  # Enable torch compat for 'triton' module only
            >>> import triton  # type: ignore[import-untyped] # All `import torch` inside `triton` will proxy to paddle
            >>> try:
            ...     import torch  # type: ignore[import-not-found] # This will raise ModuleNotFoundError
            ... except ModuleNotFoundError:
            ...     print("PyTorch compat is not enabled globally.")
            >>> paddle.disable_compat()  # Disable torch compat
    """
    assert backend == "torch", f"Unsupported backend: {backend}"
    blocked_modules = _parse_scope(blocked_modules)
    if blocked_modules is not None:
        extend_torch_proxy_blocked_modules(blocked_modules)
    scope = _parse_scope(scope)
    _register_compat_override()
    _swap_torch_modules_to_cache()
    _modify_scope_of_torch_proxy(scope, silent=silent)
    sys.meta_path.insert(0, TORCH_PROXY_FINDER)


def disable_torch_proxy() -> None:
    """
    Disable the PyTorch proxy by removing the TorchProxyMetaFinder from sys.meta_path.
    This prevents 'torch' imports from being proxied to PaddlePaddle.

    Example:
        .. code-block:: pycon

            >>> import paddle
            >>> paddle.enable_compat()  # Enable torch compat globally
            >>> import torch  # type: ignore[import-not-found] # This will import paddle as torch
            >>> assert torch.sin is paddle.sin
            >>> paddle.disable_compat()  # Disable torch compat
            >>> try:
            ...     import torch  # This will raise ModuleNotFoundError
            ... except ModuleNotFoundError:
            ...     print("PyTorch compat is disabled.")
    """
    if TORCH_PROXY_FINDER in sys.meta_path:
        sys.meta_path.remove(TORCH_PROXY_FINDER)
        _clear_torch_proxy_modules()
        _copy_torch_modules_from_cache()
        return
    warnings.warn("torch compat is not installed.")


@contextmanager
def use_torch_proxy_guard(
    *,
    enable: bool = True,
    scope: _ScopeType = None,
    silent: bool = False,
) -> Generator[None, None, None]:
    """
    Context manager to temporarily enable or disable the PyTorch compat.

    When `enable` is True (default), the PyTorch compat is enabled for the duration
    of the context and restored to its previous state afterwards. When `enable`
    is False, the PyTorch compat is disabled for the duration of the context and
    restored afterwards.

    Args:
        enable (bool, optional): Whether to enable or disable the PyTorch compat
            within the context. Defaults to True.
        scope (str or Iterable[str], optional): Specific module or modules to enable
            PyTorch compat for. If None, uses the global scope. Defaults to None.
        silent (bool, optional): If True, suppresses warnings about scope changes.
            Defaults to False.

    Example:
        .. code-block:: pycon

            >>> import paddle

            >>> with paddle.compat.use_torch_proxy_guard():
            ...     # code that requires the Torch compat to be enabled
            ...     import torch  # type: ignore[import-not-found]
            ...
            ...     assert torch.sin is paddle.sin
            ...     # Temporarily disable the Torch compat
            ...     with paddle.compat.use_torch_proxy_guard(enable=False):
            ...         try:
            ...             import torch
            ...         except ModuleNotFoundError:
            ...             print("Torch compat is disabled within this block.")
            ...     # Torch compat is re-enabled here
            ...     import torch
            ...
            ...     assert torch.sin is paddle.sin
    """
    scope = _parse_scope(scope)
    already_has_torch_proxy = TORCH_PROXY_FINDER in sys.meta_path
    original_local_enabled_scope = set(TORCH_PROXY_FINDER._local_enabled_scope)
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


def extend_torch_proxy_blocked_modules(modules: Iterable[str]) -> None:
    """Add modules to the PyTorch proxy blocked list.

    Modules in the blocked list will not use PyTorch compat when imported,
    and their functions will not trigger PyTorch compat when called.

    By default, some modules are already in the blocked list, such as 'tvm_ffi'.

    Args:
        modules(Iterable[str]): An iterable of module names to block from PyTorch compat.

    Example:
        .. code-block:: pycon

            >>> import paddle
            >>> paddle.enable_compat()  # Enable torch compat globally
            >>> # Add 'my_custom_module' to the blocked list
            >>> paddle.compat.extend_torch_proxy_blocked_modules(['my_custom_module'])
            >>> # doctest: +SKIP('my_custom_module is not available')
            >>> import my_custom_module  # type: ignore[import-not-found] # This import will not use torch compat
    """
    TORCH_PROXY_BLOCKED_MODULES.update(modules)


def paddle_triton_fun():
    """
    Enable the triton support and return triton module.
    Args: None.
    Returns: triton module

    Example:
        .. code-block:: pycon

            >>> # doctest: +REQUIRES(env:GPU)
            >>> from paddle.compat import paddle_triton_fun
            >>> triton = paddle_triton_fun()
            >>> import triton.language as tl

            >>> @triton.jit
            >>> def add_kernel(X, Y, Z, N, BLOCK: tl.constexpr):
            ...     pid = tl.program_id(0)
            ...     offs = pid * BLOCK + tl.arange(0, BLOCK)
            ...     mask = offs < N
            ...     x = tl.load(X + offs, mask=mask)
            ...     y = tl.load(Y + offs, mask=mask)
            ...     tl.store(Z + offs, x + y, mask=mask)
    """
    enable_torch_proxy(scope={"triton"})
    import triton

    return triton
