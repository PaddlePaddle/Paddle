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

import importlib
import importlib.abc
import importlib.util
import inspect
import sys
import types
import warnings
from collections.abc import Iterable
from contextlib import contextmanager
from typing import Any


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


class ProxyModule(types.ModuleType):
    def __init__(
        self,
        original_module: types.ModuleType,
        proxy_name: str,
        overrides: dict[str, Any],
    ):
        super().__init__(proxy_name)
        self._original_module = original_module
        self._proxy_name = proxy_name
        self._overrides = overrides

    def __getattr__(self, name: str) -> Any:
        if name in self._overrides:
            return self._overrides[name]
        return getattr(self._original_module, name)


GLOBAL_OVERRIDES = {}

TORCH_PROXY_BLOCKED_MODULES = {
    "tvm_ffi",
    "transformers",
}


def _is_specific_module_or_its_submodule(name: str, module: str) -> bool:
    return name == module or name.startswith(f"{module}.")


def _is_torch_module(name: str) -> bool:
    return _is_specific_module_or_its_submodule(name, "torch")


def _is_torch_proxy_blocked_module(name: str) -> bool:
    for blocked_module in TORCH_PROXY_BLOCKED_MODULES:
        if _is_specific_module_or_its_submodule(name, blocked_module):
            return True
    return False


def _is_called_by_torch_proxy_blocked_module():
    stack = inspect.stack()
    for frame_info in stack[1:]:
        if frame_info.frame.f_globals.get("__disable_torch_proxy__"):
            return True
    return False


class TorchProxyMetaFinder:
    """
    PyTorch compatibility layer for PaddlePaddle.

    This class provides a way to `import torch` but actually loads PaddlePaddle.

    Inspired by the setuptools _distutils_hack.
    """

    def find_spec(self, fullname, path, target=None):
        if _is_torch_proxy_blocked_module(fullname):
            return self._find_spec_for_torch_proxy_blocked_module(fullname)

        if not _is_torch_module(fullname):
            return None

        if _is_called_by_torch_proxy_blocked_module():
            return None

        return self._find_spec_for_torch_module(fullname)

    def _find_spec_for_torch_proxy_blocked_module(self, fullname: str):
        # Return a special loader that imports the blocked module without torch proxy
        with use_torch_proxy_guard(False):
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
                    with use_torch_proxy_guard(False):
                        original_loader.exec_module(module)
                    # Mark module as torch proxy disabled
                    module.__dict__["__disable_torch_proxy__"] = True

        spec.loader = TorchBlockedModuleLoader()
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


def enable_torch_proxy() -> None:
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
    _clear_torch_modules()
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
def use_torch_proxy_guard(enable: bool = True):
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
    if enable == already_has_torch_proxy:
        yield
        return
    if enable:
        enable_torch_proxy()
        try:
            yield
        finally:
            disable_torch_proxy()
    else:
        disable_torch_proxy()
        try:
            yield
        finally:
            enable_torch_proxy()


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
