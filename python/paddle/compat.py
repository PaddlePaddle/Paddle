# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved
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
#
# This file implements most of the public API compatible with PyTorch.
# Note that this file does not depend on PyTorch in any way.
# This is a standalone implementation.

import sys
import types
import warnings
from contextlib import contextmanager
from typing import Any

from .tensor.compat import (
    Unfold,
    max,
    median,
    min,
    nanmedian,
    pad,
    slogdet,
    sort,
    split,
)
from .tensor.compat_softmax import softmax

__all__ = [
    'slogdet',
    'softmax',
    'split',
    'sort',
    'Unfold',
    'min',
    'max',
    'median',
    'nanmedian',
    'pad',
]


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


GLOBAL_OVERRIDES = {
    "torch.Generator": create_fake_class(
        "Generator", {"manual_seed": create_fake_function("manual_seed")}
    ),
}


def _is_torch_module(name: str) -> bool:
    return name == "torch" or name.startswith("torch.")


class TorchProxyMetaFinder:
    """
    PyTorch compatibility layer for PaddlePaddle.

    This class provides a way to `import torch` but actually loads PaddlePaddle.

    Inspired by the setuptools _distutils_hack.
    """

    def find_spec(self, fullname, path, target=None):
        if not _is_torch_module(fullname):
            return None

        import importlib
        import importlib.abc
        import importlib.util

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


def enable_torch_proxy():
    sys.meta_path.insert(0, TORCH_PROXY_FINDER)


def disable_torch_proxy():
    if TORCH_PROXY_FINDER in sys.meta_path:
        sys.meta_path.remove(TORCH_PROXY_FINDER)
        for name in list(sys.modules):
            if _is_torch_module(name):
                del sys.modules[name]
        return
    warnings.warn("torch proxy is not installed.")


@contextmanager
def use_torch_proxy_guard(enable: bool = True):
    already_has_torch_proxy = TORCH_PROXY_FINDER in sys.meta_path
    if enable == already_has_torch_proxy:
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
