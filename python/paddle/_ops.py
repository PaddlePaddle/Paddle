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

import contextlib
import ctypes
import importlib
import os
import sys
import types
from functools import cached_property
from typing import Any, Callable, Generic, TypeVar

from typing_extensions import ParamSpec

import paddle

_InputT = ParamSpec("_InputT")
_RetT = TypeVar("_RetT")

PADDLE_OPS_MODULE_NAME = "paddle.ops"

# Query `hasattr` only once.
_SET_GLOBAL_FLAGS = hasattr(sys, "getdlopenflags") and hasattr(
    sys, "setdlopenflags"
)


@contextlib.contextmanager
def dl_open_guard():
    """
    Context manager to set the RTLD_GLOBAL dynamic linker flag while we open a
    shared library to load custom operators.
    """
    if not _SET_GLOBAL_FLAGS:
        yield
        return
    old_flags = sys.getdlopenflags()
    sys.setdlopenflags(old_flags | ctypes.RTLD_GLOBAL)
    try:
        yield
    finally:
        sys.setdlopenflags(old_flags)


def import_module(module: str):
    return importlib.import_module(module)


def load_library(path: str):
    """
    Load a shared library at the specified path.
    """
    path = os.path.realpath(path)
    with dl_open_guard():
        ctypes.CDLL(path)


class OverloadedOpFunction(Generic[_InputT, _RetT]):
    def __init__(self, namespace: str, name: str):
        self.namespace = namespace
        self.name = name

    @cached_property
    def callable_fn(self) -> Callable[_InputT, _RetT]:
        return paddle.base.core.torch_compat._get_operation(
            f"{self.namespace}::{self.name}"
        )

    def __getattr__(self, name: str) -> Callable[_InputT, _RetT]:
        if name == "default":
            return self.callable_fn
        raise AttributeError(
            f"'{self.namespace}.{self.name}' has no attribute '{name}'"
        )

    def __call__(self, *args: _InputT.args, **kwargs: _InputT.kwargs) -> _RetT:
        return self.callable_fn(*args, **kwargs)


class OpNameSpace(types.ModuleType):
    def __init__(self, name):
        super().__init__(f"{PADDLE_OPS_MODULE_NAME}.{name}")
        self.name = name

    def __getattr__(self, name: str) -> OverloadedOpFunction[..., Any]:
        if name == "__file__":
            return PADDLE_OPS_MODULE_NAME  # type: ignore
        return OverloadedOpFunction(self.name, name)


class PaddleOpsModule(types.ModuleType):
    __file__ = "_ops.py"

    def __init__(self):
        super().__init__(PADDLE_OPS_MODULE_NAME)

    def __getattr__(self, name: str):
        namespace = OpNameSpace(name)
        # Insert to __dict__ to avoid repeatedly __getattr__ overhead
        setattr(self, name, namespace)
        return namespace

    def import_module(self, module):
        return import_module(module)

    def load_library(self, path):
        return load_library(path)


ops = PaddleOpsModule()
