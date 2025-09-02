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

import types
from typing import Any

import paddle

from ._ops import import_module, load_library

PADDLE_CLASSES_MODULE_NAME = "paddle.classes"


class ClassesNameSpace(types.ModuleType):
    def __init__(self, name: str):
        super().__init__(f"{PADDLE_CLASSES_MODULE_NAME}.{name}")
        self.name = name

    def __getattr__(self, name: str) -> Any:
        if name == "__file__":
            return PADDLE_CLASSES_MODULE_NAME  # type: ignore
        return paddle.base.core.torch_compat._get_custom_class_python_wrapper(
            self.name, name
        )


class PaddleClassesModule(types.ModuleType):
    __file__ = "_classes.py"

    def __init__(self):
        super().__init__(PADDLE_CLASSES_MODULE_NAME)

    def __getattr__(self, name: str):
        namespace = ClassesNameSpace(name)
        # Insert to __dict__ to avoid repeatedly __getattr__ overhead
        setattr(self, name, namespace)
        return namespace

    def import_module(self, module):
        return import_module(module)

    def load_library(self, path):
        return load_library(path)


classes = PaddleClassesModule()
