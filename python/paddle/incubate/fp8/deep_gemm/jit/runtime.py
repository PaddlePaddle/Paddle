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

# The file has been adapted from DeepSeek DeepEP project
# Copyright (c) 2025 DeepSeek
# Licensed under the MIT License - https://github.com/deepseek-ai/DeepEP/blob/main/LICENSE
from __future__ import annotations

import ctypes
import os

import paddle

from .template import map_ctype


class Runtime:
    def __init__(self, path: str) -> None:
        self.path = path
        self.lib = None
        self.args = None

        assert self.is_path_valid(self.path)

    @staticmethod
    def is_path_valid(path: str) -> bool:
        # Exists and is a directory
        if not os.path.exists(path) or not os.path.isdir(path):
            return False

        # Contains all necessary files
        files = ['kernel.cu', 'kernel.args', 'kernel.so']
        return all(os.path.exists(os.path.join(path, file)) for file in files)

    def __call__(self, *args) -> int:
        # Load SO file
        if self.lib is None or self.args is None:
            self.lib = ctypes.CDLL(os.path.join(self.path, 'kernel.so'))
            with open(os.path.join(self.path, 'kernel.args'), 'r') as f:
                self.args = eval(f.read())

        # Check args and launch
        assert len(args) == len(self.args), (
            f'Expected {len(self.args)} arguments, got {len(args)}'
        )
        cargs = []
        for arg, (name, dtype) in zip(args, self.args):
            if isinstance(arg, paddle.Tensor):
                assert arg.dtype == dtype, (
                    f'Expected tensor dtype `{dtype}` for `{name}`, got `{arg.dtype}`'
                )
            else:
                assert isinstance(arg, dtype), (
                    f'Expected built-in type `{dtype}` for `{name}`, got `{type(arg)}`'
                )
            cargs.append(map_ctype(arg))

        return_code = ctypes.c_int(0)
        self.lib.launch(*cargs, ctypes.byref(return_code))
        return return_code.value


class RuntimeCache:
    def __init__(self) -> None:
        self.cache = {}

    def __getitem__(self, path: str) -> Runtime | None:
        # In Python runtime
        if path in self.cache:
            return self.cache[path]

        # Already compiled
        if os.path.exists(path) and Runtime.is_path_valid(path):
            runtime = Runtime(path)
            self.cache[path] = runtime
            return runtime
        return None

    def __setitem__(self, path, runtime) -> None:
        self.cache[path] = runtime
