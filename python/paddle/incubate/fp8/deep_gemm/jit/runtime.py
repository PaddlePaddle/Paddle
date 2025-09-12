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

import ctypes
import os
from typing import Optional


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
        files = ["kernel.cu", "kernel.args", "kernel.so"]
        return all(os.path.exists(os.path.join(path, file)) for file in files)

    def __call__(self, *args) -> int:
        # Load SO file
        if self.lib is None:
            self.lib = ctypes.CDLL(os.path.join(self.path, "kernel.so"))

        if len(args) == 9:
            cargs = [
                ctypes.c_void_p(args[0].data_ptr()),
                ctypes.c_void_p(args[1].data_ptr()),
                ctypes.c_void_p(args[2].data_ptr()),
                ctypes.c_void_p(args[3].data_ptr()),
                ctypes.c_void_p(args[4].data_ptr()),
                ctypes.c_int(args[5]),
                ctypes.c_void_p(args[6].cuda_stream),
                ctypes.c_int(args[7]),
                ctypes.c_int(args[8]),
            ]
        elif len(args) == 10:
            cargs = [
                ctypes.c_void_p(args[0].data_ptr()),
                ctypes.c_void_p(args[1].data_ptr()),
                ctypes.c_void_p(args[2].data_ptr()),
                ctypes.c_void_p(args[3].data_ptr()),
                ctypes.c_void_p(args[4].data_ptr()),
                ctypes.c_void_p(args[5].data_ptr()),
                ctypes.c_int(args[6]),
                ctypes.c_void_p(args[7].cuda_stream),
                ctypes.c_int(args[8]),
                ctypes.c_int(args[9]),
            ]
        elif len(args) == 11:
            cargs = [
                ctypes.c_void_p(args[0].data_ptr()),
                ctypes.c_void_p(args[1].data_ptr()),
                ctypes.c_void_p(args[2].data_ptr()),
                ctypes.c_void_p(args[3].data_ptr()),
                ctypes.c_void_p(args[4].data_ptr()),
                ctypes.c_void_p(args[5].data_ptr()),
                ctypes.c_int(args[6]),
                ctypes.c_int(args[7]),
                ctypes.c_void_p(args[8].cuda_stream),
                ctypes.c_int(args[9]),
                ctypes.c_int(args[10]),
            ]
        else:
            raise ValueError("Invalid number of arguments")
        return_code = ctypes.c_int(0)
        self.lib.launch(*cargs, ctypes.byref(return_code))


class RuntimeCache:
    def __init__(self) -> None:
        self.cache = {}

    def __getitem__(self, path: str) -> Optional['Runtime']:
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
