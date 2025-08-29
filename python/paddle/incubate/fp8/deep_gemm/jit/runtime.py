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

# The file has been adapted from DeepSeek DeepGEMM project
# Copyright (c) 2025 DeepSeek
# Licensed under the MIT License - https://github.com/deepseek-ai/DeepGEMM/blob/main/LICENSE
from __future__ import annotations

import os
import subprocess
import time
from typing import Any

import cuda.bindings.driver as cbd

import paddle
from paddle.utils.cpp_extension.cpp_extension import CUDA_HOME


class Runtime:
    def __init__(self, path: str) -> None:
        self.path = path
        self.lib = None
        self.kernel = None
        assert self.is_path_valid(self.path)

    @staticmethod
    def is_path_valid(path: str) -> bool:
        # Exists and is a directory
        if not os.path.exists(path) or not os.path.isdir(path):
            return False

        # Contains all necessary files
        files = ["kernel.cubin"]
        return all(os.path.exists(os.path.join(path, file)) for file in files)

    @staticmethod
    def generate(kwargs: dict[str, Any]) -> str:
        raise NotImplementedError

    @staticmethod
    def launch(kernel: cbd.CUkernel, kwargs: dict[str, Any]) -> cbd.CUresult:
        raise NotImplementedError

    def __call__(self, **kwargs) -> cbd.CUresult:
        # Load CUBIN
        if self.kernel is None:
            start_time = time.time_ns()

            # Load CUBIN
            path = bytes(os.path.join(self.path, "kernel.cubin"), "utf-8")
            result, self.lib = cbd.cuLibraryLoadFromFile(
                path, [], [], 0, [], [], 0
            )
            assert result == cbd.CUresult.CUDA_SUCCESS, (
                f"Failed to load library: {result}"
            )

            # Extract the kernel name
            # TODO: use `cuda-bindings` API to do this (requires at least 12.8)
            command = [f"{CUDA_HOME}/bin/cuobjdump", "-symbols", path]
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
            )
            assert result.returncode == 0
            illegal_names = [
                "vprintf",
                "__instantiate_kernel",
                "__internal",
                "__assertfail",
            ]
            check_illegal = lambda line: any(
                name in line for name in illegal_names
            )
            kernel_names = [
                line.split()[-1]
                for line in result.stdout.splitlines()
                if line.startswith("STT_FUNC") and not check_illegal(line)
            ]
            assert len(kernel_names) == 1, (
                f"Too many kernels in the library: {kernel_names}"
            )

            # Load kernel from the library
            result, self.kernel = cbd.cuLibraryGetKernel(
                self.lib, bytes(kernel_names[0], encoding="utf-8")
            )
            assert result == cbd.CUresult.CUDA_SUCCESS, (
                f"Failed to load kernel: {result}"
            )

            end_time = time.time_ns()
            elapsed_time = (end_time - start_time) / 1e6
            if int(os.getenv("DG_JIT_DEBUG", 0)):
                print(
                    f"Loading JIT runtime {self.path} took {elapsed_time:.2f} ms."
                )

        # noinspection PyArgumentList
        return self.launch(self.kernel, kwargs)

    def __del__(self) -> None:
        if self.lib is not None:
            res = cbd.cuLibraryUnload(self.lib)[0]
            if res != cbd.CUresult.CUDA_SUCCESS:
                raise Exception(f"Failed to unload library {self.path}: {res}")


class RuntimeCache:
    def __init__(self) -> None:
        self.cache = {}

    def __setitem__(self, path: str, runtime: Runtime) -> None:
        self.cache[path] = runtime

    def get(
        self,
        path: str,
        runtime_cls: type[Runtime],
        name: str = "",
        kwargs: dict[str, Any] | None = None,
        force_enable_cache: bool = False,
    ) -> Runtime | None:
        # In Python runtime
        if path in self.cache:
            return self.cache[path]

        # Already compiled
        use_cache = force_enable_cache or not int(
            os.getenv("DG_JIT_DISABLE_CACHE", 0)
        )
        if use_cache and os.path.exists(path) and Runtime.is_path_valid(path):
            # Print heuristic for the first time
            if name and (
                int(os.getenv("DG_JIT_DEBUG", 0))
                or int(os.getenv("DG_PRINT_CONFIGS", 0))
            ):
                simplified_kwargs = {}
                for key, value in (
                    kwargs.items() if kwargs is not None else {}.items()
                ):
                    value = (
                        f"paddle.Tensor<{value.dtype}>"
                        if isinstance(value, paddle.Tensor)
                        else value
                    )
                    value = (
                        "cuda.bindings.driver.CUtensorMap"
                        if isinstance(value, cbd.CUtensorMap)
                        else value
                    )
                    simplified_kwargs[key] = value
                print(
                    f"Put kernel {name} with {simplified_kwargs} into runtime cache"
                )

            runtime = runtime_cls(path)
            self.cache[path] = runtime
            return runtime
        return None
