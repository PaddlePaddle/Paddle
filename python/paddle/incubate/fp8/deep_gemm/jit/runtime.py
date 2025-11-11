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

import ctypes
import enum
import os
import subprocess
import time
from typing import Any

import cuda.bindings.driver as cbd

import paddle
from paddle.utils.cpp_extension.cpp_extension import CUDA_HOME


def get_num_math_warpgroups(block_m: int) -> int:
    return 1 if block_m == 64 else 2


def get_num_threads_per_sm(
    num_tma_threads: int, num_math_threads_per_group: int, block_m: int
) -> int:
    assert num_math_threads_per_group == 128, (
        "Only support 128 threads per math group"
    )
    return (
        get_num_math_warpgroups(block_m) * num_math_threads_per_group
        + num_tma_threads
    )


class GemmType(enum.Enum):
    Normal = 0
    GroupedContiguous = 1
    GroupedMasked = 2

    def __str__(self) -> str:
        return {
            0: "Normal",
            1: "GroupedContiguous",
            2: "GroupedMasked",
        }[self.value]


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


def get_cache_key(kwargs, num_tma_threads, num_math_threads_per_group):
    key_params = {
        "NUM_TMA_MULTICAST": kwargs["NUM_TMA_MULTICAST"],
        "NUM_SMS": kwargs["NUM_SMS"],
        "BLOCK_M": kwargs["BLOCK_M"],
        "SMEM_SIZE": kwargs["SMEM_SIZE"],
        "STREAM": kwargs["STREAM"],
        "num_tma_threads": num_tma_threads,
        "num_math_threads_per_group": num_math_threads_per_group,
    }
    return hash(frozenset(key_params.items()))


class KernelLaunchCache:
    def __init__(self):
        self.config_cache = {}
        self.attr_cache = {}

    @staticmethod
    def create_attr(kwargs):
        """Creates and caches property objects"""
        attr_val = cbd.CUlaunchAttributeValue()
        attr_val.clusterDim.x = kwargs["NUM_TMA_MULTICAST"]
        attr_val.clusterDim.y = 1
        attr_val.clusterDim.z = 1
        attr = cbd.CUlaunchAttribute()
        attr.id = cbd.CUlaunchAttributeID.CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION
        attr.value = attr_val
        return attr

    @staticmethod
    def create_config(
        kwargs, num_tma_threads, num_math_threads_per_group, attr
    ):
        """Creates and caches configuration objects"""
        config = cbd.CUlaunchConfig()
        config.numAttrs = 1
        config.attrs = [attr]
        config.gridDimX = kwargs["NUM_SMS"]
        config.gridDimY = 1
        config.gridDimZ = 1
        config.blockDimX = get_num_threads_per_sm(
            num_tma_threads, num_math_threads_per_group, kwargs["BLOCK_M"]
        )
        config.blockDimY = 1
        config.blockDimZ = 1
        config.sharedMemBytes = kwargs["SMEM_SIZE"]
        config.hStream = kwargs["STREAM"]
        return config

    def get_launch_config(
        self, kwargs, num_tma_threads, num_math_threads_per_group
    ):
        """Retrieves cached config or creates new instance"""
        cache_key = get_cache_key(
            kwargs, num_tma_threads, num_math_threads_per_group
        )

        if cache_key not in self.config_cache:
            # 先检查属性缓存
            attr_key = (kwargs["NUM_TMA_MULTICAST"],)
            if attr_key not in self.attr_cache:
                self.attr_cache[attr_key] = self.create_attr(kwargs)

            # 创建新配置
            config = self.create_config(
                kwargs,
                num_tma_threads,
                num_math_threads_per_group,
                self.attr_cache[attr_key],
            )
            self.config_cache[cache_key] = config

        return self.config_cache[cache_key]


launch_cache = KernelLaunchCache()


class FP8GemmRuntime(Runtime):
    def __init__(self, path: str) -> None:
        super().__init__(path)

    @staticmethod
    def generate(kwargs: dict[str, Any]) -> str:
        code = f"""
#ifdef __CUDACC_RTC__
#include <deep_gemm/nvrtc_std.cuh>
#else
#include <cuda.h>
#include <string>
#endif

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include <deep_gemm/fp8_gemm.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&fp8_gemm_kernel<
        {kwargs['N']},
        {kwargs['K']},
        {kwargs['BLOCK_M']},
        {kwargs['BLOCK_N']},
        {kwargs['BLOCK_K']},
        {kwargs['BLOCK_N_PADDING']},
        {kwargs['SWIZZLE_D_MODE']},
        {kwargs['NUM_GROUPS']},
        {kwargs['NUM_STAGES']},
        {kwargs['NUM_TMA_THREADS']},
        {kwargs['NUM_MATH_THREADS_PER_GROUP']},
        {kwargs['NUM_TMA_MULTICAST']},
        {'true' if kwargs['IS_TMA_MULTICAST_ON_A'] else 'false'},
        GemmType::{kwargs['GEMM_TYPE']}
      >);
}};
"""
        if int(os.getenv("DG_JIT_DEBUG", 0)):
            print(f"Generated FP8 GEMM code:\n{code}")
        return code

    # noinspection PyMethodOverriding
    @staticmethod
    def launch(kernel: cbd.CUkernel, kwargs: dict[str, Any]) -> cbd.CUresult:
        num_tma_threads = 128
        num_math_threads_per_group = 128

        result = cbd.cuKernelSetAttribute(
            cbd.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
            kwargs["SMEM_SIZE"],
            kernel,
            cbd.CUdevice(kwargs["DEVICE_INDEX"]),
        )[0]
        assert result == cbd.CUresult.CUDA_SUCCESS, (
            f"Failed to set max dynamic shared memory size: {result}"
        )
        config = launch_cache.get_launch_config(
            kwargs, num_tma_threads, num_math_threads_per_group
        )

        arg_values = (
            kwargs["SCALES_B"].data_ptr(),
            kwargs["GROUPED_LAYOUT"].data_ptr(),
            kwargs["M"],
            kwargs["TENSOR_MAP_A"],
            kwargs["TENSOR_MAP_B"],
            kwargs["TENSOR_MAP_SCALES_A"],
            kwargs["TENSOR_MAP_D"],
        )
        arg_types = (
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_uint32,
            None,
            None,
            None,
            None,
        )
        ret = cbd.cuLaunchKernelEx(config, kernel, (arg_values, arg_types), 0)
        return ret


class FP8WGradGemmRuntime(Runtime):
    def __init__(self, path: str) -> None:
        super().__init__(path)

    @staticmethod
    def generate(kwargs: dict[str, Any]) -> str:
        code = f"""
#ifdef __CUDACC_RTC__
#include <deep_gemm/nvrtc_std.cuh>
#else
#include <cuda.h>
#include <string>
#endif

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include <deep_gemm/fp8_wgrad_gemm.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&fp8_wgrad_gemm_kernel<
        {kwargs['M']},
        {kwargs['N']},
        {kwargs['BLOCK_M']},
        {kwargs['BLOCK_N']},
        {kwargs['BLOCK_K']},
        {kwargs['NUM_STAGES']},
        {kwargs['NUM_LAST_STAGES']},
        {kwargs['NUM_TMA_THREADS']},
        {kwargs['NUM_MATH_THREADS_PER_GROUP']},
        {kwargs['NUM_TMA_MULTICAST']},
        {'true' if kwargs['IS_TMA_MULTICAST_ON_A'] else 'false'}
      >);
}};
"""
        if int(os.getenv("DG_JIT_DEBUG", 0)):
            print(f"Generated FP8 WGrad GEMM code:\n{code}")
        return code

    # noinspection PyMethodOverriding
    @staticmethod
    def launch(kernel: cbd.CUkernel, kwargs: dict[str, Any]) -> cbd.CUresult:
        num_tma_threads = 128
        num_math_threads_per_group = 128

        result = cbd.cuKernelSetAttribute(
            cbd.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
            kwargs["SMEM_SIZE"],
            kernel,
            cbd.CUdevice(kwargs["DEVICE_INDEX"]),
        )[0]
        assert result == cbd.CUresult.CUDA_SUCCESS, (
            f"Failed to set max dynamic shared memory size: {result}"
        )
        config = launch_cache.get_launch_config(
            kwargs, num_tma_threads, num_math_threads_per_group
        )

        arg_values = (
            kwargs["K"],
            kwargs["TENSOR_MAP_A"],
            kwargs["TENSOR_MAP_B"],
            kwargs["TENSOR_MAP_SCALES_A"],
            kwargs["TENSOR_MAP_SCALES_B"],
            kwargs["TENSOR_MAP_D"],
        )
        arg_types = (
            ctypes.c_uint32,
            None,
            None,
            None,
            None,
            None,
        )
        return cbd.cuLaunchKernelEx(config, kernel, (arg_values, arg_types), 0)
