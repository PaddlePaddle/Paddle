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

import functools
import hashlib
import os
import re
import subprocess
import time
import uuid
from typing import Any

import cuda.bindings
from cuda.bindings import nvrtc

from paddle.utils.cpp_extension.cpp_extension import CUDA_HOME

from . import interleave_ffma
from .runtime import Runtime, RuntimeCache

runtime_cache = RuntimeCache()


def hash_to_hex(s: str) -> str:
    """Transforms input into a hexadecimal hash string"""
    if isinstance(s, int):
        return f"{s & 0xFFFFFFFFFFFFFFFF:016x}"
    elif isinstance(s, str):
        md5 = hashlib.md5()
        md5.update(s.encode("utf-8"))
        return md5.hexdigest()[0:12]
    else:
        raise TypeError(f"Unsupported type for hashing: {type(s)}")


@functools.cache
def get_jit_include_dir() -> str:
    return f"{os.path.dirname(os.path.abspath(__file__))}"


@functools.cache
def get_deep_gemm_version() -> str:
    md5 = hashlib.md5()

    # Update include directories
    include_dir = f"{get_jit_include_dir()}/../../../../include/paddle/fluid/fp8/deep_gemm/include"
    assert os.path.exists(include_dir), (
        f"Cannot find GEMM include directory {include_dir}"
    )
    for filename in filter(
        lambda x: x.endswith(".cuh"), sorted(os.listdir(include_dir))
    ):
        with open(os.path.join(include_dir, filename), "rb") as f:
            md5.update(f.read())

    # Update `interleave_ffma.py`
    with open(
        os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "interleave_ffma.py"
        ),
        "rb",
    ) as f:
        md5.update(f.read())
    return md5.hexdigest()[0:12]


@functools.cache
def get_nvcc_compiler() -> tuple[str, str]:
    paths = []
    if os.getenv("DG_JIT_NVCC_COMPILER"):
        paths.append(os.getenv("DG_JIT_NVCC_COMPILER"))
    paths.append(os.path.join(CUDA_HOME, "bin", "nvcc"))

    # Try to find the first available NVCC compiler
    least_version_required = "12.3"
    version_pattern = re.compile(r"release (\d+\.\d+)")
    for path in paths:
        if os.path.exists(path):
            command = [path, "--version"]
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
            )
            match = version_pattern.search(result.stdout)
            version = match.group(1)
            assert match, f"Cannot get the version of NVCC compiler {path}"
            assert version >= least_version_required, (
                f"NVCC {path} version {version} is lower than {least_version_required}"
            )
            return path, version
    raise RuntimeError("Cannot find any available NVCC compiler")


@functools.cache
def get_default_user_dir():
    if "DG_JIT_CACHE_DIR" in os.environ:
        path = os.getenv("DG_JIT_CACHE_DIR")
        os.makedirs(path, exist_ok=True)
        return path
    return os.path.join(os.path.expanduser("~"), ".deep_gemm")


@functools.cache
def get_tmp_dir():
    return os.path.join(get_default_user_dir(), "tmp")


@functools.cache
def get_cache_dir():
    return os.path.join(get_default_user_dir(), "cache")


def make_tmp_dir():
    tmp_dir = get_tmp_dir()
    os.makedirs(tmp_dir, exist_ok=True)
    return tmp_dir


def put(path, data):
    # Write and do POSIX atomic replace
    tmp_file_path = os.path.join(
        make_tmp_dir(), f"file.tmp.{uuid.uuid4()!s}.{hash_to_hex(path)}"
    )
    with open(tmp_file_path, "wb" if isinstance(data, bytes) else "w") as f:
        f.write(data)
    os.replace(tmp_file_path, path)


class Compiler:
    @classmethod
    def signature(cls) -> str:
        pass

    @staticmethod
    def __version__() -> tuple[int, int]:
        pass

    @classmethod
    def compile(cls, name: str, code: str, target_path: str) -> None:
        pass

    @staticmethod
    def flags() -> list[str]:
        cpp_standard = int(os.getenv("DG_JIT_OVERRIDE_CPP_STANDARD", 20))
        return [
            f"-std=c++{cpp_standard}",
            "--ptxas-options=--register-usage-level=10"
            + (",--verbose" if "DG_JIT_PTXAS_VERBOSE" in os.environ else ""),
            # Suppress some unnecessary warnings, such as unused variables for certain `constexpr` branch cases
            "--diag-suppress=39,161,174,177,186,940",
        ]

    @staticmethod
    def include_dirs() -> list[str]:
        return [
            f"{get_jit_include_dir()}/../../../../include/paddle/fluid/fp8/deep_gemm/include"
        ]

    @classmethod
    def build(
        cls,
        name: str,
        runtime_cls: type[Runtime],
        kwargs: dict[str, Any] | None = None,
    ) -> Runtime:
        code = runtime_cls.generate(kwargs)
        # Compiler flags
        flags = cls.flags()

        # Build signature
        enable_sass_opt = cls.__version__() <= (12, 8) and not int(
            os.getenv('DG_JIT_DISABLE_FFMA_INTERLEAVE', 0)
        )
        signature = f'{name}$${get_deep_gemm_version()}$${cls.signature()}$${flags}$${enable_sass_opt}$${code}'
        name = f"kernel.{name}.{hash_to_hex(signature)}"
        path = os.path.join(get_cache_dir(), name)

        # Check runtime cache or file system hit
        global runtime_cache
        cached_runtime = runtime_cache.get(path, runtime_cls, name, kwargs)
        if cached_runtime is not None:
            if int(os.getenv("DG_JIT_DEBUG", 0)):
                print(f"Using cached JIT runtime {name} during build")
            return cached_runtime

        # Compile into a temporary CU file
        os.makedirs(path, exist_ok=True)
        cubin_path = os.path.join(path, "kernel.cubin")
        tmp_cubin_path = os.path.join(
            make_tmp_dir(),
            f"nvcc.tmp.{uuid.uuid4()!s}.{hash_to_hex(cubin_path)}.cubin",
        )

        start_time = time.time()
        cls.compile(name, code, tmp_cubin_path)
        end_time = time.time()
        elapsed_time = end_time - start_time
        if int(os.getenv("DG_JIT_DEBUG", 0)):
            print(
                f"Compilation of JIT runtime {name} took {elapsed_time:.2f} seconds."
            )

        # Interleave FFMA reuse
        if enable_sass_opt:
            interleave_ffma.process(tmp_cubin_path)

        # Atomic replace files
        os.replace(tmp_cubin_path, cubin_path)

        # Put cache and return
        runtime = runtime_cache.get(
            path, runtime_cls, name, kwargs, force_enable_cache=True
        )
        assert runtime is not None
        return runtime


class NVCCCompiler(Compiler):
    @staticmethod
    def __version__() -> tuple[int, int]:
        _, version = get_nvcc_compiler()
        major, minor = map(int, version.split("."))
        return major, minor

    @classmethod
    def signature(cls) -> str:
        return f"{get_nvcc_compiler()[0]}+{cls.__version__()}"

    @classmethod
    def flags(cls) -> list[str]:
        cxx_flags = [
            "-fPIC",
            "-O3",
            "-fconcepts",
            "-Wno-deprecated-declarations",
            "-Wno-abi",
        ]
        return [
            *super().flags(),
            *[f"-I{d}" for d in cls.include_dirs()],
            "-gencode=arch=compute_90a,code=sm_90a",
            "-cubin",
            "-O3",
            "--expt-relaxed-constexpr",
            "--expt-extended-lambda",
            f"--compiler-options={','.join(cxx_flags)}",
        ]

    @classmethod
    def compile(cls, name: str, code: str, target_path: str) -> None:
        # Write the code
        path = os.path.join(get_cache_dir(), name)
        src_path = os.path.join(path, "kernel.cu")
        put(src_path, code)
        command = [
            get_nvcc_compiler()[0],
            src_path,
            "-o",
            target_path,
            *cls.flags(),
        ]
        if int(os.getenv("DG_JIT_DEBUG", 0)) or int(
            os.getenv("DG_JIT_PRINT_COMPILER_COMMAND", 0)
        ):
            print(f"Compiling JIT runtime {name} with command {command}")

        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            print(
                f"NVCC compilation failed: stdout: {result.stdout}, stderr: {result.stderr}"
            )
            raise AssertionError(f"Failed to compile {src_path}")


class NVRTCCompiler(Compiler):
    @staticmethod
    def __version__() -> tuple[int, int]:
        res, major, minor = nvrtc.nvrtcVersion()
        if res != nvrtc.nvrtcResult.NVRTC_SUCCESS:
            # Failed to get the actual NVRTC version, use cuda-bindings version instead
            major, minor = map(int, cuda.bindings.__version__.split(".")[:2])
        return major, minor

    @classmethod
    def signature(cls) -> str:
        return f"nvrtc+{cls.__version__()}"

    @staticmethod
    def include_dirs() -> list[str]:
        if CUDA_HOME is None:
            raise RuntimeError("CUDA_HOME is required for NVRTC compilation")
        return [get_jit_include_dir(), os.path.join(CUDA_HOME, "include")]

    @classmethod
    def flags(cls) -> list[str]:
        flags = [
            *super().flags(),
            *[f"-I{d}" for d in cls.include_dirs()],
            "--gpu-architecture=sm_90a",
            "-default-device",
        ]
        # NOTES: PCH is vital for compilation speed
        if cls.__version__() >= (12, 8):
            flags += ["--pch"]
            if int(os.getenv("DG_JIT_DEBUG", 0)):
                flags += ["--pch-verbose=true"]
        return flags

    @classmethod
    def compile(cls, name: str, code: str, target_path: str) -> None:
        # Create program
        code_bytes = bytes(code, "utf-8")
        result, program = nvrtc.nvrtcCreateProgram(
            code_bytes, bytes(name, "utf-8"), 0, [], []
        )
        assert result == nvrtc.nvrtcResult.NVRTC_SUCCESS, (
            f"Failed to create program: {result}"
        )

        # Compile
        options = [bytes(flag, "utf-8") for flag in cls.flags()]
        if int(os.getenv("DG_JIT_DEBUG", 0)) or int(
            os.getenv("DG_JIT_PRINT_COMPILER_COMMAND", 0)
        ):
            print(f"Compiling JIT runtime {name} with options: {options}")
        compile_result = nvrtc.nvrtcCompileProgram(
            program, len(options), options
        )[0]

        # Print compiler log
        if (
            int(os.getenv("DG_JIT_DEBUG", 0))
            or compile_result != nvrtc.nvrtcResult.NVRTC_SUCCESS
        ):
            result, log_size = nvrtc.nvrtcGetProgramLogSize(program)
            assert result == nvrtc.nvrtcResult.NVRTC_SUCCESS, (
                f"Failed to get program log size: {result}"
            )

            log_bytes = bytes(log_size)
            result = nvrtc.nvrtcGetProgramLog(program, log_bytes)[0]
            assert result == nvrtc.nvrtcResult.NVRTC_SUCCESS, (
                f"Failed to get program log: {result}"
            )
            print(f"Compiler log: {log_bytes.decode('utf-8')}")

        # Exit if failed
        assert compile_result == nvrtc.nvrtcResult.NVRTC_SUCCESS, (
            f"Failed to compile program: {compile_result}"
        )

        # Create CUBIN
        result, cubin_size = nvrtc.nvrtcGetCUBINSize(program)
        assert result == nvrtc.nvrtcResult.NVRTC_SUCCESS, (
            f"Failed to get CUBIN size: {result}"
        )
        cubin_bytes = bytes(cubin_size)
        result = nvrtc.nvrtcGetCUBIN(program, cubin_bytes)[0]
        assert result == nvrtc.nvrtcResult.NVRTC_SUCCESS, (
            f"Failed to get CUBIN: {result}"
        )

        # Write into the file system
        put(target_path, cubin_bytes)

        # Destroy handler
        assert (
            nvrtc.nvrtcDestroyProgram(program)[0]
            == nvrtc.nvrtcResult.NVRTC_SUCCESS
        ), f"Failed to destroy program: {result}"


def build(
    name: str, runtime_cls: type[Runtime], kwargs: dict[str, Any] | None = None
) -> Runtime:
    compiler_cls = (
        NVRTCCompiler if int(os.getenv("DG_JIT_USE_NVRTC", 0)) else NVCCCompiler
    )
    return compiler_cls.build(name, runtime_cls, kwargs)
