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

import copy
import ctypes
import os
from collections.abc import Iterable
from typing import Any

import paddle
from paddle import Tensor

# Name map for Python `eval`
typename_map: dict[Any, str] = {
    **{t: t.__name__ for t in (bool, int, float)},
    paddle.int32: "paddle.int32",
    paddle.float32: "paddle.float32",
    paddle.bfloat16: "paddle.bfloat16",
    paddle.float8_e4m3fn: "paddle.float8_e4m3fn",
    paddle.device.cuda.Stream: "paddle.device.cuda.Stream",
}
# `ctype` map for Python casting
ctype_map: dict[Any, Any] = {
    **{t: getattr(ctypes, f"c_{t.__name__}") for t in (bool, int, float)},
    **dict.fromkeys(
        (
            paddle.int32,
            paddle.float32,
            paddle.bfloat16,
            paddle.float8_e4m3fn,
            paddle.device.cuda.Stream,
        ),
        ctypes.c_void_p,
    ),
}


# Type map for both Python API and source code usages
genc_map = {
    bool: ("bool", "bool"),
    int: ("int", "int"),
    float: ("float", "float"),
    paddle.int32: ("void*", "int*"),
    paddle.float32: ("void*", "float*"),
    paddle.bfloat16: ("void*", "__nv_bfloat16*"),
    paddle.float8_e4m3fn: ("void*", "__nv_fp8_e4m3*"),
    paddle.device.cuda.Stream: ("void*", "cudaStream_t"),
}


def map_ctype(value: Any) -> Any:
    ctype = ctype_map[value.dtype if isinstance(value, Tensor) else type(value)]
    if isinstance(value, Tensor):
        return ctype(value.data_ptr())
    if isinstance(value, paddle.device.cuda.Stream):
        return ctype(value.cuda_stream)
    return ctype(value)


def cpp_format(template: str, keys: dict[str, Any]) -> str:
    # We don't use `str.format` because it's not safe for C++ {} braces
    new_template = copy.deepcopy(template)
    for key, value in keys.items():
        new_template = new_template.replace(f"{{{key}}}", f"{value}")
    return new_template


def generate(
    includes: Iterable[str], arg_defs: Iterable[tuple], body: str
) -> str:
    # Common prefix
    code = "// DeepGEMM auto-generated JIT CUDA source file\n\n"

    # Includes
    preload_sys_includes = [
        "<cuda.h>",
        "<cuda_fp8.h>",
        "<cuda_runtime.h>",
        "<iostream>",
    ]
    paddle_source_dir = os.path.dirname(paddle.__file__)
    include_dirs = (
        paddle_source_dir
        + "/include/paddle/fluid/fp8/deep_gemm/include/cutlass/cutlass.h"
    )
    preload_package_includes = [f'"{include_dirs}"']

    assert isinstance(
        includes, (list, tuple)
    ), "includes must be a list or tuple"
    sys_includes = sorted(
        set(
            preload_sys_includes
            + [include for include in includes if include.startswith("<")]
        )
    )
    package_includes = sorted(
        set(
            preload_package_includes
            + [include for include in includes if include.startswith('"')]
        )
    )
    code += (
        "\n".join(f"#include {include}" for include in sys_includes) + "\n\n"
    )
    code += (
        "\n".join(f"#include {include}" for include in package_includes)
        + "\n\n"
    )

    # Function signature
    raw = "__raw_"
    get_def = (
        lambda n, t: f"{genc_map[t][0]} "
        + (raw if genc_map[t][0] != genc_map[t][1] else "")
        + n
    )
    code += 'extern "C" void launch('
    code += ", ".join(
        [get_def(*arg_def) for arg_def in arg_defs]
        + [
            "int& __return_code",
        ]
    )
    code += ") {\n"

    # Cast raw types
    code += "    // Cast raw types (if needed)\n"
    for arg_name, arg_type in arg_defs:
        if genc_map[arg_type][0] != genc_map[arg_type][1]:
            code += f"    auto {arg_name} = reinterpret_cast<{genc_map[arg_type][1]}>({raw}{arg_name});\n"

    # Function body
    code += "\n".join(
        [(("    " if line else "") + line) for line in body.split("\n")]
    )

    # End the function
    code += "}\n\n"

    # Debug print
    if os.getenv("DG_JIT_DEBUG", None):
        print(f"Generated code:\n{code}")

    return code
