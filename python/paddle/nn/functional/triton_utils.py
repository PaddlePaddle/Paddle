# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

import re

import triton

import paddle


def SubstituteTemplate(template, values):
    text = template
    changed = True
    while changed:
        changed = False
        for key, value in values.items():
            regex = "\\$\\{%s\\}" % key
            newtext = re.sub(regex, value, text)
            if newtext != text:
                changed = True
            text = newtext
    return text


def find_so_path(generated_dir, python_package_name):
    import os

    so_path = []
    for root, dirs, files in os.walk(generated_dir):
        for file in files:
            if file.endswith(python_package_name + ".so"):
                so_path.append(os.path.join(root, file))
    if len(so_path) == 0:
        return None
    else:
        assert len(so_path) == 1
        return so_path[0]


def multi_process_do(commands):
    THREADS = 80
    import multiprocessing
    import os

    process = []

    def one_process_work(commands, thread_id):
        i = thread_id
        while i < len(commands):
            re = os.system(commands[i])
            assert re == 0
            i += THREADS

    for i in range(THREADS):
        p = multiprocessing.Process(target=one_process_work, args=(commands, i))
        process.append(p)
    for p in process:
        p.start()
    for p in process:
        p.join()


def extract_triton_kernel(kernel, file_name):
    import inspect
    import re
    import textwrap

    if type(kernel) == triton.runtime.jit.JITFunction:
        fn = kernel.fn
    elif type(kernel) == triton.runtime.autotuner.Autotuner:
        fn = kernel.fn.fn
    else:
        AssertionError("error occures")
    py_script = textwrap.dedent(inspect.getsource(fn))

    # @triton.jit must only appear once
    assert len(re.findall("@triton.jit", py_script)) == 1

    py_script = py_script[py_script.find("@triton.jit") :]
    py_script = "import triton\nimport triton.language as tl\n" + py_script

    py_script = py_script.replace("if bias_ptr is not None", "if bias_ptr")

    with open(file_name, "w") as f:
        f.write(py_script)
        f.close()


template_install = """

import os
generated_cu = []
for root, dirs, files in os.walk("./"):
    for file in files:
        if file.endswith(".c") or file.endswith(".cu"):
            generated_cu.append(os.path.join(root, file))


import paddle
from paddle.utils.cpp_extension import CUDAExtension, setup


def get_gencode_flags():
    prop = paddle.device.cuda.get_device_properties()
    cc = prop.major * 10 + prop.minor
    return ["-gencode", "arch=compute_{{0}},code=sm_{{0}}".format(cc)]


gencode_flags = get_gencode_flags()



setup(
    name="{python_package_name}",
    ext_modules=CUDAExtension(
        sources = generated_cu,
        extra_compile_args={{
            "cc": ["-lcuda"],
            "nvcc": [
                "-O3",
                "-U__CUDA_NO_HALF_OPERATORS__",
                "-U__CUDA_NO_HALF_CONVERSIONS__",
                "-U__CUDA_NO_BFLOAT16_OPERATORS__",
                "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                "-U__CUDA_NO_BFLOAT162_OPERATORS__",
                "-U__CUDA_NO_BFLOAT162_CONVERSIONS__",
            ]
            + gencode_flags,
        }},
        extra_link_args = ["-lcuda"]
    ),
)
"""


def get_value_hint(x):
    if x % 16 == 0:
        return "i32:16"
    elif x % 8 == 0:
        return "i32:8"
    elif x % 4 == 0:
        return "i32:4"
    elif x % 2 == 0:
        return "i32:2"
    elif x == 1:
        return "i32:1"
    else:
        return "i32"


def build_package(generated_dir, python_package_name):
    import os
    import sys

    setup_file_path = generated_dir + "/setup_cuda.py"
    python_path = sys.executable
    with open(setup_file_path, "w") as f:
        f.write(
            template_install.format(python_package_name=python_package_name)
        )
        f.close()
    install_command = f"cd {generated_dir} && {python_path} setup_cuda.py build"
    re = os.system(install_command)
    assert re == 0


def rename_c_to_cu(generated_dir):
    import os

    # rename the .c file to .cu
    for filename in os.listdir(generated_dir):
        if filename.endswith(".c"):
            old_path = os.path.join(generated_dir, filename)
            new_path = os.path.join(generated_dir, filename + "u")
            os.rename(old_path, new_path)


def get_pointer_hint(tensor):
    if tensor.dtype == paddle.float16:
        return "*fp16:16"
    elif tensor.dtype == paddle.uint8:
        return "*u8:16"
    elif tensor.dtype == paddle.int8:
        return "*i8:16"
    elif tensor.dtype == paddle.float32:
        return "*fp32:16"
