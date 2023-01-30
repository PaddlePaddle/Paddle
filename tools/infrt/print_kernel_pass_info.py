# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import os
import re
<<<<<<< HEAD
=======
import json
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

skip_list = []


def remove_grad_kernel(kernels):
    clean_kernels = []
    for kernel_ in kernels:
<<<<<<< HEAD
        if "_grad" not in kernel_:
=======
        if (not "_grad" in kernel_):
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            clean_kernels.append(kernel_)
    return clean_kernels


CPU_KERNEL_REGISTER = "REGISTER_OP_CPU_KERNEL("
GPU_KERNEL_REGISTER = "REGISTER_OP_CUDA_KERNEL("
XPU_KERNEL_REGISTER = "REGISTER_OP_XPU_KERNEL("


def get_compat_kernels_info(register):
    kernels_info = {}
    kernel_names = []
    for dirpath, dirnames, filenames in os.walk("../../paddle/fluid/operators"):
        for file_name in filenames:
<<<<<<< HEAD
            if ".cc" not in file_name:
=======
            if not ".cc" in file_name:
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
                continue
            with open(os.path.join(dirpath, file_name)) as f:
                txt = f.readlines()
                content = ""
                registry = False
                is_macro_defination = False
                for line in txt:
                    if line.strip().startswith(
<<<<<<< HEAD
                        "#define"
                    ) and line.strip().endswith("\\"):
=======
                            "#define") and line.strip().endswith("\\"):
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
                        is_macro_defination = True
                        continue
                    if is_macro_defination:
                        if not line.strip().endswith("\\"):
                            is_macro_defination = False
                        continue

<<<<<<< HEAD
                    if register in line:
                        content = ""
                        registry = True
                    if registry:
                        content += line
                    if registry and ";" in line:
                        kernel_name = (
                            content.replace("\n", "")
                            .replace(" ", "")
                            .strip(register)
                            .split(",")
                        )
=======
                    if (register in line):
                        content = ""
                        registry = True
                    if (registry):
                        content += line
                    if (registry and ";" in line):
                        kernel_name = content.replace("\n", "").replace(
                            " ", "").strip(register).split(",")
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
                        registry = False
                        kernel_names.append(kernel_name[0])
    return remove_grad_kernel(kernel_names)


def show_kernel_statistics(backend, kernels):
    print("=== kernels statistics === ")
<<<<<<< HEAD
    print(
        "the number of " + backend + " kernels is: " + str(len(kernels)) + "\n"
    )
=======
    print("the number of " + backend + " kernels is: " + str(len(kernels)) +
          "\n")
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    print(kernels)
    print("\n")


def show_pass_statistics(backend, passes):
    print("=== Passes Statistics === ")
    print("The number of " + backend + " passes is: " + str(len(passes)) + "\n")
    print(passes)
    print("\n")


def get_passes_info(register):
    pass_registry_func = ""
    with open("../../paddle/fluid/inference/api/paddle_pass_builder.cc") as f:
        txt = f.readlines()
        stack = []
        registry_fun_found = False
        for line in txt:
            if line.strip().startswith("//"):
                continue
            if register in line:
                registry_fun_found = True
<<<<<<< HEAD
            if registry_fun_found:
=======
            if (registry_fun_found):
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
                pass_registry_func += line
            if registry_fun_found:
                for char in line:
                    if char == "{":
                        stack.append(char)
                    if char == "}":
                        stack.pop()
            if len(stack) == 0:
                registry_fun_found = False
        pass_list = re.findall("\"(.+?)_pass\"", pass_registry_func)
        return pass_list


if __name__ == "__main__":
    cpu_kernels = get_compat_kernels_info(CPU_KERNEL_REGISTER)
    gpu_kernels = get_compat_kernels_info(GPU_KERNEL_REGISTER)
    xpu_kernels = get_compat_kernels_info(XPU_KERNEL_REGISTER)
    show_kernel_statistics("CPU", cpu_kernels)
    show_kernel_statistics("GPU", gpu_kernels)
    show_kernel_statistics("XPU", xpu_kernels)

    cpu_passes = get_passes_info("CpuPassStrategy::CpuPassStrategy()")
    gpu_passes = get_passes_info("GpuPassStrategy::GpuPassStrategy()")
    show_pass_statistics("CPU", cpu_passes)
    show_pass_statistics("GPU", gpu_passes)
