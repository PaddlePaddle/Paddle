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
import json

skip_list = []


def remove_grad_kernel(kernels):
    clean_kernels = []
    for kernel_ in kernels:
        if (not "_grad" in kernel_):
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
            if not ".cc" in file_name:
                continue
            with open(os.path.join(dirpath, file_name)) as f:
                txt = f.readlines()
                content = ""
                registry = False
                is_macro_defination = False
                for line in txt:
                    if line.strip().startswith("#define") and line.strip(
                    ).endswith("\\"):
                        is_macro_defination = True
                        continue
                    if is_macro_defination:
                        if not line.strip().endswith("\\"):
                            is_macro_defination = False
                        continue

                    if (register in line):
                        content = ""
                        registry = True
                    if (registry):
                        content += line
                    if (registry and ";" in line):
                        kernel_name = content.replace("\n", "").replace(
                            " ", "").strip(register).split(",")
                        registry = False
                        kernel_names.append(kernel_name[0])
    return remove_grad_kernel(kernel_names)


def show_kernel_statistics(backend, kernels):
    print("=== kernels statistics === ")
    print("the number of " + backend + " kernels is: " + str(len(kernels)) +
          "\n")
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
            if (registry_fun_found):
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
