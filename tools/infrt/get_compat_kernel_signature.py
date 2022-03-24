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


def parse_compat_registry(kernel_info):
    name, inputs_str, attrs_str, outputs_str = kernel_info.split(",{")
    kernel_info = {}
    kernel_info["inputs"] = inputs_str[:-1].split(",")
    kernel_info["attrs"] = attrs_str[:-1].split(",")
    kernel_info["outputs"] = outputs_str[:-1].split(",")
    return name, kernel_info


def remove_grad_registry(kernels_registry):
    clean_kernel_registry = {}
    for registry in kernels_registry:
        if (not "_grad" in registry):
            clean_kernel_registry[registry] = kernels_registry[registry]
    return clean_kernel_registry


def get_compat_kernels_info():
    kernels_info = {}
    compat_files = os.listdir("../../paddle/phi/ops/compat")
    for file_ in compat_files:
        if not ".cc" in file_:
            compat_files.remove(file_)

    for file_ in compat_files:
        if file_ in skip_list:
            continue
        with open("../../paddle/phi/ops/compat/" + file_) as in_file:
            txt = in_file.readlines()
            content = ""
            registry = False
            for line in txt:
                if ("KernelSignature(" in line):
                    content = ""
                    registry = True
                if (registry):
                    content += line
                if (registry and ";" in line):
                    data = content.replace("\n", "").replace(
                        " ",
                        "").strip("return").strip("KernelSignature(").strip(
                            "\);").replace("\"", "").replace("\\", "")
                    registry = False
                    name, registry_info = parse_compat_registry(data)

                    if name in kernels_info:
                        cur_reg = kernels_info[name]
                        kernels_info[name]["inputs"] = list(
                            set(registry_info["inputs"] + kernels_info[name][
                                "inputs"]))
                        kernels_info[name]["attrs"] = list(
                            set(registry_info["attrs"] + kernels_info[name][
                                "attrs"]))
                        kernels_info[name]["outputs"] = list(
                            set(registry_info["outputs"] + kernels_info[name][
                                "outputs"]))
                    else:
                        kernels_info[name] = registry_info

    compat_registry_ = remove_grad_registry(kernels_info)
    return compat_registry_
