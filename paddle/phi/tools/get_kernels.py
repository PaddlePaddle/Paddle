# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import pandas as pd

base_path = "/work/Paddle3/paddle/phi/kernels"
sr_path = "/work/Paddle/paddle/phi/kernels/selected_rows"
sparse_path = "/work/Paddle/paddle/phi/kernels/sparse"
string_path = "/work/Paddle/paddle/phi/kernels/strings"
# root_path = "/work/download/get_kernels/include"

kernel_signature_pattern = re.compile(
    r'template <[a-zA-Z\s\,]*> ?\n?void \w+\([^\{\}\(\)]+\)[;+ *\n?| *{?]',
    flags=re.DOTALL,
)
register_kernel_pattern = re.compile(
    r'(PD_REGISTER_KERNEL|PD_REGISTER_GENERAL_KERNEL)\([ \t\r\n]*([a-z0-9_]*)\,[[ \\\t\r\n\/]*[a-z0-9_]*]?[ \\\t\r\n]*[a-zA-Z]*\,[ \\\t\r\n]*[A-Z_]*\,[ \t\r\n]*([a-z0-9_A-Z::]*)'
)
kernel_signature = []
complement_kernel_signature = []
register_kernel_name = {}
complement_register_kernel_name = {}
result = {}


def deal_with_complement_kernels(head, source):
    with open(head, 'r') as file:
        content = file.read()
        match_result = kernel_signature_pattern.findall(content)
        kernel_signature.extend(match_result)

    with open(source, 'r') as file:
        content = file.read()
        match_result = register_kernel_pattern.findall(content)
        for each_result in match_result:
            register_kernel_name[each_result[1]] = each_result[2].split("::")[
                -1
            ]


def get_kernel_signature(path):
    files = os.listdir(path)
    for f in files:
        f_path = os.path.join(path, f)
        # if f_path.find("activation_kernel") > -1 or \
        #     f_path.find("activation_grad_kernel") > -1 or \
        #     f_path.find("cast_kernel") > -1 or \
        #     f_path.find("cast_grad_kernel") > -1 or \
        #     f_path.find("compare_kernel") > -1 or \
        #     f_path.find("logical_kernel") > -1:
        #     continue
        if f_path.endswith("kernel.h"):
            with open(f_path, 'r') as file:
                content = file.read()
                match_result = kernel_signature_pattern.findall(content)
                kernel_signature.extend(match_result)


def get_register_kernel_name(path):
    files = os.listdir(path)
    for f in files:
        f_path = os.path.join(path, f)
        # if (f_path.find("activation_kernel") > -1  and not f_path.find("kernels/activation_kernel.cc") > -1 and not f_path.find("kernels/selected_rows/activation_kernel.cc") > -1)or \
        #     (f_path.find("activation_grad_kernel") > -1  and not f_path.find("kernels/activation_grad_kernel.cc") > -1 and not f_path.find("kernels/selected_rows/activation_kernel.cc") > -1) or \
        #     f_path.find("cast_kernel") > -1 or \
        #     f_path.find("cast_grad_kernel") > -1 or \
        #     f_path.find("compare_kernel") > -1 or \
        #     f_path.find("logical_kernel") > -1:
        #     continue

        if f_path.endswith("kernel.cc") or f_path.endswith("kernel.cu"):
            with open(f_path, 'r') as file:
                content = file.read()
                match_result = register_kernel_pattern.findall(content)
                for each_result in match_result:
                    register_kernel_name[each_result[1]] = each_result[2].split(
                        "::"
                    )[-1]


def kernel_summary(register_kernel_name, kernel_signature):
    for register_name, register_kernel in register_kernel_name.items():
        for signature in kernel_signature:
            if signature.split("(")[0].split(" ")[-1] == register_kernel:
                result[register_name] = signature
                break


def get_cudnn_kernel_signature_and_name(path):
    files = os.listdir(path)
    for f in files:
        f_path = os.path.join(path, f)
        if f_path.endswith("kernel.cu"):
            with open(f_path, 'r') as file:
                content = file.read()
                cudnn_kernel_signature_pattern = re.compile(
                    r'template <[a-zA-Z\s\,]*> ?\n?void \w+\([^\{\}\(\)]+\) {+',
                    flags=re.DOTALL,
                )
                match_result = cudnn_kernel_signature_pattern.findall(content)
                kernel_signature.extend(match_result)

                match_result = register_kernel_pattern.findall(content)
                for each_result in match_result:
                    register_kernel_name[each_result[1]] = each_result[2].split(
                        "::"
                    )[-1]


# For base kernel
get_kernel_signature(base_path)
get_register_kernel_name(base_path)
get_register_kernel_name(base_path + "/cpu")
get_register_kernel_name(base_path + "/gpu")
deal_with_complement_kernels(
    "/work/download/get_kernels/include/activation_kernel_expand.h",
    "/work/download/get_kernels/include/activation_kernel_expand.cc",
)
deal_with_complement_kernels(
    "/work/download/get_kernels/include/activation_grad_kernel_expand.h",
    "/work/download/get_kernels/include/activation_grad_kernel_expand.cc",
)
deal_with_complement_kernels(
    "/work/download/get_kernels/include/cast_kernel.h",
    "/work/download/get_kernels/include/cast_kernel.cc",
)
deal_with_complement_kernels(
    "/work/download/get_kernels/include/cast_grad_kernel.h",
    "/work/download/get_kernels/include/cast_grad_kernel.cc",
)
deal_with_complement_kernels(
    "/work/download/get_kernels/include/logical_kernel_expand.h",
    "/work/download/get_kernels/include/logical_kernel_expand.cc",
)
deal_with_complement_kernels(
    "/work/download/get_kernels/include/compare_kernel_expand.h",
    "/work/download/get_kernels/include/compare_kernel_expand.cu",
)
deal_with_complement_kernels(
    "/work/download/get_kernels/include/isfinite_kernel_expand.h",
    "/work/download/get_kernels/include/isfinite_kernel.cc",
)

kernel_summary(register_kernel_name, kernel_signature)

# For sparse kernel
register_kernel_name.clear()
kernel_signature.clear()
get_kernel_signature(sparse_path)
get_register_kernel_name(sparse_path)
get_register_kernel_name(sparse_path + "/cpu")
get_register_kernel_name(sparse_path + "/gpu")
deal_with_complement_kernels(
    "/work/download/get_kernels/include/elementwise_kernel_expand.h",
    "/work/download/get_kernels/include/elementwise_kernel.cc",
)
deal_with_complement_kernels(
    "/work/download/get_kernels/include/elementwise_grad_kernel_expand.h",
    "/work/download/get_kernels/include/elementwise_grad_kernel.cc",
)
deal_with_complement_kernels(
    "/work/download/get_kernels/include/unary_kernel_expand.h",
    "/work/download/get_kernels/include/unary_kernel_expand.cc",
)
deal_with_complement_kernels(
    "/work/download/get_kernels/include/unary_grad_kernel_expand.h",
    "/work/download/get_kernels/include/unary_grad_kernel_expand.cc",
)
kernel_summary(register_kernel_name, kernel_signature)

# For selected_rows kernel
register_kernel_name.clear()
kernel_signature.clear()
get_kernel_signature(sr_path)
get_register_kernel_name(sr_path)
get_register_kernel_name(sr_path + "/cpu")
get_register_kernel_name(sr_path + "/gpu")
deal_with_complement_kernels(
    "/work/download/get_kernels/include/sr_isfinite_kernel_expand.h",
    "/work/download/get_kernels/include/sr_isfinite_kernel.cc",
)
kernel_summary(register_kernel_name, kernel_signature)

# For strings kernel
register_kernel_name.clear()
kernel_signature.clear()
get_kernel_signature(string_path)
get_register_kernel_name(string_path)
get_register_kernel_name(string_path + "/cpu")
get_register_kernel_name(string_path + "/gpu")
kernel_summary(register_kernel_name, kernel_signature)

# For gpudnn kernel
register_kernel_name.clear()
kernel_signature.clear()
get_cudnn_kernel_signature_and_name(base_path + "/gpudnn")
kernel_summary(register_kernel_name, kernel_signature)

data_frame_data = {}
data_frame_data["kernel_name"] = [k for k, v in result.items()]
data_frame_data["kernel_signature"] = [v.strip() for k, v in result.items()]
df = pd.DataFrame(data_frame_data)
df.to_excel('/work/download/get_kernels/all_kernel_signature.xls')
# df.to_csv('/work/download/get_kernels/all_kernel_signature.csv')
print("success!")
