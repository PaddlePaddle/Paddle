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

import json
import yaml
import sys
import os
from get_compat_kernel_signature import get_compat_kernels_info

#TODO @DannyIsFunny: more attr types need to be supported.
attr_type_converter = {
    "i": 'SI32Attr',
    "b": 'BoolAttr',
    "l": 'SI64Attr',
    "f": 'F32Attr',
    "NSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE": 'StrAttr',
    "St6vectorIiSaIiEE": 'I32ArrayAttr'
}

target_type_converter = {"CPU": "CPU", "GPU": "GPU", "Undefined": "UNK"}
layout_type_converter = {
    "NCHW": "NCHW",
    "NHWC": "NHWC",
    "Undefined(AnyLayout)": "ANY"
}
precision_type_converter = {
    "uint8": "UINT8",
    "int8": "INT8",
    "int16": "INT16",
    "int32": "INT32",
    "int64": "INT64",
    "float16": "FLOAT16",
    "bfloat16": "BFLOAT16",
    "float32": "FLOAT32",
    "float64": "FLOAT64",
    "complex64": "COMPLEX64",
    "complex128": "COMPLEX128",
    "bool": "BOOL",
    "Undefined": "UNK"
}

kernel_types_info_file = "./kernels.json"
kernel_signature_info_file = "./kernel_signature.json"

skipped_phi_api_list_file = "./skipped_phi_api.json"


def get_skipped_kernel_list():
    skiped_kernel_list = []
    with open(skipped_phi_api_list_file, 'r') as f:
        skiped_api_list = json.load(f)
    infer_meta_data = get_api_yaml_info("../../")
    for api in infer_meta_data:
        if "kernel" not in api or "infer_meta" not in api:
            continue
        if api["api"] in skiped_api_list["phi_apis"]:
            skiped_kernel_list.append(api["kernel"]["func"])
    skiped_kernel_list += skiped_api_list["phi_kernels"]
    return skiped_kernel_list


def get_api_yaml_info(file_path):
    f = open(file_path + "/python/paddle/utils/code_gen/api.yaml", "r")
    cont = f.read()
    return yaml.load(cont, Loader=yaml.FullLoader)


def generate_kernel_name(op_name, place_str):
    [target_, layout_, precision_] = place_str[1:-1].split(',')
    target_ = target_type_converter[target_.strip()]
    layout_ = layout_type_converter[layout_.strip()]
    precision_ = precision_type_converter[precision_.strip()]
    class_name_ = "{}{}".format(
        op_name.replace("_", "").title(), "".join([
            target_.strip().title(), precision_.strip(), layout_.strip().title()
            .title()
        ]))
    alias_ = "{}.{}".format(op_name, ".".join(
        [target_.strip(), precision_.strip(), layout_.strip()]))
    return alias_, class_name_


def generate_attrs_info(op_name, attrs_info):
    kernel_attrs_names = {}
    attrs_args_ = ""
    with open(kernel_signature_info_file) as f:
        kernel_attrs_names = json.load(f)
        kernel_attrs_names.update(get_compat_kernels_info())
    if len(kernel_attrs_names[op_name]["attrs"]) == len(attrs_info):
        for index in range(len(attrs_info)):
            attr_name = kernel_attrs_names[op_name]["attrs"][index]
            attr_type = attr_type_converter[attrs_info[index]]
            attrs_args_ += '{type_}:${name_},'.format(
                type_=attr_type, name_=attr_name)
    return attrs_args_[:-1]


def generate_inputs_info(input_info):
    input_args_ = ""
    for index in range(len(input_info)):
        [target_, layout_, precision_] = input_info[index].split(',')
        # todo: check vadility
        target_ = target_type_converter[target_.strip()]
        layout_ = layout_type_converter[layout_.strip()]
        precision_ = precision_type_converter[precision_.strip()]
        input_args_ += " DenseTensor<\"{}\",\"{}\",\"{}\">:$in{},".format(
            target_.strip(), precision_.strip(), layout_.strip(), str(index))
    input_args_ = input_args_[:-1]
    return input_args_


def generate_arguments_info(op_name, input_info, attr_info):
    input_args = generate_inputs_info(input_info)
    attr_args = generate_attrs_info(op_name, attr_info)
    context_args = "Context:$dev_ctx"
    argument_list = [context_args] + input_args.split(",") + attr_args.split(
        ",")
    while ("" in argument_list):
        argument_list.remove("")
    argument_ = ",".join(argument_list)
    return (("let arguments = (ins {});".format(argument_.strip(","))))


def generate_results_info(output_info):
    output_args_ = "let results = (outs "
    for index in range(len(output_info)):
        [target_, layout_, precision_] = output_info[index].split(',')
        # todo: check vadility
        target_ = target_type_converter[target_.strip()]
        layout_ = layout_type_converter[layout_.strip()]
        precision_ = precision_type_converter[precision_.strip()]
        output_args_ += " DenseTensor<\"{}\",\"{}\",\"{}\">:$out{},".format(
            target_.strip(), precision_.strip(), layout_.strip(), str(index))
    return ("{});".format(output_args_[:-1]))


def generate_supported_kernel_list(load_dict):
    supported_kernels_list_ = []
    kernel_attrs_names = {}
    with open(kernel_signature_info_file) as f:
        kernel_attrs_names = json.load(f)
        kernel_attrs_names.update(get_compat_kernels_info())
    for op_name in load_dict:
        kernel_list = load_dict[op_name]
        for kernel_info in kernel_list:
            for kernel_alias_ in kernel_info:
                attributes = kernel_info[kernel_alias_]["attribute"]
                flag = True
                for attribute in attributes:
                    if attribute not in attr_type_converter:
                        flag = False
                if flag and op_name in kernel_attrs_names:
                    supported_kernels_list_.append(op_name)
    supported_kernels_list_ = list(set(supported_kernels_list_))
    skipped_kernel_list = get_skipped_kernel_list()
    for skipped_kernel in skipped_kernel_list:
        if skipped_kernel in skipped_kernel_list:
            supported_kernels_list_.remove(skipped_kernel)
    return supported_kernels_list_


def scan_kernel_info(load_dict):
    target_type_ = []
    layout_type_ = []
    precision_type_ = []
    for op_name in load_dict:
        kernel_list = load_dict[op_name]
        for kernel_info in kernel_list:
            for kernel_alias_ in kernel_info:
                [target_, layout_, precision_] = kernel_alias_[1:-1].split(',')
                target_type_.append(target_.strip())
                layout_type_.append(layout_.strip())
                precision_type_.append(precision_.strip())
    target_type_ = list(set(target_type_))
    layout_type_ = list(set(layout_type_))
    precision_type_ = list(set(precision_type_))
    print(target_type_)
    print(layout_type_)
    print(precision_type_)


def generate_cpu_kernel_dialect(op_name, kernel_alias_, kernel_info):

    alias, class_name = generate_kernel_name(op_name, kernel_alias_)
    summary = 'let summary = "{name}";'.format(name=alias)
    dialect_name = alias.split(".")
    dialect_name = dialect_name[0] + "." + dialect_name[2] + "." + dialect_name[
        3]

    header = 'def {kernel_name} : PDTCPU_Kernel<"{name}",[NoSideEffect]> {left_brace}'.format(
        kernel_name=class_name, name=dialect_name.lower(), left_brace="{")

    inputs_ = kernel_info["input"]
    attributes = kernel_info["attribute"]
    arguments = generate_arguments_info(op_name, inputs_, attributes)

    outputs = kernel_info["output"]
    results = generate_results_info(outputs)

    kernel_dialect = '{header_}\n  {summary_}\n  {arguments_}\n  {results_}\n{right_brace}\n'.format(
        header_=header,
        summary_=summary,
        arguments_=arguments,
        results_=results,
        right_brace="}")
    return kernel_dialect


def generate_gpu_kernel_dialect(op_name, kernel_alias_, kernel_info):

    alias, class_name = generate_kernel_name(op_name, kernel_alias_)
    summary = 'let summary = "{name}";'.format(name=alias)
    dialect_name = alias.split(".")
    dialect_name = dialect_name[0] + "." + dialect_name[2] + "." + dialect_name[
        3]

    header = 'def {kernel_name} : PDTGPU_Kernel<"{name}",[NoSideEffect]> {left_brace}'.format(
        kernel_name=class_name, name=dialect_name.lower(), left_brace="{")
    inputs_ = kernel_info["input"]
    attributes = kernel_info["attribute"]
    arguments = generate_arguments_info(op_name, inputs_, attributes)

    outputs = kernel_info["output"]
    results = generate_results_info(outputs)

    kernel_dialect = '{header_}\n  {summary_}\n  {arguments_}\n  {results_}\n{right_brace}\n'.format(
        header_=header,
        summary_=summary,
        arguments_=arguments,
        results_=results,
        right_brace="}")
    return kernel_dialect


def generate_dialect_head():
    comment_ = "/*===- TableGen'source file -----------------------------------------------===*\\\n\
|*                                                                            *|\n\
|* Kernel Definitions                                                         *|\n\
|*                                                                            *|\n\
|* Automatically generated file, do not edit!                                 *|\n\
|* Generated by tools/infrt/generate_pten_kernel_dialect.py                   *|\n\
|*                                                                            *|\n\
\*===----------------------------------------------------------------------===*/\n"

    includes_ = "#ifndef PTEN_KERNELS\n\
#define PTEN_KERNELS\n\
include \"mlir/Interfaces/InferTypeOpInterface.td\"\n\
include \"mlir/Interfaces/LoopLikeInterface.td\"\n\
include \"mlir/IR/OpBase.td\"\n\
include \"paddle/infrt/dialect/phi/ir/infrt_phi_kernel.td\""

    return (comment_ + includes_)


def get_kernel_target(kernel_alias_):
    target = kernel_alias_[1:-1].split(",")
    return target[0]


def main():
    with open(kernel_types_info_file, "r") as f:
        load_dict = json.load(f)

        head = generate_dialect_head()

        cpu_registry_ = ""
        gpu_registry_ = ""
        supported_kernels = generate_supported_kernel_list(load_dict)

        print("Supported kernels:")
        print(supported_kernels)
        for op_name in load_dict:
            if op_name not in supported_kernels:
                continue
            kernel_list = load_dict[op_name]
            for kernel_info in kernel_list:
                for kernel_alias_ in kernel_info:
                    if get_kernel_target(kernel_alias_) == "CPU":
                        kernel_registry = generate_cpu_kernel_dialect(
                            op_name, kernel_alias_, kernel_info[kernel_alias_])
                        cpu_registry_ += kernel_registry
                    elif get_kernel_target(kernel_alias_) == "GPU":
                        kernel_registry = generate_gpu_kernel_dialect(
                            op_name, kernel_alias_, kernel_info[kernel_alias_])
                        gpu_registry_ += kernel_registry
                    else:
                        print("Unsupported backend:" + get_kernel_target(
                            kernel_alias_))
        end = "#endif  // PTEN_KERNELS"
        with open("../../paddle/infrt/dialect/phi/ir/phi_cpu_kernels.td",
                  "w") as dst:
            dst.write('{start_}\n{dialect_}\n{end_}'.format(
                start_=head, dialect_=cpu_registry_, end_=end))
        with open("../../paddle/infrt/dialect/phi/ir/phi_gpu_kernels.td",
                  "w") as dst:
            dst.write('{start_}\n{dialect_}\n{end_}'.format(
                start_=head, dialect_=gpu_registry_, end_=end))


if __name__ == '__main__':
    if not os.path.exists(kernel_types_info_file):
        print("Error: '{file_name}' not exist!".format(
            file_name=kernel_types_info_file))
    if not os.path.exists(kernel_signature_info_file):
        print("Error: '{file_name}' not exist!".format(
            file_name=kernel_signature_info_file))
    if os.path.exists(kernel_types_info_file) and os.path.exists(
            kernel_signature_info_file):
        main()
