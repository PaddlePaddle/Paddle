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

import glob
import os
import re

import yaml

from paddle.base.framework import (
    OpProtoHolder,
)


def match_operator(content, pattern1, pattern2):
    res = []
    matchs = re.findall(pattern1, content, flags=re.DOTALL)
    for match in matchs:
        if r"_grad" not in match:
            res.append(match)
    matchs = re.findall(pattern2, content, flags=re.DOTALL)
    for match in matchs:
        if r"_grad" not in match:
            res.append(match)
    return res, len(res)


def get_all_old_ir_ops():
    tool_dir = os.path.dirname(os.path.abspath(__file__))

    all_op = glob.glob(
        os.path.join(tool_dir, '../../paddle/**/*.cc'),
        recursive=True,
    )
    all_op += glob.glob(
        os.path.join(tool_dir, '../../paddle/**/*.cu'),
        recursive=True,
    )

    register_op_count = 0

    all_matches = []

    for op_file in all_op:
        op_pattern1 = r'REGISTER_OPERATOR\(.*?\);?'
        op_pattern2 = r'REGISTER_OP_WITHOUT_GRADIENT\(.*?\);?'
        # op_pattern2 = r'REGISTER_OPERATOR\(.*?_grad,.*?\);?'
        with open(op_file, 'r', encoding='utf-8') as f:
            content = ''.join(f.readlines())
            op, op_count = match_operator(content, op_pattern1, op_pattern2)
            if len(op) != 0:
                all_matches.append(op)

    opname_pattern1 = r"REGISTER_OPERATOR\(\s*(\w+),"
    opname_pattern2 = r"REGISTER_OP_WITHOUT_GRADIENT\(\s*(\w+),"

    op_list = []
    for re_op_cc in all_matches:
        for item in re_op_cc:
            match = re.search(opname_pattern1, item)
            if match:
                operator_name = match.group(1)
                op_list.append(operator_name)

    for re_op_cc in all_matches:
        for item in re_op_cc:
            match = re.search(opname_pattern2, item)
            if match:
                operator_name = match.group(1)
                op_list.append(operator_name)

    op_list = list(set(op_list))
    return op_list


def read_txt_file(file_path):
    with open(file_path, 'r') as file:
        lines = [line.strip() for line in file.readlines()]
        return lines


def get_inplace_info(inplace_info_path):
    inplace_infos_list = read_txt_file(inplace_info_path)
    inplace_infos = {}
    for i in range(len(inplace_infos_list)):
        if i % 2 == 1:
            continue
        if inplace_infos_list[i + 1] == "Null":
            inplace_infos[inplace_infos_list[i]] = ""
        else:
            inplace_infos[inplace_infos_list[i]] = inplace_infos_list[i + 1][
                :-1
            ]
    return inplace_infos


def convert_opinfo_to_dict(
    op_name,
    input_types,
    input_names,
    attr_types,
    attr_names,
    output_types,
    output_names,
    optional_names,
    inplaces,
):
    inputs = ""
    attrs = ""
    outputs = ""
    optionals = ""

    for i in range(len(input_names)):
        if i < len(input_names) - 1:
            inputs = (
                inputs + str(input_types[i]) + " " + str(input_names[i]) + ", "
            )
        else:
            inputs = inputs + str(input_types[i]) + " " + str(input_names[i])

    for i in range(len(attr_names)):
        if i < len(attr_names) - 1:
            attrs = attrs + str(attr_types[i]) + " " + str(attr_names[i]) + ", "
        else:
            attrs = attrs + str(attr_types[i]) + " " + str(attr_names[i])

    for i in range(len(output_names)):
        if i < len(output_names) - 1:
            outputs = (
                outputs
                + str(output_types[i])
                + "("
                + str(output_names[i])
                + ")"
                + ", "
            )
        else:
            outputs = (
                outputs
                + str(output_types[i])
                + "("
                + str(output_names[i])
                + ")"
            )

    for i in range(len(optional_names)):
        if i < len(optional_names) - 1:
            optionals = optionals + optional_names[i] + ","
        else:
            optionals = optionals + optional_names[i]

    inputs = "(" + inputs + ")"
    attrs = "(" + attrs + ")"
    outputs = outputs
    optionals = optionals
    inplaces = inplaces

    if len(optionals) == 0 and len(inplaces) == 0:
        data = {
            "op": op_name,
            "inputs": inputs,
            "attrs": attrs,
            "outputs": outputs,
        }
    if len(optionals) > 0 and len(inplaces) == 0:
        data = {
            "op": op_name,
            "inputs": inputs,
            "attrs": attrs,
            "outputs": outputs,
            "optionals": optionals,
        }
    if len(optionals) == 0 and len(inplaces) > 0:
        data = {
            "op": op_name,
            "inputs": inputs,
            "attrs": attrs,
            "outputs": outputs,
            "inpalces": inplaces,
        }
    if len(optionals) > 0 and len(inplaces) > 0:
        data = {
            "op": op_name,
            "inputs": inputs,
            "attrs": attrs,
            "outputs": outputs,
            "optionals": optionals,
            "inpalces": inplaces,
        }
    return data


def get_attr_type_string(attr_type):
    attr_type_dict = {
        0: 'INT',
        1: 'FLOAT',
        2: 'STRING',
        3: 'INTS',
        4: 'FLOATS',
        5: 'STRINGS',
        6: 'BOOLEAN',
        7: 'BOOLEANS',
        8: 'BLOCK',
        9: 'LONG',
        10: 'BLOCKS',
        11: 'LONGS',
        12: 'FLOAT64S',
        13: 'VAR',
        14: 'VARS',
        15: 'FLOAT64',
        16: 'SCALAR',
        17: 'SCALARS',
    }
    return attr_type_dict[attr_type]


if __name__ == '__main__':
    registed_op = read_txt_file(
        r"/home/aistudio/fix_op/Paddle/tools/count_op/fluid/registed_op.txt"
    )
    inplace_infos = get_inplace_info(
        r"/home/aistudio/fix_op/Paddle/tools/count_op/fluid/inplace_info.txt"
    )

    all_fluid_op_infos = []

    for op in registed_op:
        op_name = op
        input_types = []
        input_names = []
        optional_names = []
        attr_names = []
        attr_types = []
        output_names = []
        output_types = []

        inplaces = ""

        op_proto = OpProtoHolder.instance().get_op_proto(op_name)
        for input in op_proto.inputs:
            input_names.append(input.name)
            if input.dispensable:
                optional_names.append(input.name)
            if input.duplicable:
                input_types.append("Tensor[]")
            else:
                if "TensorArray" in input.comment:
                    input_types.append("Tensor|TensorArray?")
                else:
                    input_types.append("Tensor")

        for attr in op_proto.attrs:
            attr_names.append(attr.name)
            attr_types.append(get_attr_type_string(attr.type))

        for output in op_proto.outputs:
            output_names.append(output.name)
            if output.dispensable:
                optional_names.append(output.name)
            if output.duplicable:
                output_types.append("Tensor[]")
            else:
                if "TensorArray" in output.comment:
                    output_types.append("Tensor|TensorArray?")
                else:
                    output_types.append("Tensor")

        data = convert_opinfo_to_dict(
            op_name,
            input_types,
            input_names,
            attr_types,
            attr_names,
            output_types,
            output_names,
            optional_names,
            [],
        )

        all_fluid_op_infos.append(data)

        if len(inplace_infos[op_name]) > 0:
            data = convert_opinfo_to_dict(
                op_name + "_",
                input_types,
                input_names,
                attr_types,
                attr_names,
                output_types,
                output_names,
                optional_names,
                inplace_infos[op_name],
            )
            all_fluid_op_infos.append(data)

    print(all_fluid_op_infos)
    save_path = (
        r"/home/aistudio/fix_op/Paddle/tools/count_op/fluid/fluid_ops.yaml"
    )
    with open(save_path, 'w') as file:
        for op_info in all_fluid_op_infos:
            temp = [op_info]
            yaml.dump(
                temp,
                file,
                default_flow_style=False,
                sort_keys=False,
                indent=1,
            )
            file.write("\n")
