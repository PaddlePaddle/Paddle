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

import sys

import yaml

sys.path.append(
    r'/home/aistudio/fix_op/Paddle/paddle/fluid/operators/generator'
)

sys.path.append(
    r'/home/aistudio/test/Paddle/tools/count_op/pir_ops'
)

from parse_utils import parse_op_entry

import get_all_pir_ops


def count_ops_yaml(op_yaml_path):
    with open(op_yaml_path, "rt") as f:
        ops = yaml.safe_load(f)
        if ops is None:
            ops = []
        else:
            ops = [parse_op_entry(op, "op") for op in ops]

        pir_op_info = []

        for op in ops:
            op_type = op["name"]
            inputs = op["inputs"]
            outputs = op["outputs"]
            attrs = op["attrs"]

            input_names = []
            attr_names = []
            output_names = []

            for input in inputs:
                input_names.append(input["name"])

            for attr in attrs:
                attr_names.append(attr["name"])

            for output in outputs:
                output_names.append(output["name"])

            pir_op_info.append(
                {
                    "op_name": op_type,
                    "inputs": input_names,
                    "attrs": attr_names,
                    "outputs": output_names,
                }
            )
        print(pir_op_info)

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
                +"(" + str(output_names[i]) + ")"
                + ", "
            )
        else:
            outputs = (
                outputs + str(output_types[i])  +"(" + str(output_names[i]) + ")"
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
        data = {"op": op_name, "inputs": inputs, "attrs": attrs, "outputs": outputs}
    if len(optionals) > 0 and len(inplaces) == 0:
        data = {"op": op_name, "inputs": inputs, "attrs": attrs, "outputs": outputs,"optionals" : optionals}
    if len(optionals) == 0 and len(inplaces) > 0:
        data = {"op": op_name, "inputs": inputs, "attrs": attrs, "outputs": outputs,"inplaces" : inplaces}
    if  len(optionals) > 0 and len(inplaces) > 0:
        data = {"op": op_name, "inputs": inputs, "attrs": attrs, "outputs": outputs,"optionals" : optionals,"inplaces" : inplaces}
    return data

def extract_ops_parsed_yaml(ops_parsed_yaml_path, save_path):
    with open(ops_parsed_yaml_path, "rt") as f:
        ops = yaml.safe_load(f)

        pir_op_info = []

        for op in ops:
            op_name = op["name"]
            inputs = op["inputs"]
            outputs = op["outputs"]
            attrs = op["attrs"]

            input_names = []
            input_types = []
            attr_names = []
            attr_types = []
            output_names = []
            output_types = []
            optional_names = []
            inplaces = ""

            for input in inputs:
                input_names.append(input["name"])
                input_types.append(input["typename"])
                if input["optional"]:
                    optional_names.append(input["name"])

            for attr in attrs:
                attr_names.append(attr["name"])
                attr_types.append(attr["typename"])
                
            for output in outputs:
                output_names.append(output["name"])
                output_types.append(output["typename"])
                if output["optional"]:
                    optional_names.append(output["name"])
            
            try:
                inplace_dict = op["inplace"]
            except KeyError:
                inplace_dict = None

            if inplace_dict is  None:
                inplaces = ""
            else:
                for k,v in inplace_dict.items():
                    inplaces = inplaces + v + "->"+ k +  ","
                inplaces = inplaces[:-1]
 

            data = convert_opinfo_to_dict(
                op_name,
                input_types,
                input_names,
                attr_types,
                attr_names,
                output_types,
                output_names,
                optional_names,
                inplaces
            )
            pir_op_info.append(data)

        ops_dict = dict()
        for op_info in pir_op_info:
            ops_dict[op_info["op"]] = op_info
        
        op_info_list = list()
        for op_info in pir_op_info:
            if op_info["op"][-1] != '_':
                if "inplaces" in op_info:
                    print(op_info["op"] + "_")
                    if (op_info["op"] + "_") not in ops_dict:
                        op_info_without_inplace = dict(op_info)
                        op_info_without_inplace.pop("inplaces")
                        op_info_within_inplace = dict(op_info)
                        op_info_within_inplace["op"] = op_info["op"] + "_"
                        op_info_list.append(op_info_within_inplace)
                        op_info_list.append(op_info_without_inplace)
                    else:
                        op_info_without_inplace = dict(op_info)
                        op_info_without_inplace.pop("inplace")
                        op_info_list.append(op_info_without_inplace)
                else:
                    op_info_list.append(op_info)
            else:
                op_info_list.append(op_info)

        with open(save_path, 'w') as file:
            for op_info in op_info_list:
                temp = [op_info]
                yaml.dump(
                    temp,
                    file,
                    default_flow_style=False,
                    sort_keys=False,
                    indent=1,
                )
                file.write("\n")

        print("算子总数: ")
        print(len(op_info_list))

        # got_pir_op_names = []
        # for op_info in op_info_list:
        #     got_pir_op_names.append(op_info["op"])
        
        # all_pir_ops = get_all_pir_ops.get_all_pir_ops(get_all_pir_ops.str_info)

        # got_pir_op_names = set(got_pir_op_names)
        # all_pir_ops = set(all_pir_ops)

        # print("all_pir_ops - got_pir_op_names")
        # print(all_pir_ops - got_pir_op_names)
        # print("got_pir_op_names - all_pir_ops")
        # print(got_pir_op_names - all_pir_ops)
        



    ops_parsed_yaml_path = (
        r"/home/aistudio/test/Paddle/tools/count_op/pir_ops/ops.parsed.yaml"
    )
    save_path = r'/home/aistudio/test/Paddle/tools/count_op/example.yaml'
    extract_ops_parsed_yaml(ops_parsed_yaml_path, save_path)
