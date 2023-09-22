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

import argparse
from pathlib import Path
from typing import Dict, List, Set

import yaml
from jinja2 import Environment, FileSystemLoader, StrictUndefined

file_loader = FileSystemLoader(Path(__file__).parent)
env = Environment(
    loader=file_loader,
    keep_trailing_newline=True,
    trim_blocks=True,
    lstrip_blocks=True,
    undefined=StrictUndefined,
    extensions=['jinja2.ext.do'],
)


def OpNameNormalizerInitialization(
    op_compat_yaml_file: str = "", output_source_file: str = ""
) -> None:
    def to_phi_and_fluid_op_name(op_item):
        # Template: - op : phi_name (fluid_name)
        names = op_item.split('(')
        if len(names) == 1:
            phi_fluid_name = names[0].strip()
            return phi_fluid_name, phi_fluid_name
        else:
            phi_name = names[0].strip()
            fluid_name = names[1].split(')')[0].strip()
            return phi_name, fluid_name

    with open(op_compat_yaml_file, "r") as f:
        op_compat_infos = yaml.safe_load(f)
    op_name_mappings: Dict[str, str] = {}
    op_arg_name_mappings: Dict[str, Dict[str, str]] = {}
    op_mutable_attribues: Dict[str, Set[str]] = {}
    op_mutable_attribute_infos: Dict[str, Dict[str, List[str]]] = {}

    for op_compat_item in op_compat_infos:

        def insert_new_mappings(op_name_str: str) -> str:
            normalized_name, legacy_name = to_phi_and_fluid_op_name(op_name_str)
            if normalized_name == legacy_name:
                return normalized_name, legacy_name
            op_name_mappings[legacy_name] = normalized_name
            return normalized_name, legacy_name

        def insert_new_arg_mappings(op_name: str, arg_mapping: Dict[str, str]):
            if op_name is None:
                return
            if op_name not in op_arg_name_mappings:
                op_arg_name_mappings[op_name] = {}
            op_arg_name_mappings[op_name].update(arg_mapping)

        def insert_new_mutable_attributes(
            op_name: str, mutable_attribute_infos: Dict[str, Dict[str, str]]
        ):
            if op_name not in op_mutable_attribues:
                op_mutable_attribues[op_name] = set()
            if op_name not in op_mutable_attribute_infos:
                op_mutable_attribute_infos[op_name] = {}
            for (
                attribute_name,
                mutable_attribute_info,
            ) in mutable_attribute_infos.items():
                op_mutable_attribues[op_name].add(attribute_name)
                op_mutable_attribute_infos[op_name][attribute_name] = []
                for k, v in mutable_attribute_info.items():
                    if k == 'tensor_name' or k == 'tensors_name':
                        op_mutable_attribute_infos[op_name][
                            attribute_name
                        ].append(v)

        _, legacy_name = insert_new_mappings(op_compat_item["op"])
        legacy_backward_op_names = []
        if "backward" in op_compat_item:
            backward_op_name_mapping_paris = op_compat_item["backward"].split(
                ","
            )
            for pair in backward_op_name_mapping_paris:
                _, legacy_backward_op_name = insert_new_mappings(pair)
                legacy_backward_op_names.append(legacy_backward_op_name)

        if "inputs" in op_compat_item:
            insert_new_arg_mappings(legacy_name, op_compat_item["inputs"])
            for backward_op in legacy_backward_op_names:
                insert_new_arg_mappings(backward_op, op_compat_item["inputs"])

        if "attrs" in op_compat_item:
            insert_new_arg_mappings(legacy_name, op_compat_item["attrs"])
            for backward_op in legacy_backward_op_names:
                insert_new_arg_mappings(backward_op, op_compat_item["attrs"])
        if "outputs" in op_compat_item:
            insert_new_arg_mappings(legacy_name, op_compat_item["outputs"])
            for backward_op in legacy_backward_op_names:
                insert_new_arg_mappings(backward_op, op_compat_item["outputs"])

        if "int_array" in op_compat_item:
            insert_new_mutable_attributes(
                legacy_name, op_compat_item["int_array"]
            )
            for backward_op in legacy_backward_op_names:
                insert_new_mutable_attributes(
                    backward_op, op_compat_item["int_array"]
                )

        if "scalar" in op_compat_item:
            insert_new_mutable_attributes(legacy_name, op_compat_item["scalar"])
            for backward_op in legacy_backward_op_names:
                insert_new_mutable_attributes(
                    backward_op, op_compat_item["scalar"]
                )

    # special mapping list
    op_arg_name_mappings["set_value_grad"]["values_grad"] = "ValueTensor@GRAD"
    op_arg_name_mappings["fetch"] = {"x": "X"}

    op_name_normailzer_template = env.get_template("op_compat_info.cc.j2")
    with open(output_source_file, 'wt') as f:
        op_compat_definition = op_name_normailzer_template.render(
            op_name_pairs=op_name_mappings,
            op_arg_name_pairs=op_arg_name_mappings,
            op_mutable_attributes=op_mutable_attribues,
            op_mutable_attribute_infos=op_mutable_attribute_infos,
        )
        f.write(op_compat_definition)


# =====================================
# Script parameter parsing
# =====================================
def ParseArguments():
    parser = argparse.ArgumentParser(
        description='Generate OP Compatiable info Files By Yaml'
    )
    parser.add_argument('--op_compat_yaml_file', type=str)
    parser.add_argument('--output_source_file', type=str)
    return parser.parse_args()


# =====================================
# Main
# =====================================
if __name__ == "__main__":
    # parse arguments
    args = ParseArguments()
    OpNameNormalizerInitialization(**vars(args))
