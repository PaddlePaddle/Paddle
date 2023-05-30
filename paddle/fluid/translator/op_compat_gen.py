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
from typing import Dict

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
        # Templat: - op : phi_name (fluid_name)
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
    op_name_mappings = {}
    op_arg_name_mappings = {}
    for op_compat_item in op_compat_infos:

        def insert_new_mappings(op_name_str: str) -> str:
            normalized_name, legacy_name = to_phi_and_fluid_op_name(op_name_str)
            if normalized_name == legacy_name:
                return normalized_name
            op_name_mappings[legacy_name] = normalized_name
            return normalized_name

        def insert_new_arg_mappings(op_name: str, arg_mapping: Dict[str, str]):
            if op_name is None:
                return
            if op_name not in op_arg_name_mappings:
                op_arg_name_mappings[op_name] = {}
            op_arg_name_mappings[op_name].update(arg_mapping)

        normalized_op_name = insert_new_mappings(op_compat_item["op"])
        normailized_backward_op_name = None
        if "backward" in op_compat_item:
            normailized_backward_op_name = insert_new_mappings(
                op_compat_item["backward"]
            )
        if "inputs" in op_compat_item:
            insert_new_arg_mappings(
                normalized_op_name, op_compat_item["inputs"]
            )
            insert_new_arg_mappings(
                normailized_backward_op_name, op_compat_item["inputs"]
            )
        if "attrs" in op_compat_item:
            insert_new_arg_mappings(normalized_op_name, op_compat_item["attrs"])
            insert_new_arg_mappings(
                normailized_backward_op_name, op_compat_item["attrs"]
            )
        if "outputs" in op_compat_item:
            insert_new_arg_mappings(
                normalized_op_name, op_compat_item["outputs"]
            )
            insert_new_arg_mappings(
                normailized_backward_op_name, op_compat_item["outputs"]
            )

    # special op mappings
    op_name_mappings["fetch_v2"] = "fetch"

    op_name_normailzer_template = env.get_template("op_compat_info.cc.j2")
    with open(output_source_file, 'wt') as f:
        op_compat_definition = op_name_normailzer_template.render(
            op_name_pairs=op_name_mappings,
            op_arg_name_pairs=op_arg_name_mappings,
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
