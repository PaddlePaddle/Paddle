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

# Notice: This file will be automatically executed during building of whole paddle project.
#         You can also run this file separately if you want to preview generated file without building.

import argparse
import json
import re

import yaml


def ParseArguments():
    parser = argparse.ArgumentParser(
        description='prim ops Code Generator Args Parser'
    )
    parser.add_argument('--ops_yaml_path', type=str, help="path to ops.yaml")
    parser.add_argument(
        '--ops_legacy_yaml_path', type=str, help="path to dygraph_ops.yaml"
    )
    parser.add_argument(
        '--ops_compat_yaml_path', type=str, help="path to op_compat.yaml"
    )
    parser.add_argument(
        '--phi_ops_map_path',
        type=str,
        default="./phi_ops_map.py",
        help='path to target phi_ops_map.py',
    )

    args = parser.parse_args()
    return args


def _trans_value_type(item):
    for key in item.keys():
        for subkey in item[key]:
            value = str(item[key][subkey])
            item[key][subkey] = value


def generate_code(
    ops_yaml_path, ops_legacy_yaml_path, ops_compat_yaml_path, phi_ops_map_path
):
    """
    Generate dictionary and save to file phi_ops_map.py. The target file records gap
    of description between current op and standard ones.
    """
    dct = {}
    map_dct = {}
    for op_path in [ops_yaml_path, ops_legacy_yaml_path]:
        pattern = re.compile(r'[(](.*)[)]', re.S)
        with open(op_path, "rt") as f:
            ops = yaml.safe_load(f)
            for item in ops:
                key = item['op']
                if key in dct:
                    raise ValueError(f"There already exists op {key}")
                dct[key] = {
                    "args": re.findall(pattern, item["args"])[0],
                    "output": item["output"],
                }

        with open(ops_compat_yaml_path, "rt") as f:
            ops_compat = yaml.safe_load(f)
            for item in ops_compat:
                key = item['op']
                if key.endswith(")"):
                    tmp = re.match("(.*)\\((.*)\\)", key.replace(" ", ""))
                    phi_name, op_name = tmp.group(1), tmp.group(2)
                    map_dct[op_name] = {"phi_name": phi_name}
                else:
                    op_name = key
                    map_dct[op_name] = {"phi_name": op_name}
                for element in ["inputs", "outputs", "attrs"]:
                    if element in item.keys():
                        map_dct[op_name][element] = item[element]
                for element in ["scalar", "int_array"]:
                    if element in item.keys():
                        _trans_value_type(item[element])
                        map_dct[op_name][element] = item[element]

        with open(phi_ops_map_path, "w") as f:
            f.write("op_map = ")
            json.dump(map_dct, f, indent=4)
            f.write('\n')
            f.write("op_info = ")
            json.dump(dct, f, indent=4)
            f.write('\n')


if __name__ == "__main__":
    args = ParseArguments()
    ops_yaml_path = args.ops_yaml_path
    ops_legacy_yaml_path = args.ops_legacy_yaml_path
    ops_compat_yaml_path = args.ops_compat_yaml_path
    phi_ops_map_path = args.phi_ops_map_path
    generate_code(
        ops_yaml_path,
        ops_legacy_yaml_path,
        ops_compat_yaml_path,
        phi_ops_map_path,
    )
