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

import argparse
import os
import re
from itertools import chain
from pathlib import Path

import yaml
from jinja2 import Environment, FileSystemLoader, StrictUndefined

from filters import to_op_attr_type, to_opmaker_name, to_opmaker_name_cstr, to_pascal_case
from tests import is_base_api, is_vec, is_scalar, is_initializer_list, supports_inplace, supports_no_need_buffer
from filters import to_input_name, cartesian_prod_mapping
from parse_utils import to_named_dict

file_loader = FileSystemLoader(Path(__file__).parent / "templates")
env = Environment(loader=file_loader,
                  keep_trailing_newline=True,
                  trim_blocks=True,
                  lstrip_blocks=True,
                  undefined=StrictUndefined,
                  extensions=['jinja2.ext.do'])
env.filters["to_op_attr_type"] = to_op_attr_type
env.filters["to_opmaker_name"] = to_opmaker_name
env.filters["to_pascal_case"] = to_pascal_case
env.filters["to_input_name"] = to_input_name
env.filters["to_opmaker_name_cstr"] = to_opmaker_name_cstr
env.filters["cartesian_prod_mapping"] = cartesian_prod_mapping
env.tests["base_api"] = is_base_api
env.tests["vec"] = is_vec
env.tests["scalar"] = is_scalar
env.tests["initializer_list"] = is_initializer_list
env.tests["supports_inplace"] = supports_inplace
env.tests["supports_no_need_buffer"] = supports_no_need_buffer


def restruct_io(api):
    api["input_dict"] = to_named_dict(api["inputs"])
    api["attr_dict"] = to_named_dict(api["attrs"])
    api["output_dict"] = to_named_dict(api["outputs"])
    return api


# replace name of op and params for OpMaker
def replace_compat_name(api_op_map, forward_api_dict, backward_api_dict):

    def get_api_and_op_name(api_item):
        names = api_item.split('(')
        if len(names) == 1:
            return names[0].strip(), names[0].strip()
        else:
            return names[0].strip(), names[1].split(')')[0].strip()

    for api_args in api_op_map:
        api_name, op_name = get_api_and_op_name(api_args['op'])
        if api_name not in forward_api_dict:
            continue
        forward_api_item = forward_api_dict[api_name]
        has_backward = True if forward_api_item['backward'] else False
        if has_backward:
            backward_api_item = backward_api_dict[forward_api_item['backward']]
        if api_name != op_name:
            forward_api_item['op_name'] = op_name
        if 'backward' in api_args and has_backward:
            bw_api_name, bw_op_name = get_api_and_op_name(
                api_args['backward'].split(',')[0])
            forward_api_item['backward'] = bw_op_name
            backward_api_item['op_name'] = bw_op_name

        key_set = ['inputs', 'attrs', 'outputs']
        args_map = {}
        for key in key_set:
            if key in api_args:
                args_map.update(api_args[key])
                for args_item in forward_api_item[key]:
                    if args_item['name'] in api_args[key]:
                        args_item['name'] = api_args[key][args_item['name']]
                if has_backward:
                    for args_item in backward_api_item['forward'][key]:
                        if args_item['name'] in api_args[key]:
                            args_item['name'] = api_args[key][args_item['name']]
        forward_api_item['infer_meta']['param'] = [
            args_map[param] if param in args_map else param
            for param in forward_api_item['infer_meta']['param']
        ]
        forward_api_item['kernel']['param'] = [
            args_map[param] if param in args_map else param
            for param in forward_api_item['kernel']['param']
        ]
        if forward_api_item['kernel']['data_type']:
            forward_api_item['kernel']['data_type']['candidates'] = [
                args_map[param] if param in args_map else param for param in
                forward_api_item['kernel']['data_type']['candidates']
            ]
        if forward_api_item['kernel']['backend']:
            forward_api_item['kernel']['backend']['candidates'] = [
                args_map[param] if param in args_map else param
                for param in forward_api_item['kernel']['backend']['candidates']
            ]
        if forward_api_item['kernel']['layout']:
            forward_api_item['kernel']['layout']['candidates'] = [
                args_map[param] if param in args_map else param
                for param in forward_api_item['kernel']['layout']['candidates']
            ]
        if forward_api_item['inplace']:
            inplace_map = {}
            for key, val in forward_api_item['inplace'].items():
                if key in args_map:
                    key = args_map[key]
                if val in args_map:
                    val = args_map[val]
                key, val = val, key
                inplace_map[key] = val
            forward_api_item['inplace'] = inplace_map

        if has_backward:
            for args_item in backward_api_item['inputs']:
                if args_item['name'] in args_map:
                    args_item['name'] = args_map[args_item['name']]
                elif args_item['name'].endswith(
                        '_grad') and args_item['name'][:-5] in args_map:
                    args_map[args_item['name']] = args_map[args_item['name']
                                                           [:-5]] + '_grad'
                    args_item['name'] = args_map[args_item['name']]
            for args_item in backward_api_item['attrs']:
                if args_item['name'] in args_map:
                    args_item['name'] = args_map[args_item['name']]
            for args_item in backward_api_item['outputs']:
                if args_item['name'].endswith(
                        '_grad') and args_item['name'][:-5] in args_map:
                    args_map[args_item['name']] = args_map[args_item['name']
                                                           [:-5]] + '_grad'
                    args_item['name'] = args_map[args_item['name']]

            backward_api_item['infer_meta']['param'] = [
                args_map[param] if param in args_map else param
                for param in backward_api_item['infer_meta']['param']
            ]
            backward_api_item['kernel']['param'] = [
                args_map[param] if param in args_map else param
                for param in backward_api_item['kernel']['param']
            ]
            if backward_api_item['kernel']['data_type']:
                backward_api_item['kernel']['data_type']['candidates'] = [
                    args_map[param] if param in args_map else param for param in
                    backward_api_item['kernel']['data_type']['candidates']
                ]
            if backward_api_item['kernel']['backend']:
                backward_api_item['kernel']['backend']['candidates'] = [
                    args_map[param] if param in args_map else param for param in
                    backward_api_item['kernel']['backend']['candidates']
                ]
            if backward_api_item['kernel']['layout']:
                backward_api_item['kernel']['layout']['candidates'] = [
                    args_map[param] if param in args_map else param for param in
                    backward_api_item['kernel']['layout']['candidates']
                ]
            if backward_api_item['no_need_buffer']:
                backward_api_item['no_need_buffer'] = [
                    args_map[param] if param in args_map else param
                    for param in backward_api_item['no_need_buffer']
                ]


def main(api_yaml_path, backward_yaml_path, op_compat_yaml_path,
         api_version_yaml_path, output_op_path, output_arg_map_path):
    with open(api_yaml_path, "rt") as f:
        apis = yaml.safe_load(f)
        apis = [restruct_io(api) for api in apis]
    forward_api_dict = to_named_dict(apis)

    with open(backward_yaml_path, "rt") as f:
        backward_apis = yaml.safe_load(f)
        backward_apis = [restruct_io(api) for api in backward_apis]
    backward_api_dict = to_named_dict(backward_apis)

    with open(api_version_yaml_path, "rt") as f:
        api_versions = yaml.safe_load(f)
    # add api version info into api
    for api_version in api_versions:
        forward_api_dict[api_version['op']]['version'] = api_version['version']

    with open(op_compat_yaml_path, "rt") as f:
        api_op_map = yaml.safe_load(f)

    for api in apis:
        api['op_name'] = api['name']
    for bw_api in backward_apis:
        bw_api['op_name'] = bw_api['name']

    replace_compat_name(api_op_map, forward_api_dict, backward_api_dict)

    # fill backward field for an api if another api claims it as forward
    for name, backward_api in backward_api_dict.items():
        forward_name = backward_api["forward"]["name"]
        if forward_name in backward_api_dict:
            forward_api = backward_api_dict[forward_name]
            if forward_api["backward"] is None:
                forward_api["backward"] = name

    api_dict = {}
    api_dict.update(forward_api_dict)
    api_dict.update(backward_api_dict)

    if len(apis) == 0 and len(backward_apis) == 0:
        if os.path.isfile(output_op_path):
            os.remove(output_op_path)
        if os.path.isfile(output_arg_map_path):
            os.remove(output_arg_map_path)
        return

    op_template = env.get_template('op.c.j2')
    with open(output_op_path, "wt") as f:
        msg = op_template.render(apis=apis,
                                 backward_apis=backward_apis,
                                 api_dict=api_dict)
        f.write(msg)

    ks_template = env.get_template('ks.c.j2')
    with open(output_arg_map_path, 'wt') as f:
        msg = ks_template.render(apis=apis, backward_apis=backward_apis)
        f.write(msg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate operator file from api yaml.")
    parser.add_argument('--api_yaml_path',
                        type=str,
                        help="parsed api yaml file.")
    parser.add_argument('--backward_api_yaml_path',
                        type=str,
                        help="parsed backward api yaml file.")
    parser.add_argument('--op_compat_yaml_path',
                        type=str,
                        help="api args compat yaml file.")
    parser.add_argument('--api_version_yaml_path',
                        type=str,
                        help="api version yaml file.")
    parser.add_argument("--output_op_path",
                        type=str,
                        help="path to save generated operators.")
    parser.add_argument(
        "--output_arg_map_path",
        type=str,
        help="path to save generated argument mapping functions.")

    args = parser.parse_args()
    main(args.api_yaml_path, args.backward_api_yaml_path,
         args.op_compat_yaml_path, args.api_version_yaml_path,
         args.output_op_path, args.output_arg_map_path)
