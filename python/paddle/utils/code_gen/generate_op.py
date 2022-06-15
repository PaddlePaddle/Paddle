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
from filters import to_input_name
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
env.tests["base_api"] = is_base_api
env.tests["vec"] = is_vec
env.tests["scalar"] = is_scalar
env.tests["initializer_list"] = is_initializer_list
env.tests["supports_inplace"] = supports_inplace
env.tests["supports_no_need_buffer"] = supports_no_need_buffer


def main(api_yaml_path, backward_yaml_path, api_args_compat_yaml_path,
         api_version_yaml_path, output_op_path, output_arg_map_path):
    with open(api_yaml_path, "rt") as f:
        apis = yaml.safe_load(f)
    forward_api_dict = to_named_dict(apis)

    with open(backward_yaml_path, "rt") as f:
        backward_apis = yaml.safe_load(f)
    backward_api_dict = to_named_dict(backward_apis)

    with open(api_version_yaml_path, "rt") as f:
        api_versions = yaml.safe_load(f)
    # add api version info into api
    for api_version in api_versions:
        forward_api_dict[api_version['api']]['version'] = api_version['version']

    with open(api_args_compat_yaml_path, "rt") as f:
        api_args_map = yaml.safe_load(f)
    # replace args name for OpMaker
    for api_args in api_args_map:
        forward_api_item = forward_api_dict[api_args['api']]
        has_backward = True if forward_api_item['backward'] else False
        if has_backward:
            backward_api_item = backward_api_dict[forward_api_item['backward']]
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
        if forward_api_item['inplace']:
            inplace_map = {}
            for key, val in forward_api_item['inplace'].items():
                if key in args_map:
                    key = args_map[key]
                if val in args_map:
                    val = args_map[val]
                inplace_map[key] = val
            forward_api_item['inplace'] = inplace_map

        if has_backward:
            for args_item in backward_api_item['inputs']:
                if args_item['name'] in args_map:
                    args_item['name'] = args_map[args_item['name']]
                elif args_item['name'].endswith(
                        '_grad') and args_item['name'][:-5] in args_map:
                    args_item['name'] = args_map[args_item['name']
                                                 [:-5]] + '_grad'
            for args_item in backward_api_item['attrs']:
                if args_item['name'] in args_map:
                    args_item['name'] = args_map[args_item['name']]
            for args_item in backward_api_item['outputs']:
                if args_item['name'].endswith(
                        '_grad') and args_item['name'][:-5] in args_map:
                    args_item['name'] = args_map[args_item['name']
                                                 [:-5]] + '_grad'
            backward_api_item['infer_meta']['param'] = [
                args_map[param] if param in args_map else param
                for param in backward_api_item['infer_meta']['param']
            ]
            backward_api_item['kernel']['param'] = [
                args_map[param] if param in args_map else param
                for param in backward_api_item['kernel']['param']
            ]
            if backward_api_item['no_need_buffer']:
                backward_api_item['no_need_buffer'] = [
                    args_map[param] if param in args_map else param
                    for param in backward_api_item['no_need_buffer']
                ]

    # fill backward field for an api if another api claims it as forward
    for name, backward_api in backward_api_dict.items():
        forward_name = backward_api["forward"]["name"]
        if forward_name in backward_api_dict:
            forward_api = backward_api_dict[forward_name]
            if forward_api["backward"] is None:
                forward_api["backward"] = name

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
    parser.add_argument('--api_args_compat_yaml_path',
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
         args.api_args_compat_yaml_path, args.api_version_yaml_path,
         args.output_op_path, args.output_arg_map_path)
