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
from pathlib import Path

import yaml
from filters import (
    cartesian_prod_mapping,
    to_input_name,
    to_int_array_tensor_name,
    to_int_array_tensors_name,
    to_op_attr_type,
    to_opmaker_name,
    to_opmaker_name_cstr,
    to_pascal_case,
    to_scalar_tensor_name,
)
from jinja2 import Environment, FileSystemLoader, StrictUndefined
from parse_utils import to_named_dict
from tests import (
    is_base_op,
    is_initializer_list,
    is_scalar,
    is_vec,
    supports_inplace,
    supports_no_need_buffer,
)

file_loader = FileSystemLoader(Path(__file__).parent / "templates")
env = Environment(
    loader=file_loader,
    keep_trailing_newline=True,
    trim_blocks=True,
    lstrip_blocks=True,
    undefined=StrictUndefined,
    extensions=['jinja2.ext.do'],
)
env.filters["to_op_attr_type"] = to_op_attr_type
env.filters["to_opmaker_name"] = to_opmaker_name
env.filters["to_pascal_case"] = to_pascal_case
env.filters["to_scalar_tensor_name"] = to_scalar_tensor_name
env.filters["to_int_array_tensor_name"] = to_int_array_tensor_name
env.filters["to_int_array_tensors_name"] = to_int_array_tensors_name
env.filters["to_input_name"] = to_input_name
env.filters["to_opmaker_name_cstr"] = to_opmaker_name_cstr
env.filters["cartesian_prod_mapping"] = cartesian_prod_mapping
env.tests["base_op"] = is_base_op
env.tests["vec"] = is_vec
env.tests["scalar"] = is_scalar
env.tests["initializer_list"] = is_initializer_list
env.tests["supports_inplace"] = supports_inplace
env.tests["supports_no_need_buffer"] = supports_no_need_buffer


def restruct_io(op):
    op["input_dict"] = to_named_dict(op["inputs"])
    op["attr_dict"] = to_named_dict(op["attrs"])
    op["output_dict"] = to_named_dict(op["outputs"])
    return op


def process_scalar(op_item, scalar_configs):
    scalar_map = {
        'Scalar': 'float',
        'Scalar(float)': 'float',
        'Scalar(int)': 'int',
        'Scalar(int64_t)': 'int64_t',
    }
    if scalar_configs is not None:
        for attr_item in op_item['attrs']:
            if attr_item['name'] in scalar_configs:
                attr_type = attr_item['typename']
                assert (
                    attr_type in scalar_map
                ), f"{op_item['name']}'s scalar in op_compat.yaml is error, the data_type of {attr_item['name']} is expected to be one of Scalar, Scalar(float), Scalar(int) or Scalar(int64_t), but now is {attr_type}."

                scalar_config = scalar_configs[attr_item['name']]
                attr_item['is_support_tensor'] = (
                    True
                    if 'support_tensor' in scalar_config
                    and scalar_config['support_tensor']
                    else False
                )
                if attr_item['is_support_tensor']:
                    attr_item['typename'] = (
                        scalar_config['data_type']
                        if 'data_type' in scalar_config
                        else scalar_map[attr_type]
                    )
                else:
                    attr_item['data_type'] = (
                        scalar_config['data_type']
                        if 'data_type' in scalar_config
                        else scalar_map[attr_type]
                    )
                    attr_item['tensor_name'] = scalar_config['tensor_name']


def process_int_array(op_item, int_array_configs):
    data_type_map = {
        'int': 'std::vector<int>',
        'int64_t': 'std::vector<int64_t>',
    }
    if int_array_configs is not None:
        for attr_item in op_item['attrs']:
            if attr_item['name'] in int_array_configs:
                attr_type = attr_item['typename']
                assert (
                    attr_item['typename'] == "IntArray"
                ), f"{op_item['name']}'s int_array in op_compat.yaml is error, the data_type of {attr_item['name']} is expected to be one of IntArray, but now is {attr_type}."

                int_array_config = int_array_configs[attr_item['name']]
                attr_item['is_support_tensor'] = (
                    True
                    if 'support_tensor' in int_array_config
                    and int_array_config['support_tensor']
                    else False
                )
                if attr_item['is_support_tensor']:
                    attr_item['typename'] = (
                        data_type_map[int_array_config['data_type']]
                        if 'data_type' in int_array_config
                        else 'std::vector<int64_t>'
                    )
                else:
                    attr_item['data_type'] = (
                        data_type_map[int_array_config['data_type']]
                        if 'data_type' in int_array_config
                        else 'std::vector<int64_t>'
                    )
                    attr_item['manual_flag'] = True
                    if 'tensor_name' in int_array_config:
                        attr_item['tensor_name'] = int_array_config[
                            'tensor_name'
                        ]
                    if 'tensors_name' in int_array_config:
                        attr_item['tensors_name'] = int_array_config[
                            'tensors_name'
                        ]


# replace name of op and params for OpMaker
def replace_compat_name(op_op_map, forward_op_dict, backward_op_dict):
    def get_op_and_op_name(op_item):
        names = op_item.split('(')
        if len(names) == 1:
            return names[0].strip(), names[0].strip()
        else:
            return names[0].strip(), names[1].split(')')[0].strip()

    def update_op_attr_name(attrs, attrs_alias_map):
        for attr_item in attrs:
            if attr_item['name'] in attrs_alias_map:
                attr_item['name'] = attrs_alias_map[attr_item['name']]

    for op_args in op_op_map:
        new_op_name, op_name = get_op_and_op_name(op_args['op'])
        if new_op_name not in forward_op_dict:
            continue
        forward_op_item = forward_op_dict[new_op_name]
        has_backward = True if forward_op_item['backward'] else False
        if has_backward:
            backward_op_item = backward_op_dict[forward_op_item['backward']]
        if new_op_name != op_name:
            forward_op_item['op_name'] = op_name

        scalar_configs = None
        int_array_configs = None

        if 'scalar' in op_args:
            scalar_configs = op_args['scalar']
        if 'int_array' in op_args:
            int_array_configs = op_args['int_array']

        process_scalar(forward_op_item, scalar_configs)
        process_int_array(forward_op_item, int_array_configs)

        if 'backward' in op_args and has_backward:
            backward_op_list = op_args['backward'].split(',')
            _, bw_op_name = get_op_and_op_name(backward_op_list[0])
            forward_op_item['backward'] = bw_op_name
            backward_op_item['op_name'] = bw_op_name

            process_scalar(backward_op_item, scalar_configs)
            process_int_array(backward_op_item, int_array_configs)

            # for double grad
            if len(backward_op_list) > 1:
                (
                    new_double_grad_op_name,
                    double_grad_op_name,
                ) = get_op_and_op_name(backward_op_list[1])
                double_grad_item = backward_op_dict[new_double_grad_op_name]
                backward_op_item['backward'] = double_grad_op_name
                double_grad_item['op_name'] = double_grad_op_name
                if 'attrs' in op_args:
                    update_op_attr_name(
                        double_grad_item['attrs'], op_args['attrs']
                    )
                    update_op_attr_name(
                        double_grad_item['forward']['attrs'], op_args['attrs']
                    )

                process_scalar(double_grad_item, scalar_configs)
                process_int_array(double_grad_item, int_array_configs)

                # for triple grad
                if len(backward_op_list) > 2:
                    (
                        new_triple_grad_op_name,
                        triple_grad_op_name,
                    ) = get_op_and_op_name(backward_op_list[2])
                    triple_grad_item = backward_op_dict[new_triple_grad_op_name]
                    double_grad_item['backward'] = triple_grad_op_name
                    triple_grad_item['op_name'] = triple_grad_op_name
                    if 'attrs' in op_args:
                        update_op_attr_name(
                            triple_grad_item['attrs'], op_args['attrs']
                        )
                        update_op_attr_name(
                            triple_grad_item['forward']['attrs'],
                            op_args['attrs'],
                        )

                    process_scalar(triple_grad_item, scalar_configs)
                    process_int_array(triple_grad_item, int_array_configs)

        key_set = ['inputs', 'attrs', 'outputs']
        args_map = {}
        for key in key_set:
            if key in op_args:
                args_map.update(op_args[key])
                for args_item in forward_op_item[key]:
                    if args_item['name'] in op_args[key]:
                        args_item['name'] = op_args[key][args_item['name']]
                if has_backward:
                    for args_item in backward_op_item['forward'][key]:
                        if args_item['name'] in op_args[key]:
                            args_item['name'] = op_args[key][args_item['name']]
        forward_op_item['infer_meta']['param'] = [
            args_map[param] if param in args_map else param
            for param in forward_op_item['infer_meta']['param']
        ]
        forward_op_item['kernel']['param'] = [
            args_map[param] if param in args_map else param
            for param in forward_op_item['kernel']['param']
        ]
        if forward_op_item['kernel']['data_type']:
            forward_op_item['kernel']['data_type']['candidates'] = [
                args_map[param] if param in args_map else param
                for param in forward_op_item['kernel']['data_type'][
                    'candidates'
                ]
            ]
        if forward_op_item['kernel']['backend']:
            forward_op_item['kernel']['backend']['candidates'] = [
                args_map[param] if param in args_map else param
                for param in forward_op_item['kernel']['backend']['candidates']
            ]
        if forward_op_item['kernel']['layout']:
            forward_op_item['kernel']['layout']['candidates'] = [
                args_map[param] if param in args_map else param
                for param in forward_op_item['kernel']['layout']['candidates']
            ]
        if forward_op_item['inplace']:
            inplace_map = {}
            for key, val in forward_op_item['inplace'].items():
                if key in args_map:
                    key = args_map[key]
                if val in args_map:
                    val = args_map[val]
                inplace_map[key] = val
            forward_op_item['inplace'] = inplace_map

        if has_backward:
            for args_item in backward_op_item['inputs']:
                if args_item['name'] in args_map:
                    args_item['name'] = args_map[args_item['name']]
                elif (
                    args_item['name'].endswith('_grad')
                    and args_item['name'][:-5] in args_map
                ):
                    args_map[args_item['name']] = (
                        args_map[args_item['name'][:-5]] + '_grad'
                    )
                    args_item['name'] = args_map[args_item['name']]
            for args_item in backward_op_item['attrs']:
                if args_item['name'] in args_map:
                    args_item['name'] = args_map[args_item['name']]
            for args_item in backward_op_item['outputs']:
                if (
                    args_item['name'].endswith('_grad')
                    and args_item['name'][:-5] in args_map
                ):
                    args_map[args_item['name']] = (
                        args_map[args_item['name'][:-5]] + '_grad'
                    )
                    args_item['name'] = args_map[args_item['name']]

            if 'invoke' in backward_op_item:
                backward_op_item['invoke']['args'] = [
                    args_map[param.strip()]
                    if param.strip() in args_map
                    else param.strip()
                    for param in backward_op_item['invoke']['args'].split(',')
                ]
                continue

            backward_op_item['infer_meta']['param'] = [
                args_map[param] if param in args_map else param
                for param in backward_op_item['infer_meta']['param']
            ]
            backward_op_item['kernel']['param'] = [
                args_map[param] if param in args_map else param
                for param in backward_op_item['kernel']['param']
            ]
            if backward_op_item['kernel']['data_type']:
                backward_op_item['kernel']['data_type']['candidates'] = [
                    args_map[param] if param in args_map else param
                    for param in backward_op_item['kernel']['data_type'][
                        'candidates'
                    ]
                ]
            if backward_op_item['kernel']['backend']:
                backward_op_item['kernel']['backend']['candidates'] = [
                    args_map[param] if param in args_map else param
                    for param in backward_op_item['kernel']['backend'][
                        'candidates'
                    ]
                ]
            if backward_op_item['kernel']['layout']:
                backward_op_item['kernel']['layout']['candidates'] = [
                    args_map[param] if param in args_map else param
                    for param in backward_op_item['kernel']['layout'][
                        'candidates'
                    ]
                ]
            if backward_op_item['no_need_buffer']:
                backward_op_item['no_need_buffer'] = [
                    args_map[param] if param in args_map else param
                    for param in backward_op_item['no_need_buffer']
                ]
            if backward_op_item['inplace']:
                inplace_map = {}
                for key, val in backward_op_item['inplace'].items():
                    if key in args_map:
                        key = args_map[key]
                    if val in args_map:
                        val = args_map[val]
                    inplace_map[key] = val
                backward_op_item['inplace'] = inplace_map


def process_invoke_op(forward_op_dict, backward_op_dict):
    for bw_op in backward_op_dict.values():
        if 'invoke' in bw_op:
            invoke_op = bw_op['invoke']['func']
            args_list = bw_op['invoke']['args']
            args_index = 0
            if invoke_op in forward_op_dict:
                reuse_op = forward_op_dict[invoke_op]
                bw_op['invoke']['inputs'] = []
                bw_op['invoke']['attrs'] = []
                bw_op['invoke']['outputs'] = []
                for input_item in reuse_op['inputs']:
                    bw_op['invoke']['inputs'].append(
                        {
                            'name': input_item['name'],
                            'value': args_list[args_index],
                        }
                    )
                    args_index = args_index + 1
                for attr in reuse_op['attrs']:
                    if args_index < len(args_list):
                        attr_value = (
                            f"this->GetAttr(\"{args_list[args_index]}\")"
                            if args_list[args_index] in bw_op['attr_dict']
                            else args_list[args_index]
                        )
                        bw_op['invoke']['attrs'].append(
                            {'name': attr['name'], 'value': attr_value}
                        )
                        args_index = args_index + 1
                    else:
                        break
                for idx, output_item in enumerate(reuse_op['outputs']):
                    bw_op['invoke']['outputs'].append(
                        {
                            'name': output_item['name'],
                            'value': bw_op['outputs'][idx]['name'],
                        }
                    )


def main(
    ops_yaml_path,
    backward_yaml_path,
    op_compat_yaml_path,
    op_version_yaml_path,
    output_op_path,
    output_arg_map_path,
):
    with open(ops_yaml_path, "rt") as f:
        ops = yaml.safe_load(f)
        ops = [restruct_io(op) for op in ops]
    forward_op_dict = to_named_dict(ops)

    with open(backward_yaml_path, "rt") as f:
        backward_ops = yaml.safe_load(f)
        backward_ops = [restruct_io(op) for op in backward_ops]
    backward_op_dict = to_named_dict(backward_ops)

    with open(op_version_yaml_path, "rt") as f:
        op_versions = yaml.safe_load(f)
    # add op version info into op
    for op_version in op_versions:
        forward_op_dict[op_version['op']]['version'] = op_version['version']

    with open(op_compat_yaml_path, "rt") as f:
        op_op_map = yaml.safe_load(f)

    for op in ops:
        op['op_name'] = op['name']
    for bw_op in backward_ops:
        bw_op['op_name'] = bw_op['name']

    replace_compat_name(op_op_map, forward_op_dict, backward_op_dict)

    # prepare for invoke case
    process_invoke_op(forward_op_dict, backward_op_dict)

    # fill backward field for an op if another op claims it as forward
    for name, backward_op in backward_op_dict.items():
        forward_name = backward_op["forward"]["name"]
        if forward_name in backward_op_dict:
            forward_op = backward_op_dict[forward_name]
            if forward_op["backward"] is None:
                forward_op["backward"] = name

    op_dict = {}
    op_dict.update(forward_op_dict)
    op_dict.update(backward_op_dict)

    if len(ops) == 0 and len(backward_ops) == 0:
        if os.path.isfile(output_op_path):
            os.remove(output_op_path)
        if os.path.isfile(output_arg_map_path):
            os.remove(output_arg_map_path)
        return

    op_template = env.get_template('op.c.j2')
    with open(output_op_path, "wt") as f:
        msg = op_template.render(
            ops=ops, backward_ops=backward_ops, op_dict=op_dict
        )
        f.write(msg)

    ks_template = env.get_template('ks.c.j2')
    with open(output_arg_map_path, 'wt') as f:
        msg = ks_template.render(ops=ops, backward_ops=backward_ops)
        f.write(msg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate operator file from op yaml."
    )
    parser.add_argument(
        '--ops_yaml_path', type=str, help="parsed ops yaml file."
    )
    parser.add_argument(
        '--backward_yaml_path', type=str, help="parsed backward ops yaml file."
    )
    parser.add_argument(
        '--op_compat_yaml_path', type=str, help="ops args compat yaml file."
    )
    parser.add_argument(
        '--op_version_yaml_path', type=str, help="ops version yaml file."
    )
    parser.add_argument(
        "--output_op_path", type=str, help="path to save generated operators."
    )
    parser.add_argument(
        "--output_arg_map_path",
        type=str,
        help="path to save generated argument mapping functions.",
    )

    args = parser.parse_args()
    main(
        args.ops_yaml_path,
        args.backward_yaml_path,
        args.op_compat_yaml_path,
        args.op_version_yaml_path,
        args.output_op_path,
        args.output_arg_map_path,
    )
