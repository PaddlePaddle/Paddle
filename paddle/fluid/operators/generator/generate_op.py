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
    to_composite_grad_opmaker_name,
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
    is_composite_op,
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
env.filters["to_composite_grad_opmaker_name"] = to_composite_grad_opmaker_name
env.tests["base_op"] = is_base_op
env.tests["composite_op"] = is_composite_op
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
                        'int[]'
                        if 'data_type' in int_array_config
                        and int_array_config['data_type'] == 'int'
                        else 'int64_t[]'
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


def parse_composite_info(ops, backward_ops, backward_op_dict):
    for op in ops:
        if "backward" in op:
            op["phi_backward"] = op["backward"]
    for backward_op in backward_ops:
        if "backward" in backward_op:
            backward_op["phi_backward"] = backward_op["backward"]
    for backward_op_name, op_dict in backward_op_dict.items():
        if "composite" not in op_dict:
            continue
        op_dict["composite"]["phi_inputs"] = []
        op_dict["composite"]["phi_attrs"] = []
        op_dict["composite"]["phi_outputs"] = []
        for input in op_dict["inputs"]:
            op_dict["composite"]["phi_inputs"].append(input['name'])
        for attr in op_dict["attrs"]:
            op_dict["composite"]["phi_attrs"].append(attr['name'])
        for output in op_dict["outputs"]:
            op_dict["composite"]["phi_outputs"].append(output['name'])


# replace name of op and params for OpMaker
def replace_compat_name(op_fluid_map_list, forward_op_dict, backward_op_dict):
    def get_phi_and_fluid_op_name(op_item):
        names = op_item.split('(')
        if len(names) == 1:
            return names[0].strip(), names[0].strip()
        else:
            return names[0].strip(), names[1].split(')')[0].strip()

    def update_op_param_name(op_args, args_alias_map):
        for item in op_args:
            if item['name'] in args_alias_map:
                item['name'] = args_alias_map[item['name']]

    def update_grad_args_name(op_args, args_alias_map):
        for item in op_args:
            if (
                item['name'].endswith('_grad')
                and item['name'][:-5] in args_alias_map
            ):
                args_alias_map[item['name']] = (
                    args_alias_map[item['name'][:-5]] + '_grad'
                )
                item['name'] = args_alias_map[item['name'][:-5]] + '_grad'

    def add_fluid_info_in_composite(composite_map, args_alias_map):
        fluid_input_list = []
        fluid_attr_list = []
        fluid_output_list = []
        # add fluid op inputs
        for input in composite_map["phi_inputs"]:
            if input in args_alias_map:
                fluid_input_list.append(args_alias_map[input])
            else:
                fluid_input_list.append(input)
        # add fluid op attrs
        for attr in composite_map["phi_attrs"]:
            if attr in args_alias_map:
                fluid_attr_list.append(args_alias_map[attr])
            else:
                fluid_attr_list.append(attr)
        # add fluid op outputs
        for output in composite_map["phi_outputs"]:
            if output in args_alias_map:
                fluid_output_list.append(args_alias_map[output])
            else:
                fluid_output_list.append(output)

        composite_map.update(
            {
                "fluid_inputs": fluid_input_list,
                "fluid_attrs": fluid_attr_list,
                "fluid_outputs": fluid_output_list,
            }
        )

    def get_param_list_alias(param_list, args_map):
        return [
            args_map[param] if param in args_map else param
            for param in param_list
        ]

    def update_common_params_name(
        op_item, args_name_map, scalar_configs, int_array_configs
    ):
        if 'inplace' in op_item and op_item['inplace']:
            inplace_map = {}
            for key, val in op_item['inplace'].items():
                if key in args_map:
                    key = args_map[key]
                if val in args_map:
                    val = args_map[val]
                inplace_map[key] = val
            op_item['inplace'] = inplace_map
        if 'no_need_buffer' in op_item and op_item['no_need_buffer']:
            op_item['no_need_buffer'] = get_param_list_alias(
                op_item['no_need_buffer'], args_map
            )

        process_scalar(op_item, scalar_configs)
        process_int_array(op_item, int_array_configs)

        if 'invoke' in op_item:
            op_item['invoke']['args'] = [
                args_map[param.strip()]
                if param.strip() in args_map
                else param.strip()
                for param in op_item['invoke']['args'].split(',')
            ]
            return
        op_item['infer_meta']['param'] = get_param_list_alias(
            op_item['infer_meta']['param'], args_name_map
        )
        op_item['kernel']['param'] = get_param_list_alias(
            op_item['kernel']['param'], args_name_map
        )
        if op_item['kernel']['data_type']:
            op_item['kernel']['data_type']['candidates'] = get_param_list_alias(
                op_item['kernel']['data_type']['candidates'], args_name_map
            )
        if op_item['kernel']['backend']:
            op_item['kernel']['backend']['candidates'] = get_param_list_alias(
                op_item['kernel']['backend']['candidates'], args_name_map
            )
        if op_item['kernel']['layout']:
            op_item['kernel']['layout']['candidates'] = get_param_list_alias(
                op_item['kernel']['layout']['candidates'], args_name_map
            )

    def update_grad_op_compat_name(grad_op_item, args_name_map):
        update_op_param_name(grad_op_item['inputs'], args_name_map)
        update_op_param_name(grad_op_item['outputs'], args_name_map)
        update_op_param_name(grad_op_item['attrs'], args_name_map)
        update_op_param_name(grad_op_item['forward']['inputs'], args_name_map)
        update_op_param_name(grad_op_item['forward']['outputs'], args_name_map)
        update_op_param_name(grad_op_item['forward']['attrs'], args_name_map)
        update_grad_args_name(grad_op_item['inputs'], args_map)
        update_grad_args_name(grad_op_item['outputs'], args_map)

    for op_args in op_fluid_map_list:
        new_op_name, op_name = get_phi_and_fluid_op_name(op_args['op'])
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
        if 'extra' in op_args and 'outputs' in op_args['extra']:
            for out_item in forward_op_item['outputs']:
                if out_item['name'] in op_args['extra']['outputs']:
                    out_item['is_extra'] = True

        key_set = ['inputs', 'attrs', 'outputs']
        args_map = {}
        for key in key_set:
            if key in op_args:
                args_map.update(op_args[key])
                for args_item in forward_op_item[key]:
                    if args_item['name'] in op_args[key]:
                        if (
                            scalar_configs
                            and args_item['name'] in scalar_configs
                        ):
                            scalar_configs[
                                op_args[key][args_item['name']]
                            ] = scalar_configs[args_item['name']]
                        if (
                            int_array_configs
                            and args_item['name'] in int_array_configs
                        ):
                            int_array_configs[
                                op_args[key][args_item['name']]
                            ] = int_array_configs[args_item['name']]
                        args_item['name'] = op_args[key][args_item['name']]
                if has_backward:
                    for args_item in backward_op_item['forward'][key]:
                        if args_item['name'] in op_args[key]:
                            args_item['name'] = op_args[key][args_item['name']]
        forward_op_item["attr_dict"] = to_named_dict(forward_op_item["attrs"])
        update_common_params_name(
            forward_op_item, args_map, scalar_configs, int_array_configs
        )

        if has_backward:
            update_grad_op_compat_name(backward_op_item, args_map)
            update_common_params_name(
                backward_op_item, args_map, scalar_configs, int_array_configs
            )
            backward_op_item["attr_dict"] = to_named_dict(
                backward_op_item["attrs"]
            )

            if 'backward' not in op_args:
                continue

            backward_op_list = op_args['backward'].split(',')
            # add fluid args name in composite map
            for backward_op in backward_op_list:
                if (
                    "composite"
                    in backward_op_dict[backward_op.split('(')[0].strip()]
                ):
                    add_fluid_info_in_composite(
                        backward_op_dict[backward_op]["composite"], args_map
                    )
            _, bw_op_name = get_phi_and_fluid_op_name(backward_op_list[0])
            forward_op_item['backward'] = bw_op_name
            backward_op_item['op_name'] = bw_op_name

            # for double grad
            if len(backward_op_list) > 1:
                (
                    phi_double_grad_op_name,
                    double_grad_op_name,
                ) = get_phi_and_fluid_op_name(backward_op_list[1])
                double_grad_item = backward_op_dict[phi_double_grad_op_name]
                backward_op_item['backward'] = double_grad_op_name
                double_grad_item['op_name'] = double_grad_op_name
                update_grad_op_compat_name(double_grad_item, args_map)
                update_common_params_name(
                    double_grad_item,
                    args_map,
                    scalar_configs,
                    int_array_configs,
                )
                double_grad_item["attr_dict"] = to_named_dict(
                    double_grad_item["attrs"]
                )

                # for triple grad
                if len(backward_op_list) > 2:
                    (
                        phi_triple_grad_op_name,
                        triple_grad_op_name,
                    ) = get_phi_and_fluid_op_name(backward_op_list[2])
                    triple_grad_item = backward_op_dict[phi_triple_grad_op_name]
                    double_grad_item['backward'] = triple_grad_op_name
                    triple_grad_item['op_name'] = triple_grad_op_name
                    update_grad_op_compat_name(triple_grad_item, args_map)
                    update_common_params_name(
                        triple_grad_item,
                        args_map,
                        scalar_configs,
                        int_array_configs,
                    )
                    triple_grad_item["attr_dict"] = to_named_dict(
                        triple_grad_item["attrs"]
                    )


def process_invoke_op(forward_op_dict, backward_op_dict):
    for bw_op in backward_op_dict.values():
        if 'invoke' in bw_op:
            invoke_op = bw_op['invoke']['func']
            args_list = bw_op['invoke']['args']
            args_index = 0
            if invoke_op in forward_op_dict:
                reuse_op = forward_op_dict[invoke_op]
                bw_op['invoke']['func'] = reuse_op['op_name']
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


def parse_drop_empty_grad(op_fluid_list: list, bw_op_dict: dict):
    for op_op in op_fluid_list:
        if 'drop_empty_grad' in op_op:
            bw_names = [
                bw_name.split('(')[0].strip()
                for bw_name in op_op['backward'].split(',')
            ]
            for bw_name in bw_names:
                assert (
                    bw_name in bw_op_dict
                ), f"backward {bw_name} is not existed"
                for out_grad in op_op['drop_empty_grad']:
                    assert (
                        out_grad in bw_op_dict[bw_name]['output_dict']
                    ), f'''
                         {bw_name} with {out_grad} is not existed in output_dict '''
                    bw_op_dict[bw_name]['output_dict'][out_grad][
                        'drop_empty_grad'
                    ] = False


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
        op_fluid_map_list = yaml.safe_load(f)

    for op in ops:
        op['op_name'] = op['name']
    for bw_op in backward_ops:
        bw_op['op_name'] = bw_op['name']
        for bw_output in bw_op['outputs']:
            bw_output['drop_empty_grad'] = True

    # deal the drop_empty_grad of bw_op by op_compat.yaml
    parse_drop_empty_grad(op_fluid_map_list, backward_op_dict)

    parse_composite_info(ops, backward_ops, backward_op_dict)

    replace_compat_name(op_fluid_map_list, forward_op_dict, backward_op_dict)

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
            ops=ops,
            backward_ops=backward_ops,
            op_dict=op_dict,
            composite_gen_flag=True,
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
