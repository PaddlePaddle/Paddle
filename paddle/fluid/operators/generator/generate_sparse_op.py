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
from generate_op import process_invoke_op
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


SPARSE_OP_PREFIX = 'sparse_'


def main(op_yaml_path, backward_yaml_path, output_op_path, output_arg_map_path):
    with open(op_yaml_path, "rt") as f:
        ops = yaml.safe_load(f)
        ops = [restruct_io(op) for op in ops]
    forward_op_dict = to_named_dict(ops)

    with open(backward_yaml_path, "rt") as f:
        backward_ops = yaml.safe_load(f)
        backward_ops = [restruct_io(op) for op in backward_ops]
    backward_op_dict = to_named_dict(backward_ops)

    for op in ops:
        if op['name'][-1] == '_':
            op['name'] = op['name'][:-1]
        op['op_name'] = SPARSE_OP_PREFIX + op['name']
        op['name'] = op['op_name']
        if op["backward"] is not None:
            op["backward"] = SPARSE_OP_PREFIX + op["backward"]
    for bw_op in backward_ops:
        bw_op['op_name'] = SPARSE_OP_PREFIX + bw_op['name']
        bw_op['name'] = bw_op['op_name']
        if 'invoke' in bw_op:
            bw_op['invoke']['args'] = [
                param.strip() for param in bw_op['invoke']['args'].split(',')
            ]

    # prepare for invoke case
    process_invoke_op(forward_op_dict, backward_op_dict)
    for bw_op in backward_ops:
        if 'invoke' in bw_op:
            if bw_op['invoke']['func'] in forward_op_dict:
                bw_op['invoke']['func'] = (
                    SPARSE_OP_PREFIX + bw_op['invoke']['func']
                )

    # fill backward field for an op if another op claims it as forward
    for name, backward_op in backward_op_dict.items():
        forward_name = backward_op["forward"]["name"]
        if forward_name in backward_op_dict:
            forward_op = backward_op_dict[forward_name]
            if forward_op["backward"] is None:
                forward_op["backward"] = name
            forward_op["backward"] = SPARSE_OP_PREFIX + forward_op["backward"]

    op_dict = {}
    op_dict.update(forward_op_dict)
    op_dict.update(backward_op_dict)

    if len(ops) == 0 and len(backward_ops) == 0:
        if os.path.isfile(output_op_path):
            os.remove(output_op_path)
        if os.path.isfile(output_arg_map_path):
            os.remove(output_arg_map_path)
        return

    op_template = env.get_template('sparse_op.c.j2')
    with open(output_op_path, "wt") as f:
        msg = op_template.render(
            ops=ops, backward_ops=backward_ops, op_dict=op_dict
        )
        f.write(msg)

    ks_template = env.get_template('sparse_ks.c.j2')
    with open(output_arg_map_path, 'wt') as f:
        msg = ks_template.render(ops=ops, backward_ops=backward_ops)
        f.write(msg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate operator file from op yaml."
    )
    parser.add_argument(
        '--ops_yaml_path', type=str, help="parsed sparse ops yaml file."
    )
    parser.add_argument(
        '--backward_ops_yaml_path',
        type=str,
        help="parsed backward sparse ops yaml file.",
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
        args.backward_ops_yaml_path,
        args.output_op_path,
        args.output_arg_map_path,
    )
