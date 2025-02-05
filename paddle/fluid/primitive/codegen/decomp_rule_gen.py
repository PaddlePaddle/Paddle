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
import hashlib
import pathlib
import sys

import jinja2
import yaml

# fmt: off
# import from paddle/fluid/operators/generator
sys.path.append(
    str(pathlib.Path(__file__).resolve().parents[2] / 'operators/generator')
)
import filters as op_gen_filters
import tests_utils as op_gen_tests
from decomp_vjp_gen import extend_compat_info, filter_compat_info
from parse_utils import to_named_dict
from type_mapping import output_type_map

# import from paddle/fluid/pir/dialect/op_generator/api_gen.py
sys.path.append(
    str(pathlib.Path(__file__).resolve().parents[2] / 'pir/dialect/op_generator')
)

from decomp_interface_gen_op_list import (
    decomp_ops_contain_unused_output,
    decomp_rule_interface_implementation_gen_op_list,
)
from op_gen import attr_types_map, to_pascal_case

# fmt: on


def load(path: pathlib.Path):
    """Load config from yaml file.

    Args:
        path (pathlib.Path): The path of yaml config.

    Returns:
        dict: The config info.

    """
    with open(path, 'rt') as f:
        return yaml.safe_load(f)


def render(src_dir: pathlib.Path, dst_dir: pathlib.Path, *args, **kwargs):
    """Render and save Jinja2 templates to the destination directory.

    Args:
        src_dir (pathlib.Path): The source directory containing Jinja2 templates.
        dst_dir (pathlib.Path): The destination directory to save rendered files.
        *args: Additional positional arguments passed to the `render` function.
        **kwargs: Additional keyword arguments passed to the `render` function.

    Returns:
        None
    """
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(src_dir),
        keep_trailing_newline=True,
        trim_blocks=True,
        lstrip_blocks=True,
        undefined=jinja2.StrictUndefined,
        extensions=['jinja2.ext.do'],
    )
    env.filters.update(
        {
            'to_paddle_attr_type': op_gen_filters.to_paddle_attr_type,
            'to_paddle_input_type': op_gen_filters.to_paddle_input_type,
            'to_paddle_output_type': op_gen_filters.to_paddle_output_type,
            'trip_intermediate': op_gen_filters.filter_intermediate,
        }
    )
    env.tests.update(
        {
            'scalar': op_gen_tests.is_scalar,
            'intarray': op_gen_tests.is_intarray,
            'datatype': op_gen_tests.is_datatype,
            'exist_mutable_attribute': op_gen_tests.exist_mutable_attribute,
            'mutable_attribute': op_gen_tests.is_mutable_attribute,
            'only_composite_op': op_gen_tests.is_only_composite_op,
        }
    )

    decomp_temp = "decomp/generated_decomp.j2"
    save(
        env.get_template(decomp_temp).render(*args, **kwargs),
        pathlib.Path(dst_dir),
    )


def save(content: str, path: pathlib.Path):
    """Saves the given string contents to a file in the specified path.

    Args:
        content (str): The string content that needs to be saved.
        path (pathlib.Path): The path to save the file, a Pathlib path object

    Returns:
        None
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    dst_content = ''
    if path.is_file():
        with open(path, 'r') as f:
            dst_content = f.read()

    if (
        hashlib.md5(content.encode("UTF-8")).hexdigest()
        != hashlib.md5(dst_content.encode("UTF-8")).hexdigest()
    ):
        with open(path, 'w') as f:
            f.write(content)


def process_optional_output_info(apis):
    for api in apis:
        inputs_dict = to_named_dict(api['inputs'])
        for output in api['outputs']:
            if (
                api.get("inplace", None)
                and output['name'] in api['inplace']
                and inputs_dict[api['inplace'][output['name']]]['optional']
            ):
                output['optional'] = True
            else:
                output['optional'] = False


def gen(
    fwd_path: pathlib.Path,
    compat_path: pathlib.Path,
    fwd_pd_op_path: pathlib.Path,
    templates_dir: pathlib.Path,
    destination_dir: pathlib.Path,
):
    """The `gen` load jinja2 templates and relative config info, use jinja2
    templating engine to generate c++ code, and save the code into destination.

    Args:
        prim_path (pathlib.Path): The YAML file path of the primitive API.
        fwd_path (pathlib.Path):  The YAML file path of the forward API.
        rev_path (pathlib.Path): The YAML file path of the backward API.
        compat_path: (pathlib.Path): The YAML file path of the ops compat.
        fwd_pd_op_path (pathlib.Path): The YAML file path of the ir forward API.
        rev_pd_op_path (pathlib.Path): The YAML file path of the ir backward API.
        templates_dir (pathlib.Path): The directory of the templates.
        destination_dir (pathlib.Path): The Directory of the generated file.

    Returns:
        None
    """
    (
        fwds,
        compats,
        ir_fwds,
    ) = (
        load(fwd_path),
        load(compat_path),
        load(fwd_pd_op_path),
    )
    filter_compat_info(compats)
    apis = [
        {**api, **{'class_name': to_pascal_case(api["name"]) + "Op"}}
        for api in fwds + ir_fwds
    ]

    apis = extend_compat_info(apis, compats)

    process_optional_output_info(apis)

    for item in apis:
        for attr_item in item["attrs"]:
            if attr_item["typename"] not in attr_types_map.keys():
                raise TypeError
            attr_item["mapped_type"] = attr_types_map[attr_item["typename"]]
        for out_item in item["outputs"]:
            if out_item["typename"] not in output_type_map.keys():
                name = out_item["typename"]
                raise TypeError(f"err type {name}")
            if out_item["optional"]:
                out_item["mapped_type"] = (
                    "paddle::optional<"
                    + output_type_map[out_item["typename"]]
                    + ">"
                )
            else:
                out_item["mapped_type"] = output_type_map[out_item["typename"]]
    render(
        templates_dir,
        destination_dir,
        apis=apis,
        decomp_white_list=decomp_rule_interface_implementation_gen_op_list,
        decomp_ops_list_contain_unused_output=decomp_ops_contain_unused_output,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Generate Static Primitive API'
    )
    parser.add_argument(
        '--fwd_path', type=str, help='The parsed ops yaml file.'
    )
    parser.add_argument(
        '--compat_path',
        type=str,
        help='The parsed ops compat yaml file.',
    )
    parser.add_argument(
        '--fwd_pd_op_path',
        type=str,
        help='The ir forward ops parsed  yaml file.',
    )
    parser.add_argument(
        '--templates_dir',
        type=str,
        help='JinJa2 templates base directory.',
    )
    parser.add_argument(
        '--destination_dir',
        type=str,
        help='Destination base directory for generated file.',
    )
    args = parser.parse_args()

    gen(
        pathlib.Path(args.fwd_path),
        pathlib.Path(args.compat_path),
        pathlib.Path(args.fwd_pd_op_path),
        pathlib.Path(args.templates_dir),
        pathlib.Path(args.destination_dir),
    )
