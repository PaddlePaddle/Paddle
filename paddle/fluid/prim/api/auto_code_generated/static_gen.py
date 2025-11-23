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
import copy
import pathlib
import sys

import jinja2
import yaml

# import from paddle/fluid/operators/generator
sys.path.append(
    str(pathlib.Path(__file__).parents[3].joinpath('operators/generator'))
)
import filters as op_gen_filters
import generate_op as op_gen_utils
import parse_utils as op_gen_parse_utils
import tests_utils as op_gen_tests

# fmt: on


def load_yaml(path, mode="rt"):
    with open(path, mode) as f:
        return yaml.safe_load(f)


def render(tpl, *args, **kwargs):
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(pathlib.Path(tpl).parent),
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
            'to_pascal': op_gen_filters.to_pascal_case,
            "trip_intermediate": op_gen_filters.filter_intermediate,
        }
    )
    env.tests.update(
        {
            'scalar': op_gen_tests.is_scalar,
            'intarray': op_gen_tests.is_intarray,
            'datatype': op_gen_tests.is_datatype,
            'tensor_sequence': op_gen_tests.is_tensor_list,
        }
    )
    return env.get_template(pathlib.Path(tpl).name).render(*args, **kwargs)


def filter_prim(apis, prims):
    return [api for api in apis if api.get('name') in prims]


def extend_compat(apis, compats):
    dicts = op_gen_parse_utils.to_named_dict(copy.deepcopy(apis))
    for api in dicts.values():
        op_gen_utils.restruct_io(api)
        api['op_name'] = api['name']
        op_gen_utils.add_fluid_name(api['inputs'])
        op_gen_utils.add_fluid_name(api['attrs'])
        op_gen_utils.add_fluid_name(api['outputs'])
        api['backward'] = None
    op_gen_utils.add_compat_name(compats, dicts, {})
    return tuple(dicts.values())


def extend_version(apis, versions):
    apis = copy.deepcopy(apis)
    for api in apis:
        for version in versions:
            if version.get('op') == api.get('name'):
                api['version'] = version['version']
    return apis


def generate(
    api_prim_yaml_path,
    api_phi_yaml_path,
    api_phi_legacy_yaml_path,
    api_compat_yaml_path,
    api_version_yaml_path,
    template_path,
    output_op_path,
):
    prims, phis, legacy_phis, compats, versions = (
        load_yaml(api_prim_yaml_path),
        load_yaml(api_phi_yaml_path),
        load_yaml(api_phi_legacy_yaml_path),
        load_yaml(api_compat_yaml_path),
        load_yaml(api_version_yaml_path),
    )

    apis = phis + legacy_phis
    apis = filter_prim(apis, prims)
    apis = extend_version(apis, versions)
    apis = extend_compat(apis, compats)

    if len(apis) > 0:
        with open(output_op_path, "wt") as f:
            msg = render(template_path, apis=apis)
            f.write(msg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate Static Primitive API"
    )
    parser.add_argument(
        '--api_prim_yaml_path', type=str, help="Primitive API yaml file.."
    )
    parser.add_argument(
        '--api_phi_yaml_path', type=str, help="Parsed ops yaml file."
    )
    parser.add_argument(
        '--api_phi_legacy_yaml_path', type=str, help="Parsed ops yaml file."
    )
    parser.add_argument(
        '--api_compat_yaml_path', type=str, help="Ops args compat yaml file."
    )
    parser.add_argument(
        '--api_version_yaml_path', type=str, help="Ops version yaml file."
    )
    parser.add_argument(
        "--template_path", type=str, help="JinJa2 template file Path."
    )
    parser.add_argument("--output_path", type=str, help="Output path.")

    args = parser.parse_args()
    generate(
        args.api_prim_yaml_path,
        args.api_phi_yaml_path,
        args.api_phi_legacy_yaml_path,
        args.api_compat_yaml_path,
        args.api_version_yaml_path,
        args.template_path,
        args.output_path,
    )
