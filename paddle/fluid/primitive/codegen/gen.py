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
from typing import Dict, List

import jinja2
import yaml

# fmt: off
# import from paddle/fluid/operators/generator
sys.path.append(
    str(pathlib.Path(__file__).resolve().parents[2] / 'operators/generator')
)
import filters as op_gen_filters
import tests_utils as op_gen_tests

# import from paddle/fluid/ir/dialect/op_generator/api_gen.py
sys.path.append(
    str(pathlib.Path(__file__).resolve().parents[2] / 'ir/dialect/op_generator')
)

# fmt: on


VJPS = [
    'tanh_grad',
    'mean_grad',
    'add_grad',
    'divide_grad',
    'sum_grad',
    'concat_grad',
    'split_grad',
    'gelu_grad',
    'softmax_grad',
    'silu_grad',
    'multiply_grad',
    'subtract_grad',
    'erf_grad',
    'expand_grad',
    'exp_grad',
    'elementwise_pow_grad',
    'fused_softmax_mask_upper_triangle_grad',
    'matmul_grad',
    'pow_grad',
    'reshape_grad',
    'rsqrt_grad',
    'slice_grad',
    'transpose_grad',
    'dropout_grad',
]
VJP_COMPS = ['divide_grad', 'sum_grad']
BACKENDS = [
    'add_n',
    'mean',
    'sum',
    'divide',
    'full',
    'tanh_grad',
    'mean_grad',
    'concat',
    'add',
    'multiply',
    'elementwise_pow',
    'scale',
    'reshape',
    'expand',
    'tile',
    'add_grad',
    'divide_grad',
    'sum_grad',
    'concat_grad',
    'split_grad',
    'gelu_grad',
    'softmax_grad',
    'silu_grad',
    'multiply_grad',
    'subtract_grad',
    'erf_grad',
    'expand_grad',
    'exp_grad',
    'multiply',
    'exp',
    'erf',
    'cast',
    'elementwise_pow_grad',
    'fused_softmax_mask_upper_triangle_grad',
    'matmul_grad',
    'pow_grad',
    'reshape_grad',
    'rsqrt_grad',
    'slice_grad',
    'transpose_grad',
    'subtract',
    'assign',
    'equal',
    'greater_equal',
    'greater_than',
    'less_equal',
    'less_than',
    'matmul',
    'max',
    'maximum',
    'minimum',
    'not_equal',
    'abs',
    'bitwise_and',
    'bitwise_not',
    'bitwise_or',
    'bitwise_xor',
    'floor',
    'gather_nd',
    'log',
    'roll',
    'scatter',
    'scatter_nd_add',
    'dropout_grad',
]


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
        }
    )
    for tpl in env.list_templates(
        filter_func=lambda name: ".h" in name or ".cc" in name
    ):
        save(
            env.get_template(tpl).render(*args, **kwargs),
            dst_dir / tpl.rstrip('.j2'),
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
            print(f"Generate source file {path}")


def to_compat_dict(items: List[Dict]) -> Dict[str, Dict]:
    compat_dict = {}
    for item in items:
        name = item["op"]
        compat_dict[name] = item
    return compat_dict


def to_apis_dict(apis):
    apis_dict = {}
    for api in apis:
        apis_dict[api['name']] = api
    return apis_dict


def get_inplace_api(apis):
    inplace_apis = []
    for api in apis:
        if (
            'inplace' in api
            and api['inplace'] is not None
            and not api['name'].endswith('_')
        ):
            inplace_api = api.copy()
            inplace_api['name'] = api['name'] + '_'
            inplace_apis.append(inplace_api)
    return inplace_apis


def filter_compat_info(items):
    for item in items:
        item['op'] = item['op'].split('(')[0].strip()
        if 'backward' in item:
            item_backwards = item['backward'].split(',')
            for idx, item_backward in enumerate(item_backwards):
                item_backward = item_backward.split('(')[0].strip()
                item_backwards[idx] = item_backward
            item['backward'] = (
                ','.join(item_backwards)
                if len(item_backwards) > 0
                else item_backwards[0]
            )


def extend_compat_info(apis, compats):
    for api in apis:
        attrs = api["attrs"]
        for attr in attrs:
            if op_gen_tests.is_scalar(
                attr['typename']
            ) or op_gen_tests.is_intarray(attr['typename']):
                attr["support_tensor"] = False
    apis_dict = to_apis_dict(apis)
    for compat_item in compats:
        fwd_op_name = compat_item["op"]
        if fwd_op_name not in apis_dict:
            continue
        fwd_api = apis_dict[fwd_op_name]
        backward_op_names = []
        while fwd_op_name is not None and fwd_op_name in apis_dict:
            backward_op_names.append(apis_dict[fwd_op_name]['backward'])
            fwd_op_name = apis_dict[fwd_op_name]['backward']
        backward_apis = []
        for backward_op_name in backward_op_names:
            if backward_op_name in apis_dict:
                backward_apis.append(apis_dict[backward_op_name])
        support_tensor_attrs_names = []
        compat_attrs_data_type = {}
        if 'scalar' in compat_item:
            for attr_name, attr_info in compat_item['scalar'].items():
                if (
                    'support_tensor' in attr_info
                    and attr_info['support_tensor'] is True
                    or 'tensor_name' in attr_info
                ):
                    support_tensor_attrs_names.append(attr_name)
                if 'data_type' in attr_info:
                    compat_attrs_data_type.update(
                        {attr_name: attr_info['data_type']}
                    )
        if 'int_array' in compat_item:
            for attr_name, attr_info in compat_item['int_array'].items():
                if (
                    'support_tensor' in attr_info
                    and attr_info['support_tensor'] is True
                    or 'tensor_name' in attr_info
                    or 'tensors_name' in attr_info
                ):
                    support_tensor_attrs_names.append(attr_name)
        if len(support_tensor_attrs_names) > 0:
            for api in [fwd_api] + backward_apis:
                attrs = api["attrs"]
                for attr in attrs:
                    if attr['name'] in support_tensor_attrs_names:
                        attr['support_tensor'] = True
        for api in [fwd_api] + backward_apis:
            attrs = api["attrs"]
            for attr in attrs:
                if attr['name'] in compat_attrs_data_type:
                    attr['data_type'] = compat_attrs_data_type[attr['name']]
    return apis


def gen(
    prim_path: pathlib.Path,
    fwd_path: pathlib.Path,
    fwd_legacy_path: pathlib.Path,
    rev_path: pathlib.Path,
    rev_legacy_path: pathlib.Path,
    compat_path: pathlib.Path,
    templates_dir: pathlib.Path,
    destination_dir: pathlib.Path,
):
    """The `gen` load jinja2 templates and relative config info, use jinja2
    templating engine to generate c++ code, and save the code into destination.

    Args:
        prim_path (pathlib.Path): The YAML file path of the primitive API.
        fwd_path (pathlib.Path):  The YAML file path of the forwad API.
        fwd_legacy_path (pathlib.Path): The YAML file path of the legacy
            forwad API.
        rev_path (pathlib.Path): The YAML file path of the backward API.
        rev_legacy_path (pathlib.Path): The YAML file path of the legacy
            backward API.
        compat_path: (pathlib.Path): The YAML file path of the ops compat.
        templates_dir (pathlib.Path): The directory of the templates.
        destination_dir (pathlib.Path): The Directory of the generated file.

    Returns:
        None
    """
    prims, fwds, legacy_fwds, revs, legacy_revs, compats = (
        load(prim_path),
        load(fwd_path),
        load(fwd_legacy_path),
        load(rev_path),
        load(rev_legacy_path),
        load(compat_path),
    )
    filter_compat_info(compats)
    apis = [{**api, **{'is_fwd': True}} for api in fwds + legacy_fwds]
    apis = apis + [{**api, **{'is_fwd': False}} for api in revs + legacy_revs]
    apis = [
        {**api, **{'is_prim': True}}
        if api['name'] in prims
        else {**api, **{'is_prim': False}}
        for api in apis
    ]
    apis = extend_compat_info(apis, compats)
    apis = apis + get_inplace_api(apis)
    render(
        templates_dir,
        destination_dir,
        apis=apis,
        backend_white_list=BACKENDS,
        vjp_white_list=VJPS,
        vjp_comp_white_list=VJP_COMPS,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Generate Static Primitive API'
    )
    parser.add_argument(
        '--prim_path',
        type=str,
        help='The primitive API yaml file.',
    )
    parser.add_argument(
        '--fwd_path', type=str, help='The parsed ops yaml file.'
    )
    parser.add_argument(
        '--fwd_legacy_path',
        type=str,
        help='The parsed ops yaml file.',
    )
    parser.add_argument(
        '--rev_path', type=str, help='The parsed ops yaml file.'
    )
    parser.add_argument(
        '--rev_legacy_path',
        type=str,
        help='The parsed ops yaml file.',
    )
    parser.add_argument(
        '--compat_path',
        type=str,
        help='The parsed ops compat yaml file.',
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
        pathlib.Path(args.prim_path),
        pathlib.Path(args.fwd_path),
        pathlib.Path(args.fwd_legacy_path),
        pathlib.Path(args.rev_path),
        pathlib.Path(args.rev_legacy_path),
        pathlib.Path(args.compat_path),
        pathlib.Path(args.templates_dir),
        pathlib.Path(args.destination_dir),
    )
