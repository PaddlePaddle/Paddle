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

# import from paddle/fluid/ir/dialect/op_generator/api_gen.py
sys.path.append(
    str(pathlib.Path(__file__).resolve().parents[2] / 'ir/dialect/op_generator')
)

# fmt: on


VJPS = ['tanh_grad', 'mean_grad', 'add_grad', 'divide_grad', 'sum_grad']
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
        }
    )
    env.tests.update(
        {
            'scalar': op_gen_tests.is_scalar,
            'intarray': op_gen_tests.is_intarray,
            'datatype': op_gen_tests.is_datatype,
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


def gen(
    prim_path: pathlib.Path,
    fwd_path: pathlib.Path,
    fwd_legacy_path: pathlib.Path,
    rev_path: pathlib.Path,
    rev_legacy_path: pathlib.Path,
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
        templates_dir (pathlib.Path): The directory of the templates.
        destination_dir (pathlib.Path): The Directory of the generated file.

    Returns:
        None
    """
    prims, fwds, legacy_fwds, revs, legacy_revs = (
        load(prim_path),
        load(fwd_path),
        load(fwd_legacy_path),
        load(rev_path),
        load(rev_legacy_path),
    )

    apis = [{**api, **{'is_fwd': True}} for api in fwds + legacy_fwds]
    apis = apis + [{**api, **{'is_fwd': False}} for api in revs + legacy_revs]
    apis = [
        {**api, **{'is_prim': True}}
        if api['name'] in prims
        else {**api, **{'is_prim': False}}
        for api in apis
    ]

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
        pathlib.Path(args.templates_dir),
        pathlib.Path(args.destination_dir),
    )
