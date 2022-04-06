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
from itertools import chain
from pathlib import Path

import yaml
from jinja2 import Environment, FileSystemLoader, StrictUndefined

from filters import to_op_attr_type, to_opmaker_name, to_pascal_case
from tests import is_base_api
from filters import to_input_name, to_grad_name
from parse_utils import to_named_dict

file_loader = FileSystemLoader(Path(__file__).parent / "templates")
env = Environment(
    loader=file_loader,
    keep_trailing_newline=True,
    trim_blocks=True,
    lstrip_blocks=True,
    undefined=StrictUndefined,
    extensions=['jinja2.ext.do'])
env.filters["to_op_attr_type"] = to_op_attr_type
env.filters["to_opmaker_name"] = to_opmaker_name
env.filters["to_pascal_case"] = to_pascal_case
env.filters["to_input_name"] = to_input_name
env.filters["to_grad_name"] = to_grad_name
env.tests["base_api"] = is_base_api


def main(api_yaml_path, backward_yaml_path, gen_op_dir):
    with open(api_yaml_path, "rt") as f:
        apis = yaml.safe_load(f)
    forward_api_dict = to_named_dict(apis)

    with open(backward_yaml_path, "rt") as f:
        backward_apis = yaml.safe_load(f)
    backward_api_dict = to_named_dict(backward_apis)

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

    gen_op_dir = Path(gen_op_dir)
    gen_op_dir.mkdir(exist_ok=True)

    op_template = env.get_template('op.c.j2')
    with open(gen_op_dir / "generated_op.cc", "wt") as f:
        msg = op_template.render(
            apis=apis, backward_apis=backward_apis, api_dict=api_dict)
        f.write(msg)

    ks_template = env.get_template('ks.c.j2')
    with open(gen_op_dir / "generated_sig.cc", 'wt') as f:
        msg = ks_template.render(apis=apis, backward_apis=backward_apis)
        f.write(msg)


if __name__ == "__main__":
    current_dir = Path(__file__).parent
    gen_op_dir = (current_dir /
                  "../../../../paddle/fluid/operators/generated.tmp").resolve()
    parser = argparse.ArgumentParser(
        description="Generate operator file from api yaml.")
    parser.add_argument(
        '--api_yaml_path',
        type=str,
        default=str(current_dir / "new_api.parsed.yaml"),
        help="parsed api yaml file.")
    parser.add_argument(
        '--backward_api_yaml_path',
        type=str,
        default=str(current_dir / "new_backward_api.parsed.yaml"),
        help="parsed backward api yaml file.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(gen_op_dir),
        help="directory to save generated operator files.")

    args = parser.parse_args()
    main(args.api_yaml_path, args.backward_api_yaml_path, args.output_dir)
