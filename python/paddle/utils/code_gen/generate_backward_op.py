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
from pathlib import Path

import yaml
from jinja2 import Environment, FileSystemLoader, StrictUndefined

from filters import to_op_attr_type, to_opmaker_name, to_pascal_case
from tests import is_base_api

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
env.tests["base_api"] = is_base_api


def main(api_yaml_path, gen_op_dir):
    with open(api_yaml_path, "rt") as f:
        apis = yaml.safe_load(f)

    gen_op_dir = Path(gen_op_dir)
    gen_op_dir.mkdir(exist_ok=True)
    template = env.get_template('backward_op.c.j2')
    for api in apis:
        with open(gen_op_dir / "{}_op.cc".format(api["name"]), "wt") as f:
            msg = template.render(api=api)
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
        "--output_dir",
        type=str,
        default=str(gen_op_dir),
        help="directory to save generated operator files.")

    args = parser.parse_args()
    main(args.api_yaml_path, args.output_dir)
