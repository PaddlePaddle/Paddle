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
from parse_utils import cross_validate, to_named_dict


def main(forward_op_yaml_paths, backward_op_yaml_paths):
    ops = {}
    for op_yaml_path in chain(forward_op_yaml_paths, backward_op_yaml_paths):
        with open(op_yaml_path, "rt", encoding="utf-8") as f:
            op_list = yaml.safe_load(f)
            if op_list is not None:
                ops.update(to_named_dict((op_list)))

    cross_validate(ops)


if __name__ == "__main__":
    current_dir = Path(__file__).parent / "temp"
    parser = argparse.ArgumentParser(
        description="Parse op yaml into canonical format."
    )
    parser.add_argument(
        '--forward_yaml_paths',
        type=str,
        nargs='+',
        default=[str(current_dir / "op .parsed.yaml")],
        help="forward op yaml file.",
    )
    parser.add_argument(
        '--backward_yaml_paths',
        type=str,
        nargs='+',
        default=[str(current_dir / "backward_op .parsed.yaml")],
        help="backward op yaml file.",
    )

    args = parser.parse_args()
    main(args.forward_yaml_paths, args.backward_yaml_paths)
