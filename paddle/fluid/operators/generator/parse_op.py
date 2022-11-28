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

import yaml

from parse_utils import parse_op_entry


def main(op_yaml_path, output_path, backward):
    with open(op_yaml_path, "rt") as f:
        ops = yaml.safe_load(f)
        if ops is None:
            ops = []
        else:
            ops = [
                parse_op_entry(op, "backward_op" if backward else "op")
                for op in ops
            ]

    with open(output_path, "wt") as f:
        yaml.safe_dump(ops, f, default_flow_style=None, sort_keys=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Parse op yaml into canonical format."
    )
    parser.add_argument('--op_yaml_path', type=str, help="op yaml file.")
    parser.add_argument(
        "--output_path", type=str, help="path to save parsed yaml file."
    )
    parser.add_argument("--backward", action="store_true", default=False)

    args = parser.parse_args()
    main(args.op_yaml_path, args.output_path, args.backward)
