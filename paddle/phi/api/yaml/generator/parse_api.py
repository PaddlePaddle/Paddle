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

from parse_utils import parse_api_entry


def main(api_yaml_path, output_path, backward):
    with open(api_yaml_path, "rt") as f:
        apis = yaml.safe_load(f)
        if apis is None:
            apis = []
        else:
            apis = [
                parse_api_entry(api, "backward_api" if backward else "op")
                for api in apis
            ]

    with open(output_path, "wt") as f:
        yaml.safe_dump(apis, f, default_flow_style=None, sort_keys=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Parse api yaml into canonical format.")
    parser.add_argument('--api_yaml_path', type=str, help="api yaml file.")
    parser.add_argument("--output_path",
                        type=str,
                        help="path to save parsed yaml file.")
    parser.add_argument("--backward", action="store_true", default=False)

    args = parser.parse_args()
    main(args.api_yaml_path, args.output_path, args.backward)
