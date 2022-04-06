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
from parse_utils import cross_validate
from filters import to_named_dict


def main(forward_api_yaml_path, backward_api_yaml_path):
    with open(forward_api_yaml_path, "rt") as f:
        forward_apis = yaml.safe_load(f)

    with open(backward_api_yaml_path, "rt") as f:
        backward_apis = yaml.safe_load(f)

    apis = {}
    apis.update(to_named_dict(forward_apis))
    apis.update(to_named_dict(backward_apis))

    cross_validate(apis)


if __name__ == "__main__":
    current_dir = Path(__file__).parent / "temp"
    parser = argparse.ArgumentParser(
        description="Parse api yaml into canonical format.")
    parser.add_argument(
        '--forward_yaml_path',
        type=str,
        default=str(current_dir / "api.parsed.yaml"),
        help="forward api yaml file.")
    parser.add_argument(
        '--backward_yaml_path',
        type=str,
        default=str(current_dir / "backward.yaml.yaml"),
        help="backward api yaml file.")

    args = parser.parse_args()
    main(args.forward_yaml_path, args.backward_yaml_path)
