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


def main(forward_api_yaml_paths, backward_api_yaml_paths):
    apis = {}
    for api_yaml_path in chain(forward_api_yaml_paths, backward_api_yaml_paths):
        with open(api_yaml_path, "rt", encoding="utf-8") as f:
            api_list = yaml.safe_load(f)
            if api_list is not None:
                apis.update(to_named_dict((api_list)))

    cross_validate(apis)


if __name__ == "__main__":
    current_dir = Path(__file__).parent / "temp"
    parser = argparse.ArgumentParser(
        description="Parse api yaml into canonical format."
    )
    parser.add_argument(
        '--forward_yaml_paths',
        type=str,
        nargs='+',
        default=str(current_dir / "api.parsed.yaml"),
        help="forward api yaml file.",
    )
    parser.add_argument(
        '--backward_yaml_paths',
        type=str,
        nargs='+',
        default=str(current_dir / "backward_api.parsed.yaml"),
        help="backward api yaml file.",
    )

    args = parser.parse_args()
    main(args.forward_yaml_paths, args.backward_yaml_paths)
