# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
import yaml
import re

ATTR_TYPE_STRING_MAP = {
    'bool': 'bool',
    'int': 'int',
    'int64_t': 'int64_t',
    'float': 'float',
    'double': 'double',
    'str': 'std::string',
    'bool[]': 'std::vector<bool>',
    'int[]': 'std::vector<int>',
    'int64_t[]': 'std::vector<int64_t>',
    'float[]': 'std::vector<float>',
    'double[]': 'std::vector<double>',
    'str[]': 'std::vector<std::string>'
}


def generate_extra_info(api_compat_yaml_path, header_file_path):
    with open(api_compat_yaml_path, 'rt') as f:
        api_compat_info_map = yaml.safe_load(f)
    for api_args_map in api_compat_info_map:
        if 'extra' in api_args_map:
            api_extra_args_map = api_args_map['extra']
            print(api_extra_args_map)
            # TODO(chenweihang): add inputs and outputs
            key_set = ['attrs']


def main():
    parser = argparse.ArgumentParser(
        description='Generate PaddlePaddle extra op info map.')
    parser.add_argument('--api_compat_yaml_path',
                        help='Path to api compat yaml file.',
                        default='paddle/phi/api/yaml/api_compat.yaml')
    parser.add_argument('--extra_info_header_path',
                        help='Output of the extra op info header code file.',
                        default='paddle/phi/ops/compat/extra_info.h')

    options = parser.parse_args()
    api_compat_yaml_path = options.api_compat_yaml_path
    header_file_path = options.extra_info_header_path

    generate_extra_info(api_compat_yaml_path, header_file_path)


if __name__ == '__main__':
    main()
