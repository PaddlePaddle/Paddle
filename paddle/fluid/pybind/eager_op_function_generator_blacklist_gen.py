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

import yaml
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Eager api.yaml Parset')
    parser.add_argument('--api_yaml_path', type=str)
    parser.add_argument('--eager_impl_file_blacklist', type=str)
    args = parser.parse_args()

    api_yaml_paths = args.api_yaml_path.split(",")
    eager_impl_file_blacklist = args.eager_impl_file_blacklist

    forward_apis = []

    for i in range(len(api_yaml_paths)):
        api_yaml_path = api_yaml_paths[i]
        f = open(api_yaml_path, 'r')
        contents = yaml.load(f, Loader=yaml.FullLoader)
        contents = contents if contents is not None else []

        for item in contents:
            forward_apis.append(item['api'])

        f.close()

    file_contents = \
    """
#include <set>
#include <string>
std::set<std::string> api_black_list = {
    """

    for api in forward_apis:
        file_contents += "\"" + api + "\", "
    file_contents += "\n};"
    file_contents += "std::set<std::string> api_black_list_trans ={};\n"

    with open(eager_impl_file_blacklist, 'w') as f:
        f.write(file_contents)
