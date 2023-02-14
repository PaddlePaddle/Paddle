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
from prim_base import EagerPrimAPI


def header_include():
    return """
#include "paddle/phi/common/int_array.h"
#include "paddle/phi/common/data_type.h"
#include "paddle/phi/common/scalar.h"
#include "paddle/phi/common/place.h"
#include "paddle/utils/optional.h"
"""


def eager_source_include():
    return """
#include "paddle/fluid/eager/api/all.h"
#include "paddle/fluid/eager/api/generated/eager_generated/forwards/dygraph_functions.h"
#include "paddle/fluid/prim/api/generated_prim/prim_generated_api.h"
"""


def api_namespace():
    return (
        """
namespace paddle {
namespace prim {
""",
        """
using Tensor = paddle::experimental::Tensor;
using Scalar = paddle::experimental::Scalar;
using IntArray = paddle::experimental::IntArray;
using DataType = paddle::experimental::DataType;
""",
        """
}  // namespace prim
}  // namespace paddle
""",
    )


def generate_api(
    api_yaml_path, header_file_path, eager_prim_source_file_path, api_prim_path
):
    apis = []

    for each_api_yaml in api_yaml_path:
        with open(each_api_yaml, 'r') as f:
            api_list = yaml.load(f, Loader=yaml.FullLoader)
            if api_list:
                apis.extend(api_list)

    header_file = open(header_file_path, 'w')
    eager_prim_source_file = open(eager_prim_source_file_path, 'w')

    namespace = api_namespace()

    header_file.write("#pragma once\n")
    header_file.write(header_include())
    header_file.write(namespace[0])
    header_file.write(namespace[1])
    eager_prim_source_file.write(eager_source_include())
    eager_prim_source_file.write(namespace[0])

    with open(api_prim_path, 'rt') as f:
        api_prims = yaml.safe_load(f)

    for api in apis:
        prim_api = EagerPrimAPI(api, api_prims)
        if prim_api.is_prim_api:
            header_file.write(prim_api.gene_prim_api_declaration())
            eager_prim_source_file.write(prim_api.gene_eager_prim_api_code())

    header_file.write(namespace[2])
    eager_prim_source_file.write(namespace[2])

    header_file.close()
    eager_prim_source_file.close()


def main():
    parser = argparse.ArgumentParser(
        description='Generate PaddlePaddle C++ API files'
    )
    parser.add_argument(
        '--api_yaml_path',
        help='path to api yaml file',
        nargs='+',
        default=['paddle/phi/api/yaml/ops.yaml'],
    )

    parser.add_argument(
        '--prim_api_header_path',
        help='output of generated prim_api header code file',
        default='paddle/fluid/prim/api/generated_prim/prim_generated_api.h',
    )

    parser.add_argument(
        '--eager_prim_api_source_path',
        help='output of generated eager_prim_api source code file',
        default='paddle/fluid/prim/api/generated_prim/eager_prim_api.cc',
    )

    parser.add_argument(
        '--api_prim_yaml_path',
        help='Primitive API list yaml file.',
        default='paddle/fluid/prim/api/auto_code_generated/api.yaml',
    )

    options = parser.parse_args()

    api_yaml_path = options.api_yaml_path
    prim_api_header_file_path = options.prim_api_header_path
    eager_prim_api_source_file_path = options.eager_prim_api_source_path
    api_prim_yaml_path = options.api_prim_yaml_path

    generate_api(
        api_yaml_path,
        prim_api_header_file_path,
        eager_prim_api_source_file_path,
        api_prim_yaml_path,
    )


if __name__ == '__main__':
    main()
