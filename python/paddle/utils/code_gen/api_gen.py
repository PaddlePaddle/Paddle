# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import os
import yaml
import argparse

from api_base import BaseAPI


class ForwardAPI(BaseAPI):
    prefix_tensor_name = 'dense_'

    def __init__(self, api_item_yaml):
        super(ForwardAPI, self).__init__(api_item_yaml)

    def get_return_type(self, out_type_list):
        return out_type_list[0] if len(
            out_type_list) == 1 else "std::tuple<" + ",".join(
                out_type_list) + ">"

    def gene_output(self, output_type_list, set_out_func, code_indent):
        kernel_output = ""
        output_names = []
        output_create = ""

        if len(output_type_list) == 1:
            kernel_output = 'kernel_out'
            output_names.append('kernel_out')
            output_create = f"""
{code_indent}  {self.outputs['return_type']} out;
{code_indent}  auto kernel_out = {set_out_func}(kernel_backend, &out);"""

        elif len(output_type_list) > 1:
            output_create = f"""
{code_indent}  {self.outputs['return_type']} out;"""

            for i in range(len(output_type_list)):
                kernel_output = kernel_output + f'kernel_out_{i}, '
                output_names.append(f'kernel_out_{i}')
                output_create = output_create + f"""
{code_indent}  auto kernel_out_{i} = {set_out_func}(kernel_backend, &std::get<{i}>(out));"""

            kernel_output = kernel_output[:-2]
        else:
            raise ValueError(
                "{} : Output error: the output should not be empty.".format(
                    self.api))

        return kernel_output, output_names, output_create


def header_include():
    return """
#include <tuple>

#include "paddle/pten/api/include/tensor.h"
#include "paddle/pten/common/scalar.h"
#include "paddle/pten/common/scalar_array.h"
"""


def source_include(header_file_path):
    return f"""
#include "{header_file_path}"
#include <memory>

#include "glog/logging.h"

#include "paddle/pten/api/lib/api_registry.h"
#include "paddle/pten/api/lib/api_utils.h"
#include "paddle/pten/api/lib/data_transform.h"
#include "paddle/pten/api/lib/kernel_dispatch.h"
#include "paddle/pten/api/lib/utils/storage.h"
#include "paddle/pten/core/kernel_registry.h"
#include "paddle/pten/infermeta/binary.h"
#include "paddle/pten/infermeta/multiary.h"
#include "paddle/pten/infermeta/nullary.h"
#include "paddle/pten/infermeta/unary.h"
#include "paddle/pten/kernels/declarations.h"
"""


def api_register():
    return """
PT_REGISTER_API(Math);
"""


def api_namespace():
    return ("""
namespace paddle {
namespace experimental {

""", """

}  // namespace experimental
}  // namespace paddle
""")


def generate_api(api_yaml_path, header_file_path, source_file_path):

    with open(api_yaml_path, 'r') as f:
        apis = yaml.load(f, Loader=yaml.FullLoader)
    header_file = open(header_file_path, 'w')
    source_file = open(source_file_path, 'w')

    namespace = api_namespace()

    header_file.write("#pragma once\n")
    header_file.write(header_include())
    header_file.write(namespace[0])

    include_header_file = "paddle/pten/api/include/api.h"
    source_file.write(source_include(include_header_file))
    source_file.write(namespace[0])

    for api in apis:
        api_code = ForwardAPI(api)
        print(api_code.gene_api_declaration())
        header_file.write(api_code.gene_api_declaration())
        source_file.write(api_code.gene_api_code())

    header_file.write(namespace[1])
    source_file.write(namespace[1])

    source_file.write(api_register())

    header_file.close()
    source_file.close()


def main():
    parser = argparse.ArgumentParser(
        description='Generate PaddlePaddle C++ API files')
    parser.add_argument(
        '--api_yaml_path',
        help='path to api yaml file',
        default='python/paddle/utils/code_gen/api.yaml')
    parser.add_argument(
        '--api_header_path',
        help='output of generated api header code file',
        default='paddle/pten/api/include/api.h')

    parser.add_argument(
        '--api_source_path',
        help='output of generated api source code file',
        default='paddle/pten/api/lib/api.cc')

    options = parser.parse_args()

    api_yaml_path = options.api_yaml_path
    header_file_path = options.api_header_path
    source_file_path = options.api_source_path

    generate_api(api_yaml_path, header_file_path, source_file_path)


if __name__ == '__main__':
    main()
