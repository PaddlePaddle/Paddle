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

import os
import yaml
import argparse
import re

from sparse_api_gen import SparseAPI
from backward_api_gen import BackwardAPI


class SparseBackwardAPI(SparseAPI, BackwardAPI):
    def __init__(self, bw_api_item_yaml):
        BackwardAPI.__init__(self, bw_api_item_yaml)

    def get_api_func_name(self):
        return self.api

    def get_return_type(self, out_type_list):
        return BackwardAPI.get_return_type(self, out_type_list)

    def gene_api_declaration(self):
        return SparseAPI.gene_api_declaration(self)

    def gene_output(self,
                    output_type_list,
                    set_out_func,
                    code_indent,
                    inplace_flag=False):
        kernel_output = ""
        output_names = []
        output_create = ""

        if len(output_type_list) == 1:
            kernel_output = 'kernel_out'
            output_names.append('kernel_out')
            inplace_assign = " = " + self.inplace_map[self.outputs['names'][
                0]] if inplace_flag and self.inplace_map is not None and self.outputs[
                    'names'][0] in self.inplace_map else ""
            output_create = f"""
  {self.outputs['return_type']} api_output{inplace_assign};
  auto kernel_out = {set_out_func}(&api_output, {self.get_kernel_tensor_out_type(self.outputs['names'][0])});"""

        elif len(output_type_list) > 1:
            output_create = f"""
  {self.outputs['return_type']} api_output({len(output_type_list)});"""

            for i, out_type_item in enumerate(output_type_list):
                kernel_output = kernel_output + f'kernel_out_{i}, '
                output_names.append(f'kernel_out_{i}')
                if out_type_item == 'Tensor':
                    get_out_code = f'&api_output[{i}][0]'
                    if inplace_flag and self.inplace_map is not None and self.outputs[
                            'names'][i] in self.inplace_map:
                        output_create = output_create + f"""
  api_output[{i}].emplace_back({self.inplace_map[self.outputs['names'][i]]});"""

                    else:
                        output_create = output_create + f"""
  api_output[{i}].emplace_back();"""

                else:
                    get_out_code = f'&api_output[{i}]'
                    if inplace_flag and self.inplace_map is not None and self.outputs[
                            'names'][i] in self.inplace_map:
                        output_create = output_create + f"""
  api_output[{i}] = {self.inplace_map[self.outputs['names'][i]]};"""

                output_create = output_create + f"""
  auto kernel_out_{i} = {set_out_func}({get_out_code}, {self.get_kernel_tensor_out_type(self.outputs['names'][i])});"""

            kernel_output = kernel_output[:-2]
        else:
            raise ValueError(
                "{} : Output error: the output should not be empty.".format(
                    self.api))

        return kernel_output, output_names, output_create


def header_include():
    return """
#include "paddle/phi/api/include/tensor.h"
#include "paddle/phi/common/scalar.h"
#include "paddle/phi/common/scalar_array.h"
#include "paddle/utils/optional.h"
"""


def source_include(header_file_path):
    return f"""
#include "{header_file_path}"
#include <memory>

#include "glog/logging.h"

#include "paddle/phi/api/lib/api_gen_utils.h"
#include "paddle/phi/api/lib/kernel_dispatch.h"
#include "paddle/phi/api/lib/sparse_api_custom_impl.h"
#include "paddle/phi/core/kernel_registry.h"
"""


def api_namespace():
    return ("""
namespace paddle {
namespace experimental {
namespace sparse {

""", """

}  // namespace sparse
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

    include_header_file = "paddle/phi/api/backward/sparse_bw_api.h"
    source_file.write(source_include(include_header_file))
    source_file.write(namespace[0])

    for api in apis:
        sparse_bw_api = SparseBackwardAPI(api)
        header_file.write(sparse_bw_api.gene_api_declaration())
        source_file.write(sparse_bw_api.gene_api_code())

    header_file.write(namespace[1])
    source_file.write(namespace[1])

    header_file.close()
    source_file.close()


def main():
    parser = argparse.ArgumentParser(
        description='Generate PaddlePaddle C++ Sparse API files')
    parser.add_argument(
        '--api_yaml_path',
        help='path to sparse api yaml file',
        default='python/paddle/utils/code_gen/sparse_bw_api.yaml')

    parser.add_argument(
        '--api_header_path',
        help='output of generated api header code file',
        default='paddle/phi/api/backward/sparse_bw_api.h')

    parser.add_argument(
        '--api_source_path',
        help='output of generated api source code file',
        default='paddle/phi/api/lib/sparse_bw_api.cc')

    options = parser.parse_args()

    api_yaml_path = options.api_yaml_path
    header_file_path = options.api_header_path
    source_file_path = options.api_source_path

    generate_api(api_yaml_path, header_file_path, source_file_path)


if __name__ == '__main__':
    main()
