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

    def gene_kernel_backend_select(self):
        return BackwardAPI.gene_kernel_backend_select(self)

    def get_return_type(self, inplace_flag=False):
        return BackwardAPI.get_return_type(self)

    def gene_return_code(self):
        return "return;"

    def gene_api_declaration(self):
        return SparseAPI.gene_api_declaration(self)

    def get_declare_args(self, inplace_flag=False):
        return BackwardAPI.get_declare_args(self)

    def get_define_args(self, inplace_flag=False):
        return BackwardAPI.get_define_args(self)

    def gene_output(self,
                    out_dtype_list,
                    out_tensor_type_list=None,
                    code_indent='',
                    inplace_flag=False):
        kernel_output = []
        output_names = []
        output_create = ""
        output_type_map = {
            'dense': 'TensorType::DENSE_TENSOR',
            'sparse_coo': 'TensorType::SPARSE_COO',
            'sparse_csr': 'TensorType::SPARSE_CSR'
        }

        if len(out_dtype_list) == 1:
            kernel_output.append('kernel_out')
            output_names.append('kernel_out')
            inplace_assign = " = " + self.inplace_map[self.outputs['names'][
                0]] if inplace_flag and self.inplace_map is not None and self.outputs[
                    'names'][0] in self.inplace_map else ""
            output_create = f"""
    auto kernel_out = SetSparseKernelOutput({self.outputs['names'][0]}, {output_type_map[out_dtype_list[0]]});"""

        elif len(out_dtype_list) > 1:
            output_create = ""

            for i, out_type_item in enumerate(out_dtype_list):
                kernel_output.append(f'kernel_out_{i}')
                output_names.append(f'kernel_out_{i}')
                if inplace_flag and self.inplace_map is not None and self.outputs[
                        'names'][i] in self.inplace_map:
                    output_create = output_create + f"""
    *{self.outputs['names'][i]} = {self.inplace_map[self.outputs['names'][i]]};"""

                output_create = output_create + f"""
    auto kernel_out_{i} = SetSparseKernelOutput({self.outputs['names'][i]}, {output_type_map[out_dtype_list[i]]});"""

        else:
            raise ValueError(
                "{} : Output error: the output should not be empty.".format(
                    self.api))

        return kernel_output, output_names, output_create


def header_include():
    return """
#include "paddle/phi/api/include/tensor.h"
#include "paddle/phi/common/scalar.h"
#include "paddle/phi/common/int_array.h"
#include "paddle/utils/optional.h"
"""


def source_include(header_file_path):
    return f"""
#include "{header_file_path}"
#include <memory>

#include "glog/logging.h"

#include "paddle/phi/api/include/sparse_api.h"
#include "paddle/phi/api/lib/api_gen_utils.h"
#include "paddle/phi/api/lib/kernel_dispatch.h"
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
    parser.add_argument('--api_yaml_path',
                        help='path to sparse api yaml file',
                        default='paddle/phi/api/yaml/sparse_backward.yaml')

    parser.add_argument('--api_header_path',
                        help='output of generated api header code file',
                        default='paddle/phi/api/backward/sparse_bw_api.h')

    parser.add_argument('--api_source_path',
                        help='output of generated api source code file',
                        default='paddle/phi/api/lib/sparse_bw_api.cc')

    options = parser.parse_args()

    api_yaml_path = options.api_yaml_path
    header_file_path = options.api_header_path
    source_file_path = options.api_source_path

    generate_api(api_yaml_path, header_file_path, source_file_path)


if __name__ == '__main__':
    main()
