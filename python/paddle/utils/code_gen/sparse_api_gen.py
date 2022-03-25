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

from api_gen import ForwardAPI


class SparseAPI(ForwardAPI):
    def __init__(self, api_item_yaml):
        super(SparseAPI, self).__init__(api_item_yaml)

    def gene_api_declaration(self):
        return f"""
// {", ".join(self.outputs['names'])}
PADDLE_API {self.outputs['return_type']} {self.get_api_func_name()}({self.args_str['args_declare']});
"""

    def get_kernel_tensor_out_type(self, output_name):
        sparse_type = 'TensorType::DENSE_TENSOR'
        if output_name.endswith('@SparseCooTensor'):
            sparse_type = 'TensorType::SPARSE_COO'
        elif output_name.endswith('@SparseCsrTensor'):
            sparse_type = 'TensorType::SPARSE_CSR'
        return sparse_type

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
  auto* kernel_out = {set_out_func}(&api_output, {self.get_kernel_tensor_out_type(self.outputs['names'][0])});"""

        elif len(output_type_list) > 1:
            output_create = f"""
  {self.outputs['return_type']} api_output;"""

            for i in range(len(output_type_list)):
                kernel_output = kernel_output + f'kernel_out_{i}, '
                output_names.append(f'kernel_out_{i}')
                if inplace_flag and self.inplace_map is not None and self.outputs[
                        'names'][i] in self.inplace_map:
                    output_create = output_create + f"""
  std::get<{i}>(api_output) = {self.inplace_map[self.outputs['names'][i]]};"""

                output_create = output_create + f"""
  auto* kernel_out_{i} = {set_out_func}(&std::get<{i}>(api_output), {self.get_kernel_tensor_out_type(self.outputs['names'][i])});"""

            kernel_output = kernel_output[:-2]
        else:
            raise ValueError(
                "{} : Output error: the output should not be empty.".format(
                    self.api))

        return kernel_output, output_names, output_create

    def gen_sparse_kernel_context(self, kernel_output_names):
        input_trans_map = {
            'const Tensor&': 'const phi::TenseBase&',
            'const std::vector<Tensor>&': 'const std::vector<phi::TenseBase>&',
            'const paddle::optional<Tensor>&':
            'paddle::optional<const phi::TenseBase&>'
        }
        out_trans_map = {
            'Tensor': 'phi::TenseBase*',
            'std::vector<Tensor>': 'std::vector<phi::TenseBase*>'
        }
        input_names = self.inputs['names']
        input_infos = self.inputs['input_info']

        attr_names = self.attrs['names']
        kernel_param = self.kernel['param']
        if kernel_param is None:
            kernel_param = input_names + attr_names

        kernel_context_code = ""
        for param in kernel_param:
            if param in input_names:
                if param in self.optional_vars:
                    raise ValueError(
                        f"{self.api} : Unsupport optional input({param}) for sparse api."
                    )
                else:
                    kernel_context_code = kernel_context_code + f"""
  kernel_context.EmplaceBackInput({param}.impl().get());"""

                continue
            if param in attr_names:
                # set attr for kernel_context
                if 'ScalarArray' in self.attrs['attr_info'][param][0]:
                    param = 'phi::ScalarArray(' + param + ')'
                elif 'Scalar' in self.attrs['attr_info'][param][0]:
                    param = 'phi::Scalar(' + param + ')'
            elif isinstance(param, bool):
                param = str(param).lower()
            else:
                param + str(param) + ", "
            kernel_context_code = kernel_context_code + f"""
  kernel_context.EmplaceBackAttr({param});"""

        for out_name in kernel_output_names:
            kernel_context_code = kernel_context_code + f"""
  kernel_context.EmplaceBackOutput({out_name});"""

        return kernel_context_code

    def gen_sparse_kernel_code(self, inplace_flag=False):
        _, kernel_output_names, output_create = self.gene_output(
            self.outputs['types'], 'SetSparseKernelOutput', '', inplace_flag)

        kernel_context_code = self.gen_sparse_kernel_context(
            kernel_output_names)

        return f"""
  auto phi_kernel = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "{self.kernel['func'][0]}", {{kernel_backend, kernel_layout, kernel_data_type}});
  VLOG(6) << "{self.api} api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  VLOG(6) << "{self.api} api sparse kernel: " << phi_kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_backend);
  auto kernel_context = phi::KernelContext(dev_ctx);
{output_create}
{kernel_context_code}
  phi_kernel(&kernel_context);

  return api_output;"""

    def gene_base_api_code(self, inplace_flag=False):
        api_func_name = self.get_api_func_name()
        return f"""
PADDLE_API {self.outputs['return_type']} {api_func_name}({self.args_str["args_define"]}) {{
{self.gene_kernel_select()}
{self.gen_sparse_kernel_code(inplace_flag)}
}}
"""


def header_include():
    return """
#include <tuple>

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
#include "paddle/phi/api/lib/data_transform.h"
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

    include_header_file = "paddle/phi/api/include/sparse_api.h"
    source_file.write(source_include(include_header_file))
    source_file.write(namespace[0])

    for api in apis:
        sparse_api = SparseAPI(api)
        if sparse_api.is_dygraph_api:
            sparse_api.is_dygraph_api = False
        header_file.write(sparse_api.gene_api_declaration())
        source_file.write(sparse_api.gene_api_code())

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
        default='python/paddle/utils/code_gen/sparse_api.yaml')

    parser.add_argument(
        '--api_header_path',
        help='output of generated api header code file',
        default='paddle/phi/api/include/sparse_api.h')

    parser.add_argument(
        '--api_source_path',
        help='output of generated api source code file',
        default='paddle/phi/api/lib/sparse_api.cc')

    options = parser.parse_args()

    api_yaml_path = options.api_yaml_path
    header_file_path = options.api_header_path
    source_file_path = options.api_source_path

    generate_api(api_yaml_path, header_file_path, source_file_path)


if __name__ == '__main__':
    main()
