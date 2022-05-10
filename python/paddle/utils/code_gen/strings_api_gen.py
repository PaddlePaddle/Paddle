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
PREFIX_TENSOR_NAME = 'input_'
PREFIX_META_TENSOR_NAME = 'meta_'


class StringsAPI(ForwardAPI):
    def __init__(self, api_item_yaml):
        super(StringsAPI, self).__init__(api_item_yaml)

    def get_api_func_name(self):
        return self.api

    def gene_api_declaration(self):
        return f"""
// {", ".join(self.outputs['names'])}
PADDLE_API {self.outputs['return_type']} {self.get_api_func_name()}({self.args_str['args_declare']});
"""

    def get_kernel_tensor_out_type(self, output_name):
        strings_type = 'TensorType::DENSE_TENSOR'
        if output_name.endswith('@StringTensor'):
            strings_type = 'TensorType::STRING_TENSOR'
        return strings_type

    def get_tensor_type(self, kernel_tensor_out_type):
        tensor_type_dict = {
            "TensorType::DENSE_TENSOR": "phi::DenseTensor",
            "TensorType::STRING_TENSOR": "phi::StringTensor",
        }
        return tensor_type_dict[kernel_tensor_out_type]

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
            kernel_tensor_out_type = self.get_kernel_tensor_out_type(
                self.outputs['names'][0])
            tensor_type = self.get_tensor_type(kernel_tensor_out_type)
            inplace_assign = " = " + self.inplace_map[self.outputs['names'][
                0]] if inplace_flag and self.inplace_map is not None and self.outputs[
                    'names'][0] in self.inplace_map else ""
            output_create = f"""
  {self.outputs['return_type']} api_output{inplace_assign};
  
  {tensor_type}* kernel_out = dynamic_cast<{tensor_type}*>({set_out_func}(kernel_backend, &api_output, {kernel_tensor_out_type}));"""

        elif len(output_type_list) > 1:
            output_create = f"""
  {self.outputs['return_type']} api_output;"""

            for i in range(len(output_type_list)):
                kernel_output = kernel_output + f'kernel_out_{i}, '
                output_names.append(f'kernel_out_{i}')
                kernel_tensor_out_type = self.get_kernel_tensor_out_type(
                    self.outputs['names'][i])
                tensor_type = self.get_tensor_type(kernel_tensor_out_type)
                if inplace_flag and self.inplace_map is not None and self.outputs[
                        'names'][i] in self.inplace_map:
                    output_create = output_create + f"""
  std::get<{i}>(api_output) = {self.inplace_map[self.outputs['names'][i]]};"""

                output_create = output_create + f"""
  {tensor_type}* kernel_out_{i} = dynamic_cast<{tensor_type}*>({set_out_func}(&std::get<{i}>(api_output), {kernel_tensor_out_type}));"""

            kernel_output = kernel_output[:-2]
        else:
            raise ValueError(
                "{} : Output error: the output should not be empty.".format(
                    self.api))

        return kernel_output, output_names, output_create

    def get_kernel_args(self, code_indent):
        input_trans_map = {
            'const Tensor&': 'const phi::StringTensor&',
            'const std::vector<Tensor>&':
            'const std::vector<const phi::StringTensor*>&',
            'const paddle::optional<Tensor>&':
            'paddle::optional<const phi::StringTensor&>',
            'const paddle::optional<std::vector<Tensor>>&':
            'paddle::optional<const std::vector<phi::StringTensor>&>'
        }
        out_trans_map = {
            'Tensor': 'phi::StringTensor*',
            'std::vector<Tensor>': 'std::vector<phi::StringTensor*>&'
        }
        input_names = self.inputs['names']
        input_infos = self.inputs['input_info']
        kernel_args_type_list = ['const platform::DeviceContext&']

        attr_names = self.attrs['names']
        kernel_param = self.kernel['param']
        if kernel_param is None:
            kernel_param = input_names + attr_names
        input_tensor_code = ""
        # set input_tensor_code
        for i, input_name in enumerate(input_names):
            input_tensor_code = input_tensor_code + f"""
{code_indent}  auto {PREFIX_TENSOR_NAME}{input_name} = TensorToStringTensor({input_name});"""

        # set kernel_args
        kernel_args = "*dev_ctx, "
        for param in kernel_param:
            if param in input_names:
                if param in self.optional_vars:
                    kernel_args = kernel_args + PREFIX_TENSOR_NAME + param + ", "
                else:
                    if self.inputs['input_info'][param] == "const Tensor&":
                        kernel_args = kernel_args + "*" + PREFIX_TENSOR_NAME + param + ", "
                    elif self.inputs['input_info'][
                            input_name] == "const std::vector<Tensor>&":
                        kernel_args = kernel_args + PREFIX_TENSOR_NAME + param + ", "
                    else:
                        # do nothing
                        pass
                kernel_in_type = input_trans_map[input_infos[param]]
                kernel_args_type_list.append(kernel_in_type)
            elif param in attr_names:
                # set attr for kernel_context
                if 'IntArray' in self.attrs['attr_info'][param][0]:
                    kernel_args_type_list.append('const phi::IntArray&')
                    param = 'phi::IntArray(' + param + ')'
                elif 'Scalar' in self.attrs['attr_info'][param][0]:
                    kernel_args_type_list.append('const phi::Scalar&')
                    param = 'phi::Scalar(' + param + ')'
                else:
                    kernel_args_type_list.append(self.attrs['attr_info'][param][
                        0])
                kernel_args = kernel_args + param + ", "
            elif isinstance(param, bool):
                kernel_args = kernel_args + str(param).lower() + ", "
            else:
                kernel_args = kernel_args + str(param) + ", "

        for out_type in self.outputs['types']:
            kernel_args_type_list.append(out_trans_map[out_type])

        # set kernel_signature
        kernel_signature = "void(*)(" + ", ".join(kernel_args_type_list) + ")"

        return input_tensor_code, kernel_args[:-2], kernel_signature

    def gen_string_tensor_kernel_code(self, inplace_flag=False, code_indent=""):
        input_tensors, kernel_args, kernel_signature = self.get_kernel_args(
            code_indent)
        outputs_args, kernel_output_names, output_create = self.gene_output(
            self.outputs['types'], 'SetStringsKernelOutput', '', inplace_flag)

        return f"""
  // 1. Get kernel signature and kernel
  const auto& kernel = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "{self.kernel['func'][0]}", {{kernel_backend, kernel_layout, kernel_data_type}});
  VLOG(6) << "{self.api} api strings kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  VLOG(6) << "{self.api} api strings kernel: " << kernel;

  // 2. Get Device Context and input
  auto* dev_ctx = GetDeviceContextByBackend(kernel_backend);
  {input_tensors}

  //  3. Set output
  {output_create}
{self.gene_infer_meta(kernel_output_names, code_indent)}

  // 4. run kernel

{code_indent}  using kernel_signature = {kernel_signature};
{code_indent}  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
{code_indent}  (*kernel_fn)({kernel_args}, {outputs_args});

{code_indent}  return {self.gene_return_code()};"""

    def gene_kernel_select(self) -> str:
        api = self.api
        input_names = self.inputs['names']
        attrs = self.attrs
        kernel = self.kernel

        kernel_key_item_init = """
  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::PSTRING_UNION;
  DataType kernel_data_type = DataType::PSTRING;
"""
        # Check the tensor options
        attr_backend_count = 0
        attr_layout_count = 0
        attr_data_type_count = 0
        for attr_name in attrs['names']:
            if attrs['attr_info'][attr_name][0] == 'Backend':
                assert kernel['backend'] is not None, \
                    f"{api} api: When there is a parameter with 'Backend' type in attributes, you must set backend of kernel manually."
                attr_backend_count = attr_backend_count + 1

        # preprocess kernel configures
        kernel_select_code = ""
        if kernel['backend'] is not None:
            if '>' in kernel['backend']:
                vars_list = kernel['backend'].split('>')
                assert len(
                    vars_list
                ) == 2, f"{api} api: The number of params to set backend with '>' only allows 2, but received {len(vars_list)}."
                assert (vars_list[0].strip() in attrs['names']) and (attrs['attr_info'][vars_list[0].strip()][0] == 'const Place&'), \
                    f"{api} api: When use '>' to set kernel backend, the first param should be a attribute with Place type."
                kernel_select_code = kernel_select_code + f"""
  kernel_backend = ParseBackendWithInputOrder({vars_list[0].strip()}, {vars_list[1].strip()});
"""

            else:
                args_str = ""
                for ele in kernel['backend'].split(','):
                    args_str = args_str + ele.strip() + ', '
                kernel_select_code = kernel_select_code + f"""
  kernel_backend = ParseBackend({args_str[:-2]});
"""

        kernel_select_args = ""
        for input_name in input_names:
            kernel_select_args = kernel_select_args + input_name + ", "

        if len(kernel_select_args) > 2:
            kernel_select_args = kernel_select_args[:-2]

        kernel_select_code = kernel_key_item_init + kernel_select_code

        if len(input_names) > 0:
            if self.support_selected_rows_kernel:
                kernel_select_code = kernel_select_code + f"""
  KernelType kernel_type = ParseKernelTypeByInputArgs({", ".join(input_names)});
"""

            kernel_select_code = kernel_select_code + f"""
  auto kernel_key_set = ParseKernelKeyByInputArgs({kernel_select_args});
  auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
  kernel_backend = kernel_key.backend();"""

        return kernel_select_code

    def gene_base_api_code(self, inplace_flag=False):
        api_func_name = self.get_api_func_name()
        return f"""
PADDLE_API {self.outputs['return_type']} {api_func_name}({self.args_str["args_define"]}) {{
{self.gene_kernel_select()}
{self.gen_string_tensor_kernel_code(inplace_flag)}
}}
"""


def header_include():
    return """
#include <tuple>

#include "paddle/phi/api/include/tensor.h"
#include "paddle/phi/common/scalar.h"
#include "paddle/phi/common/int_array.h"
#include "paddle/utils/optional.h"
"""


def source_include(header_file_path):
    return f"""
#include "{header_file_path}"

#include "paddle/phi/api/lib/api_gen_utils.h"
#include "paddle/phi/core/kernel_context.h"
#include "paddle/phi/core/string_tensor.h"
#include "paddle/phi/infermeta/strings/nullary.h"
#include "paddle/phi/infermeta/strings/unary.h"
#include "paddle/phi/api/lib/api_registry.h"
#include "paddle/phi/api/lib/kernel_dispatch.h"
#include "paddle/phi/core/kernel_registry.h"
"""


def api_register():
    return """
PD_REGISTER_API(StringsApi);
"""


def api_namespace():
    return ("""
namespace paddle {
namespace experimental {
namespace strings {

""", """

}  // namespace strings
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

    include_header_file = "paddle/phi/api/include/strings_api.h"
    source_file.write(source_include(include_header_file))
    source_file.write(namespace[0])

    for api in apis:
        strings_api = StringsAPI(api)
        header_file.write(strings_api.gene_api_declaration())
        source_file.write(strings_api.gene_api_code())

    header_file.write(namespace[1])
    source_file.write(namespace[1])

    # source_file.write(api_register())

    header_file.close()
    source_file.close()


def main():
    parser = argparse.ArgumentParser(
        description='Generate PaddlePaddle C++ Strings API files')
    parser.add_argument(
        '--api_yaml_path',
        help='path to sparse api yaml file',
        default='python/paddle/utils/code_gen/strings_api.yaml')

    parser.add_argument(
        '--api_header_path',
        help='output of generated api header code file',
        default='paddle/phi/api/include/strings_api.h')

    parser.add_argument(
        '--api_source_path',
        help='output of generated api source code file',
        default='paddle/phi/api/lib/strings_api.cc')

    options = parser.parse_args()

    api_yaml_path = options.api_yaml_path
    header_file_path = options.api_header_path
    source_file_path = options.api_source_path

    generate_api(api_yaml_path, header_file_path, source_file_path)


if __name__ == '__main__':
    main()
