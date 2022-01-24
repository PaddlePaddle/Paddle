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


class API:
    prefix_tensor_name = 'dense_'

    def __init__(self, api_item_yaml):
        self.api = api_item_yaml['api']
        # args:
        #   inputs: 
        #     names : [], list of input names
        #   attrs:
        #     names : [], list of attribute names
        #     attr_info : { attr_name : (type, default_values)}    
        self.args = self.parse_args(api_item_yaml['args'])
        self.output = api_item_yaml['output']
        self.is_base_api = True
        if 'invoke' in api_item_yaml:
            self.is_base_api = False
            self.invoke = api_item_yaml['invoke']
        else:
            self.kernel = api_item_yaml['kernel']
            if 'backend' not in self.kernel or len(self.kernel['backend']) == 0:
                self.kernel['backend'] = None
            if 'layout' not in self.kernel or len(self.kernel['layout']) == 0:
                self.kernel['layout'] = None
            if 'data_type' not in self.kernel or len(self.kernel[
                    'data_type']) == 0:
                self.kernel['data_type'] = None
            if 'param' not in self.kernel:
                self.kernel['param'] = None

            self.infer_meta = api_item_yaml['infer_meta']
            if 'param' not in self.infer_meta:
                self.infer_meta['param'] = None

    def parse_args(self, args_str):
        inputs = {'names': []}
        attrs = {'names': [], 'attr_info': {}}
        args_str = args_str.strip()
        assert args_str.startswith('(') and args_str.endswith(')'), \
            f"Args declaration should start with '(' and end with ')', please check the args of {self.api} in api.yaml."
        args_str = args_str[1:-1]
        args_list = args_str.split(',')
        input_types = [
            'const Tensor&', 'const Tensor &', 'const std::vector<Tensor>&',
            'const std::vector<Tensor> &'
        ]
        attr_types = ['const Scalar&', 'const Scalar &', 'const ScalarArray&', 'const ScalarArray &', \
                      'int', 'int32_t', 'int64_t', 'size_t', 'float', 'double', 'bool', \
                      'const std::vector<int64_t>&', 'Backend', 'DataLayout', 'DataType']
        args_declare_str = ""
        args_define_str = ""
        for item in args_list:
            item = item.strip()
            # match the input tensor
            has_input = False
            for in_type in input_types:
                if item.startswith(in_type):
                    input_name = item[len(in_type):].strip()
                    assert len(input_name) > 0, \
                        f"The input tensor name should not be empty. Please check the args of {self.api} in api.yaml."
                    inputs['names'].append(input_name)
                    args_declare_str = args_declare_str + in_type + ' ' + input_name + ', '
                    args_define_str = args_define_str + in_type + ' ' + input_name + ', '
                    has_input = True
                    break
            if has_input:
                continue

            # match the attribute
            for attr_type in attr_types:
                if item.startswith(attr_type):
                    attr_name = item[len(attr_type):].strip()
                    assert len(attr_name) > 0, \
                        f"The attribute name should not be empty. Please check the args of {self.api} in api.yaml."
                    default_value = None
                    if '=' in attr_name:
                        attr_infos = attr_name.split('=')
                        attr_name = attr_infos[0].strip()
                        default_value = attr_infos[1].strip()

                    default_value_str = "" if default_value is None else '=' + default_value
                    args_declare_str = args_declare_str + attr_type + ' ' + attr_name + default_value_str + ', '
                    args_define_str = args_define_str + attr_type + ' ' + attr_name + ', '
                    attrs['names'].append(attr_name)
                    attrs['attr_info'][attr_name] = (attr_type, default_value)
                    break

        args = {
            'inputs': inputs,
            'attrs': attrs,
            'args_declare': args_declare_str[:-2],
            'args_define': args_define_str[:-2]
        }
        return args

    def gene_api_declaration(self):
        return f"""
PADDLE_API {self.output} {self.api}({self.args['args_declare']});
"""

    def gene_kernel_select(self, input_names, attrs, kernel):

        kernel_key_item_init = """
  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;
"""
        # Check the tensor options
        attr_backend_count = 0
        attr_layout_count = 0
        attr_data_type_count = 0
        for attr_name in attrs['names']:
            if attrs['attr_info'][attr_name][0] == 'Backend':
                assert kernel['backend'] is not None, \
                    f"{self.api} api: When there is a parameter with 'Backend' type in attributes, you must set backend of kernel manually."
                attr_backend_count = attr_backend_count + 1
            if attrs['attr_info'][attr_name][0] == 'DataLayout':
                assert kernel['layout'] is not None, \
                    f"{self.api} api: When there is a parameter with 'DataLayout' type in attributes, you must set layout of kernel manually."
                attr_layout_count = attr_layout_count + 1
            if attrs['attr_info'][attr_name][0] == 'DataType':
                assert kernel['data_type'] is not None, \
                    f"{self.api} api: When there is a parameter with 'DataType' type in attributes, you must set data_type of kernel manually."
                attr_data_type_count = attr_data_type_count + 1

        # preprocess kernel configures
        kernel_select_code = ""
        if kernel['backend'] is not None:
            if '>' in kernel['backend']:
                vars_list = kernel['backend'].split('>')
                assert len(
                    vars_list
                ) == 2, f"{self.api} api: The number of params to set backend with '>' only allows 2, but received {len(vars_list)}."
                assert (vars_list[0].strip() in attrs['names']) and (attrs['attr_info'][vars_list[0].strip()][0] == 'Backend'), \
                    f"{self.api} api: When use '>' to set kernel backend, the first param should be a attribute with Backend type."
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

        if kernel['layout'] is not None:
            if '>' in kernel['layout']:
                vars_list = kernel['layout'].split('>')
                assert len(
                    vars_list
                ) == 2, f"{self.api} api: The number of params to set layout with '>' only allows 2, but received {len(vars_list)}."
                assert vars_list[0].strip() in attrs['names'] and attrs['attr_info'][vars_list[0].strip()][0] == 'DataLayout', \
                    f"{self.api} api: When use '>' to set kernel layout, the first param should be a attribute with DataLayout type."
                kernel_select_code = kernel_select_code + f"""
  kernel_layout = ParseLayoutWithInputOrder({vars_list[0].strip()}, {vars_list[1].strip()});
"""

            else:
                vars_list = kernel['layout'].split(',')
                assert len(
                    vars_list
                ) == 1, f"{self.api} api: The number of params to set layout must be 1, but received {len(vars_list)}."
                kernel_select_code = kernel_select_code + f"""
  kernel_layout = ParseLayout({vars_list[0].strip()});
"""

        if kernel['data_type'] is not None:
            if '>' in kernel['data_type']:
                vars_list = kernel['data_type'].split('>')
                assert len(
                    vars_list
                ) == 2, f"{self.api} api: The number of params to set data_type with '>' only allows 2, but received {len(vars_list)}."
                assert vars_list[0].strip() in attrs['names'] and attrs['attr_info'][vars_list[0].strip()][0] == 'DataType', \
                    f"{self.api} api: When use '>' to set kernel data_type, the first param should be a attribute with DataType type."
                kernel_select_code = kernel_select_code + f"""
  kernel_data_type = ParseDataTypeWithInputOrder({vars_list[0].strip()}, {vars_list[1].strip()});
"""

            else:
                vars_list = kernel['data_type'].split(',')
                assert len(
                    vars_list
                ) == 1, f"{self.api} api: The number of params to set data_type only allows 2, but received {len(vars_list)}."
                kernel_select_code = kernel_select_code + f"""
  kernel_data_type = ParseDataType({vars_list[0].strip()});
"""

        if len(input_names) == 0:
            assert attr_backend_count > 0 and attr_layout_count > 0 and attr_data_type_count > 0, \
                f"{self.api} api: When there is no input tensor, the args must have 'Backend', 'DataLayout' and 'DataType'."

        kernel_select_args = ""
        for input_name in input_names:
            kernel_select_args = kernel_select_args + input_name + ", "

        if len(kernel_select_args) > 2:
            kernel_select_args = kernel_select_args[:-2]

        kernel_select_code = kernel_key_item_init + kernel_select_code

        if len(input_names) > 0:
            kernel_select_code = kernel_select_code + f"""
  if (kernel_backend == Backend::UNDEFINED 
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {{
    auto kernel_key_set = ParseKernelKeyByInputArgs({kernel_select_args});
    auto kernel_key = kernel_key_set.GetHigestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {{
      kernel_backend = kernel_key.backend();
    }}
    if (kernel_layout == DataLayout::UNDEFINED) {{
      kernel_layout = kernel_key.layout();
    }}
    if (kernel_data_type == DataType::UNDEFINED) {{
      kernel_data_type = kernel_key.dtype();
    }}
  }}"""

        kernel_select_code = kernel_select_code + f"""
  auto kernel = pten::KernelFactory::Instance().SelectKernelOrThrowError(
      "{kernel['func']}", {{kernel_backend, kernel_layout, kernel_data_type}});
  VLOG(6) << "{self.api} API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  VLOG(6) << "{self.api} API kernel: " << kernel;"""

        return kernel_select_code

    def gene_infer_meta(self, input_names, attr_names, infer_meta) -> str:
        infer_meta_params = infer_meta['param'] if infer_meta[
            'param'] is not None else input_names + attr_names
        param_code = ""
        for param in infer_meta_params:
            if param in input_names:
                param_code = param_code + "GetDenseTensorMeta(" + self.prefix_tensor_name + param + "), "
            elif param in attr_names:
                param_code = param_code + param + ", "
            elif isinstance(param, str):
                param_code = param_code + "\"" + param + "\", "
            elif isinstance(param, bool):
                param_code = param_code + str(param).lower() + ", "
            else:
                param_code = param_code + str(param) + ", "

        param_code = param_code[:-2]
        return f"""
  auto out_meta = pten::{infer_meta['func']}({param_code});
"""

    def get_kernel_args(self, input_names, attrs, kernel_param):
        input_tensor_code = ""
        for input_name in input_names:
            # set input code
            input_tensor_code = input_tensor_code + f"""
  auto {self.prefix_tensor_name}{input_name} = TensorToDenseTensor({input_name});"""

        attr_names = attrs['names']
        if kernel_param is None:
            kernel_param = input_names + attr_names

        kernel_args = "*dev_ctx, "
        for param in kernel_param:
            if param in input_names:
                kernel_args = kernel_args + "*" + self.prefix_tensor_name + param + ", "
            elif param in attr_names:
                # set attr for kernel_context
                if 'ScalarArray' in attrs['attr_info'][param][0]:
                    param = 'pten::ScalarArray(' + param + ')'
                elif 'Scalar' in attrs['attr_info'][param][0]:
                    param = 'pten::Scalar(' + param + ')'
                kernel_args = kernel_args + param + ", "
            elif isinstance(param, bool):
                kernel_args = kernel_args + str(param).lower() + ", "
            else:
                kernel_args = kernel_args + str(param) + ", "
        return input_tensor_code, kernel_args[:-2]

    def gene_api_code(self):
        if self.is_base_api:
            input_tensors, kernel_args = self.get_kernel_args(
                self.args['inputs']['names'], self.args['attrs'],
                self.kernel['param'])
            return f"""
PADDLE_API {self.output} {self.api}({self.args["args_define"]}) {{
{self.gene_kernel_select(self.args['inputs']['names'], self.args['attrs'], self.kernel)}

  auto* dev_ctx = GetDeviceContextByBackend(kernel_backend);
{input_tensors}
{self.gene_infer_meta(self.args['inputs']['names'], self.args['attrs']['names'], self.infer_meta)}
  auto dense_out = std::make_shared<pten::DenseTensor>(
        pten::make_intrusive<paddle::experimental::SharedStorage>(
            pten::TransToFluidPlace(kernel_backend)),
        std::move(out_meta));

  Tensor out;
  out.set_impl(dense_out);

  auto* kernel_fn = kernel.GetVariadicKernelFn<pten::{self.api}_kernel>();
  (*kernel_fn)({kernel_args}, dense_out.get());

  return out;
}}
"""

        else:
            return f"""
PADDLE_API {self.output} {self.api}({self.args["args_define"]}) {{
  return {self.invoke};
}}
"""


def header_include():
    return """
#include "paddle/pten/api/include/tensor.h"
#include "paddle/pten/common/scalar.h"
#include "paddle/pten/common/scalar_array.h"
"""


def source_include(header_file_path):
    return f"""
#include "{header_file_path}"
#include <memory>

#include "glog/logging.h"

#include "paddle/pten/api/include/kernel_signature.h"
#include "paddle/pten/api/lib/api_registry.h"
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
PT_REGISTER_API(Creation);
PT_REGISTER_API(Linalg);
PT_REGISTER_API(Manipulation);
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


def tensor_to_densetensor():
    return """
  std::shared_ptr<pten::DenseTensor> TensorToDenseTensor(const Tensor& tensor) {
      return std::dynamic_pointer_cast<pten::DenseTensor>(tensor.impl());
  }

  std::shared_ptr<std::vector<pten::DenseTensor>> TensorToDenseTensor(const std::vector<Tensor>& tensors) {
      std::vector<pten::DenseTensor> pt_tensors;

      for(auto & t : tensors) {
          pt_tensors.push_back(*std::dynamic_pointer_cast<pten::DenseTensor>(t.impl()));
      }
      return std::make_shared<std::vector<pten::DenseTensor>>(pt_tensors);
  }

   const pten::DenseTensorMeta GetDenseTensorMeta(const std::shared_ptr<pten::DenseTensor> & x) {
       return x->meta();
   }

   const std::vector<pten::DenseTensorMeta> GetDenseTensorMeta(const std::shared_ptr<std::vector<pten::DenseTensor>>& x) {
       std::vector<pten::DenseTensorMeta> metas;
       for(auto& t : *x) {
           metas.push_back(t.meta());
       }
       return metas;
   }
"""


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
    source_file.write(tensor_to_densetensor())

    for api in apis:
        api_code = API(api)
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
        help='path to yaml file directory',
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
