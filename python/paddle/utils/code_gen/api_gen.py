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
            if 'data_type' not in self.kernel or len(self.kernel['data_type']) == 0:
                self.kernel['data_type'] = None
            if 'param' not in self.kernel or len(self.kernel['param']) == 0:
                self.kernel['param'] = None

            self.infer_meta = api_item_yaml['infer_meta']
            if 'param' not in self.infer_meta or len(self.infer_meta['param']) == 0:
                self.infer_meta['param'] = None
    
    def parse_args(self, args_str) -> dict:
        inputs = {'names' : []}
        attrs = {'names' : [], 'attr_info' : {}}
        args_str = args_str.strip()
        assert args_str.startswith('(') and args_str.endswith(')'), \
            f"Args declaration should start with '(' and end with ')', please check the args of {self.api} in api.yaml."
        args_str = args_str[1:-1]
        args_list = args_str.split(',')
        input_types = ['const Tensor&', 'const Tensor &']
        attr_types = ['const Scalar&', 'const Scalar &', 'int', 'int32_t', 'int64_t', \
                      'size_t', 'float', 'double', 'bool', 'const std::vector<int64_t>&',\
                      'Backend', 'DataLayout', 'DataType']
        args_declare_str = ""
        args_define_str = ""
        for item in args_list:
            item = item.strip()
            # match the input tensor
            has_input = False
            for in_type in input_types:
                if item.startswith(in_type):
                    input_name = item[len(in_type):].strip()
                    assert len(input_name) > 0, f"The input tensor name should not be empty. Please check the args of {self.api} in api.yaml."
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
                    assert len(attr_name) > 0, f"The attribute name should not be empty. Please check the args of {self.api} in api.yaml."
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

        args = {'inputs' : inputs, 'attrs' : attrs, 'args_declare' : args_declare_str[:-2], 'args_define' : args_define_str[:-2]}
        return args

    def gene_api_declaration(self) -> str:
        return f"""
{self.output} {self.api}({self.args['args_declare']});
"""
    
    def gene_kernel_select(self, input_names, attrs, kernel) -> str:

        kernel_key_item_init = """
  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;
"""     
        kernel_key_item_by_attr = ""
        # Set kernel_key info by attr
        attr_backend_count = 0
        attr_layout_count = 0
        attr_data_type_count = 0
        for attr_name in attrs['names']:
            # Check the tensor options
            if attrs['attr_info'][attr_name][0] == 'Backend':
                assert kernel['backend'] is not None, \
                    f"{self.api} api: When there is a parameter with 'Backend' type in attributes, you must set backend of kernel manually."
            if attrs['attr_info'][attr_name][0] == 'DataLayout':
                assert kernel['layout'] is not None, \
                    f"{self.api} api: When there is a parameter with 'DataLayout' type in attributes, you must set layout of kernel manually."
            if attrs['attr_info'][attr_name][0] == 'DataType':
                assert kernel['data_type'] is not None, \
                    f"{self.api} api: When there is a parameter with 'DataType' type in attributes, you must set data_type of kernel manually."
            
            if kernel['backend'] is not None and attr_name in kernel['backend']:
                attr_backend_count = attr_backend_count + 1
                assert attrs['attr_info'][attr_name][0] == 'Backend', f"{self.api} api: The attribute to set backend only allows the type of Backend, but received {attrs['attr_info'][attr_name][0]}."
                assert attr_backend_count <= 1, f"{self.api} api: The number of attributes to set backend only allows 0 or 1, but now received more than 1."
                kernel_key_item_by_attr = kernel_key_item_by_attr + f"""
  kernel_backend = {attr_name};"""
            if kernel['layout'] is not None and attr_name in kernel['layout']:
                attr_layout_count = attr_layout_count + 1
                assert attrs['attr_info'][attr_name][0] == 'DataLayout', f"{self.api} api: The attribute to set layout only allows the type of Layout, but received {attrs['attr_info'][attr_name][0]}."
                assert attr_backend_count <= 1, f"{self.api} api: The number of attributes to set layout only allows 0 or 1, but now received more than 1."
                kernel_key_item_by_attr = kernel_key_item_by_attr + f"""
  kernel_layout = {attr_name};"""
            if kernel['data_type'] is not None and attr_name in kernel['data_type']:
                attr_data_type_count = attr_data_type_count + 1
                assert attrs['attr_info'][attr_name][0] == 'DataType', f"{self.api} api: The attribute to set data_type only allows the type of DataType, but received {attrs['attr_info'][attr_name][0]}."
                assert attr_data_type_count <= 1, f"{self.api} api: The number of attributes to set data_type only allows 0 or 1, but now received more than 1."
                kernel_key_item_by_attr = kernel_key_item_by_attr + f"""
  kernel_data_type = {attr_name};"""

        if len(input_names) == 0:
            assert attr_backend_count == 1 and attr_layout_count == 1 and attr_data_type_count == 1, \
                f"{self.api} api: When there is no input tensor, the args must have 'Backend', 'DataLayout' and 'DataType'."

        # Set kernel_key info by input
        kernel_select_args = ""
        kernel_key_item_by_input = ""

        input_backend_count = 0
        input_layout_count = 0
        input_data_type_count = 0
        
        for input_name in input_names:
            if kernel['backend'] is not None and input_name in kernel['backend']:
                input_backend_count = input_backend_count + 1
                assert input_backend_count <= 1, f"{self.api} api: Currently, the number of inputs to set backend only allows 0 or 1, but now received more than 1."
                kernel_key_item_by_input = kernel_key_item_by_input + f"""
  if (kernel_backend == Backend::UNDEFINED) {{
    kernel_backend = kernel_key_parser.ParseBackend({input_name});
  }}"""
            if kernel['layout'] is not None and input_name in kernel['layout']:
                input_layout_count = input_layout_count + 1
                assert input_layout_count <= 1, f"{self.api} api: Currently, the number of inputs to set layout only allows 0 or 1, but now received more than 1."
                kernel_key_item_by_input = kernel_key_item_by_input + f"""
  if (kernel_layout == DataLayout::UNDEFINED) {{
    kernel_layout = kernel_key_parser.ParseLayout({input_name});
  }}"""
            if kernel['data_type'] is not None and input_name in kernel['data_type']:
                input_data_type_count = input_data_type_count + 1
                assert input_data_type_count <= 1, f"{self.api} api: Currently, the number of inputs to set data_type only allows 0 or 1, but now received more than 1."
                kernel_key_item_by_input = kernel_key_item_by_input + f"""
  if (kernel_data_type == DataType::UNDEFINED) {{
    kernel_data_type = kernel_key_parser.ParseDataType({input_name});
  }}"""     
            kernel_select_args = kernel_select_args + input_name + ", "

        if len(kernel_select_args) > 2:
            kernel_select_args = kernel_select_args[:-2]
        
        if len(kernel_key_item_by_input) > 0:
            kernel_key_parse_code = """

  CustomKernelKeyParser kernel_key_parser;
"""
            kernel_key_item_by_input = kernel_key_parse_code + kernel_key_item_by_input
        
        kernel_select_code = kernel_key_item_init + kernel_key_item_by_attr + kernel_key_item_by_input

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
      "{kernel['func']}", {{kernel_backend, kernel_layout, kernel_data_type}});"""

        return kernel_select_code

    def gene_infer_meta(self, input_names, attr_names, infer_meta) -> str:
        infer_meta_params = infer_meta['param'] if infer_meta['param'] is not None else input_names + attr_names
        param_code = ""
        for param in infer_meta_params:
            if param in input_names:
                param_code = param_code + self.prefix_tensor_name + param + "->meta(), "
            elif param in attr_names:
                param_code = param_code + param + ", "
            else:
                raise ValueError(f"{self.api}: The param {param} of infer_mate is not found in api args.")
        param_code = param_code[:-2]
        return f"""
  auto out_meta = {infer_meta['func']}({param_code});
"""

    def gene_kernel_context(self, input_names, attr_names, infer_meta, kernel_param) -> str:
        if kernel_param is None:
            kernel_param = input_names + attr_names
        # set input for kernel_context
        input_code_str = ""
        for input_name in input_names:
            if input_name in kernel_param:
                input_code_str = input_code_str + f"""
  auto {self.prefix_tensor_name}{input_name} = std::dynamic_pointer_cast<pten::DenseTensor>({input_name}.impl());
  kernel_context.EmplaceBackInput({self.prefix_tensor_name}{input_name});"""
        # set attr for kernel_context
        attr_code_str = ""
        for attr_name in attr_names:
            if attr_name in kernel_param:
                attr_code_str = attr_code_str + f"""
  kernel_context.EmplaceBackAttr({attr_name});"""     
        return f"""
  auto* dev_ctx = GetDeviceContextByBackend(kernel_backend);
  auto kernel_context = pten::KernelContext(*dev_ctx);
{input_code_str}
{attr_code_str}
{self.gene_infer_meta(input_names, attr_names, infer_meta)}
  const auto allocator =
      std::make_shared<paddle::experimental::DefaultAllocator>(
          pten::TransToFluidPlace(kernel_backend));
  auto dense_out = std::make_shared<pten::DenseTensor>(allocator, out_meta);
  kernel_context.EmplaceBackOutput(dense_out);

  Tensor out;
  out.set_impl(dense_out);"""

    def gene_api_code(self):
        if self.is_base_api:
            return f"""
{self.output} {self.api}({self.args["args_define"]}) {{
{self.gene_kernel_select(self.args['inputs']['names'], self.args['attrs'], self.kernel)}
{self.gene_kernel_context(self.args['inputs']['names'], self.args['attrs']['names'], self.infer_meta, self.kernel['param'])}

  kernel(&kernel_context);
  return out;
}}
"""
        else:
            return f"""
{self.output} {self.api}({self.args["args_define"]}) {{
  return {self.invoke};
}}
"""
    
def license():
    return """
// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.    
"""

def header_include():
    return """
#include "paddle/pten/api/include/tensor.h"
#include "paddle/pten/common/data_type.h"
#include "paddle/pten/common/scalar.h"   
"""

def source_include(header_file_path):
    return f"""
#include "{header_file_path}"
#include <memory>

#include "glog/logging.h"

#include "paddle/pten/api/lib/kernel_dispatch.h"
#include "paddle/pten/api/lib/utils/allocator.h"
#include "paddle/pten/core/convert_utils.h"
#include "paddle/pten/core/dense_tensor.h"
#include "paddle/pten/core/kernel_context.h"
#include "paddle/pten/include/core.h"
#include "paddle/pten/include/infershape.h"
"""

def api_namespace():
    return ("""
namespace paddle {
namespace experimental {

""",
"""

}  // namespace experimental
}  // namespace paddle
""")


def generate_api(api_yaml_path, header_file_path, source_file_path):

    with open(api_yaml_path, 'r') as f:
        apis = yaml.load(f, Loader=yaml.FullLoader)
    header_file = open(header_file_path, 'w')
    source_file = open(source_file_path, 'w')

    namespace = api_namespace()

    header_file.write(license())
    header_file.write("#pragma once\n")
    header_file.write(header_include())
    header_file.write(namespace[0])

    source_file.write(license())
    source_file.write(source_include(header_file_path))
    source_file.write(namespace[0])

    for api in apis:
        api_code = API(api)
        print("#######", api_code.gene_api_declaration())
        header_file.write(api_code.gene_api_declaration())
        source_file.write(api_code.gene_api_code())
    
    header_file.write(namespace[1])
    source_file.write(namespace[1])


    header_file.close()
    source_file.close()
    

def main():
    parser = argparse.ArgumentParser(description='Generate PaddlePaddle C++ API files')
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
  