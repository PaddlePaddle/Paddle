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
import re

import gen_utils


class BackwardAPI:
    def __init__(self, backward_item_yaml):
        self.grad_api = backward_item_yaml['grad_api']
        self.args, self.output_type, self.return_comment = self.parse_and_check_args(
            backward_item_yaml['forward'], backward_item_yaml['args'],
            backward_item_yaml['output'])

        self.is_base_api = True
        if 'invoke' in backward_item_yaml:
            self.is_base_api = False
            self.invoke = backward_item_yaml['invoke']
        else:
            self.kernel = backward_item_yaml['kernel']
            if 'backend' not in self.kernel or len(self.kernel['backend']) == 0:
                self.kernel['backend'] = None
            if 'layout' not in self.kernel or len(self.kernel['layout']) == 0:
                self.kernel['layout'] = None
            if 'data_type' not in self.kernel or len(self.kernel[
                    'data_type']) == 0:
                self.kernel['data_type'] = None
            if 'param' not in self.kernel or len(self.kernel['param']) == 0:
                self.kernel['param'] = None

            self.infer_meta = backward_item_yaml['infer_meta']
            if 'param' not in self.infer_meta or len(self.infer_meta[
                    'param']) == 0:
                self.infer_meta['param'] = None

    def parse_forward_config(self, forward_config):
        # api_name (const Tensor& input, ... , int attr, ...) -> Tensor(out)
        result = re.search(
            r"(?P<api>[a-z][a-z0-9_]+)\s*(?P<args>\([^\)]+\))\s*->[^\(]*\((?P<outputs>[^\)]+)\)",
            forward_config)
        api = result.group('api')
        outputs = [item.strip() for item in result.group('outputs').split(',')]
        forward_args = gen_utils.parse_args(result.group('args'))

        return api, forward_args['inputs'], forward_args['attrs'], outputs

    def parse_backward_output(self, output_config):
        def parse_output_item(output_item):
            result = re.search(
                r"(?P<out_type>[a-zA-Z0-9_<>]+)\s*\((?P<name>\w+)\)",
                output_item)
            out_type = result.group('out_type')
            assert out_type in ['Tensor', 'std::vector<Tensor>'], \
                f"{self.grad_api} : Output type error: the output type only support Tensor and std::vector<Tensor>, \
                  but now is {out_type}."

            return out_type, result.group('name')

        temp_list = output_config.split(',')

        if len(temp_list) == 1:
            out_type, out_name = parse_output_item(temp_list[0])
            return out_type, 1, out_name
        else:
            out_type_list = []
            out_name_list = []
            for output_item in temp_list:
                out_type, out_name = parse_output_item(output_item)
                out_type_list.append(out_type)
                out_name_list.append(out_name)

            return "std::tuple<" + ",".join(out_type_list) + ">", len(
                temp_list), ", ".join(out_name_list)

    def parse_and_check_args(self, forward_config, args_config, output_config):
        # parse the forward and backward config
        _, fw_inputs, fw_attrs, fw_outputs = self.parse_forward_config(
            forward_config)
        bw_args = gen_utils.parse_args(args_config)

        # check the inputs of backward
        for input in bw_args['inputs']['names']:
            if input not in fw_inputs and input not in fw_outputs:
                if input.endswith('_grad'):
                    original_name = input[:-5]
                    assert original_name in fw_outputs, \
                        f"{self.grad_api} : Input Tensor error: the input tensor({input}) of backward should be an input or output or grad of output in forward api. \
                         Please check the forward of {self.grad_api} in yaml."

        # check the attributes of backward
        for attr in bw_args['attrs']['names']:
            assert attr in fw_attrs['names'] and bw_args['attrs']['attr_info'][attr][0] == fw_attrs['attr_info'][attr][0], \
                f"{self.grad_api} : Attribute error: The attribute({attr}) of backward isn't consistent with forward api. \
                 Please check the args of {self.grad_api} in yaml."

        # check the output of backward
        output_type, output_num, return_comment = self.parse_backward_output(
            output_config)
        assert output_num <= len(fw_inputs['names']), \
            f"{self.grad_api} : Output error: The number of ouputs should be less then the number of inputs of forward api. \
             Please check the output of {self.grad_api} in yaml."

        return bw_args, output_type, return_comment

    def gene_api_declaration(self):
        if self.return_comment:
            return f"""
// {self.return_comment}
{self.output_type} {self.grad_api}({self.args['args_declare']});
"""

        else:
            return f"""
{self.output_type} {self.grad_api}({self.args['args_declare']});
"""

    def gene_api_code(self):
        if self.is_base_api:
            input_tensors, kernel_args = gen_utils.get_kernel_args(
                self.args['inputs']['names'], self.args['attrs'],
                self.kernel['param'])
            outputs_args, output_create = gen_utils.gene_output(
                self.output_type)
            return f"""
// {self.return_comment}
{self.output_type} {self.grad_api}({self.args["args_define"]}) {{
{gen_utils.gene_kernel_select(self.grad_api, self.args['inputs']['names'], self.args['attrs'], self.kernel)}

  auto* dev_ctx = GetDeviceContextByBackend(kernel_backend);
{input_tensors}
{gen_utils.gene_infer_meta(self.args['inputs']['names'], self.args['attrs']['names'], self.infer_meta)}
{output_create}

  auto* kernel_fn = kernel.GetVariadicKernelFn<pten::{self.grad_api}_kernel>();
  (*kernel_fn)({kernel_args}, {outputs_args});

  return out;
}}
"""

        else:
            inveke_func_name = self.invoke.split('(')[0].strip()
            if inveke_func_name in self.args['attrs']['names']:
                # Adjust the param whose name is same with api invoked.
                pattern = '\W' + inveke_func_name + '[^A-Za-z0-9_(]'

                def adjust_name(matched):
                    matched_str = matched.group()
                    return matched_str[0:-1] + '_val' + matched_str[-1]

                invoke_code = re.sub(pattern, adjust_name, self.invoke)
                params_code = re.sub(pattern, adjust_name,
                                     self.args["args_define"])
            else:
                invoke_code = self.invoke
                params_code = self.args["args_define"]
            return f"""
// {self.return_comment}
{self.output_type} {self.grad_api}({params_code}) {{
  return {invoke_code};
}}
"""


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

#include "paddle/pten/api/include/kernel_signature.h"
#include "paddle/pten/api/lib/api_registry.h"
#include "paddle/pten/api/lib/kernel_dispatch.h"
#include "paddle/pten/api/lib/tensor_adapt.h"
#include "paddle/pten/api/lib/utils/storage.h"
#include "paddle/pten/core/kernel_registry.h"
#include "paddle/pten/api/include/api.h"
#include "paddle/pten/infermeta/grad_infermeta.h"
"""


def backward_api_namespace():
    return ("""
namespace paddle {
namespace experimental {

""", """

}  // namespace experimental
}  // namespace paddle
""")


def generate_backward_api(backward_yaml_path, header_file_path,
                          source_file_path):

    with open(backward_yaml_path, 'r') as f:
        bw_apis = yaml.load(f, Loader=yaml.FullLoader)
    header_file = open(header_file_path, 'w')
    source_file = open(source_file_path, 'w')

    namespace = backward_api_namespace()

    header_file.write("#pragma once\n")
    header_file.write(header_include())
    header_file.write(namespace[0])

    include_header_file = "paddle/pten/api/include/backward_api.h"
    source_file.write(source_include(include_header_file))
    source_file.write(namespace[0])

    for bw_api in bw_apis:
        bw_api = BackwardAPI(bw_api)
        # print(api_code.gene_api_declaration())
        header_file.write(bw_api.gene_api_declaration())
        source_file.write(bw_api.gene_api_code())

    header_file.write(namespace[1])
    source_file.write(namespace[1])

    header_file.close()
    source_file.close()


def main():
    parser = argparse.ArgumentParser(
        description='Generate PaddlePaddle C++ backward API files')
    parser.add_argument(
        '--backward_yaml_path',
        help='path to backward yaml file',
        default='python/paddle/utils/code_gen/backward.yaml')
    parser.add_argument(
        '--backward_header_path',
        help='output of generated backward header code file',
        default='paddle/pten/api/include/backward_api.h')

    parser.add_argument(
        '--backward_source_path',
        help='output of generated backward source code file',
        default='paddle/pten/api/lib/backward_api.cc')

    options = parser.parse_args()

    backward_yaml_path = options.backward_yaml_path
    header_file_path = options.backward_header_path
    source_file_path = options.backward_source_path

    generate_backward_api(backward_yaml_path, header_file_path,
                          source_file_path)


if __name__ == '__main__':
    main()
