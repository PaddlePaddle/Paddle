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
        self.backward_api = backward_item_yaml['backward_api']
        self.args, self.output_type_list, self.return_comment = self.parse_and_check_args(
            backward_item_yaml['forward'], backward_item_yaml['args'],
            backward_item_yaml['output'])
        self.return_type = self.output_type_list[0] if len(
            self.output_type_list) == 1 else "std::vector<std::vector<Tensor>>"

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
        forward_args = gen_utils.parse_args(api, result.group('args'))

        return api, forward_args['inputs'], forward_args['attrs'], outputs

    def parse_and_check_args(self, forward_config, args_config, output_config):
        # parse the forward and backward config
        _, fw_inputs, fw_attrs, fw_outputs = self.parse_forward_config(
            forward_config)
        bw_args = gen_utils.parse_args(self.backward_api, args_config)

        # check the inputs of backward
        for input in bw_args['inputs']['names']:
            if input not in fw_inputs and input not in fw_outputs:
                if input.endswith('_grad'):
                    original_name = input[:-5]
                    assert original_name in fw_outputs, \
                        f"{self.backward_api} : Input Tensor error: the input tensor({input}) of backward should be an input or output or grad of output in forward api. \
                         Please check the forward of {self.backward_api} in yaml."

        # check the attributes of backward
        for attr in bw_args['attrs']['names']:
            assert attr in fw_attrs['names'] and bw_args['attrs']['attr_info'][attr][0] == fw_attrs['attr_info'][attr][0], \
                f"{self.backward_api} : Attribute error: The attribute({attr}) of backward isn't consistent with forward api. \
                 Please check the args of {self.backward_api} in yaml."

        # check the output of backward
        out_type_list, return_comment = gen_utils.parse_output(
            self.backward_api, output_config)
        assert len(out_type_list) <= len(fw_inputs['names']), \
            f"{self.backward_api} : Output error: The number of ouputs should be less then the number of inputs of forward api. \
             Please check the output of {self.backward_api} in yaml."

        return bw_args, out_type_list, return_comment

    def gene_api_declaration(self):
        if self.return_comment:
            return f"""
// {self.return_comment}
{self.return_type} {self.backward_api}({self.args['args_declare']});
"""

        else:
            return f"""
{self.return_type} {self.backward_api}({self.args['args_declare']});
"""

    def gene_output(self, output_type_list):
        kernel_output = ""
        output_names = []
        output_create = ""

        if len(output_type_list) == 1:
            kernel_output = 'dense_out'
            output_names.append('dense_out')
            output_create = f"""
  {self.return_type} out;
  auto dense_out = SetKernelOutput(kernel_backend, &out);"""

        elif len(output_type_list) > 1:
            output_create = f"""
  {self.return_type} out({len(output_type_list)});"""

            for i, out_type_item in enumerate(output_type_list):
                kernel_output = kernel_output + f'dense_out_{i}, '
                output_names.append(f'dense_out_{i}')
                if out_type_item == 'Tensor':
                    get_out_code = f'&out[{i}][0]'
                    output_create = output_create + f"""
  out[{i}].emplace_back();"""

                else:
                    get_out_code = f'&out[{i}]'
                output_create = output_create + f"""
  auto dense_out_{i} = SetKernelOutput(kernel_backend, {get_out_code});"""

            kernel_output = kernel_output[:-2]
        else:
            raise ValueError(
                "{} : Output error: the output should not be empty.".format(
                    self.backward_api))

        return kernel_output, output_names, output_create

    def gene_api_code(self):
        if self.is_base_api:
            input_tensors, kernel_args, kernel_signature = gen_utils.get_kernel_args(
                self.args['inputs'], self.args['attrs'], self.output_type_list,
                self.kernel['param'])
            outputs_args, output_names, output_create = self.gene_output(
                self.output_type_list)
            return f"""
// {self.return_comment}
{self.return_type} {self.backward_api}({self.args["args_define"]}) {{
{gen_utils.gene_kernel_select(self.backward_api, self.args['inputs']['names'], self.args['attrs'], self.kernel)}

  auto* dev_ctx = GetDeviceContextByBackend(kernel_backend);
{input_tensors}
{output_create}
{gen_utils.gene_infer_meta(self.args['inputs']['names'], self.args['attrs']['names'], output_names, self.infer_meta)}

  using kernel_signature = {kernel_signature};
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
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
{self.return_type} {self.backward_api}({params_code}) {{
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

#include "paddle/pten/api/lib/api_registry.h"
#include "paddle/pten/api/lib/api_utils.h"
#include "paddle/pten/api/lib/kernel_dispatch.h"
#include "paddle/pten/api/lib/utils/storage.h"
#include "paddle/pten/core/kernel_registry.h"
#include "paddle/pten/api/include/api.h"
#include "paddle/pten/infermeta/backward.h"
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

    include_header_file = "paddle/pten/api/backward/backward_api.h"
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
        default='paddle/pten/api/backward/backward_api.h')

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
