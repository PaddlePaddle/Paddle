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

from api_base import BaseAPI


class BackwardAPI(BaseAPI):
    def __init__(self, backward_item_yaml):
        super(BackwardAPI, self).__init__(backward_item_yaml)
        self.check_args(backward_item_yaml['forward'])

    def get_api_name(self, api_item_yaml):
        return api_item_yaml['backward_api']

    def parse_forward_config(self, forward_config):
        # api_name (const Tensor& input, ... , int attr, ...) -> Tensor(out)
        result = re.search(
            r"(?P<api>[a-z][a-z0-9_]+)\s*(?P<args>\([^\)]+\))\s*->\s*(?P<outputs>.+)",
            forward_config)
        api = result.group('api')
        _, outputs, _ = self.parse_output(self.api, result.group('outputs'))
        fw_inputs, fw_attrs, _, = self.parse_input_and_attr(
            api, result.group('args'))

        return api, fw_inputs, fw_attrs, outputs

    def check_args(self, forward_config):
        # parse the forward and backward config
        _, fw_inputs, fw_attrs, fw_outputs = self.parse_forward_config(
            forward_config)

        # check the inputs of backward
        for input in self.inputs['names']:
            if input not in fw_inputs['names'] and input not in fw_outputs:
                if input.endswith('_grad'):
                    original_name = input[:-5]
                    assert original_name in fw_outputs, \
                        f"{self.api} : Input Tensor error: the input tensor({input}) of backward should be an input or output or grad of output in forward api. \
                         Please check the forward of {self.api} in yaml."

        # check the attributes of backward
        for attr in self.attrs['names']:
            assert attr in fw_attrs['names'] and self.attrs['attr_info'][attr][0] == fw_attrs['attr_info'][attr][0], \
                f"{self.api} : Attribute error: The attribute({attr}) of backward isn't consistent with forward api. \
                 Please check the args of {self.api} in yaml."

        # check the output of backward
        assert len(self.outputs['types']) <= len(fw_inputs['names']), \
            f"{self.api} : Output error: The number of outputs should be less then the number of inputs of forward api. \
             Please check the output of {self.api} in yaml."

    def get_return_type(self, out_type_list):
        return out_type_list[0] if len(
            out_type_list) == 1 else "std::vector<std::vector<Tensor>>"

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
{code_indent}  {self.outputs['return_type']} out{inplace_assign};
{code_indent}  auto kernel_out = {set_out_func}(kernel_backend, &out);"""

        elif len(output_type_list) > 1:
            output_create = f"""
{code_indent}  {self.outputs['return_type']} out({len(output_type_list)});"""

            for i, out_type_item in enumerate(output_type_list):
                kernel_output = kernel_output + f'kernel_out_{i}, '
                output_names.append(f'kernel_out_{i}')
                if out_type_item == 'Tensor':
                    get_out_code = f'&out[{i}][0]'
                    if inplace_flag and self.inplace_map is not None and self.outputs[
                            'names'][i] in self.inplace_map:
                        output_create = output_create + f"""
{code_indent}  out[{i}].emplace_back({self.inplace_map[self.outputs['names'][i]]});"""

                    else:
                        output_create = output_create + f"""
{code_indent}  out[{i}].emplace_back();"""

                else:
                    get_out_code = f'&out[{i}]'
                    if inplace_flag and self.inplace_map is not None and self.outputs[
                            'names'][i] in self.inplace_map:
                        output_create = output_create + f"""
{code_indent}  out[{i}] = {self.inplace_map[self.outputs['names'][i]]};"""

                output_create = output_create + f"""
{code_indent}  auto kernel_out_{i} = {set_out_func}(kernel_backend, {get_out_code});"""

            kernel_output = kernel_output[:-2]
        else:
            raise ValueError(
                "{} : Output error: the output should not be empty.".format(
                    self.api))

        return kernel_output, output_names, output_create


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

#include "paddle/phi/api/lib/api_custom_impl.h"
#include "paddle/phi/api/lib/api_registry.h"
#include "paddle/phi/api/lib/api_utils.h"
#include "paddle/phi/api/lib/data_transform.h"
#include "paddle/phi/api/lib/kernel_dispatch.h"
#include "paddle/phi/api/lib/utils/storage.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/api/include/api.h"
#include "paddle/phi/infermeta/backward.h"
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

    include_header_file = "paddle/phi/api/backward/backward_api.h"
    source_file.write(source_include(include_header_file))
    source_file.write(namespace[0])

    for bw_api in bw_apis:
        bw_api = BackwardAPI(bw_api)
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
        default='paddle/phi/api/backward/backward_api.h')

    parser.add_argument(
        '--backward_source_path',
        help='output of generated backward source code file',
        default='paddle/phi/api/lib/backward_api.cc')

    options = parser.parse_args()

    backward_yaml_path = options.backward_yaml_path
    header_file_path = options.backward_header_path
    source_file_path = options.backward_source_path

    generate_backward_api(backward_yaml_path, header_file_path,
                          source_file_path)


if __name__ == '__main__':
    main()
