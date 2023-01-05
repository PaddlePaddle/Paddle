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

import yaml
import argparse
import os
import re

from prim_base import PREFIX_TENSOR_NAME, BaseAPI

#########
# class #
#########
inplace_out_type_map = {
    "Tensor": "Tensor&",
    "std::vector<Tensor>": "std::vector<Tensor>&",
}

inplace_optional_out_type_map = {
    "Tensor": "paddle::optional<Tensor>&",
    "std::vector<Tensor>": "paddle::optional<std::vector<Tensor>>&",
}

class PrimAPI(BaseAPI):
    def __init__(self, api_item_yaml):
        super().__init__(api_item_yaml)
        self.is_prim_api = False
        if api_item_yaml['op'] in white_ops_list:
            self.is_prim_api = True

        self.inplace_map, self.view_map = self.parse_inplace_and_view(
            api_item_yaml
        )

    def is_inplace(self):
        if len(self.inplace_map) > 0:
            return True
        return False

    def get_api__func_name(self):
        api_func_name = self.api
        # if self.is_inplace:
        #     if api_func_name[-1] != '_':
        #         api_func_name += '_'
        # print("after api name", api_func_name)
        return api_func_name

    

    def gene_prim_api_declaration(self):
        api_declaration = ""
        api_func_name = self.get_api__func_name()
        if api_func_name[-1] != '_':
            api_declaration = f"""
template <typename T>
{self.get_return_type()} {api_func_name}({self.get_declare_args()});
"""
        else:
            api_declaration = (
                api_declaration
                + f"""
template <typename T>
{self.get_return_type(inplace_flag=True)} {api_func_name}({self.get_declare_args(inplace_flag=True)});
"""
            )

        return api_declaration

    def get_ad_func_input_args(self, inplace_flag=False):
        input_args = []
        for name in self.inputs['names']:
            name = name.split('@')[0]
            if inplace_flag and name in self.inplace_map.values():
                input_args.append(name)
            else:
                input_args.append(name)
        return input_args

    def get_ad_func_args(self, inplace_flag=False):
        ad_func_args = self.get_ad_func_input_args(inplace_flag)
        for name in self.attrs['names']:
            default_value = ''
            if self.attrs['attr_info'][name][1] is not None:
                default_value = ' = ' + self.attrs['attr_info'][name][1]
            ad_func_args.append(name)

        ad_func_args_str = ", ".join(ad_func_args)
        return ad_func_args_str

    def gene_ad_func_call(self):
        api_func_name = self.get_api__func_name()
        
        dygraph_ad_func_name = '::' + api_func_name + '_ad_func'
        dygraph_ad_func_parameters = self.get_ad_func_args()
        
        ad_func_call_str = f"""
VLOG(4) << "Eager Prim API {api_func_name}_ad_func call";
return {dygraph_ad_func_name}({dygraph_ad_func_parameters});
"""
        #print("ad_func_call_str: ", ad_func_call_str)
        return ad_func_call_str

    def gene_eager_prim_api_code(self):
        api_code = ""
        indent = "  "
        api_func_name = self.get_api__func_name()
        template = '<Tensor>'
        # func decalaration 
        if api_func_name[-1] != '_':
            api_code = f"""
template <>
{self.get_return_type()} {api_func_name}{template}({self.get_declare_args()})
"""
        else:
            api_code = f"""
template <>
{self.get_return_type(inplace_flag=True)} {api_func_name}{template}({self.get_declare_args(inplace_flag=True)})
"""
        #func code

        api_code = api_code + '{'
        api_code += f"""{self.gene_ad_func_call()} """
        api_code += '}'+ '\n'

        return api_code

    def parse_inplace_and_view(self, api_item_yaml):
        inplace_map, view_map = {}, {}
        for mode in ['inplace', 'view']:
            if mode in api_item_yaml:
                if mode == 'inplace':
                    inplace_map = {}
                else:
                    view_map = {}
                in_out_mapping_list = api_item_yaml[mode].split(',')
                for item in in_out_mapping_list:
                    result = re.search(r"(?P<in>\w+)\s*->\s*(?P<out>\w+)", item)
                    in_val = result.group('in')
                    out_val = result.group('out')
                    assert (
                        in_val in self.inputs['names']
                    ), f"{self.api} : {mode} input error: the input var name('{in_val}') is not found in the input args of {self.api}."
                    assert (
                        out_val in self.outputs['names']
                    ), f"{self.api} : {mode} output error: the output var name('{out_val}') is not found in the output args of {self.api}."

                    if mode == 'inplace':
                        inplace_map[out_val] = in_val
                    else:
                        view_map[out_val] = in_val

        return inplace_map, view_map

    def get_return_type(self, inplace_flag=False):
        out_type_list = []
        for i, out_type in enumerate(self.outputs['types']):
            out_name = self.outputs['names'][i].split('@')[0]
            if inplace_flag and out_name in self.inplace_map:
                if self.inplace_map[out_name] in self.optional_vars:
                    out_type_list.append(
                        inplace_optional_out_type_map[out_type]
                    )
                else:
                    out_type_list.append(inplace_out_type_map[out_type])
            else:
                out_type_list.append(out_type)
        if len(out_type_list) == 1:
            return out_type_list[0]
        else:
            return "std::tuple<" + ", ".join(out_type_list) + ">"


white_ops_list = [
    "pow",
    "scale",
    "multiply",
]


def header_include():
    return """
#include "paddle/phi/common/scalar.h"
#include "paddle/utils/optional.h"
"""


def source_include(header_file_path):
    return f"""
#include "paddle/fluid/eager/api/all.h"
#include "paddle/fluid/eager/api/generated/eager_generated/forwards/dygraph_functions.h"
#include "paddle/fluid/prim/api/generated/prim_api/prim_api.h"

"""


def api_namespace():
    return (
        """
namespace paddle {
namespace prim {
""",
        """
using Tensor = paddle::experimental::Tensor;
using Scalar = paddle::experimental::Scalar;
""",
        """
}  // namespace prim
}  // namespace paddle
""",
    )


def generate_api(api_yaml_path, header_file_path, eager_prim_source_file_path):
    apis = []

    for each_api_yaml in api_yaml_path:
        with open(each_api_yaml, 'r') as f:
            api_list = yaml.load(f, Loader=yaml.FullLoader)
            if api_list:
                apis.extend(api_list)

    header_file = open(header_file_path, 'w')
    eager_prim_source_file = open(eager_prim_source_file_path, 'w')

    namespace = api_namespace()

    header_file.write("#pragma once\n")
    header_file.write(header_include())
    header_file.write(namespace[0])
    header_file.write(namespace[1])
    include_header_file = "#include paddle/fluid/prim/api/generated/prim_api/prim_api.h"
    eager_prim_source_file.write(source_include(include_header_file))
    eager_prim_source_file.write(namespace[0])

    for api in apis:
        prim_api = PrimAPI(api)
        if prim_api.is_prim_api:
            header_file.write(prim_api.gene_prim_api_declaration())
            eager_prim_source_file.write(prim_api.gene_eager_prim_api_code())

    header_file.write(namespace[2])
    eager_prim_source_file.write(namespace[2])

    header_file.close()
    eager_prim_source_file.close()

def main():
    parser = argparse.ArgumentParser(
        description='Generate PaddlePaddle C++ API files'
    )
    parser.add_argument(
        '--api_yaml_path',
        help='path to api yaml file',
        nargs='+',
        default=['paddle/phi/api/yaml/ops.yaml'],
    )

    parser.add_argument(
        '--prim_api_header_path',
        help='output of generated prim_api header code file',
        default='paddle/fluid/prim/api/generated/prim_api/prim_api.h',
    )

    parser.add_argument(
        '--eager_prim_api_source_path',
        help='output of generated eager_prim_api source code file',
        default='paddle/fluid/prim/api/generated/prim_api/eager_prim_api.cc',
    )

    options = parser.parse_args()

    api_yaml_path = options.api_yaml_path
    prim_api_header_file_path = options.prim_api_header_path
    eager_prim_api_source_file_path = options.eager_prim_api_source_path

    generate_api(api_yaml_path, prim_api_header_file_path, eager_prim_api_source_file_path)


if __name__ == '__main__':
    main()
