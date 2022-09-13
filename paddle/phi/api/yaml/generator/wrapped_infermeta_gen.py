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

from api_gen import ForwardAPI

kernel_func_set = set()


def get_wrapped_infermeta_name(api_name):
    return api_name.capitalize() + 'InferMeta'


def gene_wrapped_infermeta_and_register(api):
    if api.is_base_api and not api.is_dygraph_api:
        register_code = f"""
PD_REGISTER_INFER_META_FN({api.kernel['func'][0]}, phi::{api.infer_meta['func']});"""

        if api.infer_meta['param'] is not None:
            if api.kernel['func'][0] in kernel_func_set:
                return '', '', ''

            kernel_params = api.kernel['param']
            if kernel_params is None:
                kernel_params = api.inputs['names'] + api.attrs['names']
            if kernel_params == api.infer_meta['param']:
                return '', '', register_code

            assert len(api.infer_meta['param']) <= len(kernel_params), \
                 f"{api.api} api: Parameters error. The params of infer_meta should be a subset of kernel params."

            tensor_type_map = {
                'const Tensor&': 'const MetaTensor&',
                'const std::vector<Tensor>&': 'const std::vector<MetaTensor>&',
                'Tensor': 'MetaTensor*',
                'std::vector<Tensor>': 'std::vector<MetaTensor>*',
                'const paddle::optional<Tensor>&': 'const MetaTensor&'
            }

            wrapped_infermeta_name = get_wrapped_infermeta_name(
                api.kernel['func'][0])
            args = []
            for input_name in api.inputs['names']:
                if input_name in kernel_params:
                    args.append(
                        tensor_type_map[api.inputs['input_info'][input_name]] +
                        ' ' + input_name)
            for attr_name in api.attrs['names']:
                if attr_name in kernel_params:
                    args.append(api.attrs['attr_info'][attr_name][0] + ' ' +
                                attr_name)
            for i, out_type in enumerate(api.outputs['types']):
                args.append(tensor_type_map[out_type] + ' ' +
                            api.outputs['names'][i])

            invoke_param = api.infer_meta['param']
            invoke_param.extend(api.outputs['names'])

            declare_code = f"""
void {wrapped_infermeta_name}({", ".join(args)});
"""

            defind_code = f"""
void {wrapped_infermeta_name}({", ".join(args)}) {{
  {api.infer_meta['func']}({", ".join(invoke_param)});
}}
"""

            register_code = f"""
PD_REGISTER_INFER_META_FN({api.kernel['func'][0]}, phi::{get_wrapped_infermeta_name(api.kernel['func'][0])});"""

            kernel_func_set.add(api.kernel['func'][0])
            return declare_code, defind_code, register_code
        else:
            return '', '', register_code
    else:
        return '', '', ''


def header_include():
    return """
#include "paddle/phi/core/meta_tensor.h"
#include "paddle/phi/common/scalar.h"
#include "paddle/phi/common/int_array.h"
"""


def source_include(header_file_path):
    return f"""
#include "{header_file_path}"
#include "paddle/phi/core/infermeta_utils.h"
#include "paddle/phi/infermeta/binary.h"
#include "paddle/phi/infermeta/multiary.h"
#include "paddle/phi/infermeta/nullary.h"
#include "paddle/phi/infermeta/unary.h"
#include "paddle/phi/infermeta/ternary.h"
"""


def api_namespace():
    return ("""
namespace phi {
""", """
}  // namespace phi
""")


def generate_wrapped_infermeta_and_register(api_yaml_path, header_file_path,
                                            source_file_path):
    apis = []
    for each_api_yaml in api_yaml_path:
        with open(each_api_yaml, 'r') as f:
            api_list = yaml.load(f, Loader=yaml.FullLoader)
            if api_list:
                apis.extend(api_list)

    header_file = open(header_file_path, 'w')
    source_file = open(source_file_path, 'w')

    namespace = api_namespace()

    header_file.write("#pragma once\n")
    header_file.write(header_include())
    header_file.write(namespace[0])

    include_header_file = "paddle/phi/infermeta/generated.h"
    source_file.write(source_include(include_header_file))
    source_file.write(namespace[0])

    infermeta_register_code = ''

    for api in apis:
        api_item = ForwardAPI(api)
        declare_code, defind_code, register_code = gene_wrapped_infermeta_and_register(
            api_item)
        header_file.write(declare_code)
        source_file.write(defind_code)
        if infermeta_register_code.find(register_code) == -1:
            infermeta_register_code = infermeta_register_code + register_code

    header_file.write(namespace[1])
    source_file.write(namespace[1])

    source_file.write(infermeta_register_code)

    header_file.close()
    source_file.close()


def main():
    parser = argparse.ArgumentParser(
        description='Generate PaddlePaddle C++ API files')
    parser.add_argument('--api_yaml_path',
                        help='path to api yaml file',
                        nargs='+',
                        default='paddle/phi/api/yaml/ops.yaml')
    parser.add_argument(
        '--wrapped_infermeta_header_path',
        help='output of generated wrapped_infermeta header code file',
        default='paddle/phi/infermeta/generated.h')

    parser.add_argument(
        '--wrapped_infermeta_source_path',
        help='output of generated wrapped_infermeta source code file',
        default='paddle/phi/infermeta/generated.cc')

    options = parser.parse_args()

    api_yaml_path = options.api_yaml_path
    header_file_path = options.wrapped_infermeta_header_path
    source_file_path = options.wrapped_infermeta_source_path

    generate_wrapped_infermeta_and_register(api_yaml_path, header_file_path,
                                            source_file_path)


if __name__ == '__main__':
    main()
