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

from api_base import BaseAPI


def get_wrapped_infermeta_name(api_name):
    return api_name.capitalize() + 'InferMeta'


def gene_wrapped_infermeta_and_register(api):
    if api.is_base_api:
        register_code = f"""
PT_REGISTER_INFER_META_FN({api.kernel['func'][0]}, pten::{api.infer_meta['func']});"""

        if api.infer_meta['param'] is not None:
            tensor_type_map = {
                'const Tensor&': 'const MetaTensor&',
                'const std::vector<Tensor>&': 'const std::vector<MetaTensor>&',
                'Tensor': 'MetaTensor*',
                'std::vector<Tensor>': 'std::vector<MetaTensor>*',
            }
            wrapped_infermeta_name = get_wrapped_infermeta_name(api.api)
            args = []
            check_args = []
            for input_name in api.inputs['names']:
                args.append(tensor_type_map[api.inputs['input_info'][
                    input_name]] + ' ' + input_name)
                check_args.append(input_name)
            for attr_name in api.attrs['names']:
                args.append(api.attrs['attr_info'][attr_name][0] + ' ' +
                            attr_name)
                check_args.append(attr_name)
            for i, out_type in enumerate(api.outputs['types']):
                args.append(tensor_type_map[out_type] + ' ' + api.outputs[
                    'names'][i])

            if check_args == api.infer_meta['param']:
                return '', '', register_code

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
PT_REGISTER_INFER_META_FN({api.kernel['func'][0]}, pten::{get_wrapped_infermeta_name(api.kernel['func'][0])});"""

            return declare_code, defind_code, register_code
        else:
            return '', '', register_code
    else:
        return '', '', ''


def gene_infermeta_register(api):
    if api.is_base_api:
        if api.infer_meta['param'] is None:
            return f"""
PT_REGISTER_INFER_META_FN({api.kernel['func'][0]}, pten::{api.infer_meta['func']});"""

        else:
            return f"""
PT_REGISTER_INFER_META_FN({api.kernel['func'][0]}, pten::{get_wrapped_infermeta_name(api.kernel['func'][0])});"""

    else:
        return ''


def header_include():
    return """
#include "paddle/pten/core/meta_tensor.h"
#include "paddle/pten/common/scalar.h"
#include "paddle/pten/common/scalar_array.h"
"""


def source_include(header_file_path):
    return f"""
#include "{header_file_path}"
#include "paddle/pten/core/infermeta_utils.h"
#include "paddle/pten/infermeta/binary.h"
#include "paddle/pten/infermeta/multiary.h"
#include "paddle/pten/infermeta/nullary.h"
#include "paddle/pten/infermeta/unary.h"
"""


def api_namespace():
    return ("""
namespace pten {
""", """
}  // namespace pten
""")


def generate_wrapped_infermeta_and_register(api_yaml_path, header_file_path,
                                            source_file_path):

    with open(api_yaml_path, 'r') as f:
        apis = yaml.load(f, Loader=yaml.FullLoader)
    header_file = open(header_file_path, 'w')
    source_file = open(source_file_path, 'w')

    namespace = api_namespace()

    header_file.write("#pragma once\n")
    header_file.write(header_include())
    header_file.write(namespace[0])

    include_header_file = "paddle/pten/infermeta/generated.h"
    source_file.write(source_include(include_header_file))
    source_file.write(namespace[0])

    infermeta_register_code = ''

    for api in apis:
        api_item = BaseAPI(api)
        declare_code, defind_code, register_code = gene_wrapped_infermeta_and_register(
            api_item)
        header_file.write(declare_code)
        source_file.write(defind_code)
        infermeta_register_code = infermeta_register_code + register_code

    header_file.write(namespace[1])
    source_file.write(namespace[1])

    source_file.write(infermeta_register_code)

    header_file.close()
    source_file.close()


def main():
    parser = argparse.ArgumentParser(
        description='Generate PaddlePaddle C++ API files')
    parser.add_argument(
        '--api_yaml_path',
        help='path to api yaml file',
        default='python/paddle/utils/code_gen/api.yaml')
    parser.add_argument(
        '--wrapped_infermeta_header_path',
        help='output of generated wrapped_infermeta header code file',
        default='paddle/pten/infermeta/generated.h')

    parser.add_argument(
        '--wrapped_infermeta_source_path',
        help='output of generated wrapped_infermeta source code file',
        default='paddle/pten/infermeta/generated.cc')

    options = parser.parse_args()

    api_yaml_path = options.api_yaml_path
    header_file_path = options.wrapped_infermeta_header_path
    source_file_path = options.wrapped_infermeta_source_path

    generate_wrapped_infermeta_and_register(api_yaml_path, header_file_path,
                                            source_file_path)


if __name__ == '__main__':
    main()
