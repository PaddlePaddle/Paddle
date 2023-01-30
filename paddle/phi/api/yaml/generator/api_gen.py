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

<<<<<<< HEAD
import argparse
import re

import yaml
from api_base import PREFIX_TENSOR_NAME, BaseAPI

inplace_out_type_map = {
    "Tensor": "Tensor&",
    "std::vector<Tensor>": "std::vector<Tensor>&",
=======
import os
import yaml
import argparse
import re

from api_base import BaseAPI, PREFIX_TENSOR_NAME

inplace_out_type_map = {
    "Tensor": "Tensor&",
    "std::vector<Tensor>": "std::vector<Tensor>&"
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
}

inplace_optional_out_type_map = {
    "Tensor": "paddle::optional<Tensor>&",
<<<<<<< HEAD
    "std::vector<Tensor>": "paddle::optional<std::vector<Tensor>>&",
=======
    "std::vector<Tensor>": "paddle::optional<std::vector<Tensor>>&"
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
}


class ForwardAPI(BaseAPI):
<<<<<<< HEAD
    def __init__(self, api_item_yaml):
        super().__init__(api_item_yaml)
        self.is_dygraph_api, self.intermediate_outs = self.parse_intermediate(
            api_item_yaml
        )
        self.inplace_map, self.view_map = self.parse_inplace_and_view(
            api_item_yaml
        )
=======

    def __init__(self, api_item_yaml):
        super(ForwardAPI, self).__init__(api_item_yaml)
        self.is_dygraph_api, self.intermediate_outs = self.parse_intermediate(
            api_item_yaml)
        self.inplace_map, self.view_map = self.parse_inplace_and_view(
            api_item_yaml)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    def get_api_func_name(self):
        if self.is_dygraph_api:
            return self.api + '_intermediate'
        else:
            return self.api

<<<<<<< HEAD
    def gene_input(self, kernel_tensor_type=None, code_indent=''):
        kernel_param = self.kernel['param']
        input_name_tensor_map, input_tensor_code = super().gene_input(
            kernel_tensor_type, code_indent
        )

        # generate the input that is in view list
        for i, input_name in enumerate(self.inputs['names']):
            if (
                input_name in self.view_map.values()
                and input_name not in input_name_tensor_map.keys()
            ):
                if (
                    kernel_tensor_type is None
                    or kernel_tensor_type[0][kernel_param.index(input_name)]
                    == 'dense'
                ):
                    trans_flag = self.gene_trans_flag(input_name)
                    input_tensor_code = (
                        input_tensor_code
                        + f"""
{code_indent}  auto {PREFIX_TENSOR_NAME}{input_name} = PrepareData({input_name}, kernel.InputAt(0), {trans_flag});"""
                    )
                else:
                    # do nothing
                    pass

        return input_name_tensor_map, input_tensor_code

=======
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def parse_intermediate(self, api_item_yaml):
        if 'intermediate' in api_item_yaml:
            intermediate_outs = [
                item.strip()
                for item in api_item_yaml['intermediate'].split(',')
            ]
            return True, intermediate_outs
        else:
            return False, []

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
<<<<<<< HEAD
                    assert (
                        in_val in self.inputs['names']
                    ), f"{self.api} : {mode} input error: the input var name('{in_val}') is not found in the input args of {self.api}."
                    assert (
                        out_val in self.outputs['names']
                    ), f"{self.api} : {mode} output error: the output var name('{out_val}') is not found in the output args of {self.api}."
=======
                    assert in_val in self.inputs['names'], \
                        f"{self.api} : {mode} input error: the input var name('{in_val}') is not found in the input args of {self.api}."
                    assert out_val in self.outputs['names'], \
                        f"{self.api} : {mode} output error: the output var name('{out_val}') is not found in the output args of {self.api}."
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

                    if mode == 'inplace':
                        inplace_map[out_val] = in_val
                    else:
                        view_map[out_val] = in_val

        return inplace_map, view_map

    def get_return_type_with_intermediate(self, inplace_flag=False):
        out_type_list = []
        for i, out_type in enumerate(self.outputs['types']):
            out_name = self.outputs['names'][i].split('@')[0]
            if inplace_flag and out_name in self.inplace_map:
                if self.inplace_map[out_name] in self.optional_vars:
                    out_type_list.append(
<<<<<<< HEAD
                        inplace_optional_out_type_map[out_type]
                    )
=======
                        inplace_optional_out_type_map[out_type])
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
                else:
                    out_type_list.append(inplace_out_type_map[out_type])
            else:
                out_type_list.append(out_type)

        if len(out_type_list) == 1:
            return out_type_list[0]
        else:
            return "std::tuple<" + ", ".join(out_type_list) + ">"

    def get_return_type(self, inplace_flag=False):
        out_type_list = []
        for i, out_type in enumerate(self.outputs['types']):
            out_name = self.outputs['names'][i].split('@')[0]
            if inplace_flag and out_name in self.inplace_map:
                if self.inplace_map[out_name] in self.optional_vars:
                    out_type_list.append(
<<<<<<< HEAD
                        inplace_optional_out_type_map[out_type]
                    )
=======
                        inplace_optional_out_type_map[out_type])
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
                else:
                    out_type_list.append(inplace_out_type_map[out_type])
            elif self.is_dygraph_api or out_name not in self.intermediate_outs:
                out_type_list.append(out_type)

        if len(out_type_list) == 1:
            return out_type_list[0]
        else:
            return "std::tuple<" + ", ".join(out_type_list) + ">"

    def gene_return_code(self):
        if self.is_dygraph_api or len(self.intermediate_outs) == 0:
            return "return api_output;"
        else:
            return_out_list = []
            for i, name in enumerate(self.outputs['names']):
                if name.split('@')[0] not in self.intermediate_outs:
                    return_out_list.append(i)
            if len(return_out_list) == 1:
                return f"return std::get<{return_out_list[0]}>(api_output);"
            else:
                selected_code = [
                    f"std::get<{i}>(api_output)" for i in return_out_list
                ]
            return 'return std::make_tuple(' + ", ".join(selected_code) + ');'

<<<<<<< HEAD
    def gene_output(
        self,
        out_dtype_list,
        out_tensor_type_list=None,
        code_indent='',
        inplace_flag=False,
    ):
=======
    def gene_output(self,
                    out_dtype_list,
                    out_tensor_type_list=None,
                    code_indent='',
                    inplace_flag=False):
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        kernel_output = []
        output_names = []
        output_create = ""
        return_type = self.get_return_type_with_intermediate(inplace_flag)

        if len(out_dtype_list) == 1:
            kernel_output.append('kernel_out')
            output_names.append('kernel_out')
<<<<<<< HEAD
            inplace_assign = (
                " = " + self.inplace_map[self.outputs['names'][0]]
                if inplace_flag and self.outputs['names'][0] in self.inplace_map
                else ""
            )
            output_create = f"""
{code_indent}  {return_type} api_output{inplace_assign};"""
            set_out_func = (
                'SetKernelOutput'
                if out_tensor_type_list is None
                or out_tensor_type_list[0] == 'dense'
                else 'SetSelectedRowsKernelOutput'
            )
            if return_type == 'std::vector<Tensor>':
                assert (
                    self.outputs['out_size_expr'][0] is not None
                ), f"{self.api}: The out size expr : '{{expr}}' should be set when output has Tensor[]. You can refer 'split' api."
                output_create = (
                    output_create
                    + f"""
{code_indent}  auto kernel_out = {set_out_func}({self.outputs['out_size_expr'][0]}, &api_output);"""
                )

            else:
                output_create = (
                    output_create
                    + f"""
{code_indent}  auto kernel_out = {set_out_func}(&api_output);"""
                )

            if (
                not inplace_flag
                and self.view_map is not None
                and self.outputs['names'][0] in self.view_map
            ):
                output_create = (
                    output_create
                    + f"""
{code_indent}  kernel_out->ShareBufferWith(*{PREFIX_TENSOR_NAME}{self.view_map[self.outputs['names'][0]]});
{code_indent}  kernel_out->ShareInplaceVersionCounterWith(*{PREFIX_TENSOR_NAME}{self.view_map[self.outputs['names'][0]]});
{code_indent}  VLOG(3) << "Perform View between Output and Input Tensor, share allocation and inplace version.";"""
                )
=======
            inplace_assign = " = " + self.inplace_map[
                self.outputs['names'][0]] if inplace_flag and self.outputs[
                    'names'][0] in self.inplace_map else ""
            output_create = f"""
{code_indent}  {return_type} api_output{inplace_assign};"""
            set_out_func = 'SetKernelOutput' if out_tensor_type_list is None or out_tensor_type_list[
                0] == 'dense' else 'SetSelectedRowsKernelOutput'
            if return_type == 'std::vector<Tensor>':
                assert self.outputs['out_size_expr'][0] is not None, \
                     f"{self.api}: The out size expr : '{{expr}}' should be set when output has Tensor[]. You can refer 'split' api."
                output_create = output_create + f"""
{code_indent}  auto kernel_out = {set_out_func}({self.outputs['out_size_expr'][0]}, &api_output);"""

            else:
                output_create = output_create + f"""
{code_indent}  auto kernel_out = {set_out_func}(&api_output);"""

            if not inplace_flag and self.view_map is not None and self.outputs[
                    'names'][0] in self.view_map:
                output_create = output_create + f"""
{code_indent}  kernel_out->ShareBufferWith(*{PREFIX_TENSOR_NAME}{self.view_map[self.outputs['names'][0]]});
{code_indent}  kernel_out->ShareInplaceVersionCounterWith(*{PREFIX_TENSOR_NAME}{self.view_map[self.outputs['names'][0]]});
{code_indent}  VLOG(3) << "Perform View between Output and Input Tensor, share allocation and inplace version.";"""
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

        elif len(out_dtype_list) > 1:
            output_create = f"""
{code_indent}  {return_type} api_output;"""

            if inplace_flag:
                output_create = f"""
{code_indent}  {return_type} api_output{{"""

                for out_name in self.outputs['names']:
                    if out_name in self.inplace_map:
                        output_create += self.inplace_map[out_name] + ', '
                    else:
                        output_create += 'Tensor(), '
                output_create = output_create[:-2] + '};'

            for i in range(len(out_dtype_list)):
                kernel_output.append(f'kernel_out_{i}')
                output_names.append(f'kernel_out_{i}')
<<<<<<< HEAD
                set_out_func = (
                    'SetKernelOutput'
                    if out_tensor_type_list is None
                    or out_tensor_type_list[i] == 'dense'
                    else 'SetSelectedRowsKernelOutput'
                )

                get_out_code = f"&std::get<{i}>(api_output)"
                if (
                    self.outputs['names'][i] in self.inplace_map
                    and self.inplace_map[self.outputs['names'][i]]
                    in self.optional_vars
                ):
                    get_out_code = f"std::get<{i}>(api_output).get_ptr()"

                if out_dtype_list[i] == 'std::vector<Tensor>':
                    assert (
                        self.outputs['out_size_expr'][i] is not None
                    ), f"{self.api}: The out size expr : '{{expr}}' should be set when output has Tensor[]. You can refer 'split' api."
                    # Special case for inplace vector and inplace optional<vector>
                    if self.outputs['names'][i] in self.inplace_map:
                        set_out_func = "SetInplaceVectorKernelOutput"
                        if (
                            self.inplace_map[self.outputs['names'][i]]
                            in self.optional_vars
                        ):
                            set_out_func = (
                                "SetInplaceOptionalVectorKernelOutput"
                            )
                            get_out_code = f"std::get<{i}>(api_output)"
                    output_create = (
                        output_create
                        + f"""
{code_indent}  auto kernel_out_{i} = {set_out_func}({self.outputs['out_size_expr'][i]}, {get_out_code});"""
                    )

                else:
                    output_create = (
                        output_create
                        + f"""
{code_indent}  auto kernel_out_{i} = {set_out_func}({get_out_code});"""
                    )

                if (
                    not inplace_flag
                    and self.view_map is not None
                    and self.outputs['names'][i] in self.view_map
                ):
                    if out_dtype_list[i] == 'Tensor':
                        output_create = (
                            output_create
                            + f"""
    {code_indent}  kernel_out_{i}->ShareBufferWith(*{PREFIX_TENSOR_NAME}{self.view_map[self.outputs['names'][i]]});
    {code_indent}  kernel_out_{i}->ShareInplaceVersionCounterWith(*{PREFIX_TENSOR_NAME}{self.view_map[self.outputs['names'][i]]});
    {code_indent}  VLOG(3) << "Perform View between Output and Input Tensor, share allocation and inplace version.";"""
                        )
                    else:
                        raise ValueError(
                            "{} : Output error: only support Tensor type when use view in yaml. But get {}".format(
                                self.api, out_dtype_list[i]
                            )
                        )
        else:
            raise ValueError(
                "{} : Output error: the output should not be empty.".format(
                    self.api
                )
            )
=======
                set_out_func = 'SetKernelOutput' if out_tensor_type_list is None or out_tensor_type_list[
                    i] == 'dense' else 'SetSelectedRowsKernelOutput'

                get_out_code = f"&std::get<{i}>(api_output)"
                if self.outputs['names'][
                        i] in self.inplace_map and self.inplace_map[
                            self.outputs['names'][i]] in self.optional_vars:
                    get_out_code = f"std::get<{i}>(api_output).get_ptr()"

                if out_dtype_list[i] == 'std::vector<Tensor>':
                    assert self.outputs['out_size_expr'][i] is not None, \
                        f"{self.api}: The out size expr : '{{expr}}' should be set when output has Tensor[]. You can refer 'split' api."
                    # Special case for inplace vector and inplace optional<vector>
                    if self.outputs['names'][i] in self.inplace_map:
                        set_out_func = "SetInplaceVectorKernelOutput"
                        if self.inplace_map[self.outputs['names']
                                            [i]] in self.optional_vars:
                            set_out_func = "SetInplaceOptionalVectorKernelOutput"
                            get_out_code = f"std::get<{i}>(api_output)"
                    output_create = output_create + f"""
{code_indent}  auto kernel_out_{i} = {set_out_func}({self.outputs['out_size_expr'][i]}, {get_out_code});"""

                else:
                    output_create = output_create + f"""
{code_indent}  auto kernel_out_{i} = {set_out_func}({get_out_code});"""

                if not inplace_flag and self.view_map is not None and self.outputs[
                        'names'][i] in self.view_map:
                    output_create = output_create + f"""
{code_indent}  kernel_out_{i}->ShareBufferWith(*{PREFIX_TENSOR_NAME}{self.view_map[self.outputs['names'][i]]});
{code_indent}  kernel_out_{i}->ShareInplaceVersionCounterWith(*{PREFIX_TENSOR_NAME}{self.view_map[self.outputs['names'][i]]});
{code_indent}  VLOG(3) << "Perform View between Output and Input Tensor, share allocation and inplace version.";"""

        else:
            raise ValueError(
                "{} : Output error: the output should not be empty.".format(
                    self.api))
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

        return kernel_output, output_names, output_create


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
#include <memory>

#include "glog/logging.h"

#include "paddle/phi/api/lib/api_custom_impl.h"
#include "paddle/phi/api/lib/api_gen_utils.h"
#include "paddle/phi/api/lib/data_transform.h"
#include "paddle/phi/api/lib/kernel_dispatch.h"
<<<<<<< HEAD
#include "paddle/phi/common/type_traits.h"
=======
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/infermeta/binary.h"
#include "paddle/phi/infermeta/multiary.h"
#include "paddle/phi/infermeta/nullary.h"
#include "paddle/phi/infermeta/unary.h"
#include "paddle/phi/infermeta/ternary.h"

#include "paddle/fluid/platform/profiler/event_tracing.h"
#include "paddle/fluid/platform/profiler/supplement_tracing.h"

DECLARE_bool(conv2d_disable_cudnn);
<<<<<<< HEAD
DECLARE_int32(low_precision_op_list);
=======
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
"""


def api_namespace():
<<<<<<< HEAD
    return (
        """
namespace paddle {
namespace experimental {

""",
        """

}  // namespace experimental
}  // namespace paddle
""",
    )
=======
    return ("""
namespace paddle {
namespace experimental {

""", """

}  // namespace experimental
}  // namespace paddle
""")
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81


def generate_api(api_yaml_path, header_file_path, source_file_path):
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

    include_header_file = "paddle/phi/api/include/api.h"
    source_file.write(source_include(include_header_file))
    source_file.write(namespace[0])

    for api in apis:
        foward_api = ForwardAPI(api)
        if foward_api.is_dygraph_api:
            foward_api.is_dygraph_api = False

        header_file.write(foward_api.gene_api_declaration())
        source_file.write(foward_api.gene_api_code())

    header_file.write(namespace[1])
    source_file.write(namespace[1])

    header_file.close()
    source_file.close()


def main():
    parser = argparse.ArgumentParser(
<<<<<<< HEAD
        description='Generate PaddlePaddle C++ API files'
    )
    parser.add_argument(
        '--api_yaml_path',
        help='path to api yaml file',
        nargs='+',
        default=['paddle/phi/api/yaml/ops.yaml'],
    )

    parser.add_argument(
        '--api_header_path',
        help='output of generated api header code file',
        default='paddle/phi/api/include/api.h',
    )

    parser.add_argument(
        '--api_source_path',
        help='output of generated api source code file',
        default='paddle/phi/api/lib/api.cc',
    )
=======
        description='Generate PaddlePaddle C++ API files')
    parser.add_argument('--api_yaml_path',
                        help='path to api yaml file',
                        nargs='+',
                        default='paddle/phi/api/yaml/ops.yaml')

    parser.add_argument('--api_header_path',
                        help='output of generated api header code file',
                        default='paddle/phi/api/include/api.h')

    parser.add_argument('--api_source_path',
                        help='output of generated api source code file',
                        default='paddle/phi/api/lib/api.cc')
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    options = parser.parse_args()

    api_yaml_path = options.api_yaml_path
    header_file_path = options.api_header_path
    source_file_path = options.api_source_path

    generate_api(api_yaml_path, header_file_path, source_file_path)


if __name__ == '__main__':
    main()
