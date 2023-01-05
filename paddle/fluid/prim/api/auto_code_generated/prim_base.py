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

import collections
import re

PREFIX_TENSOR_NAME = 'input_'
PREFIX_META_TENSOR_NAME = 'meta_'


class BaseAPI:
    def __init__(self, api_item_yaml):
        self.api = api_item_yaml['op']

        # inputs:
        #     names : [], list of input names
        #     input_info : {input_name : type}
        # attrs:
        #     names : [], list of attribute names
        #     attr_info : { attr_name : (type, default_values)}
        # outputs:
        #     names : [], list of output names
        #     types : [], list of output types
        #     out_size_expr : [], expression for getting size of vector<Tensor>
        (
            self.inputs,
            self.attrs,
            self.outputs,
            self.optional_vars,
        ) = self.parse_args(self.api, api_item_yaml)

        # self.is_base_api = True
        # if 'invoke' in api_item_yaml:
        #     self.is_base_api = False
        #     self.invoke = api_item_yaml['invoke']
        # else:
        #     if 'infer_meta' in api_item_yaml:
        #         self.infer_meta = self.parse_infer_meta(
        #             api_item_yaml['infer_meta']
        #         )
        #     self.kernel = self.parse_kernel(api_item_yaml['kernel'])
        #     self.data_transform = self.parse_data_transform(api_item_yaml)
        #     self.inplace_map, self.view_map = {}, {}

        # self.gene_input_func = {
        #     "const Tensor&": {
        #         "dense": self.gene_dense_input,
        #         "selected_rows": self.gene_selected_rows_input,
        #     },
        #     "const paddle::optional<Tensor>&": {
        #         "dense": self.gene_dense_input,
        #         "selected_rows": self.gene_selected_rows_input,
        #     },
        #     "const std::vector<Tensor>&": {"dense": self.gene_vec_dense_input},
        #     "const paddle::optional<std::vector<Tensor>>&": {
        #         "dense": self.gene_optional_vec_dense_input
        #     },
        # }


    def get_api_func_name(self):
        return self.api

    def get_input_tensor_args(self, inplace_flag=False):
        input_args = []
        inplace_type_map = {
            "const Tensor&": "Tensor&",
            "const paddle::optional<Tensor>&": "paddle::optional<Tensor>&",
            "const std::vector<Tensor>&": "std::vector<Tensor>&",
            "const paddle::optional<std::vector<Tensor>>&": "paddle::optional<std::vector<Tensor>>&",
        }
        for name in self.inputs['names']:
            name = name.split('@')[0]
            if inplace_flag and name in self.inplace_map.values():
                input_args.append(
                    inplace_type_map[self.inputs['input_info'][name]]
                    + ' '
                    + name
                )
            else:
                input_args.append(self.inputs['input_info'][name] + ' ' + name)
        return input_args

    def get_declare_args(self, inplace_flag=False):
        declare_args = self.get_input_tensor_args(inplace_flag)
        for name in self.attrs['names']:
            default_value = ''
            if self.attrs['attr_info'][name][1] is not None:
                default_value = ' = ' + self.attrs['attr_info'][name][1]
            declare_args.append(
                self.attrs['attr_info'][name][0] + ' ' + name + default_value
            )

        return ", ".join(declare_args)


    def parse_args(self, api_name, api_item_yaml):
        optional_vars = []
        if 'optional' in api_item_yaml:
            optional_vars = [
                item.strip() for item in api_item_yaml['optional'].split(',')
            ]
        inputs, attrs = self.parse_input_and_attr(
            api_name, api_item_yaml['args'], optional_vars
        )
        output_type_list, output_names, out_size_expr = self.parse_output(
            api_name, api_item_yaml['output']
        )
        return (
            inputs,
            attrs,
            {
                'names': output_names,
                'types': output_type_list,
                'out_size_expr': out_size_expr,
            },
            optional_vars,
        )

    def parse_input_and_attr(self, api_name, args_config, optional_vars=[]):
        inputs = {'names': [], 'input_info': {}}
        attrs = {'names': [], 'attr_info': {}}
        args_str = args_config.strip()
        assert args_str.startswith('(') and args_str.endswith(
            ')'
        ), f"Args declaration should start with '(' and end with ')', please check the args of {api_name} in yaml."
        args_str = args_str[1:-1]
        args_list = args_str.split(',')
        input_types_map = {
            'Tensor': 'const Tensor&',
            'Tensor[]': 'const std::vector<Tensor>&',
        }
        attr_types_map = {
            'IntArray': 'const IntArray&',
            'Scalar': 'const Scalar&',
            'Scalar(int)': 'const Scalar&',
            'Scalar(int64_t)': 'const Scalar&',
            'Scalar(float)': 'const Scalar&',
            'Scalar(dobule)': 'const Scalar&',
            'Scalar[]': 'const std::vector<phi::Scalar>&',
            'int': 'int',
            'int32_t': 'int32_t',
            'int64_t': 'int64_t',
            'long': 'long',
            'size_t': 'size_t',
            'float': 'float',
            'float[]': 'const std::vector<float>&',
            'double': 'double',
            'bool': 'bool',
            'bool[]': 'const std::vector<bool>&',
            'str': 'const std::string&',
            'str[]': 'const std::vector<std::string>&',
            'Place': 'const Place&',
            'DataLayout': 'DataLayout',
            'DataType': 'DataType',
            'int64_t[]': 'const std::vector<int64_t>&',
            'int[]': 'const std::vector<int>&',
        }
        optional_types_trans = {
            'Tensor': 'const paddle::optional<Tensor>&',
            'Tensor[]': 'const paddle::optional<std::vector<Tensor>>&',
            'int': 'paddle::optional<int>',
            'int32_t': 'paddle::optional<int32_t>',
            'int64_t': 'paddle::optional<int64_t>',
            'float': 'paddle::optional<float>',
            'double': 'paddle::optional<double>',
            'bool': 'paddle::optional<bool>',
            'Place': 'paddle::optional<const Place&>',
            'DataLayout': 'paddle::optional<DataLayout>',
            'DataType': 'paddle::optional<DataType>',
        }

        for item in args_list:
            item = item.strip()
            type_and_name = item.split(' ')
            # match the input tensor
            has_input = False
            for in_type_symbol, in_type in input_types_map.items():
                if type_and_name[0] == in_type_symbol:
                    input_name = type_and_name[1].strip()
                    assert (
                        len(input_name) > 0
                    ), f"The input tensor name should not be empty. Please check the args of {api_name} in yaml."
                    assert (
                        len(attrs['names']) == 0
                    ), f"The input Tensor should appear before attributes. please check the position of {api_name}:input({input_name}) in yaml"

                    if input_name in optional_vars:
                        in_type = optional_types_trans[in_type_symbol]

                    inputs['names'].append(input_name)
                    inputs['input_info'][input_name] = in_type
                    has_input = True
                    break
            if has_input:
                continue

            # match the attribute
            for attr_type_symbol, attr_type in attr_types_map.items():
                if type_and_name[0] == attr_type_symbol:
                    attr_name = item[len(attr_type_symbol) :].strip()
                    assert (
                        len(attr_name) > 0
                    ), f"The attribute name should not be empty. Please check the args of {api_name} in yaml."
                    default_value = None
                    if '=' in attr_name:
                        attr_infos = attr_name.split('=')
                        attr_name = attr_infos[0].strip()
                        default_value = attr_infos[1].strip()

                    if attr_name in optional_vars:
                        attr_type = optional_types_trans[attr_type_symbol]

                    default_value_str = (
                        "" if default_value is None else '=' + default_value
                    )
                    attrs['names'].append(attr_name)
                    attrs['attr_info'][attr_name] = (attr_type, default_value)
                    break

        return inputs, attrs

    def parse_output(self, api_name, output_config):
        def parse_output_item(output_item):
            output_type_map = {
                'Tensor': 'Tensor',
                'Tensor[]': 'std::vector<Tensor>',
            }
            result = re.search(
                r"(?P<out_type>[a-zA-Z0-9_[\]]+)\s*(?P<name>\([a-zA-Z0-9_@]+\))?\s*(?P<expr>\{[^\}]+\})?",
                output_item,
            )
            assert (
                result is not None
            ), f"{api_name} : the output config parse error."
            out_type = result.group('out_type')
            assert (
                out_type in output_type_map
            ), f"{api_name} : Output type error: the output type only support Tensor and Tensor[], \
                  but now is {out_type}."

            out_name = (
                'out'
                if result.group('name') is None
                else result.group('name')[1:-1]
            )
            out_size_expr = (
                None
                if result.group('expr') is None
                else result.group('expr')[1:-1]
            )
            return output_type_map[out_type], out_name, out_size_expr

        temp_list = output_config.split(',')

        if len(temp_list) == 1:
            out_type, out_name, size_expr = parse_output_item(temp_list[0])
            return [out_type], [out_name], [size_expr]
        else:
            out_type_list = []
            out_name_list = []
            out_size_expr_list = []
            for output_item in temp_list:
                out_type, out_name, size_expr = parse_output_item(output_item)
                out_type_list.append(out_type)
                out_name_list.append(out_name)
                out_size_expr_list.append(size_expr)

            return out_type_list, out_name_list, out_size_expr_list

    