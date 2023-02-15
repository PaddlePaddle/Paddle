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


# prim api list
white_ops_list = [
    "pow",
    "scale",
    "multiply",
    "unsqueeze",
    "expand",
    "full",
    "reshape",
    "divide",
    "sum",
    "exp",
]

inplace_out_type_map = {
    "Tensor": "Tensor&",
    "std::vector<Tensor>": "std::vector<Tensor>&",
}

inplace_optional_out_type_map = {
    "Tensor": "paddle::optional<Tensor>&",
    "std::vector<Tensor>": "paddle::optional<std::vector<Tensor>>&",
}


class BaseAPI:
    def __init__(self, api_item_yaml, prims=tuple()):
        # self.api = api_item_yaml['op']
        self.api = api_item_yaml['name']

        self.is_prim_api = False
        if api_item_yaml['name'] in prims:
            self.is_prim_api = True

        #######################################
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
        ########################################
        if self.is_prim_api:
            (
                self.inputs,
                self.attrs,
                self.outputs,
                self.optional_vars,
            ) = self.parse_args(self.api, api_item_yaml)

            self.inplace_map = api_item_yaml['inplace']

    def get_api_func_name(self):
        return self.api

    # def is_inplace(self):
    #     if self.inplace_map
    #         return True
    #     return False

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

    def get_declare_args_nodefault(self, inplace_flag=False):
        declare_args = self.get_input_tensor_args(inplace_flag)
        for name in self.attrs['names']:
            declare_args.append(self.attrs['attr_info'][name][0] + ' ' + name)

        return ", ".join(declare_args)

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

    def parse_args(self, api_name, api_item_yaml):
        optional_vars = []
        for input_dict in api_item_yaml['inputs']:
            if input_dict['optional']:
                optional_vars.append(input_dict['name'])

        inputs, attrs = self.parse_input_and_attr(
            api_item_yaml['inputs'], api_item_yaml['attrs']
        )

        output_type_list, output_names, out_size_expr = self.parse_output(
            api_item_yaml['outputs']
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

    def parse_input_and_attr(self, inputs_list, attrs_list):
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

        inputs = {'names': [], 'input_info': {}}
        for input_dict in inputs_list:
            inputs['names'].append(input_dict['name'])
            if input_dict['optional']:
                inputs['input_info'][input_dict['name']] = optional_types_trans[
                    input_dict['typename']
                ]
            else:
                inputs['input_info'][input_dict['name']] = input_types_map[
                    input_dict['typename']
                ]

        attrs = {'names': [], 'attr_info': {}}
        for attr_dict in attrs_list:
            attrs['names'].append(attr_dict['name'])
            if 'default_value' in attr_dict.keys():
                default_value = attr_dict['default_value']
            else:
                default_value = None

            if 'optional' in attr_dict.keys():
                attrs['attr_info'][attr_dict['name']] = (
                    optional_types_trans[attr_dict['typename']],
                    default_value,
                )
            else:
                attrs['attr_info'][attr_dict['name']] = (
                    attr_types_map[attr_dict['typename']],
                    default_value,
                )
        return inputs, attrs

    def parse_output(self, outputs_list):

        out_type_list = []
        out_name_list = []
        out_size_expr_list = []
        for output_dict in outputs_list:
            if output_dict['intermediate']:
                continue
            out_type_list.append(output_dict['typename'])
            out_name_list.append(output_dict['name'])
            if 'size' in output_dict.keys():
                out_size_expr_list.append(output_dict['size'])
            else:
                out_size_expr_list.append(None)
        return out_type_list, out_name_list, out_size_expr_list


class EagerPrimAPI(BaseAPI):
    def __init__(self, api_item_yaml, prims=tuple()):
        super().__init__(api_item_yaml, prims)

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
        # print("ad_func_call_str: ", ad_func_call_str)
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
{self.get_return_type()} {api_func_name}{template}({self.get_declare_args_nodefault()})
"""
        else:
            api_code = f"""
template <>
{self.get_return_type(inplace_flag=True)} {api_func_name}{template}({self.get_declare_args_nodefault(inplace_flag=True)})
"""
        # func code

        api_code = api_code + '{'
        api_code += f"""{self.gene_ad_func_call()}"""
        api_code += '}' + '\n'

        return api_code
