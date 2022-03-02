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

import re

PREFIX_TENSOR_NAME = 'input_'
PREFIX_META_TENSOR_NAME = 'meta_'


class BaseAPI(object):
    def __init__(self, api_item_yaml):
        self.api = self.get_api_name(api_item_yaml)

        # inputs:
        #     names : [], list of input names
        #     input_info : {input_name : type}
        # attrs:
        #     names : [], list of attribute names
        #     attr_info : { attr_name : (type, default_values)}
        # outputs:
        #     names : [], list of output names
        #     types : [], list of output types
        #     return_type : Tensor, vector<Tensor>, ..., the return type of api
        # args_str:
        #     args_declare : "str" // str of function params with default value. Example: (..., bool flag=false)
        #     args_define : "str" // str of function params without default value. Example: (..., bool flag)
        self.inputs, self.attrs, self.outputs, self.args_str, self.optional_vars = self.parse_args(
            self.api, api_item_yaml)

        self.is_base_api = True
        if 'invoke' in api_item_yaml:
            self.is_base_api = False
            self.invoke = api_item_yaml['invoke']
        else:
            self.infer_meta = self.parse_infer_meta(api_item_yaml['infer_meta'])
            self.kernel = self.parse_kernel(api_item_yaml['kernel'])
            self.support_selected_rows_kernel = False if len(self.kernel[
                'func']) == 1 else True
            self.data_transform = self.parse_data_transform(api_item_yaml)
            self.inplace_map = self.parse_inplace(api_item_yaml)

    def get_api_name(self, api_item_yaml):
        return api_item_yaml['api']

    def get_api_func_name(self):
        return self.api

    def parse_args(self, api_name, api_item_yaml):
        optional_vars = []
        if 'optional' in api_item_yaml:
            optional_vars = [
                item.strip() for item in api_item_yaml['optional'].split(',')
            ]
        inputs, attrs, args_str = self.parse_input_and_attr(
            api_name, api_item_yaml['args'], optional_vars)
        output_type_list, output_names, return_type = self.parse_output(
            api_name, api_item_yaml['output'])
        return inputs, attrs, {
            'names': output_names,
            'types': output_type_list,
            'return_type': return_type
        }, args_str, optional_vars

    def parse_input_and_attr(self, api_name, args_config, optional_vars=[]):
        inputs = {'names': [], 'input_info': {}}
        attrs = {'names': [], 'attr_info': {}}
        args_str = args_config.strip()
        assert args_str.startswith('(') and args_str.endswith(')'), \
            f"Args declaration should start with '(' and end with ')', please check the args of {api_name} in yaml."
        args_str = args_str[1:-1]
        args_list = args_str.split(',')
        input_types_map = {
            'Tensor': 'const Tensor&',
            'Tensor[]': 'const std::vector<Tensor>&'
        }
        attr_types_map = {
            'ScalarArray': 'const ScalarArray&',
            'Scalar': 'const Scalar&',
            'int': 'int',
            'int32_t': 'int32_t',
            'int64_t': 'int64_t',
            'long': 'long',
            'size_t': 'size_t',
            'float': 'float',
            'double': 'double',
            'bool': 'bool',
            'Backend': 'Backend',
            'DataLayout': 'DataLayout',
            'DataType': 'DataType',
            'int64_t[]': 'const std::vector<int64_t>&',
            'int[]': 'const std::vector<int>&',
            'long[]': 'const std::vector<int64_t>&'
        }
        optional_types_trans = {
            'Tensor': 'const paddle::optional<Tensor>&',
            'Tensor[]': 'const paddle::optional<std::vector<Tensor>>&',
            'ScalarArray': 'const paddle::optional<ScalarArray>&',
            'Scalar': 'const paddle::optional<Scalar>&',
            'int': 'paddle::optional<int>',
            'int32_t': 'paddle::optional<int32_t>',
            'int64_t': 'paddle::optional<int64_t>',
            'size_t': 'paddle::optional<size_t>',
            'float': 'paddle::optional<float>',
            'double': 'paddle::optional<double>',
            'bool': 'paddle::optional<bool>',
            'Backend': 'paddle::optional<Backend>',
            'DataLayout': 'paddle::optional<DataLayout>',
            'DataType': 'paddle::optional<DataType>',
            'int64_t[]': 'paddle::optional<std::vector<int64_t>>',
            'int[]': 'paddle::optional<std::vector<int>>'
        }

        args_declare_str = ""
        args_define_str = ""

        for item in args_list:
            item = item.strip()
            type_and_name = item.split(' ')
            # match the input tensor
            has_input = False
            for in_type_symbol, in_type in input_types_map.items():
                if type_and_name[0] == in_type_symbol:
                    input_name = type_and_name[1].strip()
                    assert len(input_name) > 0, \
                        f"The input tensor name should not be empty. Please check the args of {api_name} in yaml."
                    assert len(attrs['names']) == 0, \
                        f"The input Tensor should appear before attributes. please check the position of {api_name}:input({input_name}) in yaml"

                    if input_name in optional_vars:
                        in_type = optional_types_trans[in_type_symbol]

                    inputs['names'].append(input_name)
                    inputs['input_info'][input_name] = in_type
                    args_declare_str = args_declare_str + in_type + ' ' + input_name + ', '
                    args_define_str = args_define_str + in_type + ' ' + input_name + ', '
                    has_input = True
                    break
            if has_input:
                continue

            # match the attribute
            for attr_type_symbol, attr_type in attr_types_map.items():
                if type_and_name[0] == attr_type_symbol:
                    attr_name = item[len(attr_type_symbol):].strip()
                    assert len(attr_name) > 0, \
                        f"The attribute name should not be empty. Please check the args of {api_name} in yaml."
                    default_value = None
                    if '=' in attr_name:
                        attr_infos = attr_name.split('=')
                        attr_name = attr_infos[0].strip()
                        default_value = attr_infos[1].strip()

                    if attr_name in optional_vars:
                        attr_type = optional_types_trans[attr_type_symbol]

                    default_value_str = "" if default_value is None else '=' + default_value
                    args_declare_str = args_declare_str + attr_type + ' ' + attr_name + default_value_str + ', '
                    args_define_str = args_define_str + attr_type + ' ' + attr_name + ', '
                    attrs['names'].append(attr_name)
                    attrs['attr_info'][attr_name] = (attr_type, default_value)
                    break

        return inputs, attrs, {
            'args_declare': args_declare_str[:-2],
            'args_define': args_define_str[:-2]
        }

    def parse_output(self, api_name, output_config):
        def parse_output_item(output_item):
            output_type_map = {
                'Tensor': 'Tensor',
                'Tensor[]': 'std::vector<Tensor>'
            }
            if re.search(r'\(\w*\)', output_item):
                result = re.search(
                    r"(?P<out_type>[a-zA-Z0-9_[\]]+)\s*\((?P<name>\w+)\)",
                    output_item)
                out_type = result.group('out_type')
                assert out_type in output_type_map, \
                    f"{api_name} : Output type error: the output type only support Tensor and Tensor[], \
                      but now is {out_type}."

                return out_type, result.group('name')

            else:
                if output_item.strip() in output_type_map:
                    return output_type_map[output_item.strip()], 'out'
                else:
                    raise ValueError(
                        "{} : Output type error: the output type only support Tensor and Tensor[], \
                      but now is {}.".format(api_name, output_item.strip()))

        temp_list = output_config.split(',')

        if len(temp_list) == 1:
            out_type, out_name = parse_output_item(temp_list[0])
            return [out_type], [out_name], self.get_return_type([out_type])
        else:
            out_type_list = []
            out_name_list = []
            for output_item in temp_list:
                out_type, out_name = parse_output_item(output_item)
                out_type_list.append(out_type)
                out_name_list.append(out_name)

            return out_type_list, out_name_list, self.get_return_type(
                out_type_list)

    def parse_infer_meta(self, infer_meta_config):
        infer_meta = infer_meta_config
        if 'param' not in infer_meta_config:
            infer_meta['param'] = None

        return infer_meta

    def parse_kernel(self, kernel_config):
        # kernel :
        #    func : [], Kernel functions (example: scale, scale_sr)
        #    param : [], Input params of kernel
        #    backend : str, the names of param to choose the kernel backend, default is None
        #    layout : str, the names of param to choose the kernel layout, default is None
        #    data_type : str, the names of param to choose the kernel data_type, default is None
        kernel = {
            'func': [],
            'param': None,
            'backend': None,
            'layout': None,
            'data_type': None
        }
        if 'backend' in kernel_config and len(kernel_config['backend']) > 0:
            kernel['backend'] = kernel_config['backend']
        if 'layout' in kernel_config and len(kernel_config['layout']) > 0:
            kernel['layout'] = kernel_config['layout']
        if 'data_type' in kernel_config and len(kernel_config['data_type']) > 0:
            kernel['data_type'] = kernel_config['data_type']
        if 'param' in kernel_config:
            kernel['param'] = kernel_config['param']
        kernel['func'] = [
            kernel_fn.strip() for kernel_fn in kernel_config['func'].split(',')
        ]

        if len(kernel['func']) == 2:
            assert kernel['func'][0] == self.api, \
                    f"{self.api} : Kernel func error: If kernel has two func config, the name of first func should be same with api name({self.api}), \
                      but now is {kernel['func'][0]}."
            assert kernel['func'][1].endswith('_sr'), \
                    f"{self.api} : Kernel func error: If kernel has two func config, the name of second func should be a selected_rows kernel (the func name endwith '_sr'), \
                      but now is {kernel['func'][1]}."

        return kernel

    def parse_data_transform(self, api_item_yaml):
        data_transform = {'skip_transform': [], 'support_trans_dtype': []}
        if 'data_transform' in api_item_yaml:
            if 'skip_transform' in api_item_yaml['data_transform']:
                data_transform['skip_transform'] = api_item_yaml[
                    'data_transform']['skip_transform']
            if 'support_trans_dtype' in api_item_yaml['data_transform']:
                data_transform['support_trans_dtype'] = api_item_yaml[
                    'data_transform']['support_trans_dtype']

        return data_transform

    def parse_inplace(self, api_item_yaml):
        if 'inplace' in api_item_yaml:
            inplace_map = {}
            inplace_list = api_item_yaml['inplace'].split(',')
            for item in inplace_list:
                result = re.search(r"(?P<in>\w+)\s*->\s(?P<out>\w+)", item)
                in_val = result.group('in')
                out_val = result.group('out')
                assert in_val in self.inputs['names'], \
                    f"{self.api} : Inplace input error: the input var name('{in_val}') is not found in the input args of {self.api}."
                assert out_val in self.outputs['names'], \
                    f"{self.api} : Inplace output error: the output var name('{out_val}') is not found in the output args of {self.api}."

                inplace_map[out_val] = in_val

            return inplace_map
        else:
            return None

    # Override by child class
    def get_return_type(self, out_type_list):
        return None

    def gene_api_declaration(self):
        api_declaration = f"""
PADDLE_API {self.outputs['return_type']} {self.get_api_func_name()}({self.args_str['args_declare']});
"""

        if self.is_base_api and self.inplace_map is not None:
            api_declaration = api_declaration + f"""
PADDLE_API {self.outputs['return_type']} {self.get_api_func_name() + '_'}({self.args_str['args_declare']});
"""

        return api_declaration

    def gene_kernel_select(self) -> str:
        api = self.api
        input_names = self.inputs['names']
        attrs = self.attrs
        kernel = self.kernel

        kernel_key_item_init = """
  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;
"""
        # Check the tensor options
        attr_backend_count = 0
        attr_layout_count = 0
        attr_data_type_count = 0
        for attr_name in attrs['names']:
            if attrs['attr_info'][attr_name][0] == 'Backend':
                assert kernel['backend'] is not None, \
                    f"{api} api: When there is a parameter with 'Backend' type in attributes, you must set backend of kernel manually."
                attr_backend_count = attr_backend_count + 1
            if attrs['attr_info'][attr_name][0] == 'DataLayout':
                assert kernel['layout'] is not None, \
                    f"{api} api: When there is a parameter with 'DataLayout' type in attributes, you must set layout of kernel manually."
                attr_layout_count = attr_layout_count + 1
            if attrs['attr_info'][attr_name][0] == 'DataType':
                assert kernel['data_type'] is not None, \
                    f"{api} api: When there is a parameter with 'DataType' type in attributes, you must set data_type of kernel manually."
                attr_data_type_count = attr_data_type_count + 1

        # preprocess kernel configures
        kernel_select_code = ""
        if kernel['backend'] is not None:
            if '>' in kernel['backend']:
                vars_list = kernel['backend'].split('>')
                assert len(
                    vars_list
                ) == 2, f"{api} api: The number of params to set backend with '>' only allows 2, but received {len(vars_list)}."
                assert (vars_list[0].strip() in attrs['names']) and (attrs['attr_info'][vars_list[0].strip()][0] == 'Backend'), \
                    f"{api} api: When use '>' to set kernel backend, the first param should be a attribute with Backend type."
                kernel_select_code = kernel_select_code + f"""
  kernel_backend = ParseBackendWithInputOrder({vars_list[0].strip()}, {vars_list[1].strip()});
"""

            else:
                args_str = ""
                for ele in kernel['backend'].split(','):
                    args_str = args_str + ele.strip() + ', '
                kernel_select_code = kernel_select_code + f"""
  kernel_backend = ParseBackend({args_str[:-2]});
"""

        if kernel['layout'] is not None:
            if '>' in kernel['layout']:
                vars_list = kernel['layout'].split('>')
                assert len(
                    vars_list
                ) == 2, f"{api} api: The number of params to set layout with '>' only allows 2, but received {len(vars_list)}."
                assert vars_list[0].strip() in attrs['names'] and attrs['attr_info'][vars_list[0].strip()][0] == 'DataLayout', \
                    f"{api} api: When use '>' to set kernel layout, the first param should be a attribute with DataLayout type."
                kernel_select_code = kernel_select_code + f"""
  kernel_layout = ParseLayoutWithInputOrder({vars_list[0].strip()}, {vars_list[1].strip()});
"""

            else:
                vars_list = kernel['layout'].split(',')
                assert len(
                    vars_list
                ) == 1, f"{api} api: The number of params to set layout must be 1, but received {len(vars_list)}."
                kernel_select_code = kernel_select_code + f"""
  kernel_layout = ParseLayout({vars_list[0].strip()});
"""

        if kernel['data_type'] is not None:
            if '>' in kernel['data_type']:
                vars_list = kernel['data_type'].split('>')
                assert len(
                    vars_list
                ) == 2, f"{api} api: The number of params to set data_type with '>' only allows 2, but received {len(vars_list)}."
                assert vars_list[0].strip() in attrs['names'] and attrs['attr_info'][vars_list[0].strip()][0] == 'DataType', \
                    f"{api} api: When use '>' to set kernel data_type, the first param should be a attribute with DataType type."
                kernel_select_code = kernel_select_code + f"""
  kernel_data_type = ParseDataTypeWithInputOrder({vars_list[0].strip()}, {vars_list[1].strip()});
"""

            else:
                vars_list = kernel['data_type'].split(',')
                assert len(
                    vars_list
                ) == 1, f"{api} api: The number of params to set data_type only allows 2, but received {len(vars_list)}."
                kernel_select_code = kernel_select_code + f"""
  kernel_data_type = ParseDataType({vars_list[0].strip()});
"""

        if len(input_names) == 0:
            assert attr_backend_count > 0 and attr_data_type_count > 0, \
                f"{api} api: When there is no input tensor, the args must have 'Backend' and 'DataType'."

        kernel_select_args = ""
        for input_name in input_names:
            kernel_select_args = kernel_select_args + input_name + ", "

        if len(kernel_select_args) > 2:
            kernel_select_args = kernel_select_args[:-2]

        kernel_select_code = kernel_key_item_init + kernel_select_code

        if len(input_names) > 0:
            if self.support_selected_rows_kernel:
                kernel_select_code = kernel_select_code + f"""
  KernelType kernel_type = ParseKernelTypeByInputArgs({", ".join(input_names)});
"""

            kernel_select_code = kernel_select_code + f"""
  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {{
    auto kernel_key_set = ParseKernelKeyByInputArgs({kernel_select_args});
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
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

        return kernel_select_code

    def gene_infer_meta(self, kernel_output_names, code_indent) -> str:
        input_names = self.inputs['names']
        attr_names = self.attrs['names']
        infer_meta = self.infer_meta

        infer_meta_params = infer_meta[
            'param'] + kernel_output_names if infer_meta[
                'param'] is not None else input_names + attr_names + kernel_output_names
        # generate meta tensors
        meta_tensor_code = ""
        param_code = ""
        for param in infer_meta_params:
            if param in input_names:
                if self.inputs['input_info'][param] == "const Tensor&":
                    param_code = param_code + "MakeMetaTensor(*" + PREFIX_TENSOR_NAME + param + "), "
                elif self.inputs['input_info'][
                        param] == "const std::vector<Tensor>&":
                    meta_tensor_code = meta_tensor_code + f"""
{code_indent}  auto {param}_meta_vec = MakeMetaTensor(*{PREFIX_TENSOR_NAME}{param});
{code_indent}  std::vector<phi::MetaTensor*> {param}_metas({param}_meta_vec.size());
{code_indent}  for (size_t i = 0; i < {param}_meta_vec.size(); ++i) {{
{code_indent}    {param}_metas[i] = &{param}_meta_vec[i];
{code_indent}  }}
"""

                    param_code = param_code + param + "_metas, "
                elif param in self.optional_vars:
                    meta_tensor_code = meta_tensor_code + f"""
{code_indent}  paddle::optional<const phi::MetaTensor&> {PREFIX_TENSOR_NAME}meta_ref_{param}(paddle::none);
{code_indent}  auto {PREFIX_TENSOR_NAME}meta_{param} = MakeMetaTensor({PREFIX_TENSOR_NAME}{param});
{code_indent}  if ({PREFIX_TENSOR_NAME}meta_{param}) {{
{code_indent}    {PREFIX_TENSOR_NAME}meta_ref_{param} = paddle::make_optional<const phi::MetaTensor&>(*{PREFIX_TENSOR_NAME}meta_{param});
{code_indent}  }}"""

                    param_code = param_code + f"{PREFIX_TENSOR_NAME}meta_ref_{param}, "
                else:
                    raise ValueError(
                        f"{self.api} : Param of infer_meta error : {self.inputs['input_info'][param]} type is not supported."
                    )
            elif param in kernel_output_names:
                meta_tensor_code = meta_tensor_code + code_indent + "  phi::MetaTensor " + param.replace(
                    'kernel_', PREFIX_META_TENSOR_NAME) + "(" + param + ");\n"
                param_code = param_code + "&" + param.replace(
                    'kernel_', PREFIX_META_TENSOR_NAME) + ", "
            elif param in attr_names:
                param_code = param_code + param + ", "
            elif isinstance(param, str):
                param_code = param_code + "\"" + param + "\", "
            elif isinstance(param, bool):
                param_code = param_code + str(param).lower() + ", "
            else:
                param_code = param_code + str(param) + ", "

        param_code = param_code[:-2]
        return f"""{meta_tensor_code}
{code_indent}  phi::{infer_meta['func']}({param_code});
"""

    def get_kernel_args(self, code_indent):
        input_trans_map = {
            'const Tensor&': 'const phi::DenseTensor&',
            'const Tensor &': 'const phi::DenseTensor&',
            'const std::vector<Tensor>&':
            'const std::vector<phi::DenseTensor>&',
            'const std::vector<Tensor> &':
            'const std::vector<phi::DenseTensor>&',
            'const paddle::optional<Tensor>&':
            'paddle::optional<const phi::DenseTensor&>',
            'const paddle::optional<std::vector<Tensor>>&':
            'paddle::optional<const std::vector<phi::DenseTensor>&>'
        }
        out_trans_map = {
            'Tensor': 'phi::DenseTensor*',
            'std::vector<Tensor>': 'std::vector<phi::DenseTensor*>&'
        }
        input_names = self.inputs['names']
        input_infos = self.inputs['input_info']
        kernel_args_type_list = ['const platform::DeviceContext&']

        attr_names = self.attrs['names']
        kernel_param = self.kernel['param']
        if kernel_param is None:
            kernel_param = input_names + attr_names

        input_tensor_code = ""
        for i, input_name in enumerate(input_names):
            # set input code
            if input_name in kernel_param:
                trans_flag = "{}"
                if input_name in self.data_transform['skip_transform']:
                    trans_flag = "{true}"
                elif input_name in self.data_transform['support_trans_dtype']:
                    trans_flag = "{false, true}"
                if input_name in self.optional_vars:
                    input_tensor_code = input_tensor_code + f"""
{code_indent}  {input_trans_map[input_infos[input_name]]} {PREFIX_TENSOR_NAME}{input_name}(paddle::none);
{code_indent}  auto {PREFIX_TENSOR_NAME}{input_name}_ptr = PrepareData({input_name}, kernel.InputAt({i}), {trans_flag});
{code_indent}  if ({PREFIX_TENSOR_NAME}{input_name}_ptr) {{
{code_indent}    {PREFIX_TENSOR_NAME}{input_name} = paddle::make_optional<const phi::DenseTensor&>(*{PREFIX_TENSOR_NAME}{input_name}_ptr);
{code_indent}  }}"""

                else:
                    input_tensor_code = input_tensor_code + f"""
{code_indent}  auto {PREFIX_TENSOR_NAME}{input_name} = PrepareData({input_name}, kernel.InputAt({i}), {trans_flag});"""

            else:
                if input_name in self.optional_vars:
                    input_tensor_code = input_tensor_code + f"""
{code_indent}  {input_trans_map[input_infos[input_name]]} {PREFIX_TENSOR_NAME}{input_name}(paddle::none);
{code_indent}  auto {PREFIX_TENSOR_NAME}{input_name}_ptr = TensorToDenseTensor({input_name});
{code_indent}  if ({PREFIX_TENSOR_NAME}{input_name}_ptr) {{
{code_indent}    {PREFIX_TENSOR_NAME}{input_name} = paddle::make_optional<const phi::DenseTensor&>(*{PREFIX_TENSOR_NAME}{input_name}_ptr);
{code_indent}  }}"""

                else:
                    input_tensor_code = input_tensor_code + f"""
{code_indent}  auto {PREFIX_TENSOR_NAME}{input_name} = TensorToDenseTensor({input_name});"""

        kernel_args = "*dev_ctx, "
        for param in kernel_param:
            if param in input_names:
                if param in self.optional_vars:
                    kernel_args = kernel_args + PREFIX_TENSOR_NAME + param + ", "
                else:
                    kernel_args = kernel_args + "*" + PREFIX_TENSOR_NAME + param + ", "
                kernel_in_type = input_trans_map[input_infos[param]]
                kernel_args_type_list.append(kernel_in_type)
            elif param in attr_names:
                # set attr for kernel_context
                if 'ScalarArray' in self.attrs['attr_info'][param][0]:
                    kernel_args_type_list.append('const phi::ScalarArray&')
                    param = 'phi::ScalarArray(' + param + ')'
                elif 'Scalar' in self.attrs['attr_info'][param][0]:
                    kernel_args_type_list.append('const phi::Scalar&')
                    param = 'phi::Scalar(' + param + ')'
                else:
                    kernel_args_type_list.append(self.attrs['attr_info'][param][
                        0])
                kernel_args = kernel_args + param + ", "
            elif isinstance(param, bool):
                kernel_args = kernel_args + str(param).lower() + ", "
            else:
                kernel_args = kernel_args + str(param) + ", "

        for out_type in self.outputs['types']:
            kernel_args_type_list.append(out_trans_map[out_type])

        kernel_signature = "void(*)(" + ", ".join(kernel_args_type_list) + ")"

        return input_tensor_code, kernel_args[:-2], kernel_signature

    def get_selected_rows_kernel_args(self, code_indent):
        input_trans_map = {
            'const Tensor&': 'const phi::SelectedRows&',
            'const Tensor &': 'const phi::SelectedRows&',
            'const paddle::optional<Tensor>&':
            'paddle::optional<const phi::SelectedRows&>'
        }
        out_trans_map = {'Tensor': 'phi::SelectedRows*'}
        input_names = self.inputs['names']
        input_infos = self.inputs['input_info']
        kernel_args_type_list = ['const platform::DeviceContext&']

        attr_names = self.attrs['names']
        kernel_param = self.kernel['param']
        if kernel_param is None:
            kernel_param = input_names + attr_names

        input_tensor_code = ""
        for i, input_name in enumerate(input_names):
            # set input code
            if input_name in self.optional_vars:
                input_tensor_code = input_tensor_code + f"""

{code_indent}  {input_trans_map[input_infos[input_name]]} {PREFIX_TENSOR_NAME}{input_name}(paddle::none);
{code_indent}  auto {PREFIX_TENSOR_NAME}{input_name}_ptr = TensorToSelectedRows({input_name});
{code_indent}  if ({PREFIX_TENSOR_NAME}{input_name}_ptr) {{
{code_indent}    {PREFIX_TENSOR_NAME}{input_name} = paddle::make_optional<const phi::SelectedRows&>(*{PREFIX_TENSOR_NAME}{input_name}_ptr);
{code_indent}  }}"""

            else:
                input_tensor_code = input_tensor_code + f"""
{code_indent}  auto {PREFIX_TENSOR_NAME}{input_name} = TensorToSelectedRows({input_name});"""

        kernel_args = "*dev_ctx, "
        for param in kernel_param:
            if param in input_names:
                if param in self.optional_vars:
                    kernel_args = kernel_args + PREFIX_TENSOR_NAME + param + ", "
                else:
                    kernel_args = kernel_args + "*" + PREFIX_TENSOR_NAME + param + ", "
                kernel_in_type = input_trans_map[input_infos[param]]
                kernel_args_type_list.append(kernel_in_type)
            elif param in attr_names:
                # set attr for kernel_context
                if 'ScalarArray' in self.attrs['attr_info'][param][0]:
                    kernel_args_type_list.append('const phi::ScalarArray&')
                    param = 'phi::ScalarArray(' + param + ')'
                elif 'Scalar' in self.attrs['attr_info'][param][0]:
                    kernel_args_type_list.append('const phi::Scalar&')
                    param = 'phi::Scalar(' + param + ')'
                else:
                    kernel_args_type_list.append(self.attrs['attr_info'][param][
                        0])
                kernel_args = kernel_args + param + ", "
            elif isinstance(param, bool):
                kernel_args = kernel_args + str(param).lower() + ", "
            else:
                kernel_args = kernel_args + str(param) + ", "

        for out_type in self.outputs['types']:
            kernel_args_type_list.append(out_trans_map[out_type])

        kernel_signature = "void(*)(" + ", ".join(kernel_args_type_list) + ")"

        return input_tensor_code, kernel_args[:-2], kernel_signature

    # Override by child class
    def gene_output(self,
                    output_type_list,
                    set_out_func,
                    code_indent,
                    inplace_flag=False):
        return None, None, None

    def gen_dense_tensor_kernel_code(self, code_indent, inplace_flag=False):
        input_tensors, kernel_args, kernel_signature = self.get_kernel_args(
            code_indent)
        outputs_args, kernel_output_names, output_create = self.gene_output(
            self.outputs['types'], 'SetKernelOutput', code_indent, inplace_flag)
        return f"""
{code_indent}  auto kernel = phi::KernelFactory::Instance().SelectKernelOrThrowError(
{code_indent}      "{self.kernel['func'][0]}", {{kernel_backend, kernel_layout, kernel_data_type}});
{code_indent}  VLOG(6) << "{self.api} API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
{code_indent}  VLOG(6) << "{self.api} API kernel: " << kernel;

{code_indent}  auto* dev_ctx = GetDeviceContextByBackend(kernel_backend);
{input_tensors}
{output_create}
{self.gene_infer_meta(kernel_output_names, code_indent)}

{code_indent}  using kernel_signature = {kernel_signature};
{code_indent}  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
{code_indent}  (*kernel_fn)({kernel_args}, {outputs_args});

{code_indent}  return out;"""

    def gen_selected_rows_kernel_code(self, code_indent, inplace_flag=False):
        input_tensors, kernel_args, kernel_signature = self.get_selected_rows_kernel_args(
            code_indent)
        outputs_args, kernel_output_names, output_create = self.gene_output(
            self.outputs['types'], 'SetSelectedRowsKernelOutput', code_indent,
            inplace_flag)
        return f"""
{code_indent}  auto kernel = phi::KernelFactory::Instance().SelectKernelOrThrowError(
{code_indent}      "{self.kernel['func'][1]}", {{kernel_backend, kernel_layout, kernel_data_type}});
{code_indent}  VLOG(6) << "{self.api} API SelectedRows kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
{code_indent}  VLOG(6) << "{self.api} API SelectedRows kernel: " << kernel;

{code_indent}  auto* dev_ctx = GetDeviceContextByBackend(kernel_backend);
{input_tensors}
{output_create}
{self.gene_infer_meta(kernel_output_names, code_indent)}

{code_indent}  using kernel_signature = {kernel_signature};
{code_indent}  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
{code_indent}  (*kernel_fn)({kernel_args}, {outputs_args});

{code_indent}  return out;"""

    def gene_base_api_code(self, inplace_flag=False):
        api_func_name = self.get_api_func_name() + ('_' if inplace_flag else '')
        api_code = f"""
PADDLE_API {self.outputs['return_type']} {api_func_name}({self.args_str["args_define"]}) {{
{self.gene_kernel_select()}
"""

        if self.support_selected_rows_kernel:
            code_indent = '  '
            return api_code + f"""
  if(kernel_type == KernelType::DENSE_TENSOR_KENREL){{
{self.gen_dense_tensor_kernel_code(code_indent, inplace_flag)}
  }} else {{
{self.gen_selected_rows_kernel_code(code_indent, inplace_flag)}
  }}
}}
"""

        else:
            code_indent = ''
            return api_code + self.gen_dense_tensor_kernel_code(
                code_indent, inplace_flag) + """
}
"""

    def gene_api_code(self):
        if self.is_base_api:
            api_code = self.gene_base_api_code()
            if self.inplace_map is not None:
                api_code = api_code + self.gene_base_api_code(inplace_flag=True)
            return api_code

        else:
            inveke_func_name = self.invoke.split('(')[0].strip()
            if inveke_func_name in self.attrs['names']:
                # Adjust the param whose name is same with api invoked.
                pattern = r'\W' + inveke_func_name + '[^A-Za-z0-9_(]'

                def adjust_name(matched):
                    matched_str = matched.group()
                    return matched_str[0:-1] + '_val' + matched_str[-1]

                invoke_code = re.sub(pattern, adjust_name, self.invoke)
                params_code = re.sub(pattern, adjust_name,
                                     self.args_str["args_define"])
            else:
                invoke_code = self.invoke
                params_code = self.args_str["args_define"]
            return f"""
{self.outputs['return_type']} {self.api}({params_code}) {{
  return {invoke_code};
}}
"""
