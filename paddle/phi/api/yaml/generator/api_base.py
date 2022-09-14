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
import collections

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
        #     out_size_expr : [], expression for getting size of vector<Tensor>
        self.inputs, self.attrs, self.outputs, self.optional_vars = self.parse_args(
            self.api, api_item_yaml)

        self.is_base_api = True
        if 'invoke' in api_item_yaml:
            self.is_base_api = False
            self.invoke = api_item_yaml['invoke']
        else:
            if 'infer_meta' in api_item_yaml:
                self.infer_meta = self.parse_infer_meta(
                    api_item_yaml['infer_meta'])
            self.kernel = self.parse_kernel(api_item_yaml['kernel'])
            self.data_transform = self.parse_data_transform(api_item_yaml)
            self.inplace_map, self.view_map = {}, {}

    def get_api_name(self, api_item_yaml):
        return api_item_yaml['op']

    def get_api_func_name(self):
        return self.api

    def get_input_tensor_args(self, inplace_flag=False):
        input_args = []
        inplace_type_map = {
            "const Tensor&":
            "Tensor&",
            "const paddle::optional<Tensor>&":
            "paddle::optional<Tensor>&",
            "const std::vector<Tensor>&":
            "std::vector<Tensor>&",
            "const paddle::optional<std::vector<Tensor>>&":
            "paddle::optional<std::vector<Tensor>>&"
        }
        for name in self.inputs['names']:
            name = name.split('@')[0]
            if inplace_flag and name in self.inplace_map.values():
                input_args.append(
                    inplace_type_map[self.inputs['input_info'][name]] + ' ' +
                    name)
            else:
                input_args.append(self.inputs['input_info'][name] + ' ' + name)
        return input_args

    def get_declare_args(self, inplace_flag=False):
        declare_args = self.get_input_tensor_args(inplace_flag)
        for name in self.attrs['names']:
            default_value = ''
            if self.attrs['attr_info'][name][1] is not None:
                default_value = ' = ' + self.attrs['attr_info'][name][1]
            declare_args.append(self.attrs['attr_info'][name][0] + ' ' + name +
                                default_value)

        return ", ".join(declare_args)

    def get_define_args(self, inplace_flag=False):
        define_args = self.get_input_tensor_args(inplace_flag)
        for name in self.attrs['names']:
            define_args.append(self.attrs['attr_info'][name][0] + ' ' + name)

        return ", ".join(define_args)

    def parse_args(self, api_name, api_item_yaml):
        optional_vars = []
        if 'optional' in api_item_yaml:
            optional_vars = [
                item.strip() for item in api_item_yaml['optional'].split(',')
            ]
        inputs, attrs = self.parse_input_and_attr(api_name,
                                                  api_item_yaml['args'],
                                                  optional_vars)
        output_type_list, output_names, out_size_expr = self.parse_output(
            api_name, api_item_yaml['output'])
        return inputs, attrs, {
            'names': output_names,
            'types': output_type_list,
            'out_size_expr': out_size_expr
        }, optional_vars

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
            'DataType': 'paddle::optional<DataType>'
        }

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
                    attrs['names'].append(attr_name)
                    attrs['attr_info'][attr_name] = (attr_type, default_value)
                    break

        return inputs, attrs

    def parse_output(self, api_name, output_config):

        def parse_output_item(output_item):
            output_type_map = {
                'Tensor': 'Tensor',
                'Tensor[]': 'std::vector<Tensor>'
            }
            result = re.search(
                r"(?P<out_type>[a-zA-Z0-9_[\]]+)\s*(?P<name>\([a-zA-Z0-9_@]+\))?\s*(?P<expr>\{[^\}]+\})?",
                output_item)
            assert result is not None, f"{api_name} : the output config parse error."
            out_type = result.group('out_type')
            assert out_type in output_type_map, \
                f"{api_name} : Output type error: the output type only support Tensor and Tensor[], \
                  but now is {out_type}."

            out_name = 'out' if result.group('name') is None else result.group(
                'name')[1:-1]
            out_size_expr = None if result.group(
                'expr') is None else result.group('expr')[1:-1]
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
        #    dispatch : {}, the key is kernel_func, the value is type of inputs and outputs for kernel (example: {kernel_name : (['dense','sparse_coo']#input,['sparse_coo']#output)})
        kernel = {
            'func': [],
            'param': None,
            'backend': None,
            'layout': None,
            'data_type': None,
            'use_gpudnn': 'false',
            'dispatch': {}
        }
        if 'backend' in kernel_config and len(kernel_config['backend']) > 0:
            kernel['backend'] = kernel_config['backend']
        if 'layout' in kernel_config and len(kernel_config['layout']) > 0:
            kernel['layout'] = kernel_config['layout']
        if 'data_type' in kernel_config and len(kernel_config['data_type']) > 0:
            kernel['data_type'] = kernel_config['data_type']
        if 'param' in kernel_config:
            kernel['param'] = kernel_config['param']
        if 'use_gpudnn' in kernel_config:
            kernel['use_gpudnn'] = kernel_config['use_gpudnn']
            if isinstance(kernel['use_gpudnn'], bool):
                kernel['use_gpudnn'] = str(kernel['use_gpudnn']).lower()
        kernel_funcs = re.compile(r'([a-zA-Z0-9_]+)\s*({[^}]+})?').findall(
            kernel_config['func'])

        def parse_kernel_in_out_type(in_out_str):
            if len(in_out_str) == 0:
                return None
            tmp_in_out_list = in_out_str[1:-1].split('->')
            inputs = [item.strip() for item in tmp_in_out_list[0].split(',')]
            outputs = [item.strip() for item in tmp_in_out_list[1].split(',')]

            # check the tensor type
            for item in inputs:
                assert item in [
                    'dense', 'selected_rows', 'sparse_coo', 'sparse_csr'
                ], f"{self.api} : Invalid input tensor type ('{item}'), here we only support 'dense', 'selected_rows', 'sparse_coo' and 'sparse_csr'."
            for item in outputs:
                assert item in [
                    'dense', 'selected_rows', 'sparse_coo', 'sparse_csr'
                ], f"{self.api} : Invalid output tensor type ('{item}'), here we only support 'dense', 'selected_rows', 'sparse_coo' and 'sparse_csr'."

            return (inputs, outputs)

        for func_item in kernel_funcs:
            kernel['func'].append(func_item[0])
            kernel['dispatch'][func_item[0]] = parse_kernel_in_out_type(
                func_item[1])

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

    # Override by child class
    def get_return_type(self, inplace_flag=False):
        return None

    def gene_api_declaration(self):
        api_declaration = ""
        api_func_name = self.get_api_func_name()
        if api_func_name[-1] != '_':
            api_declaration = f"""
PADDLE_API {self.get_return_type()} {api_func_name}({self.get_declare_args()});
"""

        if self.is_base_api and len(self.inplace_map) > 0:
            if api_func_name[-1] != '_':
                api_func_name += '_'
            api_declaration = api_declaration + f"""
PADDLE_API {self.get_return_type(inplace_flag=True)} {api_func_name}({self.get_declare_args(inplace_flag=True)});
"""

        return api_declaration

    # Backward API Override this method
    def gene_kernel_backend_select(self):
        backend_select_code = ""
        if self.kernel['backend'] is not None:
            if '>' in self.kernel['backend']:
                vars_list = self.kernel['backend'].split('>')
                assert len(
                    vars_list
                ) == 2, f"{self.api} api: The number of params to set backend with '>' only allows 2, but received {len(vars_list)}."
                assert (vars_list[0].strip() in self.attrs['names']) and (self.attrs['attr_info'][vars_list[0].strip()][0] == 'const Place&'), \
                    f"{self.api} api: When use '>' to set kernel backend, the first param should be a attribute with Place type."
                backend_select_code = f"""
  kernel_backend = ParseBackendWithInputOrder({vars_list[0].strip()}, {vars_list[1].strip()});
"""

            else:
                backend_args = [
                    ele.strip() for ele in self.kernel['backend'].split(',')
                ]
                backend_select_code = f"""
  kernel_backend = ParseBackend({", ".join(backend_args)});
"""

        return backend_select_code

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
            if attrs['attr_info'][attr_name][0] == 'const Place&':
                assert kernel['backend'] is not None, \
                    f"{api} api: When there is a parameter with 'Place' type in attributes, you must set backend of kernel manually."
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
        kernel_select_code = self.gene_kernel_backend_select()

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
                ) == 1, f"{api} api: The number of params to set data_type only allows 1, but received {len(vars_list)}."
                kernel_select_code = kernel_select_code + f"""
  kernel_data_type = ParseDataType({vars_list[0].strip()});
"""

        if len(input_names) == 0:
            assert attr_backend_count > 0 and attr_data_type_count > 0, \
                f"{api} api: When there is no input tensor, the args must have 'Place' and 'DataType'."

        kernel_select_args = ""
        for input_name in input_names:
            kernel_select_args = kernel_select_args + input_name + ", "

        if len(kernel_select_args) > 2:
            kernel_select_args = kernel_select_args[:-2]

        kernel_select_code = kernel_key_item_init + kernel_select_code

        if len(input_names) > 0:
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

        infer_meta_params = infer_meta['param'] if infer_meta[
            'param'] is not None else input_names + attr_names
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
{code_indent}  auto {param}_meta_vec = MakeMetaTensor({PREFIX_TENSOR_NAME}{param});
{code_indent}  std::vector<const phi::MetaTensor*> {param}_metas({param}_meta_vec.size());
{code_indent}  for (size_t i = 0; i < {param}_meta_vec.size(); ++i) {{
{code_indent}    {param}_metas[i] = &{param}_meta_vec[i];
{code_indent}  }}
"""
                    param_code = param_code + param + "_metas, "
                elif self.inputs['input_info'][
                        param] == "const paddle::optional<std::vector<Tensor>>&":
                    meta_tensor_code = meta_tensor_code + f"""
{code_indent}  auto {param}_meta_vec = MakeMetaTensor({PREFIX_TENSOR_NAME}{param});
{code_indent}  paddle::optional<std::vector<const phi::MetaTensor*>> {param}_metas({param}_meta_vec.size());
{code_indent}  for (size_t i = 0; i < {param}_meta_vec.size(); ++i) {{
{code_indent}    {param}_metas->at(i) = &{param}_meta_vec[i];
{code_indent}  }}
"""
                    param_code = param_code + param + "_metas, "
                elif param in self.optional_vars:
                    param_code = param_code + "MakeMetaTensor(" + PREFIX_TENSOR_NAME + param + "), "
                else:
                    raise ValueError(
                        f"{self.api} : Param of infer_meta error : {self.inputs['input_info'][param]} type is not supported."
                    )
            elif param in attr_names:
                param_code = param_code + param + ", "
            elif isinstance(param, str):
                param_code = param_code + "\"" + param + "\", "
            elif isinstance(param, bool):
                param_code = param_code + str(param).lower() + ", "
            else:
                param_code = param_code + str(param) + ", "

        for i, out_name in enumerate(kernel_output_names):
            if self.outputs['types'][i] == 'std::vector<Tensor>':
                meta_tensor_code = meta_tensor_code + f"""
{code_indent}  auto {out_name}_{PREFIX_META_TENSOR_NAME}vec = MakeMetaTensor({out_name});
{code_indent}  std::vector<phi::MetaTensor*> {out_name}_metas({out_name}_{PREFIX_META_TENSOR_NAME}vec.size());
{code_indent}  for (size_t i = 0; i < {out_name}_{PREFIX_META_TENSOR_NAME}vec.size(); ++i) {{
{code_indent}    {out_name}_metas[i] = {out_name}[i] ? &{out_name}_{PREFIX_META_TENSOR_NAME}vec[i] : nullptr;
{code_indent}  }}"""

                param_code = param_code + out_name + '_metas, '
            else:
                meta_tensor_code = meta_tensor_code + code_indent + "  phi::MetaTensor " + out_name.replace(
                    'kernel_',
                    PREFIX_META_TENSOR_NAME) + "(" + out_name + ");\n"
                if len(kernel_output_names) == 1:
                    param_code = param_code + f"&{out_name.replace('kernel_', PREFIX_META_TENSOR_NAME)}, "
                else:
                    param_code = param_code + f"{out_name} ? &{out_name.replace('kernel_', PREFIX_META_TENSOR_NAME)} : nullptr, "

        param_code = param_code[:-2]
        return f"""{meta_tensor_code}
{code_indent}  phi::{infer_meta['func']}({param_code});
"""

    def get_kernel_args(self, kernel_tensor_type=None, code_indent=''):
        dense_input_trans_map = {
            'const Tensor&':
            'const phi::DenseTensor&',
            'const std::vector<Tensor>&':
            'const std::vector<const phi::DenseTensor*>&',
            'const paddle::optional<Tensor&>':
            'paddle::optional<const phi::DenseTensor&>',
            'const paddle::optional<Tensor>&':
            'const paddle::optional<phi::DenseTensor>&',
            'const paddle::optional<std::vector<Tensor>>&':
            'const paddle::optional<std::vector<const phi::DenseTensor*>>&'
        }
        dense_out_trans_map = {
            'Tensor': 'phi::DenseTensor*',
            'std::vector<Tensor>': 'std::vector<phi::DenseTensor*>&'
        }
        sr_input_trans_map = {
            'const Tensor&':
            'const phi::SelectedRows&',
            'const paddle::optional<Tensor>&':
            'const paddle::optional<phi::SelectedRows>&'
        }
        sr_out_trans_map = {'Tensor': 'phi::SelectedRows*'}
        input_names = self.inputs['names']
        input_infos = self.inputs['input_info']
        kernel_args_type_list = ['const platform::DeviceContext&']

        attr_names = self.attrs['names']
        kernel_param = self.kernel['param']
        if kernel_param is None:
            kernel_param = input_names + attr_names

        input_tensor_code = ""
        input_name_tensor_map = collections.defaultdict(list)
        for i, input_name in enumerate(input_names):
            # set input code
            if input_name in kernel_param:
                # input is dense tensor
                if kernel_tensor_type is None or kernel_tensor_type[0][
                        kernel_param.index(input_name)] == 'dense':
                    trans_flag = "{}"
                    if input_name in self.data_transform['skip_transform']:
                        trans_flag = "{true}"
                    elif input_name in self.data_transform[
                            'support_trans_dtype']:
                        trans_flag = "{false, true}"
                    if input_name in self.optional_vars:
                        if self.inputs['input_info'][
                                input_name] == "const paddle::optional<std::vector<Tensor>>&":
                            if input_name in self.inplace_map.values():
                                input_name_tensor_map[input_name].append(
                                    (f"{PREFIX_TENSOR_NAME}{input_name}", True))
                                input_tensor_code = input_tensor_code + f"""
{code_indent}  paddle::optional<std::vector<const phi::DenseTensor*>> {PREFIX_TENSOR_NAME}{input_name} = TensorToConstDenseTensorPtr({input_name});"""
                            else:
                                input_name_tensor_map[input_name].append(
                                    (f"{PREFIX_TENSOR_NAME}{input_name}_vec",
                                     True))
                                input_tensor_code = input_tensor_code + f"""
{code_indent}  auto {PREFIX_TENSOR_NAME}{input_name}_vec = PrepareData({input_name}, kernel.InputAt({kernel_param.index(input_name)}), {trans_flag});
{code_indent}  paddle::optional<std::vector<const phi::DenseTensor*>> {PREFIX_TENSOR_NAME}{input_name};
{code_indent}  if ({PREFIX_TENSOR_NAME}{input_name}_vec){{
{code_indent}    {PREFIX_TENSOR_NAME}{input_name} = paddle::optional<std::vector<const phi::DenseTensor*>>({PREFIX_TENSOR_NAME}{input_name}_vec->size());
{code_indent}    for (size_t i = 0; i < {PREFIX_TENSOR_NAME}{input_name}_vec->size(); ++i) {{
{code_indent}      {PREFIX_TENSOR_NAME}{input_name}->at(i) = &{PREFIX_TENSOR_NAME}{input_name}_vec->at(i);
{code_indent}    }}
{code_indent}  }}"""
                        else:
                            input_name_tensor_map[input_name].append(
                                (f"{PREFIX_TENSOR_NAME}{input_name}", False))
                            input_tensor_code = input_tensor_code + f"""
{code_indent}  auto {PREFIX_TENSOR_NAME}{input_name} = PrepareData({input_name}, kernel.InputAt({kernel_param.index(input_name)}), {trans_flag});"""

                    else:
                        if self.inputs['input_info'][
                                input_name] == "const Tensor&":
                            input_name_tensor_map[input_name].append(
                                (f"{PREFIX_TENSOR_NAME}{input_name}", False))
                            input_tensor_code = input_tensor_code + f"""
{code_indent}  auto {PREFIX_TENSOR_NAME}{input_name} = PrepareData({input_name}, kernel.InputAt({kernel_param.index(input_name)}), {trans_flag});"""

                        elif self.inputs['input_info'][
                                input_name] == "const std::vector<Tensor>&":
                            if input_name in self.inplace_map.values():
                                input_name_tensor_map[input_name].append(
                                    (f"{PREFIX_TENSOR_NAME}{input_name}", True))
                                input_tensor_code = input_tensor_code + f"""
{code_indent}  std::vector<const phi::DenseTensor*> {PREFIX_TENSOR_NAME}{input_name} = TensorToConstDenseTensorPtr({input_name});"""
                            else:
                                input_name_tensor_map[input_name].append(
                                    (f"{PREFIX_TENSOR_NAME}{input_name}_vec",
                                     True))
                                input_tensor_code = input_tensor_code + f"""
{code_indent}  auto {PREFIX_TENSOR_NAME}{input_name}_vec = PrepareData({input_name}, kernel.InputAt({kernel_param.index(input_name)}), {trans_flag});
{code_indent}  std::vector<const phi::DenseTensor*> {PREFIX_TENSOR_NAME}{input_name}({PREFIX_TENSOR_NAME}{input_name}_vec->size());
{code_indent}  for (size_t i = 0; i < {PREFIX_TENSOR_NAME}{input_name}.size(); ++i) {{
{code_indent}    {PREFIX_TENSOR_NAME}{input_name}[i] = &{PREFIX_TENSOR_NAME}{input_name}_vec->at(i);
{code_indent}  }}"""

                        else:
                            # do nothing
                            pass
                else:  # input is selected_rows
                    input_name_tensor_map[input_name].append(
                        (f"{PREFIX_TENSOR_NAME}{input_name}", False))
                    input_tensor_code = input_tensor_code + f"""
{code_indent}  auto {PREFIX_TENSOR_NAME}{input_name} = TensorToSelectedRows({input_name});
"""
            else:
                if input_name in self.infer_meta['param']:
                    if input_name in self.optional_vars:
                        input_tensor_code = input_tensor_code + f"""
{code_indent}  paddle::optional<phi::TensorBase> {PREFIX_TENSOR_NAME}{input_name} = {input_name} ? paddle::optional<phi::TensorBase>(*{input_name}->impl()) : paddle::none;"""

                    else:
                        if self.inputs['input_info'][
                                input_name] == "const std::vector<Tensor>&":
                            input_tensor_code = input_tensor_code + f"""
{code_indent}  auto {PREFIX_TENSOR_NAME}{input_name}_uq_ptr = TensorToDenseTensor({input_name});
{code_indent}  const auto& {PREFIX_TENSOR_NAME}{input_name} = *{PREFIX_TENSOR_NAME}{input_name}_uq_ptr;"""
                        else:
                            input_tensor_code = input_tensor_code + f"""
{code_indent}  auto {PREFIX_TENSOR_NAME}{input_name} = {input_name}.impl();"""
        input_tensor_code = input_tensor_code + f"""
{code_indent}  if(platform::RecordOpInfoSupplement::IsEnabled()){{"""
        single_tensor_names = []
        list_tensor_names = []
        for input_name, input_tensors in input_name_tensor_map.items():
            has_vector_tensor = False
            for input_tensor, is_vector in input_tensors:
                if is_vector is True:
                    has_vector_tensor = True
            if has_vector_tensor is False:
                single_tensor_names.append(input_name)
            else:
                list_tensor_names.append(input_name)
        if not single_tensor_names:
            input_tensor_code = input_tensor_code + f"""
{code_indent}     std::vector<std::pair<const char*, std::vector<phi::DDim>>> input_shapes;"""
        else:
            for input_name in single_tensor_names:
                if input_name in self.optional_vars:
                    input_tensors = input_name_tensor_map[input_name]
                    input_tensor_code = input_tensor_code + f"""
{code_indent}     std::vector<phi::DDim> {input_name}_record_shapes;"""
                    for input_tensor, _ in input_tensors:
                        input_tensor_code = input_tensor_code + f"""
{code_indent}     if({input_tensor}){{
{code_indent}       {input_name}_record_shapes.push_back((*{input_tensor}).dims());
{code_indent}     }}"""

            input_tensor_code = input_tensor_code + f"""
{code_indent}     std::vector<std::pair<const char*, std::vector<phi::DDim>>> input_shapes{{"""
            for input_name in single_tensor_names[:-1]:
                if input_name in self.optional_vars:
                    input_tensor_code = input_tensor_code + f"""            
{code_indent}     {{"{input_name}", {input_name}_record_shapes}},"""
                else:
                    input_tensor_code = input_tensor_code + f"""            
{code_indent}     {{"{input_name}", {{"""
                    input_tensors = input_name_tensor_map[input_name]
                    for input_tensor, _ in input_tensors[:-1]:
                        input_tensor_code = input_tensor_code + f"""            
{code_indent}     (*{input_tensor}).dims(),"""
                    input_tensor_code = input_tensor_code + f"""            
{code_indent}     (*{input_tensors[-1][0]}).dims()}}}},"""
            if single_tensor_names[-1] in self.optional_vars:
                input_tensor_code = input_tensor_code + f"""            
{code_indent}     {{"{single_tensor_names[-1]}",         
{code_indent}     {single_tensor_names[-1]}_record_shapes}}}};"""
            else:
                input_tensor_code = input_tensor_code + f"""            
{code_indent}     {{"{single_tensor_names[-1]}", {{"""
                input_tensors = input_name_tensor_map[single_tensor_names[-1]]
                for input_tensor, _ in input_tensors[:-1]:
                    input_tensor_code = input_tensor_code + f"""            
{code_indent}     (*{input_tensor}).dims(),"""
                input_tensor_code = input_tensor_code + f"""            
{code_indent}     (*{input_tensors[-1][0]}).dims()}}}}}};"""
        if list_tensor_names:
            input_tensor_code = input_tensor_code + f"""
{code_indent}     std::vector<phi::DDim> ddims_vec;"""
        for input_name in list_tensor_names:
            input_tensor_code = input_tensor_code + f"""
{code_indent}     ddims_vec.clear();"""
            for input_tensor, is_vector in input_name_tensor_map[input_name]:
                if is_vector:
                    input_tensor_truncate = input_tensor[:-4]
                    if input_name in self.inplace_map.values():
                        input_tensor_truncate = input_tensor

                    if input_name in self.optional_vars:
                        input_tensor_code = input_tensor_code + f"""
{code_indent}     if ({input_tensor_truncate}){{
{code_indent}       ddims_vec.reserve({input_tensor_truncate}->size());
{code_indent}       for (size_t i = 0; i < {input_tensor_truncate}->size(); ++i) {{
{code_indent}         ddims_vec.emplace_back((*{input_tensor_truncate}->at(i)).dims());
{code_indent}       }}
{code_indent}     }}"""
                    else:
                        input_tensor_code = input_tensor_code + f"""
{code_indent}     ddims_vec.reserve({input_tensor_truncate}.size());
{code_indent}     for (size_t i = 0; i < {input_tensor_truncate}.size(); ++i) {{
{code_indent}       ddims_vec.emplace_back((*{input_tensor_truncate}[i]).dims());
{code_indent}     }}"""
                else:
                    input_tensor_code = input_tensor_code + f"""  
                  ddims_vec.emplace_back((*{input_tensor}).dims());
{code_indent}     """
            input_tensor_code = input_tensor_code + f"""
{code_indent}     input_shapes.emplace_back("{input_name}", ddims_vec);"""

        input_tensor_code = input_tensor_code + f"""  
{code_indent}     platform::RecordOpInfoSupplement("{self.api}", input_shapes);  
{code_indent}  }}"""
        kernel_args = ["*dev_ctx"]
        for param in kernel_param:
            if param in input_names:
                if param in self.optional_vars:
                    kernel_args.append(PREFIX_TENSOR_NAME + param)
                else:
                    if self.inputs['input_info'][param] == "const Tensor&":
                        kernel_args.append("*" + PREFIX_TENSOR_NAME + param)
                    elif self.inputs['input_info'][
                            param] == "const std::vector<Tensor>&":
                        kernel_args.append(PREFIX_TENSOR_NAME + param)
                    else:
                        # do nothing
                        pass
                # input is dense tensor
                if kernel_tensor_type is None or kernel_tensor_type[0][
                        kernel_param.index(param)] == 'dense':
                    kernel_args_type_list.append(
                        dense_input_trans_map[input_infos[param]])
                else:  # input is selected_rows
                    kernel_args_type_list.append(
                        sr_input_trans_map[input_infos[param]])
            elif param in attr_names:
                # set attr for kernel_context
                if 'IntArray' in self.attrs['attr_info'][param][0]:
                    kernel_args_type_list.append('const phi::IntArray&')
                    param = 'phi::IntArray(' + param + ')'
                elif 'vector<phi::Scalar>' in self.attrs['attr_info'][param][0]:
                    kernel_args_type_list.append(
                        'const std::vector<phi::Scalar>&')
                    param = param
                elif 'Scalar' in self.attrs['attr_info'][param][0]:
                    kernel_args_type_list.append('const phi::Scalar&')
                    param = 'phi::Scalar(' + param + ')'
                else:
                    kernel_args_type_list.append(
                        self.attrs['attr_info'][param][0])
                kernel_args.append(param)
            elif isinstance(param, bool):
                kernel_args.append(str(param).lower())
            else:
                kernel_args.append(str(param))

        for i, out_type in enumerate(self.outputs['types']):
            # output is dense tensor
            if kernel_tensor_type is None or kernel_tensor_type[1][i] == 'dense':
                kernel_args_type_list.append(dense_out_trans_map[out_type])
            else:  # output is selected_rows
                kernel_args_type_list.append(sr_out_trans_map[out_type])

        kernel_signature = "void(*)(" + ", ".join(kernel_args_type_list) + ")"

        return input_tensor_code, ", ".join(kernel_args), kernel_signature

    # Override by child class
    def gene_return_code(self):
        return "return api_output;"

    # Override by child class
    def gene_output(self,
                    out_dtype_list,
                    out_tensor_type_list=None,
                    code_indent='',
                    inplace_flag=False):
        return None, None, None

    def gen_kernel_code(self, kernel_name, code_indent, inplace_flag=False):
        kernel_dispatch = self.kernel['dispatch'][kernel_name]
        input_tensors, kernel_args, kernel_signature = self.get_kernel_args(
            kernel_dispatch, code_indent)
        out_tensor_type_list = kernel_dispatch[1] if kernel_dispatch else None
        outputs_args, kernel_output_names, output_create = self.gene_output(
            self.outputs['types'], out_tensor_type_list, code_indent,
            inplace_flag)
        fallback_kernel_output_trans = ""
        for kernel_out in outputs_args:
            fallback_kernel_output_trans += (f"""
{code_indent}    TransDataBackend({kernel_out}, kernel_backend, {kernel_out});"""
                                             )
        cudnn_args = '' if self.kernel[
            'use_gpudnn'] == 'false' else ', ' + self.kernel['use_gpudnn']
        return f"""
{code_indent}  VLOG(6) << "{self.api} API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
{code_indent}  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
{code_indent}      "{kernel_name}", {{kernel_backend, kernel_layout, kernel_data_type}}{cudnn_args});
{code_indent}  const auto& kernel = kernel_result.kernel;
{code_indent}  VLOG(6) << "{kernel_name} kernel: " << kernel;
{code_indent}  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
{input_tensors}
{output_create}
{code_indent}  paddle::platform::RecordEvent *infer_shape_record_event = nullptr;
{code_indent}  if(paddle::platform::RecordEvent::IsEnabled()){{
{code_indent}    infer_shape_record_event = new paddle::platform::RecordEvent(\"{self.api} infer_meta\", paddle::platform::TracerEventType::OperatorInner, 1);
{code_indent}  }}
{self.gene_infer_meta(kernel_output_names, code_indent)}
{code_indent}  if(infer_shape_record_event != nullptr){{
{code_indent}    delete infer_shape_record_event;
{code_indent}  }}
{code_indent}  using kernel_signature = {kernel_signature};
{code_indent}  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
{code_indent}  paddle::platform::RecordEvent* kernel_record_event = nullptr;
{code_indent}  if(paddle::platform::RecordEvent::IsEnabled()){{
{code_indent}    kernel_record_event = new paddle::platform::RecordEvent(\"{self.api} compute\", paddle::platform::TracerEventType::OperatorInner, 1);
{code_indent}  }}
{code_indent}    (*kernel_fn)({kernel_args}, {", ".join(outputs_args)});
{code_indent}  if(kernel_record_event != nullptr){{
{code_indent}    delete kernel_record_event;
{code_indent}  }}
{code_indent}  if (kernel_result.has_fallback_cpu) {{
{fallback_kernel_output_trans}
{code_indent}  }}
{code_indent}  {self.gene_return_code()}"""

    def get_condition_code(self, kernel_name):
        assert self.kernel['dispatch'][kernel_name], \
                f"{self.api} api: the tensor type of inputs and outputs for kernel isn't set, see also 'kernel:func' of 'scale' in ops.yaml."
        input_types = self.kernel['dispatch'][kernel_name][0]
        condition_list = []
        for i, in_type in enumerate(input_types):
            if in_type == "dense":
                if self.inputs['names'][i] in self.optional_vars:
                    condition_list.append(
                        f"(!{self.inputs['names'][i]} || {self.inputs['names'][i]}->is_dense_tensor())"
                    )
                else:
                    condition_list.append(
                        f"{self.inputs['names'][i]}.is_dense_tensor()")
            else:
                if self.inputs['names'][i] in self.optional_vars:
                    condition_list.append(
                        f"(!{self.inputs['names'][i]} || {self.inputs['names'][i]}->is_selected_rows())"
                    )
                else:
                    condition_list.append(
                        f"{self.inputs['names'][i]}.is_selected_rows()")
        return " && ".join(condition_list)

    def gene_dispatch_code(self, kernel_name, inplace_flag=False):
        return f"""
  if ({self.get_condition_code(kernel_name)}) {{
{self.gen_kernel_code(kernel_name, '  ', inplace_flag)}
  }}
"""

    def gene_base_api_code(self, inplace_flag=False):
        api_func_name = self.get_api_func_name()
        if inplace_flag and api_func_name[-1] != '_':
            api_func_name += '_'
        api_code = f"""
PADDLE_API {self.get_return_type(inplace_flag)} {api_func_name}({self.get_define_args(inplace_flag)}) {{
{self.gene_kernel_select()}
"""

        if len(self.kernel['func']) > 1:
            kernel_dispatch_code = ''
            for kernel_name in self.kernel['func']:
                kernel_dispatch_code += self.gene_dispatch_code(
                    kernel_name, inplace_flag)
            return api_code + f"""
{kernel_dispatch_code}
  PADDLE_THROW(phi::errors::Unimplemented(
          "The kernel of ({self.api}) for input tensors is unimplemented, please check the type of input tensors."));
}}
"""
        else:
            return api_code + self.gen_kernel_code(self.kernel['func'][0], '',
                                                   inplace_flag) + """
}
"""

    def gene_invoke_code(self, invoke_code, params_code):
        return f"""
PADDLE_API {self.get_return_type()} {self.api}({params_code}) {{
  return {invoke_code};
}}"""

    def gene_api_code(self):
        if self.is_base_api:
            api_code = self.gene_base_api_code()
            if len(self.inplace_map) > 0:
                if self.api[-1] == '_':
                    api_code = ""
                api_code = api_code + self.gene_base_api_code(inplace_flag=True)
            return api_code

        else:
            invoke_func_name = self.invoke.split('(')[0].strip()
            if invoke_func_name in self.attrs['names']:
                # Adjust the param whose name is same with api invoked.
                pattern = r'\W' + invoke_func_name + '[^A-Za-z0-9_(]'

                def adjust_name(matched):
                    matched_str = matched.group()
                    return matched_str[0:-1] + '_val' + matched_str[-1]

                invoke_code = re.sub(pattern, adjust_name, self.invoke)
                params_code = re.sub(pattern, adjust_name,
                                     self.get_define_args())
            else:
                invoke_code = self.invoke
                params_code = self.get_define_args()
            return self.gene_invoke_code(invoke_code, params_code)
