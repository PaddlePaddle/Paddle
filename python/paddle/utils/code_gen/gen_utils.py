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

PREFIX_TENSOR_NAME = 'dense_'
PREFIX_META_TENSOR_NAME = 'meta_'


def parse_args(api_name, args_str):
    """
    Returns:
       { inputs : {
             names : [] // list of input names
             input_info : { input_name : type }
         }
         attrs: {
             names : [] // list of attribute names
             attr_info : { attr_name : (type, default_value)}
         }
         args_declare : "str" // str of funtion params with default value. Example: (..., bool flag=false)
         args_define : "str" // str of funtion params without default value. Example: (..., bool flag)
       }
    """
    inputs = {'names': [], 'input_info': {}}
    attrs = {'names': [], 'attr_info': {}}
    args_str = args_str.strip()
    assert args_str.startswith('(') and args_str.endswith(')'), \
        f"Args declaration should start with '(' and end with ')', please check the args of {api_name} in yaml."
    args_str = args_str[1:-1]
    args_list = args_str.split(',')
    input_types = [
        'const Tensor&', 'const Tensor &', 'const std::vector<Tensor>&',
        'const std::vector<Tensor> &'
    ]
    attr_types = ['const Scalar&', 'const Scalar &', 'const ScalarArray&', 'const ScalarArray &', \
                  'int', 'int32_t', 'int64_t', 'size_t', 'float', 'double', 'bool', \
                  'const std::vector<int64_t>&', 'Backend', 'DataLayout', 'DataType']
    args_declare_str = ""
    args_define_str = ""

    for item in args_list:
        item = item.strip()
        # match the input tensor
        has_input = False
        for in_type in input_types:
            if item.startswith(in_type):
                input_name = item[len(in_type):].strip()
                assert len(input_name) > 0, \
                    f"The input tensor name should not be empty. Please check the args of {api_name} in yaml."
                assert len(attrs['names']) == 0, \
                    f"The input Tensor should appear before attributes. please check the position of {api_name}:input({input_name}) in yaml"

                inputs['names'].append(input_name)
                inputs['input_info'][input_name] = in_type
                args_declare_str = args_declare_str + in_type + ' ' + input_name + ', '
                args_define_str = args_define_str + in_type + ' ' + input_name + ', '
                has_input = True
                break
        if has_input:
            continue

        # match the attribute
        for attr_type in attr_types:
            if item.startswith(attr_type):
                attr_name = item[len(attr_type):].strip()
                assert len(attr_name) > 0, \
                    f"The attribute name should not be empty. Please check the args of {api_name} in yaml."
                default_value = None
                if '=' in attr_name:
                    attr_infos = attr_name.split('=')
                    attr_name = attr_infos[0].strip()
                    default_value = attr_infos[1].strip()

                default_value_str = "" if default_value is None else '=' + default_value
                args_declare_str = args_declare_str + attr_type + ' ' + attr_name + default_value_str + ', '
                args_define_str = args_define_str + attr_type + ' ' + attr_name + ', '
                attrs['names'].append(attr_name)
                attrs['attr_info'][attr_name] = (attr_type, default_value)
                break

    args = {
        'inputs': inputs,
        'attrs': attrs,
        'args_declare': args_declare_str[:-2],
        'args_define': args_define_str[:-2]
    }
    return args


def parse_output(api_name, output_config):
    def parse_output_item(output_item):
        alllowd_output_types = ['Tensor', 'std::vector<Tensor>']
        if re.search(r'\(\w*\)', output_item):
            result = re.search(
                r"(?P<out_type>[a-zA-Z0-9_<>]+)\s*\((?P<name>\w+)\)",
                output_item)
            out_type = result.group('out_type')
            assert out_type in alllowd_output_types, \
                f"{api_name} : Output type error: the output type only support Tensor and std::vector<Tensor>, \
                  but now is {out_type}."

            return out_type, result.group('name')

        else:
            if output_item.strip() in alllowd_output_types:
                return output_item.strip(), 'out'
            else:
                raise ValueError(
                    "{} : Output type error: the output type only support Tensor and std::vector<Tensor>, \
                  but now is {}.".format(api_name, out_type))

    temp_list = output_config.split(',')

    if len(temp_list) == 1:
        out_type, out_name = parse_output_item(temp_list[0])
        return [out_type], out_name
    else:
        out_type_list = []
        out_name_list = []
        for output_item in temp_list:
            out_type, out_name = parse_output_item(output_item)
            out_type_list.append(out_type)
            out_name_list.append(out_name)

        return out_type_list, ", ".join(out_name_list)


def gene_kernel_select(api, input_names, attrs, kernel) -> str:

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
        assert attr_backend_count > 0 and attr_layout_count > 0 and attr_data_type_count > 0, \
            f"{api} api: When there is no input tensor, the args must have 'Backend', 'DataLayout' and 'DataType'."

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
    auto kernel_key = kernel_key_set.GetHigestPriorityKernelKey();
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

    kernel_select_code = kernel_select_code + f"""
  auto kernel = pten::KernelFactory::Instance().SelectKernelOrThrowError(
      "{kernel['func']}", {{kernel_backend, kernel_layout, kernel_data_type}});
  VLOG(6) << "{api} API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  VLOG(6) << "{api} API kernel: " << kernel;"""

    return kernel_select_code


def gene_infer_meta(input_names, attr_names, output_names, infer_meta) -> str:
    infer_meta_params = infer_meta['param'] + output_names if infer_meta[
        'param'] is not None else input_names + attr_names + output_names
    # generate meta tensors
    meta_tensor_code = ""
    param_code = ""
    for param in infer_meta_params:
        if param in input_names:
            param_code = param_code + "MakeMetaTensor(*" + PREFIX_TENSOR_NAME + param + "), "
        elif param in output_names:
            meta_tensor_code = meta_tensor_code + "  pten::MetaTensor " + param.replace(
                PREFIX_TENSOR_NAME,
                PREFIX_META_TENSOR_NAME) + "(" + param + ");\n"
            param_code = param_code + "&" + param.replace(
                PREFIX_TENSOR_NAME, PREFIX_META_TENSOR_NAME) + ", "
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
  pten::{infer_meta['func']}({param_code});
"""


def get_kernel_args(inputs, attrs, out_type_list, kernel_param, data_transform):
    input_trans_map = {
        'const Tensor&': 'const pten::DenseTensor&',
        'const Tensor &': 'const pten::DenseTensor&',
        'const std::vector<Tensor>&': 'const std::vector<pten::DenseTensor>&',
        'const std::vector<Tensor> &': 'const std::vector<pten::DenseTensor>&'
    }
    out_trans_map = {
        'Tensor': 'pten::DenseTensor*',
        'std::vector<Tensor>': 'std::vector<pten::DenseTensor*>&'
    }
    input_names = inputs['names']
    input_infos = inputs['input_info']
    kernel_args_type_list = ['const platform::DeviceContext&']

    input_tensor_code = ""
    for input_name in input_names:
        # set input code
        input_tensor_code = input_tensor_code + f"""
  auto {PREFIX_TENSOR_NAME}{input_name} = TensorToDenseTensor({input_name});"""

    attr_names = attrs['names']
    if kernel_param is None:
        kernel_param = input_names + attr_names

    input_tensor_code = ""
    for i, input_name in enumerate(input_names):
        # set input code
        if input_name in kernel_param:
            trans_flag = "{}"
            if input_name in data_transform['skip_transform']:
                trans_flag = "{true}"
            elif input_name in data_transform['support_trans_dtype']:
                trans_flag = "{false, true}"
            input_tensor_code = input_tensor_code + f"""
  auto {PREFIX_TENSOR_NAME}{input_name} = PrepareData({input_name}, kernel.InputAt({i}), {trans_flag});"""

        else:
            input_tensor_code = input_tensor_code + f"""
  auto {PREFIX_TENSOR_NAME}{input_name} = TensorToDenseTensor({input_name});"""

    kernel_args = "*dev_ctx, "
    for param in kernel_param:
        if param in input_names:
            kernel_args = kernel_args + "*" + PREFIX_TENSOR_NAME + param + ", "
            kernel_args_type_list.append(input_trans_map[input_infos[param]])
        elif param in attr_names:
            # set attr for kernel_context
            if 'ScalarArray' in attrs['attr_info'][param][0]:
                kernel_args_type_list.append('const pten::ScalarArray&')
                param = 'pten::ScalarArray(' + param + ')'
            elif 'Scalar' in attrs['attr_info'][param][0]:
                kernel_args_type_list.append('const pten::Scalar&')
                param = 'pten::Scalar(' + param + ')'
            else:
                kernel_args_type_list.append(attrs['attr_info'][param][0])
            kernel_args = kernel_args + param + ", "
        elif isinstance(param, bool):
            kernel_args = kernel_args + str(param).lower() + ", "
        else:
            kernel_args = kernel_args + str(param) + ", "

    for out_type in out_type_list:
        kernel_args_type_list.append(out_trans_map[out_type])

    kernel_signature = "void(*)(" + ", ".join(kernel_args_type_list) + ")"

    return input_tensor_code, kernel_args[:-2], kernel_signature
