# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

# generator interfaces
from __future__ import annotations

from typing import TYPE_CHECKING

from vjp_interface_black_list import vjp_interface_black_list

if TYPE_CHECKING:
    from op_gen import OpInfoParser

CHECK_INPUT_TEMPLATE = """
    PADDLE_ENFORCE_EQ(
      inputs_.size(),
      {inputs_size},
      common::errors::InvalidArgument("{op_name} op's inputs size should be {inputs_size}, but now is %d.", inputs_.size()));
    PADDLE_ENFORCE_EQ(
      outputs.size(),
      {outputs_size},
      common::errors::InvalidArgument("{op_name} op's outputs size should be {outputs_size}, but now is %d.", outputs.size()));
"""

OP_VJP_FORWARD_INPUT_OR_OUTPUT_TEMPLATE = """
    {input_type} {input_name}(std::make_shared<primitive::LazyTensor>({vjp_param_name}[{input_idx}][0]));"""

OP_VJP_FORWARD_MULTI_INPUT_TEMPLATE = """
    std::vector<Tensor> {input_name};
    for (size_t idx = 0; idx < {vjp_param_name}[{input_idx}].size(); idx++) {{
        {input_name}.emplace_back(
            std::make_shared<primitive::LazyTensor>({vjp_param_name}[{input_idx}][idx]));
    }}"""

OP_VJP_FORWARD_OPTIONAL_INPUT_TEMPLATE = """
    paddle::optional<Tensor> {input_name};
    if (!IsEmptyValue({vjp_param_name}[{input_idx}][0])){{
        {input_name} = paddle::make_optional<Tensor>(Tensor(std::make_shared<primitive::LazyTensor>({vjp_param_name}[{input_idx}][0])));
    }}"""

OP_VJP_FORWARD_OPTIONAL_VECTOR_INPUT_TEMPLATE = """
    paddle::optional<std::vector<Tensor>> {input_name};
    std::vector<Tensor> optional_{input_name};
    if (!IsEmptyValue({vjp_param_name}[{input_idx}][0])){{
        for (size_t idx = 0; idx < {vjp_param_name}[{input_idx}].size(); idx++) {{
            optional_{input_name}.emplace_back(
                std::make_shared<primitive::LazyTensor>({vjp_param_name}[{input_idx}][idx]));
        }}
        {input_name} = paddle::make_optional<std::vector<Tensor>>(optional_{input_name});
    }}"""

OP_VJP_ATTRIBUTE_TEMPLATE = """
    {attr_type} {attr_name} = op->attribute("{attr_name}").dyn_cast<{attr_parse_type}>().{func}();"""

OP_VJP_ATTRIBUTE_DEFAULT_TEMPLATE = """
    {attr_type} {attr_name} = {default_value};"""

OP_VJP_ATTRIBUTE_ARRAY_TEMPLATE = """
    {attr_type} {attr_name};
    for (size_t i = 0; i < op->attribute("{attr_name}").dyn_cast<pir::ArrayAttribute>().size(); i++) {{
        {attr_name}.push_back(op->attribute("{attr_name}").dyn_cast<pir::ArrayAttribute>().at(i).dyn_cast<{inner_type}>().{func}());
    }}"""

OP_VJP_CALL_VJP_TEMPLATE = """
    std::vector<std::vector<Tensor>> tensor_res =
        primitive::{op_phi_name}_vjp(
        {inputs_list}stop_gradients);"""

OP_VJP_STOPGRADIENT_TEMPLATE = """
    std::vector<std::vector<pir::Value>> res(tensor_res.size());
    for (size_t i = 0; i < tensor_res.size(); ++i) {
        res[i].resize(tensor_res[i].size());
        for (size_t j = 0; j < tensor_res[i].size(); ++j) {
            if(tensor_res[i][j].defined()){
                res[i][j] = std::static_pointer_cast<primitive::LazyTensor>(tensor_res[i][j].impl())->value();
            }
        }
    }"""

OP_VJP_DEFINE_TEMPLATE = """
std::vector<std::vector<pir::Value>> {op_class_name}::Vjp(pir::Operation* op, const std::vector<std::vector<pir::Value>>& inputs_, const std::vector<std::vector<pir::Value>>& outputs, const std::vector<std::vector<pir::Value>>& out_grads, const std::vector<std::vector<bool>>& stop_gradients){{
{check_param}
    VLOG(6) << "Prepare inputs of {op_grad_name}";
{backward_input_code}

    VLOG(6) << "Vjp prepare Prepare attributes of {op_grad_name}";
{attribute_code}

    VLOG(6) << "Vjp prepare call {op_phi_name}'s vjp interface";
{call_vjp_code}

    VLOG(6) << "Vjp prepare stop gradient of {op_grad_name}";
{stop_gradient_input_grad_code}
    return res;
}}
"""

input_types_map = {
    'paddle::dialect::DenseTensorType': 'Tensor',
    'pir::VectorType<paddle::dialect::DenseTensorType>': 'Tensor[]',
}


def gen_op_vjp_str(
    op_class_name: str,
    op_grad_name: str,
    op_phi_name: str,
    op_info: OpInfoParser,
    op_grad_info: OpInfoParser,
):
    bw_input_list = op_grad_info.input_name_list
    fwd_input_and_mutable_attr_name_list = (
        op_info.input_name_list + op_info.mutable_attribute_name_list
    )
    if op_grad_info.forward_input_name_list:
        fwd_inputs_list = op_grad_info.forward_input_name_list
    else:
        fwd_inputs_list = fwd_input_and_mutable_attr_name_list
    if op_grad_info.forward_output_name_list:
        fwd_outputs_list = op_grad_info.forward_output_name_list
    else:
        fwd_outputs_list = op_info.output_name_list

    backward_input_code = ''
    build_args_str = ''
    grad_idx = -1
    for idx in range(len(bw_input_list)):
        bw_input_name = bw_input_list[idx]
        build_args_str += bw_input_name + ", "
        input_type = input_types_map[op_grad_info.input_type_list[idx]]

        vjp_param_name = ''
        index_0 = -1
        if bw_input_name in fwd_inputs_list:
            vjp_param_name = 'inputs_'
            index_0 = fwd_inputs_list.index(bw_input_name)
        elif bw_input_name in fwd_outputs_list:
            vjp_param_name = 'outputs'
            index_0 = fwd_outputs_list.index(bw_input_name)
        else:
            vjp_param_name = 'out_grads'
            offset = len('_grad')
            grad_idx = fwd_outputs_list.index(bw_input_name[:-offset])
            index_0 = grad_idx
        if op_grad_info.input_optional_list[idx] == 'true':
            if input_type == 'Tensor':
                backward_input_code += (
                    OP_VJP_FORWARD_OPTIONAL_INPUT_TEMPLATE.format(
                        vjp_param_name=vjp_param_name,
                        input_name=bw_input_name,
                        input_idx=index_0,
                    )
                )
            else:
                backward_input_code += (
                    OP_VJP_FORWARD_OPTIONAL_VECTOR_INPUT_TEMPLATE.format(
                        vjp_param_name=vjp_param_name,
                        input_name=bw_input_name,
                        input_idx=index_0,
                    )
                )
        else:
            if input_type == 'Tensor':
                backward_input_code += (
                    OP_VJP_FORWARD_INPUT_OR_OUTPUT_TEMPLATE.format(
                        vjp_param_name=vjp_param_name,
                        input_type=input_type,
                        input_name=bw_input_name,
                        input_idx=index_0,
                    )
                )
            else:
                backward_input_code += (
                    OP_VJP_FORWARD_MULTI_INPUT_TEMPLATE.format(
                        vjp_param_name=vjp_param_name,
                        input_name=bw_input_name,
                        input_idx=index_0,
                    )
                )

    op_attribute_list = op_grad_info.attribute_name_list
    attribute_code = ''
    build_attr_str = ''
    array_attr_str = "pir::ArrayAttribute"
    for idx in range(len(op_attribute_list)):
        if op_attribute_list[idx] in op_info.attribute_name_list:
            if op_attribute_list[idx] in op_info.mutable_attribute_name_list:
                attribute_code += (
                    OP_VJP_FORWARD_INPUT_OR_OUTPUT_TEMPLATE.format(
                        vjp_param_name='inputs_',
                        input_type="Tensor",
                        input_name=op_attribute_list[idx],
                        input_idx=fwd_input_and_mutable_attr_name_list.index(
                            op_attribute_list[idx]
                        ),
                    )
                )
                build_args_str += op_attribute_list[idx] + ", "
            else:
                func = "data"
                attr_type = op_grad_info.attribute_gen_arg_type_list[idx]
                attr_type = attr_type.replace("const ", "")
                attr_type = attr_type.replace("&", "")
                if array_attr_str in op_grad_info.attribute_type_list[idx]:
                    inner_type = op_grad_info.attribute_type_list[idx][
                        len(array_attr_str) + 1 : -1
                    ]
                    func = "data"
                    if inner_type == "pir::StrAttribute":
                        func = "AsString"
                    attribute_code += OP_VJP_ATTRIBUTE_ARRAY_TEMPLATE.format(
                        attr_type=attr_type,
                        attr_name=op_attribute_list[idx],
                        inner_type=inner_type,
                        func=func,
                    )
                else:
                    if (
                        op_grad_info.attribute_type_list[idx]
                        == "pir::StrAttribute"
                    ):
                        func = "AsString"
                    attribute_code += OP_VJP_ATTRIBUTE_TEMPLATE.format(
                        attr_type=attr_type,
                        attr_name=op_attribute_list[idx],
                        attr_parse_type=op_grad_info.attribute_type_list[idx],
                        func=func,
                    )
                build_attr_str += op_attribute_list[idx] + ", "

        else:
            attribute_code += OP_VJP_ATTRIBUTE_DEFAULT_TEMPLATE.format(
                attr_type=op_grad_info.attribute_gen_arg_type_list[idx],
                attr_name=op_attribute_list[idx],
                default_value=op_grad_info.attribute_default_value_list[idx],
            )
            build_attr_str += op_attribute_list[idx] + ", "
    build_args_str += build_attr_str
    if op_info.is_sparse_op:
        if op_info.op_phi_name[0].endswith('_'):
            op_phi_name_suffix = 'sp_'
        else:
            op_phi_name_suffix = '_sp'
    else:
        op_phi_name_suffix = ''
    op_phi_name_format = op_info.op_yaml_item['name'] + op_phi_name_suffix
    call_vjp_code = OP_VJP_CALL_VJP_TEMPLATE.format(
        op_phi_name=op_phi_name_format,
        inputs_list=build_args_str,
    )
    stop_gradient_input_grad_code = OP_VJP_STOPGRADIENT_TEMPLATE
    check_param = CHECK_INPUT_TEMPLATE.format(
        op_name=op_phi_name_format,
        inputs_size=len(fwd_input_and_mutable_attr_name_list),
        outputs_size=len(op_info.output_name_list),
        out_grads_size=grad_idx + 1,
    )
    str = OP_VJP_DEFINE_TEMPLATE.format(
        check_param=check_param,
        op_class_name=op_class_name,
        op_grad_name=op_grad_name,
        op_phi_name=op_phi_name,
        backward_input_code=backward_input_code,
        attribute_code=attribute_code,
        call_vjp_code=call_vjp_code,
        stop_gradient_input_grad_code=stop_gradient_input_grad_code,
    )
    return str


def gen_exclusive_interface_str(op_info: OpInfoParser, op_info_items):
    exclusive_interface_str = ""
    if op_info.op_phi_name[0] not in vjp_interface_black_list:
        exclusive_interface_str += "\n  static std::vector<std::vector<pir::Value>> Vjp(pir::Operation* op, const std::vector<std::vector<pir::Value>>& inputs_, const std::vector<std::vector<pir::Value>>& outputs, const std::vector<std::vector<pir::Value>>& out_grads, const std::vector<std::vector<bool>>& stop_gradients);"
    return exclusive_interface_str
