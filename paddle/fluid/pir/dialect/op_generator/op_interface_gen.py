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

from decomp_interface_gen_op_list import decomp_interface_declare_gen_op_list

# generator interfaces
from vjp_interface_black_list import vjp_interface_black_list

OP_INFER_SHAPE_TEMPLATE = """
void {op_name}::InferMeta( phi::InferMetaContext *infer_meta ) {{
  auto fn = PD_INFER_META(phi::{infer_meta_func});
  fn(infer_meta);
}}
"""

OP_VJP_FORWARD_INPUT_OR_OUTPUT_TEMPLATE = """
    {input_type} {input_name}(std::make_shared<primitive::LazyTensor>(op_obj.{input_name}()));"""

OP_VJP_FORWARD_MULTI_INPUT_TEMPLATE = """
    pir::CombineOp combine_op_obj_{input_name} =
      op_obj.{input_name}().dyn_cast<pir::OpResult>().owner()->dyn_cast<pir::CombineOp>();
    std::vector<Tensor> {input_name};
    for (size_t idx = 0; idx < combine_op_obj_{input_name}.inputs().size(); idx++) {{
        {input_name}.emplace_back(
            std::make_shared<primitive::LazyTensor>(combine_op_obj_{input_name}.inputs()[idx]));
    }}"""

OP_VJP_FORWARD_OPTIONAL_INPUT_TEMPLATE = """
    paddle::optional<Tensor> {input_name};
    if (!IsEmptyValue(op_obj.{input_name}())){{
        {input_name} = paddle::make_optional<Tensor>(Tensor(std::make_shared<primitive::LazyTensor>(op_obj.{input_name}())));
    }}"""

OP_VJP_FORWARD_OPTIONAL_VECTOR_INPUT_TEMPLATE = """
    paddle::optional<std::vector<Tensor>> {input_name};
    if (!IsEmptyValue(op_obj.{input_name}())){{
        pir::CombineOp combine_op_obj =
            op_obj.{input_name}().dyn_cast<pir::OpResult>().owner()->dyn_cast<pir::CombineOp>();
        std::vector<Tensor> optional_{input_name};
        for (size_t idx = 0; idx < combine_op_obj.inputs().size(); idx++) {{
            optional_{input_name}.emplace_back(
                std::make_shared<primitive::LazyTensor>(combine_op_obj.inputs()[idx]));
        }}
        {input_name} = paddle::make_optional<std::vector<Tensor>>(optional_{input_name});
    }}"""

OP_VJP_FORWARD_OUTPUT_GRAD_TEMPLATE = """
    Tensor {output_grad_name}(std::make_shared<primitive::LazyTensor>(out_grads[{idx1}][{idx2}]));"""

OP_VJP_FORWARD_OUTPUT_GRAD_LIST_TEMPLATE = """
    std::vector<Tensor> {output_grad_name};
    for (size_t idx = 0; idx < out_grads[{index}].size(); idx++) {{
        {output_grad_name}.emplace_back(
            std::make_shared<primitive::LazyTensor>(out_grads[{index}][idx]));
    }}"""

OP_VJP_FORWARD_OPTIONAL_OUTPUT_GRAD_TEMPLATE = """
    paddle::optional<Tensor> {output_grad_name};
    if (!IsEmptyValue(out_grads[{idx1}][{idx2}])){{
        {output_grad_name} = paddle::make_optional<Tensor>(Tensor(std::make_shared<primitive::LazyTensor>(out_grads[{idx1}][{idx2}])));
    }}"""

OP_VJP_FORWARD_OPTIONAL_VECTOR_OUTPUT_GRAD_TEMPLATE = """
    paddle::optional<std::vector<Tensor>> {output_grad_name};
    std::vector<Tensor> optional_{output_grad_name};
    if (!IsEmptyValue(out_grads[{index}])){{
        for (size_t idx = 0; idx < out_grads[{index}].size(); idx++) {{
            optional_{output_grad_name}.emplace_back(
                std::make_shared<primitive::LazyTensor>(out_grads[{index}][idx]));
        }}
        {output_grad_name} = paddle::make_optional<std::vector<Tensor>>(optional_{output_grad_name});
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
    std::vector<std::vector<pir::OpResult>> res(tensor_res.size());
    for (size_t i = 0; i < tensor_res.size(); ++i) {
        res[i].resize(tensor_res[i].size());
        for (size_t j = 0; j < tensor_res[i].size(); ++j) {
            if(tensor_res[i][j].defined()){
                res[i][j] = std::static_pointer_cast<primitive::LazyTensor>(tensor_res[i][j].impl())->value().dyn_cast<pir::OpResult>();
            }
        }
    }"""

OP_VJP_DEFINE_TEMPLATE = """
std::vector<std::vector<pir::OpResult>> {op_class_name}::Vjp(pir::Operation* op, const std::vector<std::vector<pir::Value>>& out_grads, const std::vector<std::vector<bool>>& stop_gradients){{
    {op_class_name} op_obj = op->dyn_cast<{op_class_name}>(); (void)op_obj;

    VLOG(6) << "Prepare inputs of {op_grad_name}";
{forward_input_output_code}
{forward_output_grad_code}

    VLOG(6) << "Vjp prepare Prepare attributes of {op_grad_name}";
{attribute_code}

    VLOG(6) << "Vjp prepare call {op_phi_name}'s vjp inteface";
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
    op_class_name,
    op_grad_name,
    op_phi_name,
    op_info,
    op_grad_info,
):
    bw_input_list = op_grad_info.input_name_list
    forward_input_output_code = ''
    forward_output_grad_code = ''
    build_args_str = ''
    grad_idx = -1
    for idx in range(len(bw_input_list)):
        build_args_str += bw_input_list[idx] + ", "
        input_type = input_types_map[op_grad_info.input_type_list[idx]]
        if (
            bw_input_list[idx] in op_info.input_name_list
            or bw_input_list[idx] in op_info.output_name_list
        ):
            if op_grad_info.input_optional_list[idx] == 'true':
                if input_type == 'Tensor':
                    forward_input_output_code += (
                        OP_VJP_FORWARD_OPTIONAL_INPUT_TEMPLATE.format(
                            input_name=bw_input_list[idx],
                        )
                    )
                else:
                    forward_input_output_code += (
                        OP_VJP_FORWARD_OPTIONAL_VECTOR_INPUT_TEMPLATE.format(
                            input_name=bw_input_list[idx],
                        )
                    )
            else:
                if input_type == 'Tensor':
                    forward_input_output_code += (
                        OP_VJP_FORWARD_INPUT_OR_OUTPUT_TEMPLATE.format(
                            input_type=input_type,
                            input_name=bw_input_list[idx],
                        )
                    )
                else:
                    forward_input_output_code += (
                        OP_VJP_FORWARD_MULTI_INPUT_TEMPLATE.format(
                            input_name=bw_input_list[idx],
                        )
                    )
        else:
            grad_idx += 1
            if op_grad_info.input_optional_list[idx] == 'true':
                if input_type == 'Tensor':
                    forward_input_output_code += (
                        OP_VJP_FORWARD_OPTIONAL_OUTPUT_GRAD_TEMPLATE.format(
                            output_grad_name=bw_input_list[idx],
                            idx1=grad_idx,
                            idx2=0,
                        )
                    )
                else:
                    forward_input_output_code += OP_VJP_FORWARD_OPTIONAL_VECTOR_OUTPUT_GRAD_TEMPLATE.format(
                        output_grad_name=bw_input_list[idx], index=grad_idx
                    )
            else:
                if input_type == 'Tensor':
                    forward_output_grad_code += (
                        OP_VJP_FORWARD_OUTPUT_GRAD_TEMPLATE.format(
                            output_grad_name=bw_input_list[idx],
                            idx1=grad_idx,
                            idx2=0,
                        )
                    )
                else:
                    forward_input_output_code += (
                        OP_VJP_FORWARD_OUTPUT_GRAD_LIST_TEMPLATE.format(
                            output_grad_name=bw_input_list[idx], index=grad_idx
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
                        input_type="Tensor",
                        input_name=op_attribute_list[idx],
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
    op_phi_name_format = op_info.op_yaml_item['name']
    call_vjp_code = OP_VJP_CALL_VJP_TEMPLATE.format(
        op_phi_name=op_phi_name_format,
        inputs_list=build_args_str,
    )
    stop_gradient_input_grad_code = OP_VJP_STOPGRADIENT_TEMPLATE

    str = OP_VJP_DEFINE_TEMPLATE.format(
        op_class_name=op_class_name,
        op_grad_name=op_grad_name,
        op_phi_name=op_phi_name,
        res_size=len(op_info.input_name_list),
        forward_input_output_code=forward_input_output_code,
        forward_output_grad_code=forward_output_grad_code,
        attribute_code=attribute_code,
        call_vjp_code=call_vjp_code,
        stop_gradient_input_grad_code=stop_gradient_input_grad_code,
    )
    return str


def gen_op_infer_meta_str(op_info, op_class_name, op_info_items):
    op_infer_meta_str = ""
    if op_info.infer_meta_func:
        op_infer_meta_str = OP_INFER_SHAPE_TEMPLATE.format(
            op_name=op_class_name,
            infer_meta_func=op_info.infer_meta_func,
        )
    elif op_info.invoke_map and op_info.invoke_map['func'] in op_info_items:
        if op_info_items[op_info.invoke_map['func']].infer_meta_func:
            op_infer_meta_str = OP_INFER_SHAPE_TEMPLATE.format(
                op_name=op_class_name,
                infer_meta_func=op_info_items[
                    op_info.invoke_map['func']
                ].infer_meta_func,
            )
    return op_infer_meta_str


def gen_exclusive_interface_str(op_info, op_info_items):
    exclusive_interface_str = ""
    if op_info.infer_meta_func:
        exclusive_interface_str += (
            "  static void InferMeta( phi::InferMetaContext *infer_meta );"
        )
    elif op_info.invoke_map and op_info.invoke_map['func'] in op_info_items:
        if op_info_items[op_info.invoke_map['func']].infer_meta_func:
            exclusive_interface_str += (
                "  static void InferMeta( phi::InferMetaContext *infer_meta );"
            )
    if op_info.op_phi_name[0] not in vjp_interface_black_list:
        exclusive_interface_str += "\n  static std::vector<std::vector<pir::OpResult>> Vjp(pir::Operation* op, const std::vector<std::vector<pir::Value>>& out_grads, const std::vector<std::vector<bool>>& stop_gradients);"
    if op_info.op_phi_name[0] in decomp_interface_declare_gen_op_list:
        exclusive_interface_str += "\n  static std::vector<std::vector<pir::OpResult>> Decomp(pir::Operation* op);"
    return exclusive_interface_str
