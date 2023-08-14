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
from vjp_interface_gen_op_list import vjp_interface_gen_op_list

OP_INFER_SHAPE_TEMPLATE = """
void {op_name}::InferMeta( phi::InferMetaContext *infer_meta ) {{
  auto fn = PD_INFER_META(phi::{infer_meta_func});
  fn(infer_meta);
}}
"""

OP_VJP_FORWARD_INPUT_OR_OUTPUT_TEMPLATE = """
    {input_type} {input_name}(std::make_shared<primitive::experimental::DescTensor>(op_obj.{input_name}()));
"""

OP_VJP_FORWARD_OUTPUT_GRAD_TEMPLATE = """
    Tensor {output_grad_name}(std::make_shared<primitive::experimental::DescTensor>((out_grads[{idx1}][{idx2}]);
"""

OP_VJP_FORWARD_OUTPUT_GRAD_LIST_TEMPLATE = """
    std::vector<Tensor> {output_grad_name}(std::make_shared<primitive::experimental::DescTensor>((out_grads[{idx1}]);
"""

OP_VJP_CALL_VJP_TEMPLATE = """
    Tensor std::vector<std::vector<Tensor>> tensor_res =
      primitive::experimental::{op_phi_name}_vjp({inputs_list}, stop_gradients);
"""

OP_VJP_STOPGRADIENT_TEMPLATE = """
    if(!stop_gradients[{idx1}][{idx2}]){{
        res[{idx1}][{idx2}] = std::static_pointer_cast<primitive::experimental::DescTensor>(
            tensor_res[idx1][idx2].impl())
            ->getValue()
            .dyn_cast<ir::OpResult>();
    }}
"""

OP_VJP_DEFINE_TEMPLATE = """
std::vector<std::vector<ir::OpResult>> {op_class_name}::Vjp(
    ir::Operation* op,
    const std::vector<std::vector<ir::OpResult>>& out_grads,
    const std::vector<std::vector<bool>>& stop_gradients){{
  {op_class_name} op_obj = op->dyn_cast<{op_class_name}>();

  VLOG(6) << "Prepare inputs of {op_grad_name}";

  {forward_input_code}
  {forward_output_code}
  {forward_output_grad_code}

  VLOG(6) << "Vjp prepare Prepare attributes of {op_grad_name}";
  {attribute_code}

  VLOG(4) << "Vjp prepare call {op_phi_name}'s vjp inteface";
  {call_vjp_code}

  std::vector<std::vector<ir::OpResult>> res(1, std::vector<ir::OpResult>(1));
  {stop_gradient_input_grad_code}

  return res;
}}
"""


def gen_op_vjp_str(
    op_class_name,
    op_grad_name,
    op_phi_name,
    op_info,
    op_grad_info,
):
    forward_input_code = ''
    forward_output_code = ''
    forward_output_grad_code = ''
    attribute_code = ''
    call_vjp_code = ''
    stop_gradient_input_grad_code = ''

    str = OP_VJP_DEFINE_TEMPLATE.format(
        op_class_name=op_class_name,
        op_grad_name=op_grad_name,
        op_phi_name=op_phi_name,
        forward_input_code=forward_input_code,
        forward_output_code=forward_output_code,
        forward_output_grad_code=forward_output_grad_code,
        attribute_code=attribute_code,
        call_vjp_code=call_vjp_code,
        stop_gradient_input_grad_code=stop_gradient_input_grad_code,
    )
    return str


def gen_op_infer_meta_str(op_info, op_class_name):
    op_infer_meta_str = ""
    if op_info.infer_meta_func:
        op_infer_meta_str = OP_INFER_SHAPE_TEMPLATE.format(
            op_name=op_class_name,
            infer_meta_func=op_info.infer_meta_func,
        )
    return op_infer_meta_str


def gen_exclusive_interface_str(op_info):
    exclusive_interface_str = ""
    if op_info.infer_meta_func:
        exclusive_interface_str += (
            "  static void InferMeta( phi::InferMetaContext *infer_meta );"
        )
    if op_info.op_phi_name[0] in vjp_interface_gen_op_list:
        exclusive_interface_str += "\n  static std::vector<std::vector<ir::OpResult>> Vjp(ir::Operation* op, const std::vector<std::vector<ir::OpResult>>& out_grads, const std::vector<std::vector<bool>>& stop_gradients);"
    return exclusive_interface_str
