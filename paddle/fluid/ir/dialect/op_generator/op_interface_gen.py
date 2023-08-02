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
        exclusive_interface_str += "\n  static std::vector<std::vector<ir::OpResult>> Vjp(ir::Operation* op, const std::vector<std::vector<ir::OpResult>>& out_grads, const std::vector<std::vector<int>>& stop_gradients);"
    return exclusive_interface_str
