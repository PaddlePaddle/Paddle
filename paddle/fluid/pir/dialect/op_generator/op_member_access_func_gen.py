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

# generator op member function

OP_GET_INPUT_TEMPLATE = """  pir::Value {input_name}() {{ return operand_source({input_index}); }}
"""
OP_GET_OUTPUT_TEMPLATE = """  pir::Value {output_name}() {{ return result({output_index}); }}
"""


# =================================== #
#  gen get input/output methods str   #
# =================================== #
def gen_op_member_access_func(args, op_info, op_info_items):
    op_input_name_list = op_info.input_name_list
    op_mutable_attribute_name_list = op_info.mutable_attribute_name_list
    op_output_name_list = op_info.output_name_list
    op_get_inputs_outputs_str = ""
    for idx in range(len(op_input_name_list)):
        op_get_inputs_outputs_str += OP_GET_INPUT_TEMPLATE.format(
            input_name=op_input_name_list[idx],
            input_index=idx,
        )
    for idx in range(len(op_mutable_attribute_name_list)):
        op_get_inputs_outputs_str += OP_GET_INPUT_TEMPLATE.format(
            input_name=op_mutable_attribute_name_list[idx],
            input_index=idx + len(op_input_name_list),
        )
    for idx in range(len(op_output_name_list)):
        op_get_inputs_outputs_str += OP_GET_OUTPUT_TEMPLATE.format(
            output_name=op_output_name_list[idx],
            output_index=idx,
        )
    return [], op_get_inputs_outputs_str, None
