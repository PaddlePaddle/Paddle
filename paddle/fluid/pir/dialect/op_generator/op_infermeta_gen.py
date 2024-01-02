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

from op_build_gen import GenBuildOutputsPart2, GetAttributes

OP_INFERMETA_TEMPLATE = """
std::vector<pir::Type> {op_name}::InferMeta(std::vector<pir::Value>& input_values, pir::AttributeMap attributes) {{
{infermeta_inputs}
{get_attributes_str}
{infermeta_outputs}
  return argument_outputs;
}}
"""

CREATE_INPUT_VALUE_TEMPLATE = """
  pir::Value {input_name}_ = input_values[{index}];"""

GET_ATTRIBUTES_FROM_MAP_TEMPLATE = """
  IR_ENFORCE(input_values.size() == {op_input_name_list_size},
      "Num of inputs is expected to be {op_input_name_list_size} but got %d.", input_values.size());
"""


def get_infermeta_inputs_str(
    op_input_name_list,
    op_input_type_list,
    op_input_optional_list,
    op_mutable_attribute_name_list,
    mutable_attr_is_input,
):
    op_input_name_list_size = len(op_input_name_list)
    if mutable_attr_is_input:
        op_input_name_list_size += len(op_mutable_attribute_name_list)

    infermeta_inputs_str = GET_ATTRIBUTES_FROM_MAP_TEMPLATE.format(
        op_input_name_list_size=str(op_input_name_list_size),
    )

    for i in range(len(op_input_name_list)):
        infermeta_inputs_str += CREATE_INPUT_VALUE_TEMPLATE.format(
            input_name=op_input_name_list[i], index=str(i)
        )

    if mutable_attr_is_input:
        # add mutable attributes as inputs
        if len(op_mutable_attribute_name_list) > 0:
            for i in range(len(op_mutable_attribute_name_list)):
                infermeta_inputs_str += CREATE_INPUT_VALUE_TEMPLATE.format(
                    input_name=op_mutable_attribute_name_list[i],
                    index=str(i + len(op_input_name_list)),
                )
    infermeta_inputs_str += "\n"

    infermeta_inputs_str += '  VLOG(4) << "Builder construction outputs";\n'
    # Prepar input type
    for idx in range(len(op_input_name_list)):
        # is a vector<Tensor>
        if 'pir::VectorType' in op_input_type_list[idx]:
            if op_input_optional_list[idx] == 'false':
                infermeta_inputs_str += "  pir::VectorType {name} = {name}_.type().dyn_cast<pir::VectorType>(); (void){name};\n".format(
                    name=op_input_name_list[idx]
                )
        # is a Tensor
        else:
            if op_input_optional_list[idx] == 'false':
                infermeta_inputs_str += "  {type} {name} = {name}_.type().dyn_cast<{type}>(); (void){name};\n".format(
                    type=op_input_type_list[idx], name=op_input_name_list[idx]
                )

    return infermeta_inputs_str


def gen_infermeta_func_str(
    op_class_name,
    op_input_name_list,
    op_input_type_list,
    op_input_optional_list,
    op_mutable_attribute_name_list,
    op_mutable_attribute_type_list,
    op_output_name_list,
    op_output_type_list,
    op_output_size_list,
    op_output_optional_list,
    op_infer_meta_map,
    op_inplace_map,
    op_attribute_name_list,
    op_attribute_type_list,
    op_attribute_build_arg_type_list,
    op_non_mutable_attribute_name_list,
    op_non_mutable_attribute_type_list,
    op_non_mutable_attribute_build_arg_type_list,
    muta_attr_is_input=False,
    attr_args_is_map=True,
):
    infermeta_inputs_str = get_infermeta_inputs_str(
        op_input_name_list,
        op_input_type_list,
        op_input_optional_list,
        op_mutable_attribute_name_list,
        muta_attr_is_input,
    )

    get_attributes_str = GetAttributes(
        op_class_name,
        muta_attr_is_input,
        op_attribute_name_list,
        op_attribute_type_list,
        op_attribute_build_arg_type_list,
        op_non_mutable_attribute_name_list,
        op_non_mutable_attribute_type_list,
        op_non_mutable_attribute_build_arg_type_list,
        attr_args_is_map,
    )

    infermeta_outputs_str = GenBuildOutputsPart2(
        op_class_name,
        op_input_name_list,
        op_input_type_list,
        op_input_optional_list,
        op_mutable_attribute_name_list,
        op_mutable_attribute_type_list,
        op_output_name_list,
        op_output_type_list,
        op_output_size_list,
        op_output_optional_list,
        op_infer_meta_map,
        op_inplace_map,
        muta_attr_is_input,
    )

    infermeta_func = OP_INFERMETA_TEMPLATE.format(
        op_name=op_class_name,
        infermeta_inputs=infermeta_inputs_str,
        get_attributes_str=get_attributes_str,
        infermeta_outputs=infermeta_outputs_str,
    )

    return infermeta_func
