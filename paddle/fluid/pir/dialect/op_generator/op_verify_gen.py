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

# verify
OP_VERIFY_TEMPLATE = """
void {op_name}::VerifySig() {{
  VLOG(4) << "Start Verifying inputs, outputs and attributes for: {op_name}.";
  VLOG(4) << "Verifying inputs:";
  {{
  auto input_size = num_operands();
  PADDLE_ENFORCE_EQ(input_size , {inputs_size}, common::errors::InvalidArgument(
                    "The size of inputs must be equal to {inputs_size}."));{inputs_type_check}
  }}
  VLOG(4) << "Verifying attributes:";
  {{{attributes_check}
  }}
  VLOG(4) << "Verifying outputs:";
  {{
  auto output_size = num_results();
  PADDLE_ENFORCE_EQ(output_size, {outputs_size}, common::errors::InvalidArgument(
                    "The size of outputs must be equal to {outputs_size}."));{outputs_type_check}
  }}
  VLOG(4) << "End Verifying for: {op_name}.";
}}
"""

GRAD_OP_VERIFY_TEMPLATE = """
void {op_name}::VerifySig() {{}}
"""

INPUT_TYPE_CHECK_TEMPLATE = """
  PADDLE_ENFORCE_EQ((*this)->operand_source({index}).type().isa<{standard}>(), true,
                  common::errors::InvalidArgument("Type validation failed for the {index}th input, got %s.", (*this)->operand_source({index}).type()));"""
INPUT_VECTORTYPE_CHECK_TEMPLATE = """
  if (auto vec_type = (*this)->operand_source({index}).type().dyn_cast<pir::VectorType>()) {{
      for (size_t i = 0; i < vec_type.size(); ++i) {{
        PADDLE_ENFORCE_EQ(vec_type[i].isa<{standard}>(), true, common::errors::InvalidArgument(
                       "Type validation failed for the {index}th input, got %s.", (*this)->operand_source({index}).type()));
      }}
  }}
  else {{
    PADDLE_ENFORCE_EQ((*this)->operand_source({index}).type().isa<{standard}>(), true, common::errors::InvalidArgument(
                   "Type validation failed for the {index}th input, got %s.", (*this)->operand_source({index}).type()));
  }}"""
INPUT_OPTIONAL_TYPE_CHECK_TEMPLATE = """
  if (auto val = (*this)->operand({index})) {{
    PADDLE_ENFORCE_EQ(val.type().isa<{standard}>(), true, common::errors::InvalidArgument(
                   "Type validation failed for the {index}th input, got %s.", (*this)->operand_source({index}).type()));
  }}"""
INPUT_OPTIONAL_VECTORTYPE_CHECK_TEMPLATE = """
  if (auto val =  (*this)->operand({index})) {{
    if (auto vec_type = val.type().dyn_cast<pir::VectorType>()) {{
      for (size_t i = 0; i < vec_type.size(); i++) {{
        PADDLE_ENFORCE_EQ(vec_type[i].isa<{standard}>(), true, common::errors::InvalidArgument(
                          "Type validation failed for the {index}th input, got %s.", (*this)->operand_source({index}).type()));
      }}
    }}
    else {{
      PADDLE_ENFORCE_EQ(val.type().isa<{standard}>(), true, common::errors::InvalidArgument(
                        "Type validation failed for the {index}th input, got %s.", (*this)->operand_source({index}).type()));
    }}
  }}"""
ATTRIBUTE_CHECK_TEMPLATE = """
  PADDLE_ENFORCE_GT(attributes.count("{attribute_name}"), 0, common::errors::InvalidArgument(
                 "{attribute_name} does not exist."));
  PADDLE_ENFORCE_EQ(attributes.at("{attribute_name}").isa<{standard}>(), true, common::errors::InvalidArgument(
                 "Type of attribute: {attribute_name} is not {standard}."));
"""
ATTRIBUTE_VECTOR_CHECK_TEMPLATE = """
  PADDLE_ENFORCE_GT(attributes.count("{attribute_name}"), 0, common::errors::InvalidArgument(
                 "{attribute_name} does not exist."));
  PADDLE_ENFORCE_EQ(attributes.at("{attribute_name}").isa<pir::ArrayAttribute>(), true, common::errors::InvalidArgument(
                 "Type of attribute: {attribute_name} is not pir::ArrayAttribute."));
  for (size_t i = 0; i < attributes.at("{attribute_name}").dyn_cast<pir::ArrayAttribute>().size(); i++) {{
    PADDLE_ENFORCE_EQ(attributes.at("{attribute_name}").dyn_cast<pir::ArrayAttribute>().at(i).isa<{standard}>(), true, common::errors::InvalidArgument(
                   "Type of attribute: {attribute_name} is not right."));
  }}"""
OUTPUT_TYPE_CHECK_TEMPLATE = """
  PADDLE_ENFORCE_EQ((*this)->result({index}).type().isa<{standard}>(), true, common::errors::InvalidArgument(
                 "Type validation failed for the {index}th output."));"""
OUTPUT_VECTORTYPE_CHECK_TEMPLATE = """
  auto output_{index}_type = (*this)->result({index}).type();
  if (auto vec_type = output_{index}_type.dyn_cast<pir::VectorType>()) {{
    for (size_t i = 0; i < vec_type.size(); i++) {{
      PADDLE_ENFORCE_EQ(vec_type[i].isa<{standard}>(), true, common::errors::InvalidArgument(
                     "Type validation failed for the {index}th output."));
    }}
  }}
  else {{
    PADDLE_ENFORCE_EQ(output_{index}_type.isa<{standard}>(), true, common::errors::InvalidArgument(
                   "Type validation failed for the {index}th output."));
  }}"""
OUTPUT_OPTIONAL_TYPE_CHECK_TEMPLATE = """
  if (auto output_{index}_type = (*this)->result({index}).type()) {{
    PADDLE_ENFORCE_EQ(output_{index}_type.isa<{standard}>(),true, common::errors::InvalidArgument(
                   "Type validation failed for the {index}th output."));
  }}"""
OUTPUT_OPTIONAL_VECTORTYPE_CHECK_TEMPLATE = """
  if (auto output_{index}_type = (*this)->result({index}).type()) {{
    if (auto vec_type = output_{index}_type.dyn_cast<pir::VectorType>()) {{
      for (size_t i = 0; i < vec_type.size(); ++i) {{
        PADDLE_ENFORCE_EQ(vec_type[i].isa<{standard}>(), true, common::errors::InvalidArgument(
                       "Type validation failed for the {index}th output."));
      }}
    }}
    else {{
      PADDLE_ENFORCE_EQ(output_{index}_type.isa<{standard}>(), true, common::errors::InvalidArgument(
                     "Type validation failed for the {index}th output."));
    }}
  }}"""


# generate inputs_type_check_str
def gen_inputs_type_check_str(
    op_input_type_list,
    op_input_optional_list,
    op_mutable_attribute_name_list,
    op_mutable_attribute_type_list,
):
    if (len(op_input_type_list) + len(op_mutable_attribute_name_list)) == 0:
        inputs_type_check_str = """
  // Inputs num is 0, not need to check inputs type."""
    else:
        inputs_type_check_str = ""
    vector_type_str = "pir::VectorType<"
    for idx in range(len(op_input_type_list)):
        input_type = op_input_type_list[idx]
        is_optional = op_input_optional_list[idx]
        is_vector = False
        if input_type.startswith(vector_type_str):
            is_vector = True
            input_type = input_type[len(vector_type_str) : -1]
        check_str = ""
        if is_optional == "true":
            if is_vector:
                check_str = INPUT_OPTIONAL_VECTORTYPE_CHECK_TEMPLATE.format(
                    index=idx, standard=input_type
                )
            else:
                check_str = INPUT_OPTIONAL_TYPE_CHECK_TEMPLATE.format(
                    index=idx, standard=input_type
                )
        else:
            if is_vector:
                check_str = INPUT_VECTORTYPE_CHECK_TEMPLATE.format(
                    index=idx, standard=input_type
                )
            else:
                check_str = INPUT_TYPE_CHECK_TEMPLATE.format(
                    index=idx, standard=input_type
                )
        inputs_type_check_str += check_str
    for idx in range(len(op_mutable_attribute_name_list)):
        mutable_attribute_type = op_mutable_attribute_type_list[idx][0]
        check_str = ""
        if mutable_attribute_type == "paddle::dialect::ScalarAttribute":
            check_str = INPUT_TYPE_CHECK_TEMPLATE.format(
                index=idx + len(op_input_type_list),
                standard="paddle::dialect::DenseTensorType",
            )
        else:
            check_str = INPUT_VECTORTYPE_CHECK_TEMPLATE.format(
                index=idx + len(op_input_type_list),
                standard="paddle::dialect::DenseTensorType",
            )
        inputs_type_check_str += check_str
    return inputs_type_check_str


# generate attributes_check_str
def gen_attributes_type_check_str(
    op_non_mutable_attribute_name_list, op_non_mutable_attribute_type_list
):
    if len(op_non_mutable_attribute_name_list) == 0:
        attributes_check_str = """
  // Attributes num is 0, not need to check attributes type."""
    else:
        attributes_check_str = """
  auto& attributes = this->attributes();"""
    array_attr_str = "pir::ArrayAttribute<"
    for idx in range(len(op_non_mutable_attribute_name_list)):
        attribute_name = op_non_mutable_attribute_name_list[idx]
        attribute_type = op_non_mutable_attribute_type_list[idx]

        if attribute_type.startswith(array_attr_str):
            attribute_type = attribute_type[len(array_attr_str) : -1]
            attributes_check_str += ATTRIBUTE_VECTOR_CHECK_TEMPLATE.format(
                attribute_name=attribute_name,
                standard=attribute_type,
            )
        else:
            attributes_check_str += ATTRIBUTE_CHECK_TEMPLATE.format(
                attribute_name=attribute_name, standard=attribute_type
            )
    return attributes_check_str


# generate outputs_type_check_str
def gen_outputs_type_check_str(op_output_type_list, op_output_optional_list):
    if len(op_output_type_list) == 0:
        outputs_type_check_str = """
  // Outputs num is 0, not need to check outputs type."""
    else:
        outputs_type_check_str = ""
    vector_type_str = "pir::VectorType<"
    for idx in range(len(op_output_type_list)):
        output_type = op_output_type_list[idx]
        is_optional = op_output_optional_list[idx]
        is_vector = False
        if output_type.startswith(vector_type_str):
            is_vector = True
            output_type = output_type[len(vector_type_str) : -1]
        check_str = ""
        if is_optional == "true":
            if is_vector:
                check_str = OUTPUT_OPTIONAL_VECTORTYPE_CHECK_TEMPLATE.format(
                    index=idx, standard=output_type
                )
            else:
                check_str = OUTPUT_OPTIONAL_TYPE_CHECK_TEMPLATE.format(
                    index=idx, standard=output_type
                )
        else:
            if is_vector:
                check_str = OUTPUT_VECTORTYPE_CHECK_TEMPLATE.format(
                    index=idx, standard=output_type
                )
            else:
                check_str = OUTPUT_TYPE_CHECK_TEMPLATE.format(
                    index=idx, standard=output_type
                )
        outputs_type_check_str += check_str
    return outputs_type_check_str


# generate op verify function
def gen_verify_func_str(
    op_class_name,
    op_input_type_list,
    op_input_optional_list,
    op_mutable_attribute_name_list,
    op_mutable_attribute_type_list,
    op_non_mutable_attribute_name_list,
    op_non_mutable_attribute_type_list,
    op_output_type_list,
    op_output_optional_list,
):
    if "GradOp" in op_class_name or "Grad_Op" in op_class_name:
        return GRAD_OP_VERIFY_TEMPLATE.format(op_name=op_class_name)

    inputs_type_check_str = gen_inputs_type_check_str(
        op_input_type_list,
        op_input_optional_list,
        op_mutable_attribute_name_list,
        op_mutable_attribute_type_list,
    )
    attributes_type_check_str = gen_attributes_type_check_str(
        op_non_mutable_attribute_name_list, op_non_mutable_attribute_type_list
    )

    outputs_type_check_str = gen_outputs_type_check_str(
        op_output_type_list, op_output_optional_list
    )
    # skip AddNOp type check, because it would be select rows.
    if op_class_name == "AddNOp":
        inputs_type_check_str = ""
        outputs_type_check_str = ""

    return OP_VERIFY_TEMPLATE.format(
        op_name=op_class_name,
        inputs_size=len(op_input_type_list)
        + len(op_mutable_attribute_type_list),
        inputs_type_check=inputs_type_check_str,
        attributes_check=attributes_type_check_str,
        outputs_size=len(op_output_type_list),
        outputs_type_check=outputs_type_check_str,
    )
