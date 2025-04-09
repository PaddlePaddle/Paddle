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

from gen_utils import to_pascal_case
from op_build_gen import (
    _INFERMETA_NEED_META_CONFIG,
    _PREPARE_DATA_WITH_VECTOR_INT64_MTTABLE_ATTRIBUTE,
)

OP_INFERMETA_DECL_STRING = (
    "  static void InferMeta(phi::InferMetaContext *infer_meta );\n"
    "  static std::vector<pir::Type> InferMeta( const std::vector<pir::Value>& input_values, pir::AttributeMap* p_attributes );"
)

OP_INFERMETA_IMPL_TEMPLATE_1 = """
void {op_name}::InferMeta( phi::InferMetaContext *infer_meta ) {{
  auto fn = PD_INFER_META(phi::{infer_meta_func});
  fn(infer_meta);
}}
"""

OP_INFERMETA_IMPL_TEMPLATE_2 = """
std::vector<pir::Type> {op_name}::InferMeta(const std::vector<pir::Value>& input_values, pir::AttributeMap* p_attributes) {{
  PADDLE_ENFORCE_NOT_NULL(
        p_attributes, common::errors::Fatal("AttrtibueMap pointer in InferMeta function is nullptr."));
  auto& attributes = *p_attributes; (void)attributes;
{infermeta_inputs}
{get_attributes_str}
{infermeta_outputs}
  return argument_outputs;
}}
"""

OP_INFERMETA_IMPL_TEMPLATE_2_BY_INVOKE = """
std::vector<pir::Type> {op_name}::InferMeta(const std::vector<pir::Value>& input_values, pir::AttributeMap* attributes) {{
  return {invoke_class}::InferMeta(input_values, attributes);
}}
"""

CREATE_INPUT_VALUE_TEMPLATE = """
  pir::Value {input_name}_ = input_values[{index}]; (void){input_name}_;"""

ENFORCE_INPUT_NUM_TEMPLATE = """
  PADDLE_ENFORCE_EQ(input_values.size() == {op_input_name_list_size}, true, common::errors::InvalidArgument(
      "Num of inputs is expected to be {op_input_name_list_size} but got %d.", input_values.size()));
"""

GET_INPUT_TYPE_TEMPLATE = """
  {type} {name};
  if ({name}_.type().isa<{type}>()) {{
    {name} = {name}_.type().dyn_cast<{type}>(); (void){name};
  }} else {{
    PADDLE_THROW(common::errors::Unimplemented("Only support {type} or {allocated_type}"));
  }}
"""


def get_infermeta_inputs_str(
    op_info,
    inuse_infer_meta_args,
    op_input_name_list,
    op_input_type_list,
    op_input_optional_list,
    op_mutable_attribute_name_list,
    mutable_attr_is_input,
):
    op_input_name_list_size = len(op_info.input_name_list)
    if mutable_attr_is_input:
        op_input_name_list_size += len(op_mutable_attribute_name_list)

    infermeta_inputs_str = ENFORCE_INPUT_NUM_TEMPLATE.format(
        op_input_name_list_size=str(op_input_name_list_size),
    )

    for i in range(len(op_info.input_name_list)):
        if op_info.input_name_list[i] not in inuse_infer_meta_args:
            continue
        infermeta_inputs_str += CREATE_INPUT_VALUE_TEMPLATE.format(
            input_name=op_info.input_name_list[i], index=str(i)
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
    infermeta_inputs_str += (
        '  bool is_from_tensor = false; (void) is_from_tensor;\n'
    )
    # Prepare input type
    for idx in range(len(op_input_name_list)):
        if op_input_name_list[idx] not in inuse_infer_meta_args:
            continue
        # is a vector<Tensor>
        if 'pir::VectorType' in op_input_type_list[idx]:
            infermeta_inputs_str += f"  pir::VectorType {op_input_name_list[idx]} = {op_input_name_list[idx]}_.type().dyn_cast<pir::VectorType>(); (void){op_input_name_list[idx]};\n"
        # is a Tensor
        else:
            if op_input_optional_list[idx] == 'false':
                type = op_input_type_list[idx]
                allocated_type = (
                    type.replace('DenseTensorType', 'AllocatedDenseTensorType')
                    .replace("SelectedRowsType", "AllocatedSelectedRowsType")
                    .replace(
                        "SparseCooTensorType", "AllocatedSparseCooTensorType"
                    )
                    .replace(
                        "SparseCsrTensorType", "AllocatedSparseCsrTensorType"
                    )
                )
                infermeta_inputs_str += GET_INPUT_TYPE_TEMPLATE.format(
                    type=type,
                    name=op_input_name_list[idx],
                    allocated_type=allocated_type,
                )

    return infermeta_inputs_str


def GenBuildOutputsPart2(
    args,
    op_info,
    inuse_infer_meta_args,
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
    mutable_attr_is_input,
):
    CREATE_INPUT_METATENSOR_TEMPLATE = """
  VLOG(4) << "Builder construction  dense_{name}";
  paddle::dialect::IrTensor ir_tensor_{name}(paddle::dialect::TransToPhiDataType({name}.dtype()),
                                                      {name}.dims(),
                                                      {name}.data_layout(),
                                                      {name}.lod(),
                                                      {name}.offset());
  VLOG(4) << "Builder construction  meta_{name}";
  paddle::dialect::IrMetaTensor meta_{name}(&ir_tensor_{name});
"""

    CREATE_SPARSECOO_INPUT_METATENSOR_TEMPLATE = """
  VLOG(4) << "Builder construction  sparse_{name}";
  paddle::dialect::IrSparseCooTensor ir_tensor_{name}(paddle::dialect::TransToPhiDataType({name}.dtype()),
                                                      {name}.dims(),
                                                      {name}.non_zero_dims(),
                                                      {name}.data_layout(),
                                                      {name}.coalesced());
  VLOG(4) << "Builder construction  meta_{name}";
  paddle::dialect::IrMetaTensor meta_{name}(&ir_tensor_{name});
"""

    CREATE_SPARSECSR_INPUT_METATENSOR_TEMPLATE = """
  VLOG(4) << "Builder construction  dense_{name}";
  paddle::dialect::IrSparseCsrTensor ir_tensor_{name}(paddle::dialect::TransToPhiDataType({name}.dtype()),
                                                      {name}.dims(),
                                                      {name}.data_layout(),
                                                      {name}.non_zero_crows(),
                                                      {name}.non_zero_cols(),
                                                      {name}.non_zero_elements());
  VLOG(4) << "Builder construction  meta_{name}";
  paddle::dialect::IrMetaTensor meta_{name}(&ir_tensor_{name});
"""

    CREATE_OPTIONAL_INPUT_METATENSOR_TEMPLATE = """
  paddle::dialect::IrMetaTensor meta_{name};
  paddle::dialect::IrTensor ir_tensor_{name};

  if ({name}_.impl() != nullptr) {{
    VLOG(4) << "Builder construction  dense_{name}";
    {type} {name};
    if ({name}_.type().isa<{type}>()) {{
      {name} = {name}_.type().dyn_cast<{type}>();
    }} else {{
      PADDLE_THROW(common::errors::Unimplemented("Only support {type} or {allocated_type}"));
    }}
    ir_tensor_{name} = paddle::dialect::IrTensor(paddle::dialect::TransToPhiDataType({name}.dtype()),
                                                        {name}.dims(),
                                                        {name}.data_layout(),
                                                        {name}.lod(),
                                                        {name}.offset());
    VLOG(4) << "Builder construction  meta_{name}";
    meta_{name} = paddle::dialect::IrMetaTensor(&ir_tensor_{name});
  }}

"""
    CREATE_SPARSECOO_OPTIONAL_INPUT_METATENSOR_TEMPLATE = """
  paddle::dialect::IrMetaTensor meta_{name};
  paddle::dialect::IrSparseCooTensor ir_tensor_{name};

  if ({name}_.impl() != nullptr) {{
    VLOG(4) << "Builder construction  sparse_{name}";
    {type} {name};
    if ({name}_.type().isa<{type}>()) {{
      {name} = {name}_.type().dyn_cast<{type}>();
    }} else {{
      PADDLE_THROW(common::errors::Unimplemented("Only support {type}"));
    }}
    ir_tensor_{name} = paddle::dialect::IrSparseCooTensor(paddle::dialect::TransToPhiDataType({name}.dtype()),
                                                        {name}.dims(),
                                                        {name}.non_zero_dims(),
                                                        {name}.data_layout(),
                                                        {name}.coalesced());
    VLOG(4) << "Builder construction  meta_{name}";
    meta_{name} = paddle::dialect::IrMetaTensor(&ir_tensor_{name});
  }}

"""
    CREATE_SPARSECSR_OPTIONAL_INPUT_METATENSOR_TEMPLATE = """
  paddle::dialect::IrMetaTensor meta_{name};
  paddle::dialect::IrSparseCsrTensor ir_tensor_{name};

  if ({name}_.impl() != nullptr) {{
    VLOG(4) << "Builder construction  sparse_{name}";
    {type} {name};
    if ({name}_.type().isa<{type}>()) {{
      {name} = {name}_.type().dyn_cast<{type}>();
    }} else {{
      PADDLE_THROW(common::errors::Unimplemented("Only support {type}"));
    }}
    ir_tensor_{name} = paddle::dialect::IrSparseCsrTensor(paddle::dialect::TransToPhiDataType({name}.dtype()),
                                                        {name}.dims(),
                                                        {name}.data_layout(),
                                                        {name}.non_zero_crows(),
                                                        {name}.non_zero_cols(),
                                                        {name}.non_zero_elements());
    VLOG(4) << "Builder construction  meta_{name}";
    meta_{name} = paddle::dialect::IrMetaTensor(&ir_tensor_{name});
  }}

"""
    CREATE_INPUT_VEC_METATENSOR_TEMPLATE = """  std::vector<paddle::dialect::IrTensor> vec_ir_tensor_{name};
  for (size_t i=0; i < static_cast<size_t>({name}.size()); i++) {{
    if({name}[i].isa<paddle::dialect::DenseTensorType>()) {{
        auto {name}_type = {name}[i].dyn_cast<paddle::dialect::DenseTensorType>();
        vec_ir_tensor_{name}.push_back(paddle::dialect::IrTensor(paddle::dialect::TransToPhiDataType({name}_type.dtype()),
                                                                    {name}_type.dims(),
                                                                    {name}_type.data_layout(),
                                                                    {name}_type.lod(),
                                                                    {name}_type.offset()));
    }} else {{
        PADDLE_THROW(common::errors::Unimplemented("Only support DenseTensorType or AllocatedDenseTensorType"));
    }}
  }}
  std::vector<paddle::dialect::IrMetaTensor> vec_meta_{name};
  for (size_t i=0; i < vec_ir_tensor_{name}.size(); i++) {{
    vec_meta_{name}.push_back(paddle::dialect::IrMetaTensor(&vec_ir_tensor_{name}[i]));
  }}

  std::vector<const phi::MetaTensor*> meta_{name};
  for (size_t i=0; i < static_cast<size_t>(vec_meta_{name}.size()); i++) {{
    meta_{name}.push_back(&vec_meta_{name}[i]);
  }}
 """

    # In cudnn_lstm operator, the output weight_list_grad requires the use of optional input weight_list,
    # so "pir::VectorType {name}" outside the "if" block.
    CREATE_OPTIONAL_INPUT_VEC_METATENSOR_TEMPLATE = """  std::vector<paddle::dialect::IrTensor> vec_ir_tensor_{name};
  if ({name}_.impl() != nullptr) {{
    for (size_t i=0; i < static_cast<size_t>({name}.size()); i++) {{
        if({name}[i].isa<paddle::dialect::DenseTensorType>()) {{
          auto {name}_type = {name}[i].dyn_cast<paddle::dialect::DenseTensorType>();
          vec_ir_tensor_{name}.push_back(paddle::dialect::IrTensor(paddle::dialect::TransToPhiDataType({name}_type.dtype()),
                                                                        {name}_type.dims(),
                                                                        {name}_type.data_layout(),
                                                                        {name}_type.lod(),
                                                                        {name}_type.offset()));
        }} else {{
            PADDLE_THROW(common::errors::Unimplemented("Only support DenseTensorType or AllocatedDenseTensorType"));
        }}
    }}
  }}

  std::vector<paddle::dialect::IrMetaTensor> vec_meta_{name};
  for (size_t i=0; i < vec_ir_tensor_{name}.size(); i++) {{
    vec_meta_{name}.push_back(paddle::dialect::IrMetaTensor(&vec_ir_tensor_{name}[i]));
  }}

  std::vector<const phi::MetaTensor*> meta_{name};
  for (size_t i=0; i < static_cast<size_t>(vec_meta_{name}.size()); i++) {{
    meta_{name}.push_back(&vec_meta_{name}[i]);
  }}

"""

    CREATE_INTARRAY_MUTABLE_ATTRIBUTE_WITH_UNKNOWN_DATA_TEMPLATE = """  is_from_tensor = false;
  phi::IntArray {name} = phi::IntArray(paddle::dialect::ParseValueShape({name}_, &is_from_tensor));
  if (is_from_tensor) {name}.SetFromTensor(true);\n"""

    CREATE_VECTOR_INT_MUTABLE_ATTRIBUTE_WITH_UNKNOWN_DATA_TEMPLATE = """  std::vector<int64_t> {name};
  if ({name}_.isa<pir::OpResult>() && {name}_.defining_op()->isa<paddle::dialect::FullIntArrayOp>()) {{
    {name} = paddle::dialect::GetInt64Vector(
                    {name}_.defining_op()
                    ->dyn_cast<paddle::dialect::FullIntArrayOp>()
                    .attribute("value"));
  }} else if ({name}_.type().isa<pir::VectorType>()) {{
    size_t {name}_size = {name}_.type().dyn_cast<pir::VectorType>().size();
    {name} = std::vector<int64_t>({name}_size, -1);
  }} else if ({name}_.type().isa<paddle::dialect::DenseTensorType>()) {{
    common::DDim {name}_dim = {name}_.type().dyn_cast<paddle::dialect::DenseTensorType>().dims();
    size_t {name}_size = common::product({name}_dim);
    if (common::contain_unknown_dim({name}_dim)) {{
      {name}_size = 1;
    }}
    {name} = std::vector<int64_t>({name}_size, -1);
  }} else if ({name}_.type().isa<paddle::dialect::SparseCooTensorType>()) {{
    common::DDim {name}_dim = {name}_.type().dyn_cast<paddle::dialect::SparseCooTensorType>().dims();
    size_t {name}_size = common::product({name}_dim);
    if (common::contain_unknown_dim({name}_dim)) {{
      {name}_size = 1;
    }}
    {name} = std::vector<int64_t>({name}_size, -1);
  }}else if ({name}_.type().isa<paddle::dialect::SparseCsrTensorType>()) {{
    common::DDim {name}_dim = {name}_.type().dyn_cast<paddle::dialect::SparseCsrTensorType>().dims();
    size_t {name}_size = common::product({name}_dim);
    if (common::contain_unknown_dim({name}_dim)) {{
      {name}_size = 1;
    }}
    {name} = std::vector<int64_t>({name}_size, -1);
  }}else {{
    PADDLE_THROW(common::errors::Unimplemented("Only support VectorType or DenseTensorType or AllocatedDenseTensorType"));
  }}\n"""

    CREATE_SCALAR_MUTABLE_ATTRIBUTE_WITH_UNKNOWN_DATA_TEMPLATE = """  phi::Scalar {name};
  if ({name}_.isa<pir::OpResult>() && {name}_.defining_op()->isa<paddle::dialect::FullOp>()) {{
    {name} = phi::Scalar({name}_.defining_op()
                                  ->dyn_cast<paddle::dialect::FullOp>()
                                  .attribute("value")
                                  .dyn_cast<paddle::dialect::ScalarAttribute>()
                                  .data()
                                  .to<{dtype}>());
  }}
  else {{
    {name} = phi::Scalar(-1);
    {name}.SetFromTensor(true);
  }}\n"""

    CREATE_OUTPUT_METATENSOR_TEMPLATE = """  paddle::dialect::IrTensor dense_{name};
  paddle::dialect::IrMetaTensor meta_{name}(&dense_{name});
"""
    CREATE_OUTPUT_METASELETEROWS_TEMPLATE = """  paddle::dialect::IrSelectedRows dense_{name};
  paddle::dialect::IrMetaTensor meta_{name}(&dense_{name});
"""
    CREATE_OUTPUT_METASPARSECOOTENSOR_TEMPLATE = """  paddle::dialect::IrSparseCooTensor dense_{name};
  paddle::dialect::IrMetaTensor meta_{name}(&dense_{name});
"""

    CREATE_OUTPUT_METASPARSECSRTENSOR_TEMPLATE = """  paddle::dialect::IrSparseCsrTensor dense_{name};
  paddle::dialect::IrMetaTensor meta_{name}(&dense_{name});
"""

    CREATE_OUTPUT_VEC_METATENSOR_TEMPLATE = """  std::vector<paddle::dialect::IrTensor> vec_dense_{name}(({output_size}), paddle::dialect::IrTensor());
  std::vector<paddle::dialect::IrMetaTensor> vec_meta_{name};
  for (size_t i=0; i < static_cast<size_t>({output_size}); i++) {{
    vec_meta_{name}.push_back(paddle::dialect::IrMetaTensor(&vec_dense_{name}[i]));
  }}
  std::vector<phi::MetaTensor*> meta_{name};
  for (size_t i=0; i < static_cast<size_t>(vec_meta_{name}.size()); i++) {{
    meta_{name}.push_back(&vec_meta_{name}[i]);
  }}
"""
    build_output_str = ""
    # Prepare mutable attributes
    if mutable_attr_is_input:
        for idx in range(len(op_mutable_attribute_name_list)):
            attr_dtype = op_mutable_attribute_type_list[idx]
            # int_array
            if attr_dtype[0] == "paddle::dialect::IntArrayAttribute":
                if (
                    op_info.class_name
                    in _PREPARE_DATA_WITH_VECTOR_INT64_MTTABLE_ATTRIBUTE
                ):
                    build_output_str += CREATE_VECTOR_INT_MUTABLE_ATTRIBUTE_WITH_UNKNOWN_DATA_TEMPLATE.format(
                        name=op_mutable_attribute_name_list[idx]
                    )
                else:
                    build_output_str += CREATE_INTARRAY_MUTABLE_ATTRIBUTE_WITH_UNKNOWN_DATA_TEMPLATE.format(
                        name=op_mutable_attribute_name_list[idx]
                    )
            # scalar
            elif attr_dtype[0] == "paddle::dialect::ScalarAttribute":
                build_output_str += CREATE_SCALAR_MUTABLE_ATTRIBUTE_WITH_UNKNOWN_DATA_TEMPLATE.format(
                    name=op_mutable_attribute_name_list[idx],
                    dtype=attr_dtype[1],
                )
            # string
            elif attr_dtype[0] == "pir::StrAttribute":
                build_output_str += ""
            else:
                assert "mutable attribute type is not right."
        build_output_str += "\n"

    # Prepare inputs_meta_tensor & attributes for infer meta
    infer_meta_args = []
    for idx in range(len(op_infer_meta_map['param'])):
        # is input
        if op_infer_meta_map['param'][idx] in op_input_name_list:
            if (
                "meta_" + op_infer_meta_map['param'][idx]
            ) not in infer_meta_args:
                # is a vector<Tensor>
                if (
                    'pir::VectorType'
                    in op_input_type_list[
                        op_input_name_list.index(
                            op_infer_meta_map['param'][idx]
                        )
                    ]
                ):
                    input_index = op_input_name_list.index(
                        op_infer_meta_map['param'][idx]
                    )
                    if op_input_optional_list[input_index] == 'true':
                        build_output_str += CREATE_OPTIONAL_INPUT_VEC_METATENSOR_TEMPLATE.format(
                            name=op_infer_meta_map['param'][idx]
                        )
                    else:
                        build_output_str += (
                            CREATE_INPUT_VEC_METATENSOR_TEMPLATE.format(
                                name=op_infer_meta_map['param'][idx]
                            )
                        )
                # is a Tensor
                else:
                    input_index = op_input_name_list.index(
                        op_infer_meta_map['param'][idx]
                    )
                    if op_input_optional_list[input_index] == 'true':
                        type = op_input_type_list[input_index]
                        allocated_type = (
                            type.replace(
                                'DenseTensorType', 'AllocatedDenseTensorType'
                            )
                            .replace(
                                "SelectedRowsType", "AllocatedSelectedRowsType"
                            )
                            .replace(
                                "SparseCooTensorType",
                                "AllocatedSparseCooTensorType",
                            )
                            .replace(
                                "SparseCsrTensorType",
                                "AllocatedSparseCsrTensorType",
                            )
                        )
                        if op_info.is_sparse_op:
                            if (
                                op_input_type_list[input_index]
                                == 'paddle::dialect::SparseCooTensorType'
                            ):
                                build_output_str += CREATE_SPARSECOO_OPTIONAL_INPUT_METATENSOR_TEMPLATE.format(
                                    name=op_infer_meta_map['param'][idx],
                                    type=op_input_type_list[input_index],
                                    allocated_type=allocated_type,
                                )
                            elif (
                                op_input_type_list[input_index]
                                == 'paddle::dialect::SparseCsrTensorType'
                            ):
                                build_output_str += CREATE_SPARSECSR_OPTIONAL_INPUT_METATENSOR_TEMPLATE.format(
                                    name=op_infer_meta_map['param'][idx],
                                    type=op_input_type_list[input_index],
                                    allocated_type=allocated_type,
                                )
                            else:
                                build_output_str += CREATE_OPTIONAL_INPUT_METATENSOR_TEMPLATE.format(
                                    name=op_infer_meta_map['param'][idx],
                                    type=op_input_type_list[input_index],
                                    allocated_type=allocated_type,
                                )
                        else:
                            build_output_str += CREATE_OPTIONAL_INPUT_METATENSOR_TEMPLATE.format(
                                name=op_infer_meta_map['param'][idx],
                                type=op_input_type_list[input_index],
                                allocated_type=allocated_type,
                            )
                    else:
                        if op_info.is_sparse_op:
                            if (
                                op_input_type_list[input_index]
                                == 'paddle::dialect::SparseCooTensorType'
                            ):
                                build_output_str += CREATE_SPARSECOO_INPUT_METATENSOR_TEMPLATE.format(
                                    name=op_infer_meta_map['param'][idx],
                                    type=op_input_type_list[input_index],
                                )
                            elif (
                                op_input_type_list[input_index]
                                == 'paddle::dialect::SparseCsrTensorType'
                            ):
                                build_output_str += CREATE_SPARSECSR_INPUT_METATENSOR_TEMPLATE.format(
                                    name=op_infer_meta_map['param'][idx],
                                    type=op_input_type_list[input_index],
                                )
                            else:
                                build_output_str += (
                                    CREATE_INPUT_METATENSOR_TEMPLATE.format(
                                        name=op_infer_meta_map['param'][idx]
                                    )
                                )
                        else:
                            build_output_str += (
                                CREATE_INPUT_METATENSOR_TEMPLATE.format(
                                    name=op_infer_meta_map['param'][idx]
                                )
                            )

            infer_meta_args.append("meta_" + op_infer_meta_map['param'][idx])
        # is attribute
        else:
            infer_meta_args.append(str(op_infer_meta_map['param'][idx]))

    # Prepare outputs_meta_tensor for infer meta

    for idx in range(len(op_output_name_list)):
        # is a vector<Tensor>
        if 'pir::VectorType' in op_output_type_list[idx]:
            build_output_str += CREATE_OUTPUT_VEC_METATENSOR_TEMPLATE.format(
                name=op_output_name_list[idx],
                output_size=op_output_size_list[idx],
            )
            infer_meta_args.append(f"meta_{op_output_name_list[idx]}")
        # is a Tensor
        else:
            if op_output_type_list[idx] == "paddle::dialect::DenseTensorType":
                build_output_str += CREATE_OUTPUT_METATENSOR_TEMPLATE.format(
                    name=op_output_name_list[idx]
                )
                infer_meta_args.append(f"&meta_{op_output_name_list[idx]}")
            elif (
                op_output_type_list[idx]
                == "paddle::dialect::SparseCooTensorType"
            ):
                build_output_str += (
                    CREATE_OUTPUT_METASPARSECOOTENSOR_TEMPLATE.format(
                        name=op_output_name_list[idx]
                    )
                )
                infer_meta_args.append(f"&meta_{op_output_name_list[idx]}")
            elif (
                op_output_type_list[idx]
                == "paddle::dialect::SparseCsrTensorType"
            ):
                build_output_str += (
                    CREATE_OUTPUT_METASPARSECSRTENSOR_TEMPLATE.format(
                        name=op_output_name_list[idx]
                    )
                )
                infer_meta_args.append(f"&meta_{op_output_name_list[idx]}")
            else:
                build_output_str += (
                    CREATE_OUTPUT_METASELETEROWS_TEMPLATE.format(
                        name=op_output_name_list[idx]
                    )
                )
                infer_meta_args.append(f"&meta_{op_output_name_list[idx]}")

    # Execute infer meta function
    CREATE_INFER_META_FUNC_TEMPLATE = """
  phi::{func}({args});
"""
    CREATE_INFER_META_FUNC_WITH_META_CONFIG_TEMPLATE = """
  phi::{func}({args}, phi::MetaConfig(false, false));
"""
    infer_meta_args = [
        arg.lower() if arg in ['False', 'True'] else arg
        for arg in infer_meta_args
    ]
    if op_infer_meta_map['func'] in _INFERMETA_NEED_META_CONFIG:
        build_output_str += (
            CREATE_INFER_META_FUNC_WITH_META_CONFIG_TEMPLATE.format(
                func=op_infer_meta_map['func'], args=", ".join(infer_meta_args)
            )
        )
    else:
        build_output_str += CREATE_INFER_META_FUNC_TEMPLATE.format(
            func=op_infer_meta_map['func'], args=", ".join(infer_meta_args)
        )

    # use dense_{name} or vec_dense_{name} to create Outputs type
    build_output_str += "\n  std::vector<pir::Type> argument_outputs;"

    CREATE_OUTPUT_DENSE_TENSOR_TEMPLATE = """
  pir::Type {name}_type = CvtTo{type}(dense_{name});
"""
    CREATE_OUTPUT_INPLACE_OPTIONAL_DENSE_TENSOR_TEMPLATE = """
  pir::Type {name}_type;
  if ({input_name}_.impl() != nullptr) {{
    {name}_type = CvtTo{type}(dense_{name});
  }}
"""

    CREATE_OUTPUT_VEC_DENSE_TENSOR_TEMPLATE = """
  std::vector<pir::Type> {name}_types;
  for (size_t i=0; i < static_cast<size_t>({output_size}); i++) {{
    {name}_types.push_back(CvtToDenseTensorType(vec_dense_{name}[i]));
  }}
  pir::Type {name}_type = pir::VectorType::get(pir::IrContext::Instance(), {name}_types);
"""
    for idx in range(len(op_output_name_list)):
        # is a vector<Tensor>
        if 'pir::VectorType' in op_output_type_list[idx]:
            build_output_str += CREATE_OUTPUT_VEC_DENSE_TENSOR_TEMPLATE.format(
                name=op_output_name_list[idx],
                output_size=op_output_size_list[idx],
            )
        # is a Tensor
        else:
            output_name = op_output_name_list[idx]
            has_input_inplace = (
                op_inplace_map is not None
                and output_name in op_inplace_map.keys()
            )
            if op_output_optional_list[idx] == 'true' and has_input_inplace:
                # is a inplace optional output
                build_output_str += (
                    CREATE_OUTPUT_INPLACE_OPTIONAL_DENSE_TENSOR_TEMPLATE.format(
                        input_name=op_inplace_map[output_name],
                        name=output_name,
                        type=op_output_type_list[idx][17:],
                    )
                )
            else:
                build_output_str += CREATE_OUTPUT_DENSE_TENSOR_TEMPLATE.format(
                    type=op_output_type_list[idx][17:], name=output_name
                )
    build_output_str += GenDistBranch(args, op_info)

    PUSH_BACK_OUTPUT_TYPE_TEMPLATE = """
  argument_outputs.push_back({name});
"""
    for idx in range(len(op_output_name_list)):
        build_output_str += PUSH_BACK_OUTPUT_TYPE_TEMPLATE.format(
            name=op_output_name_list[idx] + "_type",
        )
    return build_output_str


def GetAttributes(
    op_info,
    mutable_attr_is_input,
    inuse_infer_meta_args,
    attr_args_is_map,
):
    GET_ATTRIBUTES_FROM_MAP_TEMPLATE = """
  PADDLE_ENFORCE_NE(
      attributes.find("{attribute_name}"),
      attributes.end(),
      common::errors::InvalidArgument(
          "'{attribute_name}' Attribute is expected for {op_name}. "));
  {attr_type} {attribute_name} = attributes.at("{attribute_name}").dyn_cast<{attr_ir_type}>().data();
"""
    GET_STR_ATTRIBUTES_FROM_MAP_TEMPLATE = """
  PADDLE_ENFORCE_NE(
      attributes.find("{attribute_name}"),
      attributes.end(),
      common::errors::InvalidArgument(
          "'{attribute_name}' Attribute is expected for {op_name}. "));
  {attr_type} {attribute_name} = attributes.at("{attribute_name}").dyn_cast<pir::StrAttribute>().AsString();
"""
    GET_ARRAY_ATTRIBUTE_FROM_MAP_TEMPLATE = """
  PADDLE_ENFORCE_NE(
      attributes.find("{attribute_name}"),
      attributes.end(),
      common::errors::InvalidArgument(
          "'{attribute_name}' Attribute is expected for {op_name}. "));
  {attr_type} {attribute_name};
  for (size_t i = 0; i < attributes.at("{attribute_name}").dyn_cast<pir::ArrayAttribute>().size(); i++) {{
    {attribute_name}.push_back(attributes.at("{attribute_name}").dyn_cast<pir::ArrayAttribute>().at(i).dyn_cast<{inner_type}>().{data_name}());
  }}
"""
    GET_INTARRAY_ATTRIBUTE_FROM_MAP_TEMPLATE = """
  PADDLE_ENFORCE_NE(
      attributes.find("{attribute_name}"),
      attributes.end(),
      common::errors::InvalidArgument(
          "'{attribute_name}' Attribute is expected for {op_name}. "));
  {attr_type} {attribute_name} = attributes.at("{attribute_name}").dyn_cast<paddle::dialect::IntArrayAttribute>().data().GetData();
"""
    GET_SCALAR_ATTRIBUTE_FROM_MAP_TEMPLATE = """
  PADDLE_ENFORCE_NE(
      attributes.find("{attribute_name}"),
      attributes.end(),
      common::errors::InvalidArgument(
          "'{attribute_name}' Attribute is expected for {op_name}. "));
  {attr_type} {attribute_name} = attributes.at("{attribute_name}").dyn_cast<paddle::dialect::ScalarAttribute>().data().to<{attr_type}>();
"""

    get_attributes_str = ""
    array_attr_str = "pir::ArrayAttribute"

    attr_names = []
    attr_types = []
    attr_build_arg_types = []
    if not mutable_attr_is_input:
        attr_names = op_info.attribute_name_list
        attr_types = op_info.attribute_type_list
        attr_build_arg_types = op_info.attribute_build_arg_type_list
    else:
        attr_names = op_info.non_mutable_attribute_name_list
        attr_types = op_info.non_mutable_attribute_type_list
        attr_build_arg_types = op_info.non_mutable_attribute_build_arg_type_list
    if attr_args_is_map:
        for idx in range(len(attr_names)):
            if attr_names[idx] not in inuse_infer_meta_args:
                continue
            attr_type = attr_build_arg_types[idx]
            attr_type = attr_type.replace("const ", "")
            attr_type = attr_type.replace("&", "")
            # if attr_build_arg_types[idx] == "const std::vector<int>&":
            #     attr_type = "std::vector<int>"

            if array_attr_str in attr_types[idx]:
                inner_type = attr_types[idx][len(array_attr_str) + 1 : -1]
                data_name = "data"
                if inner_type == "pir::StrAttribute":
                    data_name = "AsString"
                get_attributes_str += (
                    GET_ARRAY_ATTRIBUTE_FROM_MAP_TEMPLATE.format(
                        op_name=op_info.class_name,
                        attr_type=attr_type,
                        attribute_name=attr_names[idx],
                        inner_type=inner_type,
                        data_name=data_name,
                    )
                )
            elif "paddle::dialect::IntArrayAttribute" in attr_types[idx]:
                get_attributes_str += (
                    GET_INTARRAY_ATTRIBUTE_FROM_MAP_TEMPLATE.format(
                        op_name=op_info.class_name,
                        attr_type=attr_type,
                        attribute_name=attr_names[idx],
                    )
                )
            elif "paddle::dialect::ScalarAttribute" in attr_types[idx]:
                get_attributes_str += (
                    GET_SCALAR_ATTRIBUTE_FROM_MAP_TEMPLATE.format(
                        op_name=op_info.class_name,
                        attr_type=attr_type,
                        attribute_name=attr_names[idx],
                    )
                )
            elif "pir::StrAttribute" in attr_types[idx]:
                get_attributes_str += (
                    GET_STR_ATTRIBUTES_FROM_MAP_TEMPLATE.format(
                        op_name=op_info.class_name,
                        attr_type=attr_type,
                        attribute_name=attr_names[idx],
                        attr_ir_type=attr_types[idx],
                    )
                )
            else:
                get_attributes_str += GET_ATTRIBUTES_FROM_MAP_TEMPLATE.format(
                    op_name=op_info.class_name,
                    attr_type=attr_type,
                    attribute_name=attr_names[idx],
                    attr_ir_type=attr_types[idx],
                )
    return get_attributes_str


def GenDistBranch(args, op_info):
    if not args.with_distributed:
        return ""
    TEMPLATE = """
  // Auto Parallel condition
  ProcessMeshAttribute op_mesh;
  if(HasDistInput(input_values, &op_mesh)) {{
    {}
    {}
    CvtAllInputsToDist(input_values, op_mesh);
    auto ctx = pir::IrContext::Instance();
    std::vector<pir::Attribute> dist_operand_attrs, dist_result_attrs;
    """

    extra_call = ""
    for name in op_info.spmd_params:
        if name == "learning_rate":
            extra_call = "CopyLeafOpToMesh(learning_rate_, op_mesh);"
            break
    merge_input_meshes = ""
    if (
        op_info.class_name == 'CheckFiniteAndUnscale_Op'
        or op_info.class_name == 'UpdateLossScaling_Op'
    ):
        merge_input_meshes = "op_mesh = CreateGlobalMesh(input_values);"
    if op_info.class_name == 'CheckFiniteAndUnscale_Op':
        extra_call = "CopyLeafOpToMesh(scale_, op_mesh);"
    dist_branch_str = TEMPLATE.format(merge_input_meshes, extra_call)
    infer_spmd_args_list = []
    spmd_input_value_num = 0
    map_input_idx = {}
    # Prepare inputs_meta_tensor & attributes for infer spmd
    for spmd_arg_idx, name in enumerate(op_info.spmd_params):
        # is input
        if name in op_info.input_name_list:
            input_index = op_info.input_name_list.index(name)
            # is a vector<Tensor>
            if 'pir::VectorType' in op_info.input_type_list[input_index]:
                TEMPLATE = """
    std::vector<phi::distributed::DistMetaTensor> vec_dist_meta_{name};
    {name} = {name}_.type().dyn_cast<pir::VectorType>();
    for(auto& sub_ir_tensor: {name}.data()) {{
      vec_dist_meta_{name}.push_back(CvtToDistMetaTensor(sub_ir_tensor.dyn_cast<DistDenseTensorType>()));
    }}"""
                dist_branch_str += TEMPLATE.format(name=name)
                infer_spmd_args_list.append("vec_dist_meta_" + name)
            # is a Tensor
            else:
                if op_info.input_optional_list[input_index] == 'true':
                    TEMPLATE = """
    phi::distributed::DistMetaTensor dist_meta_{name};
    if({name}_) {{
      dist_meta_{name} = CvtToDistMetaTensor({name}_.type().dyn_cast<DistDenseTensorType>());
    }}"""
                    dist_branch_str += TEMPLATE.format(name=name)
                else:
                    TEMPLATE = """
    auto dist_meta_{name} = CvtToDistMetaTensor({name}_.type().dyn_cast<DistDenseTensorType>());"""
                    dist_branch_str += TEMPLATE.format(name=name)
                infer_spmd_args_list.append("dist_meta_" + name)
            spmd_input_value_num += 1
            map_input_idx[input_index] = spmd_arg_idx
        else:
            attr_index = op_info.attribute_name_list.index(name)
            param_type = op_info.attribute_gen_arg_type_list[attr_index]
            infer_spmd_args_list.append(name)
            if param_type == "phi::IntArray":
                if name in op_info.mutable_attribute_name_list:
                    attr_index = op_info.mutable_attribute_name_list.index(name)
                    attr_type = op_info.mutable_attribute_type_list[attr_index]
                    if attr_type[0] == "paddle::dialect::IntArrayAttribute":
                        infer_spmd_args_list[-1] = name + ".GetData()"
    spmd_rule_func = op_info.spmd_rule_func
    if spmd_rule_func is None:
        spmd_rule_func = "VariadicReplicatedInferSpmdDynamic"
    TEMPLATE = """
    auto spmd_info = phi::distributed::{spmd_func}({args});
    DebugInfoForInferSpmd("{op_name}", spmd_info);
    PADDLE_ENFORCE_EQ(spmd_info.first.size(), {input_size}u, common::errors::Unavailable(
        "Size of spmd_info.first for op[{op_name}]is unexpected."));
"""
    dist_branch_str += TEMPLATE.format(
        spmd_func=spmd_rule_func,
        args=', '.join(infer_spmd_args_list),
        input_size=spmd_input_value_num,
        op_name=op_info.class_name,
    )

    if spmd_input_value_num == len(op_info.input_name_list):
        TEMPLATE = """
    for(auto& arg_dist : spmd_info.first) {
      dist_operand_attrs.push_back(CvtToPirAttr(arg_dist));
    }
"""
        dist_branch_str += TEMPLATE
    else:
        for i in range(len(op_info.input_name_list)):
            if i in map_input_idx:
                spmd_idx = map_input_idx[i]
                TEMPLATE = """
    dist_operand_attrs.push_back(CvtToPirAttr(spmd_info.first[{idx}]));"""
                dist_branch_str += TEMPLATE.format(idx=spmd_idx)
            # vector<Tensor> input
            elif "pir::VectorType" in op_info.input_type_list[i]:
                TEMPLATE = """
    dist_operand_attrs.push_back(GetTensorDistAttr({name}));"""
                dist_branch_str += TEMPLATE.format(
                    name=op_info.input_name_list[i]
                )
            # Tensor input
            else:
                TEMPLATE = """
    dist_operand_attrs.push_back(GetTensorDistAttr({name}.dtype()));"""
                dist_branch_str += TEMPLATE.format(
                    name=op_info.input_name_list[i]
                )

    if len(op_info.mutable_attribute_name_list) > 0:
        TEMPLATE = """
    for(int i = {input_size}; i < {all_input_size}; ++i) {{
        dist_operand_attrs.push_back(GetTensorDistAttr(input_values[i].type()));
    }}
"""
        dist_branch_str += TEMPLATE.format(
            input_size=len(op_info.input_name_list),
            all_input_size=len(op_info.input_name_list)
            + len(op_info.mutable_attribute_name_list),
        )

    for idx, output_name in enumerate(op_info.output_name_list):
        if op_info.spmd_rule_func is None:
            TEMPLATE = """
    auto dist_attr_{name} = CreateReplicatedDistAttr({name}_type, op_mesh);
"""
            dist_branch_str += TEMPLATE.format(name=output_name)
        else:
            TEMPLATE = """
    auto dist_attr_{name} = CvtToPirAttr(spmd_info.second[{idx}]);
"""
            dist_branch_str += TEMPLATE.format(idx=idx, name=output_name)
        TEMPLATE = """
    dist_result_attrs.push_back(dist_attr_{name});
    argument_outputs.push_back(CvtToPirDistType({name}_type, dist_attr_{name}));
"""
        dist_branch_str += TEMPLATE.format(name=output_name)
    TEMPLATE = """
    attributes[kAttrOpDistAttr] = OperationDistAttribute::get(
        ctx,
        op_mesh,
        dist_operand_attrs,
        dist_result_attrs
    );
    return argument_outputs;
  }}
"""
    dist_branch_str += TEMPLATE.format()
    return dist_branch_str


def gen_infermeta_func_str(args, op_info):
    attr_args_is_map = True
    mutable_attr_is_input = (
        True if len(op_info.mutable_attribute_name_list) > 0 else False
    )
    inuse_infer_meta_args = []
    for idx in range(len(op_info.infer_meta_map['param'])):
        inuse_infer_meta_args.append(op_info.infer_meta_map['param'][idx])

    # Prepare outputs_meta_tensor for infer meta
    for idx in range(len(op_info.output_name_list)):
        if op_info.output_name_list[idx].endswith('_grad'):
            inuse_infer_meta_args.append(
                f"{op_info.output_name_list[idx][0:-5]}"
            )
        if op_info.output_name_list[idx].endswith('_grad_'):
            inuse_infer_meta_args.append(
                f"{op_info.output_name_list[idx][0:-6]}"
            )
        inuse_infer_meta_args.append(f"{op_info.output_name_list[idx]}")

    if op_info.inplace_map is not None:
        for input in op_info.inplace_map.keys():
            inuse_infer_meta_args.append(f"{op_info.inplace_map[input]}")

    spmd_params = []
    if args.with_distributed:
        if op_info.spmd_rule_func is not None:
            spmd_params = op_info.input_name_list + op_info.attribute_name_list
            if op_info.kernel_map is not None:
                spmd_params = op_info.kernel_map['param']
        else:
            spmd_params = op_info.input_name_list
    op_info.spmd_params = spmd_params

    infermeta_inputs_str = get_infermeta_inputs_str(
        op_info,
        inuse_infer_meta_args + spmd_params,
        op_info.input_name_list,
        op_info.kernel_input_type_list,
        op_info.input_optional_list,
        op_info.mutable_attribute_name_list,
        mutable_attr_is_input,
    )

    get_attributes_str = GetAttributes(
        op_info,
        mutable_attr_is_input,
        inuse_infer_meta_args + spmd_params,
        attr_args_is_map,
    )
    infermeta_outputs_str = GenBuildOutputsPart2(
        args,
        op_info,
        inuse_infer_meta_args + spmd_params,
        op_info.input_name_list,
        op_info.kernel_input_type_list,
        op_info.input_optional_list,
        op_info.mutable_attribute_name_list,
        op_info.mutable_attribute_type_list,
        op_info.output_name_list,
        op_info.kernel_output_type_list,
        op_info.output_size_list,
        op_info.output_optional_list,
        op_info.infer_meta_map,
        op_info.inplace_map,
        mutable_attr_is_input,
    )
    infermeta_func = OP_INFERMETA_IMPL_TEMPLATE_2.format(
        op_name=op_info.class_name,
        infermeta_inputs=infermeta_inputs_str,
        get_attributes_str=get_attributes_str,
        infermeta_outputs=infermeta_outputs_str,
    )

    return infermeta_func


def gen_infermeta_impl_str(args, op_info):
    return (
        OP_INFERMETA_IMPL_TEMPLATE_1.format(
            op_name=op_info.class_name,
            infer_meta_func=op_info.infer_meta_func,
        )
        + "\n"
        + gen_infermeta_func_str(args, op_info)
    )


def gen_infermeta_by_invoke_impl_str(op_info, op_info_items):
    invoke_class_name = to_pascal_case(op_info.invoke_map['func']) + "Op"
    return (
        OP_INFERMETA_IMPL_TEMPLATE_1.format(
            op_name=op_info.class_name,
            infer_meta_func=op_info_items[
                op_info.invoke_map['func']
            ].infer_meta_func,
        )
        + "\n"
        + OP_INFERMETA_IMPL_TEMPLATE_2_BY_INVOKE.format(
            op_name=op_info.class_name, invoke_class=invoke_class_name
        )
    )


def gen_op_infermeta_func(args, op_info, op_info_items):
    interface = []
    declare_str = ""
    impl_str = ""
    if op_info.infer_meta_func:
        interface = ["paddle::dialect::InferMetaInterface"]
        declare_str = OP_INFERMETA_DECL_STRING
        impl_str = gen_infermeta_impl_str(args, op_info)
    elif op_info.invoke_map and op_info.invoke_map['func'] in op_info_items:
        if op_info_items[op_info.invoke_map['func']].infer_meta_func:
            interface = ["paddle::dialect::InferMetaInterface"]
            declare_str = OP_INFERMETA_DECL_STRING
            impl_str = gen_infermeta_by_invoke_impl_str(op_info, op_info_items)

    return interface, declare_str, impl_str
