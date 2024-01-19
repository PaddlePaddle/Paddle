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

from op_build_gen import (
    _INFERMETA_NEED_META_CONFIG,
    _PREPARE_DATA_WITH_VECTOR_INT64_MTTABLE_ATTRIBUTE,
)

OP_INFERMETA_TEMPLATE = """
std::vector<pir::Type> {op_name}::InferMeta(const std::vector<pir::Value>& input_values, const pir::AttributeMap& attributes) {{
{infermeta_inputs}
{get_attributes_str}
{infermeta_outputs}
  return argument_outputs;
}}
"""

CREATE_INPUT_VALUE_TEMPLATE = """
  pir::Value {input_name}_ = input_values[{index}]; (void){input_name}_;"""

ENFORCE_INPUT_NUM_TEMPLATE = """
  IR_ENFORCE(input_values.size() == {op_input_name_list_size},
      "Num of inputs is expected to be {op_input_name_list_size} but got %d.", input_values.size());
"""

OP_INFERMETA_BY_INVOKE_TEMPLATE = """
std::vector<pir::Type> {op_name}::InferMeta(const std::vector<pir::Value>& input_values, const pir::AttributeMap& attributes) {{
  return {invoke_class}::InferMeta(input_values, attributes);
}}
"""

GET_INPUT_TYPE_TEMPLATE = """
  {type} {name};
  if ({name}_.type().isa<{type}>()) {{
    {name} = {name}_.type().dyn_cast<{type}>(); (void){name};
  }} else if ({name}_.type().isa<{allocated_type}>()) {{
    {allocated_type} allocated_{name} = {name}_.type().dyn_cast<{allocated_type}>();
    {name} = {type}::get(pir::IrContext::Instance(),
                                            allocated_{name}.dtype(),
                                            allocated_{name}.dims(),
                                            allocated_{name}.data_layout(),
                                            allocated_{name}.lod(),
                                            allocated_{name}.offset());
    (void){name};
  }} else {{
    PADDLE_THROW(phi::errors::Unimplemented("Only support {type} or {allocated_type}"));
  }}
"""


def get_infermeta_inputs_str(
    inuse_infer_meta_args,
    op_input_name_list,
    op_input_type_list,
    op_input_optional_list,
    op_mutable_attribute_name_list,
    mutable_attr_is_input,
):
    op_input_name_list_size = len(op_input_name_list)
    if mutable_attr_is_input:
        op_input_name_list_size += len(op_mutable_attribute_name_list)

    infermeta_inputs_str = ENFORCE_INPUT_NUM_TEMPLATE.format(
        op_input_name_list_size=str(op_input_name_list_size),
    )

    for i in range(len(op_input_name_list)):
        if op_input_name_list[i] not in inuse_infer_meta_args:
            continue
        infermeta_inputs_str += CREATE_INPUT_VALUE_TEMPLATE.format(
            input_name=op_input_name_list[i], index=str(i)
        )

    if mutable_attr_is_input:
        # add mutable attributes as inputs
        if len(op_mutable_attribute_name_list) > 0:
            for i in range(len(op_mutable_attribute_name_list)):
                if (
                    op_mutable_attribute_name_list[i]
                    not in inuse_infer_meta_args
                ):
                    continue
                infermeta_inputs_str += CREATE_INPUT_VALUE_TEMPLATE.format(
                    input_name=op_mutable_attribute_name_list[i],
                    index=str(i + len(op_input_name_list)),
                )
    infermeta_inputs_str += "\n"

    infermeta_inputs_str += '  VLOG(4) << "Builder construction outputs";\n'
    # Prepare input type
    for idx in range(len(op_input_name_list)):
        if op_input_name_list[idx] not in inuse_infer_meta_args:
            continue
        # is a vector<Tensor>
        if 'pir::VectorType' in op_input_type_list[idx]:
            if op_input_optional_list[idx] == 'false':
                infermeta_inputs_str += "  pir::VectorType {name} = {name}_.type().dyn_cast<pir::VectorType>(); (void){name};\n".format(
                    name=op_input_name_list[idx]
                )
        # is a Tensor
        else:
            if op_input_optional_list[idx] == 'false':
                type = op_input_type_list[idx]
                allocated_type = type.replace(
                    'DenseTensorType', 'AllocatedDenseTensorType'
                ).replace("SelectedRowsType", "AllocatedSelectedRowsType")
                infermeta_inputs_str += GET_INPUT_TYPE_TEMPLATE.format(
                    type=type,
                    name=op_input_name_list[idx],
                    allocated_type=allocated_type,
                )

    return infermeta_inputs_str


def GenBuildOutputsPart2(
    op_class_name,
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

    CREATE_OPTIONAL_INPUT_METATENSOR_TEMPLATE = """
  paddle::dialect::IrMetaTensor meta_{name};
  paddle::dialect::IrTensor ir_tensor_{name};


  if ({name}_.impl() != nullptr) {{
    VLOG(4) << "Builder construction  dense_{name}";
    {type} {name};
    if ({name}_.type().isa<{type}>()) {{
      {name} = {name}_.type().dyn_cast<{type}>();
    }} else if ({name}_.type().isa<{allocated_type}>()) {{
      {allocated_type} allocated_{name} = {name}_.type().dyn_cast<{allocated_type}>();
      {name} = {type}::get(pir::IrContext::Instance(),
                            allocated_{name}.dtype(),
                            allocated_{name}.dims(),
                            allocated_{name}.data_layout(),
                            allocated_{name}.lod(),
                            allocated_{name}.offset());
    }} else {{
      PADDLE_THROW(phi::errors::Unimplemented("Only support {type} or {allocated_type}"));
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

    CREATE_INPUT_VEC_METATENSOR_TEMPLATE = """  std::vector<paddle::dialect::IrTensor> vec_ir_tensor_{name};
  for (size_t i=0; i < static_cast<size_t>({name}.size()); i++) {{
    if({name}[i].isa<paddle::dialect::DenseTensorType>()) {{
        auto {name}_type = {name}[i].dyn_cast<paddle::dialect::DenseTensorType>();
        vec_ir_tensor_{name}.push_back(paddle::dialect::IrTensor(paddle::dialect::TransToPhiDataType({name}_type.dtype()),
                                                                    {name}_type.dims(),
                                                                    {name}_type.data_layout(),
                                                                    {name}_type.lod(),
                                                                    {name}_type.offset()));
    }} else if({name}[i].isa<paddle::dialect::AllocatedDenseTensorType>()){{
        auto {name}_type = {name}[i].dyn_cast<paddle::dialect::AllocatedDenseTensorType>();
        vec_ir_tensor_{name}.push_back(paddle::dialect::IrTensor(paddle::dialect::TransToPhiDataType({name}_type.dtype()),
                                                                    {name}_type.dims(),
                                                                    {name}_type.data_layout(),
                                                                    {name}_type.lod(),
                                                                    {name}_type.offset()));
    }} else {{
        PADDLE_THROW(phi::errors::Unimplemented("Only support DenseTensorType or AllocatedDenseTensorType"));
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

    CREATE_OPTIONAL_INPUT_VEC_METATENSOR_TEMPLATE = """  std::vector<paddle::dialect::IrTensor> vec_ir_tensor_{name};
  if ({name}_.impl() != nullptr) {{
    pir::VectorType {name} = {name}_.type().dyn_cast<pir::VectorType>();
    for (size_t i=0; i < static_cast<size_t>({name}.size()); i++) {{
        if({name}[i].isa<paddle::dialect::DenseTensorType>()) {{
          auto {name}_type = {name}[i].dyn_cast<paddle::dialect::DenseTensorType>();
          vec_ir_tensor_{name}.push_back(paddle::dialect::IrTensor(paddle::dialect::TransToPhiDataType({name}_type.dtype()),
                                                                        {name}_type.dims(),
                                                                        {name}_type.data_layout(),
                                                                        {name}_type.lod(),
                                                                        {name}_type.offset()));
        }} else if({name}[i].isa<paddle::dialect::AllocatedDenseTensorType>()){{
          auto {name}_type = {name}[i].dyn_cast<paddle::dialect::AllocatedDenseTensorType>();
          vec_ir_tensor_{name}.push_back(paddle::dialect::IrTensor(paddle::dialect::TransToPhiDataType({name}_type.dtype()),
                                                                        {name}_type.dims(),
                                                                        {name}_type.data_layout(),
                                                                        {name}_type.lod(),
                                                                        {name}_type.offset()));
        }} else {{
            PADDLE_THROW(phi::errors::Unimplemented("Only support DenseTensorType or AllocatedDenseTensorType"));
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

    CREATE_INTARRAY_MUTABLE_ATTRIBUE_WITH_UNKONW_DATA_TEMPLATE = """  phi::IntArray {name};
  if ({name}_.dyn_cast<pir::OpResult>() && {name}_.dyn_cast<pir::OpResult>().owner()->isa<paddle::dialect::FullIntArrayOp>()) {{
    {name} = std::move(phi::IntArray(paddle::dialect::GetInt64Vector(
                          {name}_.dyn_cast<pir::OpResult>().owner()
                          ->dyn_cast<paddle::dialect::FullIntArrayOp>()
                          .attribute("value"))));
  }} else if ({name}_.type().isa<pir::VectorType>()) {{
    size_t {name}_size = {name}_.type().dyn_cast<pir::VectorType>().size();
    {name} = std::move(phi::IntArray(std::vector<int64_t>({name}_size, -1)));
    {name}.SetFromTensor(true);
  }} else if ({name}_.type().isa<paddle::dialect::DenseTensorType>()) {{
    common::DDim {name}_dim = {name}_.type().dyn_cast<paddle::dialect::DenseTensorType>().dims();
    size_t {name}_size = common::product({name}_dim);
    if (common::contain_unknown_dim({name}_dim)) {{
      {name}_size = 1;
    }}
    {name} = std::move(phi::IntArray(std::vector<int64_t>({name}_size, -1)));
    {name}.SetFromTensor(true);
  }} else if ({name}_.type().isa<paddle::dialect::AllocatedDenseTensorType>()) {{
    common::DDim {name}_dim = {name}_.type().dyn_cast<paddle::dialect::AllocatedDenseTensorType>().dims();
    size_t {name}_size = common::product({name}_dim);
    if (common::contain_unknown_dim({name}_dim)) {{
      {name}_size = 1;
    }}
    {name} = std::move(phi::IntArray(std::vector<int64_t>({name}_size, -1)));
    {name}.SetFromTensor(true);
  }} else {{
    PADDLE_THROW(phi::errors::Unimplemented("Only support VectorType or DenseTensorType or AllocatedDenseTensorType"));
  }}\n"""

    CREATE_VECTOR_INT_MUTABLE_ATTRIBUE_WITH_UNKONW_DATA_TEMPLATE = """  std::vector<int64_t> {name};
  if ({name}_.dyn_cast<pir::OpResult>() && {name}_.dyn_cast<pir::OpResult>().owner()->isa<paddle::dialect::FullIntArrayOp>()) {{
    {name} = paddle::dialect::GetInt64Vector(
                    {name}_.dyn_cast<pir::OpResult>().owner()
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
  }} else if ({name}_.type().isa<paddle::dialect::AllocatedDenseTensorType>()) {{
    common::DDim {name}_dim = {name}_.type().dyn_cast<paddle::dialect::AllocatedDenseTensorType>().dims();
    size_t {name}_size = common::product({name}_dim);
    if (common::contain_unknown_dim({name}_dim)) {{
      {name}_size = 1;
    }}
    {name} = std::vector<int64_t>({name}_size, -1);
  }} else {{
    PADDLE_THROW(phi::errors::Unimplemented("Only support VectorType or DenseTensorType or AllocatedDenseTensorType"));
  }}\n"""

    CREATE_SCALAR_MUTABLE_ATTRIBUE_WITH_UNKONW_DATA_TEMPLATE = """  phi::Scalar {name};
  if ({name}_.dyn_cast<pir::OpResult>() && {name}_.dyn_cast<pir::OpResult>().owner()->isa<paddle::dialect::FullOp>()) {{
    {name} = std::move(phi::Scalar({name}_.dyn_cast<pir::OpResult>().owner()
                                  ->dyn_cast<paddle::dialect::FullOp>()
                                  .attribute("value")
                                  .dyn_cast<paddle::dialect::ScalarAttribute>()
                                  .data()
                                  .to<int>()));
  }}
  else {{
    {name} = std::move(phi::Scalar(-1));
    {name}.SetFromTensor(true);
  }}\n"""

    CREATE_OUTPUT_METATENSOR_TEMPLATE = """  paddle::dialect::IrTensor dense_{name};
  paddle::dialect::IrMetaTensor meta_{name}(&dense_{name});
"""
    CREATE_OUTPUT_METASELETEROWS_TEMPLATE = """  paddle::dialect::IrSelectedRows dense_{name};
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
            if op_mutable_attribute_name_list[idx] not in inuse_infer_meta_args:
                continue
            attr_dtype = op_mutable_attribute_type_list[idx]
            # int_array
            if attr_dtype[0] == "paddle::dialect::IntArrayAttribute":
                if (
                    op_class_name
                    in _PREPARE_DATA_WITH_VECTOR_INT64_MTTABLE_ATTRIBUTE
                ):
                    build_output_str += CREATE_VECTOR_INT_MUTABLE_ATTRIBUE_WITH_UNKONW_DATA_TEMPLATE.format(
                        name=op_mutable_attribute_name_list[idx]
                    )
                else:
                    build_output_str += CREATE_INTARRAY_MUTABLE_ATTRIBUE_WITH_UNKONW_DATA_TEMPLATE.format(
                        name=op_mutable_attribute_name_list[idx]
                    )
            # scalar
            elif attr_dtype[0] == "paddle::dialect::ScalarAttribute":
                build_output_str += CREATE_SCALAR_MUTABLE_ATTRIBUE_WITH_UNKONW_DATA_TEMPLATE.format(
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
                        type = op_input_type_list[idx]
                        allocated_type = type.replace(
                            'DenseTensorType', 'AllocatedDenseTensorType'
                        ).replace(
                            "SelectedRowsType", "AllocatedSelectedRowsType"
                        )
                        build_output_str += (
                            CREATE_OPTIONAL_INPUT_METATENSOR_TEMPLATE.format(
                                name=op_infer_meta_map['param'][idx],
                                type=op_input_type_list[idx],
                                allocated_type=allocated_type,
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
            infer_meta_args.append(op_infer_meta_map['param'][idx])

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
    CREATE_INFER_META_FUNC_WITH_METACINFIG_TEMPLATE = """
  phi::{func}({args}, phi::MetaConfig(false, false));
"""
    if op_infer_meta_map['func'] in _INFERMETA_NEED_META_CONFIG:
        build_output_str += (
            CREATE_INFER_META_FUNC_WITH_METACINFIG_TEMPLATE.format(
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
  pir::Type {name}_dense_tensor_type = {type}::get(pir::IrContext::Instance(), paddle::dialect::TransToIrDataType(dense_{name}.dtype()), dense_{name}.dims(), dense_{name}.layout(), dense_{name}.lod(), dense_{name}.offset());
  argument_outputs.push_back({name}_dense_tensor_type);
"""

    CREATE_OUTPUT_INPLACE_OPTIONAL_DENSE_TENSOR_TEMPLATE = """
  if ({input_name}_.impl() != nullptr) {{
    pir::Type {output_name}_dense_tensor_type = {type}::get(pir::IrContext::Instance(), paddle::dialect::TransToIrDataType(dense_{output_name}.dtype()), dense_{output_name}.dims(), dense_{output_name}.layout(), dense_{output_name}.lod(), dense_{output_name}.offset());
    argument_outputs.push_back({output_name}_dense_tensor_type);
  }} else {{
    pir::Type {output_name}_type;
    argument_outputs.push_back({output_name}_type);
  }}

"""

    CREATE_OUTPUT_VEC_DENSE_TENSOR_TEMPLATE = """
  std::vector<pir::Type> {name}_types;
  for (size_t i=0; i < static_cast<size_t>({output_size}); i++) {{
    {name}_types.push_back(paddle::dialect::DenseTensorType::get(pir::IrContext::Instance(), paddle::dialect::TransToIrDataType(vec_dense_{name}[i].dtype()), vec_dense_{name}[i].dims(), vec_dense_{name}[i].layout(), vec_dense_{name}[i].lod(), vec_dense_{name}[i].offset()));
  }}
  pir::Type {name}_vector_type = pir::VectorType::get(pir::IrContext::Instance(), {name}_types);
  argument_outputs.push_back({name}_vector_type);
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
                        output_name=output_name,
                        type=op_output_type_list[idx],
                    )
                )
            else:
                build_output_str += CREATE_OUTPUT_DENSE_TENSOR_TEMPLATE.format(
                    type=op_output_type_list[idx], name=output_name
                )
    return build_output_str


def GetAttributes(
    op_class_name,
    muta_attr_is_input,
    inuse_infer_meta_args,
    op_attribute_name_list,
    op_attribute_type_list,
    op_attribute_build_arg_type_list,
    op_non_mutable_attribute_name_list,
    op_non_mutable_attribute_type_list,
    op_non_mutable_attribute_build_arg_type_list,
    attr_args_is_map,
):
    GET_ATTRIBUTES_FROM_MAP_TEMPLATE = """
  IR_ENFORCE(
      attributes.find("{attribute_name}") != attributes.end(),
          "'{attribute_name}' Attribute is expected for {op_name}. ");
  {attr_type} {attribute_name} = attributes.at("{attribute_name}").dyn_cast<{attr_ir_type}>().data();
"""
    GET_STR_ATTRIBUTES_FROM_MAP_TEMPLATE = """
  IR_ENFORCE(
      attributes.find("{attribute_name}") != attributes.end(),
          "'{attribute_name}' Attribute is expected for {op_name}. ");
  {attr_type} {attribute_name} = attributes.at("{attribute_name}").dyn_cast<pir::StrAttribute>().AsString();
"""
    GET_ARRAY_ATTRIBUTE_FROM_MAP_TEMPLATE = """
  IR_ENFORCE(
      attributes.find("{attribute_name}") != attributes.end(),
          "'{attribute_name}' Attribute is expected for {op_name}. ");
  {attr_type} {attribute_name};
  for (size_t i = 0; i < attributes.at("{attribute_name}").dyn_cast<pir::ArrayAttribute>().size(); i++) {{
    {attribute_name}.push_back(attributes.at("{attribute_name}").dyn_cast<pir::ArrayAttribute>().at(i).dyn_cast<{inner_type}>().{data_name}());
  }}
"""
    GET_INTARRAY_ATTRIBUTE_FROM_MAP_TEMPLATE = """
  IR_ENFORCE(
      attributes.find("{attribute_name}") != attributes.end(),
          "'{attribute_name}' Attribute is expected for {op_name}. ");
  {attr_type} {attribute_name} = attributes.at("{attribute_name}").dyn_cast<paddle::dialect::IntArrayAttribute>().data().GetData();
"""
    GET_SCALAR_ATTRIBUTE_FROM_MAP_TEMPLATE = """
  IR_ENFORCE(
      attributes.find("{attribute_name}") != attributes.end(),
          "'{attribute_name}' Attribute is expected for {op_name}. ");
  {attr_type} {attribute_name} = attributes.at("{attribute_name}").dyn_cast<paddle::dialect::ScalarAttribute>().data().to<{attr_type}>();
"""

    get_attributes_str = ""
    array_attr_str = "pir::ArrayAttribute"

    attr_names = []
    attr_types = []
    attr_build_arg_types = []
    if not muta_attr_is_input:
        attr_names = op_attribute_name_list
        attr_types = op_attribute_type_list
        attr_build_arg_types = op_attribute_build_arg_type_list
    else:
        attr_names = op_non_mutable_attribute_name_list
        attr_types = op_non_mutable_attribute_type_list
        attr_build_arg_types = op_non_mutable_attribute_build_arg_type_list
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
                        op_name=op_class_name,
                        attr_type=attr_type,
                        attribute_name=attr_names[idx],
                        inner_type=inner_type,
                        data_name=data_name,
                    )
                )
            elif "paddle::dialect::IntArrayAttribute" in attr_types[idx]:
                get_attributes_str += (
                    GET_INTARRAY_ATTRIBUTE_FROM_MAP_TEMPLATE.format(
                        op_name=op_class_name,
                        attr_type=attr_type,
                        attribute_name=attr_names[idx],
                    )
                )
            elif "paddle::dialect::ScalarAttribute" in attr_types[idx]:
                get_attributes_str += (
                    GET_SCALAR_ATTRIBUTE_FROM_MAP_TEMPLATE.format(
                        op_name=op_class_name,
                        attr_type=attr_type,
                        attribute_name=attr_names[idx],
                    )
                )
            elif "pir::StrAttribute" in attr_types[idx]:
                get_attributes_str += (
                    GET_STR_ATTRIBUTES_FROM_MAP_TEMPLATE.format(
                        op_name=op_class_name,
                        attr_type=attr_type,
                        attribute_name=attr_names[idx],
                        attr_ir_type=attr_types[idx],
                    )
                )
            else:
                get_attributes_str += GET_ATTRIBUTES_FROM_MAP_TEMPLATE.format(
                    op_name=op_class_name,
                    attr_type=attr_type,
                    attribute_name=attr_names[idx],
                    attr_ir_type=attr_types[idx],
                )
    return get_attributes_str


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
    inuse_infer_meta_args = []
    for idx in range(len(op_infer_meta_map['param'])):
        inuse_infer_meta_args.append(op_infer_meta_map['param'][idx])

    # Prepare outputs_meta_tensor for infer meta
    for idx in range(len(op_output_name_list)):
        if op_output_name_list[idx].endswith('_grad'):
            inuse_infer_meta_args.append(f"{op_output_name_list[idx][0:-5]}")
        if op_output_name_list[idx].endswith('_grad_'):
            inuse_infer_meta_args.append(f"{op_output_name_list[idx][0:-6]}")
        inuse_infer_meta_args.append(f"{op_output_name_list[idx]}")

    infermeta_inputs_str = get_infermeta_inputs_str(
        inuse_infer_meta_args,
        op_input_name_list,
        op_input_type_list,
        op_input_optional_list,
        op_mutable_attribute_name_list,
        muta_attr_is_input,
    )

    get_attributes_str = GetAttributes(
        op_class_name,
        muta_attr_is_input,
        inuse_infer_meta_args,
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
        muta_attr_is_input,
    )

    infermeta_func = OP_INFERMETA_TEMPLATE.format(
        op_name=op_class_name,
        infermeta_inputs=infermeta_inputs_str,
        get_attributes_str=get_attributes_str,
        infermeta_outputs=infermeta_outputs_str,
    )

    return infermeta_func


def gen_infermeta_by_invoke_func_str(op_class_name, invoke_class_name):
    return OP_INFERMETA_BY_INVOKE_TEMPLATE.format(
        op_name=op_class_name, invoke_class=invoke_class_name
    )
