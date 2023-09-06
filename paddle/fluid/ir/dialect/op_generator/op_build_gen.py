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

# generator build function
_INFERMETA_NEED_META_CONFIG = {'SplitInferMeta'}

_PREPARE_DATA_WITH_UNKNOW_ATTRIBUTE = {'SplitOp'}

OP_BUILD_TEMPLATE = """
void {op_name}::Build({build_args}) {{
{get_attributes}
{build_mutable_attributes}
{build_inputs}
{build_attributes}
{build_outputs}
}}
"""


def GenBuildInputArgsStr(
    op_input_name_list,
    op_attribute_name_list,
    op_attribute_build_arg_type_list,
    op_attribute_default_value_list,
    op_mutable_attribute_name_list,
    op_non_mutable_attribute_name_list,
    op_non_mutable_attribute_build_arg_type_list,
    op_non_mutable_attribute_default_value_list,
    for_func_define=True,
    mutable_attr_is_input=False,
    attr_args_is_map=False,
):
    '''
    Example: ir::Builder &builder, ir::OperationArgument &argument, ir::OpResult x_, phi::DataType dtype=phi::DataType::UNDEFINED, phi::Place place={}
    '''
    # add inputs
    build_args_str = "ir::Builder &builder, ir::OperationArgument &argument"
    if len(op_input_name_list) > 0:
        for input_name in op_input_name_list:
            build_args_str += ", ir::OpResult " + input_name + "_"

    if attr_args_is_map:
        build_args_str += ", ir::AttributeMap attributes"
    else:
        if not mutable_attr_is_input:
            # add attributes
            for attr_idx in range(len(op_attribute_name_list)):
                build_args_str += (
                    ", "
                    + op_attribute_build_arg_type_list[attr_idx]
                    + " "
                    + op_attribute_name_list[attr_idx]
                )
                if for_func_define:
                    if op_attribute_default_value_list[attr_idx] is not None:
                        default_value = op_attribute_default_value_list[
                            attr_idx
                        ]
                        if (
                            op_attribute_build_arg_type_list[attr_idx]
                            != "const std::string&"
                        ):
                            if (
                                default_value[0] == "'"
                                or default_value[0] == '"'
                            ):
                                default_value = default_value[1:]
                            if (
                                default_value[-1] == "'"
                                or default_value[-1] == '"'
                            ):
                                default_value = default_value[0:-1]
                        build_args_str += "=" + default_value
        else:
            # add mutable attributes as inputs
            if len(op_mutable_attribute_name_list) > 0:
                for mutable_attr in op_mutable_attribute_name_list:
                    build_args_str += ", ir::OpResult " + mutable_attr + "_"

            # add non-mutable attributes
            for attr_idx in range(len(op_non_mutable_attribute_name_list)):
                build_args_str += (
                    ", "
                    + op_non_mutable_attribute_build_arg_type_list[attr_idx]
                    + " "
                    + op_non_mutable_attribute_name_list[attr_idx]
                )
                if for_func_define:
                    if (
                        op_non_mutable_attribute_default_value_list[attr_idx]
                        is not None
                    ):
                        default_value = (
                            op_non_mutable_attribute_default_value_list[
                                attr_idx
                            ]
                        )
                        if (
                            op_non_mutable_attribute_build_arg_type_list[
                                attr_idx
                            ]
                            != "const std::string&"
                        ):
                            if (
                                default_value[0] == "'"
                                or default_value[0] == '"'
                            ):
                                default_value = default_value[1:]
                            if (
                                default_value[-1] == "'"
                                or default_value[-1] == '"'
                            ):
                                default_value = default_value[0:-1]
                        build_args_str += "=" + default_value

    return build_args_str


mutable_attribute_phi_type_maps = {
    'int': 'phi::DataType::INT32',
    'int64_t': 'phi::DataType::INT64',
    'float': 'phi::DataType::FLOAT32',
    'std::vector<int64_t>': 'phi::DataType::INT64',
    'const std::vector<int64_t>&': 'phi::DataType::INT64',
    'bool': 'phi::DataType::BOOL',
}


def GenBuildInserFullForMutableAttribute(
    op_attribute_name_list,
    op_attribute_build_arg_type_list,
    op_mutable_attribute_name_list,
    op_mutable_attribute_type_list,
):
    build_mutable_attribute = ""
    BUILD_INTARRAY_ATTRIBUTE_TEMPLATE = """  // Generate int_array mutable attribute: {attr_name}
  paddle::dialect::FullIntArrayOp full_{attr_name}_op = builder.Build<paddle::dialect::FullIntArrayOp>({attr_name}, {phi_dtype}, phi::CPUPlace());
  ir::OpResult {attr_name}_ = full_{attr_name}_op->result(0);
    """
    BUILD_SCALAR_ATTRIBUTE_TEMPLATE = """  // Generate scalar mutable attribute: {attr_name}
  paddle::dialect::FullOp full_{attr_name}_op = builder.Build<paddle::dialect::FullOp>(std::vector<int64_t>{{1}}, {attr_name}, {phi_dtype}, phi::CPUPlace());
  ir::OpResult {attr_name}_ = full_{attr_name}_op->result(0);
    """
    for idx in range(len(op_mutable_attribute_name_list)):
        attr_name = op_mutable_attribute_name_list[idx]
        attr_type = op_mutable_attribute_type_list[idx][0]
        if attr_name in op_attribute_name_list:
            phi_dtype = mutable_attribute_phi_type_maps[
                op_attribute_build_arg_type_list[
                    op_attribute_name_list.index(attr_name)
                ]
            ]
        else:
            phi_dtype = mutable_attribute_phi_type_maps[
                op_mutable_attribute_type_list[idx][1]
            ]
        if attr_type == "paddle::dialect::IntArrayAttribute":
            build_mutable_attribute += BUILD_INTARRAY_ATTRIBUTE_TEMPLATE.format(
                attr_name=attr_name, phi_dtype=phi_dtype
            )
        else:
            build_mutable_attribute += BUILD_SCALAR_ATTRIBUTE_TEMPLATE.format(
                attr_name=attr_name, phi_dtype=phi_dtype
            )
    return build_mutable_attribute


def GenBuildInputs(op_input_name_list, op_mutable_attribute_name_list):
    BUILD_INPUT_TEMPLATE = """  std::vector<ir::OpResult> argument_inputs = {{{inputs_args}}};
  argument.AddOperands(argument_inputs.begin(), argument_inputs.end());
"""
    build_input_str = '  VLOG(4) << "Builder construction inputs";\n'
    input_name_list = op_input_name_list + op_mutable_attribute_name_list
    if len(input_name_list) > 0:
        inputs_args_str = ""
        inputs_args_str += "_, ".join(input_name_list) + "_"
        build_input_str += BUILD_INPUT_TEMPLATE.format(
            inputs_args=inputs_args_str
        )
    return build_input_str


def GenBuildAttributes(
    op_non_mutable_attribute_name_list, op_non_mutable_attribute_type_list
):
    INTARRAY_STR_TEMPLATE = """  ir::Attribute attr_{attr_name} = {op_attribute_type}::get(ir::IrContext::Instance(), phi::IntArray({attr}));
"""
    SCALAR_STR_TEMPLATE = """  ir::Attribute attr_{attr_name} = paddle::dialect::TransToIrAttribute({attr}, ir::IrContext::Instance());
"""
    STR_TEMPLATE = """  ir::Attribute attr_{attr_name} = {op_attribute_type}::get(ir::IrContext::Instance(), {attr});
"""
    ARRAY_ATTRIBUTE_TEMPLATE = """  std::vector<ir::Attribute> vec_{attr_name};
  for (size_t i = 0; i < static_cast<size_t>({attr_size}); i++) {{
    {create_attribute}
    vec_{attr_name}.push_back(attr_{attr_name});
  }}
  ir::Attribute attr_{attr_name} = ir::ArrayAttribute::get(ir::IrContext::Instance(), vec_{attr_name});
"""
    attr_str = '  VLOG(4) << "Builder construction attributes";\n'
    for idx in range(len(op_non_mutable_attribute_name_list)):
        if "ir::ArrayAttribute<" in op_non_mutable_attribute_type_list[idx]:
            inner_attribute_type = op_non_mutable_attribute_type_list[idx][
                19:-1
            ]
            if inner_attribute_type == "paddle::dialect::IntArrayAttribute":
                attr_str += ARRAY_ATTRIBUTE_TEMPLATE.format(
                    attr_name=op_non_mutable_attribute_name_list[idx],
                    attr_size=op_non_mutable_attribute_name_list[idx]
                    + ".size()",
                    create_attribute=INTARRAY_STR_TEMPLATE.format(
                        attr_name=op_non_mutable_attribute_name_list[idx],
                        op_attribute_type=inner_attribute_type,
                        attr=op_non_mutable_attribute_name_list[idx] + "[i]",
                    ),
                )
            elif inner_attribute_type == "paddle::dialect::ScalarAttribute":
                attr_str += ARRAY_ATTRIBUTE_TEMPLATE.format(
                    attr_name=op_non_mutable_attribute_name_list[idx],
                    attr_size=op_non_mutable_attribute_name_list[idx]
                    + ".size()",
                    create_attribute=SCALAR_STR_TEMPLATE.format(
                        attr_name=op_non_mutable_attribute_name_list[idx],
                        attr=op_non_mutable_attribute_name_list[idx] + "[i]",
                    ),
                )
            else:
                attr_str += ARRAY_ATTRIBUTE_TEMPLATE.format(
                    attr_name=op_non_mutable_attribute_name_list[idx],
                    attr_size=op_non_mutable_attribute_name_list[idx]
                    + ".size()",
                    create_attribute=STR_TEMPLATE.format(
                        attr_name=op_non_mutable_attribute_name_list[idx],
                        op_attribute_type=inner_attribute_type,
                        attr=op_non_mutable_attribute_name_list[idx] + "[i]",
                    ),
                )
        elif (
            op_non_mutable_attribute_type_list[idx]
            == "paddle::dialect::IntArrayAttribute"
        ):
            attr_str += INTARRAY_STR_TEMPLATE.format(
                attr_name=op_non_mutable_attribute_name_list[idx],
                op_attribute_type=op_non_mutable_attribute_type_list[idx],
                attr=op_non_mutable_attribute_name_list[idx],
            )

        elif (
            op_non_mutable_attribute_type_list[idx]
            == "paddle::dialect::ScalarAttribute"
        ):
            attr_str += SCALAR_STR_TEMPLATE.format(
                attr_name=op_non_mutable_attribute_name_list[idx],
                attr=op_non_mutable_attribute_name_list[idx],
            )
        else:
            attr_str += STR_TEMPLATE.format(
                attr_name=op_non_mutable_attribute_name_list[idx],
                op_attribute_type=op_non_mutable_attribute_type_list[idx],
                attr=op_non_mutable_attribute_name_list[idx],
            )
        attr_str += """  argument.AddAttribute("{attr_name}", attr_{attr_name});\n""".format(
            attr_name=op_non_mutable_attribute_name_list[idx]
        )

    return attr_str


def GenBuildOutputs(
    op_class_name,
    op_input_name_list,
    op_input_type_list,
    op_mutable_attribute_name_list,
    op_mutable_attribute_type_list,
    op_output_name_list,
    op_output_type_list,
    op_output_size_list,
    op_infer_meta_map,
    mutable_attr_is_input=False,
):
    build_output_str = '  VLOG(4) << "Builder construction outputs";\n'
    CREATE_INPUT_METATENSOR_TEMPLATE = """
  VLOG(4) << "Builder construction  dense_{name}";
  paddle::dialect::IrMetaTensor ir_meta_tensor_{name}(paddle::dialect::TransToPhiDataType({name}.dtype()),
                                                      {name}.dims(),
                                                      {name}.data_layout(),
                                                      {name}.lod(),
                                                      {name}.offset());
  VLOG(4) << "Builder construction  meta_{name}";
  phi::MetaTensor meta_{name}(&ir_meta_tensor_{name});
"""
    CREATE_INPUT_VEC_METATENSOR_TEMPLATE = """  std::vector<paddle::dialect::IrMetaTensor> vec_ir_meta_tensor_{name};
  for (size_t i=0; i < static_cast<size_t>({name}.size()); i++) {{
    vec_ir_meta_tensor_{name}.push_back(paddle::dialect::IrMetaTensor(paddle::dialect::TransToPhiDataType({name}[i].dyn_cast<paddle::dialect::DenseTensorType>().dtype()),
                                                                     {name}[i].dyn_cast<paddle::dialect::DenseTensorType>().dims(),
                                                                     {name}[i].dyn_cast<paddle::dialect::DenseTensorType>().data_layout(),
                                                                     {name}[i].dyn_cast<paddle::dialect::DenseTensorType>().lod(),
                                                                     {name}[i].dyn_cast<paddle::dialect::DenseTensorType>().offset()));
  }}
  std::vector<phi::MetaTensor> vec_meta_{name};
  for (size_t i=0; i < vec_ir_meta_tensor_{name}.size(); i++) {{
    vec_meta_{name}.push_back(phi::MetaTensor(&vec_ir_meta_tensor_{name}[i]));
  }}

  std::vector<const phi::MetaTensor*> meta_{name};
  for (size_t i=0; i < static_cast<size_t>(vec_meta_{name}.size()); i++) {{
    meta_{name}.push_back(&vec_meta_{name}[i]);
  }}
 """

    CREATE_INTARRAY_MUTABLE_ATTRIBUE_TEMPLATE = """  std::vector<int64_t> {name} = {name}_.owner()->dyn_cast<paddle::dialect::FullIntArrayOp>().attributes().at("value").dyn_cast<paddle::dialect::IntArrayAttribute>().data().GetData(); (void){name};\n"""
    CREATE_SCALAR_MUTABLE_ATTRIBUE_TEMPLATE = """  {dtype} {name} = {name}_.owner()->dyn_cast<paddle::dialect::FullOp>().attributes().at("value").dyn_cast<paddle::dialect::ScalarAttribute>().data().to<{dtype}>(); (void){name};\n"""

    CREATE_INTARRAY_MUTABLE_ATTRIBUE_WITH_UNKONW_DATA_TEMPLATE = """  phi::IntArray {name};
  if ({name}_.owner()->info().id() == ir::TypeId::get<paddle::dialect::FullIntArrayOp>()) {{
    {name} = std::move(phi::IntArray({name}_.owner()
                          ->dyn_cast<paddle::dialect::FullIntArrayOp>()
                          .attributes()
                          .at("value")
                          .dyn_cast<paddle::dialect::IntArrayAttribute>()
                          .data()
                          .GetData()));
  }}
  else {{
    PADDLE_ENFORCE(
        {name}_.type().isa<ir::VectorType>(),
        phi::errors::PreconditionNotMet("section Type should be VectorType."));
    size_t {name}_size = {name}_.type().dyn_cast<ir::VectorType>().size();
    {name} = std::move(phi::IntArray(std::vector<int64_t>({name}_size, -1)));
    {name}.SetFromTensor(true);
  }}\n"""

    CREATE_SCALAR_MUTABLE_ATTRIBUE_WITH_UNKONW_DATA_TEMPLATE = """  phi::Scalar {name};
  if ({name}_.owner()->info().id() == ir::TypeId::get<paddle::dialect::FullOp>()) {{
    {name} = std::move(phi::Scalar({name}_.owner()
                                  ->dyn_cast<paddle::dialect::FullOp>()
                                  .attributes()
                                  .at("value")
                                  .dyn_cast<paddle::dialect::ScalarAttribute>()
                                  .data()
                                  .to<int>()));
  }}
  else {{
    {name} = std::move(phi::Scalar(-1));
    {name}.SetFromTensor(true);
  }}\n"""

    CREATE_OUTPUT_METATENSOR_TEMPLATE = """  phi::DenseTensor dense_{name};
  phi::MetaTensor meta_{name}(&dense_{name});
"""
    CREATE_OUTPUT_VEC_METATENSOR_TEMPLATE = """  std::vector<phi::DenseTensor> vec_dense_{name}(({output_size}), phi::DenseTensor());
  std::vector<phi::MetaTensor> vec_meta_{name};
  for (size_t i=0; i < static_cast<size_t>({output_size}); i++) {{
    vec_meta_{name}.push_back(phi::MetaTensor(&vec_dense_{name}[i]));
  }}
  std::vector<phi::MetaTensor*> meta_{name};
  for (size_t i=0; i < static_cast<size_t>(vec_meta_{name}.size()); i++) {{
    meta_{name}.push_back(&vec_meta_{name}[i]);
  }}
"""
    # Prepar input type
    for idx in range(len(op_input_name_list)):
        # is a vector<Tensor>
        if 'ir::VectorType' in op_input_type_list[idx]:
            build_output_str += "  ir::VectorType {name} = {name}_.type().dyn_cast<ir::VectorType>(); (void){name};\n".format(
                name=op_input_name_list[idx]
            )
        # is a Tensor
        else:
            build_output_str += "  paddle::dialect::DenseTensorType {name} = {name}_.type().dyn_cast<paddle::dialect::DenseTensorType>(); (void){name};\n".format(
                name=op_input_name_list[idx]
            )

    # Prepare mutable attributes
    if mutable_attr_is_input:
        for idx in range(len(op_mutable_attribute_name_list)):
            attr_dtype = op_mutable_attribute_type_list[idx]
            # int_array
            if attr_dtype[0] == "paddle::dialect::IntArrayAttribute":
                if op_class_name in _PREPARE_DATA_WITH_UNKNOW_ATTRIBUTE:
                    build_output_str += CREATE_INTARRAY_MUTABLE_ATTRIBUE_WITH_UNKONW_DATA_TEMPLATE.format(
                        name=op_mutable_attribute_name_list[idx]
                    )
                else:
                    build_output_str += (
                        CREATE_INTARRAY_MUTABLE_ATTRIBUE_TEMPLATE.format(
                            name=op_mutable_attribute_name_list[idx]
                        )
                    )
            # scalar
            elif attr_dtype[0] == "paddle::dialect::ScalarAttribute":
                if op_class_name in _PREPARE_DATA_WITH_UNKNOW_ATTRIBUTE:
                    build_output_str += CREATE_SCALAR_MUTABLE_ATTRIBUE_WITH_UNKONW_DATA_TEMPLATE.format(
                        name=op_mutable_attribute_name_list[idx],
                        dtype=attr_dtype[1],
                    )
                else:
                    build_output_str += (
                        CREATE_SCALAR_MUTABLE_ATTRIBUE_TEMPLATE.format(
                            name=op_mutable_attribute_name_list[idx],
                            dtype=attr_dtype[1],
                        )
                    )
            # string
            elif attr_dtype[0] == "ir::StrAttribute":
                build_output_str += ""
            else:
                assert "mutable attribtue type is not right."
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
                    'ir::VectorType'
                    in op_input_type_list[
                        op_input_name_list.index(
                            op_infer_meta_map['param'][idx]
                        )
                    ]
                ):
                    build_output_str += (
                        CREATE_INPUT_VEC_METATENSOR_TEMPLATE.format(
                            name=op_infer_meta_map['param'][idx]
                        )
                    )
                # is a Tensor
                else:
                    build_output_str += CREATE_INPUT_METATENSOR_TEMPLATE.format(
                        name=op_infer_meta_map['param'][idx]
                    )

            infer_meta_args.append("meta_" + op_infer_meta_map['param'][idx])
        # is attribute
        else:
            infer_meta_args.append(op_infer_meta_map['param'][idx])

    # Prepare outputs_meta_tensor for infer meta
    for idx in range(len(op_output_name_list)):
        # is a vector<Tensor>
        if 'ir::VectorType' in op_output_type_list[idx]:
            build_output_str += CREATE_OUTPUT_VEC_METATENSOR_TEMPLATE.format(
                name=op_output_name_list[idx],
                output_size=op_output_size_list[idx],
            )
            infer_meta_args.append(f"meta_{op_output_name_list[idx]}")
        # is a Tensor
        else:
            build_output_str += CREATE_OUTPUT_METATENSOR_TEMPLATE.format(
                name=op_output_name_list[idx]
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
    build_output_str += "\n  std::vector<ir::Type> argument_outputs;"

    CREATE_OUTPUT_DENSE_TENSOR_TEMPLATE = """
  ir::Type {name}_dense_tensor_type = paddle::dialect::DenseTensorType::get(ir::IrContext::Instance(), paddle::dialect::TransToIrDataType(dense_{name}.dtype()), dense_{name}.dims(), dense_{name}.layout(), dense_{name}.lod(), dense_{name}.offset());
  argument_outputs.push_back({name}_dense_tensor_type);
"""
    CREATE_OUTPUT_VEC_DENSE_TENSOR_TEMPLATE = """
  std::vector<ir::Type> {name}_types;
  for (size_t i=0; i < static_cast<size_t>({output_size}); i++) {{
    {name}_types.push_back(paddle::dialect::DenseTensorType::get(ir::IrContext::Instance(), paddle::dialect::TransToIrDataType(vec_dense_{name}[i].dtype()), vec_dense_{name}[i].dims(), vec_dense_{name}[i].layout(), vec_dense_{name}[i].lod(), vec_dense_{name}[i].offset()));
  }}
  ir::Type {name}_vector_type = ir::VectorType::get(ir::IrContext::Instance(), {name}_types);
  argument_outputs.push_back({name}_vector_type);
"""
    for idx in range(len(op_output_name_list)):
        # is a vector<Tensor>
        if 'ir::VectorType' in op_output_type_list[idx]:
            build_output_str += CREATE_OUTPUT_VEC_DENSE_TENSOR_TEMPLATE.format(
                name=op_output_name_list[idx],
                output_size=op_output_size_list[idx],
            )
        # is a Tensor
        else:
            build_output_str += CREATE_OUTPUT_DENSE_TENSOR_TEMPLATE.format(
                name=op_output_name_list[idx]
            )

    build_output_str += "  argument.AddOutputs(argument_outputs.begin(), argument_outputs.end());\n"

    return build_output_str


def gen_build_func_str(
    op_class_name,
    op_input_name_list,
    op_input_type_list,
    op_attribute_name_list,
    op_attribute_type_list,
    op_attribute_build_arg_type_list,
    op_attribute_default_value_list,
    op_mutable_attribute_name_list,
    op_mutable_attribute_type_list,
    op_non_mutable_attribute_name_list,
    op_non_mutable_attribute_type_list,
    op_non_mutable_attribute_build_arg_type_list,
    op_non_mutable_attribute_default_value_list,
    op_output_name_list,
    op_output_type_list,
    op_output_size_list,
    op_infer_meta_map,
    muta_attr_is_input=False,
    attr_args_is_map=False,
):
    build_args_for_declare = ""
    build_func = ""

    build_args_for_declare = GenBuildInputArgsStr(
        op_input_name_list,
        op_attribute_name_list,
        op_attribute_build_arg_type_list,
        op_attribute_default_value_list,
        op_mutable_attribute_name_list,
        op_non_mutable_attribute_name_list,
        op_non_mutable_attribute_build_arg_type_list,
        op_non_mutable_attribute_default_value_list,
        True,
        muta_attr_is_input,
        attr_args_is_map,
    )

    build_args_for_define = GenBuildInputArgsStr(
        op_input_name_list,
        op_attribute_name_list,
        op_attribute_build_arg_type_list,
        op_attribute_default_value_list,
        op_mutable_attribute_name_list,
        op_non_mutable_attribute_name_list,
        op_non_mutable_attribute_build_arg_type_list,
        op_non_mutable_attribute_default_value_list,
        False,
        muta_attr_is_input,
        attr_args_is_map,
    )
    inset_full_for_mutable_attributes_str = ""
    if not muta_attr_is_input:
        inset_full_for_mutable_attributes_str = (
            GenBuildInserFullForMutableAttribute(
                op_attribute_name_list,
                op_attribute_build_arg_type_list,
                op_mutable_attribute_name_list,
                op_mutable_attribute_type_list,
            )
        )

    build_inputs_str = GenBuildInputs(
        op_input_name_list, op_mutable_attribute_name_list
    )
    build_attributes_str = GenBuildAttributes(
        op_non_mutable_attribute_name_list,
        op_non_mutable_attribute_type_list,
    )
    build_outputs_str = GenBuildOutputs(
        op_class_name,
        op_input_name_list,
        op_input_type_list,
        op_mutable_attribute_name_list,
        op_mutable_attribute_type_list,
        op_output_name_list,
        op_output_type_list,
        op_output_size_list,
        op_infer_meta_map,
        muta_attr_is_input,
    )

    GET_ATTRIBUTES_FROM_MAP_TEMPLATE = """
  {attr_type} {attribute_name} = attributes.at("{attribute_name}").dyn_cast<{attr_ir_type}>().data();
"""
    GET_STR_ATTRIBUTES_FROM_MAP_TEMPLATE = """
  {attr_type} {attribute_name} = attributes.at("{attribute_name}").dyn_cast<ir::StrAttribute>().AsString();
"""
    GET_ARRAY_ATTRIBUTE_FROM_MAP_TEMPLATE = """
  {attr_type} {attribute_name};
  for (size_t i = 0; i < attributes.at("{attribute_name}").dyn_cast<ir::ArrayAttribute>().size(); i++) {{
    {attribute_name}.push_back(attributes.at("{attribute_name}").dyn_cast<ir::ArrayAttribute>().at(i).dyn_cast<{inner_type}>().{data_name}());
  }}
"""
    GET_INTARRAY_ATTRIBUTE_FROM_MAP_TEMPLATE = """
  {attr_type} {attribute_name} = attributes.at("{attribute_name}").dyn_cast<paddle::dialect::IntArrayAttribute>().data().GetData();
"""
    GET_SCALAR_ATTRIBUTE_FROM_MAP_TEMPLATE = """
  {attr_type} {attribute_name} = attributes.at("{attribute_name}").dyn_cast<paddle::dialect::ScalarAttribute>().data().to<{attr_type}>();
"""

    get_attributes_str = ""
    if attr_args_is_map:
        for idx in range(len(op_attribute_name_list)):
            attr_type = op_attribute_build_arg_type_list[idx]
            attr_type = attr_type.replace("const ", "")
            attr_type = attr_type.replace("&", "")
            # if op_attribute_build_arg_type_list[idx] == "const std::vector<int>&":
            #     attr_type = "std::vector<int>"
            if "ir::ArrayAttribute" in op_attribute_type_list[idx]:
                inner_type = op_attribute_type_list[idx][19:-1]
                data_name = "data"
                if inner_type == "ir::StrAttribute":
                    data_name = "AsString"
                get_attributes_str += (
                    GET_ARRAY_ATTRIBUTE_FROM_MAP_TEMPLATE.format(
                        attr_type=attr_type,
                        attribute_name=op_attribute_name_list[idx],
                        inner_type=inner_type,
                        data_name=data_name,
                    )
                )
            elif (
                "paddle::dialect::IntArrayAttribute"
                in op_attribute_type_list[idx]
            ):
                get_attributes_str += (
                    GET_INTARRAY_ATTRIBUTE_FROM_MAP_TEMPLATE.format(
                        attr_type=attr_type,
                        attribute_name=op_attribute_name_list[idx],
                    )
                )
            elif (
                "paddle::dialect::ScalarAttribute"
                in op_attribute_type_list[idx]
            ):
                get_attributes_str += (
                    GET_SCALAR_ATTRIBUTE_FROM_MAP_TEMPLATE.format(
                        attr_type=attr_type,
                        attribute_name=op_attribute_name_list[idx],
                    )
                )
            elif "ir::StrAttribute" in op_attribute_type_list[idx]:
                get_attributes_str += (
                    GET_STR_ATTRIBUTES_FROM_MAP_TEMPLATE.format(
                        attr_type=attr_type,
                        attribute_name=op_attribute_name_list[idx],
                        attr_ir_type=op_attribute_type_list[idx],
                    )
                )
            else:
                get_attributes_str += GET_ATTRIBUTES_FROM_MAP_TEMPLATE.format(
                    attr_type=attr_type,
                    attribute_name=op_attribute_name_list[idx],
                    attr_ir_type=op_attribute_type_list[idx],
                )

    build_func = OP_BUILD_TEMPLATE.format(
        op_name=op_class_name,
        build_args=build_args_for_define,
        build_mutable_attributes=inset_full_for_mutable_attributes_str,
        get_attributes=get_attributes_str,
        build_inputs=build_inputs_str,
        build_attributes=build_attributes_str,
        build_outputs=build_outputs_str,
    )

    return (build_args_for_declare, build_func)
