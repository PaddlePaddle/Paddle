// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#ifdef GET_OP_LIST
#undef GET_OP_LIST
paddle::onednn::dialect::ExpandOp
#else

#include "paddle/fluid/pir/dialect/operator/ir/manual_onednn_op.h"
#include "paddle/fluid/pir/dialect/kernel/ir/kernel_type.h"
#include "paddle/fluid/pir/dialect/operator/ir/ir_meta_tensor.h"
#include "paddle/fluid/pir/dialect/operator/ir/ir_selected_rows.h"
#include "paddle/fluid/pir/dialect/operator/ir/ir_tensor.h"
#include "paddle/fluid/pir/dialect/operator/ir/onednn_op.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_attribute.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_type.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/fluid/primitive/vjp_interface/vjp.h"
#include "paddle/phi/api/lib/data_type_set.h"
#include "paddle/phi/api/lib/utils/allocator.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/infermeta/backward.h"
#include "paddle/phi/infermeta/binary.h"
#include "paddle/phi/infermeta/fusion.h"
#include "paddle/phi/infermeta/multiary.h"
#include "paddle/phi/infermeta/nullary.h"
#include "paddle/phi/infermeta/ternary.h"
#include "paddle/phi/infermeta/unary.h"
#include "paddle/pir/include/core/builtin_attribute.h"
#include "paddle/pir/include/core/builtin_op.h"
#include "paddle/pir/include/core/builtin_type.h"
#include "paddle/pir/include/core/ir_context.h"
#include "paddle/pir/include/core/op_base.h"

namespace paddle {
namespace onednn {
namespace dialect {

const char* ExpandOp::attributes_name[1] = {"mkldnn_data_type"};  // NOLINT

OpInfoTuple ExpandOp::GetOpInfo() {
  std::vector<paddle::dialect::OpInputInfo> inputs = {
      paddle::dialect::OpInputInfo(
          "x", "paddle::dialect::DenseTensorType", false, false, false, true),
      paddle::dialect::OpInputInfo("shape",
                                   "paddle::dialect::IntArrayAttribute",
                                   false,
                                   false,
                                   true,
                                   false)};
  std::vector<paddle::dialect::OpAttributeInfo> attributes = {
      paddle::dialect::OpAttributeInfo(
          "mkldnn_data_type", "pir::StrAttribute", "")};
  std::vector<paddle::dialect::OpOutputInfo> outputs = {
      paddle::dialect::OpOutputInfo(
          "out", "paddle::dialect::DenseTensorType", false, false)};
  pir::AttributeMap extra_attr_default_value;
  pir::Attribute attr_mkldnn_data_type =
      pir::StrAttribute::get(pir::IrContext::Instance(), "float32");
  extra_attr_default_value["mkldnn_data_type"] = attr_mkldnn_data_type;

  paddle::dialect::OpRunTimeInfo run_time_info =
      paddle::dialect::OpRunTimeInfo("ExpandInferMeta",
                                     {"x", "shape"},
                                     "expand",
                                     {"x", "shape"},
                                     {"x"},
                                     {},
                                     {},
                                     {},
                                     {"mkldnn_data_type"},
                                     {},
                                     extra_attr_default_value,
                                     {},
                                     false,
                                     false);
  return std::make_tuple(inputs, attributes, outputs, run_time_info, "expand");
}

void ExpandOp::Build(pir::Builder& builder,
                     pir::OperationArgument& argument,
                     pir::Value x_,
                     const std::vector<int64_t>& shape,
                     const std::string& mkldnn_data_type) {
  VLOG(4) << "Start build ExpandOp";

  // Generate int_array mutable attribute: shape
  paddle::dialect::FullIntArrayOp full_shape_op =
      builder.Build<paddle::dialect::FullIntArrayOp>(
          shape, phi::DataType::INT64, phi::CPUPlace());
  pir::Value shape_ = full_shape_op->result(0);

  VLOG(4) << "Builder construction inputs";
  std::vector<pir::Value> argument_inputs = {x_, shape_};
  argument.AddInputs(argument_inputs);

  VLOG(4) << "Builder construction attributes";
  pir::AttributeMap argument_attributes = {};
  pir::Attribute attr_mkldnn_data_type =
      pir::StrAttribute::get(pir::IrContext::Instance(), mkldnn_data_type);
  argument.AddAttribute("mkldnn_data_type", attr_mkldnn_data_type);
  argument_attributes.insert({"mkldnn_data_type", attr_mkldnn_data_type});

  std::vector<pir::Type> argument_outputs =
      ExpandOp::InferMeta(argument_inputs, &argument_attributes);
  argument.AddOutputs(argument_outputs.begin(), argument_outputs.end());
  ::pir::PassStopGradientsDefaultly(argument);
}

void ExpandOp::Build(pir::Builder& builder,
                     pir::OperationArgument& argument,
                     pir::Value x_,
                     pir::AttributeMap attributes) {
  VLOG(4) << "Start build ExpandOp";

  PADDLE_ENFORCE_NE(attributes.find("shape"),
                    attributes.end(),
                    common::errors::InvalidArgument(
                        "'shape' Attribute is expected for ExpandOp. "));
  std::vector<int64_t> shape =
      attributes.at("shape")
          .dyn_cast<paddle::dialect::IntArrayAttribute>()
          .data()
          .GetData();

  PADDLE_ENFORCE_NE(
      attributes.find("mkldnn_data_type"),
      attributes.end(),
      common::errors::InvalidArgument(
          "'mkldnn_data_type' Attribute is expected for ExpandOp. "));
  std::string mkldnn_data_type = attributes.at("mkldnn_data_type")
                                     .dyn_cast<pir::StrAttribute>()
                                     .AsString();

  // Generate int_array mutable attribute: shape
  paddle::dialect::FullIntArrayOp full_shape_op =
      builder.Build<paddle::dialect::FullIntArrayOp>(
          shape, phi::DataType::INT64, phi::CPUPlace());
  pir::Value shape_ = full_shape_op->result(0);

  VLOG(4) << "Builder construction inputs";
  std::vector<pir::Value> argument_inputs = {x_, shape_};
  argument.AddInputs(argument_inputs);

  VLOG(4) << "Builder construction attributes";
  pir::AttributeMap argument_attributes = {};
  pir::Attribute attr_mkldnn_data_type =
      pir::StrAttribute::get(pir::IrContext::Instance(), mkldnn_data_type);
  argument.AddAttribute("mkldnn_data_type", attr_mkldnn_data_type);
  argument_attributes.insert({"mkldnn_data_type", attr_mkldnn_data_type});

  std::vector<pir::Type> argument_outputs =
      ExpandOp::InferMeta(argument_inputs, &argument_attributes);
  argument.AddOutputs(argument_outputs.begin(), argument_outputs.end());
  ::pir::PassStopGradientsDefaultly(argument);
}

void ExpandOp::Build(pir::Builder& builder,
                     pir::OperationArgument& argument,
                     pir::Value x_,
                     pir::Value shape_,
                     const std::string& mkldnn_data_type) {
  VLOG(4) << "Start build ExpandOp";

  VLOG(4) << "Builder construction inputs";
  std::vector<pir::Value> argument_inputs = {x_, shape_};
  argument.AddInputs(argument_inputs);

  VLOG(4) << "Builder construction attributes";
  pir::AttributeMap argument_attributes = {};
  pir::Attribute attr_mkldnn_data_type =
      pir::StrAttribute::get(pir::IrContext::Instance(), mkldnn_data_type);
  argument.AddAttribute("mkldnn_data_type", attr_mkldnn_data_type);
  argument_attributes.insert({"mkldnn_data_type", attr_mkldnn_data_type});

  std::vector<pir::Type> argument_outputs =
      ExpandOp::InferMeta(argument_inputs, &argument_attributes);
  argument.AddOutputs(argument_outputs.begin(), argument_outputs.end());
  ::pir::PassStopGradientsDefaultly(argument);
}

void ExpandOp::VerifySig() {
  VLOG(4) << "Start Verifying inputs, outputs and attributes for: ExpandOp.";
  VLOG(4) << "Verifying inputs:";
  {
    auto input_size = num_operands();
    PADDLE_ENFORCE_EQ(
        input_size,
        2u,
        common::errors::InvalidArgument(
            "The size %d of inputs must be equal to 2.", input_size));
    PADDLE_ENFORCE_EQ((*this)
                          ->operand_source(0)
                          .type()
                          .isa<paddle::dialect::DenseTensorType>(),
                      true,
                      common::errors::InvalidArgument(
                          "Type validation failed for the 0th input, got %s.",
                          (*this)->operand_source(0).type()));
    if (auto vec_type =
            (*this)->operand_source(1).type().dyn_cast<pir::VectorType>()) {
      for (size_t i = 0; i < vec_type.size(); ++i) {
        PADDLE_ENFORCE_EQ(
            vec_type[i].isa<paddle::dialect::DenseTensorType>(),
            true,
            common::errors::InvalidArgument(
                "Type validation failed for the 1th input, got %s.",
                (*this)->operand_source(1).type()));
      }
    } else {
      PADDLE_ENFORCE_EQ((*this)
                            ->operand_source(1)
                            .type()
                            .isa<paddle::dialect::DenseTensorType>(),
                        true,
                        common::errors::InvalidArgument(
                            "Type validation failed for the 1th input, got %s.",
                            (*this)->operand_source(1).type()));
    }
  }
  VLOG(4) << "Verifying attributes:";
  {
    auto& attributes = this->attributes();
    PADDLE_ENFORCE_GT(
        attributes.count("mkldnn_data_type"),
        0,
        common::errors::InvalidArgument("mkldnn_data_type does not exist."));
    PADDLE_ENFORCE_EQ(
        attributes.at("mkldnn_data_type").isa<pir::StrAttribute>(),
        true,
        common::errors::InvalidArgument(
            "Type of attribute: mkldnn_data_type is not pir::StrAttribute."));
  }
  VLOG(4) << "Verifying outputs:";
  {
    auto output_size = num_results();
    PADDLE_ENFORCE_EQ(
        output_size,
        1u,
        common::errors::InvalidArgument(
            "The size %d of outputs must be equal to 1.", output_size));
    PADDLE_ENFORCE_EQ(
        (*this)->result(0).type().isa<paddle::dialect::DenseTensorType>(),
        true,
        common::errors::InvalidArgument(
            "Type validation failed for the 0th output."));
  }
  VLOG(4) << "End Verifying for: ExpandOp.";
}

void ExpandOp::InferMeta(phi::InferMetaContext* infer_meta) {
  auto fn = PD_INFER_META(phi::ExpandInferMeta);
  fn(infer_meta);
}

std::vector<pir::Type> ExpandOp::InferMeta(
    const std::vector<pir::Value>& input_values,
    pir::AttributeMap* p_attributes) {
  PADDLE_ENFORCE_NOT_NULL(
      p_attributes,
      common::errors::Fatal(
          "AttributeMap pointer in InferMeta function is nullptr."));
  PADDLE_ENFORCE_EQ(input_values.size(),
                    2,
                    common::errors::InvalidArgument(
                        "Num of inputs is expected to be 2 but got %d.",
                        input_values.size()));

  pir::Value x_ = input_values[0];
  pir::Value shape_ = input_values[1];
  VLOG(4) << "Builder construction outputs";

  paddle::dialect::DenseTensorType x;
  if (x_.type().isa<paddle::dialect::DenseTensorType>()) {
    x = x_.type().dyn_cast<paddle::dialect::DenseTensorType>();
  } else {
    PADDLE_THROW(common::errors::Unimplemented(
        "Only support paddle::dialect::DenseTensorType or "
        "paddle::dialect::AllocatedDenseTensorType"));
  }

  phi::IntArray shape;
  if (shape_.defining_op()->isa<paddle::dialect::FullIntArrayOp>()) {
    shape = phi::IntArray(paddle::dialect::GetInt64Vector(
        shape_.defining_op()
            ->dyn_cast<paddle::dialect::FullIntArrayOp>()
            .attribute("value")));
  } else if (shape_.type().isa<pir::VectorType>()) {
    size_t shape_size = shape_.type().dyn_cast<pir::VectorType>().size();
    // In ExpandInferMeta use -2 to represent the element in expand_shape is a
    // var.
    shape = phi::IntArray(std::vector<int64_t>(shape_size, -2));
    shape.SetFromTensor(true);
  } else if (shape_.type().isa<paddle::dialect::DenseTensorType>()) {
    size_t shape_size = common::product(
        shape_.type().dyn_cast<paddle::dialect::DenseTensorType>().dims());
    // In ExpandInferMeta use -2 to represent the element in expand_shape is a
    // var.
    shape = phi::IntArray(std::vector<int64_t>(shape_size, -2));
    shape.SetFromTensor(true);
  } else {
    PADDLE_THROW(common::errors::Unimplemented(
        "Only support VectorType or DenseTensorType"));
  }

  VLOG(4) << "Builder construction  dense_x";
  paddle::dialect::IrTensor ir_tensor_x(
      paddle::dialect::TransToPhiDataType(x.dtype()),
      x.dims(),
      x.data_layout(),
      x.lod(),
      x.offset());
  VLOG(4) << "Builder construction  meta_x";
  paddle::dialect::IrMetaTensor meta_x(&ir_tensor_x);
  paddle::dialect::IrTensor dense_out;
  paddle::dialect::IrMetaTensor meta_out(&dense_out);

  phi::ExpandInferMeta(meta_x, shape, &meta_out);

  std::vector<pir::Type> argument_outputs;
  pir::Type out_dense_tensor_type = paddle::dialect::DenseTensorType::get(
      pir::IrContext::Instance(),
      paddle::dialect::TransToIrDataType(dense_out.dtype()),
      dense_out.dims(),
      dense_out.layout(),
      dense_out.lod(),
      dense_out.offset());
  argument_outputs.push_back(out_dense_tensor_type);

  return argument_outputs;
}

phi::DataType ExpandOp::GetKernelTypeForVar(
    const std::string& var_name,
    const phi::DataType& tensor_dtype,
    const phi::DataType& expected_kernel_dtype) {
  VLOG(4) << "Get KernelType for Var of op: ExpandOp";

  return expected_kernel_dtype;
}

}  // namespace dialect
}  // namespace onednn
}  // namespace paddle

IR_DEFINE_EXPLICIT_TYPE_ID(paddle::onednn::dialect::ExpandOp)
#endif
