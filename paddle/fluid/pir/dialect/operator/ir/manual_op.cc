// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
paddle::dialect::AddNOp, paddle::dialect::AddN_Op, paddle::dialect::AddNArrayOp,
    paddle::dialect::FusedGemmEpilogueOp, paddle::dialect::AssignOut_Op,
    paddle::dialect::FusedGemmEpilogueGradOp, paddle::dialect::SplitGradOp,
    paddle::dialect::ExpandOp, paddle::dialect::CreateArrayOp,
    paddle::dialect::CreateArrayLikeOp, paddle::dialect::ArrayLengthOp,
    paddle::dialect::ArrayReadOp, paddle::dialect::ArrayWrite_Op,
    paddle::dialect::SliceArrayOp, paddle::dialect::SliceArrayDenseOp,
    paddle::dialect::AssignArrayOp, paddle::dialect::AssignArray_Op,
    paddle::dialect::ArrayToTensorOp, paddle::dialect::TensorToArrayOp,
    paddle::dialect::IncrementOp, paddle::dialect::Increment_Op,
    paddle::dialect::ShapeBroadcastOp, paddle::dialect::MemcpyD2hMultiIoOp,
    paddle::dialect::ArrayPopOp
#else

#include "paddle/fluid/pir/dialect/operator/ir/manual_op.h"
#include "paddle/fluid/pir/dialect/kernel/ir/kernel_type.h"
#include "paddle/fluid/pir/dialect/operator/ir/ir_meta_tensor.h"
#include "paddle/fluid/pir/dialect/operator/ir/ir_selected_rows.h"
#include "paddle/fluid/pir/dialect/operator/ir/ir_tensor.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_attribute.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_type.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/fluid/primitive/rule/vjp/vjp.h"
#include "paddle/phi/api/lib/data_type_set.h"
#include "paddle/phi/api/lib/utils/allocator.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/infermeta/backward.h"
#include "paddle/phi/infermeta/binary.h"
#include "paddle/phi/infermeta/fusion.h"
#include "paddle/phi/infermeta/multiary.h"
#include "paddle/phi/infermeta/nullary.h"
#include "paddle/phi/infermeta/unary.h"
#include "paddle/pir/include/core/builtin_attribute.h"
#include "paddle/pir/include/core/builtin_op.h"
#include "paddle/pir/include/core/builtin_type.h"
#include "paddle/pir/include/core/ir_context.h"

namespace paddle {
namespace dialect {

OpInfoTuple AddNOp::GetOpInfo() {
  std::vector<paddle::dialect::OpInputInfo> inputs = {
      OpInputInfo("inputs",
                  "pir::VectorType<paddle::dialect::DenseTensorType>",
                  false,
                  false,
                  false,
                  true)};
  std::vector<paddle::dialect::OpAttributeInfo> attributes = {};
  std::vector<paddle::dialect::OpOutputInfo> outputs = {
      OpOutputInfo("out", "paddle::dialect::DenseTensorType", false, false)};
  paddle::dialect::OpRunTimeInfo run_time_info = OpRunTimeInfo(
      "AddNInferMeta", {"inputs"}, "add_n", {"inputs"}, {}, {}, {}, {});

  return std::make_tuple(inputs, attributes, outputs, run_time_info, "add_n");
}

void AddNOp::VerifySig() {
  VLOG(4) << "Start Verifying inputs, outputs and attributes for: AddNOp.";
  VLOG(4) << "Verifying inputs:";
  {
    auto input_size = num_operands();
    PADDLE_ENFORCE_EQ(
        input_size,
        1u,
        phi::errors::PreconditionNotMet(
            "The size %d of inputs must be equal to 1.", input_size));
    if (auto vec_type =
            (*this)->operand(0).type().dyn_cast<pir::VectorType>()) {
      for (size_t i = 0; i < vec_type.size(); ++i) {
        PADDLE_ENFORCE(vec_type[i].isa<paddle::dialect::DenseTensorType>() ||
                           vec_type[i].isa<paddle::dialect::SelectedRowsType>(),
                       phi::errors::PreconditionNotMet(
                           "Type validation failed for the 0th input."));
      }
    } else {
      PADDLE_ENFORCE(
          (*this)->operand(0).type().isa<paddle::dialect::DenseTensorType>() ||
              (*this)
                  ->operand(0)
                  .type()
                  .isa<paddle::dialect::SelectedRowsType>(),
          phi::errors::PreconditionNotMet(
              "Type validation failed for the 0th input."));
    }
  }
  VLOG(4) << "Verifying attributes:";
  {
    // Attributes num is 0, not need to check attributes type.
  }
  VLOG(4) << "Verifying outputs:";
  {
    auto output_size = num_results();
    PADDLE_ENFORCE_EQ(
        output_size,
        1u,
        phi::errors::PreconditionNotMet(
            "The size %d of outputs must be equal to 1.", output_size));
    PADDLE_ENFORCE(
        (*this)->result(0).type().isa<paddle::dialect::DenseTensorType>() ||
            (*this)->result(0).type().isa<paddle::dialect::SelectedRowsType>(),
        phi::errors::PreconditionNotMet(
            "Type validation failed for the 0th output."));
  }
  VLOG(4) << "End Verifying for: AddNOp.";
}

void AddNOp::Build(pir::Builder &builder,             // NOLINT
                   pir::OperationArgument &argument,  // NOLINT
                   pir::Value inputs) {
  VLOG(4) << "Start build AddNOp";

  VLOG(4) << "Builder construction inputs";
  std::vector<pir::Value> argument_inputs = {inputs};
  argument.AddInput(inputs);

  VLOG(4) << "Builder construction attributes";
  pir::AttributeMap argument_attributes = {};
  std::vector<pir::Type> argument_outputs =
      AddNOp::InferMeta(argument_inputs, &argument_attributes);

  argument.AddOutputs(argument_outputs.begin(), argument_outputs.end());
  ::pir::PassStopGradientsDefaultly(argument);
}

void AddNOp::InferMeta(phi::InferMetaContext *infer_meta) {
  auto fn = PD_INFER_META(phi::AddNInferMeta);
  fn(infer_meta);
}

std::vector<pir::Type> AddNOp::InferMeta(
    const std::vector<pir::Value> &input_values,
    pir::AttributeMap *p_attributes) {
  VLOG(4) << "Start infermeta AddNOp";
  IR_ENFORCE(input_values.size() == 1,
             "Num of inputs is expected to be 1 but got %d.",
             input_values.size());
  pir::Value inputs_ = input_values[0];

  VLOG(4) << "Builder construction outputs";
  pir::VectorType x = inputs_.type().dyn_cast<pir::VectorType>();

  std::vector<paddle::dialect::IrTensor> vec_dense_x;
  for (size_t i = 0; i < x.size(); i++) {
    if (x[i].isa<paddle::dialect::DenseTensorType>()) {
      vec_dense_x.push_back(paddle::dialect::IrTensor(
          TransToPhiDataType(
              x[i].dyn_cast<paddle::dialect::DenseTensorType>().dtype()),
          x[i].dyn_cast<paddle::dialect::DenseTensorType>().dims(),
          x[i].dyn_cast<paddle::dialect::DenseTensorType>().data_layout(),
          x[i].dyn_cast<paddle::dialect::DenseTensorType>().lod(),
          x[i].dyn_cast<paddle::dialect::DenseTensorType>().offset()));
    } else {
      PADDLE_THROW(phi::errors::Unimplemented(
          "Only support paddle::dialect::DenseTensorType or "
          "paddle::dialect::AllocatedDenseTensorType"));
    }
  }
  std::vector<paddle::dialect::IrMetaTensor> vec_meta_x;
  for (size_t i = 0; i < vec_dense_x.size(); i++) {
    vec_meta_x.push_back(paddle::dialect::IrMetaTensor(&vec_dense_x[i]));
  }

  std::vector<const phi::MetaTensor *> meta_x;
  for (size_t i = 0; i < static_cast<size_t>(vec_meta_x.size()); i++) {
    meta_x.push_back(&vec_meta_x[i]);
  }

  paddle::dialect::IrTensor dense_out;
  paddle::dialect::IrMetaTensor meta_out(&dense_out);

  phi::AddNInferMeta(meta_x, &meta_out, phi::MetaConfig(false, false));

  std::vector<pir::Type> argument_outputs;
  pir::Type out_dense_tensor_type = paddle::dialect::DenseTensorType::get(
      pir::IrContext::Instance(),
      TransToIrDataType(dense_out.dtype()),
      dense_out.dims(),
      dense_out.layout(),
      dense_out.lod(),
      dense_out.offset());
  argument_outputs.push_back(out_dense_tensor_type);
  return argument_outputs;
}

OpInfoTuple AddN_Op::GetOpInfo() {
  std::vector<paddle::dialect::OpInputInfo> inputs = {
      paddle::dialect::OpInputInfo(
          "inputs",
          "pir::VectorType<paddle::dialect::DenseTensorType>",
          false,
          false,
          false,
          true)};
  std::vector<paddle::dialect::OpAttributeInfo> attributes = {};
  std::vector<paddle::dialect::OpOutputInfo> outputs = {
      paddle::dialect::OpOutputInfo(
          "out", "paddle::dialect::DenseTensorType", false, false)};
  paddle::dialect::OpRunTimeInfo run_time_info = paddle::dialect::OpRunTimeInfo(
      "AddNInferMeta", {"inputs"}, "add_n", {"inputs"}, {}, {}, {}, {});
  return std::make_tuple(inputs, attributes, outputs, run_time_info, "add_n_");
}

void AddN_Op::Build(pir::Builder &builder,
                    pir::OperationArgument &argument,
                    pir::Value inputs_) {
  VLOG(4) << "Start build AddN_Op";

  VLOG(4) << "Builder construction inputs";
  std::vector<pir::Value> argument_inputs = {inputs_};
  argument.AddInput(inputs_);

  VLOG(4) << "Builder construction attributes";
  pir::AttributeMap argument_attributes = {};
  std::vector<pir::Type> argument_outputs =
      AddN_Op::InferMeta(argument_inputs, &argument_attributes);

  argument.AddOutputs(argument_outputs.begin(), argument_outputs.end());
}

void AddN_Op::VerifySig() {
  VLOG(4) << "Start Verifying inputs, outputs and attributes for: AddN_Op.";
  VLOG(4) << "Verifying inputs:";
  {
    auto input_size = num_operands();
    PADDLE_ENFORCE_EQ(
        input_size,
        1u,
        phi::errors::PreconditionNotMet(
            "The size %d of inputs must be equal to 1.", input_size));
    if (auto vec_type =
            (*this)->operand_source(0).type().dyn_cast<pir::VectorType>()) {
      for (size_t i = 0; i < vec_type.size(); ++i) {
        PADDLE_ENFORCE(vec_type[i].isa<paddle::dialect::DenseTensorType>() ||
                           vec_type[i].isa<paddle::dialect::SelectedRowsType>(),
                       phi::errors::PreconditionNotMet(
                           "Type validation failed for the 0th input."));
      }
    } else {
      PADDLE_ENFORCE((*this)->operand_source(0)
                             .type()
                             .isa<paddle::dialect::DenseTensorType>() ||
                         (*this)
                             ->operand_source(0)
                             .type()
                             .isa<paddle::dialect::SelectedRowsType>(),
                     phi::errors::PreconditionNotMet(
                         "Type validation failed for the 0th input."));
    }
  }
  VLOG(4) << "Verifying attributes:";
  {
    // Attributes num is 0, not need to check attributes type.
  }
  VLOG(4) << "Verifying outputs:";
  {
    auto output_size = num_results();
    PADDLE_ENFORCE_EQ(
        output_size,
        1u,
        phi::errors::PreconditionNotMet(
            "The size %d of outputs must be equal to 1.", output_size));
    PADDLE_ENFORCE(
        (*this)->result(0).type().isa<paddle::dialect::DenseTensorType>() ||
            (*this)->result(0).type().isa<paddle::dialect::SelectedRowsType>(),
        phi::errors::PreconditionNotMet(
            "Type validation failed for the 0th output."));
  }
  VLOG(4) << "End Verifying for: AddN_Op.";
}

void AddN_Op::InferMeta(phi::InferMetaContext *infer_meta) {
  auto fn = PD_INFER_META(phi::AddNInferMeta);
  fn(infer_meta);
}

std::vector<pir::Type> AddN_Op::InferMeta(
    const std::vector<pir::Value> &input_values,
    pir::AttributeMap *p_attributes) {
  VLOG(4) << "Start infermeta AddN_Op";
  IR_ENFORCE(input_values.size() == 1,
             "Num of inputs is expected to be 1 but got %d.",
             input_values.size());
  pir::Value inputs_ = input_values[0];

  VLOG(4) << "Builder construction outputs";
  pir::VectorType inputs = inputs_.type().dyn_cast<pir::VectorType>();
  std::vector<paddle::dialect::IrTensor> vec_dense_inputs;
  for (size_t i = 0; i < static_cast<size_t>(inputs.size()); i++) {
    if (inputs[i].isa<paddle::dialect::DenseTensorType>()) {
      vec_dense_inputs.push_back(paddle::dialect::IrTensor(
          paddle::dialect::TransToPhiDataType(
              inputs[i].dyn_cast<paddle::dialect::DenseTensorType>().dtype()),
          inputs[i].dyn_cast<paddle::dialect::DenseTensorType>().dims(),
          inputs[i].dyn_cast<paddle::dialect::DenseTensorType>().data_layout(),
          inputs[i].dyn_cast<paddle::dialect::DenseTensorType>().lod(),
          inputs[i].dyn_cast<paddle::dialect::DenseTensorType>().offset()));
    } else {
      PADDLE_THROW(phi::errors::Unimplemented(
          "Only support paddle::dialect::DenseTensorType or "
          "paddle::dialect::AllocatedDenseTensorType"));
    }
  }

  std::vector<paddle::dialect::IrMetaTensor> vec_meta_inputs;
  for (size_t i = 0; i < vec_dense_inputs.size(); i++) {
    vec_meta_inputs.push_back(
        paddle::dialect::IrMetaTensor(&vec_dense_inputs[i]));
  }

  std::vector<const phi::MetaTensor *> meta_inputs;
  for (size_t i = 0; i < static_cast<size_t>(vec_meta_inputs.size()); i++) {
    meta_inputs.push_back(&vec_meta_inputs[i]);
  }
  paddle::dialect::IrTensor dense_out;
  paddle::dialect::IrMetaTensor meta_out(&dense_out);

  phi::AddNInferMeta(meta_inputs, &meta_out, phi::MetaConfig(false, false));

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

OpInfoTuple AddNArrayOp::GetOpInfo() {
  std::vector<paddle::dialect::OpInputInfo> inputs = {
      OpInputInfo("inputs",
                  "pir::VectorType<paddle::dialect::DenseTensorArrayType>",
                  false,
                  false,
                  false,
                  true)};
  std::vector<paddle::dialect::OpAttributeInfo> attributes = {};
  std::vector<paddle::dialect::OpOutputInfo> outputs = {OpOutputInfo(
      "out", "paddle::dialect::DenseTensorArrayType", false, false)};
  paddle::dialect::OpRunTimeInfo run_time_info =
      OpRunTimeInfo("AddNTensorArrayInferMeta",
                    {"inputs"},
                    "add_n_array",
                    {"inputs"},
                    {},
                    {},
                    {},
                    {});

  return std::make_tuple(
      inputs, attributes, outputs, run_time_info, "add_n_array");
}

void AddNArrayOp::VerifySig() {
  VLOG(4) << "Start Verifying inputs, outputs and attributes for: AddNArrayOp.";
  VLOG(4) << "Verifying inputs:";
  {
    auto input_size = num_operands();
    PADDLE_ENFORCE_EQ(
        input_size,
        1u,
        phi::errors::PreconditionNotMet(
            "The size %d of inputs must be equal to 1.", input_size));
    if (auto vec_type =
            (*this)->operand(0).type().dyn_cast<pir::VectorType>()) {
      for (size_t i = 0; i < vec_type.size(); ++i) {
        PADDLE_ENFORCE(vec_type[i].isa<paddle::dialect::DenseTensorArrayType>(),
                       phi::errors::PreconditionNotMet(
                           "Type validation failed for the 0th input."));
      }
    } else {
      PADDLE_ENFORCE((*this)
                         ->operand(0)
                         .type()
                         .isa<paddle::dialect::DenseTensorArrayType>(),
                     phi::errors::PreconditionNotMet(
                         "Type validation failed for the 0th input."));
    }
  }
  VLOG(4) << "Verifying attributes:";
  {
    // Attributes num is 0, not need to check attributes type.
  }
  VLOG(4) << "Verifying outputs:";
  {
    auto output_size = num_results();
    PADDLE_ENFORCE_EQ(
        output_size,
        1u,
        phi::errors::PreconditionNotMet(
            "The size %d of outputs must be equal to 1.", output_size));
    PADDLE_ENFORCE(
        (*this)->result(0).type().isa<paddle::dialect::DenseTensorArrayType>(),
        phi::errors::PreconditionNotMet(
            "Type validation failed for the 0th output."));
  }
  VLOG(4) << "End Verifying for: AddNArrayOp.";
}

void AddNArrayOp::Build(pir::Builder &builder,             // NOLINT
                        pir::OperationArgument &argument,  // NOLINT
                        pir::Value inputs_) {
  VLOG(4) << "Start build AddNArrayOp";

  VLOG(4) << "Builder construction inputs";
  std::vector<pir::Value> argument_inputs = {inputs_};
  argument.AddInput(inputs_);

  VLOG(4) << "Builder construction attributes";
  pir::AttributeMap argument_attributes = {};
  std::vector<pir::Type> argument_outputs =
      AddNArrayOp::InferMeta(argument_inputs, &argument_attributes);

  argument.AddOutputs(argument_outputs.begin(), argument_outputs.end());
  argument.AddAttributes(argument_attributes);
  ::pir::PassStopGradientsDefaultly(argument);
}

void AddNArrayOp::InferMeta(phi::InferMetaContext *infer_meta) {
  auto fn = PD_INFER_META(phi::AddNTensorArrayInferMeta);
  fn(infer_meta);
}

std::vector<pir::Type> AddNArrayOp::InferMeta(
    const std::vector<pir::Value> &input_values,
    pir::AttributeMap *p_attributes) {
  VLOG(4) << "Start infermeta AddNArrayOp";
  IR_ENFORCE(input_values.size() == 1,
             "Num of inputs is expected to be 1 but got %d.",
             input_values.size());
  pir::Value inputs_ = input_values[0];
  VLOG(4) << "Builder construction outputs";
  pir::VectorType inputs = inputs_.type().dyn_cast<pir::VectorType>();

  std::vector<paddle::dialect::IrTensor> vec_dense_inputs;
  for (size_t i = 0; i < inputs.size(); i++) {
    if (inputs[i].isa<paddle::dialect::DenseTensorArrayType>()) {
      vec_dense_inputs.push_back(paddle::dialect::IrTensor(
          TransToPhiDataType(
              inputs[i]
                  .dyn_cast<paddle::dialect::DenseTensorArrayType>()
                  .dtype()),
          inputs[i].dyn_cast<paddle::dialect::DenseTensorArrayType>().dims(),
          inputs[i]
              .dyn_cast<paddle::dialect::DenseTensorArrayType>()
              .data_layout(),
          {}));
    } else {
      PADDLE_THROW(phi::errors::Unimplemented(
          "Only support paddle::dialect::DenseTensorArrayType or "
          "paddle::dialect::AllocatedDenseTensorArrayType"));
    }
  }

  std::vector<paddle::dialect::IrMetaTensor> vec_meta_inputs;
  for (size_t i = 0; i < vec_dense_inputs.size(); i++) {
    vec_meta_inputs.push_back(
        paddle::dialect::IrMetaTensor(&vec_dense_inputs[i]));
  }

  std::vector<const phi::MetaTensor *> meta_inputs;
  for (size_t i = 0; i < static_cast<size_t>(vec_meta_inputs.size()); i++) {
    meta_inputs.push_back(&vec_meta_inputs[i]);
  }

  paddle::dialect::IrTensor dense_out;
  paddle::dialect::IrMetaTensor meta_out(&dense_out);

  phi::AddNTensorArrayInferMeta(
      meta_inputs, &meta_out, phi::MetaConfig(false, false));
  std::vector<pir::Type> argument_outputs;
  pir::Type out_dense_tensor_type = paddle::dialect::DenseTensorArrayType::get(
      pir::IrContext::Instance(),
      TransToIrDataType(dense_out.dtype()),
      dense_out.dims(),
      dense_out.layout());

  argument_outputs.push_back(out_dense_tensor_type);
  return argument_outputs;
}

const char *FusedGemmEpilogueOp::attributes_name[3] = {  // NOLINT
    "trans_x",
    "trans_y",
    "activation"};

OpInfoTuple FusedGemmEpilogueOp::GetOpInfo() {
  std::vector<paddle::dialect::OpInputInfo> inputs = {
      paddle::dialect::OpInputInfo(
          "x", "paddle::dialect::DenseTensorType", false, false, false, false),
      paddle::dialect::OpInputInfo(
          "y", "paddle::dialect::DenseTensorType", false, false, false, false),
      paddle::dialect::OpInputInfo("bias",
                                   "paddle::dialect::DenseTensorType",
                                   false,
                                   false,
                                   false,
                                   false)};
  std::vector<paddle::dialect::OpAttributeInfo> attributes = {
      paddle::dialect::OpAttributeInfo("trans_x", "pir::BoolAttribute", ""),
      paddle::dialect::OpAttributeInfo("trans_y", "pir::BoolAttribute", ""),
      paddle::dialect::OpAttributeInfo("activation", "pir::StrAttribute", "")};
  std::vector<paddle::dialect::OpOutputInfo> outputs = {
      paddle::dialect::OpOutputInfo(
          "out", "paddle::dialect::DenseTensorType", false, false),
      paddle::dialect::OpOutputInfo(
          "reserve_space", "paddle::dialect::DenseTensorType", true, false)};
  paddle::dialect::OpRunTimeInfo run_time_info(
      "FusedGemmEpilogueInferMeta",
      {"x", "y", "bias", "trans_x", "trans_y", "activation"},
      {"fused_gemm_epilogue"},
      {"x", "y", "bias", "trans_x", "trans_y", "activation"},
      {},
      {},
      {},
      {});

  return std::make_tuple(
      inputs, attributes, outputs, run_time_info, "fused_gemm_epilogue");
}

void FusedGemmEpilogueOp::Build(pir::Builder &builder,
                                pir::OperationArgument &argument,
                                pir::Value x_,
                                pir::Value y_,
                                pir::Value bias_,
                                pir::AttributeMap attributes) {
  VLOG(4) << "Start build FusedGemmEpilogueOp";

  PADDLE_ENFORCE(
      attributes.find("trans_x") != attributes.end(),
      phi::errors::NotFound(
          "'trans_x' Attribute is expected for FusedGemmEpilogueOp"));
  bool trans_x = attributes.at("trans_x").dyn_cast<pir::BoolAttribute>().data();

  PADDLE_ENFORCE(
      attributes.find("trans_y") != attributes.end(),
      phi::errors::NotFound(
          "'trans_y' Attribute is expected for FusedGemmEpilogueOp"));
  bool trans_y = attributes.at("trans_y").dyn_cast<pir::BoolAttribute>().data();

  PADDLE_ENFORCE(
      attributes.find("activation") != attributes.end(),
      phi::errors::NotFound(
          "'activation' Attribute is expected for FusedGemmEpilogueOp"));
  std::string activation =
      attributes.at("activation").dyn_cast<pir::StrAttribute>().AsString();

  VLOG(4) << "Builder construction inputs";
  std::vector<pir::Value> argument_inputs = {x_, y_, bias_};
  argument.AddInputs({x_, y_, bias_});

  VLOG(4) << "Builder construction attributes";
  pir::AttributeMap argument_attributes = {};
  pir::Attribute attr_trans_x =
      pir::BoolAttribute::get(pir::IrContext::Instance(), trans_x);
  argument.AddAttribute("trans_x", attr_trans_x);
  argument_attributes.insert({"trans_x", attr_trans_x});
  pir::Attribute attr_trans_y =
      pir::BoolAttribute::get(pir::IrContext::Instance(), trans_y);
  argument.AddAttribute("trans_y", attr_trans_y);
  argument_attributes.insert({"trans_y", attr_trans_y});
  pir::Attribute attr_activation =
      pir::StrAttribute::get(pir::IrContext::Instance(), activation);
  argument.AddAttribute("activation", attr_activation);
  argument_attributes.insert({"activation", attr_activation});
  std::vector<pir::Type> argument_outputs =
      FusedGemmEpilogueOp::InferMeta(argument_inputs, &argument_attributes);

  argument.AddOutputs(argument_outputs.begin(), argument_outputs.end());
}

void FusedGemmEpilogueOp::VerifySig() {
  VLOG(4) << "Start Verifying inputs, outputs and attributes for: "
             "FusedGemmEpilogueOp.";
  VLOG(4) << "Verifying inputs:";
  {
    auto input_size = num_operands();
    PADDLE_ENFORCE_EQ(
        input_size,
        3u,
        phi::errors::PreconditionNotMet(
            "The size %d of inputs must be equal to 3.", input_size));
    PADDLE_ENFORCE((*this)
                       ->operand_source(0)
                       .type()
                       .isa<paddle::dialect::DenseTensorType>(),
                   phi::errors::PreconditionNotMet(
                       "Type validation failed for the 0th input."));
    PADDLE_ENFORCE((*this)
                       ->operand_source(1)
                       .type()
                       .isa<paddle::dialect::DenseTensorType>(),
                   phi::errors::PreconditionNotMet(
                       "Type validation failed for the 1th input."));
    PADDLE_ENFORCE((*this)
                       ->operand_source(2)
                       .type()
                       .isa<paddle::dialect::DenseTensorType>(),
                   phi::errors::PreconditionNotMet(
                       "Type validation failed for the 2th input."));
  }
  VLOG(4) << "Verifying attributes:";
  {
    auto &attributes = this->attributes();
    PADDLE_ENFORCE(attributes.count("trans_x") > 0 &&
                       attributes.at("trans_x").isa<pir::BoolAttribute>(),
                   phi::errors::PreconditionNotMet(
                       "Type of attribute: trans_x is not right."));
    PADDLE_ENFORCE(attributes.count("trans_y") > 0 &&
                       attributes.at("trans_y").isa<pir::BoolAttribute>(),
                   phi::errors::PreconditionNotMet(
                       "Type of attribute: trans_y is not right."));
    PADDLE_ENFORCE(attributes.count("activation") > 0 &&
                       attributes.at("activation").isa<pir::StrAttribute>(),
                   phi::errors::PreconditionNotMet(
                       "Type of attribute: activation is not right."));
  }
  VLOG(4) << "Verifying outputs:";
  {
    auto output_size = num_results();
    PADDLE_ENFORCE_EQ(
        output_size,
        2u,
        phi::errors::PreconditionNotMet(
            "The size %d of outputs must be equal to 2.", output_size));
    PADDLE_ENFORCE(
        (*this)->result(0).type().isa<paddle::dialect::DenseTensorType>(),
        phi::errors::PreconditionNotMet(
            "Type validation failed for the 0th output."));
    if (auto output_1_type = (*this)->result(1).type()) {
      PADDLE_ENFORCE(output_1_type.isa<paddle::dialect::DenseTensorType>(),
                     phi::errors::PreconditionNotMet(
                         "Type validation failed for the 1th output."));
    }
  }
  VLOG(4) << "End Verifying for: FusedGemmEpilogueOp.";
}

void FusedGemmEpilogueOp::InferMeta(phi::InferMetaContext *infer_meta) {
  auto fn = PD_INFER_META(phi::FusedGemmEpilogueInferMeta);
  fn(infer_meta);
}

std::vector<pir::Type> FusedGemmEpilogueOp::InferMeta(
    const std::vector<pir::Value> &input_values,
    pir::AttributeMap *p_attributes) {
  PADDLE_ENFORCE_NOT_NULL(
      p_attributes,
      common::errors::Fatal(
          "AttrtibueMap pointer in InferMeta function is nullptr."));
  auto &attributes = *p_attributes;
  VLOG(4) << "Start infermeta FusedGemmEpilogueOp";
  IR_ENFORCE(input_values.size() == 3,
             "Num of inputs is expected to be 3 but got %d.",
             input_values.size());
  pir::Value x_ = input_values[0];
  pir::Value y_ = input_values[1];
  pir::Value bias_ = input_values[2];

  PADDLE_ENFORCE(
      attributes.find("trans_x") != attributes.end(),
      phi::errors::NotFound(
          "'trans_x' Attribute is expected for FusedGemmEpilogueOp"));
  bool trans_x = attributes.at("trans_x").dyn_cast<pir::BoolAttribute>().data();

  PADDLE_ENFORCE(
      attributes.find("trans_y") != attributes.end(),
      phi::errors::NotFound(
          "'trans_y' Attribute is expected for FusedGemmEpilogueOp"));
  bool trans_y = attributes.at("trans_y").dyn_cast<pir::BoolAttribute>().data();

  PADDLE_ENFORCE(
      attributes.find("activation") != attributes.end(),
      phi::errors::NotFound(
          "'activation' Attribute is expected for FusedGemmEpilogueOp"));
  std::string activation =
      attributes.at("activation").dyn_cast<pir::StrAttribute>().AsString();

  VLOG(4) << "Builder construction outputs";
  paddle::dialect::DenseTensorType x;
  if (x_.type().isa<paddle::dialect::DenseTensorType>()) {
    x = x_.type().dyn_cast<paddle::dialect::DenseTensorType>();
  } else {
    PADDLE_THROW(phi::errors::Unimplemented(
        "Only support paddle::dialect::DenseTensorType or "
        "paddle::dialect::AllocatedDenseTensorType"));
  }

  paddle::dialect::DenseTensorType y;
  if (y_.type().isa<paddle::dialect::DenseTensorType>()) {
    y = y_.type().dyn_cast<paddle::dialect::DenseTensorType>();
  } else {
    PADDLE_THROW(phi::errors::Unimplemented(
        "Only support paddle::dialect::DenseTensorType or "
        "paddle::dialect::AllocatedDenseTensorType"));
  }

  paddle::dialect::DenseTensorType bias;
  if (bias_.type().isa<paddle::dialect::DenseTensorType>()) {
    bias = bias_.type().dyn_cast<paddle::dialect::DenseTensorType>();
  } else {
    PADDLE_THROW(phi::errors::Unimplemented(
        "Only support paddle::dialect::DenseTensorType or "
        "paddle::dialect::AllocatedDenseTensorType"));
  }

  VLOG(4) << "Builder construction  dense_x";
  paddle::dialect::IrTensor dense_x(
      paddle::dialect::TransToPhiDataType(x.dtype()),
      x.dims(),
      x.data_layout(),
      x.lod(),
      x.offset());
  VLOG(4) << "Builder construction  meta_x";
  paddle::dialect::IrMetaTensor meta_x(&dense_x);

  VLOG(4) << "Builder construction  dense_y";
  paddle::dialect::IrTensor dense_y(
      paddle::dialect::TransToPhiDataType(y.dtype()),
      y.dims(),
      y.data_layout(),
      y.lod(),
      y.offset());
  VLOG(4) << "Builder construction  meta_y";
  paddle::dialect::IrMetaTensor meta_y(&dense_y);

  VLOG(4) << "Builder construction  dense_bias";
  paddle::dialect::IrTensor dense_bias(
      paddle::dialect::TransToPhiDataType(bias.dtype()),
      bias.dims(),
      bias.data_layout(),
      bias.lod(),
      bias.offset());
  VLOG(4) << "Builder construction  meta_bias";
  paddle::dialect::IrMetaTensor meta_bias(&dense_bias);
  paddle::dialect::IrTensor dense_out;
  paddle::dialect::IrMetaTensor meta_out(&dense_out);
  paddle::dialect::IrTensor dense_reserve_space;
  paddle::dialect::IrMetaTensor meta_reserve_space(&dense_reserve_space);

  phi::FusedGemmEpilogueInferMeta(
      meta_x,
      meta_y,
      meta_bias,
      trans_x,
      trans_y,
      activation,
      &meta_out,
      activation == "none" ? nullptr : &meta_reserve_space);

  std::vector<pir::Type> argument_outputs;
  pir::Type out_dense_tensor_type = paddle::dialect::DenseTensorType::get(
      pir::IrContext::Instance(),
      paddle::dialect::TransToIrDataType(dense_out.dtype()),
      dense_out.dims(),
      dense_out.layout(),
      dense_out.lod(),
      dense_out.offset());
  argument_outputs.push_back(out_dense_tensor_type);

  pir::Type reserve_space_dense_tensor_type =
      activation == "none"
          ? pir::Type()
          : paddle::dialect::DenseTensorType::get(
                pir::IrContext::Instance(),
                paddle::dialect::TransToIrDataType(dense_reserve_space.dtype()),
                dense_reserve_space.dims(),
                dense_reserve_space.layout(),
                dense_reserve_space.lod(),
                dense_reserve_space.offset());
  argument_outputs.push_back(reserve_space_dense_tensor_type);
  return argument_outputs;
}

const char *FusedGemmEpilogueGradOp::attributes_name[3] = {  // NOLINT
    "trans_x",
    "trans_y",
    "activation_grad"};

OpInfoTuple FusedGemmEpilogueGradOp::GetOpInfo() {
  std::vector<paddle::dialect::OpInputInfo> inputs = {
      paddle::dialect::OpInputInfo(
          "x", "paddle::dialect::DenseTensorType", false, false, false, false),
      paddle::dialect::OpInputInfo(
          "y", "paddle::dialect::DenseTensorType", false, false, false, false),
      paddle::dialect::OpInputInfo("reserve_space",
                                   "paddle::dialect::DenseTensorType",
                                   true,
                                   false,
                                   false,
                                   false),
      paddle::dialect::OpInputInfo("out_grad",
                                   "paddle::dialect::DenseTensorType",
                                   false,
                                   false,
                                   false,
                                   false)};
  std::vector<paddle::dialect::OpAttributeInfo> attributes = {
      paddle::dialect::OpAttributeInfo("trans_x", "pir::BoolAttribute", ""),
      paddle::dialect::OpAttributeInfo("trans_y", "pir::BoolAttribute", ""),
      paddle::dialect::OpAttributeInfo(
          "activation_grad", "pir::StrAttribute", "")};
  std::vector<paddle::dialect::OpOutputInfo> outputs = {
      paddle::dialect::OpOutputInfo(
          "x_grad", "paddle::dialect::DenseTensorType", false, false),
      paddle::dialect::OpOutputInfo(
          "y_grad", "paddle::dialect::DenseTensorType", false, false),
      paddle::dialect::OpOutputInfo(
          "bias_grad", "paddle::dialect::DenseTensorType", false, false)};
  paddle::dialect::OpRunTimeInfo run_time_info("FusedGemmEpilogueGradInferMeta",
                                               {"x",
                                                "y",
                                                "reserve_space",
                                                "out_grad",
                                                "trans_x",
                                                "trans_y",
                                                "activation_grad"},
                                               {"fused_gemm_epilogue_grad"},
                                               {"x",
                                                "y",
                                                "reserve_space",
                                                "out_grad",
                                                "trans_x",
                                                "trans_y",
                                                "activation_grad"},
                                               {},
                                               {},
                                               {},
                                               {});

  return std::make_tuple(
      inputs, attributes, outputs, run_time_info, "fused_gemm_epilogue_grad");
}

void FusedGemmEpilogueGradOp::Build(pir::Builder &builder,
                                    pir::OperationArgument &argument,
                                    pir::Value x_,
                                    pir::Value y_,
                                    pir::Value reserve_space_,
                                    pir::Value out_grad_,
                                    pir::AttributeMap attributes) {
  VLOG(4) << "Start build FusedGemmEpilogueGradOp";

  PADDLE_ENFORCE(
      attributes.find("trans_x") != attributes.end(),
      phi::errors::NotFound(
          "'trans_x' Attribute is expected for FusedGemmEpilogueGradOp"));
  bool trans_x = attributes.at("trans_x").dyn_cast<pir::BoolAttribute>().data();

  PADDLE_ENFORCE(
      attributes.find("trans_y") != attributes.end(),
      phi::errors::NotFound(
          "'trans_y' Attribute is expected for FusedGemmEpilogueGradOp"));
  bool trans_y = attributes.at("trans_y").dyn_cast<pir::BoolAttribute>().data();

  PADDLE_ENFORCE(
      attributes.find("activation_grad") != attributes.end(),
      phi::errors::NotFound("'activation_grad' Attribute is expected for"
                            "FusedGemmEpilogueGradOp"));
  std::string activation_grad =
      attributes.at("activation_grad").dyn_cast<pir::StrAttribute>().AsString();

  VLOG(4) << "Builder construction inputs";
  std::vector<pir::Value> argument_inputs = {x_, y_, reserve_space_, out_grad_};
  argument.AddInputs({x_, y_, reserve_space_, out_grad_});

  VLOG(4) << "Builder construction attributes";
  pir::AttributeMap argument_attributes = {};
  pir::Attribute attr_trans_x =
      pir::BoolAttribute::get(pir::IrContext::Instance(), trans_x);
  argument.AddAttribute("trans_x", attr_trans_x);
  argument_attributes.insert({"trans_x", attr_trans_x});
  pir::Attribute attr_trans_y =
      pir::BoolAttribute::get(pir::IrContext::Instance(), trans_y);
  argument.AddAttribute("trans_y", attr_trans_y);
  argument_attributes.insert({"trans_y", attr_trans_y});
  pir::Attribute attr_activation_grad =
      pir::StrAttribute::get(pir::IrContext::Instance(), activation_grad);
  argument.AddAttribute("activation_grad", attr_activation_grad);
  argument_attributes.insert({"activation_grad", attr_activation_grad});
  std::vector<pir::Type> argument_outputs =
      FusedGemmEpilogueGradOp::InferMeta(argument_inputs, &argument_attributes);

  argument.AddOutputs(argument_outputs.begin(), argument_outputs.end());
}

void FusedGemmEpilogueGradOp::VerifySig() {}

void FusedGemmEpilogueGradOp::InferMeta(phi::InferMetaContext *infer_meta) {
  auto fn = PD_INFER_META(phi::FusedGemmEpilogueGradInferMeta);
  fn(infer_meta);
}

std::vector<pir::Type> FusedGemmEpilogueGradOp::InferMeta(
    const std::vector<pir::Value> &input_values,
    pir::AttributeMap *p_attributes) {
  PADDLE_ENFORCE_NOT_NULL(
      p_attributes,
      common::errors::Fatal(
          "AttrtibueMap pointer in InferMeta function is nullptr."));
  auto &attributes = *p_attributes;
  IR_ENFORCE(input_values.size() == 4,
             "Num of inputs is expected to be 4 but got %d.",
             input_values.size());

  pir::Value x_ = input_values[0];
  pir::Value y_ = input_values[1];
  pir::Value reserve_space_ = input_values[2];
  pir::Value out_grad_ = input_values[3];
  VLOG(4) << "Start build FusedGemmEpilogueGradOp";

  PADDLE_ENFORCE(
      attributes.find("trans_x") != attributes.end(),
      phi::errors::NotFound(
          "'trans_x' Attribute is expected for FusedGemmEpilogueGradOp"));
  bool trans_x = attributes.at("trans_x").dyn_cast<pir::BoolAttribute>().data();

  PADDLE_ENFORCE(
      attributes.find("trans_y") != attributes.end(),
      phi::errors::NotFound(
          "'trans_y' Attribute is expected for FusedGemmEpilogueGradOp"));
  bool trans_y = attributes.at("trans_y").dyn_cast<pir::BoolAttribute>().data();

  PADDLE_ENFORCE(
      attributes.find("activation_grad") != attributes.end(),
      phi::errors::NotFound("'activation_grad' Attribute is expected for"
                            "FusedGemmEpilogueGradOp"));
  std::string activation_grad =
      attributes.at("activation_grad").dyn_cast<pir::StrAttribute>().AsString();

  VLOG(4) << "Builder construction outputs";
  paddle::dialect::DenseTensorType x;
  if (x_.type().isa<paddle::dialect::DenseTensorType>()) {
    x = x_.type().dyn_cast<paddle::dialect::DenseTensorType>();
  } else {
    PADDLE_THROW(phi::errors::Unimplemented(
        "Only support paddle::dialect::DenseTensorType or "
        "paddle::dialect::AllocatedDenseTensorType"));
  }

  paddle::dialect::DenseTensorType y;
  if (y_.type().isa<paddle::dialect::DenseTensorType>()) {
    y = y_.type().dyn_cast<paddle::dialect::DenseTensorType>();
  } else {
    PADDLE_THROW(phi::errors::Unimplemented(
        "Only support paddle::dialect::DenseTensorType or "
        "paddle::dialect::AllocatedDenseTensorType"));
  }

  paddle::dialect::DenseTensorType reserve_space;
  if (reserve_space_) {
    if (reserve_space_.type().isa<paddle::dialect::DenseTensorType>()) {
      reserve_space =
          reserve_space_.type().dyn_cast<paddle::dialect::DenseTensorType>();
    } else {
      PADDLE_THROW(phi::errors::Unimplemented(
          "Only support paddle::dialect::DenseTensorType or "
          "paddle::dialect::AllocatedDenseTensorType"));
    }
  } else {
    reserve_space = paddle::dialect::DenseTensorType();
  }

  paddle::dialect::DenseTensorType out_grad;
  if (out_grad_.type().isa<paddle::dialect::DenseTensorType>()) {
    out_grad = out_grad_.type().dyn_cast<paddle::dialect::DenseTensorType>();
  } else {
    PADDLE_THROW(phi::errors::Unimplemented(
        "Only support paddle::dialect::DenseTensorType or "
        "paddle::dialect::AllocatedDenseTensorType"));
  }

  VLOG(4) << "Builder construction  dense_x";
  paddle::dialect::IrTensor dense_x(
      paddle::dialect::TransToPhiDataType(x.dtype()),
      x.dims(),
      x.data_layout(),
      x.lod(),
      x.offset());
  VLOG(4) << "Builder construction  meta_x";
  paddle::dialect::IrMetaTensor meta_x(&dense_x);

  VLOG(4) << "Builder construction  dense_y";
  paddle::dialect::IrTensor dense_y(
      paddle::dialect::TransToPhiDataType(y.dtype()),
      y.dims(),
      y.data_layout(),
      y.lod(),
      y.offset());
  VLOG(4) << "Builder construction  meta_y";
  paddle::dialect::IrMetaTensor meta_y(&dense_y);

  VLOG(4) << "Builder construction  dense_reserve_space";
  std::unique_ptr<paddle::dialect::IrTensor> dense_reserve_space =
      reserve_space_
          ? std::make_unique<paddle::dialect::IrTensor>(
                paddle::dialect::TransToPhiDataType(reserve_space.dtype()),
                reserve_space.dims(),
                reserve_space.data_layout(),
                reserve_space.lod(),
                reserve_space.offset())
          : nullptr;
  VLOG(4) << "Builder construction  meta_reserve_space";
  paddle::dialect::IrMetaTensor meta_reserve_space(dense_reserve_space.get());

  VLOG(4) << "Builder construction  dense_out_grad";
  paddle::dialect::IrTensor dense_out_grad(
      paddle::dialect::TransToPhiDataType(out_grad.dtype()),
      out_grad.dims(),
      out_grad.data_layout(),
      out_grad.lod(),
      out_grad.offset());
  VLOG(4) << "Builder construction  meta_out_grad";
  paddle::dialect::IrMetaTensor meta_out_grad(&dense_out_grad);
  paddle::dialect::IrTensor dense_x_grad;
  paddle::dialect::IrMetaTensor meta_x_grad(&dense_x_grad);
  paddle::dialect::IrTensor dense_y_grad;
  paddle::dialect::IrMetaTensor meta_y_grad(&dense_y_grad);
  paddle::dialect::IrTensor dense_bias_grad;
  paddle::dialect::IrMetaTensor meta_bias_grad(&dense_bias_grad);

  phi::FusedGemmEpilogueGradInferMeta(meta_x,
                                      meta_y,
                                      meta_reserve_space,
                                      meta_out_grad,
                                      trans_x,
                                      trans_y,
                                      activation_grad,
                                      &meta_x_grad,
                                      &meta_y_grad,
                                      &meta_bias_grad);

  std::vector<pir::Type> argument_outputs;
  pir::Type x_grad_dense_tensor_type = paddle::dialect::DenseTensorType::get(
      pir::IrContext::Instance(),
      paddle::dialect::TransToIrDataType(dense_x_grad.dtype()),
      dense_x_grad.dims(),
      dense_x_grad.layout(),
      dense_x_grad.lod(),
      dense_x_grad.offset());
  argument_outputs.push_back(x_grad_dense_tensor_type);

  pir::Type y_grad_dense_tensor_type = paddle::dialect::DenseTensorType::get(
      pir::IrContext::Instance(),
      paddle::dialect::TransToIrDataType(dense_y_grad.dtype()),
      dense_y_grad.dims(),
      dense_y_grad.layout(),
      dense_y_grad.lod(),
      dense_y_grad.offset());
  argument_outputs.push_back(y_grad_dense_tensor_type);

  pir::Type bias_grad_dense_tensor_type = paddle::dialect::DenseTensorType::get(
      pir::IrContext::Instance(),
      paddle::dialect::TransToIrDataType(dense_bias_grad.dtype()),
      dense_bias_grad.dims(),
      dense_bias_grad.layout(),
      dense_bias_grad.lod(),
      dense_bias_grad.offset());
  argument_outputs.push_back(bias_grad_dense_tensor_type);
  return argument_outputs;
}

const char *SplitGradOp::attributes_name[1] = {"axis"};  // NOLINT

OpInfoTuple SplitGradOp::GetOpInfo() {
  std::vector<paddle::dialect::OpInputInfo> inputs = {
      OpInputInfo("out_grad",
                  "pir::VectorType<paddle::dialect::DenseTensorType>",
                  false,
                  false,
                  false,
                  true),
      OpInputInfo("axis",
                  "paddle::dialect::ScalarAttribute",
                  false,
                  false,
                  true,
                  false)};
  std::vector<paddle::dialect::OpAttributeInfo> attributes = {};
  std::vector<paddle::dialect::OpOutputInfo> outputs = {
      OpOutputInfo("x_grad", "paddle::dialect::DenseTensorType", false, false)};
  paddle::dialect::OpRunTimeInfo run_time_info =
      OpRunTimeInfo("ConcatInferMeta",
                    {"out_grad", "axis"},
                    "concat",
                    {"out_grad", "axis"},
                    {"out_grad"},
                    {},
                    {},
                    {});

  return std::make_tuple(
      inputs, attributes, outputs, run_time_info, "split_grad");
}

void SplitGradOp::Build(pir::Builder &builder,
                        pir::OperationArgument &argument,
                        pir::Value out_grad_,
                        float axis) {
  VLOG(4) << "Start build SplitGradOp";

  // Generate scalar mutable attribute: axis
  paddle::dialect::FullOp full_axis_op = builder.Build<paddle::dialect::FullOp>(
      std::vector<int64_t>{1}, axis, phi::DataType::FLOAT32, phi::CPUPlace());
  pir::Value axis_ = full_axis_op->result(0);

  VLOG(4) << "Builder construction inputs";
  std::vector<pir::Value> argument_inputs = {out_grad_, axis_};
  argument.AddInputs({out_grad_, axis_});

  VLOG(4) << "Builder construction attributes";
  pir::AttributeMap argument_attributes = {};
  std::vector<pir::Type> argument_outputs =
      SplitGradOp::InferMeta(argument_inputs, &argument_attributes);

  argument.AddOutputs(argument_outputs.begin(), argument_outputs.end());
  ::pir::PassStopGradientsDefaultly(argument);
}

void SplitGradOp::Build(pir::Builder &builder,
                        pir::OperationArgument &argument,
                        pir::Value out_grad_,
                        pir::Value axis_) {
  VLOG(4) << "Start build SplitGradOp";

  VLOG(4) << "Builder construction inputs";
  std::vector<pir::Value> argument_inputs = {out_grad_, axis_};
  argument.AddInputs({out_grad_, axis_});

  VLOG(4) << "Builder construction attributes";
  pir::AttributeMap argument_attributes = {};
  std::vector<pir::Type> argument_outputs =
      SplitGradOp::InferMeta(argument_inputs, &argument_attributes);

  argument.AddOutputs(argument_outputs.begin(), argument_outputs.end());
  ::pir::PassStopGradientsDefaultly(argument);
}

void SplitGradOp::VerifySig() {
  VLOG(4) << "Start Verifying inputs, outputs and attributes for: SplitGradOp.";
  VLOG(4) << "Verifying inputs:";
  {
    auto input_size = num_operands();
    PADDLE_ENFORCE_EQ(
        input_size,
        2u,
        phi::errors::PreconditionNotMet(
            "The size %d of inputs must be equal to 2.", input_size));
    if (auto vec_type =
            (*this)->operand_source(0).type().dyn_cast<pir::VectorType>()) {
      for (size_t i = 0; i < vec_type.size(); ++i) {
        PADDLE_ENFORCE(vec_type[i].isa<paddle::dialect::DenseTensorType>(),
                       phi::errors::PreconditionNotMet(
                           "Type validation failed for the 0th input."));
      }
    } else {
      PADDLE_ENFORCE((*this)
                         ->operand_source(0)
                         .type()
                         .isa<paddle::dialect::DenseTensorType>(),
                     phi::errors::PreconditionNotMet(
                         "Type validation failed for the 0th input."));
    }
    PADDLE_ENFORCE((*this)
                       ->operand_source(1)
                       .type()
                       .isa<paddle::dialect::DenseTensorType>(),
                   phi::errors::PreconditionNotMet(
                       "Type validation failed for the 1th input."));
  }
  VLOG(4) << "Verifying attributes:";
  {
    // Attributes num is 0, not need to check attributes type.
  }
  VLOG(4) << "Verifying outputs:";
  {
    auto output_size = num_results();
    PADDLE_ENFORCE_EQ(
        output_size,
        1u,
        phi::errors::PreconditionNotMet(
            "The size %d of outputs must be equal to 1.", output_size));
    PADDLE_ENFORCE(
        (*this)->result(0).type().isa<paddle::dialect::DenseTensorType>(),
        phi::errors::PreconditionNotMet(
            "Type validation failed for the 0th output."));
  }
  VLOG(4) << "End Verifying for: SplitGradOp.";
}

void SplitGradOp::InferMeta(phi::InferMetaContext *infer_meta) {
  auto fn = PD_INFER_META(phi::ConcatInferMeta);
  fn(infer_meta);
}

std::vector<pir::Type> SplitGradOp::InferMeta(
    const std::vector<pir::Value> &input_values,
    pir::AttributeMap *p_attributes) {
  VLOG(4) << "Start infermeta SplitGradOp";

  IR_ENFORCE(input_values.size() == 2,
             "Num of inputs is expected to be 2 but got %d.",
             input_values.size());
  pir::Value out_grad_ = input_values[0];
  pir::Value axis_ = input_values[1];

  VLOG(4) << "Builder construction outputs";
  pir::VectorType out_grad = out_grad_.type().dyn_cast<pir::VectorType>();
  int axis = axis_.defining_op()
                 ->dyn_cast<paddle::dialect::FullOp>()
                 .attribute<paddle::dialect::ScalarAttribute>("value")
                 .data()
                 .to<int>();

  std::vector<paddle::dialect::IrTensor> vec_dense_out_grad;
  for (size_t i = 0; i < static_cast<size_t>(out_grad.size()); i++) {
    vec_dense_out_grad.push_back(paddle::dialect::IrTensor(
        TransToPhiDataType(
            out_grad[i].dyn_cast<paddle::dialect::DenseTensorType>().dtype()),
        out_grad[i].dyn_cast<paddle::dialect::DenseTensorType>().dims(),
        out_grad[i].dyn_cast<paddle::dialect::DenseTensorType>().data_layout(),
        out_grad[i].dyn_cast<paddle::dialect::DenseTensorType>().lod(),
        out_grad[i].dyn_cast<paddle::dialect::DenseTensorType>().offset()));
  }
  std::vector<paddle::dialect::IrMetaTensor> vec_meta_out_grad;
  for (size_t i = 0; i < vec_dense_out_grad.size(); i++) {
    vec_meta_out_grad.push_back(
        paddle::dialect::IrMetaTensor(&vec_dense_out_grad[i]));
  }

  std::vector<const phi::MetaTensor *> meta_out_grad;
  for (size_t i = 0; i < static_cast<size_t>(vec_meta_out_grad.size()); i++) {
    meta_out_grad.push_back(&vec_meta_out_grad[i]);
  }
  paddle::dialect::IrTensor dense_x_grad;
  paddle::dialect::IrMetaTensor meta_x_grad(&dense_x_grad);

  phi::ConcatInferMeta(meta_out_grad, axis, &meta_x_grad);

  std::vector<pir::Type> argument_outputs;
  pir::Type x_grad_dense_tensor_type = paddle::dialect::DenseTensorType::get(
      pir::IrContext::Instance(),
      TransToIrDataType(dense_x_grad.dtype()),
      dense_x_grad.dims(),
      dense_x_grad.layout(),
      dense_x_grad.lod(),
      dense_x_grad.offset());
  argument_outputs.push_back(x_grad_dense_tensor_type);
  return argument_outputs;
}

const char *CreateArrayOp::attributes_name[1] = {"dtype"};  // NOLINT

OpInfoTuple CreateArrayOp::GetOpInfo() {
  std::vector<paddle::dialect::OpInputInfo> inputs = {};

  std::vector<paddle::dialect::OpAttributeInfo> attributes = {
      paddle::dialect::OpAttributeInfo(
          "dtype", "paddle::dialect::DataTypeAttribute", "")};

  std::vector<paddle::dialect::OpOutputInfo> outputs = {OpOutputInfo(
      "out", "paddle::dialect::DenseTensorArrayType", false, false)};

  paddle::dialect::OpRunTimeInfo run_time_info =
      OpRunTimeInfo("CreateArrayInferMeta",
                    {"dtype"},
                    "create_array",
                    {"dtype"},
                    {"dtype"},
                    {},
                    {},
                    {});

  return std::make_tuple(
      inputs, attributes, outputs, run_time_info, "create_array");
}

void CreateArrayOp::Build(pir::Builder &builder,
                          pir::OperationArgument &argument,
                          phi::DataType dtype) {
  VLOG(4) << "Start build CreateArrayOp";
  VLOG(4) << "Builder construction inputs";
  std::vector<pir::Value> argument_inputs = {};
  VLOG(4) << "Builder construction attributes";
  pir::AttributeMap argument_attributes = {};
  pir::Attribute attr_dtype = paddle::dialect::DataTypeAttribute::get(
      pir::IrContext::Instance(), dtype);
  argument.AddAttribute("dtype", attr_dtype);
  argument_attributes.insert({"dtype", attr_dtype});
  std::vector<pir::Type> argument_outputs =
      CreateArrayOp::InferMeta(argument_inputs, &argument_attributes);

  argument.AddOutputs(argument_outputs.begin(), argument_outputs.end());
  ::pir::PassStopGradientsDefaultly(argument);
}

void CreateArrayOp::VerifySig() {
  VLOG(4) << "Start Verifying inputs, outputs and attributes for: "
             "CreateArrayOp.";
  VLOG(4) << "Verifying inputs:";
  {
    auto input_size = num_operands();
    PADDLE_ENFORCE_EQ(
        input_size,
        0u,
        phi::errors::PreconditionNotMet(
            "The size %d of inputs must be equal to 1.", input_size));
  }
  VLOG(4) << "Verifying attributes:";
  {
    auto &attributes = this->attributes();
    PADDLE_ENFORCE(attributes.count("dtype") > 0, "dtype does not exist.");
  }
  VLOG(4) << "Verifying outputs:";
  {
    auto output_size = num_results();
    PADDLE_ENFORCE_EQ(
        output_size,
        1u,
        phi::errors::PreconditionNotMet(
            "The size %d of outputs must be equal to 1.", output_size));
    PADDLE_ENFORCE(
        (*this)->result(0).type().isa<paddle::dialect::DenseTensorArrayType>(),
        phi::errors::PreconditionNotMet(
            "Type validation failed for the 0th output."));
  }
  VLOG(4) << "End Verifying for: CreateArrayOp.";
}

void CreateArrayOp::InferMeta(phi::InferMetaContext *infer_meta) {
  auto fn = PD_INFER_META(phi::CreateArrayInferMeta);
  fn(infer_meta);
}

std::vector<pir::Type> CreateArrayOp::InferMeta(
    const std::vector<pir::Value> &input_values,
    pir::AttributeMap *p_attributes) {
  PADDLE_ENFORCE_NOT_NULL(
      p_attributes,
      common::errors::Fatal(
          "AttrtibueMap pointer in InferMeta function is nullptr."));
  auto &attributes = *p_attributes;
  VLOG(4) << "Start infermeta CreateArrayOp";

  PADDLE_ENFORCE(
      attributes.find("dtype") != attributes.end(),
      phi::errors::NotFound("'dtype' Attribute is expected for CreateArrayOp"));
  phi::DataType dtype = attributes.at("dtype")
                            .dyn_cast<paddle::dialect::DataTypeAttribute>()
                            .data();

  VLOG(4) << "Builder construction outputs";
  paddle::dialect::IrTensor dense_out;
  paddle::dialect::IrMetaTensor meta_out(&dense_out);

  phi::CreateArrayInferMeta(dtype, &meta_out);
  std::vector<pir::Type> argument_outputs;
  pir::Type out_dense_tensor_type = paddle::dialect::DenseTensorArrayType::get(
      pir::IrContext::Instance(),
      paddle::dialect::TransToIrDataType(dense_out.dtype()),
      common::make_ddim({0}),  // TODO(chenxi67): set shape
      dense_out.layout());
  argument_outputs.push_back(out_dense_tensor_type);
  return argument_outputs;
}

const char *CreateArrayLikeOp::attributes_name[1] = {"val"};

OpInfoTuple CreateArrayLikeOp::GetOpInfo() {
  std::vector<paddle::dialect::OpInputInfo> inputs = {
      paddle::dialect::OpInputInfo("input",
                                   "paddle::dialect::DenseTensorArrayType",
                                   false,
                                   false,
                                   false,
                                   false)};

  std::vector<paddle::dialect::OpAttributeInfo> attributes = {
      paddle::dialect::OpAttributeInfo("val", "pir::FloatAttribute", "")};

  std::vector<paddle::dialect::OpOutputInfo> outputs = {OpOutputInfo(
      "out", "paddle::dialect::DenseTensorArrayType", false, false)};

  paddle::dialect::OpRunTimeInfo run_time_info =
      OpRunTimeInfo("CreateArrayLikeInferMeta",
                    {"input"},
                    "create_array_like",
                    {"input", "val"},
                    {},
                    {},
                    {},
                    {});

  return std::make_tuple(
      inputs, attributes, outputs, run_time_info, "create_array_like");
}

void CreateArrayLikeOp::Build(pir::Builder &builder,             // NOLINT
                              pir::OperationArgument &argument,  // NOLINT
                              pir::Value &input_,                // NOLINT
                              float &val) {
  VLOG(4) << "Start build CreateArrayLikeOp";
  VLOG(4) << "Builder construction inputs";
  std::vector<pir::Value> argument_inputs = {input_};
  argument.AddInputs(argument_inputs);

  VLOG(4) << "Builder construction attributes";
  pir::AttributeMap argument_attributes = {};
  pir::Attribute attr_val =
      pir::FloatAttribute::get(pir::IrContext::Instance(), val);
  argument.AddAttribute("val", attr_val);
  argument_attributes.insert({"val", attr_val});
  std::vector<pir::Type> argument_outputs =
      CreateArrayLikeOp::InferMeta(argument_inputs, &argument_attributes);

  argument.AddOutputs(argument_outputs.begin(), argument_outputs.end());
  ::pir::PassStopGradientsDefaultly(argument);
}

void CreateArrayLikeOp::VerifySig() {
  VLOG(4) << "Start Verifying inputs, outputs and attributes for: "
             "CreateArrayLikeOp.";
  VLOG(4) << "Verifying inputs:";
  {
    auto input_size = num_operands();
    PADDLE_ENFORCE_EQ(
        input_size,
        1u,
        phi::errors::PreconditionNotMet(
            "The size %d of inputs must be equal to 1.", input_size));
  }
  VLOG(4) << "Verifying attributes:";
  {
    auto &attributes = this->attributes();
    PADDLE_ENFORCE(attributes.count("val") > 0, "val does not exist.");
  }
  VLOG(4) << "Verifying outputs:";
  {
    auto output_size = num_results();
    PADDLE_ENFORCE_EQ(
        output_size,
        1u,
        phi::errors::PreconditionNotMet(
            "The size %d of outputs must be equal to 1.", output_size));
    PADDLE_ENFORCE(
        (*this)->result(0).type().isa<paddle::dialect::DenseTensorArrayType>(),
        phi::errors::PreconditionNotMet(
            "Type validation failed for the 0th output."));
  }
  VLOG(4) << "End Verifying for: CreateArrayLikeOp.";
}

void CreateArrayLikeOp::InferMeta(phi::InferMetaContext *infer_meta) {
  auto fn = PD_INFER_META(phi::CreateArrayLikeInferMeta);
  fn(infer_meta);
}

std::vector<pir::Type> CreateArrayLikeOp::InferMeta(
    const std::vector<pir::Value> &input_values,
    pir::AttributeMap *p_attributes) {
  VLOG(4) << "Start infermeta CreateArrayLikeOp";
  IR_ENFORCE(input_values.size() == 1,
             "Num of inputs is expected to be 1 but got %d.",
             input_values.size());
  pir::Value input_ = input_values[0];

  VLOG(4) << "Builder construction outputs";
  paddle::dialect::DenseTensorArrayType input_type;
  if (input_.type().isa<paddle::dialect::DenseTensorArrayType>()) {
    input_type =
        input_.type().dyn_cast<paddle::dialect::DenseTensorArrayType>();
  } else {
    PADDLE_THROW(phi::errors::Unimplemented(
        "Only support paddle::dialect::DenseTensorArrayType or "
        "paddle::dialect::AllocatedDenseTensorArrayType"));
  }

  paddle::dialect::IrTensor dense_input(
      paddle::dialect::TransToPhiDataType(input_type.dtype()),
      input_type.dims(),
      input_type.data_layout(),
      {});

  paddle::dialect::IrMetaTensor meta_input(&dense_input);

  paddle::dialect::IrTensor dense_out;
  paddle::dialect::IrMetaTensor meta_out(&dense_out);

  phi::CreateArrayLikeInferMeta(meta_input, &meta_out);

  std::vector<pir::Type> argument_outputs;
  pir::Type out_dense_tensor_type = paddle::dialect::DenseTensorArrayType::get(
      pir::IrContext::Instance(),
      paddle::dialect::TransToIrDataType(dense_out.dtype()),
      dense_out.dims(),
      dense_out.layout());
  argument_outputs.push_back(out_dense_tensor_type);

  return argument_outputs;
}

OpInfoTuple ArrayLengthOp::GetOpInfo() {
  std::vector<paddle::dialect::OpInputInfo> inputs = {
      OpInputInfo("x",
                  "paddle::dialect::DenseTensorArrayType",
                  false,
                  false,
                  false,
                  false)};

  std::vector<paddle::dialect::OpAttributeInfo> attributes = {};

  std::vector<paddle::dialect::OpOutputInfo> outputs = {
      OpOutputInfo("out", "paddle::dialect::DenseTensorType", false, false)};

  paddle::dialect::OpRunTimeInfo run_time_info = OpRunTimeInfo(
      "ArrayLengthInferMeta", {"x"}, "array_length", {"x"}, {}, {}, {}, {});

  return std::make_tuple(
      inputs, attributes, outputs, run_time_info, "array_length");
}

void ArrayLengthOp::Build(pir::Builder &builder,
                          pir::OperationArgument &argument,
                          pir::Value x) {
  VLOG(4) << "Start build ArrayLengthOp";
  VLOG(4) << "Builder construction inputs";
  std::vector<pir::Value> argument_inputs = {x};
  argument.AddInputs({x});
  VLOG(4) << "Builder construction attributes";
  pir::AttributeMap argument_attributes = {};
  std::vector<pir::Type> argument_outputs =
      ArrayLengthOp::InferMeta(argument_inputs, &argument_attributes);

  argument.AddOutputs(argument_outputs.begin(), argument_outputs.end());
}

void ArrayLengthOp::VerifySig() {
  VLOG(4) << "Start Verifying inputs, outputs and attributes for: "
             "ArrayLengthOp.";
  VLOG(4) << "Verifying inputs:";
  {
    auto input_size = num_operands();
    PADDLE_ENFORCE_EQ(
        input_size,
        1u,
        phi::errors::PreconditionNotMet(
            "The size %d of inputs must be equal to 1.", input_size));

    PADDLE_ENFORCE((*this)
                       ->operand_source(0)
                       .type()
                       .isa<paddle::dialect::DenseTensorArrayType>(),
                   phi::errors::PreconditionNotMet(
                       "Type validation failed for the 0th input."));
  }
  VLOG(4) << "Verifying attributes:";
  VLOG(4) << "Verifying outputs:";
  {
    auto output_size = num_results();
    PADDLE_ENFORCE_EQ(
        output_size,
        1u,
        phi::errors::PreconditionNotMet(
            "The size %d of outputs must be equal to 1.", output_size));
    PADDLE_ENFORCE(
        (*this)->result(0).type().isa<paddle::dialect::DenseTensorType>(),
        phi::errors::PreconditionNotMet(
            "Type validation failed for the 0th output."));
  }
  VLOG(4) << "End Verifying for: ArrayLengthOp.";
}

void ArrayLengthOp::InferMeta(phi::InferMetaContext *infer_meta) {
  auto fn = PD_INFER_META(phi::ArrayLengthInferMeta);
  fn(infer_meta);
}

std::vector<pir::Type> ArrayLengthOp::InferMeta(
    const std::vector<pir::Value> &input_values,
    pir::AttributeMap *p_attributes) {
  VLOG(4) << "Start infermeta ArrayLengthOp";
  IR_ENFORCE(input_values.size() == 1,
             "Num of inputs is expected to be 1 but got %d.",
             input_values.size());
  pir::Value x_ = input_values[0];

  paddle::dialect::DenseTensorArrayType x_type;
  if (x_.type().isa<paddle::dialect::DenseTensorArrayType>()) {
    x_type = x_.type().dyn_cast<paddle::dialect::DenseTensorArrayType>();
  } else {
    PADDLE_THROW(phi::errors::Unimplemented(
        "Only support paddle::dialect::DenseTensorArrayType or "
        "paddle::dialect::AllocatedDenseTensorArrayType"));
  }

  paddle::dialect::IrTensor dense_x(
      paddle::dialect::TransToPhiDataType(x_type.dtype()),
      x_type.dims(),
      x_type.data_layout(),
      {});
  paddle::dialect::IrMetaTensor meta_x(&dense_x);

  paddle::dialect::IrTensor dense_out;
  paddle::dialect::IrMetaTensor meta_out(&dense_out);

  phi::ArrayLengthInferMeta(meta_x, &meta_out);

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

OpInfoTuple ArrayReadOp::GetOpInfo() {
  std::vector<paddle::dialect::OpInputInfo> inputs = {
      OpInputInfo("array",
                  "paddle::dialect::DenseTensorArrayType",
                  false,
                  false,
                  false,
                  false),
      OpInputInfo(
          "i", "paddle::dialect::ScalarAttribute", false, false, true, false)};

  std::vector<paddle::dialect::OpAttributeInfo> attributes = {};

  std::vector<paddle::dialect::OpOutputInfo> outputs = {
      OpOutputInfo("out", "paddle::dialect::DenseTensorType", false, false)};

  paddle::dialect::OpRunTimeInfo run_time_info =
      OpRunTimeInfo("ArrayReadInferMeta",
                    {"array", "i"},
                    "array_read",
                    {"array", "i"},
                    {"array"},
                    {},
                    {},
                    {});

  return std::make_tuple(
      inputs, attributes, outputs, run_time_info, "array_read");
}

void ArrayReadOp::Build(pir::Builder &builder,
                        pir::OperationArgument &argument,
                        pir::Value array,
                        int64_t i) {
  VLOG(4) << "Start build ArrayReadOp";
  paddle::dialect::FullOp full_i_op = builder.Build<paddle::dialect::FullOp>(
      std::vector<int64_t>{1}, i, phi::DataType::INT64, phi::CPUPlace());

  VLOG(4) << "Builder construction inputs";
  std::vector<pir::Value> argument_inputs = {array, full_i_op.result(0)};
  argument.AddInputs({array, full_i_op.result(0)});
  VLOG(4) << "Builder construction attributes";
  pir::AttributeMap argument_attributes = {};
  std::vector<pir::Type> argument_outputs =
      ArrayReadOp::InferMeta(argument_inputs, &argument_attributes);

  argument.AddOutputs(argument_outputs.begin(), argument_outputs.end());
  ::pir::PassStopGradientsDefaultly(argument);
}

void ArrayReadOp::Build(pir::Builder &builder,
                        pir::OperationArgument &argument,
                        pir::Value array,
                        pir::Value i) {
  VLOG(4) << "Start build ArrayReadOp";
  VLOG(4) << "Builder construction inputs";
  std::vector<pir::Value> argument_inputs = {array, i};
  argument.AddInputs({array, i});
  VLOG(4) << "Builder construction attributes";
  pir::AttributeMap argument_attributes = {};
  std::vector<pir::Type> argument_outputs =
      ArrayReadOp::InferMeta(argument_inputs, &argument_attributes);

  argument.AddOutputs(argument_outputs.begin(), argument_outputs.end());
  ::pir::PassStopGradientsDefaultly(argument);
}

void ArrayReadOp::VerifySig() {
  VLOG(4) << "Start Verifying inputs, outputs and attributes for: "
             "ArrayReadOp.";
  VLOG(4) << "Verifying inputs:";
  {
    auto input_size = num_operands();
    PADDLE_ENFORCE_EQ(
        input_size,
        2u,
        phi::errors::PreconditionNotMet(
            "The size %d of inputs must be equal to 3.", input_size));

    PADDLE_ENFORCE((*this)
                       ->operand_source(0)
                       .type()
                       .isa<paddle::dialect::DenseTensorArrayType>(),
                   phi::errors::PreconditionNotMet(
                       "Type validation failed for the 0th input."));
    PADDLE_ENFORCE((*this)
                       ->operand_source(1)
                       .type()
                       .isa<paddle::dialect::DenseTensorType>(),
                   phi::errors::PreconditionNotMet(
                       "Type validation failed for the 1th input."));
  }
  VLOG(4) << "Verifying attributes:";
  VLOG(4) << "Verifying outputs:";
  {
    auto output_size = num_results();
    PADDLE_ENFORCE_EQ(
        output_size,
        1u,
        phi::errors::PreconditionNotMet(
            "The size %d of outputs must be equal to 1.", output_size));
    PADDLE_ENFORCE(
        (*this)->result(0).type().isa<paddle::dialect::DenseTensorType>(),
        phi::errors::PreconditionNotMet(
            "Type validation failed for the 0th output."));
  }
  VLOG(4) << "End Verifying for: ArrayReadOp.";
}

void ArrayReadOp::InferMeta(phi::InferMetaContext *infer_meta) {
  auto fn = PD_INFER_META(phi::ArrayReadInferMeta);
  fn(infer_meta);
}

std::vector<pir::Type> ArrayReadOp::InferMeta(
    const std::vector<pir::Value> &input_values,
    pir::AttributeMap *p_attributes) {
  VLOG(4) << "Start infermeta ArrayLengthOp";
  IR_ENFORCE(input_values.size() == 2,
             "Num of inputs is expected to be 2 but got %d.",
             input_values.size());
  pir::Value array_ = input_values[0];
  pir::Value i_ = input_values[1];

  VLOG(4) << "Builder construction outputs";
  paddle::dialect::DenseTensorArrayType array_type;
  if (array_.type().isa<paddle::dialect::DenseTensorArrayType>()) {
    array_type =
        array_.type().dyn_cast<paddle::dialect::DenseTensorArrayType>();
  } else {
    PADDLE_THROW(phi::errors::Unimplemented(
        "Only support paddle::dialect::DenseTensorArrayType or "
        "paddle::dialect::AllocatedDenseTensorArrayType"));
  }
  paddle::dialect::IrTensor dense_array(
      paddle::dialect::TransToPhiDataType(array_type.dtype()),
      array_type.dims(),
      array_type.data_layout(),
      {});
  paddle::dialect::IrMetaTensor meta_array(&dense_array);

  phi::Scalar i_scalar;
  if (i_.isa<pir::OpResult>() &&
      i_.defining_op()->isa<paddle::dialect::FullOp>()) {
    i_scalar = phi::Scalar(i_.defining_op()
                               ->dyn_cast<paddle::dialect::FullOp>()
                               .attribute("value")
                               .dyn_cast<paddle::dialect::ScalarAttribute>()
                               .data()
                               .to<int64_t>());
  } else {
    i_scalar = phi::Scalar(-1);
    i_scalar.SetFromTensor(true);
  }

  paddle::dialect::IrTensor dense_out;
  paddle::dialect::IrMetaTensor meta_out(&dense_out);

  phi::ArrayReadInferMeta(
      meta_array, i_scalar, &meta_out, phi::MetaConfig(false, false));

  std::vector<pir::Type> argument_outputs;
  pir::Type out_type = paddle::dialect::DenseTensorType::get(
      pir::IrContext::Instance(),
      paddle::dialect::TransToIrDataType(dense_out.dtype()),
      dense_out.dims(),
      dense_out.layout(),
      dense_out.lod());
  argument_outputs.push_back(out_type);
  return argument_outputs;
}

OpInfoTuple ArrayWrite_Op::GetOpInfo() {
  std::vector<paddle::dialect::OpInputInfo> inputs = {
      OpInputInfo("array",
                  "paddle::dialect::DenseTensorArrayType",
                  false,
                  false,
                  false,
                  false),
      OpInputInfo(
          "x", "paddle::dialect::DenseTensorType", false, false, false, true),
      OpInputInfo(
          "i", "paddle::dialect::ScalarAttribute", false, false, true, false)};

  std::vector<paddle::dialect::OpAttributeInfo> attributes = {};

  std::vector<paddle::dialect::OpOutputInfo> outputs = {OpOutputInfo(
      "out", "paddle::dialect::DenseTensorArrayType", false, false)};

  paddle::dialect::OpRunTimeInfo run_time_info =
      OpRunTimeInfo("ArrayWriteInferMeta",
                    {"array", "x"},
                    "array_write",
                    {"array", "x", "i"},
                    {"array"},
                    {"array"},
                    {{"out", "array"}},
                    {});

  return std::make_tuple(
      inputs, attributes, outputs, run_time_info, "array_write");
}

void ArrayWrite_Op::Build(pir::Builder &builder,
                          pir::OperationArgument &argument,
                          pir::Value array,
                          pir::Value x,
                          pir::Value i) {
  VLOG(4) << "Start build ArrayWrite_Op";
  VLOG(4) << "Builder construction inputs";
  std::vector<pir::Value> argument_inputs = {array, x, i};
  argument.AddInputs({array, x, i});
  VLOG(4) << "Builder construction attributes";
  pir::AttributeMap argument_attributes = {};
  std::vector<pir::Type> argument_outputs =
      ArrayWrite_Op::InferMeta(argument_inputs, &argument_attributes);

  argument.AddOutputs(argument_outputs.begin(), argument_outputs.end());
  constexpr char kStopGradientAttrName[] = "stop_gradient";
  auto stop_gradient0 =
      argument.inputs[0].attribute<pir::BoolAttribute>(kStopGradientAttrName);
  auto stop_gradient1 =
      argument.inputs[1].attribute<pir::BoolAttribute>(kStopGradientAttrName);
  auto stop_gradient = stop_gradient0.data() && stop_gradient1.data();
  argument.inputs[0].set_attribute(kStopGradientAttrName,
                                   builder.bool_attr(stop_gradient));
}

void ArrayWrite_Op::VerifySig() {
  VLOG(4) << "Start Verifying inputs, outputs and attributes for: "
             "ArrayWrite_Op.";
  VLOG(4) << "Verifying inputs:";
  {
    auto input_size = num_operands();
    PADDLE_ENFORCE_EQ(
        input_size,
        3u,
        phi::errors::PreconditionNotMet(
            "The size %d of inputs must be equal to 3.", input_size));

    PADDLE_ENFORCE((*this)
                       ->operand_source(0)
                       .type()
                       .isa<paddle::dialect::DenseTensorArrayType>(),
                   phi::errors::PreconditionNotMet(
                       "Type validation failed for the 0th input."));
    PADDLE_ENFORCE((*this)
                       ->operand_source(1)
                       .type()
                       .isa<paddle::dialect::DenseTensorType>(),
                   phi::errors::PreconditionNotMet(
                       "Type validation failed for the 1th input."));
    PADDLE_ENFORCE((*this)
                       ->operand_source(2)
                       .type()
                       .isa<paddle::dialect::DenseTensorType>(),
                   phi::errors::PreconditionNotMet(
                       "Type validation failed for the 2th input."));
  }
  VLOG(4) << "Verifying attributes:";
  VLOG(4) << "Verifying outputs:";
  {
    auto output_size = num_results();
    PADDLE_ENFORCE_EQ(
        output_size,
        1u,
        phi::errors::PreconditionNotMet(
            "The size %d of outputs must be equal to 1.", output_size));
    PADDLE_ENFORCE(
        (*this)->result(0).type().isa<paddle::dialect::DenseTensorArrayType>(),
        phi::errors::PreconditionNotMet(
            "Type validation failed for the 0th output."));
  }
  VLOG(4) << "End Verifying for: ArrayWrite_Op.";
}

void ArrayWrite_Op::InferMeta(phi::InferMetaContext *infer_meta) {
  auto fn = PD_INFER_META(phi::ArrayWriteInferMeta);
  fn(infer_meta);
}

std::vector<pir::Type> ArrayWrite_Op::InferMeta(
    const std::vector<pir::Value> &input_values,
    pir::AttributeMap *p_attributes) {
  VLOG(4) << "Start infermeta ArrayWrite_Op";
  IR_ENFORCE(input_values.size() == 3,
             "Num of inputs is expected to be 3 but got %d.",
             input_values.size());
  pir::Value array_ = input_values[0];
  pir::Value x_ = input_values[1];

  VLOG(4) << "Builder construction outputs";
  paddle::dialect::DenseTensorArrayType array_type;
  if (array_.type().isa<paddle::dialect::DenseTensorArrayType>()) {
    array_type =
        array_.type().dyn_cast<paddle::dialect::DenseTensorArrayType>();
  } else {
    PADDLE_THROW(phi::errors::Unimplemented(
        "Only support paddle::dialect::DenseTensorArrayType or "
        "paddle::dialect::AllocatedDenseTensorArrayType"));
  }

  paddle::dialect::IrTensor dense_array(
      paddle::dialect::TransToPhiDataType(array_type.dtype()),
      array_type.dims(),
      array_type.data_layout(),
      {});
  paddle::dialect::IrMetaTensor meta_array(&dense_array);

  paddle::dialect::DenseTensorType x_type;
  phi::Place place = phi::CPUPlace();
  if (x_.type().isa<paddle::dialect::DenseTensorType>()) {
    x_type = x_.type().dyn_cast<paddle::dialect::DenseTensorType>();
  } else {
    PADDLE_THROW(phi::errors::Unimplemented(
        "Only support paddle::dialect::DenseTensorType or "
        "paddle::dialect::AllocatedDenseTensorType"));
  }
  paddle::dialect::IrTensor dense_x(
      paddle::dialect::TransToPhiDataType(x_type.dtype()),
      x_type.dims(),
      x_type.data_layout(),
      x_type.lod(),
      x_type.offset());
  paddle::dialect::IrMetaTensor meta_x(&dense_x);

  paddle::dialect::IrTensor dense_out;
  paddle::dialect::IrMetaTensor meta_out(&dense_out);

  phi::ArrayWriteInferMeta(
      meta_array, meta_x, &meta_out, phi::MetaConfig(false, false));

  std::vector<pir::Type> argument_outputs;
  pir::Type out_type = paddle::dialect::DenseTensorArrayType::get(
      pir::IrContext::Instance(),
      paddle::dialect::TransToIrDataType(dense_array.dtype()),
      x_type.dims(),
      dense_array.layout());
  // update array's dims as x's dims.
  // TOOD(chenxi67) Do not change if dim is set by custom
  if (array_.type().isa<paddle::dialect::AllocatedDenseTensorArrayType>()) {
    array_.set_type(paddle::dialect::AllocatedDenseTensorArrayType::get(
        pir::IrContext::Instance(),
        place,
        array_type.dtype(),
        x_type.dims(),
        array_type.data_layout()));
  } else if (array_.type().isa<paddle::dialect::DenseTensorArrayType>()) {
    array_.set_type(
        paddle::dialect::DenseTensorArrayType::get(pir::IrContext::Instance(),
                                                   array_type.dtype(),
                                                   x_type.dims(),
                                                   array_type.data_layout()));
  }

  argument_outputs.push_back(out_type);
  return argument_outputs;
}

const char *ArrayToTensorOp::attributes_name[2] = {"axis", "use_stack"};

OpInfoTuple ArrayToTensorOp::GetOpInfo() {
  std::vector<paddle::dialect::OpInputInfo> inputs = {
      paddle::dialect::OpInputInfo("x",
                                   "paddle::dialect::DenseTensorArrayType",
                                   false,
                                   false,
                                   false,
                                   true)};

  std::vector<paddle::dialect::OpAttributeInfo> attributes = {
      paddle::dialect::OpAttributeInfo("axis", "pir::Int32Attribute", ""),
      paddle::dialect::OpAttributeInfo("use_stack", "pir::BoolAttribute", "")};

  std::vector<paddle::dialect::OpOutputInfo> outputs = {
      paddle::dialect::OpOutputInfo(
          "out", "paddle::dialect::DenseTensorType", false, false),
      paddle::dialect::OpOutputInfo(
          "out_index", "paddle::dialect::DenseTensorType", false, false)};

  paddle::dialect::OpRunTimeInfo run_time_info =
      paddle::dialect::OpRunTimeInfo("ArrayToTensorInferMeta",
                                     {"x", "axis", "use_stack"},
                                     "array_to_tensor",
                                     {"x", "axis", "use_stack"},
                                     {"x"},
                                     {},
                                     {},
                                     {});
  return std::make_tuple(
      inputs, attributes, outputs, run_time_info, "array_to_tensor");
}

void ArrayToTensorOp::Build(pir::Builder &builder,             // NOLINT
                            pir::OperationArgument &argument,  // NOLINT
                            pir::Value x,
                            int axis,
                            bool use_stack) {
  VLOG(4) << "Start build ArrayToTensorOp";
  VLOG(4) << "Builder construction inputs";
  std::vector<pir::Value> argument_inputs = {x};
  argument.AddInputs({x});

  VLOG(4) << "Builder construction attributes";
  pir::AttributeMap argument_attributes = {};
  pir::Attribute attr_axis =
      pir::Int32Attribute::get(pir::IrContext::Instance(), axis);
  argument.AddAttribute("axis", attr_axis);
  argument_attributes.insert({"axis", attr_axis});
  pir::Attribute attr_use_stack =
      pir::BoolAttribute::get(pir::IrContext::Instance(), use_stack);
  argument.AddAttribute("use_stack", attr_use_stack);
  argument_attributes.insert({"use_stack", attr_use_stack});
  std::vector<pir::Type> argument_outputs =
      ArrayToTensorOp::InferMeta(argument_inputs, &argument_attributes);

  argument.AddOutputs(argument_outputs.begin(), argument_outputs.end());
  ::pir::PassStopGradientsDefaultly(argument);
}

void ArrayToTensorOp::VerifySig() {
  VLOG(4) << "Start Verifying inputs, outputs and attributes for: "
             "ArrayToTensorOp.";
  VLOG(4) << "Verifying inputs:";
  {
    auto input_size = num_operands();
    PADDLE_ENFORCE_EQ(
        input_size,
        1u,
        phi::errors::PreconditionNotMet(
            "The size %d of inputs must be equal to 1.", input_size));

    PADDLE_ENFORCE((*this)
                       ->operand_source(0)
                       .type()
                       .isa<paddle::dialect::DenseTensorArrayType>(),
                   phi::errors::PreconditionNotMet(
                       "Type validation failed for the 0th input."));
  }

  VLOG(4) << "Verifying attributes:";
  {
    auto &attributes = this->attributes();
    PADDLE_ENFORCE(attributes.count("axis") > 0, "axis does not exist.");
    PADDLE_ENFORCE(attributes.count("use_stack") > 0,
                   "use_stack does not exist.");
  }

  VLOG(4) << "Verifying outputs:";
  {
    auto output_size = num_results();
    PADDLE_ENFORCE_EQ(
        output_size,
        2u,
        phi::errors::PreconditionNotMet(
            "The size %d of outputs must be equal to 1.", output_size));
    PADDLE_ENFORCE(
        (*this)->result(0).type().isa<paddle::dialect::DenseTensorType>(),
        phi::errors::PreconditionNotMet(
            "Type validation failed for the 0th output."));
    PADDLE_ENFORCE(
        (*this)->result(1).type().isa<paddle::dialect::DenseTensorType>(),
        phi::errors::PreconditionNotMet(
            "Type validation failed for the 0th output."));
  }
  VLOG(4) << "End Verifying for: ArrayToTensorOp.";
}

void ArrayToTensorOp::InferMeta(phi::InferMetaContext *infer_meta) {
  auto fn = PD_INFER_META(phi::ArrayToTensorInferMeta);
  fn(infer_meta);
}

std::vector<pir::Type> ArrayToTensorOp::InferMeta(
    const std::vector<pir::Value> &input_values,
    pir::AttributeMap *p_attributes) {
  PADDLE_ENFORCE_NOT_NULL(
      p_attributes,
      common::errors::Fatal(
          "AttrtibueMap pointer in InferMeta function is nullptr."));
  auto &attributes = *p_attributes;
  VLOG(4) << "Start infermeta ArrayToTensorOp";
  IR_ENFORCE(input_values.size() == 1,
             "Num of inputs is expected to be 1 but got %d.",
             input_values.size());
  pir::Value x_ = input_values[0];

  IR_ENFORCE(attributes.find("axis") != attributes.end(),
             "'value' Attribute is expected for IncrementOp. ");
  int32_t axis = attributes.at("axis").dyn_cast<pir::Int32Attribute>().data();

  IR_ENFORCE(attributes.find("use_stack") != attributes.end(),
             "'value' Attribute is expected for IncrementOp. ");
  bool use_stack =
      attributes.at("use_stack").dyn_cast<pir::BoolAttribute>().data();

  VLOG(4) << "Builder construction outputs";
  paddle::dialect::DenseTensorArrayType x_type;
  if (x_.type().isa<paddle::dialect::DenseTensorArrayType>()) {
    x_type = x_.type().dyn_cast<paddle::dialect::DenseTensorArrayType>();
  } else {
    PADDLE_THROW(phi::errors::Unimplemented(
        "Only support paddle::dialect::DenseTensorArrayType or "
        "paddle::dialect::AllocatedDenseTensorArrayType"));
  }
  paddle::dialect::IrTensor dense_x(
      paddle::dialect::TransToPhiDataType(x_type.dtype()),
      x_type.dims(),
      x_type.data_layout(),
      {});
  paddle::dialect::IrMetaTensor meta_x(&dense_x);

  paddle::dialect::IrTensor dense_out;
  paddle::dialect::IrMetaTensor meta_out(&dense_out);

  paddle::dialect::IrTensor dense_out_index;
  paddle::dialect::IrMetaTensor meta_out_index(&dense_out_index);

  phi::ArrayToTensorInferMeta(meta_x,
                              axis,
                              use_stack,
                              &meta_out,
                              &meta_out_index,
                              phi::MetaConfig(false, false));

  std::vector<pir::Type> argument_outputs;
  pir::Type out_dense_tensor_type = paddle::dialect::DenseTensorType::get(
      pir::IrContext::Instance(),
      paddle::dialect::TransToIrDataType(dense_out.dtype()),
      dense_out.dims(),
      dense_out.layout(),
      dense_out.lod(),
      dense_out.offset());
  argument_outputs.push_back(out_dense_tensor_type);
  pir::Type out_index_dense_tensor_type = paddle::dialect::DenseTensorType::get(
      pir::IrContext::Instance(),
      paddle::dialect::TransToIrDataType(dense_out_index.dtype()),
      dense_out_index.dims(),
      dense_out_index.layout(),
      dense_out_index.lod(),
      dense_out_index.offset());
  argument_outputs.push_back(out_index_dense_tensor_type);
  return argument_outputs;
}

const char *TensorToArrayOp::attributes_name[2] = {"axis", "use_stack"};

OpInfoTuple TensorToArrayOp::GetOpInfo() {
  std::vector<paddle::dialect::OpInputInfo> inputs = {
      paddle::dialect::OpInputInfo("x",
                                   "paddle::dialect::DenseTensorArrayType",
                                   false,
                                   false,
                                   false,
                                   true),
      paddle::dialect::OpInputInfo("out_grad",
                                   "paddle::dialect::DenseTensorType",
                                   false,
                                   false,
                                   false,
                                   true)};

  std::vector<paddle::dialect::OpAttributeInfo> attributes = {
      paddle::dialect::OpAttributeInfo("axis", "pir::Int32Attribute", ""),
      paddle::dialect::OpAttributeInfo("use_stack", "pir::BoolAttribute", "")};

  std::vector<paddle::dialect::OpOutputInfo> outputs = {
      paddle::dialect::OpOutputInfo(
          "out", "paddle::dialect::DenseTensorArrayType", false, false)};

  paddle::dialect::OpRunTimeInfo run_time_info =
      paddle::dialect::OpRunTimeInfo("TensorToArrayInferMeta",
                                     {"x", "axis", "use_stack"},
                                     "tensor_to_array",
                                     {"x", "out_grad", "axis", "use_stack"},
                                     {"x"},
                                     {},
                                     {},
                                     {});
  return std::make_tuple(
      inputs, attributes, outputs, run_time_info, "tensor_to_array");
}

void TensorToArrayOp::Build(pir::Builder &builder,             // NOLINT
                            pir::OperationArgument &argument,  // NOLINT
                            pir::Value x_,
                            pir::Value out_grad_,
                            int axis,
                            bool use_stack) {
  VLOG(4) << "Start build TensorToArrayOp";
  VLOG(4) << "Builder construction inputs";
  std::vector<pir::Value> argument_inputs = {x_, out_grad_};
  argument.AddInputs({x_, out_grad_});

  VLOG(4) << "Builder construction attributes";
  pir::AttributeMap argument_attributes = {};
  pir::Attribute attr_axis =
      pir::Int32Attribute::get(pir::IrContext::Instance(), axis);
  argument.AddAttribute("axis", attr_axis);
  argument_attributes.insert({"axis", attr_axis});

  pir::Attribute attr_use_stack =
      pir::BoolAttribute::get(pir::IrContext::Instance(), use_stack);
  argument.AddAttribute("use_stack", attr_use_stack);
  argument_attributes.insert({"use_stack", attr_use_stack});
  std::vector<pir::Type> argument_outputs =
      TensorToArrayOp::InferMeta(argument_inputs, &argument_attributes);

  argument.AddOutputs(argument_outputs.begin(), argument_outputs.end());
  ::pir::PassStopGradientsDefaultly(argument);
}

void TensorToArrayOp::VerifySig() {
  VLOG(4) << "Start Verifying inputs, outputs and attributes for: "
             "TensorToArrayOp.";
  VLOG(4) << "Verifying inputs:";
  {
    auto input_size = num_operands();
    PADDLE_ENFORCE_EQ(
        input_size,
        2u,
        phi::errors::PreconditionNotMet(
            "The size %d of inputs must be equal to 2.", input_size));

    PADDLE_ENFORCE((*this)
                       ->operand_source(0)
                       .type()
                       .isa<paddle::dialect::DenseTensorArrayType>(),
                   phi::errors::PreconditionNotMet(
                       "Type validation failed for the 0th input."));
    PADDLE_ENFORCE((*this)
                       ->operand_source(1)
                       .type()
                       .isa<paddle::dialect::DenseTensorType>(),
                   phi::errors::PreconditionNotMet(
                       "Type validation failed for the 1th input."));
  }

  VLOG(4) << "Verifying attributes:";
  {
    auto &attributes = this->attributes();
    PADDLE_ENFORCE(attributes.count("axis") > 0, "axis does not exist.");
    PADDLE_ENFORCE(attributes.count("use_stack") > 0,
                   "use_stack does not exist.");
  }

  VLOG(4) << "Verifying outputs:";
  {
    auto output_size = num_results();
    PADDLE_ENFORCE_EQ(
        output_size,
        1u,
        phi::errors::PreconditionNotMet(
            "The size %d of outputs must be equal to 1.", output_size));
    PADDLE_ENFORCE(
        (*this)->result(0).type().isa<paddle::dialect::DenseTensorArrayType>(),
        phi::errors::PreconditionNotMet(
            "Type validation failed for the 0th output."));
  }
  VLOG(4) << "End Verifying for: TensorToArrayOp.";
}

void TensorToArrayOp::InferMeta(phi::InferMetaContext *infer_meta) {
  auto fn = PD_INFER_META(phi::TensorToArrayInferMeta);
  fn(infer_meta);
}

std::vector<pir::Type> TensorToArrayOp::InferMeta(
    const std::vector<pir::Value> &input_values,
    pir::AttributeMap *p_attributes) {
  PADDLE_ENFORCE_NOT_NULL(
      p_attributes,
      common::errors::Fatal(
          "AttrtibueMap pointer in InferMeta function is nullptr."));
  auto &attributes = *p_attributes;
  VLOG(4) << "Start infermeta TensorToArrayOp";
  IR_ENFORCE(input_values.size() == 2,
             "Num of inputs is expected to be 2 but got %d.",
             input_values.size());
  pir::Value x_ = input_values[0];
  pir::Value out_grad_ = input_values[1];

  VLOG(4) << "Builder construction attributes";
  pir::AttributeMap argument_attributes = {};

  IR_ENFORCE(attributes.find("axis") != attributes.end(),
             "'value' Attribute is expected for IncrementOp. ");
  int32_t axis = attributes.at("axis").dyn_cast<pir::Int32Attribute>().data();

  IR_ENFORCE(attributes.find("use_stack") != attributes.end(),
             "'value' Attribute is expected for IncrementOp. ");
  bool use_stack =
      attributes.at("use_stack").dyn_cast<pir::BoolAttribute>().data();

  VLOG(4) << "Builder construction outputs";
  paddle::dialect::DenseTensorArrayType x;

  if (x_.type().isa<paddle::dialect::DenseTensorArrayType>()) {
    x = x_.type().dyn_cast<paddle::dialect::DenseTensorArrayType>();
  } else {
    PADDLE_THROW(phi::errors::Unimplemented(
        "Only support paddle::dialect::DenseTensorArrayType or "
        "paddle::dialect::AllocatedDenseTensorArrayType"));
  }

  paddle::dialect::IrTensor dense_x(
      paddle::dialect::TransToPhiDataType(x.dtype()),
      x.dims(),
      x.data_layout(),
      {});

  paddle::dialect::DenseTensorType out_grad;
  if (out_grad_.type().isa<paddle::dialect::DenseTensorType>()) {
    out_grad = out_grad_.type().dyn_cast<paddle::dialect::DenseTensorType>();
  } else {
    PADDLE_THROW(phi::errors::Unimplemented(
        "Only support paddle::dialect::DenseTensorType or "
        "paddle::dialect::AllocatedDenseTensorType"));
  }

  paddle::dialect::IrTensor dense_out_grad(
      paddle::dialect::TransToPhiDataType(out_grad.dtype()),
      out_grad.dims(),
      out_grad.data_layout(),
      out_grad.lod(),
      out_grad.offset());

  VLOG(4) << "Builder construction  meta_x, meta_out_grad";
  paddle::dialect::IrMetaTensor meta_out_grad(&dense_out_grad);
  paddle::dialect::IrMetaTensor meta_x(&dense_x);

  paddle::dialect::IrTensor dense_x_grad;
  paddle::dialect::IrMetaTensor meta_x_grad(&dense_x_grad);

  phi::TensorToArrayInferMeta(
      meta_x, meta_out_grad, axis, use_stack, &meta_x_grad);

  std::vector<pir::Type> argument_outputs;
  pir::Type out_dense_tensor_array_type =
      paddle::dialect::DenseTensorArrayType::get(
          pir::IrContext::Instance(),
          paddle::dialect::TransToIrDataType(dense_x_grad.dtype()),
          dense_x_grad.dims(),
          dense_x_grad.layout());
  argument_outputs.push_back(out_dense_tensor_array_type);
  return argument_outputs;
}

OpInfoTuple SliceArrayOp::GetOpInfo() {
  std::vector<paddle::dialect::OpInputInfo> inputs = {
      paddle::dialect::OpInputInfo("input",
                                   "paddle::dialect::DenseTensorArrayType",
                                   false,
                                   false,
                                   false,
                                   false),
      paddle::dialect::OpInputInfo("starts",
                                   "paddle::dialect::IntArrayAttribute",
                                   false,
                                   false,
                                   true,
                                   false),
      paddle::dialect::OpInputInfo("ends",
                                   "paddle::dialect::IntArrayAttribute",
                                   false,
                                   false,
                                   true,
                                   false)};
  std::vector<paddle::dialect::OpAttributeInfo> attributes = {};
  std::vector<paddle::dialect::OpOutputInfo> outputs = {
      paddle::dialect::OpOutputInfo(
          "out", "paddle::dialect::DenseTensorArrayType", false, false)};
  paddle::dialect::OpRunTimeInfo run_time_info =
      paddle::dialect::OpRunTimeInfo("SliceArrayInferMeta",
                                     {"input", "starts", "ends"},
                                     "slice_array",
                                     {"input", "starts", "ends"},
                                     {},
                                     {},
                                     {},
                                     {});
  return std::make_tuple(
      inputs, attributes, outputs, run_time_info, "slice_array");
}

void SliceArrayOp::VerifySig() {
  VLOG(4)
      << "Start Verifying inputs, outputs and attributes for: SliceArrayOp.";
  VLOG(4) << "Verifying inputs:";
  {
    auto input_size = num_operands();
    IR_ENFORCE(input_size == 3u,
               "The size %d of inputs must be equal to 3.",
               input_size);
    IR_ENFORCE((*this)
                   ->operand_source(0)
                   .type()
                   .isa<paddle::dialect::DenseTensorArrayType>(),
               "Type validation failed for the 0th input, got %s.",
               (*this)->operand_source(0).type());
    IR_ENFORCE((*this)->operand_source(1).type().isa<pir::VectorType>() ||
                   (*this)
                       ->operand_source(1)
                       .type()
                       .isa<paddle::dialect::DenseTensorType>(),
               "Type validation failed for the 1st input, got %s.",
               (*this)->operand_source(1).type());
    IR_ENFORCE((*this)->operand_source(2).type().isa<pir::VectorType>() ||
                   (*this)
                       ->operand_source(2)
                       .type()
                       .isa<paddle::dialect::DenseTensorType>(),
               "Type validation failed for the 1st input, got %s.",
               (*this)->operand_source(2).type());
  }
  VLOG(4) << "Verifying outputs:";
  {
    auto output_size = num_results();
    IR_ENFORCE(output_size == 1u,
               "The size %d of outputs must be equal to 1.",
               output_size);
    IR_ENFORCE(
        (*this)->result(0).type().isa<paddle::dialect::DenseTensorArrayType>(),
        "Type validation failed for the 0th output.");
  }
  VLOG(4) << "End Verifying for: SliceArrayOp.";
}

phi::IntArray CalcSliceBoundsFromValue(pir::Value starts_or_ends) {
  phi::IntArray starts_or_ends_list;
  if (starts_or_ends.defining_op()->isa<paddle::dialect::FullIntArrayOp>()) {
    starts_or_ends_list = phi::IntArray(paddle::dialect::GetInt64Vector(
        starts_or_ends.defining_op()
            ->dyn_cast<paddle::dialect::FullIntArrayOp>()
            .attribute("value")));
  } else if (starts_or_ends.type().isa<pir::VectorType>()) {
    size_t starts_or_ends_size =
        starts_or_ends.type().dyn_cast<pir::VectorType>().size();
    starts_or_ends_list =
        phi::IntArray(std::vector<int64_t>(starts_or_ends_size, -1));
    starts_or_ends_list.SetFromTensor(true);
  } else if (starts_or_ends.type().isa<paddle::dialect::DenseTensorType>()) {
    common::DDim starts_or_ends_dim =
        starts_or_ends.type()
            .dyn_cast<paddle::dialect::DenseTensorType>()
            .dims();
    size_t starts_or_ends_size = common::product(starts_or_ends_dim);
    if (common::contain_unknown_dim(starts_or_ends_dim)) {
      starts_or_ends_size = 1;
    }
    starts_or_ends_list =
        phi::IntArray(std::vector<int64_t>(starts_or_ends_size, -1));
    starts_or_ends_list.SetFromTensor(true);
  } else {
    PADDLE_THROW(
        phi::errors::Unimplemented("Only support VectorType or DenseTensorType "
                                   "or AllocatedDenseTensorType"));
  }
  return starts_or_ends_list;
}

void SliceArrayOp::Build(pir::Builder &builder,             // NOLINT
                         pir::OperationArgument &argument,  // NOLINT
                         pir::Value input,
                         pir::Value starts,
                         pir::Value ends) {
  VLOG(4) << "Start build SliceArrayOp";
  VLOG(4) << "Builder construction inputs";
  std::vector<pir::Value> argument_inputs = {input, starts, ends};
  argument.AddInputs(argument_inputs);
  VLOG(4) << "Builder construction attributes";
  pir::AttributeMap argument_attributes = {};
  VLOG(4) << "Builder construction outputs";
  std::vector<pir::Type> argument_outputs =
      SliceArrayOp::InferMeta(argument_inputs, &argument_attributes);
  argument.AddOutputs(argument_outputs.begin(), argument_outputs.end());
  ::pir::PassStopGradientsDefaultly(argument);
}

void SliceArrayOp::InferMeta(phi::InferMetaContext *infer_meta) {
  auto fn = PD_INFER_META(phi::SliceArrayInferMeta);
  fn(infer_meta);
}

std::vector<pir::Type> SliceArrayOp::InferMeta(
    const std::vector<pir::Value> &input_values,
    pir::AttributeMap *p_attributes) {
  VLOG(4) << "Start infermeta SliceArrayOp";
  IR_ENFORCE(input_values.size() == 3,
             "Num of inputs is expected to be 3 but got %d.",
             input_values.size());
  pir::Value input = input_values[0];
  pir::Value starts = input_values[1];
  pir::Value ends = input_values[2];

  VLOG(4) << "Builder construction outputs";
  paddle::dialect::DenseTensorArrayType input_type;
  if (input.type().isa<paddle::dialect::DenseTensorArrayType>()) {
    input_type = input.type().dyn_cast<paddle::dialect::DenseTensorArrayType>();
  } else {
    PADDLE_THROW(phi::errors::Unimplemented(
        "Only support paddle::dialect::AllocatedDenseTensorArrayType or "
        "paddle::dialect::AllocatedDenseTensorArrayType"));
  }

  paddle::dialect::IrTensor dense_input(
      paddle::dialect::TransToPhiDataType(input_type.dtype()),
      input_type.dims(),
      input_type.data_layout(),
      {});
  paddle::dialect::IrMetaTensor meta_input(&dense_input);

  phi::IntArray starts_list = CalcSliceBoundsFromValue(starts);
  phi::IntArray ends_list = CalcSliceBoundsFromValue(ends);

  paddle::dialect::IrTensor dense_out;
  paddle::dialect::IrMetaTensor meta_out(&dense_out);

  phi::SliceArrayInferMeta(meta_input,
                           starts_list,
                           ends_list,
                           &meta_out,
                           phi::MetaConfig(false, false));

  std::vector<pir::Type> argument_outputs;
  pir::Type out_dense_tensor_array_type =
      paddle::dialect::DenseTensorArrayType::get(
          pir::IrContext::Instance(),
          TransToIrDataType(dense_out.dtype()),
          dense_out.dims(),
          dense_out.layout());
  argument_outputs.push_back(out_dense_tensor_array_type);
  return argument_outputs;
}

phi::DataType SliceArrayOp::GetKernelTypeForVar(
    const std::string &var_name,
    const phi::DataType &tensor_dtype,
    const phi::DataType &expected_kernel_dtype) {
  VLOG(4) << "Get KernelType for Var of op: SliceArrayOp";

  return expected_kernel_dtype;
}

OpInfoTuple SliceArrayDenseOp::GetOpInfo() {
  std::vector<paddle::dialect::OpInputInfo> inputs = {
      paddle::dialect::OpInputInfo("input",
                                   "paddle::dialect::DenseTensorArrayType",
                                   false,
                                   false,
                                   false,
                                   false),
      paddle::dialect::OpInputInfo("starts",
                                   "paddle::dialect::IntArrayAttribute",
                                   false,
                                   false,
                                   true,
                                   false)};

  std::vector<paddle::dialect::OpAttributeInfo> attributes = {};

  std::vector<paddle::dialect::OpOutputInfo> outputs = {
      paddle::dialect::OpOutputInfo(
          "out", "paddle::dialect::DenseTensorType", false, false)};
  paddle::dialect::OpRunTimeInfo run_time_info =
      paddle::dialect::OpRunTimeInfo("SliceArrayDenseInferMeta",
                                     {"input", "starts"},
                                     "slice_array_dense",
                                     {"input", "starts"},
                                     {"input"},
                                     {"input"},
                                     {},
                                     {});
  return std::make_tuple(
      inputs, attributes, outputs, run_time_info, "slice_array_dense");
}

void SliceArrayDenseOp::VerifySig() {
  VLOG(4)
      << "Start Verifying inputs, outputs and attributes for: SliceArrayOp.";
  VLOG(4) << "Verifying inputs:";
  {
    auto input_size = num_operands();
    IR_ENFORCE(input_size == 2u,
               "The size %d of inputs must be equal to 2.",
               input_size);
    IR_ENFORCE((*this)
                   ->operand_source(0)
                   .type()
                   .isa<paddle::dialect::DenseTensorArrayType>(),
               "Type validation failed for the 0th input, got %s.",
               (*this)->operand_source(0).type());
    IR_ENFORCE((*this)->operand_source(1).type().isa<pir::VectorType>() ||
                   (*this)
                       ->operand_source(1)
                       .type()
                       .isa<paddle::dialect::DenseTensorType>(),
               "Type validation failed for the 1st input, got %s.",
               (*this)->operand_source(1).type());
  }
  VLOG(4) << "Verifying outputs:";
  {
    auto output_size = num_results();
    IR_ENFORCE(output_size == 1u,
               "The size %d of outputs must be equal to 1.",
               output_size);
    IR_ENFORCE(
        (*this)->result(0).type().isa<paddle::dialect::DenseTensorType>(),
        "Type validation failed for the 0th output.");
  }
  VLOG(4) << "End Verifying for: SliceArrayOp.";
}

void SliceArrayDenseOp::Build(pir::Builder &builder,             // NOLINT
                              pir::OperationArgument &argument,  // NOLINT
                              pir::Value input,
                              pir::Value starts) {
  VLOG(4) << "Start build SliceArrayDenseOp";
  VLOG(4) << "Builder construction inputs";
  std::vector<pir::Value> argument_inputs = {input, starts};
  argument.AddInputs({input, starts});
  VLOG(4) << "Builder construction attributes";
  pir::AttributeMap argument_attributes = {};
  std::vector<pir::Type> argument_outputs =
      SliceArrayDenseOp::InferMeta(argument_inputs, &argument_attributes);

  argument.AddOutputs(argument_outputs.begin(), argument_outputs.end());
  ::pir::PassStopGradientsDefaultly(argument);
}

void SliceArrayDenseOp::InferMeta(phi::InferMetaContext *infer_meta) {
  auto fn = PD_INFER_META(phi::SliceArrayDenseInferMeta);
  fn(infer_meta);
}

std::vector<pir::Type> SliceArrayDenseOp::InferMeta(
    const std::vector<pir::Value> &input_values,
    pir::AttributeMap *p_attributes) {
  VLOG(4) << "Start infermeta SliceArrayDenseOp";
  IR_ENFORCE(input_values.size() == 2,
             "Num of inputs is expected to be 2 but got %d.",
             input_values.size());
  pir::Value input = input_values[0];
  pir::Value starts = input_values[1];

  VLOG(4) << "Builder construction outputs";
  paddle::dialect::DenseTensorArrayType input_type;
  if (input.type().isa<paddle::dialect::DenseTensorArrayType>()) {
    input_type = input.type().dyn_cast<paddle::dialect::DenseTensorArrayType>();
  } else {
    PADDLE_THROW(phi::errors::Unimplemented(
        "Only support paddle::dialect::DenseTensorArrayType or "
        "paddle::dialect::AllocatedDenseTensorArrayType"));
  }
  paddle::dialect::IrTensor dense_input(
      paddle::dialect::TransToPhiDataType(input_type.dtype()),
      input_type.dims(),
      input_type.data_layout(),
      {});
  paddle::dialect::IrMetaTensor meta_input(&dense_input);

  phi::IntArray starts_list = CalcSliceBoundsFromValue(starts);

  paddle::dialect::IrTensor dense_out;
  paddle::dialect::IrMetaTensor meta_out(&dense_out);

  phi::SliceArrayDenseInferMeta(
      meta_input, starts_list, &meta_out, phi::MetaConfig(false, false));

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

phi::DataType SliceArrayDenseOp::GetKernelTypeForVar(
    const std::string &var_name,
    const phi::DataType &tensor_dtype,
    const phi::DataType &expected_kernel_dtype) {
  VLOG(4) << "Get KernelType for Var of op: SliceArrayOp";

  return expected_kernel_dtype;
}

OpInfoTuple AssignArrayOp::GetOpInfo() {
  std::vector<paddle::dialect::OpInputInfo> inputs = {
      paddle::dialect::OpInputInfo("x",
                                   "paddle::dialect::DenseTensorArrayType",
                                   false,
                                   false,
                                   false,
                                   true)};
  std::vector<paddle::dialect::OpAttributeInfo> attributes = {};
  std::vector<paddle::dialect::OpOutputInfo> outputs = {
      paddle::dialect::OpOutputInfo(
          "out", "paddle::dialect::DenseTensorArrayType", false, false)};
  paddle::dialect::OpRunTimeInfo run_time_info = paddle::dialect::OpRunTimeInfo(
      "UnchangedArrayInferMeta", {"x"}, "assign_array", {"x"}, {}, {}, {}, {});
  return std::make_tuple(
      inputs, attributes, outputs, run_time_info, "assign_array");
}

void AssignArrayOp::Build(pir::Builder &builder,
                          pir::OperationArgument &argument,
                          pir::Value x_) {
  VLOG(4) << "Start build AssignArrayOp";

  VLOG(4) << "Builder construction inputs";
  std::vector<pir::Value> argument_inputs = {x_};
  argument.AddInputs(argument_inputs);

  VLOG(4) << "Builder construction attributes";
  pir::AttributeMap argument_attributes = {};

  VLOG(4) << "Builder construction outputs";
  std::vector<pir::Type> argument_outputs =
      AssignArrayOp::InferMeta(argument_inputs, &argument_attributes);
  argument.AddOutputs(argument_outputs.begin(), argument_outputs.end());
  ::pir::PassStopGradientsDefaultly(argument);
}

void AssignArrayOp::VerifySig() {
  VLOG(4)
      << "Start Verifying inputs, outputs and attributes for: AssignArrayOp.";
  VLOG(4) << "Verifying inputs:";
  {
    auto input_size = num_operands();
    IR_ENFORCE(input_size == 1u,
               "The size %d of inputs must be equal to 1.",
               input_size);
    IR_ENFORCE((*this)
                   ->operand_source(0)
                   .type()
                   .isa<paddle::dialect::DenseTensorArrayType>(),
               "Type validation failed for the 0th input, got %s.",
               (*this)->operand_source(0).type());
  }
  VLOG(4) << "Verifying attributes:";
  {
    // Attributes num is 0, not need to check attributes type.
  }
  VLOG(4) << "Verifying outputs:";
  {
    auto output_size = num_results();
    IR_ENFORCE(output_size == 1u,
               "The size %d of outputs must be equal to 1.",
               output_size);
    IR_ENFORCE(
        (*this)->result(0).type().isa<paddle::dialect::DenseTensorArrayType>(),
        "Type validation failed for the 0th output.");
  }
  VLOG(4) << "End Verifying for: AssignArrayOp.";
}

void AssignArrayOp::InferMeta(phi::InferMetaContext *infer_meta) {
  auto fn = PD_INFER_META(phi::UnchangedArrayInferMeta);
  fn(infer_meta);
}

phi::DataType AssignArrayOp::GetKernelTypeForVar(
    const std::string &var_name,
    const phi::DataType &tensor_dtype,
    const phi::DataType &expected_kernel_dtype) {
  VLOG(4) << "Get KernelType for Var of op: AssignArrayOp";

  return expected_kernel_dtype;
}

std::vector<pir::Type> AssignArrayOp::InferMeta(
    const std::vector<pir::Value> &input_values,
    pir::AttributeMap *p_attributes) {
  VLOG(4) << "Start infermeta AssignArrayOp";
  IR_ENFORCE(input_values.size() == 1,
             "Num of inputs is expected to be 1 but got %d.",
             input_values.size());
  pir::Value x_ = input_values[0];

  VLOG(4) << "Builder construction outputs";
  paddle::dialect::DenseTensorArrayType x_type;
  if (x_.type().isa<paddle::dialect::DenseTensorArrayType>()) {
    x_type = x_.type().dyn_cast<paddle::dialect::DenseTensorArrayType>();
  } else {
    PADDLE_THROW(phi::errors::Unimplemented(
        "Only support paddle::dialect::DenseTensorArrayType or "
        "paddle::dialect::AllocatedDenseTensorArrayType"));
  }
  paddle::dialect::IrTensor dense_input(
      paddle::dialect::TransToPhiDataType(x_type.dtype()),
      x_type.dims(),
      x_type.data_layout(),
      {});
  paddle::dialect::IrMetaTensor meta_input(&dense_input);

  paddle::dialect::IrTensor dense_out;
  paddle::dialect::IrMetaTensor meta_out(&dense_out);

  phi::UnchangedArrayInferMeta(meta_input, &meta_out);

  std::vector<pir::Type> argument_outputs;
  pir::Type out_dense_tensor_array_type =
      paddle::dialect::DenseTensorArrayType::get(
          pir::IrContext::Instance(),
          TransToIrDataType(dense_out.dtype()),
          dense_out.dims(),
          dense_out.layout());
  argument_outputs.push_back(out_dense_tensor_array_type);
  return argument_outputs;
}

OpInfoTuple AssignArray_Op::GetOpInfo() {
  std::vector<paddle::dialect::OpInputInfo> inputs = {
      paddle::dialect::OpInputInfo("x",
                                   "paddle::dialect::DenseTensorArrayType",
                                   false,
                                   false,
                                   false,
                                   false)};
  std::vector<paddle::dialect::OpAttributeInfo> attributes = {};
  std::vector<paddle::dialect::OpOutputInfo> outputs = {
      paddle::dialect::OpOutputInfo(
          "out", "paddle::dialect::DenseTensorArrayType", false, false)};
  paddle::dialect::OpRunTimeInfo run_time_info =
      paddle::dialect::OpRunTimeInfo("UnchangedArrayInferMeta",
                                     {"x"},
                                     "assign_array",
                                     {"x"},
                                     {},
                                     {},
                                     {{"out", "x"}},
                                     {});
  return std::make_tuple(
      inputs, attributes, outputs, run_time_info, "assign_array");
}

void AssignArray_Op::VerifySig() {
  VLOG(4)
      << "Start Verifying inputs, outputs and attributes for: AssignArray_Op.";
  VLOG(4) << "Verifying inputs:";
  {
    auto input_size = num_operands();
    IR_ENFORCE(input_size == 1u,
               "The size %d of inputs must be equal to 1.",
               input_size);
    IR_ENFORCE((*this)
                   ->operand_source(0)
                   .type()
                   .isa<paddle::dialect::DenseTensorArrayType>(),
               "Type validation failed for the 0th input, but got %s.",
               (*this)->operand_source(0).type());
  }
  VLOG(4) << "Verifying attributes:";
  VLOG(4) << "Verifying outputs:";
  {
    auto output_size = num_results();
    IR_ENFORCE(output_size == 1u,
               "The size %d of outputs must be equal to 1.",
               output_size);
    IR_ENFORCE(
        (*this)->result(0).type().isa<paddle::dialect::DenseTensorArrayType>(),
        "Type validation failed for the 0th output.");
  }
  VLOG(4) << "End Verifying for: AssignArray_Op.";
}

void AssignArray_Op::InferMeta(phi::InferMetaContext *infer_meta) {
  auto fn = PD_INFER_META(phi::UnchangedArrayInferMeta);
  fn(infer_meta);
}

std::vector<pir::Type> AssignArray_Op::InferMeta(
    const std::vector<pir::Value> &input_values,
    pir::AttributeMap *p_attributes) {
  VLOG(4) << "Start infermeta AssignArray_Op";
  IR_ENFORCE(input_values.size() == 1,
             "Num of inputs is expected to be 1 but got %d.",
             input_values.size());
  pir::Value x_ = input_values[0];

  VLOG(4) << "Builder construction outputs";
  paddle::dialect::DenseTensorArrayType x_type;
  if (x_.type().isa<paddle::dialect::DenseTensorArrayType>()) {
    x_type = x_.type().dyn_cast<paddle::dialect::DenseTensorArrayType>();
  } else {
    PADDLE_THROW(phi::errors::Unimplemented(
        "Only support paddle::dialect::DenseTensorArrayType or "
        "paddle::dialect::AllocatedDenseTensorArrayType"));
  }
  paddle::dialect::IrTensor dense_input(
      paddle::dialect::TransToPhiDataType(x_type.dtype()),
      x_type.dims(),
      x_type.data_layout(),
      {});
  paddle::dialect::IrMetaTensor meta_input(&dense_input);

  paddle::dialect::IrTensor dense_out;
  paddle::dialect::IrMetaTensor meta_out(&dense_out);

  phi::UnchangedArrayInferMeta(meta_input, &meta_out);

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

phi::DataType AssignArray_Op::GetKernelTypeForVar(
    const std::string &var_name,
    const phi::DataType &tensor_dtype,
    const phi::DataType &expected_kernel_dtype) {
  VLOG(4) << "Get KernelType for Var of op: AssignArray_Op";

  return expected_kernel_dtype;
}

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
  std::vector<paddle::dialect::OpAttributeInfo> attributes = {};
  std::vector<paddle::dialect::OpOutputInfo> outputs = {
      paddle::dialect::OpOutputInfo(
          "out", "paddle::dialect::DenseTensorType", false, false)};
  paddle::dialect::OpRunTimeInfo run_time_info =
      paddle::dialect::OpRunTimeInfo("ExpandInferMeta",
                                     {"x", "shape"},
                                     "expand",
                                     {"x", "shape"},
                                     {"x"},
                                     {},
                                     {},
                                     {});
  return std::make_tuple(inputs, attributes, outputs, run_time_info, "expand");
}

void ExpandOp::Build(pir::Builder &builder,
                     pir::OperationArgument &argument,
                     pir::Value x_,
                     const std::vector<int64_t> &shape) {
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
  std::vector<pir::Type> argument_outputs =
      ExpandOp::InferMeta(argument_inputs, &argument_attributes);

  argument.AddOutputs(argument_outputs.begin(), argument_outputs.end());
  ::pir::PassStopGradientsDefaultly(argument);
}

void ExpandOp::Build(pir::Builder &builder,
                     pir::OperationArgument &argument,
                     pir::Value x_,
                     pir::AttributeMap attributes) {
  VLOG(4) << "Start build ExpandOp";

  IR_ENFORCE(attributes.find("shape") != attributes.end(),
             "'shape' Attribute is expected for ExpandOp. ");
  std::vector<int64_t> shape =
      attributes.at("shape")
          .dyn_cast<paddle::dialect::IntArrayAttribute>()
          .data()
          .GetData();

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
  std::vector<pir::Type> argument_outputs =
      ExpandOp::InferMeta(argument_inputs, &argument_attributes);

  argument.AddOutputs(argument_outputs.begin(), argument_outputs.end());
  ::pir::PassStopGradientsDefaultly(argument);
}

void ExpandOp::Build(pir::Builder &builder,
                     pir::OperationArgument &argument,
                     pir::Value x_,
                     pir::Value shape_) {
  VLOG(4) << "Start build ExpandOp";

  VLOG(4) << "Builder construction inputs";
  std::vector<pir::Value> argument_inputs = {x_, shape_};
  argument.AddInputs(argument_inputs);

  VLOG(4) << "Builder construction attributes";
  pir::AttributeMap argument_attributes = {};
  std::vector<pir::Type> argument_outputs =
      ExpandOp::InferMeta(argument_inputs, &argument_attributes);

  argument.AddOutputs(argument_outputs.begin(), argument_outputs.end());
  ::pir::PassStopGradientsDefaultly(argument);
}

bool ExpandOp::InferSymbolicShape(
    pir::ShapeConstraintIRAnalysis *shape_analysis) {
  const auto &x_shape_or_data = shape_analysis->GetShapeOrDataForValue(x());
  const auto &expand_shape_shape_or_data =
      shape_analysis->GetShapeOrDataForValue(shape());

  const std::vector<symbol::DimExpr> &x_dims = [&] {
    std::vector<symbol::DimExpr> dims;
    if (x_shape_or_data.data().has_value()) {
      dims = x_shape_or_data.data().value();
    } else {
      dims = x_shape_or_data.shape();
    }
    return dims;
  }();

  const std::vector<symbol::DimExpr> &expand_shape = [&] {
    std::vector<symbol::DimExpr> dims;

    if (expand_shape_shape_or_data
            .isa<symbol::TensorListShapeOrDataDimExprs>()) {
      const auto &dims_list =
          expand_shape_shape_or_data
              .dyn_cast<symbol::TensorListShapeOrDataDimExprs>();
      for (const auto &shape_data : dims_list) {
        const auto &dim_expr = shape_data.data().has_value()
                                   ? shape_data.data().value()[0]
                                   : shape_data.shape()[0];
        dims.emplace_back(dim_expr);
      }
    } else {
      dims = expand_shape_shape_or_data.data().has_value()
                 ? expand_shape_shape_or_data.data().value()
                 : expand_shape_shape_or_data.shape();
    }
    if (dims.empty()) {
      dims = std::vector<symbol::DimExpr>(x_dims.size(), -1);
    }
    return dims;
  }();

  std::vector<symbol::DimExpr> out_shape = expand_shape;
  for (size_t i = 0; i < expand_shape.size(); i++) {
    if (expand_shape[i] == -1) {  // copy the dim from x
      // the shape is right aligned
      int index = i - (expand_shape.size() - x_dims.size());
      IR_ENFORCE(index >= 0,
                 "in ExpandOpInferSymbolicShape, the dim to copy must >= 0, "
                 "but got %d",
                 index);

      out_shape[i] = x_dims[index];
    }
  }

  shape_analysis->SetShapeOrDataForValue(
      out(),
      symbol::ShapeOrDataDimExprs{
          symbol::TensorShapeOrDataDimExprs(out_shape)});

  return true;
}

void ExpandOp::VerifySig() {
  VLOG(4) << "Start Verifying inputs, outputs and attributes for: ExpandOp.";
  VLOG(4) << "Verifying inputs:";
  {
    auto input_size = num_operands();
    IR_ENFORCE(input_size == 2u,
               "The size %d of inputs must be equal to 2.",
               input_size);
    IR_ENFORCE((*this)
                   ->operand_source(0)
                   .type()
                   .isa<paddle::dialect::DenseTensorType>(),
               "Type validation failed for the 0th input.");
    if (auto vec_type =
            (*this)->operand_source(1).type().dyn_cast<pir::VectorType>()) {
      for (size_t i = 0; i < vec_type.size(); ++i) {
        IR_ENFORCE(vec_type[i].isa<paddle::dialect::DenseTensorType>(),
                   "Type validation failed for the 1th input.");
      }
    } else {
      IR_ENFORCE((*this)
                     ->operand_source(1)
                     .type()
                     .isa<paddle::dialect::DenseTensorType>(),
                 "Type validation failed for the 1th input.");
    }
  }
  VLOG(4) << "Verifying attributes:";
  {
    // Attributes num is 0, not need to check attributes type.
  }
  VLOG(4) << "Verifying outputs:";
  {
    auto output_size = num_results();
    IR_ENFORCE(output_size == 1u,
               "The size %d of outputs must be equal to 1.",
               output_size);
    IR_ENFORCE(
        (*this)->result(0).type().isa<paddle::dialect::DenseTensorType>(),
        "Type validation failed for the 0th output.");
  }
  VLOG(4) << "End Verifying for: ExpandOp.";
}

void ExpandOp::InferMeta(phi::InferMetaContext *infer_meta) {
  auto fn = PD_INFER_META(phi::ExpandInferMeta);
  fn(infer_meta);
}

std::vector<pir::Type> ExpandOp::InferMeta(
    const std::vector<pir::Value> &input_values,
    pir::AttributeMap *p_attributes) {
  VLOG(4) << "Start infermeta ExpandOp";
  IR_ENFORCE(input_values.size() == 2,
             "Num of inputs is expected to be 2 but got %d.",
             input_values.size());
  pir::Value x_ = input_values[0];
  pir::Value shape_ = input_values[1];

  VLOG(4) << "Builder construction outputs";
  bool is_from_tensor = false;
  paddle::dialect::DenseTensorType x;
  if (x_.type().isa<paddle::dialect::DenseTensorType>()) {
    x = x_.type().dyn_cast<paddle::dialect::DenseTensorType>();
  } else {
    PADDLE_THROW(phi::errors::Unimplemented(
        "Only support paddle::dialect::DenseTensorType or "
        "paddle::dialect::AllocatedDenseTensorType"));
  }

  std::function<std::vector<int64_t>(const pir::Value &, bool *)>
      ParseValueShape = [&](const pir::Value &shape,
                            bool *is_from_tensor) -> std::vector<int64_t> {
    std::vector<int64_t> vec_shape;
    if (shape.isa<pir::OpResult>() &&
        shape.defining_op()->isa<paddle::dialect::FullIntArrayOp>()) {
      vec_shape = paddle::dialect::GetInt64Vector(
          shape.defining_op()
              ->dyn_cast<paddle::dialect::FullIntArrayOp>()
              .attribute("value"));
    } else if (shape.isa<pir::OpResult>() &&
               shape.defining_op()->isa<paddle::dialect::FullOp>()) {
      auto shape_item = shape.defining_op()
                            ->dyn_cast<paddle::dialect::FullOp>()
                            .attribute("value")
                            .dyn_cast<pir::FloatAttribute>()
                            .data();
      vec_shape = {static_cast<int64_t>(shape_item)};
    } else if (shape.isa<pir::OpResult>() &&
               shape.defining_op()->isa<paddle::dialect::StackOp>()) {
      std::vector<pir::Value> inputs = shape.defining_op()
                                           ->operand_source(0)
                                           .defining_op()
                                           ->operands_source();
      for (auto item : inputs) {
        auto tmp = ParseValueShape(item, is_from_tensor);
        vec_shape.insert(vec_shape.end(), tmp.begin(), tmp.end());
      }
    } else if (shape.type().isa<pir::VectorType>()) {
      size_t shape_size = shape.type().dyn_cast<pir::VectorType>().size();
      vec_shape = std::vector<int64_t>(shape_size, -2);
      *is_from_tensor = true;
    } else if (shape.type().isa<paddle::dialect::DenseTensorType>()) {
      common::DDim shape_dim =
          shape.type().dyn_cast<paddle::dialect::DenseTensorType>().dims();
      size_t shape_size = common::product(shape_dim);
      if (common::contain_unknown_dim(shape_dim)) {
        shape_size = 1;
      }
      vec_shape = std::vector<int64_t>(shape_size, -2);
      *is_from_tensor = true;
    } else {
      PADDLE_THROW(phi::errors::Unimplemented(
          "Only support VectorType or DenseTensorType "
          "or AllocatedDenseTensorType"));
    }
    return vec_shape;
  };

  is_from_tensor = false;
  phi::IntArray shape = phi::IntArray(ParseValueShape(shape_, &is_from_tensor));
  if (is_from_tensor) shape.SetFromTensor(true);

  VLOG(4) << "Builder construction  dense_x";
  paddle::dialect::IrTensor ir_meta_tensor_x(
      paddle::dialect::TransToPhiDataType(x.dtype()),
      x.dims(),
      x.data_layout(),
      x.lod(),
      x.offset());
  VLOG(4) << "Builder construction  meta_x";
  paddle::dialect::IrMetaTensor meta_x(&ir_meta_tensor_x);
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
    const std::string &var_name,
    const phi::DataType &tensor_dtype,
    const phi::DataType &expected_kernel_dtype) {
  VLOG(4) << "Get KernelType for Var of op: ExpandOp";
  return expected_kernel_dtype;
}

const char *IncrementOp::attributes_name[1] = {"value"};

OpInfoTuple IncrementOp::GetOpInfo() {
  std::vector<paddle::dialect::OpInputInfo> inputs = {
      paddle::dialect::OpInputInfo(
          "x", "paddle::dialect::DenseTensorType", false, false, false, true)};
  std::vector<paddle::dialect::OpAttributeInfo> attributes = {
      paddle::dialect::OpAttributeInfo("value", "pir::FloatAttribute", "")};
  std::vector<paddle::dialect::OpOutputInfo> outputs = {
      paddle::dialect::OpOutputInfo(
          "out", "paddle::dialect::DenseTensorType", false, false)};
  paddle::dialect::OpRunTimeInfo run_time_info =
      paddle::dialect::OpRunTimeInfo("IncrementInferMeta",
                                     {"x", "value"},
                                     "increment",
                                     {"x", "value"},
                                     {},
                                     {},
                                     {},
                                     {});
  return std::make_tuple(
      inputs, attributes, outputs, run_time_info, "increment");
}

void IncrementOp::Build(pir::Builder &builder,
                        pir::OperationArgument &argument,
                        pir::Value x_,
                        float value) {
  VLOG(4) << "Start build IncrementOp";

  VLOG(4) << "Builder construction inputs";
  std::vector<pir::Value> argument_inputs = {x_};
  argument.AddInputs(argument_inputs);

  VLOG(4) << "Builder construction attributes";
  pir::AttributeMap argument_attributes = {};
  pir::Attribute attr_value =
      pir::FloatAttribute::get(pir::IrContext::Instance(), value);
  argument.AddAttribute("value", attr_value);
  argument_attributes.insert({"value", attr_value});
  std::vector<pir::Type> argument_outputs =
      IncrementOp::InferMeta(argument_inputs, &argument_attributes);

  argument.AddOutputs(argument_outputs.begin(), argument_outputs.end());
  ::pir::PassStopGradientsDefaultly(argument);
}

void IncrementOp::Build(pir::Builder &builder,
                        pir::OperationArgument &argument,
                        pir::Value x_,
                        pir::AttributeMap attributes) {
  VLOG(4) << "Start build IncrementOp";

  IR_ENFORCE(attributes.find("value") != attributes.end(),
             "'value' Attribute is expected for IncrementOp. ");
  float value = attributes.at("value").dyn_cast<pir::FloatAttribute>().data();

  VLOG(4) << "Builder construction inputs";
  std::vector<pir::Value> argument_inputs = {x_};
  argument.AddInputs(argument_inputs);

  VLOG(4) << "Builder construction attributes";
  pir::AttributeMap argument_attributes = {};
  pir::Attribute attr_value =
      pir::FloatAttribute::get(pir::IrContext::Instance(), value);
  argument.AddAttribute("value", attr_value);
  argument_attributes.insert({"value", attr_value});
  std::vector<pir::Type> argument_outputs =
      IncrementOp::InferMeta(argument_inputs, &argument_attributes);

  argument.AddOutputs(argument_outputs.begin(), argument_outputs.end());
  ::pir::PassStopGradientsDefaultly(argument);
}

void IncrementOp::VerifySig() {
  VLOG(4) << "Start Verifying inputs, outputs and attributes for: IncrementOp.";
  VLOG(4) << "Verifying inputs:";
  {
    auto input_size = num_operands();
    IR_ENFORCE(input_size == 1u,
               "The size %d of inputs must be equal to 1.",
               input_size);
    IR_ENFORCE((*this)
                   ->operand_source(0)
                   .type()
                   .isa<paddle::dialect::DenseTensorType>(),
               "Type validation failed for the 0th input, got %s.",
               (*this)->operand_source(0).type());
  }
  VLOG(4) << "Verifying attributes:";
  {
    auto &attributes = this->attributes();
    IR_ENFORCE(attributes.count("value") > 0, "value does not exist.");
    IR_ENFORCE(attributes.at("value").isa<pir::FloatAttribute>(),
               "Type of attribute: value is not pir::FloatAttribute.");
  }
  VLOG(4) << "Verifying outputs:";
  {
    auto output_size = num_results();
    IR_ENFORCE(output_size == 1u,
               "The size %d of outputs must be equal to 1.",
               output_size);
    IR_ENFORCE(
        (*this)->result(0).type().isa<paddle::dialect::DenseTensorType>(),
        "Type validation failed for the 0th output.");
  }
  VLOG(4) << "End Verifying for: IncrementOp.";
}

void IncrementOp::InferMeta(phi::InferMetaContext *infer_meta) {
  auto fn = PD_INFER_META(phi::IncrementInferMeta);
  fn(infer_meta);
}

std::vector<pir::Type> IncrementOp::InferMeta(
    const std::vector<pir::Value> &input_values,
    pir::AttributeMap *p_attributes) {
  PADDLE_ENFORCE_NOT_NULL(
      p_attributes,
      common::errors::Fatal(
          "AttrtibueMap pointer in InferMeta function is nullptr."));
  auto &attributes = *p_attributes;
  VLOG(4) << "Start infermeta IncrementOp";
  IR_ENFORCE(input_values.size() == 1,
             "Num of inputs is expected to be 1 but got %d.",
             input_values.size());
  pir::Value x_ = input_values[0];

  IR_ENFORCE(attributes.find("value") != attributes.end(),
             "'value' Attribute is expected for IncrementOp. ");
  float value = attributes.at("value").dyn_cast<pir::FloatAttribute>().data();

  VLOG(4) << "Builder construction outputs";
  paddle::dialect::DenseTensorType x;
  if (x_.type().isa<paddle::dialect::DenseTensorType>()) {
    x = x_.type().dyn_cast<paddle::dialect::DenseTensorType>();
  } else {
    PADDLE_THROW(phi::errors::Unimplemented(
        "Only support paddle::dialect::DenseTensorType or "
        "paddle::dialect::AllocatedDenseTensorType"));
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

  phi::IncrementInferMeta(meta_x, value, &meta_out);

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

phi::DataType IncrementOp::GetKernelTypeForVar(
    const std::string &var_name,
    const phi::DataType &tensor_dtype,
    const phi::DataType &expected_kernel_dtype) {
  VLOG(4) << "Get KernelType for Var of op: IncrementOp";

  return expected_kernel_dtype;
}

bool IncrementOp::InferSymbolicShape(
    pir::ShapeConstraintIRAnalysis *shape_analysis) {
  const symbol::ShapeOrDataDimExprs &operand_shape_or_data =
      shape_analysis->GetShapeOrDataForValue(x());
  shape_analysis->SetShapeOrDataForValue(out(), operand_shape_or_data);
  return true;
}

const char *Increment_Op::attributes_name[1] = {"value"};

OpInfoTuple Increment_Op::GetOpInfo() {
  std::vector<paddle::dialect::OpInputInfo> inputs = {
      paddle::dialect::OpInputInfo(
          "x", "paddle::dialect::DenseTensorType", false, false, false, false)};
  std::vector<paddle::dialect::OpAttributeInfo> attributes = {
      paddle::dialect::OpAttributeInfo("value", "pir::FloatAttribute", "")};
  std::vector<paddle::dialect::OpOutputInfo> outputs = {
      paddle::dialect::OpOutputInfo(
          "out", "paddle::dialect::DenseTensorType", false, false)};
  paddle::dialect::OpRunTimeInfo run_time_info =
      paddle::dialect::OpRunTimeInfo("IncrementInferMeta",
                                     {"x", "value"},
                                     "increment",
                                     {"x", "value"},
                                     {},
                                     {},
                                     {{"out", "x"}},
                                     {});
  return std::make_tuple(
      inputs, attributes, outputs, run_time_info, "increment");
}

void Increment_Op::Build(pir::Builder &builder,
                         pir::OperationArgument &argument,
                         pir::Value x_,
                         float value) {
  VLOG(4) << "Start build Increment_Op";

  VLOG(4) << "Builder construction inputs";
  std::vector<pir::Value> argument_inputs = {x_};
  argument.AddInputs(argument_inputs);

  VLOG(4) << "Builder construction attributes";
  pir::AttributeMap argument_attributes = {};
  pir::Attribute attr_value =
      pir::FloatAttribute::get(pir::IrContext::Instance(), value);
  argument.AddAttribute("value", attr_value);
  argument_attributes.insert({"value", attr_value});
  std::vector<pir::Type> argument_outputs =
      Increment_Op::InferMeta(argument_inputs, &argument_attributes);

  argument.AddOutputs(argument_outputs.begin(), argument_outputs.end());
  ::pir::PassStopGradientsDefaultly(argument);
}

void Increment_Op::Build(pir::Builder &builder,
                         pir::OperationArgument &argument,
                         pir::Value x_,
                         pir::AttributeMap attributes) {
  VLOG(4) << "Start build Increment_Op";

  IR_ENFORCE(attributes.find("value") != attributes.end(),
             "'value' Attribute is expected for Increment_Op. ");
  float value = attributes.at("value").dyn_cast<pir::FloatAttribute>().data();

  VLOG(4) << "Builder construction inputs";
  std::vector<pir::Value> argument_inputs = {x_};
  argument.AddInputs(argument_inputs);

  VLOG(4) << "Builder construction attributes";
  pir::AttributeMap argument_attributes = {};
  pir::Attribute attr_value =
      pir::FloatAttribute::get(pir::IrContext::Instance(), value);
  argument.AddAttribute("value", attr_value);
  argument_attributes.insert({"value", attr_value});
  std::vector<pir::Type> argument_outputs =
      Increment_Op::InferMeta(argument_inputs, &argument_attributes);

  argument.AddOutputs(argument_outputs.begin(), argument_outputs.end());
  ::pir::PassStopGradientsDefaultly(argument);
}

void Increment_Op::VerifySig() {
  VLOG(4)
      << "Start Verifying inputs, outputs and attributes for: Increment_Op.";
  VLOG(4) << "Verifying inputs:";
  {
    auto input_size = num_operands();
    IR_ENFORCE(input_size == 1u,
               "The size %d of inputs must be equal to 1.",
               input_size);
    IR_ENFORCE((*this)
                   ->operand_source(0)
                   .type()
                   .isa<paddle::dialect::DenseTensorType>(),
               "Type validation failed for the 0th input, got %s.",
               (*this)->operand_source(0).type());
  }
  VLOG(4) << "Verifying attributes:";
  {
    auto &attributes = this->attributes();
    IR_ENFORCE(attributes.count("value") > 0, "value does not exist.");
    IR_ENFORCE(attributes.at("value").isa<pir::FloatAttribute>(),
               "Type of attribute: value is not pir::FloatAttribute.");
  }
  VLOG(4) << "Verifying outputs:";
  {
    auto output_size = num_results();
    IR_ENFORCE(output_size == 1u,
               "The size %d of outputs must be equal to 1.",
               output_size);
    IR_ENFORCE(
        (*this)->result(0).type().isa<paddle::dialect::DenseTensorType>(),
        "Type validation failed for the 0th output.");
  }
  VLOG(4) << "End Verifying for: Increment_Op.";
}

void Increment_Op::InferMeta(phi::InferMetaContext *infer_meta) {
  auto fn = PD_INFER_META(phi::IncrementInferMeta);
  fn(infer_meta);
}

std::vector<pir::Type> Increment_Op::InferMeta(
    const std::vector<pir::Value> &input_values,
    pir::AttributeMap *p_attributes) {
  PADDLE_ENFORCE_NOT_NULL(
      p_attributes,
      common::errors::Fatal(
          "AttrtibueMap pointer in InferMeta function is nullptr."));
  auto &attributes = *p_attributes;
  VLOG(4) << "Start infermeta Increment_Op";
  IR_ENFORCE(input_values.size() == 1,
             "Num of inputs is expected to be 1 but got %d.",
             input_values.size());
  pir::Value x_ = input_values[0];

  IR_ENFORCE(attributes.find("value") != attributes.end(),
             "'value' Attribute is expected for Increment_Op. ");
  float value = attributes.at("value").dyn_cast<pir::FloatAttribute>().data();

  VLOG(4) << "Builder construction outputs";
  paddle::dialect::DenseTensorType x;
  if (x_.type().isa<paddle::dialect::DenseTensorType>()) {
    x = x_.type().dyn_cast<paddle::dialect::DenseTensorType>();
  } else {
    PADDLE_THROW(phi::errors::Unimplemented(
        "Only support paddle::dialect::DenseTensorType or "
        "paddle::dialect::AllocatedDenseTensorType"));
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

  phi::IncrementInferMeta(meta_x, value, &meta_out);

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

phi::DataType Increment_Op::GetKernelTypeForVar(
    const std::string &var_name,
    const phi::DataType &tensor_dtype,
    const phi::DataType &expected_kernel_dtype) {
  VLOG(4) << "Get KernelType for Var of op: Increment_Op";

  return expected_kernel_dtype;
}

bool Increment_Op::InferSymbolicShape(
    pir::ShapeConstraintIRAnalysis *shape_analysis) {
  const symbol::ShapeOrDataDimExprs &operand_shape_or_data =
      shape_analysis->GetShapeOrDataForValue(x());
  shape_analysis->SetShapeOrDataForValue(out(), operand_shape_or_data);
  return true;
}

OpInfoTuple AssignOut_Op::GetOpInfo() {
  std::vector<paddle::dialect::OpInputInfo> inputs = {
      paddle::dialect::OpInputInfo(
          "x", "paddle::dialect::DenseTensorType", false, false, false, true),
      paddle::dialect::OpInputInfo("output",
                                   "paddle::dialect::DenseTensorType",
                                   false,
                                   false,
                                   false,
                                   true)};
  std::vector<paddle::dialect::OpAttributeInfo> attributes = {};
  std::vector<paddle::dialect::OpOutputInfo> outputs = {
      paddle::dialect::OpOutputInfo(
          "out", "paddle::dialect::DenseTensorType", false, false)};
  paddle::dialect::OpRunTimeInfo run_time_info =
      paddle::dialect::OpRunTimeInfo("UnchangedInferMeta",
                                     {"x"},
                                     "assign",
                                     {"x"},
                                     {},
                                     {},
                                     {{"out", "output"}},
                                     {});
  return std::make_tuple(
      inputs, attributes, outputs, run_time_info, "assign_out_");
}

void AssignOut_Op::Build(pir::Builder &builder,
                         pir::OperationArgument &argument,
                         pir::Value x_,
                         pir::Value output_) {
  VLOG(4) << "Start build AssignOut_Op";

  VLOG(4) << "Builder construction inputs";
  std::vector<pir::Value> argument_inputs = {x_, output_};
  argument.AddInputs(argument_inputs);

  VLOG(4) << "Builder construction attributes";
  pir::AttributeMap argument_attributes = {};

  std::vector<pir::Type> argument_outputs =
      AssignOut_Op::InferMeta(argument_inputs, &argument_attributes);
  argument.AddOutputs(argument_outputs.begin(), argument_outputs.end());
  constexpr char kStopGradientAttrName[] = "stop_gradient";
  auto stop_gradient0 =
      argument.inputs[0].attribute<pir::BoolAttribute>(kStopGradientAttrName);
  argument.inputs[1].set_attribute(kStopGradientAttrName,
                                   builder.bool_attr(stop_gradient0.data()));
}

void AssignOut_Op::VerifySig() {
  VLOG(4)
      << "Start Verifying inputs, outputs and attributes for: AssignOut_Op.";
  VLOG(4) << "Verifying inputs:";
  {
    auto input_size = num_operands();
    IR_ENFORCE(input_size == 2u,
               "The size %d of inputs must be equal to 2.",
               input_size);
    IR_ENFORCE((*this)
                   ->operand_source(0)
                   .type()
                   .isa<paddle::dialect::DenseTensorType>(),
               "Type validation failed for the 0th input, got %s.",
               (*this)->operand_source(0).type());
    IR_ENFORCE((*this)
                   ->operand_source(1)
                   .type()
                   .isa<paddle::dialect::DenseTensorType>(),
               "Type validation failed for the 1th input, got %s.",
               (*this)->operand_source(1).type());
  }
  VLOG(4) << "Verifying attributes:";
  {
    // Attributes num is 0, not need to check attributes type.
  }
  VLOG(4) << "Verifying outputs:";
  {
    auto output_size = num_results();
    IR_ENFORCE(output_size == 1u,
               "The size %d of outputs must be equal to 1.",
               output_size);
    IR_ENFORCE(
        (*this)->result(0).type().isa<paddle::dialect::DenseTensorType>(),
        "Type validation failed for the 0th output.");
  }
  VLOG(4) << "End Verifying for: AssignOut_Op.";
}

void AssignOut_Op::InferMeta(phi::InferMetaContext *infer_meta) {
  auto fn = PD_INFER_META(phi::UnchangedInferMeta);
  fn(infer_meta);
}

std::vector<pir::Type> AssignOut_Op::InferMeta(
    const std::vector<pir::Value> &input_values,
    pir::AttributeMap *p_attributes) {
  IR_ENFORCE(input_values.size() == 2,
             "Num of inputs is expected to be 2 but got %d.",
             input_values.size());

  pir::Value x_ = input_values[0];
  VLOG(4) << "Builder construction outputs";

  paddle::dialect::DenseTensorType x;
  if (x_.type().isa<paddle::dialect::DenseTensorType>()) {
    x = x_.type().dyn_cast<paddle::dialect::DenseTensorType>();
  } else {
    PADDLE_THROW(phi::errors::Unimplemented(
        "Only support paddle::dialect::DenseTensorType or "
        "paddle::dialect::AllocatedDenseTensorType"));
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

  phi::UnchangedInferMeta(meta_x, &meta_out);

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

phi::DataType AssignOut_Op::GetKernelTypeForVar(
    const std::string &var_name,
    const phi::DataType &tensor_dtype,
    const phi::DataType &expected_kernel_dtype) {
  VLOG(4) << "Get KernelType for Var of op: AssignOut_Op";

  return expected_kernel_dtype;
}

void ShapeBroadcastOp::Build(pir::Builder &builder,
                             pir::OperationArgument &argument,
                             pir::Value x_,
                             pir::Value y_) {
  VLOG(4) << "Start build ShapeBroadcastOp";

  VLOG(4) << "Builder construction inputs";
  std::vector<pir::Value> argument_inputs = {x_, y_};
  argument.AddInputs(argument_inputs);

  VLOG(4) << "Builder construction attributes";
  pir::AttributeMap argument_attributes = {};
  std::vector<pir::Type> argument_outputs =
      ShapeBroadcastOp::InferMeta(argument_inputs, &argument_attributes);

  argument.AddOutputs(argument_outputs.begin(), argument_outputs.end());
  ::pir::PassStopGradientsDefaultly(argument);
}

void ShapeBroadcastOp::InferMeta(phi::InferMetaContext *infer_meta) {
  auto fn = PD_INFER_META(phi::ElementwiseInferMeta);
  fn(infer_meta);
}

std::vector<pir::Type> ShapeBroadcastOp::InferMeta(
    const std::vector<pir::Value> &input_values,
    pir::AttributeMap *p_attributes) {
  VLOG(4) << "Start infermeta ShapeBroadcastOp";
  IR_ENFORCE(input_values.size() == 2,
             "Num of inputs is expected to be 2 but got %d.",
             input_values.size());
  pir::Value x_ = input_values[0];
  pir::Value y_ = input_values[1];

  VLOG(4) << "Builder construction outputs";
  paddle::dialect::DenseTensorType x;
  if (x_.type().isa<paddle::dialect::DenseTensorType>()) {
    x = x_.type().dyn_cast<paddle::dialect::DenseTensorType>();
  } else {
    PADDLE_THROW(phi::errors::Unimplemented(
        "Only support paddle::dialect::DenseTensorType or "
        "paddle::dialect::AllocatedDenseTensorType"));
  }

  paddle::dialect::DenseTensorType y;
  if (y_.type().isa<paddle::dialect::DenseTensorType>()) {
    y = y_.type().dyn_cast<paddle::dialect::DenseTensorType>();
  } else {
    PADDLE_THROW(phi::errors::Unimplemented(
        "Only support paddle::dialect::DenseTensorType or "
        "paddle::dialect::AllocatedDenseTensorType"));
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

  VLOG(4) << "Builder construction  dense_y";
  paddle::dialect::IrTensor ir_tensor_y(
      paddle::dialect::TransToPhiDataType(y.dtype()),
      y.dims(),
      y.data_layout(),
      y.lod(),
      y.offset());
  VLOG(4) << "Builder construction  meta_y";
  paddle::dialect::IrMetaTensor meta_y(&ir_tensor_y);
  paddle::dialect::IrTensor dense_out;
  paddle::dialect::IrMetaTensor meta_out(&dense_out);

  phi::ElementwiseInferMeta(meta_x, meta_y, &meta_out);

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

namespace {

symbol::DimExpr GetBroadcastDimExpr(const symbol::DimExpr &lhs,
                                    const symbol::DimExpr &rhs) {
  if (lhs == rhs) {
    return lhs;
  } else if (lhs == 1 || symbol::IsDimExprGreaterThanOne(rhs)) {
    return rhs;
  } else if (rhs == 1 || symbol::IsDimExprGreaterThanOne(lhs)) {
    return lhs;
  } else {
    return symbol::Broadcast<symbol::DimExpr>{
        symbol::List<symbol::DimExpr>{lhs, rhs}};
  }
  LOG(FATAL) << "Dead code";
}

}  // namespace

std::vector<symbol::DimExpr> ComputeBroadcastShape(
    const std::vector<symbol::DimExpr> &large_shape,
    const std::vector<symbol::DimExpr> &small_shape) {
  IR_ENFORCE(large_shape.size() >= small_shape.size(),
             "Size of large_shape is expected to be greater or equal size of "
             "small_shape, but got [%d] >= [%d].",
             large_shape.size(),
             small_shape.size());
  std::vector<symbol::DimExpr> output_data;
  output_data.reserve(large_shape.size());
  auto rank_gap = large_shape.size() - small_shape.size();
  for (size_t i = 0; i < rank_gap; ++i) {
    output_data.emplace_back(large_shape.at(i));
  }
  for (size_t i = 0; i < small_shape.size(); ++i) {
    output_data.emplace_back(
        GetBroadcastDimExpr(large_shape.at(i + rank_gap), small_shape.at(i)));
  }
  return output_data;
}

bool ShapeBroadcastOp::InferSymbolicShape(
    pir::ShapeConstraintIRAnalysis *shape_analysis) {
  pir::Value x = operand_source(0);
  pir::Value y = operand_source(1);

  IR_ENFORCE(shape_analysis->HasShapeOrDataForValue(x) > 0,
             "Value x does not exist.");
  IR_ENFORCE(shape_analysis->HasShapeOrDataForValue(y) > 0,
             "Value y does not exist.");
  const auto &x_data_shape = shape_analysis->GetShapeOrDataForValue(x);
  const auto &y_data_shape = shape_analysis->GetShapeOrDataForValue(y);
  IR_ENFORCE(x_data_shape.data().has_value(),
             "Value x comes from ShapeOp, it must have data");
  IR_ENFORCE(y_data_shape.data().has_value(),
             "Value y comes from ShapeOp, it must have data");
  const auto &x_data = x_data_shape.data().value();
  const auto &y_data = y_data_shape.data().value();

  std::vector<symbol::DimExpr> output_data =
      x_data.size() > y_data.size() ? ComputeBroadcastShape(x_data, y_data)
                                    : ComputeBroadcastShape(y_data, x_data);

  pir::Value res = result(0);
  std::vector<symbol::DimExpr> shape{std::int64_t(output_data.size())};

  symbol::ShapeOrDataDimExprs output_data_shape{
      symbol::TensorShapeOrDataDimExprs(shape, output_data)};
  shape_analysis->SetShapeOrDataForValue(res, output_data_shape);
  return true;
}

const char *MemcpyD2hMultiIoOp::attributes_name[1] = {"dst_place_type"};

OpInfoTuple MemcpyD2hMultiIoOp::GetOpInfo() {
  std::vector<paddle::dialect::OpInputInfo> inputs = {
      paddle::dialect::OpInputInfo("x",
                                   "paddle::dialect::DenseTensorArrayType",
                                   false,
                                   false,
                                   false,
                                   false)};
  std::vector<paddle::dialect::OpAttributeInfo> attributes = {
      paddle::dialect::OpAttributeInfo(
          "dst_place_type", "pir::Int32Attribute", "")};
  std::vector<paddle::dialect::OpOutputInfo> outputs = {
      paddle::dialect::OpOutputInfo(
          "out", "paddle::dialect::DenseTensorArrayType", false, false)};
  paddle::dialect::OpRunTimeInfo run_time_info =
      paddle::dialect::OpRunTimeInfo("UnchangedInferMeta",
                                     {"x"},
                                     "memcpy_d2h_multi_io",
                                     {"x", "dst_place_type"},
                                     {},
                                     {},
                                     {},
                                     {});
  return std::make_tuple(
      inputs, attributes, outputs, run_time_info, "memcpy_d2h_multi_io");
}

void MemcpyD2hMultiIoOp::VerifySig() {
  VLOG(4) << "Start Verifying inputs, outputs and attributes for: "
             "MemcpyD2hMultiIoOp.";
  VLOG(4) << "Verifying inputs:";
  {
    auto input_size = num_operands();
    IR_ENFORCE(input_size == 1u,
               "The size %d of inputs must be equal to 1.",
               input_size);
    IR_ENFORCE((*this)
                   ->operand_source(0)
                   .type()
                   .isa<paddle::dialect::DenseTensorArrayType>(),
               "Type validation failed for the 0th input, got %s.",
               (*this)->operand_source(0).type());
  }
  VLOG(4) << "Verifying attributes:";
  {
    auto &attributes = this->attributes();
    IR_ENFORCE(attributes.count("dst_place_type") > 0,
               "dst_place_type does not exist.");
    IR_ENFORCE(attributes.at("dst_place_type").isa<pir::Int32Attribute>(),
               "Type of attribute: dst_place_type is not pir::Int32Attribute.");
  }
  VLOG(4) << "Verifying outputs:";
  {
    auto output_size = num_results();
    IR_ENFORCE(output_size == 1u,
               "The size %d of outputs must be equal to 1.",
               output_size);
    auto output_0_type = (*this)->result(0).type();

    IR_ENFORCE(output_0_type.isa<paddle::dialect::DenseTensorArrayType>(),
               "Type validation failed for the 0th output.");
  }
  VLOG(4) << "End Verifying for: MemcpyD2hMultiIoOp.";
}

void MemcpyD2hMultiIoOp::InferMeta(phi::InferMetaContext *infer_meta) {
  auto fn = PD_INFER_META(phi::UnchangedArrayInferMeta);
  fn(infer_meta);
}

std::vector<pir::Type> MemcpyD2hMultiIoOp::InferMeta(
    const std::vector<pir::Value> &input_values,
    pir::AttributeMap *p_attributes) {
  IR_ENFORCE(input_values.size() == 1,
             "Num of inputs is expected to be 1 but got %d.",
             input_values.size());

  pir::Value x_ = input_values[0];
  VLOG(4) << "Builder construction outputs";
  paddle::dialect::DenseTensorArrayType x_type;
  if (x_.type().isa<paddle::dialect::DenseTensorArrayType>()) {
    x_type = x_.type().dyn_cast<paddle::dialect::DenseTensorArrayType>();
  } else {
    PADDLE_THROW(phi::errors::Unimplemented(
        "Only support paddle::dialect::DenseTensorArrayType or "
        "paddle::dialect::AllocatedDenseTensorArrayType"));
  }
  paddle::dialect::IrTensor dense_input(
      paddle::dialect::TransToPhiDataType(x_type.dtype()),
      x_type.dims(),
      x_type.data_layout(),
      {});
  paddle::dialect::IrMetaTensor meta_input(&dense_input);

  paddle::dialect::IrTensor dense_out;
  paddle::dialect::IrMetaTensor meta_out(&dense_out);

  phi::UnchangedArrayInferMeta(meta_input, &meta_out);

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

phi::DataType MemcpyD2hMultiIoOp::GetKernelTypeForVar(
    const std::string &var_name,
    const phi::DataType &tensor_dtype,
    const phi::DataType &expected_kernel_dtype) {
  VLOG(4) << "Get KernelType for Var of op: MemcpyD2hMultiIoOp";

  return expected_kernel_dtype;
}

const char *ArrayPopOp::attributes_name[1] = {"index"};

OpInfoTuple ArrayPopOp::GetOpInfo() {
  std::vector<paddle::dialect::OpInputInfo> inputs = {
      paddle::dialect::OpInputInfo("array",
                                   "paddle::dialect::DenseTensorArrayType",
                                   false,
                                   false,
                                   false,
                                   false)};

  std::vector<paddle::dialect::OpAttributeInfo> attributes = {
      paddle::dialect::OpAttributeInfo("index", "pir::Int32Attribute", ""),
  };

  std::vector<paddle::dialect::OpOutputInfo> outputs = {
      paddle::dialect::OpOutputInfo(
          "array_out", "paddle::dialect::DenseTensorArrayType", false, false),
      paddle::dialect::OpOutputInfo(
          "out", "paddle::dialect::DenseTensorType", false, false)};
  paddle::dialect::OpRunTimeInfo run_time_info =
      paddle::dialect::OpRunTimeInfo("ArrayPopInferMeta",
                                     {"array", "index"},
                                     "array_pop",
                                     {"array", "index"},
                                     {"array"},
                                     {"array"},
                                     {{"array_out", "array"}},
                                     {});
  return std::make_tuple(
      inputs, attributes, outputs, run_time_info, "array_pop");
}

void ArrayPopOp::VerifySig() {
  VLOG(4) << "Start Verifying inputs, outputs and attributes for: ArrayPopOp.";
  VLOG(4) << "Verifying inputs:";
  {
    auto input_size = num_operands();
    IR_ENFORCE(input_size == 1u,
               "The size %d of inputs must be equal to 1.",
               input_size);
    IR_ENFORCE((*this)
                   ->operand_source(0)
                   .type()
                   .isa<paddle::dialect::DenseTensorArrayType>(),
               "Type validation failed for the 0th input, got %s.",
               (*this)->operand_source(0).type());
  }
  VLOG(4) << "Verifying attributes:";
  {
    auto &attributes = this->attributes();
    IR_ENFORCE(attributes.count("index") > 0, "index does not exist.");
    IR_ENFORCE(attributes.at("index").isa<pir::Int32Attribute>(),
               "Type of attribute: index is not pir::Int32Attribute.");
  }
  VLOG(4) << "Verifying outputs:";
  {
    auto output_size = num_results();
    IR_ENFORCE(output_size == 2u,
               "The size %d of outputs must be equal to 2.",
               output_size);
    IR_ENFORCE(
        (*this)->result(0).type().isa<paddle::dialect::DenseTensorArrayType>(),
        "Type validation failed for the 0th output.");
    IR_ENFORCE(
        (*this)->result(1).type().isa<paddle::dialect::DenseTensorType>(),
        "Type validation failed for the 1st output.");
  }
  VLOG(4) << "End Verifying for: ArrayPopOp.";
}

void ArrayPopOp::Build(pir::Builder &builder,             // NOLINT
                       pir::OperationArgument &argument,  // NOLINT
                       pir::Value array,
                       int index) {
  VLOG(4) << "Start build ArrayPopOp";
  VLOG(4) << "Builder construction inputs";
  std::vector<pir::Value> argument_inputs = {array};
  argument.AddInputs({array});
  VLOG(4) << "Builder construction attributes";
  pir::AttributeMap argument_attributes = {};
  pir::Attribute attr_index =
      pir::Int32Attribute::get(pir::IrContext::Instance(), index);
  argument.AddAttribute("index", attr_index);
  argument_attributes.insert({"index", attr_index});
  std::vector<pir::Type> argument_outputs =
      ArrayPopOp::InferMeta(argument_inputs, &argument_attributes);

  argument.AddOutputs(argument_outputs.begin(), argument_outputs.end());
  ::pir::PassStopGradientsDefaultly(argument);
}

void ArrayPopOp::InferMeta(phi::InferMetaContext *infer_meta) {
  auto fn = PD_INFER_META(phi::ArrayPopInferMeta);
  fn(infer_meta);
}

std::vector<pir::Type> ArrayPopOp::InferMeta(
    const std::vector<pir::Value> &input_values,
    pir::AttributeMap *p_attributes) {
  PADDLE_ENFORCE_NOT_NULL(
      p_attributes,
      common::errors::Fatal(
          "AttrtibueMap pointer in InferMeta function is nullptr."));
  auto &attributes = *p_attributes;
  VLOG(4) << "Start infermeta ArrayPopOp";
  IR_ENFORCE(input_values.size() == 1,
             "Num of inputs is expected to be 1 but got %d.",
             input_values.size());
  pir::Value input = input_values[0];

  VLOG(4) << "Builder construction outputs";
  paddle::dialect::DenseTensorArrayType input_type;
  if (input.type().isa<paddle::dialect::DenseTensorArrayType>()) {
    input_type = input.type().dyn_cast<paddle::dialect::DenseTensorArrayType>();
  } else {
    PADDLE_THROW(phi::errors::Unimplemented(
        "Only support paddle::dialect::DenseTensorArrayType or "
        "paddle::dialect::AllocatedDenseTensorArrayType"));
  }

  IR_ENFORCE(attributes.find("index") != attributes.end(),
             "'index' Attribute is expected for ArrayPopOp. ");
  int index = attributes.at("index").dyn_cast<pir::Int32Attribute>().data();

  paddle::dialect::IrTensor dense_input(
      paddle::dialect::TransToPhiDataType(input_type.dtype()),
      input_type.dims(),
      input_type.data_layout(),
      {});
  paddle::dialect::IrMetaTensor meta_input(&dense_input);
  paddle::dialect::IrTensor array_out;
  paddle::dialect::IrMetaTensor meta_array_out(&array_out);
  paddle::dialect::IrTensor dense_out;
  paddle::dialect::IrMetaTensor meta_out(&dense_out);

  phi::ArrayPopInferMeta(meta_input,
                         index,
                         &meta_array_out,
                         &meta_out,
                         phi::MetaConfig(false, false));

  std::vector<pir::Type> argument_outputs;
  pir::Type out_array_type = paddle::dialect::DenseTensorArrayType::get(
      pir::IrContext::Instance(),
      paddle::dialect::TransToIrDataType(array_out.dtype()),
      array_out.dims(),
      array_out.layout());
  pir::Type out_dense_tensor_type = paddle::dialect::DenseTensorType::get(
      pir::IrContext::Instance(),
      paddle::dialect::TransToIrDataType(dense_out.dtype()),
      dense_out.dims(),
      dense_out.layout(),
      dense_out.lod(),
      dense_out.offset());
  argument_outputs.push_back(out_array_type);
  argument_outputs.push_back(out_dense_tensor_type);
  return argument_outputs;
}

phi::DataType ArrayPopOp::GetKernelTypeForVar(
    const std::string &var_name,
    const phi::DataType &tensor_dtype,
    const phi::DataType &expected_kernel_dtype) {
  VLOG(4) << "Get KernelType for Var of op: ArrayPopOp";

  return expected_kernel_dtype;
}

}  // namespace dialect
}  // namespace paddle

IR_DEFINE_EXPLICIT_TYPE_ID(paddle::dialect::AddNOp)
IR_DEFINE_EXPLICIT_TYPE_ID(paddle::dialect::SplitGradOp)
IR_DEFINE_EXPLICIT_TYPE_ID(paddle::dialect::AddN_Op)
IR_DEFINE_EXPLICIT_TYPE_ID(paddle::dialect::AddNArrayOp)
IR_DEFINE_EXPLICIT_TYPE_ID(paddle::dialect::AssignOut_Op)
IR_DEFINE_EXPLICIT_TYPE_ID(paddle::dialect::FusedGemmEpilogueOp)
IR_DEFINE_EXPLICIT_TYPE_ID(paddle::dialect::FusedGemmEpilogueGradOp)
IR_DEFINE_EXPLICIT_TYPE_ID(paddle::dialect::CreateArrayOp)
IR_DEFINE_EXPLICIT_TYPE_ID(paddle::dialect::CreateArrayLikeOp)
IR_DEFINE_EXPLICIT_TYPE_ID(paddle::dialect::ArrayLengthOp)
IR_DEFINE_EXPLICIT_TYPE_ID(paddle::dialect::ArrayReadOp)
IR_DEFINE_EXPLICIT_TYPE_ID(paddle::dialect::ArrayWrite_Op)
IR_DEFINE_EXPLICIT_TYPE_ID(paddle::dialect::SliceArrayOp)
IR_DEFINE_EXPLICIT_TYPE_ID(paddle::dialect::SliceArrayDenseOp)
IR_DEFINE_EXPLICIT_TYPE_ID(paddle::dialect::AssignArrayOp)
IR_DEFINE_EXPLICIT_TYPE_ID(paddle::dialect::AssignArray_Op)
IR_DEFINE_EXPLICIT_TYPE_ID(paddle::dialect::ArrayToTensorOp)
IR_DEFINE_EXPLICIT_TYPE_ID(paddle::dialect::TensorToArrayOp)
IR_DEFINE_EXPLICIT_TYPE_ID(paddle::dialect::ExpandOp)
IR_DEFINE_EXPLICIT_TYPE_ID(paddle::dialect::IncrementOp)
IR_DEFINE_EXPLICIT_TYPE_ID(paddle::dialect::Increment_Op)
IR_DEFINE_EXPLICIT_TYPE_ID(paddle::dialect::MemcpyD2hMultiIoOp)
IR_DEFINE_EXPLICIT_TYPE_ID(paddle::dialect::ShapeBroadcastOp)
IR_DEFINE_EXPLICIT_TYPE_ID(paddle::dialect::ArrayPopOp)
#endif
