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
paddle::dialect::AddNOp, paddle::dialect::AddN_Op,
    paddle::dialect::AddNWithKernelOp, paddle::dialect::FusedGemmEpilogueOp,
    paddle::dialect::FusedGemmEpilogueGradOp, paddle::dialect::SplitGradOp,
    paddle::dialect::ExpandOp, paddle::dialect::CreateArrayOp,
    paddle::dialect::ArrayLengthOp, paddle::dialect::ArrayReadOp,
    paddle::dialect::ArrayWrite_Op, paddle::dialect::SliceArrayOp,
    paddle::dialect::SliceArrayDenseOp, paddle::dialect::AssignArray_Op,
    paddle::dialect::ArrayToTensorOp, paddle::dialect::SelectInputOp
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
#include "paddle/phi/api/lib/utils/allocator.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/infermeta/backward.h"
#include "paddle/phi/infermeta/binary.h"
#include "paddle/phi/infermeta/fusion.h"
#include "paddle/phi/infermeta/multiary.h"
#include "paddle/phi/infermeta/nullary.h"
#include "paddle/phi/infermeta/unary.h"
#include "paddle/pir/core/builtin_attribute.h"
#include "paddle/pir/core/builtin_op.h"
#include "paddle/pir/core/builtin_type.h"
#include "paddle/pir/core/ir_context.h"

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
  argument.AddInput(inputs);

  VLOG(4) << "Builder construction attributes";

  VLOG(4) << "Builder construction outputs";
  pir::VectorType x = inputs.type().dyn_cast<pir::VectorType>();

  std::vector<paddle::dialect::IrTensor> vec_dense_x;
  for (size_t i = 0; i < x.size(); i++) {
    vec_dense_x.push_back(paddle::dialect::IrTensor(
        TransToPhiDataType(
            x[i].dyn_cast<paddle::dialect::DenseTensorType>().dtype()),
        x[i].dyn_cast<paddle::dialect::DenseTensorType>().dims(),
        x[i].dyn_cast<paddle::dialect::DenseTensorType>().data_layout(),
        x[i].dyn_cast<paddle::dialect::DenseTensorType>().lod(),
        x[i].dyn_cast<paddle::dialect::DenseTensorType>().offset()));
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

  phi::AddNInferMeta(meta_x, &meta_out);

  std::vector<pir::Type> argument_outputs;
  pir::Type out_dense_tensor_type = paddle::dialect::DenseTensorType::get(
      pir::IrContext::Instance(),
      TransToIrDataType(dense_out.dtype()),
      dense_out.dims(),
      dense_out.layout(),
      dense_out.lod(),
      dense_out.offset());
  argument_outputs.push_back(out_dense_tensor_type);
  argument.AddOutputs(argument_outputs.begin(), argument_outputs.end());
  ::pir::PassStopGradientsDefaultly(argument);
}

void AddNOp::InferMeta(phi::InferMetaContext *infer_meta) {
  auto fn = PD_INFER_META(phi::AddNInferMeta);
  fn(infer_meta);
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
  argument.AddInput(inputs_);

  VLOG(4) << "Builder construction attributes";

  VLOG(4) << "Builder construction outputs";
  pir::VectorType inputs = inputs_.type().dyn_cast<pir::VectorType>();
  std::vector<paddle::dialect::IrTensor> vec_dense_inputs;
  for (size_t i = 0; i < static_cast<size_t>(inputs.size()); i++) {
    vec_dense_inputs.push_back(paddle::dialect::IrTensor(
        paddle::dialect::TransToPhiDataType(
            inputs[i].dyn_cast<paddle::dialect::DenseTensorType>().dtype()),
        inputs[i].dyn_cast<paddle::dialect::DenseTensorType>().dims(),
        inputs[i].dyn_cast<paddle::dialect::DenseTensorType>().data_layout(),
        inputs[i].dyn_cast<paddle::dialect::DenseTensorType>().lod(),
        inputs[i].dyn_cast<paddle::dialect::DenseTensorType>().offset()));
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

  phi::AddNInferMeta(meta_inputs, &meta_out);

  std::vector<pir::Type> argument_outputs;
  pir::Type out_dense_tensor_type = paddle::dialect::DenseTensorType::get(
      pir::IrContext::Instance(),
      paddle::dialect::TransToIrDataType(dense_out.dtype()),
      dense_out.dims(),
      dense_out.layout(),
      dense_out.lod(),
      dense_out.offset());
  argument_outputs.push_back(out_dense_tensor_type);
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

OpInfoTuple AddNWithKernelOp::GetOpInfo() {
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
  return std::make_tuple(
      inputs, attributes, outputs, run_time_info, "add_n_with_kernel");
}

void AddNWithKernelOp::Build(pir::Builder &builder,
                             pir::OperationArgument &argument,
                             pir::Value inputs_) {
  VLOG(4) << "Start build AddNWithKernelOp";

  VLOG(4) << "Builder construction inputs";
  argument.AddInput(inputs_);

  VLOG(4) << "Builder construction attributes";

  VLOG(4) << "Builder construction outputs";
  pir::VectorType inputs = inputs_.type().dyn_cast<pir::VectorType>();
  std::vector<paddle::dialect::IrTensor> vec_dense_inputs;
  for (size_t i = 0; i < static_cast<size_t>(inputs.size()); i++) {
    vec_dense_inputs.push_back(paddle::dialect::IrTensor(
        paddle::dialect::TransToPhiDataType(
            inputs[i].dyn_cast<paddle::dialect::DenseTensorType>().dtype()),
        inputs[i].dyn_cast<paddle::dialect::DenseTensorType>().dims(),
        inputs[i].dyn_cast<paddle::dialect::DenseTensorType>().data_layout(),
        inputs[i].dyn_cast<paddle::dialect::DenseTensorType>().lod(),
        inputs[i].dyn_cast<paddle::dialect::DenseTensorType>().offset()));
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

  phi::AddNInferMeta(meta_inputs, &meta_out);

  std::vector<pir::Type> argument_outputs;
  pir::Type out_dense_tensor_type = paddle::dialect::DenseTensorType::get(
      pir::IrContext::Instance(),
      paddle::dialect::TransToIrDataType(dense_out.dtype()),
      dense_out.dims(),
      dense_out.layout(),
      dense_out.lod(),
      dense_out.offset());
  argument_outputs.push_back(out_dense_tensor_type);
  argument.AddOutputs(argument_outputs.begin(), argument_outputs.end());
}

void AddNWithKernelOp::VerifySig() {
  VLOG(4) << "Start Verifying inputs, outputs and attributes for: "
             "AddNWithKernelOp.";
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
  VLOG(4) << "End Verifying for: AddNWithKernelOp.";
}

void AddNWithKernelOp::InferMeta(phi::InferMetaContext *infer_meta) {
  auto fn = PD_INFER_META(phi::AddNInferMeta);
  fn(infer_meta);
}

const char *FusedGemmEpilogueOp::attributes_name[3] = {
    "trans_x", "trans_y", "activation"};

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
  argument.AddInputs({x_, y_, bias_});

  VLOG(4) << "Builder construction attributes";
  pir::Attribute attr_trans_x =
      pir::BoolAttribute::get(pir::IrContext::Instance(), trans_x);
  argument.AddAttribute("trans_x", attr_trans_x);
  pir::Attribute attr_trans_y =
      pir::BoolAttribute::get(pir::IrContext::Instance(), trans_y);
  argument.AddAttribute("trans_y", attr_trans_y);
  pir::Attribute attr_activation =
      pir::StrAttribute::get(pir::IrContext::Instance(), activation);
  argument.AddAttribute("activation", attr_activation);

  VLOG(4) << "Builder construction outputs";
  paddle::dialect::DenseTensorType x =
      x_.type().dyn_cast<paddle::dialect::DenseTensorType>();
  (void)x;
  paddle::dialect::DenseTensorType y =
      y_.type().dyn_cast<paddle::dialect::DenseTensorType>();
  (void)y;
  paddle::dialect::DenseTensorType bias =
      bias_.type().dyn_cast<paddle::dialect::DenseTensorType>();
  (void)bias;

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

const char *FusedGemmEpilogueGradOp::attributes_name[3] = {
    "trans_x", "trans_y", "activation_grad"};

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
  argument.AddInputs({x_, y_, reserve_space_, out_grad_});

  VLOG(4) << "Builder construction attributes";
  pir::Attribute attr_trans_x =
      pir::BoolAttribute::get(pir::IrContext::Instance(), trans_x);
  argument.AddAttribute("trans_x", attr_trans_x);
  pir::Attribute attr_trans_y =
      pir::BoolAttribute::get(pir::IrContext::Instance(), trans_y);
  argument.AddAttribute("trans_y", attr_trans_y);
  pir::Attribute attr_activation_grad =
      pir::StrAttribute::get(pir::IrContext::Instance(), activation_grad);
  argument.AddAttribute("activation_grad", attr_activation_grad);

  VLOG(4) << "Builder construction outputs";
  paddle::dialect::DenseTensorType x =
      x_.type().dyn_cast<paddle::dialect::DenseTensorType>();
  (void)x;
  paddle::dialect::DenseTensorType y =
      y_.type().dyn_cast<paddle::dialect::DenseTensorType>();
  (void)y;
  paddle::dialect::DenseTensorType reserve_space =
      reserve_space_
          ? reserve_space_.type().dyn_cast<paddle::dialect::DenseTensorType>()
          : paddle::dialect::DenseTensorType();
  (void)reserve_space;
  paddle::dialect::DenseTensorType out_grad =
      out_grad_.type().dyn_cast<paddle::dialect::DenseTensorType>();
  (void)out_grad;

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
  argument.AddOutputs(argument_outputs.begin(), argument_outputs.end());
}

void FusedGemmEpilogueGradOp::VerifySig() {}

void FusedGemmEpilogueGradOp::InferMeta(phi::InferMetaContext *infer_meta) {
  auto fn = PD_INFER_META(phi::FusedGemmEpilogueGradInferMeta);
  fn(infer_meta);
}

const char *SplitGradOp::attributes_name[1] = {"axis"};

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
  pir::OpResult axis_ = full_axis_op->result(0);

  VLOG(4) << "Builder construction inputs";
  argument.AddInputs({out_grad_, axis_});

  VLOG(4) << "Builder construction attributes";

  VLOG(4) << "Builder construction outputs";
  pir::VectorType out_grad = out_grad_.type().dyn_cast<pir::VectorType>();
  std::vector<paddle::dialect::IrTensor> vec_dense_out_grad;
  for (size_t i = 0; i < static_cast<size_t>(out_grad.size()); i++) {
    vec_dense_out_grad.push_back(paddle::dialect::IrTensor(
        paddle::dialect::TransToPhiDataType(
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
      paddle::dialect::TransToIrDataType(dense_x_grad.dtype()),
      dense_x_grad.dims(),
      dense_x_grad.layout(),
      dense_x_grad.lod(),
      dense_x_grad.offset());
  argument_outputs.push_back(x_grad_dense_tensor_type);
  argument.AddOutputs(argument_outputs.begin(), argument_outputs.end());
}

void SplitGradOp::Build(pir::Builder &builder,
                        pir::OperationArgument &argument,
                        pir::Value out_grad_,
                        pir::Value axis_) {
  VLOG(4) << "Start build SplitGradOp";

  VLOG(4) << "Builder construction inputs";
  argument.AddInputs({out_grad_, axis_});

  VLOG(4) << "Builder construction attributes";

  VLOG(4) << "Builder construction outputs";
  pir::VectorType out_grad = out_grad_.type().dyn_cast<pir::VectorType>();
  int axis = axis_.dyn_cast<pir::OpResult>()
                 .owner()
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
  argument.AddOutputs(argument_outputs.begin(), argument_outputs.end());
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

const char *CreateArrayOp::attributes_name[1] = {"dtype"};

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
  VLOG(4) << "Builder construction attributes";
  pir::Attribute attr_dtype = paddle::dialect::DataTypeAttribute::get(
      pir::IrContext::Instance(), dtype);
  argument.AddAttribute("dtype", attr_dtype);
  VLOG(4) << "Builder construction outputs";

  paddle::dialect::IrTensor dense_out;
  paddle::dialect::IrMetaTensor meta_out(&dense_out);

  phi::CreateArrayInferMeta(dtype, &meta_out);

  std::vector<pir::Type> argument_outputs;
  pir::Type out_dense_tensor_type = paddle::dialect::DenseTensorArrayType::get(
      pir::IrContext::Instance(),
      paddle::dialect::TransToIrDataType(dense_out.dtype()),
      dense_out.layout());
  argument_outputs.push_back(out_dense_tensor_type);
  argument.AddOutputs(argument_outputs.begin(), argument_outputs.end());
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
  argument.AddInputs({x});
  VLOG(4) << "Builder construction attributes";
  VLOG(4) << "Builder construction outputs";

  paddle::dialect::DenseTensorArrayType x_type =
      x.type().dyn_cast<paddle::dialect::DenseTensorArrayType>();
  paddle::dialect::IrTensor dense_x(
      paddle::dialect::TransToPhiDataType(x_type.dtype()),
      {},
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
  argument.AddInputs({array, full_i_op.result(0)});
  VLOG(4) << "Builder construction attributes";
  VLOG(4) << "Builder construction outputs";
  paddle::dialect::DenseTensorArrayType array_type =
      array.type().dyn_cast<paddle::dialect::DenseTensorArrayType>();
  paddle::dialect::IrTensor dense_array(
      paddle::dialect::TransToPhiDataType(array_type.dtype()),
      {},
      array_type.data_layout(),
      {});
  paddle::dialect::IrMetaTensor meta_array(&dense_array);

  paddle::dialect::IrTensor dense_out;
  paddle::dialect::IrMetaTensor meta_out(&dense_out);

  phi::ArrayReadInferMeta(
      meta_array, i, &meta_out, phi::MetaConfig(false, false));

  std::vector<pir::Type> argument_outputs;
  pir::Type out_type = paddle::dialect::DenseTensorType::get(
      pir::IrContext::Instance(),
      paddle::dialect::TransToIrDataType(dense_out.dtype()),
      dense_out.dims(),
      dense_out.layout(),
      dense_out.lod());
  argument_outputs.push_back(out_type);
  argument.AddOutputs(argument_outputs.begin(), argument_outputs.end());
}

void ArrayReadOp::Build(pir::Builder &builder,
                        pir::OperationArgument &argument,
                        pir::Value array,
                        pir::Value i) {
  VLOG(4) << "Start build ArrayReadOp";
  VLOG(4) << "Builder construction inputs";
  argument.AddInputs({array, i});
  VLOG(4) << "Builder construction attributes";
  VLOG(4) << "Builder construction outputs";
  paddle::dialect::DenseTensorArrayType array_type =
      array.type().dyn_cast<paddle::dialect::DenseTensorArrayType>();
  paddle::dialect::IrTensor dense_array(
      paddle::dialect::TransToPhiDataType(array_type.dtype()),
      {},
      array_type.data_layout(),
      {});
  paddle::dialect::IrMetaTensor meta_array(&dense_array);

  phi::Scalar i_scalar;
  if (i.dyn_cast<pir::OpResult>().owner()->isa<paddle::dialect::FullOp>()) {
    i_scalar =
        std::move(phi::Scalar(i.dyn_cast<pir::OpResult>()
                                  .owner()
                                  ->dyn_cast<paddle::dialect::FullOp>()
                                  .attribute("value")
                                  .dyn_cast<paddle::dialect::ScalarAttribute>()
                                  .data()
                                  .to<int64_t>()));
  } else {
    i_scalar = std::move(phi::Scalar(-1));
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
  argument.AddOutputs(argument_outputs.begin(), argument_outputs.end());
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
  VLOG(4) << "End Verifying for: ArrayWrite_Op.";
}

void ArrayReadOp::InferMeta(phi::InferMetaContext *infer_meta) {
  auto fn = PD_INFER_META(phi::ArrayReadInferMeta);
  fn(infer_meta);
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
          "x", "paddle::dialect::DenseTensorType", false, false, false, false),
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
                    {},
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
  argument.AddInputs({array, x, i});
  VLOG(4) << "Builder construction attributes";
  VLOG(4) << "Builder construction outputs";
  paddle::dialect::DenseTensorArrayType array_type =
      array.type().dyn_cast<paddle::dialect::DenseTensorArrayType>();
  paddle::dialect::IrTensor dense_array(
      paddle::dialect::TransToPhiDataType(array_type.dtype()),
      {},
      array_type.data_layout(),
      {});
  paddle::dialect::IrMetaTensor meta_array(&dense_array);

  paddle::dialect::DenseTensorType x_type =
      x.type().dyn_cast<paddle::dialect::DenseTensorType>();
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
      paddle::dialect::TransToIrDataType(dense_out.dtype()),
      dense_out.layout());
  argument_outputs.push_back(out_type);
  argument.AddOutputs(argument_outputs.begin(), argument_outputs.end());
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
  argument.AddInputs({x});

  VLOG(4) << "Builder construction attributes";
  pir::Attribute attr_axis =
      pir::Int32Attribute::get(pir::IrContext::Instance(), axis);
  argument.AddAttribute("axis", attr_axis);
  pir::Attribute attr_use_stack =
      pir::BoolAttribute::get(pir::IrContext::Instance(), use_stack);
  argument.AddAttribute("use_stack", attr_use_stack);

  VLOG(4) << "Builder construction outputs";
  paddle::dialect::DenseTensorArrayType x_type =
      x.type().dyn_cast<paddle::dialect::DenseTensorArrayType>();
  paddle::dialect::IrTensor dense_x(
      paddle::dialect::TransToPhiDataType(x_type.dtype()),
      {},
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
  argument.AddOutputs(argument_outputs.begin(), argument_outputs.end());
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

const char *SliceArrayOp::attributes_name[2] = {"starts", "ends"};

OpInfoTuple SliceArrayOp::GetOpInfo() {
  std::vector<paddle::dialect::OpInputInfo> inputs = {
      paddle::dialect::OpInputInfo("input",
                                   "paddle::dialect::DenseTensorArrayType",
                                   false,
                                   false,
                                   false,
                                   false)};
  std::vector<paddle::dialect::OpAttributeInfo> attributes = {
      paddle::dialect::OpAttributeInfo("starts",
                                       "paddle::dialect::IntArrayAttribute",
                                       "std::vector<int64_t>"),
      paddle::dialect::OpAttributeInfo("ends",
                                       "paddle::dialect::IntArrayAttribute",
                                       "std::vector<int64_t>")};
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
    IR_ENFORCE(attributes.count("starts") > 0, "starts does not exist.");
    IR_ENFORCE(
        attributes.at("starts").isa<paddle::dialect::IntArrayAttribute>(),
        "Type of attribute: starts is not paddle::dialect::IntArrayAttribute.");

    IR_ENFORCE(attributes.count("ends") > 0, "ends does not exist.");
    IR_ENFORCE(
        attributes.at("ends").isa<paddle::dialect::IntArrayAttribute>(),
        "Type of attribute: ends is not paddle::dialect::IntArrayAttribute.");
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

void SliceArrayOp::InferMeta(phi::InferMetaContext *infer_meta) {
  auto fn = PD_INFER_META(phi::SliceArrayInferMeta);
  fn(infer_meta);
}

phi::DataType SliceArrayOp::GetKernelTypeForVar(
    const std::string &var_name,
    const phi::DataType &tensor_dtype,
    const phi::DataType &expected_kernel_dtype) {
  VLOG(4) << "Get KernelType for Var of op: SliceArrayOp";

  return expected_kernel_dtype;
}

const char *SliceArrayDenseOp::attributes_name[1] = {"starts"};

OpInfoTuple SliceArrayDenseOp::GetOpInfo() {
  std::vector<paddle::dialect::OpInputInfo> inputs = {
      paddle::dialect::OpInputInfo("input",
                                   "paddle::dialect::DenseTensorArrayType",
                                   false,
                                   false,
                                   false,
                                   false)};
  std::vector<paddle::dialect::OpAttributeInfo> attributes = {
      paddle::dialect::OpAttributeInfo("starts",
                                       "paddle::dialect::IntArrayAttribute",
                                       "std::vector<int64_t>")};
  std::vector<paddle::dialect::OpOutputInfo> outputs = {
      paddle::dialect::OpOutputInfo(
          "out", "paddle::dialect::DenseTensorType", false, false)};
  paddle::dialect::OpRunTimeInfo run_time_info =
      paddle::dialect::OpRunTimeInfo("SliceArrayInferMeta",
                                     {"input", "starts"},
                                     "slice_array_dense",
                                     {"input", "starts"},
                                     {},
                                     {},
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
    IR_ENFORCE(attributes.count("starts") > 0, "starts does not exist.");
    IR_ENFORCE(
        attributes.at("starts").isa<paddle::dialect::IntArrayAttribute>(),
        "Type of attribute: starts is not paddle::dialect::IntArrayAttribute.");
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

void SliceArrayDenseOp::InferMeta(phi::InferMetaContext *infer_meta) {
  auto fn = PD_INFER_META(phi::SliceArrayDenseInferMeta);
  fn(infer_meta);
}

phi::DataType SliceArrayDenseOp::GetKernelTypeForVar(
    const std::string &var_name,
    const phi::DataType &tensor_dtype,
    const phi::DataType &expected_kernel_dtype) {
  VLOG(4) << "Get KernelType for Var of op: SliceArrayOp";

  return expected_kernel_dtype;
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
  pir::OpResult shape_ = full_shape_op->result(0);

  VLOG(4) << "Builder construction inputs";
  std::vector<pir::Value> argument_inputs = {x_, shape_};
  argument.AddInputs(argument_inputs);

  VLOG(4) << "Builder construction attributes";

  VLOG(4) << "Builder construction outputs";
  paddle::dialect::DenseTensorType x =
      x_.type().dyn_cast<paddle::dialect::DenseTensorType>();
  (void)x;

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
  pir::OpResult shape_ = full_shape_op->result(0);

  VLOG(4) << "Builder construction inputs";
  std::vector<pir::Value> argument_inputs = {x_, shape_};
  argument.AddInputs(argument_inputs);

  VLOG(4) << "Builder construction attributes";

  VLOG(4) << "Builder construction outputs";
  paddle::dialect::DenseTensorType x =
      x_.type().dyn_cast<paddle::dialect::DenseTensorType>();
  (void)x;

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

  VLOG(4) << "Builder construction outputs";
  paddle::dialect::DenseTensorType x =
      x_.type().dyn_cast<paddle::dialect::DenseTensorType>();
  (void)x;
  phi::IntArray shape;
  if (shape_.dyn_cast<pir::OpResult>()
          .owner()
          ->isa<paddle::dialect::FullIntArrayOp>()) {
    shape = std::move(phi::IntArray(paddle::dialect::GetInt64Vector(
        shape_.dyn_cast<pir::OpResult>()
            .owner()
            ->dyn_cast<paddle::dialect::FullIntArrayOp>()
            .attribute("value"))));
  } else if (shape_.type().isa<pir::VectorType>()) {
    size_t shape_size = shape_.type().dyn_cast<pir::VectorType>().size();
    // In ExpandInferMeta use -2 to represent the element in expand_shape is a
    // var.
    shape = std::move(phi::IntArray(std::vector<int64_t>(shape_size, -2)));
    shape.SetFromTensor(true);
  } else if (shape_.type().isa<paddle::dialect::DenseTensorType>()) {
    size_t shape_size = common::product(
        shape_.type().dyn_cast<paddle::dialect::DenseTensorType>().dims());
    // In ExpandInferMeta use -2 to represent the element in expand_shape is a
    // var.
    shape = std::move(phi::IntArray(std::vector<int64_t>(shape_size, -2)));
    shape.SetFromTensor(true);
  } else {
    PADDLE_THROW(phi::errors::Unimplemented(
        "Only support VectorType or DenseTensorType"));
  }

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
  argument.AddOutputs(argument_outputs.begin(), argument_outputs.end());
  ::pir::PassStopGradientsDefaultly(argument);
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

phi::DataType ExpandOp::GetKernelTypeForVar(
    const std::string &var_name,
    const phi::DataType &tensor_dtype,
    const phi::DataType &expected_kernel_dtype) {
  VLOG(4) << "Get KernelType for Var of op: ExpandOp";
  return expected_kernel_dtype;
}

void SelectInputOp::VerifySig() {
  VLOG(4) << "Verifying inputs, outputs and attributes for: SelectInputOp.";
  VLOG(4) << "Verifying inputs:";
  {
    auto in_size = num_operands();
    IR_ENFORCE(in_size == 3u, "Size %d of inputs must be >= 3.", in_size);
    auto input1 = (*this)->operand_source(1).type();
    auto input2 = (*this)->operand_source(2).type();
    if (input1.isa<paddle::dialect::DenseTensorType>() &&
        input2.isa<paddle::dialect::DenseTensorType>()) {
      auto tensor1 = input1.dyn_cast<paddle::dialect::DenseTensorType>();
      auto tensor2 = input1.dyn_cast<paddle::dialect::DenseTensorType>();
      IR_ENFORCE(
          tensor1.dtype() == tensor2.dtype(),
          "The 1st input dtype %s should be equal to 2ed input dtype %s.",
          tensor1.dtype(),
          tensor2.dtype());
      IR_ENFORCE(tensor1.data_layout() == tensor2.data_layout(),
                 "The 1st input data_layout %s should be equal to 2ed input "
                 "data_layout %s.",
                 tensor1.data_layout(),
                 tensor2.data_layout());
      IR_ENFORCE(tensor1.lod() == tensor2.lod(),
                 "The 1st input lod %s should be equal to 2ed input lod %s.",
                 tensor1.lod(),
                 tensor2.lod());
      IR_ENFORCE(
          tensor1.offset() == tensor2.offset(),
          "The 1st input offset %s should be equal to 2ed input offset %s.",
          tensor1.offset(),
          tensor2.offset());
    } else if (input1.isa<paddle::dialect::AllocatedDenseTensorType>() &&
               input2.isa<paddle::dialect::AllocatedDenseTensorType>()) {
      auto tensor1 =
          input1.dyn_cast<paddle::dialect::AllocatedDenseTensorType>();
      auto tensor2 =
          input1.dyn_cast<paddle::dialect::AllocatedDenseTensorType>();
      IR_ENFORCE(
          tensor1.dtype() == tensor2.dtype(),
          "The 1st input dtype %s should be equal to 2ed input dtype %s.",
          tensor1.dtype(),
          tensor2.dtype());
      IR_ENFORCE(tensor1.data_layout() == tensor2.data_layout(),
                 "The 1st input data_layout %s should be equal to 2ed input "
                 "data_layout %s.",
                 tensor1.data_layout(),
                 tensor2.data_layout());
      IR_ENFORCE(tensor1.lod() == tensor2.lod(),
                 "The 1st input lod %s should be equal to 2ed input lod %s.",
                 tensor1.lod(),
                 tensor2.lod());
      IR_ENFORCE(
          tensor1.offset() == tensor2.offset(),
          "The 1st input offset %s should be equal to 2ed input offset %s.",
          tensor1.offset(),
          tensor2.offset());
      IR_ENFORCE(
          tensor1.place() == tensor2.place(),
          "The 1st input place %s should be equal to 2ed input place %s.",
          tensor1.place(),
          tensor2.place());
    } else {
      IR_ENFORCE(input1 == input2,
                 "The 1st input type %s should be equal to 2ed input type %s.",
                 input1,
                 input2);
    }
  }
  VLOG(4) << "Verifying outputs:";
  {
    auto out_size = num_results();
    IR_ENFORCE(
        out_size == 1u, "Size %d of outputs must be equal to 1.", out_size);
  }
  VLOG(4) << "End Verifying for: AssignArray_Op.";
}

}  // namespace dialect
}  // namespace paddle

IR_DEFINE_EXPLICIT_TYPE_ID(paddle::dialect::AddNOp)
IR_DEFINE_EXPLICIT_TYPE_ID(paddle::dialect::SplitGradOp)
IR_DEFINE_EXPLICIT_TYPE_ID(paddle::dialect::AddN_Op)
IR_DEFINE_EXPLICIT_TYPE_ID(paddle::dialect::AddNWithKernelOp)
IR_DEFINE_EXPLICIT_TYPE_ID(paddle::dialect::FusedGemmEpilogueOp)
IR_DEFINE_EXPLICIT_TYPE_ID(paddle::dialect::FusedGemmEpilogueGradOp)
IR_DEFINE_EXPLICIT_TYPE_ID(paddle::dialect::CreateArrayOp)
IR_DEFINE_EXPLICIT_TYPE_ID(paddle::dialect::ArrayLengthOp)
IR_DEFINE_EXPLICIT_TYPE_ID(paddle::dialect::ArrayReadOp)
IR_DEFINE_EXPLICIT_TYPE_ID(paddle::dialect::ArrayWrite_Op)
IR_DEFINE_EXPLICIT_TYPE_ID(paddle::dialect::SliceArrayOp)
IR_DEFINE_EXPLICIT_TYPE_ID(paddle::dialect::SliceArrayDenseOp)
IR_DEFINE_EXPLICIT_TYPE_ID(paddle::dialect::AssignArray_Op)
IR_DEFINE_EXPLICIT_TYPE_ID(paddle::dialect::ArrayToTensorOp)
IR_DEFINE_EXPLICIT_TYPE_ID(paddle::dialect::ExpandOp)
IR_DEFINE_EXPLICIT_TYPE_ID(paddle::dialect::SelectInputOp)
#endif
