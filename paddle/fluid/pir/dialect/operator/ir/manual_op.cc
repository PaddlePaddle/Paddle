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

#include "paddle/fluid/pir/dialect/operator/ir/manual_op.h"
#include "paddle/fluid/pir/dialect/operator/ir/meta_tensor.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_attribute.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_type.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/fluid/primitive/rule/vjp/vjp.h"
#include "paddle/phi/api/lib/utils/allocator.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/infermeta/backward.h"
#include "paddle/phi/infermeta/fusion.h"
#include "paddle/phi/infermeta/multiary.h"
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

void AddNOp::Verify() {
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

  std::vector<paddle::dialect::IrMetaTensor> vec_dense_x;
  for (size_t i = 0; i < x.size(); i++) {
    vec_dense_x.push_back(paddle::dialect::IrMetaTensor(
        TransToPhiDataType(
            x[i].dyn_cast<paddle::dialect::DenseTensorType>().dtype()),
        x[i].dyn_cast<paddle::dialect::DenseTensorType>().dims(),
        x[i].dyn_cast<paddle::dialect::DenseTensorType>().data_layout(),
        x[i].dyn_cast<paddle::dialect::DenseTensorType>().lod(),
        x[i].dyn_cast<paddle::dialect::DenseTensorType>().offset()));
  }
  std::vector<phi::MetaTensor> vec_meta_x;
  for (size_t i = 0; i < vec_dense_x.size(); i++) {
    vec_meta_x.push_back(phi::MetaTensor(&vec_dense_x[i]));
  }

  std::vector<const phi::MetaTensor *> meta_x;
  for (size_t i = 0; i < static_cast<size_t>(vec_meta_x.size()); i++) {
    meta_x.push_back(&vec_meta_x[i]);
  }

  paddle::dialect::IrMetaTensor dense_out;
  phi::MetaTensor meta_out(&dense_out);

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
  std::vector<paddle::dialect::IrMetaTensor> vec_dense_inputs;
  for (size_t i = 0; i < static_cast<size_t>(inputs.size()); i++) {
    vec_dense_inputs.push_back(paddle::dialect::IrMetaTensor(
        paddle::dialect::TransToPhiDataType(
            inputs[i].dyn_cast<paddle::dialect::DenseTensorType>().dtype()),
        inputs[i].dyn_cast<paddle::dialect::DenseTensorType>().dims(),
        inputs[i].dyn_cast<paddle::dialect::DenseTensorType>().data_layout(),
        inputs[i].dyn_cast<paddle::dialect::DenseTensorType>().lod(),
        inputs[i].dyn_cast<paddle::dialect::DenseTensorType>().offset()));
  }
  std::vector<phi::MetaTensor> vec_meta_inputs;
  for (size_t i = 0; i < vec_dense_inputs.size(); i++) {
    vec_meta_inputs.push_back(phi::MetaTensor(&vec_dense_inputs[i]));
  }

  std::vector<const phi::MetaTensor *> meta_inputs;
  for (size_t i = 0; i < static_cast<size_t>(vec_meta_inputs.size()); i++) {
    meta_inputs.push_back(&vec_meta_inputs[i]);
  }
  paddle::dialect::IrMetaTensor dense_out;
  phi::MetaTensor meta_out(&dense_out);

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

void AddN_Op::Verify() {
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
  std::vector<paddle::dialect::IrMetaTensor> vec_dense_inputs;
  for (size_t i = 0; i < static_cast<size_t>(inputs.size()); i++) {
    vec_dense_inputs.push_back(paddle::dialect::IrMetaTensor(
        paddle::dialect::TransToPhiDataType(
            inputs[i].dyn_cast<paddle::dialect::DenseTensorType>().dtype()),
        inputs[i].dyn_cast<paddle::dialect::DenseTensorType>().dims(),
        inputs[i].dyn_cast<paddle::dialect::DenseTensorType>().data_layout(),
        inputs[i].dyn_cast<paddle::dialect::DenseTensorType>().lod(),
        inputs[i].dyn_cast<paddle::dialect::DenseTensorType>().offset()));
  }
  std::vector<phi::MetaTensor> vec_meta_inputs;
  for (size_t i = 0; i < vec_dense_inputs.size(); i++) {
    vec_meta_inputs.push_back(phi::MetaTensor(&vec_dense_inputs[i]));
  }

  std::vector<const phi::MetaTensor *> meta_inputs;
  for (size_t i = 0; i < static_cast<size_t>(vec_meta_inputs.size()); i++) {
    meta_inputs.push_back(&vec_meta_inputs[i]);
  }
  paddle::dialect::IrMetaTensor dense_out;
  phi::MetaTensor meta_out(&dense_out);

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

void AddNWithKernelOp::Verify() {
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
      "",
      {""},
      {""},
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
  paddle::dialect::IrMetaTensor dense_x(
      paddle::dialect::TransToPhiDataType(x.dtype()),
      x.dims(),
      x.data_layout(),
      x.lod(),
      x.offset());
  VLOG(4) << "Builder construction  meta_x";
  phi::MetaTensor meta_x(&dense_x);

  VLOG(4) << "Builder construction  dense_y";
  paddle::dialect::IrMetaTensor dense_y(
      paddle::dialect::TransToPhiDataType(y.dtype()),
      y.dims(),
      y.data_layout(),
      y.lod(),
      y.offset());
  VLOG(4) << "Builder construction  meta_y";
  phi::MetaTensor meta_y(&dense_y);

  VLOG(4) << "Builder construction  dense_bias";
  paddle::dialect::IrMetaTensor dense_bias(
      paddle::dialect::TransToPhiDataType(bias.dtype()),
      bias.dims(),
      bias.data_layout(),
      bias.lod(),
      bias.offset());
  VLOG(4) << "Builder construction  meta_bias";
  phi::MetaTensor meta_bias(&dense_bias);
  paddle::dialect::IrMetaTensor dense_out;
  phi::MetaTensor meta_out(&dense_out);
  paddle::dialect::IrMetaTensor dense_reserve_space;
  phi::MetaTensor meta_reserve_space(&dense_reserve_space);

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

void FusedGemmEpilogueOp::Verify() {
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
                                               "",
                                               {""},
                                               {""},
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
  paddle::dialect::IrMetaTensor dense_x(
      paddle::dialect::TransToPhiDataType(x.dtype()),
      x.dims(),
      x.data_layout(),
      x.lod(),
      x.offset());
  VLOG(4) << "Builder construction  meta_x";
  phi::MetaTensor meta_x(&dense_x);

  VLOG(4) << "Builder construction  dense_y";
  paddle::dialect::IrMetaTensor dense_y(
      paddle::dialect::TransToPhiDataType(y.dtype()),
      y.dims(),
      y.data_layout(),
      y.lod(),
      y.offset());
  VLOG(4) << "Builder construction  meta_y";
  phi::MetaTensor meta_y(&dense_y);

  VLOG(4) << "Builder construction  dense_reserve_space";
  std::unique_ptr<paddle::dialect::IrMetaTensor> dense_reserve_space =
      reserve_space_
          ? std::make_unique<paddle::dialect::IrMetaTensor>(
                paddle::dialect::TransToPhiDataType(reserve_space.dtype()),
                reserve_space.dims(),
                reserve_space.data_layout(),
                reserve_space.lod(),
                reserve_space.offset())
          : nullptr;
  VLOG(4) << "Builder construction  meta_reserve_space";
  phi::MetaTensor meta_reserve_space(dense_reserve_space.get());

  VLOG(4) << "Builder construction  dense_out_grad";
  paddle::dialect::IrMetaTensor dense_out_grad(
      paddle::dialect::TransToPhiDataType(out_grad.dtype()),
      out_grad.dims(),
      out_grad.data_layout(),
      out_grad.lod(),
      out_grad.offset());
  VLOG(4) << "Builder construction  meta_out_grad";
  phi::MetaTensor meta_out_grad(&dense_out_grad);
  paddle::dialect::IrMetaTensor dense_x_grad;
  phi::MetaTensor meta_x_grad(&dense_x_grad);
  paddle::dialect::IrMetaTensor dense_y_grad;
  phi::MetaTensor meta_y_grad(&dense_y_grad);
  paddle::dialect::IrMetaTensor dense_bias_grad;
  phi::MetaTensor meta_bias_grad(&dense_bias_grad);

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

void FusedGemmEpilogueGradOp::Verify() {}

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
  std::vector<paddle::dialect::IrMetaTensor> vec_dense_out_grad;
  for (size_t i = 0; i < static_cast<size_t>(out_grad.size()); i++) {
    vec_dense_out_grad.push_back(paddle::dialect::IrMetaTensor(
        paddle::dialect::TransToPhiDataType(
            out_grad[i].dyn_cast<paddle::dialect::DenseTensorType>().dtype()),
        out_grad[i].dyn_cast<paddle::dialect::DenseTensorType>().dims(),
        out_grad[i].dyn_cast<paddle::dialect::DenseTensorType>().data_layout(),
        out_grad[i].dyn_cast<paddle::dialect::DenseTensorType>().lod(),
        out_grad[i].dyn_cast<paddle::dialect::DenseTensorType>().offset()));
  }
  std::vector<phi::MetaTensor> vec_meta_out_grad;
  for (size_t i = 0; i < vec_dense_out_grad.size(); i++) {
    vec_meta_out_grad.push_back(phi::MetaTensor(&vec_dense_out_grad[i]));
  }

  std::vector<const phi::MetaTensor *> meta_out_grad;
  for (size_t i = 0; i < static_cast<size_t>(vec_meta_out_grad.size()); i++) {
    meta_out_grad.push_back(&vec_meta_out_grad[i]);
  }
  paddle::dialect::IrMetaTensor dense_x_grad;
  phi::MetaTensor meta_x_grad(&dense_x_grad);

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

  std::vector<paddle::dialect::IrMetaTensor> vec_dense_out_grad;
  for (size_t i = 0; i < static_cast<size_t>(out_grad.size()); i++) {
    vec_dense_out_grad.push_back(paddle::dialect::IrMetaTensor(
        TransToPhiDataType(
            out_grad[i].dyn_cast<paddle::dialect::DenseTensorType>().dtype()),
        out_grad[i].dyn_cast<paddle::dialect::DenseTensorType>().dims(),
        out_grad[i].dyn_cast<paddle::dialect::DenseTensorType>().data_layout(),
        out_grad[i].dyn_cast<paddle::dialect::DenseTensorType>().lod(),
        out_grad[i].dyn_cast<paddle::dialect::DenseTensorType>().offset()));
  }
  std::vector<phi::MetaTensor> vec_meta_out_grad;
  for (size_t i = 0; i < vec_dense_out_grad.size(); i++) {
    vec_meta_out_grad.push_back(phi::MetaTensor(&vec_dense_out_grad[i]));
  }

  std::vector<const phi::MetaTensor *> meta_out_grad;
  for (size_t i = 0; i < static_cast<size_t>(vec_meta_out_grad.size()); i++) {
    meta_out_grad.push_back(&vec_meta_out_grad[i]);
  }
  paddle::dialect::IrMetaTensor dense_x_grad;
  phi::MetaTensor meta_x_grad(&dense_x_grad);

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

void SplitGradOp::Verify() {
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

void IfOp::Build(pir::Builder &builder,             // NOLINT
                 pir::OperationArgument &argument,  // NOLINT
                 pir::Value cond,
                 std::vector<pir::Type> &&output_types) {
  VLOG(4) << "Start build IfOp";

  argument.AddRegions(2u);
  argument.AddInput(cond);
  argument.output_types.swap(output_types);
}
pir::Block *IfOp::true_block() {
  pir::Region &true_region = (*this)->region(0);
  if (true_region.empty()) true_region.emplace_back();
  return true_region.front();
}
pir::Block *IfOp::false_block() {
  pir::Region &false_region = (*this)->region(1);
  if (false_region.empty()) false_region.emplace_back();
  return false_region.front();
}
void IfOp::Print(pir::IrPrinter &printer) {
  auto &os = printer.os;
  auto op = operation();
  printer.PrintOpResult(op);
  os << " = pd_op.if";
  printer.PrintOpOperands(op);
  os << " -> ";
  printer.PrintOpReturnType(op);
  os << "{";
  for (auto item : *true_block()) {
    os << "\n  ";
    printer.PrintOperation(item);
  }
  os << "\n } else {";
  for (auto item : *false_block()) {
    os << "\n  ";
    printer.PrintOperation(item);
  }
  os << "\n }";
}
void IfOp::Verify() {}

void WhileOp::Build(pir::Builder &builder,             // NOLINT
                    pir::OperationArgument &argument,  // NOLINT
                    const std::vector<pir::Value> &inputs,
                    const std::vector<pir::Type> &output_types) {
  // auto insert_point = builder.insert_point();
  argument.AddInputs(inputs);
  argument.AddOutputs(output_types);
  argument.AddRegion(nullptr);
  argument.AddRegion(nullptr);
}
pir::Block *WhileOp::cond_block() {
  pir::Region &cond_region = (*this)->region(0);
  if (cond_region.empty()) cond_region.emplace_back();
  return cond_region.front();
}
pir::Block *WhileOp::body_block() {
  pir::Region &body_region = (*this)->region(1);
  if (body_region.empty()) body_region.emplace_back();
  return body_region.front();
}
}  // namespace dialect
}  // namespace paddle

IR_DEFINE_EXPLICIT_TYPE_ID(paddle::dialect::AddNOp)
IR_DEFINE_EXPLICIT_TYPE_ID(paddle::dialect::SplitGradOp)
IR_DEFINE_EXPLICIT_TYPE_ID(paddle::dialect::AddN_Op)
IR_DEFINE_EXPLICIT_TYPE_ID(paddle::dialect::AddNWithKernelOp)
IR_DEFINE_EXPLICIT_TYPE_ID(paddle::dialect::FusedGemmEpilogueOp)
IR_DEFINE_EXPLICIT_TYPE_ID(paddle::dialect::FusedGemmEpilogueGradOp)
IR_DEFINE_EXPLICIT_TYPE_ID(paddle::dialect::IfOp)
IR_DEFINE_EXPLICIT_TYPE_ID(paddle::dialect::WhileOp)
