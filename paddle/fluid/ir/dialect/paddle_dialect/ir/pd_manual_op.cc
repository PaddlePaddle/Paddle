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

#include "paddle/fluid/ir/dialect/paddle_dialect/ir/pd_manual_op.h"
#include "paddle/fluid/ir/dialect/paddle_dialect/ir/pd_attribute.h"
#include "paddle/fluid/ir/dialect/paddle_dialect/ir/pd_op.h"
#include "paddle/fluid/ir/dialect/paddle_dialect/ir/pd_type.h"
#include "paddle/ir/core/builtin_attribute.h"
#include "paddle/ir/core/builtin_op.h"
#include "paddle/ir/core/builtin_type.h"
#include "paddle/ir/core/ir_context.h"
#include "paddle/phi/api/lib/utils/allocator.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/infermeta/backward.h"
#include "paddle/phi/infermeta/multiary.h"

namespace paddle {
namespace dialect {

OpInfoTuple AddNOp::GetOpInfo() {
  std::vector<paddle::dialect::OpInputInfo> inputs = {
      OpInputInfo("inputs",
                  "ir::VectorType<paddle::dialect::DenseTensorType>",
                  false,
                  false,
                  false,
                  true)};
  std::vector<paddle::dialect::OpAttributeInfo> attributes = {};
  std::vector<paddle::dialect::OpOutputInfo> outputs = {
      OpOutputInfo("out", "paddle::dialect::DenseTensorType", false, false)};
  paddle::dialect::OpRunTimeInfo run_time_info = OpRunTimeInfo(
      "AddNInferMeta", {"inputs"}, {"add_n"}, {"inputs"}, {}, {}, {}, {});

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
    if (auto vec_type = (*this)->operand(0).type().dyn_cast<ir::VectorType>()) {
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

void AddNOp::Build(ir::Builder &builder,             // NOLINT
                   ir::OperationArgument &argument,  // NOLINT
                   ir::OpResult inputs) {
  VLOG(4) << "Builder construction inputs";
  std::vector<ir::OpResult> argument_inputs = {inputs};
  argument.AddOperands(argument_inputs.begin(), argument_inputs.end());

  VLOG(4) << "Builder construction attributes";

  VLOG(4) << "Builder construction outputs";
  ir::VectorType x = inputs.type().dyn_cast<ir::VectorType>();
  (void)x;

  std::vector<phi::DenseTensor> vec_dense_x;
  for (size_t i = 0; i < x.size(); i++) {
    vec_dense_x.push_back(phi::DenseTensor(
        std::make_unique<paddle::experimental::DefaultAllocator>(
            paddle::platform::CPUPlace())
            .get(),
        phi::DenseTensorMeta(
            TransToPhiDataType(
                x[i].dyn_cast<paddle::dialect::DenseTensorType>().dtype()),
            x[i].dyn_cast<paddle::dialect::DenseTensorType>().dims(),
            x[i].dyn_cast<paddle::dialect::DenseTensorType>().data_layout(),
            x[i].dyn_cast<paddle::dialect::DenseTensorType>().lod(),
            x[i].dyn_cast<paddle::dialect::DenseTensorType>().offset())));
  }
  std::vector<phi::MetaTensor> vec_meta_x;
  for (size_t i = 0; i < vec_dense_x.size(); i++) {
    vec_meta_x.push_back(phi::MetaTensor(&vec_dense_x[i]));
  }

  std::vector<const phi::MetaTensor *> meta_x;
  for (size_t i = 0; i < static_cast<size_t>(vec_meta_x.size()); i++) {
    meta_x.push_back(&vec_meta_x[i]);
  }
  phi::DenseTensor dense_out;
  phi::MetaTensor meta_out(&dense_out);

  phi::AddNInferMeta(meta_x, &meta_out);

  std::vector<ir::Type> argument_outputs;
  ir::Type out_dense_tensor_type = paddle::dialect::DenseTensorType::get(
      ir::IrContext::Instance(),
      TransToIrDataType(dense_out.dtype()),
      dense_out.dims(),
      dense_out.layout(),
      dense_out.lod(),
      dense_out.offset());
  argument_outputs.push_back(out_dense_tensor_type);
  argument.AddOutputs(argument_outputs.begin(), argument_outputs.end());
}

void AddNOp::InferMeta(phi::InferMetaContext *infer_meta) {
  auto fn = PD_INFER_META(phi::AddNInferMeta);
  fn(infer_meta);
}

OpInfoTuple AddN_Op::GetOpInfo() {
  std::vector<paddle::dialect::OpInputInfo> inputs = {
      paddle::dialect::OpInputInfo(
          "inputs",
          "ir::VectorType<paddle::dialect::DenseTensorType>",
          false,
          false,
          false,
          true)};
  std::vector<paddle::dialect::OpAttributeInfo> attributes = {};
  std::vector<paddle::dialect::OpOutputInfo> outputs = {
      paddle::dialect::OpOutputInfo(
          "out", "paddle::dialect::DenseTensorType", false, false)};
  paddle::dialect::OpRunTimeInfo run_time_info = paddle::dialect::OpRunTimeInfo(
      "AddNInferMeta", {"inputs"}, {"add_n"}, {"inputs"}, {}, {}, {}, {});
  return std::make_tuple(inputs, attributes, outputs, run_time_info, "add_n_");
}

void AddN_Op::Build(ir::Builder &builder,
                    ir::OperationArgument &argument,
                    ir::OpResult inputs_) {
  VLOG(4) << "Builder construction inputs";
  std::vector<ir::OpResult> argument_inputs = {inputs_};
  argument.AddOperands(argument_inputs.begin(), argument_inputs.end());

  VLOG(4) << "Builder construction attributes";

  VLOG(4) << "Builder construction outputs";
  ir::VectorType inputs = inputs_.type().dyn_cast<ir::VectorType>();
  (void)inputs;
  std::vector<phi::DenseTensor> vec_dense_inputs;
  for (size_t i = 0; i < static_cast<size_t>(inputs.size()); i++) {
    vec_dense_inputs.push_back(phi::DenseTensor(
        std::make_unique<paddle::experimental::DefaultAllocator>(
            paddle::platform::CPUPlace())
            .get(),
        phi::DenseTensorMeta(
            paddle::dialect::TransToPhiDataType(
                inputs[i].dyn_cast<paddle::dialect::DenseTensorType>().dtype()),
            inputs[i].dyn_cast<paddle::dialect::DenseTensorType>().dims(),
            inputs[i]
                .dyn_cast<paddle::dialect::DenseTensorType>()
                .data_layout(),
            inputs[i].dyn_cast<paddle::dialect::DenseTensorType>().lod(),
            inputs[i].dyn_cast<paddle::dialect::DenseTensorType>().offset())));
  }
  std::vector<phi::MetaTensor> vec_meta_inputs;
  for (size_t i = 0; i < vec_dense_inputs.size(); i++) {
    vec_meta_inputs.push_back(phi::MetaTensor(&vec_dense_inputs[i]));
  }

  std::vector<const phi::MetaTensor *> meta_inputs;
  for (size_t i = 0; i < static_cast<size_t>(vec_meta_inputs.size()); i++) {
    meta_inputs.push_back(&vec_meta_inputs[i]);
  }
  phi::DenseTensor dense_out;
  phi::MetaTensor meta_out(&dense_out);

  phi::AddNInferMeta(meta_inputs, &meta_out);

  std::vector<ir::Type> argument_outputs;
  ir::Type out_dense_tensor_type = paddle::dialect::DenseTensorType::get(
      ir::IrContext::Instance(),
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
            (*this)->operand_source(0).type().dyn_cast<ir::VectorType>()) {
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
          "ir::VectorType<paddle::dialect::DenseTensorType>",
          false,
          false,
          false,
          true)};
  std::vector<paddle::dialect::OpAttributeInfo> attributes = {};
  std::vector<paddle::dialect::OpOutputInfo> outputs = {
      paddle::dialect::OpOutputInfo(
          "out", "paddle::dialect::DenseTensorType", false, false)};
  paddle::dialect::OpRunTimeInfo run_time_info = paddle::dialect::OpRunTimeInfo(
      "AddNInferMeta", {"inputs"}, {"add_n"}, {"inputs"}, {}, {}, {}, {});
  return std::make_tuple(
      inputs, attributes, outputs, run_time_info, "add_n_with_kernel");
}

void AddNWithKernelOp::Build(ir::Builder &builder,
                             ir::OperationArgument &argument,
                             ir::OpResult inputs_) {
  VLOG(4) << "Builder construction inputs";
  std::vector<ir::OpResult> argument_inputs = {inputs_};
  argument.AddOperands(argument_inputs.begin(), argument_inputs.end());

  VLOG(4) << "Builder construction attributes";

  VLOG(4) << "Builder construction outputs";
  ir::VectorType inputs = inputs_.type().dyn_cast<ir::VectorType>();
  (void)inputs;
  std::vector<phi::DenseTensor> vec_dense_inputs;
  for (size_t i = 0; i < static_cast<size_t>(inputs.size()); i++) {
    vec_dense_inputs.push_back(phi::DenseTensor(
        std::make_unique<paddle::experimental::DefaultAllocator>(
            paddle::platform::CPUPlace())
            .get(),
        phi::DenseTensorMeta(
            paddle::dialect::TransToPhiDataType(
                inputs[i].dyn_cast<paddle::dialect::DenseTensorType>().dtype()),
            inputs[i].dyn_cast<paddle::dialect::DenseTensorType>().dims(),
            inputs[i]
                .dyn_cast<paddle::dialect::DenseTensorType>()
                .data_layout(),
            inputs[i].dyn_cast<paddle::dialect::DenseTensorType>().lod(),
            inputs[i].dyn_cast<paddle::dialect::DenseTensorType>().offset())));
  }
  std::vector<phi::MetaTensor> vec_meta_inputs;
  for (size_t i = 0; i < vec_dense_inputs.size(); i++) {
    vec_meta_inputs.push_back(phi::MetaTensor(&vec_dense_inputs[i]));
  }

  std::vector<const phi::MetaTensor *> meta_inputs;
  for (size_t i = 0; i < static_cast<size_t>(vec_meta_inputs.size()); i++) {
    meta_inputs.push_back(&vec_meta_inputs[i]);
  }
  phi::DenseTensor dense_out;
  phi::MetaTensor meta_out(&dense_out);

  phi::AddNInferMeta(meta_inputs, &meta_out);

  std::vector<ir::Type> argument_outputs;
  ir::Type out_dense_tensor_type = paddle::dialect::DenseTensorType::get(
      ir::IrContext::Instance(),
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
            (*this)->operand_source(0).type().dyn_cast<ir::VectorType>()) {
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

const char *SplitGradOp::attributes_name[1] = {"axis"};

OpInfoTuple SplitGradOp::GetOpInfo() {
  std::vector<paddle::dialect::OpInputInfo> inputs = {
      OpInputInfo("out_grad",
                  "ir::VectorType<paddle::dialect::DenseTensorType>",
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
                    {"concat"},
                    {"out_grad", "axis"},
                    {"out_grad"},
                    {},
                    {},
                    {});

  return std::make_tuple(
      inputs, attributes, outputs, run_time_info, "split_grad");
}

void SplitGradOp::Build(ir::Builder &builder,
                        ir::OperationArgument &argument,
                        ir::OpResult out_grad_,
                        float axis) {
  // Generate scalar mutable attribute: axis
  paddle::dialect::FullOp full_axis_op = builder.Build<paddle::dialect::FullOp>(
      std::vector<int64_t>{1}, axis, phi::DataType::FLOAT32, phi::CPUPlace());
  ir::OpResult axis_ = full_axis_op->result(0);

  VLOG(4) << "Builder construction inputs";
  std::vector<ir::OpResult> argument_inputs = {out_grad_, axis_};
  argument.AddOperands(argument_inputs.begin(), argument_inputs.end());

  VLOG(4) << "Builder construction attributes";

  VLOG(4) << "Builder construction outputs";
  ir::VectorType out_grad = out_grad_.type().dyn_cast<ir::VectorType>();
  std::vector<phi::DenseTensor> vec_dense_out_grad;
  for (size_t i = 0; i < static_cast<size_t>(out_grad.size()); i++) {
    vec_dense_out_grad.push_back(phi::DenseTensor(
        std::make_unique<paddle::experimental::DefaultAllocator>(
            paddle::platform::CPUPlace())
            .get(),
        phi::DenseTensorMeta(
            paddle::dialect::TransToPhiDataType(
                out_grad[i]
                    .dyn_cast<paddle::dialect::DenseTensorType>()
                    .dtype()),
            out_grad[i].dyn_cast<paddle::dialect::DenseTensorType>().dims(),
            out_grad[i]
                .dyn_cast<paddle::dialect::DenseTensorType>()
                .data_layout(),
            out_grad[i].dyn_cast<paddle::dialect::DenseTensorType>().lod(),
            out_grad[i]
                .dyn_cast<paddle::dialect::DenseTensorType>()
                .offset())));
  }
  std::vector<phi::MetaTensor> vec_meta_out_grad;
  for (size_t i = 0; i < vec_dense_out_grad.size(); i++) {
    vec_meta_out_grad.push_back(phi::MetaTensor(&vec_dense_out_grad[i]));
  }

  std::vector<const phi::MetaTensor *> meta_out_grad;
  for (size_t i = 0; i < static_cast<size_t>(vec_meta_out_grad.size()); i++) {
    meta_out_grad.push_back(&vec_meta_out_grad[i]);
  }
  phi::DenseTensor dense_x_grad;
  phi::MetaTensor meta_x_grad(&dense_x_grad);

  phi::ConcatInferMeta(meta_out_grad, axis, &meta_x_grad);

  std::vector<ir::Type> argument_outputs;
  ir::Type x_grad_dense_tensor_type = paddle::dialect::DenseTensorType::get(
      ir::IrContext::Instance(),
      paddle::dialect::TransToIrDataType(dense_x_grad.dtype()),
      dense_x_grad.dims(),
      dense_x_grad.layout(),
      dense_x_grad.lod(),
      dense_x_grad.offset());
  argument_outputs.push_back(x_grad_dense_tensor_type);
  argument.AddOutputs(argument_outputs.begin(), argument_outputs.end());
}

void SplitGradOp::Build(ir::Builder &builder,
                        ir::OperationArgument &argument,
                        ir::OpResult out_grad_,
                        ir::OpResult axis_) {
  VLOG(4) << "Builder construction inputs";
  std::vector<ir::OpResult> argument_inputs = {out_grad_, axis_};
  argument.AddOperands(argument_inputs.begin(), argument_inputs.end());

  VLOG(4) << "Builder construction attributes";

  VLOG(4) << "Builder construction outputs";
  ir::VectorType out_grad = out_grad_.type().dyn_cast<ir::VectorType>();
  int axis = axis_.owner()
                 ->dyn_cast<paddle::dialect::FullOp>()
                 .attributes()
                 .at("value")
                 .dyn_cast<paddle::dialect::ScalarAttribute>()
                 .data()
                 .to<int>();

  std::vector<phi::DenseTensor> vec_dense_out_grad;
  for (size_t i = 0; i < static_cast<size_t>(out_grad.size()); i++) {
    vec_dense_out_grad.push_back(phi::DenseTensor(
        std::make_unique<paddle::experimental::DefaultAllocator>(
            paddle::platform::CPUPlace())
            .get(),
        phi::DenseTensorMeta(
            TransToPhiDataType(out_grad[i]
                                   .dyn_cast<paddle::dialect::DenseTensorType>()
                                   .dtype()),
            out_grad[i].dyn_cast<paddle::dialect::DenseTensorType>().dims(),
            out_grad[i]
                .dyn_cast<paddle::dialect::DenseTensorType>()
                .data_layout(),
            out_grad[i].dyn_cast<paddle::dialect::DenseTensorType>().lod(),
            out_grad[i]
                .dyn_cast<paddle::dialect::DenseTensorType>()
                .offset())));
  }
  std::vector<phi::MetaTensor> vec_meta_out_grad;
  for (size_t i = 0; i < vec_dense_out_grad.size(); i++) {
    vec_meta_out_grad.push_back(phi::MetaTensor(&vec_dense_out_grad[i]));
  }

  std::vector<const phi::MetaTensor *> meta_out_grad;
  for (size_t i = 0; i < static_cast<size_t>(vec_meta_out_grad.size()); i++) {
    meta_out_grad.push_back(&vec_meta_out_grad[i]);
  }
  phi::DenseTensor dense_x_grad;
  phi::MetaTensor meta_x_grad(&dense_x_grad);

  phi::ConcatInferMeta(meta_out_grad, axis, &meta_x_grad);

  std::vector<ir::Type> argument_outputs;
  ir::Type x_grad_dense_tensor_type = paddle::dialect::DenseTensorType::get(
      ir::IrContext::Instance(),
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
            (*this)->operand_source(0).type().dyn_cast<ir::VectorType>()) {
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

void IfOp::Build(ir::Builder &builder,             // NOLINT
                 ir::OperationArgument &argument,  // NOLINT
                 ir::OpResult cond,
                 std::vector<ir::Type> &&output_types) {
  argument.num_regions = 2;
  argument.AddOperand(cond);
  argument.output_types.swap(output_types);
}
ir::Block *IfOp::true_block() {
  ir::Region &true_region = (*this)->region(0);
  if (true_region.empty()) true_region.emplace_back();
  return true_region.front();
}
ir::Block *IfOp::false_block() {
  ir::Region &false_region = (*this)->region(1);
  if (false_region.empty()) false_region.emplace_back();
  return false_region.front();
}
void IfOp::Print(ir::IrPrinter &printer) {
  auto &os = printer.os;
  auto op = operation();
  printer.PrintOpResult(op);
  os << " = pd.if";
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
}  // namespace dialect
}  // namespace paddle

IR_DEFINE_EXPLICIT_TYPE_ID(paddle::dialect::AddNOp)
IR_DEFINE_EXPLICIT_TYPE_ID(paddle::dialect::SplitGradOp)
IR_DEFINE_EXPLICIT_TYPE_ID(paddle::dialect::AddN_Op)
IR_DEFINE_EXPLICIT_TYPE_ID(paddle::dialect::AddNWithKernelOp)
IR_DEFINE_EXPLICIT_TYPE_ID(paddle::dialect::IfOp)
