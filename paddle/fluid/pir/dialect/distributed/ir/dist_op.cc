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

#include "paddle/fluid/pir/dialect/distributed/ir/dist_op.h"
#include "paddle/fluid/pir/dialect/distributed/ir/dist_attribute.h"
#include "paddle/fluid/pir/dialect/distributed/ir/dist_type.h"
#include "paddle/fluid/pir/dialect/operator/ir/api_builder.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_attribute.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_type.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/pir/include/core/builtin_attribute.h"
#include "paddle/pir/include/core/builtin_op.h"
#include "paddle/pir/include/core/ir_context.h"

namespace paddle::dialect {

const char* ShardTensorOp::attributes_name[1] = {"op_dist_attr"};  // NOLINT
const char* ReshardOp::attributes_name[1] = {"op_dist_attr"};      // NOLINT

void ShardTensorOp::VerifySig() {
  VLOG(4)
      << "Start Verifying inputs, outputs and attributes for: ShardTensorOp.";
  VLOG(4) << "Verifying inputs:";
  {
    auto input_size = num_operands();
    PADDLE_ENFORCE_EQ(input_size,
                      1u,
                      common::errors::PreconditionNotMet(
                          "The size of inputs must be equal to 1."));
    PADDLE_ENFORCE_EQ((*this)
                          ->operand_source(0)
                          .type()
                          .isa<paddle::dialect::DenseTensorType>(),
                      true,
                      common::errors::PreconditionNotMet(
                          "Type validation failed for the 0th input."));
  }
  VLOG(4) << "Verifying attributes:";
  {
    auto& attributes = this->attributes();
    PADDLE_ENFORCE_EQ((attributes.count("op_dist_attr") > 0 &&
                       attributes.at("op_dist_attr")
                           .isa<paddle::dialect::OperationDistAttribute>()),
                      true,
                      common::errors::PreconditionNotMet(
                          "Type of attribute: op_dist_attr is not right."));
  }
  VLOG(4) << "Verifying outputs:";
  {
    auto output_size = num_results();
    PADDLE_ENFORCE_EQ(
        output_size,
        1u,
        common::errors::PreconditionNotMet(
            "The size %d of outputs must be equal to 1.", output_size));
    PADDLE_ENFORCE_EQ(
        (*this)->result(0).type().isa<paddle::dialect::DistDenseTensorType>(),
        true,
        common::errors::PreconditionNotMet(
            "Type validation failed for the 0th output."));
  }

  VLOG(4) << "Verifying op dist attrs:";
  {
    auto op_dist_attr =
        this->attribute<paddle::dialect::OperationDistAttribute>(
            "op_dist_attr");
    PADDLE_ENFORCE_EQ(op_dist_attr.num_operands(),
                      0u,
                      phi::errors::PreconditionNotMet(
                          "The op_dist_attr input size must be equal to 0."));

    PADDLE_ENFORCE_EQ(
        op_dist_attr.num_results(),
        num_results(),
        phi::errors::PreconditionNotMet("The op_dist_attr output size %d must "
                                        "be equal to op output size %d.",
                                        op_dist_attr.num_results(),
                                        num_results()));
  }
  VLOG(4) << "End Verifying for: ShardTensorOp.";
}

void ShardTensorOp::Build(pir::Builder& builder,
                          pir::OperationArgument& argument,
                          pir::Value input,
                          pir::AttributeMap attributes) {
  VLOG(4) << "Start build ShardOp";

  // Temporary restriction, will support input use_empty false in the future
  PADDLE_ENFORCE_EQ(
      input.use_empty(),
      true,
      common::errors::PreconditionNotMet("'input' use_empty is not true"));

  paddle::dialect::DenseTensorType input_tensor_type;
  if (input.type().isa<paddle::dialect::DenseTensorType>()) {
    input_tensor_type =
        input.type().dyn_cast<paddle::dialect::DenseTensorType>();
  } else {
    PADDLE_THROW(common::errors::Unimplemented(
        "Only support paddle::dialect::DenseTensorType"));
  }

  PADDLE_ENFORCE_NE(
      attributes.find("tensor_dist_attr"),
      attributes.end(),
      common::errors::NotFound(
          "'tensor_dist_attr' Attribute is expected for ShardOp"));
  paddle::dialect::TensorDistAttribute tensor_dist_attr =
      attributes.at("tensor_dist_attr")
          .dyn_cast<paddle::dialect::TensorDistAttribute>();

  VLOG(4) << "Builder construction inputs";
  argument.AddInput(input);

  VLOG(4) << "Builder construction attributes";
  auto process_mesh_attr = tensor_dist_attr.process_mesh_attr();
  auto dims_mapping = tensor_dist_attr.dims_mapping();

  pir::Attribute op_dist_attr = OperationDistAttribute::get(
      pir::IrContext::Instance(),
      process_mesh_attr,
      std::vector<pir::Attribute>(),
      std::vector<pir::Attribute>{tensor_dist_attr});
  argument.AddAttribute("op_dist_attr", op_dist_attr);

  VLOG(4) << "Builder construction outputs";
  auto global_dims = input_tensor_type.dims();
  auto process_mesh_shape = process_mesh_attr.shape();
  PADDLE_ENFORCE_EQ(static_cast<int>(dims_mapping.size()),
                    global_dims.size(),
                    common::errors::PreconditionNotMet(
                        "dims_mapping size %d does not match input size %d",
                        dims_mapping.size(),
                        global_dims.size()));
  auto local_shape = InferLocalDDim(global_dims, tensor_dist_attr);
  pir::Type out_dist_tensor_type =
      paddle::dialect::DistDenseTensorType::get(pir::IrContext::Instance(),
                                                input_tensor_type,
                                                tensor_dist_attr,
                                                local_shape);
  argument.AddOutput(out_dist_tensor_type);
  ::pir::PassStopGradientsDefaultly(argument);
}

OpInfoTuple ReshardOp::GetOpInfo() {
  return OpInfoTuple(
      {OpInputInfo()}, {}, {OpOutputInfo()}, OpRunTimeInfo(), "reshard");
}

std::vector<std::vector<pir::Value>> ReshardOp::Vjp(
    pir::Operation* op,
    const std::vector<std::vector<pir::Value>>& inputs_,
    const std::vector<std::vector<pir::Value>>& outputs,
    const std::vector<std::vector<pir::Value>>& out_grads,
    const std::vector<std::vector<bool>>& stop_gradients) {
  VLOG(6) << "Start call vjp for reshard op.";
  PADDLE_ENFORCE_EQ(
      inputs_.size(),
      1,
      common::errors::InvalidArgument("reshard op's inputs' size should be 1"));
  PADDLE_ENFORCE_EQ(inputs_[0].size(),
                    1,
                    common::errors::InvalidArgument(
                        "reshard op's inputs[0]'s size should be 1"));
  auto dist_type = inputs_[0][0].type().dyn_cast<DistTypeInterface>();

  PADDLE_ENFORCE_NOT_NULL(
      dist_type,
      common::errors::InvalidArgument(
          "Currently, reshard op's inputs type must be dist type."));

  PADDLE_ENFORCE_EQ(out_grads.size(),
                    1,
                    common::errors::InvalidArgument(
                        "reshard op's outputs  grad size should be 1"));

  PADDLE_ENFORCE_EQ(out_grads[0].size(),
                    1,
                    common::errors::InvalidArgument(
                        "reshard op's outputs grad[0] size should be 1"));

  auto& builder = *ApiBuilder::Instance().GetBuilder();

  auto grad_op =
      builder.Build<ReshardOp>(out_grads[0][0], dist_type.tensor_dist_attr());

  VLOG(6) << "End call vjp for reshard op.";

  return {std::vector<pir::Value>{grad_op->result(0)}};
}
void ReshardOp::VerifySig() {
  VLOG(4) << "Start Verifying inputs, outputs and attributes for: ReshardOp.";
  VLOG(4) << "Verifying inputs:";
  {
    auto input_size = num_operands();
    PADDLE_ENFORCE_EQ(
        input_size,
        1u,
        common::errors::PreconditionNotMet(
            "The size %d of inputs must be equal to 1.", input_size));
    PADDLE_ENFORCE_EQ(!(*this)->operand_source(0) ||
                          (*this)  // reshard allow NULL TYPE as input
                              ->operand_source(0)
                              .type()
                              .isa<paddle::dialect::DistDenseTensorType>(),
                      true,
                      common::errors::PreconditionNotMet(
                          "Type validation failed for the 0th input."));
  }
  VLOG(4) << "Verifying attributes:";
  {
    auto& attributes = this->attributes();
    PADDLE_ENFORCE_EQ((attributes.count("op_dist_attr") > 0 &&
                       attributes.at("op_dist_attr")
                           .isa<paddle::dialect::OperationDistAttribute>()),
                      true,
                      common::errors::PreconditionNotMet(
                          "Type of attribute: op_dist_attr is not right."));
  }
  VLOG(4) << "Verifying outputs:";
  {
    auto output_size = num_results();
    PADDLE_ENFORCE_EQ(
        output_size,
        1u,
        common::errors::PreconditionNotMet(
            "The size %d of outputs must be equal to 1.", output_size));
    PADDLE_ENFORCE_EQ(
        !(*this)->result(0) ||
            (*this)
                ->result(0)
                .type()
                .isa<paddle::dialect::DistDenseTensorType>(),  // reshard allow
                                                               // NULL TYPE as
                                                               // output
        true,
        common::errors::PreconditionNotMet(
            "Type validation failed for the 0th output."));
  }

  VLOG(4) << "Verifying op dist attrs:";
  {
    auto op_dist_attr =
        this->attribute<paddle::dialect::OperationDistAttribute>(
            "op_dist_attr");
    PADDLE_ENFORCE_EQ(
        op_dist_attr.num_operands(),
        1u,
        common::errors::PreconditionNotMet(
            "The op_dist_attr input size of reshard op must be equal to 1."));

    PADDLE_ENFORCE_EQ(op_dist_attr.num_results(),
                      num_results(),
                      phi::errors::PreconditionNotMet(
                          "The op_dist_attr output size of reshard op must be "
                          "equal to op output size."));
  }
  VLOG(4) << "End Verifying for: ShardTensorOp.";
}

ProcessMeshAttribute MergeMeshes(const ProcessMeshAttribute& mesh1,
                                 const ProcessMeshAttribute& mesh2) {
  if (mesh1 == mesh2) return mesh1;
  // Combine the two ids
  std::vector<int64_t> merged_ids;
  std::vector<int64_t> ids1 = mesh1.process_ids();
  std::vector<int64_t> ids2 = mesh2.process_ids();

  merged_ids.reserve(ids1.size() + ids2.size());
  merged_ids.insert(merged_ids.end(), ids1.begin(), ids1.end());
  merged_ids.insert(merged_ids.end(), ids2.begin(), ids2.end());

  // Remove duplicates
  std::sort(merged_ids.begin(), merged_ids.end());
  auto last = std::unique(merged_ids.begin(), merged_ids.end());
  merged_ids.erase(last, merged_ids.end());

  return ProcessMeshAttribute::get(
      pir::IrContext::Instance(),
      {static_cast<int64_t>(merged_ids.size())},  // flatten mesh shape
      merged_ids,
      {"merged"});
}

void ReshardOp::Build(pir::Builder& builder,
                      pir::OperationArgument& argument,
                      pir::Value input,
                      TensorDistAttribute tensor_dist_attr) {
  paddle::dialect::DistDenseTensorType input_tensor_type;
  if (input.type().isa<paddle::dialect::DistDenseTensorType>()) {
    input_tensor_type =
        input.type().dyn_cast<paddle::dialect::DistDenseTensorType>();
  } else {
    PADDLE_THROW(common::errors::Unimplemented(
        "Only support paddle::dialect::DistDenseTensorType"));
  }

  VLOG(4) << "Builder construction inputs";
  argument.AddInput(input);

  VLOG(4) << "Builder construction attributes";
  pir::Attribute op_dist_attr = OperationDistAttribute::get(
      pir::IrContext::Instance(),
      MergeMeshes(input_tensor_type.tensor_dist_attr().process_mesh_attr(),
                  tensor_dist_attr.process_mesh_attr()),
      std::vector<pir::Attribute>{input_tensor_type.tensor_dist_attr()},
      std::vector<pir::Attribute>{tensor_dist_attr});
  argument.AddAttribute("op_dist_attr", op_dist_attr);

  VLOG(4) << "Builder construction outputs";
  auto global_dims = input_tensor_type.global_ddim();
  auto process_mesh_attr = tensor_dist_attr.process_mesh_attr();
  auto dims_mapping = tensor_dist_attr.dims_mapping();

  auto process_mesh_shape = process_mesh_attr.shape();
  PADDLE_ENFORCE_EQ(static_cast<int>(dims_mapping.size()),
                    global_dims.size(),
                    common::errors::PreconditionNotMet(
                        "dst dims_mapping size %d does not match input size %d",
                        dims_mapping.size(),
                        global_dims.size()));

  auto local_shape = InferLocalDDim(global_dims, tensor_dist_attr);
  pir::Type out_dist_tensor_type = paddle::dialect::DistDenseTensorType::get(
      pir::IrContext::Instance(),
      input_tensor_type.dense_tensor_type(),
      tensor_dist_attr,
      local_shape);
  argument.AddOutput(out_dist_tensor_type);
  ::pir::PassStopGradientsDefaultly(argument);
}

}  // namespace paddle::dialect

IR_DEFINE_EXPLICIT_TYPE_ID(paddle::dialect::ShardTensorOp)
IR_DEFINE_EXPLICIT_TYPE_ID(paddle::dialect::ReshardOp)
