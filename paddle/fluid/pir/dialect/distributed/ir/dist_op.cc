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
#include "paddle/fluid/pir/dialect/distributed/ir/dist_tools.h"
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
const char* MoESubMeshTensorsOp::attributes_name[1] = {
    "op_dist_attr"};  // NOLINT
const char* MoEGlobalMeshTensorOp::attributes_name[1] = {
    "op_dist_attr"};  // NOLINT
const char* DistReshapeOp::attributes_name[3] = {
    "op_dist_attr", "x_placements", "out_placements"};  // NOLINT

template <typename T>
void VerifyOpArgNum(const pir::OpBase* op,
                    size_t num_inputs,
                    size_t num_outputs,
                    size_t op_dist_attr_ninputs,
                    size_t op_dist_attr_noutputs) {
  VLOG(4) << "Start Verifying inputs, outputs and attributes for: "
          << T::name();
  VLOG(4) << "Verifying inputs num:";
  {
    auto input_size = op->num_operands();
    PADDLE_ENFORCE_EQ(input_size,
                      num_inputs,
                      common::errors::PreconditionNotMet(
                          "Mismatched inputs size, expected:%u, "
                          "but received:%u.",
                          num_inputs,
                          input_size));
  }

  VLOG(4) << "Verifying outputs num:";
  {
    auto output_size = op->num_results();
    PADDLE_ENFORCE_EQ(output_size,
                      num_outputs,
                      common::errors::PreconditionNotMet(
                          "Mismatched outputs size, expected:%u, "
                          "but received:%u.",
                          num_outputs,
                          output_size));
  }

  VLOG(4) << "Verifying attributes:";
  {
    auto& attributes = op->attributes();
    PADDLE_ENFORCE_EQ(
        attributes.size(),
        T::attributes_num +
            1,  // stop_gradient is added as an additional attribute
        common::errors::PreconditionNotMet(
            "The attributes num must be equal to %d.", T::attributes_num));

    PADDLE_ENFORCE_EQ((attributes.count("op_dist_attr") > 0 &&
                       attributes.at("op_dist_attr")
                           .isa<paddle::dialect::OperationDistAttribute>()),
                      true,
                      common::errors::PreconditionNotMet(
                          "Type of attribute: op_dist_attr is not right."));
  }

  VLOG(4) << "Verifying op dist attrs:";
  {
    auto op_dist_attr =
        op->attribute<paddle::dialect::OperationDistAttribute>("op_dist_attr");
    PADDLE_ENFORCE_EQ(op_dist_attr.num_operands(),
                      op_dist_attr_ninputs,
                      common::errors::PreconditionNotMet(
                          "The op_dist_attr input size must be equal to %u.",
                          op_dist_attr_ninputs));

    PADDLE_ENFORCE_EQ(op_dist_attr.num_results(),
                      op_dist_attr_noutputs,
                      common::errors::PreconditionNotMet(
                          "The op_dist_attr output size must be equal to %u.",
                          op_dist_attr_noutputs));
  }
  VLOG(4) << "End Verifying inputs, outputs and attributes num for: "
          << T::name();
}

void ShardTensorOp::VerifySig() {
  VerifyOpArgNum<ShardTensorOp>(this, 1u, 1u, 0u, 1u);

  PADDLE_ENFORCE_EQ(
      (*this)->operand_source(0).type().isa<paddle::dialect::DenseTensorType>(),
      true,
      common::errors::PreconditionNotMet(
          "Mismatched input type. ShardTensorOp requires 'DenseTensorType' for "
          "input."));

  PADDLE_ENFORCE_EQ(
      (*this)->result(0).type().isa<paddle::dialect::DistDenseTensorType>(),
      true,
      common::errors::PreconditionNotMet(
          "Mismatched output type. ShardTensorOp requires "
          "'DistDenseTensorType' for output."));

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
  VerifyOpArgNum<ReshardOp>(this, 1u, 1u, 1u, num_results());

  PADDLE_ENFORCE_EQ(!(*this)->operand_source(0) ||
                        (*this)  // reshard allow NULL TYPE as input
                            ->operand_source(0)
                            .type()
                            .isa<paddle::dialect::DistDenseTensorType>(),
                    true,
                    common::errors::PreconditionNotMet(
                        "Type validation failed for the 0th input."));

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

  VLOG(4) << "End Verifying for: ReshardOp.";
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

void DtensorFromLocalOp::Build(pir::Builder& builder,
                               pir::OperationArgument& argument,
                               pir::Value input,
                               TensorDistAttribute tensor_dist_attr) {
  VLOG(4) << "Start build DtensorFromLocalOp";

  paddle::dialect::DenseTensorType local_tensor_type;
  if (input.type().isa<paddle::dialect::DenseTensorType>()) {
    local_tensor_type =
        input.type().dyn_cast<paddle::dialect::DenseTensorType>();
  } else {
    PADDLE_THROW(common::errors::Unimplemented(
        "Only support paddle::dialect::DenseTensorType"));
  }

  VLOG(4) << "Builder construction inputs";
  argument.AddInput(input);

  VLOG(4) << "Builder construction attributes";

  VLOG(4) << "Builder construction outputs";

  auto global_ddim =
      InferGlobalDDim(local_tensor_type.dims(), tensor_dist_attr);
  auto global_tensor =
      dialect::DenseTensorType::get(pir::IrContext::Instance(),
                                    local_tensor_type.dtype(),
                                    global_ddim,
                                    local_tensor_type.data_layout(),
                                    local_tensor_type.lod(),
                                    local_tensor_type.offset());

  pir::Type out_dist_tensor_type =
      paddle::dialect::DistDenseTensorType::get(pir::IrContext::Instance(),
                                                global_tensor,
                                                tensor_dist_attr,
                                                local_tensor_type.dims());
  argument.AddOutput(out_dist_tensor_type);
  ::pir::PassStopGradientsDefaultly(argument);
}

OpInfoTuple DtensorFromLocalOp::GetOpInfo() {
  return OpInfoTuple({OpInputInfo()},
                     {},
                     {OpOutputInfo()},
                     OpRunTimeInfo(),
                     "dtensor_from_local");
}
std::vector<std::vector<pir::Value>> DtensorFromLocalOp::Vjp(
    pir::Operation* op,
    const std::vector<std::vector<pir::Value>>& inputs,
    const std::vector<std::vector<pir::Value>>& outputs,
    const std::vector<std::vector<pir::Value>>& out_grads,
    const std::vector<std::vector<bool>>& stop_gradients) {
  VLOG(6) << "Start call vjp for dtensor_from_local op.";
  PADDLE_ENFORCE_EQ(inputs.size(),
                    1,
                    common::errors::InvalidArgument(
                        "dtensor_from_local op's inputs' size should be 1"));
  PADDLE_ENFORCE_EQ(
      inputs[0].size(),
      1,
      common::errors::InvalidArgument(
          "dtensor_from_local op's inputs[0]'s size should be 1"));

  PADDLE_ENFORCE_EQ(outputs.size(),
                    1,
                    common::errors::InvalidArgument(
                        "dtensor_from_local op's outputs' size should be 1"));
  PADDLE_ENFORCE_EQ(
      outputs[0].size(),
      1,
      common::errors::InvalidArgument(
          "dtensor_from_local op's outputs[0]'s size should be 1"));
  auto dist_type = outputs[0][0].type().dyn_cast<DistTypeInterface>();

  PADDLE_ENFORCE_NOT_NULL(
      dist_type,
      common::errors::InvalidArgument("Currently, dtensor_from_local op's "
                                      "outputs type must be dist type."));

  PADDLE_ENFORCE_EQ(
      out_grads.size(),
      1,
      common::errors::InvalidArgument(
          "dtensor_from_local op's outputs  grad size should be 1"));

  PADDLE_ENFORCE_EQ(
      out_grads[0].size(),
      1,
      common::errors::InvalidArgument(
          "dtensor_from_local op's outputs grad[0] size should be 1"));

  auto& builder = *ApiBuilder::Instance().GetBuilder();

  auto out_grad = out_grads[0][0];

  if (out_grad.type() != outputs[0][0].type()) {
    out_grad = builder.Build<ReshardOp>(out_grad, dist_type.tensor_dist_attr())
                   ->result(0);
  }

  auto grad_op = builder.Build<DtensorToLocalOp>(out_grad);

  VLOG(6) << "End call vjp for dtensor_from_local op.";

  return {std::vector<pir::Value>{grad_op->result(0)}};
}

void DtensorToLocalOp::Build(pir::Builder& builder,
                             pir::OperationArgument& argument,
                             pir::Value input) {
  VLOG(4) << "Start build DtensorToLocalOp";

  VLOG(4) << "Builder construction inputs";
  argument.AddInput(input);

  VLOG(4) << "Builder construction attributes";

  VLOG(4) << "Builder construction outputs";

  auto dist_type = input.type().dyn_cast<DistTypeInterface>();
  if (!dist_type) {
    PADDLE_THROW(common::errors::Unimplemented(
        "The input of DtensorToLocalOp must be dist type."));
  }

  argument.AddOutput(dist_type.local_type());
  ::pir::PassStopGradientsDefaultly(argument);
}

OpInfoTuple DtensorToLocalOp::GetOpInfo() {
  return OpInfoTuple({OpInputInfo()},
                     {},
                     {OpOutputInfo()},
                     OpRunTimeInfo(),
                     "dtensor_to_local");
}

std::vector<std::vector<pir::Value>> DtensorToLocalOp::Vjp(
    pir::Operation* op,
    const std::vector<std::vector<pir::Value>>& inputs,
    const std::vector<std::vector<pir::Value>>& outputs,
    const std::vector<std::vector<pir::Value>>& out_grads,
    const std::vector<std::vector<bool>>& stop_gradients) {
  VLOG(6) << "Start call vjp for dtensor_to_local op.";
  PADDLE_ENFORCE_EQ(inputs.size(),
                    1,
                    common::errors::InvalidArgument(
                        "dtensor_to_local op's inputs' size should be 1"));
  PADDLE_ENFORCE_EQ(inputs[0].size(),
                    1,
                    common::errors::InvalidArgument(
                        "dtensor_to_local op's inputs[0]'s size should be 1"));

  PADDLE_ENFORCE_EQ(outputs.size(),
                    1,
                    common::errors::InvalidArgument(
                        "dtensor_to_local op's outputs' size should be 1"));
  PADDLE_ENFORCE_EQ(outputs[0].size(),
                    1,
                    common::errors::InvalidArgument(
                        "dtensor_to_local op's outputs[0]'s size should be 1"));
  auto dist_type = inputs[0][0].type().dyn_cast<DistTypeInterface>();

  PADDLE_ENFORCE_NOT_NULL(
      dist_type,
      common::errors::InvalidArgument(
          "Currently, dtensor_to_local op's inputs type must be dist type."));

  PADDLE_ENFORCE_EQ(
      out_grads.size(),
      1,
      common::errors::InvalidArgument(
          "dtensor_from_local op's outputs  grad size should be 1"));

  PADDLE_ENFORCE_EQ(
      out_grads[0].size(),
      1,
      common::errors::InvalidArgument(
          "dtensor_from_local op's outputs grad[0] size should be 1"));

  auto& builder = *ApiBuilder::Instance().GetBuilder();

  auto grad_op = builder.Build<DtensorFromLocalOp>(
      out_grads[0][0], dist_type.tensor_dist_attr());

  VLOG(6) << "End call vjp for dtensor_from_local op.";

  return {std::vector<pir::Value>{grad_op->result(0)}};
}

TEST_API void paddle::dialect::MoESubMeshTensorsOp::Build(
    pir::Builder& builder,
    pir::OperationArgument& argument,
    pir::Value input,
    const std::vector<TensorDistAttribute>& local_dist_attrs,
    const TensorDistAttribute& global_dist_attr) {
  VLOG(4) << "Build MoESubMeshTensorsOp";
  paddle::dialect::DistDenseTensorType input_tensor_type;
  if (input.type().isa<paddle::dialect::DistDenseTensorType>()) {
    input_tensor_type =
        input.type().dyn_cast<paddle::dialect::DistDenseTensorType>();
  } else {
    PADDLE_THROW(common::errors::Unimplemented(
        "Input's type must be paddle::dialect::DistDenseTensorType"));
  }

  VLOG(4) << "Builder construction inputs";
  argument.AddInput(input);

  VLOG(4) << "Builder construction attributes";
  std::vector<pir::Attribute> local_dist_attrs_;
  for (const TensorDistAttribute& local_dist_attr : local_dist_attrs) {
    local_dist_attrs_.emplace_back(local_dist_attr);
  }
  pir::Attribute op_dist_attr =
      OperationDistAttribute::get(pir::IrContext::Instance(),
                                  global_dist_attr.process_mesh_attr(),
                                  std::vector<pir::Attribute>{global_dist_attr},
                                  local_dist_attrs_);
  argument.AddAttribute("op_dist_attr", op_dist_attr);

  VLOG(4) << "Builder construction outputs";
  phi::DDim global_dims = input_tensor_type.global_ddim();
  phi::DDim local_dims = InferLocalDDim(global_dims, global_dist_attr);
  pir::DenseTensorType input_dense_tensor_type =
      input_tensor_type.dense_tensor_type();
  for (auto local_dist_attr : local_dist_attrs) {
    phi::DDim local_tensor_dims(local_dims);  // global shape of local tensor
    const std::vector<int64_t>& dims_mapping = local_dist_attr.dims_mapping();
    ProcessMeshAttribute mesh = local_dist_attr.process_mesh_attr();
    const std::vector<int64_t>& mesh_shape = mesh.shape();
    PADDLE_ENFORCE_EQ(
        static_cast<int>(dims_mapping.size()),
        local_tensor_dims.size(),
        common::errors::PreconditionNotMet(
            "local dims_mapping size %d does not match local size %d",
            dims_mapping.size(),
            local_tensor_dims.size()));

    for (size_t i = 0; i < dims_mapping.size(); ++i) {
      if (dims_mapping[i] != -1) {
        int64_t dim_size = mesh_shape.at(dims_mapping.at(i));
        local_tensor_dims[i] *= dim_size;
      }
    }

    pir::DenseTensorType out_dense_tensor_type =
        paddle::dialect::DenseTensorType::get(
            pir::IrContext::Instance(),
            input_dense_tensor_type.dtype(),
            local_tensor_dims,
            input_dense_tensor_type.data_layout(),
            input_dense_tensor_type.lod(),
            input_dense_tensor_type.offset());

    pir::Type out_dist_tensor_type =
        paddle::dialect::DistDenseTensorType::get(pir::IrContext::Instance(),
                                                  out_dense_tensor_type,
                                                  local_dist_attr,
                                                  local_dims);
    argument.AddOutput(out_dist_tensor_type);
  }

  ::pir::PassStopGradientsDefaultly(argument);
}

OpInfoTuple MoESubMeshTensorsOp::GetOpInfo() {
  return OpInfoTuple({OpInputInfo()},
                     {},
                     {OpOutputInfo()},
                     OpRunTimeInfo(),
                     "moe_sub_mesh_tensors");
}

std::vector<std::vector<pir::Value>> MoESubMeshTensorsOp::Vjp(
    pir::Operation* op,
    const std::vector<std::vector<pir::Value>>& inputs_,
    const std::vector<std::vector<pir::Value>>& outputs,
    const std::vector<std::vector<pir::Value>>& out_grads,
    const std::vector<std::vector<bool>>& stop_gradients) {
  VLOG(6) << "Start call vjp for moe_sub_mesh_tensors op.";
  PADDLE_ENFORCE_EQ(inputs_.size(),
                    1,
                    common::errors::InvalidArgument(
                        "moe_sub_mesh_tensors op's inputs' size should be 1"));
  PADDLE_ENFORCE_EQ(
      inputs_[0].size(),
      1,
      common::errors::InvalidArgument(
          "moe_sub_mesh_tensors op's inputs[0]'s size should be 1"));
  auto dist_type = inputs_[0][0].type().dyn_cast<DistTypeInterface>();

  PADDLE_ENFORCE_NOT_NULL(
      dist_type,
      common::errors::InvalidArgument(
          "moe_sub_mesh_tensors op's inputs type must be dist type."));

  auto& builder = *ApiBuilder::Instance().GetBuilder();

  std::vector<TensorDistAttribute> local_dist_attrs;
  // the grad_op is dtensor_from_local_tensors, whose input
  // type is std::vector<pir::Value>.
  std::vector<pir::Value> input_for_grad_op;
  for (size_t i = 0; i < out_grads.size(); i++) {
    // the input dist_attr of grad op should be equal
    // to the output dist_attr of the forward op
    DistTypeInterface grad_dist_type =
        outputs[i][0].type().dyn_cast<DistTypeInterface>();
    local_dist_attrs.emplace_back(grad_dist_type.tensor_dist_attr());
    input_for_grad_op.emplace_back(out_grads[i][0]);
  }

  DistDenseTensorType global_dist_type =
      inputs_[0][0].type().dyn_cast<DistDenseTensorType>();
  TensorDistAttribute global_dist_attr = global_dist_type.tensor_dist_attr();
  auto grad_op =
      builder.Build<MoEGlobalMeshTensorOp>(input_for_grad_op,
                                           local_dist_attrs,
                                           global_dist_attr,
                                           global_dist_type.global_ddim());

  VLOG(6) << "End call vjp for moe_sub_mesh_tensors op.";

  return {std::vector<pir::Value>{grad_op->result(0)}};
}

void MoESubMeshTensorsOp::VerifySig() {
  VerifyOpArgNum<MoESubMeshTensorsOp>(
      this, 1u, num_results(), 1u, num_results());

  PADDLE_ENFORCE_EQ((*this)
                        ->operand_source(0)
                        .type()
                        .isa<paddle::dialect::DistDenseTensorType>(),
                    true,
                    common::errors::PreconditionNotMet(
                        "Mismatched input type. MoESubMeshTensorsOp requires "
                        "'DistDenseTensorType' for input."));

  auto output_size = num_results();
  for (size_t i = 0; i < output_size; ++i) {
    PADDLE_ENFORCE_EQ(
        (*this)->result(i).type().isa<paddle::dialect::DistDenseTensorType>(),
        true,
        common::errors::PreconditionNotMet(
            "Mismatched type of %u'th output. MoESubMeshTensorsOp requires "
            "'DistDenseTensorType.",
            i));
  }

  VLOG(4) << "End Verifying for: moe_sub_mesh_tensors op.";
}

TEST_API void paddle::dialect::MoEGlobalMeshTensorOp::Build(
    pir::Builder& builder,
    pir::OperationArgument& argument,
    const std::vector<pir::Value>& inputs,
    const std::vector<TensorDistAttribute>& local_dist_attrs,
    const TensorDistAttribute& global_dist_attr,
    const phi::DDim& global_dims) {
  VLOG(4) << "Build moe_global_mesh_tensor op";
  paddle::dialect::DistDenseTensorType input_tensor_type;
  for (pir::Value input : inputs) {
    if (input.type().isa<paddle::dialect::DistDenseTensorType>()) {
      input_tensor_type =
          input.type().dyn_cast<paddle::dialect::DistDenseTensorType>();
    } else {
      PADDLE_THROW(common::errors::Unimplemented(
          "Input's type must be paddle::dialect::DistDenseTensorType"));
    }
  }

  VLOG(4) << "Builder construction inputs";
  argument.AddInputs(inputs);

  VLOG(4) << "Builder construction attributes";
  std::vector<pir::Attribute> local_dist_attrs_;
  for (const TensorDistAttribute& local_dist_attr : local_dist_attrs) {
    local_dist_attrs_.emplace_back(local_dist_attr);
  }
  pir::Attribute op_dist_attr = OperationDistAttribute::get(
      pir::IrContext::Instance(),
      global_dist_attr.process_mesh_attr(),
      local_dist_attrs_,
      std::vector<pir::Attribute>{global_dist_attr});
  argument.AddAttribute("op_dist_attr", op_dist_attr);

  VLOG(4) << "Builder construction outputs";
  pir::DenseTensorType input_dense_tensor_type =
      input_tensor_type.dense_tensor_type();
  pir::DenseTensorType out_dense_tensor_type =
      paddle::dialect::DenseTensorType::get(
          pir::IrContext::Instance(),
          input_dense_tensor_type.dtype(),
          global_dims,
          input_dense_tensor_type.data_layout(),
          input_dense_tensor_type.lod(),
          input_dense_tensor_type.offset());
  pir::Type out_dist_tensor_type = paddle::dialect::DistDenseTensorType::get(
      pir::IrContext::Instance(), out_dense_tensor_type, global_dist_attr);
  argument.AddOutput(out_dist_tensor_type);

  ::pir::PassStopGradientsDefaultly(argument);
}

OpInfoTuple MoEGlobalMeshTensorOp::GetOpInfo() {
  return OpInfoTuple({OpInputInfo()},
                     {},
                     {OpOutputInfo()},
                     OpRunTimeInfo(),
                     "moe_global_mesh_tensor");
}

void MoEGlobalMeshTensorOp::VerifySig() {
  VerifyOpArgNum<MoEGlobalMeshTensorOp>(
      this, num_operands(), 1u, num_operands(), 1u);

  auto input_size = num_operands();
  for (size_t i = 0; i < input_size; ++i) {
    PADDLE_ENFORCE_EQ(
        (*this)
            ->operand_source(i)
            .type()
            .isa<paddle::dialect::DistDenseTensorType>(),
        true,
        common::errors::PreconditionNotMet(
            "Mismatched type of %u'th input. MoEGlobalMeshTensorOp requires "
            "'DistDenseTensorType'.",
            i));
  }

  PADDLE_ENFORCE_EQ(
      (*this)->result(0).type().isa<paddle::dialect::DistDenseTensorType>(),
      true,
      common::errors::PreconditionNotMet(
          "Type validation failed for the 0th input."));

  VLOG(4) << "End Verifying for: moe_global_mesh_tensor op.";
}

std::vector<std::vector<pir::Value>> MoEGlobalMeshTensorOp::Vjp(
    pir::Operation* op,
    const std::vector<std::vector<pir::Value>>& inputs_,
    const std::vector<std::vector<pir::Value>>& outputs,
    const std::vector<std::vector<pir::Value>>& out_grads,
    const std::vector<std::vector<bool>>& stop_gradients) {
  VLOG(6) << "Start call vjp for moe_global_mesh_tensor op.";

  std::vector<TensorDistAttribute> local_dist_attrs;
  for (size_t i = 0; i < inputs_.size(); i++) {
    auto dist_type = inputs_[i][0].type().dyn_cast<DistTypeInterface>();
    PADDLE_ENFORCE_NOT_NULL(
        dist_type,
        common::errors::InvalidArgument(
            "Currently, %s's inputs type must be dist type.", name()));
    local_dist_attrs.push_back(dist_type.tensor_dist_attr());
  }

  PADDLE_ENFORCE_EQ(out_grads.size(),
                    1,
                    common::errors::InvalidArgument(
                        "reshard op's outputs  grad size should be 1"));

  PADDLE_ENFORCE_EQ(out_grads[0].size(),
                    1,
                    common::errors::InvalidArgument(
                        "reshard op's outputs grad[0] size should be 1"));

  TensorDistAttribute global_dist_attr =
      outputs[0][0].type().dyn_cast<DistTypeInterface>().tensor_dist_attr();

  auto& builder = *ApiBuilder::Instance().GetBuilder();

  auto grad_op = builder.Build<MoESubMeshTensorsOp>(
      out_grads[0][0], local_dist_attrs, global_dist_attr);

  VLOG(6) << "End call vjp for " << name() << " op.";

  std::vector<std::vector<pir::Value>> res;
  for (const pir::Value& value : grad_op->results()) {
    res.emplace_back(std::vector<pir::Value>{value});
  }
  return res;
}

void DistReshapeOp::VerifySig() {
  VerifyOpArgNum<DistReshapeOp>(this, 1u, 1u, 1u, 1u);

  PADDLE_ENFORCE_EQ((*this)
                        ->operand_source(0)
                        .type()
                        .isa<paddle::dialect::DistDenseTensorType>(),
                    true,
                    common::errors::PreconditionNotMet(
                        "Type validation failed for the 0th input."));

  PADDLE_ENFORCE_EQ(
      (*this)->result(0).type().isa<paddle::dialect::DistDenseTensorType>(),
      true,
      common::errors::PreconditionNotMet(
          "Type validation failed for the 0th output."));

  VLOG(4) << "End Verifying for: DistReshapeOp.";
}

OpInfoTuple DistReshapeOp::GetOpInfo() {
  return OpInfoTuple(
      {OpInputInfo()}, {}, {OpOutputInfo()}, OpRunTimeInfo(), "dist_reshape");
}

void DistReshapeOp::Build(pir::Builder& builder,
                          pir::OperationArgument& argument,
                          pir::Value input,
                          const PlacementsAttribute& x_placements,
                          const common::DDim& global_shape,
                          const common::DDim& local_shape,
                          const TensorDistAttribute& out_dist_attr) {
  paddle::dialect::DistDenseTensorType input_tensor_type;
  if (input.type().isa<paddle::dialect::DistDenseTensorType>()) {
    input_tensor_type =
        input.type().dyn_cast<paddle::dialect::DistDenseTensorType>();
  } else {
    PADDLE_THROW(common::errors::Unimplemented(
        "Only support paddle::dialect::DistDenseTensorType"));
  }
  argument.AddInput(input);

  pir::Attribute op_dist_attr = OperationDistAttribute::get(
      pir::IrContext::Instance(),
      out_dist_attr.process_mesh_attr(),
      std::vector<pir::Attribute>{input_tensor_type.tensor_dist_attr()},
      std::vector<pir::Attribute>{out_dist_attr});
  argument.AddAttribute("op_dist_attr", op_dist_attr);

  argument.AddAttribute("x_placements", x_placements);
  std::optional<PlacementsAttribute> placements =
      out_dist_attr.placements_attr();
  PADDLE_ENFORCE_EQ(
      placements.has_value(),
      true,
      common::errors::InvalidArgument("PlacementsAttribute is not set."));
  argument.AddAttribute("out_placements", placements.value());

  pir::DenseTensorType input_dense_tensor_type =
      input_tensor_type.dense_tensor_type();
  pir::DenseTensorType out_dense_tensor_type =
      paddle::dialect::DenseTensorType::get(
          pir::IrContext::Instance(),
          input_dense_tensor_type.dtype(),
          global_shape,
          input_dense_tensor_type.data_layout(),
          input_dense_tensor_type.lod(),
          input_dense_tensor_type.offset());
  pir::Type out_dist_tensor_type =
      paddle::dialect::DistDenseTensorType::get(pir::IrContext::Instance(),
                                                out_dense_tensor_type,
                                                out_dist_attr,
                                                local_shape);
  argument.AddOutput(out_dist_tensor_type);

  ::pir::PassStopGradientsDefaultly(argument);
}

std::vector<std::vector<pir::Value>> DistReshapeOp::Vjp(
    pir::Operation* op,
    const std::vector<std::vector<pir::Value>>& inputs_,
    const std::vector<std::vector<pir::Value>>& outputs,
    const std::vector<std::vector<pir::Value>>& out_grads,
    const std::vector<std::vector<bool>>& stop_gradients) {
  VLOG(6) << "Start call vjp for dist_reshape op.";
  PADDLE_ENFORCE_EQ(
      inputs_.size(),
      1,
      common::errors::InvalidArgument("reshard op's inputs' size should be 1"));
  PADDLE_ENFORCE_EQ(inputs_[0].size(),
                    1,
                    common::errors::InvalidArgument(
                        "dist_reshape op's inputs[0]'s size should be 1"));

  PADDLE_ENFORCE_EQ(out_grads.size(),
                    1,
                    common::errors::InvalidArgument(
                        "dist_reshape op's outputs grad size should be 1"));

  PADDLE_ENFORCE_EQ(out_grads[0].size(),
                    1,
                    common::errors::InvalidArgument(
                        "dist_reshape op's outputs grad[0] size should be 1"));

  DistDenseTensorType input_type =
      inputs_[0][0].type().dyn_cast<DistDenseTensorType>();
  DistDenseTensorType out_grad_type =
      out_grads[0][0].type().dyn_cast<DistDenseTensorType>();
  PADDLE_ENFORCE_NOT_NULL(
      input_type,
      common::errors::InvalidArgument(
          "dist_reshape op's inputs type must be dist type."));
  PADDLE_ENFORCE_NOT_NULL(
      out_grad_type,
      common::errors::InvalidArgument(
          "dist_reshape op's outputs grad type must be dist type."));

  PlacementsAttribute x_placements =
      op->attribute<PlacementsAttribute>("x_placements");
  PlacementsAttribute out_placements =
      op->attribute<PlacementsAttribute>("out_placements");

  pir::IrContext* ctx = pir::IrContext::Instance();
  TensorDistAttribute tmp_attr = input_type.tensor_dist_attr();
  TensorDistAttribute input_dist_attr =
      TensorDistAttribute::get(ctx,
                               tmp_attr.process_mesh_attr(),
                               tmp_attr.dims_mapping(),
                               tmp_attr.partial_status(),
                               x_placements);

  auto& builder = *ApiBuilder::Instance().GetBuilder();

  auto grad_op = builder.Build<DistReshapeOp>(out_grads[0][0],
                                              out_placements,
                                              input_type.global_ddim(),
                                              input_type.local_ddim(),
                                              input_dist_attr);

  VLOG(6) << "End call vjp for dist_reshape op.";

  return {std::vector<pir::Value>{grad_op->result(0)}};
}

}  // namespace paddle::dialect

IR_DEFINE_EXPLICIT_TYPE_ID(paddle::dialect::ShardTensorOp)
IR_DEFINE_EXPLICIT_TYPE_ID(paddle::dialect::ReshardOp)
IR_DEFINE_EXPLICIT_TYPE_ID(paddle::dialect::DtensorFromLocalOp)
IR_DEFINE_EXPLICIT_TYPE_ID(paddle::dialect::DtensorToLocalOp)
IR_DEFINE_EXPLICIT_TYPE_ID(paddle::dialect::MoESubMeshTensorsOp)
IR_DEFINE_EXPLICIT_TYPE_ID(paddle::dialect::MoEGlobalMeshTensorOp)
IR_DEFINE_EXPLICIT_TYPE_ID(paddle::dialect::DistReshapeOp)
