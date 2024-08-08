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
const char* LocalTensorsFromDistOp::attributes_name[1] = {
    "op_dist_attr"};  // NOLINT
const char* DistTensorFromLocalsOp::attributes_name[1] = {
    "op_dist_attr"};  // NOLINT

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
                      common::errors::PreconditionNotMet(
                          "The op_dist_attr input size must be equal to 0."));

    PADDLE_ENFORCE_EQ(op_dist_attr.num_results(),
                      num_results(),
                      common::errors::PreconditionNotMet(
                          "The op_dist_attr output size %d must "
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
                      common::errors::PreconditionNotMet(
                          "The op_dist_attr output size of reshard op must be "
                          "equal to op output size."));
  }
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

TEST_API void paddle::dialect::LocalTensorsFromDistOp::Build(
    pir::Builder& builder,
    pir::OperationArgument& argument,
    pir::Value input,
    const std::vector<TensorDistAttribute>& local_dist_attrs,
    const TensorDistAttribute& global_dist_attr) {
  VLOG(4) << "Build LocalTensorsFromDistOp";
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

OpInfoTuple LocalTensorsFromDistOp::GetOpInfo() {
  return OpInfoTuple({OpInputInfo()},
                     {},
                     {OpOutputInfo()},
                     OpRunTimeInfo(),
                     "local_tensors_from_dtensor");
}

std::vector<std::vector<pir::Value>> LocalTensorsFromDistOp::Vjp(
    pir::Operation* op,
    const std::vector<std::vector<pir::Value>>& inputs_,
    const std::vector<std::vector<pir::Value>>& outputs,
    const std::vector<std::vector<pir::Value>>& out_grads,
    const std::vector<std::vector<bool>>& stop_gradients) {
  VLOG(6) << "Start call vjp for local_tensors_from_dtensor op.";
  PADDLE_ENFORCE_EQ(
      inputs_.size(),
      1,
      common::errors::InvalidArgument(
          "local_tensors_from_dtensor op's inputs' size should be 1"));
  PADDLE_ENFORCE_EQ(
      inputs_[0].size(),
      1,
      common::errors::InvalidArgument(
          "local_tensors_from_dtensor op's inputs[0]'s size should be 1"));
  auto dist_type = inputs_[0][0].type().dyn_cast<DistTypeInterface>();

  PADDLE_ENFORCE_NOT_NULL(
      dist_type,
      common::errors::InvalidArgument(
          "local_tensors_from_dtensor op's inputs type must be dist type."));

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
      builder.Build<DistTensorFromLocalsOp>(input_for_grad_op,
                                            local_dist_attrs,
                                            global_dist_attr,
                                            global_dist_type.global_ddim());

  VLOG(6) << "End call vjp for local_tensors_from_dtensor op.";

  return {std::vector<pir::Value>{grad_op->result(0)}};
}

void LocalTensorsFromDistOp::VerifySig() {
  VLOG(4) << "Start Verifying inputs, outputs and attributes for: "
             "local_tensors_from_dtensor op.";
  VLOG(4) << "Verifying inputs:";
  {
    auto input_size = num_operands();
    PADDLE_ENFORCE_EQ(
        input_size,
        1u,
        common::errors::PreconditionNotMet(
            "The size of inputs must be equal to 1, but got %d.", input_size));
    PADDLE_ENFORCE_EQ((*this)
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
    for (size_t i = 0; i < output_size; ++i) {
      PADDLE_ENFORCE_EQ(
          (*this)->result(i).type().isa<paddle::dialect::DistDenseTensorType>(),
          true,
          common::errors::PreconditionNotMet(
              "Type validation failed for the %d input.", i));
    }
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

    PADDLE_ENFORCE_EQ(
        op_dist_attr.num_results(),
        num_results(),
        common::errors::PreconditionNotMet("The op_dist_attr output size of "
                                           "local_tensors_from_dist op must be "
                                           "equal to op output size."));
  }
  VLOG(4) << "End Verifying for: local_tensors_from_dtensor op.";
}

TEST_API void paddle::dialect::DistTensorFromLocalsOp::Build(
    pir::Builder& builder,
    pir::OperationArgument& argument,
    const std::vector<pir::Value>& inputs,
    const std::vector<TensorDistAttribute>& local_dist_attrs,
    const TensorDistAttribute& global_dist_attr,
    const phi::DDim& global_dims) {
  VLOG(4) << "Build dtensor_from_local_tensors op";
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
  pir::Type out_dist_tensor_type =
      paddle::dialect::DistDenseTensorType::get(pir::IrContext::Instance(),
                                                out_dense_tensor_type,
                                                global_dist_attr,
                                                global_dims);
  argument.AddOutput(out_dist_tensor_type);

  ::pir::PassStopGradientsDefaultly(argument);
}

OpInfoTuple DistTensorFromLocalsOp::GetOpInfo() {
  return OpInfoTuple({OpInputInfo()},
                     {},
                     {OpOutputInfo()},
                     OpRunTimeInfo(),
                     "dtensor_from_local_tensors");
}

void DistTensorFromLocalsOp::VerifySig() {
  VLOG(4) << "Start Verifying inputs, outputs and attributes for: "
             "dtensor_from_local_tensors op.";
  VLOG(4) << "Verifying inputs:";
  {
    auto input_size = num_operands();
    for (size_t i = 0; i < input_size; ++i) {
      PADDLE_ENFORCE_EQ((*this)
                            ->operand_source(i)
                            .type()
                            .isa<paddle::dialect::DistDenseTensorType>(),
                        true,
                        common::errors::PreconditionNotMet(
                            "Type validation failed for the %d input.", i));
    }
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
    PADDLE_ENFORCE_EQ(output_size,
                      1u,
                      common::errors::PreconditionNotMet(
                          "The size of outputs must be equal to 1, but got %d.",
                          output_size));
    PADDLE_ENFORCE_EQ(
        (*this)->result(0).type().isa<paddle::dialect::DistDenseTensorType>(),
        true,
        common::errors::PreconditionNotMet(
            "Type validation failed for the 0th input."));
  }

  VLOG(4) << "Verifying op dist attrs:";
  {
    auto op_dist_attr =
        this->attribute<paddle::dialect::OperationDistAttribute>(
            "op_dist_attr");
    PADDLE_ENFORCE_EQ(
        op_dist_attr.num_operands(),
        num_operands(),
        common::errors::PreconditionNotMet(
            "The op_dist_attr input size of dtensor_from_local_tensors "
            "op must be equal to op input size."));

    PADDLE_ENFORCE_EQ(
        op_dist_attr.num_results(),
        num_results(),
        common::errors::PreconditionNotMet(
            "The op_dist_attr output size of dtensor_from_local_tensors op "
            "must be equal to op output size."));
  }
  VLOG(4) << "End Verifying for: dtensor_from_local_tensors op.";
}

std::vector<std::vector<pir::Value>> DistTensorFromLocalsOp::Vjp(
    pir::Operation* op,
    const std::vector<std::vector<pir::Value>>& inputs_,
    const std::vector<std::vector<pir::Value>>& outputs,
    const std::vector<std::vector<pir::Value>>& out_grads,
    const std::vector<std::vector<bool>>& stop_gradients) {
  VLOG(6) << "Start call vjp for dist_tensor_from_local_tensors op.";

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

  auto grad_op = builder.Build<LocalTensorsFromDistOp>(
      out_grads[0][0], local_dist_attrs, global_dist_attr);

  VLOG(6) << "End call vjp for " << name() << " op.";

  std::vector<std::vector<pir::Value>> res;
  for (const pir::Value& value : grad_op->results()) {
    res.emplace_back(std::vector<pir::Value>{value});
  }
  return res;
}

}  // namespace paddle::dialect

IR_DEFINE_EXPLICIT_TYPE_ID(paddle::dialect::ShardTensorOp)
IR_DEFINE_EXPLICIT_TYPE_ID(paddle::dialect::ReshardOp)
IR_DEFINE_EXPLICIT_TYPE_ID(paddle::dialect::LocalTensorsFromDistOp)
IR_DEFINE_EXPLICIT_TYPE_ID(paddle::dialect::DistTensorFromLocalsOp)
