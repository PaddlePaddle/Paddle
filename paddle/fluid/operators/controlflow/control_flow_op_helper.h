// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#pragma once

#include "paddle/fluid/framework/new_executor/standalone_executor.h"
#include "paddle/fluid/operators/assign_op.h"
#include "paddle/phi/kernels/funcs/math_function.h"

namespace paddle {
namespace operators {

static void BuildScopeForControlFlowOp(
    const framework::InterpreterCore &interpreter_core,
    const framework::BlockDesc &block,
    framework::Scope *scope) {
  for (auto &var_desc : block.AllVars()) {
    auto var_name = var_desc->Name();
    if (var_name == framework::kEmptyVarName) {
      continue;
    }
    VLOG(5) << "[BuildScopeForControlFlowOp]"
            << "start:" << var_name;
    if (var_desc->Persistable()) {
      VLOG(5) << "[BuildScopeForControlFlowOp]"
              << "Don't process persistent: " << var_name;
    } else {
      auto *ptr = scope->Var(var_name);
      InitializeVariable(ptr, var_desc->GetType());
      VLOG(5) << "[BuildScopeForControlFlowOp]"
              << "Not Found locally and created: " << var_name;
    }
  }

  auto &data_transfer_added_vars =
      interpreter_core.GetVariableScope()->DataTransferAddedVars();
  for (size_t i = 0; i < data_transfer_added_vars.size(); i++) {
    auto *ptr = scope->Var(data_transfer_added_vars[i].first);
    InitializeVariable(ptr,
                       static_cast<paddle::framework::proto::VarType::Type>(
                           data_transfer_added_vars[i].second));
    VLOG(5) << "[BuildScopeForControlFlowOp]"
            << "Initialize Transfer Added Variable "
            << data_transfer_added_vars[i].first;
  }
}

static void AssignZeroToOutsideTensor(const phi::Place &place,
                                      const framework::Scope &cur_scope,
                                      const phi::DenseTensor &input_tensor,
                                      phi::DenseTensor *outside_tensor) {
  if (!input_tensor.IsInitialized() || input_tensor.numel() == 0) {
    return;
  }
  VLOG(4) << "Assigning zero to " << outside_tensor;
  outside_tensor->Resize(input_tensor.dims());
  outside_tensor->mutable_data(place, input_tensor.dtype());
  const phi::DeviceContext *dev_ctx =
      phi::DeviceContextPool::Instance().Get(place);
  phi::funcs::set_constant(*dev_ctx, outside_tensor, 0.0f);
  outside_tensor->set_lod(input_tensor.lod());
}

static void AssignZeroToParentScope(
    const phi::Place &place,
    const framework::Scope &scope,
    const std::vector<std::string> &inputs,
    const std::vector<std::string> &outside_grads) {
  for (size_t i = 0; i < outside_grads.size(); ++i) {
    const std::string &outside_grad_name = outside_grads[i];
    const std::string &input_name = inputs[i];
    VLOG(4) << "[assign zero]"
            << "input_name = " << input_name
            << ", outside_grad_name = " << outside_grad_name;
    framework::Variable *input_var = scope.FindVar(input_name);
    if (input_var == nullptr) {
      continue;
    }
    framework::Variable *outside_var = scope.FindVar(outside_grad_name);
    if (outside_var == nullptr) {
      continue;
    }

    if (input_var->IsType<phi::DenseTensor>()) {
      PADDLE_ENFORCE_EQ(
          outside_var->IsType<phi::DenseTensor>(),
          true,
          common::errors::InvalidArgument(
              "Type of outside_var %s is NOT phi::DenseTensor, which "
              "doesn't match input_var %s.",
              outside_grad_name,
              input_name));
      AssignZeroToOutsideTensor(place,
                                scope,
                                input_var->Get<phi::DenseTensor>(),
                                outside_var->GetMutable<phi::DenseTensor>());
    } else if (input_var->IsType<phi::TensorArray>()) {
      PADDLE_ENFORCE_EQ(outside_var->IsType<phi::TensorArray>(),
                        true,
                        common::errors::InvalidArgument(
                            "Type of outside_var %s is NOT phi::TensorArray, "
                            "which doesn't match input_var %s.",
                            outside_grad_name,
                            input_name));
      const auto &input_tensors = input_var->Get<phi::TensorArray>();
      auto *outside_tensors = outside_var->GetMutable<phi::TensorArray>();
      if (outside_tensors->empty()) {
        outside_tensors->resize(input_tensors.size());
      }
      PADDLE_ENFORCE_EQ(input_tensors.size(),
                        outside_tensors->size(),
                        common::errors::InvalidArgument(
                            "DenseTensorArray outside_var %s doesn't have same "
                            "size as input_var %s.",
                            outside_grad_name,
                            input_name));
      for (size_t j = 0; j < input_tensors.size(); ++j) {
        AssignZeroToOutsideTensor(
            place, scope, input_tensors[j], &((*outside_tensors)[j]));
      }
    } else {
      // TODO(huihuangzheng): add support for SelectedRows
      PADDLE_THROW(common::errors::InvalidArgument(
          "Conditional block grad op doesn't support non-phi::DenseTensor "
          "output "
          "now."));
    }
  }
}

static void AssignLocalGradientToParentScope(
    const phi::Place &place,
    const framework::Scope &cur_scope,
    const framework::Scope &parent_scope,
    const std::vector<std::string> &inside_grads,
    const std::vector<std::string> &outside_grads,
    const std::vector<std::string> &inputs) {
  std::vector<std::string> assign_zero_outside_grads;
  std::vector<std::string> assign_zero_inputs;
  for (size_t i = 0; i < outside_grads.size(); ++i) {
    const std::string &outside_grad_name = outside_grads[i];
    const std::string &inside_grad_name = inside_grads[i];
    VLOG(4) << "[assign local]"
            << "inside_grad_name = " << inside_grad_name
            << ", outside_grad_name = " << outside_grad_name;
    framework::Variable *outside_var = parent_scope.FindVar(outside_grad_name);
    if (outside_var == nullptr) {
      continue;
    }
    framework::Variable *inside_var = cur_scope.FindLocalVar(inside_grad_name);
    if (inside_var == nullptr) {
      assign_zero_outside_grads.emplace_back(outside_grad_name);
      assign_zero_inputs.emplace_back(inputs[i]);
      continue;
    }
    phi::DeviceContext *dev_ctx = phi::DeviceContextPool::Instance().Get(place);
    framework::VisitVarType(*inside_var, AssignFunctor(outside_var, *dev_ctx));
  }
  // Assign zero to the grad_vars that are in outside_grads but not in
  // inside_grads
  AssignZeroToParentScope(
      place, parent_scope, assign_zero_inputs, assign_zero_outside_grads);
}
}  // namespace operators
}  // namespace paddle
