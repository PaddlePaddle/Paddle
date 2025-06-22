// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/eager/to_static/run_program_node.h"

#include "paddle/fluid/eager/grad_node_info.h"
#include "paddle/fluid/eager/tensor_wrapper.h"
#include "paddle/fluid/eager/to_static/run_program_impl.h"
#include "paddle/fluid/eager/to_static/run_program_utils.h"
#include "paddle/phi/core/platform/profiler/event_tracing.h"

GradNodeRunProgram::~GradNodeRunProgram() {
  if (!(*executed_)) {
    auto *out_scope_vec = &step_scope_;
    VLOG(4) << "~GradNodeRunProgram";
    // Normally out_scope_vec.size() == 1. for safety, we add for-loop here.
    for (size_t i = 0; i < out_scope_vec->size(); ++i) {
      paddle::framework::Scope *global_inner_scope = out_scope_vec->at(i);
      global_inner_scope->SetCanReused(true);
      egr::to_static::details::GcScope(global_inner_scope);
      VLOG(4) << "global_inner_scope SetCanReused";
    }
  }
}

void GradNodeRunProgram::ConstructXGradTensors(
    const std::vector<paddle::Tensor> &x, std::vector<paddle::Tensor> *x_grad) {
  auto x_grad_names =
      PADDLE_GET_CONST(std::vector<std::string>, attrs_.at("bx_g_names"));
  PADDLE_ENFORCE_EQ(x.size(),
                    x_grad_names.size(),
                    common::errors::InvalidArgument(
                        "The x.size() and x_grad_names.size() should be equal. "
                        "But received x.size() = %d, x_grad_names.size() = %d",
                        x.size(),
                        x_grad_names.size()));

  // TODO(dev): Need an elegant way to determine information of grad_tensor,
  // such as: name, tensor type (DenseTensor, SelectedRows or
  // VariableRefArray).
  for (size_t i = 0; i < x.size(); i++) {
    if (x[i].is_dense_tensor()) {
      x_grad->emplace_back(std::make_shared<phi::DenseTensor>());
    } else if (x[i].is_selected_rows()) {
      x_grad->emplace_back(std::make_shared<phi::SelectedRows>());
    } else if (egr::to_static::IsVariableRefArray(x[i])) {
      x_grad->emplace_back(
          std::make_shared<paddle::framework::VariableRefArray>());
    } else {
      PADDLE_THROW(common::errors::InvalidArgument(
          "The grad tensor type is not supported."));
    }
  }
}

void GradNodeRunProgram::ConstructParamGradTensors(
    const std::vector<paddle::Tensor> &params,
    std::vector<paddle::Tensor> *param_grads) {
  auto p_grad_names =
      PADDLE_GET_CONST(std::vector<std::string>, attrs_.at("bp_g_names"));
  PADDLE_ENFORCE_EQ(params.size(),
                    p_grad_names.size(),
                    common::errors::InvalidArgument(
                        "The param.size() and "
                        "param_grad_names.size() should be equal."));

  for (size_t i = 0; i < params.size(); ++i) {
    auto &p = params[i];
    auto &p_grad = egr::EagerUtils::unsafe_autograd_meta(p)->Grad();
    // In eager mode, the number of param_grad should be the same as
    // param, so here an empty Tensor is added for the param with
    // stop_gradient=True
    if (!p_grad.defined()) {
      param_grads->emplace_back();
    } else if (p_grad.is_dense_tensor()) {
      param_grads->emplace_back(std::make_shared<phi::DenseTensor>());
    } else if (p_grad.is_selected_rows()) {
      param_grads->emplace_back(std::make_shared<phi::SelectedRows>());
    }
  }
}

paddle::small_vector<std::vector<paddle::Tensor>, egr::kSlotSmallVectorSize>
GradNodeRunProgram::operator()(
    paddle::small_vector<std::vector<paddle::Tensor>,
                         egr::kSlotSmallVectorSize> &grads,  // NOLINT
    bool create_graph UNUSED,
    bool is_new_grad UNUSED) {
  VLOG(3) << "Running Eager Backward Node: GradNodeRunProgram";
  paddle::small_vector<std::vector<paddle::Tensor>, egr::kSlotSmallVectorSize>
      hooked_grads = GradNodeRunProgram::ApplyGradientHooks(grads);
  PADDLE_ENFORCE_EQ(hooked_grads.size(),
                    1,
                    common::errors::InvalidArgument(
                        "The hooked_grads.size() of RunProgramGradOp should "
                        "be equal to 1."));

  std::vector<paddle::Tensor> x_grad;
  std::vector<paddle::Tensor> params_grad;
  std::vector<paddle::Tensor *> x_grad_ptr;
  std::vector<paddle::Tensor *> params_grad_ptr;
  {
    phi::RecordEvent record_event(
        "construct_grad_tensor", phi::TracerEventType::UserDefined, 1);

    egr::EagerUtils::FillZeroForEmptyOptionalGradInput(&hooked_grads[0],
                                                       this->InputMeta()[0]);
    VLOG(3) << "hooked_grads[0].size() : " << hooked_grads[0].size();
    ConstructXGradTensors(x_, &x_grad);
    ConstructParamGradTensors(params_, &params_grad);
    for (auto &i : x_grad) {
      x_grad_ptr.emplace_back(&i);
    }
    for (auto &i : params_grad) {
      params_grad_ptr.emplace_back(&i);
    }
  }

  const auto &out_grad_names =
      PADDLE_GET_CONST(std::vector<std::string>, attrs_.at("bo_g_names"));
  PADDLE_ENFORCE_EQ(hooked_grads[0].size(),
                    out_grad_names.size(),
                    common::errors::InvalidArgument(
                        "The hooked_grads[0].size() and "
                        "out_grad_values.size() should be equal."));

  egr::to_static::RunProgramGradImpl(hooked_grads[0],
                                     step_scope_,
                                     attrs_,
                                     x_grad_ptr,
                                     params_grad_ptr,
                                     place_hash_key_);
  VLOG(3) << "End Eager Backward Node: GradNodeRunProgram";

  *executed_ = true;
  egr::EagerUtils::FillZeroForEmptyOptionalGradOutput(&x_grad,
                                                      this->OutputMeta()[0]);
  egr::EagerUtils::FillZeroForEmptyOptionalGradOutput(&params_grad,
                                                      this->OutputMeta()[1]);
  return {x_grad, params_grad};
}

// TODO(cleanup-legacy-ir): Cleanup below code after legacy IR is removed.

GradNodeLegacyRunProgram::~GradNodeLegacyRunProgram() {
  if (!(*executed_)) {
    auto *out_scope_vec = &step_scope_;
    VLOG(4) << "~GradNodeLegacyRunProgram: " << this;
    // Normally out_scope_vec.size() == 1. for safety, we add for-loop here.
    for (size_t i = 0; i < out_scope_vec->size(); ++i) {
      paddle::framework::Scope *global_inner_scope = out_scope_vec->at(i);
      global_inner_scope->SetCanReused(true);
      egr::to_static::details::GcScope(global_inner_scope);
      VLOG(4) << "global_inner_scope SetCanReused";
    }
  }
}

void GradNodeLegacyRunProgram::ConstructXGradTensors(
    const std::vector<paddle::Tensor> &x, std::vector<paddle::Tensor> *x_grad) {
  const auto &x_grad_names =
      PADDLE_GET_CONST(std::vector<std::string>, attrs_.at("x_grad_names"));
  PADDLE_ENFORCE_EQ(x.size(),
                    x_grad_names.size(),
                    common::errors::InvalidArgument(
                        "The x.size() and x_grad_names.size() should be equal. "
                        "But received x.size() = %d, x_grad_names.size() = %d",
                        x.size(),
                        x_grad_names.size()));

  // TODO(dev): Need an elegant way to determine information of grad_tensor,
  // such as: name, tensor type(DenseTensor or SelectedRows).
  for (size_t i = 0; i < x.size(); i++) {
    if (x[i].is_dense_tensor()) {
      x_grad->emplace_back(std::make_shared<phi::DenseTensor>());
    } else if (x[i].is_selected_rows()) {
      x_grad->emplace_back(std::make_shared<phi::SelectedRows>());
    }
    x_grad->back().set_name(x_grad_names[i]);
  }
}

void GradNodeLegacyRunProgram::ConstructParamGradTensors(
    const std::vector<paddle::Tensor> &params,
    std::vector<paddle::Tensor> *param_grads) {
  const auto &param_grad_names =
      PADDLE_GET_CONST(std::vector<std::string>, attrs_.at("param_grad_names"));
  PADDLE_ENFORCE_EQ(params.size(),
                    param_grad_names.size(),
                    common::errors::InvalidArgument(
                        "The param.size() and "
                        "param_grad_names.size() should be equal."));

  for (size_t i = 0; i < params.size(); ++i) {
    auto &p = params[i];
    auto &p_grad = egr::EagerUtils::unsafe_autograd_meta(p)->Grad();
    // In eager mode, the number of param_grad should be the same as
    // param, so here an empty Tensor is added for the param with
    // stop_gradient=True
    if (!p_grad.defined()) {
      param_grads->emplace_back();
    } else if (p_grad.is_dense_tensor()) {
      param_grads->emplace_back(std::make_shared<phi::DenseTensor>());
    } else if (p_grad.is_selected_rows()) {
      param_grads->emplace_back(std::make_shared<phi::SelectedRows>());
    }
    param_grads->back().set_name(param_grad_names[i]);
  }
}

paddle::small_vector<std::vector<paddle::Tensor>, egr::kSlotSmallVectorSize>
GradNodeLegacyRunProgram::operator()(
    paddle::small_vector<std::vector<paddle::Tensor>,
                         egr::kSlotSmallVectorSize> &grads,  // NOLINT
    bool create_graph UNUSED,
    bool is_new_grad UNUSED) {
  VLOG(3) << "Running Eager Backward Node: GradNodeLegacyRunProgram";
  paddle::small_vector<std::vector<paddle::Tensor>, egr::kSlotSmallVectorSize>
      hooked_grads = GradNodeLegacyRunProgram::ApplyGradientHooks(grads);
  PADDLE_ENFORCE_EQ(hooked_grads.size(),
                    1,
                    common::errors::InvalidArgument(
                        "The hooked_grads.size() of RunProgramGradOp should "
                        "be equal to 1."));

  std::vector<paddle::Tensor> x_grad;
  std::vector<paddle::Tensor> params_grad;
  std::vector<paddle::Tensor *> x_grad_ptr;
  std::vector<paddle::Tensor *> params_grad_ptr;
  {
    phi::RecordEvent record_event(
        "construct_grad_tensor", phi::TracerEventType::UserDefined, 1);

    egr::EagerUtils::FillZeroForEmptyOptionalGradInput(&hooked_grads[0],
                                                       this->InputMeta()[0]);
    VLOG(3) << "hooked_grads[0].size() : " << hooked_grads[0].size();
    ConstructXGradTensors(x_, &x_grad);
    ConstructParamGradTensors(params_, &params_grad);
    for (auto &i : x_grad) {
      x_grad_ptr.emplace_back(&i);
    }
    for (auto &i : params_grad) {
      if (i.defined()) {
        params_grad_ptr.emplace_back(&i);
      }
    }
  }

  const auto &out_grad_names =
      PADDLE_GET_CONST(std::vector<std::string>, attrs_.at("out_grad_names"));
  PADDLE_ENFORCE_EQ(hooked_grads[0].size(),
                    out_grad_names.size(),
                    common::errors::InvalidArgument(
                        "The hooked_grads[0].size() and "
                        "out_grad_names.size() should be equal."));
  for (size_t i = 0; i < out_grad_names.size(); ++i) {
    hooked_grads[0][i].set_name(out_grad_names[i]);
  }
  egr::to_static::LegacyRunProgramGradImpl(hooked_grads[0],
                                           step_scope_,
                                           attrs_,
                                           x_grad_ptr,
                                           params_grad_ptr,
                                           place_hash_key_);
  VLOG(3) << "End Eager Backward Node: GradNodeLegacyRunProgram: Ptr " << this;

  *executed_ = true;
  egr::EagerUtils::FillZeroForEmptyOptionalGradOutput(&x_grad,
                                                      this->OutputMeta()[0]);
  egr::EagerUtils::FillZeroForEmptyOptionalGradOutput(&params_grad,
                                                      this->OutputMeta()[1]);
  return {x_grad, params_grad};
}
