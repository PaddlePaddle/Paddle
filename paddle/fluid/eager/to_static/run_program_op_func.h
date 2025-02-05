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

#include <vector>

#include "paddle/fluid/eager/autograd_meta.h"
#include "paddle/fluid/eager/eager_tensor.h"
#include "paddle/fluid/eager/to_static/run_program_op_node.h"
#include "paddle/fluid/eager/utils.h"
#include "paddle/fluid/framework/tensor_ref_array.h"
#include "paddle/phi/core/memory/allocation/allocator.h"
#include "paddle/pir/include/core/block.h"
#include "paddle/pir/include/core/builtin_type.h"
#include "paddle/pir/include/core/value.h"
#include "paddle/pir/include/dialect/control_flow/ir/cf_type.h"

static void clear_no_grad_edges_with_partial_block(
    const std::vector<paddle::Tensor>& params,
    const paddle::framework::BlockDesc* forward_block_desc,
    const paddle::framework::BlockDesc* backward_block_desc,
    egr::GradNodeBase* grad_node,
    size_t slot_id) {
  for (size_t i = 0; i < params.size(); ++i) {
    auto p_grad_name = paddle::framework::GradVarName(params[i].name());
    if (!forward_block_desc->HasVar(p_grad_name) &&
        !backward_block_desc->HasVar(p_grad_name)) {
      VLOG(3) << "clear edge of " << p_grad_name;
      grad_node->MutableOutputMeta()[slot_id][i].GetMutableEdge().Clear();
    }
  }
}

static bool IsFakeValue(const pir::Value& value) {
  return value.impl() == nullptr || !value.type();
}

// Filter params without grads in global block. In this case, we will
// tag its AutogradMeta with stop_gradient = True to avoid fault from
// reducer while training on multi-cards.
static void pir_clear_no_grad_edges(
    const std::vector<paddle::Tensor>& params,
    const std::vector<pir::Value>& backward_params_grad,
    const pir::Block* backward_block,
    egr::GradNodeBase* grad_node,
    size_t slot_id) {
  for (size_t i = 0; i < params.size(); ++i) {
    if (IsFakeValue(backward_params_grad[i])) {
      VLOG(3) << "clear edge of " << params[i].name();
      grad_node->MutableOutputMeta()[slot_id][i].GetMutableEdge().Clear();
    }
  }
}

static void clear_unused_out_var_in_backward(
    const std::vector<paddle::Tensor*>& out,
    const paddle::framework::BlockDesc* backward_block,
    paddle::framework::Scope* scope) {
  std::deque<std::shared_ptr<paddle::memory::Allocation>>* garbages =
      new std::deque<std::shared_ptr<paddle::memory::Allocation>>();
  for (auto* out_tensor : out) {
    if (!backward_block->HasVar(out_tensor->name())) {
      auto var = scope->FindVar(out_tensor->name());
      if (var == nullptr) {
        continue;
      }
      if (var->IsType<phi::DenseTensor>()) {
        garbages->emplace_back(
            var->GetMutable<phi::DenseTensor>()->MoveMemoryHolder());
      }
    }
  }
  delete garbages;
}

static void pir_clear_unused_out_var_in_backward(
    const std::vector<pir::Value>& fo,
    const pir::Block* backward_block,
    paddle::framework::Scope* scope) {
  auto out_names = details::GetNameFromValue(fo);
  std::deque<std::shared_ptr<paddle::memory::Allocation>>* garbages =
      new std::deque<std::shared_ptr<paddle::memory::Allocation>>();
  for (auto out_name : out_names) {
    if (!backward_block->kwargs().count(out_name)) {
      auto var = scope->FindVar(out_name);
      if (var == nullptr) {
        continue;
      }
      if (var->IsType<phi::DenseTensor>()) {
        garbages->emplace_back(
            var->GetMutable<phi::DenseTensor>()->MoveMemoryHolder());
      }
    }
  }
  delete garbages;
}

static std::vector<paddle::Tensor> filter_unused_input_var_in_backward(
    const std::vector<paddle::Tensor>& x,
    const std::vector<std::string>& x_names,
    const paddle::framework::BlockDesc* backward_block) {
  auto filter_x = std::vector<paddle::Tensor>(x);
  for (size_t i = 0; i < x.size(); i++) {
    if (!backward_block->HasVar(x_names[i])) {
      auto fake = paddle::Tensor(std::make_shared<phi::DenseTensor>());
      fake.set_name(paddle::framework::kFakeVarName);
      filter_x[i] = fake;
    }
  }
  return filter_x;
}

static std::vector<paddle::Tensor> pir_filter_unused_input_var_in_backward(
    const std::vector<paddle::Tensor>& x,
    const std::string x_key_name,
    const paddle::framework::AttributeMap& attrs) {
  auto values =
      PADDLE_GET_CONST(std::vector<::pir::Value>, attrs.at(x_key_name));
  auto filter_x = std::vector<paddle::Tensor>(x);
  for (size_t i = 0; i < x.size(); i++) {
    if (values[i].impl() == nullptr) {
      auto fake = paddle::Tensor(std::make_shared<phi::DenseTensor>());
      fake.set_name(paddle::framework::kFakeVarName);
      filter_x[i] = fake;
    }
  }
  return filter_x;
}

static std::vector<paddle::Tensor>
pir_filter_no_need_buffer_input_var_in_backward(
    const std::vector<paddle::Tensor>& x,
    const paddle::framework::AttributeMap& attrs) {
  auto forward_inputs_values =
      PADDLE_GET_CONST(std::vector<::pir::Value>, attrs.at("fx"));
  auto no_need_buffers_values =
      PADDLE_GET_CONST(std::vector<::pir::Value>, attrs.at("no_need_buffers"));
  auto filter_x = std::vector<paddle::Tensor>(x);
  std::deque<std::shared_ptr<paddle::memory::Allocation>>* garbages =
      new std::deque<std::shared_ptr<paddle::memory::Allocation>>();
  for (size_t i = 0; i < x.size(); i++) {
    if (std::find(no_need_buffers_values.begin(),
                  no_need_buffers_values.end(),
                  forward_inputs_values[i]) != no_need_buffers_values.end()) {
      auto& tensor = filter_x[i];
      if (tensor.initialized() && tensor.is_dense_tensor()) {
        auto copied_dense_tensor = std::make_shared<phi::DenseTensor>(
            *std::dynamic_pointer_cast<phi::DenseTensor>(tensor.impl()));
        garbages->emplace_back(copied_dense_tensor->MoveMemoryHolder());
        auto meta_only_tensor = paddle::Tensor(
            copied_dense_tensor, tensor.mutable_autograd_meta(), tensor.name());
        filter_x[i] = meta_only_tensor;
      }
    }
  }
  delete garbages;
  return filter_x;
}

static std::vector<paddle::Tensor> Trans2ContiguousTensors(
    const std::vector<paddle::Tensor>& tensors) {
  std::vector<paddle::Tensor> res;
  for (const auto& t : tensors) {
    if (t.initialized() && t.is_dense_tensor() &&
        !std::dynamic_pointer_cast<phi::DenseTensor>(t.impl())
             ->meta()
             .is_contiguous()) {
      res.emplace_back(
          std::make_shared<phi::DenseTensor>(
              paddle::experimental::Trans2Contiguous(
                  *(std::dynamic_pointer_cast<phi::DenseTensor>(t.impl())))),
          t.mutable_autograd_meta(),
          t.name());
    } else {
      res.emplace_back(t);
    }
  }
  return res;
}

int64_t hash_with_seed(int64_t value, int64_t seed) {
  return seed + 0x9e3779b9 + (value << 6) + (value >> 2);
}

inline void run_program_ad_func(
    const std::vector<paddle::Tensor>& x,
    const std::vector<paddle::Tensor>& params,
    std::vector<paddle::Tensor*>& out,                   // NOLINT
    std::vector<paddle::framework::Scope*>& step_scope,  // NOLINT
    const paddle::framework::AttributeMap& attrs) {
  // Prepare Autograd Meta
  VLOG(2) << "start run run_program ad function.";
  auto deref_out = details::DereferenceTensors(out);
  std::vector<egr::AutogradMeta*> p_autograd_x =
      egr::EagerUtils::nullable_autograd_meta(x);
  std::vector<egr::AutogradMeta*> p_autograd_params =
      egr::EagerUtils::nullable_autograd_meta(params);
  std::vector<egr::AutogradMeta*> p_autograd_outs =
      egr::EagerUtils::nullable_autograd_meta(deref_out);

  bool trace_backward = egr::Controller::Instance().HasGrad();
  bool require_any_grad = egr::EagerUtils::ComputeRequireGrad(
      trace_backward, &p_autograd_x, &p_autograd_params);

  VLOG(2) << "start run run_program with require_any_grad = "
          << require_any_grad;
  auto x_tmp = Trans2ContiguousTensors(x);
  auto params_tmp = Trans2ContiguousTensors(params);
  // Call forward function
  // if require_any_grad is False, don't save any middle vars.
  int64_t place_hash_key = 0;
  for (const paddle::Tensor& tensor : x) {
    int64_t device_type = static_cast<int64_t>(tensor.place().GetType());
    place_hash_key = hash_with_seed(place_hash_key, device_type);
  }
  RunProgramAPI(x_tmp,
                params_tmp,
                out,
                step_scope,
                require_any_grad,
                attrs,
                place_hash_key);
  VLOG(2) << "start run run_program grad";
  auto is_test = false;
  if (attrs.count("is_test")) {
    is_test = PADDLE_GET_CONST(bool, attrs.at("is_test"));
  }
  if (!is_test && require_any_grad) {
    auto x_names =
        PADDLE_GET_CONST(std::vector<std::string>, attrs.at("x_names"));

    // Create GradOpNode (1 means [out_grad], 2 means [x_grad, paramx_grad])
    auto grad_node = std::make_shared<GradNodeRunProgram>(1, 2);

    // Set place hash keys for backward
    grad_node->SetPlaceHashKey(place_hash_key);

    // Set Attributes
    grad_node->SetAttrMap(attrs);

    auto* forward_global_block = PADDLE_GET_CONST(
        paddle::framework::BlockDesc*, attrs.at("forward_global_block"));
    auto* backward_global_block = PADDLE_GET_CONST(
        paddle::framework::BlockDesc*, attrs.at("backward_global_block"));
    // Clear unused x vars
    auto filter_x = filter_unused_input_var_in_backward(
        x_tmp, x_names, backward_global_block);
    // Set TensorWrappers
    grad_node->SetFwdX(filter_x);
    // Clear unused out vars
    clear_unused_out_var_in_backward(out, backward_global_block, step_scope[0]);

    grad_node->SetFwdParams(params_tmp);
    grad_node->SetStepScope(step_scope);

    grad_node->SetGradOutMeta(x, /*slot id*/ 0);
    grad_node->SetGradOutMeta(params, /*slot id*/ 1);

    VLOG(2) << "clear_no_grad_edges.";
    clear_no_grad_edges_with_partial_block(params,
                                           forward_global_block,
                                           backward_global_block,
                                           grad_node.get(),
                                           /*slot id*/ 1);

    grad_node->SetGradInMeta(deref_out, 0);

    egr::EagerUtils::SetOutRankWithSlot(&p_autograd_outs, 0);

    // Set History for output set current Grad Node for
    egr::EagerUtils::SetHistory(&p_autograd_outs, grad_node);
  }
}

inline void pir_run_program_ad_func(
    const std::vector<paddle::Tensor>& x,
    const std::vector<paddle::Tensor>& params,
    std::vector<paddle::Tensor*>& out,                   // NOLINT
    std::vector<paddle::framework::Scope*>& step_scope,  // NOLINT
    const paddle::framework::AttributeMap& attrs) {
  // Prepare Autograd Meta
  VLOG(2) << "start run pir run_program ad function.";
  auto deref_out = details::DereferenceTensors(out);
  std::vector<egr::AutogradMeta*> p_autograd_x =
      egr::EagerUtils::nullable_autograd_meta(x);
  std::vector<egr::AutogradMeta*> p_autograd_params =
      egr::EagerUtils::nullable_autograd_meta(params);
  std::vector<egr::AutogradMeta*> p_autograd_outs =
      egr::EagerUtils::nullable_autograd_meta(deref_out);

  bool trace_backward = egr::Controller::Instance().HasGrad();
  bool require_any_grad = egr::EagerUtils::ComputeRequireGrad(
      trace_backward, &p_autograd_x, &p_autograd_params);

  // Create Middle Output for GradNode.
  auto middle_values =
      PADDLE_GET_CONST(std::vector<::pir::Value>, attrs.at("fm"));

  auto is_test = false;
  if (attrs.count("is_test")) {
    is_test = PADDLE_GET_CONST(bool, attrs.at("is_test"));
  }
  VLOG(2) << "start run run_program with require_any_grad = "
          << require_any_grad << ", is_test = " << is_test;
  auto x_tmp = Trans2ContiguousTensors(x);
  auto params_tmp = Trans2ContiguousTensors(params);
  // Call forward function
  // if require_any_grad is False, don't save any middle vars.
  int64_t place_hash_key = 0x9e3779b9;
  for (const paddle::Tensor& tensor : x) {
    int64_t device_type = static_cast<int64_t>(tensor.place().GetType());
    place_hash_key = hash_with_seed(place_hash_key, device_type);
  }
  PirRunProgramAPI(x_tmp,
                   params_tmp,
                   out,
                   step_scope,
                   require_any_grad,
                   attrs,
                   place_hash_key);
  if (!is_test && require_any_grad) {
    // Create GradOpNode (1 means [out_grad], 2 means [x_grad, paramx_grad])
    auto grad_node = std::make_shared<PirGradNodeRunProgram>(1, 2);

    // Set place hash keys for backward
    grad_node->SetPlaceHashKey(place_hash_key);

    // Set Attributes
    grad_node->SetAttrMap(attrs);

    // Clear unused x vars
    // NOTE(SigureMo): There are 2 kinds Tensor need to be filtered:
    // 1. The input Tensor unused in backward block.
    // 2. The input Tensor use meta only in backward block.
    // We need to filter both of them.
    // For the first kind, we can create a empty Tensor to replace it.
    // For the second kind, we need to keep the meta only Tensor.
    auto filter_x = pir_filter_no_need_buffer_input_var_in_backward(
        pir_filter_unused_input_var_in_backward(x_tmp, "bx", attrs), attrs);
    // Set TensorWrappers
    grad_node->SetFwdX(filter_x);

    std::shared_ptr<::pir::Program> backward_program = PADDLE_GET_CONST(
        std::shared_ptr<::pir::Program>, attrs.at("backward_program"));
    auto forward_outputs =
        PADDLE_GET_CONST(std::vector<::pir::Value>, attrs.at("fo"));
    auto backward_params_grad =
        PADDLE_GET_CONST(std::vector<::pir::Value>, attrs.at("bp_g"));

    pir_clear_unused_out_var_in_backward(
        forward_outputs, backward_program->block(), step_scope[0]);

    grad_node->SetFwdParams(params_tmp);

    grad_node->SetStepScope(step_scope);  // just for set useable.

    grad_node->SetGradOutMeta(x, /*slot id*/ 0);
    grad_node->SetGradOutMeta(params, /*slot id*/ 1);

    // Clear no grad edges
    VLOG(2) << "clear no grad edges.";
    pir_clear_no_grad_edges(params,
                            backward_params_grad,
                            backward_program->block(),
                            grad_node.get(),
                            /*slot id*/ 1);

    grad_node->SetGradInMeta(deref_out, 0);

    egr::EagerUtils::SetOutRankWithSlot(&p_autograd_outs, 0);

    // Set History for output set current Grad Node for
    egr::EagerUtils::SetHistory(&p_autograd_outs, grad_node);
  }
}
