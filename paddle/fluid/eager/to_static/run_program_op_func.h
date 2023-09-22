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
#include "paddle/fluid/memory/allocation/allocator.h"
#include "paddle/pir/core/block.h"
#include "paddle/pir/core/value.h"

// Filter params without grads in global block. In this case, we will
// tag its AutogradMeta with stop_gradient = True to avoid fault from
// reducer while training on multi-cards.
static void clear_no_grad_edges(const std::vector<paddle::Tensor>& params,
                                const paddle::framework::BlockDesc* block_desc,
                                egr::GradNodeBase* grad_node,
                                size_t slot_id) {
  for (size_t i = 0; i < params.size(); ++i) {
    auto p_grad_name = paddle::framework::GradVarName(params[i].name());
    if (!block_desc->HasVar(p_grad_name)) {
      VLOG(3) << "clear edge of " << p_grad_name;
      grad_node->MutableOutputMeta()[slot_id][i].GetMutableEdge().Clear();
    }
  }
}

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

static std::vector<paddle::Tensor> newir_filter_unused_input_var_in_backward(
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

static std::vector<paddle::Tensor> Trans2ContiguousTensors(
    const std::vector<paddle::Tensor>& tensors) {
  std::vector<paddle::Tensor> res;
  for (const auto& t : tensors) {
    if (t.is_initialized() && t.is_dense_tensor() &&
        !std::dynamic_pointer_cast<phi::DenseTensor>(t.impl())
             ->meta()
             .is_contiguous()) {
      res.emplace_back(
          std::make_shared<phi::DenseTensor>(
              std::move(paddle::experimental::Trans2Contiguous(
                  *(std::dynamic_pointer_cast<phi::DenseTensor>(t.impl()))))),
          t.mutable_autograd_meta());
    } else {
      res.emplace_back(t);
    }
  }
  return res;
}

inline void run_program_ad_func(
    const std::vector<paddle::Tensor>& x,
    const std::vector<paddle::Tensor>& params,
    std::vector<paddle::Tensor*>& out,                   // NOLINT
    std::vector<paddle::framework::Scope*>& step_scope,  // NOLINT
    std::vector<paddle::Tensor*>& dout,                  // NOLINT
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
  RunProgramAPI(
      x_tmp, params_tmp, out, step_scope, dout, require_any_grad, attrs);
  VLOG(2) << "start run run_program grad";
  auto is_test = false;
  if (attrs.count("is_test")) {
    is_test = PADDLE_GET_CONST(bool, attrs.at("is_test"));
  }
  if (!is_test && require_any_grad) {
    auto x_names =
        PADDLE_GET_CONST(std::vector<std::string>, attrs.at("x_names"));

    egr::EagerUtils::PassStopGradient(false, &p_autograd_outs);
    // Create GradOpNode (1 means [out_grad], 2 means [x_grad, paramx_grad])
    auto grad_node = std::make_shared<GradNodeRunProgram>(1, 2);

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

    // Set Grad out rank as same as fwd input and set stop gradient to bwd
    // NOTE(@xiongkun): Not every tensor in x(list of tensor) is required
    // gradient. for example: x[1] is not used for output, the x[1] is ignored.

    std::vector<const paddle::Tensor*> x_require_grad;
    for (size_t i = 0; i < x.size(); ++i) {
      auto& name = x_names[i];
      if (forward_global_block->HasVar(name) ||
          backward_global_block->HasVar(name)) {
        x_require_grad.push_back(&x[i]);
      }
    }

    grad_node->SetGradOutMeta(x_require_grad, /*slot id*/ 0);
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

inline void newir_run_program_ad_func(
    const std::vector<paddle::Tensor>& x,
    const std::vector<paddle::Tensor>& params,
    std::vector<paddle::Tensor*>& out,                   // NOLINT
    std::vector<paddle::framework::Scope*>& step_scope,  // NOLINT
    std::vector<paddle::Tensor*>& dout,                  // NOLINT
    const paddle::framework::AttributeMap& attrs) {
  // Prepare Autograd Meta
  VLOG(2) << "start run newir run_program ad function.";
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
  auto middle_size =
      PADDLE_GET_CONST(std::vector<::pir::Value>, attrs.at("fm")).size();
  auto output_size =
      PADDLE_GET_CONST(std::vector<::pir::Value>, attrs.at("fo")).size();
  auto middles = std::vector<paddle::Tensor*>();
  std::shared_ptr<NewIRGradNodeRunProgram> grad_node;
  VLOG(2) << "start run run_program with require_any_grad = "
          << require_any_grad;

  if (require_any_grad) {
    // Create GradOpNode (1 means [out_grad], 2 means [x_grad, paramx_grad])
    grad_node = std::make_shared<NewIRGradNodeRunProgram>(1, 2);
    grad_node->GetMiddle().resize(middle_size);
    grad_node->GetOutputs().resize(output_size);
    for (size_t i = 0; i < middle_size; ++i) {
      grad_node->GetMiddle()[i] =
          paddle::Tensor(std::make_shared<phi::DenseTensor>());
      middles.push_back(&grad_node->GetMiddle()[i]);
    }

    auto backward_outs =
        PADDLE_GET_CONST(std::vector<::pir::Value>, attrs.at("bo"));
    for (size_t i = 0; i < output_size; ++i) {
      if (backward_outs[i] != nullptr) {
        grad_node->GetOutputs()[i] = *out[i];
      } else {  // not used by backward program
        auto fake = paddle::Tensor(std::make_shared<phi::DenseTensor>());
        fake.set_name(paddle::framework::kFakeVarName);
        grad_node->GetOutputs()[i] = fake;
      }
    }
  }

  // Call forward function
  // if require_any_grad is False, don't save any middle vars.
  NewIRRunProgramAPI(
      x, params, out, middles, step_scope, dout, require_any_grad, attrs);
  if (require_any_grad) {
    egr::EagerUtils::PassStopGradient(false, &p_autograd_outs);

    // Set Attributes
    grad_node->SetAttrMap(attrs);

    // Clear unused x vars
    auto filter_x = newir_filter_unused_input_var_in_backward(x, "bx", attrs);
    // Set TensorWrappers
    grad_node->SetFwdX(filter_x);

    auto filter_params =
        newir_filter_unused_input_var_in_backward(params, "bp", attrs);
    grad_node->SetFwdParams(filter_params);

    grad_node->SetStepScope(step_scope);  // just for set useable.

    // Set Grad out rank as same as fwd input and set stop gradient to bwd
    // NOTE(@xiongkun): Not every tensor in x(list of tensor) is required
    // gradient. for example: x[1] is not used for output, the x[1] is ignored.

    std::vector<const paddle::Tensor*> x_require_grad;
    for (size_t i = 0; i < x.size(); ++i) {
      x_require_grad.push_back(&x[i]);
    }

    grad_node->SetGradOutMeta(x_require_grad, /*slot id*/ 0);
    grad_node->SetGradOutMeta(params, /*slot id*/ 1);

    // TODO(@xiongkun): rewrite by new ir representation.
    // VLOG(2) << "clear_no_grad_edges.";
    // clear_no_grad_edges_with_partial_block(params,
    // forward_global_block,
    // backward_global_block,
    // grad_node.get(),
    // [>slot id<] 1);

    grad_node->SetGradInMeta(deref_out, 0);

    egr::EagerUtils::SetOutRankWithSlot(&p_autograd_outs, 0);

    // Set History for output set current Grad Node for
    egr::EagerUtils::SetHistory(&p_autograd_outs, grad_node);
  }
}
