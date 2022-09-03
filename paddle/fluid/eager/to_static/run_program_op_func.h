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

// Filter params without grads in global block. In this case, we will
// tag its AutogradMeta with stop_gradient = True to avoid fault from
// reducer while training on multi-cards.
static void clear_no_grad_edges(
    const std::vector<paddle::experimental::Tensor>& params,
    const paddle::framework::BlockDesc* block_desc,
    egr::GradNodeBase* grad_node,
    size_t slot_id) {
  for (size_t i = 0; i < params.size(); ++i) {
    auto p_grad_name = paddle::framework::GradVarName(params[i].name());
    if (!block_desc->HasVar(p_grad_name)) {
      VLOG(1) << "clear edge of " << p_grad_name;
      grad_node->MutableOutputMeta()[slot_id][i].GetMutableEdge().Clear();
    }
  }
}

static void clear_no_grad_edges_with_partial_block(
    const std::vector<paddle::experimental::Tensor>& params,
    const paddle::framework::BlockDesc* forward_block_desc,
    const paddle::framework::BlockDesc* backward_block_desc,
    egr::GradNodeBase* grad_node,
    size_t slot_id) {
  for (size_t i = 0; i < params.size(); ++i) {
    auto p_grad_name = paddle::framework::GradVarName(params[i].name());
    if (!forward_block_desc->HasVar(p_grad_name) &&
        !backward_block_desc->HasVar(p_grad_name)) {
      VLOG(1) << "clear edge of " << p_grad_name;
      grad_node->MutableOutputMeta()[slot_id][i].GetMutableEdge().Clear();
    }
  }
}

inline void run_program_dygraph_function(
    const std::vector<paddle::experimental::Tensor>& x,
    const std::vector<paddle::experimental::Tensor>& params,
    std::vector<paddle::experimental::Tensor*>& out,     // NOLINT
    std::vector<paddle::framework::Scope*>& step_scope,  // NOLINT
    std::vector<paddle::experimental::Tensor*>& dout,    // NOLINT
    const paddle::framework::AttributeMap& attrs) {
  VLOG(2) << "start run run_program";
  // Call forward function
  RunProgramAPI(x, params, out, step_scope, dout, attrs);
  VLOG(2) << "start run run_program grad";

  // Prepare Autograd Meta
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

  if (require_any_grad) {
    std::vector<std::string> out_names;
    for (auto& t : deref_out) {
      out_names.emplace_back(t.name());
    }

    egr::EagerUtils::PassStopGradient(false, &p_autograd_outs);
    // Create GradOpNode (1 means [out_grad], 2 means [x_grad, paramx_grad])
    auto grad_node = std::make_shared<GradNodeRunProgram>(1, 2);

    grad_node->SetFwdOutNames(out_names);
    // Set Attributes
    grad_node->SetAttrMap(attrs);
    // Set TensorWrappers
    grad_node->SetFwdX(x);

    grad_node->SetFwdParams(params);
    grad_node->SetStepScope(step_scope);

    // Set Grad out rank as same as fwd input and set stop gradient to bwd
    grad_node->SetGradOutMeta(x, /*slot id*/ 0);
    grad_node->SetGradOutMeta(params, /*slot id*/ 1);

    bool use_interpretorcore =
        PADDLE_GET_CONST(bool, attrs.at("use_interpretorcore"));
    VLOG(2) << "clear_no_grad_edges.";
    if (use_interpretorcore) {
      auto* forward_global_block = PADDLE_GET_CONST(
          paddle::framework::BlockDesc*, attrs.at("forward_global_block"));
      auto* backward_global_block = PADDLE_GET_CONST(
          paddle::framework::BlockDesc*, attrs.at("backward_global_block"));
      clear_no_grad_edges_with_partial_block(params,
                                             forward_global_block,
                                             backward_global_block,
                                             grad_node.get(),
                                             /*slot id*/ 1);

    } else {
      auto* global_block = PADDLE_GET_CONST(paddle::framework::BlockDesc*,
                                            attrs.at("global_block"));
      clear_no_grad_edges(params, global_block, grad_node.get(), /*slot id*/ 1);
    }

    grad_node->SetGradInMeta(deref_out, 0);

    egr::EagerUtils::SetOutRankWithSlot(&p_autograd_outs, 0);

    // Set History for output set current Grad Node for
    egr::EagerUtils::SetHistory(&p_autograd_outs, grad_node);
    egr::EagerUtils::CheckAndRetainGrad(deref_out);
  }
}
