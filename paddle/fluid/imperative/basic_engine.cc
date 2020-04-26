// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/imperative/basic_engine.h"

#include <algorithm>
#include <memory>
#include <queue>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#include "paddle/fluid/imperative/gradient_accumulator.h"
#include "paddle/fluid/imperative/layer.h"
#include "paddle/fluid/imperative/op_base.h"
#include "paddle/fluid/imperative/tracer.h"
#include "paddle/fluid/operators/math/math_function.h"
#include "paddle/fluid/platform/profiler.h"

namespace paddle {
namespace imperative {

void BasicEngine::Init(VarBase* var, const detail::BackwardStrategy& strategy) {
  backward_strategy_ = strategy;
  init_node_ = var->GradVarBase()->GradNode();
  var->GradVarBase()->ClearGradNode();

  if (init_node_ == nullptr || var->OverridedStopGradient()) {
    VLOG(3) << "Skip auto grad since there is no grad op for var or loss is "
               "stop_gradient=True: "
            << var->Name();
    return;
  }

  VLOG(3) << "start backward";

  PADDLE_ENFORCE_EQ(
      var->HasGradVar(), true,
      platform::errors::NotFound("Grad variable not exist for variable %s",
                                 var->Name()));

  auto& fwd_var = var->Var().Get<framework::LoDTensor>();
  auto* grad_var =
      var->GradVarBase()->MutableVar()->GetMutable<framework::LoDTensor>();
  VLOG(6) << "init loss grad:" << var->GradVarBase()->Name()
          << " as stop_gradient false";
  var->GradVarBase()->InnerSetOverridedStopGradient(false);
  auto* dev_ctx = platform::DeviceContextPool::Instance().Get(fwd_var.place());
  grad_var->Resize(fwd_var.dims());
  grad_var->mutable_data(fwd_var.place(), fwd_var.type());
  operators::math::set_constant(*dev_ctx, grad_var, 1.0);
}

void BasicEngine::CheckBackwardInputs(const OpBase& op) {
  for (auto& pair : op.GetInsMap()) {
    if (!pair.second.IsGrad()) {
      continue;
    }

    for (auto& var : pair.second) {
      if (!var) {
        continue;
      }

      auto* inner_var = var->MutableVar();
      framework::Tensor* tensor = nullptr;
      if (!inner_var->IsInitialized() ||
          inner_var->IsType<framework::LoDTensor>()) {
        tensor = inner_var->GetMutable<framework::LoDTensor>();
      }

      if (tensor && !tensor->IsInitialized()) {
        VLOG(6) << "Set ungenerated Grad: " << var->Name() << " as zero";
        auto* dev_ctx = platform::DeviceContextPool::Instance().Get(op.place());
        tensor->mutable_data(op.place(), var->DataType());
        operators::math::set_constant(*dev_ctx, tensor, 0.0);
      }
    }
  }
}

void BasicEngine::PrepareGradAccumulators(const OpBase& op) {
  for (const auto& pair : op.GetOutsMap()) {
    if (!pair.second.IsGrad()) {
      continue;
    }

    for (const auto& var : pair.second) {
      if (!var) continue;

      auto& accumulator = accumulators_[var.get()];
      if (!accumulator) {
        if (backward_strategy_.sorted_sum_gradient_) {
          accumulator.reset(new SortedGradientAccumulator(var.get()));
        } else {
          accumulator.reset(new EagerGradientAccumulator(var.get()));
        }
      }

      accumulator->IncreaseRefCnt();

      VLOG(3) << "Prepare to acccumulate variable grad " << var->Name() << "("
              << var.get() << ")  with reference count "
              << accumulator->RefCnt();
    }
  }
}

void BasicEngine::PrepareDeps() {
  PADDLE_ENFORCE_EQ(
      node_deps_.empty(), true,
      platform::errors::AlreadyExists("Op deps must be initialized here"));
  PADDLE_ENFORCE_EQ(
      accumulators_.empty(), true,
      platform::errors::AlreadyExists("Accumulators must be initialized here"));

  std::queue<GradOpNode*> q;
  std::unordered_set<GradOpNode*> visited;

  q.push(init_node_.get());
  visited.insert(init_node_.get());

  while (!q.empty()) {
    auto* cur_node = q.front();
    q.pop();

    for (auto& cur_op : *cur_node) {
      cur_op.EnforceHasInOut();
      PrepareGradAccumulators(cur_op);
    }

    const auto& grad_pending_nodes = cur_node->GradPendingNodes();
    for (auto& grad_pending_node : grad_pending_nodes) {
      PADDLE_ENFORCE_NOT_NULL(
          grad_pending_node,
          platform::errors::NotFound("Grad pending node should not be null"));
      ++node_deps_[grad_pending_node.get()];
      if (visited.count(grad_pending_node.get()) == 0) {
        visited.insert(grad_pending_node.get());
        q.push(grad_pending_node.get());
      }
    }
  }
}

void BasicEngine::Execute() {
  if (init_node_ == nullptr) {
    return;
  }

  PrepareDeps();
  // Start execute Computation graph
  std::queue<std::shared_ptr<GradOpNode>> q;
  q.push(std::move(init_node_));

  size_t op_num = 0;

  while (!q.empty()) {
    auto shared_cur_node = std::move(q.front());
    q.pop();

    for (auto& cur_op : *shared_cur_node) {
      ++op_num;

      // CheckBackWardInput
      CheckBackwardInputs(cur_op);

      // Step 1: Run Backward
      auto& bwd_ins = cur_op.GetInsMap();
      auto& bwd_outs = cur_op.GetOutsMap();

      NameVarMap<VariableWrapper> tmp_outs(bwd_outs);
      // 1. construct the output map 2. replace the element in the map
      // A var may be coresponding to several grad var in one op
      for (auto& pair : tmp_outs) {
        if (!pair.second.IsGrad()) {
          continue;
        }

        for (auto& var : pair.second) {
          if (!var) {
            continue;
          }

          auto iter = accumulators_.find(var.get());
          PADDLE_ENFORCE_EQ(
              iter != accumulators_.end(), true,
              platform::errors::NotFound("Cannot find gradient of variable %s",
                                         var->Name()));

          if (!var->OverridedStopGradient() && iter->second->RefCnt() == 1) {
            continue;
          }

          var = std::make_shared<VariableWrapper>(var->Name());
          need_accu_var_list_.emplace_back(iter->second.get(), var);
        }
      }

      {
        VLOG(3) << "Start to execute grad op " << cur_op.Type();
        OpBase::Run(cur_op.InnerOp(), bwd_ins, tmp_outs, cur_op.Attrs(),
                    cur_op.place());
      }

      // Step 2: Sum Gradient
      for (auto& pair : need_accu_var_list_) {
        pair.first->Add(std::move(pair.second), cur_op.id());
      }

      need_accu_var_list_.clear();

      VLOG(3) << "Remove op after op " << cur_op.Type() << " runs";
      cur_op.ClearBackwardTrace();
    }

    // Step 3: Collect ready ops
    for (auto& grad_pending_node : shared_cur_node->GradPendingNodes()) {
      PADDLE_ENFORCE_NOT_NULL(grad_pending_node,
                              platform::errors::NotFound(
                                  "Grad pending node should not be nullptr"));
      auto iter = node_deps_.find(grad_pending_node.get());
      if (iter == node_deps_.end()) {
        continue;
      }

      if (--(iter->second) == 0) {
        q.push(grad_pending_node);
      }
    }
  }
  Clear();

  VLOG(1) << "Backward op number: " << op_num;
}

void BasicEngine::Clear() {
  init_node_.reset();
  node_deps_.clear();
  accumulators_.clear();
  need_accu_var_list_.clear();
}

}  // namespace imperative
}  // namespace paddle
