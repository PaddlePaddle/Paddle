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

DECLARE_bool(sort_sum_gradient);

namespace paddle {
namespace imperative {

void BasicEngine::Init(VarBase* var, bool retain_graph) {
  retain_graph_ = retain_graph;
  init_node_ = var->GradVarBase()->GradNode();
  PADDLE_ENFORCE_EQ(var->GradVarBase()->GraphIsFreed(), false,
                    platform::errors::Unavailable(
                        "%s trying to backward through the same graph a second "
                        "time, but this graph have already been freed. Please "
                        "specify Tensor.backward(retain_graph=True) when "
                        "calling backward at the first time.",
                        var->Name()));

  if (!retain_graph) {
    VLOG(5) << "Clear the auto-grad graph from grad var " << var->Name()
            << " because of retain_graph=False when calling backward";
    var->GradVarBase()->SetGraphIsFreed(true);
    var->GradVarBase()->ClearGradNode();
  }

  if (init_node_ == nullptr || var->OverridedStopGradient()) {
    VLOG(3) << "Skip auto grad since there is no grad op for var or loss is "
               "stop_gradient=True: "
            << var->Name();
    return;
  }

  VLOG(3) << "Init first node of backward";

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
        if (FLAGS_sort_sum_gradient) {
          accumulator.reset(new SortedGradientAccumulator(var.get()));
        } else {
          accumulator.reset(new EagerGradientAccumulator(var.get()));
        }
      }

      accumulator->IncreaseRefCnt();

      VLOG(3) << "Prepare to acccumulate variable grad " << var->Name() << "("
              << var.get() << ")  with reference count "
              << accumulator->RefCnt();

      if (var->HasLeafHooks()) {
        VLOG(3) << "Grad variable wrapper (" << var->Name()
                << ") has leaf grad hooks.";
        PADDLE_ENFORCE_NE(var->HasGradNode(), true,
                          platform::errors::PermissionDenied(
                              "Only leaf Tensor's gradient can append hook to "
                              "Gradientaccumulator."));
        accumulator->SetPostHooks(var->GetLeafHooks());
      }
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

      // Step 1: Run Backward OP
      auto& bwd_ins = cur_op.GetInsMap();
      auto& bwd_outs = cur_op.GetOutsMap();

      NameVarMap<VariableWrapper> tmp_outs(bwd_outs);
      // 1. construct the temp output map, avoid to disrupt graph
      // 2. replace the element in the map by temp var, because a
      // var may be coresponding to several grad var in one op
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

          // leaf_accumulators_ : hooks and accumulate-grad for leaf tensor
          if (var->IsLeafGrad()) {
            leaf_accumulators_.insert(iter->second.get());

            if (iter->second->HasInnerVar()) {
              var = iter->second->InnerVar();
            }
          }

          if (var->OverridedStopGradient() || iter->second->RefCnt() > 1) {
            auto tmp_var = std::make_shared<VariableWrapper>(var->Name());
            tmp_var->SetType(var->Type());
            var = tmp_var;
            need_accu_var_list_.emplace_back(iter->second.get(), var);
            VLOG(10) << "create temporary var of " << var->Name()
                     << " for sum gradient within this graph!";
          }
        }
      }

      VLOG(4) << "Check whether there is any inplace operation affecting "
                 "gradient calculation.";
      for (auto& pair : bwd_ins) {
        for (auto& var_wrapper : pair.second) {
          auto wrapper_version_snapshot = var_wrapper->InplaceVersionSnapshot();
          auto tensor_version =
              var_wrapper->MutableVar()->CurrentInplaceVersion();
          PADDLE_ENFORCE_EQ(
              tensor_version, wrapper_version_snapshot,
              platform::errors::PermissionDenied(
                  "Tensor '%s' used in gradient computation in grad op '%s' "
                  "has been "
                  "modified by an inplace operation. "
                  "Its version is %s but the expected version is %s. "
                  "Please fix your code to void calling an inplace operator "
                  "after using the Tensor which will used in gradient "
                  "computation.",
                  var_wrapper->Name(), cur_op.Type(), tensor_version,
                  wrapper_version_snapshot));

          VLOG(6) << " The version of Tensor '" << var_wrapper->Name()
                  << "' is [ " << wrapper_version_snapshot << " ]";
        }
      }

      {
        VLOG(3) << "Start to execute grad op " << cur_op.Type();
        OpBase::Run(cur_op.InnerOp(), bwd_ins, tmp_outs, cur_op.Attrs(),
                    cur_op.place());
      }

      // Step 2: Sum Gradient of This graph
      for (auto& pair : need_accu_var_list_) {
        pair.first->SumGrad(std::move(pair.second), cur_op.id());
      }

      // Step 3: Call Hooks && Sum Gradient with Pre-Graph && Call BackwardHooks
      for (auto* accumulator : leaf_accumulators_) {
        if (!accumulator->SumGradCompleted()) {
          continue;
        }
        // 1. Call Hooks for **inner_var_**

        // 2. Sum Gradient with Previous Graph
        accumulator->AccumulateGrad();

        // 3. Call backward Hooks for **var_**
        if (accumulator->HasPostHooks()) {
          accumulator->CallBackwardPostHooks();
        }
      }

      need_accu_var_list_.clear();
      leaf_accumulators_.clear();

      if (!retain_graph_) {
        VLOG(3) << "Remove op after op " << cur_op.Type() << " runs";
        cur_op.ClearBackwardTrace();
      }
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
  leaf_accumulators_.clear();
}

}  // namespace imperative
}  // namespace paddle
