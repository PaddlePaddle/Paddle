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

#include "paddle/fluid/imperative/engine.h"

#include <algorithm>
#include <memory>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#include "paddle/fluid/imperative/gradient_accumulator.h"
#include "paddle/fluid/imperative/layer.h"
#include "paddle/fluid/imperative/tracer.h"
#include "paddle/fluid/operators/math/math_function.h"
#include "paddle/fluid/platform/profiler.h"

namespace paddle {
namespace imperative {

void BasicEngine::Init(VarBase* var, const detail::BackwardStrategy& strategy) {
  backward_strategy_ = strategy;
  const auto& ops = var->GradVarBase()->GradOps();
  var->ClearGradOps();

  if (ops.empty() || var->OverridedStopGradient()) {
    VLOG(3) << "Skip auto grad since there is no grad op for var or loss is "
               "stop_gradient=True: "
            << var->Name();
    return;
  } else {
    bool valid = false;
    for (const auto& op : ops) {
      if (op) {
        valid = true;
      }
    }
    if (!valid) {
      VLOG(3) << "Skip auto grad since all grad op of start VarBase is nullptr";
      return;
    }
  }

  init_ops_ = ops;
  var->GradVarBase()->ClearGradOps();
  VLOG(3) << "start backward";

  PADDLE_ENFORCE_EQ(var->HasGradVar(), true,
                    "Grad variable not exist for variable %s", var->Name());

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

void BasicEngine::CheckBackwardInputs(OpBase* op) {
  for (auto& pair : op->GetInsMap()) {
    for (auto& var : pair.second) {
      if (!var || op->IsAllowedEmptyVar(var.get())) {
        continue;
      }

      auto* inner_var = var->MutableVar();
      framework::Tensor* tensor = nullptr;
      if (!inner_var->IsInitialized() ||
          inner_var->IsType<framework::LoDTensor>()) {
        tensor = inner_var->GetMutable<framework::LoDTensor>();
      }

      if (tensor && !tensor->IsInitialized()) {
        // if grad var has OverridedStopGradient skip this Op
        VLOG(6) << "Set ungenerated Grad: " << var->Name() << " as zero";
        auto* dev_ctx =
            platform::DeviceContextPool::Instance().Get(op->place());
        tensor->mutable_data(op->place(), var->DataType());
        operators::math::set_constant(*dev_ctx, tensor, 0.0);
      }
    }
  }
}

void BasicEngine::PrepareGradAccumulators(OpBase* op) {
  for (const auto& pair : op->GetOutsMap()) {
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

      VLOG(3) << "Prepare to acccumulate variable grad " << var->Name()
              << "with reference count " << accumulator->RefCnt();
    }
  }
}

void BasicEngine::PrepareDeps() {
  PADDLE_ENFORCE_EQ(op_deps_.empty(), true, "Op deps must be initialized here");
  PADDLE_ENFORCE_EQ(accumulators_.empty(), true,
                    "Accumulators must be initialized here");

  std::queue<OpBase*> q;
  std::unordered_set<OpBase*> visited;
  for (const auto& init_op : init_ops_) {
    q.push(init_op.get());
    visited.insert(init_op.get());
  }

  while (!q.empty()) {
    auto* cur_op = q.front();
    q.pop();

    PADDLE_ENFORCE_NE(
        cur_op->GetInsMap().empty() && cur_op->GetOutsMap().empty(), true,
        platform::errors::NotFound(
            "Inputs and outputs of %s do not exist. "
            "This may be because you call \"backward()\" twice for the same "
            "subgraph. Please try to call \"stop_gradient = True\" or "
            "\"detach()\" if you use some same vars between two \"backward()\" "
            "calls.",
            cur_op->Type()));

    PrepareGradAccumulators(cur_op);

    const auto& grad_pending_ops = cur_op->GradPendingOps();
    for (auto& grad_pending_op : grad_pending_ops) {
      PADDLE_ENFORCE_NOT_NULL(grad_pending_op);
      ++op_deps_[grad_pending_op.get()];
      if (visited.count(grad_pending_op.get()) == 0) {
        visited.insert(grad_pending_op.get());
        q.push(grad_pending_op.get());
      }
    }
  }
}

void BasicEngine::SumGradient(OpBase* op, std::shared_ptr<VariableWrapper> src,
                              VariableWrapper* dst) {
  auto iter = accumulators_.find(dst);
  PADDLE_ENFORCE_EQ(iter != accumulators_.end(), true,
                    "Cannot find gradient of variable %s", dst->Name());
  iter->second->Add(std::move(src), op->id());
}

void BasicEngine::Execute() {
  PrepareDeps();
  // Start execute Computation graph
  std::queue<std::shared_ptr<OpBase>> q;
  for (const auto& init_op : init_ops_) {
    q.push(std::move(init_op));
  }

  size_t op_num = 0;

  while (!q.empty()) {
    auto shared_cur_op = std::move(q.front());
    q.pop();

    auto* cur_op = shared_cur_op.get();
    ++op_num;

    // CheckBackWardInput
    CheckBackwardInputs(cur_op);

    // Step 1: Run Backward
    auto& bwd_ins = cur_op->GetInsMap();
    auto& bwd_outs = cur_op->GetOutsMap();

    NameVarMap<VariableWrapper> tmp_outs(bwd_outs);
    // 1. construct the output map 2. replace the element in the map
    // A var may be coresponding to several grad var in one op
    for (auto it = tmp_outs.begin(); it != tmp_outs.end(); ++it) {
      for (size_t i = 0; i < it->second.size(); ++i) {
        auto tmp_var =
            std::make_shared<VariableWrapper>("Gtmp@");  // Do not need grad

        auto var = it->second[i];
        it->second[i] = tmp_var;
        if (var) {
          need_accu_var_list_.emplace_back(var.get(), std::move(tmp_var));
        }
      }
    }

    {
      VLOG(3) << "Start to execute grad op " << cur_op->Type();
      OpBase::Run(cur_op->InnerOp(), bwd_ins, tmp_outs, cur_op->Attrs(),
                  cur_op->place());
    }

    // Step 2: Sum Gradient

    if (need_accu_var_list_.size() > 0) {
      for (auto& pair : need_accu_var_list_) {
        SumGradient(cur_op, std::move(pair.second), pair.first);
      }
    }

    need_accu_var_list_.clear();

    // Step 3: Collect ready ops

    for (auto& grad_pending_op : cur_op->GradPendingOps()) {
      PADDLE_ENFORCE_NOT_NULL(grad_pending_op);
      auto iter = op_deps_.find(grad_pending_op.get());
      if (iter == op_deps_.end()) {
        continue;
      }

      VLOG(3) << "Found grad_pending op of " << cur_op->Type();
      // An Op is ready to go while its deps comes to zero

      if (--(iter->second) == 0) {
        q.push(grad_pending_op);
        VLOG(3) << "Push grad_pending op " << grad_pending_op->Type()
                << " into queue";
      }
    }

    // Step 4: Delete op to collect unused variables
    VLOG(3) << "Remove op after op " << cur_op->Type() << " runs";
    cur_op->ClearBackwardTrace();
  }
  Clear();

  VLOG(1) << "Backward op number: " << op_num;
}
}  // namespace imperative
}  // namespace paddle
