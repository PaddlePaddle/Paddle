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

void Engine::RunOp(paddle::imperative::OpBase* op,
                   const paddle::imperative::NameVarBaseMap& ins,
                   const paddle::imperative::NameVarBaseMap& outs,
                   const paddle::platform::Place& place) {
  platform::RecordEvent event(op->Type());

  op->Run(ins, outs);
}

void BasicEngine::Init(VarBase* var, const detail::BackwardStrategy& strategy) {
  backward_strategy_ = strategy;
  const std::vector<OpBase*> ops = var->GradVarBase()->GradOps();
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
  platform::RecordEvent record_event("Imperative Backward");
  VLOG(3) << "start backward";

  PADDLE_ENFORCE_EQ(var->HasGradVar(), true,
                    "Grad variable not exist for variable %s", var->Name());

  auto& fwd_var = var->Var().Get<framework::LoDTensor>();
  auto* grad_var =
      var->GradVarBase()->MutableVar()->GetMutable<framework::LoDTensor>();
  VLOG(6) << "init loss grad:" << var->GradVarBase()->Name()
          << " as stop_gradient false";
  var->GradVarBase()->InnerSetOverridedStopGradient(false);
  var->GradVarBase()->SetGradGenerated(true);
  auto* dev_ctx = platform::DeviceContextPool::Instance().Get(fwd_var.place());
  grad_var->Resize(fwd_var.dims());
  grad_var->mutable_data(fwd_var.place(), fwd_var.type());
  operators::math::set_constant(*dev_ctx, grad_var, 1.0);
}

void BasicEngine::CheckBackwardInputs(OpBase* op) {
  for (auto& pair : op->GetInsMap()) {
    for (auto& var : pair.second) {
      if (var && IsGrad(var.get())) {
        // if grad var has OverridedStopGradient skip this Op
        if (!var->GradGenerated()) {
          VLOG(6) << "Set ungenerated Grad: " << var->Name() << " as zero";
          auto* dev_ctx =
              platform::DeviceContextPool::Instance().Get(op->place());
          auto* tensor = var->MutableVar()->GetMutable<framework::LoDTensor>();
          tensor->mutable_data(op->place(), var->DataType());
          operators::math::set_constant(*dev_ctx, tensor, 0.0);
        } else {
          continue;
        }
      }
    }
  }
}

void BasicEngine::SetBackwardOutputs(paddle::imperative::OpBase* op) {
  for (auto& pair : op->GetOutsMap()) {
    for (auto& var : pair.second) {
      if (var) {
        // Set Backward outputs's generate_grad as true
        var->SetGradGenerated(true);
        VLOG(6) << "Set backward output: " << var->Name()
                << "'s SetGeneratedGrad as True";
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
    q.push(init_op);
    visited.insert(init_op);
  }

  while (!q.empty()) {
    auto* cur_op = q.front();
    q.pop();
    VLOG(3) << "Checking grads of op " << cur_op->Type();

    SetBackwardOutputs(cur_op);

    PrepareGradAccumulators(cur_op);

    auto& grad_pending_ops = cur_op->GradPendingOps();
    for (auto* grad_pending_op : grad_pending_ops) {
      PADDLE_ENFORCE_NOT_NULL(grad_pending_op);
      ++op_deps_[grad_pending_op];
      if (visited.count(grad_pending_op) == 0) {
        visited.insert(grad_pending_op);
        q.push(grad_pending_op);
      }
    }
  }
}

void BasicEngine::SumGradient(OpBase* op, std::shared_ptr<VarBase> src,
                              VarBase* dst) {
  auto iter = accumulators_.find(dst);

  PADDLE_ENFORCE_EQ(iter != accumulators_.end(), true,
                    "Cannot find gradient of variable %s", dst->Name());
  iter->second->Add(std::move(src), op->id());
}
void BasicEngine::Execute() {
  PrepareDeps();
  // Start execute Computation graph
  std::queue<OpBase*> q;
  for (const auto& init_op : init_ops_) {
    q.push(init_op);
  }
  while (!q.empty()) {
    OpBase* cur_op = q.front();
    q.pop();

    // CheckBackWardInput
    CheckBackwardInputs(cur_op);

    // Step 1: Run Backward
    auto& bwd_ins = cur_op->GetInsMap();
    auto& bwd_outs = cur_op->GetOutsMap();

    NameVarBaseMap tmp_outs;
    // A var may be coresponding to several grad var in one op
    std::unordered_map<VarBase*, std::vector<std::shared_ptr<VarBase>>> var_map;
    for (auto& bwd_out : bwd_outs) {
      auto& tmp_var_list = tmp_outs[bwd_out.first];
      tmp_var_list.reserve(bwd_out.second.size());
      for (auto& var : bwd_out.second) {
        auto tmp_var =
            std::make_shared<VarBase>(false, "Gtmp@");  // Do not need grad
        tmp_var_list.emplace_back(tmp_var);
        if (var) {
          var_map[var.get()].emplace_back(std::move(tmp_var));

          var->ClearGradOps();
        }
      }
    }
    VLOG(3) << "Start to execute grad op " << cur_op->Type();
    RunOp(cur_op, bwd_ins, tmp_outs, cur_op->place());
    // Step 2: Sum Gradient
    {
      platform::RecordEvent record_event("merge_grads");
      for (auto& var_pair : var_map) {
        auto* dst_var = var_pair.first;
        if (dst_var == nullptr) continue;
        for (auto& src_var : var_pair.second) {
          VLOG(3) << "Sum gradient of variable " << dst_var->Name()
                  << " after op " << cur_op->Type();
          SumGradient(cur_op, std::move(src_var), dst_var);
        }
      }
    }

    // Step 3: Collect ready ops

    for (auto* grad_pending_op : cur_op->GradPendingOps()) {
      PADDLE_ENFORCE_NOT_NULL(grad_pending_op);
      auto iter = op_deps_.find(grad_pending_op);
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
    RemoveOp(cur_op);
  }
  VLOG(3) << "Clean properties of BasicEngine";
  CleanEngine();
}
}  // namespace imperative
}  // namespace paddle
