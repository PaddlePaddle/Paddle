// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/imperative/autograd.h"
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

class AutoGradFunctor {
 public:
  AutoGradFunctor(OpBase* op, const detail::BackwardStrategy& strategy);

  void operator()();

 private:
  void PrepareDeps();

  bool CheckBackwardInputs(OpBase* op);

  void PrepareGradAccumulators(OpBase* op);

  void SumGradient(OpBase* op, std::shared_ptr<VarBase> src, VarBase* dst);

  Tracer* tracer_;
  OpBase* op_;
  detail::BackwardStrategy backward_strategy_;
  std::unordered_map<OpBase*, size_t> op_deps_;
  std::unordered_map<VarBase*, std::unique_ptr<GradientAccumulator>>
      accumulators_;
};

void AutoGrad(VarBase* var, const detail::BackwardStrategy& strategy) {
  auto* op = var->GeneratedOp();
  if (!op) {
    VLOG(3) << "Skip auto grad since generated op is nullptr";
    return;
  }

  platform::RecordEvent record_event("Imperative Backward");
  VLOG(3) << "start backward";

  PADDLE_ENFORCE(var->HasGradVar(), "Grad variable not exist for variable %s",
                 var->Name());

  auto& fwd_var = var->Var().Get<framework::LoDTensor>();
  auto* grad_var =
      var->GradVarBase()->MutableVar()->GetMutable<framework::LoDTensor>();
  auto* dev_ctx = platform::DeviceContextPool::Instance().Get(fwd_var.place());
  grad_var->Resize(fwd_var.dims());
  grad_var->mutable_data(fwd_var.place(), fwd_var.type());
  operators::math::set_constant(*dev_ctx, grad_var, 1.0);

  AutoGradFunctor functor(op, strategy);
  functor();
}

AutoGradFunctor::AutoGradFunctor(OpBase* op,
                                 const detail::BackwardStrategy& strategy)
    : tracer_(op->HoldedTracer()), op_(op), backward_strategy_(strategy) {
  PrepareDeps();
}

bool AutoGradFunctor::CheckBackwardInputs(OpBase* op) {
  for (auto& in : op->BackwardInputs()) {
    for (auto& pair : in) {
      for (auto& var : pair.second) {
        if (var && !var->StopGradient()) {
          return true;
        }
      }
    }
  }
  return false;
}

void AutoGradFunctor::PrepareGradAccumulators(OpBase* op) {
  for (auto& out : op->BackwardOutputs()) {
    for (auto& pair : out) {
      for (auto& var : pair.second) {
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
}

void AutoGradFunctor::PrepareDeps() {
  PADDLE_ENFORCE(op_deps_.empty(), "Op deps must be initialized here");
  PADDLE_ENFORCE(accumulators_.empty(),
                 "Accumulators must be initialized here");

  std::queue<OpBase*> q;
  std::unordered_set<OpBase*> visited;
  q.push(op_);
  visited.insert(op_);

  while (!q.empty()) {
    auto* cur_op = q.front();
    q.pop();
    VLOG(3) << "Checking grads of op " << cur_op->Type();

    if (!CheckBackwardInputs(cur_op)) {
      // TODO(zjl): clear ops that do not need grad before running autograd
      VLOG(3) << "Stop checking preceding ops of " << cur_op->Type()
              << " because all of its backward inputs is stop_gradient=True";
      continue;
    }

    PrepareGradAccumulators(cur_op);

    auto& preceding_ops = cur_op->PrecedingOps();
    for (auto* preceding_op : preceding_ops) {
      PADDLE_ENFORCE_NOT_NULL(preceding_op);
      ++op_deps_[preceding_op];
      if (visited.count(preceding_op) == 0) {
        visited.insert(preceding_op);
        q.push(preceding_op);
      }
    }
  }
}

void AutoGradFunctor::SumGradient(OpBase* op, std::shared_ptr<VarBase> src,
                                  VarBase* dst) {
  auto iter = accumulators_.find(dst);
  PADDLE_ENFORCE(iter != accumulators_.end(),
                 "Cannot find gradient of variable %s", dst->Name());
  iter->second->Add(std::move(src), op->id());
}

void AutoGradFunctor::operator()() {
  std::queue<OpBase*> q;
  q.push(op_);

  while (!q.empty()) {
    OpBase* cur_op = q.front();
    q.pop();

    // Step 1: Run Backward
    auto& grad_descs = cur_op->GradOpDescs();
    auto& bwd_ins = cur_op->BackwardInputs();
    auto& bwd_outs = cur_op->BackwardOutputs();
    size_t grad_op_num = grad_descs.size();

    PADDLE_ENFORCE_EQ(grad_op_num, bwd_ins.size());
    PADDLE_ENFORCE_EQ(grad_op_num, bwd_outs.size());

    for (size_t i = 0; i < grad_op_num; ++i) {
      NameVarBaseMap tmp_outs;
      // A var may be coresponding to several grad var in one op
      std::unordered_map<VarBase*, std::vector<std::shared_ptr<VarBase>>>
          var_map;
      size_t counter = 0;
      for (auto& bwd_out : bwd_outs[i]) {
        auto& tmp_var_list = tmp_outs[bwd_out.first];
        tmp_var_list.reserve(bwd_out.second.size());
        for (auto& var : bwd_out.second) {
          auto tmp_var = std::make_shared<VarBase>(false);  // Do not need grad
          tmp_var->SetName("Gtmp@" + std::to_string(counter++));
          tmp_var_list.emplace_back(tmp_var);
          if (var) {
            var_map[var.get()].emplace_back(std::move(tmp_var));
          }
        }
      }

      VLOG(3) << "Start to trace grad op " << grad_descs[i]->Type();
      tracer_->TraceOp(*(grad_descs[i]), bwd_ins[i], tmp_outs, cur_op->place(),
                       false);
      // Step 2: Sum Gradient
      {
        platform::RecordEvent record_event("merge_grads");
        for (auto& var_pair : var_map) {
          auto* dst_var = var_pair.first;
          if (dst_var == nullptr) continue;
          for (auto& src_var : var_pair.second) {
            VLOG(3) << "Sum gradient of variable " << dst_var->Name()
                    << " after op " << grad_descs[i]->Type();
            SumGradient(cur_op, std::move(src_var), dst_var);
          }
        }
      }
    }

    // Step 3: Collect ready ops
    for (auto* preceding_op : cur_op->PrecedingOps()) {
      PADDLE_ENFORCE_NOT_NULL(preceding_op);
      auto iter = op_deps_.find(preceding_op);
      if (iter == op_deps_.end()) {
        continue;
      }

      VLOG(3) << "Found preceding op of " << cur_op->Type();
      if (--(iter->second) == 0) {
        q.push(preceding_op);
        VLOG(3) << "Push preceding op " << preceding_op->Type()
                << " into queue";
      }
    }

    // Step 4: Delete op to collect unused variables
    VLOG(3) << "Remove op after op " << cur_op->Type() << " runs";
    tracer_->RemoveOp(cur_op);
  }

  VLOG(3) << "Clear left op in tracer";
  tracer_->Clear();
}

}  // namespace imperative
}  // namespace paddle
