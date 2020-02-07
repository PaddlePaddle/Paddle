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

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#include "paddle/fluid/imperative/backward_strategy.h"
#include "paddle/fluid/imperative/gradient_accumulator.h"
#include "paddle/fluid/imperative/layer.h"

namespace paddle {
namespace imperative {

// It seems there is no need for Engine to be an
// singleton, we can have multi-engine to run
// mutil-graoh. For future use we may expose a interface
// to Python to support
class Engine {
 public:
  virtual ~Engine() = default;
  virtual void Execute() = 0;
  virtual void Init(VarBase* var, const detail::BackwardStrategy& strategy) = 0;
  virtual void RunOp(imperative::OpBase* op, const NameVarBaseMap& ins,
                     const NameVarBaseMap& outs, const platform::Place& place);

  virtual void RemoveOp(OpBase* op) {
    PADDLE_ENFORCE_NOT_NULL(op, "Cannot remove null op");
    auto iter = grad_ops_.find(op);
    PADDLE_ENFORCE_EQ(iter != grad_ops_.end(), true, "Op is not inside tracer");
    grad_ops_.erase(iter);
  }

  void InsertOp(OpBase* op, std::shared_ptr<OpBase> op_shared) {
    grad_ops_[op] = std::move(op_shared);
  }

  const std::unordered_set<VarBase*>& GradVars() const { return grad_vars_; }

  const std::unordered_map<OpBase*, std::shared_ptr<OpBase>>& GradOps() const {
    return grad_ops_;
  }

  void InsertGradVar(VarBase* grad) { grad_vars_.emplace(grad); }

  bool IsGrad(VarBase* var) { return grad_vars_.count(var) > 0; }

  void Clear() {
    grad_ops_.clear();
    grad_vars_.clear();
  }

 private:
  std::unordered_map<OpBase*, std::shared_ptr<OpBase>>
      grad_ops_;  // opBase for remove - grad_op
  std::unordered_set<VarBase*> grad_vars_;
};

class BasicEngine : public Engine {
 public:
  BasicEngine() = default;

  void Init(VarBase* var, const detail::BackwardStrategy& strategy) override;

  ~BasicEngine() override = default;

  void Execute() override;

 private:
  void PrepareDeps();

  void CheckBackwardInputs(OpBase* op);

  void SetBackwardOutputs(OpBase* op);

  void PrepareGradAccumulators(OpBase* op);

  void SumGradient(OpBase* op, std::shared_ptr<VarBase> src, VarBase* dst);

  // TODO(jiabin): maybe we can optimize the performance of engine by cache the
  // result
  void CleanEngine() {
    init_ops_.clear();
    op_deps_.clear();
    accumulators_.clear();
    Clear();
  }

  std::vector<OpBase*> init_ops_;
  detail::BackwardStrategy backward_strategy_;
  std::unordered_map<OpBase*, size_t> op_deps_;
  std::unordered_map<VarBase*, std::unique_ptr<GradientAccumulator>>
      accumulators_;
};

}  // namespace imperative
}  // namespace paddle
