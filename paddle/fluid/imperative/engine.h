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
};

class BasicEngine : public Engine {
 public:
  void Init(VarBase* var, const detail::BackwardStrategy& strategy) override;

  void Execute() override;

 private:
  void PrepareDeps();

  void CheckBackwardInputs(OpBase* op);

  void PrepareGradAccumulators(OpBase* op);

  void SumGradient(OpBase* op, std::shared_ptr<VariableWrapper> src,
                   VariableWrapper* dst);

  // TODO(jiabin): maybe we can optimize the performance of engine by cache the
  // result
  void Clear() {
    init_ops_.clear();
    op_deps_.clear();
    accumulators_.clear();
  }

  std::vector<std::shared_ptr<OpBase>> init_ops_;
  detail::BackwardStrategy backward_strategy_;
  std::unordered_map<OpBase*, size_t> op_deps_;
  std::unordered_map<VariableWrapper*, std::unique_ptr<GradientAccumulator>>
      accumulators_;

  std::vector<std::pair<VariableWrapper*, std::shared_ptr<VariableWrapper>>>
      need_accu_var_list_;
};

}  // namespace imperative
}  // namespace paddle
