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

#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>
#include "paddle/fluid/imperative/backward_strategy.h"
#include "paddle/fluid/imperative/engine.h"
#include "paddle/fluid/imperative/gradient_accumulator.h"

namespace paddle {
namespace imperative {

class VarBase;
class OpBase;

class BasicEngine : public Engine {
 public:
  void Init(VarBase* var, const detail::BackwardStrategy& strategy);

  void Execute() override;

 private:
  void PrepareDeps();

  void CheckBackwardInputs(const OpBase& op);

  void PrepareGradAccumulators(const OpBase& op);

  void Clear();

 private:
  std::shared_ptr<GradOpNode> init_node_;
  detail::BackwardStrategy backward_strategy_;
  std::unordered_map<GradOpNode*, size_t> node_deps_;
  std::unordered_map<VariableWrapper*, std::unique_ptr<GradientAccumulator>>
      accumulators_;
  std::vector<std::pair<GradientAccumulator*, std::shared_ptr<VariableWrapper>>>
      need_accu_var_list_;
};

}  // namespace imperative
}  // namespace paddle
