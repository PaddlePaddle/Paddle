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
#include <unordered_set>
#include <utility>
#include <vector>

#include "paddle/fluid/imperative/engine.h"
#include "paddle/fluid/imperative/gradient_accumulator.h"
#include "paddle/utils/test_macros.h"

namespace paddle {
namespace imperative {

class VarBase;
class OpBase;

class TEST_API BasicEngine : public Engine {
 public:
  void Init(const std::vector<std::shared_ptr<VarBase>>& tensors,
            const std::vector<std::shared_ptr<VarBase>>& grad_tensors,
            bool retain_graph = false);

  void Execute() override;

 private:
  void PrepareDeps();

  void CheckBackwardInputs(const OpBase& op);

  void PrepareGradAccumulators(
      const OpBase& op,
      const std::vector<std::shared_ptr<GradOpNode>>& grad_pending_nodes);

  void Clear();

 private:
  std::vector<std::shared_ptr<GradOpNode>> init_nodes_;
  std::unordered_map<GradOpNode*, size_t> node_deps_;
  // The input and output of Inplace op are the same. If only `var` is used
  // as the key, then the input and output of inplace op must be gradient
  // accumulated. Therefore, add the `grad_node` as the key to prevent the
  // problem of gradient accumulation in inplace op.
  std::unordered_map<std::shared_ptr<GradOpNode>,
                     std::unordered_map<VariableWrapper*,
                                        std::unique_ptr<GradientAccumulator>>>
      accumulators_with_grad_node_;
  // Leaf var doesn't have grad_node, and leaf var with `stop_gradient=False`
  // can't use Inplace strategy. If a var doesn't have grad_node, only use
  // `var` as the key.
  std::unordered_map<VariableWrapper*, std::unique_ptr<GradientAccumulator>>
      accumulators_;
  // The output grad var of Inplace grad op. Because Inplace grad op does not
  // use the Inplace strategy, a new output grad var needs to be created.
  std::vector<std::pair<std::shared_ptr<VariableWrapper>,
                        std::shared_ptr<VariableWrapper>>>
      inplace_output_grad_var_list_;
  std::vector<std::pair<GradientAccumulator*, std::shared_ptr<VariableWrapper>>>
      need_accu_var_list_;
  // leaf_accumulators_ is only for leaf tensor(hooks/accumulate grad)
  // It should be orderly and not repeated, because multiple cards must ensure
  // that the order of vars is the same.
  std::vector<GradientAccumulator*> leaf_accumulators_;

  bool retain_graph_;
};

}  // namespace imperative
}  // namespace paddle
