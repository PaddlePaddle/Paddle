// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/cinn/operator_fusion/pattern_node.h"
#include "paddle/cinn/operator_fusion/policy/policy_manager.h"
#include "paddle/cinn/operator_fusion/policy/relative_judge_policy.h"
#include "paddle/cinn/operator_fusion/utils.h"
#include "paddle/common/enforce.h"

namespace cinn::fusion {

template <typename T>
using PatternNodePtrSet = std::unordered_set<PatternNodePtr<T>>;
template <typename T>
using MergePatternFn =
    std::function<StmtPattern<T>(const StmtPattern<T>&, const StmtPattern<T>&)>;

template <typename T>
class PatternGraph {
 public:
  PatternGraph(const std::vector<PatternContent<T>>& nodes,
               const std::vector<pir::Value>& outputs,
               const PolicyManager<T> policy_manager);

  std::vector<PatternNodePtr<T>> ClusterOps();

  void SinkTrivialPattern();
  void HorizontalFusion();
  void ReduceLiftReduceTree();
  void ReduceTreeGrown();
  void ReduceTree_Trivial_Fusion();
  void LiftToAnchorPattern();
  void AnchorPatternFusion();
  void SplitRecomputePattern();

  void RemoveNode(const PatternNodePtr<T>& node);
  void AppendNode(const PatternNodePtr<T>& node);
  std::string GraphInfo() const;
  PatternNodePtr<T> MergeNode(const PatternNodePtr<T>& upstream,
                              const PatternNodePtr<T>& downstream,
                              MergePatternFn<T> merge_pattern_fn);
  std::vector<PatternNodePtr<T>> SortByTopoOrder();

  const PatternNodePtrSet<T>& all_pattern_nodes() const {
    return all_pattern_nodes_;
  }
  const std::vector<pir::Value>& outputs() const { return outputs_; }
  const PolicyManager<T>& policy_manager() const { return policy_manager_; }

 private:
  PatternNodePtrSet<T> all_pattern_nodes_;
  std::vector<pir::Value> outputs_;
  PolicyManager<T> policy_manager_;
};

}  // namespace cinn::fusion
