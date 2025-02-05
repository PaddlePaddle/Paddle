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
#include "paddle/cinn/operator_fusion/utils.h"
#include "paddle/common/enforce.h"

namespace cinn::fusion {

using MergePatternFn =
    std::function<StmtPattern(const StmtPattern&, const StmtPattern&)>;
class PatternGraph {
 public:
  PatternGraph(const std::vector<PatternContent>& nodes,
               const std::vector<pir::Value>& outputs,
               const PolicyManager policy_manager);

  std::vector<PatternNodePtr> ClusterOps();

  void SinkTrivialPattern();
  void HorizontalFusion();
  void ReduceLiftReduceTree();
  void ReduceTreeGrown();
  void ReduceTree_Trivial_Fusion();
  void LiftToItersPermutationPattern();
  void LimitedAnchorFusion();
  void ItersPermutationFusion();
  void SplitRecomputePattern();
  std::vector<PatternNodePtr> ReturnFusionResults();

  void RemoveNode(const PatternNodePtr& node);
  void AppendNode(const PatternNodePtr& node);
  void PrintGraphInfo() const;
  PatternNodePtr MergeNode(const PatternNodePtr& upstream,
                           const PatternNodePtr& downstream,
                           MergePatternFn merge_pattern_fn);
  std::vector<PatternNodePtr> SortByTopoOrder() const;
  std::vector<PatternNodePtr> SortByReverseTopoOrder() const;

  const PatternNodePtrSet& all_pattern_nodes() const {
    return all_pattern_nodes_;
  }
  const std::vector<pir::Value>& outputs() const { return outputs_; }
  const PolicyManager& policy_manager() const { return policy_manager_; }
  std::shared_ptr<ItersFusionPolicy> iters_fusion_policy() {
    return policy_manager_.template GetPolicy<ItersFusionPolicy>();
  }

 private:
  PatternNodePtrSet all_pattern_nodes_;
  std::vector<pir::Value> outputs_;
  PolicyManager policy_manager_;
};

}  // namespace cinn::fusion
