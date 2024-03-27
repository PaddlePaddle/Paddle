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

#include "paddle/cinn/frontend/group_cluster/cluster_policy/policy_manager.h"
#include "paddle/cinn/frontend/group_cluster/common_utils.h"
#include "paddle/cinn/frontend/group_cluster/pattern_node.h"

namespace cinn::frontend::group_cluster {

struct PatternNodePtrHash {
  size_t operator()(const PatternNodePtr& node) const {
    return std::hash<PatternNode*>()(node.get());
  }
};

struct PatternNodePtrCompare {
  bool operator()(const std::shared_ptr<PatternNode>& a,
                  const std::shared_ptr<PatternNode>& b) const {
    return a.get() == b.get();
  }
};

using PatternNodePtrSet = std::
    unordered_set<PatternNodePtr, PatternNodePtrHash, PatternNodePtrCompare>;

class PatternGraph {
 public:
  PatternGraph(const std::vector<pir::Operation*>& ops,
               const policy::PolicyManager policy_manager);

  std::unordered_set<PatternNodePtr> ClusterOps();

 private:
  void SinkTrivialPattern();
  void FuseReducePattern();
  void ReduceLiftReduceTree();
  void ReduceTreeGrown();
  void ReduceTree_Trivial_Fusion();

  void RemoveNode(const PatternNodePtr& node);
  void AppendNode(const PatternNodePtr& node);
  void PrintGraph();
  void MergeNode(const PatternNodePtr& upstream,
                 const PatternNodePtr& downstream);

 private:
  PatternNodePtrSet all_pattern_nodes_;
  PatternNodePtrSet entrance_nodes_;
  PatternNodePtrSet exit_nodes_;
  const policy::PolicyManager policy_manager_;
};

}  // namespace cinn::frontend::group_cluster
