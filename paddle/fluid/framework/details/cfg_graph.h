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

#include <list>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <utility>

#include "paddle/fluid/framework/ir/graph.h"

namespace paddle {
namespace framework {
namespace details {

constexpr char kGlobalUnlivedNodePool[] = "global_unused_node_pool";
constexpr char kGlobalReusedNodePairMap[] = "global_reused_nodepair_map";

using UnlivedNodePool = std::set<ir::Node*>; // order matters
using ReusedNodePairMap = std::unordered_map<Node* /*op*/,
                                             std::pair<Node*/*var*/, Node*/*reused var*/>>;

class ControlFlowGraph {
 public:
  explicit ControlFlowGraph(const ir::Graph& graph);

  void LiveVariableAnalysis();

  void UpdateGraph(ir::Node* old_node, ir::Node* new_node, int beign_idx);

  const std::unordered_set<ir::Node*>& LiveIn(ir::Node* op) const;
  const std::unordered_set<ir::Node*>& LiveOut(ir::Node* op) const;
  const std::unordered_set<ir::Node*>& Def(ir::Node* op) const;
  const std::unordered_set<ir::Node*>& Use(ir::Node* op) const;
  const std::vector<ir::Node*>& Ops() const;

 private:
  typedef std::unordered_map<ir::Node*, std::list<ir::Node*>> NodeListType;
  typedef std::unordered_map<ir::Node*, std::unordered_set<ir::Node*>>
      NodeSetType;

  std::vector<ir::Node*> ops_;  // topology sort ops
  NodeListType successors_;
  NodeListType predecessors_;
  NodeSetType live_in_;
  NodeSetType live_out_;
  NodeSetType uses_;
  NodeSetType defs_;
};

}  // namespace details
}  // namespace framework
}  // namespace paddle
