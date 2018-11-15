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
#include <algorithm>
#include <list>
#include <map>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#include "paddle/fluid/framework/data_type.h"
#include "paddle/fluid/framework/details/memory_reuse_types.h"
#include "paddle/fluid/framework/ir/graph.h"

namespace paddle {
namespace framework {
namespace details {

class ControlFlowGraph {
 public:
  ControlFlowGraph() {}
  explicit ControlFlowGraph(const ir::Graph& graph);

  void LiveVariableAnalysis();

  void UpdateGraph(ir::Node* old_node, ir::Node* new_node, int beign_idx);

  const std::unordered_set<ir::Node*>& LiveIn(ir::Node* op) const;
  const std::unordered_set<ir::Node*>& LiveOut(ir::Node* op) const;
  const std::unordered_set<ir::Node*>& Def(ir::Node* op) const;
  const std::unordered_set<ir::Node*>& Use(ir::Node* op) const;
  const std::vector<ir::Node*>& Ops() const;
  std::vector<ir::Node*>& Ops();

 private:
  void ConnectNodes();
  using NodeListMap = std::unordered_map<ir::Node*, std::list<ir::Node*>>;
  using NodeSetMap =
      std::unordered_map<ir::Node*, std::unordered_set<ir::Node*>>;
  // successors ops use the output variables.
  NodeListMap successors_;
  // predecessors ops generated input variables.
  NodeListMap predecessors_;
  // variables lived before run current op.
  NodeSetMap live_in_;
  // variables lived after run current op.
  NodeSetMap live_out_;
  NodeSetMap uses_;             // op inputs
  NodeSetMap defs_;             // op outputs
  std::vector<ir::Node*> ops_;  // op sequence by topology sort
};

}  // namespace details
}  // namespace framework
}  // namespace paddle
