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
#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "paddle/fluid/framework/data_type.h"
#include "paddle/fluid/framework/details/memory_reuse_types.h"
#include "paddle/fluid/framework/ir/graph.h"
#include "paddle/fluid/framework/ir/pass.h"

namespace paddle {
namespace framework {
namespace details {
constexpr char kAllOpDescs[] = "all_op_descs";

std::vector<ir::Node*> SortOpLikeDescOrder(const ir::Graph& graph);
// sort op in bfs order
std::vector<ir::Node*> BFSSortGraphOps(const ir::Graph& graph);

class ControlFlowGraph;

class AnalysisVarPass : public ir::Pass {
 protected:
  std::unique_ptr<ir::Graph> ApplyImpl(
      std::unique_ptr<ir::Graph> graph) const override;

 private:
  // fill the variable map(var_nodes) by version.
  void InitSSAGraphNodes() const;
  // update program descs
  void RenameVarInGraphDesc(const std::string& var,
                            const std::string& cache_var, size_t idx) const;
  // update ir nodes
  void RenameVarInGraphNode(const std::string& var,
                            const std::string& cache_var, size_t idx,
                            ir::Graph* graph) const;

  void SubGraphOptimize(OpDesc* op_desc) const;
  // valid a tensor can be reuse or not
  bool NodeCanReused(ir::Node* node) const;
  // scan subblock and collect the output/input variables.
  std::unordered_set<std::string> GetSubBlockVars(
      const std::unordered_set<ir::Node*>&) const;
  // check op has subblock or not
  bool OpHasSubBlock(OpDesc* desc) const;

 private:
  // Reuse Node Pool, Owned.
  mutable OrderedNodePairPool pool_;
  // controlflow Graph
  mutable std::unique_ptr<ControlFlowGraph> cfg_;
  // skip set
  mutable std::unordered_set<std::string> skip_set_;
  // var nodes
  mutable std::map<std::string, std::vector<ir::Node*>> var_nodes_;
};

class ControlFlowGraph {
 public:
  ControlFlowGraph() = default;
  // For IR Graph in parallelexecutor
  explicit ControlFlowGraph(const ir::Graph& graph);

  void LiveVariableAnalysis();

  void RenameVarInCFGGraph(const std::string& old_node,
                           const std::string& new_node, int begin_idx);

  const std::set<std::string> LiveIn(ir::Node* op) const;
  const std::set<std::string> LiveOut(ir::Node* op) const;
  const std::set<std::string> Use(ir::Node* op) const;
  const std::vector<ir::Node*> Ops() const;
  std::vector<ir::Node*>& Ops();

  // for ssa-graph nodes
  ir::Node* GetNodeFromVarName(const std::string& name, ir::Node* op) const;

 private:
  void BuildCFGGraph();
  void ConnectNodes();
  using NodeListMap = std::unordered_map<ir::Node*, std::set<ir::Node*>>;
  using VarSetMap = std::map<ir::Node*, std::set<std::string>>;
  // successors ops use the output variables.
  NodeListMap successors_;
  // predecessors ops generated input variables.
  NodeListMap predecessors_;
  // variables lived before run current op.
  VarSetMap live_in_;
  // variables lived after run current op.
  VarSetMap live_out_;
  VarSetMap uses_;  // op inputs
  VarSetMap defs_;  // op outputs

  std::vector<ir::Node*> ops_;  // op sequence by topology sort
};

}  // namespace details
}  // namespace framework
}  // namespace paddle
