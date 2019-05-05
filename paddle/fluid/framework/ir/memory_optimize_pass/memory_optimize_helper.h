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
#include <iostream>
#include <iterator>
#include <list>
#include <map>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#include "paddle/fluid/framework/data_type.h"
#include "paddle/fluid/framework/ir/graph.h"

namespace paddle {
namespace framework {
namespace ir {

/// this attribute is used to avoid some core variables removed/reused
/// in memory optimize related passes
constexpr char kMemOptSkipVars[] = "@MEM_OPT_SKIP_VARS@";
typedef std::unordered_set<std::string> MemOptSkipVars;

std::vector<ir::Node*> SortOpLikeDescOrder(const ir::Graph& graph);

// NOTE(dzh): A ordered set for node reuse in memory optimize.
// the orderedset sort node in ascend order(by node bytes size).
// in fluid, -1 means the batch_size, which is determined in runtime.
// So the reuse happens between nodes who's batch_size both are -1
// simultaneously or not.
//
// sort rule:
// rule 0 : smaller node ranking in front.
// rule 1 : batch_size equal -1 ranking in the front than the node not.
//
// For example,
// node0[-1, 1] node1[-1, 1, 1], node2[1,1], node3[1,1024], ..

class OrderedSet {
 public:
  // nodes with same name exists in pool.
  using NodeVector = std::vector<ir::Node*>;
  using Iter = typename std::list<NodeVector>::iterator;
  using ConstIter = typename std::list<NodeVector>::const_iterator;

  void Insert(ir::Node* var);
  void Erase(ir::Node* var);
  void Erase(const std::string& var);
  bool Has(ir::Node* var) const;
  void Clear() {
    mark_table_.clear();
    nodes_.clear();
  }
  // find the bestfit shape node block with var.
  ir::Node* FindBestFitNode(ir::Node* var) const;
  ir::Node* FindNextBestFitNode(ir::Node* var, ir::Node* prev) const;
  // map store non-const iterator, can not promise const
  int GetNodeIndexInPool(ir::Node* var);
  // pool all node to string
  std::string ToString() const;

  Iter begin() { return nodes_.begin(); }
  Iter end() { return nodes_.end(); }
  ConstIter begin() const { return nodes_.begin(); }
  ConstIter end() const { return nodes_.end(); }

  size_t size() const { return nodes_.size(); }

 private:
  // for searching.
  std::unordered_map<std::string, Iter> mark_table_;
  // node pool
  std::list<NodeVector> nodes_;
};

class ControlFlowGraph {
 public:
  ControlFlowGraph() = default;
  // IR Graph
  explicit ControlFlowGraph(const ir::Graph& graph);

  void LiveVariableAnalysis();

  void RenameVarInCFGGraph(const std::string& old_node,
                           const std::string& new_node, int begin_idx);

  const std::set<std::string>& LiveIn(ir::Node* op) const;
  const std::set<std::string>& LiveOut(ir::Node* op) const;
  const std::set<std::string>& Use(ir::Node* op) const;
  const std::set<std::string>& Unlived(ir::Node* op) const;
  const std::vector<ir::Node*>& Ops() const;
  std::vector<ir::Node*>& Ops();

  // for ssa-graph nodes
  ir::Node* GetNodeByName(const std::string& name, ir::Node* op) const;

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
  std::unordered_map<ir::Node*, std::set<std::string>> unlived_vars_;

  std::vector<ir::Node*> ops_;  // op sequence by topology sort
};

// valid a tensor can be reuse or not
bool NodeCanReused(ir::Node* node);

// valid a tensor can be reuse or not.
bool NodeCanReused(const VarDesc& node);

// check op has subblock or not
bool OpHasSubBlock(OpDesc* desc);

// node memory size in bytes
size_t NodeSize(ir::Node* n);

// node memory size in bytes
size_t NodeSize(const VarDesc&);

std::string DebugString(ir::Node* var);

VarDesc* GetVarDesc(ir::Node* n);

static inline bool IsSameDesc(OpDesc* op1, OpDesc* op2) {
  return op1->Type() == op2->Type() && op1->Inputs() == op2->Inputs() &&
         op1->Outputs() == op2->Outputs();
}

template <typename Container, typename Callback>
class FilterVariableImpl {
 public:
  void operator()(const Container& nodes, Callback callback) {
    for (auto* node : nodes) {
      callback(node);
    }
  }
};

// filter var node for op->inputs/outputs
template <typename Callback>
class FilterVariableImpl<std::vector<ir::Node*>, Callback> {
 public:
  void operator()(const std::vector<ir::Node*>& nodes, Callback callback) {
    for (auto* var : nodes) {
      if (var->IsVar() && !var->IsCtrlVar()) {
        callback(var);
      }
    }
  }
};

template <typename Container, typename Callback>
void FilterVariables(const Container& nodes, Callback callback) {
  FilterVariableImpl<Container, Callback>()(nodes, callback);
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle
