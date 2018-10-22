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

#include <algorithm>

#include "paddle/fluid/framework/details/cfg_graph.h"
#include "paddle/fluid/framework/ir/graph_helper.h"

namespace paddle {
namespace framework {
namespace details {

ControlFlowGraph::ControlFlowGraph(const ir::Graph& graph) {
  ops_ = ir::TopologySortOperations(graph);
  for (auto& op : ops_) {
    PADDLE_ENFORCE(
        op->NodeType() == ir::Node::Type::kOperation,
        string::Sprintf("ControlFlow expect op, but get %s ", op->Name()));
    if (successors_.find(op) == successors_.end()) {
      // op info init
      successors_[op] = std::list<ir::Node*>();
      predecessors_[op] = std::list<ir::Node*>();
      live_in_[op] = std::unordered_set<ir::Node*>();
      live_out_[op] = std::unordered_set<ir::Node*>();
      uses_[op] = std::unordered_set<ir::Node*>();
      defs_[op] = std::unordered_set<ir::Node*>();
    }
    for (auto& input_var : op->inputs) {
      for (auto& generated_op : input_var->inputs) {
        predecessors_[op].push_back(generated_op);
        successors_[generated_op].push_back(op);
      }
      PADDLE_ENFORCE(
          input_var->NodeType() == ir::Node::Type::kVariable,
          string::Sprintf("Expect var, but get %s", input_var->Name()));
      uses_[op].insert(input_var);
    }
    for (auto& output_var : op->outputs) {
      PADDLE_ENFORCE(
          output_var->NodeType() == ir::Node::Type::kVariable,
          string::Sprintf("Expect var, but get %s", output_var->Name()));
      defs_[op].insert(output_var);
    }
  }
}

void ControlFlowGraph::LiveVariableAnalysis() {
  // compute the liveness of for each variable though worklist algorithm.
  // It iterates the operators from end to begin, compute the live in/live out
  // variable set for each op, then the diff between in/out will be used for
  // the variable reuse. For detail refer to
  // http://www.cs.cornell.edu/courses/cs4120/2013fa/lectures/lec26-fa13.pdf

  std::unordered_set<ir::Node*> node_live_in;
  std::list<ir::Node*> worklist(ops_.begin(), ops_.end());
  std::reverse(worklist.begin(), worklist.end());
  auto set_equal = [](const std::unordered_set<ir::Node*>& lhs,
                      const std::unordered_set<ir::Node*>& rhs) -> bool {
    if (lhs.size() != rhs.size()) return false;
    for (auto& item : lhs) {
      if (rhs.find(item) == rhs.end()) {
        return false;
      }
    }
    return true;
  };
  while (!worklist.empty()) {
    ir::Node* op = worklist.front();
    worklist.pop_front();
    PADDLE_ENFORCE(live_in_.find(op) != live_in_.end(),
                   string::Sprintf("Expect var %s in live_in, but Not Found.",
                                   op->Name()));
    node_live_in = live_in_[op];
    for (auto& s : successors_[op]) {
      for (auto& var : live_in_[s]) {
        if (live_out_[op].find(var) == live_out_[op].end()) {
          live_out_[op].insert(var);
        }
      }
    }
    live_in_[op].clear();
    for (auto it = live_out_[op].begin(); it != live_out_[op].end(); ++it) {
      if (defs_[op].find(*it) == defs_[op].end()) {
        live_in_[op].insert(*it);
      }
    }
    for (auto it = uses_[op].begin(); it != uses_[op].end(); ++it) {
      if (live_in_[op].find(*it) == live_in_[op].end()) {
        live_in_[op].insert(*it);
      }
    }

    if (!set_equal(live_in_[op], node_live_in)) {
      for (auto& pre : predecessors_[op]) {
        worklist.push_back(pre);
      }
    }
  }
}

void ControlFlowGraph::UpdateGraph(ir::Node* old_node, ir::Node* new_node,
                                   int begin_idx) {
  // update graph from begin idx to the end
  for (size_t i = begin_idx; i != ops_.size(); ++i) {
    auto* op = ops_[i];
    if (uses_[op].find(old_node) != uses_[op].end()) {
      uses_[op].erase(old_node);
      uses_[op].insert(new_node);
    }
    if (defs_[op].find(old_node) != defs_[op].end()) {
      defs_[op].erase(old_node);
      defs_[op].insert(new_node);
    }
    if (live_in_[op].find(old_node) != live_in_[op].end()) {
      live_in_[op].erase(old_node);
      live_in_[op].insert(new_node);
    }
    if (live_out_[op].find(old_node) != live_out_[op].end()) {
      live_out_[op].erase(old_node);
      live_out_[op].insert(new_node);
    }
  }
}

const std::unordered_set<ir::Node*>& ControlFlowGraph::LiveIn(
    ir::Node* op) const {
  auto it = live_in_.find(op);
  PADDLE_ENFORCE(
      it != live_in_.end(),
      string::Sprintf("Expect %s in live_in, but Not Found.", op->Name()));
  return it->second;
}

const std::unordered_set<ir::Node*>& ControlFlowGraph::LiveOut(
    ir::Node* op) const {
  auto it = live_out_.find(op);
  PADDLE_ENFORCE(
      it != live_out_.end(),
      string::Sprintf("Expect %s in live_out, but Not Found.", op->Name()));
  return it->second;
}
const std::unordered_set<ir::Node*>& ControlFlowGraph::Def(ir::Node* op) const {
  auto it = defs_.find(op);
  PADDLE_ENFORCE(
      it != defs_.end(),
      string::Sprintf("Expect %s in defs, but Not Found.", op->Name()));
  return it->second;
}

const std::unordered_set<ir::Node*>& ControlFlowGraph::Use(ir::Node* op) const {
  auto it = uses_.find(op);
  PADDLE_ENFORCE(
      it != uses_.end(),
      string::Sprintf("Expect %s in live_out, but Not Found.", op->Name()));
  return it->second;
}

const std::vector<ir::Node*>& ControlFlowGraph::Ops() const { return ops_; }

}  // namespace details
}  // namespace framework
}  // namespace paddle
