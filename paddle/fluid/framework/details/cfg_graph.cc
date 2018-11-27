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
#include <string>
#include <utility>

#include "paddle/fluid/framework/details/cfg_graph.h"
#include "paddle/fluid/framework/ir/graph_helper.h"

namespace paddle {
namespace framework {
namespace details {

ControlFlowGraph::ControlFlowGraph(const ir::Graph& graph) {
  ops_ = ir::SortOperationsInSequence(graph);
  ConnectNodes();
}

void ControlFlowGraph::BuildCFGGraph() {
  // FIXME(dzh): same effect with ConnectNodes, but use the control
  // link to build dependency graph, it goes wrong in transformer.
  for (ir::Node* op : ops_) {
    for (auto& input_var : op->inputs) {
      if (!input_var->inputs.empty()) {
        PADDLE_ENFORCE(
            input_var->inputs.size() == 1 && input_var->inputs[0]->IsOp(),
            "Preceding Op Node of Var Node must be unique");
        auto* pred_op = input_var->inputs[0];
        if (pred_op->Op() != nullptr) {
          predecessors_[op].insert(pred_op);
          successors_[pred_op].insert(op);
        }
      }
      if (input_var->IsVar() && !input_var->IsCtrlVar()) {
        uses_[op].insert(input_var->Name());
      }
    }
    for (auto& output_var : op->outputs) {
      // output var may be used by many op
      for (auto* succ_op : output_var->outputs) {
        if (succ_op->Op() != nullptr) {
          successors_[op].insert(succ_op);
          predecessors_[succ_op].insert(op);
        }
      }
      if (output_var->IsVar() && !output_var->IsCtrlVar()) {
        defs_[op].insert(output_var->Name());
      }
    }
  }
}

void ControlFlowGraph::ConnectNodes() {
  for (size_t i = 0; i < ops_.size(); ++i) {
    auto& op = ops_[i];
    try {
      auto& next_op = ops_.at(i + 1);
      successors_[op].insert(next_op);
      predecessors_[next_op].insert(op);
    } catch (...) {
      // do nothing
    }

    FilterVariables(op->inputs,
                    [&](ir::Node* var) { uses_[op].emplace(var->Name()); });

    FilterVariables(op->outputs,
                    [&](ir::Node* var) { defs_[op].emplace(var->Name()); });
  }
}

void ControlFlowGraph::LiveVariableAnalysis() {
  // NOTE(dzh): variable liveless analysis (a.k.a worklist algorithm)
  // compute the liveness of for each variable though worklist algorithm.
  // It iterates the operators from end to begin, compute the live in/live out
  // variable set for each op, then the diff between in/out will be used for
  // the variable reuse. For detail refer to
  // http://www.cs.cornell.edu/courses/cs4120/2013fa/lectures/lec26-fa13.pdf
  std::set<std::string> node_live_in;
  std::list<ir::Node*> worklist(ops_.rbegin(), ops_.rend());
  while (!worklist.empty()) {
    ir::Node* op = worklist.front();
    worklist.pop_front();
    node_live_in = std::move(live_in_[op]);
    for (auto& s : successors_[op]) {
      for (auto& var : live_in_[s]) {
        live_out_[op].insert(var);
      }
    }
    for (auto& var : uses_[op]) {
      live_in_[op].insert(var);
    }
    for (auto& var : live_out_[op]) {
      live_in_[op].insert(var);
    }
    for (auto& var : defs_[op]) {
      live_in_[op].erase(var);
    }
    if (live_in_[op] != node_live_in) {
      for (auto& pre : predecessors_[op]) {
        worklist.push_back(pre);
      }
    }
  }
}

void ControlFlowGraph::RenameVarInCFGGraph(const std::string& old_node,
                                           const std::string& new_node,
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

const std::set<std::string>& ControlFlowGraph::LiveIn(ir::Node* op) const {
  auto it = live_in_.find(op);
  PADDLE_ENFORCE(
      it != live_in_.end(),
      string::Sprintf("Expect %s in live_in, but Not Found.", op->Name()));
  return it->second;
}

const std::set<std::string>& ControlFlowGraph::LiveOut(ir::Node* op) const {
  auto it = live_out_.find(op);
  PADDLE_ENFORCE(
      it != live_out_.end(),
      string::Sprintf("Expect %s in live_out, but Not Found.", op->Name()));
  return it->second;
}
const std::set<std::string>& ControlFlowGraph::Def(ir::Node* op) const {
  auto it = defs_.find(op);
  PADDLE_ENFORCE(
      it != defs_.end(),
      string::Sprintf("Expect %s in defs, but Not Found.", op->Name()));
  return it->second;
}

const std::set<std::string>& ControlFlowGraph::Use(ir::Node* op) const {
  auto it = uses_.find(op);
  PADDLE_ENFORCE(
      it != uses_.end(),
      string::Sprintf("Expect %s in live_out, but Not Found.", op->Name()));
  return it->second;
}

const std::vector<ir::Node*>& ControlFlowGraph::Ops() const { return ops_; }

std::vector<ir::Node*>& ControlFlowGraph::Ops() { return ops_; }

ir::Node* ControlFlowGraph::GetNodeFromVarName(const std::string& name,
                                               ir::Node* op) const {
  // in ssa-graph, different version nodes have same name,
  // this function get the latest version var before target op
  // It may return nullptr, such as data node.
  ir::Node* found_node = nullptr;
  for (auto* node : ops_) {
    if (node == op) break;
    for (auto& output : node->outputs) {
      if (output->Name() == name) {
        found_node = output;
      }
    }
  }
  return found_node;
}

}  // namespace details
}  // namespace framework
}  // namespace paddle
