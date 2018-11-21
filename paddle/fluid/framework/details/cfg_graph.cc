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
  ops_ = ir::SortOperationsInSequence(graph);
  for (auto& op : ops_) {
    PADDLE_ENFORCE(op->IsOp(), "expect operator in graph.");
    successors_[op] = std::list<ir::Node*>();
    predecessors_[op] = std::list<ir::Node*>();
    live_in_[op] = std::unordered_set<std::string>();
    live_out_[op] = std::unordered_set<std::string>();
    uses_[op] = std::unordered_set<std::string>();
    defs_[op] = std::unordered_set<std::string>();
  }
  ConnectNodes();
}

void ControlFlowGraph::ConnectNodes() {
  // auto is_same_node = [](ir::Node* lhs, ir::Node* rhs) {
  //   return lhs->Name() == rhs->Name();
  // };
  // for (ir::Node* op: ops_) {
  //   for (auto& input_var : op->inputs) {
  //     if (!input_var->inputs.empty()){
  //       PADDLE_ENFORCE(input_var->inputs.size() == 1 &&
  //       input_var->inputs[0]->IsOp(),
  //                      "Preceding Op Node of Var Node must be unique");
  //       if (input_var->inputs[0]->Op() != nullptr) {
  //         predecessors_[op].push_back(input_var->inputs[0]);
  //       }
  //     }
  //     if (input_var->IsVar() && !input_var->IsCtrlVar()) {
  //       uses_[op].insert(input_var);
  //     }
  //   }
  //   for (auto& output_var : op->outputs) {
  //     if (!output_var->outputs.empty()) {
  //       // output var may be used by many op
  //       for (auto* op_use_output_var : output_var->outputs){
  //         if (op_use_output_var->Op() != nullptr) {
  //           successors_[op].push_back(op_use_output_var);
  //         }
  //       }
  //     }
  //     if (output_var->IsVar() && !output_var->IsCtrlVar()) {
  //       defs_[op].insert(output_var);
  //     }
  //   }
  //   // avoid the op use same op output
  //   std::unique(predecessors_[op].begin(), predecessors_[op].end(),
  //   is_same_node);
  //   std::unique(successors_[op].begin(), successors_[op].end(),
  //   is_same_node);
  // }

  for (size_t i = 0; i < ops_.size(); ++i) {
    auto& op = ops_[i];
    if (i < ops_.size() - 1) {
      auto& next_op = ops_[i + 1];
      successors_[op].push_back(next_op);
      predecessors_[next_op].push_back(op);
    }
    for (auto& input_var : op->inputs) {
      if (input_var->IsVar() && !input_var->IsCtrlVar()) {
        uses_[op].insert(input_var->Name());
      }
    }
    for (auto& output_var : op->outputs) {
      if (output_var->IsVar() && !output_var->IsCtrlVar()) {
        defs_[op].insert(output_var->Name());
      }
    }
  }
}

void ControlFlowGraph::LiveVariableAnalysis() {
  // NOTE(dzh): variable liveless analysis (a.k.a worklist algorithm)
  // compute the liveness of for each variable though worklist algorithm.
  // It iterates the operators from end to begin, compute the live in/live out
  // variable set for each op, then the diff between in/out will be used for
  // the variable reuse. For detail refer to
  // http://www.cs.cornell.edu/courses/cs4120/2013fa/lectures/lec26-fa13.pdf
  auto set_equal = [](const std::unordered_set<std::string>& lhs,
                      const std::unordered_set<std::string>& rhs) {
    if (lhs.size() != rhs.size()) return false;
    for (auto& item : lhs) {
      if (rhs.find(item) == rhs.end()) {
        return false;
      }
    }
    return true;
  };

  std::unordered_set<std::string> node_live_in;
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
    for (auto& var : live_out_[op]) {
      live_in_[op].insert(var);
    }
    for (auto& var : uses_[op]) {
      live_in_[op].insert(var);
    }
    for (auto& var : defs_[op]) {
      live_in_[op].erase(var);
    }
    if (!set_equal(live_in_[op], node_live_in)) {
      for (auto& pre : predecessors_[op]) {
        worklist.push_back(pre);
      }
    }
  }

  // for (size_t i = 0; i < ops_.size(); ++i) {
  //   auto& op = ops_[i];
  //   VLOG(3) << i << " "<< op->Name();
  //   VLOG(3) << "live in " << this->LiveIn(op).size();
  //   VLOG(3) << "live out " << this->LiveOut(op).size();
  //   VLOG(3) << "use " << this->Use(op).size();
  //   VLOG(3) << "def " << this->Def(op).size();
  //   VLOG(3) << "successors" << successors_[op].size();
  //   VLOG(3) << "predecessors" << predecessors_[op].size();
  // }
}

void ControlFlowGraph::UpdateGraph(const std::string& old_node,
                                   const std::string& new_node, int begin_idx) {
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

const std::unordered_set<std::string>& ControlFlowGraph::LiveIn(
    ir::Node* op) const {
  auto it = live_in_.find(op);
  PADDLE_ENFORCE(
      it != live_in_.end(),
      string::Sprintf("Expect %s in live_in, but Not Found.", op->Name()));
  return it->second;
}

const std::unordered_set<std::string>& ControlFlowGraph::LiveOut(
    ir::Node* op) const {
  auto it = live_out_.find(op);
  PADDLE_ENFORCE(
      it != live_out_.end(),
      string::Sprintf("Expect %s in live_out, but Not Found.", op->Name()));
  return it->second;
}
const std::unordered_set<std::string>& ControlFlowGraph::Def(
    ir::Node* op) const {
  auto it = defs_.find(op);
  PADDLE_ENFORCE(
      it != defs_.end(),
      string::Sprintf("Expect %s in defs, but Not Found.", op->Name()));
  return it->second;
}

const std::unordered_set<std::string>& ControlFlowGraph::Use(
    ir::Node* op) const {
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
  // in ssa-graph, different version nodes have same name, this get the latest
  // version var
  // before op.
  ir::Node* found_node = nullptr;
  for (auto* node : ops_) {
    if (node == op) break;
    for (auto& output : node->outputs) {
      if (output->Name() == name) {
        found_node = output;
      }
    }
  }
  // PADDLE_ENFORCE(found_node != nullptr, string::Sprintf("Not found %s before
  // op %s", name, op->Name()));
  return found_node;
}

}  // namespace details
}  // namespace framework
}  // namespace paddle
