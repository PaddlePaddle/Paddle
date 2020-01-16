/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include "paddle/fluid/framework/ir/fusion_group/operation.h"
#include "paddle/fluid/framework/ir/node.h"
#include "paddle/fluid/framework/ir/pass_tester_helper.h"

namespace paddle {
namespace framework {
namespace ir {
namespace fusion_group {

struct SubGraph {
  int type{-1};
  std::string func_name;
  bool save_intermediate_out{false};

  SubGraph() = default;
  SubGraph(int t, std::string f, bool s, const std::unordered_set<Node*>& n)
      : type(t), func_name(f), save_intermediate_out(s), nodes_set(n) {}

  bool IsEmpty() { return nodes_set.empty(); }

  const std::unordered_set<Node*>& Nodes() const { return nodes_set; }

  const std::vector<Node*>& SortedNodes() {
    if (!is_sorted) {
      Sort();
    }
    return sorted_nodes;
  }

  size_t GetNumNodes() { return nodes_set.size(); }

  bool Has(Node* n) { return nodes_set.find(n) != nodes_set.end(); }

  void Insert(Node* n) {
    if (nodes_set.find(n) == nodes_set.end()) {
      VLOG(5) << "Insert " << n->Name() << " to subgraph " << this;
      nodes_set.insert(n);
      is_sorted = false;
    }
  }

  int GetNumOperations() {
    int num_operations = 0;
    for (auto* n : nodes_set) {
      if (n && n->IsOp() && n->Op()) {
        num_operations++;
      }
    }
    return num_operations;
  }

  std::vector<Node*> GetInputVarNodes() {
    // The order of input nodes should be consistent anywhere.
    std::vector<Node*> input_vars;
    for (auto* n : SortedNodes()) {
      if (n && n->IsVar() && n->Var()) {
        bool is_found = true;
        // When the inputs size is 0, it is also considered the input var of
        // subgraph.
        if (n->inputs.size() == 0U) {
          is_found = false;
        }
        // Normally a var node has only one input op node.
        for (auto* in : n->inputs) {
          if (!Has(in)) {
            is_found = false;
          }
        }
        if (!is_found) {
          input_vars.push_back(n);
        }
      }
    }
    return input_vars;
  }

  std::vector<Node*> GetOutputVarNodes() {
    // The order of output nodes should be consistant anywhere..
    std::vector<Node*> output_vars;
    for (auto* n : SortedNodes()) {
      if (n && n->IsVar() && n->Var()) {
        if (save_intermediate_out) {
          // If the var_node is the output of some op_node in the subgraph, it
          // is considered the output var node of the subgraph.
          bool is_found = false;
          for (auto* in : n->inputs) {
            if (Has(in)) {
              is_found = true;
            }
          }
          if (is_found) {
            output_vars.push_back(n);
          }
        } else {
          // If one of the var_node's outputs is the input of some operator
          // outside the subgraph, it is considered the output var node of the
          // subgraph.
          bool is_found = true;
          if (n->outputs.size() == 0U) {
            is_found = false;
          }
          for (auto* out : n->outputs) {
            if (!Has(out)) {
              is_found = false;
            }
          }
          if (!is_found) {
            output_vars.push_back(n);
          }
        }
      }
    }
    return output_vars;
  }

 private:
  int FindIndexInSortedNodes(Node* n) {
    for (size_t i = 0; i < sorted_nodes.size(); ++i) {
      if (n == sorted_nodes[i]) {
        return static_cast<int>(i);
      }
    }
    return -1;
  }

  void SortVarsBasedOnSortedOps() {
    // Insert var nodes to sorted_nodes.
    std::unordered_map<std::string, Node*> sorted_vars;
    for (auto* n : nodes_set) {
      if (n && n->IsVar() && n->Var()) {
        int from = 0;
        int to = sorted_nodes.size();

        for (auto* in : n->inputs) {
          if (in && in->IsOp() && in->Op()) {
            int index = FindIndexInSortedNodes(in);
            // Insert after input op node
            if (index >= 0) {
              from = index + 1 > from ? index + 1 : from;
            }
          }
        }

        for (auto* out : n->outputs) {
          if (out && out->IsOp() && out->Op()) {
            int index = FindIndexInSortedNodes(out);
            // Insert before output op node
            if (index >= 0) {
              to = index < to ? index : to;
            }
          }
        }

        if (from > to) {
          LOG(INFO) << "subgraph: {\n" << DebugString(Nodes()) << "}\n";
          LOG(INFO) << "sorted nodes: {\n"
                    << DebugString(sorted_nodes) << "}\n";
        }
        PADDLE_ENFORCE_LE(from, to, "Range [%d, %d] is invalid.", from, to);
        sorted_nodes.insert(sorted_nodes.begin() + to, n);
        sorted_vars[n->Name()] = n;
      }
    }
  }

  std::vector<Node*> SortedOps() {
    Node* start_op_n = nullptr;
    std::unordered_set<Node*> ops;
    for (auto* op_n : nodes_set) {
      if (op_n && op_n->IsOp() && op_n->Op()) {
        // Initialize ops to all ops in the subgraph.
        ops.insert(op_n);

        if (!start_op_n) {
          // Find start op node whose inputs are produced outside the subgraph.
          bool is_found = false;
          for (auto* prev_op_n : GetPrevOpNodes(op_n)) {
            if (Has(prev_op_n)) {
              is_found = true;
              break;
            }
          }
          if (!is_found) {
            start_op_n = op_n;
          }
        }
      }
    }

    std::vector<Node*> sorted_ops;
    sorted_ops.push_back(start_op_n);
    ops.erase(start_op_n);
    while (ops.size() > 0U) {
      std::unordered_set<Node*> erased_ops;
      for (auto* op_n : ops) {
        bool found_connected_ops = false;
        int from = 1;
        int to = sorted_ops.size();
        std::unordered_set<Node*> prev_op_nodes = GetPrevOpNodes(op_n);
        std::unordered_set<Node*> next_op_nodes = GetNextOpNodes(op_n);
        for (int i = sorted_ops.size(); i >= 0; --i) {
          if (prev_op_nodes.find(sorted_ops[i]) != prev_op_nodes.end()) {
            // Insert after i (i + 1)
            found_connected_ops = true;
            from = (i + 1 > from) ? i + 1 : from;
          }
          if (next_op_nodes.find(sorted_ops[i]) != next_op_nodes.end()) {
            // Insert before i
            found_connected_ops = true;
            to = (i < to) ? i : to;
          }
        }
        if (found_connected_ops) {
          if (from > to) {
            LOG(INFO) << "subgraph: {\n" << DebugString(Nodes()) << "}\n";
          }
          PADDLE_ENFORCE_LE(from, to, "Range [%d, %d] is invalid.", from, to);
          sorted_ops.insert(sorted_ops.begin() + to, op_n);
          erased_ops.insert(op_n);
        }
      }
      PADDLE_ENFORCE_GT(erased_ops.size(), 0U);
      for (auto* op_n : erased_ops) {
        ops.erase(op_n);
      }
    }
    return sorted_ops;
  }

  std::unordered_set<Node*> GetPrevOpNodes(Node* op_n) {
    PADDLE_ENFORCE_EQ(op_n && op_n->IsOp() && op_n->Op(), true,
                      "Node %p is not a op node.", op_n);

    std::unordered_set<Node*> prev_op_nodes;
    for (auto* in_var : op_n->inputs) {
      if (in_var && in_var->IsVar() && in_var->Var()) {
        for (auto* prev_op_n : in_var->inputs) {
          if (prev_op_n && prev_op_n->IsOp() && prev_op_n->Op()) {
            prev_op_nodes.insert(prev_op_n);
          }
        }
      }
    }
    return prev_op_nodes;
  }

  std::unordered_set<Node*> GetNextOpNodes(Node* op_n) {
    PADDLE_ENFORCE_EQ(op_n && op_n->IsOp() && op_n->Op(), true,
                      "Node %p is not a op node.", op_n);

    std::unordered_set<Node*> next_op_nodes;
    for (auto* out_var : op_n->outputs) {
      if (out_var && out_var->IsVar() && out_var->Var()) {
        for (auto* next_op_n : out_var->outputs) {
          if (next_op_n && next_op_n->IsOp() && next_op_n->Op()) {
            next_op_nodes.insert(next_op_n);
          }
        }
      }
    }
    return next_op_nodes;
  }

  void Sort() {
    if (!is_sorted) {
      sorted_nodes = SortedOps();
      SortVarsBasedOnSortedOps();
    }
    is_sorted = true;
  }

 private:
  std::unordered_set<Node*> nodes_set;
  bool is_sorted{false};
  std::vector<Node*> sorted_nodes;
};

}  // namespace fusion_group
}  // namespace ir
}  // namespace framework
}  // namespace paddle
