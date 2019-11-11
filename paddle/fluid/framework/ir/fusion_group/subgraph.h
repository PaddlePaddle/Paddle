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
#include <unordered_set>
#include <vector>
#include "paddle/fluid/framework/ir/node.h"

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
    // The order of input nodes should be consistent with that of the generated
    // code.
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
    // The order of output nodes should be consistant with that of the generated
    // code.
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

  void InsertVarInOrder(Node* n) {
    PADDLE_ENFORCE_NOT_NULL(n, "Node should not be null.");
    PADDLE_ENFORCE_EQ(n->IsVar() && n->Var(), true,
                      "Node %s is not a var node.", n->Name());

    int from = 0;
    int to = sorted_nodes.size();

    for (auto* in : n->inputs) {
      if (in && in->IsOp() && in->Op()) {
        int index = FindIndexInSortedNodes(in);
        // Insert after input op node
        if (index >= 0 && index >= from) {
          from = index + 1;
        }
      }
    }

    for (auto* out : n->outputs) {
      if (out && out->IsOp() && out->Op()) {
        int index = FindIndexInSortedNodes(out);
        // Insert before output op node
        if (index >= 0 && index < to) {
          to = index;
        }
        //      auto* out_op = out->Op();
        //      bool is_found = false;
        //      for (size_t i = 0; i < out_op->InputArgumentNames().size(); ++i)
        //      {
        //        auto arg_name = out_op->InputArgumentNames()[i];
        //        if (out_op->Input(arg_name) == n->Name()) {
        //          is_found = true;
        //        }
        //      }
      }
    }

    PADDLE_ENFORCE_LE(from, to, "Range [%d, %d] is invalid");
    LOG(INFO) << "Var " << n->Name() << ", insert from:" << from
              << ", to:" << to;
    sorted_nodes.insert(sorted_nodes.begin() + from, n);
  }

  void InsertOpInOrder(Node* n) {
    PADDLE_ENFORCE_NOT_NULL(n, "Node should not be null.");
    PADDLE_ENFORCE_EQ(n->IsOp() && n->Op(), true, "Node %p is not a op node.",
                      n);

    int from = 0;
    int to = sorted_nodes.size();

    for (auto* in : n->inputs) {
      if (in && in->IsVar() && in->Var()) {
        int index = FindIndexInSortedNodes(in);
        // Insert after input var node
        if (index >= 0 && index >= from) {
          from = index + 1;
        }
      }
    }

    for (auto* out : n->outputs) {
      if (out && out->IsVar() && out->Var()) {
        int index = FindIndexInSortedNodes(out);
        // Insert before output var node
        if (index >= 0 && index < to) {
          to = index;
        }
      }
    }

    PADDLE_ENFORCE_LE(from, to, "Range [%d, %d] is invalid");
    LOG(INFO) << "Op " << n->Op()->Type() << ", insert from:" << from
              << ", to:" << to;
    sorted_nodes.insert(sorted_nodes.begin() + from, n);
  }

  void Sort() {
    if (!is_sorted) {
      sorted_nodes.clear();
      for (auto* n : nodes_set) {
        if (sorted_nodes.size() == 0U) {
          sorted_nodes.push_back(n);
        } else {
          if (n && n->IsVar() && n->Var()) {
            InsertVarInOrder(n);
          } else if (n && n->IsOp() && n->Op()) {
            InsertOpInOrder(n);
          }
        }
      }
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
