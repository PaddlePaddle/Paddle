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
#include "paddle/fluid/framework/ir/graph.h"
#include "paddle/fluid/framework/ir/graph_traits.h"
#include "paddle/fluid/framework/ir/node.h"
#include "paddle/fluid/framework/ir/subgraph_detector.h"

namespace paddle {
namespace framework {
namespace ir {
namespace fusion_group {

class SubGraph {
 public:
  SubGraph() = default;
  explicit SubGraph(int type) : type_(type) {}
  SubGraph(int type, std::string func_name, bool save_intermediate_out,
           const std::unordered_set<Node*>& nodes_set)
      : type_(type),
        func_name_(func_name),
        save_intermediate_out_(save_intermediate_out) {
    for (auto* n : nodes_set) {
      nodes_set_.insert(n);
      if (n && n->IsOp() && n->Op()) {
        // If the node is an op node, then add its input/output var nodes
        //  into the subgraph.
        for (auto* in : n->inputs) {
          nodes_set_.insert(in);
        }
        for (auto* out : n->outputs) {
          nodes_set_.insert(out);
        }
      }
    }
  }

  bool IsValid(int min_subgraph_size) {
    int num_operations = GetNumOperations();
    if (num_operations < min_subgraph_size) {
      VLOG(2) << "There are only " << num_operations
              << " operations in the subgraph. Expected at least "
              << min_subgraph_size;
      return false;
    }

    return true;
  }

  int GetType() const { return type_; }

  void SetFuncName(std::string func_name) { func_name_ = func_name; }
  std::string GetFuncName() const { return func_name_; }

  bool SaveIntermediateOut() const { return save_intermediate_out_; }

  const std::unordered_set<Node*>& Nodes() const { return nodes_set_; }
  const std::vector<Node*>& SortedNodes() {
    if (!is_sorted_) {
      TopologicalSort();
    }
    return sorted_nodes_;
  }

  size_t GetNumNodes() { return nodes_set_.size(); }

  bool Has(Node* n) { return nodes_set_.find(n) != nodes_set_.end(); }

  int GetNumOperations() {
    int num_operations = 0;
    for (auto* n : nodes_set_) {
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

  std::vector<Node*> GetOutputVarNodes(bool with_intermediate_out) {
    // The order of output nodes should be consistant anywhere..
    std::vector<Node*> output_vars;
    for (auto* n : SortedNodes()) {
      if (IsOutputOfInternalOp(n)) {
        // If the var_node is the output of some op_node in the subgraph, it
        // is considered the output var node of the subgraph.
        if (with_intermediate_out) {
          output_vars.push_back(n);
        } else {
          if (n->outputs.empty() || IsInputOfExternalOp(n)) {
            output_vars.push_back(n);
          }
        }
      }
    }
    return output_vars;
  }

  std::vector<Node*> GetIntermediateOutVarNodes() {
    // Intermediate output var nodes: the output of some op_node in the
    // subgraph, but not referenced outside the subgraph.
    std::vector<Node*> intermediate_out_vars;
    for (auto* n : SortedNodes()) {
      if (IsOutputOfInternalOp(n) && IsInputOfInternalOp(n) &&
          !IsInputOfExternalOp(n)) {
        // When the outputs size is 0, it is also considered a intermidiate
        // output. It maybe an unused output or the fetching vars, so that we
        // cannot eleiminate it directly here.
        intermediate_out_vars.push_back(n);
      }
    }
    return intermediate_out_vars;
  }

  std::unordered_set<Node*> GetIntermediateOutVarNodesSet() {
    std::vector<Node*> intermediate_out_vars = GetIntermediateOutVarNodes();
    return std::unordered_set<Node*>(intermediate_out_vars.begin(),
                                     intermediate_out_vars.end());
  }

 private:
  bool IsInputOfInternalOp(Node* n) {
    bool is_input_of_internal_op = false;
    if (Has(n) && n && n->IsVar() && n->Var()) {
      for (auto* out : n->outputs) {
        if (Has(out)) {
          is_input_of_internal_op = true;
          break;
        }
      }
    }
    return is_input_of_internal_op;
  }

  bool IsInputOfExternalOp(Node* n) {
    // If n is the input any one node outside the subgraph.
    bool is_input_of_external_op = false;
    if (Has(n) && n && n->IsVar() && n->Var()) {
      for (auto* out : n->outputs) {
        if (!Has(out)) {
          is_input_of_external_op = true;
          break;
        }
      }
    }
    return is_input_of_external_op;
  }

  bool IsOutputOfInternalOp(Node* n) {
    bool is_output_of_internal_op = false;
    if (Has(n) && n && n->IsVar() && n->Var()) {
      for (auto* in : n->inputs) {
        if (Has(in)) {
          is_output_of_internal_op = true;
          break;
        }
      }
    }
    return is_output_of_internal_op;
  }

  void TopologicalSort() {
    if (!is_sorted_) {
      std::unordered_map<Node*, std::vector<Node*>> inputs_map;
      std::unordered_map<Node*, std::vector<Node*>> outputs_map;
      for (auto* n : nodes_set_) {
        inputs_map[n] = n->inputs;
        outputs_map[n] = n->outputs;
      }

      for (auto* n : nodes_set_) {
        if (n && ((n->IsVar() && n->Var()) || n->IsCtrlVar())) {
          // Set the input of subgraph's input var node to null.
          std::vector<Node*> inputs;
          for (auto* in : n->inputs) {
            if (Has(in)) {
              inputs.push_back(in);
            }
          }
          // Set the output of subgraph's output var node to null.
          std::vector<Node*> outputs;
          for (auto* out : n->outputs) {
            if (Has(out)) {
              outputs.push_back(out);
            }
          }
          n->inputs = inputs;
          n->outputs = outputs;
        }
      }
      // Collect the start points of the subgraph.
      std::vector<Node*> start_points;
      for (auto* n : nodes_set_) {
        if (n->inputs.empty()) {
          start_points.push_back(n);
        }
      }
      // Sort the subgraph.
      NodesTSIterator x(start_points);
      for (auto& n : iterator_range<NodesTSIterator>(
               NodesTSIterator(start_points), NodesTSIterator())) {
        sorted_nodes_.push_back(&n);
      }
      // Reset the inputs, outputs.
      for (auto* n : nodes_set_) {
        n->inputs = inputs_map[n];
        n->outputs = outputs_map[n];
      }
    }
    is_sorted_ = true;
  }

 private:
  int type_{-1};
  std::string data_type_;
  std::string func_name_;
  bool save_intermediate_out_{true};

  std::unordered_set<Node*> nodes_set_;
  bool is_sorted_{false};
  std::vector<Node*> sorted_nodes_;
};

}  // namespace fusion_group
}  // namespace ir
}  // namespace framework
}  // namespace paddle
