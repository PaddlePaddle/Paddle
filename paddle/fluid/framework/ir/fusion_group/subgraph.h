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
  std::unordered_set<Node*> nodes_set;

  bool IsEmpty() { return nodes_set.empty(); }

  size_t GetNumNodes() { return nodes_set.size(); }

  int GetNumOperations() {
    int num_operations = 0;
    for (auto* n : nodes_set) {
      if (n && n->IsOp() && n->Op()) {
        num_operations++;
      }
    }
    return num_operations;
  }

  std::vector<Node*> GetInputVarNodes() const {
    // The order of input nodes should be consistent with that of the generated
    // code.
    std::vector<Node*> input_vars;
    for (auto* n : nodes_set) {
      if (n && n->IsVar() && n->Var()) {
        bool is_found = true;
        // When the inputs size is 0, it is also considered the input var of
        // subgraph.
        if (n->inputs.size() == 0U) {
          is_found = false;
        }
        // Normally a var node has only one input op node.
        for (auto* in : n->inputs) {
          if (nodes_set.find(in) == nodes_set.end()) {
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

  std::vector<Node*> GetOutputVarNodes() const {
    // The order of output nodes should be consistant with that of the generated
    // code.
    std::vector<Node*> output_vars;
    for (auto* n : nodes_set) {
      if (n && n->IsVar() && n->Var()) {
        bool is_found = true;
        if (n->outputs.size() == 0U) {
          is_found = false;
        }
        for (auto* out : n->outputs) {
          if (nodes_set.find(out) == nodes_set.end()) {
            is_found = false;
          }
        }
        if (!is_found) {
          output_vars.push_back(n);
        }
      }
    }
    return output_vars;
  }
};

}  // namespace fusion_group
}  // namespace ir
}  // namespace framework
}  // namespace paddle
