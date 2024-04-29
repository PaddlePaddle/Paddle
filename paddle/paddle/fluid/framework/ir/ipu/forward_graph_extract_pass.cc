// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/ir/ipu/forward_graph_extract_pass.h"

#include "paddle/fluid/framework/ir/pass_tester_helper.h"

namespace paddle {
namespace framework {
namespace ir {

void ForwardGraphExtractPass::ApplyImpl(ir::Graph* graph) const {
  VLOG(10) << "enter ForwardGraphExtractPass::ApplyImpl";

  std::unordered_map<OpRole, std::unordered_set<ir::Node*>> all_ops{
      {OpRole::kForward, {}},
      {OpRole::kBackward, {}},
      {OpRole::kOptimize, {}},
      {OpRole::kRPC, {}},
      {OpRole::kDist, {}},
      {OpRole::kLRSched, {}},
      {OpRole::kLoss, {}},
      {OpRole::kNotSpecified, {}}};
  for (auto* node : graph->Nodes()) {
    if (!node->IsOp()) {
      continue;
    }
    auto op_role = PADDLE_GET_MUTABLE(int, node->Op()->GetAttr("op_role"));
    if (op_role == static_cast<int>(OpRole::kForward)) {
      all_ops[OpRole::kForward].insert(node);
    } else if (op_role == static_cast<int>(OpRole::kBackward)) {
      all_ops[OpRole::kBackward].insert(node);
    } else if (op_role == static_cast<int>(OpRole::kOptimize)) {
      all_ops[OpRole::kOptimize].insert(node);
    } else if (op_role == static_cast<int>(OpRole::kRPC)) {
    } else if (op_role == static_cast<int>(OpRole::kDist)) {
    } else if (op_role == static_cast<int>(OpRole::kLRSched)) {
    } else if (op_role == static_cast<int>(OpRole::kLoss)) {
      all_ops[OpRole::kLoss].insert(node);
    } else if (op_role == static_cast<int>(OpRole::kNotSpecified)) {
      LOG(WARNING) << "Op: " << node->Name() << " OpRole is NotSpecified ";
    }
  }

  std::unordered_set<ir::Node*> forward_vars;
  std::unordered_set<ir::Node*> backward_vars;
  std::unordered_set<ir::Node*> control_vars;
  // forward_vars
  for (auto& nodes : std::array<std::unordered_set<ir::Node*>, 2>{
           all_ops[OpRole::kForward], all_ops[OpRole::kLoss]}) {
    for (auto* node : nodes) {
      for (auto* in_node : node->inputs) {
        forward_vars.insert(in_node);
      }
      for (auto* out_node : node->outputs) {
        forward_vars.insert(out_node);
      }
    }
  }
  // learning_rate var
  for (auto* node : all_ops[OpRole::kOptimize]) {
    if (node->Op()->Inputs().count("LearningRate") &&
        !node->Op()->Inputs().at("LearningRate").empty()) {
      auto lr_var_name = node->Op()->Inputs().at("LearningRate").front();
      for (auto* in_var : node->inputs) {
        if (in_var->Name() == lr_var_name) {
          VLOG(10) << "found LearningRate var: " << in_var->Name();
          forward_vars.insert(in_var);
        }
      }
    }
  }
  // control_vars &  backward_vars
  for (auto* node : graph->Nodes()) {
    if (!node->IsVar()) {
      continue;
    }
    if (node->IsCtrlVar()) {
      control_vars.insert(node);
    }
    for (auto* in_node : node->inputs) {
      if (all_ops[OpRole::kOptimize].count(in_node)) {
        backward_vars.insert(node);
      }
    }
  }
  // all removed node
  std::unordered_set<ir::Node*> rm_nodes;
  for (auto* node : graph->Nodes()) {
    if (backward_vars.count(node)) {
      rm_nodes.insert(node);
    } else if (control_vars.count(node)) {
      rm_nodes.insert(node);
    } else if (all_ops[OpRole::kBackward].count(node)) {
      rm_nodes.insert(node);
    } else if (all_ops[OpRole::kForward].count(node) == 0 &&
               all_ops[OpRole::kLoss].count(node) == 0 &&
               forward_vars.count(node) == 0) {
      rm_nodes.insert(node);
    } else if (node->Name() == "feed" || node->Name() == "fetch") {
      rm_nodes.insert(node);
    }
  }

  VLOG(10) << "Remove Node: ";
  for (auto* node : rm_nodes) {
    // rm node relations
    for (auto* node_in : node->inputs) {
      for (size_t i = 0; i < node_in->outputs.size(); ++i) {
        if (node_in->outputs[i] == node) {
          node_in->outputs.erase(node_in->outputs.begin() + i);
          break;
        }
      }
    }
    for (auto* node_out : node->outputs) {
      for (size_t i = 0; i < node_out->inputs.size(); ++i) {
        if (node_out->inputs[i] == node) {
          node_out->inputs.erase(node_out->inputs.begin() + i);
          break;
        }
      }
    }
    VLOG(10) << "\t" << node->Name();
    graph->RemoveNode(node);
  }

  VLOG(10) << "Post Graph: ";
  VLOG(10) << DebugString(graph);

  VLOG(10) << "leave ForwardGraphExtractPass::ApplyImpl";
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(forward_graph_extract_pass,
              paddle::framework::ir::ForwardGraphExtractPass);
