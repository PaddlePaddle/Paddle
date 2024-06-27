// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/ir/inplace_op_var_pass.h"

#include "paddle/fluid/framework/ir/graph_helper.h"
#include "paddle/fluid/framework/ir/graph_pattern_detector.h"
#include "paddle/fluid/framework/ir/node.h"
#include "paddle/fluid/framework/op_version_registry.h"

namespace paddle::framework::ir {

class Graph;

bool InplaceOpVarPass::IsValidInplaceOp(
    Node* node, const std::unordered_set<std::string>& deny_var_names) const {
  if (!node->IsOp() || inplace_ops_.count(node->Op()->Type()) == 0)
    return false;

  // in_var_node should only has one out_op_node
  auto x_name = node->Op()->Input("X").front();
  for (auto* var_node : node->inputs) {
    if (var_node->Name() != x_name) continue;
    if (var_node->Var()->Persistable() || var_node->outputs.size() != 1)
      return false;
    // The op type in front of in_var_node should not be feed.
    for (auto* pre_op : var_node->inputs) {
      if (pre_op->Op()->Type() == "feed") {
        return false;
      }
    }
  }

  // in/out_var_node should be not used in multi graphs.
  auto out_name = node->Op()->Output("Out").front();
  if (deny_var_names.count(x_name) > 0 || deny_var_names.count(out_name) > 0)
    return false;

  return true;
}

int InplaceOpVarPass::ApplyImpl(
    ir::Graph* graph,
    const std::unordered_set<std::string>& deny_var_names) const {
  int found_subgraph_count = 0;
  // inplace all reshape op.
  auto topo_nodes = TopologySortOperations(*graph);
  for (auto* node : topo_nodes) {
    if (!IsValidInplaceOp(node, deny_var_names)) continue;
    auto* op_node = node->Op();
    auto input_name = op_node->Input("X")[0];
    auto output_name = op_node->Output("Out")[0];
    for (auto* out_var : node->outputs) {
      if (out_var->Name() == output_name) {
        out_var->RenameVar(input_name);
        for (auto* next_op : out_var->outputs) {
          next_op->Op()->RenameInput(output_name, input_name);
          next_op->Op()->Flush();
        }
      }
    }
    op_node->RenameOutput(output_name, input_name);
    op_node->Flush();
    found_subgraph_count++;
  }
  return found_subgraph_count;
}

std::vector<std::string> InplaceOpVarPass::GetControlFlowVarNames(
    ir::Graph* graph) const {
  std::vector<std::string> control_flow_var_names;
  for (auto* node : graph->Nodes()) {
    if (!node->IsOp() || control_flow_ops_.count(node->Op()->Type()) == 0)
      continue;
    for (auto const& in_names : node->Op()->Inputs()) {
      auto var_names = in_names.second;
      control_flow_var_names.insert(
          control_flow_var_names.end(), var_names.begin(), var_names.end());
    }
    for (auto const& out_names : node->Op()->Outputs()) {
      auto var_names = out_names.second;
      control_flow_var_names.insert(
          control_flow_var_names.end(), var_names.begin(), var_names.end());
    }
  }
  return control_flow_var_names;
}

void InplaceOpVarPass::ApplyImpl(ir::Graph* graph) const {
  FusePassBase::Init("inplace_op_var", graph);
  if (!graph->IsMainGraph()) {
    VLOG(3) << "Pass(apply in main graph) will work on all subgraphs.";
    return;
  }

  std::unordered_set<std::string> deny_var_names;
  for (size_t i = 0; i < graph->SubGraphsSize(); i++) {
    auto control_flow_var_names = GetControlFlowVarNames(graph->GetSubGraph(i));
    deny_var_names.insert(control_flow_var_names.begin(),
                          control_flow_var_names.end());
  }

  int found_subgraph_count = 0;
  for (size_t i = 0; i < graph->SubGraphsSize(); i++) {
    found_subgraph_count += ApplyImpl(graph->GetSubGraph(i), deny_var_names);
  }
  AddStatis(found_subgraph_count);
}

}  // namespace paddle::framework::ir

REGISTER_PASS(inplace_op_var_pass, paddle::framework::ir::InplaceOpVarPass);
REGISTER_PASS_CAPABILITY(inplace_op_var_pass)
    .AddCombination(
        paddle::framework::compatible::OpVersionComparatorCombination().EQ(
            "reshape2", 0))
    .AddCombination(
        paddle::framework::compatible::OpVersionComparatorCombination().EQ(
            "unsqueeze2", 0))
    .AddCombination(
        paddle::framework::compatible::OpVersionComparatorCombination().EQ(
            "unsqueeze", 0))
    .AddCombination(
        paddle::framework::compatible::OpVersionComparatorCombination().EQ(
            "squeeze2", 0))
    .AddCombination(
        paddle::framework::compatible::OpVersionComparatorCombination().EQ(
            "squeeze", 0));
