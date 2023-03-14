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

namespace paddle {
namespace framework {
namespace ir {

class Graph;

void InplaceOpVarPass::ApplyImpl(ir::Graph* graph) const {
  FusePassBase::Init("inplace_op_var", graph);
  int found_subgraph_count = 0;

  auto nodes = graph->Nodes();
  auto is_valid_reshape = [](Node* node) {
    // Some cases need to consider, please refer to
    // https://github.com/PaddlePaddle/Paddle/pull/49146
    if (node->IsOp() && node->Op()->Type() == "reshape2") {
      auto x_name = node->Op()->Input("X").front();
      for (auto* var_node : node->inputs) {
        if (var_node->Name() == x_name) {
          if (!var_node->Var()->Persistable() && var_node->outputs.size() == 1)
            return true;
        }
      }
    }
    return false;
  };

  // Record all reshape2 op's input name and output name in block 0.
  // If the name used in other block, we can not inplace reshape op.
  std::unordered_set<std::string> var_names, deny_var_names;
  for (auto* node : nodes) {
    if (is_valid_reshape(node)) {
      for (auto n : node->inputs) var_names.insert(n->Name());
      for (auto n : node->outputs) var_names.insert(n->Name());
    }
  }
  for (size_t i = 1; i < graph->SubGraphsSize(); ++i) {
    auto sub_graph = graph->GetSubGraph(i);
    for (auto* node : sub_graph->Nodes()) {
      if (node->IsOp()) {
        for (auto var_node : node->inputs) {
          if (var_names.count(var_node->Name()))
            deny_var_names.insert(var_node->Name());
        }
        for (auto var_node : node->outputs) {
          if (var_names.count(var_node->Name()))
            deny_var_names.insert(var_node->Name());
        }
      }
    }
  }

  // inplace all reshape op.
  auto topo_nodes = TopologySortOperations(*graph);
  for (auto* node : topo_nodes) {
    if (!is_valid_reshape(node)) continue;
    auto* op_node = node->Op();
    auto input_name = op_node->Input("X")[0];
    auto output_name = op_node->Output("Out")[0];
    if (deny_var_names.count(input_name) || deny_var_names.count(output_name)) {
      continue;
    }
    ++found_subgraph_count;
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
  }
  AddStatis(found_subgraph_count);
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(inplace_op_var_pass, paddle::framework::ir::InplaceOpVarPass);
REGISTER_PASS_CAPABILITY(inplace_op_var_pass)
    .AddCombination(
        paddle::framework::compatible::OpVersionComparatorCombination().EQ(
            "reshape2", 0));
