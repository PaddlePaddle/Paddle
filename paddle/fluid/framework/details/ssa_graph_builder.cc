//   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
#include "paddle/fluid/framework/details/ssa_graph_builder.h"
#include <utility>

namespace paddle {
namespace framework {
namespace details {
VarHandle *SSAGraphBuilder::CreateOrGetLatestVarHandle(
    ir::Graph *graph, ir::Node *node, const platform::Place &place,
    size_t place_offset) {
  auto &var_holders = graph->Get<GraphVars>("vars")[place_offset];
  auto &var_holder = var_holders[node->Name()];
  VarHandle *var = nullptr;
  if (var_holder.empty()) {
    if (node->Var()) {
      var = new VarHandle(graph->CreateVarNode(node->Var()), 0, place_offset,
                          node->Name(), place);
    } else {
      var = new VarHandle(
          graph->CreateEmptyNode(node->Name(), ir::Node::Type::kVariable), 0,
          place_offset, node->Name(), place);
    }
    var_holder.emplace_back(var);
  } else {
    var = var_holder.rbegin()->get();
  }
  return var;
}

void SSAGraphBuilder::CreateOpOutput(ir::Graph *graph, OpHandleBase *op_handle,
                                     ir::Node *new_node,
                                     const platform::Place &place,
                                     size_t place_offset) {
  auto &vars = graph->Get<GraphVars>("vars")[place_offset][new_node->Name()];
  size_t version = vars.size();
  auto var =
      new VarHandle(new_node, version, place_offset, new_node->Name(), place);
  vars.emplace_back(var);
  op_handle->AddOutput(var);
}

void SSAGraphBuilder::AddOutputToLeafOps(ir::Graph *graph) {
  for (auto &op : graph->Get<GraphOps>("ops")) {
    if (!op->Outputs().empty()) {
      continue;
    }
    auto *dummy_leaf = new DummyVarHandle(
        graph->CreateEmptyNode("dummy", ir::Node::Type::kVariable));
    graph->Get<GraphDepVars>("dep_vars").emplace(dummy_leaf);
    op->AddOutput(dummy_leaf);
  }
}
}  // namespace details
}  // namespace framework
}  // namespace paddle
