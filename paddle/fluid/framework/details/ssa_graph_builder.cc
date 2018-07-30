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
void SSAGraphBuilder::PolishGraphToSupportDataHazards(ir::Graph *graph) {
  for (auto &var_map : graph->Get<GraphVars>(kGraphVars)) {
    for (auto &name_pair : var_map) {
      if (name_pair.second.size() <= 1) {
        continue;
      }
      auto it_new = name_pair.second.rbegin();
      auto it_old = name_pair.second.rbegin();
      ++it_old;
      for (; it_old != name_pair.second.rend(); it_new = it_old, ++it_old) {
        OpHandleBase *write_op = (*it_new)->GeneratedOp();
        const auto &read_ops = (*it_old)->PendingOps();

        for (auto *read_op : read_ops) {
          // Manually add a dependency var from read_op to write_op;
          if (read_op == write_op) {
            // Read Write is the same op.
            continue;
          }
          bool has_dep = false;
          for (auto *r_out : read_op->Outputs()) {
            for (auto *w_in : write_op->Inputs()) {
              if (r_out->Node() == w_in->Node()) {
                has_dep = true;
                break;
              }
            }
          }
          if (has_dep) continue;

          auto *dep_var = new DummyVarHandle(graph->CreateControlDepVar());
          read_op->AddOutput(dep_var);
          write_op->AddInput(dep_var);
          graph->Get<GraphDepVars>(kGraphDepVars).emplace(dep_var);
        }
      }
    }
  }
}

VarHandle *SSAGraphBuilder::CreateOrGetLatestVarHandle(
    ir::Graph *graph, ir::Node *node, const platform::Place &place,
    size_t place_offset) {
  auto &var_holders = graph->Get<GraphVars>(kGraphVars)[place_offset];
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
  auto &vars =
      graph->Get<GraphVars>(kGraphVars)[place_offset][new_node->Name()];
  size_t version = vars.size();
  auto var =
      new VarHandle(new_node, version, place_offset, new_node->Name(), place);
  vars.emplace_back(var);
  op_handle->AddOutput(var);
}

void SSAGraphBuilder::AddOutputToLeafOps(ir::Graph *graph) {
  for (auto &op : graph->Get<GraphOps>(kGraphOps)) {
    if (!op->Outputs().empty()) {
      continue;
    }
    auto *dummy_leaf = new DummyVarHandle(graph->CreateControlDepVar());
    graph->Get<GraphDepVars>(kGraphDepVars).emplace(dummy_leaf);
    op->AddOutput(dummy_leaf);
  }
}
}  // namespace details
}  // namespace framework
}  // namespace paddle
