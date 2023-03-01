// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include <string>
#include "paddle/fluid/framework/ir/graph_pattern_detector.h"
#include "paddle/fluid/framework/ir/pass.h"
#include "paddle/fluid/framework/scope.h"

namespace phi {
class DenseTensor;
}  // namespace phi

namespace paddle {
namespace framework {
class Scope;
}  // namespace framework
}  // namespace paddle

namespace paddle {
namespace framework {
namespace ir {

class DeleteIsolatedNodePass : public Pass {
 protected:
  void ApplyImpl(ir::Graph* graph) const override;

 private:
  void CollectReservedPersistableNodeNames(
      ir::Graph* graph,
      std::unordered_set<std::string>* reserved_persistable_node_names) const;

  int RemoveIsolatedNodes(ir::Graph* graph,
                          const std::unordered_set<std::string>&
                              reserved_persistable_node_names) const;

  const std::map<std::string, std::string> control_flow_op_input_map_{
      {"while", "X"},
      {"conditional_block", "Input"},
  };
};

void DeleteIsolatedNodePass::ApplyImpl(ir::Graph* graph) const {
  PADDLE_ENFORCE_NOT_NULL(
      graph, platform::errors::PreconditionNotMet("graph should not be null."));
  PADDLE_ENFORCE(graph->IsMainGraph(),
                 platform::errors::PreconditionNotMet(
                     "Pass(apply in main graph) will delete isolated nodes in "
                     "all subgraphs. Do not apply pass in subgraph."));

  std::unordered_set<std::string> reserved_persistable_node_names;
  for (size_t i = 0; i < graph->SubGraphsSize(); i++) {
    CollectReservedPersistableNodeNames(graph->GetSubGraph(i),
                                        &reserved_persistable_node_names);
  }

  int delete_counts = 0;
  for (size_t i = 0; i < graph->SubGraphsSize(); i++) {
    delete_counts += RemoveIsolatedNodes(graph->GetSubGraph(i),
                                         reserved_persistable_node_names);
  }

  if (delete_counts > 0) {
    LOG(INFO) << "---  delete " << delete_counts << " isolated nodes";
  }
}

void DeleteIsolatedNodePass::CollectReservedPersistableNodeNames(
    ir::Graph* graph,
    std::unordered_set<std::string>* reserved_persistable_node_names) const {
  for (auto* node : graph->Nodes()) {
    if (!node->IsVar() || !node->Var()->Persistable()) continue;
    for (auto* out_node : node->outputs) {
      auto op_type = out_node->Op()->Type();
      if (control_flow_op_input_map_.count(op_type) == 0) {
        reserved_persistable_node_names->insert(node->Var()->Name());
        break;
      }
    }
  }
}

int DeleteIsolatedNodePass::RemoveIsolatedNodes(
    ir::Graph* graph,
    const std::unordered_set<std::string>& reserved_persistable_node_names)
    const {
  BlockDesc* block = nullptr;
  for (auto* node : graph->Nodes()) {
    if (node->IsOp()) {
      block = node->Op()->Block();
    }
  }
  Scope& scope = graph->Get<framework::Scope>("__param_scope__");

  // If graph has nodes to delete:
  // 1. Clear var_desc in block
  // 2. Clear tensor in variable
  // 3. Clear variable in scope
  int delete_node_counts = 0;
  std::unordered_set<const Node*> delete_nodes;
  const std::unordered_set<ir::Node*> nodes = graph->Nodes();
  for (auto* node : nodes) {
    if (!node->IsVar() || !node->Var()->Persistable()) continue;
    auto name = node->Var()->Name();
    if (reserved_persistable_node_names.count(name) > 0) continue;
    delete_nodes.insert(node);
    block->RemoveVar(name);
    auto* var = scope.FindVar(name);
    if (var != nullptr) {
      var->Clear();
      scope.EraseVars({name});
    }
    delete_node_counts++;
  }

  std::unordered_map<std::string, Node*> persistable_nodes_map;
  for (auto* node : nodes) {
    if (node->IsVar() && node->Var()->Persistable()) {
      persistable_nodes_map[node->Var()->Name()] = node;
    }
  }

  // Update node links and inputs map of ontrol flow ops.
  for (auto* node : nodes) {
    if (!node->IsOp()) continue;
    auto op_type = node->Op()->Type();
    if (control_flow_op_input_map_.count(op_type) == 0) continue;
    auto in_arg_name = control_flow_op_input_map_.at(op_type);
    auto in_names = node->Op()->Inputs().at(in_arg_name);
    std::unordered_set<std::string> in_names_set(in_names.begin(),
                                                 in_names.end());
    for (auto* delete_node : delete_nodes) {
      auto delete_node_name = delete_node->Var()->Name();
      if (in_names_set.count(delete_node_name) == 0) continue;
      in_names_set.erase(delete_node_name);
      std::string trans_node_name = delete_node_name + "_int16";
      if (persistable_nodes_map.count(trans_node_name) > 0) {
        std::string trans_node_max_name = delete_node_name + "_max";
        auto* trans_node = persistable_nodes_map.at(trans_node_name);
        auto* trans_max_node = persistable_nodes_map.at(trans_node_max_name);
        in_names_set.insert(trans_node_name);
        in_names_set.insert(trans_node_max_name);
        IR_NODE_LINK_TO(trans_node, node);
        IR_NODE_LINK_TO(trans_max_node, node);
      }
    }
    std::vector<std::string> new_in_names(in_names_set.begin(),
                                          in_names_set.end());
    node->Op()->SetInput(in_arg_name, new_in_names);
  }

  GraphSafeRemoveNodes(graph, delete_nodes);
  return delete_node_counts;
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(delete_isolated_node_pass,
              paddle::framework::ir::DeleteIsolatedNodePass);
