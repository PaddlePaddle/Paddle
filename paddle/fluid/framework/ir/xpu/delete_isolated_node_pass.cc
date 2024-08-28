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

#include "glog/logging.h"

#include "paddle/fluid/framework/ir/graph_pattern_detector.h"
#include "paddle/fluid/framework/ir/pass.h"
#include "paddle/fluid/framework/ir/xpu/pass_utils.h"
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
  void ApplyImpl(Graph* graph) const override;

 private:
  void CollectReservedPersistableNodeNames(
      Graph* graph,
      std::unordered_set<std::string>* reserved_persistable_node_names) const;

  int RemoveIsolatedNodes(
      Graph* graph,
      const std::unordered_set<std::string>& reserved_persistable_node_names,
      std::unordered_set<std::string>* delete_node_names) const;

  int UpdateControlFlowOp(
      int current_graph_index,
      Graph* graph,
      const std::unordered_set<std::string>& delete_node_names) const;

  const std::map<std::string, std::string> control_flow_op_input_map_{
      {"while", "X"},
      {"conditional_block", "Input"},
  };
};

void DeleteIsolatedNodePass::ApplyImpl(Graph* graph) const {
  PADDLE_ENFORCE_NOT_NULL(
      graph, common::errors::PreconditionNotMet("graph should not be null."));
  if (!graph->IsMainGraph()) {
    VLOG(3) << "Pass(apply in main graph) will delete isolated nodes in all "
               "subgraphs.";
    return;
  }

  std::unordered_set<std::string> reserved_persistable_node_names;
  for (size_t i = 0; i < graph->SubGraphsSize(); i++) {
    CollectReservedPersistableNodeNames(graph->GetSubGraph(i),
                                        &reserved_persistable_node_names);
  }

  int delete_counts = 0;
  std::unordered_set<std::string> delete_node_names;
  for (size_t i = 0; i < graph->SubGraphsSize(); i++) {
    delete_counts += RemoveIsolatedNodes(graph->GetSubGraph(i),
                                         reserved_persistable_node_names,
                                         &delete_node_names);
  }
  if (delete_counts > 0) {
    LOG(INFO) << "---  delete " << delete_counts << " isolated nodes";
  }

  int update_counts = 0;
  for (size_t i = 0; i < graph->SubGraphsSize(); i++) {
    update_counts += UpdateControlFlowOp(i, graph, delete_node_names);
  }
  if (update_counts > 0) {
    LOG(INFO) << "---  update " << update_counts << " control flow ops";
  }
}

void DeleteIsolatedNodePass::CollectReservedPersistableNodeNames(
    Graph* graph,
    std::unordered_set<std::string>* reserved_persistable_node_names) const {
  for (auto* node : graph->Nodes()) {
    if (!node || node->Name() == "fetch" || node->Name() == "feed") continue;
    if (!node->IsVar() || !node->Var() || !node->Var()->Persistable()) continue;
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
    Graph* graph,
    const std::unordered_set<std::string>& reserved_persistable_node_names,
    std::unordered_set<std::string>* delete_node_names) const {
  BlockDesc* block = nullptr;
  for (auto* node : graph->Nodes()) {
    if (node->IsOp()) {
      block = node->Op()->Block();
      if (block != nullptr) {
        break;
      }
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
    if (!node || node->Name() == "fetch" || node->Name() == "feed") continue;
    if (!node->IsVar() || !node->Var() || !node->Var()->Persistable()) continue;
    auto name = node->Var()->Name();
    if (reserved_persistable_node_names.count(name) > 0) continue;
    delete_nodes.insert(node);
    delete_node_names->insert(node->Name());
    block->RemoveVar(name);
    auto* var = scope.FindVar(name);
    if (var != nullptr) {
      var->Clear();
      scope.EraseVars({name});
    }
    delete_node_counts++;
  }

  GraphSafeRemoveNodes(graph, delete_nodes);
  return delete_node_counts;
}

int DeleteIsolatedNodePass::UpdateControlFlowOp(
    int current_graph_index,
    Graph* graph,
    const std::unordered_set<std::string>& delete_node_names) const {
  auto* cur_graph = graph->GetSubGraph(current_graph_index);
  int update_counts = 0;
  for (auto* node : cur_graph->Nodes()) {
    if (!node->IsOp()) continue;
    auto op_type = node->Op()->Type();
    if (control_flow_op_input_map_.count(op_type) == 0) continue;

    auto in_arg_name = control_flow_op_input_map_.at(op_type);
    auto in_name = node->Op()->Input(in_arg_name);
    std::unordered_set<std::string> in_names_set(in_name.begin(),
                                                 in_name.end());
    for (auto delete_node_name : delete_node_names) {
      if (in_names_set.count(delete_node_name) > 0) {
        in_names_set.erase(delete_node_name);
      }
    }

    auto* sub_block = PADDLE_GET_CONST(framework::BlockDesc*,
                                       node->Op()->GetAttr("sub_block"));
    auto* sub_graph = graph->GetSubGraph(sub_block->ID());
    std::unordered_set<std::string> sub_persistable_node_names;
    CollectReservedPersistableNodeNames(sub_graph, &sub_persistable_node_names);
    for (auto sub_name : sub_persistable_node_names) {
      if (in_names_set.count(sub_name) > 0) continue;
      auto* in_node = FindNodeWithName(graph, sub_name);
      if (in_node == nullptr) continue;
      in_names_set.insert(sub_name);
      IR_NODE_LINK_TO(in_node, node);
    }
    std::vector<std::string> new_in_names(in_names_set.begin(),
                                          in_names_set.end());
    node->Op()->SetInput(in_arg_name, new_in_names);
    update_counts++;
  }
  return update_counts;
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(delete_isolated_node_pass,
              paddle::framework::ir::DeleteIsolatedNodePass);
