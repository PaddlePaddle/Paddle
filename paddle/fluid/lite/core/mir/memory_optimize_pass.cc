// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/lite/core/mir/memory_optimize_pass.h"
#include <algorithm>
#include <fstream>
#include <limits>
#include <map>
#include <unordered_map>
#include <utility>
#include <vector>
#include "paddle/fluid/lite/core/mir/pass_registry.h"
#include "paddle/fluid/lite/core/type_system.h"

namespace paddle {
namespace lite {
namespace mir {

void MemoryOptimizePass::Apply(const std::unique_ptr<SSAGraph>& graph) {
  SSAGraph* ssa_graph = graph.get();
  CHECK(graph);

  std::unordered_map<std::string, lifecycle_t> lifecycles;
  std::unordered_map<std::string, size_t> space_table;
  std::unordered_map<std::string, std::string> node2cluster;
  std::vector<MemNode> mem_nodes;

  CollectLifeCycle(ssa_graph, &lifecycles);
  CollectVarMemorySize(ssa_graph, &space_table);
  CollectOverlapInfo(lifecycles, space_table, &mem_nodes);
  MakeReusePlan(ssa_graph, &node2cluster, &mem_nodes);
  // For kernel runtime
  UpdateScopeVarsByReuseTable(ssa_graph, node2cluster);
  // For OpDesc and SSAGraph display
  UpdateSSAGraphByReuseTable(ssa_graph, node2cluster);
}

void MemoryOptimizePass::CollectLifeCycle(
    SSAGraph* graph,
    std::unordered_map<std::string, lifecycle_t>* lifecycles) const {
  int max_lifecycle = 0;
  for (auto& node : graph->StmtTopologicalOrder()) {
    auto& inst = node->AsStmt();
    // Disable reuse of feed variables.
    if (inst.op_type == "feed") {
      CollectLifeCycleHelper(graph, max_lifecycle, lifecycles,
                             inst.op_info()->output_names());
    } else if (inst.op_type == "fetch") {
      CollectLifeCycleHelper(graph, max_lifecycle, lifecycles,
                             inst.op_info()->input_names());
    } else {
      // Normal operators.
      CollectLifeCycleHelper(graph, max_lifecycle, lifecycles,
                             inst.op_info()->input_names());
      CollectLifeCycleHelper(graph, max_lifecycle, lifecycles,
                             inst.op_info()->output_names());
    }
    ++max_lifecycle;
  }
}

// Collect the memory size of the tensors.
void MemoryOptimizePass::CollectVarMemorySize(
    SSAGraph* graph,
    std::unordered_map<std::string, size_t>* space_table) const {
  // Collect tensors from graph.
  LOG(INFO) << "CollectVarMemorySize: ===========================";
  for (auto& node : graph->mutable_nodes()) {
    if (!node.IsArg() || !IsVarCanBeReused(graph, node.AsArg().name)) continue;
    auto& var = node.AsArg();
    LOG(INFO) << var.name << " " << var.var_info()->data_size() << " "
              << var.var_info()->data_type_space();
    (*space_table)[var.name] = var.var_info()->space_size();
  }
}

void MemoryOptimizePass::CollectOverlapInfo(
    const std::unordered_map<std::string, std::pair<int, int>>& lifecycles,
    const std::unordered_map<std::string, size_t>& space_table,
    std::vector<MemNode>* mem_nodes) {
  // Make sure candidate reuse variables match rules.
  // Exclude: FeedOp's inputs, FetchOp's outputs and persistable variables
  for (auto& data : lifecycles) {
    CHECK(space_table.count(data.first))
        << data.first
        << " variable should be in the spacetable during memory optimize";
    mem_nodes->emplace_back();
    auto& temp_node = mem_nodes->back();
    temp_node.name = data.first;
    temp_node.size = space_table.at(data.first);
    temp_node.cluster = -1;
    temp_node.lifetime = data.second;
  }
  auto overlap = [](std::pair<int, int> a, std::pair<int, int> b) -> bool {
    return b.second >= a.first && a.second >= b.first;
  };
  // If the lifetime of two nodes is overwritten, we set them as adjacent nodes.
  for (size_t i = 0; i < mem_nodes->size(); i++) {
    for (size_t j = i + 1; j < mem_nodes->size(); j++) {
      if (overlap((*mem_nodes)[i].lifetime, (*mem_nodes)[j].lifetime)) {
        (*mem_nodes)[i].adj.insert((*mem_nodes)[j].name);
        (*mem_nodes)[j].adj.insert((*mem_nodes)[i].name);
      }
    }
  }
  // Sort the nodes according to the node memory size.
  auto sort_func = [](MemNode a, MemNode b) { return a.size > b.size; };
  std::sort(mem_nodes->begin(), mem_nodes->end(), sort_func);
}

void MemoryOptimizePass::MakeReusePlan(
    SSAGraph* graph, std::unordered_map<std::string, std::string>* node2cluster,
    std::vector<MemNode>* mem_nodes) {
  switch (memory_optimize_kind_) {
    case MemoryOptimizeKind::kGreedy:
      MemoryOptimizeGreedy(graph, node2cluster, mem_nodes);
      break;
    case MemoryOptimizeKind::kAdapt:
      LOG(FATAL) << "Adapt memory optimize is not supported now";
      break;
    default:
      LOG(FATAL) << "Unknown memory optimize kind";
  }
}

void MemoryOptimizePass::MemoryOptimizeGreedy(
    SSAGraph* graph, std::unordered_map<std::string, std::string>* node2cluster,
    std::vector<MemNode>* mem_nodes) {
  struct ClusterInfo {
    std::unordered_map<std::string, int> cluster_size;
    std::unordered_map<std::string, std::string> cluster;
  };

  std::map<TargetType, ClusterInfo> clusters;

  // Generating Memory Reuse Strategy Based on Greedy Way
  for (size_t i = 0; i < mem_nodes->size(); i++) {
    if ((*mem_nodes)[i].cluster >= 0) continue;
    auto& center_name = (*mem_nodes)[i].name;
    auto var_type = graph->Argument(center_name)->AsArg().type->target();
    // var_type = TargetType::kX86;
    auto& cluster_size = clusters[var_type].cluster_size;
    auto& cluster = clusters[var_type].cluster;

    int cluster_index = cluster_size.size();
    (*mem_nodes)[i].cluster = cluster_index;
    cluster_size[center_name] = (*mem_nodes)[i].size;
    cluster[center_name] = center_name;
    std::unordered_set<std::string> cluster_adj = (*mem_nodes)[i].adj;

    for (size_t j = i + 1; j < mem_nodes->size(); j++) {
      auto& candi_name = (*mem_nodes)[j].name;
      auto candi_var_type = graph->Argument(candi_name)->AsArg().type->target();
      if (var_type != candi_var_type) continue;
      if ((*mem_nodes)[j].cluster < 0 &&
          (cluster_adj.find(candi_name) == cluster_adj.end())) {
        cluster[candi_name] = center_name;
        (*mem_nodes)[j].cluster = cluster_index;
        for (auto& n : (*mem_nodes)[j].adj) {
          cluster_adj.insert(n);
        }
      }
    }
  }

  // Merge clusters together
  for (auto& item : clusters) {
    node2cluster->insert(item.second.cluster.begin(),
                         item.second.cluster.end());
  }
}

void MemoryOptimizePass::UpdateScopeVarsByReuseTable(
    SSAGraph* graph,
    const std::unordered_map<std::string, std::string>& reuse_table) const {
  for (auto& item : reuse_table) {
    if (item.first == item.second) continue;
    auto* src = graph->scope()->FindVar(item.second);
    auto* tgt = graph->scope()->FindVar(item.first);
    CHECK(src);
    CHECK(tgt);
    auto* src_tensor = src->GetMutable<lite::Tensor>();
    auto* tgt_tensor = tgt->GetMutable<lite::Tensor>();
// All variable in scope must be lite::Tensor
#ifndef LITE_WITH_LIGHT_WEIGHT_FRAMEWORK
    // HvyTensor(framework::LoDTensor) must call mutable_data first.
    // Now just init memory with default type FLOAT
    src_tensor->template mutable_data<float>();
#endif
    tgt_tensor->ShareDataWith(*src_tensor);
  }
}

void MemoryOptimizePass::UpdateSSAGraphByReuseTable(
    SSAGraph* graph,
    const std::unordered_map<std::string, std::string>& reuse_table) const {
  UpdateVarNodesByReuseTable(graph, reuse_table);
  UpdateOpNodesByReuseTable(graph, reuse_table);
}

void MemoryOptimizePass::UpdateVarNodesByReuseTable(
    SSAGraph* graph,
    const std::unordered_map<std::string, std::string>& reuse_table) const {
  // Update SSAGraph VarNodes(IsArg() == true) for display
  auto& graph_nodes = graph->mutable_nodes();
  std::unordered_map<std::string, mir::Node::Arg*> name2node;
  for (auto& node : graph_nodes) {
    if (!node.IsArg()) continue;
    auto& var = node.AsArg();
    name2node.insert({var.name, &var});
  }

  for (auto& item : reuse_table) {
    CHECK(name2node.count(item.first));
    CHECK(name2node.count(item.second));
    auto* tgt = name2node.at(item.first);
    auto* src = name2node.at(item.second);
    tgt->ShareDataWith(*src);
  }
}

void MemoryOptimizePass::UpdateOpNodesByReuseTable(
    SSAGraph* graph,
    const std::unordered_map<std::string, std::string>& reuse_table) const {
  using op_data_t = std::map<std::string, std::vector<std::string>>;
  // Debug information.
  std::unordered_map<std::string, std::pair<std::string, std::string>>
      arg2param;
  std::unordered_set<std::string> free_vars;
  auto debug_func = std::function<void(mir::Node::Stmt&, const op_data_t&)>(
      [&](mir::Node::Stmt& node, const op_data_t& vars) {
        for (const auto& item : vars) {
          for (const auto& x : item.second) {
            if (arg2param.count(x)) continue;
            arg2param.emplace(
                std::make_pair(x, std::make_pair(node.op_type, item.first)));
          }
        }
      });
  for (auto& node : graph->StmtTopologicalOrder()) {
    auto& inst = node->AsStmt();
    debug_func(inst, inst.op_info()->outputs());
    debug_func(inst, inst.op_info()->inputs());
  }

  auto update_opdesc_func = std::function<void(mir::Node::Stmt&, op_data_t*)>(
      [&](mir::Node::Stmt& node, op_data_t* vars) {
        for (auto& item : *vars) {
          for (auto& x : item.second) {
            if (reuse_table.count(x) && reuse_table.at(x) != x) {
              const auto& target = reuse_table.at(x);
              LOG(INFO) << node.op_type << "/" << item.first << "/" << x
                        << " -> " << arg2param[target].first << "/"
                        << arg2param[target].second << "/" << target;
              free_vars.insert(x);
              x = target;
            }
          }
        }
      });
  for (auto& node : graph->StmtTopologicalOrder()) {
    auto& inst = node->AsStmt();
    // Replace the original inputs.
    update_opdesc_func(inst, inst.mutable_op_info()->mutable_inputs());
    // Replace the original outputs.
    update_opdesc_func(inst, inst.mutable_op_info()->mutable_outputs());
  }

  size_t total_size = arg2param.size();
  size_t free_size = free_vars.size();
  LOG(INFO) << "Total Vars: " << total_size << " Reuse Vars: " << free_size
            << " Reuse Ratio: " << (static_cast<float>(free_size) / total_size);
}

bool MemoryOptimizePass::IsVarCanBeReused(SSAGraph* graph,
                                          const std::string& name) const {
  CHECK(graph);
  if (name == "feed" || name == "fetch") {
    return false;
  }

  auto* node = graph->Argument(name);
  CHECK(node);
  return !node->AsArg().is_weight;
}

void MemoryOptimizePass::CollectLifeCycleHelper(
    SSAGraph* graph, int max_lifecycle,
    std::unordered_map<std::string, lifecycle_t>* lifecycles,
    const std::vector<std::string>& var_names) const {
  for (auto& var : var_names) {
    if (!IsVarCanBeReused(graph, var)) continue;
    if (!lifecycles->count(var)) {
      (*lifecycles)[var] = std::make_pair(max_lifecycle, max_lifecycle);
    } else {
      (*lifecycles)[var].second =
          std::max(max_lifecycle, lifecycles->at(var).second);  // max()
    }
  }
}

}  // namespace mir
}  // namespace lite
}  // namespace paddle

REGISTER_MIR_PASS(memory_optimize_pass, paddle::lite::mir::MemoryOptimizePass);
