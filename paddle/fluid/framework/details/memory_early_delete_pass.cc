// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/details/memory_early_delete_pass.h"
#include <queue>
#include <string>
#include <vector>
#include "paddle/fluid/framework/details/memory_reuse_types.h"
#include "paddle/fluid/framework/details/multi_devices_helper.h"
#include "paddle/fluid/framework/details/reference_count_pass_helper.h"
#include "paddle/fluid/framework/ir/graph_helper.h"

namespace paddle {
namespace framework {
namespace details {

static ComputationOpHandle* FindNextComputationOpHandle(VarHandle* var_in) {
  std::queue<VarHandleBase*> queue;
  queue.push(var_in);
  do {
    auto* var = queue.front();
    queue.pop();
    for (auto* op : var->PendingOps()) {
      auto* compute_op = dynamic_cast<ComputationOpHandle*>(op);
      if (compute_op != nullptr && compute_op->GetPlace() == var_in->place()) {
        return compute_op;
      }
      for (auto* out_var : op->Outputs()) {
        queue.push(out_var);
      }
    }
  } while (!queue.empty());
  return nullptr;
}

std::unique_ptr<ir::Graph> MemoryEarlyDeletePass::ApplyImpl(
    std::unique_ptr<ir::Graph> graph) const {
  auto& graph_pool = Get<GraphNodePool>(kGraphNodePool);
  auto& gcs = Get<GarbageCollectorMap>(kGarbageCollector);

  std::unordered_map<std::string, std::unordered_set<OpDesc*>> unlived_vars;
  unlived_vars.reserve(graph_pool.size());
  for (auto& pair : graph_pool) {
    unlived_vars.insert(std::make_pair(pair.first, pair.second));
  }

  auto compare_and_insert_early_delete_op = [&](
      OpHandleBase* op, const std::vector<VarHandleBase*>& vars) {
    if (unlived_vars.empty()) return;
    // unlived vars can be deleted after the last used op has finished.
    auto* compute_op = dynamic_cast<ComputationOpHandle*>(op);
    const auto& places = Get<std::vector<platform::Place>>(kAllPlaces);
    for (auto& var : vars) {
      auto* var_handle = dynamic_cast<VarHandle*>(var);
      auto var_name = var->Node()->Name();
      auto& var_place = var_handle->place();
      if (unlived_vars.count(var_name) == 0) continue;
      if (!unlived_vars[var_name].empty()) {
        if (compute_op != nullptr &&
            unlived_vars[var_name].count(compute_op->Node()->Op()) != 0) {
          unlived_vars[var_name].erase(compute_op->Node()->Op());
        }
        continue;
      }

      if (var_handle == nullptr || !var_handle->Node()->IsVar() ||
          var_handle->Node()->IsCtrlVar())
        continue;

      // shameless copyed from reference count pass.
      if (compute_op == nullptr) {
        // use next computation op scope
        compute_op = FindNextComputationOpHandle(var_handle);
      }
      auto* early_delete_node =
          graph->CreateEmptyNode("early_delete", ir::Node::Type::kOperation);
      GarbageCollector* gc = gcs.at(places[compute_op->GetScopeIdx()]).get();
      auto* early_delete_handle = new EarlyDeleteOpHandle(
          early_delete_node, compute_op->GetScope(), var_place, {var_name}, gc);
      if (compute_op->Outputs().empty()) {
        auto* dep_var = new DummyVarHandle(graph->CreateControlDepVar());
        compute_op->AddOutput(dep_var);
        graph->Get<GraphDepVars>(kGraphDepVars).emplace(dep_var);
      }
      early_delete_handle->AddInput(compute_op->Outputs().front());
      VLOG(5) << "Add early delete op " << var_name << " to Operator"
              << compute_op->Name();
    }
  };

  auto all_ops = ir::FilterByNodeWrapper<OpHandleBase>(*graph);
  for (auto& op : all_ops) {
    compare_and_insert_early_delete_op(op, op->Inputs());
    compare_and_insert_early_delete_op(op, op->Outputs());
  }
  return graph;
}

}  // namespace details
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(memory_early_delete_pass,
              paddle::framework::details::MemoryEarlyDeletePass)
    .RequireGraphAttr(paddle::framework::details::kGraphNodePool)
    .RequireGraphAttr(paddle::framework::details::kGarbageCollector);
