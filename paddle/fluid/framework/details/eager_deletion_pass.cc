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

#include <queue>
#include <string>
#include <vector>

#include "paddle/fluid/framework/details/computation_op_handle.h"
#include "paddle/fluid/framework/details/eager_deletion_op_handle.h"
#include "paddle/fluid/framework/details/eager_deletion_pass.h"
#include "paddle/fluid/framework/details/multi_devices_helper.h"
#include "paddle/fluid/framework/ir/graph_helper.h"

namespace paddle {
namespace framework {
namespace details {

std::unique_ptr<ir::Graph> EagerDeletionPass::ApplyImpl(
    std::unique_ptr<ir::Graph> graph) const {
  auto &ref_cnts =
      Get<std::vector<AtomicReferenceCountMap>>(kRuntimeReferenceCount);
  PADDLE_ENFORCE(ref_cnts.empty(),
                 "kRuntimeReferenceCount should be initialized here!");

  const auto &vars = graph->Get<GraphVars>(kGraphVars);
  ref_cnts.resize(vars.size());

  const auto &last_live_ops =
      Get<std::vector<LastLiveOpsOfVars>>(kLastLiveOpsOfVars);
  const auto &gcs = Get<GarbageCollectorMap>(kGarbageCollector);
  const auto &places = Get<std::vector<platform::Place>>(kAllPlaces);

  // a reverse map of last_live_ops
  //   i.e., last op --> variable names which can be deleted.
  std::unordered_map<ComputationOpHandle *, std::unordered_set<std::string>>
      op_vars_map;

  for (auto &var_ops_map : last_live_ops) {
    for (auto &var_ops_pair : var_ops_map) {
      const std::string &var_name = var_ops_pair.first;
      for (auto *op : var_ops_pair.second) {
        op_vars_map[op].insert(var_name);
      }
    }
  }

  for (auto &pair : op_vars_map) {
    auto *op = pair.first;
    auto &var_names = pair.second;

    auto *eager_deletion_node =
        graph->CreateEmptyNode("eager_deletion", ir::Node::Type::kOperation);
    auto *eager_deletion_op = new EagerDeletionOpHandle(
        eager_deletion_node, op->GetScope(), op->GetPlace(), var_names,
        gcs.at(places[op->GetScopeIdx()]).get(),
        &(ref_cnts[op->GetScopeIdx()]));

    auto it = std::find_if(
        op->Outputs().begin(), op->Outputs().end(), [](VarHandleBase *var) {
          return dynamic_cast<DummyVarHandle *>(var) != nullptr;
        });

    if (it != op->Outputs().end()) {
      eager_deletion_op->AddInput(*it);
    } else {
      auto *dep_var = new DummyVarHandle(graph->CreateControlDepVar());
      graph->Get<GraphDepVars>(kGraphDepVars).emplace(dep_var);
      op->AddOutput(dep_var);
      eager_deletion_op->AddInput(dep_var);
    }

    auto *dummy_leaf = new DummyVarHandle(graph->CreateControlDepVar());
    graph->Get<GraphDepVars>(kGraphDepVars).emplace(dummy_leaf);
    eager_deletion_op->AddOutput(dummy_leaf);
  }

  VLOG(10) << "Create " << op_vars_map.size() << " EagerDeletionOpHandle(s)";
  return graph;
}

}  // namespace details
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(eager_deletion_pass,
              paddle::framework::details::EagerDeletionPass)
    .RequirePassAttr(paddle::framework::details::kRuntimeReferenceCount)
    .RequirePassAttr(paddle::framework::details::kLastLiveOpsOfVars)
    .RequirePassAttr(paddle::framework::details::kAllPlaces)
    .RequirePassAttr(paddle::framework::details::kGarbageCollector);
