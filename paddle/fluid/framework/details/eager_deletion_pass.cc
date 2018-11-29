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

static void AddDependencyBetween(OpHandleBase *in, OpHandleBase *out,
                                 ir::Graph *graph) {
  auto it = std::find_if(
      in->Outputs().begin(), in->Outputs().end(), [](VarHandleBase *var) {
        return dynamic_cast<DummyVarHandle *>(var) != nullptr;
      });

  if (it != in->Outputs().end()) {
    out->AddInput(*it);
  } else {
    auto *dep_var = new DummyVarHandle(graph->CreateControlDepVar());
    graph->Get<GraphDepVars>(kGraphDepVars).emplace(dep_var);
    in->AddOutput(dep_var);
    out->AddInput(dep_var);
  }

  // Add leaf node to eager_deletion_node
  if (out->Outputs().empty()) {
    auto *dummy_leaf = new DummyVarHandle(graph->CreateControlDepVar());
    graph->Get<GraphDepVars>(kGraphDepVars).emplace(dummy_leaf);
    out->AddOutput(dummy_leaf);
  }
}

std::unique_ptr<ir::Graph> EagerDeletionPass::ApplyImpl(
    std::unique_ptr<ir::Graph> graph) const {
  auto &vars = graph->Get<GraphVars>(kGraphVars);

  auto &ref_cnts =
      Get<std::vector<AtomicReferenceCountMap>>(kCurReferenceCount);
  auto &last_live_ops = Get<std::vector<LastLiveOpsOfVars>>(kLastLiveOpsOfVars);
  auto &gcs = Get<GarbageCollectorList>(kGarbageCollector);

  ref_cnts = std::vector<AtomicReferenceCountMap>(vars.size());

  std::unordered_map<ComputationOpHandle *, EagerDeletionOpHandle *> op_map;
  for (auto &var_ops_map : last_live_ops) {
    for (auto &var_ops_pair : var_ops_map) {
      const std::string &var_name = var_ops_pair.first;
      for (ComputationOpHandle *op : var_ops_pair.second) {
        auto it = op_map.find(op);
        if (it != op_map.end()) {
          it->second->AddVar(var_name);
        } else {
          auto *eager_deletion_node = graph->CreateEmptyNode(
              "eager_deletion", ir::Node::Type::kOperation);
          auto *eager_deletion_op = new EagerDeletionOpHandle(
              eager_deletion_node, op->GetScope(), op->GetPlace(), {var_name},
              gcs[op->GetScopeIdx()].get(), &(ref_cnts[op->GetScopeIdx()]));
          AddDependencyBetween(op, eager_deletion_op, graph.get());
          op_map[op] = eager_deletion_op;
        }
      }
    }
  }
  VLOG(10) << "Create " << op_map.size() << " EagerDeletionOpHandle(s)";
  return graph;
}

}  // namespace details
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(eager_deletion_pass,
              paddle::framework::details::EagerDeletionPass)
    .RequirePassAttr(paddle::framework::details::kCurReferenceCount)
    .RequirePassAttr(paddle::framework::details::kLastLiveOpsOfVars)
    .RequirePassAttr(paddle::framework::details::kGarbageCollector);
