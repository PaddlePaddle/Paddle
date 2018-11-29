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
#include "paddle/fluid/framework/details/multi_devices_helper.h"
#include "paddle/fluid/framework/details/reference_count_pass.h"
#include "paddle/fluid/framework/details/reference_count_pass_helper.h"
#include "paddle/fluid/framework/ir/graph_helper.h"

namespace paddle {
namespace framework {
namespace details {

static ComputationOpHandle *FindNextComputationOpHandleOrReturnItself(
    OpHandleBase *op, size_t scope_idx) {
  std::queue<OpHandleBase *> q;
  std::unordered_set<OpHandleBase *> visited;
  q.push(op);
  do {
    auto *op = q.front();
    q.pop();
    auto *compute_op = dynamic_cast<ComputationOpHandle *>(op);
    if (compute_op != nullptr && compute_op->GetScopeIdx() == scope_idx) {
      return compute_op;
    }
    for (auto *out_var : op->Outputs()) {
      for (auto *pending_op : out_var->PendingOps()) {
        if (visited.count(pending_op)) continue;
        visited.insert(pending_op);
      }
    }
  } while (!q.empty());
  return nullptr;
}

std::unique_ptr<ir::Graph> ReferenceCountPass::ApplyImpl(
    std::unique_ptr<ir::Graph> graph) const {
  auto &vars = graph->Get<GraphVars>(kGraphVars);
  auto &ref_cnts = Get<std::vector<ReferenceCountMap>>(kGlobalReferenceCount);
  auto &last_live_ops_of_vars =
      Get<std::vector<LastLiveOpsOfVars>>(kLastLiveOpsOfVars);

  last_live_ops_of_vars = std::vector<LastLiveOpsOfVars>(vars.size());
  ref_cnts = std::vector<ReferenceCountMap>(vars.size());

  for (size_t i = 0; i < vars.size(); ++i) {
    for (auto &name_var_pair : vars[i]) {
      if (name_var_pair.second.empty()) continue;
      auto *last_ver_var = name_var_pair.second.back();

      VarDesc *var_desc = nullptr;
      std::find_if(name_var_pair.second.rbegin(), name_var_pair.second.rend(),
                   [&](VarHandle *var_handle) -> bool {
                     var_desc = var_handle->Node()->Var();
                     return var_desc != nullptr;
                   });

      if (var_desc == nullptr || var_desc->Persistable()) {
        continue;
      }

      auto var_type = var_desc->Proto()->type().type();
      if (var_type != proto::VarType::LOD_TENSOR &&
          var_type != proto::VarType::SELECTED_ROWS &&
          var_type != proto::VarType::LOD_TENSOR_ARRAY) {
        continue;
      }

      std::unordered_set<ComputationOpHandle *> last_live_op;
      auto add_last_live_op = [&](OpHandleBase *op) {
        auto *compute_op = FindNextComputationOpHandleOrReturnItself(op, i);
        if (compute_op) {
          last_live_op.insert(compute_op);
        }
      };
      const std::string &var_name = name_var_pair.first;
      auto &pending_ops = last_ver_var->PendingOps();
      if (pending_ops.empty()) {
        auto *generated_op = last_ver_var->GeneratedOp();
        if (generated_op) {
          ref_cnts[i].emplace(var_name, 1);
          add_last_live_op(generated_op);
        }
      } else {
        ref_cnts[i].emplace(var_name, pending_ops.size());
        for (auto *pending_op : pending_ops) {
          add_last_live_op(pending_op);
        }
      }

      last_live_ops_of_vars[i].emplace(var_name, std::move(last_live_op));
    }
  }
  return graph;
}

}  // namespace details
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(reference_count_pass,
              paddle::framework::details::ReferenceCountPass)
    .RequirePassAttr(paddle::framework::details::kGlobalReferenceCount)
    .RequirePassAttr(paddle::framework::details::kLastLiveOpsOfVars);
