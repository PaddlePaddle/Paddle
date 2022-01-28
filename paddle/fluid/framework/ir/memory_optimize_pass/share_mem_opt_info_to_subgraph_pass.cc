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

#include <algorithm>
#include "paddle/fluid/framework/details/computation_op_handle.h"
#include "paddle/fluid/framework/details/eager_deletion_op_handle.h"
#include "paddle/fluid/framework/ir/graph_helper.h"
#include "paddle/fluid/framework/ir/memory_optimize_pass/memory_optimization_var_info.h"
#include "paddle/fluid/framework/paddle2cinn/build_cinn_pass.h"
#include "paddle/fluid/framework/paddle2cinn/cinn_compiler.h"
#include "paddle/fluid/operators/cinn/cinn_launch_op.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace framework {
namespace ir {

using Name2VarInfoMap =
    std::unordered_map<std::string, std::shared_ptr<MemOptVarInfo>>;

static details::EagerDeletionOpHandle* FindFollowedEagerDeletionOp(
    details::ComputationOpHandle* compute_op) {
  for (auto* var : compute_op->Outputs()) {
    if (!var->Node()->IsCtrlVar()) {
      continue;
    }
    for (auto* op : var->PendingOps()) {
      auto* eager_deletion_op =
          dynamic_cast<details::EagerDeletionOpHandle*>(op);
      if (eager_deletion_op) {
        return eager_deletion_op;
      }
    }
  }
  return nullptr;
}

static void ShareVarInfoToCinnLaunch(
    const MemOptVarInfoMapList& var_infos,
    details::ComputationOpHandle* cinn_launch_op) {
  auto* followed_eager_deletion_op =
      FindFollowedEagerDeletionOp(cinn_launch_op);
  if (!followed_eager_deletion_op) {
    VLOG(4) << "No eager_deletion op found after this cinn_launch op";
    return;
  }
  auto vars_to_delete = followed_eager_deletion_op->VarsToDelete();
  if (vars_to_delete.empty()) {
    VLOG(4) << "No var to be deleted after this cinn_launch op";
    return;
  }

  const auto& subgraph = paddle2cinn::CinnCompiler::GetInstance()->FindGraph(
      cinn_launch_op->GetOp()->Attr<std::string>(operators::kCompilationKey));
  auto& varinfo_from_maingraph =
      subgraph.Get<Name2VarInfoMap>(paddle2cinn::kMemOptVarInfoFromMainGraph);
  const auto& cur_place_var_infos = var_infos.at(cinn_launch_op->GetScopeIdx());

  // collect all MemOptVarInfos of external variables
  // that would be eager deleted after the cinn_launch subgraph executed,
  // and store them as attribute of the subgraph
  for (const auto& var_name : vars_to_delete) {
    auto it = cur_place_var_infos.find(var_name);
    PADDLE_ENFORCE_NE(it, cur_place_var_infos.end(),
                      platform::errors::NotFound(
                          "MemOptVarInfo of Var[%s] not found", var_name));
    varinfo_from_maingraph.emplace(var_name, it->second);
  }
}

static void TakeVarInfoFromMainGraph(
    const Name2VarInfoMap& parent_var_infos,
    const MemOptVarInfoMapList& var_infos,
    details::EagerDeletionOpHandle* eager_deletion_op) {
  const auto& cur_place_var_infos =
      var_infos.at(eager_deletion_op->GetScopeIdx());
  for (auto&& var_name : eager_deletion_op->VarsToDelete()) {
    auto cur_it = cur_place_var_infos.find(var_name);
    PADDLE_ENFORCE_NE(cur_it, cur_place_var_infos.end(),
                      platform::errors::NotFound(
                          "MemOptVarInfo of Var[%s] not found", var_name));
    auto parent_it = parent_var_infos.find(var_name);
    if (parent_it != parent_var_infos.end()) {
      VLOG(4) << "Var[" << var_name << "] set parent holder";
      cur_it->second->SetParentHolder(parent_it->second);
    }
  }
}

// This pass will be applied on both the main graph and all subgraphs,
// and it distinguishs them according to whether the graph has the
// kMemOptVarInfoFromMainGraph attribute or not.
// On the main graph, it finds all cinn_launch ops and shares MemOptVarInfos
// to their subgraphs.
// On a subgraph, it iterates each variable that will be deleted by a
// eager_deletion op, and take the MemOptVarInfo from the main graph
// if such one found.
class ShareMemOptInfoToSubGraphPass : public ir::Pass {
 protected:
  void ApplyImpl(ir::Graph* graph) const override {
    auto all_ops = ir::FilterByNodeWrapper<details::OpHandleBase>(*graph);
    const auto& var_infos = Get<MemOptVarInfoMapList>(kMemOptVarInfoMapList);

    // the main graph
    if (!graph->Has(paddle2cinn::kMemOptVarInfoFromMainGraph)) {
      for (auto* op : all_ops) {
        auto compute_op = dynamic_cast<details::ComputationOpHandle*>(op);
        if (compute_op && compute_op->Name() == "cinn_launch") {
          ShareVarInfoToCinnLaunch(var_infos, compute_op);
        }
      }
    } else {  // a subgraph
      const auto& parent_var_infos =
          graph->Get<Name2VarInfoMap>(paddle2cinn::kMemOptVarInfoFromMainGraph);
      for (auto* op : all_ops) {
        auto eager_deletion_op =
            dynamic_cast<details::EagerDeletionOpHandle*>(op);
        if (eager_deletion_op) {
          TakeVarInfoFromMainGraph(parent_var_infos, var_infos,
                                   eager_deletion_op);
        }
      }
    }
  }
};

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(share_mem_opt_info_to_subgraph_pass,
              paddle::framework::ir::ShareMemOptInfoToSubGraphPass)
    .RequirePassAttr(paddle::framework::ir::kMemOptVarInfoMapList);
