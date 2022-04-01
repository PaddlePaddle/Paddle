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
#include "paddle/fluid/string/string_helper.h"

namespace paddle::framework::ir {

using Name2VarInfoMap =
    std::unordered_map<std::string, std::shared_ptr<MemOptVarInfo>>;

static details::EagerDeletionOpHandle* FindFollowedEagerDeletionOp(
    details::ComputationOpHandle* compute_op) {
  for (details::VarHandleBase* var : compute_op->Outputs()) {
    if (!var->Node()->IsCtrlVar()) {
      continue;
    }
    for (details::OpHandleBase* op : var->PendingOps()) {
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
    const MemOptVarInfoMapList& varinfo_maps,
    details::ComputationOpHandle* cinn_launch_op) {
  details::EagerDeletionOpHandle* followed_eager_deletion_op =
      FindFollowedEagerDeletionOp(cinn_launch_op);
  if (!followed_eager_deletion_op) {
    VLOG(4) << "No eager_deletion op found after this cinn_launch op";
    return;
  }

  std::vector<std::string> vars_to_delete =
      followed_eager_deletion_op->VarsToDelete();
  if (vars_to_delete.empty()) {
    VLOG(4) << "No var to be deleted after this cinn_launch op";
    return;
  }
  VLOG(4) << "Variables would be deleted by the eager_deletion_op"
          << " following the cinn_launch:"
          << paddle::string::join_strings(vars_to_delete, ',');

  const Graph& subgraph = paddle2cinn::CinnCompiler::GetInstance()->FindGraph(
      cinn_launch_op->GetOp()->Attr<std::string>(operators::kCompilationKey));
  auto& dst_varinfo_map =
      subgraph.Get<Name2VarInfoMap>(paddle2cinn::kMemOptVarInfoFromMainGraph);
  const Name2VarInfoMap& src_varinfo_map =
      varinfo_maps.at(cinn_launch_op->GetScopeIdx());

  // collect all MemOptVarInfos of external variables
  // that were eager deleted after the cinn_launch subgraph executed,
  // and we will delete them in advance among eager_deletion_ops
  // inside cinn_launch subgraph, so store them as attribute of the subgraph
  // to pass to the inner eager_deletion_ops.
  for (const auto& var_name : vars_to_delete) {
    auto it = src_varinfo_map.find(var_name);
    PADDLE_ENFORCE_NE(it, src_varinfo_map.end(),
                      platform::errors::NotFound(
                          "MemOptVarInfo of var[%s] not found", var_name));
    dst_varinfo_map.emplace(var_name, it->second);
  }
  // skip running of the followed eager_deletion_op
  followed_eager_deletion_op->SetSkipRunning(true);
}

static void TakeVarInfoFromMainGraph(
    const Name2VarInfoMap& src_varinfo_map,
    const MemOptVarInfoMapList& varinfo_maps,
    details::EagerDeletionOpHandle* eager_deletion_op) {
  const Name2VarInfoMap& dst_varinfo_map =
      varinfo_maps.at(eager_deletion_op->GetScopeIdx());
  for (auto&& var_name : eager_deletion_op->VarsToDelete()) {
    auto dst_it = dst_varinfo_map.find(var_name);
    PADDLE_ENFORCE_NE(dst_it, dst_varinfo_map.end(),
                      platform::errors::NotFound(
                          "MemOptVarInfo of var[%s] not found", var_name));
    auto src_it = src_varinfo_map.find(var_name);
    if (src_it != src_varinfo_map.end()) {
      VLOG(4) << "MemOptVarInfo of var[" << var_name << "] set parent holder";
      dst_it->second->SetParentHolder(src_it->second);
    }
  }
}

// This pass will be applied on both the main graph and all cinn subgraphs,
// and it distinguishs them according to whether the graph has the
// kMemOptVarInfoFromMainGraph attribute or not.
// On the main graph, it finds all cinn_launch ops and shares MemOptVarInfos
// to their subgraphs.
// On a cinn subgraph, it iterates each variable that will be deleted by a
// eager_deletion op, and take the MemOptVarInfo from the main graph
// if such one found.
class ShareMemOptInfoToSubGraphPass : public ir::Pass {
 protected:
  void ApplyImpl(ir::Graph* graph) const override {
    auto all_ops = ir::FilterByNodeWrapper<details::OpHandleBase>(*graph);
    const auto& varinfo_maps = Get<MemOptVarInfoMapList>(kMemOptVarInfoMapList);

    // the main graph
    if (!graph->Has(paddle2cinn::kMemOptVarInfoFromMainGraph)) {
      for (details::OpHandleBase* op : all_ops) {
        auto compute_op = dynamic_cast<details::ComputationOpHandle*>(op);
        if (compute_op && compute_op->Name() == "cinn_launch") {
          ShareVarInfoToCinnLaunch(varinfo_maps, compute_op);
        }
      }
    } else {  // a cinn subgraph
      const auto& parent_varinfo_map =
          graph->Get<Name2VarInfoMap>(paddle2cinn::kMemOptVarInfoFromMainGraph);
      for (details::OpHandleBase* op : all_ops) {
        auto eager_deletion_op =
            dynamic_cast<details::EagerDeletionOpHandle*>(op);
        if (eager_deletion_op) {
          TakeVarInfoFromMainGraph(parent_varinfo_map, varinfo_maps,
                                   eager_deletion_op);
        }
      }
    }
  }
};

}  // namespace paddle::framework::ir

REGISTER_PASS(share_varinfo_into_cinn_pass,
              paddle::framework::ir::ShareMemOptInfoToSubGraphPass)
    .RequirePassAttr(paddle::framework::ir::kMemOptVarInfoMapList);
