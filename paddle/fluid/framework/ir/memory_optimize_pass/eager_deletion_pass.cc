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

#include <algorithm>
#include <functional>
#include <queue>
#include <string>
#include <tuple>
#include <vector>

#include "paddle/fluid/framework/details/computation_op_handle.h"
#include "paddle/fluid/framework/details/eager_deletion_op_handle.h"
#include "paddle/fluid/framework/details/multi_devices_helper.h"
#include "paddle/fluid/framework/garbage_collector.h"
#include "paddle/fluid/framework/ir/graph_helper.h"
#include "paddle/fluid/framework/ir/memory_optimize_pass/memory_optimization_var_info.h"

namespace paddle {
namespace framework {
namespace ir {

// op -> variables which can be deleted after op runs
using OpToVarNameSetMap = std::unordered_map<details::ComputationOpHandle *,
                                             std::unordered_set<std::string>>;

static std::map<size_t, std::unordered_set<std::string>> VarsGroupByScopeIdx(
    const OpToVarNameSetMap &map) {
  std::map<size_t, std::unordered_set<std::string>> result;
  for (auto &pair : map) {
    size_t scope_idx = pair.first->GetScopeIdx();
    auto &var_set = result[scope_idx];
    for (auto &var : pair.second) {
      var_set.insert(var);
    }
  }
  return result;
}

// Check whether the variable is LoDTensor based on static VarDesc info
static bool IsLoDTensor(VarDesc *var) {
  return var->Proto()->type().type() == proto::VarType::LOD_TENSOR;
}

// Get memory size of LoDTensor
static int64_t GetMemorySize(
    const std::unordered_map<std::string, std::vector<details::VarHandle *>>
        &vars,
    const std::string &var_name) {
  auto *var_desc = TryGetLatestVarDesc(vars.at(var_name));
  PADDLE_ENFORCE_NOT_NULL(
      var_desc,
      platform::errors::NotFound("Var(%s) can not find VarDesc.", var_name));
  PADDLE_ENFORCE_EQ(IsLoDTensor(var_desc), true,
                    platform::errors::InvalidArgument(
                        "Var(%s) must be LoDTensor.", var_name));
  auto dims = var_desc->GetShape();
  return SizeOfType(var_desc->GetDataType()) *
         std::accumulate(dims.begin(), dims.end(), static_cast<int64_t>(1),
                         std::multiplies<int64_t>());
}

// Split all variables in the graph into LoDTensor and Non-LoDTensor (e.g.
// SelectedRows, LoDTensorArray)
// Since partial GC is based on static analysis of memory size of each variable
// So we should skip SelectedRows and LoDTensorArray here
static void SplitIntoLoDTensorAndNonLoDTensorVars(
    const OpToVarNameSetMap &m, const details::GraphVars &vars,
    OpToVarNameSetMap *lod_tensors, OpToVarNameSetMap *other_vars) {
  lod_tensors->clear();
  other_vars->clear();

  for (auto &op_vars_pair : m) {
    for (auto var_name : op_vars_pair.second) {
      auto *var_desc = TryGetLatestVarDesc(
          vars[op_vars_pair.first->GetScopeIdx()].at(var_name));
      if (IsLoDTensor(var_desc)) {
        (*lod_tensors)[op_vars_pair.first].insert(var_name);
      } else {
        (*other_vars)[op_vars_pair.first].insert(var_name);
      }
    }
  }
}

struct GCVarInfo {
  GCVarInfo(const std::string &name, int64_t memory_size,
            details::ComputationOpHandle *op, size_t scope_idx)
      : name_(name),
        memory_size_(memory_size),
        op_(op),
        scope_idx_(scope_idx) {}

  std::string name_;     // variable name
  int64_t memory_size_;  // memory size
  details::ComputationOpHandle
      *op_;           // op after which the variable could be deleted
  size_t scope_idx_;  // scope index where the variable locates

  int64_t AbsMemorySize() const { return std::abs(memory_size_); }
};

// Delete delete_lod_tensor_only is not used currently
static OpToVarNameSetMap ShrinkGCVars(
    const OpToVarNameSetMap &m, const details::GraphVars &vars,
    const std::vector<platform::Place> &places, double fraction_of_memory_size,
    bool delete_lod_tensor_only = false) {
  // Do not perform gc when fraction_of_memory_size = 0
  if (fraction_of_memory_size <= 0.0) return {};

  /**
   * Step 1: Split all variables into LoDTensor and Non-LoDTensor.
   * We can only calculate memory size of LoDTensors
   */
  OpToVarNameSetMap lod_tensors, other_vars;
  SplitIntoLoDTensorAndNonLoDTensorVars(m, vars, &lod_tensors, &other_vars);

  // Perform complete gc when fraction_of_memory_size >= 1
  if (fraction_of_memory_size >= 1.0) {
    return delete_lod_tensor_only ? lod_tensors : m;
  }

  /**
   * Step 2: build GCVarInfos, and calculate total memory sizes of each device
   */

  // place -> variable info (name, memory size, place, scope_idx)
  std::map<platform::Place, std::vector<GCVarInfo>> place_to_vars;

  // place -> total memory sizes
  std::map<platform::Place, int64_t> place_to_size;
  for (auto &op_vars_pair : lod_tensors) {
    auto *op = op_vars_pair.first;
    auto &var_names = op_vars_pair.second;
    auto scope_idx = op->GetScopeIdx();
    auto &place = places[scope_idx];

    for (auto &var_name : var_names) {
      auto var_size = GetMemorySize(vars[scope_idx], var_name);
      GCVarInfo var_info(var_name, var_size, op, scope_idx);
      place_to_size[place] += var_info.AbsMemorySize();
      place_to_vars[place].emplace_back(std::move(var_info));
    }
  }

  /**
   * Step 3: sort GCVarInfos, and only delete the largest variables.
   */
  OpToVarNameSetMap partial_vars;
  for (auto &place_to_var_pair : place_to_vars) {
    auto &place = place_to_var_pair.first;
    auto &gc_vars = place_to_var_pair.second;
    std::sort(gc_vars.begin(), gc_vars.end(),
              [](const GCVarInfo &var1, const GCVarInfo &var2) {
                return var1.AbsMemorySize() > var2.AbsMemorySize();
              });

    int64_t accumulated_size = 0;
    int64_t size_threshold =
        static_cast<int64_t>(fraction_of_memory_size * place_to_size[place]);
    for (size_t i = 0; i < gc_vars.size() && accumulated_size < size_threshold;
         ++i) {
      partial_vars[gc_vars[i].op_].insert(gc_vars[i].name_);
      accumulated_size += gc_vars[i].AbsMemorySize();
    }
  }

  /**
   * Step 4: Combine other vars (SelectedRows, LoDTensorArray)
   */
  if (!delete_lod_tensor_only) {
    for (auto &op_vars_pair : other_vars) {
      partial_vars[op_vars_pair.first].insert(op_vars_pair.second.begin(),
                                              op_vars_pair.second.end());
    }
  }

  return partial_vars;
}

class EagerDeletionPass : public ir::Pass {
 protected:
  void ApplyImpl(ir::Graph *graph) const override;
};

void EagerDeletionPass::ApplyImpl(ir::Graph *graph) const {
  auto &var_infos = Get<MemOptVarInfoMapList>(kMemOptVarInfoMapList);

  const auto &vars = graph->Get<details::GraphVars>(details::kGraphVars);

  const auto &last_live_ops =
      Get<std::vector<LastLiveOpsOfVars>>(kLastLiveOpsOfVars);
  const auto &gcs = Get<GarbageCollectorMap>(kGarbageCollector);
  const auto &places = Get<std::vector<platform::Place>>(kAllPlaces);

  // a reverse map of last_live_ops
  //   i.e., last op --> variable names which can be deleted.
  OpToVarNameSetMap op_vars_map;
  for (auto &var_ops_map : last_live_ops) {
    for (auto &var_ops_pair : var_ops_map) {
      const std::string &var_name = var_ops_pair.first;
      for (auto *op : var_ops_pair.second.ops()) {
        op_vars_map[op].insert(var_name);
      }
    }
  }

  double memory_fraction = framework::GetEagerDeletionMemoryFraction();

  op_vars_map = ShrinkGCVars(op_vars_map, vars, places, memory_fraction);

  for (auto &pair : op_vars_map) {
    auto *op = pair.first;
    auto &var_names = pair.second;

    auto *eager_deletion_node =
        graph->CreateEmptyNode("eager_deletion", ir::Node::Type::kOperation);

    std::unordered_set<MemOptVarInfo *> var_info;
    for (auto &var_name : var_names) {
      var_info.insert(var_infos[op->GetScopeIdx()].at(var_name).get());
    }

    auto *eager_deletion_op = new details::EagerDeletionOpHandle(
        eager_deletion_node, op->GetScope(), op->GetScopeIdx(), op->GetPlace(),
        std::move(var_info), gcs.at(places[op->GetScopeIdx()]).get());

    auto it = std::find_if(
        op->Outputs().begin(), op->Outputs().end(),
        [](details::VarHandleBase *var) {
          return dynamic_cast<details::DummyVarHandle *>(var) != nullptr;
        });

    if (it != op->Outputs().end()) {
      eager_deletion_op->AddInput(*it);
    } else {
      auto *dep_var = new details::DummyVarHandle(graph->CreateControlDepVar());
      graph->Get<details::GraphDepVars>(details::kGraphDepVars)
          .emplace(dep_var);
      op->AddOutput(dep_var);
      eager_deletion_op->AddInput(dep_var);
    }

    auto *dummy_leaf =
        new details::DummyVarHandle(graph->CreateControlDepVar());
    graph->Get<details::GraphDepVars>(details::kGraphDepVars)
        .emplace(dummy_leaf);
    eager_deletion_op->AddOutput(dummy_leaf);

    eager_deletion_op->SetDeviceContext(
        places[op->GetScopeIdx()],
        platform::DeviceContextPool::Instance().Get(places[op->GetScopeIdx()]));
  }

  VLOG(10) << "FLAGS_memory_fraction_of_eager_deletion = " << memory_fraction;
  VLOG(10) << "Create " << op_vars_map.size() << " EagerDeletionOpHandle(s)";

  if (VLOG_IS_ON(10)) {
    auto vars_group_by_scope_idx = VarsGroupByScopeIdx(op_vars_map);
    for (auto &pair : vars_group_by_scope_idx) {
      VLOG(10) << "Scope " << pair.first << " has " << pair.second.size()
               << " vars";
    }
  }

  auto conditional_block_op_eager_deletion_pass =
      ir::PassRegistry::Instance().Get(
          "conditional_block_op_eager_deletion_pass");
  conditional_block_op_eager_deletion_pass->Apply(graph);

  auto while_op_eager_deletion_pass =
      ir::PassRegistry::Instance().Get("while_op_eager_deletion_pass");
  while_op_eager_deletion_pass->Apply(graph);

  auto recurrent_op_eager_deletion_pass =
      ir::PassRegistry::Instance().Get("recurrent_op_eager_deletion_pass");
  recurrent_op_eager_deletion_pass->Apply(graph);

#ifdef PADDLE_WITH_CINN
  auto share_varinfo_into_cinn_pass =
      ir::PassRegistry::Instance().Get("share_varinfo_into_cinn_pass");
  share_varinfo_into_cinn_pass->SetNotOwned(kMemOptVarInfoMapList, &var_infos);
  share_varinfo_into_cinn_pass->Apply(graph);
#endif
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(eager_deletion_pass, paddle::framework::ir::EagerDeletionPass)
    .RequirePassAttr(paddle::framework::ir::kMemOptVarInfoMapList)
    .RequirePassAttr(paddle::framework::ir::kLastLiveOpsOfVars)
    .RequirePassAttr(paddle::framework::ir::kAllPlaces)
    .RequirePassAttr(paddle::framework::ir::kGarbageCollector);

USE_PASS(conditional_block_op_eager_deletion_pass);
USE_PASS(while_op_eager_deletion_pass);
USE_PASS(recurrent_op_eager_deletion_pass);
#ifdef PADDLE_WITH_CINN
USE_PASS(share_varinfo_into_cinn_pass);
#endif
