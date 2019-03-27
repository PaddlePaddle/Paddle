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

namespace paddle {
namespace framework {
namespace details {

// op -> variables which can be deleted after op runs
using OpToVarNameSetMap =
    std::unordered_map<ComputationOpHandle *, std::unordered_set<std::string>>;

// Check whether the variable is LoDTensor based on static VarDesc info
static bool IsLoDTensor(VarDesc *var) {
  return var->Proto()->type().type() == proto::VarType::LOD_TENSOR;
}

// Get memory size of LoDTensor
static int64_t GetMemorySize(
    const std::unordered_map<std::string, std::vector<VarHandle *>> &vars,
    const std::string &var_name) {
  auto *var_desc = TryGetLatestVarDesc(vars.at(var_name));
  PADDLE_ENFORCE_NOT_NULL(var_desc);
  PADDLE_ENFORCE(IsLoDTensor(var_desc));
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
    const OpToVarNameSetMap &m, const GraphVars &vars,
    OpToVarNameSetMap *lod_tensors, OpToVarNameSetMap *other_vars) {
  lod_tensors->clear();
  other_vars->clear();

  for (auto &op_vars_pair : m) {
    for (auto &var_name : op_vars_pair.second) {
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
            ComputationOpHandle *op, size_t scope_idx)
      : name_(name),
        memory_size_(memory_size),
        op_(op),
        scope_idx_(scope_idx) {}

  std::string name_;         // variable name
  int64_t memory_size_;      // memory size
  ComputationOpHandle *op_;  // op after which the variable could be deleted
  size_t scope_idx_;         // scope index where the variable locates

  int64_t AbsMemorySize() const { return std::abs(memory_size_); }
};

// Delete delete_lod_tensor_only is not used currently
static OpToVarNameSetMap ShrinkGCVars(
    const OpToVarNameSetMap &m, const GraphVars &vars,
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
  std::unique_ptr<ir::Graph> ApplyImpl(
      std::unique_ptr<ir::Graph> graph) const override;
};

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
  OpToVarNameSetMap op_vars_map;
  for (auto &var_ops_map : last_live_ops) {
    for (auto &var_ops_pair : var_ops_map) {
      const std::string &var_name = var_ops_pair.first;
      for (auto *op : var_ops_pair.second) {
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

  VLOG(10) << "FLAGS_memory_fraction_of_eager_deletion = " << memory_fraction;
  VLOG(10) << "Create " << op_vars_map.size() << " EagerDeletionOpHandle(s)";

  auto loop_op_eager_deletion_pass =
      ir::PassRegistry::Instance().Get("loop_op_eager_deletion_pass");
  return loop_op_eager_deletion_pass->Apply(std::move(graph));
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

USE_PASS(loop_op_eager_deletion_pass);
