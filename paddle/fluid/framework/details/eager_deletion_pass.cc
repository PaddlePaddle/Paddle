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
#include "paddle/fluid/framework/details/eager_deletion_pass.h"
#include "paddle/fluid/framework/details/multi_devices_helper.h"
#include "paddle/fluid/framework/ir/graph_helper.h"

DEFINE_double(fraction_of_eager_deletion, 1.0, "Fraction of eager deletion");
DEFINE_bool(eager_delete_tensor_only, false, "");

namespace paddle {
namespace framework {
namespace details {

namespace {  // NOLINT
using OpToVarNameSetMap =
    std::unordered_map<ComputationOpHandle *, std::unordered_set<std::string>>;
}  // NOLINT

static bool IsLoDTensor(VarDesc *var) {
  return var->Proto()->type().type() == proto::VarType::LOD_TENSOR;
}

static int64_t GetNumel(const GraphVars &vars, const std::string &var_name,
                        size_t scope_idx) {
  auto *var_desc = TryGetLatestVarDesc(vars[scope_idx].at(var_name));
  PADDLE_ENFORCE(IsLoDTensor(var_desc));
  auto dims = var_desc->GetShape();
  return std::accumulate(dims.begin(), dims.end(), static_cast<int64_t>(1),
                         std::multiplies<int64_t>());
}

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

static OpToVarNameSetMap ShrinkGCVars(const OpToVarNameSetMap &m,
                                      const GraphVars &vars,
                                      double fraction_of_memory_size,
                                      bool delete_lod_tensor_only = false) {
  // Do not perform gc
  if (fraction_of_memory_size <= 0.0) return {};

  // Perform complete gc
  if (fraction_of_memory_size >= 1.0) {
    if (delete_lod_tensor_only) {
      OpToVarNameSetMap lod_tensors, other_vars;
      SplitIntoLoDTensorAndNonLoDTensorVars(m, vars, &lod_tensors, &other_vars);
      return lod_tensors;
    } else {
      return m;
    }
  }

  // Perform partial gc
  OpToVarNameSetMap lod_tensors, other_vars;
  SplitIntoLoDTensorAndNonLoDTensorVars(m, vars, &lod_tensors, &other_vars);

  using TupleType = std::tuple<std::string, ComputationOpHandle *, int64_t>;

  std::unordered_map<size_t, std::vector<TupleType>> place_to_vars;
  std::unordered_map<size_t, int64_t> total_memory_size;
  for (auto &op_vars_pair : lod_tensors) {
    auto scope_idx = op_vars_pair.first->GetScopeIdx();
    int64_t size = 0;
    for (auto &var_name : op_vars_pair.second) {
      auto var_size = GetNumel(vars, var_name, scope_idx);
      size += std::abs(var_size);
      place_to_vars[scope_idx].emplace_back(var_name, op_vars_pair.first,
                                            var_size);
    }
    total_memory_size.emplace(scope_idx, size);
  }

  for (auto &pair : place_to_vars) {
    std::sort(pair.second.begin(), pair.second.end(),
              [](const TupleType &t1, const TupleType &t2) {
                return std::abs(std::get<2>(t1)) > std::abs(std::get<2>(t2));
              });
  }

  OpToVarNameSetMap ret;
  for (auto &pair : place_to_vars) {
    auto desired_delete_size = static_cast<int64_t>(
        fraction_of_memory_size * total_memory_size.at(pair.first));
    int64_t cur_size = 0;
    for (size_t i = 0; i < pair.second.size() && cur_size < desired_delete_size;
         ++i) {
      auto &var_name = std::get<0>(pair.second[i]);
      auto *op = std::get<1>(pair.second[i]);
      cur_size += std::get<2>(pair.second[i]);
      ret[op].insert(var_name);
    }
  }

  if (!delete_lod_tensor_only) {
    for (auto &op_vars_pair : other_vars) {
      for (auto &var_name : op_vars_pair.second) {
        ret[op_vars_pair.first].insert(var_name);
      }
    }
  }

  return ret;
}

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

  op_vars_map =
      ShrinkGCVars(op_vars_map, vars, FLAGS_fraction_of_eager_deletion,
                   FLAGS_eager_delete_tensor_only);

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

  VLOG(10) << "FLAGS_fraction_of_eager_deletion = "
           << FLAGS_fraction_of_eager_deletion;
  VLOG(10) << "FLAGS_eager_delete_tensor_only = "
           << FLAGS_eager_delete_tensor_only;
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
