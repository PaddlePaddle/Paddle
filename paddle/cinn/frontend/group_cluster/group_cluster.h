// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#pragma once

#include "paddle/cinn/frontend/group_cluster/cluster_policy/general_topo_policy.h"
#include "paddle/cinn/frontend/group_cluster/cluster_policy/shardable_axes_policy/shardable_axes_policy.h"
#include "paddle/cinn/frontend/group_cluster/pattern_graph.h"

namespace cinn::frontend {

inline std::vector<std::vector<const pir::Operation*>> ClusterOps(
    const cinn::dialect::GroupOp& group_op) {
  const auto& ops = [&] {
    std::vector<const pir::Operation*> ops;
    for (const auto& op : group_op.GetOperators()) {
      ops.emplace_back(op);
    }
    return ops;
  }();

  VLOG(4) << "Start Cluster Ops!";
  VLOG(4) << "Input Group with size " << ops.size() << " :\n"
          << group_cluster::OpsDebugStr(ops);

  const auto* shape_analysis =
      &pir::ShapeAnalysisManager::Instance().Get(group_op->GetParentProgram());

  auto shardable_axes_policy =
      std::make_shared<group_cluster::policy::ShardableAxesPolicy>(
          ops, shape_analysis);
  auto general_topo_policy =
      std::make_shared<group_cluster::policy::GeneralTopoPolicy>();

  auto policy_manager = group_cluster::policy::PolicyManager(
      {shardable_axes_policy, general_topo_policy});

  group_cluster::PatternGraph graph(ops, policy_manager);
  return graph.ClusterOps();
}

}  // namespace cinn::frontend
