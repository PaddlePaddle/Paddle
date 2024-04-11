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
#include "paddle/cinn/frontend/group_cluster/cluster_policy/relative_judge_policy.h"
#include "paddle/cinn/frontend/group_cluster/cluster_policy/shardable_axes_policy/shardable_axes_policy.h"
#include "paddle/cinn/frontend/group_cluster/pattern_graph.h"

namespace cinn::frontend {

inline std::vector<group_cluster::PatternNodePtr> ClusterOps(
    const std::vector<pir::Operation*>& origin_ops,
    bool with_horizontal_fusion = false) {
  CHECK_GT(origin_ops.size(), 0);
  VLOG(4) << "Start Cluster Ops!";
  VLOG(4) << "Input Group with size " << origin_ops.size() << " :\n"
          << group_cluster::OpsDebugStr(origin_ops);

  std::vector<pir::Value> outputs;
  const auto& ops = [&] {
    std::vector<pir::Operation*> ops;
    for (const auto& op : origin_ops) {
      if (op->name() == "cf.yield") {  // just skip cf.yield.
        for (auto& operand : op->operands()) {
          outputs.push_back(operand.source());
        }
        continue;
      }
      ops.emplace_back(op);
    }
    return ops;
  }();

  pir::Program* program = ops.at(0)->GetParentProgram();

  const auto* shape_analysis =
      &pir::ShapeAnalysisManager::Instance().Get(program);

  // const auto& shardable_axes_policy =
  // std::make_shared<group_cluster::policy::RelativeJudgePolicy>(
  // ops, shape_analysis);
  VLOG(4) << "Start Create Policies and PolicyManager!";
  const auto& relative_judge_policy =
      std::make_shared<group_cluster::policy::RelativeJudgePolicy>(
          ops, shape_analysis);

  const auto& general_topo_policy =
      std::make_shared<group_cluster::policy::GeneralTopoPolicy>();

  auto policy_manager = group_cluster::policy::PolicyManager(
      {relative_judge_policy, general_topo_policy});

  auto topo_manager = group_cluster::policy::PolicyManager(
      {relative_judge_policy, general_topo_policy});

  VLOG(4) << "Start Create PatternGraph";
  group_cluster::PatternGraph graph(ops, outputs, policy_manager, topo_manager);
  auto result = graph.ClusterOps(with_horizontal_fusion);

  VLOG(4) << "End Cluster Ops! result size:" << result.size();
  for (const auto& node : result) {
    VLOG(4) << "\n"
            << node->DebugStr() << "\n"
            << group_cluster::StmtPatternDebugStr(node->stmt_pattern_);
  }

  return result;
}

}  // namespace cinn::frontend
