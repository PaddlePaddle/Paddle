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

#include "paddle/cinn/frontend/cluster_ops/clustering_engine.h"

namespace cinn::api {

frontend::cluster_ops::ClusteringResult ClusterOps(
    const cinn::dialect::GroupOp& group_op) {
  const auto& ops = [&] {
    std::vector<const pir::Operation*> ops;
    for (const auto& op : *group_op.block()) {
      ops.push_back(&op);
    }
    return ops;
  }();

  auto shardable_axes_provider = [&] {
    auto* program = group_op->GetParentProgram();
    const auto* shape_analysis =
        &pir::ShapeAnalysisManager::Instance().Get(program);
    return frontend::cluster_ops::MakeDefaultShardableAxesProvider(
        shape_analysis);
  }();

  auto cluster_policy = [&] {
    auto* program = group_op->GetParentProgram();
    const auto* shape_analysis =
        &pir::ShapeAnalysisManager::Instance().Get(program);
    return frontend::cluster_ops::MakeLoopAlignableClusteringPolicy(
        shape_analysis);
  }();

  frontend::cluster_ops::ShardableAxesInferer inferer(shardable_axes_provider);
  frontend::cluster_ops::ClusteringEngine engine(ops, inferer, cluster_policy);

  return engine.ClusterOps();
}
}  // namespace cinn::api
