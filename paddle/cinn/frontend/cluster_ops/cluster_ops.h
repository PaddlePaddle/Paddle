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

namespace cinn::api {



  auto shardable_axes_provider = [&] {
    auto* program = group_op->GetParentProgram();
    const auto* shape_analysis =
        &pir::ShapeAnalysisManager::Instance().Get(program);
    return frontend::MakeDefaultShardableAxesProvider(shape_analysis);
  }();

  auto cluster_policy = [&] {
    auto* program = group_op->GetParentProgram();
    const auto* shape_analysis =
        &pir::ShapeAnalysisManager::Instance().Get(program);
    return frontend::MakeLoopAlignableClusteringPolicy(shape_analysis);
  }();

ClusteringResult ClusterOps(
    const std::vector<const pir::Operation*>& ops,
    const std::shared_ptr<ShardableAxesProvider>& shardable_axes_provider,
    const std::shared_ptr<ClusteringPolicy>& clustering_policy) {
  VLOG(4) << "Initializing Inferer";
  ShardableAxesInferer inferer(shardable_axes_provider);
  VLOG(4) << "Initializing Clustering Engine";
  ClusteringEngine engine(ops, inferer, clustering_policy);
  VLOG(4) << "Engine calls ClusterOps()";
  return engine.ClusterOps();
}
}
