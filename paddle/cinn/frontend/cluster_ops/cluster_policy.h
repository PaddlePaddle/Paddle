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

#include "paddle/cinn/frontend/cluster_ops/pattern_utils.h"
#include "paddle/cinn/frontend/cluster_ops/shardable_axes_inferer.h"
#include "paddle/cinn/frontend/cluster_ops/shardable_axes_provider.h"
#include "paddle/cinn/hlir/dialect/operator/ir/manual_op.h"

namespace cinn::frontend::cluster_ops {

class ClusteringPolicy {
 public:
  virtual ~ClusteringPolicy() = default;

  using ShardableAxes4ValueT =
      std::function<std::optional<const ShardableAxes*>(pir::Value)>;

  virtual bool CanActAsSink(const ShardableAxes4ValueT& ShardableAxes4Value,
                            const api::StmtPattern<FrontendPattern>& node) = 0;

  virtual bool IsEdgeFusible(const ShardableAxes4ValueT& ShardableAxes4Value,
                             const api::StmtPattern<FrontendPattern>& src,
                             const api::StmtPattern<FrontendPattern>& dst) = 0;

  using StmtPatternPtrs = std::vector<const api::StmtPattern<FrontendPattern>*>;
  virtual ClusteringResult MakeClusteringResult(
      const std::vector<StmtPatternPtrs>& stmts) = 0;

 protected:
  ClusteringPolicy() = default;
};

std::shared_ptr<ClusteringPolicy> MakeLoopAlignableClusteringPolicy(
    const pir::ShapeConstraintIRAnalysis* shape_analysis);
}  // namespace cinn::frontend::cluster_ops
