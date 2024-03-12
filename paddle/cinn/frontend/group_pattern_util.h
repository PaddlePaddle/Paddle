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

#include "paddle/cinn/frontend/group_pattern.h"
#include "paddle/cinn/hlir/dialect/operator/ir/manual_op.h"

namespace cinn::frontend {

struct OpsClusteringSpec {
  // shardable_dim_size(reduce_op) = size(reduce_op.result(0)).
  // The infered_shardable_dim_size(reduce_op) may be less than shardable_dim_size(reduce_op) because:
  //   infered_shardable_dim_size(reduce_op) =
  //     min(shardable_dim_size(reduce_op), infered_shardable_dim_size(downstreams(reduce_op)))
  const size_t reduce_op_minimal_infered_shardable_dim_size;
};

std::vector<ConditionalGroupPattern> ClusterIntoGroupPatternsFromOpList(
    const pir::ShapeConstraintIRAnalysis* shape_analysis,
    const std::vector<const pir::Operation*>& ops,
    const OpsClusteringSpec& clustering_spec);

GroupPattern GenerateGroupPatternFromOpList(
    const std::vector<const pir::Operation*>& ops);

std::unordered_map<pir::Value, ShardableAxes> InferShardableAxes(
    const std::shared_ptr<std::unordered_set<const pir::Operation*>>& ops);

}  // namespace cinn::frontend
