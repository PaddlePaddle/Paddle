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
#include "paddle/cinn/frontend/group_cluster/cluster_policy/policy_manager.h"
#include "paddle/cinn/frontend/group_cluster/cluster_policy/shardable_axes_policy/shardable_axes_base.h"

namespace cinn::frontend::group_cluster::policy {

class ShardableAxesRRFusePolicy final : public Policy {
 public:
  ShardableAxesRRFusePolicy(
      const std::vector<pir::Operation*>& ops,               // NOLINT
      const pir::ShapeConstraintIRAnalysis* shape_analysis)  // NOLINT
      : axes_info_(ops, shape_analysis) {}
  bool CanFuse(const PatternNodePtr& upstream,
               const PatternNodePtr& downstream) override;
  std::string Name() { return "ShardableAxesRRFusePolicy"; }

 private:
  bool ReduceTreeGrownCanMerge(const PatternNodePtr&, const PatternNodePtr&);
  std::optional<ReducePattern> GetDownstreamFromCandidate(
      const ReducePattern& upstream,
      const std::vector<ReducePattern>& candidates);
  ShardableAxesInfoManager axes_info_;
  bool IsDownstreamStmtDependReduceOp(pir::Operation* reduce,
                                      const StmtPattern& downstream);
};

}  // namespace cinn::frontend::group_cluster::policy
