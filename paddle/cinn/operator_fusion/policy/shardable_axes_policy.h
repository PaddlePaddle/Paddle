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
#include "paddle/cinn/operator_fusion/policy/policy_manager.h"
#include "paddle/cinn/operator_fusion/policy/shardable_axes_base.h"

namespace cinn::fusion {

template <typename T>
class ShardableAxesRRFusePolicy final : public Policy<T> {
 public:
  ShardableAxesRRFusePolicy(
      const std::vector<pir::Operation*>& ops,         // NOLINT
      pir::ShapeConstraintIRAnalysis* shape_analysis)  // NOLINT
      : axes_info_(ops, shape_analysis) {}
  bool CanFuse(const PatternNodePtr<T>& upstream,
               const PatternNodePtr<T>& downstream) override;
  std::string Name() { return "ShardableAxesRRFusePolicy"; }

 private:
  bool ReduceTreeGrownCanMerge(const PatternNodePtr<T>&,
                               const PatternNodePtr<T>&);
  std::optional<ReducePattern<T>> GetDownstreamFromCandidate(
      const ReducePattern<T>& upstream,
      const std::vector<ReducePattern<T>>& candidates);
  ShardableAxesInfoManager axes_info_;
  bool IsDownstreamStmtDependReduceOp(pir::Operation* reduce,
                                      const StmtPattern<T>& downstream);
};

}  // namespace cinn::fusion
