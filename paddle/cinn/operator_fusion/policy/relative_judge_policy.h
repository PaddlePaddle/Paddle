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
#include <functional>
#include "paddle/cinn/operator_fusion/policy/dim_relation.h"
#include "paddle/cinn/operator_fusion/policy/policy_manager.h"
#include "paddle/cinn/operator_fusion/policy/shardable_axes_base.h"
#include "paddle/cinn/operator_fusion/utils.h"
#include "paddle/common/enforce.h"

namespace cinn::fusion {

template <typename T>
class RelativeJudgePolicy final : public Policy<T> {
 public:
  RelativeJudgePolicy(const std::vector<pir::Operation*>& ops,
                      pir::ShapeConstraintIRAnalysis* shape_analysis)
      : axes_info_(ops, shape_analysis) {
    VLOG(4) << "[relative_judge_policy] Start AnalysisIndexExprRelation.";
    index_expr_map_ = AnalysisIndexExprRelation(ops);
    VLOG(4) << "[relative_judge_policy] End AnalysisIndexExprRelation.";
  }

  ShardableAxesInfoManager& GetAxesInfoManager() { return axes_info_; }

  bool CanFuse(const PatternNodePtr<T>& upstream,
               const PatternNodePtr<T>& downstream) override;

  std::string Name() { return "RelativeJudgePolicy"; }

  std::vector<size_t> GetFakeReduceIterIdx(
      const PatternNodePtr<T>& upstream,
      const PatternNodePtr<T>& downstream) override;

  bool IsRelated(DimUsage in, DimUsage out) {
    return index_expr_map_[in].count(out) == 1;
  }

 private:
  DimUsageRelation index_expr_map_;
  ShardableAxesInfoManager axes_info_;
  bool ReduceTreeGrownCanMerge(const PatternNodePtr<T>&,
                               const PatternNodePtr<T>&);
  bool ReducePlusTrivialCanMerge(const PatternNodePtr<T>&,
                                 const PatternNodePtr<T>&);
  std::pair<std::vector<DimUsage>, std::vector<DimUsage>>
  SplitFirstIfRelatedBySecond(const std::vector<DimUsage>& targets,
                              const std::vector<DimUsage>& related_with);
  std::optional<ReducePattern<T>> GetDownstreamFromCandidate(
      const ReducePattern<T>& upstream,
      const std::vector<ReducePattern<T>>& candidates);
  bool IsDownstreamStmtDependReduceOp(pir::Operation* reduce,
                                      const StmtPattern<T>& downstream);
};

}  // namespace cinn::fusion
