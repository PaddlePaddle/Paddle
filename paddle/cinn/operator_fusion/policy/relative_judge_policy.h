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
#include "paddle/cinn/operator_fusion/pattern_node.h"
#include "paddle/cinn/operator_fusion/pir_graph_analyzing/dim_relation.h"
#include "paddle/cinn/operator_fusion/pir_graph_analyzing/shardable_axes_base.h"
#include "paddle/cinn/operator_fusion/policy/policy_base.h"
#include "paddle/cinn/operator_fusion/utils.h"
#include "paddle/common/enforce.h"

namespace cinn::fusion {

class RelativeJudgePolicy final : public PolicyBase {
 public:
  static constexpr PolicyKind Kind = PolicyKind::RelativeJudge;
  RelativeJudgePolicy(const std::vector<pir::Operation*>& ops,
                      pir::ShapeConstraintIRAnalysis* shape_analysis)
      : axes_info_(ops, shape_analysis) {
    VLOG(4) << "[relative_judge_policy] Start AnalysisIndexExprRelation.";
    index_expr_map_ = AnalysisIndexExprRelation(ops);
    VLOG(4) << "[relative_judge_policy] End AnalysisIndexExprRelation.";
  }
  bool CanFuse(const PatternNodePtr& upstream,
               const PatternNodePtr& downstream);

  ShardableAxesInfoManager& GetAxesInfoManager() { return axes_info_; }

  std::string Name() { return "RelativeJudgePolicy"; }

  std::vector<size_t> GetFakeReduceIterIdx(const PatternNodePtr& upstream,
                                           const PatternNodePtr& downstream);

  bool IsRelated(DimUsage in, DimUsage out) {
    return index_expr_map_[in].count(out) == 1;
  }

 private:
  DimUsageRelation index_expr_map_;
  ShardableAxesInfoManager axes_info_;
  bool ReduceTreeGrownCanMerge(const PatternNodePtr&, const PatternNodePtr&);
  bool ReducePlusTrivialCanMerge(const PatternNodePtr&, const PatternNodePtr&);
  std::pair<std::vector<DimUsage>, std::vector<DimUsage>>
  SplitFirstIfRelatedBySecond(const std::vector<DimUsage>& targets,
                              const std::vector<DimUsage>& related_with);
  std::optional<ReducePattern> GetDownstreamFromCandidate(
      const ReducePattern& upstream,
      const std::vector<ReducePattern>& candidates);
  bool IsDownstreamStmtDependReduceOp(pir::Operation* reduce,
                                      const StmtPattern& downstream);
};

}  // namespace cinn::fusion
