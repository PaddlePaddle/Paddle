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
#include "paddle/cinn/common/broadcast_tree.h"
#include "paddle/cinn/hlir/dialect/operator/transforms/lowering_pass/utils.h"
#include "paddle/pir/include/pattern_rewrite/pattern_match.h"

namespace cinn::dialect::ir::details {
using SharedGroupHasher = OpLoweringGroup::SharedGroupHasher;
using SharedGroupComparator = OpLoweringGroup::SharedGroupComparator;
class BroadcastTreeInfo;

using BroadcastTreeInfoMap =
    std::unordered_map<OpLoweringGroupPtr,
                       std::shared_ptr<BroadcastTreeInfo>,
                       SharedGroupHasher,
                       SharedGroupComparator>;

struct GroupDimExprInfo {
  adt::List<std::vector<symbol::DimExpr>> all_value_dim_exprs;
  std::unordered_map<pir::Value, size_t> value_to_dim_expr_idx;
};

class BroadcastTreeInfo final {
 public:
  explicit BroadcastTreeInfo(
      const adt::List<std::vector<symbol::DimExpr>>& leaves) {
    ConstructBroadcastTree(leaves);
  }
  const std::shared_ptr<cinn::common::BroadcastTree>& GetBroadcastTree() const;

 private:
  void ConstructBroadcastTree(
      const adt::List<std::vector<symbol::DimExpr>>& leaves);

  std::shared_ptr<cinn::common::BroadcastTree> broadcast_tree_;
};

bool NeedBroadcastWithCF(const OpLoweringGroupPtr& group);
bool NeedBroadcastWithCF(const adt::List<std::vector<symbol::DimExpr>>& leaves);
GroupDimExprInfo GetGroupDimExprInfo(const OpLoweringGroupPtr& group);

pir::Operation* CompileBroadcastTreeToConditionBlock(
    const OpLoweringGroupPtr& group,
    const BroadcastTreeInfo& broadcast_tree_info,
    pir::ShapeConstraintIRAnalysis& shape_analysis,  // NOLINT
    const std::unordered_map<pir::Value, size_t>& value_to_dim_expr_idx,
    const std::vector<pir::Value>& group_inputs,
    const std::vector<pir::Type>& output_types,
    pir::PatternRewriter& rewriter  // NOLINT
);
}  // namespace cinn::dialect::ir::details
