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
using cinn::common::BroadcastTree;

class BroadcastTreeInfo;

struct GroupDimExprInfo {
  common::BroadcastLeaf all_value_dim_exprs;
  std::unordered_map<pir::Value, size_t> value_to_dim_expr_idx;
};

std::optional<std::shared_ptr<BroadcastTree>> ConstructBroadcastTree(
    const common::BroadcastLeaf& leaves);

std::optional<std::shared_ptr<BroadcastTree>> GetBroadcastTreeForOptimize(
    const OpLoweringGroupPtr& group);
bool ContainBroadcastShape(const common::BroadcastLeaf& leaves);
GroupDimExprInfo GetGroupDimExprInfo(const OpLoweringGroupPtr& group);

pir::Operation* CompileBroadcastTreeToConditionBlock(
    const OpLoweringGroupPtr& group,
    const BroadcastTree& broadcast_tree,
    const std::unordered_map<pir::Value, size_t>& value_to_dim_expr_idx,
    const std::vector<pir::Value>& group_inputs,
    const std::vector<pir::Type>& output_types,
    pir::PatternRewriter& rewriter  // NOLINT
);
}  // namespace cinn::dialect::ir::details
