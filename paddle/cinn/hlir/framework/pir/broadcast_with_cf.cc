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

#include "paddle/cinn/hlir/framework/pir/broadcast_with_cf.h"
#include "paddle/cinn/common/broadcast_tree.h"
#include "paddle/cinn/hlir/dialect/operator/ir/cinn_op.h"
#include "paddle/cinn/hlir/dialect/operator/transforms/lowering_pass/utils.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"

PD_DECLARE_bool(cinn_bc_branch_optimize);

using BroadcastCond = std::pair<symbol::Broadcastable<symbol::DimExpr>,
                                OpLoweringGroup::BranchType>;

namespace {
void UpdateGroupShapeExprs(
    const OpLoweringGroupPtr& new_group,
    const OpLoweringGroupPtr& origin_group,
    const cinn::common::BroadcastLeaf& value_dim_exprs_list,
    const std::unordered_map<pir::Value, size_t>& value_to_dim_expr_idx) {
  for (const auto& [value, idx] : value_to_dim_expr_idx) {
    const auto& shape_dim_expr =
        value_dim_exprs_list->at(value_to_dim_expr_idx.at(value));
    const auto& origin_shape_or_data = origin_group->GetShapeOrDataExprs(value);
    if (origin_shape_or_data.data()) {
      std::vector<symbol::DimExpr> shape_dim_expr_shape = {
          symbol::DimExpr(static_cast<int64_t>(shape_dim_expr.size()))};
      new_group->SetShapeOrDataExprs(
          value,
          symbol::ShapeOrDataDimExprs{symbol::TensorShapeOrDataDimExprs(
              shape_dim_expr_shape, shape_dim_expr)});
    } else {
      new_group->SetShapeOrDataExprs(
          value,
          symbol::ShapeOrDataDimExprs{
              symbol::TensorShapeOrDataDimExprs(shape_dim_expr)});
    }
  }
}

OpLoweringGroupPtr CloneGroup(const OpLoweringGroupPtr& group,
                              const std::string& name_suffix) {
  return group->Clone(name_suffix);
}

void SetBroadcastLeafGroup(
    const OpLoweringGroupPtr& origin_group,
    const cinn::common::BroadcastLeaf& value_dim_exprs_list,
    const std::unordered_map<pir::Value, size_t>& value_to_dim_expr_idx,
    std::vector<OpLoweringGroupPtr>* group_list,
    const std::vector<BroadcastCond>& broadcast_conditions) {
  auto new_group =
      CloneGroup(origin_group, std::to_string(group_list->size() + 1));
  new_group->SetIsBroadcastLeaf(true);
  new_group->SetBroadcastConditions(broadcast_conditions);
  PADDLE_ENFORCE_EQ(
      origin_group->ops().size(),
      new_group->ops().size(),
      ::common::errors::InvalidArgument(
          "The size of origin group ops and new group ops is not equal,"
          "where the size of origin group ops:%d but the size of new group "
          "ops:%d.",
          origin_group->ops().size(),
          new_group->ops().size()));

  UpdateGroupShapeExprs(
      new_group, origin_group, value_dim_exprs_list, value_to_dim_expr_idx);

  group_list->emplace_back(new_group);
}

// Visit broadcast_tree by dfs
void ConstructBroadcastGroupList(
    const cinn::common::BroadcastTree& broadcast_tree,
    const OpLoweringGroupPtr& origin_group,
    std::vector<BroadcastCond>* current_branch_conditions,
    const std::unordered_map<pir::Value, size_t>& value_to_dim_expr_idx,
    std::vector<OpLoweringGroupPtr>* group_list) {
  if (broadcast_tree.Has<cinn::common::BroadcastLeaf>()) {
    const auto& broadcast_leaf =
        broadcast_tree.Get<cinn::common::BroadcastLeaf>();
    SetBroadcastLeafGroup(origin_group,
                          broadcast_leaf,
                          value_to_dim_expr_idx,
                          group_list,
                          *current_branch_conditions);
  } else {
    const auto& branch =
        broadcast_tree
            .Get<cinn::common::BroadcastBranch<cinn::common::BroadcastTree>>();
    const symbol::Broadcastable<symbol::DimExpr> broadcastable_condition =
        branch.Get<0>();

    // lhs == rhs
    current_branch_conditions->emplace_back(
        broadcastable_condition, OpLoweringGroup::BranchType::LHS_EQ_RHS);
    ConstructBroadcastGroupList(branch.Get<1>(),
                                origin_group,
                                current_branch_conditions,
                                value_to_dim_expr_idx,
                                group_list);
    current_branch_conditions->pop_back();

    // lhs != rhs && lhs == 1
    current_branch_conditions->emplace_back(
        broadcastable_condition, OpLoweringGroup::BranchType::LHS_EQ_ONE);
    ConstructBroadcastGroupList(branch.Get<2>(),
                                origin_group,
                                current_branch_conditions,
                                value_to_dim_expr_idx,
                                group_list);
    current_branch_conditions->pop_back();

    // lhs != rhs && rhs == 1
    current_branch_conditions->emplace_back(
        broadcastable_condition, OpLoweringGroup::BranchType::RHS_EQ_ONE);
    ConstructBroadcastGroupList(branch.Get<3>(),
                                origin_group,
                                current_branch_conditions,
                                value_to_dim_expr_idx,
                                group_list);
    current_branch_conditions->pop_back();
  }
}

struct GroupDimExprInfo {
  cinn::common::BroadcastLeaf all_value_dim_exprs;
  std::unordered_map<pir::Value, size_t> value_to_dim_expr_idx;
};

GroupDimExprInfo GetGroupDimExprInfo(const OpLoweringGroupPtr& group) {
  std::unordered_set<pir::Value> value_view;
  group->WalkOps([&group, &value_view](pir::Operation* op) {
    for (size_t i = 0; i < op->num_operands(); ++i) {
      value_view.insert(op->operand_source(i));
    }
    for (size_t i = 0; i < op->num_results(); ++i) {
      value_view.insert(op->result(i));
    }
  });

  GroupDimExprInfo group_dim_expr_info;
  for (const auto& value : value_view) {
    const auto& shape_dim_expr = group->GetShapeOrDataExprs(value);
    const auto& data_shape = shape_dim_expr.data();
    if (data_shape) {
      group_dim_expr_info.all_value_dim_exprs->push_back(*data_shape);
    } else {
      group_dim_expr_info.all_value_dim_exprs->push_back(
          shape_dim_expr.shape());
    }
    group_dim_expr_info.value_to_dim_expr_idx[value] =
        group_dim_expr_info.all_value_dim_exprs->size() - 1;
  }
  return group_dim_expr_info;
}

bool ContainBroadcastShape(const cinn::common::BroadcastLeaf& leaves) {
  std::optional<symbol::Broadcastable<symbol::DimExpr>>
      broadcastable_condition = cinn::common::GetFirstCstrBroadcastable(leaves);
  return broadcastable_condition.has_value();
}

std::optional<std::shared_ptr<cinn::common::BroadcastTree>>
ConstructBroadcastTree(const cinn::common::BroadcastLeaf& leaves) {
  VLOG(6) << "before constructed. broadcast-leaf: \n"
          << ToTxtString(cinn::common::BroadcastTree(leaves));
  int num_of_leaves = 0;
  auto broadcast_tree = std::make_shared<cinn::common::BroadcastTree>(
      cinn::common::ConstructBroadcastTree(cinn::common::BroadcastLeaf(leaves),
                                           &num_of_leaves));
  if (num_of_leaves > FLAGS_pir_broadcast_tree_limit) {
    LOG(WARNING) << "the number of leaf nodes in broadcast tree exceeds "
                    "limit.";
    return std::nullopt;
  }
  VLOG(4) << "broadcast-tree: \n" << ToTxtString(*broadcast_tree);
  return broadcast_tree;
}

}  // namespace

namespace cinn::hlir::framework::pir {
std::optional<std::vector<OpLoweringGroupPtr>> GetBroadcastGroupListForOptimize(
    const OpLoweringGroupPtr& group) {
  if (!FLAGS_cinn_bc_branch_optimize) return std::nullopt;

  GroupDimExprInfo group_dim_expr_info = GetGroupDimExprInfo(group);
  if (!ContainBroadcastShape(group_dim_expr_info.all_value_dim_exprs))
    return std::nullopt;

  const auto& optional_broadcast_tree =
      ConstructBroadcastTree(group_dim_expr_info.all_value_dim_exprs);

  if (!optional_broadcast_tree.has_value()) return std::nullopt;

  const auto& broadcast_tree = optional_broadcast_tree.value();

  const auto& ChangeBroadcastTreeToGroupList =
      [&]() -> std::vector<OpLoweringGroupPtr> {
    std::vector<OpLoweringGroupPtr> group_list;
    std::vector<BroadcastCond> current_branch_conditions;
    const auto& value_to_dim_expr_idx =
        group_dim_expr_info.value_to_dim_expr_idx;
    ConstructBroadcastGroupList(*broadcast_tree,
                                group,
                                &current_branch_conditions,
                                value_to_dim_expr_idx,
                                &group_list);
    return group_list;
  };

  return ChangeBroadcastTreeToGroupList();
}
}  // namespace cinn::hlir::framework::pir
