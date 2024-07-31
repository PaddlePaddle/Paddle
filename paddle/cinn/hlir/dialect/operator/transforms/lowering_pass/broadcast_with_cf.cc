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

#include "paddle/cinn/hlir/dialect/operator/transforms/lowering_pass/broadcast_with_cf.h"
#include "paddle/cinn/hlir/dialect/operator/ir/cinn_op.h"
#include "paddle/cinn/hlir/dialect/operator/ir/generate_shape_util.h"
#include "paddle/cinn/hlir/dialect/operator/transforms/lowering_pass/utils.h"
#include "paddle/cinn/hlir/dialect/runtime/ir/jit_kernel_op.h"
#include "paddle/fluid/pir/dialect/operator/ir/control_flow_op.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/pir/include/dialect/control_flow/ir/cf_op.h"
#include "paddle/pir/include/pattern_rewrite/frozen_rewrite_pattern_set.h"

using OpLoweringGroup = cinn::hlir::framework::pir::OpLoweringGroup;
using OpLoweringGroupPtr = std::shared_ptr<OpLoweringGroup>;
using BroadcastCond = std::pair<symbol::Broadcastable<symbol::DimExpr>,
                                OpLoweringGroup::BranchType>;
using cinn::dialect::ir::details::CompileBroadcastGroupsAsOpAttribute;
using cinn::dialect::ir::details::GetBlockOutsideInput;

PD_DECLARE_bool(cinn_bc_branch_optimize);

namespace {
std::vector<pir::Value> GetOpOuputValues(const pir::Operation* op) {
  std::vector<pir::Value> outputs;
  outputs.reserve(op->num_results());
  for (size_t i = 0; i < op->num_results(); ++i) {
    outputs.push_back(op->result(i));
  }
  return outputs;
}

using ShapeOrDataDimExprs4ValueT =
    std::function<const symbol::ShapeOrDataDimExprs&(pir::Value)>;

static bool SameInputOutputShape(
    paddle::dialect::ExpandOp expand_op,
    const ShapeOrDataDimExprs4ValueT& ShapeOrDataDimExprs4Value) {
  const auto& x = ShapeOrDataDimExprs4Value(expand_op.x());
  const auto& shape = ShapeOrDataDimExprs4Value(expand_op.shape());
  const auto& out = ShapeOrDataDimExprs4Value(expand_op.out());
  if (x.data().has_value()) return false;
  if (!shape.data().has_value()) return false;
  if (out.data().has_value()) return false;
  CHECK(shape.data().value() == out.shape());
  return x.shape() == out.shape();
}

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
                              const int& group_idx) {
  return group->Clone(group_idx);
}

void SetBroadcastLeafGroup(
    const OpLoweringGroupPtr& origin_group,
    const cinn::common::BroadcastLeaf& value_dim_exprs_list,
    const std::unordered_map<pir::Value, size_t>& value_to_dim_expr_idx,
    std::vector<OpLoweringGroupPtr>* group_list,
    const std::vector<BroadcastCond>& broadcast_conditions) {
  auto new_group = CloneGroup(origin_group, group_list->size() + 1);
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
}  // namespace

namespace cinn::dialect::ir::details {

std::optional<std::shared_ptr<BroadcastTree>> ConstructBroadcastTree(
    const cinn::common::BroadcastLeaf& leaves) {
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
  for (auto value : value_view) {
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

std::optional<std::shared_ptr<BroadcastTree>> GetBroadcastTreeForOptimize(
    const OpLoweringGroupPtr& group) {
  if (!FLAGS_cinn_bc_branch_optimize) return std::nullopt;

  const common::BroadcastLeaf leaves = [&]() {
    // NOTE(dev): Need UpdateShapeOrDataExprs firstly and the logic
    // will be migated into BucketLower later.
    UpdateGroupShapeOrDataExprs(const_cast<OpLoweringGroupPtr&>(group));
    GroupDimExprInfo group_dim_expr_info = GetGroupDimExprInfo(group);
    return group_dim_expr_info.all_value_dim_exprs;
  }();

  if (!ContainBroadcastShape(leaves)) return std::nullopt;

  return ConstructBroadcastTree(leaves);
}

bool ContainBroadcastShape(const cinn::common::BroadcastLeaf& leaves) {
  std::optional<symbol::Broadcastable<symbol::DimExpr>>
      broadcastable_condition = cinn::common::GetFirstCstrBroadcastable(leaves);
  return broadcastable_condition.has_value();
}

std::unordered_map<std::string, pir::Attribute> CompileBroadcastTree(
    const OpLoweringGroupPtr& group,
    const BroadcastTree& broadcast_tree,
    const std::unordered_map<pir::Value, size_t>& value_to_dim_expr_idx) {
  auto ShapeOrDataDimExprs4Value =
      [&group](pir::Value value) -> const symbol::ShapeOrDataDimExprs& {
    return group->GetShapeOrDataExprs(value);
  };
  // 1. broadcast tree to condition op
  VLOG(4) << "broadcast tree to condition op";
  std::vector<OpLoweringGroupPtr> group_list;
  std::vector<BroadcastCond> current_branch_conditions;
  ConstructBroadcastGroupList(broadcast_tree,
                              group,
                              &current_branch_conditions,
                              value_to_dim_expr_idx,
                              &group_list);

  // 2. compile condition block to jit_kernel_op
  auto op_attr = CompileBroadcastGroupsAsOpAttribute(group_list, group);

  return op_attr;
}
}  // namespace cinn::dialect::ir::details
