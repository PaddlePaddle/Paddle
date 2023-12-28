#include "paddle/cinn/common/broadcast_tree.h"
#include <optional>

namespace cinn::common {

std::optional<symbol::Broadcastable<symbol::DimExpr>> GetFirstCstrBroadcastable(const BroadcastLeaf& leaves) {
  TODO();
}

symbol::DimExpr GetCstrLhsEqRhsDimExpr(
    const symbol::Broadcastable<symbol::DimExpr>& broadcastable_condition,
    const symbol::DimExpr& dim_expr) {
  TODO();
}

symbol::DimExpr GetCstrLhsEqOneDimExpr(
    const symbol::Broadcastable<symbol::DimExpr>& broadcastable_condition,
    const symbol::DimExpr& dim_expr) {
  TODO();
}

symbol::DimExpr GetCstrRhsEqOneDimExpr(
    const symbol::Broadcastable<symbol::DimExpr>& broadcastable_condition,
    const symbol::DimExpr& dim_expr) {
  TODO();
}

typedef symbol::DimExpr (*ConvertDimExprT)(
    const symbol::Broadcastable<symbol::DimExpr>& broadcastable_condition,
    const symbol::DimExpr& dim_expr);

template <ConvertDimExprT ConvertDimExpr>
BroadcastLeaf ConvertBroadcastLeaf(
    const symbol::Broadcastable<symbol::DimExpr>& broadcastable_condition,
    const BroadcastLeaf& leaves) {
  BroadcastLeaf ret{};
  for (const auto& dim_exprs : *leaves) {
    std::vector<symbol::DimExpr> converted{};
    converted.reserve(dim_exprs.size());
    for (const auto& dim_expr : dim_exprs) {
      converted.push_back(ConvertDimExpr(broadcastable_condition, dim_expr));
    }
    ret->emplace_back(std::move(converted));
  }
  return ret;
}

BroadcastLeaf GetCstrLhsEqRhsLeaves(
    const symbol::Broadcastable<symbol::DimExpr>& broadcastable_condition,
    const BroadcastLeaf& leaves) {
  return ConvertBroadcastLeaf<&GetCstrLhsEqRhsDimExpr>(broadcastable_condition, leaves);
}

BroadcastLeaf GetCstrLhsEqOneLeaves(
    const symbol::Broadcastable<symbol::DimExpr>& broadcastable_condition,
    const BroadcastLeaf& leaves) {
  return ConvertBroadcastLeaf<&GetCstrLhsEqOneDimExpr>(broadcastable_condition, leaves);
}

BroadcastLeaf GetCstrRhsEqOneLeaves(
    const symbol::Broadcastable<symbol::DimExpr>& broadcastable_condition,
    const BroadcastLeaf& leaves) {
  return ConvertBroadcastLeaf<&GetCstrRhsEqOneDimExpr>(broadcastable_condition, leaves);
}

BroadcastBranch<BroadcastTree> ConstructBroadcastBranch(
    const symbol::Broadcastable<symbol::DimExpr>& broadcastable_condition,
    const BroadcastLeaf& leaves) {
  BroadcastLeaf cstr_lhs_eq_rhs_leaves = GetCstrLhsEqRhsLeaves(broadcastable_condition, leaves);
  BroadcastLeaf cstr_lhs_eq_one_leaves = GetCstrLhsEqOneLeaves(broadcastable_condition, leaves);
  BroadcastLeaf cstr_rhs_eq_one_leaves = GetCstrRhsEqOneLeaves(broadcastable_condition, leaves);
  return BroadcastBranch<BroadcastTree>{
    .broadcastable_condition=broadcastable_condition,
    .cstr_lhs_eq_rhs_branch=ConstructBroadcastTree(cstr_lhs_eq_rhs_leaves),
    .cstr_lhs_eq_one_branch=ConstructBroadcastTree(cstr_lhs_eq_one_leaves),
    .cstr_rhs_eq_one_branch=ConstructBroadcastTree(cstr_rhs_eq_one_leaves),
  };
}

BroadcastTree ConstructBroadcastTree(const BroadcastLeaf& leaves) {
  std::optional<symbol::Broadcastable<symbol::DimExpr>> broadcastable_condition =
      GetFirstCstrBroadcastable(leaves);
  if (!broadcastable_condition.has_value()) return leaves;
  return ConstructBroadcastBranch(broadcastable_condition.value(), leaves);
}

}