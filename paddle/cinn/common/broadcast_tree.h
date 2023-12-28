#pragma once

#include "paddle/cinn/adt/tree.h"
#include "paddle/pir/dialect/shape/utils/dim_expr.h"

namespace cinn::common {
  
template <typename T>
struct BroadcastBranch {
  symbol::Broadcastable<symbol::DimExpr> broadcastable_condition;
  T cstr_lhs_eq_rhs_branch;
  T cstr_lhs_eq_one_branch;
  T cstr_rhs_eq_one_branch;
};

using BroadcastLeaf = adt::List<std::vector<symbol::DimExpr>>;

using BroadcastTree = adt::Tree<BroadcastBranch, BroadcastLeaf>;

BroadcastTree ConstructBroadcastTree(const BroadcastLeaf& leaves);

}