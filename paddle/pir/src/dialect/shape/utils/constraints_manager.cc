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

#include "paddle/pir/include/dialect/shape/utils/constraints_manager.h"
#include "paddle/pir/include/dialect/shape/utils/dim_expr_builder.h"
#include "paddle/pir/include/dialect/shape/utils/dim_expr_util.h"

namespace symbol {
void ConstraintsManager::AddEqCstr(const DimExpr& lhs, const DimExpr& rhs) {
  if (lhs == rhs) {
    return;
  }
  equals_.Union(lhs, rhs);
  VLOG(8) << "AddEqCstr the constraint: " << lhs << " == " << rhs;

  SubstituteDimExprInConstraint(lhs, rhs);
}

void ConstraintsManager::AddBroadcastableCstr(const DimExpr& lhs,
                                              const DimExpr& rhs) {
  broadcastables_.push_back(Broadcastable<DimExpr>(lhs, rhs));

  bool lhs_gtone = IsDimExprGTOne(lhs);
  bool rhs_gtone = IsDimExprGTOne(rhs);
  DimExprBuilder builder;
  if (lhs_gtone && rhs_gtone) {
    AddEqCstr(lhs, rhs);
  } else if (lhs_gtone) {
    AddEqCstr(lhs, builder.Broadcast(lhs, rhs));
  } else if (rhs_gtone) {
    AddEqCstr(rhs, builder.Broadcast(lhs, rhs));
  }
}

void ConstraintsManager::AddGTOneCstr(const DimExpr& dim_expr) {
  gtones_.insert(dim_expr);

  auto InsertEqualCstr = [&](const DimExpr& gtone_dim_expr,
                             const DimExpr& other_dim_expr) {
    if (IsDimExprGTOne(other_dim_expr)) {
      AddEqCstr(gtone_dim_expr, other_dim_expr);
    } else {
      DimExprBuilder builder;
      AddEqCstr(gtone_dim_expr,
                builder.Broadcast(gtone_dim_expr, other_dim_expr));
    }
  };

  for (auto broadcastable : broadcastables_) {
    if (broadcastable->lhs == dim_expr) {
      InsertEqualCstr(dim_expr, broadcastable->rhs);
    } else if (broadcastable->rhs == dim_expr) {
      InsertEqualCstr(dim_expr, broadcastable->lhs);
    }
  }
}

bool ConstraintsManager::IsDimExprGTOne(const DimExpr& dim_expr) {
  return gtones_.count(dim_expr) || IsDimExprGreaterThanOne(dim_expr);
}

bool ConstraintsManager::IsDimExprEqual(const DimExpr& lhs,
                                        const DimExpr& rhs) {
  if (equals_.IsConnect(lhs, rhs)) {
    return true;
  }
  return false;
}

void ConstraintsManager::PrintDimExprClusters(std::stringstream& ss) {
  std::vector<std::vector<DimExpr>> equal_clusters = equals_.Clusters();
  ss << "to print DimExpr equal constraints clusters" << std::endl;
  for (auto equal_cluster : equal_clusters) {
    ss << "Equal Clusters {" << std::endl;
    for (auto dim_expr : equal_cluster) {
      ss << dim_expr << std::endl;
    }
    ss << "}" << std::endl;
  }
}

namespace {

int GetDimExprPriority(const symbol::DimExpr& dim_expr) {
  return std::visit(
      symbol::Overloaded{
          [&](std::int64_t) { return 0; },
          [&](const std::string&) { return 1; },
          [&](const symbol::Negative<symbol::DimExpr>&) { return 2; },
          [&](const symbol::Reciprocal<symbol::DimExpr>&) { return 2; },
          [&](const symbol::Add<symbol::DimExpr>&) { return 2; },
          [&](const symbol::Mul<symbol::DimExpr>&) { return 2; },
          [&](const symbol::Max<symbol::DimExpr>&) { return 2; },
          [&](const symbol::Min<symbol::DimExpr>&) { return 2; },
          [&](const symbol::Broadcast<symbol::DimExpr>&) { return 2; },
      },
      dim_expr.variant());
}

/**
 * @brief Compare the two dim exprs
 *
 * @param lhs The left-hand side dim expr
 * @param rhs The right-hand side dim expr
 *
 * @return -1 if lhs is less than rhs, 1 if lhs is greater than rhs, and 0 if
 * they are equal
 */
int CompareDimExprPriority(const symbol::DimExpr& lhs,
                           const symbol::DimExpr& rhs) {
  int lhs_priority = GetDimExprPriority(lhs);
  int rhs_priority = GetDimExprPriority(rhs);
  if (lhs_priority >= 2 || rhs_priority >= 2) {
    return 0;
  }
  if (lhs_priority != rhs_priority) {
    return lhs_priority < rhs_priority ? -1 : 1;
  }

  // if the priority is same, we compare the string value to find the smallest
  // one
  if (lhs.isa<std::string>()) {
    const auto& lhs_str = lhs.dyn_cast<std::string>();
    const auto& rhs_str = rhs.dyn_cast<std::string>();
    if (lhs_str.size() != rhs_str.size()) {
      return lhs_str.size() < rhs_str.size() ? -1 : 1;
    }
    return lhs_str.compare(rhs_str);
  }
  return 0;
}

}  // namespace

void ConstraintsManager::SubstituteDimExprInConstraint(const DimExpr& lhs,
                                                       const DimExpr& rhs) {
  std::unordered_map<symbol::DimExpr, symbol::DimExpr> substitution_pattern;
  int compare_lhs_to_rhs = CompareDimExprPriority(lhs, rhs);
  if (compare_lhs_to_rhs == 0) {
    return;
  } else if (compare_lhs_to_rhs < 0) {
    substitution_pattern[rhs] = lhs;
  } else {
    substitution_pattern[lhs] = rhs;
  }

  auto equals_parents = equals_.GetMap();
  for (auto it = equals_parents.begin(); it != equals_parents.end();) {
    DimExpr key = SubstituteDimExpr(it->first, substitution_pattern);
    DimExpr value = SubstituteDimExpr(it->first, substitution_pattern);
    if (key != it->first) {
      equals_parents.erase(it);
      equals_parents[key] = value;
    } else if (value != it->second) {
      equals_parents[key] = value;
      it++;
    } else {
      it++;
    }
  }

  for (auto it = gtones_.begin(); it != gtones_.end();) {
    DimExpr substituted_dim_expr = SubstituteDimExpr(*it, substitution_pattern);
    if (substituted_dim_expr != *it) {
      it = gtones_.erase(it);
      gtones_.insert(substituted_dim_expr);
    } else {
      it++;
    }
  }

  for (auto it = broadcastables_.begin(); it != broadcastables_.end(); it++) {
    DimExpr substituted_lhs =
        SubstituteDimExpr(it->data->lhs, substitution_pattern);
    DimExpr substituted_rhs =
        SubstituteDimExpr(it->data->rhs, substitution_pattern);
    if (substituted_lhs != it->data->lhs) {
      it->data->lhs = substituted_lhs;
    }
    if (substituted_rhs != it->data->rhs) {
      it->data->rhs = substituted_rhs;
    }
  }
}

}  // namespace symbol
