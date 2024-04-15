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
namespace {

bool CanSubstituteInConstraint(const DimExpr& lhs, const DimExpr& rhs) {
  int lhs_priority = GetDimExprPriority(lhs);
  int rhs_priority = GetDimExprPriority(rhs);
  if (lhs_priority >= 2 || rhs_priority >= 2) {
    return false;
  }
  return true;
}

bool CanEqualCStrInsert(const DimExpr& lhs, const DimExpr& rhs) {
  int lhs_priority = GetDimExprPriority(lhs);
  int rhs_priority = GetDimExprPriority(rhs);
  if (lhs_priority < 2 && rhs_priority < 2) {
    return false;
  }
  return true;
}

template <template <class> class OpT>
std::pair<DimExpr, DimExpr> FindDifferences(const OpT<DimExpr>& lhs,
                                            const OpT<DimExpr>& rhs) {
  List<DimExpr> lhs_list = lhs.operands;
  List<DimExpr> rhs_list = rhs.operands;
  List<DimExpr> lhs_diffs, rhs_diffs;
  std::unordered_map<DimExpr, int> lhs_hash;
  for (const auto& lhs_dim_expr : *lhs_list) {
    lhs_hash[lhs_dim_expr]++;
  }
  for (const auto& rhs_dim_expr : *rhs_list) {
    if (lhs_hash.count(rhs_dim_expr) && lhs_hash.at(rhs_dim_expr) >= 1) {
      lhs_hash[rhs_dim_expr]--;
      continue;
    }
    rhs_diffs->push_back(rhs_dim_expr);
  }
  for (const auto& lhs_dim_expr : *lhs_list) {
    while (lhs_hash.at(lhs_dim_expr) >= 1) {
      lhs_hash[lhs_dim_expr]--;
      lhs_diffs->push_back(lhs_dim_expr);
    }
  }
  if (lhs_diffs->size() == 0 || rhs_diffs->size() == 0)
    return std::pair(lhs, rhs);
  auto lhs_diff =
      lhs_diffs->size() == 1 ? lhs_diffs->at(0) : OpT<DimExpr>{lhs_diffs};
  auto rhs_diff =
      rhs_diffs->size() == 1 ? rhs_diffs->at(0) : OpT<DimExpr>{rhs_diffs};
  return std::pair(lhs_diff, rhs_diff);
}

std::pair<DimExpr, DimExpr> SimplifyEqCstr(const DimExpr& lhs,
                                           const DimExpr& rhs) {
  auto DoSimplify = Overloaded{
      [](const Add<DimExpr>& lhs,
         const Add<DimExpr>& rhs) -> std::pair<DimExpr, DimExpr> {
        return FindDifferences<Add>(lhs, rhs);
      },
      [](const Mul<DimExpr>& lhs,
         const Mul<DimExpr>& rhs) -> std::pair<DimExpr, DimExpr> {
        return FindDifferences<Mul>(lhs, rhs);
      },
      [](const auto& lhs, const auto& rhs) -> std::pair<DimExpr, DimExpr> {
        return std::make_pair(DimExpr(lhs), DimExpr(rhs));
      }};
  return std::visit(DoSimplify, lhs.variant(), rhs.variant());
}

}  // namespace

void ConstraintsManager::AddEqCstr(const DimExpr& lhs, const DimExpr& rhs) {
  if (lhs == rhs) {
    return;
  }

  auto simplify_result = SimplifyEqCstr(lhs, rhs);
  if (simplify_result.first != lhs && simplify_result.second != rhs) {
    AddEqCstr(simplify_result.first, simplify_result.second);
    return;
  }
  if (CanEqualCStrInsert(lhs, rhs)) {
    equals_.Union(lhs, rhs);
  }
  DimExpr origin, subsutituted;
  auto comp_result = CompareDimExprPriority(lhs, rhs);
  if (comp_result == PriorityComparisonStatus::LOWER) {
    origin = lhs;
    subsutituted = rhs;
  } else if (comp_result == PriorityComparisonStatus::HIGHER) {
    origin = rhs;
    subsutituted = lhs;
  } else {
    return;
  }
  if (CanSubstituteInConstraint(origin, subsutituted)) {
    SubstituteInConstraint(origin, subsutituted);
  }
  if (equal_callback_func_) {
    equal_callback_func_(origin, subsutituted);
  }
}

bool ConstraintsManager::IsEqual(const DimExpr& lhs, const DimExpr& rhs) const {
  return lhs == rhs || equals_.IsConnect(lhs, rhs);
}

std::vector<std::vector<DimExpr>> ConstraintsManager::GetEqualClusters() const {
  return equals_.Clusters();
}

void ConstraintsManager::SetEqualCallbackFunc(
    EqualCallbackFunc equal_callback_func) {
  equal_callback_func_ = equal_callback_func;
}

void ConstraintsManager::SubstituteInConstraint(const DimExpr& origin,
                                                const DimExpr& substituted) {
  std::unordered_map<DimExpr, DimExpr> substitution_pattern;
  substitution_pattern[origin] = substituted;

  EqualConstraints substituted_equals;
  auto substituted_equals_map = substituted_equals.GetMap();
  EqualConstraintsVisitor([&](auto it) {
    DimExpr key = SubstituteDimExpr(it->first, substitution_pattern);
    DimExpr value = SubstituteDimExpr(it->second, substitution_pattern);
    (*substituted_equals_map)[key] = value;
  });
  equals_ = substituted_equals;

  GTOneConstraints substituted_gtones;
  GTOneConstraintsVisitor([&](auto it) {
    substituted_gtones.insert(SubstituteDimExpr(*it, substitution_pattern));
  });
  gtones_ = substituted_gtones;

  BroadcastableConstraints substituted_broadcastables;
  BroadcastableConstraintsVisitor([&](auto it) {
    const DimExpr& substituted_lhs =
        SubstituteDimExpr(it->data->lhs, substitution_pattern);
    const DimExpr& substituted_rhs =
        SubstituteDimExpr(it->data->rhs, substitution_pattern);
    substituted_broadcastables.emplace_back(
        Broadcastable<DimExpr>(substituted_lhs, substituted_rhs));
  });
  broadcastables_ = substituted_broadcastables;
}

template <typename DoEachT>
void ConstraintsManager::EqualConstraintsVisitor(const DoEachT& DoEach) {
  auto equals_parents = equals_.GetMap();
  for (auto it = equals_parents->begin(); it != equals_parents->end(); it++) {
    DoEach(it);
  }
}

template <typename DoEachT>
void ConstraintsManager::GTOneConstraintsVisitor(const DoEachT& DoEach) {
  for (auto it = gtones_.begin(); it != gtones_.end(); it++) {
    DoEach(it);
  }
}

template <typename DoEachT>
void ConstraintsManager::BroadcastableConstraintsVisitor(
    const DoEachT& DoEach) {
  for (auto it = broadcastables_.begin(); it != broadcastables_.end();) {
    DoEach(it);
  }
}

std::ostream& operator<<(std::ostream& stream,
                         const ConstraintsManager& constraints_manager) {
  const std::vector<std::vector<DimExpr>>& equal_clusters =
      constraints_manager.GetEqualClusters();
  stream << "Equal Constraints Clusters:" << std::endl;
  for (auto equal_cluster : equal_clusters) {
    stream << "{" << std::endl;
    for (auto dim_expr : equal_cluster) {
      stream << dim_expr << std::endl;
    }
    stream << "}" << std::endl;
  }
  return stream;
}

}  // namespace symbol
