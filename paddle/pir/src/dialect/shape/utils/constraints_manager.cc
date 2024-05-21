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
std::pair<DimExpr, DimExpr> EliminateCommonFactor(const OpT<DimExpr>& lhs,
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
  auto DoSimplify = common::Overloaded{
      [](const Add<DimExpr>& lhs,
         const Add<DimExpr>& rhs) -> std::pair<DimExpr, DimExpr> {
        return EliminateCommonFactor<Add>(lhs, rhs);
      },
      [](const Mul<DimExpr>& lhs,
         const Mul<DimExpr>& rhs) -> std::pair<DimExpr, DimExpr> {
        return EliminateCommonFactor<Mul>(lhs, rhs);
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
  return lhs == rhs || equals_.HasSameRoot(lhs, rhs);
}

template <typename DoEachClusterT>
void ConstraintsManager::VisitEqualClusters(
    const DoEachClusterT& DoEachCluster) const {
  equals_.VisitCluster(DoEachCluster);
}

void ConstraintsManager::AddGTOneCstr(const DimExpr& dim_expr) {
  gtones_.insert(dim_expr);

  auto InsertEqualCstr = [&](const DimExpr& gtone_dim_expr,
                             const DimExpr& other_dim_expr) {
    if (IsGTOne(other_dim_expr)) {
      AddEqCstr(gtone_dim_expr, other_dim_expr);
    } else {
      AddEqCstr(
          gtone_dim_expr,
          Broadcast<DimExpr>{List<DimExpr>{gtone_dim_expr, other_dim_expr}});
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

namespace {

bool IsGTOneBaseOnValue(const DimExpr& dim_expr) {
  auto AllOperandGTOne = [](List<DimExpr> dim_exprs) {
    for (const auto& dim_expr : *dim_exprs) {
      if (IsGTOneBaseOnValue(dim_expr) == false) return false;
    }
    return true;
  };
  auto GTOneWithSomeOperandsGEOne = [](List<DimExpr> dim_exprs) {
    bool flag_exist_gtone = false;
    for (const auto& dim_expr : *dim_exprs) {
      if (dim_expr.isa<Broadcast<DimExpr>>() ||
          (dim_expr.isa<std::int64_t>() && dim_expr.Get<std::int64_t>() >= 1))
        flag_exist_gtone = true;
      else if (!dim_expr.isa<std::string>())
        return false;
    }
    return flag_exist_gtone;
  };

  auto IsGTOnePredicater =
      common::Overloaded{[&](std::int64_t dim_expr) { return dim_expr > 1; },
                         [&](const Add<DimExpr>& dim_expr) {
                           if (AllOperandGTOne(dim_expr.operands)) return true;
                           if (GTOneWithSomeOperandsGEOne(dim_expr.operands))
                             return true;
                           return false;
                         },
                         [&](const Mul<DimExpr>& dim_expr) {
                           if (AllOperandGTOne(dim_expr.operands)) return true;
                           if (GTOneWithSomeOperandsGEOne(dim_expr.operands))
                             return true;
                           return false;
                         },
                         [&](const auto& dim_expr) { return false; }};

  return std::visit(IsGTOnePredicater, dim_expr.variant());
}

}  // namespace

bool ConstraintsManager::IsGTOne(const DimExpr& dim_expr) const {
  return gtones_.count(dim_expr) || IsGTOneBaseOnValue(dim_expr);
}

void ConstraintsManager::AddBroadcastableCstr(const DimExpr& lhs,
                                              const DimExpr& rhs) {
  broadcastables_.insert(Broadcastable<DimExpr>(lhs, rhs));

  bool lhs_gtone = IsGTOne(lhs);
  bool rhs_gtone = IsGTOne(rhs);
  if (lhs_gtone && rhs_gtone) {
    AddEqCstr(lhs, rhs);
  } else if (lhs_gtone) {
    AddEqCstr(lhs, Broadcast<DimExpr>{List<DimExpr>{lhs, rhs}});
  } else if (rhs_gtone) {
    AddEqCstr(rhs, Broadcast<DimExpr>{List<DimExpr>{lhs, rhs}});
  }
}

bool ConstraintsManager::IsBroadcastable(const DimExpr& lhs,
                                         const DimExpr& rhs) const {
  for (auto broadcastable : broadcastables_) {
    if ((broadcastable->lhs == lhs && broadcastable->rhs == rhs) ||
        (broadcastable->rhs == lhs && broadcastable->lhs == rhs)) {
      return true;
    }
  }
  return false;
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
  EqualConstraintsVisitor([&](auto it) {
    DimExpr key = SubstituteDimExpr(it->first, substitution_pattern);
    DimExpr value = SubstituteDimExpr(it->second, substitution_pattern);
    substituted_equals.Union(key, value);
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
    if (substituted_lhs != substituted_rhs) {
      substituted_broadcastables.insert(
          Broadcastable<DimExpr>(substituted_lhs, substituted_rhs));
    }
  });
  broadcastables_ = substituted_broadcastables;
}

template <typename DoEachT>
void ConstraintsManager::EqualConstraintsVisitor(const DoEachT& DoEach) {
  auto equals_parents = equals_.MutMap();
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
void ConstraintsManager::GTOneConstraintsVisitor(const DoEachT& DoEach) const {
  for (auto it = gtones_.begin(); it != gtones_.end(); it++) {
    DoEach(it);
  }
}

template <typename DoEachT>
void ConstraintsManager::BroadcastableConstraintsVisitor(
    const DoEachT& DoEach) {
  for (auto it = broadcastables_.begin(); it != broadcastables_.end(); it++) {
    DoEach(it);
  }
}

template <typename DoEachT>
void ConstraintsManager::BroadcastableConstraintsVisitor(
    const DoEachT& DoEach) const {
  for (auto it = broadcastables_.begin(); it != broadcastables_.end(); it++) {
    DoEach(it);
  }
}

std::ostream& operator<<(std::ostream& stream,
                         const ConstraintsManager& constraints_manager) {
  stream << "ConstraintsManager:" << std::endl;
  stream << "Equal Constraints Clusters:" << std::endl;
  constraints_manager.VisitEqualClusters([&](const auto& cluster) {
    stream << "  {" << std::endl;
    for (const auto& dim_expr : cluster) {
      stream << "  " << dim_expr << std::endl;
    }
    stream << "  }" << std::endl;
  });
  stream << "Broadcastable Constraints:" << std::endl;
  constraints_manager.BroadcastableConstraintsVisitor([&](const auto& it) {
    stream << "  Broadcastable[ " << it->data->lhs << "," << it->data->rhs
           << " ]" << std::endl;
  });
  stream << "GreatThanOne Constraints:" << std::endl;
  constraints_manager.GTOneConstraintsVisitor(
      [&](const auto& it) { stream << "  " << *it << std::endl; });
  return stream;
}

}  // namespace symbol
