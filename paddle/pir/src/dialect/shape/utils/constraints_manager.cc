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

bool CanSubstituteInShapeAnalysis(const DimExpr& lhs, const DimExpr& rhs) {
  int lhs_priority = GetDimExprPriority(lhs);
  int rhs_priority = GetDimExprPriority(rhs);
  if (lhs_priority >= 2 && rhs_priority >= 2) {
    return false;
  }
  return true;
}

}  // namespace

void ConstraintsManager::AddEqCstr(const DimExpr& lhs, const DimExpr& rhs) {
  if (lhs == rhs) {
    return;
  }

  auto simplify_result = SimplifyEqualCstr(lhs, rhs);
  if (simplify_result.first != lhs && simplify_result.second != rhs) {
    AddEqCstr(simplify_result.first, simplify_result.second);
  } else {
    equals_.Union(lhs, rhs);
    VLOG(8) << "AddEqCstr the constraint: " << lhs << " == " << rhs;

    DimExpr origin, subsutituted;
    if (CompareDimExprPriority(lhs, rhs) > 0) {
      origin = lhs;
      subsutituted = rhs;
    } else {
      origin = rhs;
      subsutituted = lhs;
    }
    if (CanSubstituteInConstraint(simplify_result.first,
                                  simplify_result.second)) {
      SubstituteInConstraint(simplify_result.first, simplify_result.second);
    }
    if (equal_callback_func_ &&
        CanSubstituteInShapeAnalysis(simplify_result.first,
                                     simplify_result.second)) {
      equal_callback_func_(simplify_result.first, simplify_result.second);
    }
  }
}

bool ConstraintsManager::IsEqual(const DimExpr& lhs, const DimExpr& rhs) {
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

void ConstraintsManager::SetEqualCallbackFunc(
    EqualCallbackFunc equal_callback_func) {
  equal_callback_func_ = equal_callback_func;
}

void ConstraintsManager::SubstituteInConstraint(const DimExpr& origin,
                                                const DimExpr& substituted) {
  std::unordered_map<DimExpr, DimExpr> substitution_pattern;
  substitution_pattern[origin] = substituted;

  auto equals_parents = equals_.GetMap();
  for (auto it = equals_parents.begin(); it != equals_parents.end();) {
    DimExpr key = SubstituteDimExpr(it->first, substitution_pattern);
    DimExpr value = SubstituteDimExpr(it->first, substitution_pattern);
    if (key != it->first) {
      it = equals_parents.erase(it);
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

  for (auto it = bcables_.begin(); it != bcables_.end(); it++) {
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

namespace {

std::pair<List<DimExpr>, List<DimExpr>> GetEqualFromAddAndMul(
    const List<DimExpr>& lhs_list, const List<DimExpr>& rhs_list) {
  if (lhs_list->size() != rhs_list->size()) {
    return std::pair(lhs_list, rhs_list);
  }
  int diff_count = 0;
  List<DimExpr> lhs_diffs, rhs_diffs;
  std::unordered_map<DimExpr, int> lhs_hash;
  for (const auto& lhs_dim_expr : *lhs_list) {
    lhs_hash[lhs_dim_expr]++;
  }
  for (const auto& rhs_dim_expr : *rhs_list) {
    if (lhs_hash.count(rhs_dim_expr)) {
      if (lhs_hash[rhs_dim_expr] >= 1) {
        lhs_hash[rhs_dim_expr]--;
        continue;
      }
    }
    rhs_diffs->push_back(rhs_dim_expr);
  }
  for (const auto& lhs_dim_expr : *lhs_list) {
    if (lhs_hash.at(lhs_dim_expr) >= 1) {
      lhs_hash[lhs_dim_expr]--;
      lhs_diffs->push_back(lhs_dim_expr);
    }
  }
  return std::pair(lhs_diffs, rhs_diffs);
}

}  // namespace

std::pair<DimExpr, DimExpr> ConstraintsManager::SimplifyEqualCstr(
    const DimExpr& lhs, const DimExpr& rhs) {
  auto DoSimplify = Overloaded{
      [](const Add<DimExpr>& lhs,
         const Add<DimExpr>& rhs) -> std::pair<DimExpr, DimExpr> {
        List<DimExpr> lhs_list = lhs.operands;
        List<DimExpr> rhs_list = rhs.operands;
        std::tie(lhs_list, rhs_list) =
            GetEqualFromAddAndMul(lhs_list, rhs_list);
        auto lhs_diff =
            lhs_list->size() == 1 ? lhs_list->at(0) : Add<DimExpr>{lhs_list};
        auto rhs_diff =
            rhs_list->size() == 1 ? rhs_list->at(0) : Add<DimExpr>{rhs_list};
        return std::pair(lhs_diff, rhs_diff);
      },
      [](const Mul<DimExpr>& lhs,
         const Mul<DimExpr>& rhs) -> std::pair<DimExpr, DimExpr> {
        List<DimExpr> lhs_list = lhs.operands;
        List<DimExpr> rhs_list = rhs.operands;
        std::tie(lhs_list, rhs_list) =
            GetEqualFromAddAndMul(lhs_list, rhs_list);
        auto lhs_diff =
            lhs_list->size() == 1 ? lhs_list->at(0) : Mul<DimExpr>{lhs_list};
        auto rhs_diff =
            rhs_list->size() == 1 ? rhs_list->at(0) : Mul<DimExpr>{rhs_list};
        return std::pair(lhs_diff, rhs_diff);
      },
      [](const auto& lhs, const auto& rhs) -> std::pair<DimExpr, DimExpr> {
        return std::make_pair(DimExpr(lhs), DimExpr(rhs));
      }};
  return std::visit(DoSimplify, lhs.variant(), rhs.variant());
}

}  // namespace symbol
