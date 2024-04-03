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
    return 0;
  }
}

bool CanSubstituteInShapeAnalysis(const DimExpr& lhs, const DimExpr& rhs) {
  int lhs_priority = GetDimExprPriority(lhs);
  int rhs_priority = GetDimExprPriority(rhs);
  if (lhs_priority >= 2 || rhs_priority >= 2) {
    return 0;
  }
}
}  // namespace

void ConstraintsManager::AddEqCstr(const DimExpr& lhs, const DimExpr& rhs) {
  if (lhs == rhs) {
    return;
  }
  equals_.Union(lhs, rhs);
  VLOG(8) << "AddEqCstr the constraint: " << lhs << " == " << rhs;

  auto simplify_result = SimpliyEqualCstr(lhs, rhs);
  if (simplify_result.has_value()) {
    AddEqCstr(simplify_result->first, simplify_result->second);
  } else {
    if (CanSubstituteInConstraint(simplify_result->first,
                                  simplify_result->second)) {
      SubstituteDimExprInConstraint(simplify_result->first,
                                    simplify_result->second);
    }
    if (equal_callback_func_ &&
        CanSubstituteInShapeAnalysis(simplify_result->first,
                                     simplify_result->second)) {
      equal_callback_func_(simplify_result->first, simplify_result->second);
    }
  }
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

namespace {

bool IsDimExprGreaterThanOne(const DimExpr& dim_expr) {
  if (std::holds_alternative<std::int64_t>(dim_expr)) {
    if (std::get<std::int64_t>(dim_expr) > 1) return true;
  } else if (std::holds_alternative<Add<DimExpr>>(dim_expr)) {
    const auto& sub_dim_exprs = *std::get<Add<DimExpr>>(dim_expr).operands;
    for (auto sub_dim_expr : sub_dim_exprs) {
      if (std::holds_alternative<std::int64_t>(sub_dim_expr)) {
        if (std::get<std::int64_t>(sub_dim_expr) > 1) return true;
      }
    }
  }
  return false;
}

}  // namespace

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

void ConstraintsManager::SetEqualCallbackFunc(
    EqualCallbackFunc equal_callback_func) {
  equal_callback_func_ = equal_callback_func;
}

namespace {

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

void ConstraintsManager::SubstituteDimExprInConstraint(
    const DimExpr& origin, const DimExpr& substituted) {
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

std::optional<std::pair<DimExpr, DimExpr>> ConstraintsManager::SimpliyEqualCstr(
    const DimExpr& lhs, const DimExpr& rhs) {
  if (!((lhs.isa<Add<DimExpr>>() && rhs.isa<Add<DimExpr>>()) ||
        (lhs.isa<Mul<DimExpr>>() && rhs.isa<Mul<DimExpr>>()))) {
    return std::nullopt;
  }
  List<DimExpr> lhs_list, rhs_list;
  if (lhs.isa<Add<DimExpr>>()) {
    lhs_list = lhs.Get<Add<DimExpr>>().operands;
    rhs_list = lhs.Get<Add<DimExpr>>().operands;
  } else {
    lhs_list = lhs.Get<Mul<DimExpr>>().operands;
    rhs_list = lhs.Get<Mul<DimExpr>>().operands;
  }
  if (lhs_list->size() != rhs_list->size()) {
    return std::nullopt;
  }

  int diff_count = 0;
  DimExpr lhs_diff, rhs_diff;
  std::unordered_map<DimExpr, int> lhs_hash;
  for (DimExpr lhs_dim_expr : *lhs_list) {
    if (lhs_hash.count(lhs_dim_expr)) {
      lhs_hash[lhs_dim_expr] = lhs_hash[lhs_dim_expr]++;
    } else {
      lhs_hash[lhs_dim_expr] = 1;
    }
  }
  for (DimExpr rhs_dim_expr : *rhs_list) {
    if (lhs_hash.count(rhs_dim_expr)) {
      if (lhs_hash[rhs_dim_expr] >= 1) {
        lhs_hash[rhs_dim_expr]--;
        continue;
      }
    }
    if (++diff_count > 1) return std::nullopt;
    rhs_diff = rhs_dim_expr;
  }
  for (auto it = lhs_hash.begin(); it != lhs_hash.end(); it++) {
    if (it->second < 0) {
      lhs_diff = it->first;
      break;
    }
  }
  return std::pair(lhs_diff, rhs_diff);
}

}  // namespace symbol
