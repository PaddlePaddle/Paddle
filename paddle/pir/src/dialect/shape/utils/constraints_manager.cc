// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

namespace symbol {

namespace {

symbol::TensorShapeOrDataDimExprs SubstituteTensorShapeOrData(
    const symbol::TensorShapeOrDataDimExprs& shape_or_data,
    const std::unordered_map<symbol::DimExpr, symbol::DimExpr>&
        substitution_pattern) {
  auto SubstituteOneDimExpr =
      [](const std::vector<symbol::DimExpr>& original_dim_expr,
         const std::unordered_map<symbol::DimExpr, symbol::DimExpr>&
             substitution_pattern) -> std::vector<symbol::DimExpr> {
    std::vector<symbol::DimExpr> substituted_dim_expr{};
    for (const symbol::DimExpr& dim_expr : original_dim_expr) {
      const auto& tmp_dim_expr =
          symbol::SubstituteDimExpr(dim_expr, substitution_pattern);
      substituted_dim_expr.push_back(symbol::SimplifyDimExpr(tmp_dim_expr));
    }
    return substituted_dim_expr;
  };

  std::vector<symbol::DimExpr> substituted_shape =
      SubstituteOneDimExpr(shape_or_data.shape(), substitution_pattern);
  if (!shape_or_data.data().has_value()) {
    return symbol::ShapeOrData<symbol::DimExpr>(substituted_shape);
  } else {
    std::vector<symbol::DimExpr> substituted_data = SubstituteOneDimExpr(
        shape_or_data.data().value(), substitution_pattern);
    return symbol::ShapeOrData<symbol::DimExpr>(substituted_shape,
                                                substituted_data);
  }
}

symbol::ShapeOrDataDimExprs SubstituteShapeOrData(
    const symbol::ShapeOrDataDimExprs& shape_or_data,
    const std::unordered_map<symbol::DimExpr, symbol::DimExpr>&
        substitution_pattern) {
  auto lambdas = symbol::Overloaded{
      [&](const symbol::TensorShapeOrDataDimExprs& tensor_shape_or_data) {
        return symbol::ShapeOrDataDimExprs(SubstituteTensorShapeOrData(
            tensor_shape_or_data, substitution_pattern));
      },
      [&](const symbol::TensorListShapeOrDataDimExprs& tensor_list) {
        symbol::TensorListShapeOrDataDimExprs substituted_tensor_list;
        for (symbol::TensorShapeOrDataDimExprs tensor_shape_or_data :
             tensor_list) {
          substituted_tensor_list.push_back(SubstituteTensorShapeOrData(
              tensor_shape_or_data, substitution_pattern));
        }
        return symbol::ShapeOrDataDimExprs(substituted_tensor_list);
      }};
  return std::visit(lambdas, shape_or_data.variant());
}

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
int CompareDimExpr(const symbol::DimExpr& lhs, const symbol::DimExpr& rhs) {
  int lhs_priority = GetDimExprPriority(lhs);
  int rhs_priority = GetDimExprPriority(rhs);
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

void ConstraintsManager::SubstituteRhsToLhs(const DimExpr& lhs,
                                            const DimExpr& rhs) {
  std::unordered_map<symbol::DimExpr, symbol::DimExpr> substitution_pattern;
  int compare_lhs_to_rhs = CompareDimExpr(lhs, rhs);
  if (compare_lhs_to_rhs == 0) {
    return;
  } else if (compare_lhs_to_rhs < 0) {
    substitution_pattern[rhs] = lhs;
  } else {
    substitution_pattern[lhs] = rhs;
  }
  for (auto it = value_to_shape_or_data_->begin();
       it != value_to_shape_or_data_->end();
       it++) {
    const symbol::ShapeOrDataDimExprs& substituted_shape_or_data =
        SubstituteShapeOrData(it->second, substitution_pattern);
    auto iter = value_to_shape_or_data_->find(it->first);
    if (iter == value_to_shape_or_data_->end()) {
      value_to_shape_or_data_->emplace(it->first, substituted_shape_or_data);
    } else {
      iter->second = substituted_shape_or_data;
    }
  }
}

void ConstraintsManager::SetValueToShapeOrData(
    std::unordered_map<pir::Value, ShapeOrDataDimExprs>*
        value_to_shape_or_data) {
  value_to_shape_or_data_ = value_to_shape_or_data;
}

void ConstraintsManager::AddEqCstr(const DimExpr& lhs, const DimExpr& rhs) {
  if (lhs == rhs) {
    return;
  }
  equals_.Union(lhs, rhs);
  VLOG(8) << "AddEqCstr the constraint: " << lhs << " == " << rhs;

  SubstituteRhsToLhs(lhs, rhs);
}

}  // namespace symbol
