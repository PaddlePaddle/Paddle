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

#include "paddle/cinn/hlir/dialect/operator/transforms/lowering_pass/collect_sym_expr.h"
#include "paddle/pir/include/dialect/shape/utils/dim_expr_util.h"

namespace {

bool IsComplicatedDimExpr(const symbol::DimExpr& dim_expr) {
  auto lambdas = symbol::Overloaded{
      [](std::int64_t dim_expr) { return false; },
      [](const std::string& dim_expr) { return false; },
      [](const symbol::Negative<symbol::DimExpr>& dim_expr) { return true; },
      [](const symbol::Reciprocal<symbol::DimExpr>& dim_expr) { return true; },
      [](const symbol::Add<symbol::DimExpr>& dim_expr) { return true; },
      [](const symbol::Mul<symbol::DimExpr>& dim_expr) { return true; },
      [](const symbol::Max<symbol::DimExpr>& dim_expr) { return true; },
      [](const symbol::Min<symbol::DimExpr>& dim_expr) { return true; },
      [](const symbol::Broadcast<symbol::DimExpr>& dim_expr) { return true; }};
  return std::visit(lambdas, dim_expr.variant());
}

template <typename DoEachT>
void VisitEachInputValue(const OpLoweringGroupPtr& group,
                         const DoEachT& DoEach) {
  for (pir::Value value : GetBlockOutsideInput(group->ops())) {
    DoEach(value);
  }
}

template <typename DoEachT>
void VisitEachDimExprFromTensorShapeOrData(
    const symbol::TensorShapeOrDataDimExprs& shape_or_data,
    const DoEachT& DoEach) {
  for (const auto& dim_expr : shape_or_data.shape()) {
    DoEach(dim_expr);
  }
  if (!shape_or_data.data().has_value()) {
    return;
  }
  for (const auto& dim_expr : shape_or_data.data().value()) {
    DoEach(dim_expr);
  }
}

template <typename DoEachT>
void VisitEachDimExpr(const symbol::ShapeOrDataDimExprs& shape_or_data,
                      const DoEachT& DoEach) {
  auto lambdas = symbol::Overloaded{
      [&](const symbol::TensorShapeOrDataDimExprs& tensor_shape_or_data) {
        VisitEachDimExprFromTensorShapeOrData(tensor_shape_or_data, DoEach);
      },
      [&](const symbol::TensorListShapeOrDataDimExprs& tensor_list) {
        symbol::TensorListShapeOrDataDimExprs simplified_tensor_list;
        for (const symbol::TensorShapeOrDataDimExprs& tensor_shape_or_data :
             tensor_list) {
          VisitEachDimExprFromTensorShapeOrData(tensor_shape_or_data, DoEach);
        }
      }};
  return std::visit(lambdas, shape_or_data.variant());
}

std::unordered_map<symbol::DimExpr, symbol::DimExpr>
CollectSubstituteDimExprMap(
    const OpLoweringGroupPtr& group,
    pir::ShapeConstraintIRAnalysis& shape_analysis) {  // NOLINT
  std::unordered_map<symbol::DimExpr, symbol::DimExpr> dim_expr_map;
  std::unordered_set<std::string> base_dim_expr_set;

  VisitEachInputValue(group, [&](::pir::Value value) {
    if (!shape_analysis.HasShapeOrDataForValue(value)) {
      return;
    }
    auto& shape_or_data = shape_analysis.GetShapeOrDataForValue(value);
    VisitEachDimExpr(shape_or_data, [&](const symbol::DimExpr& dim_expr) {
      if (IsComplicatedDimExpr(dim_expr) &&
          dim_expr_map.find(dim_expr) == dim_expr_map.end()) {
        dim_expr_map[dim_expr] =
            symbol::DimExpr(shape_analysis.GetNextSymName());
      }
      if (dim_expr.isa<std::string>()) {
        base_dim_expr_set.insert(dim_expr.Get<std::string>());
      }
    });
  });

  const std::unordered_set<symbol::DimExpr> dim_exprs_no_outer_symbol = [&] {
    auto HasOuterBasicSymbol = [&](const symbol::DimExpr& dim_expr) {
      for (const auto& symbol : symbol::CollectDimExprSymbols(dim_expr)) {
        if (base_dim_expr_set.count(symbol) == 0) {
          return true;
        }
      }
      return false;
    };
    std::unordered_set<symbol::DimExpr> result;
    for (const auto& kv : dim_expr_map) {
      if (IsComplicatedDimExpr(kv.first) && !HasOuterBasicSymbol(kv.first)) {
        result.insert(kv.first);
      }
    }
    return result;
  }();
  for (const auto& dim_expr : dim_exprs_no_outer_symbol) {
    dim_expr_map.erase(dim_expr);
  }

  return dim_expr_map;
}

bool IsShapeOrDataNeedSubstitute(
    const symbol::ShapeOrDataDimExprs& shape_or_data,
    const std::unordered_map<symbol::DimExpr, symbol::DimExpr>& dim_expr_map) {
  bool ret = false;
  VisitEachDimExpr(shape_or_data, [&](const symbol::DimExpr& dim_expr) {
    if (dim_expr_map.find(dim_expr) != dim_expr_map.end()) {
      ret = true;
    }
  });
  return ret;
}

symbol::TensorShapeOrDataDimExprs SubstituteTensorShapeOrData(
    const symbol::TensorShapeOrDataDimExprs& shape_or_data,
    const std::unordered_map<symbol::DimExpr, symbol::DimExpr>& dim_expr_map) {
  const auto& SimplifyDimExpr =
      [&](const std::vector<symbol::DimExpr>& original_dim_expr)
      -> std::vector<symbol::DimExpr> {
    std::vector<symbol::DimExpr> simplified_dim_expr{};
    for (const symbol::DimExpr& dim_expr : original_dim_expr) {
      simplified_dim_expr.push_back(symbol::SimplifyDimExpr(
          symbol::SubstituteDimExpr(dim_expr, dim_expr_map)));
    }
    return simplified_dim_expr;
  };

  std::vector<symbol::DimExpr> simplified_shape =
      SimplifyDimExpr(shape_or_data.shape());
  if (!shape_or_data.data().has_value()) {
    return symbol::ShapeOrData<symbol::DimExpr>(simplified_shape);
  }
  std::vector<symbol::DimExpr> simplified_data =
      SimplifyDimExpr(shape_or_data.data().value());
  return symbol::ShapeOrData<symbol::DimExpr>(simplified_shape,
                                              simplified_data);
}

symbol::ShapeOrDataDimExprs SubstituteShapeOrData(
    const symbol::ShapeOrDataDimExprs& shape_or_data,
    const std::unordered_map<symbol::DimExpr, symbol::DimExpr>& dim_expr_map) {
  auto lambdas = symbol::Overloaded{
      [&](const symbol::TensorShapeOrDataDimExprs& tensor_shape_or_data) {
        return symbol::ShapeOrDataDimExprs(
            SubstituteTensorShapeOrData(tensor_shape_or_data, dim_expr_map));
      },
      [&](const symbol::TensorListShapeOrDataDimExprs& tensor_list) {
        symbol::TensorListShapeOrDataDimExprs simplified_tensor_list;
        for (symbol::TensorShapeOrDataDimExprs tensor_shape_or_data :
             tensor_list) {
          simplified_tensor_list.push_back(
              SubstituteTensorShapeOrData(tensor_shape_or_data, dim_expr_map));
        }
        return symbol::ShapeOrDataDimExprs(simplified_tensor_list);
      }};
  return std::visit(lambdas, shape_or_data.variant());
}

symbol::ShapeOrDataDimExprs TrySubstitute(
    const symbol::ShapeOrDataDimExprs& shape_or_data,
    const std::unordered_map<symbol::DimExpr, symbol::DimExpr>& dim_expr_map) {
  if (!IsShapeOrDataNeedSubstitute(shape_or_data, dim_expr_map)) {
    return shape_or_data;
  }
  return SubstituteShapeOrData(shape_or_data, dim_expr_map);
}

}  // namespace

namespace cinn::hlir::dialect::details {

std::unordered_map<::pir::Value, symbol::ShapeOrDataDimExprs>
CreateGroupShapeOrDataExprs(
    const OpLoweringGroupPtr& group,
    pir::ShapeConstraintIRAnalysis& shape_analysis) {  // NOLINT
  std::unordered_map<symbol::DimExpr, symbol::DimExpr> dim_expr_map =
      CollectSubstituteDimExprMap(group, shape_analysis);
  std::unordered_map<::pir::Value, symbol::ShapeOrDataDimExprs> value2shape;
  for (auto* op : group->ops()) {
    for (size_t i = 0; i < op->num_operands(); ++i) {
      auto operand = op->operand_source(i);
      if (operand && value2shape.find(operand) == value2shape.end() &&
          shape_analysis.HasShapeOrDataForValue(operand)) {
        VLOG(6) << "Add value_to_shape_or_data_exprs for " << operand.impl();
        value2shape.insert(
            {operand,
             TrySubstitute(shape_analysis.GetShapeOrDataForValue(operand),
                           dim_expr_map)});
      }
    }
    for (size_t i = 0; i < op->num_results(); ++i) {
      auto result = op->result(i);
      if (result && value2shape.find(result) == value2shape.end() &&
          shape_analysis.HasShapeOrDataForValue(result)) {
        VLOG(6) << "Add value_to_shape_or_data_exprs for " << result.impl();
        value2shape.insert(
            {result,
             TrySubstitute(shape_analysis.GetShapeOrDataForValue(result),
                           dim_expr_map)});
      }
    }
  }
  VLOG(5) << group.get()
          << " value_to_shape_or_data_exprs.size() : " << value2shape.size();
  return value2shape;
}

}  // namespace cinn::hlir::dialect::details
