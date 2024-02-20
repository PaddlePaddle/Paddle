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

#pragma once

#include "paddle/cinn/hlir/dialect/operator/transforms/group_merge/substitute_dim_expr_based_on_constraint_pass.h"

#include <unordered_map>
#include "paddle/cinn/common/union_find.h"
#include "paddle/cinn/hlir/dialect/operator/ir/generate_shape_util.h"
#include "paddle/cinn/hlir/dialect/operator/ir/op_dialect.h"
#include "paddle/pir/include/dialect/shape/ir/shape_attribute.h"
#include "paddle/pir/include/dialect/shape/utils/dim_expr_simplify.h"

namespace cinn {
namespace dialect {
namespace ir {

namespace {

template <typename DoEachT>
void VisitEachValue(pir::ModuleOp module_op, const DoEachT& DoEach) {
  for (uint32_t i = 0; i < module_op->num_regions(); i++) {
    for (pir::Block& block : module_op->region(i)) {
      for (pir::Operation& op : block) {
        for (std::size_t i = 0; i < op.num_operands(); ++i) {
          DoEach(op.operand_source(i));
        }
        for (std::size_t i = 0; i < op.num_results(); ++i) {
          DoEach(op.result(i));
        }
      }
    }
  }
}

symbol::TensorShapeOrDataDimExprs SubstituteTensorShapeOrData(
    const symbol::TensorShapeOrDataDimExprs& shape_or_data,
    const std::unordered_map<symbol::DimExpr, symbol::DimExpr>&
        dim_expr_substitution) {
  const auto& SubstituteDimExpr =
      [](const std::vector<symbol::DimExpr>& original_dim_expr,
         const std::unordered_map<symbol::DimExpr, symbol::DimExpr>&
             dim_expr_substitution) -> std::vector<symbol::DimExpr> {
    std::vector<symbol::DimExpr> substituted_dim_expr{};
    for (const symbol::DimExpr& dim_expr : original_dim_expr) {
      const auto& dim_expr_to_substitute = dim_expr_substitution.find(dim_expr);
      if (dim_expr_to_substitute != dim_expr_substitution.end()) {
        substituted_dim_expr.push_back(dim_expr_to_substitute->second);
      }
    }
    return substituted_dim_expr;
  };

  std::vector<symbol::DimExpr> substituted_shape =
      SubstituteDimExpr(shape_or_data.shape(), dim_expr_substitution);
  if (!shape_or_data.data().has_value()) {
    return symbol::ShapeOrData<symbol::DimExpr>(substituted_shape);
  } else {
    std::vector<symbol::DimExpr> substituted_data =
        SubstituteDimExpr(shape_or_data.data().value(), dim_expr_substitution);
    return symbol::ShapeOrData<symbol::DimExpr>(substituted_shape,
                                                substituted_data);
  }
}

symbol::ShapeOrDataDimExprs SubstituteShapeOrData(
    const symbol::ShapeOrDataDimExprs& shape_or_data,
    const std::unordered_map<symbol::DimExpr, symbol::DimExpr>&
        dim_expr_substitution) {
  auto lambdas = symbol::Overloaded{
      [&](const symbol::TensorShapeOrDataDimExprs& tensor_shape_or_data) {
        return symbol::ShapeOrDataDimExprs(SubstituteTensorShapeOrData(
            tensor_shape_or_data, dim_expr_substitution));
      },
      [&](const symbol::TensorListShapeOrDataDimExprs& tensor_list) {
        symbol::TensorListShapeOrDataDimExprs substituted_tensor_list;
        for (symbol::TensorShapeOrDataDimExprs tensor_shape_or_data :
             tensor_list) {
          substituted_tensor_list.push_back(SubstituteTensorShapeOrData(
              tensor_shape_or_data, dim_expr_substitution));
        }
        return symbol::ShapeOrDataDimExprs(substituted_tensor_list);
      }};
  return std::visit(lambdas, shape_or_data.variant());
}

std::unordered_map<symbol::DimExpr, symbol::DimExpr> GetDimExprSubstitution(
    pir::ShapeConstraintIRAnalysis shape_analysis) {
  std::vector<symbol::DimExprConstraint> dim_expr_constraints =
      shape_analysis.CreateDimExprBuilder().constraints();
  cinn::common::UnionFindSet<symbol::DimExpr> union_find_set;
  for (const auto& constraint : dim_expr_constraints) {
    CHECK(std::holds_alternative<symbol::Equal<symbol::DimExpr>>(constraint))
        << "the DimExprConstraint type is no Equal<DimExpr>, this part is to "
           "be completed";
    const auto& data =
        std::get<symbol::Equal<symbol::DimExpr>>(constraint).data;
    union_find_set.Union(data->lhs, data->rhs);
  }
  std::vector<std::vector<symbol::DimExpr>> dim_expr_clusters =
      union_find_set.Clusters();
  std::unordered_map<symbol::DimExpr, symbol::DimExpr> dim_expr_substitution;
  for (const auto& dim_expr_cluster : dim_expr_clusters) {
    auto dim_expr_best = dim_expr_cluster[0];
    for (const auto& dim_expr : dim_expr_cluster) {
      if (std::holds_alternative<std::int64_t>(dim_expr)) {
        dim_expr_best = dim_expr;
        break;
      }
    }
    for (const auto& dim_expr : dim_expr_cluster) {
      dim_expr_substitution[dim_expr] = dim_expr_best;
    }
  }
  return dim_expr_substitution;
}

void SubstituteDimExprBasedOnConstraint(pir::ModuleOp module_op) {
  VLOG(4) << "SubstituteDimExprBasedOnConstraint start";
  pir::ShapeConstraintIRAnalysis shape_analysis =
      pir::ShapeAnalysisManager::Instance().Get(module_op.program());
  const std::unordered_map<symbol::DimExpr, symbol::DimExpr>&
      dim_expr_substitution = GetDimExprSubstitution(shape_analysis);

  VisitEachValue(module_op, [&](pir::Value value) {
    if (!shape_analysis.HasShapeOrDataForValue(value)) {
      VLOG(4) << "Can not find ShapeOrData for value int shape_analysis";
    } else {
      const symbol::ShapeOrDataDimExprs& origin_shape_or_data =
          shape_analysis.GetShapeOrDataForValue(value);
      symbol::ShapeOrDataDimExprs substituted_shape_or_data =
          SubstituteShapeOrData(origin_shape_or_data, dim_expr_substitution);
      shape_analysis.SetShapeOrDataForValue(value, substituted_shape_or_data);
    }
  });

  VLOG(4) << "SubstituteDimExprBasedOnConstraint end";
}

class SubstituteDimExprBasedOnConstraintPass : public pir::Pass {
 public:
  SubstituteDimExprBasedOnConstraintPass()
      : pir::Pass("substitute_dim_expr_based_on_constraint_pass", 1) {}

  void Run(pir::Operation* op) override {
    pir::ModuleOp module_op = op->dyn_cast<pir::ModuleOp>();
    SubstituteDimExprBasedOnConstraint(module_op);
  }

  bool CanApplyOn(pir::Operation* op) const override {
    return op->isa<pir::ModuleOp>() && op->num_regions() > 0;
  }
};

}  // namespace

std::unique_ptr<::pir::Pass> CreateSubstituteDimExprBasedOnConstraintPass() {
  return std::make_unique<SubstituteDimExprBasedOnConstraintPass>();
}

}  // namespace ir
}  // namespace dialect
}  // namespace cinn
