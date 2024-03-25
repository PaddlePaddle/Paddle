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

#include "paddle/cinn/hlir/dialect/operator/transforms/group_merge/substitute_dim_expr_based_on_constraints_pass.h"

#include <regex>

#include "paddle/cinn/common/union_find.h"
#include "paddle/cinn/hlir/dialect/operator/ir/manual_op.h"
#include "paddle/common/enforce.h"
#include "paddle/pir/include/dialect/shape/ir/shape_attribute.h"
#include "paddle/pir/include/dialect/shape/utils/dim_expr_util.h"

PD_DECLARE_string(cinn_symbol_dim_constraints);

namespace cinn {
namespace dialect {
namespace ir {

namespace {

namespace detail {

void Trim(std::string* str) {
  str->erase(0, str->find_first_not_of(' '));
  str->erase(str->find_last_not_of(' ') + 1);
}

std::vector<std::string> Split(const std::string& str,
                               const std::string& delimiter) {
  std::vector<std::string> result;
  std::regex reg(delimiter);
  std::sregex_token_iterator pos(str.begin(), str.end(), reg, -1);
  decltype(pos) end;
  for (; pos != end; ++pos) {
    std::string tmp = pos->str();
    Trim(&tmp);
    if (!tmp.empty()) {
      result.emplace_back(tmp);
    }
  }
  return result;
}

std::vector<symbol::DimExprConstraint> ParseDimExprConstraintsFLAGS() {
  const auto& CreateConstrain = [&](const std::string& constraint,
                                    symbol::DimExprBuilder* builder) {
    std::vector<std::string> expr_pair = Split(constraint, "==");
    PADDLE_ENFORCE_EQ(expr_pair.size(),
                      2UL,
                      ::common::errors::InvalidArgument(
                          "The constraint is invalid. the result size of "
                          "constraint after split by '==' should be 2, but now "
                          "is %d, original constrain is '%s'",
                          expr_pair.size(),
                          constraint));
    const std::string& lhs_expr = expr_pair[0];
    const std::string& rhs_expr = expr_pair[1];
    VLOG(0) << "######### ParseDimExprConstraintsFLAG " << lhs_expr
            << " == " << rhs_expr;
    builder->CstrEq(symbol::DimExpr{lhs_expr}, symbol::DimExpr{rhs_expr});
  };
  std::vector<symbol::DimExprConstraint> dim_expr_constraints = [&] {
    const std::string& constraints = FLAGS_cinn_symbol_dim_constraints;
    std::vector<symbol::DimExprConstraint> dim_expr_constraints;
    symbol::DimExprBuilder builder(&dim_expr_constraints);
    for (const std::string& constraint : Split(constraints, ",")) {
      CreateConstrain(constraint, &builder);
    }
    return dim_expr_constraints;
  }();
  return dim_expr_constraints;
}

}  // namespace detail

template <typename DoEachT>
void VisitEachOp(pir::Operation* op, const DoEachT& DoEach) {
  DoEach(op);
  for (auto& region : *op) {
    for (auto& block : region) {
      for (auto& op_in_block : block) {
        DoEach(&op_in_block);
      }
    }
  }
}

template <typename DoEachT>
void VisitEachValue(const pir::Operation* op, const DoEachT& DoEach) {
  for (std::size_t i = 0; i < op->num_operands(); ++i) {
    DoEach(op->operand_source(i));
  }
  for (std::size_t i = 0; i < op->num_results(); ++i) {
    DoEach(op->result(i));
  }
}

const std::vector<symbol::DimExprConstraint>& ParseDimExprConstraintsFLAGS() {
  static std::vector<symbol::DimExprConstraint> dim_expr_constraints{
      detail::ParseDimExprConstraintsFLAGS()};
  return dim_expr_constraints;
}

symbol::TensorShapeOrDataDimExprs SubstituteTensorShapeOrData(
    const symbol::TensorShapeOrDataDimExprs& shape_or_data,
    const std::unordered_map<symbol::DimExpr, symbol::DimExpr>&
        substitution_pattern) {
  for (const auto& it : substitution_pattern) {
    VLOG(4) << "substitution_pattern: " << it.first << " -> " << it.second;
  }

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

void SimplifyUnionSet(
    cinn::common::UnionFindSet<symbol::DimExpr>* union_find_set) {
  const std::vector<std::vector<symbol::DimExpr>>& dim_expr_clusters =
      union_find_set->Clusters();
  const std::unordered_map<symbol::DimExpr, symbol::DimExpr>
      substitution_pattern = [&] {
        std::unordered_map<symbol::DimExpr, symbol::DimExpr>
            substitution_pattern;
        for (const auto& dim_expr_cluster : dim_expr_clusters) {
          CHECK(!dim_expr_cluster.empty());
          auto dim_expr_root = dim_expr_cluster[0];
          for (const auto& dim_expr : dim_expr_cluster) {
            if (CompareDimExpr(dim_expr, dim_expr_root) < 0) {
              dim_expr_root = dim_expr;
            }
          }
          for (const auto& dim_expr : dim_expr_cluster) {
            if (dim_expr.isa<std::string>() && dim_expr != dim_expr_root) {
              substitution_pattern[dim_expr] = dim_expr_root;
            }
          }
        }
        return substitution_pattern;
      }();

  bool is_update = false;
  for (const auto& dim_expr_cluster : dim_expr_clusters) {
    for (const auto& dim_expr : dim_expr_cluster) {
      if (!dim_expr.isa<int64_t>() && !dim_expr.isa<std::string>()) {
        const auto& tmp_dim_expr = symbol::SimplifyDimExpr(
            symbol::SubstituteDimExpr(dim_expr, substitution_pattern));
        if (tmp_dim_expr != dim_expr && union_find_set->Find(tmp_dim_expr) !=
                                            union_find_set->Find(dim_expr)) {
          union_find_set->Union(tmp_dim_expr, dim_expr);
          is_update = true;
        }
      }
    }
  }
  if (is_update) {
    SimplifyUnionSet(union_find_set);
  }
}

std::unordered_map<symbol::DimExpr, symbol::DimExpr> GetDimExprSubstitution(
    pir::ShapeConstraintIRAnalysis* shape_analysis) {
  const std::vector<std::vector<symbol::DimExpr>>& dim_expr_clusters = [&]() {
    cinn::common::UnionFindSet<symbol::DimExpr> union_find_set;
    auto AddEqualCstr = [&](const symbol::DimExprConstraint& constraint) {
      if (!std::holds_alternative<symbol::Equal<symbol::DimExpr>>(constraint)) {
        VLOG(0) << "The DimExprConstraint type is no Equal<DimExpr>, this part "
                   "is to be completed.";
        return;
      }
      const auto& data =
          std::get<symbol::Equal<symbol::DimExpr>>(constraint).data;
      if (data->lhs == data->rhs) {
        return;
      }
      union_find_set.Union(data->lhs, data->rhs);
    };
    const auto& shape_analysis_constraints =
        shape_analysis->DimExprBuilder().constraints();
    for (const auto& constraint : shape_analysis_constraints) {
      AddEqualCstr(constraint);
    }
    const auto& dim_expr_constraints = ParseDimExprConstraintsFLAGS();
    for (const auto& constraint : dim_expr_constraints) {
      AddEqualCstr(constraint);
    }
    SimplifyUnionSet(&union_find_set);
    std::vector<std::vector<symbol::DimExpr>> dim_expr_clusters =
        union_find_set.Clusters();
    return dim_expr_clusters;
  }();

  std::unordered_map<symbol::DimExpr, symbol::DimExpr> substitution_pattern;
  for (const auto& dim_expr_cluster : dim_expr_clusters) {
    VLOG(0) << "####### dim_expr_cluster: " << dim_expr_cluster;
    CHECK(!dim_expr_cluster.empty());
    auto dim_expr_root = dim_expr_cluster[0];
    for (const auto& dim_expr : dim_expr_cluster) {
      if (CompareDimExpr(dim_expr, dim_expr_root) < 0) {
        dim_expr_root = dim_expr;
      }
    }
    for (const auto& dim_expr : dim_expr_cluster) {
      if (dim_expr != dim_expr_root) {
        if (!dim_expr.isa<std::string>() && !dim_expr.isa<int64_t>()) {
          continue;
        }
        substitution_pattern[dim_expr] = dim_expr_root;
      }
    }
  }
  return substitution_pattern;
}

void SubstituteDimExprBasedOnConstraints(pir::Operation* region_op) {
  VLOG(4) << "SubstituteDimExprBasedOnConstraints start";
  pir::ShapeConstraintIRAnalysis* shape_analysis =
      &pir::ShapeAnalysisManager::Instance().Get(region_op->GetParentProgram());
  const std::unordered_map<symbol::DimExpr, symbol::DimExpr>&
      substitution_pattern = GetDimExprSubstitution(shape_analysis);

  VisitEachOp(region_op, [&](pir::Operation* op) {
    VisitEachValue(op, [&](pir::Value value) {
      if (!shape_analysis->HasShapeOrDataForValue(value)) {
        VLOG(4) << "Can not find ShapeOrData for value of op(" << op->name()
                << ") in shape_analysis";
      } else {
        const symbol::ShapeOrDataDimExprs& origin_shape_or_data =
            shape_analysis->GetShapeOrDataForValue(value);
        VLOG(8) << op->name()
                << "      origin_shape_or_data: " << origin_shape_or_data;
        const symbol::ShapeOrDataDimExprs& substituted_shape_or_data =
            SubstituteShapeOrData(origin_shape_or_data, substitution_pattern);
        VLOG(8) << op->name()
                << " substituted_shape_or_data: " << substituted_shape_or_data;
        shape_analysis->SetShapeOrDataForValue(value,
                                               substituted_shape_or_data);
      }
    });
    if (op->num_regions() > 0) {
      return;
    }
    if (op->num_results() > 0) {
      pir::shape::SetShapeAttrForOp(
          op, shape_analysis->GetShapeOrDataForValue(op->result(0)));
    } else {
      pir::shape::SetShapeAttrForOp(
          op, shape_analysis->GetShapeOrDataForValue(op->operand_source(0)));
    }
  });
  VLOG(4) << "SubstituteDimExprBasedOnConstraints end";
}

class SubstituteDimExprBasedOnConstraintsPass : public pir::Pass {
 public:
  SubstituteDimExprBasedOnConstraintsPass()
      : pir::Pass("substitute_dim_expr_based_on_constraints_pass", 1) {}

  void Run(pir::Operation* op) override {
    SubstituteDimExprBasedOnConstraints(op);
  }

  bool CanApplyOn(pir::Operation* op) const override {
    return op->num_regions() > 0;
  }
};

}  // namespace

std::unique_ptr<::pir::Pass> CreateSubstituteDimExprBasedOnConstraintsPass() {
  return std::make_unique<SubstituteDimExprBasedOnConstraintsPass>();
}

}  // namespace ir
}  // namespace dialect
}  // namespace cinn
