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

#include "paddle/pir/include/dialect/shape/utils/shape_analysis.h"
#include <string>
#include "paddle/pir/include/dialect/shape/utils/dim_expr_util.h"

namespace pir {

const symbol::DimExpr& UnionFindSet::Find(const symbol::DimExpr& x) {
  if (parent_.find(x) == parent_.end()) {
    return x;
  }
  if (parent_[x] != x) {
    parent_[x] = Find(parent_[x]);
  }
  return parent_[x];
}

void UnionFindSet::Union(const symbol::DimExpr& p, const symbol::DimExpr& q) {
  if (parent_.find(p) == parent_.end()) {
    parent_[p] = p;
  }
  if (parent_.find(q) == parent_.end()) {
    parent_[q] = q;
  }
  parent_[Find(q)] = Find(p);
}

std::vector<std::vector<symbol::DimExpr>> UnionFindSet::Clusters() {
  std::unordered_map<symbol::DimExpr, std::vector<symbol::DimExpr>>
      clusters_map;
  for (auto it = parent_.begin(); it != parent_.end(); it++) {
    clusters_map[Find(it->first)].emplace_back(it->first);
  }
  std::vector<std::vector<symbol::DimExpr>> clusters;
  for (auto it = clusters_map.begin(); it != clusters_map.end(); it++) {
    clusters.emplace_back(it->second);
  }
  return clusters;
}

static std::string GetValueId(Value val) {
  auto op_id = val.defining_op()->id();
  auto val_idx = val.dyn_cast<OpResult>().index();

  return val.defining_op()->name() + "_" + std::to_string(op_id) + "_rst_" +
         std::to_string(val_idx);
}

void ShapeConstraintIRAnalysis::Init() {
  value_to_shape_or_data_.clear();
  next_sym_idx_ = 0;
}

const std::string ShapeConstraintIRAnalysis::GetNextSymName() {
  return "S" + std::to_string(next_sym_idx_++);
}

bool ShapeConstraintIRAnalysis::HasShapeOrDataForValue(Value val) const {
  return value_to_shape_or_data_.count(val) > 0;
}

const symbol::ShapeOrDataDimExprs&
ShapeConstraintIRAnalysis::GetShapeOrDataForValue(Value val) const {
  // TODO(zhangbopd): Uncomment this part and remove `if` later.
  // IR_ENFORCE(this->HasShapeOrDataForValue(val),
  //            "No shape_or_data for this value.");
  if (!HasShapeOrDataForValue(val)) {
    static symbol::ShapeOrDataDimExprs empty{
        symbol::TensorShapeOrDataDimExprs{}};
    return empty;
  }

  return value_to_shape_or_data_.at(val);
}

void ShapeConstraintIRAnalysis::SetShapeOrDataForValue(
    Value val, const symbol::ShapeOrDataDimExprs& shape_or_data) {
  auto iter = value_to_shape_or_data_.find(val);
  if (iter == value_to_shape_or_data_.end()) {
    value_to_shape_or_data_.emplace(val, shape_or_data);
  } else {
    iter->second = shape_or_data;
  }
}

symbol::DimExprBuilder ShapeConstraintIRAnalysis::DimExprBuilder() {
  return symbol::DimExprBuilder(&constraints_);
}

void ShapeConstraintIRAnalysis::PrintShapeOrDatas() const {
  LOG(INFO) << "shape analysis : @" << this
            << " value_to_shape_or_data_ size : "
            << value_to_shape_or_data_.size();
  LOG(INFO) << "----------- ShapeOrData for Values ------------";
  for (const auto& [value, shape_or_data] : value_to_shape_or_data_) {
    if (value) {
      LOG(INFO) << GetValueId(value) << " : " << shape_or_data;
    }
  }
}

// Currently, we only support TensorShapeOrDataDimExprs but not
// TensorListShapeOrDataDimExprs to compare the shape.
bool ShapeConstraintIRAnalysis::IsShapeEqual(Value lhs, Value rhs) const {
  if (lhs == rhs) return true;

  if (!HasShapeOrDataForValue(lhs) || !HasShapeOrDataForValue(rhs)) {
    return false;
  }

  auto lhs_type = lhs.type().dyn_cast<ShapedTypeInterface>();
  auto rhs_type = rhs.type().dyn_cast<ShapedTypeInterface>();

  if (!lhs_type || !rhs_type || !lhs_type.HasRank() || !rhs_type.HasRank())
    return false;

  auto lhs_shape_data = GetShapeOrDataForValue(lhs);
  auto rhs_shape_data = GetShapeOrDataForValue(rhs);

  IR_ENFORCE(lhs_shape_data.isa<symbol::TensorShapeOrDataDimExprs>() &&
                 rhs_shape_data.isa<symbol::TensorShapeOrDataDimExprs>(),
             "Currently, IsShapeEqual only support TensorShapeOrDataDimExprs "
             "but not TensorListShapeOrDataDimExprs.");

  // For static shape, directly compare the shapes.
  if (lhs_type.IsStaticShape() && rhs_type.IsStaticShape()) {
    return lhs_type.GetShape() == rhs_type.GetShape();
  }

  // For dynamic shape, compare the symbolic dimensions.
  return lhs_shape_data.variant() == rhs_shape_data.variant();
}

bool ShapeConstraintIRAnalysis::IsProductEqual(
    Value lhs,
    const std::vector<int>& lhs_dim_idxs,
    Value rhs,
    const std::vector<int>& rhs_dim_idxs) const {
  if (lhs == rhs) return true;

  if (!HasShapeOrDataForValue(lhs) || !HasShapeOrDataForValue(rhs)) {
    return false;
  }

  auto lhs_type = lhs.type().dyn_cast<ShapedTypeInterface>();
  auto rhs_type = rhs.type().dyn_cast<ShapedTypeInterface>();

  if (!lhs_type || !rhs_type || !lhs_type.HasRank() || !rhs_type.HasRank())
    return false;

  auto lhs_shape_data = GetShapeOrDataForValue(lhs);
  auto rhs_shape_data = GetShapeOrDataForValue(rhs);

  IR_ENFORCE(lhs_shape_data.isa<symbol::TensorShapeOrDataDimExprs>() &&
                 rhs_shape_data.isa<symbol::TensorShapeOrDataDimExprs>(),
             "Currently, IsProductEqual only support TensorShapeOrDataDimExprs "
             "but not TensorListShapeOrDataDimExprs.");

  // For static shape
  if (lhs_type.IsStaticShape() && rhs_type.IsStaticShape()) {
    int64_t lhs_product = 1;
    int64_t rhs_product = 1;
    for (int i : lhs_dim_idxs) {
      lhs_product *= lhs_type.GetShape()[i];
    }
    for (int i : rhs_dim_idxs) {
      rhs_product *= rhs_type.GetShape()[i];
    }
    return lhs_product == rhs_product;
  }

  // For dynamic shape
  symbol::DimExpr lhs_product(1);
  symbol::DimExpr rhs_product(1);
  for (int i : lhs_dim_idxs) {
    lhs_product = lhs_product * lhs_shape_data.shape()[i];
  }
  for (int i : rhs_dim_idxs) {
    rhs_product = rhs_product * rhs_shape_data.shape()[i];
  }
  return symbol::SimplifyDimExpr(lhs_product) ==
         symbol::SimplifyDimExpr(rhs_product);
}

bool ShapeConstraintIRAnalysis::IsProductEqual(Value lhs,
                                               int lhs_from,
                                               int lhs_to,
                                               Value rhs,
                                               int rhs_from,
                                               int rhs_to) const {
  std::vector<int> lhs_dim_idxs, rhs_dim_idxs;

  lhs_dim_idxs.reserve(lhs_to - lhs_from);
  rhs_dim_idxs.reserve(rhs_to - rhs_from);

  for (int i = lhs_from; i < lhs_to; ++i) lhs_dim_idxs.push_back(i);
  for (int i = rhs_from; i < rhs_to; ++i) rhs_dim_idxs.push_back(i);

  return IsProductEqual(lhs, lhs_dim_idxs, rhs, rhs_dim_idxs);
}

bool ShapeConstraintIRAnalysis::IsSameNumel(Value lhs, Value rhs) const {
  if (lhs == rhs) return true;

  auto lhs_type = lhs.type().dyn_cast<ShapedTypeInterface>();
  auto rhs_type = rhs.type().dyn_cast<ShapedTypeInterface>();

  if (!lhs_type || !rhs_type || !lhs_type.HasRank() || !rhs_type.HasRank())
    return false;

  // For static shape
  if (lhs_type.IsStaticShape() && rhs_type.IsStaticShape()) {
    auto lhs_shape = lhs_type.GetShape();
    auto rhs_shape = rhs_type.GetShape();
    if (lhs_shape == rhs_shape) {
      return true;
    }
    return common::product(lhs_shape) == common::product(rhs_shape);
  }

  return IsProductEqual(lhs,
                        0,
                        static_cast<int>(lhs_type.GetRank()),
                        rhs,
                        0,
                        static_cast<int>(rhs_type.GetRank()));
}

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

bool CanDimExprSubstitute(const symbol::DimExpr& lhs,
                          const symbol::DimExpr& rhs) {
  int priority_lhs = GetDimExprPriority(lhs);
  int priority_rhs = GetDimExprPriority(rhs);
  if (priority_lhs >= 2 && priority_rhs >= 2) return false;
  return true;
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

void ShapeConstraintIRAnalysis::AddEqCstr(const symbol::DimExpr& lhs,
                                          const symbol::DimExpr& rhs) {
  if (lhs == rhs) return;
  eq_cstr_set.Union(lhs, rhs);
  VLOG(8) << "AddEqCstr the constraint: " << lhs << " == " << rhs;

  if (CanDimExprSubstitute(lhs, rhs)) {
    std::unordered_map<symbol::DimExpr, symbol::DimExpr> substitution_pattern;
    if (CompareDimExpr(lhs, rhs) < 0) {
      substitution_pattern[rhs] = lhs;
    } else {
      substitution_pattern[lhs] = rhs;
    }

    for (auto it = value_to_shape_or_data_.begin();
         it != value_to_shape_or_data_.end();
         it++) {
      const symbol::ShapeOrDataDimExprs& substituted_shape_or_data =
          SubstituteShapeOrData(it->second, substitution_pattern);
      SetShapeOrDataForValue(it->first, substituted_shape_or_data);
    }
  }
}

pir::PrintHooks ShapeConstraintIRAnalysis::PrintHook() const {
  pir::PrintHooks print_hook;
  print_hook.op_print_hook = [&](Operation* op, IrPrinter& printer) {
    printer.IrPrinter::PrintOperation(op);
    printer.os << " { ";
    for (uint32_t i = 0; i < op->num_results(); ++i) {
      if (this->HasShapeOrDataForValue(op->result(i))) {
        printer.os << "(" << this->GetShapeOrDataForValue(op->result(i)) << ")";
      } else {
        printer.os << "()";
      }
      if (i < op->num_results() - 1) {
        printer.os << ", ";
      }
    }
    printer.os << " }";
    printer.os << "\t(op_" << op->id() << ")";
  };
  return print_hook;
}

ShapeAnalysisManager& ShapeAnalysisManager::Instance() {
  static ShapeAnalysisManager instance;
  return instance;
}

ShapeConstraintIRAnalysis& ShapeAnalysisManager::Get(pir::Program* program) {
  auto it = tables_.find(program->module_op().operation()->id());

  if (it == tables_.end()) {
    it = tables_
             .emplace(program->module_op().operation()->id(),
                      ShapeConstraintIRAnalysis())
             .first;
  }

  return it->second;
}

}  // namespace pir
