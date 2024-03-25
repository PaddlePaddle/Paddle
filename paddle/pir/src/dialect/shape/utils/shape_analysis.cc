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

void ShapeConstraintIRAnalysis::AddEqCstr(const symbol::DimExpr& lhs,
                                          const symbol::DimExpr& rhs) {
  eq_cstr_set.Union(lhs, rhs);
}

bool ShapeConstraintIRAnalysis::IsDimExprEqual(const symbol::DimExpr& lhs,
                                               const symbol::DimExpr& rhs) {
  if (lhs == rhs) return true;

  if (eq_cstr_set.Find(lhs) == eq_cstr_set.Find(rhs)) {
    return true;
  }

  return false;
}

void ShapeConstraintIRAnalysis::PrintDimExprClusters() {
  const auto& clusters = eq_cstr_set.Clusters();
  VLOG(0) << "##### shape analysis clusters: ";
  for (auto& cluster : clusters) {
    VLOG(0) << "  cluster: " << cluster;
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