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

#include "paddle/pir/dialect/shape/utils/shape_utils.h"
#include <string>
namespace pir {

bool ShapeAnalysis::IsSameNumElements(Value lhs, Value rhs) {
  if (lhs == rhs) return true;
  auto lhs_type = lhs.type().dyn_cast<ShapedTypeInterface>();
  auto rhs_type = rhs.type().dyn_cast<ShapedTypeInterface>();

  if (!lhs_type || !rhs_type || !lhs_type.HasRank() || !rhs_type.HasRank())
    return false;

  return IsProductEqual(lhs,
                        0,
                        static_cast<int>(lhs_type.GetRank()),
                        rhs,
                        0,
                        static_cast<int>(rhs_type.GetRank()));
}

bool ShapeAnalysis::IsProductEqual(
    Value lhs, int lhs_from, int lhs_to, Value rhs, int rhs_from, int rhs_to) {
  std::vector<int> lhs_dim_idxs, rhs_dim_idxs;

  lhs_dim_idxs.reserve(lhs_to - lhs_from);
  rhs_dim_idxs.reserve(rhs_to - rhs_from);

  for (int i = lhs_from; i < lhs_to; ++i) lhs_dim_idxs.push_back(i);
  for (int i = rhs_from; i < rhs_to; ++i) rhs_dim_idxs.push_back(i);

  return IsProductEqual(lhs, lhs_dim_idxs, rhs, rhs_dim_idxs);
}

ShapeConstraintIRAnalysis::ShapeConstraintIRAnalysis(ModuleOp m)
    : m_(m), mgr_(m) {
  for (auto& op : m.block()) {
    auto tie_shape_op = op.dyn_cast<shape::TieShapeOp>();
    if (!tie_shape_op) continue;
    Value result = tie_shape_op.input();
    auto& symbols = value_to_sym_dims_[result];
    auto attrs =
        tie_shape_op
            .attribute<ArrayAttribute>(SymbolicDimOp::GetSymbolicDimAttrName())
            .AsVector();
    for (const auto& attr : attrs) {
      auto sym_op = mgr_.symbolTable().Lookup<SymbolicDimOp>(
          attr.dyn_cast<StrAttribute>().AsString());
      if (!sym_op) continue;
      symbols.push_back(sym_op);
    }
  }
}

ShapeConstraintIRAnalysis::~ShapeConstraintIRAnalysis() {}

bool ShapeConstraintIRAnalysis::IsShapeEqual(Value lhs, Value rhs) {
  if (lhs == rhs) return true;

  auto lhs_type = lhs.type().dyn_cast<ShapedTypeInterface>();
  auto rhs_type = rhs.type().dyn_cast<ShapedTypeInterface>();

  if (!lhs_type || !rhs_type || !lhs_type.HasRank() || !rhs_type.HasRank())
    return false;

  if (lhs_type.HasStaticShape() && rhs_type.HasStaticShape()) {
    return lhs_type.GetDyShape() == rhs_type.GetDyShape();
  }

  const auto& lhs_sym_shape = GetShapeOrDataForValue(&lhs);
  const auto& rhs_sym_shape = GetShapeOrDataForValue(&rhs);
  VLOG(1) << "######## " << GetValueId(&lhs) << " : " << lhs_sym_shape;
  VLOG(1) << "######## " << GetValueId(&rhs) << " : " << rhs_sym_shape;

  if (lhs_sym_shape.shape() == rhs_sym_shape.shape() &&
      lhs_sym_shape.data() == lhs_sym_shape.data()) {
    return true;
  }

  return false;
}

bool ShapeConstraintIRAnalysis::IsProductEqual(Value lhs,
                                               std::vector<int> lhs_dim_idxs,
                                               Value rhs,
                                               std::vector<int> rhs_dim_idxs) {
  SymbolicDimProduct lhs_prod;
  SymbolicDimProduct rhs_prod;

  auto build_symbolic_dim_product =
      [&](SymbolicDimProduct& prod, Value value, std::vector<int> dim_idxs) {
        auto type = value.type().dyn_cast<ShapedTypeInterface>();
        auto it = value_to_sym_dims_.find(value);
        if (!type || !type.HasRank()) return false;
        for (int idx : dim_idxs) {
          if (type.GetDyShape()[idx] == ShapedTypeInterface::kDynamic) {
            if (it == value_to_sym_dims_.end() ||
                static_cast<int>(it->second.size()) <= idx)
              return false;
            prod.symbols.push_back(it->second[idx]);
          } else {
            prod.factor *= type.GetDyShape()[idx];
          }
        }
        return true;
      };

  if (!build_symbolic_dim_product(lhs_prod, lhs, lhs_dim_idxs) ||
      !build_symbolic_dim_product(rhs_prod, rhs, rhs_dim_idxs)) {
    return false;
  }

  return mgr_.IsSymbolicDimProductEqual(lhs_prod, rhs_prod);
}

std::vector<shape::SymbolicDimOp>&
ShapeConstraintIRAnalysis::GetOrCreateSymbolicDimsForRankedValue(
    const Value& value) {
  if (value_to_sym_dims_.find(value) == value_to_sym_dims_.end()) {
    CHECK(value_to_sym_dims_
              .emplace(value, mgr_.CreateSymbolicDimsForRankedValue(value))
              .second);
  }
  return value_to_sym_dims_.at(value);
}

symbol::DimExprBuilder ShapeConstraintIRAnalysis::CreateDimExprBuilder() {
  return symbol::DimExprBuilder(&constraints_);
}

ShapeAnalysisManager& ShapeAnalysisManager::Instance() {
  static ShapeAnalysisManager instance;
  return instance;
}

ShapeConstraintIRAnalysis& ShapeAnalysisManager::Get(pir::Program* program) {
  if (tables_.empty()) {
    tables_.emplace(program->module_op().operation()->id(),
                    ShapeConstraintIRAnalysis(program->module_op()));
  }
  return tables_.begin()->second;
}

std::string GetValueId(const Value* val) {
  auto op_id = val->defining_op()->id();
  auto val_idx = val->dyn_cast<OpResult>().index();

  return "op_" + std::to_string(op_id) + "_rst_" + std::to_string(val_idx);
}

const symbol::ShapeOrDataDimExprs&
ShapeConstraintIRAnalysis::GetShapeOrDataForValue(const Value* val) {
  auto val_id = GetValueId(val);
  CHECK(value_to_shape_or_data_.count(*val))
      << "Cannot find shape or data for value: " << val_id;
  return value_to_shape_or_data_.at(*val);
}

void ShapeConstraintIRAnalysis::SetShapeOrDataForValue(
    const Value* val, const symbol::ShapeOrDataDimExprs& shape_or_data) {
  value_to_shape_or_data_[*val] = shape_or_data;
}

void ShapeConstraintIRAnalysis::PrintAllShapeOrDataDimExprs() const {
  for (const auto& [value, shape_or_data] : value_to_shape_or_data_) {
    LOG(INFO) << GetValueId(&value) << " : " << shape_or_data;
  }
}

}  // namespace pir
