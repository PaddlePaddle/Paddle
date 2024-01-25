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

#include "paddle/pir/dialect/shape/utils/shape_analysis.h"
#include <string>

namespace pir {

static std::string GetValueId(const Value& val) {
  auto op_id = val.defining_op()->id();
  auto val_idx = val.dyn_cast<OpResult>().index();

  return val.defining_op()->name() + "_" + std::to_string(op_id) + "_rst_" +
         std::to_string(val_idx);
}

ShapeConstraintIRAnalysis::ShapeConstraintIRAnalysis(ModuleOp m) : m_(m) {}

ShapeConstraintIRAnalysis::ShapeConstraintIRAnalysis(
    std::shared_ptr<pir::Program>&& program)
    : ShapeConstraintIRAnalysis(program->module_op()) {
  program_ = std::move(program);
}
ShapeConstraintIRAnalysis::ShapeConstraintIRAnalysis(pir::IrContext* ctx)
    : ShapeConstraintIRAnalysis(std::make_shared<pir::Program>(ctx)) {}

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

bool ShapeConstraintIRAnalysis::SetShapeOrDataForValue(
    Value val, const symbol::ShapeOrDataDimExprs& shape_or_data) {
  return value_to_shape_or_data_.emplace(val, shape_or_data).second;
}

symbol::DimExprBuilder ShapeConstraintIRAnalysis::CreateDimExprBuilder() {
  return symbol::DimExprBuilder(&constraints_);
}

void ShapeConstraintIRAnalysis::PrintShapeOrDatas() const {
  LOG(INFO) << "shape analysis : @" << this
            << " value_to_shape_or_data_ size : "
            << value_to_shape_or_data_.size();
  LOG(INFO) << "----------- ShapeOrData for Values ------------";
  for (const auto& [value, shape_or_data] : value_to_shape_or_data_) {
    LOG(INFO) << GetValueId(value) << " : " << shape_or_data;
  }
}

bool ShapeConstraintIRAnalysis::IsSameNumElements(Value lhs, Value rhs) {
  if (lhs == rhs) return true;
  // auto lhs_type = lhs.type().dyn_cast<ShapedTypeInterface>();
  // auto rhs_type = rhs.type().dyn_cast<ShapedTypeInterface>();

  // if (!lhs_type || !rhs_type || !lhs_type.HasRank() || !rhs_type.HasRank())
  //   return false;

  // return IsProductEqual(lhs,
  //                       0,
  //                       static_cast<int>(lhs_type.GetRank()),
  //                       rhs,
  //                       0,
  //                       static_cast<int>(rhs_type.GetRank()));
  return true;
}

bool ShapeConstraintIRAnalysis::IsProductEqual(
    Value lhs, int lhs_from, int lhs_to, Value rhs, int rhs_from, int rhs_to) {
  // std::vector<int> lhs_dim_idxs, rhs_dim_idxs;

  // lhs_dim_idxs.reserve(lhs_to - lhs_from);
  // rhs_dim_idxs.reserve(rhs_to - rhs_from);

  // for (int i = lhs_from; i < lhs_to; ++i) lhs_dim_idxs.push_back(i);
  // for (int i = rhs_from; i < rhs_to; ++i) rhs_dim_idxs.push_back(i);

  // return IsProductEqual(lhs, lhs_dim_idxs, rhs, rhs_dim_idxs);
  return true;
}

bool ShapeConstraintIRAnalysis::IsShapeEqual(Value lhs, Value rhs) {
  if (lhs == rhs) return true;

  // auto lhs_type = lhs.type().dyn_cast<ShapedTypeInterface>();
  // auto rhs_type = rhs.type().dyn_cast<ShapedTypeInterface>();

  // if (!lhs_type || !rhs_type || !lhs_type.HasRank() || !rhs_type.HasRank())
  //   return false;

  // if (lhs_type.HasStaticShape() && rhs_type.HasStaticShape()) {
  //   return vectorize(lhs_type.GetShape()) == vectorize(rhs_type.GetShape());
  // }

  // auto lhs_it = value_to_sym_dims_.find(lhs);
  // auto rhs_it = value_to_sym_dims_.find(rhs);

  // if (lhs_it == value_to_sym_dims_.end() ||
  //     rhs_it == value_to_sym_dims_.end() ||
  //     lhs_it->second.size() != rhs_it->second.size())
  //   return false;

  // std::vector<SymbolicDimOp> lhs_syms;
  // std::vector<SymbolicDimOp> rhs_syms;
  // for (auto sym : lhs_it->second) {
  //   lhs_syms.push_back(mgr_.GetRootSymbolicDim(sym));
  // }
  // for (auto sym : rhs_it->second) {
  //   rhs_syms.push_back(mgr_.GetRootSymbolicDim(sym));
  // }
  // return lhs_syms == rhs_syms;
  return true;
}

bool ShapeConstraintIRAnalysis::IsProductEqual(Value lhs,
                                               std::vector<int> lhs_dim_idxs,
                                               Value rhs,
                                               std::vector<int> rhs_dim_idxs) {
  // SymbolicDimProduct lhs_prod;
  // SymbolicDimProduct rhs_prod;

  // auto build_symbolic_dim_product =
  //     [&](SymbolicDimProduct& prod, Value value, std::vector<int> dim_idxs) {
  //       auto type = value.type().dyn_cast<ShapedTypeInterface>();
  //       auto it = value_to_sym_dims_.find(value);
  //       if (!type || !type.HasRank()) return false;
  //       for (int idx : dim_idxs) {
  //         if (type.GetShape()[idx] == ShapedTypeInterface::kDynamic) {
  //           if (it == value_to_sym_dims_.end() ||
  //               static_cast<int>(it->second.size()) <= idx)
  //             return false;
  //           prod.symbols.push_back(it->second[idx]);
  //         } else {
  //           prod.factor *= type.GetShape()[idx];
  //         }
  //       }
  //       return true;
  //     };

  // if (!build_symbolic_dim_product(lhs_prod, lhs, lhs_dim_idxs) ||
  //     !build_symbolic_dim_product(rhs_prod, rhs, rhs_dim_idxs)) {
  //   return false;
  // }

  // return mgr_.IsSymbolicDimProductEqual(lhs_prod, rhs_prod);
  return true;
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
                      ShapeConstraintIRAnalysis(program->module_op()))
             .first;
  }

  return it->second;
}

}  // namespace pir
