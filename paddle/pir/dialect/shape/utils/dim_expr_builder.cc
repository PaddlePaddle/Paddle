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

#include "paddle/pir/dialect/shape/utils/dim_expr_builder.h"

namespace symbol {

using BroadcastDimExpr = Broadcast<DimExpr>;
using MinDimExpr = Min<DimExpr>;
using MaxDimExpr = Max<DimExpr>;

DimExpr DimExprBuilder::ConstSize(std::int64_t dim) { SYMBOL_NOT_IMPLEMENTED; }

DimExpr DimExprBuilder::Symbol(const std::string& symbol_name) {
  SYMBOL_NOT_IMPLEMENTED;
}

DimExpr DimExprBuilder::Add(const DimExpr& lhs, const DimExpr& rhs) {
  return lhs + rhs;
}

DimExpr DimExprBuilder::Any(const DimExpr& lhs, const DimExpr& rhs) {
  SYMBOL_NOT_IMPLEMENTED;
}

DimExpr DimExprBuilder::Mul(const DimExpr& lhs, const DimExpr& rhs) {
  return lhs * rhs;
}

DimExpr DimExprBuilder::Div(const DimExpr& lhs, const DimExpr& rhs) {
  return lhs / rhs;
}

DimExpr DimExprBuilder::Max(const DimExpr& lhs, const DimExpr& rhs) {
  return MaxDimExpr{List<DimExpr>{lhs, rhs}};
}

DimExpr DimExprBuilder::Min(const DimExpr& lhs, const DimExpr& rhs) {
  return MinDimExpr{List<DimExpr>{lhs, rhs}};
}

DimExpr DimExprBuilder::Broadcast(const DimExpr& lhs, const DimExpr& rhs) {
  return BroadcastDimExpr{List<DimExpr>{lhs, rhs}};
}

std::vector<DimExpr> DimExprBuilder::ConstShape(
    const std::vector<std::int64_t>& dims) {
  SYMBOL_NOT_IMPLEMENTED;
}

void DimExprBuilder::CstrBroadcastable(const DimExpr& lhs, const DimExpr& rhs) {
  SYMBOL_NOT_IMPLEMENTED;
}

void DimExprBuilder::CstrBroadcastable(const std::vector<DimExpr>& lhs,
                                       const std::vector<DimExpr>& rhs) {
  SYMBOL_NOT_IMPLEMENTED;
}

void DimExprBuilder::CstrEq(const DimExpr& lhs, const DimExpr& rhs) {
  constraints_->emplace_back(Equal<DimExpr>(lhs, rhs));
}

void DimExprBuilder::CstrEq(const std::vector<DimExpr>& lhs,
                            const std::vector<DimExpr>& rhs) {
  SYMBOL_NOT_IMPLEMENTED;
}

std::vector<DimExpr> DimExprBuilder::Concat(const std::vector<DimExpr>& lhs,
                                            const std::vector<DimExpr>& rhs) {
  SYMBOL_NOT_IMPLEMENTED;
}

std::pair<std::vector<DimExpr>, std::vector<DimExpr>> DimExprBuilder::SplitAt(
    const std::vector<DimExpr>, int index) {
  SYMBOL_NOT_IMPLEMENTED;
}

const std::vector<DimExprConstraint>& DimExprBuilder::constraints() const {
  SYMBOL_NOT_IMPLEMENTED;
}

}  // namespace symbol
