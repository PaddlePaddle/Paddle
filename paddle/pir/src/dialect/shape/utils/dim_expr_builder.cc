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

#include "paddle/pir/include/dialect/shape/utils/dim_expr_builder.h"
#include "paddle/common/enforce.h"
#include "paddle/pir/include/dialect/shape/utils/dim_expr_util.h"

namespace symbol {

using BroadcastDimExpr = Broadcast<DimExpr>;
using MinDimExpr = Min<DimExpr>;
using MaxDimExpr = Max<DimExpr>;

DimExpr DimExprBuilder::ConstSize(std::int64_t dim) { return DimExpr(dim); }

DimExpr DimExprBuilder::Symbol(const std::string& symbol_name) {
  return DimExpr(symbol_name);
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
  return SimplifyDimExpr(MaxDimExpr{List<DimExpr>{lhs, rhs}});
}

DimExpr DimExprBuilder::Min(const DimExpr& lhs, const DimExpr& rhs) {
  return SimplifyDimExpr(MinDimExpr{List<DimExpr>{lhs, rhs}});
}

DimExpr DimExprBuilder::Broadcast(const DimExpr& lhs, const DimExpr& rhs) {
  return SimplifyDimExpr(BroadcastDimExpr{List<DimExpr>{lhs, rhs}});
}

std::vector<DimExpr> DimExprBuilder::ConstShape(
    const std::vector<std::int64_t>& dims) {
  std::vector<DimExpr> ret{};
  ret.reserve(dims.size());
  for (std::int64_t dim : dims) {
    ret.emplace_back(dim);
  }
  return ret;
}

std::vector<DimExpr> DimExprBuilder::Concat(const std::vector<DimExpr>& lhs,
                                            const std::vector<DimExpr>& rhs) {
  std::vector<DimExpr> ret{};
  const auto& EmplaceDimExpr = [&](const std::vector<DimExpr>& exprs) {
    for (const auto& expr : exprs) {
      ret.emplace_back(expr);
    }
  };
  EmplaceDimExpr(lhs);
  EmplaceDimExpr(rhs);
  return ret;
}

std::pair<std::vector<DimExpr>, std::vector<DimExpr>> DimExprBuilder::SplitAt(
    const std::vector<DimExpr> dim_exprs, int index) {
  PADDLE_ENFORCE_EQ(
      index > 0 && index < static_cast<int>(dim_exprs.size()),
      true,
      common::errors::InvalidArgument(
          "Index invalid, index = %d, dim_exprs.size() = %d. Please check "
          "your inputs.",
          index,
          dim_exprs.size()));
  std::vector<DimExpr> lhs(dim_exprs.begin(), dim_exprs.begin() + index);
  std::vector<DimExpr> rhs(dim_exprs.begin() + index, dim_exprs.end());
  return std::make_pair(lhs, rhs);
}

}  // namespace symbol
