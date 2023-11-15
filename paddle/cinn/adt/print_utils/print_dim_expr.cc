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

#include "paddle/cinn/adt/print_utils/print_dim_expr.h"

namespace cinn::adt {

std::string ToTxtString(const DimExpr& loop_size);

namespace {

std::string ToTxtStringImpl(std::int64_t dim_expr) {
  return std::to_string(dim_expr);
}

std::string ToTxtStringImpl(const SymbolicDim& dim_expr) {
  return std::string("sym_") + std::to_string(dim_expr.value().unique_id());
}

std::string ToTxtStringImpl(const Negative<DimExpr>& dim_expr) {
  const auto& [item] = dim_expr.tuple();
  return std::string("-") + ToTxtString(item);
}

std::string ToTxtStringImpl(const Reciprocal<DimExpr>& dim_expr) {
  const auto& [item] = dim_expr.tuple();
  return std::string("1 / (") + ToTxtString(item) + std::string(")");
}

std::string ListDimExprToTxtString(const List<DimExpr>& dim_exprs) {
  std::string ret;
  for (std::size_t i = 0; i < dim_exprs->size(); ++i) {
    if (i > 0) {
      ret += ", ";
    }
    ret += ToTxtString(dim_exprs->at(i));
  }
  return ret;
}

std::string ToTxtStringImpl(const Sum<DimExpr>& dim_expr) {
  const auto& [operands] = dim_expr;
  return std::string() + "Sum(" + ListDimExprToTxtString(operands) + ")";
}

std::string ToTxtStringImpl(const Product<DimExpr>& dim_expr) {
  const auto& [operands] = dim_expr;
  return std::string() + "Prod(" + ListDimExprToTxtString(operands) + ")";
}

std::string ToTxtStringImpl(const BroadcastedDim<DimExpr>& dim_expr) {
  const auto& [operands] = dim_expr;
  return std::string() + "BD(" + ListDimExprToTxtString(operands) + ")";
}

}  // namespace

std::string ToTxtString(const DimExpr& loop_size) {
  return std::visit([&](const auto& impl) { return ToTxtStringImpl(impl); },
                    loop_size.variant());
}

std::string ToTxtString(const List<DimExpr>& loop_sizes) {
  std::string ret;
  ret += "[";
  for (std::size_t idx = 0; idx < loop_sizes->size(); ++idx) {
    if (idx != 0) {
      ret += ", ";
    }
    ret += ToTxtString(loop_sizes.Get(idx));
  }
  ret += "]";
  return ret;
}

}  // namespace cinn::adt
