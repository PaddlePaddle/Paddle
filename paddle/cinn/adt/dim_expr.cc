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

#include "paddle/cinn/adt/dim_expr.h"
#include <type_traits>
#include "paddle/cinn/adt/print_utils/print_dim_expr.h"

namespace cinn::adt {

namespace {

template <typename T0, typename T1>
bool DimExprEqualImpl(const T0&, const T1&) {
  LOG(FATAL) << "Dead code";
}

bool DimExprEqualImpl(std::int64_t lhs, std::int64_t rhs) { return lhs == rhs; }

bool DimExprEqualImpl(const SymbolicDim& lhs, const SymbolicDim& rhs) {
  return lhs == rhs;
}

bool DimExprEqualImpl(const Negative<DimExpr>& lhs,
                      const Negative<DimExpr>& rhs) {
  const auto& [lhs_arg0] = lhs.tuple();
  const auto& [rhs_arg0] = rhs.tuple();
  return lhs_arg0 == rhs_arg0;
}

bool DimExprEqualImpl(const Reciprocal<DimExpr>& lhs,
                      const Reciprocal<DimExpr>& rhs) {
  const auto& [lhs_arg0] = lhs.tuple();
  const auto& [rhs_arg0] = rhs.tuple();
  return lhs_arg0 == rhs_arg0;
}

bool DimExprEqualImpl(const Sum<DimExpr>& lhs, const Sum<DimExpr>& rhs) {
  const auto& [lhs_operands] = lhs;
  const auto& [rhs_operands] = rhs;
  return lhs_operands == rhs_operands;
}

bool DimExprEqualImpl(const Product<DimExpr>& lhs,
                      const Product<DimExpr>& rhs) {
  const auto& [lhs_operands] = lhs;
  const auto& [rhs_operands] = rhs;
  return lhs_operands == rhs_operands;
}

bool DimExprEqualImpl(const BroadcastedDim<DimExpr>& lhs,
                      const BroadcastedDim<DimExpr>& rhs) {
  const auto& [lhs_operands] = lhs;
  const auto& [rhs_operands] = rhs;
  return lhs_operands == rhs_operands;
}

}  // namespace

bool operator==(const DimExpr& lhs, const DimExpr& rhs) {
  return std::visit(
      [](const auto& lhs, const auto& rhs) {
        if (std::is_same_v<std::decay_t<decltype(lhs)>,
                           std::decay_t<decltype(rhs)>>) {
          return DimExprEqualImpl(lhs, rhs);
        } else {
          return false;
        }
      },
      lhs.variant(),
      rhs.variant());
}

namespace {

std::size_t GetHashValueImpl(std::int64_t expr) { return expr; }

std::size_t GetHashValueImpl(const SymbolicDim& expr) {
  return expr.value().unique_id();
}

std::size_t GetHashValueImpl(const Negative<DimExpr>& expr) {
  const auto& [item] = expr.tuple();
  return -GetHashValue(item);
}

std::size_t GetHashValueImpl(const Reciprocal<DimExpr>& expr) {
  const auto& [item] = expr.tuple();
  return -GetHashValue(item);
}

std::size_t GetHashValueImpl(const List<DimExpr>& exprs) {
  std::size_t ret = 0;
  for (const auto& expr : *exprs) {
    ret = hash_combine(ret, GetHashValue(expr));
  }
}

std::size_t GetHashValueImpl(const Sum<DimExpr>& expr) {
  const auto& [operands] = expr;
  return GetHashValueImpl(operands);
}

std::size_t GetHashValueImpl(const Product<DimExpr>& expr) {
  const auto& [operands] = expr;
  return GetHashValueImpl(operands);
}

std::size_t GetHashValueImpl(const BroadcastedDim<DimExpr>& expr) {
  const auto& [operands] = expr;
  return GetHashValueImpl(operands);
}

}  // namespace

std::size_t GetHashValue(const DimExpr& expr) {
  return std::visit([&](const auto& impl) { return GetHashValueImpl(impl); },
                    expr.variant());
}

DimExpr operator+(const DimExpr& lhs, const DimExpr& rhs) {
  return Sum<DimExpr>{List<DimExpr>{lhs, rhs}};
}

DimExpr operator-(const DimExpr& lhs, const DimExpr& rhs) {
  return Sum<DimExpr>{List<DimExpr>{lhs, Negative<DimExpr>{rhs}}};
}

DimExpr operator*(const DimExpr& lhs, const DimExpr& rhs) {
  return Product<DimExpr>{List<DimExpr>{lhs, rhs}};
}

DimExpr operator/(const DimExpr& lhs, const DimExpr& rhs) {
  return Product<DimExpr>{List<DimExpr>{lhs, Reciprocal<DimExpr>{rhs}}};
}

DimExpr MakeBroadcastedDim(const DimExpr& lhs, const DimExpr& rhs) {
  return BroadcastedDim<DimExpr>{List<DimExpr>{lhs, rhs}};
}

std::ostream& operator<<(std::ostream& stream, const DimExpr& expr) {
  stream << ToTxtString(expr);
  return stream;
}
}  // namespace cinn::adt
