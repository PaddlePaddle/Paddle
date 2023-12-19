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

#include "paddle/pir/dialect/shape/utils/dim_expr.h"

namespace symbol {

DimExpr DimExpr::operator+(const DimExpr& other) const {
  return Add<DimExpr>(std::vector{*this, other});
}

DimExpr DimExpr::operator-(const DimExpr& other) const {
  const DimExpr& neg = Negative<DimExpr>(other);
  return Add<DimExpr>(std::vector{*this, neg});
}

DimExpr DimExpr::operator*(const DimExpr& other) const {
  return Mul<DimExpr>(std::vector{*this, other});
}

DimExpr DimExpr::operator/(const DimExpr& other) const {
  const DimExpr& reciprocal = Reciprocal<DimExpr>(other);
  return Mul<DimExpr>(std::vector{*this, reciprocal});
}

namespace {

bool DimExprEqual(std::int64_t lhs, std::int64_t rhs) { return lhs == rhs; }

bool DimExprEqual(const std::string& lhs, const std::string& rhs) {
  return lhs == rhs;
}

bool DimExprEqual(const Negative<DimExpr>& lhs, const Negative<DimExpr>& rhs) {
  return lhs->data == rhs->data;
}

bool DimExprEqual(const Reciprocal<DimExpr>& lhs,
                  const Reciprocal<DimExpr>& rhs) {
  return lhs->data == rhs->data;
}

template <template <typename> class Op>
bool DimExprEqual(const Op<DimExpr>& lhs, const Op<DimExpr>& rhs) {
  if (lhs->size() != rhs->size()) {
    return false;
  }
  for (std::size_t i = 0; i < lhs->size(); ++i) {
    if (lhs->at(i) != rhs->at(i)) {
      return false;
    }
  }
  return true;
}

bool DimExprEqual(const Add<DimExpr>& lhs, const Add<DimExpr>& rhs) {
  return DimExprEqual<Add>(lhs, rhs);
}

bool DimExprEqual(const Mul<DimExpr>& lhs, const Mul<DimExpr>& rhs) {
  return DimExprEqual<Mul>(lhs, rhs);
}

bool DimExprEqual(const Max<DimExpr>& lhs, const Max<DimExpr>& rhs) {
  return DimExprEqual<Max>(lhs, rhs);
}

bool DimExprEqual(const Min<DimExpr>& lhs, const Min<DimExpr>& rhs) {
  return DimExprEqual<Min>(lhs, rhs);
}

bool DimExprEqual(const Broadcast<DimExpr>& lhs,
                  const Broadcast<DimExpr>& rhs) {
  return DimExprEqual<Broadcast>(lhs, rhs);
}

}  // namespace

bool DimExpr::operator==(const DimExpr& other) const {
  if (this == &other) {
    return true;
  }
  return std::visit(
      [](const auto& lhs, const auto& rhs) {
        if constexpr (std::is_same_v<std::decay_t<decltype(lhs)>,
                                     std::decay_t<decltype(rhs)>>) {
          return DimExprEqual(lhs, rhs);
        } else {
          return false;
        }
      },
      this->variant(),
      other.variant());
}

bool DimExpr::operator!=(const DimExpr& other) const {
  return !(*this == other);
}

}  // namespace symbol
