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
  return std::make_shared<Add<DimExpr>>(std::vector{*this, other});
}

DimExpr DimExpr::operator-(const DimExpr& other) const {
  const DimExpr& neg = std::make_shared<Negative<DimExpr>>(other);
  return std::make_shared<Add<DimExpr>>(std::vector{*this, neg});
}

DimExpr DimExpr::operator*(const DimExpr& other) const {
  return std::make_shared<Mul<DimExpr>>(std::vector{*this, other});
}

DimExpr DimExpr::operator/(const DimExpr& other) const {
  const DimExpr& reciprocal = std::make_shared<Reciprocal<DimExpr>>(other);
  return std::make_shared<Mul<DimExpr>>(std::vector{*this, reciprocal});
}

}  // namespace symbol
