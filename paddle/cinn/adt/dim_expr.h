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

#pragma once

#include <ostream>
#include "paddle/cinn/adt/adt.h"
#include "paddle/cinn/adt/arithmetic.h"
#include "paddle/cinn/adt/logical.h"
#include "paddle/cinn/adt/symbolic_dim.h"

namespace cinn::adt {

template <typename T>
struct BroadcastedDim final {
  List<T> operands;

  const BroadcastedDim& tuple() const { return *this; }
};

// DimExpr = std::int64_t
//                 | SymbolicDim
//                 | Negative DimExpr
//                 | Reciprocal DimExpr
//                 | Sum DimExpr
//                 | Product DimExpr
//                 | BroadcastedDim DimExpr
DEFINE_ADT_UNION(DimExpr,
                 std::int64_t,
                 SymbolicDim,
                 Negative<DimExpr>,
                 Reciprocal<DimExpr>,
                 Sum<DimExpr>,
                 Product<DimExpr>,
                 BroadcastedDim<DimExpr>);

DimExpr operator+(const DimExpr& lhs, const DimExpr& rhs);
DimExpr operator-(const DimExpr& lhs, const DimExpr& rhs);
DimExpr operator*(const DimExpr& lhs, const DimExpr& rhs);
DimExpr operator/(const DimExpr& lhs, const DimExpr& rhs);

DimExpr MakeBroadcastedDim(const DimExpr& lhs, const DimExpr& rhs);

bool operator==(const DimExpr& lhs, const DimExpr& rhs);

inline bool operator!=(const DimExpr& lhs, const DimExpr& rhs) {
  return !(lhs == rhs);
}

std::size_t GetHashValue(const DimExpr& expr);

std::ostream& operator<<(std::ostream&, const DimExpr& expr);

}  // namespace cinn::adt
