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

#include <memory>
#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace symbol {

template <typename T>
struct BinaryDimExpr {
  T lhs;
  T rhs;
};

#define DEFINE_DIM_EXPR_SUBCLASS(class_name, base) \
  template <typename T>                            \
  struct class_name : public base {                \
    using base::base;                              \
  };

DEFINE_DIM_EXPR_SUBCLASS(Add, BinaryDimExpr);
DEFINE_DIM_EXPR_SUBCLASS(Any, BinaryDimExpr);
DEFINE_DIM_EXPR_SUBCLASS(Mul, BinaryDimExpr);
DEFINE_DIM_EXPR_SUBCLASS(Div, BinaryDimExpr);
DEFINE_DIM_EXPR_SUBCLASS(Max, BinaryDimExpr);
DEFINE_DIM_EXPR_SUBCLASS(Min, BinaryDimExpr);
DEFINE_DIM_EXPR_SUBCLASS(Broadcast, BinaryDimExpr);
DEFINE_DIM_EXPR_SUBCLASS(Equal, BinaryDimExpr);
DEFINE_DIM_EXPR_SUBCLASS(Broadcastable, BinaryDimExpr);

class DimExpr;

// DimExpr = std::int64_t
//         | std::string
//         | Add DimExpr DimExpr
//         | Any DimExpr DimExpr
//         | Mul DimExpr DimExpr
//         | Div DimExpr DimExpr
//         | Max DimExpr DimExpr
//         | Min DimExpr DimExpr
//         | Broadcast DimExpr DimExpr
using DimExprBase = std::variant<std::int64_t,
                                 std::string,
                                 std::shared_ptr<Add<DimExpr>>,
                                 std::shared_ptr<Any<DimExpr>>,
                                 std::shared_ptr<Mul<DimExpr>>,
                                 std::shared_ptr<Div<DimExpr>>,
                                 std::shared_ptr<Max<DimExpr>>,
                                 std::shared_ptr<Min<DimExpr>>,
                                 std::shared_ptr<Broadcast<DimExpr>>>;

class DimExpr : public DimExprBase {
 public:
  using DimExprBase::DimExprBase;

  DimExpr operator+(const DimExpr& other) const;
  DimExpr operator-(const DimExpr& other) const;
  DimExpr operator*(const DimExpr& other) const;
  DimExpr operator/(const DimExpr& other) const;
};

// DimExprConstraint = Equal DimExpr DimExpr
//                   | Broadcastable DimExpr DimExpr
using DimExprConstraint = std::variant<std::shared_ptr<Equal<DimExpr>>,
                                       std::shared_ptr<Broadcastable<DimExpr>>>;

// ValueShapeDimExprs = tShape DimExpr | tValue DimExpr
struct ValueShapeDimExprs {
  std::optional<std::vector<DimExpr>> shape;
  std::optional<std::vector<DimExpr>> value;
};

}  // namespace symbol
