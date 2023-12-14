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

#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <variant>
#include <vector>

#include "glog/logging.h"

namespace symbol {

#define SYMBOL_NOT_IMPLEMENTED LOG(FATAL) << "Not Implemented"

template <typename T>
struct UnaryDimExpr {
  explicit UnaryDimExpr(const T& d) : data(d) {}
  T data;
};

template <typename T>
struct BinaryDimExpr {
  explicit BinaryDimExpr(const T& l, const T& r) : lhs(l), rhs(r) {}

  T lhs;
  T rhs;
};

template <typename T>
struct VariadicDimExpr {
  explicit VariadicDimExpr(const std::vector<T>& vec) : data(vec) {}

  std::vector<T> data;
};

#define DEFINE_DIM_EXPR_SUBCLASS(class_name, base) \
  template <typename T>                            \
  struct class_name : public base<T> {             \
    using base<T>::base;                           \
  };

DEFINE_DIM_EXPR_SUBCLASS(Negative, UnaryDimExpr);
DEFINE_DIM_EXPR_SUBCLASS(Reciprocal, UnaryDimExpr);
DEFINE_DIM_EXPR_SUBCLASS(Add, VariadicDimExpr);
DEFINE_DIM_EXPR_SUBCLASS(Mul, VariadicDimExpr);
DEFINE_DIM_EXPR_SUBCLASS(Max, VariadicDimExpr);
DEFINE_DIM_EXPR_SUBCLASS(Min, VariadicDimExpr);
DEFINE_DIM_EXPR_SUBCLASS(Broadcast, VariadicDimExpr);
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
                                 std::shared_ptr<Negative<DimExpr>>,
                                 std::shared_ptr<Reciprocal<DimExpr>>,
                                 std::shared_ptr<Add<DimExpr>>,
                                 std::shared_ptr<Mul<DimExpr>>,
                                 std::shared_ptr<Max<DimExpr>>,
                                 std::shared_ptr<Min<DimExpr>>,
                                 std::shared_ptr<Broadcast<DimExpr>>>;

class DimExpr : public DimExprBase {
 public:
  using DimExprBase::DimExprBase;

  template <typename T>
  bool isa() const {
    return std::holds_alternative<T>(*this);
  }

  template <typename T>
  const T& dyn_cast() const {
    return std::get<T>(*this);
  }

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
template <typename T>
class ValueShape {
 public:
  ValueShape(const std::vector<T>& shape,
             const std::function<std::optional<T>(int i)>& ValueGetter)
      : shape_(shape), value_(CalcValue(ValueGetter, shape)) {}

  explicit ValueShape(const std::vector<T>& shape)
      : ValueShape(shape, [](int i) { return std::nullopt; }) {}

  const std::optional<std::vector<T>>& shape() const { return shape_; }
  const std::optional<std::vector<T>>& value() const { return value_; }

 private:
  static std::optional<std::vector<T>> CalcValue(
      const std::function<std::optional<T>(int i)>& ValueGetter,
      const std::vector<T>& shape) {
    SYMBOL_NOT_IMPLEMENTED;
  }

  std::optional<std::vector<T>> shape_;
  std::optional<std::vector<T>> value_;
};

using ValueShapeDimExprs = ValueShape<DimExpr>;

}  // namespace symbol
