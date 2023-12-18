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
  explicit UnaryDimExpr(const T& d) : data(std::make_shared<Data>(d)) {}
  struct Data {
    explicit Data(const T& d) : data(d) {}
    T data;
  };

  const Data& operator*() const { return *data; }
  Data& operator*() { return *data; }
  const Data* operator->() const { return data.get(); }
  Data* operator->() { return data.get(); }

  std::shared_ptr<Data> data;
};

template <typename T>
struct BinaryDimExpr {
  explicit BinaryDimExpr(const T& l, const T& r)
      : data(std::make_shared<Data>(l, r)) {}

  struct Data {
    explicit Data(const T& l, const T& r) : lhs(l), rhs(r) {}
    T lhs;
    T rhs;
  };

  const Data& operator*() const { return *data; }
  Data& operator*() { return *data; }
  const Data* operator->() const { return data.get(); }
  Data* operator->() { return data.get(); }

  std::shared_ptr<Data> data;
};

template <typename T>
struct VariadicDimExpr {
  explicit VariadicDimExpr(const std::vector<T>& vec)
      : data(std::make_shared<Data>(vec)) {}

  using Data = std::vector<T>;

  const Data& operator*() const { return *data; }
  Data& operator*() { return *data; }
  const Data* operator->() const { return data.get(); }
  Data* operator->() { return data.get(); }

  std::shared_ptr<Data> data;
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
//         | Negative DimExpr
//         | Reciprocal DimExpr
//         | Add DimExpr
//         | Mul DimExpr
//         | Max DimExpr
//         | Min DimExpr
//         | Broadcast DimExpr
using DimExprBase = std::variant<std::int64_t,
                                 std::string,
                                 Negative<DimExpr>,
                                 Reciprocal<DimExpr>,
                                 Add<DimExpr>,
                                 Mul<DimExpr>,
                                 Max<DimExpr>,
                                 Min<DimExpr>,
                                 Broadcast<DimExpr>>;

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

  const DimExprBase& variant() const {
    return static_cast<const DimExprBase&>(*this);
  }

  DimExpr operator+(const DimExpr& other) const;
  DimExpr operator-(const DimExpr& other) const;
  DimExpr operator*(const DimExpr& other) const;
  DimExpr operator/(const DimExpr& other) const;
  bool operator==(const DimExpr& other) const;
  bool operator!=(const DimExpr& other) const;
};

// DimExprConstraint = Equal DimExpr
//                   | Broadcastable DimExpr
using DimExprConstraint = std::variant<Equal<DimExpr>, Broadcastable<DimExpr>>;

// ValueShapeDimExprs = tValue [DimExpr] | tShape [DimExpr]
template <typename T>
class ValueShape {
 public:
  explicit ValueShape(const std::vector<T>& shape)
      : value_(std::nullopt), shape_(shape) {}
  ValueShape() = default;
  ValueShape(const ValueShape&) = default;
  ValueShape(ValueShape&&) = default;
  ValueShape& operator=(const ValueShape&) = default;
  ValueShape& operator=(ValueShape&&) = default;

  static ValueShape MakeConsistentValueShape(
      const std::vector<T>& shape,
      const std::function<std::optional<T>(int)>& GetValueByIndex) {
    if (shape.empty()) {
      if (!GetValueByIndex(0).has_value()) {
        return ValueShape(std::nullopt, shape);
      }
      return ValueShape({GetValueByIndex(0).value()}, shape);
    } else if (shape.size() == 1) {
      if (!shape.at(0).template Has<std::int64_t>()) {
        return ValueShape(std::nullopt, shape);
      }
      std::vector<T> value{};
      std::int64_t value_size = shape.at(0).template Get<std::int64_t>();
      for (std::size_t i = 0; i < value_size; ++i) {
        if (!GetValueByIndex(i).has_value()) {
          return ValueShape(std::nullopt, shape);
        }
        value.push_back(GetValueByIndex(i).value());
      }
      return ValueShape(value, shape);
    } else {
      return ValueShape(std::nullopt, shape);
    }
  }

  static ValueShape MakeConsistentValueShape(const std::vector<T>& value) {
    T shape(std::int64_t(value.size()));
    return ValueShape(std::vector<T>{shape}, value);
  }

  const std::optional<std::vector<T>>& shape() const { return shape_; }
  const std::optional<std::vector<T>>& value() const { return value_; }

 private:
  explicit ValueShape(const std::vector<T>& value, const std::vector<T>& shape)
      : value_(value), shape_(shape) {}

  std::optional<std::vector<T>> value_;
  std::optional<std::vector<T>> shape_;
};

using ValueShapeDimExprs = ValueShape<DimExpr>;

}  // namespace symbol
