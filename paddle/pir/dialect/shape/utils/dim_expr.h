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
class List final {
 public:
  List(const List&) = default;
  List(List&&) = default;
  List& operator=(const List&) = default;
  List& operator=(List&&) = default;

  using value_type = T;

  explicit List() : vector_(std::make_shared<std::vector<T>>()) {}

  template <
      typename Arg,
      std::enable_if_t<!std::is_same_v<std::decay_t<Arg>, List>, bool> = true>
  explicit List(Arg&& arg)
      : vector_(std::make_shared<std::vector<T>>(
            std::vector<T>{std::forward<Arg>(arg)})) {}

  template <typename Arg0, typename Arg1, typename... Args>
  List(Arg0&& arg0, Arg1&& arg1, Args&&... args)
      : vector_(std::make_shared<std::vector<T>>(
            std::vector<T>{std::forward<Arg0>(arg0),
                           std::forward<Arg1>(arg1),
                           std::forward<Args>(args)...})) {}

  bool operator==(const List& other) const {
    if (&vector() == &other.vector()) {
      return true;
    }
    return vector() == other.vector();
  }

  bool operator!=(const List& other) const { return !(*this == other); }

  std::vector<T>& operator*() const { return *vector_; }
  std::vector<T>* operator->() const { return vector_.get(); }

  const std::vector<T>& vector() const { return *vector_; }

  const auto& Get(std::size_t idx) const { return vector_->at(idx); }

 private:
  std::shared_ptr<std::vector<T>> vector_;
};

template <typename T>
struct Negative final : public UnaryDimExpr<T> {
  using UnaryDimExpr<T>::UnaryDimExpr;
};

template <typename T>
struct Reciprocal final : public UnaryDimExpr<T> {
  using UnaryDimExpr<T>::UnaryDimExpr;
};

template <typename T>
struct Add final {
  List<T> operands;
};

template <typename T>
struct Mul final {
  List<T> operands;
};

template <typename T>
struct Max final {
  List<T> operands;
};

template <typename T>
struct Min final {
  List<T> operands;
};

template <typename T>
struct Broadcast final {
  List<T> operands;
};

template <typename T>
struct Equal final : public BinaryDimExpr<T> {
  using BinaryDimExpr<T>::BinaryDimExpr;
};

template <typename T>
struct Broadcastable final : public BinaryDimExpr<T> {
  using BinaryDimExpr<T>::BinaryDimExpr;
};

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

  template <typename T>
  bool Has() const {
    return std::holds_alternative<T>(*this);
  }

  template <typename T>
  const T& Get() const {
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

  static ValueShape MakeConsistentValueShape(const std::vector<T>& value) {
    T shape(std::int64_t(value.size()));
    return ValueShape(value, std::vector<T>{shape});
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
