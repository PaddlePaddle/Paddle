// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

// TODO(dev): Replace it with framework class
#include <memory>
#include <string>
#include <vector>

#include "paddle/phi/api/include/tensor.h"

namespace paddle {
namespace jit {

using Tensor = paddle::experimental::Tensor;

enum class Tag : int8_t { None = -1, Tensor, Bool, Int, Double };

// TODO(dev): make it into inner union of IValue or Pimpl
union Holder {
  union ConstValue {
    ConstValue() : as_int(0) {}

    int64_t as_int;
    double as_double;
    bool as_bool;
    // TODO(dev): Should add a member to log device info.
    // TODO(dev): Add a ptr to represent List<int64_t/double/bool> or tensor*
    // as_intrusive_ptr*
  };

  ConstValue cv;
  Tensor as_tensor;

  Holder() : cv() {}
  ~Holder() {}
};

class IValue {
 public:
  IValue() = default;
  ~IValue() {
    if (IsTensor()) {
      holder_.as_tensor.~Tensor();
    }
  }

  IValue(const Holder& holder, Tag tag) : tag_(tag) {
    if (IsTensor()) {
      new (&holder_.as_tensor) Tensor();
      holder_.as_tensor = holder.as_tensor;
    } else {
      holder_.cv = holder.cv;
    }
  }

  IValue(const IValue& rhs) : tag_(rhs.tag_) {
    if (IsTensor()) {
      holder_.as_tensor = rhs.holder_.as_tensor;
    } else {
      holder_.cv = rhs.holder_.cv;
    }
  }

  IValue(IValue&& rhs) noexcept { MoveFrom(std::move(rhs)); }

  IValue& operator=(IValue&& rhs) noexcept {
    if (&rhs == this) {
      return *this;
    }
    MoveFrom(std::move(rhs));
    return *this;
  }

  IValue& operator=(const IValue& rhs) {
    *this = IValue(rhs);
    return *this;
  }

  explicit IValue(const Tensor& t) : tag_(Tag::Tensor) {
    new (&holder_.as_tensor) Tensor();
    holder_.as_tensor = t;
    std::cout << holder_.as_tensor.defined() << std::endl;
    std::cout << holder_.as_tensor.is_initialized() << std::endl;
  }
  explicit IValue(int val) : tag_(Tag::Int) { holder_.cv.as_int = val; }
  explicit IValue(double val) : tag_(Tag::Double) {
    holder_.cv.as_double = val;
  }
  explicit IValue(bool val) : tag_(Tag::Bool) { holder_.cv.as_bool = val; }

  bool IsTensor() const { return tag_ == Tag::Tensor; }
  bool IsBool() const { return tag_ == Tag::Bool; }
  bool IsInt() const { return tag_ == Tag::Int; }
  bool IsDouble() const { return tag_ == Tag::Double; }

  // Tensor AsTensor() &&;
  // Tensor& AsTensor() &;
  const Tensor& AsTensor() const {
    std::cout << holder_.as_tensor.defined() << std::endl;
    std::cout << holder_.as_tensor.is_initialized() << std::endl;
    return holder_.as_tensor;
  }

  int64_t AsInt() const { return holder_.cv.as_int; }
  double AsDouble() const { return holder_.cv.as_double; }
  bool AsBool() const { return holder_.cv.as_bool; }

  // std::vector<int64_t> AsInts() const;
  // std::vector<double> AsDoubles() const;
  // std::vector<bool> AsBools() const;

 private:
  // TODO(dev): implement it.
  void MoveFrom(IValue&& rhs) {}
  // Inner data struct to represent both Tensor and Plain constant value;
  Holder holder_;
  // TODO(dev): Use Tag to distingwish Tensor/bool/int64_t/double
  Tag tag_{Tag::None};
};
}  // namespace jit
}  // namespace paddle
