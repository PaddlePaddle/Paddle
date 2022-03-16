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

// TODO(dev): make it into inner union of IValue or Pimpl
union Holder {
  union ConstValue {
    ConstValue() : as_int(0) {}

    int64_t as_int;
    double as_double;
    bool as_bool;
    // TODO(dev): Should add a member to log device info.
    // TODO(dev): Add a ptr to represent List<int64_t/double/bool>
  };

  ConstValue cv;
  Tensor as_tensor;

  Holder() : cv() {}
  ~Holder() {}
};

class IValue {
 public:
  IValue(const Holder& holder, bool is_tensor) : is_tensor_(is_tensor) {
    if (IsTensor()) {
      holder_.as_tensor = holder.as_tensor;
    } else {
      holder_.cv = holder.cv;
    }
  }

  ~IValue() {}

  IValue(const IValue& rhs) {
    if (rhs.is_tensor_) {
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

  explicit IValue(const Tensor& t) : is_tensor_(true) { holder_.as_tensor = t; }

  explicit IValue(int val) : is_tensor_(false) { holder_.cv.as_int = val; }

  explicit IValue(double val) : is_tensor_(false) {
    holder_.cv.as_double = val;
  }

  explicit IValue(bool val) : is_tensor_(false) { holder_.cv.as_bool = val; }

  bool IsTensor() const { return is_tensor_; }

  // Tensor AsTensor() &&;
  // Tensor& AsTensor() &;
  const Tensor& AsTensor() const { return holder_.as_tensor; }

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
  bool is_tensor_;
};
}  // namespace jit
}  // namespace paddle
