// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include <cstdint>
#include "paddle/fluid/framework/unroll_array_ops.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace framework {

template <typename T, size_t N>
class Array {
 public:
  static constexpr size_t kSize = N;

  HOSTDEVICE inline Array() = default;

  template <typename... Args>
  HOSTDEVICE inline explicit Array(const T &val, Args... args) {
    UnrollVarArgsAssign<T, N>::Run(data_, val, args...);
  }

  HOSTDEVICE inline void Fill(const T &val) {
    UnrollFillConstant<N>::Run(data_, val);
  }

  HOSTDEVICE inline const T *Get() const { return data_; }

  HOSTDEVICE inline T *GetMutable() { return data_; }

  HOSTDEVICE inline T &operator[](size_t index) { return data_[index]; }

  HOSTDEVICE inline const T &operator[](size_t index) const {
    return data_[index];
  }

  HOSTDEVICE constexpr size_t size() const { return N; }

  HOSTDEVICE inline bool operator==(const Array<T, N> &other) const {
    return UnrollCompare<N>::Run(data_, other.data_);
  }

  HOSTDEVICE inline bool operator!=(const Array<T, N> &other) const {
    return !(*this == other);
  }

 private:
  T data_[N];
};

template <typename T>
class Array<T, 0> {
 public:
  static constexpr size_t kSize = 0;

  HOSTDEVICE inline Array() = default;

  HOSTDEVICE inline void Fill(const T &val) {}

  HOSTDEVICE inline constexpr T *Get() const { return nullptr; }

  // Add constexpr to GetMutable() cause warning in MAC
  HOSTDEVICE inline T *GetMutable() { return nullptr; }

  HOSTDEVICE inline T &operator[](size_t index) {
#ifndef __CUDA_ARCH__
    PADDLE_THROW("Array<T, 0> has no element");
#endif
  }

  HOSTDEVICE inline const T &operator[](size_t index) const {
#ifndef __CUDA_ARCH__
    PADDLE_THROW("Array<T, 0> has no element");
#endif
  }

  HOSTDEVICE constexpr size_t size() const { return 0; }

  HOSTDEVICE constexpr bool operator==(const Array<T, 0> &other) const {
    return true;
  }

  HOSTDEVICE constexpr bool operator!=(const Array<T, 0> &other) const {
    return false;
  }
};

}  // namespace framework
}  // namespace paddle
