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
#include "paddle/fluid/platform/hostdevice.h"

namespace paddle {
namespace framework {
template <typename T, size_t N>
class Array {
  static_assert(N > 0, "The size of array must be larger than 0");

 public:
  HOSTDEVICE Array() {}

  HOSTDEVICE explicit Array(const T &val) {
    for (size_t i = 0; i < N; ++i) data_[i] = val;
  }

  HOSTDEVICE const T *Get() const { return data_; }

  HOSTDEVICE T *GetMutable() { return data_; }

  HOSTDEVICE T &operator[](size_t index) { return data_[index]; }

  HOSTDEVICE const T &operator[](size_t index) const { return data_[index]; }

  HOSTDEVICE constexpr size_t size() const { return N; }

 private:
  T data_[N];
};

}  // namespace framework
}  // namespace paddle
