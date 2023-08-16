// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
#include <vector>

#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace framework {

template <typename T, size_t N>
class InlinedVector {
  static_assert(N > 0, "N must be larger than 0");

 public:
  inline InlinedVector() { len_ = 0; }

  inline size_t size() const { return len_; }

  inline T& operator[](size_t i) { return i < N ? head_[i] : tail_[i - N]; }

  inline const T& operator[](size_t i) const {
    return i < N ? head_[i] : tail_[i - N];
  }

  inline void emplace_back(const T& item) {
    if (LIKELY(len_ < N)) {
      head_[len_++] = item;
    } else {
      tail_.emplace_back(item);
      ++len_;
    }
  }

  inline void pop_back() {
    if (UNLIKELY(len_ > N)) {
      tail_.pop_back();
    }
    --len_;
  }

  inline T& back() {
    if (LIKELY(len_ <= N)) {
      return head_[len_ - 1];
    } else {
      return tail_.back();
    }
  }

 private:
  T head_[N];
  size_t len_;
  std::vector<T> tail_;
};

}  // namespace framework
}  // namespace paddle
