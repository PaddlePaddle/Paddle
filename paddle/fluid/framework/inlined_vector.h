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

#include <vector>
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace framework {

template <typename T, size_t N>
class InlinedVector {
  static_assert(N > 0, "N must be larger than 0");

 public:
  inline void push_back(const T& item) {
    if (size_ < N) {
      head_[size_] = item;
    } else {
      tail_.emplace_back(item);
    }
    ++size_;
  }

  inline void pop_back() {
    PADDLE_ENFORCE(!empty(), "Try to pop back element from empty vector.");
    if (size_ > N) {
      tail_.pop_back();
    }
    --size_;
  }

  inline const T& back() const {
    PADDLE_ENFORCE(!empty(), "Try to get back element of empty vector.");
    return size_ <= N ? head_[size_ - 1] : tail_.back();
  }

  inline T& back() {
    PADDLE_ENFORCE(!empty(), "Try to get back element of empty vector.");
    return size_ <= N ? head_[size_ - 1] : tail_.back();
  }

  inline bool empty() const { return size_ == 0; }

  inline size_t size() const { return size_; }

  // This API can only be used in unittest
  T& operator[](size_t i) { return i < N ? head_[i] : tail_[i - N]; }

  const T& operator[](size_t i) const {
    return i < N ? head_[i] : tail_[i - N];
  }

  operator std::vector<T>() const {
    std::vector<T> ret;
    ret.reserve(size_);
    for (size_t i = 0; i < size_; ++i) {
      ret.emplace_back((*this)[i]);
    }
    return ret;
  }

 private:
  T head_[N];
  size_t size_{0};
  std::vector<T> tail_;
};

}  // namespace framework
}  // namespace paddle
