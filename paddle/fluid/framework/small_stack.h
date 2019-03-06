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

#include <array>
#include <deque>
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace framework {

template <typename T, size_t N>
class SmallStack {
  static_assert(N > 0, "N must be larger than 0");

 public:
  inline void push(const T& item) {
    if (size_ < N) {
      head_[size_] = item;
    } else {
      tail_.emplace_back(item);
    }
    ++size_;
  }

  inline void pop() {
    PADDLE_ENFORCE(!empty(), "Try to pop element from empty stack.");
    if (size_ > N) {
      tail_.pop_back();
    }
    --size_;
  }

  inline const T& top() const {
    PADDLE_ENFORCE(!empty(), "Try to get top element of empty stack.");
    return size_ <= N ? head_[size_ - 1] : tail_.back();
  }

  inline T& top() {
    PADDLE_ENFORCE(!empty(), "Try to get top element of empty stack.");
    return size_ <= N ? head_[size_ - 1] : tail_.back();
  }

  inline bool empty() const { return size_ == 0; }

  inline size_t size() const { return size_; }

  // This API can only be used in unittest
  T& operator[](size_t i) { return i < N ? head_[i] : tail_[i - N]; }

  const T& operator[](size_t i) const {
    return i < N ? head_[i] : tail_[i - N];
  }

 private:
  T head_[N];
  std::deque<T> tail_;
  size_t size_;
};

}  // namespace framework
}  // namespace paddle
