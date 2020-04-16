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
#include <algorithm>
#include <cstdint>
#include <utility>
#include <vector>
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace framework {

template <typename T, size_t N>
class InlinedVector {
  static_assert(N > 0, "N must be larger than 0");

 public:
  InlinedVector() {}

  explicit InlinedVector(size_t size) { resize(size); }

  template <typename U>
  InlinedVector(std::initializer_list<U> list) {
    reserve(list.size());
    for (const auto& item : list) {
      push_back(item);
    }
  }

  InlinedVector(const InlinedVector<T, N>& other)
      : len_(other.len_), tail_(other.tail_) {
    size_t head_size = HeadSize();
    for (size_t i = 0; i < head_size; ++i) {
      new (MutableHead(i)) T(other.Head(i));
    }
  }

  InlinedVector(InlinedVector<T, N>&& other)
      : len_(other.len_), tail_(std::move(other.tail_)) {
    other.len_ = 0;
    size_t head_size = HeadSize();
    for (size_t i = 0; i < head_size; ++i) {
      auto* other_item = other.MutableHead(i);
      new (MutableHead(i)) T(std::move(*other_item));
      other_item->~T();
    }
  }

  ~InlinedVector() {
    size_t head_size = HeadSize();
    for (size_t i = 0; i < head_size; ++i) {
      MutableHead(i)->~T();
    }
  }

  InlinedVector<T, N>& operator=(const InlinedVector<T, N>& other) {
    if (LIKELY(this != &other)) {
      size_t head_size = HeadSize();
      size_t other_head_size = other.HeadSize();
      size_t min_size = std::min(head_size, other_head_size);
      for (size_t i = 0; i < min_size; ++i) {
        *MutableHead(i) = other.Head(i);
      }

      for (size_t i = min_size; i < other_head_size; ++i) {
        new (MutableHead(i)) T(other.Head(i));
      }

      for (size_t i = other_head_size; i < head_size; ++i) {
        MutableHead(i)->~T();
      }

      len_ = other.len_;
      tail_ = other.tail_;
    }
    return *this;
  }

  InlinedVector<T, N>& operator=(InlinedVector<T, N>&& other) {
    if (LIKELY(this != &other)) {
      size_t head_size = HeadSize();
      size_t other_head_size = other.HeadSize();
      size_t min_size = std::min(head_size, other_head_size);
      for (size_t i = 0; i < min_size; ++i) {
        *MutableHead(i) = std::move(*other.MutableHead(i));
      }

      for (size_t i = min_size; i < other_head_size; ++i) {
        new (MutableHead(i)) T(std::move(*other.MutableHead(i)));
      }

      for (size_t i = other_head_size; i < head_size; ++i) {
        MutableHead(i)->~T();
      }

      for (size_t i = 0; i < other_head_size; ++i) {
        other.MutableHead(i)->~T();
      }

      len_ = other.len_;
      other.len_ = 0;
      tail_ = std::move(other.tail_);
    }
    return *this;
  }

  bool empty() const { return len_ == 0; }

  size_t size() const { return len_; }

  size_t capacity() const { return N + tail_.capactiy(); }

  void reserve(size_t size) {
    if (size > N) {
      tail_.reserve(size - N);
    }
  }

  void resize(size_t size) {
    if (size > N) {
      tail_.resize(size - N);
    } else {
      tail_.clear();
    }

    size_t head_size = HeadSize();
    size_t max_head_size = std::min(size, N);

    if (head_size <= max_head_size) {
      for (size_t i = head_size; i < max_head_size; ++i) {
        new (MutableHead(i)) T();
      }
    } else {
      for (size_t i = max_head_size; i < head_size; ++i) {
        MutableHead(i)->~T();
      }
    }
    len_ = size;
  }

  T& operator[](size_t i) { return i < N ? *MutableHead(i) : tail_[i - N]; }

  const T& operator[](size_t i) const { return i < N ? Head(i) : tail_[i - N]; }

  template <typename... Args>
  void emplace_back(Args&&... args) {
    if (LIKELY(len_ < N)) {
      new (MutableHead(len_)) T(std::forward<Args>(args)...);
    } else {
      tail_.emplace_back(std::forward<Args>(args)...);
    }
    ++len_;
  }

  void push_back(const T& item) { emplace_back(item); }

  void push_back(T&& item) { emplace_back(std::move(item)); }

  void pop_back() {
    if (LIKELY(--len_ < N)) {
      MutableHead(len_)->~T();
    } else {
      tail_.pop_back();
    }
  }

  T& front() { return (*this)[0]; }

  const T& front() const { return (*this)[0]; }

  T& back() { return (*this)[len_ - 1]; }

  const T& back() const { return (*this)[len_ - 1]; }

  void clear() {
    size_t head_size = HeadSize();
    for (size_t i = 0; i < head_size; ++i) {
      MutableHead(i)->~T();
    }
    len_ = 0;
    tail_.clear();
  }

 private:
  T* MutableHead(size_t i) { return reinterpret_cast<T*>(&head_[0]) + i; }

  const T& Head(size_t i) const {
    return *(reinterpret_cast<const T*>(&head_[0]) + i);
  }

  size_t HeadSize() const { return len_ < N ? len_ : N; }

 private:
  uint8_t head_[N * sizeof(T)];
  size_t len_{0};
  std::vector<T> tail_;
};

}  // namespace framework
}  // namespace paddle
