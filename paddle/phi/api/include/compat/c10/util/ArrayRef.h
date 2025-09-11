// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

// #The file has been adapted from pytorch project
// #Licensed under  BSD-style license -
// https://github.com/pytorch/pytorch/blob/main/LICENSE

#pragma once

#include <c10/util/Exception.h>
#include <cstdint>
#include <functional>
#include <iterator>
#include <vector>

#include "paddle/phi/common/int_array.h"

namespace c10 {

#define TORCH_CHECK_CONSTEXPR(COND, MSG) \
  ((COND) ? void(0) : throw std::runtime_error(MSG))

template <typename T>
class ArrayRef {
 private:
  /// The start of the array, in an external buffer.
  const T* Data;

  /// The number of elements.
  size_t Length;

 public:
  using iterator = const T*;
  using const_iterator = const T*;
  using size_type = size_t;
  using value_type = T;

  using reverse_iterator = std::reverse_iterator<iterator>;

  /* implicit */ constexpr ArrayRef() : Data(nullptr), Length(0) {}

  constexpr ArrayRef(const T& OneElt) : Data(&OneElt), Length(1) {}  // NOLINT

  constexpr ArrayRef(const T* data, size_t length)
      : Data(data), Length(length) {}

  constexpr ArrayRef(const T* begin, const T* end)
      : Data(begin), Length(end - begin) {}

  template <typename Container,
            typename U = decltype(std::declval<Container>().data()),
            typename = std::enable_if_t<(std::is_same_v<U, T*> ||
                                         std::is_same_v<U, T const*>)>>
  /* implicit */ ArrayRef(const Container& container)  // NOLINT
      : Data(container.data()), Length(container.size()) {}

  template <typename A>
  /* implicit */ ArrayRef(const std::vector<T, A>& Vec)  // NOLINT
      : Data(Vec.data()), Length(Vec.size()) {
    static_assert(!std::is_same_v<T, bool>,
                  "ArrayRef<bool> cannot be constructed from a "
                  "std::vector<bool> bitfield.");
  }

  template <size_t N>
  /* implicit */ constexpr ArrayRef(const std::array<T, N>& Arr)  // NOLINT
      : Data(Arr.data()), Length(N) {}

  template <size_t N>
  /* implicit */ constexpr ArrayRef(const T (&Arr)[N])  // NOLINT
      : Data(Arr), Length(N) {}

  /* implicit */ constexpr ArrayRef(const std::initializer_list<T>& Vec)
      : Data(std::begin(Vec) == std::end(Vec) ? static_cast<T*>(nullptr)
                                              : std::begin(Vec)),
        Length(Vec.size()) {}

  constexpr iterator begin() const { return Data; }
  constexpr iterator end() const { return Data + Length; }

  constexpr const_iterator cbegin() const { return Data; }
  constexpr const_iterator cend() const { return Data + Length; }

  constexpr reverse_iterator rbegin() const { return reverse_iterator(end()); }
  constexpr reverse_iterator rend() const { return reverse_iterator(begin()); }

  constexpr bool allMatch(const std::function<bool(const T&)>& pred) const {
    return std::all_of(cbegin(), cend(), pred);
  }

  constexpr bool empty() const { return Length == 0; }

  constexpr const T* data() const { return Data; }

  constexpr size_t size() const { return Length; }

  constexpr const T& front() const {
    TORCH_CHECK_CONSTEXPR(
        !empty(), "ArrayRef: attempted to access front() of empty list");
    return Data[0];
  }

  constexpr const T& back() const {
    TORCH_CHECK_CONSTEXPR(!empty(),
                          "ArrayRef: attempted to access back() of empty list");
    return Data[Length - 1];
  }

  constexpr bool equals(ArrayRef RHS) const {
    return Length == RHS.Length && std::equal(begin(), end(), RHS.begin());
  }

  /// slice(n, m) - Take M elements of the array starting at element N
  constexpr ArrayRef<T> slice(size_t N, size_t M) const {
    TORCH_CHECK_CONSTEXPR(N + M <= size(), "ArrayRef: invalid slice");
    return ArrayRef<T>(data() + N, M);
  }

  /// slice(n) - Chop off the first N elements of the array.
  constexpr ArrayRef<T> slice(size_t N) const {
    TORCH_CHECK_CONSTEXPR(N <= size(), "ArrayRef: invalid slice");
    return slice(N, size() - N);
  }

  constexpr const T& operator[](size_t Index) const { return Data[Index]; }

  /// Vector compatibility
  constexpr const T& at(size_t Index) const {
    TORCH_CHECK_CONSTEXPR(Index < Length, "ArrayRef: invalid index");
    return Data[Index];
  }

  template <typename U>
  std::enable_if_t<std::is_same_v<U, T>, ArrayRef<T>>& operator=(
      U&& Temporary) = delete;

  template <typename U>
  std::enable_if_t<std::is_same_v<U, T>, ArrayRef<T>>& operator=(
      std::initializer_list<U>) = delete;

  std::vector<T> vec() const { return std::vector<T>(Data, Data + Length); }

  const paddle::experimental::IntArray _PD_ToPaddleIntArray() const {
    return paddle::experimental::IntArray(Data, Length);
  }
};

template <typename T>
bool operator==(c10::ArrayRef<T> a1, c10::ArrayRef<T> a2) {
  return a1.equals(a2);
}

template <typename T>
bool operator!=(c10::ArrayRef<T> a1, c10::ArrayRef<T> a2) {
  return !a1.equals(a2);
}

template <typename T>
bool operator==(const std::vector<T>& a1, c10::ArrayRef<T> a2) {
  return c10::ArrayRef<T>(a1).equals(a2);
}

template <typename T>
bool operator!=(const std::vector<T>& a1, c10::ArrayRef<T> a2) {
  return !c10::ArrayRef<T>(a1).equals(a2);
}

template <typename T>
bool operator==(c10::ArrayRef<T> a1, const std::vector<T>& a2) {
  return a1.equals(c10::ArrayRef<T>(a2));
}

template <typename T>
bool operator!=(c10::ArrayRef<T> a1, const std::vector<T>& a2) {
  return !a1.equals(c10::ArrayRef<T>(a2));
}
using IntArrayRef = ArrayRef<int64_t>;

}  // namespace c10

namespace at {
using c10::ArrayRef;
using c10::IntArrayRef;
}  // namespace at

namespace torch {
using c10::ArrayRef;
using c10::IntArrayRef;
}  // namespace torch
