/***************************************************************************************************
 * Copyright (c) 2023 - 2023 NVIDIA CORPORATION & AFFILIATES. All rights
 *reserved. SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 *ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 *LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 *CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 *SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 *INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 *CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 *ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
#pragma once

#include <cstddef>
#include <utility>

#include <cute/config.hpp>

namespace cute {

template <class T, std::size_t N>
struct array_view {
  using value_type = T;
  using size_type = std::size_t;
  using difference_type = std::ptrdiff_t;
  using reference = value_type&;
  using const_reference = const value_type&;
  using pointer = value_type*;
  using const_pointer = const value_type*;
  using iterator = pointer;
  using const_iterator = const_pointer;

  array_view(array<T, N>& a) : __elems_(a.data()) {}

  CUTE_HOST_DEVICE
  reference operator[](size_type pos) { return begin()[pos]; }

  CUTE_HOST_DEVICE
  const_reference operator[](size_type pos) const { return begin()[pos]; }

  CUTE_HOST_DEVICE
  reference front() { return *begin(); }

  CUTE_HOST_DEVICE
  const_reference front() const { return *begin(); }

  CUTE_HOST_DEVICE
  reference back() {
    // return *rbegin();
    return operator[](N - 1);
  }

  CUTE_HOST_DEVICE
  const_reference back() const {
    // return *rbegin();
    return operator[](N - 1);
  }

  CUTE_HOST_DEVICE
  T* data() { return __elems_; }

  CUTE_HOST_DEVICE
  const T* data() const { return __elems_; }

  CUTE_HOST_DEVICE
  iterator begin() { return data(); }

  CUTE_HOST_DEVICE
  const_iterator begin() const { return data(); }

  CUTE_HOST_DEVICE
  const_iterator cbegin() { return begin(); }

  CUTE_HOST_DEVICE
  const_iterator cbegin() const { return begin(); }

  CUTE_HOST_DEVICE
  iterator end() { return data() + size(); }

  CUTE_HOST_DEVICE
  const_iterator end() const { return data() + size(); }

  CUTE_HOST_DEVICE
  const_iterator cend() { return end(); }

  CUTE_HOST_DEVICE
  const_iterator cend() const { return end(); }

  CUTE_HOST_DEVICE constexpr bool empty() const { return size() == 0; }

  CUTE_HOST_DEVICE constexpr size_type size() const { return N; }

  CUTE_HOST_DEVICE constexpr size_type max_size() const { return size(); }

  CUTE_HOST_DEVICE
  void fill(const T& value) {
    for (auto& e : *this) {
      e = value;
    }
  }

  CUTE_HOST_DEVICE
  void swap(array_view& other) {
    using std::swap;
    swap(__elems_, other.__elems_);
  }

  value_type* __elems_;
};

template <class T, std::size_t N>
CUTE_HOST_DEVICE bool operator==(const array_view<T, N>& lhs,
                                 const array_view<T, N>& rhs) {
  for (std::size_t i = 0; i < N; ++i) {
    if (lhs[i] != rhs[i]) return false;
  }

  return true;
}

template <typename T, std::size_t N>
CUTE_HOST_DEVICE void clear(array_view<T, N>& a) {
  a.fill(T(0));
}

template <class T, std::size_t N>
CUTE_HOST_DEVICE void swap(array_view<T, N>& a, array_view<T, N>& b) {
  a.swap(b);
}

}  // namespace cute

//
// Specialize tuple-related functionality for cute::array_view
//

#include <tuple>

namespace cute {

template <std::size_t I, class T, std::size_t N>
CUTE_HOST_DEVICE constexpr T& get(array_view<T, N>& a) {
  static_assert(I < N, "Index out of range");
  return a[I];
}

template <std::size_t I, class T, std::size_t N>
CUTE_HOST_DEVICE constexpr const T& get(const array_view<T, N>& a) {
  static_assert(I < N, "Index out of range");
  return a[I];
}

template <std::size_t I, class T, std::size_t N>
CUTE_HOST_DEVICE constexpr T&& get(array_view<T, N>&& a) {
  static_assert(I < N, "Index out of range");
  return std::move(a[I]);
}

}  // end namespace cute

namespace std {

template <class T, std::size_t N>
struct tuple_size<cute::array_view<T, N>>
    : std::integral_constant<std::size_t, N> {};

template <std::size_t I, class T, std::size_t N>
struct tuple_element<I, cute::array_view<T, N>> {
  using type = T;
};

}  // namespace std
