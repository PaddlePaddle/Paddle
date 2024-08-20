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
struct array {
  using value_type = T;
  using size_type = std::size_t;
  using difference_type = std::ptrdiff_t;
  using reference = value_type&;
  using const_reference = const value_type&;
  using pointer = value_type*;
  using const_pointer = const value_type*;
  using iterator = pointer;
  using const_iterator = const_pointer;

  CUTE_HOST_DEVICE constexpr reference operator[](size_type pos) {
    return begin()[pos];
  }

  CUTE_HOST_DEVICE constexpr const_reference operator[](size_type pos) const {
    return begin()[pos];
  }

  CUTE_HOST_DEVICE constexpr reference front() { return *begin(); }

  CUTE_HOST_DEVICE constexpr const_reference front() const { return *begin(); }

  CUTE_HOST_DEVICE constexpr reference back() {
    // return *rbegin();
    return operator[](N - 1);
  }

  CUTE_HOST_DEVICE constexpr const_reference back() const {
    // return *rbegin();
    return operator[](N - 1);
  }

  CUTE_HOST_DEVICE constexpr T* data() { return __elems_; }

  CUTE_HOST_DEVICE constexpr T const* data() const { return __elems_; }

  CUTE_HOST_DEVICE constexpr iterator begin() { return data(); }

  CUTE_HOST_DEVICE constexpr const_iterator begin() const { return data(); }

  CUTE_HOST_DEVICE constexpr const_iterator cbegin() { return begin(); }

  CUTE_HOST_DEVICE constexpr const_iterator cbegin() const { return begin(); }

  CUTE_HOST_DEVICE constexpr iterator end() { return data() + size(); }

  CUTE_HOST_DEVICE constexpr const_iterator end() const {
    return data() + size();
  }

  CUTE_HOST_DEVICE constexpr const_iterator cend() { return end(); }

  CUTE_HOST_DEVICE constexpr const_iterator cend() const { return end(); }

  CUTE_HOST_DEVICE constexpr bool empty() const { return size() == 0; }

  CUTE_HOST_DEVICE constexpr size_type size() const { return N; }

  CUTE_HOST_DEVICE constexpr size_type max_size() const { return size(); }

  CUTE_HOST_DEVICE constexpr void fill(const T& value) {
    for (auto& e : *this) {
      e = value;
    }
  }

  CUTE_HOST_DEVICE constexpr void clear() { fill(T(0)); }

  CUTE_HOST_DEVICE constexpr void swap(array& other) {
    using std::swap;
    for (size_type i = 0; i < size(); ++i) {
      swap((*this)[i], other[i]);
    }
  }

  value_type __elems_[N > 0 ? N : 1];
};

template <class T, std::size_t N>
CUTE_HOST_DEVICE constexpr bool operator==(array<T, N> const& lhs,
                                           array<T, N> const& rhs) {
  for (std::size_t i = 0; i < N; ++i) {
    if (lhs[i] != rhs[i]) {
      return false;
    }
  }
  return true;
}

template <typename T, std::size_t N>
CUTE_HOST_DEVICE constexpr void clear(array<T, N>& a) {
  a.fill(T(0));
}

template <typename T, std::size_t N>
CUTE_HOST_DEVICE constexpr void fill(array<T, N>& a, T const& value) {
  a.fill(value);
}

template <class T, std::size_t N>
CUTE_HOST_DEVICE constexpr void swap(array<T, N>& a, array<T, N>& b) {
  a.swap(b);
}

}  // namespace cute

//
// Specialize tuple-related functionality for cute::array
//

#include <tuple>

namespace cute {

template <std::size_t I, class T, std::size_t N>
CUTE_HOST_DEVICE constexpr T& get(array<T, N>& a) {
  static_assert(I < N, "Index out of range");
  return a[I];
}

template <std::size_t I, class T, std::size_t N>
CUTE_HOST_DEVICE constexpr T const& get(array<T, N> const& a) {
  static_assert(I < N, "Index out of range");
  return a[I];
}

template <std::size_t I, class T, std::size_t N>
CUTE_HOST_DEVICE constexpr T&& get(array<T, N>&& a) {
  static_assert(I < N, "Index out of range");
  return std::move(a[I]);
}

}  // end namespace cute

namespace std {

template <class T, std::size_t N>
struct tuple_size<cute::array<T, N>> : std::integral_constant<std::size_t, N> {
};

template <std::size_t I, class T, std::size_t N>
struct tuple_element<I, cute::array<T, N>> {
  using type = T;
};

}  // namespace std
