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

#include <cute/config.hpp>

#include <cute/container/alignment.hpp>
#include <cute/numeric/int.hpp>
#include <cute/numeric/math.hpp>

namespace cute {

template <typename T, std::size_t N, std::size_t Alignment = 16>
struct array_aligned : public aligned_struct<Alignment> {
  /// Make sure the Alignment makes sense wrt the size of elements.
  static_assert(Alignment == 16 || Alignment >= sizeof(T),
                "Alignment is too small");
  /// Alignment must be a power of two
  static_assert(has_single_bit(Alignment), "Alignment must be a power of two");

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

  CUTE_HOST_DEVICE constexpr T* data() { return reinterpret_cast<T*>(storage); }

  CUTE_HOST_DEVICE constexpr T const* data() const {
    return reinterpret_cast<T const*>(storage);
  }

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

  CUTE_HOST_DEVICE constexpr void fill(T const& value) {
    for (auto& e : *this) {
      e = value;
    }
  }

  CUTE_HOST_DEVICE constexpr void clear() { fill(T(0)); }

  // Not private, we want trivial type
  // private:

  /// Storage type to use for Elements
  using StorageType = typename uint_byte<static_cast<int>(Alignment)>::type;

  /// Ensure that there's enough storage for all elements
  static_assert(sizeof(StorageType) <= Alignment,
                "StorageType is too big for given alignment");

  /// Number of elements in the storage
  static constexpr std::size_t storageN =
      (sizeof(T) * N + sizeof(StorageType) - 1) / sizeof(StorageType);

  /// The storage.
  StorageType storage[storageN > 0 ? storageN : 1];
};

//
// Operators
//

template <typename T, std::size_t N, std::size_t Alignment>
CUTE_HOST_DEVICE constexpr void clear(array_aligned<T, N, Alignment>& a) {
  a.clear();
}

template <typename T, std::size_t N, std::size_t Alignment>
CUTE_HOST_DEVICE constexpr void fill(array_aligned<T, N, Alignment>& a,
                                     T const& value) {
  a.fill(value);
}

}  // end namespace cute

//
// Specialize tuple-related functionality for cute::array
//

#include <tuple>

namespace cute {

template <std::size_t I, class T, std::size_t N>
CUTE_HOST_DEVICE constexpr T& get(array_aligned<T, N>& a) {
  static_assert(I < N, "Index out of range");
  return a[I];
}

template <std::size_t I, class T, std::size_t N>
CUTE_HOST_DEVICE constexpr T const& get(array_aligned<T, N> const& a) {
  static_assert(I < N, "Index out of range");
  return a[I];
}

template <std::size_t I, class T, std::size_t N>
CUTE_HOST_DEVICE constexpr T&& get(array_aligned<T, N>&& a) {
  static_assert(I < N, "Index out of range");
  return std::move(a[I]);
}

}  // end namespace cute

namespace std {

template <class T, std::size_t N>
struct tuple_size<cute::array_aligned<T, N>>
    : std::integral_constant<std::size_t, N> {};

template <std::size_t I, class T, std::size_t N>
struct tuple_element<I, cute::array_aligned<T, N>> {
  using type = T;
};

}  // namespace std
