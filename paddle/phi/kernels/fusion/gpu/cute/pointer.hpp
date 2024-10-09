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

#include <cute/numeric/integral_constant.hpp>
#include <cute/numeric/math.hpp>
#include <cute/util/type_traits.hpp>

namespace cute {

//
// has_dereference to determine if a type is a pointer concept
//

template <class T, class = void>
struct has_dereference : std::false_type {};

template <class T>
struct has_dereference<T, void_t<decltype(*std::declval<T>())>>
    : std::true_type {};

//
// Pointer categories
//

template <class T>
struct is_gmem : false_type {};

template <class T>
struct is_smem : false_type {};

// Anything that is not gmem or smem is rmem
template <class T>
struct is_rmem : bool_constant<not(is_gmem<T>::value || is_smem<T>::value)> {};

//
// A very simplified wrapper for pointers -- use for constructing tagged
// pointers
//
template <class T, class DerivedType>
struct device_ptr {
  using value_type = T;

  CUTE_HOST_DEVICE constexpr device_ptr(T* ptr) : ptr_(ptr) {}

  CUTE_HOST_DEVICE constexpr T* get() const { return ptr_; }

  CUTE_HOST_DEVICE constexpr T& operator*() const { return *ptr_; }

  template <class Index>
  CUTE_HOST_DEVICE constexpr T& operator[](Index const& i) const {
    return ptr_[i];
  }

  template <class Index>
  CUTE_HOST_DEVICE constexpr DerivedType operator+(Index const& i) const {
    return {ptr_ + i};
  }

  CUTE_HOST_DEVICE constexpr friend std::ptrdiff_t operator-(
      device_ptr<T, DerivedType> const& a,
      device_ptr<T, DerivedType> const& b) {
    return a.ptr_ - b.ptr_;
  }

  T* ptr_;
};

//
// gmem_ptr
//

template <class T>
struct gmem_ptr : device_ptr<T, gmem_ptr<T>> {
  using device_ptr<T, gmem_ptr<T>>::device_ptr;
};

template <class T>
CUTE_HOST_DEVICE constexpr gmem_ptr<T> make_gmem_ptr(T* ptr) {
  return {ptr};
}

template <class T>
CUTE_HOST_DEVICE constexpr gmem_ptr<T> make_gmem_ptr(void* ptr) {
  return {reinterpret_cast<T*>(ptr)};
}

template <class T>
struct is_gmem<gmem_ptr<T>> : true_type {};

//
// smem_ptr
//

template <class T>
struct smem_ptr : device_ptr<T, smem_ptr<T>> {
  using device_ptr<T, smem_ptr<T>>::device_ptr;
};

template <class T>
CUTE_HOST_DEVICE constexpr smem_ptr<T> make_smem_ptr(T* ptr) {
  return {ptr};
}

template <class T>
CUTE_HOST_DEVICE constexpr smem_ptr<T> make_smem_ptr(void* ptr) {
  return {reinterpret_cast<T*>(ptr)};
}

template <class T>
struct is_smem<smem_ptr<T>> : true_type {};

//
// rmem_ptr
//

template <class T>
struct rmem_ptr : device_ptr<T, rmem_ptr<T>> {
  using device_ptr<T, rmem_ptr<T>>::device_ptr;
};

template <class T>
CUTE_HOST_DEVICE constexpr rmem_ptr<T> make_rmem_ptr(T* ptr) {
  return {ptr};
}

template <class T>
CUTE_HOST_DEVICE constexpr rmem_ptr<T> make_rmem_ptr(void* ptr) {
  return {reinterpret_cast<T*>(ptr)};
}

template <class T>
struct is_rmem<rmem_ptr<T>> : true_type {};

//
// counting iterator -- quick and dirty
//

struct counting {
  using index_type = int;
  using value_type = index_type;

  CUTE_HOST_DEVICE constexpr counting() : n_(0) {}
  CUTE_HOST_DEVICE constexpr counting(index_type const& n) : n_(n) {}

  CUTE_HOST_DEVICE constexpr index_type operator[](index_type const& i) const {
    return n_ + i;
  }

  CUTE_HOST_DEVICE constexpr index_type const& operator*() const { return n_; }

  CUTE_HOST_DEVICE constexpr counting operator+(index_type const& i) const {
    return {n_ + i};
  }
  CUTE_HOST_DEVICE constexpr counting& operator++() {
    ++n_;
    return *this;
  }

  CUTE_HOST_DEVICE constexpr bool operator==(counting const& other) const {
    return n_ == other.n_;
  }
  CUTE_HOST_DEVICE constexpr bool operator!=(counting const& other) const {
    return n_ != other.n_;
  }

  CUTE_HOST_DEVICE constexpr bool operator<(counting const& other) const {
    return n_ < other.n_;
  }

  index_type n_;
};

//
// recast
//

template <class NewT, class T>
CUTE_HOST_DEVICE constexpr auto recast(T* ptr) {
  return reinterpret_cast<NewT*>(ptr);
}

template <class NewT, class T>
CUTE_HOST_DEVICE constexpr auto recast(T const* ptr) {
  return reinterpret_cast<NewT const*>(ptr);
}

template <class NewT, class T>
CUTE_HOST_DEVICE constexpr auto recast(gmem_ptr<T> const& ptr) {
  return make_gmem_ptr(recast<NewT>(ptr.ptr_));
}

template <class NewT, class T>
CUTE_HOST_DEVICE constexpr auto recast(gmem_ptr<T const> const& ptr) {
  return make_gmem_ptr(recast<NewT const>(ptr.ptr_));
}

template <class NewT, class T>
CUTE_HOST_DEVICE constexpr auto recast(smem_ptr<T> const& ptr) {
  return make_smem_ptr(recast<NewT>(ptr.ptr_));
}

template <class NewT, class T>
CUTE_HOST_DEVICE constexpr auto recast(smem_ptr<T const> const& ptr) {
  return make_smem_ptr(recast<NewT const>(ptr.ptr_));
}

template <class NewT, class T>
CUTE_HOST_DEVICE constexpr auto recast(rmem_ptr<T> const& ptr) {
  return make_rmem_ptr(recast<NewT>(ptr.ptr_));
}

template <class NewT, class T>
CUTE_HOST_DEVICE constexpr auto recast(rmem_ptr<T const> const& ptr) {
  return make_rmem_ptr(recast<NewT const>(ptr.ptr_));
}

//
// Display utilities
//

template <class T>
CUTE_HOST_DEVICE void print(T const* const ptr) {
  printf("raw_ptr_%db(%p)", int(8 * sizeof(T)), ptr);
}

template <class T>
CUTE_HOST_DEVICE void print(gmem_ptr<T> const& ptr) {
  printf("gmem_ptr_%db(%p)", int(8 * sizeof(T)), ptr.get());
}

template <class T>
CUTE_HOST_DEVICE void print(smem_ptr<T> const& ptr) {
  printf("smem_ptr_%db(%p)", int(8 * sizeof(T)), ptr.get());
}

template <class T>
CUTE_HOST_DEVICE void print(rmem_ptr<T> const& ptr) {
  printf("rmem_ptr_%db(%p)", int(8 * sizeof(T)), ptr.get());
}

template <class T>
CUTE_HOST std::ostream& operator<<(std::ostream& os, gmem_ptr<T> const& ptr) {
  return os << "gmem_ptr_" << int(8 * sizeof(T)) << "b";
}

template <class T>
CUTE_HOST std::ostream& operator<<(std::ostream& os, smem_ptr<T> const& ptr) {
  return os << "smem_ptr_" << int(8 * sizeof(T)) << "b";
}

template <class T>
CUTE_HOST std::ostream& operator<<(std::ostream& os, rmem_ptr<T> const& ptr) {
  return os << "rmem_ptr_" << int(8 * sizeof(T)) << "b";
}

}  // end namespace cute
