// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
#include <cstddef>
#include <type_traits>

#include "paddle/phi/core/hostdevice.h"

namespace phi {
namespace detail {
template <size_t kStart, size_t kEnd, bool kStop>
struct UnrollFillConstant {
  template <typename T>
  HOSTDEVICE inline static void Run(T *data, T val) {
    data[kStart] = val;
    UnrollFillConstant<kStart + 1, kEnd, kStart + 1 == kEnd>::Run(data, val);
  }
};

template <size_t kStart, size_t kEnd>
struct UnrollFillConstant<kStart, kEnd, true> {
  template <typename T>
  HOSTDEVICE inline static void Run(T *data, T val) {}
};

template <size_t kStart, size_t kEnd, bool kStop>
struct UnrollAssign {
  template <typename Tin, typename Tout>
  HOSTDEVICE inline static void Run(const Tin *d1, Tout *d2) {
    d2[kStart] = static_cast<Tout>(d1[kStart]);
    UnrollAssign<kStart + 1, kEnd, kStart + 1 == kEnd>::Run(d1, d2);
  }
};

template <size_t kStart, size_t kEnd>
struct UnrollAssign<kStart, kEnd, true> {
  template <typename Tin, typename Tout>
  HOSTDEVICE inline static void Run(const Tin *d1, Tout *d2) {}
};

template <typename T, size_t kStart, size_t kEnd, bool kStop>
struct UnrollVarArgsAssignImpl {
  template <typename... Args>
  HOSTDEVICE inline static void Run(T *d, T val, Args... args) {
    static_assert(sizeof...(args) + 1 == kEnd - kStart, "Wrong argument");
    d[kStart] = val;
    UnrollVarArgsAssignImpl<T, kStart + 1, kEnd, kStart + 1 == kEnd>::Run(
        d, args...);
  }
};

template <typename T, size_t kStart, size_t kEnd>
struct UnrollVarArgsAssignImpl<T, kStart, kEnd, true> {
  HOSTDEVICE inline static void Run(T *d) {}
};

template <typename T>
struct UnrollVarArgsAssign {
  template <typename... Args>
  HOSTDEVICE inline static void Run(T *d, Args... args) {
    UnrollVarArgsAssignImpl<T, 0, sizeof...(Args), sizeof...(Args) == 0>::Run(
        d, args...);
  }
};

template <size_t kStart, size_t kEnd, bool kStop>
struct UnrollCompare {
  template <typename T>
  HOSTDEVICE inline static bool Run(const T *d1, const T *d2) {
    return d1[kStart] == d2[kStart] &&
           UnrollCompare<kStart + 1, kEnd, kStart + 1 == kEnd>::Run(d1, d2);
  }
};

template <size_t kStart, size_t kEnd>
struct UnrollCompare<kStart, kEnd, true> {
  template <typename T>
  HOSTDEVICE inline constexpr static bool Run(const T *d1, const T *d2) {
    return true;
  }
};

template <size_t kStart, size_t kEnd, bool kStop>
struct UnrollProduct {
  template <typename T>
  HOSTDEVICE inline static T Run(const T *d) {
    return d[kStart] *
           UnrollProduct<kStart + 1, kEnd, kStart + 1 == kEnd>::Run(d);
  }
};

template <size_t kStart, size_t kEnd>
struct UnrollProduct<kStart, kEnd, true> {
  template <typename T>
  HOSTDEVICE inline constexpr static T Run(const T *d) {
    return 1;
  }
};
}  // namespace detail

template <size_t N>
using UnrollFillConstant = detail::UnrollFillConstant<0, N, N == 0>;

template <size_t N>
using UnrollAssign = detail::UnrollAssign<0, N, N == 0>;

template <typename T>
using UnrollVarArgsAssign = detail::UnrollVarArgsAssign<T>;

template <size_t N>
using UnrollCompare = detail::UnrollCompare<0, N, N == 0>;

template <size_t N>
using UnrollProduct = detail::UnrollProduct<0, N, N == 0>;

}  // namespace phi
