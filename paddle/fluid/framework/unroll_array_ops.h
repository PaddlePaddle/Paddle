// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
#include <type_traits>
#include "paddle/fluid/platform/hostdevice.h"

namespace paddle {
namespace framework {

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
struct UnrollVarArgsAssign {
  template <typename... Args>
  HOSTDEVICE inline static void Run(T *d, T val, Args... args) {
    static_assert(sizeof...(args) + 1 == kEnd - kStart, "Wrong argument");
    d[kStart] = val;
    UnrollVarArgsAssign<T, kStart + 1, kEnd, kStart + 1 == kEnd>::Run(d,
                                                                      args...);
  }
};

template <typename T, size_t kStart, size_t kEnd>
struct UnrollVarArgsAssign<T, kStart, kEnd, true> {
  HOSTDEVICE inline static void Run(T *d) {}
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
struct UnrollAdd {
  template <typename T>
  HOSTDEVICE inline static void Run(const T *d1, const T *d2, T *d3) {
    d3[kStart] = d1[kStart] + d2[kStart];
    UnrollAdd<kStart + 1, kEnd, kStart + 1 == kEnd>::Run(d1, d2, d3);
  }
};

template <size_t kStart, size_t kEnd>
struct UnrollAdd<kStart, kEnd, true> {
  template <typename T>
  HOSTDEVICE inline static void Run(const T *d1, const T *d2, T *d3) {}
};

template <size_t kStart, size_t kEnd, bool kStop>
struct UnrollMul {
  template <typename T>
  HOSTDEVICE inline static void Run(const T *d1, const T *d2, T *d3) {
    d3[kStart] = d1[kStart] * d2[kStart];
    UnrollMul<kStart + 1, kEnd, kStart + 1 == kEnd>::Run(d1, d2, d3);
  }
};

template <size_t kStart, size_t kEnd>
struct UnrollMul<kStart, kEnd, true> {
  template <typename T>
  HOSTDEVICE inline static void Run(const T *d1, const T *d2, T *d3) {}
};

template <size_t kStart, size_t kEnd, bool kStop>
struct UnrollProduct {
  template <typename T>
  HOSTDEVICE inline static T Run(const T *d) {
    return d[kStart] *
           UnrollProduct<kStart + 1, kEnd, kStart + 1 == kEnd>::Run(d);
  }

  template <typename T>
  HOSTDEVICE inline static T Run(const T *d1, const T *d2) {
    return d1[kStart] * d2[kStart] +
           UnrollProduct<kStart + 1, kEnd, kStart + 1 == kEnd>::Run(d1, d2);
  }
};

template <size_t kStart, size_t kEnd>
struct UnrollProduct<kStart, kEnd, true> {
  template <typename T>
  HOSTDEVICE inline constexpr static T Run(const T *d) {
    return 1;
  }

  template <typename T>
  HOSTDEVICE inline constexpr static T Run(const T *d1, const T *d2) {
    return 0;
  }
};

}  // namespace detail

template <size_t N>
using UnrollFillConstant = detail::UnrollFillConstant<0, N, N == 0>;

template <size_t N>
using UnrollAssign = detail::UnrollAssign<0, N, N == 0>;

template <typename T, size_t N>
using UnrollVarArgsAssign = detail::UnrollVarArgsAssign<T, 0, N, N == 0>;

template <size_t N>
using UnrollCompare = detail::UnrollCompare<0, N, N == 0>;

template <size_t N>
using UnrollAdd = detail::UnrollAdd<0, N, N == 0>;

template <size_t N>
using UnrollMul = detail::UnrollMul<0, N, N == 0>;

template <size_t N>
using UnrollProduct = detail::UnrollProduct<0, N, N == 0>;

}  // namespace framework
}  // namespace paddle
