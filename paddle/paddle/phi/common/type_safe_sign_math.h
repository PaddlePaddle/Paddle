// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include <cmath>
#include <type_traits>

#include "paddle/common/hostdevice.h"
#include "paddle/phi/common/bfloat16.h"
#include "paddle/phi/common/float16.h"

namespace phi {

// Returns false since we cannot have x < 0 if x is unsigned.
template <typename T>
static inline constexpr bool is_negative(const T& x, std::true_type) {
  return false;
}

// Returns true if a signed variable x < 0
template <typename T>
static inline constexpr bool is_negative(const T& x, std::false_type) {
  return x < T(0);
}

// Returns true if x < 0
template <typename T>
inline constexpr bool is_negative(const T& x) {
  return is_negative(x, std::is_unsigned<T>());
}

// Note: Explicit implementation of copysign for float16 and bfloat16
// is needed to workaround g++-7/8 crash on aarch64, but also makes
// copysign faster for the half-precision types
template <typename T, typename U>
inline HOSTDEVICE auto copysign(const T& a, const U& b) {
  return std::copysign(a, b);
}

// Implement copysign for half precision floats using bit ops
// Sign is the most significant bit for both float16 and bfloat16 types
inline HOSTDEVICE phi::dtype::float16 copysign(phi::dtype::float16 a,
                                               phi::dtype::float16 b) {
  return phi::dtype::raw_uint16_to_float16((a.x & 0x7fff) | (b.x & 0x8000));
}

inline HOSTDEVICE phi::dtype::bfloat16 copysign(phi::dtype::bfloat16 a,
                                                phi::dtype::bfloat16 b) {
  return phi::dtype::raw_uint16_to_bfloat16((a.x & 0x7fff) | (b.x & 0x8000));
}

}  // namespace phi
