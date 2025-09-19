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

#if defined(_M_X64) || defined(__x86_64__) || defined(_M_IX86) || \
    defined(__i386__)
#define __PADDLE_x86__
// Note(risemeup1):undef __SSE2__ to avoid fp16 conflict between cuda and gcc12
#ifdef __SSE2__
#undef __SSE2__
#include <immintrin.h>
#define __SSE2__
#else
#include <immintrin.h>
#endif
#endif
#include <stdint.h>

#include <cmath>
#include <iostream>
#include <limits>
#include "paddle/common/hostdevice.h"
namespace phi {
namespace dtype {
inline std::ostream& operator<<(std::ostream& os, const float16& a) {
  os << static_cast<float>(a);
  return os;
}
}  // namespace dtype
}  // namespace phi
namespace std {

// Override the std::is_pod::value for float16
// The reason is that different compilers implemented std::is_pod based on
// different C++ standards. float16 class is a plain old data in C++11 given
// that it is both trivial and standard_layout.
// However, std::is_pod in nvcc 8.0 host c++ compiler follows C++0x and is
// more restricted in that you cannot provide any customized
// constructor in float16. Hence, we override is_pod here following C++11
// so that .cu files can be successfully compiled by nvcc.
template <>
struct is_pod<phi::dtype::float16> {
  static const bool value = is_trivial<phi::dtype::float16>::value &&
                            is_standard_layout<phi::dtype::float16>::value;
};

#if !(defined(PADDLE_WITH_CUSTOM_KERNEL) && defined(PADDLE_WITH_HIP))
template <>
struct is_floating_point<phi::dtype::float16>
    : std::integral_constant<
          bool,
          std::is_same<
              phi::dtype::float16,
              typename std::remove_cv<phi::dtype::float16>::type>::value> {};
#endif

template <>
struct is_signed<phi::dtype::float16> {
  static const bool value = true;
};

template <>
struct is_unsigned<phi::dtype::float16> {
  static const bool value = false;
};

inline bool isnan(const phi::dtype::float16& a) { return phi::dtype::isnan(a); }

inline bool isinf(const phi::dtype::float16& a) { return phi::dtype::isinf(a); }

inline bool isfinite(const phi::dtype::float16& a) {
  return phi::dtype::isfinite(a);
}

template <>
struct numeric_limits<phi::dtype::float16> {
  static const bool is_specialized = true;
  static const bool is_signed = true;
  static const bool is_integer = false;
  static const bool is_exact = false;
  static const bool has_infinity = true;
  static const bool has_quiet_NaN = true;
  static const bool has_signaling_NaN = true;
  static const float_denorm_style has_denorm = denorm_present;
  static const bool has_denorm_loss = false;
  static const std::float_round_style round_style = std::round_to_nearest;
  static const bool is_iec559 = false;
  static const bool is_bounded = false;
  static const bool is_modulo = false;
  static const int digits = 11;
  static const int digits10 = 3;
  static const int max_digits10 = 5;
  static const int radix = 2;
  static const int min_exponent = -13;
  static const int min_exponent10 = -4;
  static const int max_exponent = 16;
  static const int max_exponent10 = 4;
  static const bool traps = true;
  static const bool tinyness_before = false;

  HOSTDEVICE static phi::dtype::float16(min)() {
    return phi::dtype::raw_uint16_to_float16(0x400);
  }
  HOSTDEVICE static phi::dtype::float16 lowest() {
    return phi::dtype::raw_uint16_to_float16(0xfbff);
  }
  HOSTDEVICE static phi::dtype::float16(max)() {
    return phi::dtype::raw_uint16_to_float16(0x7bff);
  }
  HOSTDEVICE static phi::dtype::float16 epsilon() {
    return phi::dtype::raw_uint16_to_float16(0x1400);
  }
  HOSTDEVICE static phi::dtype::float16 round_error() {
    return phi::dtype::float16(0.5);
  }
  HOSTDEVICE static phi::dtype::float16 infinity() {
    return phi::dtype::raw_uint16_to_float16(0x7c00);
  }
  HOSTDEVICE static phi::dtype::float16 quiet_NaN() {
    return phi::dtype::raw_uint16_to_float16(0x7e00);
  }
  HOSTDEVICE static phi::dtype::float16 signaling_NaN() {
    return phi::dtype::raw_uint16_to_float16(0x7e00);
  }
  HOSTDEVICE static phi::dtype::float16 denorm_min() {
    return phi::dtype::raw_uint16_to_float16(0x1);
  }
};

HOSTDEVICE inline phi::dtype::float16 abs(const phi::dtype::float16& a) {
  return phi::dtype::abs(a);
}

}  // namespace std
