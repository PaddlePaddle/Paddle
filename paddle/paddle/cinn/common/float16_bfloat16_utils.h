// Copyright (c) 2022 CINN Authors. All Rights Reserved.
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

#include <iostream>
#include <limits>

#include "paddle/cinn/common/bfloat16.h"
#include "paddle/cinn/common/float16.h"

namespace std {
// Override the std::is_pod::value for float16 and bfloat16
// The reason is that different compilers implemented std::is_pod based on
// different C++ standards. float16 class is a plain old data in C++11 given
// that it is both trivial and standard_layout.
// However, std::is_pod in nvcc 8.0 host c++ compiler follows C++0x and is
// more restricted in that you cannot provide any customized
// constructor in float16. Hence, we override is_pod here following C++11
// so that .cu files can be successfully compiled by nvcc.

// for float16
template <>
struct is_pod<cinn::common::float16> {
  static const bool value = is_trivial<cinn::common::float16>::value &&
                            is_standard_layout<cinn::common::float16>::value;
};

template <>
struct is_floating_point<cinn::common::float16>
    : std::integral_constant<
          bool,
          std::is_same<
              cinn::common::float16,
              typename std::remove_cv<cinn::common::float16>::type>::value> {};
template <>
struct is_signed<cinn::common::float16> {
  static const bool value = true;
};

template <>
struct is_unsigned<cinn::common::float16> {
  static const bool value = false;
};

__host__ __device__ inline cinn::common::float16 abs(
    const cinn::common::float16& a) {
  return cinn::common::abs(a);
}

inline bool isnan(const cinn::common::float16& a) {
  return cinn::common::isnan(a);
}

inline bool isinf(const cinn::common::float16& a) {
  return cinn::common::isinf(a);
}

inline bool isfinite(const cinn::common::float16& a) {
  return cinn::common::isfinite(a);
}

template <>
struct numeric_limits<cinn::common::float16> {
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

  __host__ __device__ static cinn::common::float16(min)() {
    return cinn::common::raw_uint16_to_float16(0x400);
  }
  __host__ __device__ static cinn::common::float16 lowest() {
    return cinn::common::raw_uint16_to_float16(0xfbff);
  }
  __host__ __device__ static cinn::common::float16(max)() {
    return cinn::common::raw_uint16_to_float16(0x7bff);
  }
  __host__ __device__ static cinn::common::float16 epsilon() {
    return cinn::common::raw_uint16_to_float16(0x0800);
  }
  __host__ __device__ static cinn::common::float16 round_error() {
    return cinn::common::float16(0.5);
  }
  __host__ __device__ static cinn::common::float16 infinity() {
    return cinn::common::raw_uint16_to_float16(0x7c00);
  }
  __host__ __device__ static cinn::common::float16 quiet_NaN() {
    return cinn::common::raw_uint16_to_float16(0x7e00);
  }
  __host__ __device__ static cinn::common::float16 signaling_NaN() {
    return cinn::common::raw_uint16_to_float16(0x7e00);
  }
  __host__ __device__ static cinn::common::float16 denorm_min() {
    return cinn::common::raw_uint16_to_float16(0x1);
  }
};

// for bfloat16
template <>
struct is_pod<cinn::common::bfloat16> {
  static const bool value = is_trivial<cinn::common::bfloat16>::value &&
                            is_standard_layout<cinn::common::bfloat16>::value;
};

template <>
struct is_floating_point<cinn::common::bfloat16>
    : std::integral_constant<
          bool,
          std::is_same<
              cinn::common::bfloat16,
              typename std::remove_cv<cinn::common::bfloat16>::type>::value> {};
template <>
struct is_signed<cinn::common::bfloat16> {
  static const bool value = true;
};

template <>
struct is_unsigned<cinn::common::bfloat16> {
  static const bool value = false;
};

inline bool isnan(const cinn::common::bfloat16& a) {
  return cinn::common::isnan(a);
}

inline bool isinf(const cinn::common::bfloat16& a) {
  return cinn::common::isinf(a);
}

template <>
struct numeric_limits<cinn::common::bfloat16> {
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
  static const int digits = 8;
  static const int digits10 = 2;
  static const int max_digits10 = 9;
  static const int radix = 2;
  static const int min_exponent = -125;
  static const int min_exponent10 = -37;
  static const int max_exponent = 128;
  static const int max_exponent10 = 38;
  static const bool traps = true;
  static const bool tinyness_before = false;

  __host__ __device__ static cinn::common::bfloat16(min)() {
    return cinn::common::raw_uint16_to_bfloat16(0x007f);
  }
  __host__ __device__ static cinn::common::bfloat16 lowest() {
    return cinn::common::raw_uint16_to_bfloat16(0xff7f);
  }
  __host__ __device__ static cinn::common::bfloat16(max)() {
    return cinn::common::raw_uint16_to_bfloat16(0x7f7f);
  }
  __host__ __device__ static cinn::common::bfloat16 epsilon() {
    return cinn::common::raw_uint16_to_bfloat16(0x3400);
  }
  __host__ __device__ static cinn::common::bfloat16 round_error() {
    return cinn::common::bfloat16(0.5);
  }
  __host__ __device__ static cinn::common::bfloat16 infinity() {
    return cinn::common::raw_uint16_to_bfloat16(0x7f80);
  }
  __host__ __device__ static cinn::common::bfloat16 quiet_NaN() {
    return cinn::common::raw_uint16_to_bfloat16(0xffc1);
  }
  __host__ __device__ static cinn::common::bfloat16 signaling_NaN() {
    return cinn::common::raw_uint16_to_bfloat16(0xff81);
  }
  __host__ __device__ static cinn::common::bfloat16 denorm_min() {
    return cinn::common::raw_uint16_to_bfloat16(0x0001);
  }
};

}  // namespace std

namespace cinn {
namespace common {
inline std::ostream& operator<<(std::ostream& os, const float16& a) {
  os << std::showpoint << static_cast<float>(a);
  return os;
}

inline std::ostream& operator<<(std::ostream& os, const bfloat16& a) {
  os << std::showpoint << static_cast<float>(a);
  return os;
}
}  // namespace common
}  // namespace cinn
