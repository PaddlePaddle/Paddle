// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include <stdint.h>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include "paddle/common/hostdevice.h"

namespace phi {
namespace dtype {

inline std::ostream& operator<<(std::ostream& os, const float8_e4m3fn& a) {
  os << static_cast<float>(a);
  return os;
}

}  // namespace dtype
}  // namespace phi

namespace cinn {
namespace common {
using float8_e4m3fn = ::phi::dtype::float8_e4m3fn;
}  // namespace common
}  // namespace cinn

namespace std {

template <>
struct is_pod<phi::dtype::float8_e4m3fn> {
  static const bool value =
      is_trivial<phi::dtype::float8_e4m3fn>::value &&
      is_standard_layout<phi::dtype::float8_e4m3fn>::value;
};

template <>
struct is_floating_point<phi::dtype::float8_e4m3fn>
    : std::integral_constant<
          bool,
          std::is_same<phi::dtype::float8_e4m3fn,
                       typename std::remove_cv<
                           phi::dtype::float8_e4m3fn>::type>::value> {};
template <>
struct is_signed<phi::dtype::float8_e4m3fn> {
  static const bool value = true;
};

template <>
struct is_unsigned<phi::dtype::float8_e4m3fn> {
  static const bool value = false;
};

inline bool isnan(const phi::dtype::float8_e4m3fn& a) {
  return phi::dtype::isnan(a);
}

inline bool isinf(const phi::dtype::float8_e4m3fn& a) {
  return phi::dtype::isinf(a);
}

template <>
struct numeric_limits<phi::dtype::float8_e4m3fn> {
  static constexpr bool is_specialized = true;
  static constexpr bool is_signed = true;
  static constexpr bool is_integer = false;
  static constexpr bool is_exact = false;
  static constexpr bool has_infinity = false;
  static constexpr bool has_quiet_NaN = true;
  static constexpr bool has_signaling_NaN = false;
  static constexpr auto has_denorm = true;
  static constexpr auto has_denorm_loss = true;
  static constexpr auto round_style = numeric_limits<float>::round_style;
  static constexpr bool is_iec559 = false;
  static constexpr bool is_bounded = true;
  static constexpr bool is_modulo = false;
  static constexpr int digits = 4;
  static constexpr int digits10 = 0;
  static constexpr int max_digits10 = 3;
  static constexpr int radix = 2;
  static constexpr int min_exponent = -5;
  static constexpr int min_exponent10 = -1;
  static constexpr int max_exponent = 8;
  static constexpr int max_exponent10 = 2;
  static constexpr auto traps = numeric_limits<float>::traps;
  static constexpr auto tinyness_before = false;

  HOSTDEVICE static phi::dtype::float8_e4m3fn(min)() {
    return phi::dtype::raw_uint8_to_float8_e4m3fn(0x08);
  }
  HOSTDEVICE static phi::dtype::float8_e4m3fn lowest() {
    return phi::dtype::raw_uint8_to_float8_e4m3fn(0xFE);
  }
  HOSTDEVICE static phi::dtype::float8_e4m3fn(max)() {
    return phi::dtype::raw_uint8_to_float8_e4m3fn(0x7E);
  }
  HOSTDEVICE static phi::dtype::float8_e4m3fn epsilon() {
    return phi::dtype::raw_uint8_to_float8_e4m3fn(0x20);
  }
  HOSTDEVICE static phi::dtype::float8_e4m3fn round_error() {
    return phi::dtype::raw_uint8_to_float8_e4m3fn(0x30);
  }

  HOSTDEVICE static phi::dtype::float8_e4m3fn infinity() {
    return phi::dtype::raw_uint8_to_float8_e4m3fn(0x7F);
  }
  HOSTDEVICE static phi::dtype::float8_e4m3fn quiet_NaN() {
    return phi::dtype::raw_uint8_to_float8_e4m3fn(0x7F);
  }
  HOSTDEVICE static phi::dtype::float8_e4m3fn denorm_min() {
    return phi::dtype::raw_uint8_to_float8_e4m3fn(0x01);
  }
};
template <>
struct common_type<float, phi::dtype::float8_e4m3fn> {
  using type = float;
};
}  // namespace std
