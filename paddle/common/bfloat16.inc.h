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

#include <stdint.h>

#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include "paddle/common/backend_header.h"
#include "paddle/common/hostdevice.h"
namespace phi {
namespace dtype {
inline std::ostream& operator<<(std::ostream& os, const bfloat16& a) {
  os << static_cast<float>(a);
  return os;
}

}  // namespace dtype
}  // namespace phi

namespace std {

template <>
struct is_pod<phi::dtype::bfloat16> {
  static const bool value = is_trivial<phi::dtype::bfloat16>::value &&
                            is_standard_layout<phi::dtype::bfloat16>::value;
};

template <>
struct is_floating_point<phi::dtype::bfloat16>
    : std::integral_constant<
          bool,
          std::is_same<
              phi::dtype::bfloat16,
              typename std::remove_cv<phi::dtype::bfloat16>::type>::value> {};
template <>
struct is_signed<phi::dtype::bfloat16> {
  static const bool value = true;
};

template <>
struct is_unsigned<phi::dtype::bfloat16> {
  static const bool value = false;
};

inline bool isnan(const phi::dtype::bfloat16& a) {
  return phi::dtype::isnan(a);
}

inline bool isinf(const phi::dtype::bfloat16& a) {
  return phi::dtype::isinf(a);
}

template <>
struct numeric_limits<phi::dtype::bfloat16> {
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

  HOSTDEVICE static phi::dtype::bfloat16(min)() {
    return phi::dtype::raw_uint16_to_bfloat16(0x0080);
  }
  HOSTDEVICE static phi::dtype::bfloat16 lowest() {
    return phi::dtype::raw_uint16_to_bfloat16(0xff7f);
  }
  HOSTDEVICE static phi::dtype::bfloat16(max)() {
    return phi::dtype::raw_uint16_to_bfloat16(0x7f7f);
  }
  HOSTDEVICE static phi::dtype::bfloat16 epsilon() {
    return phi::dtype::raw_uint16_to_bfloat16(0x3C00);
  }
  HOSTDEVICE static phi::dtype::bfloat16 round_error() {
    return phi::dtype::bfloat16(0.5);
  }
  HOSTDEVICE static phi::dtype::bfloat16 infinity() {
    return phi::dtype::raw_uint16_to_bfloat16(0x7f80);
  }
  HOSTDEVICE static phi::dtype::bfloat16 quiet_NaN() {
    return phi::dtype::raw_uint16_to_bfloat16(0xffc1);
  }
  HOSTDEVICE static phi::dtype::bfloat16 signaling_NaN() {
    return phi::dtype::raw_uint16_to_bfloat16(0xff81);
  }
  HOSTDEVICE static phi::dtype::bfloat16 denorm_min() {
    return phi::dtype::raw_uint16_to_bfloat16(0x0001);
  }
};

}  // namespace std
