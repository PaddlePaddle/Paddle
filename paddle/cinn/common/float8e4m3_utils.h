// Copyright (c) 2025 CINN Authors. All Rights Reserved.
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

#include "paddle/cinn/common/float8e4m3.h"

namespace std {
// Override the std::is_pod::value for float8e4m3
template <>
struct is_pod<cinn::common::float8e4m3> {
  static const bool value = is_trivial<cinn::common::float8e4m3>::value &&
                            is_standard_layout<cinn::common::float8e4m3>::value;
};

template <>
struct is_floating_point<cinn::common::float8e4m3>
    : std::integral_constant<
          bool,
          std::is_same<
              cinn::common::float8e4m3,
              typename std::remove_cv<cinn::common::float8e4m3>::type>::value> {
};

template <>
struct is_signed<cinn::common::float8e4m3> {
  static const bool value = true;
};

template <>
struct is_unsigned<cinn::common::float8e4m3> {
  static const bool value = false;
};

__host__ __device__ inline cinn::common::float8e4m3 abs(
    const cinn::common::float8e4m3& a) {
#if defined(CINN_CUDA_FP8)
  __half fp16_val = __nv_cvt_fp8_to_halfraw(a.x, __NV_E4M3);
  __half fp16_abs = __habs(fp16_val);
  return cinn::common::float8e4m3(__nv_cvt_halfraw_to_fp8(fp16_abs, __NV_E4M3));
#else
  return cinn::common::float8e4m3(a.x & 0x7F);
#endif
}

__host__ __device__ inline bool isnan(const cinn::common::float8e4m3& a) {
#if defined(CINN_CUDA_FP8)
  __half fp16_val = __nv_cvt_fp8_to_halfraw(a.x, __NV_E4M3);
  return __hisnan(fp16_val);
#else
  return (a.x & 0x7F) == 0x7F &&
         (a.x & 0x07) != 0;  // E4M3 NaN: exp=15, mantissa≠0
#endif
}

__host__ __device__ inline bool isinf(const cinn::common::float8e4m3& a) {
#if defined(CINN_CUDA_FP8)
  __half fp16_val = __nv_cvt_fp8_to_halfraw(a.x, __NV_E4M3);
  return __hisinf(fp16_val);
#else
  return (a.x & 0x7F) == 0x7F &&
         (a.x & 0x07) == 0;  // E4M3 Inf: exp=15, mantissa=0
#endif
}

__host__ __device__ inline bool isfinite(const cinn::common::float8e4m3& a) {
#if defined(CINN_CUDA_FP8)
  __half fp16_val = __nv_cvt_fp8_to_halfraw(a.x, __NV_E4M3);
  return !__hisnan(fp16_val) && !__hisinf(fp16_val);
#else
  return !isnan(a) && !isinf(a);
#endif
}

template <>
struct numeric_limits<cinn::common::float8e4m3> {
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
  static const int digits = 4;        // 3 mantissa + implicit 1
  static const int digits10 = 0;      // floor(3 * log10(2));
  static const int max_digits10 = 3;  // ceil(4 * log10(2) + 1)
  static const int radix = 2;
  static const int min_exponent = -5;
  static const int min_exponent10 = -1;
  static const int max_exponent = 9;
  static const int max_exponent10 = 2;
  static const bool traps = true;
  static const bool tinyness_before = false;

  __host__ __device__ static cinn::common::float8e4m3(min)() {
    return cinn::common::raw_uint8_to_float8e4m3(0x04);
  }
  __host__ __device__ static cinn::common::float8e4m3 lowest() {
    return cinn::common::raw_uint8_to_float8e4m3(0xFB);
  }
  __host__ __device__ static cinn::common::float8e4m3(max)() {
    return cinn::common::raw_uint8_to_float8e4m3(0x7B);
  }
  __host__ __device__ static cinn::common::float8e4m3 epsilon() {
    return cinn::common::raw_uint8_to_float8e4m3(0x08);
  }
  __host__ __device__ static cinn::common::float8e4m3 round_error() {
    return cinn::common::float8e4m3(0.5);
  }
  __host__ __device__ static cinn::common::float8e4m3 infinity() {
    return cinn::common::raw_uint8_to_float8e4m3(0x7C);
  }
  __host__ __device__ static cinn::common::float8e4m3 quiet_NaN() {
    return cinn::common::raw_uint8_to_float8e4m3(0x7E);
  }
  __host__ __device__ static cinn::common::float8e4m3 signaling_NaN() {
    return cinn::common::raw_uint8_to_float8e4m3(0x7E);
  }
  __host__ __device__ static cinn::common::float8e4m3 denorm_min() {
    return cinn::common::raw_uint8_to_float8e4m3(0x01);
  }
};

}  // namespace std

namespace cinn {
namespace common {
inline std::ostream& operator<<(std::ostream& os, const float8e4m3& a) {
  os << std::showpoint << static_cast<float>(a);
  return os;
}
}  // namespace common
}  // namespace cinn
