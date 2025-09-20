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
#include "paddle/common/backend_header.h"
#include "paddle/common/hostdevice.h"

namespace phi {
namespace dtype {

inline uint16_t cpu_float_to_bfloat16(float f) {
  uint32_t int_raw;
  memcpy(&int_raw, &f, sizeof(float));

  uint16_t high_part = (int_raw >> 16);
  uint16_t x;
  uint32_t abs_int = int_raw & 0x7FFFFFFF;

  if (abs_int == 0) {  // Zero
    x = high_part & 0x8000;
  } else if (abs_int >= 0x7F800000) {  // Inf or NaN
    if (abs_int == 0x7F800000) {       // Infinity
      x = high_part;
    } else {  // NaN
      x = 0x7FFF;
    }
  } else {  // Normal or subnormal
    // round to nearest even and truncate
    const uint32_t rounding_bias = 0x00007FFF + (high_part & 0x1);
    int_raw = int_raw + rounding_bias;
    x = (int_raw >> 16);
  }
  return x;
}

struct PADDLE_ALIGN(2) bfloat16 {
 public:
  uint16_t x;

  // Constructors
  bfloat16() = default;
  bfloat16(const bfloat16& o) = default;
  bfloat16& operator=(const bfloat16& o) = default;
  bfloat16(bfloat16&& o) = default;
  bfloat16& operator=(bfloat16&& o) = default;
  ~bfloat16() = default;

  HOSTDEVICE inline explicit bfloat16(float val) {
#ifdef PADDLE_WITH_HIP
    uint32_t res = 0;
    uint32_t* tempRes;
    // We should be using memcpy in order to respect the strict aliasing rule
    // but it fails in the HIP environment.
    tempRes = reinterpret_cast<uint32_t*>(&val);
    res = *tempRes;
    x = res >> 16;
#else
#if defined(PADDLE_CUDA_BF16)
    __nv_bfloat16 tmp = __float2bfloat16(val);
    x = *reinterpret_cast<uint16_t*>(&tmp);
#else
    x = cpu_float_to_bfloat16(val);
#endif
#endif
  }

#if defined(PADDLE_CUDA_BF16)
  HOSTDEVICE inline explicit bfloat16(const __nv_bfloat16& val) {
    x = *reinterpret_cast<const unsigned short*>(&val);  // NOLINT
  }
#endif

  template <class T>
  HOSTDEVICE inline explicit bfloat16(const T& val)
      : x(bfloat16(static_cast<float>(val)).x) {}

// Assignment operators
#if defined(PADDLE_CUDA_BF16)
  HOSTDEVICE inline bfloat16& operator=(const __nv_bfloat16& val) {
    x = *reinterpret_cast<const unsigned short*>(&val);  // NOLINT
    return *this;
  }
#endif

  HOSTDEVICE inline bfloat16& operator=(bool b) {
    x = b ? 0x3f80 : 0;
    return *this;
  }

  HOSTDEVICE inline bfloat16& operator=(int8_t val) {
    x = bfloat16(val).x;
    return *this;
  }

  HOSTDEVICE inline bfloat16& operator=(uint8_t val) {
    x = bfloat16(val).x;
    return *this;
  }

  HOSTDEVICE inline bfloat16& operator=(int16_t val) {
    x = bfloat16(val).x;
    return *this;
  }

  HOSTDEVICE inline bfloat16& operator=(uint16_t val) {
    x = bfloat16(val).x;
    return *this;
  }

  HOSTDEVICE inline bfloat16& operator=(int32_t val) {
    x = bfloat16(val).x;
    return *this;
  }

  HOSTDEVICE inline bfloat16& operator=(uint32_t val) {
    x = bfloat16(val).x;
    return *this;
  }

  HOSTDEVICE inline bfloat16& operator=(int64_t val) {
    x = bfloat16(val).x;
    return *this;
  }

  HOSTDEVICE inline bfloat16& operator=(uint64_t val) {
    x = bfloat16(val).x;
    return *this;
  }

  HOSTDEVICE inline bfloat16& operator=(float val) {
    x = bfloat16(val).x;
    return *this;
  }

  HOSTDEVICE inline bfloat16& operator=(double val) {
    x = bfloat16(val).x;
    return *this;
  }

  // Conversion operators
  HOSTDEVICE inline operator float() const {
#ifdef PADDLE_WITH_HIP
    uint32_t res = 0;
    // We should be using memcpy in order to respect the strict aliasing rule
    // but it fails in the HIP environment.
    uint16_t temp = x;
    uint16_t* temp_ptr = reinterpret_cast<uint16_t*>(&temp);
    res = *temp_ptr;
    // return res;
    res = res << 16;
    return *reinterpret_cast<float*>(&res);
#else
#ifdef PADDLE_CUDA_BF16
    return __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&x));
#else
    float val = 0.f;
    uint16_t temp = x;
    std::memcpy(
        reinterpret_cast<char*>(&val) + 2, reinterpret_cast<char*>(&temp), 2);
    return val;
#endif
#endif
  }

#ifdef PADDLE_CUDA_BF16
  HOSTDEVICE inline __nv_bfloat16 to_nv_bfloat16() const {
    return *reinterpret_cast<const __nv_bfloat16*>(&x);
  }
#endif

  HOSTDEVICE inline explicit operator bool() const { return (x & 0x7fff) != 0; }

  HOSTDEVICE inline explicit operator int8_t() const {
    return static_cast<int8_t>(static_cast<float>(*this));
  }

  HOSTDEVICE inline explicit operator uint8_t() const {
    return static_cast<uint8_t>(static_cast<float>(*this));
  }

  HOSTDEVICE inline explicit operator int16_t() const {
    return static_cast<int16_t>(static_cast<float>(*this));
  }

  HOSTDEVICE inline explicit operator uint16_t() const {
    return static_cast<uint16_t>(static_cast<float>(*this));
  }

  HOSTDEVICE inline explicit operator int32_t() const {
    return static_cast<int32_t>(static_cast<float>(*this));
  }

  HOSTDEVICE inline explicit operator uint32_t() const {
    return static_cast<uint32_t>(static_cast<float>(*this));
  }

  HOSTDEVICE inline explicit operator int64_t() const {
    return static_cast<int64_t>(static_cast<float>(*this));
  }

  HOSTDEVICE inline explicit operator uint64_t() const {
    return static_cast<uint64_t>(static_cast<float>(*this));
  }

  HOSTDEVICE inline operator double() const {
    return static_cast<double>(static_cast<float>(*this));
  }
};

struct PADDLE_ALIGN(16) bfloat168 {
  bfloat16 x, y, z, w, v, u, t, s;
};

struct PADDLE_ALIGN(8) bfloat164 {
  bfloat16 x, y, z, w;
};

struct PADDLE_ALIGN(4) bfloat162 {
  bfloat16 x, y;
};

HOSTDEVICE inline bfloat16 operator+(const bfloat16& a, const bfloat16& b) {
#if defined(PADDLE_CUDA_BF16) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  return bfloat16(__hadd(a.to_nv_bfloat16(), b.to_nv_bfloat16()));
#else
  return bfloat16(static_cast<float>(a) + static_cast<float>(b));
#endif
}

HOSTDEVICE inline bfloat16 operator-(const bfloat16& a, const bfloat16& b) {
#if defined(PADDLE_CUDA_BF16) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  return bfloat16(__hsub(a.to_nv_bfloat16(), b.to_nv_bfloat16()));
#else
  return bfloat16(static_cast<float>(a) - static_cast<float>(b));
#endif
}

HOSTDEVICE inline bfloat16 operator*(const bfloat16& a, const bfloat16& b) {
#if defined(PADDLE_CUDA_BF16) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  return bfloat16(__hmul(a.to_nv_bfloat16(), b.to_nv_bfloat16()));
#else
  return bfloat16(static_cast<float>(a) * static_cast<float>(b));
#endif
}

HOSTDEVICE inline bfloat16 operator/(const bfloat16& a, const bfloat16& b) {
#if defined(PADDLE_CUDA_BF16) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  return bfloat16(__hdiv(a.to_nv_bfloat16(), b.to_nv_bfloat16()));
#else
  return bfloat16(static_cast<float>(a) / static_cast<float>(b));
#endif
}

HOSTDEVICE inline bfloat16 operator-(const bfloat16& a) {
  bfloat16 res;
  res.x = a.x ^ 0x8000;
  return res;
}

HOSTDEVICE inline bfloat16& operator+=(bfloat16& a,  // NOLINT
                                       const bfloat16& b) {
  a = a + b;
  return a;
}

HOSTDEVICE inline bfloat16& operator-=(bfloat16& a,  // NOLINT
                                       const bfloat16& b) {
  a = a - b;
  return a;
}

HOSTDEVICE inline bfloat16& operator*=(bfloat16& a,  // NOLINT
                                       const bfloat16& b) {
  a = a * b;
  return a;
}

HOSTDEVICE inline bfloat16& operator/=(bfloat16& a,  // NOLINT
                                       const bfloat16& b) {
  a = a / b;
  return a;
}

HOSTDEVICE inline bfloat16 raw_uint16_to_bfloat16(uint16_t a) {
  bfloat16 res;
  res.x = a;
  return res;
}

// Comparison operators
HOSTDEVICE inline bool operator==(const bfloat16& a, const bfloat16& b) {
#if defined(PADDLE_CUDA_BF16) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  return __heq(a.to_nv_bfloat16(), b.to_nv_bfloat16());
#else
  return static_cast<float>(a) == static_cast<float>(b);
#endif
}

HOSTDEVICE inline bool operator!=(const bfloat16& a, const bfloat16& b) {
#if defined(PADDLE_CUDA_BF16) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  return __hne(a.to_nv_bfloat16(), b.to_nv_bfloat16());
#else
  return static_cast<float>(a) != static_cast<float>(b);
#endif
}

HOSTDEVICE inline bool operator<(const bfloat16& a, const bfloat16& b) {
#if defined(PADDLE_CUDA_BF16) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  return __hlt(a.to_nv_bfloat16(), b.to_nv_bfloat16());
#else
  return static_cast<float>(a) < static_cast<float>(b);
#endif
}

HOSTDEVICE inline bool operator<=(const bfloat16& a, const bfloat16& b) {
#if defined(PADDLE_CUDA_BF16) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  return __hle(a.to_nv_bfloat16(), b.to_nv_bfloat16());
#else
  return static_cast<float>(a) <= static_cast<float>(b);
#endif
}

HOSTDEVICE inline bool operator>(const bfloat16& a, const bfloat16& b) {
#if defined(PADDLE_CUDA_BF16) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  return __hgt(a.to_nv_bfloat16(), b.to_nv_bfloat16());
#else
  return static_cast<float>(a) > static_cast<float>(b);
#endif
}

HOSTDEVICE inline bool operator>=(const bfloat16& a, const bfloat16& b) {
#if defined(PADDLE_CUDA_BF16) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  return __hge(a.to_nv_bfloat16(), b.to_nv_bfloat16());
#else
  return static_cast<float>(a) >= static_cast<float>(b);
#endif
}

HOSTDEVICE inline bool(isnan)(const bfloat16& a) {
#if defined(PADDLE_CUDA_BF16) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  return __hisnan(a.to_nv_bfloat16());
#else
  return (a.x & 0x7FFF) > 0x7F80;
#endif
}

HOSTDEVICE inline bool(isinf)(const bfloat16& a) {
#if defined(PADDLE_CUDA_BF16) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  return __hisinf(a.to_nv_bfloat16());
#else
  return (a.x & 0x7F80) == 0x7F80;
#endif
}

HOSTDEVICE inline bool(isfinite)(const bfloat16& a) {
  return !((isnan)(a)) && !((isinf)(a));
}
HOSTDEVICE inline bfloat16(abs)(const bfloat16& a) {
#if defined(PADDLE_CUDA_BF16) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  return bfloat16(__habs(a.to_nv_bfloat16()));
#else
  return bfloat16(std::abs(static_cast<float>(a)));
#endif
}

}  // namespace dtype
}  // namespace phi

// for runtime calls
#if defined(PADDLE_CUDA_BF16)
__device__ inline phi::dtype::bfloat16 __shfl_sync(unsigned mask,
                                                   phi::dtype::bfloat16 var,
                                                   int srcLane,
                                                   int width = warpSize) {
  return phi::dtype::bfloat16(
      __shfl_sync(mask, var.to_nv_bfloat16(), srcLane, width));
}

__device__ inline phi::dtype::bfloat16 __shfl_up_sync(unsigned mask,
                                                      phi::dtype::bfloat16 var,
                                                      unsigned int delta,
                                                      int width = warpSize) {
  return phi::dtype::bfloat16(
      __shfl_up_sync(mask, var.to_nv_bfloat16(), delta, width));
}

__device__ inline phi::dtype::bfloat16 __shfl_down_sync(
    unsigned mask,
    phi::dtype::bfloat16 var,
    unsigned int delta,
    int width = warpSize) {
  return phi::dtype::bfloat16(
      __shfl_down_sync(mask, var.to_nv_bfloat16(), delta, width));
}

__device__ inline phi::dtype::bfloat16 __shfl_xor_sync(unsigned mask,
                                                       phi::dtype::bfloat16 var,
                                                       int laneMask,
                                                       int width = warpSize) {
  return phi::dtype::bfloat16(
      __shfl_xor_sync(mask, var.to_nv_bfloat16(), laneMask, width));
}

__host__ __device__ inline phi::dtype::bfloat16 max(
    const phi::dtype::bfloat16& a, const phi::dtype::bfloat16& b) {
  return a > b ? a : b;
}
__host__ __device__ inline phi::dtype::bfloat16 min(
    const phi::dtype::bfloat16& a, const phi::dtype::bfloat16& b) {
  return a < b ? a : b;
}
#endif  // PADDLE_CUDA_BF16

namespace cinn {
namespace common {
using namespace phi::dtype;  // NOLINT
}  // namespace common
}  // namespace cinn
