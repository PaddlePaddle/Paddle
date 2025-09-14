/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.1 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.1

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include <cstdint>

#include "paddle/phi/kernels/funcs/aligned_vector.h"

#define INT_BITS 32
#if defined(__xpu__)
#define __forceinline__ __inline__
#endif

namespace phi {
namespace funcs {

/**
 * Fast division : Replace division in CUDA with multiplication to improve
 * kernel performance.
 * 1. Complete the division calculation on the CPU, and record the calculation
 *    results by using the divider and shift_val.
 * 2. Set the divisor on the GPU through Div() to complete the calculation.
 */
template <typename IndexT>
struct FastDivMod {
  // 1st value represents the result of input number divides by recorded divisor
  // 2nd value represents the result of input number modulo by recorded divisor
  using DivModT = phi::AlignedVector<uint32_t, 2>;

  FastDivMod() {}
  HOSTDEVICE FastDivMod(uint32_t d) : divisor(d) {
    static_assert(sizeof(unsigned int) == 4,
                  "Only Support 32-bit unsigned int.");

    for (shift_val = 0; shift_val < INT_BITS; ++shift_val) {
      auto shift_limit = 1 << shift_val;
      if (shift_limit >= divisor) break;
    }
    uint64_t long_one = 1;
    uint64_t temp_div =
        ((long_one << INT_BITS) * ((long_one << shift_val) - divisor)) /
            divisor +
        1;
    multiplier = temp_div;
  }

  __device__ __forceinline__ uint32_t Div(uint32_t n) const {
    uint32_t t = __umulhi(n, multiplier);
    return (t + n) >> shift_val;
  }

  __device__ __forceinline__ DivModT Divmod(uint32_t n) const {
    uint32_t q = Div(n);
    DivModT result = {q, n - q * divisor};
    return result;
  }

  __device__ __forceinline__ uint32_t DivCeil(uint32_t n) const {
    DivModT res = Divmod(n);
    return res.val[1] > 0 ? res.val[0] + 1 : res.val[0];
  }

  int32_t shift_val;
  uint32_t divisor;
  uint32_t multiplier;
};

template <>
struct FastDivMod<int64_t> {
  using DivModT = phi::AlignedVector<uint64_t, 2>;

  FastDivMod() {}
  HOSTDEVICE FastDivMod(uint64_t d) : divisor(d) {
    for (shift_val = 0; shift_val < 64; ++shift_val) {
      uint64_t shift_limit = uint64_t(1) << shift_val;
      if (shift_limit >= divisor) break;
    }

    // quotient = ((uint128_t)n_hi << 64) / d
    uint64_t quotient = 0;
    uint64_t n_hi = (uint64_t(1) << shift_val) - d, n_lo = 0;
    for (int i = 63; i >= 0; --i) {
      uint64_t d_hi = i == 0 ? 0 : d >> (64 - i);
      uint64_t d_lo = d << i;
      if (n_hi == 0 && n_lo == 0) break;
      if ((d_hi < n_hi) || (d_hi <= n_hi && d_lo <= n_lo)) {
        quotient |= uint64_t(1) << i;
        n_hi -= d_hi + (d_lo > n_lo);
        n_lo -= d_lo;
      }
    }
    multiplier = quotient + 1;
  }

  __device__ __forceinline__ uint64_t Div(uint64_t n) const {
    uint64_t t = __umul64hi(n, multiplier);
    return (t + n) >> shift_val;
  }

  __device__ __forceinline__ DivModT Divmod(uint64_t n) const {
    uint64_t q = Div(n);
    return {q, n - q * divisor};
  }
  __device__ __forceinline__ uint64_t DivCeil(uint64_t n) const {
    DivModT res = Divmod(n);
    return res.val[1] > 0 ? res.val[0] + 1 : res.val[0];
  }

  int shift_val;
  uint64_t divisor;
  uint64_t multiplier;
};

}  // namespace funcs
}  // namespace phi
