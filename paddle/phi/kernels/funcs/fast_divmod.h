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

  int32_t shift_val;
  uint32_t divisor;
  uint32_t multiplier;
};

template <typename IndexT>
struct GeneralDivMod {
 public:
  explicit GeneralDivMod(IndexT d) { divmoder = phi::funcs::FastDivMod(d); }
  __device__ inline phi::funcs::FastDivMod::DivModT div_mod(IndexT val) {
    return divmoder.Divmod(val);
  }

  phi::funcs::FastDivMod divmoder;
};

template <>
struct GeneralDivMod<int64_t> {
 public:
  using DivModT = phi::AlignedVector<int64_t, 2>;

  explicit GeneralDivMod(int64_t d) { divisor = d; }
  __device__ inline DivModT div_mod(int64_t val) {
    DivModT data;
    data[0] = val / divisor;
    data[1] = val - data[0] * divisor;
    return data;
  }

  int64_t divisor;
};

}  // namespace funcs
}  // namespace phi
