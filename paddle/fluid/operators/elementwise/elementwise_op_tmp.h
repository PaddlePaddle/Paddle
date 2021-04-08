// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

template <typename INDEX_T>
struct DivMod {
  INDEX_T div, mod;

  __forceinline__ __device__ DivMod(INDEX_T div, INDEX_T mod)
      : div(div), mod(mod) {}
};

template <typename INDEX_T>
struct FastDivMod {
  FastDivMod() {}

  explicit FastDivMod(INDEX_T d) : divisor(d) {
    // assert(divisor >= 1 && divisor <= std::numeric_limits<INDEX_T>::max);
    for (shift_val = 0; shift_val < INT_BITS; ++shift_val) {
      if ((1 << shift_val) >= divisor) {
        break;
      }
    }
    uint64_t one_uint64 = 1;
    uint64_t temp_div =
        ((one_uint64 << INT_BITS) * ((one_uint64 << shift_val) - divisor)) /
            divisor +
        1;
    multiplier = temp_div;
  }

  __forceinline__ __device__ INDEX_T div(INDEX_T n) const {
    INDEX_T t = __umulhi(n, multiplier);
    return (t + n) >> shift_val;
  }

  __forceinline__ __device__ DivMod<INDEX_T> divmod(INDEX_T n) const {
    INDEX_T q = div(n);
    return DivMod<INDEX_T>(q, n - q * divisor);
  }

  INDEX_T divisor;
  INDEX_T multiplier;
  INDEX_T shift_val;
};
