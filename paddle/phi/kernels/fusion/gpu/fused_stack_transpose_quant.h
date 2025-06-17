// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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
#ifndef PADDLE_PHI_KERNELS_FUSION_GPU_FUSED_STACK_TRANSPOSE_QUANT_H_
#define PADDLE_PHI_KERNELS_FUSION_GPU_FUSED_STACK_TRANSPOSE_QUANT_H_

#include "paddle/phi/kernels/fusion/gpu/quant_utils.h"

template <typename T, int VecSize>
struct __align__(sizeof(T) * VecSize) VecType {
  T val[VecSize];
  __host__ __device__ inline T& operator[](size_t i) { return val[i]; }
  __host__ __device__ inline const T& operator[](size_t i) const {
    return val[i];
  }
};

struct FastDiv {
  FastDiv() {}
  explicit FastDiv(
      uint64_t d) {  // Single-parameter constructors should be marked explicit.
    for (shift_val = 0; shift_val < 64; ++shift_val) {
      uint64_t shift_limit = uint64_t(1) << shift_val;
      if (shift_limit >= d) break;
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

  __device__ uint64_t Div(uint64_t n) const {
    uint64_t t = __umul64hi(n, multiplier);
    return (t + n) >> shift_val;
  }

  int shift_val;
  uint64_t multiplier;
};

__device__ __forceinline__ void BlockLoad(const int64_t* __restrict__ X_ptrs,
                                          __nv_bfloat16 input[4][4],
                                          size_t K,
                                          size_t block_y,
                                          size_t block_x) {
  const __nv_bfloat16* X =
      reinterpret_cast<const __nv_bfloat16*>(X_ptrs[blockIdx.z]);

  for (size_t i = 0; i < 4; i++) {
    size_t idx_m = block_y * 128 + threadIdx.y + i * 32;
    size_t idx_k = block_x * 128 + threadIdx.x * 4;
    size_t idx = idx_m * K + idx_k;

    using LoadT = VecType<__nv_bfloat16, 4>;
    LoadT data = *reinterpret_cast<const LoadT*>(X + idx);
    for (int j = 0; j < 4; j++) {
      input[i][j] = data[j];
    }
  }
}

__device__ __forceinline__ __nv_bfloat16 WarpReduceMax(__nv_bfloat16 x) {
  for (int offset = 16; offset > 0; offset /= 2) {
    __nv_bfloat16 t = __shfl_down_sync(0xffffffff, x, offset);
    x = BF16_MAX(x, t);
  }
  return x;
}

__device__ __forceinline__ __nv_bfloat16
BlockReduceMax(__nv_bfloat16 input[4][4]) {
  // [(4), 32, 32, (4)] => [32, 32]
  __nv_bfloat16 local_max;
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      __nv_bfloat16 t = BF16_ABS(input[i][j]);
      local_max = (i == 0 && j == 0) ? t : BF16_MAX(local_max, t);
    }
  }

  // [32, (32)] => [32]
  __nv_bfloat16 warp_max = WarpReduceMax(local_max);

  // [(32)] => [1]
  __shared__ __nv_bfloat16 block_max[32];
  if (threadIdx.x == 0) {
    block_max[threadIdx.y] = warp_max;
  }
  __syncthreads();
  if (threadIdx.y == 0) {
    warp_max = WarpReduceMax(block_max[threadIdx.x]);
    if (threadIdx.x == 0) {
      block_max[0] = warp_max;
    }
  }
  __syncthreads();

  return block_max[0];
}
#endif  // PADDLE_PHI_KERNELS_FUSION_GPU_FUSED_STACK_TRANSPOSE_QUANT_H_
