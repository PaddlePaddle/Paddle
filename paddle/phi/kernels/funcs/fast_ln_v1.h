/* Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include "paddle/phi/common/amp_type_traits.h"
#include "paddle/phi/kernels/funcs/aligned_vector.h"

namespace phi {
namespace funcs {
namespace fast_ln_v1 {

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
template <typename T,
          typename U,
          typename ScaleT = U,
          int VecSize = 8,
          int WARPS_M = 4,
          int WARPS_N = 1,
          int BYTES_PER_LDG = 16,
          int ELTS_PER_ROW = 1024,
          int THREADS_PER_WARP = 32,
          int THREADS_PER_ROW = WARPS_N *THREADS_PER_WARP,
          int THREADS_PER_CTA = WARPS_M *THREADS_PER_ROW,
          int ROWS_PER_CTA = WARPS_M,
          int ELTS_PER_ROW_PER_CTA = THREADS_PER_ROW *VecSize,
          int LDGS = ELTS_PER_ROW / ELTS_PER_ROW_PER_CTA>
__global__ __launch_bounds__(THREADS_PER_CTA) void fast_ln_v1_fwd_kernel(
    int rows,
    int cols,
    const float epsilon,
    const T *__restrict__ x_ptr,
    const ScaleT *__restrict__ gamma_ptr,
    const ScaleT *__restrict__ beta_ptr,
    U *__restrict__ mean_out_ptr,
    U *__restrict__ var_out_ptr,
    T *__restrict__ y_ptr) {
  __shared__ U smem[WARPS_M * WARPS_N];
  using Vec = phi::AlignedVector<T, VecSize>;
  using Vec_scale = phi::AlignedVector<ScaleT, VecSize>;

  const int tidx = threadIdx.x;
  const int bidx = blockIdx.x;
  const int lane = tidx % THREADS_PER_WARP;  // 0, 1, ..., 31
  const int warp = tidx / THREADS_PER_WARP;  // 0, 1, 2, 3
  const int warp_n = warp % WARPS_N;         // 0
  const int warp_m = warp / WARPS_N;         // 0, 1, 2, 3

  const int c = warp_n * THREADS_PER_WARP + lane;  // lane
  const int r = bidx * ROWS_PER_CTA + warp_m;      // row id

  Vec_scale gamma[LDGS];
  Vec_scale beta[LDGS];
#pragma unroll
  for (int it = 0, col = c; it < LDGS; it++) {
    if (col < cols) {
      phi::Load<ScaleT, VecSize>(gamma_ptr + col * VecSize, &gamma[it]);
      phi::Load<ScaleT, VecSize>(beta_ptr + col * VecSize, &beta[it]);
    } else {
      gamma[it] = Vec_scale{};
      beta[it] = Vec_scale{};
    }
    col += THREADS_PER_ROW;
  }

  constexpr U rn = 1.f / U(ELTS_PER_ROW);
  for (int row = r; row < rows; row += gridDim.x * ROWS_PER_CTA) {
    Vec x[LDGS];
#pragma unroll
    for (int it = 0, col = c; it < LDGS; it++) {
      if (col < cols) {
        phi::Load<T, VecSize>(
            x_ptr + static_cast<int64_t>(row) * ELTS_PER_ROW + col * VecSize,
            &x[it]);
      } else {
        x[it] = Vec{};
      }
      col += THREADS_PER_ROW;
    }
    U xf[LDGS * VecSize];

    U mu_local = 0.f;

#pragma unroll
    for (int it = 0; it < LDGS; it++) {
#pragma unroll
      for (int jt = 0; jt < VecSize; jt++) {
        xf[it * VecSize + jt] = U(x[it][jt]);
        mu_local += xf[it * VecSize + jt];
      }
    }

#pragma unroll
    for (int it = 1; it < THREADS_PER_WARP; it *= 2) {
#ifdef PADDLE_WITH_HIP
      mu_local += __shfl_xor(mu_local, it);
#else
      mu_local += __shfl_xor_sync(uint32_t(-1), mu_local, it);
#endif
    }
    if (WARPS_N > 1) {
      if (lane == 0) {
        smem[warp_m * WARPS_N + warp_n] = mu_local;
      }
      __syncthreads();
      if (tidx % THREADS_PER_ROW == 0) {
        mu_local = 0.f;
#pragma unroll
        for (int it = 0; it < WARPS_N; ++it) {
          mu_local += smem[warp_m * WARPS_N + it];
        }
        smem[warp_m * WARPS_N] = mu_local;
      }
      __syncthreads();
      mu_local = smem[warp_m * WARPS_N];
    }

    mu_local *= rn;
    if (lane == 0) {
      mean_out_ptr[row] = mu_local;
    }
    U var_local = 0.f;

#pragma unroll
    for (int it = 0; it < LDGS; it++) {
#pragma unroll
      for (int jt = 0; jt < VecSize; jt++) {
        U diff = xf[it * VecSize + jt] - mu_local;
        var_local += diff * diff;
      }
    }

#pragma unroll
    for (int it = 1; it < THREADS_PER_WARP; it *= 2) {
#ifdef PADDLE_WITH_HIP
      var_local += __shfl_xor(var_local, it);
#else
      var_local += __shfl_xor_sync(uint32_t(-1), var_local, it);
#endif
    }

    if (WARPS_N > 1) {
      __syncthreads();
      if (lane == 0) {
        smem[warp_m * WARPS_N + warp_n] = var_local;
      }
      __syncthreads();
      if (tidx % THREADS_PER_ROW == 0) {
        var_local = 0.f;
#pragma unroll
        for (int it = 0; it < WARPS_N; ++it) {
          var_local += smem[warp_m * WARPS_N + it];
        }
        smem[warp_m * WARPS_N] = var_local;
      }
      __syncthreads();
      var_local = smem[warp_m * WARPS_N];
    }

    // Note: to assure if it is right for double
    U rsigma = rsqrtf(var_local * rn + epsilon);
    if (lane == 0) {
      var_out_ptr[row] = var_local * rn;
    }

#pragma unroll
    for (int it = 0; it < LDGS; it++) {
#pragma unroll
      for (int jt = 0; jt < VecSize; jt++) {
        // use fp16 to compute
        // ScaleT tmp = static_cast<ScaleT>(rsigma * (xf[it * VecSize + jt] -
        // mu_local));
        // x[it][jt] = gamma[it][jt] *  tmp + beta[it][jt];
        // cast to fp32 to compute
        U tmp = (rsigma * (static_cast<U>(xf[it * VecSize + jt]) - mu_local));
        x[it][jt] = static_cast<T>(static_cast<U>(gamma[it][jt]) * tmp +
                                   static_cast<U>(beta[it][jt]));
      }
    }

#pragma unroll
    for (int it = 0, col = c; it < LDGS; it++) {
      if (col < cols) {
        phi::Store<T, VecSize>(
            x[it],
            y_ptr + static_cast<int64_t>(row) * ELTS_PER_ROW + col * VecSize);
      }
      col += THREADS_PER_ROW;
    }
  }
}
#endif
}  // namespace fast_ln_v1
}  // namespace funcs
}  // namespace phi
