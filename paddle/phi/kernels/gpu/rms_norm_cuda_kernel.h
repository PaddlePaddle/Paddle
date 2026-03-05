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

#pragma once

#include <cuda_runtime.h>

#include "paddle/common/ddim.h"
#include "paddle/common/flags.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/common/amp_type_traits.h"
#include "paddle/phi/common/data_type.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/kernels/reduce_sum_kernel.h"

COMMON_DECLARE_bool(use_accuracy_compatible_kernel);

namespace phi {

// -----------------------------------------------------------------------
//  Constants
// -----------------------------------------------------------------------

static constexpr int kCUDANumThreads = 256;
static constexpr int kCUDABlockReduceNumThreads = 512;
static constexpr int kWarpSize = 32;

// -----------------------------------------------------------------------
//  Helper Functions & Structs
// -----------------------------------------------------------------------

template <typename T>
__device__ __forceinline__ T Rsqrt_(T x);

template <>
__device__ __forceinline__ float Rsqrt_<float>(float x) {
  return rsqrtf(x);
}

template <>
__device__ __forceinline__ double Rsqrt_<double>(double x) {
  return rsqrt(x);
}

template <typename T, int kVecSize>
struct alignas(sizeof(T) * kVecSize) aligned_vector {
  T val[kVecSize];
};

template <typename T1, typename T2>
struct SimplePair {
  T1 first;
  T2 second;

  __host__ __device__ SimplePair() {}
  __host__ __device__ SimplePair(T1 f, T2 s) : first(f), second(s) {}
};

template <typename T>
bool can_vectorize(const T* ptr, int alignment) {
  uint64_t addr = reinterpret_cast<uint64_t>(ptr);
  return addr % alignment == 0;
}

// -----------------------------------------------------------------------
//  Welford Algorithms
// -----------------------------------------------------------------------
template <typename scalar_t, typename index_t>
struct WelfordData {
  scalar_t mean;
  scalar_t m2;
  index_t n;
  scalar_t nf;

  __host__ __device__ WelfordData() : mean(0), m2(0), n(0), nf(0) {}

  __host__ __device__
  WelfordData(scalar_t mean, scalar_t m2, index_t n, scalar_t nf)
      : mean(mean), m2(m2), n(n), nf(nf) {}
};

// -----------------------------------------------------------------------
//  Warp & Block Reductions
// -----------------------------------------------------------------------

template <typename T>
__device__ __forceinline__ T WARP_SHFL_DOWN_(T value,
                                             int delta,
                                             int width = kWarpSize,
                                             unsigned int mask = 0xffffffff) {
#ifndef __HIP_PLATFORM_HCC__
  return __shfl_down_sync(mask, value, delta, width);
#else
  return __shfl_down(value, delta, width);
#endif
}

template <typename T>
__device__ __forceinline__ T WARP_SHFL_(T value,
                                        int srcLane,
                                        int width = kWarpSize,
                                        unsigned int mask = 0xffffffff) {
#ifndef __HIP_PLATFORM_HCC__
  return __shfl_sync(mask, value, srcLane, width);
#else
  return __shfl(value, srcLane, width);
#endif
}

template <typename T>
__device__ __forceinline__ T WARP_SHFL_XOR_(T value,
                                            int laneMask,
                                            int width = kWarpSize,
                                            unsigned int mask = 0xffffffff) {
#ifndef __HIP_PLATFORM_HCC__
  return __shfl_xor_sync(mask, value, laneMask, width);
#else
  return __shfl_xor(value, laneMask, width);
#endif
}

template <typename T>
__device__ T BlockReduceSum(T val, T* shared) {
  int lane = threadIdx.x % kWarpSize;
  int wid = threadIdx.x / kWarpSize;

  for (int offset = kWarpSize >> 1; offset > 0; offset >>= 1) {
    val += WARP_SHFL_DOWN_(val, offset);
  }

  if (lane == 0) {
    shared[wid] = val;
  }
  __syncthreads();

  // Assuming blockDim.x <= 1024, max 32 warps
  val = (threadIdx.x < blockDim.x / kWarpSize) ? shared[lane] : T(0);

  if (wid == 0) {
    for (int offset = kWarpSize >> 1; offset > 0; offset >>= 1) {
      val += WARP_SHFL_DOWN_(val, offset);
    }
  }
  return val;
}

template <typename scalar_t,
          typename acc_scalar_t,
          typename index_t,
          typename res_t>
struct WelfordOps {
  acc_scalar_t correction;
  bool take_sqrt;

 public:
  using acc_t = WelfordData<acc_scalar_t, index_t>;
  inline __device__ acc_t reduce(acc_t acc,
                                 scalar_t data,
                                 index_t /*idx*/) const {
    index_t new_n = acc.n + 1;
    acc_scalar_t new_nf = static_cast<acc_scalar_t>(new_n);
    acc_scalar_t delta = data - acc.mean;
    acc_scalar_t new_mean = acc.mean + delta / new_nf;
    acc_scalar_t new_delta = data - new_mean;
    return {
        new_mean,
        acc.m2 + delta * new_delta,
        new_n,
        new_nf,
    };
  }
  inline __device__ acc_t combine(acc_t a, acc_t b) const {
    if (a.nf == 0) {
      return b;
    }
    if (b.nf == 0) {
      return a;
    }
    acc_scalar_t delta = b.mean - a.mean;
    acc_scalar_t new_count = a.nf + b.nf;
    acc_scalar_t nb_over_n = b.nf / new_count;
    return {a.mean + delta * nb_over_n,
            a.m2 + b.m2 + delta * delta * a.nf * nb_over_n,
            -1,
            new_count};
  }
  inline __device__ res_t project(acc_t acc) const {
    const scalar_t mean = static_cast<scalar_t>(acc.mean);
    const acc_scalar_t divisor = acc.nf > correction ? acc.nf - correction : 0;
    const acc_scalar_t var = acc.m2 / divisor;
    res_t results(take_sqrt ? std::sqrt(var) : var, mean);
    return results;
  }

  static __device__ acc_t translate_idx(acc_t acc, int64_t /*base_idx*/) {
    return acc;
  }

#if defined(__CUDACC__) || defined(__HIPCC__)
  inline __device__ acc_t warp_shfl_down(acc_t acc, int offset) const {
    return {WARP_SHFL_DOWN_(acc.mean, offset),
            WARP_SHFL_DOWN_(acc.m2, offset),
            WARP_SHFL_DOWN_(acc.n, offset),
            WARP_SHFL_DOWN_(acc.nf, offset)};
  }
#endif
  __host__ __device__ WelfordOps(acc_scalar_t correction, bool take_sqrt)
      : correction(correction), take_sqrt(take_sqrt) {}
};

// -----------------------------------------------------------------------
//  Forward Kernels
// -----------------------------------------------------------------------

// Non-vectorized RowwiseMoments for RMSNorm
template <typename T, typename T_ACC>
__global__ void RowwiseMomentsCUDAKernel(int64_t N,
                                         T_ACC eps,
                                         const T* X,
                                         T_ACC* rstd) {
  using WelfordType = WelfordData<T_ACC, int64_t>;
  using WelfordOp = WelfordOps<T_ACC, T_ACC, int64_t, SimplePair<T_ACC, T_ACC>>;

  const int64_t i = blockIdx.x;
  WelfordOp welford_op = {/*correction=*/0, /*take_sqrt=*/false};
  WelfordType val(0, 0, 0, 0);

  for (int64_t j = threadIdx.x; j < N; j += blockDim.x) {
    const int64_t index = i * N + j;
    val = welford_op.reduce(val, static_cast<T_ACC>(X[index]), index);
  }

  // Block Reduce
  // 1. Warp Reduce
  for (int offset = kWarpSize >> 1; offset > 0; offset >>= 1) {
    WelfordType wdB = welford_op.warp_shfl_down(val, offset);
    val = welford_op.combine(val, wdB);
  }

  // 2. Block Reduce (via shared memory)
  __shared__
      typename std::aligned_storage<sizeof(WelfordType),
                                    alignof(WelfordType)>::type val_shared[32];
  WelfordType* val_shared_ptr = reinterpret_cast<WelfordType*>(val_shared);

  int lane = threadIdx.x % kWarpSize;
  int wid = threadIdx.x / kWarpSize;

  __syncthreads();
  if (lane == 0) {
    val_shared_ptr[wid] = val;
  }
  __syncthreads();

  val = (threadIdx.x < blockDim.x / kWarpSize) ? val_shared_ptr[lane]
                                               : WelfordType(0, 0, 0, 0);

  // Final Warp Reduce for the first warp
  if (wid == 0) {
    for (int offset = kWarpSize >> 1; offset > 0; offset >>= 1) {
      WelfordType wdB = welford_op.warp_shfl_down(val, offset);
      val = welford_op.combine(val, wdB);
    }
  }

  if (threadIdx.x == 0) {
    T_ACC m1;  // mean
    T_ACC m2;  // var
    SimplePair<T_ACC, T_ACC> res = welford_op.project(val);
    m2 = res.first;
    m1 = res.second;
    rstd[i] = Rsqrt_<T_ACC>(m2 + m1 * m1 + eps);
  }
}

// Non-vectorized Forward for RMSNorm
template <typename T, typename T_ACC>
__global__ void RMSNormForwardCUDAKernel(
    int64_t N, const T* X, const T_ACC* rstd, const T* scale, T* Y) {
  const int64_t i = blockIdx.x;
  for (int64_t j = threadIdx.x; j < N; j += blockDim.x) {
    const int64_t index = i * N + j;
    const T_ACC scale_v =
        scale == nullptr ? T_ACC(1) : static_cast<T_ACC>(scale[j]);
    Y[index] = static_cast<T>((static_cast<T_ACC>(X[index])) *
                              static_cast<T_ACC>(rstd[i]) * scale_v);
  }
}

// Vectorized Helper
template <typename T, typename T_ACC, int kVecSize>
__device__ T_ACC compute_stats(const T* __restrict__ X,
                               const int N,
                               T_ACC* buf) {
  using vec_t = aligned_vector<T, kVecSize>;
  const vec_t* X_vec = reinterpret_cast<const vec_t*>(X);
  const int numx = blockDim.x * blockDim.y;
  const int thrx = threadIdx.x + threadIdx.y * blockDim.x;
  const int n_vec_to_read = N / kVecSize;
  T_ACC sigma2 = 0;

  for (int i = thrx; i < n_vec_to_read; i += numx) {
    vec_t data = X_vec[i];
#pragma unroll
    for (int ii = 0; ii < kVecSize; ii++) {
      T_ACC val = static_cast<T_ACC>(data.val[ii]);
      sigma2 += val * val;
    }
  }

  // Intra-warp reduction
  for (int offset = (kWarpSize >> 1); offset > 0; offset >>= 1) {
    sigma2 += WARP_SHFL_DOWN_(sigma2, offset);
  }

  // Inter-warp reductions
  if (blockDim.y > 1) {
    T_ACC* meansigmabuf = buf;
    // Use simpler layout: just sigma2
    for (int offset = blockDim.y >> 1; offset > 0; offset >>= 1) {
      if (threadIdx.x == 0 && threadIdx.y >= offset &&
          threadIdx.y < 2 * offset) {
        const int wrt_y = threadIdx.y - offset;
        meansigmabuf[wrt_y] = sigma2;
      }
      __syncthreads();
      if (threadIdx.x == 0 && threadIdx.y < offset) {
        sigma2 += meansigmabuf[threadIdx.y];
      }
      __syncthreads();
    }
    if (threadIdx.x == 0 && threadIdx.y == 0) {
      meansigmabuf[0] = sigma2 / static_cast<T_ACC>(N);
    }
    __syncthreads();
    return meansigmabuf[0];

  } else {
    return WARP_SHFL_(sigma2, 0) / static_cast<T_ACC>(N);
  }
}

template <typename T, typename T_ACC, int kVecSize>
__global__ void vectorized_rms_norm_kernel(const int N,
                                           T_ACC eps,
                                           const T* __restrict__ X,
                                           const T* scale,
                                           T_ACC* rstd,
                                           T* Y) {
  extern __shared__ char s_data_raw[];
  T_ACC* s_data = reinterpret_cast<T_ACC*>(s_data_raw);

  auto i1 = blockIdx.x;
  const T* block_row = X + i1 * N;

  // Compute stats
  T_ACC sigma2 = compute_stats<T, T_ACC, kVecSize>(block_row, N, s_data);

  using vec_t = aligned_vector<T, kVecSize>;
  const vec_t* X_vec = reinterpret_cast<const vec_t*>(block_row);
  const vec_t* scale_vec =
      (scale != nullptr) ? reinterpret_cast<const vec_t*>(scale) : nullptr;
  vec_t* Y_vec = reinterpret_cast<vec_t*>(Y + i1 * N);

  const int numx = blockDim.x * blockDim.y;
  const int thrx = threadIdx.x + threadIdx.y * blockDim.x;
  const int n_vec_to_read = N / kVecSize;

  T_ACC rstd_val = Rsqrt_<T_ACC>(sigma2 + eps);

  if (scale_vec != nullptr) {
    for (int i = thrx; i < n_vec_to_read; i += numx) {
      vec_t data = X_vec[i];
      vec_t out;
#pragma unroll
      for (int ii = 0; ii < kVecSize; ii++) {
        out.val[ii] =
            static_cast<T>(static_cast<T_ACC>(scale_vec[i].val[ii]) *
                           (rstd_val * static_cast<T_ACC>(data.val[ii])));
      }
      Y_vec[i] = out;
    }
  } else {
    for (int i = thrx; i < n_vec_to_read; i += numx) {
      vec_t data = X_vec[i];
      vec_t out;
#pragma unroll
      for (int ii = 0; ii < kVecSize; ii++) {
        out.val[ii] =
            static_cast<T>(rstd_val * static_cast<T_ACC>(data.val[ii]));
      }
      Y_vec[i] = out;
    }
  }

  if (thrx == 0) {
    rstd[i1] = rstd_val;
  }
}

template <typename T, typename T_ACC, int kVecSize>
void launch_vectorized_rms_norm_kernel_driver(int N,
                                              int64_t M,
                                              T_ACC eps,
                                              const T* X_data,
                                              const T* scale_data,
                                              T* Y_data,
                                              T_ACC* rstd_data,
                                              cudaStream_t stream) {
  const int num_threads = 128;
  const dim3 threads(kWarpSize, num_threads / kWarpSize, 1);
  dim3 blocks(M);

  // Shared memory for reduction: need size proportional to threads.y and T_ACC
  int nshared = threads.y > 1 ? threads.y * 3 / 2 * sizeof(T_ACC) : 0;

  vectorized_rms_norm_kernel<T, T_ACC, kVecSize>
      <<<blocks, threads, nshared, stream>>>(
          N, eps, X_data, scale_data, rstd_data, Y_data);
}

// -----------------------------------------------------------------------
//  Backward Kernels
// -----------------------------------------------------------------------

template <typename T, typename T_ACC>
__device__ __inline__ void compute_gI(const T* __restrict__ dY,
                                      const T* __restrict__ X,
                                      const T_ACC* __restrict__ rstd,
                                      const T* __restrict__ scale,
                                      T* dX,
                                      const int N,
                                      T_ACC* buf) {
  const auto i1 = blockIdx.x;
  const T_ACC rstd_val = rstd[i1];
  T_ACC stats_x2{0};
  constexpr int unroll = 4;
  auto l = unroll * threadIdx.x;
  const T* X_i = X + i1 * N;
  const T* dY_i = dY + i1 * N;
  T* dX_i = dX + i1 * N;

  for (; l + unroll - 1 < N; l += blockDim.x * unroll) {
#pragma unroll
    for (int k = 0; k < unroll; k++) {
      const auto scale_val =
          (scale != nullptr) ? static_cast<T_ACC>(scale[l + k]) : T_ACC(1);
      const auto c_h = static_cast<T_ACC>(X_i[l + k]);
      const auto c_loss = static_cast<T_ACC>(dY_i[l + k]);
      stats_x2 += c_loss * scale_val * (c_h)*rstd_val;
    }
  }
  for (; l < N; l++) {
    const auto scale_val =
        (scale != nullptr) ? static_cast<T_ACC>(scale[l]) : T_ACC(1);
    const auto c_h = static_cast<T_ACC>(X_i[l]);
    const auto c_loss = static_cast<T_ACC>(dY_i[l]);
    stats_x2 += c_loss * scale_val * (c_h)*rstd_val;
  }

  stats_x2 = BlockReduceSum(stats_x2, buf);

  if (threadIdx.x == 0) {
    buf[0] = stats_x2;
  }
  __syncthreads();
  stats_x2 = buf[0];

  T_ACC fH = N;
  T_ACC term1 = (T_ACC(1) / fH) * rstd_val;

  for (int l = threadIdx.x; l < N; l += blockDim.x) {
    const auto x = static_cast<T_ACC>(X_i[l]);
    const auto dy = static_cast<T_ACC>(dY_i[l]);
    const auto scale_val =
        (scale != nullptr) ? static_cast<T_ACC>(scale[l]) : T_ACC(1);

    T_ACC f_grad_input = fH * scale_val * dy;
    f_grad_input -= (x)*rstd_val * stats_x2;
    f_grad_input *= term1;
    dX_i[l] = static_cast<T>(f_grad_input);
  }
}

template <typename T, typename T_ACC>
__global__ void rms_norm_grad_input_kernel(const T* __restrict__ dY,
                                           const T* __restrict__ X,
                                           const T_ACC* __restrict__ rstd,
                                           const T* __restrict__ scale,
                                           T* dX,
                                           const int N) {
  alignas(sizeof(double)) extern __shared__ char s_data1[];
  T_ACC* buf = reinterpret_cast<T_ACC*>(&s_data1);
  compute_gI<T, T_ACC>(dY, X, rstd, scale, dX, N, buf);
}

template <typename T, typename T_ACC, int kVecSize>
__global__ void rms_norm_grad_input_kernel_vectorized(
    const T* __restrict__ dY,
    const T* __restrict__ X,
    const T_ACC* __restrict__ rstd,
    const T* __restrict__ scale,
    T* dX,
    const int N) {
  alignas(sizeof(double)) extern __shared__ char shared_data[];
  T_ACC* reduce_buf = reinterpret_cast<T_ACC*>(&shared_data);

  const auto bIdx = blockIdx.x;
  const T_ACC rstd_val = rstd[bIdx];
  const T* X_i = X + bIdx * N;
  const T* dY_i = dY + bIdx * N;
  T* dX_i = dX + bIdx * N;

  using vec_t = aligned_vector<T, kVecSize>;
  const vec_t* const X_i_vec_ptr = reinterpret_cast<const vec_t*>(X_i);
  const vec_t* const dY_i_vec_ptr = reinterpret_cast<const vec_t*>(dY_i);
  const vec_t* const scale_vec_ptr =
      (scale != nullptr) ? reinterpret_cast<const vec_t*>(scale) : nullptr;
  vec_t* const dX_i_vec = reinterpret_cast<vec_t*>(dX_i);

  vec_t X_i_vec_reg, dY_i_vec_reg, scale_vec_reg, dX_i_vec_reg;
  for (int k = 0; k < kVecSize; ++k) {
    scale_vec_reg.val[k] = T(1);
  }

  T_ACC stats_x2{0};
  unsigned int l = threadIdx.x * kVecSize;
  for (; l + kVecSize - 1 < N; l += blockDim.x * kVecSize) {
    unsigned int vec_idx = l / kVecSize;
    if (scale != nullptr) {
      scale_vec_reg = scale_vec_ptr[vec_idx];
    }

    X_i_vec_reg = X_i_vec_ptr[vec_idx];
    dY_i_vec_reg = dY_i_vec_ptr[vec_idx];

    for (int k = 0; k < kVecSize; ++k) {
      const auto scale_val = static_cast<T_ACC>(scale_vec_reg.val[k]);
      const auto c_h = static_cast<T_ACC>(X_i_vec_reg.val[k]);
      const auto c_loss = static_cast<T_ACC>(dY_i_vec_reg.val[k]);
      stats_x2 += c_loss * scale_val * (c_h)*rstd_val;
    }
  }

  // Tail Loop
  for (; l < N; l++) {
    const auto scale_val =
        (scale != nullptr) ? static_cast<T_ACC>(scale[l]) : T_ACC(1);
    const auto c_h = static_cast<T_ACC>(X_i[l]);
    const auto c_loss = static_cast<T_ACC>(dY_i[l]);
    stats_x2 += c_loss * scale_val * (c_h)*rstd_val;
  }

  stats_x2 = BlockReduceSum(stats_x2, reduce_buf);
  if (threadIdx.x == 0) {
    reduce_buf[0] = stats_x2;
  }
  __syncthreads();
  stats_x2 = reduce_buf[0];

  T_ACC fH = N;
  T_ACC term1 = (T_ACC(1) / fH) * rstd_val;

  l = threadIdx.x * kVecSize;
  for (; l + kVecSize - 1 < N; l += blockDim.x * kVecSize) {
    unsigned int vec_idx = l / kVecSize;
    if (scale != nullptr) {
      scale_vec_reg = scale_vec_ptr[vec_idx];
    }

    X_i_vec_reg = X_i_vec_ptr[vec_idx];
    dY_i_vec_reg = dY_i_vec_ptr[vec_idx];

    for (int k = 0; k < kVecSize; ++k) {
      const auto scale_val = static_cast<T_ACC>(scale_vec_reg.val[k]);
      const auto x = static_cast<T_ACC>(X_i_vec_reg.val[k]);
      const auto dy = static_cast<T_ACC>(dY_i_vec_reg.val[k]);

      T_ACC f_grad_input = fH * scale_val * dy;
      f_grad_input -= (x)*rstd_val * stats_x2;
      f_grad_input *= term1;
      dX_i_vec_reg.val[k] = static_cast<T>(f_grad_input);
    }

    dX_i_vec[vec_idx] = dX_i_vec_reg;
  }

  // Tail Loop
  for (; l < N; l += blockDim.x) {
    const auto x = static_cast<T_ACC>(X_i[l]);
    const auto dy = static_cast<T_ACC>(dY_i[l]);
    const auto scale_val =
        (scale != nullptr) ? static_cast<T_ACC>(scale[l]) : T_ACC(1);

    T_ACC f_grad_input = fH * scale_val * dy;
    f_grad_input -= (x)*rstd_val * stats_x2;
    f_grad_input *= term1;
    dX_i[l] = static_cast<T>(f_grad_input);
  }
}

template <typename T,
          typename T_ACC,
          unsigned int block_dim_x,
          unsigned int block_dim_y,
          unsigned int rows_per_block_y,
          bool check_x,
          bool check_y>
__device__ __forceinline__ void blockReduceScaleBackwardHelper(
    int64_t M_start,
    int64_t M,
    int64_t N,
    const T* __restrict__ dY,
    const T* __restrict__ X,
    const T_ACC* __restrict__ rstd,
    T* __restrict__ dscale,
    T_ACC* dscale_sum) {
  constexpr int rows_per_thread_y = rows_per_block_y / block_dim_y;
  int64_t thread_x = static_cast<int64_t>(blockIdx.x) * block_dim_x +
                     static_cast<int64_t>(threadIdx.x);

  int lane_id = (threadIdx.y * blockDim.x + threadIdx.x) & (kWarpSize - 1);
  int64_t mean_index =
      M_start + static_cast<int64_t>(threadIdx.y) * rows_per_thread_y;
  T_ACC warp_rstd = 0;
  if (lane_id < rows_per_thread_y && mean_index + lane_id < M) {
    warp_rstd = rstd[mean_index + lane_id];
  }

#if defined(__CUDACC__) || defined(__HIPCC__)
  __syncwarp();
#endif

  T_ACC dY_regs[rows_per_thread_y] = {0};
  T_ACC X_regs[rows_per_thread_y] = {0};
#pragma unroll
  for (int i = 0; i < rows_per_thread_y; ++i) {
    int64_t current_y =
        M_start + static_cast<int64_t>(threadIdx.y) * rows_per_thread_y + i;
    bool active = true;
    if (check_x && thread_x >= N) {
      active = false;
    }
    if (check_y && current_y >= M) {
      active = false;
    }
    if (active) {
      dY_regs[i] = static_cast<T_ACC>(dY[current_y * N + thread_x]);
      X_regs[i] = static_cast<T_ACC>(X[current_y * N + thread_x]);
    }
  }

#pragma unroll
  for (int i = 0; i < rows_per_thread_y; ++i) {
    T_ACC rstd_reg = WARP_SHFL_(warp_rstd, i, kWarpSize);
    *dscale_sum += dY_regs[i] * (X_regs[i]) * rstd_reg;
  }
}

template <typename T,
          typename T_ACC,
          unsigned int block_dim_x,
          unsigned int block_dim_y,
          unsigned int rows_per_block_y,
          bool check_x,
          bool check_y>
__device__ __forceinline__ void blockReduceScaleBackwardWithChecks(
    int64_t M,
    int64_t N,
    const T* __restrict__ dY,
    const T* __restrict__ X,
    const T_ACC* __restrict__ rstd,
    T* __restrict__ dscale,
    T_ACC* dscale_sum) {
  for (int64_t M_start = static_cast<int64_t>(blockIdx.y) * rows_per_block_y;
       M_start < M;
       M_start += rows_per_block_y * gridDim.y) {
    int64_t M_end = M_start + rows_per_block_y - 1;
    if (!check_y || M_end < M) {
      blockReduceScaleBackwardHelper<T,
                                     T_ACC,
                                     block_dim_x,
                                     block_dim_y,
                                     rows_per_block_y,
                                     check_x,
                                     false>(
          M_start, M, N, dY, X, rstd, dscale, dscale_sum);
    } else {
      blockReduceScaleBackwardHelper<T,
                                     T_ACC,
                                     block_dim_x,
                                     block_dim_y,
                                     rows_per_block_y,
                                     check_x,
                                     true>(
          M_start, M, N, dY, X, rstd, dscale, dscale_sum);
    }
  }
}

template <typename T,
          typename T_ACC,
          unsigned int block_dim_x,
          unsigned int block_dim_y,
          unsigned int rows_per_block_y,
          bool partial_reduction,
          bool aligned_grid>
__global__ void ScaleBackwardCUDAKernelTemplate(int64_t M,
                                                int64_t N,
                                                const T* __restrict__ dY,
                                                const T* __restrict__ X,
                                                const T_ACC* __restrict__ rstd,
                                                T* __restrict__ dscale) {
  constexpr int rows_per_thread_y = rows_per_block_y / block_dim_y;
  static_assert(rows_per_thread_y <= kWarpSize);

  T_ACC dscale_sum = 0;

  // Template : Boundary check of x and y
  if (aligned_grid) {
    blockReduceScaleBackwardWithChecks<T,
                                       T_ACC,
                                       block_dim_x,
                                       block_dim_y,
                                       rows_per_block_y,
                                       false,
                                       false>(
        M, N, dY, X, rstd, dscale, &dscale_sum);
  } else {
    if (static_cast<int64_t>(blockIdx.x) * block_dim_x + block_dim_x - 1 < N) {
      blockReduceScaleBackwardWithChecks<T,
                                         T_ACC,
                                         block_dim_x,
                                         block_dim_y,
                                         rows_per_block_y,
                                         false,
                                         true>(
          M, N, dY, X, rstd, dscale, &dscale_sum);
    } else {
      blockReduceScaleBackwardWithChecks<T,
                                         T_ACC,
                                         block_dim_x,
                                         block_dim_y,
                                         rows_per_block_y,
                                         true,
                                         true>(
          M, N, dY, X, rstd, dscale, &dscale_sum);
    }
  }

  int64_t thread_x =
      (static_cast<int64_t>(blockIdx.x)) * block_dim_x + threadIdx.x;

  if (partial_reduction || (blockDim.y == 1 && gridDim.y == 1)) {
    if (aligned_grid || thread_x < N) {
      int64_t thread_y =
          (static_cast<int64_t>(blockIdx.y)) * blockDim.y + threadIdx.y;
      if (dscale) {
        dscale[thread_y * N + thread_x] = static_cast<T>(dscale_sum);
      }
    }
  } else {
    // Full reduction using shared memory
    static_assert(rows_per_thread_y <= kWarpSize);
    alignas(sizeof(double)) extern __shared__ char s_data1[];
    T_ACC* s_data_typed = reinterpret_cast<T_ACC*>(&s_data1);
    T_ACC* s_dscale;
    int padded_bx = (block_dim_x + 1);
    s_dscale = s_data_typed;
    s_dscale[threadIdx.y * padded_bx + threadIdx.x] = dscale_sum;
    __syncthreads();

    static_assert(block_dim_x * block_dim_y % kWarpSize == 0);
    constexpr int warps_available_to_reduce =
        block_dim_x * block_dim_y / kWarpSize;
    int thread_id = threadIdx.y * block_dim_x + threadIdx.x;
    int warp_id = thread_id / kWarpSize;
    int lane_id = thread_id & (kWarpSize - 1);
#pragma unroll
    for (int i = warp_id; i < block_dim_x; i += warps_available_to_reduce) {
      T_ACC reg_dscale;
      if (lane_id < block_dim_y) {
        reg_dscale = s_dscale[lane_id * padded_bx + i];
      }
#pragma unroll
      for (unsigned delta = block_dim_y >> 1; delta >= 1; delta >>= 1) {
        reg_dscale += WARP_SHFL_XOR_(reg_dscale, delta, kWarpSize);
      }

      int64_t out_index = static_cast<int64_t>(blockIdx.x) * block_dim_x + i;
      if (threadIdx.x == 0 && (aligned_grid || out_index < N)) {
        if (dscale) {
          dscale[out_index] = static_cast<T>(reg_dscale);
        }
      }
    }
  }
}

template <typename T,
          typename T_ACC,
          int block_dim_x,
          int block_dim_y,
          int rows_per_block_y>
void ConfigureAndLaunchScaleBackwardKernel(const T* dY_data,
                                           const T* X_data,
                                           const T_ACC* rstd_data,
                                           int64_t M,
                                           int64_t N,
                                           T* dscale_data,
                                           cudaStream_t cuda_stream) {
  bool aligned_grid = (M % rows_per_block_y == 0) && (N % block_dim_x == 0);
  dim3 threads{block_dim_x, block_dim_y};
  dim3 blocks;
  blocks.x = (N + block_dim_x - 1) / block_dim_x;
  blocks.y = 1;
  size_t shmem_sz = (block_dim_x + 1) * block_dim_y * sizeof(T_ACC) * 2;

  if (blocks.y == 1 && threads.y == 1) {
    if (aligned_grid) {
      ScaleBackwardCUDAKernelTemplate<T,
                                      T_ACC,
                                      block_dim_x,
                                      block_dim_y,
                                      rows_per_block_y,
                                      true,
                                      true>
          <<<blocks, threads, shmem_sz, cuda_stream>>>(
              M, N, dY_data, X_data, rstd_data, dscale_data);
    } else {
      ScaleBackwardCUDAKernelTemplate<T,
                                      T_ACC,
                                      block_dim_x,
                                      block_dim_y,
                                      rows_per_block_y,
                                      true,
                                      false>
          <<<blocks, threads, shmem_sz, cuda_stream>>>(
              M, N, dY_data, X_data, rstd_data, dscale_data);
    }
  } else {
    if (aligned_grid) {
      ScaleBackwardCUDAKernelTemplate<T,
                                      T_ACC,
                                      block_dim_x,
                                      block_dim_y,
                                      rows_per_block_y,
                                      false,
                                      true>
          <<<blocks, threads, shmem_sz, cuda_stream>>>(
              M, N, dY_data, X_data, rstd_data, dscale_data);
    } else {
      ScaleBackwardCUDAKernelTemplate<T,
                                      T_ACC,
                                      block_dim_x,
                                      block_dim_y,
                                      rows_per_block_y,
                                      false,
                                      false>
          <<<blocks, threads, shmem_sz, cuda_stream>>>(
              M, N, dY_data, X_data, rstd_data, dscale_data);
    }
  }
}

// -----------------------------------------------------------------------
//  Host API Implementations
// -----------------------------------------------------------------------

template <typename T, typename Context>
void RMSNormFwdKernel(const Context& dev_ctx,
                      const DenseTensor& x,
                      const paddle::optional<DenseTensor>& scale_opt,
                      const std::vector<int64_t>& normalized_shape,
                      double epsilon,
                      DenseTensor* y,
                      DenseTensor* invvar) {
  using T_ACC = typename phi::dtype::MPTypeTrait<T>::Type;

  if (x.numel() == 0) {
    dev_ctx.template Alloc<T>(y);
    dev_ctx.template Alloc<T_ACC>(invvar);
    return;
  }

  int begin_norm_axis = x.dims().size() - normalized_shape.size();

  auto matrix_dim = common::flatten_to_2d(x.dims(), begin_norm_axis);
  int64_t rows = matrix_dim[0];
  int64_t cols = matrix_dim[1];

  auto* scale_ptr = scale_opt.get_ptr();
  const DenseTensor& scale = *scale_ptr;

  auto* x_data = x.data<T>();
  auto* scale_data = scale_ptr ? scale.data<T>() : nullptr;
  auto* y_data = dev_ctx.template Alloc<T>(y);
  auto* rstd_data = dev_ctx.template Alloc<T_ACC>(invvar);

  auto stream = dev_ctx.stream();

  // When using a vectorization size of 8 in fp16 and bf16, there may be
  // misalignment of accuracy and torch alignment.
  if (!FLAGS_use_accuracy_compatible_kernel && rows <= 1024 &&
      (cols / rows >= 32)) {
    constexpr int num_vec_elems2 = 8;
    constexpr int alignment2 = num_vec_elems2 * sizeof(T);
    bool can_vec_X2 = can_vectorize(x_data, alignment2);
    bool can_vec_Y2 = can_vectorize(y_data, alignment2);
    bool can_vec_scale2 = can_vectorize(scale_data, alignment2);
    bool is_supported_type2 = (std::is_same<T, phi::dtype::float16>::value ||
                               std::is_same<T, phi::dtype::bfloat16>::value);
    if (is_supported_type2 &&
        cols <=
            static_cast<int64_t>(1ULL << std::numeric_limits<float>::digits) &&
        cols % num_vec_elems2 == 0 && can_vec_X2 && can_vec_Y2 &&
        can_vec_scale2) {
      launch_vectorized_rms_norm_kernel_driver<T, T_ACC, 8>(
          cols,
          rows,
          static_cast<T_ACC>(epsilon),
          x_data,
          scale_data,
          y_data,
          rstd_data,
          stream);
      return;
    }
  }

  // Check vectorization conditions
  constexpr int num_vec_elems = 4;
  constexpr int alignment = num_vec_elems * sizeof(T);
  bool can_vec_X = can_vectorize(x_data, alignment);
  bool can_vec_Y = can_vectorize(y_data, alignment);
  bool can_vec_scale = can_vectorize(scale_data, alignment);
  bool is_supported_type = (std::is_same<T, float>::value ||
                            std::is_same<T, phi::dtype::float16>::value ||
                            std::is_same<T, phi::dtype::bfloat16>::value);

  if (is_supported_type &&
      cols <=
          static_cast<int64_t>(1ULL << std::numeric_limits<float>::digits) &&
      cols % num_vec_elems == 0 && can_vec_X && can_vec_Y && can_vec_scale) {
    launch_vectorized_rms_norm_kernel_driver<T, T_ACC, 4>(
        cols,
        rows,
        static_cast<T_ACC>(epsilon),
        x_data,
        scale_data,
        y_data,
        rstd_data,
        stream);

  } else {
    RowwiseMomentsCUDAKernel<T, T_ACC>
        <<<rows, kCUDABlockReduceNumThreads, 0, stream>>>(
            cols, static_cast<T_ACC>(epsilon), x_data, rstd_data);

    RMSNormForwardCUDAKernel<T, T_ACC><<<rows, kCUDANumThreads, 0, stream>>>(
        cols, x_data, rstd_data, scale_data, y_data);
  }
}

template <typename T, typename Context>
void RMSNormBwdKernel(const Context& dev_ctx,
                      const DenseTensor& X,
                      const paddle::optional<DenseTensor>& scale_opt,
                      const DenseTensor& invvar,
                      const DenseTensor& dY,
                      const std::vector<int64_t>& normalized_shape,
                      double epsilon,
                      DenseTensor* dX,
                      DenseTensor* dscale) {
  using T_ACC = typename phi::dtype::MPTypeTrait<T>::Type;

  if (X.numel() == 0) {
    if (dX) {
      dev_ctx.template Alloc<T>(dX);
    }
    if (dscale) {
      dev_ctx.template Alloc<T_ACC>(dscale);
    }
    return;
  }

  int begin_norm_axis = X.dims().size() - normalized_shape.size();

  // X, dY: [Batch, ..., Feature] -> flatten to [M, N]
  // scale, dscale: [Feature] -> [N]
  // invvar: [Batch, ...] -> [M]

  auto matrix_dim = common::flatten_to_2d(X.dims(), begin_norm_axis);
  int64_t M = matrix_dim[0];
  int64_t N = matrix_dim[1];

  auto* scale_ptr = scale_opt.get_ptr();
  const DenseTensor& scale = *scale_ptr;

  auto* dY_data = dY.data<T>();
  auto* X_data = X.data<T>();
  auto* scale_data = scale_ptr ? scale.data<T>() : nullptr;
  auto* invvar_data = invvar.data<T_ACC>();

  auto* dX_data = dX ? dev_ctx.template Alloc<T>(dX) : nullptr;
  auto* dscale_data = dscale ? dev_ctx.template Alloc<T>(dscale) : nullptr;

  auto stream = dev_ctx.stream();

  // 1. Compute dX
  if (dX_data) {
    static constexpr int kVecSize = 4;
    bool bVectorSizeMultiple = (N % kVecSize == 0);
    const unsigned int alignment = sizeof(T) * kVecSize;
    bool bAlignedBuffers = can_vectorize(dY_data, alignment) &&
                           can_vectorize(X_data, alignment) &&
                           can_vectorize(scale_data, alignment) &&
                           can_vectorize(dX_data, alignment);
    bool is_supported_type = (std::is_same<T, float>::value ||
                              std::is_same<T, phi::dtype::float16>::value ||
                              std::is_same<T, phi::dtype::bfloat16>::value);

    const unsigned int alignment2 = sizeof(T) * 8;
    bool bAlignedBuffers2 = can_vectorize(dY_data, alignment2) &&
                            can_vectorize(X_data, alignment2) &&
                            can_vectorize(scale_data, alignment2) &&
                            can_vectorize(dX_data, alignment2);
    bool is_supported_type2 = (std::is_same<T, phi::dtype::float16>::value ||
                               std::is_same<T, phi::dtype::bfloat16>::value);

    dim3 blocks(M);
    constexpr int num_threads = 128;
    constexpr int nshared = (num_threads / kWarpSize) * sizeof(T_ACC);

    // When using a vectorization size of 8 in fp16 and bf16, there may be
    // misalignment of accuracy and torch alignment.
    if (!FLAGS_use_accuracy_compatible_kernel && is_supported_type2 &&
        bAlignedBuffers2 && (N % 8 == 0 && M <= 1024 && (N / M >= 32))) {
      rms_norm_grad_input_kernel_vectorized<T, T_ACC, 8>
          <<<blocks, num_threads, nshared, stream>>>(
              dY_data, X_data, invvar_data, scale_data, dX_data, N);
    } else if (is_supported_type && bAlignedBuffers && bVectorSizeMultiple) {
      rms_norm_grad_input_kernel_vectorized<T, T_ACC, kVecSize>
          <<<blocks, num_threads, nshared, stream>>>(
              dY_data, X_data, invvar_data, scale_data, dX_data, N);
    } else {
      rms_norm_grad_input_kernel<T, T_ACC>
          <<<blocks, num_threads, nshared, stream>>>(
              dY_data, X_data, invvar_data, scale_data, dX_data, N);
    }
  }

  // 2. Compute dscale
  if (dscale_data) {
    constexpr int block_dim_x = 32;
    const int sm_count = dev_ctx.GetSMCount();
    if (M > 64 * 1024 && N / block_dim_x < sm_count / 2) {
      // When M>>N and N is very small. We can parallelize and accelerate
      // computation by starting multiple blocks on the M-dimension (y).
      constexpr int block_dim_y = 1;
      constexpr int rows_per_block_y = 32;
      bool aligned_grid = (M % rows_per_block_y == 0) && (N % block_dim_x == 0);
      dim3 threads{block_dim_x, block_dim_y};
      dim3 blocks;
      blocks.x = (N + block_dim_x - 1) / block_dim_x;
      blocks.y = (M + rows_per_block_y - 1) / rows_per_block_y;
      constexpr int max_grid_size = 64 * 1024 / 2;
      blocks.y = std::min<unsigned int>(max_grid_size / blocks.x, blocks.y);

      DenseTensor dscale_blocks;
      dscale_blocks.Resize({static_cast<int64_t>(blocks.y * threads.y), N});
      T* dscale_blocks_ptr = dev_ctx.template Alloc<T>(&dscale_blocks);

      if (aligned_grid) {
        ScaleBackwardCUDAKernelTemplate<T,
                                        T_ACC,
                                        block_dim_x,
                                        block_dim_y,
                                        rows_per_block_y,
                                        true,
                                        true><<<blocks, threads, 0, stream>>>(
            M, N, dY_data, X_data, invvar_data, dscale_blocks_ptr);
      } else {
        ScaleBackwardCUDAKernelTemplate<T,
                                        T_ACC,
                                        block_dim_x,
                                        block_dim_y,
                                        rows_per_block_y,
                                        true,
                                        false><<<blocks, threads, 0, stream>>>(
            M, N, dY_data, X_data, invvar_data, dscale_blocks_ptr);
      }

      // Sum reduction along blocks.y dimension to get final dscale
      phi::SumKernel<T, Context>(
          dev_ctx, dscale_blocks, {0}, dscale->dtype(), false, dscale);

    } else {
      if (M < 64) {
        ConfigureAndLaunchScaleBackwardKernel<T, T_ACC, block_dim_x, 1, 8>(
            dY_data, X_data, invvar_data, M, N, dscale_data, stream);
      } else if (M < 128) {
        ConfigureAndLaunchScaleBackwardKernel<T, T_ACC, block_dim_x, 8, 64>(
            dY_data, X_data, invvar_data, M, N, dscale_data, stream);
      } else if (M < 256) {
        ConfigureAndLaunchScaleBackwardKernel<T, T_ACC, block_dim_x, 16, 128>(
            dY_data, X_data, invvar_data, M, N, dscale_data, stream);
      } else {
        ConfigureAndLaunchScaleBackwardKernel<T, T_ACC, block_dim_x, 32, 256>(
            dY_data, X_data, invvar_data, M, N, dscale_data, stream);
      }
    }
  }
}

}  // namespace phi
