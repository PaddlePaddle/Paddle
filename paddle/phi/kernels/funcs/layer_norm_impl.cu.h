/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

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

#ifdef __NVCC__
#include "cub/cub.cuh"
#endif
#ifdef __HIPCC__
#include <hipcub/hipcub.hpp>
namespace cub = hipcub;
#endif

#include <iostream>

#include "glog/logging.h"

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/backends/gpu/gpu_device_function.h"
#include "paddle/phi/backends/gpu/gpu_dnn.h"
#include "paddle/phi/common/memory_utils.h"
#include "paddle/phi/core/ddim.h"
#include "paddle/phi/kernels/funcs/aligned_vector.h"

namespace phi {
namespace funcs {

template <typename T>
using CudnnDataType = phi::backends::gpu::CudnnDataType<T>;
template <typename T>
using LayerNormParamType = typename CudnnDataType<T>::BatchNormParamType;

inline static int GetDesiredBlockDim(int64_t block_dim) {
#ifdef __HIPCC__
  const int kMaxBlockDim = 256;
  const int lwarpSize = 64;
#else
  const int kMaxBlockDim = 512;
  const int lwarpSize = 32;
#endif
  return block_dim >= kMaxBlockDim ? kMaxBlockDim : lwarpSize;
}

template <typename U>
static __forceinline__ __device__ U WarpReduceSum(U val) {
  unsigned mask = 0u;
  CREATE_SHFL_MASK(mask, true);
  for (int offset = warpSize / 2; offset > 0; offset /= 2) {
    val += phi::backends::gpu::CudaShuffleDownSync(mask, val, offset);
  }
  return val;
}

template <typename U>
__forceinline__ __device__ U BlockReduceSum(U val, U *shared) {
  int lane = threadIdx.x % warpSize;
  int wid = threadIdx.x / warpSize;

  val = WarpReduceSum(val);          // Each warp performs partial reduction
  if (lane == 0) shared[wid] = val;  // Write reduced value to shared memory
  __syncthreads();                   // Wait for all partial reductions
  // read from shared memory only if that warp existed
  val =
      (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : static_cast<U>(0);

  if (wid == 0) val = WarpReduceSum(val);  // Final reduce within first warp

  return val;
}

#define FIXED_BLOCK_DIM_CASE_BASE(log2_block_dim, ...)  \
  case (1 << (log2_block_dim)): {                       \
    constexpr auto kBlockDim = (1 << (log2_block_dim)); \
    __VA_ARGS__;                                        \
  } break

#define FIXED_BLOCK_DIM_CASE(...)              \
  FIXED_BLOCK_DIM_CASE_BASE(9, ##__VA_ARGS__); \
  FIXED_BLOCK_DIM_CASE_BASE(8, ##__VA_ARGS__); \
  FIXED_BLOCK_DIM_CASE_BASE(7, ##__VA_ARGS__); \
  FIXED_BLOCK_DIM_CASE_BASE(6, ##__VA_ARGS__); \
  FIXED_BLOCK_DIM_CASE_BASE(5, ##__VA_ARGS__); \
  FIXED_BLOCK_DIM_CASE_BASE(4, ##__VA_ARGS__); \
  FIXED_BLOCK_DIM_CASE_BASE(3, ##__VA_ARGS__); \
  FIXED_BLOCK_DIM_CASE_BASE(2, ##__VA_ARGS__); \
  FIXED_BLOCK_DIM_CASE_BASE(1, ##__VA_ARGS__)

#define FIXED_BLOCK_DIM_FIXED_BLOCK_NUM_CASE_BASE(                          \
    log2_block_dim, feature_size, kMaxBlockNum, ...)                        \
  case (1 << (log2_block_dim)): {                                           \
    for (int64_t i = 0; i < std::ceil(feature_size / (1.0 * kMaxBlockNum)); \
         i++) {                                                             \
      int64_t col_offset = i * static_cast<int64_t>(kMaxBlockNum);          \
      int block_num = static_cast<int>(std::min(                            \
          feature_size - col_offset, static_cast<int64_t>(kMaxBlockNum)));  \
      constexpr auto kBlockDim = (1 << (log2_block_dim));                   \
      __VA_ARGS__;                                                          \
    }                                                                       \
  } break

#define FIXED_BLOCK_DIM_FIXED_BLOCK_NUM_CASE(feature_size, kMaxBlockNum, ...) \
  FIXED_BLOCK_DIM_FIXED_BLOCK_NUM_CASE_BASE(                                  \
      9, feature_size, kMaxBlockNum, ##__VA_ARGS__);                          \
  FIXED_BLOCK_DIM_FIXED_BLOCK_NUM_CASE_BASE(                                  \
      8, feature_size, kMaxBlockNum, ##__VA_ARGS__);                          \
  FIXED_BLOCK_DIM_FIXED_BLOCK_NUM_CASE_BASE(                                  \
      7, feature_size, kMaxBlockNum, ##__VA_ARGS__);                          \
  FIXED_BLOCK_DIM_FIXED_BLOCK_NUM_CASE_BASE(                                  \
      6, feature_size, kMaxBlockNum, ##__VA_ARGS__);                          \
  FIXED_BLOCK_DIM_FIXED_BLOCK_NUM_CASE_BASE(                                  \
      5, feature_size, kMaxBlockNum, ##__VA_ARGS__);                          \
  FIXED_BLOCK_DIM_FIXED_BLOCK_NUM_CASE_BASE(                                  \
      4, feature_size, kMaxBlockNum, ##__VA_ARGS__);                          \
  FIXED_BLOCK_DIM_FIXED_BLOCK_NUM_CASE_BASE(                                  \
      3, feature_size, kMaxBlockNum, ##__VA_ARGS__);                          \
  FIXED_BLOCK_DIM_FIXED_BLOCK_NUM_CASE_BASE(                                  \
      2, feature_size, kMaxBlockNum, ##__VA_ARGS__);                          \
  FIXED_BLOCK_DIM_FIXED_BLOCK_NUM_CASE_BASE(                                  \
      1, feature_size, kMaxBlockNum, ##__VA_ARGS__)

static __device__ __forceinline__ float real_sqrt(float x) { return sqrtf(x); }
static __device__ __forceinline__ double real_sqrt(double x) {
  return ::sqrt(x);
}

template <typename T>
struct PairForLayerNorm {
  __device__ __forceinline__ PairForLayerNorm() {}
  __device__ __forceinline__ PairForLayerNorm(const T &first, const T &second)
      : first_(first), second_(second) {}

  T first_;
  T second_;
};

template <typename T>
struct PairForLayerNormAddFunctor {
  __device__ __forceinline__ PairForLayerNorm<T> operator()(
      const PairForLayerNorm<T> &p1, const PairForLayerNorm<T> &p2) {
    return PairForLayerNorm<T>(p1.first_ + p2.first_, p1.second_ + p2.second_);
  }
};

template <typename T>
__inline__ __device__ T rsqrt_(const T val) {
  return static_cast<T>(1) / sqrt(val);
}

template <>
__inline__ __device__ float rsqrt_(const float val) {
  return rsqrtf(val);
}

template <>
__inline__ __device__ double rsqrt_(const double val) {
  return ::rsqrt(val);
}

#if CUDA_ARCH_FP16_SUPPORTED(__CUDA_ARCH__)
template <>
__inline__ __device__ half rsqrt_(const half val) {
  return hrsqrt(val);
}
#endif

#ifdef PADDLE_WITH_CUDA
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
__global__ __launch_bounds__(THREADS_PER_CTA) void fast_ln_fwd_kernel(
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
    phi::Load<ScaleT, VecSize>(gamma_ptr + col * VecSize, &gamma[it]);
    phi::Load<ScaleT, VecSize>(beta_ptr + col * VecSize, &beta[it]);
    col += THREADS_PER_ROW;
  }

  constexpr U rn = 1.f / U(ELTS_PER_ROW);
  for (int row = r; row < rows; row += gridDim.x * ROWS_PER_CTA) {
    Vec x[LDGS];
#pragma unroll
    for (int it = 0, col = c; it < LDGS; it++) {
      phi::Load<T, VecSize>(x_ptr + row * ELTS_PER_ROW + col * VecSize, &x[it]);
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
      mu_local += __shfl_xor_sync(uint32_t(-1), mu_local, it);
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
        smem[warp_m] = mu_local;
      }
      __syncthreads();
      mu_local = smem[warp_m];
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
      var_local += __shfl_xor_sync(uint32_t(-1), var_local, it);
    }

    if (WARPS_N > 1) {
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
        smem[warp_m] = var_local;
      }
      __syncthreads();
      var_local = smem[warp_m];
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
      phi::Store<T, VecSize>(x[it], y_ptr + row * ELTS_PER_ROW + col * VecSize);
      col += THREADS_PER_ROW;
    }
  }
}
#endif

template <typename T>
inline HOSTDEVICE T roundWithTiesToEven(T x) {
  T xLower = floor(x);
  T xUpper = ceil(x);
  // x is in interval [xl,xu]. Choose closest of two bounds, breaking ties to
  // even.
  T dLower = x - xLower;
  T dUpper = xUpper - x;
  return static_cast<T>(
      (dLower == dUpper ? fmod(xLower, 2.0F) == 0.0F : dLower < dUpper)
          ? xLower
          : xUpper);
}

template <typename T>
__forceinline__ __device__ int8_t quant_helper(const T input,
                                               const float scale,
                                               const int round_type,
                                               const float max_bound,
                                               const float min_bound) {
  float quant_value = max_bound * scale * static_cast<float>(input);

  if (round_type == 0) {
    quant_value = static_cast<float>(roundWithTiesToEven(quant_value));
  } else {
    quant_value = static_cast<float>(round(quant_value));
  }
  quant_value = quant_value > max_bound ? max_bound : quant_value;
  quant_value = quant_value < min_bound ? min_bound : quant_value;
  return static_cast<int8_t>(quant_value);
}

template <typename T, typename U, bool ScaleBiasWithSameTypeX>
using LayerNormScaleBiasT =
    typename std::conditional<ScaleBiasWithSameTypeX, T, U>::type;

template <typename T,
          typename U,
          int BlockDim,
          bool ScaleBiasWithSameTypeX = false,
          typename InType = T,
          typename OutType = T>
__global__ void LayerNormForward(
    const InType *x,
    const LayerNormScaleBiasT<T, U, ScaleBiasWithSameTypeX> *scale,
    const LayerNormScaleBiasT<T, U, ScaleBiasWithSameTypeX> *bias,
    OutType *y,
    U *mean,
    U *var,
    float epsilon,
    int64_t feature_size,
    const float *dequant_out_scale_data = nullptr,
    const int quant_out_scale_offset = 0,
    const float quant_in_scale = 1.0,
    const int quant_round_type = 1,
    const float quant_max_bound = 127.0,
    const float quant_min_bound = -127.0) {
  __shared__ U mean_share;
  __shared__ U var_share;
  __shared__ U shared_mean[32];  // threadIdx.x / warpSize <= kMaxBlockDim /
                                 // warpSize <= 1024/32 = 32;
  __shared__ U shared_var[32];

  int64_t beg_idx = blockIdx.x * feature_size + threadIdx.x;
  int64_t end_idx = (blockIdx.x + 1) * feature_size;

  // Step 1: Reduce to calculate mean and var
  U mean_val = 0;
  U var_val = 0;
  for (int64_t i = beg_idx; i < end_idx; i += BlockDim) {
    U tmp = static_cast<U>(x[i]);
    mean_val += tmp;
    var_val += (tmp * tmp);
  }

  mean_val = BlockReduceSum<U>(mean_val, shared_mean);
  var_val = BlockReduceSum<U>(var_val, shared_var);

  if (threadIdx.x == 0) {
    auto scale = static_cast<U>(static_cast<float>(1.) /
                                static_cast<float>(feature_size));
    auto tmp = mean_val * scale;
    mean[blockIdx.x] = mean_share = static_cast<U>(tmp);
    var_share = static_cast<U>(var_val * scale - mean_share * mean_share);
    var_share = var_share > U(0) ? var_share : U(0);
    var[blockIdx.x] = var_share;
  }
  __syncthreads();

  mean_val = mean_share;
  U invvar = rsqrt_<U>(var_share + static_cast<U>(epsilon));

  // Step 2: Calculate y
  if (scale != nullptr) {
    if (bias != nullptr) {
      for (int64_t i = beg_idx, j = threadIdx.x; i < end_idx;
           i += BlockDim, j += BlockDim) {
        if (std::is_same<OutType, int8_t>::value) {
          y[i] = quant_helper(
              static_cast<T>(static_cast<U>(scale[j]) *
                                 (static_cast<U>(x[i]) - mean_val) * invvar +
                             static_cast<U>(bias[j])),
              quant_in_scale,
              quant_round_type,
              quant_max_bound,
              quant_min_bound);
        } else {
          y[i] = static_cast<OutType>(static_cast<U>(scale[j]) *
                                          (static_cast<U>(x[i]) - mean_val) *
                                          invvar +
                                      static_cast<U>(bias[j]));
        }
      }
    } else {
      for (int64_t i = beg_idx, j = threadIdx.x; i < end_idx;
           i += BlockDim, j += BlockDim) {
        if (std::is_same<OutType, int8_t>::value) {
          y[i] = quant_helper(
              static_cast<T>(static_cast<U>(scale[j]) *
                             (static_cast<U>(x[i]) - mean_val) * invvar),
              quant_in_scale,
              quant_round_type,
              quant_max_bound,
              quant_min_bound);
        } else {
          y[i] =
              static_cast<OutType>(static_cast<U>(scale[j]) *
                                   (static_cast<U>(x[i]) - mean_val) * invvar);
        }
      }
    }
  } else {  // scale == nullptr
    if (bias != nullptr) {
      for (int64_t i = beg_idx, j = threadIdx.x; i < end_idx;
           i += BlockDim, j += BlockDim) {
        if (std::is_same<OutType, int8_t>::value) {
          y[i] = quant_helper(
              static_cast<T>((static_cast<U>(x[i]) - mean_val) * invvar +
                             static_cast<U>(bias[j])),
              quant_in_scale,
              quant_round_type,
              quant_max_bound,
              quant_min_bound);
        } else {
          y[i] =
              static_cast<OutType>((static_cast<U>(x[i]) - mean_val) * invvar +
                                   static_cast<U>(bias[j]));
        }
      }
    } else {
      for (int64_t i = beg_idx, j = threadIdx.x; i < end_idx;
           i += BlockDim, j += BlockDim) {
        if (std::is_same<OutType, int8_t>::value) {
          y[i] = quant_helper(
              static_cast<T>((static_cast<U>(x[i]) - mean_val) * invvar),
              quant_in_scale,
              quant_round_type,
              quant_max_bound,
              quant_min_bound);
        } else {
          y[i] =
              static_cast<OutType>((static_cast<U>(x[i]) - mean_val) * invvar);
        }
      }
    }
  }
}

template <typename T, typename U, int VPT>
__inline__ __device__ void cuLoadAddStridedInputs(const int64_t i1_block,
                                                  const int thr_load_row_off,
                                                  const int thr_load_col_off,
                                                  const int i2_off,
                                                  const int row_stride,
                                                  U *warp_buf1,
                                                  U *warp_buf2,
                                                  const T *__restrict__ input,
                                                  const T *__restrict__ dout,
                                                  const int64_t i1_end,
                                                  const int64_t n2,
                                                  const U *__restrict__ mean,
                                                  const U *__restrict__ var,
                                                  const float epsilon) {
  const int64_t i1 = i1_block + thr_load_row_off;
  if (i1 >= i1_end) return;
  U curr_mean = mean[i1];
  U curr_invvar = rsqrt_<U>(var[i1] + epsilon);
#pragma unroll
  for (int k = 0; k < VPT; ++k) {
    const int i2 = i2_off + k;
    const int64_t load_idx = i1 * n2 + i2;
    const int write_idx = thr_load_row_off * row_stride + thr_load_col_off + k;
    if (i2 < n2) {
      U curr_input = static_cast<U>(input[load_idx]);
      U curr_dout = static_cast<U>(dout[load_idx]);
      warp_buf1[write_idx] += curr_dout;
      warp_buf2[write_idx] +=
          curr_dout * (curr_input - curr_mean) * curr_invvar;
    }
  }
}

#ifdef PADDLE_WITH_CUDA
template <bool IsFusedDropoutResidualLn,
          bool NeedDDropoutSrcPtr,
          typename T,
          typename U,
          typename ScaleT = U,
          typename MaskType = uint8_t,
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
__global__ __launch_bounds__(THREADS_PER_CTA) void fused_ln_bwd_fast_kernel(
    const int rows,
    float epsilon,
    const T *__restrict__ x_ptr,
    const ScaleT *__restrict__ gamma_ptr,
    const U *__restrict__ mean_ptr,
    const U *__restrict__ var_ptr,
    const T *__restrict__ dout_ptr,
    U *__restrict__ dgamma_temp_ptr,
    U *__restrict__ dbeta_temp_ptr,
    T *__restrict__ dx_ptr,
    const MaskType *mask_ptr = nullptr,
    T factor = static_cast<T>(0),
    T *d_dropout_src_ptr = nullptr) {
  static_assert(
      !IsFusedDropoutResidualLn || NeedDDropoutSrcPtr,
      "When IsFusedDropoutResidualLn = true, NeedDDropoutSrcPtr must be true.");

  using Vec = phi::AlignedVector<T, VecSize>;
  using Vec_scale = phi::AlignedVector<ScaleT, VecSize>;
  using MaskLoadT = phi::AlignedVector<MaskType, VecSize>;

  const int tidx = threadIdx.x;
  const int bidx = blockIdx.x;
  const int lane = tidx % THREADS_PER_WARP;            // 0, 1, ..., 31
  const int warp = tidx / THREADS_PER_WARP;            // 0, 1, 2, 3
  const int warp_m = warp / WARPS_N;                   // 0, 1, 2, 3
  const int warp_n = warp % WARPS_N;                   // 0
  const int tid_r = warp_n * THREADS_PER_WARP + lane;  // 0, 1, ..., 31

  const int r = bidx * ROWS_PER_CTA + warp_m;
  const int c = warp_n * THREADS_PER_WARP + lane;

  static_assert(ELTS_PER_ROW == THREADS_PER_ROW * LDGS * VecSize, "");

  // smem for column reduction
  __shared__ U smem_[ROWS_PER_CTA * ELTS_PER_ROW];

  U dgamma_sum[LDGS * VecSize];
  U dbeta_sum[LDGS * VecSize];

  memset(dgamma_sum, 0, sizeof(U) * LDGS * VecSize);
  memset(dbeta_sum, 0, sizeof(U) * LDGS * VecSize);

  // Note: it is no use for WARP_N = 1
  __shared__ U smem_sum_loss1[ROWS_PER_CTA * WARPS_N];  // 4
  __shared__ U smem_sum_loss2[ROWS_PER_CTA * WARPS_N];  // 4
  U *sum_loss1_shared = &smem_sum_loss1[warp_m * WARPS_N];
  U *sum_loss2_shared = &smem_sum_loss2[warp_m * WARPS_N];

  // step-1: compute dx and local results of dscale and dbias
  constexpr float rn = 1.f / static_cast<float>(ELTS_PER_ROW);
  Vec_scale gamma[LDGS];
  int col = c;
#pragma unroll
  for (int it = 0; it < LDGS; it++) {
    phi::Load<ScaleT, VecSize>(gamma_ptr + col * VecSize, &gamma[it]);
    col += THREADS_PER_ROW;
  }

#pragma unroll 1
  for (int row = r; row < rows; row += gridDim.x * ROWS_PER_CTA) {
    const U mean_cur_row = mean_ptr[row];
    const U var_cur_row = rsqrt_<U>(var_ptr[row] + epsilon);
    Vec dout[LDGS], x[LDGS];
    MaskLoadT mask_vec[LDGS];
    int col = c;
#pragma unroll
    for (int it = 0; it < LDGS; it++) {
      phi::Load<T, VecSize>(dout_ptr + row * ELTS_PER_ROW + col * VecSize,
                            &dout[it]);
      phi::Load<T, VecSize>(x_ptr + row * ELTS_PER_ROW + col * VecSize, &x[it]);
      if (IsFusedDropoutResidualLn) {
        phi::Load<MaskType, VecSize>(
            mask_ptr + row * ELTS_PER_ROW + col * VecSize, &mask_vec[it]);
      }

      col += THREADS_PER_ROW;
    }

    // local reductions
    U dy[LDGS * VecSize];
    U y[LDGS * VecSize];

    U sum_loss1 = 0.f;
    U sum_loss2 = 0.f;
#pragma unroll
    for (int it = 0; it < LDGS; it++) {
#pragma unroll
      for (int jt = 0; jt < VecSize; jt++) {
        U x_tmp = static_cast<U>(x[it][jt]);
        U y_tmp = var_cur_row * (x_tmp - mean_cur_row);
        U dy_tmp = static_cast<U>(gamma[it][jt]) *
                   static_cast<U>(dout[it][jt]);    // scale * dy
        U dout_tmp = static_cast<U>(dout[it][jt]);  // dy

        // used for get dx (row reduction)
        sum_loss1 += dy_tmp;          // scale * dy, sum_1
        sum_loss2 += dy_tmp * y_tmp;  // scale * dy * y, sum_2

        dy[it * VecSize + jt] = dy_tmp;  // scale * dy
        y[it * VecSize + jt] = y_tmp;    // y

        // used for get dscale and dbias (column reduction)
        dgamma_sum[it * VecSize + jt] += dout_tmp * y_tmp;  // dy * y
        dbeta_sum[it * VecSize + jt] += dout_tmp;           // dy
      }
    }

    // reduction across row for sum_loss1, sum_loss2
    if (WARPS_N == 1) {
#pragma unroll
      // row reduction among 32 threads.
      for (int it = 1; it < THREADS_PER_WARP; it *= 2) {
        sum_loss1 += __shfl_xor_sync(uint32_t(-1), sum_loss1, it);
        sum_loss2 += __shfl_xor_sync(uint32_t(-1), sum_loss2, it);
      }
      sum_loss1 *= rn;
      sum_loss2 *= rn;
    } else {
#pragma unroll
      for (int it = 16; it > 0; it /= 2) {
        sum_loss1 += __shfl_down_sync(uint32_t(-1), sum_loss1, it);
        sum_loss2 += __shfl_down_sync(uint32_t(-1), sum_loss2, it);
      }

      if (lane == 0) {
        sum_loss1_shared[warp_n] = sum_loss1;
        sum_loss2_shared[warp_n] = sum_loss2;
      }

      __syncthreads();
      if (warp_n == 0 && lane == 0) {
        sum_loss1 = 0.f;
        sum_loss2 = 0.f;
        for (int it = 0; it < WARPS_N; it++) {
          sum_loss1 += sum_loss1_shared[it];
          sum_loss2 += sum_loss2_shared[it];
        }
        sum_loss1_shared[0] = sum_loss1;
        sum_loss2_shared[0] = sum_loss2;
      }
      __syncthreads();

      sum_loss1 = sum_loss1_shared[0] * rn;
      sum_loss2 = sum_loss2_shared[0] * rn;
    }

#pragma unroll
    for (int it = 0; it < LDGS; it++) {
#pragma unroll
      for (int jt = 0; jt < VecSize; jt++) {
        U dy_tmp = dy[it * VecSize + jt];  // scale * dy
        U y_tmp = y[it * VecSize + jt];    // y
        // dx = var * (scale * dy - sum_loss2 * y - sum_loss1)
        U dx_tmp = var_cur_row * (dy_tmp - sum_loss2 * y_tmp - sum_loss1);
        // Note: reuse x and dout vec register to store dx and d_dropout_src.
        x[it][jt] = static_cast<T>(dx_tmp);
        if (IsFusedDropoutResidualLn) {
          dout[it][jt] = x[it][jt] * static_cast<T>(mask_vec[it][jt]) * factor;
        }
      }
    }

    // store dx to global memory
    col = c;
#pragma unroll
    for (int it = 0; it < LDGS; it++) {
      phi::Store<T, VecSize>(x[it],
                             dx_ptr + row * ELTS_PER_ROW + col * VecSize);
      if (IsFusedDropoutResidualLn) {
        phi::Store<T, VecSize>(
            dout[it], d_dropout_src_ptr + row * ELTS_PER_ROW + col * VecSize);
      } else if (NeedDDropoutSrcPtr) {
        phi::Store<T, VecSize>(
            x[it], d_dropout_src_ptr + row * ELTS_PER_ROW + col * VecSize);
      }
      col += THREADS_PER_ROW;
    }
  }

  // step-2: column reduction of dscale and dbias for each thread block.
  // each block's sum: [4 * 1024] -> [1 * 1024]
  enum { NUM_RES = ELTS_PER_ROW / THREADS_PER_CTA };  // 1024/128 = 8
  static_assert(NUM_RES * THREADS_PER_CTA == ELTS_PER_ROW, "");

  U *smem_write;

  smem_write = &smem_[warp_m * ELTS_PER_ROW + tid_r * VecSize];  // [4 * 1024]
#pragma unroll
  for (int it = 0; it < LDGS; it++) {
#pragma unroll
    for (int jt = 0; jt < VecSize; jt++) {
      smem_write[jt] = dbeta_sum[it * VecSize + jt];
    }
    smem_write += THREADS_PER_ROW * VecSize;  // 32*8
  }
  __syncthreads();
  U cta_dbeta_sum[NUM_RES];
  memset(cta_dbeta_sum, 0, sizeof(U) * NUM_RES);
  // column reduction for elems in smem: 4*1024 -> 1*1024.
  for (int it = 0; it < ROWS_PER_CTA; it++) {
    for (int jt = 0; jt < NUM_RES; jt++) {
      cta_dbeta_sum[jt] +=
          smem_[it * ELTS_PER_ROW + tidx + jt * THREADS_PER_CTA];
    }
  }
  __syncthreads();

  smem_write = &smem_[warp_m * ELTS_PER_ROW + tid_r * VecSize];
#pragma unroll
  for (int it = 0; it < LDGS; it++) {
#pragma unroll
    for (int jt = 0; jt < VecSize; jt++) {
      smem_write[jt] = dgamma_sum[it * VecSize + jt];
    }
    smem_write += THREADS_PER_ROW * VecSize;
  }
  __syncthreads();
  U cta_dgamma_sum[NUM_RES];
  memset(cta_dgamma_sum, 0, sizeof(U) * NUM_RES);
  for (int it = 0; it < ROWS_PER_CTA; it++) {
    for (int jt = 0; jt < NUM_RES; jt++) {
      cta_dgamma_sum[jt] +=
          smem_[it * ELTS_PER_ROW + tidx + jt * THREADS_PER_CTA];
    }
  }

  // the shape of results：(#blocks, 1024)
  U *dgamma_part =
      static_cast<U *>(dgamma_temp_ptr) + bidx * ELTS_PER_ROW + tidx;
  for (int jt = 0; jt < NUM_RES; jt++) {
    *dgamma_part = cta_dgamma_sum[jt];
    dgamma_part += THREADS_PER_CTA;
  }

  U *dbeta_part = static_cast<U *>(dbeta_temp_ptr) + bidx * ELTS_PER_ROW + tidx;
  for (int jt = 0; jt < NUM_RES; jt++) {
    *dbeta_part = cta_dbeta_sum[jt];
    dbeta_part += THREADS_PER_CTA;
  }
}

/* This function carry out column reduction whose input is [rows, 1024] and
 * output is [1, 1024].
 * #blocks: 32
 * #threads: 512
 */
// todo(@limin29): to think if there are better impl strategies
template <typename U,
          typename ScaleT = U,
          int VecSize = 1,
          int WARPS_M = 16,
          int WARPS_N = 1,
          int BYTES_PER_LDG = 4,
          int ELTS_PER_ROW = 1024,
          int THREADS_PER_WARP = 32,
          int THREADS_PER_ROW = WARPS_N *THREADS_PER_WARP,
          int THREADS_PER_CTA = WARPS_M *THREADS_PER_ROW,
          int ROWS_PER_CTA = WARPS_M,
          int ELTS_PER_ROW_PER_CTA = THREADS_PER_ROW *VecSize,
          int LDGS = ELTS_PER_ROW / ELTS_PER_ROW_PER_CTA,
          int VEC_COLS = ELTS_PER_ROW / VecSize>
__global__ __launch_bounds__(THREADS_PER_CTA) void ln_bwd_fast_final_kernel(
    const int rows,
    U *__restrict__ dg_part_,
    U *__restrict__ db_part_,
    ScaleT *__restrict__ dg_,
    ScaleT *__restrict__ db_) {
  using Vec = phi::AlignedVector<U, VecSize>;
  static_assert(VEC_COLS == ELTS_PER_ROW / VecSize, "");

  const int tidx = threadIdx.x;
  const int bidx = blockIdx.x;
  const int lane = tidx % THREADS_PER_WARP;
  const int warp = tidx / THREADS_PER_WARP;
  const int warp_m = warp / WARPS_N;
  const int warp_n = warp % WARPS_N;
  const int tid_c = warp_n * THREADS_PER_WARP + lane;

  const int c = bidx * THREADS_PER_ROW + tid_c;
  const int r = warp_m;

  __shared__ U smem_space[(WARPS_M - 1) * THREADS_PER_ROW * VecSize];

  for (int col = c; col < VEC_COLS; col += gridDim.x * THREADS_PER_ROW) {
    const U *dg_part_ptr = (dg_part_) + r * ELTS_PER_ROW + col * VecSize;
    const U *db_part_ptr = (db_part_) + r * ELTS_PER_ROW + col * VecSize;

    U dg_sum[VecSize];
    U db_sum[VecSize];
    memset(dg_sum, 0, sizeof(U) * VecSize);
    memset(db_sum, 0, sizeof(U) * VecSize);
#pragma unroll
    for (int row = r; row < rows; row += ROWS_PER_CTA) {
      Vec dg;
      Vec db;
      phi::Load<U, VecSize>(dg_part_ptr, &dg);
      phi::Load<U, VecSize>(db_part_ptr, &db);
      dg_part_ptr += ROWS_PER_CTA * ELTS_PER_ROW;
      db_part_ptr += ROWS_PER_CTA * ELTS_PER_ROW;

#pragma unroll
      for (int jt = 0; jt < VecSize; jt++) {
        dg_sum[jt] += dg[jt];
        db_sum[jt] += db[jt];
      }
    }

    // reduction across rows of the thread block
    U *smem_write;
    smem_write = smem_space + (warp_m - 1) * THREADS_PER_ROW * VecSize + tid_c;

    if (warp_m > 0) {
#pragma unroll
      for (int jt = 0; jt < VecSize; jt++) {
        *smem_write = dg_sum[jt];
        smem_write += THREADS_PER_ROW;
      }
    }
    __syncthreads();

    U *smem_read;
    smem_read = smem_space + tid_c;
    if (warp_m == 0) {
#pragma unroll
      for (int it = 0; it < WARPS_M - 1; it++) {
#pragma unroll
        for (int jt = 0; jt < VecSize; jt++) {
          dg_sum[jt] += *smem_read;
          smem_read += THREADS_PER_ROW;
        }
      }
    }

    __syncthreads();

    smem_write = smem_space + (warp_m - 1) * THREADS_PER_ROW * VecSize + tid_c;

    if (warp_m > 0) {
#pragma unroll
      for (int jt = 0; jt < VecSize; jt++) {
        *smem_write = db_sum[jt];
        smem_write += THREADS_PER_ROW;
      }
    }
    __syncthreads();

    smem_read = smem_space + tid_c;
    if (warp_m == 0) {
#pragma unroll
      for (int it = 0; it < WARPS_M - 1; it++) {
#pragma unroll
        for (int jt = 0; jt < VecSize; jt++) {
          db_sum[jt] += *smem_read;
          smem_read += THREADS_PER_ROW;
        }
      }

      union {
        ScaleT raw;
        ScaleT elt[VecSize];
      } dg_out, db_out;

#pragma unroll
      for (int jt = 0; jt < VecSize; jt++) {
        dg_out.elt[jt] = dg_sum[jt];
        db_out.elt[jt] = db_sum[jt];
      }
      ScaleT *dg_ptr = reinterpret_cast<ScaleT *>(dg_) + col;
      ScaleT *db_ptr = reinterpret_cast<ScaleT *>(db_) + col;
      *dg_ptr = dg_out.raw;
      *db_ptr = db_out.raw;
    }
  }
}

/* This function support two kinds of computations (only for float and fp16
 * type):
 *
 * Case-1: compute layer_norm_grad for layernorm op by setting mask_ptr and
 * d_dropout_src_ptr to nullptr. Here, d_x_ptr returns the grad of layernorm
 * input.
 *
 * Case-2: compute layer_norm_grad + residual_grad + dropout_grad for
 * fused_dropout_residual_layernorm op. Here, dx_ptr returns residual_grad.
 *
 */
template <typename T,
          typename U,
          typename ScaleT = U,
          typename MaskType = uint8_t>
void ln_bwd_fast_kernel_driver(const phi::GPUContext &dev_ctx,
                               const int rows,
                               const int cols,
                               float epsilon,
                               const T *x_ptr,
                               const ScaleT *scale_ptr,
                               const U *mean_ptr,
                               const U *var_ptr,
                               const T *dout_ptr,
                               T *dx_ptr,
                               ScaleT *dscale_ptr,
                               ScaleT *dbias_ptr,
                               const MaskType *mask_ptr = nullptr,
                               T factor = static_cast<T>(0),
                               T *d_dropout_src_ptr = nullptr) {
  auto stream = dev_ctx.stream();
  if (cols == 1024 || cols == 384 || cols == 256) {
    // step-1: compute dx and reduced part results of dscale and dbias.
    const int WARPS_M = 4;  // how many rows delt in a cta.
    const int WARPS_N = 1;  // how many warps to deal with a row.
    const int BYTES_PER_LDG = 16;
    const int VecSize = BYTES_PER_LDG / sizeof(T);

    const int THREADS_PER_WARP = 32;
    const int THREADS_PER_ROW = WARPS_N * THREADS_PER_WARP;
    const int THREADS_PER_CTA = WARPS_M * THREADS_PER_ROW;
    const int ROWS_PER_CTA = WARPS_M;

    // 4 * 1024 * 4
    const int SMEM_BYTES = ROWS_PER_CTA * cols * sizeof(U);

    // #blocks = 2 * #SM
    const int gridx = 2 * dev_ctx.GetSMCount();

    // get temp space for dscale and dbias.
    phi::DenseTensor dscale_temp;
    dscale_temp.Resize({gridx, cols});
    dev_ctx.template Alloc<U>(&dscale_temp);
    U *dscale_temp_ptr = dscale_temp.data<U>();

    phi::DenseTensor dbias_temp;
    dbias_temp.Resize({gridx, cols});
    dev_ctx.template Alloc<U>(&dbias_temp);
    U *dbias_temp_ptr = dbias_temp.data<U>();

    if (mask_ptr != nullptr) {
      if (d_dropout_src_ptr == nullptr) {
        PADDLE_THROW(phi::errors::InvalidArgument(
            "To compute fused_dropout_residual_ln grad, d_dropout_src_ptr "
            "can't be null"));
      }
#define LAUNCH_MASK_FUSED_LN_BWD_FAST_KERNEL(vec_size, ele_per_row) \
  fused_ln_bwd_fast_kernel<true,                                    \
                           true,                                    \
                           T,                                       \
                           U,                                       \
                           ScaleT,                                  \
                           MaskType,                                \
                           vec_size,                                \
                           WARPS_M,                                 \
                           WARPS_N,                                 \
                           BYTES_PER_LDG,                           \
                           ele_per_row>                             \
      <<<gridx, THREADS_PER_CTA, 0, stream>>>(rows,                 \
                                              epsilon,              \
                                              x_ptr,                \
                                              scale_ptr,            \
                                              mean_ptr,             \
                                              var_ptr,              \
                                              dout_ptr,             \
                                              dscale_temp_ptr,      \
                                              dbias_temp_ptr,       \
                                              dx_ptr,               \
                                              mask_ptr,             \
                                              factor,               \
                                              d_dropout_src_ptr);

      if (cols == 1024) {
        LAUNCH_MASK_FUSED_LN_BWD_FAST_KERNEL(VecSize, 1024);
      } else {
        switch (cols) {
          case 384:
            LAUNCH_MASK_FUSED_LN_BWD_FAST_KERNEL(1, 384);
            break;
          case 256:
            LAUNCH_MASK_FUSED_LN_BWD_FAST_KERNEL(VecSize, 256);
            break;
        }
      }
#undef LAUNCH_MASK_FUSED_LN_BWD_FAST_KERNEL

    } else {
#define LAUNCH_FUSED_LN_BWD_FAST_KERNEL_BASE(                  \
    vec_size, ele_per_row, need_d_dropout_src_ptr)             \
  fused_ln_bwd_fast_kernel<false,                              \
                           need_d_dropout_src_ptr,             \
                           T,                                  \
                           U,                                  \
                           ScaleT,                             \
                           MaskType,                           \
                           vec_size,                           \
                           WARPS_M,                            \
                           WARPS_N,                            \
                           BYTES_PER_LDG,                      \
                           ele_per_row>                        \
      <<<gridx, THREADS_PER_CTA, 0, stream>>>(rows,            \
                                              epsilon,         \
                                              x_ptr,           \
                                              scale_ptr,       \
                                              mean_ptr,        \
                                              var_ptr,         \
                                              dout_ptr,        \
                                              dscale_temp_ptr, \
                                              dbias_temp_ptr,  \
                                              dx_ptr,          \
                                              nullptr,         \
                                              factor,          \
                                              d_dropout_src_ptr);

#define LAUNCH_FUSED_LN_BWD_FAST_KERNEL(vec_size, ele_per_row)            \
  do {                                                                    \
    if (d_dropout_src_ptr != nullptr) {                                   \
      LAUNCH_FUSED_LN_BWD_FAST_KERNEL_BASE(vec_size, ele_per_row, true);  \
    } else {                                                              \
      LAUNCH_FUSED_LN_BWD_FAST_KERNEL_BASE(vec_size, ele_per_row, false); \
    }                                                                     \
  } while (0)

      if (cols == 1024) {
        LAUNCH_FUSED_LN_BWD_FAST_KERNEL(VecSize, 1024);
      } else {
        switch (cols) {
          case 384:
            LAUNCH_FUSED_LN_BWD_FAST_KERNEL(1, 384);
            break;
          case 256:
            LAUNCH_FUSED_LN_BWD_FAST_KERNEL(VecSize, 256);
            break;
        }
      }

#undef LAUNCH_FUSED_LN_BWD_FAST_KERNEL
    }

    const int WARPS_M_2 = 16;
    const int WARPS_N_2 = 1;
    const int BYTES_PER_LDG_2 = 4;
    const int VecSize_2 =
        std::max(1, static_cast<int>(BYTES_PER_LDG_2 / sizeof(U)));  // 1

    const int THREADS_PER_WARP_2 = 32;
    const int THREADS_PER_ROW_2 = WARPS_N_2 * THREADS_PER_WARP_2;  // 32
    const int THREADS_PER_CTA_2 =
        WARPS_M_2 * THREADS_PER_ROW_2;     // 16 * 32 = 512
    const int ROWS_PER_CTA_2 = WARPS_M_2;  // 16

    // #blocks: 32, #threads_per_block: 512
    // Note: it is not supported for double type.
    if (sizeof(U) > 4) {
      PADDLE_THROW(
          phi::errors::InvalidArgument("Only support float and fp16 type"));
    } else {
      int gridx_2 = 0;

#define LAUNCH_LN_BWD_BETA_GAMMMA_KERNEL(vec_size, ele_per_row)         \
  gridx_2 = static_cast<int>(std::ceil(                                 \
      ele_per_row / static_cast<float>(THREADS_PER_ROW_2 * vec_size))); \
  ln_bwd_fast_final_kernel<U,                                           \
                           ScaleT,                                      \
                           vec_size,                                    \
                           WARPS_M_2,                                   \
                           WARPS_N_2,                                   \
                           BYTES_PER_LDG_2,                             \
                           ele_per_row>                                 \
      <<<gridx_2, THREADS_PER_CTA_2, 0, stream>>>(                      \
          gridx, dscale_temp_ptr, dbias_temp_ptr, dscale_ptr, dbias_ptr);

      if (cols == 1024) {
        LAUNCH_LN_BWD_BETA_GAMMMA_KERNEL(VecSize_2, 1024);
      } else {
        switch (cols) {
          case 384:
            LAUNCH_LN_BWD_BETA_GAMMMA_KERNEL(1, 384);
            break;
          case 256:
            LAUNCH_LN_BWD_BETA_GAMMMA_KERNEL(VecSize_2, 256);
            break;
        }
      }

#undef LAUNCH_LN_BWD_BETA_GAMMMA_KERNEL
    }
  } else {
    PADDLE_THROW(phi::errors::InvalidArgument(
        "Fast layer_norm kernel is only used when feature_size is 1024"));
  }
}
#endif

template <typename T, typename U, int BDIMX, int BDIMY, int VPTX>
__global__ void LayerNormBackwardPartGradGammaBeta(const T *__restrict__ dout,
                                                   const T *__restrict__ input,
                                                   const int64_t n1,
                                                   const int64_t n2,
                                                   const U *__restrict__ mean,
                                                   const U *__restrict__ var,
                                                   float epsilon,
                                                   U *part_grad_gamma,
                                                   U *part_grad_beta) {
  // VPTX -> value per thread.x, BDIMX -> blockDim.x,
  // BDIMY -> blockDim.y, template for compile time optimizations.
  constexpr int RowStride = BDIMX + 1;
  constexpr int BLOCK_SIZE = BDIMX * BDIMY;
  constexpr int VPTX_MUL_BDIMY = VPTX * BDIMY;
  constexpr int SharedSize = (BLOCK_SIZE > 2 * VPTX_MUL_BDIMY * RowStride)
                                 ? BLOCK_SIZE
                                 : 2 * VPTX_MUL_BDIMY * RowStride;

  const int thr_load_col_off = (threadIdx.x * VPTX) & (BDIMX - 1);
  const int thr_load_row_off =
      (threadIdx.x * VPTX) / BDIMX + threadIdx.y * BDIMY;
  const int i2_off = blockIdx.x * BDIMX + thr_load_col_off;

  __shared__ U buf[SharedSize];
  U *warp_buf1 = reinterpret_cast<U *>(buf);
  U *warp_buf2 = warp_buf1 + VPTX_MUL_BDIMY * RowStride;

  for (int idx = threadIdx.y * BDIMX + threadIdx.x;
       idx < 2 * VPTX_MUL_BDIMY * RowStride;
       idx += BLOCK_SIZE) {
    buf[idx] = U(0);
  }
  __syncthreads();

  for (int64_t i1_block = blockIdx.y * BDIMY * VPTX; i1_block < n1;
       i1_block += VPTX_MUL_BDIMY * gridDim.y) {
    cuLoadAddStridedInputs<T, U, VPTX>(i1_block,
                                       thr_load_row_off,
                                       thr_load_col_off,
                                       i2_off,
                                       RowStride,
                                       warp_buf1,
                                       warp_buf2,
                                       input,
                                       dout,
                                       n1,
                                       n2,
                                       mean,
                                       var,
                                       epsilon);
  }
  __syncthreads();

  // inter-warp reductions, sum within each warp
  U acc1 = U(0);
  U acc2 = U(0);
#pragma unroll
  for (int k = 0; k < VPTX; ++k) {
    int row1 = threadIdx.y + k * VPTX;
    int idx1 = row1 * RowStride + threadIdx.x;
    acc1 += warp_buf1[idx1];
    acc2 += warp_buf2[idx1];
  }
  warp_buf1[threadIdx.y * RowStride + threadIdx.x] = acc1;
  warp_buf2[threadIdx.y * RowStride + threadIdx.x] = acc2;
  __syncthreads();

  // sum all warps
#pragma unroll
  for (int offset = VPTX >> 1; offset > 1; offset >>= 1) {
    if (threadIdx.y < offset) {
      int row1 = threadIdx.y;
      int row2 = threadIdx.y + offset;
      int idx1 = row1 * RowStride + threadIdx.x;
      int idx2 = row2 * RowStride + threadIdx.x;
      warp_buf1[idx1] += warp_buf1[idx2];
      warp_buf2[idx1] += warp_buf2[idx2];
    }
    __syncthreads();
  }
  int64_t i2 = blockIdx.x * BDIMX + threadIdx.x;
  if (threadIdx.y == 0 && i2 < n2) {
    int row1 = threadIdx.y;
    int row2 = threadIdx.y + 1;
    int idx1 = row1 * RowStride + threadIdx.x;
    int idx2 = row2 * RowStride + threadIdx.x;
    part_grad_beta[blockIdx.y * n2 + i2] = warp_buf1[idx1] + warp_buf1[idx2];
    part_grad_gamma[blockIdx.y * n2 + i2] = warp_buf2[idx1] + warp_buf2[idx2];
  }
}

template <typename T, typename U, int BDIMX, int BDIMY, typename ScaleT>
__global__ void LayerNormBackwardSumGradGammaBeta(const U *part_grad_gamma,
                                                  const U *part_grad_beta,
                                                  const int part_size,
                                                  const int n1,
                                                  const int n2,
                                                  ScaleT *grad_gamma,
                                                  ScaleT *grad_beta) {
  // sum partial gradients for gamma and beta
  __shared__ U buf[BDIMX * BDIMY];
  int64_t i2 = blockIdx.x * BDIMX + threadIdx.x;
  if (i2 < n2) {
    // each warp does sequential reductions until reduced part_size is num_warps
    int num_warp_reductions = part_size / BDIMY;
    U sum_gamma = U(0);
    U sum_beta = U(0);
    const U *part_grad_gamma_ptr =
        part_grad_gamma + threadIdx.y * num_warp_reductions * n2 + i2;
    const U *part_grad_beta_ptr =
        part_grad_beta + threadIdx.y * num_warp_reductions * n2 + i2;
    for (int warp_offset = 0; warp_offset < num_warp_reductions;
         ++warp_offset) {
      sum_gamma += part_grad_gamma_ptr[warp_offset * n2];
      sum_beta += part_grad_beta_ptr[warp_offset * n2];
    }
    // inter-warp reductions
    constexpr int nbsize3 = BDIMX * BDIMY / 2;
    for (int offset = BDIMY / 2; offset >= 1; offset /= 2) {
      // top half write to shared memory
      if (threadIdx.y >= offset && threadIdx.y < 2 * offset) {
        const int write_idx = (threadIdx.y - offset) * blockDim.x + threadIdx.x;
        buf[write_idx] = sum_gamma;
        buf[write_idx + nbsize3] = sum_beta;
      }
      __syncthreads();
      // bottom half sums
      if (threadIdx.y < offset) {
        const int read_idx = threadIdx.y * BDIMX + threadIdx.x;
        sum_gamma += buf[read_idx];
        sum_beta += buf[read_idx + nbsize3];
      }
      __syncthreads();
    }
    // write out fully summed gradients
    if (threadIdx.y == 0) {
      grad_gamma[i2] = static_cast<ScaleT>(sum_gamma);
      grad_beta[i2] = static_cast<ScaleT>(sum_beta);
    }
  }
}

template <typename T, typename U, int BDIMX, int BDIMY, typename ScaleT>
__global__ void LayerNormBackwardComputeGradInput(const T *__restrict__ dout,
                                                  const T *__restrict__ input,
                                                  const int n1,
                                                  const int n2,
                                                  const U *__restrict__ mean,
                                                  const U *__restrict__ var,
                                                  const float epsilon,
                                                  const ScaleT *gamma,
                                                  T *grad_input) {
#ifdef __HIPCC__
  for (int64_t i1 = hipBlockIdx_x; i1 < n1; i1 += hipGridDim_x) {
#else
  for (int64_t i1 = blockIdx.x; i1 < n1; i1 += gridDim.x) {
#endif
    U sum_loss1 = U(0);
    U sum_loss2 = U(0);
    const U c_mean = mean[i1];
    const U c_invvar = rsqrt_<U>(var[i1] + epsilon);
    const T *k_input = input + i1 * n2;
    const T *k_dout = dout + i1 * n2;
    constexpr int numx = BDIMX * BDIMY;
    const int thrx = threadIdx.x + threadIdx.y * BDIMX;
    if (gamma != NULL) {
      int l = 4 * thrx;
      for (; l + 3 < n2; l += 4 * numx) {
        for (int k = 0; k < 4; ++k) {
          const U c_h = static_cast<U>(k_input[l + k]);
          const U c_loss = static_cast<U>(k_dout[l + k]);
          sum_loss1 += c_loss * static_cast<U>(gamma[l + k]);
          sum_loss2 +=
              c_loss * static_cast<U>(gamma[l + k]) * (c_h - c_mean) * c_invvar;
        }
      }
      for (; l < n2; ++l) {
        const U c_h = static_cast<U>(k_input[l]);
        const U c_loss = static_cast<U>(k_dout[l]);
        sum_loss1 += c_loss * static_cast<U>(gamma[l]);
        sum_loss2 +=
            c_loss * static_cast<U>(gamma[l]) * (c_h - c_mean) * c_invvar;
      }
    } else {
      int l = 4 * thrx;
      for (; l + 3 < n2; l += 4 * numx) {
        for (int k = 0; k < 4; ++k) {
          const U c_h = static_cast<U>(k_input[l + k]);
          const U c_loss = static_cast<U>(k_dout[l + k]);
          sum_loss1 += c_loss;
          sum_loss2 += c_loss * (c_h - c_mean) * c_invvar;
        }
      }
      for (; l < n2; ++l) {
        const U c_h = static_cast<U>(k_input[l]);
        const U c_loss = static_cast<U>(k_dout[l]);
        sum_loss1 += c_loss;
        sum_loss2 += c_loss * (c_h - c_mean) * c_invvar;
      }
    }
    // intra-warp reductions
#pragma unroll
    for (int mask = BDIMX / 2; mask > 0; mask /= 2) {
#ifdef PADDLE_WITH_HIP
      // WARP_SHFL_XOR(sum_loss, mask);
      sum_loss1 += __shfl_xor(sum_loss1, mask, warpSize);
      sum_loss2 += __shfl_xor(sum_loss2, mask, warpSize);
#else
      // WARP_SHFL_XOR(sum_loss, mask);
      sum_loss1 += __shfl_xor_sync(0xffffffff, sum_loss1, mask, warpSize);
      sum_loss2 += __shfl_xor_sync(0xffffffff, sum_loss2, mask, warpSize);
#endif
    }
    // inter-warp reductions
    if (BDIMY > 1) {
      __shared__ U buf[BDIMX * BDIMY];
      for (int offset = BDIMY / 2; offset > 0; offset /= 2) {
        // upper half of warps write to shared
        if (threadIdx.y >= offset && threadIdx.y < 2 * offset) {
          const int wrt_i = (threadIdx.y - offset) * BDIMX + threadIdx.x;
          buf[2 * wrt_i] = sum_loss1;
          buf[2 * wrt_i + 1] = sum_loss2;
        }
        __syncthreads();
        // lower half merges
        if (threadIdx.y < offset) {
          const int read_i = threadIdx.y * blockDim.x + threadIdx.x;
          sum_loss1 += buf[2 * read_i];
          sum_loss2 += buf[2 * read_i + 1];
        }
        __syncthreads();
      }
      if (threadIdx.y == 0) {
        buf[2 * threadIdx.x] = sum_loss1;
        buf[2 * threadIdx.x + 1] = sum_loss2;
      }
      __syncthreads();
      if (threadIdx.y != 0) {
        sum_loss1 = buf[2 * threadIdx.x];
        sum_loss2 = buf[2 * threadIdx.x + 1];
      }
    }
    // all threads now have the two sums over l
    U fH = (U)n2;
    U term1 = (U(1) / fH) * c_invvar;
    T *k_grad_input = grad_input + i1 * n2;
    if (gamma != NULL) {
      for (int l = thrx; l < n2; l += numx) {
        const U c_h = static_cast<U>(k_input[l]);
        const U c_loss = static_cast<U>(k_dout[l]);
        U f_grad_input = fH * c_loss * static_cast<U>(gamma[l]);
        f_grad_input -= sum_loss1;
        f_grad_input -= (c_h - c_mean) * c_invvar * sum_loss2;
        f_grad_input *= term1;
        k_grad_input[l] = static_cast<T>(f_grad_input);
      }
    } else {
      for (int l = thrx; l < n2; l += numx) {
        const U c_h = static_cast<U>(k_input[l]);
        const U c_loss = static_cast<U>(k_dout[l]);
        U f_grad_input = fH * c_loss;
        f_grad_input -= sum_loss1;
        f_grad_input -= (c_h - c_mean) * c_invvar * sum_loss2;
        f_grad_input *= term1;
        k_grad_input[l] = static_cast<T>(f_grad_input);
      }
    }
  }
}

template <typename T, typename U, typename ScaleT, int DataPerTid>
__global__ void LayerNormBackwardComputeGradInputWithSmallFeatureSize(
    const T *__restrict__ dout,
    const T *__restrict__ input,
    const int n1,
    const int n2,
    const U *__restrict__ mean,
    const U *__restrict__ var,
    const float epsilon,
    const ScaleT *__restrict__ gamma,
    T *grad_input) {
  constexpr int WarpSize = 32;
#ifdef __HIPCC__
  for (int64_t bid = hipBlockIdx_x; bid < n1; bid += hipGridDim_x) {
#else
  for (int64_t bid = blockIdx.x; bid < n1; bid += gridDim.x) {
#endif
    U sum_loss1 = U(0);
    U sum_loss2 = U(0);
    const U c_mean = mean[bid];
    const U c_invvar = rsqrt_<U>(var[bid] + epsilon);

    const int main_vec_n2 = n2 / DataPerTid;
    const int tid_num = WarpSize * blockDim.y;
    const int thrx = threadIdx.x + threadIdx.y * WarpSize;

    // One feature-size per block.
    const T *__restrict__ k_dout = dout + bid * n2;
    const T *__restrict__ k_input = input + bid * n2;
    T *k_grad_input = grad_input + bid * n2;

    // Data storage location in local register.
    using VecT = phi::AlignedVector<T, DataPerTid>;
    using VecScaleT = phi::AlignedVector<ScaleT, DataPerTid>;

    const VecT *__restrict__ v_k_dout =
        reinterpret_cast<const VecT *__restrict__>(k_dout);
    const VecT *__restrict__ v_k_input =
        reinterpret_cast<const VecT *__restrict__>(k_input);
    const VecScaleT *__restrict__ v_gamma =
        reinterpret_cast<const VecScaleT *__restrict__>(gamma);
    VecT *v_grad = reinterpret_cast<VecT *>(k_grad_input);

    // Each thread shall deal with no more than 8 data.
    U dout_data[8];
    U input_data[8];
    U gamma_data[8];

    if (gamma != NULL) {
      int tid = thrx;
      for (int i = 0; tid < main_vec_n2; tid += tid_num, ++i) {
        VecT v_tmp_dout = v_k_dout[tid];
        VecT v_tmp_input = v_k_input[tid];
        VecScaleT v_tmp_gamma = v_gamma[tid];
#pragma unroll
        for (int k = 0; k < DataPerTid; ++k) {
          const int idx = k + i * DataPerTid;
          dout_data[idx] = static_cast<U>(v_tmp_dout[k]);
          input_data[idx] = static_cast<U>(v_tmp_input[k]);
          gamma_data[idx] = static_cast<U>(v_tmp_gamma[k]);
          sum_loss1 += dout_data[idx] * gamma_data[idx];
          sum_loss2 += dout_data[idx] * gamma_data[idx] *
                       (input_data[idx] - c_mean) * c_invvar;
        }
      }
    } else {
      int tid = thrx;
      for (int i = 0; tid < main_vec_n2; tid += tid_num, ++i) {
        VecT v_tmp_dout = v_k_dout[tid];
        VecT v_tmp_input = v_k_input[tid];
#pragma unroll
        for (int k = 0; k < DataPerTid; ++k) {
          const int idx = k + i * DataPerTid;
          dout_data[idx] = static_cast<U>(v_tmp_dout[k]);
          input_data[idx] = static_cast<U>(v_tmp_input[k]);
          sum_loss1 += dout_data[idx];
          sum_loss2 += dout_data[idx] * (input_data[idx] - c_mean) * c_invvar;
        }
      }
    }

    // intra-warp reductions
#pragma unroll
    for (int mask = WarpSize / 2; mask > 0; mask /= 2) {
#ifdef PADDLE_WITH_HIP
      // WARP_SHFL_XOR(sum_loss, mask);
      sum_loss1 += __shfl_xor(sum_loss1, mask, warpSize);
      sum_loss2 += __shfl_xor(sum_loss2, mask, warpSize);
#else
      // WARP_SHFL_XOR(sum_loss, mask);
      sum_loss1 += __shfl_xor_sync(0xffffffff, sum_loss1, mask, WarpSize);
      sum_loss2 += __shfl_xor_sync(0xffffffff, sum_loss2, mask, WarpSize);
#endif
    }

    // inter-warp reductions
    if (blockDim.y > 1) {
      __shared__ U buf[512];
      for (int offset = blockDim.y / 2; offset > 0; offset /= 2) {
        // upper half of warps write to shared
        if (threadIdx.y >= offset && threadIdx.y < 2 * offset) {
          const int wrt_i = (threadIdx.y - offset) * WarpSize + threadIdx.x;
          buf[2 * wrt_i] = sum_loss1;
          buf[2 * wrt_i + 1] = sum_loss2;
        }
        __syncthreads();
        // lower half merges
        if (threadIdx.y < offset) {
          const int read_i = threadIdx.y * blockDim.x + threadIdx.x;
          sum_loss1 += buf[2 * read_i];
          sum_loss2 += buf[2 * read_i + 1];
        }
        __syncthreads();
      }
      if (threadIdx.y == 0) {
        buf[2 * threadIdx.x] = sum_loss1;
        buf[2 * threadIdx.x + 1] = sum_loss2;
      }
      __syncthreads();
      if (threadIdx.y != 0) {
        sum_loss1 = buf[2 * threadIdx.x];
        sum_loss2 = buf[2 * threadIdx.x + 1];
      }
    }

    U fH = static_cast<U>(n2);
    U ratio_term = (static_cast<U>(1) / fH) * c_invvar;
    if (gamma != NULL) {
      int tid = thrx;
      for (int i = 0; tid < main_vec_n2; tid += tid_num, ++i) {
        VecT temp_grad;
#pragma unroll
        for (int k = 0; k < DataPerTid; ++k) {
          const int idx = i * DataPerTid + k;
          const U c_h = input_data[idx];
          const U c_loss = dout_data[idx];
          U f_grad_input = fH * c_loss * gamma_data[idx] - sum_loss1;
          f_grad_input -= (c_h - c_mean) * c_invvar * sum_loss2;
          temp_grad[k] = static_cast<T>(f_grad_input * ratio_term);
        }
        v_grad[tid] = temp_grad;
      }
    } else {
      int tid = thrx;
      for (int i = 0; tid < main_vec_n2; tid += tid_num, ++i) {
        VecT temp_grad;
#pragma unroll
        for (int k = 0; k < DataPerTid; ++k) {
          const int idx = i * DataPerTid + k;
          const U c_h = input_data[idx];
          const U c_loss = dout_data[idx];
          U f_grad_input = fH * c_loss - sum_loss1;
          f_grad_input -= (c_h - c_mean) * c_invvar * sum_loss2;
          temp_grad[k] = static_cast<T>(f_grad_input * ratio_term);
        }
        v_grad[tid] = temp_grad;
      }
    }
  }
}

// Make sure that d_scale != nullptr && d_bias != nullptr
// Since d_scale != nullptr, scale would not be nullptr
template <typename T,
          typename U,
          int BlockDim,
          bool HasDx,
          bool ScaleBiasWithSameTypeX>
__global__ void LayerNormBackwardGradientAll(
    const T *x,
    const T *d_y,
    LayerNormScaleBiasT<T, U, ScaleBiasWithSameTypeX> *d_scale,
    LayerNormScaleBiasT<T, U, ScaleBiasWithSameTypeX> *d_bias,
    T *d_x,
    const U *mean,
    const U *var,
    const LayerNormScaleBiasT<T, U, ScaleBiasWithSameTypeX> *scale,
    float epsilon,
    int64_t batch_size,
    int64_t feature_size,
    int64_t col_offset) {
  using ScaleBiasT = LayerNormScaleBiasT<T, U, ScaleBiasWithSameTypeX>;
  int64_t beg_idx = threadIdx.x * feature_size + (blockIdx.x + col_offset);
  int64_t end_idx = batch_size * feature_size + (blockIdx.x + col_offset);
  int64_t stride = BlockDim * feature_size;

  U d_scale_partial = static_cast<U>(0), d_bias_partial = static_cast<U>(0);

  for (int64_t i = beg_idx; i < end_idx; i += stride) {
    int row_idx = i / feature_size;
    auto var_val = rsqrt_(static_cast<U>(var[row_idx]) + epsilon);
    d_scale_partial += static_cast<U>(d_y[i]) *
                       (static_cast<U>(x[i]) - mean[row_idx]) * var_val;
    d_bias_partial += static_cast<U>(d_y[i]);
    if (HasDx) {
      d_x[i] = static_cast<T>(static_cast<U>(d_y[i]) *
                              static_cast<U>(scale[blockIdx.x + col_offset]) *
                              var_val);
    }
  }

  __shared__ U shared_scale[32];  // threadIdx.x / warpSize <= kMaxBlockDim /
                                  // warpSize <= 1024/32 = 32;
  __shared__ U shared_bias[32];
  d_scale_partial = BlockReduceSum<U>(d_scale_partial, shared_scale);
  d_bias_partial = BlockReduceSum<U>(d_bias_partial, shared_bias);

  if (threadIdx.x == 0) {
    d_scale[blockIdx.x + col_offset] = static_cast<ScaleBiasT>(d_scale_partial);
    d_bias[blockIdx.x + col_offset] = static_cast<ScaleBiasT>(d_bias_partial);
  }
}

// Make sure that there is only one true expression: d_scale != nullptr
// or d_bias != nullptr
// Notice: scale may be nullptr
template <typename T,
          typename U,
          int BlockDim,
          bool HasDx,
          bool HasDScale,
          bool ScaleBiasWithSameTypeX>
__global__ void LayerNormBackwardGradientScaleOrBias(
    const T *x,
    const T *d_y,
    LayerNormScaleBiasT<T, U, ScaleBiasWithSameTypeX> *d_scale,
    LayerNormScaleBiasT<T, U, ScaleBiasWithSameTypeX> *d_bias,
    T *d_x,
    const U *mean,
    const U *var,
    const LayerNormScaleBiasT<T, U, ScaleBiasWithSameTypeX> *scale,
    float epsilon,
    int64_t batch_size,
    int64_t feature_size,
    int col_offset) {
  using ScaleBiasT = LayerNormScaleBiasT<T, U, ScaleBiasWithSameTypeX>;
  using BlockReduce = cub::BlockReduce<U, BlockDim>;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  int64_t beg_idx = threadIdx.x * feature_size + blockIdx.x + col_offset;
  int64_t end_idx = batch_size * feature_size + blockIdx.x + col_offset;
  int stride = BlockDim * feature_size;
  U d_scale_or_d_bias_partial = static_cast<U>(0);

  for (int64_t i = beg_idx; i < end_idx; i += stride) {
    int row_idx = i / feature_size;
    auto var_val =
        static_cast<U>(rsqrt_(static_cast<float>(var[row_idx]) + epsilon));
    if (HasDScale) {
      d_scale_or_d_bias_partial += static_cast<U>(d_y[i]) *
                                   (static_cast<U>(x[i]) - mean[row_idx]) *
                                   var_val;
    } else {  // d_bias != nullptr
      d_scale_or_d_bias_partial += static_cast<U>(d_y[i]);
    }

    if (HasDx) {
      if (scale != nullptr) {
        d_x[i] = static_cast<T>(static_cast<U>(d_y[i]) *
                                static_cast<U>(scale[blockIdx.x + col_offset]) *
                                var_val);
      } else {
        d_x[i] = static_cast<T>(static_cast<U>(d_y[i]) * var_val);
      }
    }
  }

  d_scale_or_d_bias_partial =
      BlockReduce(temp_storage).Reduce(d_scale_or_d_bias_partial, cub::Sum());

  if (threadIdx.x == 0) {
    if (HasDScale) {
      d_scale[blockIdx.x + col_offset] =
          static_cast<ScaleBiasT>(d_scale_or_d_bias_partial);
    } else {
      d_bias[blockIdx.x + col_offset] =
          static_cast<ScaleBiasT>(d_scale_or_d_bias_partial);
    }
  }
}

template <typename T, typename U, int BlockDim>
__global__ void LayerNormBackwardPostProcessToCalculateDX(
    const T *x,
    T *d_x,
    const U *mean,
    const U *var,
    float epsilon,
    int64_t feature_size) {
  using BlockReduce = cub::BlockReduce<PairForLayerNorm<U>, BlockDim>;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  __shared__ U d_x_reduce_tmp[2];

  int64_t beg_idx = blockIdx.x * feature_size + threadIdx.x;
  int64_t end_idx = (blockIdx.x + 1) * feature_size;

  U block_mean = mean[blockIdx.x];
  U block_var = var[blockIdx.x];
  U d_x_mean_partial = static_cast<U>(0), d_x_var_partial = static_cast<U>(0);
  for (int64_t i = beg_idx; i < end_idx; i += BlockDim) {
    d_x_mean_partial += static_cast<U>(d_x[i]);
    d_x_var_partial +=
        static_cast<U>(d_x[i]) * (static_cast<U>(x[i]) - block_mean);
  }

  auto pair =
      BlockReduce(temp_storage)
          .Reduce(PairForLayerNorm<U>(d_x_mean_partial, d_x_var_partial),
                  PairForLayerNormAddFunctor<U>());

  if (threadIdx.x == 0) {
    d_x_reduce_tmp[0] = static_cast<float>(pair.first_) / feature_size;
    d_x_reduce_tmp[1] =
        static_cast<float>(pair.second_) /
        (feature_size * (static_cast<float>(block_var) + epsilon));
  }
  __syncthreads();

  d_x_mean_partial = d_x_reduce_tmp[0];
  d_x_var_partial = d_x_reduce_tmp[1];
  for (int64_t i = beg_idx; i < end_idx; i += BlockDim) {
    d_x[i] -= static_cast<T>(d_x_mean_partial);
    d_x[i] -=
        static_cast<T>((static_cast<U>(x[i]) - block_mean) * d_x_var_partial);
  }
}

// Here, we only calculate d_x
template <typename T, typename U, int BlockDim, bool ScaleBiasWithSameTypeX>
__global__ void LayerNormBackwardGradientOnlyDX(
    const T *x,
    const T *d_y,
    T *d_x,
    const U *mean,
    const U *var,
    const LayerNormScaleBiasT<T, U, ScaleBiasWithSameTypeX> *scale,
    float epsilon,
    int64_t feature_size) {
  using ScaleBiasT = LayerNormScaleBiasT<T, U, ScaleBiasWithSameTypeX>;
  using BlockReduce = cub::BlockReduce<PairForLayerNorm<U>, BlockDim>;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  __shared__ U d_x_reduce_tmp[2];

  int64_t beg_idx = blockIdx.x * feature_size + threadIdx.x;
  int64_t end_idx = (blockIdx.x + 1) * feature_size;

  U block_mean = mean[blockIdx.x], block_var = var[blockIdx.x];
  U d_x_mean_partial = static_cast<U>(0), d_x_var_partial = static_cast<U>(0);
  for (int64_t i = beg_idx; i < end_idx; i += BlockDim) {
    auto var_val =
        static_cast<U>(rsqrt_(static_cast<float>(block_var) + epsilon));
    if (scale != nullptr) {
      int col_idx = i % feature_size;
      d_x[i] = static_cast<T>(static_cast<U>(d_y[i]) *
                              static_cast<U>(scale[col_idx]) * var_val);
    } else {
      d_x[i] = static_cast<T>(static_cast<U>(d_y[i]) * var_val);
    }
    d_x_mean_partial += static_cast<U>(d_x[i]);
    d_x_var_partial +=
        static_cast<U>(d_x[i]) * (static_cast<U>(x[i]) - block_mean);
  }

  auto pair =
      BlockReduce(temp_storage)
          .Reduce(PairForLayerNorm<U>(d_x_mean_partial, d_x_var_partial),
                  PairForLayerNormAddFunctor<U>());

  if (threadIdx.x == 0) {
    d_x_reduce_tmp[0] = static_cast<float>(pair.first_) / feature_size;
    d_x_reduce_tmp[1] =
        static_cast<float>(pair.second_) /
        (feature_size * (static_cast<float>(block_var) + epsilon));
  }
  __syncthreads();

  d_x_mean_partial = d_x_reduce_tmp[0];
  d_x_var_partial = d_x_reduce_tmp[1];
  for (int64_t i = beg_idx; i < end_idx; i += BlockDim) {
    d_x[i] -= static_cast<T>(d_x_mean_partial);
    d_x[i] -=
        static_cast<T>((static_cast<U>(x[i]) - block_mean) * d_x_var_partial);
  }
}

template <typename T, typename U, bool ScaleBiasWithSameTypeX>
__global__ void LayerNormBackwardWhenBatchSizeIsOne(
    const T *x,
    const T *d_y,
    T *d_x,
    LayerNormScaleBiasT<T, U, ScaleBiasWithSameTypeX> *d_scale,
    LayerNormScaleBiasT<T, U, ScaleBiasWithSameTypeX> *d_bias,
    const U *mean,
    const U *var,
    const LayerNormScaleBiasT<T, U, ScaleBiasWithSameTypeX> *scale,
    float epsilon,
    int64_t feature_size) {
  int64_t idx = threadIdx.x + blockIdx.x * blockDim.x;
  using ScaleBiasT = LayerNormScaleBiasT<T, U, ScaleBiasWithSameTypeX>;
  if (idx < feature_size) {
    auto var_val = static_cast<U>(rsqrt_(static_cast<float>(var[0]) + epsilon));
    if (d_x != nullptr) {
      if (d_scale == nullptr) {
        d_x[idx] = static_cast<T>(static_cast<U>(d_y[idx]) * var_val);
      } else {
        d_x[idx] = static_cast<T>(static_cast<U>(d_y[idx]) *
                                  static_cast<U>(scale[idx]) * var_val);
      }
    }

    if (d_scale != nullptr) {
      d_scale[idx] =
          static_cast<ScaleBiasT>(static_cast<U>(d_y[idx]) *
                                  (static_cast<U>(x[idx]) - mean[0]) * var_val);
    }

    if (d_bias != nullptr) {
      d_bias[idx] = static_cast<ScaleBiasT>(d_y[idx]);
    }
  }
}

inline int VecSizeJudgeForeGradInput(const int feature_size,
                                     const int vec_size) {
  if (!(feature_size & (vec_size - 1))) {
    return vec_size;
  } else if (vec_size == 4) {
    if (!(feature_size & 1)) {
      return 2;
    }
  }
  return 1;
}

template <typename T, typename U, bool ScaleBiasWithSameTypeX = false>
static void LayerNormBackward(
    const T *x,
    const T *d_y,
    const LayerNormScaleBiasT<T, U, ScaleBiasWithSameTypeX> *scale,
    const U *mean,
    const U *var,
    T *d_x,
    LayerNormScaleBiasT<T, U, ScaleBiasWithSameTypeX> *d_scale,
    LayerNormScaleBiasT<T, U, ScaleBiasWithSameTypeX> *d_bias,
    float epsilon,
    int64_t batch_size,
    int64_t feature_size,
    const phi::GPUContext &dev_ctx) {
  auto stream = dev_ctx.stream();
#ifdef __HIPCC__
  const int kMaxBlockDim = 256;
#else
  const int kMaxBlockDim = 512;
#endif
  const int kMaxBlockNum = 128;
  int gradient_flag = ((d_x != nullptr ? 1 : 0) << 2) |
                      ((d_scale != nullptr ? 1 : 0) << 1) |
                      ((d_bias != nullptr ? 1 : 0));
  if (gradient_flag == 0) return;

  if (batch_size == 1) {
    LayerNormBackwardWhenBatchSizeIsOne<T, U, ScaleBiasWithSameTypeX>
        <<<(feature_size + kMaxBlockDim - 1) / kMaxBlockDim,
           kMaxBlockDim,
           0,
           stream>>>(x,
                     d_y,
                     d_x,
                     d_scale,
                     d_bias,
                     mean,
                     var,
                     scale,
                     epsilon,
                     feature_size);

    if (d_x != nullptr) {
      switch (GetDesiredBlockDim(feature_size)) {
        FIXED_BLOCK_DIM_CASE(
            LayerNormBackwardPostProcessToCalculateDX<T, U, kBlockDim>
            <<<1, kBlockDim, 0, stream>>>(
                x, d_x, mean, var, epsilon, feature_size));
      }
    }
    return;
  }

  auto block_dim = GetDesiredBlockDim(batch_size);
  switch (gradient_flag) {
    case 1:  // d_x == nulptr, d_scale == nullptr, d_bias != nullptr
      switch (block_dim) {
        FIXED_BLOCK_DIM_FIXED_BLOCK_NUM_CASE(
            feature_size,
            kMaxBlockNum,
            LayerNormBackwardGradientScaleOrBias<T,
                                                 U,
                                                 kBlockDim,
                                                 false,
                                                 false,
                                                 ScaleBiasWithSameTypeX>
            <<<block_num, kBlockDim, 0, stream>>>(x,
                                                  d_y,
                                                  d_scale,
                                                  d_bias,
                                                  d_x,
                                                  mean,
                                                  var,
                                                  scale,
                                                  epsilon,
                                                  batch_size,
                                                  feature_size,
                                                  col_offset));
      }
      break;
    case 2:  // d_x == nullptr, d_scale != nullptr, d_bias == nullptr
      switch (block_dim) {
        FIXED_BLOCK_DIM_FIXED_BLOCK_NUM_CASE(
            feature_size,
            kMaxBlockNum,
            LayerNormBackwardGradientScaleOrBias<T,
                                                 U,
                                                 kBlockDim,
                                                 false,
                                                 true,
                                                 ScaleBiasWithSameTypeX>
            <<<block_num, kBlockDim, 0, stream>>>(x,
                                                  d_y,
                                                  d_scale,
                                                  d_bias,
                                                  d_x,
                                                  mean,
                                                  var,
                                                  scale,
                                                  epsilon,
                                                  batch_size,
                                                  feature_size,
                                                  col_offset));
      }
      break;
    case 3:  // d_x == nullptr, d_scale != nulptr, d_bias != nullptr
      switch (block_dim) {
        FIXED_BLOCK_DIM_FIXED_BLOCK_NUM_CASE(
            feature_size,
            kMaxBlockNum,
            LayerNormBackwardGradientAll<T,
                                         U,
                                         kBlockDim,
                                         false,
                                         ScaleBiasWithSameTypeX>
            <<<block_num, kBlockDim, 0, stream>>>(x,
                                                  d_y,
                                                  d_scale,
                                                  d_bias,
                                                  d_x,
                                                  mean,
                                                  var,
                                                  scale,
                                                  epsilon,
                                                  batch_size,
                                                  feature_size,
                                                  col_offset));
      }
      break;
    case 4:  // d_x != nullptr, d_scale == nullptr, d_bias == nullptr
      switch (GetDesiredBlockDim(feature_size)) {
        FIXED_BLOCK_DIM_CASE(
            LayerNormBackwardGradientOnlyDX<T,
                                            U,
                                            kBlockDim,
                                            ScaleBiasWithSameTypeX>
            <<<batch_size, kBlockDim, 0, stream>>>(
                x, d_y, d_x, mean, var, scale, epsilon, feature_size));
      }
      break;
    case 5:  // d_x != nulptr, d_scale == nullptr, d_bias != nullptr
      switch (block_dim) {
        FIXED_BLOCK_DIM_FIXED_BLOCK_NUM_CASE(
            feature_size,
            kMaxBlockNum,
            LayerNormBackwardGradientScaleOrBias<T,
                                                 U,
                                                 kBlockDim,
                                                 true,
                                                 false,
                                                 ScaleBiasWithSameTypeX>
            <<<block_num, kBlockDim, 0, stream>>>(x,
                                                  d_y,
                                                  d_scale,
                                                  d_bias,
                                                  d_x,
                                                  mean,
                                                  var,
                                                  scale,
                                                  epsilon,
                                                  batch_size,
                                                  feature_size,
                                                  col_offset));
      }
      switch (GetDesiredBlockDim(feature_size)) {
        FIXED_BLOCK_DIM_CASE(
            LayerNormBackwardPostProcessToCalculateDX<T, U, kBlockDim>
            <<<batch_size, kBlockDim, 0, stream>>>(
                x, d_x, mean, var, epsilon, feature_size));
      }
      break;
    case 6:  // d_x != nullptr, d_scale != nullptr, d_bias == nullptr
      switch (block_dim) {
        FIXED_BLOCK_DIM_FIXED_BLOCK_NUM_CASE(
            feature_size,
            kMaxBlockNum,
            LayerNormBackwardGradientScaleOrBias<T,
                                                 U,
                                                 kBlockDim,
                                                 true,
                                                 true,
                                                 ScaleBiasWithSameTypeX>
            <<<block_num, kBlockDim, 0, stream>>>(x,
                                                  d_y,
                                                  d_scale,
                                                  d_bias,
                                                  d_x,
                                                  mean,
                                                  var,
                                                  scale,
                                                  epsilon,
                                                  batch_size,
                                                  feature_size,
                                                  col_offset));
      }
      switch (GetDesiredBlockDim(feature_size)) {
        FIXED_BLOCK_DIM_CASE(
            LayerNormBackwardPostProcessToCalculateDX<T, U, kBlockDim>
            <<<batch_size, kBlockDim, 0, stream>>>(
                x, d_x, mean, var, epsilon, feature_size));
      }
      break;
    case 7:  // d_x != nullptr, d_scale != nullptr, d_bias != nullptr
    {
#ifdef PADDLE_WITH_CUDA
      bool can_call_fast_kernel = false;
      // todo: rule out double type.
      if ((feature_size == 1024 || feature_size == 384 ||
           feature_size == 256) &&
          sizeof(T) <= 4) {
        can_call_fast_kernel = true;
      }

      VLOG(6) << "can_call_fast_kernel = " << can_call_fast_kernel;
      if (can_call_fast_kernel) {
        ln_bwd_fast_kernel_driver<
            T,
            U,
            LayerNormScaleBiasT<T, U, ScaleBiasWithSameTypeX>>(dev_ctx,
                                                               batch_size,
                                                               feature_size,
                                                               epsilon,
                                                               x,
                                                               scale,
                                                               mean,
                                                               var,
                                                               d_y,
                                                               d_x,
                                                               d_scale,
                                                               d_bias);
      } else {
#endif
        using ScaleT = LayerNormScaleBiasT<T, U, ScaleBiasWithSameTypeX>;
        constexpr int BDIMX = 32;

        constexpr int VPT = 4;
        constexpr int BDIMY1 = 4;
        constexpr int PartSize = BDIMY1 * VPT;
        dim3 threads2(BDIMX, BDIMY1, 1);
        dim3 blocks2((feature_size + BDIMX - 1) / BDIMX, PartSize, 1);

        int64_t param_num = PartSize * feature_size;
        auto part_grad_param_ptr = phi::memory_utils::Alloc(
            dev_ctx.GetPlace(),
            param_num * sizeof(U) * 2,  // for both gamma and beta
            phi::Stream(reinterpret_cast<phi::StreamId>(dev_ctx.stream())));

        U *part_grad_gamma = reinterpret_cast<U *>(part_grad_param_ptr->ptr());
        U *part_grad_beta = reinterpret_cast<U *>(part_grad_gamma + param_num);

        LayerNormBackwardPartGradGammaBeta<T, U, BDIMX, BDIMY1, VPT>
            <<<blocks2, threads2, 0, stream>>>(d_y,
                                               x,
                                               batch_size,
                                               feature_size,
                                               mean,
                                               var,
                                               epsilon,
                                               part_grad_gamma,
                                               part_grad_beta);

        constexpr int BDIMY2 = 8;
        dim3 threads3(BDIMX, BDIMY2, 1);
        const dim3 blocks3((feature_size + BDIMX - 1) / BDIMX, 1, 1);
        LayerNormBackwardSumGradGammaBeta<T, U, BDIMX, BDIMY2, ScaleT>
            <<<blocks3, threads3, 0, stream>>>(part_grad_gamma,
                                               part_grad_beta,
                                               PartSize,
                                               batch_size,
                                               feature_size,
                                               d_scale,
                                               d_bias);

        uint64_t addr = reinterpret_cast<uint64_t>(d_y) |
                        reinterpret_cast<uint64_t>(x) |
                        reinterpret_cast<uint64_t>(d_x);
        int vec_size = phi::GetVectorizedSize<T>(reinterpret_cast<T *>(addr));
        int real_vec = VecSizeJudgeForeGradInput(feature_size, vec_size);

        if (feature_size <= 2048) {
          // One thread must work with at least real_vec quantity data, at most
          // 8 data.
          int data_per_warp = BDIMX * real_vec;
          uint32_t warp_num =
              feature_size < data_per_warp ? 1 : (feature_size / data_per_warp);
#if defined(__clang__) || defined(__GNUC__)
          int block_dim_y = std::min(8, 1 << (31 - __builtin_clz(warp_num)));
#else
        int block_dim_y = 1;
        while (warp_num != 0) {
          warp_num = warp_num >> 1;
          block_dim_y <<= 1;
        }
        block_dim_y = std::min(8, (block_dim_y / 2));
#endif  // __GNUCC__

          dim3 threads1(BDIMX, block_dim_y, 1);
#define IMPL_BACKWARD_FOR_INPUT(num)                                       \
  LayerNormBackwardComputeGradInputWithSmallFeatureSize<T, U, ScaleT, num> \
      <<<batch_size, threads1, 0, stream>>>(                               \
          d_y, x, batch_size, feature_size, mean, var, epsilon, scale, d_x);

          switch (real_vec) {
            case 4: {
              IMPL_BACKWARD_FOR_INPUT(4);
            } break;
            case 2: {
              IMPL_BACKWARD_FOR_INPUT(2);
            } break;
            default: {
              IMPL_BACKWARD_FOR_INPUT(1);
            }
          }
#undef IMPL_BACKWARD_FOR_INPUT

        } else {
          constexpr int BDIMY3 = 4;
          dim3 threads1(BDIMX, BDIMY3, 1);
          LayerNormBackwardComputeGradInput<T, U, BDIMX, BDIMY3, ScaleT>
              <<<batch_size, threads1, 0, stream>>>(d_y,
                                                    x,
                                                    batch_size,
                                                    feature_size,
                                                    mean,
                                                    var,
                                                    epsilon,
                                                    scale,
                                                    d_x);
        }
#ifdef PADDLE_WITH_CUDA
      }
#endif

      break;
    }
    default:
      break;
  }
}

}  // namespace funcs
}  // namespace phi
