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

#include "paddle/fluid/framework/ddim.h"
#include "paddle/fluid/platform/aligned_vector.h"
#include "paddle/fluid/platform/device/gpu/gpu_device_function.h"
#include "paddle/fluid/platform/device/gpu/gpu_dnn.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
template <typename T>
using CudnnDataType = platform::CudnnDataType<T>;
template <typename T>
using LayerNormParamType = typename CudnnDataType<T>::BatchNormParamType;

#define LN_NUM_COLS 1024

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
    val += paddle::platform::CudaShuffleDownSync(mask, val, offset);
  }
  return val;
}

template <typename U>
__forceinline__ __device__ U BlockReduceSum(U val, U *shared) {
  int lane = threadIdx.x % warpSize;
  int wid = threadIdx.x / warpSize;

  val = WarpReduceSum(val);  // Each warp performs partial reduction

  __syncthreads();
  if (lane == 0) shared[wid] = val;  // Write reduced value to shared memory

  __syncthreads();  // Wait for all partial reductions
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
  FIXED_BLOCK_DIM_FIXED_BLOCK_NUM_CASE_BASE(9, feature_size, kMaxBlockNum,    \
                                            ##__VA_ARGS__);                   \
  FIXED_BLOCK_DIM_FIXED_BLOCK_NUM_CASE_BASE(8, feature_size, kMaxBlockNum,    \
                                            ##__VA_ARGS__);                   \
  FIXED_BLOCK_DIM_FIXED_BLOCK_NUM_CASE_BASE(7, feature_size, kMaxBlockNum,    \
                                            ##__VA_ARGS__);                   \
  FIXED_BLOCK_DIM_FIXED_BLOCK_NUM_CASE_BASE(6, feature_size, kMaxBlockNum,    \
                                            ##__VA_ARGS__);                   \
  FIXED_BLOCK_DIM_FIXED_BLOCK_NUM_CASE_BASE(5, feature_size, kMaxBlockNum,    \
                                            ##__VA_ARGS__);                   \
  FIXED_BLOCK_DIM_FIXED_BLOCK_NUM_CASE_BASE(4, feature_size, kMaxBlockNum,    \
                                            ##__VA_ARGS__);                   \
  FIXED_BLOCK_DIM_FIXED_BLOCK_NUM_CASE_BASE(3, feature_size, kMaxBlockNum,    \
                                            ##__VA_ARGS__);                   \
  FIXED_BLOCK_DIM_FIXED_BLOCK_NUM_CASE_BASE(2, feature_size, kMaxBlockNum,    \
                                            ##__VA_ARGS__);                   \
  FIXED_BLOCK_DIM_FIXED_BLOCK_NUM_CASE_BASE(1, feature_size, kMaxBlockNum,    \
                                            ##__VA_ARGS__)

static __device__ __forceinline__ float real_sqrt(float x) { return sqrtf(x); }
static __device__ __forceinline__ double real_sqrt(double x) { return sqrt(x); }

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
  return rsqrt(val);
}

#if CUDA_ARCH_FP16_SUPPORTED(__CUDA_ARCH__)
template <>
__inline__ __device__ half rsqrt_(const half val) {
  return hrsqrt(val);
}
#endif

template <typename T, typename U, typename ScaleT = U, int VecSize = 8,
          int WARPS_M = 4, int WARPS_N = 1, int BYTES_PER_LDG = 16,
          int ELTS_PER_ROW = 1024, int THREADS_PER_WARP = 32,
          int THREADS_PER_ROW = WARPS_N *THREADS_PER_WARP,
          int THREADS_PER_CTA = WARPS_M *THREADS_PER_ROW,
          int ROWS_PER_CTA = WARPS_M,
          int ELTS_PER_ROW_PER_CTA = THREADS_PER_ROW *VecSize,
          int LDGS = ELTS_PER_ROW / ELTS_PER_ROW_PER_CTA>
__global__ __launch_bounds__(THREADS_PER_CTA) void ln_fwd_1024_kernel(
    int rows, int cols, const float epsilon, const T *__restrict__ x_ptr,
    const ScaleT *__restrict__ gamma_ptr, const ScaleT *__restrict__ beta_ptr,
    U *__restrict__ mean_out_ptr, U *__restrict__ var_out_ptr,
    T *__restrict__ y_ptr) {
  using Vec = platform::AlignedVector<T, VecSize>;
  using Vec_scale = platform::AlignedVector<ScaleT, VecSize>;

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
    platform::Load<ScaleT, VecSize>(gamma_ptr + col * VecSize, &gamma[it]);
    platform::Load<ScaleT, VecSize>(beta_ptr + col * VecSize, &beta[it]);
    col += THREADS_PER_ROW;
  }

  constexpr U rn = 1.f / U(LN_NUM_COLS);
  for (int row = r; row < rows; row += gridDim.x * ROWS_PER_CTA) {
    Vec x[LDGS];
#pragma unroll
    for (int it = 0, col = c; it < LDGS; it++) {
      platform::Load<T, VecSize>(x_ptr + row * LN_NUM_COLS + col * VecSize,
                                 &x[it]);
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
      platform::Store<T, VecSize>(x[it],
                                  y_ptr + row * LN_NUM_COLS + col * VecSize);
      col += THREADS_PER_ROW;
    }
  }
}

template <typename T, typename U, bool ScaleBiasWithSameTypeX>
using LayerNormScaleBiasT =
    typename std::conditional<ScaleBiasWithSameTypeX, T, U>::type;

template <typename T, typename U, int BlockDim,
          bool ScaleBiasWithSameTypeX = false>
__global__ void LayerNormForward(
    const T *x, const LayerNormScaleBiasT<T, U, ScaleBiasWithSameTypeX> *scale,
    const LayerNormScaleBiasT<T, U, ScaleBiasWithSameTypeX> *bias, T *y,
    U *mean, U *var, float epsilon, int64_t feature_size) {
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
    auto scale = static_cast<float>(1.) / static_cast<float>(feature_size);
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
        y[i] = static_cast<T>(static_cast<U>(scale[j]) *
                                  (static_cast<U>(x[i]) - mean_val) * invvar +
                              static_cast<U>(bias[j]));
      }
    } else {
      for (int64_t i = beg_idx, j = threadIdx.x; i < end_idx;
           i += BlockDim, j += BlockDim) {
        y[i] = static_cast<T>(static_cast<U>(scale[j]) *
                              (static_cast<U>(x[i]) - mean_val) * invvar);
      }
    }
  } else {  // scale == nullptr
    if (bias != nullptr) {
      for (int64_t i = beg_idx, j = threadIdx.x; i < end_idx;
           i += BlockDim, j += BlockDim) {
        y[i] = static_cast<T>((static_cast<U>(x[i]) - mean_val) * invvar +
                              static_cast<U>(bias[j]));
      }
    } else {
      for (int64_t i = beg_idx, j = threadIdx.x; i < end_idx;
           i += BlockDim, j += BlockDim) {
        y[i] = static_cast<T>((static_cast<U>(x[i]) - mean_val) * invvar);
      }
    }
  }
}

template <typename T, typename U, int VPT>
__inline__ __device__ void cuLoadAddStridedInputs(
    const int64_t i1_block, const int thr_load_row_off,
    const int thr_load_col_off, const int i2_off, const int row_stride,
    U *warp_buf1, U *warp_buf2, const T *input, const T *dout,
    const int64_t i1_end, const int64_t n2, const U *__restrict__ mean,
    const U *__restrict__ var, const float epsilon) {
  const int64_t i1 = i1_block + thr_load_row_off;
  if (i1 >= i1_end) return;
  U curr_mean = mean[i1];
  U curr_invvar = rsqrt_<U>(var[i1] + epsilon);
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

// 2*sms blocks
// 32 * 4 threads per block.
#define COLS_ 1024
template <typename T, typename U, typename ScaleT = U, int VecSize = 8,
          int WARPS_M = 4, int WARPS_N = 1, int BYTES_PER_LDG = 16,
          int ELTS_PER_ROW = 1024, int THREADS_PER_WARP = 32,
          int THREADS_PER_ROW = WARPS_N *THREADS_PER_WARP,
          int THREADS_PER_CTA = WARPS_M *THREADS_PER_ROW,
          int ROWS_PER_CTA = WARPS_M,
          int ELTS_PER_ROW_PER_CTA = THREADS_PER_ROW *VecSize,
          int LDGS = ELTS_PER_ROW / ELTS_PER_ROW_PER_CTA>
__global__ __launch_bounds__(THREADS_PER_CTA) void ln_bwd_1024_kernel(
    const int rows, const T *__restrict__ x_ptr,
    const ScaleT *__restrict__ gamma_ptr, const U *__restrict__ mu_ptr,
    const U *__restrict__ rs_ptr, const T *__restrict__ dout_ptr,
    U *__restrict__ dgamma_temp_ptr, U *__restrict__ dbeta_temp_ptr,
    T *__restrict__ dx_ptr) {
  using Vec = platform::AlignedVector<T, VecSize>;
  using Vec_scale = platform::AlignedVector<ScaleT, VecSize>;

  const int tidx = threadIdx.x;
  const int bidx = blockIdx.x;
  const int lane = tidx % THREADS_PER_WARP;            // 0, 1, ..., 31
  const int warp = tidx / THREADS_PER_WARP;            // 0, 1, 2, 3
  const int warp_m = warp / WARPS_N;                   // 0, 1, 2, 3
  const int warp_n = warp % WARPS_N;                   // 0
  const int tid_r = warp_n * THREADS_PER_WARP + lane;  // 0, 1, ..., 31

  const int r = bidx * ROWS_PER_CTA + warp_m;
  const int c = warp_n * THREADS_PER_WARP + lane;

  static_assert(COLS_ == THREADS_PER_ROW * LDGS * VecSize, "");

  // smem for final reduction
  // __shared__ U smem_[ROWS_PER_CTA * COLS];
  extern __shared__ U smem_[];
  // static_assert(sizeof(smem_dw_sum) == 32*1024,"");
  // Using the grid stride loop we can assign multiple rows to each thread
  // by using a number of CTAs smaller than rows / ROWS_PER_CTA
  // We accumulate them here, one in smem, one in registers, because the smem
  // capacity is limited U * dw_sum = &smem_dw_sum[warp_m * COLS + tid_r
  // * LDGS * VecSize];
  U dwy_sum[LDGS * VecSize];
  U dw_sum[LDGS * VecSize];

  memset(dwy_sum, 0, sizeof(U) * LDGS * VecSize);
  memset(dw_sum, 0, sizeof(U) * LDGS * VecSize);
  // Debug 8 rows, 4B, 1024 cols

  __shared__ U smem_mdy[ROWS_PER_CTA * WARPS_N];   // 4
  __shared__ U smem_mdyy[ROWS_PER_CTA * WARPS_N];  // 4
  U *mdy_shared = &smem_mdy[warp_m * WARPS_N];     //
  U *mdyy_shared = &smem_mdyy[warp_m * WARPS_N];   //

  constexpr float rn = 1.f / static_cast<float>(COLS_);
  Vec_scale gamma[LDGS];
  int col = c;
#pragma unroll
  for (int it = 0; it < LDGS; it++) {
    // gamma[it].load_from(g_ptr + col * BYTES_PER_LDG);
    platform::Load<ScaleT, VecSize>(gamma_ptr + col * VecSize, &gamma[it]);
    col += THREADS_PER_ROW;
  }
// if ROWS_PER_CTA does not divice rows, we might get divergence in the
// last blocks with syncthreads!
// grid stride over rows
#pragma unroll 1
  for (int row = r; row < rows; row += gridDim.x * ROWS_PER_CTA) {
    const U mu_r = mu_ptr[row];
    const U rs_r = rs_ptr[row];
    Vec dout[LDGS], x[LDGS], dx[LDGS];
    int col = c;
#pragma unroll
    for (int it = 0; it < LDGS; it++) {
      // dw[it].load_from(dw_ptr + row * BYTES_PER_ROW + col * BYTES_PER_LDG);
      // x[it].load_from(x_ptr + row * BYTES_PER_ROW + col * BYTES_PER_LDG);
      platform::Load<T, VecSize>(dout_ptr + row * COLS_ + col * VecSize,
                                 &dout[it]);
      platform::Load<T, VecSize>(x_ptr + row * COLS_ + col * VecSize, &x[it]);
      col += THREADS_PER_ROW;
    }
    // local reductions
    U dy[LDGS * VecSize];
    U y[LDGS * VecSize];

    U mdy_local = 0.f;
    U mdyy_local = 0.f;
#pragma unroll
    for (int it = 0; it < LDGS; it++) {
#pragma unroll
      for (int jt = 0; jt < VecSize; jt++) {
        // U x_tmp = x[it].data.elt[jt];
        U x_tmp = x[it][jt];
        U y_tmp = rs_r * (x_tmp - mu_r);
        // U dy_tmp = gamma[it].data.elt[jt] * dw[it].data.elt[jt]; // scale *
        // dy
        // U dw_tmp = dw[it].data.elt[jt]; // dy
        U dy_tmp = gamma[it][jt] * dout[it][jt];  // scale * dy
        U dw_tmp = dout[it][jt];                  // dy

        // 求dx的，后面进行行规约
        mdy_local += dy_tmp;           // scale * dy, sum_1
        mdyy_local += dy_tmp * y_tmp;  // scale * dy * y, sum_2

        dy[it * VecSize + jt] = dy_tmp;  // scale* dy
        y[it * VecSize + jt] = y_tmp;    // y

        // 求dscale和dbias的，列规约.
        dwy_sum[it * VecSize + jt] += dw_tmp * y_tmp;  // dy * y
        dw_sum[it * VecSize + jt] += dw_tmp;           // dy
      }
    }

    // reduction across row for mdy, mdyy
    if (WARPS_N == 1) {  // no need to go through smem!
#pragma unroll
      // 32个线程的结果进行行规约，
      for (int it = 1; it < THREADS_PER_WARP; it *= 2) {
        mdy_local += __shfl_xor_sync(uint32_t(-1), mdy_local, it);
        mdyy_local += __shfl_xor_sync(uint32_t(-1), mdyy_local, it);
      }

      mdy_local *= rn;
      mdyy_local *= rn;

    } else {
#pragma unroll
      for (int it = 16; it > 0; it /= 2) {
        mdy_local += __shfl_down_sync(uint32_t(-1), mdy_local, it);
        mdyy_local += __shfl_down_sync(uint32_t(-1), mdyy_local, it);
      }  // lane 0 holds the result!

      if (lane == 0) {
        mdy_shared[warp_n] = mdy_local;
        mdyy_shared[warp_n] = mdyy_local;
      }

      __syncthreads();
      if (warp_n == 0 && lane == 0) {
        mdy_local = 0.f;
        mdyy_local = 0.f;
        for (int it = 0; it < WARPS_N; it++) {
          mdy_local += mdy_shared[it];
          mdyy_local += mdyy_shared[it];
        }
        mdy_shared[0] = mdy_local;
        mdyy_shared[0] = mdyy_local;
      }
      __syncthreads();

      mdy_local = mdy_shared[0] * rn;
      mdyy_local = mdyy_shared[0] * rn;
    }

//
#pragma unroll
    for (int it = 0; it < LDGS; it++) {
#pragma unroll
      for (int jt = 0; jt < VecSize; jt++) {
        U dy_tmp = dy[it * VecSize + jt];  // scale* dy
        U y_tmp = y[it * VecSize + jt];    // y
        // var * (scale*dy - sum_2 * y - sum_1)
        U dx_tmp = U(rs_r) * (dy_tmp - mdyy_local * y_tmp - mdy_local);
        // dx[it].data.elt[jt] = dx_tmp;
        // todo: static_cast<T>
        dx[it][jt] = dx_tmp;
      }
    }

    col = c;  // 0, ... 31
              // 将dx的计算结果存回.
#pragma unroll
    for (int it = 0; it < LDGS; it++) {
      // dx[it].store_to(dx_ptr + row * BYTES_PER_ROW + col * BYTES_PER_LDG);
      platform::Store<T, VecSize>(dx[it], dx_ptr + row * COLS_ + col * VecSize);
      col += THREADS_PER_ROW;
    }
  }  // end: grid stride loop

  // Finalize reduction of part dgamma and dbeta for this CTA
  // by reducing over the rows held across the WARPS_M warps

  enum { NUM_RES = COLS_ / THREADS_PER_CTA };  // 1024/128 = 8
  static_assert(NUM_RES * THREADS_PER_CTA == COLS_, "");

  U *smem_write;

  // tid_r: 0, ..., 31(lane id)
  smem_write = &smem_[warp_m * COLS_ + tid_r * VecSize];  // 4 * 1024大小
#pragma unroll
  for (int it = 0; it < LDGS; it++) {
#pragma unroll
    for (int jt = 0; jt < VecSize; jt++) {
      smem_write[jt] = dw_sum[it * VecSize + jt];
    }
    smem_write += THREADS_PER_ROW * VecSize;  // 32*8
  }
  __syncthreads();
  U cta_dw_sum[NUM_RES];
  memset(cta_dw_sum, 0, sizeof(U) * NUM_RES);
  // 对smem中的元素按照列进行规约: 4*1024 -> 1*1024.
  for (int it = 0; it < ROWS_PER_CTA; it++) {
    for (int jt = 0; jt < NUM_RES; jt++) {
      cta_dw_sum[jt] += smem_[it * COLS_ + tidx + jt * THREADS_PER_CTA];
    }
  }
  __syncthreads();

  smem_write = &smem_[warp_m * COLS_ + tid_r * VecSize];
#pragma unroll
  for (int it = 0; it < LDGS; it++) {
#pragma unroll
    for (int jt = 0; jt < VecSize; jt++) {
      smem_write[jt] = dwy_sum[it * VecSize + jt];
    }
    smem_write += THREADS_PER_ROW * VecSize;
  }
  __syncthreads();
  U cta_dwy_sum[NUM_RES];
  memset(cta_dwy_sum, 0, sizeof(U) * NUM_RES);
  for (int it = 0; it < ROWS_PER_CTA; it++) {
    for (int jt = 0; jt < NUM_RES; jt++) {
      cta_dwy_sum[jt] += smem_[it * COLS_ + tidx + jt * THREADS_PER_CTA];
    }
  }

  // 结果的shape：(#blocks, 1024)
  U *dgamma_part = static_cast<U *>(dgamma_temp_ptr) + bidx * COLS_ + tidx;
  for (int jt = 0; jt < NUM_RES; jt++) {
    *dgamma_part = cta_dwy_sum[jt];  //
    dgamma_part += THREADS_PER_CTA;  // 128
  }

  U *dbeta_part = static_cast<U *>(dbeta_temp_ptr) + bidx * COLS_ + tidx;
  for (int jt = 0; jt < NUM_RES; jt++) {
    *dbeta_part = cta_dw_sum[jt];
    dbeta_part += THREADS_PER_CTA;
  }
}

// todo: driver routine
template <typename T, typename U, typename ScaleT = U, int VecSize = 8>
void ln_bwd_1024_kernel_driver(const platform::CUDADeviceContext &dev_ctx,
                               cudaStream_t stream, const int rows,
                               const int cols, const T *x_ptr,
                               const ScaleT *gamma_ptr, const U *mean_ptr,
                               const U *var_ptr, const T *dout_ptr, T *dx_ptr,
                               ScaleT *dgamma_ptr, ScaleT *dbeta_ptr) {
  if (cols == 1024) {
    // step-1: compute dx and reduced intermediate results of dscale and dbias.
    const int WARPS_M = 4;
    const int WARPS_N = 1;
    const int BYTES_PER_LDG = 16;
    const int THREADS_PER_WARP = 32;
    const int THREADS_PER_ROW = WARPS_N * THREADS_PER_WARP;
    const int THREADS_PER_CTA = WARPS_M * THREADS_PER_ROW;
    const int ROWS_PER_CTA = WARPS_M;
    // int ELTS_PER_ROW = 1024;
    // int ELTS_PER_ROW_PER_CTA=THREADS_PER_ROW*VecSize,
    // int LDGS = ELTS_PER_ROW / ELTS_PER_ROW_PER_CTA;

    // 4 * 1024 * 4
    const int SMEM_BYTES = ROWS_PER_CTA * cols * sizeof(U);
#if 0
    if (SMEM_BYTES >= 48 * 1024) {
      AT_CUDA_CHECK(cudaFuncSetAttribute(
          ln_bwd_kernel<Ktraits>, cudaFuncAttributeMaxDynamicSharedMemorySize,
          Ktraits::SMEM_BYTES));
    }
#endif

    // blocks数目：2*SM数目
    int gridx = 216;

    // get temp space for dgamma and dbeta.
    framework::Tensor dgamma_temp;
    dgamma_temp.Resize({gridx, cols});
    dgamma_temp.mutable_data<U>(dev_ctx.GetPlace());
    U *dgamma_temp_ptr = dgamma_temp.data<U>();

    framework::Tensor dbeta_temp;
    dbeta_temp.Resize({gridx, cols});
    dbeta_temp.mutable_data<U>(dev_ctx.GetPlace());
    U *dbeta_temp_ptr = dbeta_temp.data<U>();

    ln_bwd_1024_kernel<
        T, U, ScaleT, VecSize, WARPS_M, WARPS_N,
        BYTES_PER_LDG><<<gridx, THREADS_PER_CTA, SMEM_BYTES, stream>>>(
        rows, x_ptr, gamma_ptr, mean_ptr, var_ptr, dout_ptr, dgamma_temp_ptr,
        dbeta_temp_ptr, dx_ptr);

    // using Ktraits2 = Kernel_traits<float, 1024, 16, 1, 4>;
    // // 向量化长度为1.
    // constexpr int grid2 =
    //     DIVUP(1024, Ktraits2::THREADS_PER_ROW * Ktraits2::Vec::VecSize); //
    //     1024/(32*1)= 32
    // // 32个block，块内512个线程.
    // ln_bwd_finalize_kernel<Ktraits2, scalar_t>
    //     <<<grid2, Ktraits2::THREADS_PER_CTA, 0, stream>>>(
    //         dgamma_ptr, dbeta_ptr, dgamma_temp_ptr,
    //         dbeta_temp_ptr, gridx);
  } else {
    PADDLE_THROW(platform::errors::InvalidArgument(
        "Fast layer_norm kernel is only used when feature_size is 1024"));
  }
  // AT_CUDA_CHECK(cudaPeekAtLastError());
}

template <typename T, typename U, int BDIMX, int BDIMY, int VPTX>
__global__ void LayerNormBackwardPartGradGammaBeta(
    const T *__restrict__ dout, const T *__restrict__ input, const int64_t n1,
    const int64_t n2, const U *__restrict__ mean, const U *__restrict__ var,
    float epsilon, U *part_grad_gamma, U *part_grad_beta) {
  // VPTX -> value per thread.x, BDIMX -> blockDim.x, BDIMY -> blockDim.y, BDIMX
  // -> blockDim.x
  // template for compile time optimizations

  constexpr int row_stride = BDIMX + 1;
  const int thr_load_col_off = (threadIdx.x * VPTX) & (BDIMX - 1);
  const int thr_load_row_off =
      (threadIdx.x * VPTX) / BDIMX + threadIdx.y * BDIMY;
  const int i2_off = blockIdx.x * BDIMX + thr_load_col_off;

  constexpr int shared_cap = (BDIMX * BDIMY > 2 * VPTX * BDIMY * row_stride)
                                 ? BDIMX * BDIMY
                                 : 2 * VPTX * BDIMY * row_stride;
  __shared__ U buf[shared_cap];

  U *warp_buf1 = reinterpret_cast<U *>(buf);
  U *warp_buf2 = warp_buf1 + VPTX * BDIMY * row_stride;

  for (int idx = threadIdx.y * blockDim.x + threadIdx.x;
       idx < 2 * VPTX * BDIMY * row_stride; idx += BDIMX * BDIMY) {
    buf[idx] = U(0);
  }
  __syncthreads();

  for (int64_t i1_block = blockIdx.y * BDIMY * VPTX; i1_block < n1;
       i1_block += VPTX * BDIMY * gridDim.y) {
    cuLoadAddStridedInputs<T, U, VPTX>(
        i1_block, thr_load_row_off, thr_load_col_off, i2_off, row_stride,
        warp_buf1, warp_buf2, input, dout, n1, n2, mean, var, epsilon);
  }
  __syncthreads();

  // inter-warp reductions
  // sum within each warp
  U acc1 = U(0);
  U acc2 = U(0);
  for (int k = 0; k < VPTX; ++k) {
    int row1 = threadIdx.y + k * VPTX;
    int idx1 = row1 * row_stride + threadIdx.x;
    acc1 += warp_buf1[idx1];
    acc2 += warp_buf2[idx1];
  }
  warp_buf1[threadIdx.y * row_stride + threadIdx.x] = acc1;
  warp_buf2[threadIdx.y * row_stride + threadIdx.x] = acc2;
  __syncthreads();
  // sum all warps
  for (int offset = VPTX >> 1; offset > 1; offset >>= 1) {
    if (threadIdx.y < offset) {
      int row1 = threadIdx.y;
      int row2 = threadIdx.y + offset;
      int idx1 = row1 * row_stride + threadIdx.x;
      int idx2 = row2 * row_stride + threadIdx.x;
      warp_buf1[idx1] += warp_buf1[idx2];
      warp_buf2[idx1] += warp_buf2[idx2];
    }
    __syncthreads();
  }
  int64_t i2 = blockIdx.x * blockDim.x + threadIdx.x;
  if (threadIdx.y == 0 && i2 < n2) {
    int row1 = threadIdx.y;
    int row2 = threadIdx.y + 1;
    int idx1 = row1 * row_stride + threadIdx.x;
    int idx2 = row2 * row_stride + threadIdx.x;
    part_grad_beta[blockIdx.y * n2 + i2] = warp_buf1[idx1] + warp_buf1[idx2];
    part_grad_gamma[blockIdx.y * n2 + i2] = warp_buf2[idx1] + warp_buf2[idx2];
  }
}

template <typename T, typename U, int BDIMX, int BDIMY, bool ScaleBiasSameTypeX>
__global__ void LayerNormBackwardSumGradGammaBeta(
    const U *part_grad_gamma, const U *part_grad_beta, const int part_size,
    // const int n1, const int n2, T* grad_gamma, T* grad_beta) {
    const int n1, const int n2,
    LayerNormScaleBiasT<T, U, ScaleBiasSameTypeX> *grad_gamma,
    LayerNormScaleBiasT<T, U, ScaleBiasSameTypeX> *grad_beta) {
  // sum partial gradients for gamma and beta
  using ScaleBiasT = LayerNormScaleBiasT<T, U, ScaleBiasSameTypeX>;
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
      grad_gamma[i2] = static_cast<ScaleBiasT>(sum_gamma);
      grad_beta[i2] = static_cast<ScaleBiasT>(sum_beta);
    }
  }
}

template <typename T, typename U, int BDIMX, int BDIMY, bool ScaleBiasSameTypeX>
__global__ void LayerNormBackwardComputeGradInput(
    const T *__restrict__ dout, const T *__restrict__ input, const int n1,
    const int n2, const U *__restrict__ mean, const U *__restrict__ var,
    const float epsilon,
    const LayerNormScaleBiasT<T, U, ScaleBiasSameTypeX> *gamma, T *grad_input) {
#ifdef __HIPCC__
  for (auto i1 = hipBlockIdx_x; i1 < n1; i1 += hipGridDim_x) {
#else
  for (auto i1 = blockIdx.x; i1 < n1; i1 += gridDim.x) {
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
    for (int mask = BDIMX / 2; mask > 0; mask /= 2) {
#ifdef PADDLE_WITH_HIP
      sum_loss1 += __shfl_xor(sum_loss1, mask,
                              warpSize);  // WARP_SHFL_XOR(sum_loss1, mask);
      sum_loss2 += __shfl_xor(sum_loss2, mask,
                              warpSize);  // WARP_SHFL_XOR(sum_loss2, mask);
#else
      sum_loss1 +=
          __shfl_xor_sync(0xffffffff, sum_loss1, mask,
                          warpSize);  // WARP_SHFL_XOR(sum_loss1, mask);
      sum_loss2 +=
          __shfl_xor_sync(0xffffffff, sum_loss2, mask,
                          warpSize);  // WARP_SHFL_XOR(sum_loss2, mask);
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

// Make sure that d_scale != nullptr && d_bias != nullptr
// Since d_scale != nullptr, scale would not be nullptr
template <typename T, typename U, int BlockDim, bool HasDx,
          bool ScaleBiasWithSameTypeX>
__global__ void LayerNormBackwardGradientAll(
    const T *x, const T *d_y,
    LayerNormScaleBiasT<T, U, ScaleBiasWithSameTypeX> *d_scale,
    LayerNormScaleBiasT<T, U, ScaleBiasWithSameTypeX> *d_bias, T *d_x,
    const U *mean, const U *var,
    const LayerNormScaleBiasT<T, U, ScaleBiasWithSameTypeX> *scale,
    float epsilon, int64_t batch_size, int64_t feature_size,
    int64_t col_offset) {
  using ScaleBiasT = LayerNormScaleBiasT<T, U, ScaleBiasWithSameTypeX>;
  int64_t beg_idx = threadIdx.x * feature_size + (blockIdx.x + col_offset);
  int64_t end_idx = batch_size * feature_size + (blockIdx.x + col_offset);
  int64_t stride = BlockDim * feature_size;

  U d_scale_partial = static_cast<U>(0), d_bias_partial = static_cast<U>(0);

  for (int64_t i = beg_idx; i < end_idx; i += stride) {
    int row_idx = i / feature_size;
    auto var_val = real_sqrt(static_cast<U>(var[row_idx]) + epsilon);
    d_scale_partial += static_cast<U>(d_y[i]) *
                       (static_cast<U>(x[i]) - mean[row_idx]) / var_val;
    d_bias_partial += static_cast<U>(d_y[i]);
    if (HasDx) {
      d_x[i] = static_cast<T>(static_cast<U>(d_y[i]) *
                              static_cast<U>(scale[blockIdx.x + col_offset]) /
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
template <typename T, typename U, int BlockDim, bool HasDx, bool HasDScale,
          bool ScaleBiasWithSameTypeX>
__global__ void LayerNormBackwardGradientScaleOrBias(
    const T *x, const T *d_y,
    LayerNormScaleBiasT<T, U, ScaleBiasWithSameTypeX> *d_scale,
    LayerNormScaleBiasT<T, U, ScaleBiasWithSameTypeX> *d_bias, T *d_x,
    const U *mean, const U *var,
    const LayerNormScaleBiasT<T, U, ScaleBiasWithSameTypeX> *scale,
    float epsilon, int64_t batch_size, int64_t feature_size, int col_offset) {
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
        static_cast<U>(real_sqrt(static_cast<float>(var[row_idx]) + epsilon));
    if (HasDScale) {
      d_scale_or_d_bias_partial += static_cast<U>(d_y[i]) *
                                   (static_cast<U>(x[i]) - mean[row_idx]) /
                                   var_val;
    } else {  // d_bias != nullptr
      d_scale_or_d_bias_partial += static_cast<U>(d_y[i]);
    }

    if (HasDx) {
      if (scale != nullptr) {
        d_x[i] = static_cast<T>(static_cast<U>(d_y[i]) *
                                static_cast<U>(scale[blockIdx.x + col_offset]) /
                                var_val);
      } else {
        d_x[i] = static_cast<T>(static_cast<U>(d_y[i]) / var_val);
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
    const T *x, T *d_x, const U *mean, const U *var, float epsilon,
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
    const T *x, const T *d_y, T *d_x, const U *mean, const U *var,
    const LayerNormScaleBiasT<T, U, ScaleBiasWithSameTypeX> *scale,
    float epsilon, int64_t feature_size) {
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
        static_cast<U>(real_sqrt(static_cast<float>(block_var) + epsilon));
    if (scale != nullptr) {
      int col_idx = i % feature_size;
      d_x[i] = static_cast<T>(static_cast<U>(d_y[i]) *
                              static_cast<U>(scale[col_idx]) / var_val);
    } else {
      d_x[i] = static_cast<T>(static_cast<U>(d_y[i]) / var_val);
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
    const T *x, const T *d_y, T *d_x,
    LayerNormScaleBiasT<T, U, ScaleBiasWithSameTypeX> *d_scale,
    LayerNormScaleBiasT<T, U, ScaleBiasWithSameTypeX> *d_bias, const U *mean,
    const U *var,
    const LayerNormScaleBiasT<T, U, ScaleBiasWithSameTypeX> *scale,
    float epsilon, int64_t feature_size) {
  int64_t idx = threadIdx.x + blockIdx.x * blockDim.x;
  using ScaleBiasT = LayerNormScaleBiasT<T, U, ScaleBiasWithSameTypeX>;
  if (idx < feature_size) {
    auto var_val =
        static_cast<U>(real_sqrt(static_cast<float>(var[0]) + epsilon));
    if (d_x != nullptr) {
      if (d_scale == nullptr) {
        d_x[idx] = static_cast<T>(static_cast<U>(d_y[idx]) / var_val);
      } else {
        d_x[idx] = static_cast<T>(static_cast<U>(d_y[idx]) *
                                  static_cast<U>(scale[idx]) / var_val);
      }
    }

    if (d_scale != nullptr) {
      d_scale[idx] =
          static_cast<ScaleBiasT>(static_cast<U>(d_y[idx]) *
                                  (static_cast<U>(x[idx]) - mean[0]) / var_val);
    }

    if (d_bias != nullptr) {
      d_bias[idx] = static_cast<ScaleBiasT>(d_y[idx]);
    }
  }
}

template <typename T, typename U, bool ScaleBiasWithSameTypeX = false>
static void LayerNormBackward(
    const T *x, const T *d_y,
    const LayerNormScaleBiasT<T, U, ScaleBiasWithSameTypeX> *scale,
    const U *mean, const U *var, T *d_x,
    LayerNormScaleBiasT<T, U, ScaleBiasWithSameTypeX> *d_scale,
    LayerNormScaleBiasT<T, U, ScaleBiasWithSameTypeX> *d_bias, float epsilon,
    int64_t batch_size, int64_t feature_size,
    const platform::CUDADeviceContext &dev_ctx) {
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
    LayerNormBackwardWhenBatchSizeIsOne<T, U, ScaleBiasWithSameTypeX><<<
        (feature_size + kMaxBlockDim - 1) / kMaxBlockDim, kMaxBlockDim, 0,
        stream>>>(x, d_y, d_x, d_scale, d_bias, mean, var, scale, epsilon,
                  feature_size);

    if (d_x != nullptr) {
      switch (GetDesiredBlockDim(feature_size)) {
        FIXED_BLOCK_DIM_CASE(LayerNormBackwardPostProcessToCalculateDX<
                             T, U, kBlockDim><<<1, kBlockDim, 0, stream>>>(
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
            feature_size, kMaxBlockNum,
            LayerNormBackwardGradientScaleOrBias<
                T, U, kBlockDim, false, false,
                ScaleBiasWithSameTypeX><<<block_num, kBlockDim, 0, stream>>>(
                x, d_y, d_scale, d_bias, d_x, mean, var, scale, epsilon,
                batch_size, feature_size, col_offset));
      }
      break;
    case 2:  // d_x == nullptr, d_scale != nullptr, d_bias == nullptr
      switch (block_dim) {
        FIXED_BLOCK_DIM_FIXED_BLOCK_NUM_CASE(
            feature_size, kMaxBlockNum,
            LayerNormBackwardGradientScaleOrBias<
                T, U, kBlockDim, false, true,
                ScaleBiasWithSameTypeX><<<block_num, kBlockDim, 0, stream>>>(
                x, d_y, d_scale, d_bias, d_x, mean, var, scale, epsilon,
                batch_size, feature_size, col_offset));
      }
      break;
    case 3:  // d_x == nullptr, d_scale != nulptr, d_bias != nullptr
      switch (block_dim) {
        FIXED_BLOCK_DIM_FIXED_BLOCK_NUM_CASE(
            feature_size, kMaxBlockNum,
            LayerNormBackwardGradientAll<
                T, U, kBlockDim, false,
                ScaleBiasWithSameTypeX><<<block_num, kBlockDim, 0, stream>>>(
                x, d_y, d_scale, d_bias, d_x, mean, var, scale, epsilon,
                batch_size, feature_size, col_offset));
      }
      break;
    case 4:  // d_x != nullptr, d_scale == nullptr, d_bias == nullptr
      switch (GetDesiredBlockDim(feature_size)) {
        FIXED_BLOCK_DIM_CASE(
            LayerNormBackwardGradientOnlyDX<
                T, U, kBlockDim,
                ScaleBiasWithSameTypeX><<<batch_size, kBlockDim, 0, stream>>>(
                x, d_y, d_x, mean, var, scale, epsilon, feature_size));
      }
      break;
    case 5:  // d_x != nulptr, d_scale == nullptr, d_bias != nullptr
      switch (block_dim) {
        FIXED_BLOCK_DIM_FIXED_BLOCK_NUM_CASE(
            feature_size, kMaxBlockNum,
            LayerNormBackwardGradientScaleOrBias<
                T, U, kBlockDim, true, false,
                ScaleBiasWithSameTypeX><<<block_num, kBlockDim, 0, stream>>>(
                x, d_y, d_scale, d_bias, d_x, mean, var, scale, epsilon,
                batch_size, feature_size, col_offset));
      }
      switch (GetDesiredBlockDim(feature_size)) {
        FIXED_BLOCK_DIM_CASE(
            LayerNormBackwardPostProcessToCalculateDX<
                T, U, kBlockDim><<<batch_size, kBlockDim, 0, stream>>>(
                x, d_x, mean, var, epsilon, feature_size));
      }
      break;
    case 6:  // d_x != nullptr, d_scale != nullptr, d_bias == nullptr
      switch (block_dim) {
        FIXED_BLOCK_DIM_FIXED_BLOCK_NUM_CASE(
            feature_size, kMaxBlockNum,
            LayerNormBackwardGradientScaleOrBias<
                T, U, kBlockDim, true, true,
                ScaleBiasWithSameTypeX><<<block_num, kBlockDim, 0, stream>>>(
                x, d_y, d_scale, d_bias, d_x, mean, var, scale, epsilon,
                batch_size, feature_size, col_offset));
      }
      switch (GetDesiredBlockDim(feature_size)) {
        FIXED_BLOCK_DIM_CASE(
            LayerNormBackwardPostProcessToCalculateDX<
                T, U, kBlockDim><<<batch_size, kBlockDim, 0, stream>>>(
                x, d_x, mean, var, epsilon, feature_size));
      }
      break;
    case 7:  // d_x != nullptr, d_scale != nullptr, d_bias != nullptr
    {
      constexpr int VPT = 4;
      constexpr int BDIMX2 = 32;
      constexpr int BDIMY2 = 4;
      dim3 threads2(BDIMX2, BDIMY2, 1);
      constexpr int part_size = BDIMY2 * VPT;
      const dim3 blocks2((feature_size + BDIMX2 - 1) / BDIMX2, part_size, 1);

      auto part_grad_gamma_ptr =
          memory::Alloc(dev_ctx, part_size * feature_size * sizeof(U));
      auto part_grad_beta_ptr =
          memory::Alloc(dev_ctx, part_size * feature_size * sizeof(U));
      U *part_grad_gamma = reinterpret_cast<U *>(part_grad_gamma_ptr->ptr());
      U *part_grad_beta = reinterpret_cast<U *>(part_grad_beta_ptr->ptr());

      LayerNormBackwardPartGradGammaBeta<T, U, BDIMX2, BDIMY2,
                                         VPT><<<blocks2, threads2, 0, stream>>>(
          d_y, x, batch_size, feature_size, mean, var, epsilon, part_grad_gamma,
          part_grad_beta);  // compute part_grad_gamma, beta

      constexpr int BDIMX3 = 32;
      constexpr int BDIMY3 = 8;
      dim3 threads3(BDIMX3, BDIMY3, 1);
      const dim3 blocks3((feature_size + BDIMX2 - 1) / BDIMX2, 1, 1);
      LayerNormBackwardSumGradGammaBeta<
          T, U, BDIMX3, BDIMY3,
          ScaleBiasWithSameTypeX><<<blocks3, threads3, 0, stream>>>(
          part_grad_gamma, part_grad_beta, part_size, batch_size, feature_size,
          d_scale, d_bias);

      constexpr int BDIMX1 = 32;
      constexpr int BDIMY1 = 4;
      dim3 threads1(BDIMX1, BDIMY1, 1);
      LayerNormBackwardComputeGradInput<
          T, U, BDIMX1, BDIMY1,
          ScaleBiasWithSameTypeX><<<batch_size, threads1, 0, stream>>>(
          d_y, x, batch_size, feature_size, mean, var, epsilon, scale, d_x);
      break;
    }
    default:
      break;
  }
}

}  // namespace operators
}  // namespace paddle
