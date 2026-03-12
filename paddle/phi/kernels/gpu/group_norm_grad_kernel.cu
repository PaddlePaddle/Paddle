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

#include "paddle/phi/kernels/group_norm_grad_kernel.h"

#include "paddle/common/layout.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/backends/gpu/gpu_device_function.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/full_kernel.h"
#include "paddle/phi/kernels/funcs/math_function.h"
#include "paddle/phi/kernels/gpu/group_norm_utils.h"

namespace phi {

// ============================================================================
// NHWC backward kernels (kept unchanged for NHWC data layout)
// ============================================================================

template <typename T, typename AccT, int flags>
__global__ void GroupNormBackwardGetMeanAndVar(const T* x,
                                               const T* scale,
                                               const T* bias,
                                               const T* d_y,
                                               int64_t N,
                                               int64_t C,
                                               int64_t W,
                                               int64_t imsize,
                                               int groups,
                                               int64_t group_size,
                                               double epsilon,
                                               AccT* d_mean,
                                               AccT* d_var,
                                               T* d_scale,
                                               T* d_bias) {
  int gid = blockIdx.y;
  for (int64_t cid = blockIdx.x; cid < group_size; cid += gridDim.x) {
    for (int64_t bid = blockIdx.z; bid < N; bid += gridDim.z) {
      int64_t H = imsize / W;
      int64_t number = min(group_size, C - gid * group_size);
      int64_t ccid = gid * group_size + cid;
      if (ccid >= C) return;
      T x_scale = (flags & kHasScale) ? scale[ccid] : static_cast<T>(1);
      T x_bias = (flags & kHasBias) ? bias[ccid] : static_cast<T>(0);
      T x_scale_inv = static_cast<T>(0);
      if (x_scale != static_cast<T>(0))
        x_scale_inv = static_cast<T>(1.0) / x_scale;
      AccT d_mean_data = static_cast<AccT>(0);
      AccT d_var_data = static_cast<AccT>(0);
      AccT d_scale_data = static_cast<AccT>(0);
      AccT d_bias_data = static_cast<AccT>(0);

      for (int64_t imid = threadIdx.x; imid < imsize; imid += blockDim.x) {
        AccT val, dval;

        int64_t hid = imid / W;
        int64_t wid = imid % W;
        val = static_cast<AccT>(x[(bid * H + hid) * W * C + wid * C + ccid]) -
              static_cast<AccT>(x_bias);
        dval = static_cast<AccT>(d_y[(bid * H + hid) * W * C + wid * C + ccid]);

        d_var_data += val * dval;
        d_mean_data += dval * static_cast<AccT>(x_scale);

        val = val * static_cast<AccT>(x_scale_inv);
        d_bias_data += dval;
        d_scale_data += val * dval;
      }
      CudaAtomicAddWithWarp(&(d_mean[bid * groups + gid]),
                            static_cast<AccT>(d_mean_data));
      CudaAtomicAddWithWarp(&(d_var[bid * groups + gid]),
                            static_cast<AccT>(d_var_data));

      if (flags & kHasScale) {
        CudaAtomicAdd(&(d_scale[ccid]), static_cast<T>(d_scale_data));
      }
      if (flags & kHasBias) {
        CudaAtomicAdd(&(d_bias[ccid]), static_cast<T>(d_bias_data));
      }
    }
  }
}

template <typename T, typename AccT, int flags>
__global__ void GroupNormBackward(const T* x,
                                  const T* d_y,
                                  const T* scale,
                                  const T* bias,
                                  const AccT* var,
                                  const AccT* d_mean,
                                  const AccT* d_var,
                                  int64_t N,
                                  int64_t C,
                                  int64_t W,
                                  int64_t imsize,
                                  int groups,
                                  int64_t group_size,
                                  double epsilon,
                                  T* d_x) {
  int gid = blockIdx.y;
  for (int64_t cid = blockIdx.x; cid < group_size; cid += gridDim.x) {
    for (int64_t bid = blockIdx.z; bid < N; bid += gridDim.z) {
      int64_t H = imsize / W;
      int64_t number = min(group_size, C - gid * group_size);
      int64_t ccid = gid * group_size + cid;
      if (ccid >= C) return;
      AccT x_var = var[bid * groups + gid];
      AccT d_x_mean = static_cast<AccT>(d_mean[bid * groups + gid]);
      AccT d_x_var = static_cast<AccT>(d_var[bid * groups + gid]);

      AccT x_var_inv = static_cast<AccT>(1.0) / sqrt((x_var) + epsilon);
      AccT number_inv =
          static_cast<AccT>(1.0) / static_cast<AccT>((number * imsize));

      AccT x_scale = (flags & kHasScale) ? static_cast<AccT>(scale[ccid])
                                         : static_cast<AccT>(1);
      AccT x_bias = (flags & kHasBias) ? static_cast<AccT>(bias[ccid])
                                       : static_cast<AccT>(0);
      AccT x_scale_inv = static_cast<AccT>(0);
      if (x_scale != static_cast<AccT>(0))
        x_scale_inv = static_cast<AccT>(1.0) / x_scale;

      for (int64_t imid = threadIdx.x; imid < imsize; imid += blockDim.x) {
        int64_t hid = imid / W;
        int64_t wid = imid % W;
        AccT tmp =
            static_cast<AccT>(x[(bid * H + hid) * W * C + wid * C + ccid]);
        AccT v_y = (tmp - x_bias) * x_scale_inv;
        AccT dly =
            static_cast<AccT>(d_y[(bid * H + hid) * W * C + wid * C + ccid]);
        d_x[(bid * H + hid) * W * C + wid * C + ccid] =
            static_cast<T>(x_var_inv * ((dly) * (x_scale)-number_inv * d_x_var *
                                        (v_y)-number_inv * d_x_mean));
      }
    }
  }
}

// ============================================================================
// PyTorch-aligned NCHW backward implementation
// ============================================================================

// Constant matching PyTorch's kCUDABlockReduceNumThreads
constexpr int kGradBlockReduceNumThreads = 512;
constexpr int kGradReduceTileSize = 32;

// Warp reduce sum using shuffle-down (matching PyTorch's WarpReduceSum)
template <typename AccT>
__device__ __forceinline__ AccT GradWarpReduceSum(AccT val) {
#pragma unroll
  for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
    val += phi::backends::gpu::CudaShuffleDownSync(0xffffffff, val, offset);
  }
  return val;
}

// Block reduce sum (matching PyTorch's BlockReduceSum)
template <typename AccT>
__device__ __forceinline__ AccT GradBlockReduceSum(AccT val, AccT* shared) {
  const int tid = threadIdx.x;
  const int lid = tid % warpSize;
  const int wid = tid / warpSize;
  const int num_warps = blockDim.x / warpSize;
  val = GradWarpReduceSum(val);
  __syncthreads();
  if (lid == 0) {
    shared[wid] = val;
  }
  __syncthreads();
  val = (tid < num_warps) ? shared[lid] : AccT(0);
  if (wid == 0) {
    val = GradWarpReduceSum(val);
  }
  return val;
}

// Stage 1: Compute internal gradients ds[n,c] and db[n,c]
// ds[nc] = sum_hw(dY * X), db[nc] = sum_hw(dY)
// Grid: N*C blocks. Matches PyTorch's ComputeInternalGradientsCUDAKernel.
template <typename T, typename AccT>
__global__ void ComputeInternalGradientsCUDAKernel(
    int64_t HxW, const T* dY, const T* X, AccT* ds, AccT* db) {
  const int64_t nc = blockIdx.x;
  AccT sum1 = 0;
  AccT sum2 = 0;
  for (int64_t hw = threadIdx.x; hw < HxW; hw += blockDim.x) {
    const int64_t index = nc * HxW + hw;
    sum1 += static_cast<AccT>(dY[index]) * static_cast<AccT>(X[index]);
    sum2 += static_cast<AccT>(dY[index]);
  }
  if (blockDim.x <= static_cast<unsigned>(warpSize)) {
    sum1 = GradWarpReduceSum(sum1);
    sum2 = GradWarpReduceSum(sum2);
  } else {
    constexpr int kMaxWarps = 32;
    __shared__ AccT ds_shared[kMaxWarps];
    __shared__ AccT db_shared[kMaxWarps];
    sum1 = GradBlockReduceSum(sum1, ds_shared);
    sum2 = GradBlockReduceSum(sum2, db_shared);
  }
  if (threadIdx.x == 0) {
    ds[nc] = sum1;
    db[nc] = sum2;
  }
}

// Stage 1 vectorized: float4 version for T=float, HxW divisible by 4
// Processes 4 elements per thread per iteration for better memory throughput.
template <typename T, typename AccT>
__global__ void ComputeInternalGradientsVec4CUDAKernel(
    int64_t HxW_vec, const T* dY, const T* X, AccT* ds, AccT* db) {
  const int64_t nc = blockIdx.x;
  AccT sum1 = 0;
  AccT sum2 = 0;
  const float4* dY_vec = reinterpret_cast<const float4*>(dY + nc * HxW_vec * 4);
  const float4* X_vec = reinterpret_cast<const float4*>(X + nc * HxW_vec * 4);
  for (int64_t i = threadIdx.x; i < HxW_vec; i += blockDim.x) {
    float4 dy4 = dY_vec[i];
    float4 x4 = X_vec[i];
    sum1 += static_cast<AccT>(dy4.x) * static_cast<AccT>(x4.x);
    sum1 += static_cast<AccT>(dy4.y) * static_cast<AccT>(x4.y);
    sum1 += static_cast<AccT>(dy4.z) * static_cast<AccT>(x4.z);
    sum1 += static_cast<AccT>(dy4.w) * static_cast<AccT>(x4.w);
    sum2 += static_cast<AccT>(dy4.x);
    sum2 += static_cast<AccT>(dy4.y);
    sum2 += static_cast<AccT>(dy4.z);
    sum2 += static_cast<AccT>(dy4.w);
  }
  if (blockDim.x <= static_cast<unsigned>(warpSize)) {
    sum1 = GradWarpReduceSum(sum1);
    sum2 = GradWarpReduceSum(sum2);
  } else {
    constexpr int kMaxWarps = 32;
    __shared__ AccT ds_shared[kMaxWarps];
    __shared__ AccT db_shared[kMaxWarps];
    sum1 = GradBlockReduceSum(sum1, ds_shared);
    sum2 = GradBlockReduceSum(sum2, db_shared);
  }
  if (threadIdx.x == 0) {
    ds[nc] = sum1;
    db[nc] = sum2;
  }
}

// Stage 1 vectorized: double2 version for T=double, HxW divisible by 2
template <typename T, typename AccT>
__global__ void ComputeInternalGradientsVec2DoubleCUDAKernel(
    int64_t HxW_vec, const T* dY, const T* X, AccT* ds, AccT* db) {
  const int64_t nc = blockIdx.x;
  AccT sum1 = 0;
  AccT sum2 = 0;
  const double2* dY_vec =
      reinterpret_cast<const double2*>(dY + nc * HxW_vec * 2);
  const double2* X_vec = reinterpret_cast<const double2*>(X + nc * HxW_vec * 2);
  for (int64_t i = threadIdx.x; i < HxW_vec; i += blockDim.x) {
    double2 dy2 = dY_vec[i];
    double2 x2 = X_vec[i];
    sum1 += static_cast<AccT>(dy2.x) * static_cast<AccT>(x2.x);
    sum1 += static_cast<AccT>(dy2.y) * static_cast<AccT>(x2.y);
    sum2 += static_cast<AccT>(dy2.x);
    sum2 += static_cast<AccT>(dy2.y);
  }
  if (blockDim.x <= static_cast<unsigned>(warpSize)) {
    sum1 = GradWarpReduceSum(sum1);
    sum2 = GradWarpReduceSum(sum2);
  } else {
    constexpr int kMaxWarps = 32;
    __shared__ AccT ds_shared[kMaxWarps];
    __shared__ AccT db_shared[kMaxWarps];
    sum1 = GradBlockReduceSum(sum1, ds_shared);
    sum2 = GradBlockReduceSum(sum2, db_shared);
  }
  if (threadIdx.x == 0) {
    ds[nc] = sum1;
    db[nc] = sum2;
  }
}

// Stage 2: Compute backward fused params c2[n,g] and c3[n,g]
// Reduces ds*gamma and db*gamma across channels in group.
// Grid: dim3(N, G). Matches PyTorch's ComputeBackwardFusedParamsCUDAKernel.
// Rounds mean/rstd through T to match PyTorch's precision behavior.
template <typename T, typename AccT>
__global__ void ComputeBackwardFusedParamsCUDAKernel(int64_t C,
                                                     int64_t HxW,
                                                     int64_t G,
                                                     const AccT* mean,
                                                     const AccT* var,
                                                     const T* gamma,
                                                     const AccT* ds,
                                                     const AccT* db,
                                                     AccT eps,
                                                     AccT* c2,
                                                     AccT* c3) {
  const int64_t D = C / G;
  const int64_t n = blockIdx.x;
  const int64_t g = blockIdx.y;
  const int64_t ng = n * G + g;
  AccT sum1 = 0;
  AccT sum2 = 0;
  for (int64_t i = threadIdx.x; i < D; i += blockDim.x) {
    const int64_t index = ng * D + i;
    const int64_t c_idx = g * D + i;
    const AccT gamma_v =
        gamma == nullptr ? AccT(1) : static_cast<AccT>(gamma[c_idx]);
    sum1 += ds[index] * gamma_v;
    sum2 += db[index] * gamma_v;
  }
  if (blockDim.x <= static_cast<unsigned>(warpSize)) {
    sum1 = GradWarpReduceSum(sum1);
    sum2 = GradWarpReduceSum(sum2);
  } else {
    constexpr int kMaxWarps = 32;
    __shared__ AccT ds_shared[kMaxWarps];
    __shared__ AccT db_shared[kMaxWarps];
    sum1 = GradBlockReduceSum(sum1, ds_shared);
    sum2 = GradBlockReduceSum(sum2, db_shared);
  }
  if (threadIdx.x == 0) {
    // Round mean and rstd through T to match PyTorch
    AccT mean_val = static_cast<AccT>(static_cast<T>(mean[ng]));
    AccT rstd_val = static_cast<AccT>(static_cast<T>(rsqrt(var[ng] + eps)));
    const AccT s = AccT(1) / static_cast<AccT>(D * HxW);
    const AccT x =
        (sum2 * mean_val - sum1) * rstd_val * rstd_val * rstd_val * s;
    c2[ng] = x;
    c3[ng] = -x * mean_val - sum2 * rstd_val * s;
  }
}

// ============================================================================
// Optimized dX kernels for NCHW backward path
// ============================================================================

// Stage 3: Pre-compute c1[n,c] = rstd[n,g] * gamma[c]
// This removes per-element rsqrt and gamma lookup from the inner dX loop.
// Grid: ceil(N*C / 256) blocks.
// Rounds rstd through T to match PyTorch's precision behavior.
template <typename T, typename AccT>
__global__ void ComputeC1CUDAKernel(int64_t N_C,
                                    int64_t C,
                                    int64_t G,
                                    AccT eps,
                                    const AccT* var,
                                    const T* gamma,
                                    AccT* c1) {
  const int64_t D = C / G;
  for (int64_t index =
           static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < N_C;
       index += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int64_t ng = index / D;
    const int64_t c = index % C;
    // Round rstd through T to match PyTorch
    AccT rstd_val = static_cast<AccT>(static_cast<T>(rsqrt(var[ng] + eps)));
    AccT gamma_val = (gamma != nullptr) ? static_cast<AccT>(gamma[c]) : AccT(1);
    c1[index] = rstd_val * gamma_val;
  }
}

// Stage 4 optimized: Vectorized dX kernel using float4
// dX[idx] = c1[nc] * dY[idx] + c2[ng] * X[idx] + c3[ng]
// Processes 4 float elements at a time within each (n,c) spatial plane.
template <typename T, typename AccT>
__global__ void GroupNormBackwardDxVec4CUDAKernel(int64_t N_C,
                                                  int64_t HxW_vec,
                                                  int64_t D,
                                                  const T* __restrict__ dY,
                                                  const T* __restrict__ X,
                                                  const AccT* __restrict__ c1,
                                                  const AccT* __restrict__ c2,
                                                  const AccT* __restrict__ c3,
                                                  T* __restrict__ dX) {
  const int64_t total_vec = N_C * HxW_vec;
  for (int64_t vidx =
           static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       vidx < total_vec;
       vidx += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int64_t nc = vidx / HxW_vec;
    const int64_t ng = nc / D;

    const AccT c1v = c1[nc];
    const AccT c2v = c2[ng];
    const AccT c3v = c3[ng];

    // Vectorized load: 4 consecutive float elements
    float4 dy4 = reinterpret_cast<const float4*>(dY)[vidx];
    float4 x4 = reinterpret_cast<const float4*>(X)[vidx];

    float4 dx4;
    dx4.x = static_cast<T>(c1v * static_cast<AccT>(dy4.x) +
                           c2v * static_cast<AccT>(x4.x) + c3v);
    dx4.y = static_cast<T>(c1v * static_cast<AccT>(dy4.y) +
                           c2v * static_cast<AccT>(x4.y) + c3v);
    dx4.z = static_cast<T>(c1v * static_cast<AccT>(dy4.z) +
                           c2v * static_cast<AccT>(x4.z) + c3v);
    dx4.w = static_cast<T>(c1v * static_cast<AccT>(dy4.w) +
                           c2v * static_cast<AccT>(x4.w) + c3v);

    reinterpret_cast<float4*>(dX)[vidx] = dx4;
  }
}

// Stage 4 optimized: Vectorized dX kernel using double2 for float64
template <typename T, typename AccT>
__global__ void GroupNormBackwardDxVec2DoubleCUDAKernel(
    int64_t N_C,
    int64_t HxW_vec,
    int64_t D,
    const T* __restrict__ dY,
    const T* __restrict__ X,
    const AccT* __restrict__ c1,
    const AccT* __restrict__ c2,
    const AccT* __restrict__ c3,
    T* __restrict__ dX) {
  const int64_t total_vec = N_C * HxW_vec;
  for (int64_t vidx =
           static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       vidx < total_vec;
       vidx += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int64_t nc = vidx / HxW_vec;
    const int64_t ng = nc / D;

    const AccT c1v = c1[nc];
    const AccT c2v = c2[ng];
    const AccT c3v = c3[ng];

    // Vectorized load: 2 consecutive double elements
    double2 dy2 = reinterpret_cast<const double2*>(dY)[vidx];
    double2 x2 = reinterpret_cast<const double2*>(X)[vidx];

    double2 dx2;
    dx2.x = static_cast<T>(c1v * static_cast<AccT>(dy2.x) +
                           c2v * static_cast<AccT>(x2.x) + c3v);
    dx2.y = static_cast<T>(c1v * static_cast<AccT>(dy2.y) +
                           c2v * static_cast<AccT>(x2.y) + c3v);

    reinterpret_cast<double2*>(dX)[vidx] = dx2;
  }
}

// Stage 4 optimized: Scalar fallback for when HxW is not divisible by vec_size
// Uses pre-computed c1[n,c].
template <typename T, typename AccT>
__global__ void GroupNormBackwardDxOptCUDAKernel(int64_t N_C,
                                                 int64_t HxW,
                                                 int64_t D,
                                                 const T* __restrict__ dY,
                                                 const T* __restrict__ X,
                                                 const AccT* __restrict__ c1,
                                                 const AccT* __restrict__ c2,
                                                 const AccT* __restrict__ c3,
                                                 T* __restrict__ dX) {
  const int64_t total = N_C * HxW;
  for (int64_t idx =
           static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < total;
       idx += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int64_t nc = idx / HxW;
    const int64_t ng = nc / D;
    dX[idx] = static_cast<T>(c1[nc] * static_cast<AccT>(dY[idx]) +
                             c2[ng] * static_cast<AccT>(X[idx]) + c3[ng]);
  }
}

// Stage 4 optimized: Vectorized no-gamma dX kernel using float4
// When gamma is null, c1 = rstd[ng], constant across D*HxW per (n,g) group.
template <typename T, typename AccT>
__global__ void GroupNormBackwardDxNoGammaVec4CUDAKernel(
    int64_t N_G,
    int64_t D_HxW_vec,
    const T* __restrict__ dY,
    const T* __restrict__ X,
    const AccT* __restrict__ c1_ng,
    const AccT* __restrict__ c2,
    const AccT* __restrict__ c3,
    T* __restrict__ dX) {
  const int64_t total_vec = N_G * D_HxW_vec;
  for (int64_t vidx =
           static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       vidx < total_vec;
       vidx += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int64_t ng = vidx / D_HxW_vec;

    const AccT rstd_val = c1_ng[ng];
    const AccT c2v = c2[ng];
    const AccT c3v = c3[ng];

    float4 dy4 = reinterpret_cast<const float4*>(dY)[vidx];
    float4 x4 = reinterpret_cast<const float4*>(X)[vidx];

    float4 dx4;
    dx4.x = static_cast<T>(rstd_val * static_cast<AccT>(dy4.x) +
                           c2v * static_cast<AccT>(x4.x) + c3v);
    dx4.y = static_cast<T>(rstd_val * static_cast<AccT>(dy4.y) +
                           c2v * static_cast<AccT>(x4.y) + c3v);
    dx4.z = static_cast<T>(rstd_val * static_cast<AccT>(dy4.z) +
                           c2v * static_cast<AccT>(x4.z) + c3v);
    dx4.w = static_cast<T>(rstd_val * static_cast<AccT>(dy4.w) +
                           c2v * static_cast<AccT>(x4.w) + c3v);

    reinterpret_cast<float4*>(dX)[vidx] = dx4;
  }
}

// Stage 4 optimized: Vectorized no-gamma dX kernel using double2
template <typename T, typename AccT>
__global__ void GroupNormBackwardDxNoGammaVec2DoubleCUDAKernel(
    int64_t N_G,
    int64_t D_HxW_vec,
    const T* __restrict__ dY,
    const T* __restrict__ X,
    const AccT* __restrict__ c1_ng,
    const AccT* __restrict__ c2,
    const AccT* __restrict__ c3,
    T* __restrict__ dX) {
  const int64_t total_vec = N_G * D_HxW_vec;
  for (int64_t vidx =
           static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       vidx < total_vec;
       vidx += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int64_t ng = vidx / D_HxW_vec;

    const AccT rstd_val = c1_ng[ng];
    const AccT c2v = c2[ng];
    const AccT c3v = c3[ng];

    double2 dy2 = reinterpret_cast<const double2*>(dY)[vidx];
    double2 x2 = reinterpret_cast<const double2*>(X)[vidx];

    double2 dx2;
    dx2.x = static_cast<T>(rstd_val * static_cast<AccT>(dy2.x) +
                           c2v * static_cast<AccT>(x2.x) + c3v);
    dx2.y = static_cast<T>(rstd_val * static_cast<AccT>(dy2.y) +
                           c2v * static_cast<AccT>(x2.y) + c3v);

    reinterpret_cast<double2*>(dX)[vidx] = dx2;
  }
}

// Stage 4 optimized: Scalar no-gamma dX using pre-computed rstd
template <typename T, typename AccT>
__global__ void GroupNormBackwardDxNoGammaOptCUDAKernel(
    int64_t N_G,
    int64_t D_HxW,
    const T* __restrict__ dY,
    const T* __restrict__ X,
    const AccT* __restrict__ c1_ng,
    const AccT* __restrict__ c2,
    const AccT* __restrict__ c3,
    T* __restrict__ dX) {
  const int64_t total = N_G * D_HxW;
  for (int64_t idx =
           static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < total;
       idx += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int64_t ng = idx / D_HxW;
    const AccT rstd_val = c1_ng[ng];
    dX[idx] = static_cast<T>(rstd_val * static_cast<AccT>(dY[idx]) +
                             c2[ng] * static_cast<AccT>(X[idx]) + c3[ng]);
  }
}

// Pre-compute rstd values rounded through T for the no-gamma path.
// c1_ng[ng] = round_through_T(rsqrt(var[ng] + eps))
template <typename T, typename AccT>
__global__ void ComputeRstdCUDAKernel(int64_t N_G,
                                      AccT eps,
                                      const AccT* var,
                                      AccT* c1_ng) {
  for (int64_t index =
           static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < N_G;
       index += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    c1_ng[index] = static_cast<AccT>(static_cast<T>(rsqrt(var[index] + eps)));
  }
}

// ============================================================================
// End of optimized dX kernels
// ============================================================================

// dgamma/dbeta variant 1: small batch (N <= 128)
// Per-channel thread, loop over batch dimension.
// Matches PyTorch's GammaBetaBackwardCUDAKernel1.
// Rounds mean/rstd through T to match PyTorch.
template <typename T, typename AccT>
__global__ void GammaBetaBackwardCUDAKernel1(int64_t N,
                                             int64_t C,
                                             int64_t G,
                                             const AccT* mean,
                                             const AccT* var,
                                             const AccT* ds,
                                             const AccT* db,
                                             AccT eps,
                                             T* dgamma,
                                             T* dbeta) {
  const int64_t c = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (c < C) {
    const int64_t D = C / G;
    AccT sum1 = 0;
    AccT sum2 = 0;
    for (int64_t n = 0; n < N; ++n) {
      const int64_t nc = n * C + c;
      const int64_t ng = n * G + c / D;
      // Round through T to match PyTorch
      AccT mean_val = static_cast<AccT>(static_cast<T>(mean[ng]));
      AccT rstd_val = static_cast<AccT>(static_cast<T>(rsqrt(var[ng] + eps)));
      sum1 += (dgamma == nullptr) ? AccT(0)
                                  : ((ds[nc] - db[nc] * mean_val) * rstd_val);
      sum2 += (dbeta == nullptr) ? AccT(0) : db[nc];
    }
    if (dgamma != nullptr) {
      dgamma[c] = static_cast<T>(sum1);
    }
    if (dbeta != nullptr) {
      dbeta[c] = static_cast<T>(sum2);
    }
  }
}

// dgamma/dbeta variant 2: large batch (N > 128)
// 32x32 tile with shared memory.
// Matches PyTorch's GammaBetaBackwardCUDAKernel2.
// Rounds mean/rstd through T to match PyTorch.
template <typename T, typename AccT>
__global__ void GammaBetaBackwardCUDAKernel2(int64_t N,
                                             int64_t C,
                                             int64_t G,
                                             const AccT* mean,
                                             const AccT* var,
                                             const AccT* ds,
                                             const AccT* db,
                                             AccT eps,
                                             T* dgamma,
                                             T* dbeta) {
  __shared__ AccT g_shared[kGradReduceTileSize][kGradReduceTileSize + 1];
  __shared__ AccT b_shared[kGradReduceTileSize][kGradReduceTileSize + 1];
  const int64_t c = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  AccT dg_sum1 = 0;
  AccT dg_sum2 = 0;
  AccT db_sum1 = 0;
  AccT db_sum2 = 0;
  if (c < C) {
    const int64_t D = C / G;
    for (int64_t n = threadIdx.y; n < N; n += blockDim.y * 2) {
      const int64_t n1 = n;
      const int64_t n2 = n + blockDim.y;
      const int64_t nc1 = n1 * C + c;
      const int64_t nc2 = n2 * C + c;
      const int64_t ng1 = n1 * G + c / D;
      const int64_t ng2 = n2 * G + c / D;
      // Round through T to match PyTorch
      AccT mean_val1 = static_cast<AccT>(static_cast<T>(mean[ng1]));
      AccT rstd1 = static_cast<AccT>(static_cast<T>(rsqrt(var[ng1] + eps)));
      dg_sum1 += dgamma == nullptr ? AccT(0)
                                   : ((ds[nc1] - db[nc1] * mean_val1) * rstd1);
      db_sum1 += dbeta == nullptr ? AccT(0) : db[nc1];
      if (n2 < N) {
        AccT mean_val2 = static_cast<AccT>(static_cast<T>(mean[ng2]));
        AccT rstd2 = static_cast<AccT>(static_cast<T>(rsqrt(var[ng2] + eps)));
        dg_sum2 += dgamma == nullptr
                       ? AccT(0)
                       : ((ds[nc2] - db[nc2] * mean_val2) * rstd2);
        db_sum2 += dbeta == nullptr ? AccT(0) : db[nc2];
      }
    }
  }

  g_shared[threadIdx.y][threadIdx.x] = dg_sum1;
  g_shared[threadIdx.y + blockDim.y][threadIdx.x] = dg_sum2;
  b_shared[threadIdx.y][threadIdx.x] = db_sum1;
  b_shared[threadIdx.y + blockDim.y][threadIdx.x] = db_sum2;
  __syncthreads();

  // Warp reduce for 1st 16 cols in tile
  AccT sum1 = g_shared[threadIdx.x][threadIdx.y];
  AccT sum2 = b_shared[threadIdx.x][threadIdx.y];
  sum1 = GradWarpReduceSum(sum1);
  sum2 = GradWarpReduceSum(sum2);
  if (threadIdx.x == 0) {
    const int64_t c_out = blockIdx.x * blockDim.x + threadIdx.y;
    if (c_out < C) {
      if (dgamma != nullptr) {
        dgamma[c_out] = static_cast<T>(sum1);
      }
      if (dbeta != nullptr) {
        dbeta[c_out] = static_cast<T>(sum2);
      }
    }
  }

  // Warp reduce for 2nd 16 cols in tile
  sum1 = g_shared[threadIdx.x][threadIdx.y + blockDim.y];
  sum2 = b_shared[threadIdx.x][threadIdx.y + blockDim.y];
  sum1 = GradWarpReduceSum(sum1);
  sum2 = GradWarpReduceSum(sum2);
  if (threadIdx.x == 0) {
    const int64_t c_out = blockIdx.x * blockDim.x + threadIdx.y + blockDim.y;
    if (c_out < C) {
      if (dgamma != nullptr) {
        dgamma[c_out] = static_cast<T>(sum1);
      }
      if (dbeta != nullptr) {
        dbeta[c_out] = static_cast<T>(sum2);
      }
    }
  }
}

// ============================================================================
// Main backward kernel
// ============================================================================

template <typename T, typename Context>
void GroupNormGradKernel(const Context& dev_ctx,
                         const DenseTensor& x,
                         const optional<DenseTensor>& scale,
                         const optional<DenseTensor>& bias,
                         const DenseTensor& y,
                         const DenseTensor& mean,
                         const DenseTensor& var,
                         const DenseTensor& d_y,
                         double epsilon,
                         int groups,
                         const std::string& data_layout_str,
                         DenseTensor* d_x,
                         DenseTensor* d_scale,
                         DenseTensor* d_bias) {
  if (x.numel() == 0) {
    dev_ctx.template Alloc<T>(d_x);
    if (d_scale) {
      if (x.dims().size() > 0 && x.dims()[0] == 0) {
        Full<T, Context>(dev_ctx, d_scale->dims(), 0, d_scale);
      } else {
        Full<T, Context>(dev_ctx, d_scale->dims(), NAN, d_scale);
      }
    }
    if (d_bias) {
      Full<T, Context>(dev_ctx, d_bias->dims(), 0, d_bias);
    }
    return;
  }
  using AccT = typename phi::dtype::MPTypeTrait<T>::Type;
  const DataLayout data_layout = StringToDataLayout(data_layout_str);
  const auto scale_ptr = scale.get_ptr();
  const auto bias_ptr = bias.get_ptr();

  const auto& x_dims = x.dims();
  const int64_t N = x_dims[0];
  const int64_t C =
      (data_layout == DataLayout::NCHW ? x_dims[1] : x_dims[x_dims.size() - 1]);
  const int64_t group_size = C / groups;
  const int64_t W =
      (data_layout == DataLayout::NCHW ? x_dims[x_dims.size() - 1]
                                       : x_dims[x_dims.size() - 2]);
  const int64_t G = groups;
  const int64_t D = C / G;

  if (d_x) {
    dev_ctx.template Alloc<T>(d_x);
  }
  funcs::SetConstant<GPUContext, T> set_zero;
  funcs::SetConstant<GPUContext, AccT> set_zero_AccT;

  auto* x_data = x.data<T>();
  T* d_x_data = nullptr;
  if (d_x) d_x_data = d_x->data<T>();
  auto* dy_data = d_y.data<T>();
  auto* var_data = var.data<AccT>();
  auto* mean_data = mean.data<AccT>();
  T* d_scale_data = nullptr;
  if (d_scale) {
    dev_ctx.template Alloc<T>(d_scale);
    d_scale_data = d_scale->data<T>();
  }
  T* d_bias_data = nullptr;
  if (d_bias) {
    dev_ctx.template Alloc<T>(d_bias);
    d_bias_data = d_bias->data<T>();
  }

  const T* scale_data = nullptr;
  if (scale_ptr) scale_data = scale_ptr->data<T>();
  const T* bias_data = nullptr;
  if (bias_ptr) bias_data = bias_ptr->data<T>();

  int64_t imsize = 1;
  if (data_layout == DataLayout::NCHW) {
    for (int i = 2; i < x_dims.size(); ++i) {
      imsize *= x_dims[i];
    }
  } else {
    for (int i = 1; i < x_dims.size() - 1; ++i) {
      imsize *= x_dims[i];
    }
  }

  if (data_layout == DataLayout::NCHW) {
    // =========================================================
    // PyTorch-aligned NCHW backward path (optimized)
    // =========================================================
    const int64_t HxW = imsize;

    // Stage 1: Compute internal gradients ds[n,c] = sum_hw(dY*X),
    //          db[n,c] = sum_hw(dY)
    DenseTensor ds_tensor, db_tensor;
    ds_tensor.Resize({N, C});
    db_tensor.Resize({N, C});
    AccT* ds_data = dev_ctx.template Alloc<AccT>(&ds_tensor);
    AccT* db_data = dev_ctx.template Alloc<AccT>(&db_tensor);

    {
      // Use scalar kernel to match PyTorch's exact accumulation order
      // (Vec4 changes thread-to-element mapping, causing 1-2 ULP differences)
      {
        int64_t num_threads =
            HxW < kGradBlockReduceNumThreads ? 32 : kGradBlockReduceNumThreads;
        ComputeInternalGradientsCUDAKernel<T, AccT>
            <<<N * C, num_threads, 0, dev_ctx.stream()>>>(
                HxW, dy_data, x_data, ds_data, db_data);
      }
    }

    // Stage 2+3+4: Compute dX
    if (d_x_data != nullptr) {
      DenseTensor c2_tensor, c3_tensor;
      c2_tensor.Resize({N, G});
      c3_tensor.Resize({N, G});
      AccT* c2_data = dev_ctx.template Alloc<AccT>(&c2_tensor);
      AccT* c3_data = dev_ctx.template Alloc<AccT>(&c3_tensor);

      // Stage 2: Compute backward fused params c2, c3
      {
        int64_t num_threads =
            D < kGradBlockReduceNumThreads ? 32 : kGradBlockReduceNumThreads;
        ComputeBackwardFusedParamsCUDAKernel<T, AccT>
            <<<dim3(N, G), num_threads, 0, dev_ctx.stream()>>>(
                C,
                HxW,
                G,
                mean_data,
                var_data,
                scale_data,
                ds_data,
                db_data,
                static_cast<AccT>(epsilon),
                c2_data,
                c3_data);
      }

      // Stage 4: dX = c1 * dY + c2 * X + c3 (optimized)
      int64_t max_grid_x = dev_ctx.GetCUDAMaxGridDimSize()[0];
      constexpr int64_t kElemThreads = 256;

      if (scale_data != nullptr) {
        // Pre-compute c1[n,c] = rstd[n,g] * gamma[c]
        // This eliminates per-element rsqrt and gamma lookup.
        DenseTensor c1_tensor;
        c1_tensor.Resize({N, C});
        AccT* c1_data = dev_ctx.template Alloc<AccT>(&c1_tensor);

        {
          const int64_t N_C = N * C;
          const int64_t c1_blocks =
              std::min((N_C + kElemThreads - 1) / kElemThreads, max_grid_x);
          ComputeC1CUDAKernel<T, AccT>
              <<<c1_blocks, kElemThreads, 0, dev_ctx.stream()>>>(
                  N_C,
                  C,
                  G,
                  static_cast<AccT>(epsilon),
                  var_data,
                  scale_data,
                  c1_data);
        }

        // Dispatch vectorized or scalar dX kernel
        if (std::is_same<T, float>::value && (HxW % 4 == 0)) {
          const int64_t HxW_vec = HxW / 4;
          const int64_t total_vec = N * C * HxW_vec;
          const int64_t elem_blocks = std::min(
              (total_vec + kElemThreads - 1) / kElemThreads, max_grid_x);
          GroupNormBackwardDxVec4CUDAKernel<T, AccT>
              <<<elem_blocks, kElemThreads, 0, dev_ctx.stream()>>>(N * C,
                                                                   HxW_vec,
                                                                   D,
                                                                   dy_data,
                                                                   x_data,
                                                                   c1_data,
                                                                   c2_data,
                                                                   c3_data,
                                                                   d_x_data);
        } else if (std::is_same<T, double>::value && (HxW % 2 == 0)) {
          const int64_t HxW_vec = HxW / 2;
          const int64_t total_vec = N * C * HxW_vec;
          const int64_t elem_blocks = std::min(
              (total_vec + kElemThreads - 1) / kElemThreads, max_grid_x);
          GroupNormBackwardDxVec2DoubleCUDAKernel<T, AccT>
              <<<elem_blocks, kElemThreads, 0, dev_ctx.stream()>>>(N * C,
                                                                   HxW_vec,
                                                                   D,
                                                                   dy_data,
                                                                   x_data,
                                                                   c1_data,
                                                                   c2_data,
                                                                   c3_data,
                                                                   d_x_data);
        } else {
          // Scalar fallback with pre-computed c1
          const int64_t total = N * C * HxW;
          const int64_t elem_blocks =
              std::min((total + kElemThreads - 1) / kElemThreads, max_grid_x);
          GroupNormBackwardDxOptCUDAKernel<T, AccT>
              <<<elem_blocks, kElemThreads, 0, dev_ctx.stream()>>>(N * C,
                                                                   HxW,
                                                                   D,
                                                                   dy_data,
                                                                   x_data,
                                                                   c1_data,
                                                                   c2_data,
                                                                   c3_data,
                                                                   d_x_data);
        }
      } else {
        // No gamma path: c1 = rstd[ng], pre-compute rstd rounded through T
        DenseTensor c1_ng_tensor;
        c1_ng_tensor.Resize({N, G});
        AccT* c1_ng_data = dev_ctx.template Alloc<AccT>(&c1_ng_tensor);

        {
          const int64_t N_G = N * G;
          const int64_t rstd_blocks =
              std::min((N_G + kElemThreads - 1) / kElemThreads, max_grid_x);
          ComputeRstdCUDAKernel<T, AccT>
              <<<rstd_blocks, kElemThreads, 0, dev_ctx.stream()>>>(
                  N_G, static_cast<AccT>(epsilon), var_data, c1_ng_data);
        }

        const int64_t D_HxW = D * HxW;

        if (std::is_same<T, float>::value && (D_HxW % 4 == 0)) {
          const int64_t D_HxW_vec = D_HxW / 4;
          const int64_t total_vec = N * G * D_HxW_vec;
          const int64_t elem_blocks = std::min(
              (total_vec + kElemThreads - 1) / kElemThreads, max_grid_x);
          GroupNormBackwardDxNoGammaVec4CUDAKernel<T, AccT>
              <<<elem_blocks, kElemThreads, 0, dev_ctx.stream()>>>(N * G,
                                                                   D_HxW_vec,
                                                                   dy_data,
                                                                   x_data,
                                                                   c1_ng_data,
                                                                   c2_data,
                                                                   c3_data,
                                                                   d_x_data);
        } else if (std::is_same<T, double>::value && (D_HxW % 2 == 0)) {
          const int64_t D_HxW_vec = D_HxW / 2;
          const int64_t total_vec = N * G * D_HxW_vec;
          const int64_t elem_blocks = std::min(
              (total_vec + kElemThreads - 1) / kElemThreads, max_grid_x);
          GroupNormBackwardDxNoGammaVec2DoubleCUDAKernel<T, AccT>
              <<<elem_blocks, kElemThreads, 0, dev_ctx.stream()>>>(N * G,
                                                                   D_HxW_vec,
                                                                   dy_data,
                                                                   x_data,
                                                                   c1_ng_data,
                                                                   c2_data,
                                                                   c3_data,
                                                                   d_x_data);
        } else {
          // Scalar fallback with pre-computed rstd
          const int64_t total = N * G * D_HxW;
          const int64_t elem_blocks =
              std::min((total + kElemThreads - 1) / kElemThreads, max_grid_x);
          GroupNormBackwardDxNoGammaOptCUDAKernel<T, AccT>
              <<<elem_blocks, kElemThreads, 0, dev_ctx.stream()>>>(N * G,
                                                                   D_HxW,
                                                                   dy_data,
                                                                   x_data,
                                                                   c1_ng_data,
                                                                   c2_data,
                                                                   c3_data,
                                                                   d_x_data);
        }
      }
    }

    // Stage 5: dgamma/dbeta
    if (d_scale || d_bias) {
      if (N <= 128) {
        constexpr int kNumThreads = 256;
        const int64_t B = (C + kNumThreads - 1) / kNumThreads;
        GammaBetaBackwardCUDAKernel1<T, AccT>
            <<<B, kNumThreads, 0, dev_ctx.stream()>>>(
                N,
                C,
                G,
                mean_data,
                var_data,
                ds_data,
                db_data,
                static_cast<AccT>(epsilon),
                d_scale_data,
                d_bias_data);
      } else {
        const int64_t B = (C + kGradReduceTileSize - 1) / kGradReduceTileSize;
        constexpr int kThreadX = kGradReduceTileSize;
        constexpr int kThreadY = kGradReduceTileSize / 2;
        GammaBetaBackwardCUDAKernel2<T, AccT>
            <<<B, dim3(kThreadX, kThreadY), 0, dev_ctx.stream()>>>(
                N,
                C,
                G,
                mean_data,
                var_data,
                ds_data,
                db_data,
                static_cast<AccT>(epsilon),
                d_scale_data,
                d_bias_data);
      }
    }
  } else {
    // =========================================================
    // NHWC backward path (kept unchanged)
    // =========================================================
    int block_size = std::min(static_cast<int64_t>(1024), imsize);
    int64_t max_grid_x = dev_ctx.GetCUDAMaxGridDimSize()[0];
    int64_t max_grid_z = dev_ctx.GetCUDAMaxGridDimSize()[2];
    dim3 grid(
        std::min(max_grid_x, group_size), groups, std::min(max_grid_z, N));
    dim3 threads(block_size, 1, 1);

    auto* y_data = y.data<T>();

    if (d_scale) {
      set_zero(dev_ctx, d_scale, static_cast<T>(0));
    }
    if (d_bias) {
      set_zero(dev_ctx, d_bias, static_cast<T>(0));
    }

    DenseTensor temp_var;
    temp_var.Resize(var.dims());
    dev_ctx.template Alloc<AccT>(&temp_var);
    set_zero_AccT(dev_ctx, &temp_var, static_cast<AccT>(0));
    auto* temp_var_data = temp_var.data<AccT>();

    DenseTensor temp_mean;
    temp_mean.Resize(var.dims());
    dev_ctx.template Alloc<AccT>(&temp_mean);
    set_zero_AccT(dev_ctx, &temp_mean, static_cast<AccT>(0));
    auto* temp_mean_data = temp_mean.data<AccT>();

    int flags =
        (scale_data != nullptr) * kHasScale + (bias_data != nullptr) * kHasBias;
    UNROLL_ALL_CASES(flags,
                     GroupNormBackwardGetMeanAndVar,
                     y_data,
                     scale_data,
                     bias_data,
                     dy_data,
                     N,
                     C,
                     W,
                     imsize,
                     groups,
                     group_size,
                     epsilon,
                     temp_mean_data,
                     temp_var_data,
                     d_scale_data,
                     d_bias_data);
    if (d_x_data != nullptr) {
      UNROLL_ALL_CASES(flags,
                       GroupNormBackward,
                       y_data,
                       dy_data,
                       scale_data,
                       bias_data,
                       var_data,
                       temp_mean_data,
                       temp_var_data,
                       N,
                       C,
                       W,
                       imsize,
                       groups,
                       group_size,
                       epsilon,
                       d_x_data);
    }
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(group_norm_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::GroupNormGradKernel,
                   float,
                   double,
                   phi::bfloat16,
                   phi::float16) {}
