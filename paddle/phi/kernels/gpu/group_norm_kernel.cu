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

#include "paddle/phi/kernels/group_norm_kernel.h"

#include "paddle/common/layout.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/backends/gpu/gpu_device_function.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/math_function.h"
#include "paddle/phi/kernels/gpu/group_norm_utils.h"

#include "paddle/phi/common/data_type.h"
#include "paddle/phi/core/device_context.h"
#include "paddle/phi/kernels/full_kernel.h"

#include "paddle/common/flags.h"
COMMON_DECLARE_bool(use_accuracy_compatible_kernel);

namespace phi {

template <typename T>
static inline T divUp(T m, T n) {
  return (m + n - 1) / n;
}

static inline __device__ __host__ float sigmoid(float x) {
  return 1.F / (1.F + expf(-x));
}

#ifdef PADDLE_CUDA_BF16
__host__ __device__ inline float2 bfloat1622float2(const __nv_bfloat162 a) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800))
  return __bfloat1622float2(a);
#else
  float hi_float;
  float lo_float;
  lo_float = __internal_bfloat162float(((__nv_bfloat162_raw)a).x);
  hi_float = __internal_bfloat162float(((__nv_bfloat162_raw)a).y);
  return make_float2(lo_float, hi_float);
#endif
}

__host__ __device__ inline __nv_bfloat162 float22bfloat162_rn(const float2 a) {
  __nv_bfloat162 val;
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800))
  val = __float22bfloat162_rn(a);
#else
  val.x = __float2bfloat16_rn(a.x);
  val.y = __float2bfloat16_rn(a.y);
#endif
  return val;
}

#endif

template <typename T>
__host__ __device__ inline float __2float(const T a) {
  return static_cast<float>(a);
}

template <>
__host__ __device__ inline float __2float<__half>(const __half a) {
  return __half2float(a);
}

template <typename T>
__host__ __device__ inline T __2dst(const float a) {
  return static_cast<T>(a);
}

template <>
__host__ __device__ inline __half __2dst<__half>(const float a) {
  return __float2half(a);
}

struct GroupSums {
  // Is it the 1st element of the group?
  int32_t flag;
  // The sum.
  float sum;
  // The sum of squares.
  float sumSq;
};

struct GroupSumsOp {
  inline __device__ GroupSums operator()(GroupSums const& a,
                                         GroupSums const& b) {
    GroupSums dst;
    dst.sum = b.flag ? b.sum : (a.sum + b.sum);
    dst.sumSq = b.flag ? b.sumSq : (a.sumSq + b.sumSq);
    dst.flag = a.flag + b.flag;
    return dst;
  }
};

static int64_t findMaxDivisor(int64_t n, int64_t maxAllowedDivisor) {
  int64_t maxDivisor = -1;
  for (int64_t i = 1; i <= std::sqrt(n); i++) {
    if (n % i == 0) {
      int64_t divisor1 = n / i;
      int64_t divisor2 = i;

      if (divisor1 > maxDivisor && divisor1 < maxAllowedDivisor) {
        maxDivisor = divisor1;
      }
      if (divisor2 > maxDivisor && divisor2 < maxAllowedDivisor) {
        maxDivisor = divisor2;
      }
    }
  }
  return maxDivisor;
}

template <typename T, int THREADS_PER_CHANNEL>
inline __device__ void UpdateSum(const T* srcX, float* sum, float* sumSq) {
  float src_data = __2float<T>(*srcX);
  *sum += src_data;
  *sumSq += src_data * src_data;
}

template <typename T, int THREADS_PER_CHANNEL>
inline __device__ void UpdateSum(const T* srcX,
                                 const T* srcR,
                                 float* sum,
                                 float* sumSq) {
  float src_data = __2float<T>(*srcX);
  float srcy_data = __2float<T>(*srcR);
  *sum += src_data + srcy_data;
  *sumSq += (src_data + srcy_data) * (src_data + srcy_data);
}

template <>
inline __device__ void UpdateSum<__half, 2>(const __half* srcX,
                                            float* sum,
                                            float* sumSq) {
  __half2 h2 = *reinterpret_cast<__half2 const*>(srcX);
  float2 f2 = __half22float2(h2);
  *sum += f2.x + f2.y;
  *sumSq += f2.x * f2.x + f2.y * f2.y;
}

template <>
inline __device__ void UpdateSum<__half, 2>(const __half* srcX,
                                            const __half* srcR,
                                            float* sum,
                                            float* sumSq) {
  __half2 h2 = *reinterpret_cast<__half2 const*>(srcX);
  __half2 h2_r = *reinterpret_cast<__half2 const*>(srcR);
  float2 f2 = __half22float2(h2);
  float2 f2_r = __half22float2(h2_r);
  *sum += f2.x + f2_r.x + f2.y + f2_r.y;
  *sumSq +=
      (f2.x + f2_r.x) * (f2.x + f2_r.x) + (f2.y + f2_r.y) * (f2.y + f2_r.y);
}

template <>
inline __device__ void UpdateSum<float16, 2>(const float16* srcX,
                                             float* sum,
                                             float* sumSq) {
  __half2 h2 = *reinterpret_cast<__half2 const*>(srcX);
  float2 f2 = __half22float2(h2);
  *sum += f2.x + f2.y;
  *sumSq += f2.x * f2.x + f2.y * f2.y;
}

template <>
inline __device__ void UpdateSum<float16, 2>(const float16* srcX,
                                             const float16* srcR,
                                             float* sum,
                                             float* sumSq) {
  __half2 h2 = *reinterpret_cast<__half2 const*>(srcX);
  __half2 h2_r = *reinterpret_cast<__half2 const*>(srcR);
  float2 f2 = __half22float2(h2);
  float2 f2_r = __half22float2(h2_r);
  *sum += f2.x + f2_r.x + f2.y + f2_r.y;
  *sumSq +=
      (f2.x + f2_r.x) * (f2.x + f2_r.x) + (f2.y + f2_r.y) * (f2.y + f2_r.y);
}

#ifdef PADDLE_CUDA_BF16
template <>
inline __device__ void UpdateSum<bfloat16, 2>(const bfloat16* srcX,
                                              float* sum,
                                              float* sumSq) {
  __nv_bfloat162 h2 = *reinterpret_cast<__nv_bfloat162 const*>(srcX);
  float2 f2 = bfloat1622float2(h2);
  *sum += f2.x + f2.y;
  *sumSq += f2.x * f2.x + f2.y * f2.y;
}

template <>
inline __device__ void UpdateSum<bfloat16, 2>(const bfloat16* srcX,
                                              const bfloat16* srcR,
                                              float* sum,
                                              float* sumSq) {
  __nv_bfloat162 h2 = *reinterpret_cast<__nv_bfloat162 const*>(srcX);
  __nv_bfloat162 h2_r = *reinterpret_cast<__nv_bfloat162 const*>(srcR);
  float2 f2 = bfloat1622float2(h2);
  float2 f2_r = bfloat1622float2(h2_r);
  *sum += f2.x + f2_r.x + f2.y + f2_r.y;
  *sumSq +=
      (f2.x + f2_r.x) * (f2.x + f2_r.x) + (f2.y + f2_r.y) * (f2.y + f2_r.y);
}
#endif

template <typename T, int THREADS_PER_BLOCK>
__global__ void groupNormNDHWCSumSingerChannelKernel(
    const GroupNormNDHWCParams<T> params) {
  // The instance in the batch.
  __shared__ float2 smem[THREADS_PER_BLOCK];
  int32_t ni = blockIdx.z;
  int64_t ci = static_cast<int64_t>(blockIdx.x) *
                   static_cast<int64_t>(params.cPerBlock) +
               static_cast<int64_t>(threadIdx.x);
  if (ci >= params.c) {
    return;
  }
  // The first activation loaded by that block.
  int64_t dhwBegin = static_cast<int64_t>(blockIdx.y) * params.dhwPerBlock;
  // The last activation loaded by that block.
  int64_t dhwEnd = min(dhwBegin + params.dhwPerBlock, params.dhw);

  // The sums.
  float sum = 0.F;
  float sumSq = 0.F;

  for (int64_t dhwi = dhwBegin; dhwi < dhwEnd; ++dhwi) {
    // The offset.
    int64_t offset = ni * params.dhwc + dhwi * params.c + ci;
    float src_data = *reinterpret_cast<float const*>(&params.srcX[offset]);
    if (params.srcR != nullptr) {
      int64_t g_offset = params.y_same_with_x ? offset : ci;
      UpdateSum<T, 1>(
          &params.srcX[offset], &params.srcR[g_offset], &sum, &sumSq);
    } else {
      UpdateSum<T, 1>(&params.srcX[offset], &sum, &sumSq);
    }
  }

  smem[threadIdx.x] = make_float2(sum, sumSq);

  __syncthreads();

  float2 sums = smem[threadIdx.x];
  atomicAdd(&params.redBuffer[(2 * ni + 0) * params.groups + ci],
            sums.x * params.invDHWC);
  atomicAdd(&params.redBuffer[(2 * ni + 1) * params.groups + ci], sums.y);
}

template <typename T, int THREADS_PER_BLOCK, int THREADS_PER_CHANNEL>
__global__ void groupNormNDHWCSumKernel(const GroupNormNDHWCParams<T> params) {
  // The object in charge of doing the sums for the different blocks.
  typedef cub::BlockScan<GroupSums, THREADS_PER_BLOCK> BlockScan;
  __shared__ typename BlockScan::TempStorage tempStorage;
  // Allocate shared memory for BlockScan.
  // Allocate shared memory for the groups. We could reduce the amount of shared
  // memory reserved.
  __shared__ float2 smem[THREADS_PER_BLOCK];

  // The instance in the batch.
  int32_t ni = blockIdx.z;
  // The channel loaded by that thread (2 channels per thread for F16x2).
  int64_t ci = static_cast<int64_t>(blockIdx.x) *
                   static_cast<int64_t>(params.cPerBlock) +
               static_cast<int64_t>(threadIdx.x) * THREADS_PER_CHANNEL;
  if (ci >= params.c || threadIdx.x * THREADS_PER_CHANNEL >= params.cPerBlock) {
    return;
  }
  int32_t gj = ci / params.cPerGroup;
  int32_t cj = ci % params.cPerGroup;
  int64_t dhwBegin = static_cast<int64_t>(blockIdx.y) * params.dhwPerBlock;
  // The last activation loaded by that block.
  int64_t dhwEnd = min(dhwBegin + params.dhwPerBlock, params.dhw);

  // The sums.
  float sum = 0.F;
  float sumSq = 0.F;

  for (int64_t dhwi = dhwBegin; dhwi < dhwEnd; ++dhwi) {
    // The offset.
    int64_t offset = ni * params.dhwc + dhwi * params.c + ci;
    float src_data = *reinterpret_cast<float const*>(&params.srcX[offset]);
    if (params.srcR != nullptr) {
      int64_t g_offset =
          params.y_same_with_x ? offset : gj * params.cPerGroup + cj;
      UpdateSum<T, THREADS_PER_CHANNEL>(
          &params.srcX[offset], &params.srcR[g_offset], &sum, &sumSq);
    } else {
      UpdateSum<T, THREADS_PER_CHANNEL>(&params.srcX[offset], &sum, &sumSq);
    }
  }

  // The group that thread works on and the channel in the group (modulus).
  int32_t gi =
      ci / params.cPerGroup - blockIdx.x * params.cPerBlock / params.cPerGroup;
  int flag = (cj == 0 || threadIdx.x == 0) ? 1 : 0;
  GroupSums inp{flag, sum, sumSq};
  GroupSums out;
  BlockScan(tempStorage).InclusiveScan(inp, out, GroupSumsOp());

  if (cj == params.cPerGroup - THREADS_PER_CHANNEL ||
      threadIdx.x * THREADS_PER_CHANNEL ==
          params.cPerBlock - THREADS_PER_CHANNEL) {
    smem[gi] = make_float2(out.sum, out.sumSq);
  }

  __syncthreads();

  if (cj == params.cPerGroup - THREADS_PER_CHANNEL ||
      threadIdx.x * THREADS_PER_CHANNEL ==
          params.cPerBlock - THREADS_PER_CHANNEL) {
    float2 sums = smem[gi];
    atomicAdd(&params.redBuffer[(2 * ni + 0) * params.groups + gj],
              sums.x * params.invDHWC);
    atomicAdd(&params.redBuffer[(2 * ni + 1) * params.groups + gj], sums.y);
  }
}

template <typename T>
void groupNormNDHWCSum<T>::operator()(GroupNormNDHWCParams<T>* params,
                                      gpuStream_t stream) {
  dim3 grid;
  grid.x = divUp(params->c, params->cPerBlock);
  grid.y = divUp(params->dhw, params->dhwPerBlock);
  grid.z = params->n;
  if (params->cPerGroup % 2 == 0) {
    switch (params->cPerBlock) {
      case 512:
      case 480:
        groupNormNDHWCSumKernel<T, 256, 2><<<grid, 256, 0, stream>>>(*params);
        break;
      case 320:
        groupNormNDHWCSumKernel<T, 160, 2><<<grid, 160, 0, stream>>>(*params);
        break;
      case 256:
        groupNormNDHWCSumKernel<T, 128, 2><<<grid, 128, 0, stream>>>(*params);
        break;
      case 128:
        groupNormNDHWCSumKernel<T, 64, 2><<<grid, 64, 0, stream>>>(*params);
        break;
      default:
        grid.x = divUp(params->c, 128);
        params->cPerBlock = 128;
        groupNormNDHWCSumKernel<T, 64, 2><<<grid, 64, 0, stream>>>(*params);
    }
  } else {
    if (params->cPerGroup != 1) {
      switch (params->cPerBlock) {
        case 512:
          groupNormNDHWCSumKernel<T, 512, 1><<<grid, 512, 0, stream>>>(*params);
          break;
        case 480:
          groupNormNDHWCSumKernel<T, 480, 1><<<grid, 480, 0, stream>>>(*params);
          break;
        case 320:
          groupNormNDHWCSumKernel<T, 320, 1><<<grid, 320, 0, stream>>>(*params);
          break;
        case 256:
          groupNormNDHWCSumKernel<T, 256, 1><<<grid, 256, 0, stream>>>(*params);
          break;
        case 128:
          groupNormNDHWCSumKernel<T, 128, 1><<<grid, 128, 0, stream>>>(*params);
          break;
        default:
          grid.x = divUp(params->c, 128);
          params->cPerBlock = 128;
          groupNormNDHWCSumKernel<T, 128, 1><<<grid, 128, 0, stream>>>(*params);
      }
    } else {
      switch (params->cPerBlock) {
        case 512:
          groupNormNDHWCSumSingerChannelKernel<T, 512>
              <<<grid, 512, 0, stream>>>(*params);
          break;
        case 480:
          groupNormNDHWCSumSingerChannelKernel<T, 480>
              <<<grid, 480, 0, stream>>>(*params);
          break;
        case 320:
          groupNormNDHWCSumSingerChannelKernel<T, 320>
              <<<grid, 320, 0, stream>>>(*params);
          break;
        case 256:
          groupNormNDHWCSumSingerChannelKernel<T, 256>
              <<<grid, 256, 0, stream>>>(*params);
          break;
        case 128:
          groupNormNDHWCSumSingerChannelKernel<T, 128>
              <<<grid, 128, 0, stream>>>(*params);
          break;
        default:
          grid.x = divUp(params->c, 128);
          params->cPerBlock = 128;
          groupNormNDHWCSumSingerChannelKernel<T, 128>
              <<<grid, 128, 0, stream>>>(*params);
      }
    }
  }
}
template class PADDLE_API groupNormNDHWCSum<half>;

template <typename T, int THREADS_PER_CHANNEL>
inline __device__ void GroupNormCompute(int64_t dhwBegin,
                                        int64_t dhwEnd,
                                        int32_t ci,
                                        const GroupNormNDHWCParams<T>& params,
                                        float mean,
                                        float invStdDev) {
  float gamma = __2float<T>(*(reinterpret_cast<T const*>(params.gamma) + ci));
  float beta = __2float<T>(*(reinterpret_cast<T const*>(params.beta) + ci));
  for (int64_t dhwi = dhwBegin; dhwi < dhwEnd; ++dhwi) {
    // The src/dst offset.
    int64_t offset =
        static_cast<int64_t>(blockIdx.z) * params.dhwc + dhwi * params.c + ci;
    float src_data = __2float<T>(params.srcX[offset]);
    if (params.srcR != nullptr) {
      auto gi = ci / params.cPerGroup;
      auto gj = ci % params.cPerGroup;
      int64_t g_offset =
          params.y_same_with_x ? offset : gi * params.cPerGroup + gj;
      src_data += __2float<T>(params.srcR[g_offset]);
      *reinterpret_cast<T*>(&params.eleOut[offset]) = __2dst<T>(src_data);
    }
    // Normalize the channels.
    float dst_data = (src_data - mean) * invStdDev;
    // Scale by gamma and add beta.
    dst_data = gamma * dst_data + beta;

    // Apply Silu if needed.
    if (params.withSilu) {
      dst_data = dst_data * sigmoid(dst_data);
    }

    // Store the scaled values.
    *reinterpret_cast<T*>(&params.dst[offset]) = __2dst<T>(dst_data);
  }
}

template <>
inline __device__ void GroupNormCompute<float16, 2>(
    int64_t dhwBegin,
    int64_t dhwEnd,
    int32_t ci,
    const GroupNormNDHWCParams<float16>& params,
    float mean,
    float invStdDev) {
  float2 gammaF2, betaF2;
  gammaF2 = __half22float2(*reinterpret_cast<__half2 const*>(
      reinterpret_cast<half const*>(params.gamma) + ci));
  betaF2 = __half22float2(*reinterpret_cast<__half2 const*>(
      reinterpret_cast<half const*>(params.beta) + ci));

  // Iterate over the activations to compute the sums.
  for (int64_t dhwi = dhwBegin; dhwi < dhwEnd; ++dhwi) {
    // The src/dst offset.
    int64_t offset =
        static_cast<int64_t>(blockIdx.z) * params.dhwc + dhwi * params.c + ci;

    // Fetch two channels per thread.
    __half2 h2 = *reinterpret_cast<__half2 const*>(&params.srcX[offset]);

    // Extract the two half values.
    float2 f2 = __half22float2(h2);

    if (params.srcR != nullptr) {
      auto gi = ci / params.cPerGroup;
      auto gj = ci % params.cPerGroup;
      int64_t g_offset =
          params.y_same_with_x ? offset : gi * params.cPerGroup + gj;
      __half2 r2 = *reinterpret_cast<__half2 const*>(&params.srcR[g_offset]);
      float2 r_f2 = __half22float2(r2);
      f2.x += r_f2.x;
      f2.y += r_f2.y;
      *reinterpret_cast<__half2*>(&params.eleOut[offset]) =
          __float22half2_rn(f2);
    }
    // Normalize the channels.
    f2.x = (f2.x - mean) * invStdDev;
    f2.y = (f2.y - mean) * invStdDev;

    // Scale by gamma and add beta.
    f2.x = gammaF2.x * f2.x + betaF2.x;
    f2.y = gammaF2.y * f2.y + betaF2.y;

    // Apply Silu if needed.
    if (params.withSilu) {
      f2.x = f2.x * sigmoid(f2.x);
      f2.y = f2.y * sigmoid(f2.y);
    }
    // Store the scaled values.
    *reinterpret_cast<__half2*>(&params.dst[offset]) = __float22half2_rn(f2);
  }
}

template <>
inline __device__ void GroupNormCompute<__half, 2>(
    int64_t dhwBegin,
    int64_t dhwEnd,
    int32_t ci,
    const GroupNormNDHWCParams<__half>& params,
    float mean,
    float invStdDev) {
  float2 gammaF2, betaF2;
  gammaF2 = __half22float2(*reinterpret_cast<__half2 const*>(
      reinterpret_cast<half const*>(params.gamma) + ci));
  betaF2 = __half22float2(*reinterpret_cast<__half2 const*>(
      reinterpret_cast<half const*>(params.beta) + ci));

  // Iterate over the activations to compute the sums.
  for (int64_t dhwi = dhwBegin; dhwi < dhwEnd; ++dhwi) {
    // The src/dst offset.
    int64_t offset =
        static_cast<int64_t>(blockIdx.z) * params.dhwc + dhwi * params.c + ci;

    // Fetch two channels per thread.
    __half2 h2 = *reinterpret_cast<__half2 const*>(&params.srcX[offset]);

    // Extract the two half values.
    float2 f2 = __half22float2(h2);
    if (params.srcR != nullptr) {
      auto gi = ci / params.cPerGroup;
      auto gj = ci % params.cPerGroup;
      int64_t g_offset =
          params.y_same_with_x ? offset : gi * params.cPerGroup + gj;
      __half2 r2 = *reinterpret_cast<__half2 const*>(&params.srcR[g_offset]);
      float2 r_f2 = __half22float2(r2);
      f2.x += r_f2.x;
      f2.y += r_f2.y;
      *reinterpret_cast<__half2*>(&params.eleOut[offset]) =
          __float22half2_rn(f2);
    }
    // Normalize the channels.
    f2.x = (f2.x - mean) * invStdDev;
    f2.y = (f2.y - mean) * invStdDev;

    // Scale by gamma and add beta.
    f2.x = gammaF2.x * f2.x + betaF2.x;
    f2.y = gammaF2.y * f2.y + betaF2.y;

    // Apply Silu if needed.
    if (params.withSilu) {
      f2.x = f2.x * sigmoid(f2.x);
      f2.y = f2.y * sigmoid(f2.y);
    }
    // Store the scaled values.
    *reinterpret_cast<__half2*>(&params.dst[offset]) = __float22half2_rn(f2);
  }
}

#ifdef PADDLE_CUDA_BF16
template <>
inline __device__ void GroupNormCompute<bfloat16, 2>(
    int64_t dhwBegin,
    int64_t dhwEnd,
    int32_t ci,
    const GroupNormNDHWCParams<bfloat16>& params,
    float mean,
    float invStdDev) {
  float2 gammaF2, betaF2;
  gammaF2 = bfloat1622float2(*reinterpret_cast<__nv_bfloat162 const*>(
      reinterpret_cast<__nv_bfloat16 const*>(params.gamma) + ci));
  betaF2 = bfloat1622float2(*reinterpret_cast<__nv_bfloat162 const*>(
      reinterpret_cast<__nv_bfloat16 const*>(params.beta) + ci));

  // Iterate over the activations to compute the sums.
  for (int64_t dhwi = dhwBegin; dhwi < dhwEnd; ++dhwi) {
    // The src/dst offset.
    int64_t offset =
        static_cast<int64_t>(blockIdx.z) * params.dhwc + dhwi * params.c + ci;

    // Fetch two channels per thread.
    __nv_bfloat162 h2 =
        *reinterpret_cast<__nv_bfloat162 const*>(&params.srcX[offset]);

    // Extract the two half values.
    float2 f2 = bfloat1622float2(h2);

    if (params.srcR != nullptr) {
      auto gi = ci / params.cPerGroup;
      auto gj = ci % params.cPerGroup;
      int64_t g_offset =
          params.y_same_with_x ? offset : gi * params.cPerGroup + gj;
      __nv_bfloat162 r2 =
          *reinterpret_cast<__nv_bfloat162 const*>(&params.srcR[g_offset]);
      float2 r_f2 = bfloat1622float2(r2);
      f2.x += r_f2.x;
      f2.y += r_f2.y;
      *reinterpret_cast<__nv_bfloat162*>(&params.eleOut[offset]) =
          float22bfloat162_rn(f2);
    }
    // Normalize the channels.
    f2.x = (f2.x - mean) * invStdDev;
    f2.y = (f2.y - mean) * invStdDev;

    // Scale by gamma and add beta.
    f2.x = gammaF2.x * f2.x + betaF2.x;
    f2.y = gammaF2.y * f2.y + betaF2.y;

    // Apply Silu if needed.
    if (params.withSilu) {
      f2.x = f2.x * sigmoid(f2.x);
      f2.y = f2.y * sigmoid(f2.y);
    }
    // Store the scaled values.
    *reinterpret_cast<__nv_bfloat162*>(&params.dst[offset]) =
        float22bfloat162_rn(f2);
  }
}
#endif

template <typename T, int THREADS_PER_CHANNEL>
__global__ void groupNormNDHWCScaleKernel(
    const GroupNormNDHWCParams<T> params) {
  // The instance in the batch.
  int32_t ni = blockIdx.z;
  // The channel loaded by that thread (2 channels per thread for F16x2).
  int64_t ci = static_cast<int64_t>(blockIdx.x) *
                   static_cast<int64_t>(params.cPerBlock) +
               static_cast<int64_t>(threadIdx.x) * THREADS_PER_CHANNEL;

  // The group that thread works on and the channel in the group (modulus).
  int32_t gi = ci / params.cPerGroup;
  int32_t gj = ci % params.cPerGroup;

  if (ci >= params.c || gi >= params.groups) {
    return;
  }

  // Load the sum and sum of squares for the group.

  float mean = params.redBuffer[(2 * ni + 0) * params.groups + gi];
  float sumSq = params.redBuffer[(2 * ni + 1) * params.groups + gi];

  // Compute the variance.
  float var = sumSq * params.invDHWC - (mean * mean);

  if (params.var_data != nullptr) {
    params.var_data[ni * params.groups + gi] = var;
  }
  // Compute the inverse of the stddev.
  float invStdDev = rsqrtf(var + params.eps);

  // The first activation loaded by that block.
  int64_t dhwBegin = static_cast<int64_t>(blockIdx.y) * params.dhwPerBlock;
  // The last activation loaded by that block.
  int64_t dhwEnd = min(dhwBegin + params.dhwPerBlock, params.dhw);
  GroupNormCompute<T, THREADS_PER_CHANNEL>(
      dhwBegin, dhwEnd, ci, params, mean, invStdDev);
}

template <typename T>
void groupNormNDHWCScale<T>::operator()(const GroupNormNDHWCParams<T>& params,
                                        gpuStream_t stream) {
  dim3 grid;

  // The number of blocks to compute all the channels.
  grid.x = divUp(params.c, params.cPerBlock);
  // The number of blocks to compute all the activations in a given instance.
  grid.y = divUp(params.dhw, params.dhwPerBlock);
  // The number of instances.
  grid.z = params.n;

  if (params.cPerGroup % 2 == 0) {
    switch (params.cPerBlock) {
      case 512:
      case 480:
        groupNormNDHWCScaleKernel<T, 2><<<grid, 256, 0, stream>>>(params);
        break;
      case 320:
        groupNormNDHWCScaleKernel<T, 2><<<grid, 160, 0, stream>>>(params);
        break;
      case 256:
        groupNormNDHWCScaleKernel<T, 2><<<grid, 128, 0, stream>>>(params);
        break;
      case 128:
        groupNormNDHWCScaleKernel<T, 2><<<grid, 64, 0, stream>>>(params);
        break;
      default:
        grid.x = divUp(params.c, 128);
        groupNormNDHWCScaleKernel<T, 2><<<grid, 64, 0, stream>>>(params);
    }
  } else {
    switch (params.cPerBlock) {
      case 512:
        groupNormNDHWCScaleKernel<T, 1><<<grid, 512, 0, stream>>>(params);
        break;
      case 480:
        groupNormNDHWCScaleKernel<T, 1><<<grid, 480, 0, stream>>>(params);
        break;
      case 320:
        groupNormNDHWCScaleKernel<T, 1><<<grid, 320, 0, stream>>>(params);
        break;
      case 256:
        groupNormNDHWCScaleKernel<T, 1><<<grid, 256, 0, stream>>>(params);
        break;
      case 128:
        groupNormNDHWCScaleKernel<T, 1><<<grid, 128, 0, stream>>>(params);
        break;
      default:
        grid.x = divUp(params.c, 128);
        groupNormNDHWCScaleKernel<T, 1><<<grid, 128, 0, stream>>>(params);
    }
  }
}
template class PADDLE_API groupNormNDHWCScale<half>;

template <typename T, typename Context>
void GroupNormNDHWCKernel(const Context& dev_ctx,
                          const DenseTensor& x,
                          const optional<DenseTensor>& residual,
                          const optional<DenseTensor>& scale,
                          const optional<DenseTensor>& bias,
                          double epsilon,
                          int groups,
                          const std::string& data_layout_str,
                          const std::string& activation,
                          DenseTensor* y,
                          DenseTensor* residual_out,
                          DenseTensor* mean,
                          DenseTensor* var) {
  const DataLayout data_layout = StringToDataLayout(data_layout_str);
  if (data_layout != DataLayout::NHWC) {
    PD_THROW("data_layout only supports NHWC and NDHWC");
  }
  using AccT = typename MPTypeTrait<T>::Type;
  GroupNormNDHWCParams<T> params_;
  params_.withSilu = activation == "silu" ? true : false;

  const auto x_dims = x.dims();
  dev_ctx.template Alloc<T>(y);
  const T* x_data = x.data<T>();
  T* y_data = y->data<T>();
  const auto scale_ptr = scale.get_ptr();
  const auto bias_ptr = bias.get_ptr();
  const T* scale_data = nullptr;
  if (scale_ptr) scale_data = scale_ptr->data<T>();
  const T* bias_data = nullptr;
  if (bias_ptr) bias_data = bias_ptr->data<T>();
  const auto d_dim = x_dims.size();
  params_.n = x_dims[0];
  if (d_dim == 3) {
    params_.c = x_dims[2];
    params_.d = 1;
    params_.h = 1;
    params_.w = x_dims[1];
  } else if (d_dim == 4) {
    params_.c = x_dims[3];
    params_.d = 1;
    params_.h = x_dims[1];
    params_.w = x_dims[2];
  } else {
    // d_dim == 5
    params_.c = x_dims[4];
    params_.d = x_dims[1];
    params_.h = x_dims[2];
    params_.w = x_dims[3];
  }

  const T* residual_data = nullptr;
  const auto residual_ptr = residual.get_ptr();
  T* residual_out_data = nullptr;
  if (residual_ptr) {
    dev_ctx.template Alloc<T>(residual_out);
    residual_data = residual_ptr->data<T>();
    residual_out_data = residual_out->data<T>();
    const auto r_dims = residual_ptr->dims();
    int64_t r_dim = 1;
    for (size_t i = 0; i < r_dims.size(); i++) {
      r_dim *= r_dims[i];
    }
    params_.y_same_with_x = r_dim == static_cast<int64_t>(params_.n) *
                                         params_.c * params_.d * params_.h *
                                         params_.w
                                ? true
                                : false;
  }
  dev_ctx.template Alloc<AccT>(mean);
  dev_ctx.template Alloc<AccT>(var);
  auto* mean_data = mean->data<AccT>();
  auto* var_data = var->data<AccT>();
  params_.var_data = var_data;

  int32_t cPerBlock = 320;
  int32_t maxBlocksPerDHW = 1024;
  switch (params_.c) {
    case 2048:
    case 1024:
      cPerBlock = 512;
      break;
    case 960:
    case 1920:
      cPerBlock = 480;
      break;
    case 512:
    case 256:
      cPerBlock = 256;
      break;
    case 128:
      cPerBlock = 128;
      break;
    default:
      cPerBlock = 320;
  }
  params_.groups = groups;
  params_.cPerGroup = params_.c / params_.groups;
  if (cPerBlock % params_.cPerGroup != 0) {
    cPerBlock = params_.cPerGroup;
  }
  params_.srcX = reinterpret_cast<const T*>(x_data);
  params_.dst = reinterpret_cast<T*>(y_data);
  if (residual_ptr) {
    params_.srcR = reinterpret_cast<const T*>(residual_data);
    params_.eleOut = reinterpret_cast<T*>(residual_out_data);
  }
  params_.gamma = scale_data;
  params_.beta = bias_data;
  params_.dhw = static_cast<int64_t>(params_.d) * params_.h * params_.w;
  const int64_t blocksPerDHW =
      findMaxDivisor(params_.dhw, static_cast<int64_t>(maxBlocksPerDHW));
  params_.dhwPerBlock = divUp(params_.dhw, blocksPerDHW);
  params_.cPerBlock = cPerBlock;
  params_.dhwc = params_.dhw * params_.c;
  params_.invDHWC = 1.F / static_cast<float>(params_.dhw * params_.cPerGroup);
  params_.eps = epsilon;
  auto stream = dev_ctx.stream();
  DenseTensor redBuffer;
  int buffer_sizes = 2 * params_.n * groups;
  redBuffer.Resize({1, buffer_sizes});
  params_.redBuffer = dev_ctx.template Alloc<float>(&redBuffer);
  int64_t max_grid_x = dev_ctx.GetCUDAMaxGridDimSize()[0];
  int64_t max_grid_y = dev_ctx.GetCUDAMaxGridDimSize()[1];
  int64_t max_grid_z = dev_ctx.GetCUDAMaxGridDimSize()[2];
  if (params_.n > max_grid_z) {
    PADDLE_THROW(common::errors::Unimplemented(
        "GroupNorm kernel launch failed: 'batch_size' (%ld) exceeds the "
        "maximum supported limit (%ld). "
        "Please reduce the batch size or modify the kernel configuration.",
        params_.n,
        max_grid_z));
  }

  if (divUp(params_.dhw, params_.dhwPerBlock) > max_grid_y) {
    PADDLE_THROW(common::errors::Unimplemented(
        "GroupNorm kernel launch failed: computed gridDim.y (%ld) exceeds the "
        "hardware limit (%ld). "
        "This may be due to excessively large 'dhw' (%ld). Consider reducing "
        "input spatial dimensions "
        "or adjusting 'dhwPerBlock'.",
        divUp(params_.dhw, params_.dhwPerBlock),
        max_grid_y,
        params_.dhw));
  }

  if (divUp(params_.c, std::max(params_.cPerBlock, 128)) > max_grid_x) {
    PADDLE_THROW(common::errors::Unimplemented(
        "GroupNorm kernel launch failed: computed gridDim.x (%ld) exceeds the "
        "hardware limit (%ld). "
        "This may be due to excessively large channel count 'c' (%ld). "
        "Consider reducing the number of channels "
        "or adjusting 'cPerBlock'.",
        divUp(params_.c, std::max(params_.cPerBlock, 128)),
        max_grid_x,
        params_.c));
  }
#ifdef PADDLE_WITH_HIP
  hipMemset(params_.redBuffer, 0, buffer_sizes * sizeof(float));
#else
  cudaMemset(params_.redBuffer, 0, buffer_sizes * sizeof(float));
#endif
  groupNormNDHWCSum<T> ndhwc_sum;
  ndhwc_sum(&params_, stream);
  groupNormNDHWCScale<T> ndhwc_scale;
  ndhwc_scale(params_, stream);
#ifdef PADDLE_WITH_HIP
  backends::gpu::GpuMemcpyAsync(mean_data,
                                params_.redBuffer,
                                params_.n * groups * sizeof(float),
                                hipMemcpyDeviceToHost,
                                stream);
#else
  backends::gpu::GpuMemcpyAsync(mean_data,
                                params_.redBuffer,
                                params_.n * groups * sizeof(float),
                                cudaMemcpyDeviceToHost,
                                stream);
#endif
}

// ============================================================================
// PyTorch-aligned NCHW forward implementation using Welford algorithm
// ============================================================================

// Welford data structure for online mean/variance computation
template <typename AccT>
struct WelfordData {
  AccT mean;
  AccT m2;
  int64_t n;
  AccT nf;

  __host__ __device__ WelfordData() : mean(0), m2(0), n(0), nf(0) {}
  __host__ __device__ WelfordData(AccT mean, AccT m2, int64_t n, AccT nf)
      : mean(mean), m2(m2), n(n), nf(nf) {}
};

// Welford online update: incorporate a single data point
template <typename AccT>
__device__ __forceinline__ WelfordData<AccT> WelfordReduce(
    WelfordData<AccT> acc, AccT data) {
  int64_t new_n = acc.n + 1;
  AccT new_nf = static_cast<AccT>(new_n);
  AccT delta = data - acc.mean;
  AccT new_mean = acc.mean + delta / new_nf;
  AccT new_delta = data - new_mean;
  return {new_mean, acc.m2 + delta * new_delta, new_n, new_nf};
}

// Welford combine: merge two partial aggregates
template <typename AccT>
__device__ __forceinline__ WelfordData<AccT> WelfordCombine(
    WelfordData<AccT> a, WelfordData<AccT> b) {
  if (a.nf == 0) return b;
  if (b.nf == 0) return a;
  AccT delta = b.mean - a.mean;
  AccT new_count = a.nf + b.nf;
  AccT nb_over_n = b.nf / new_count;
  return {a.mean + delta * nb_over_n,
          a.m2 + b.m2 + delta * delta * a.nf * nb_over_n,
          -1,
          new_count};
}

// Warp-level Welford reduce using shuffle-down
template <typename AccT>
__device__ __forceinline__ WelfordData<AccT> WelfordWarpReduce(
    WelfordData<AccT> val) {
#pragma unroll
  for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
    WelfordData<AccT> other;
    other.mean =
        backends::gpu::CudaShuffleDownSync(0xffffffff, val.mean, offset);
    other.m2 = backends::gpu::CudaShuffleDownSync(0xffffffff, val.m2, offset);
    other.n = backends::gpu::CudaShuffleDownSync(0xffffffff, val.n, offset);
    other.nf = backends::gpu::CudaShuffleDownSync(0xffffffff, val.nf, offset);
    val = WelfordCombine(val, other);
  }
  return val;
}

// Block-level Welford reduce (for > warpSize threads)
template <typename AccT>
__device__ __forceinline__ WelfordData<AccT> WelfordBlockReduce(
    WelfordData<AccT> val) {
  // Shared memory for warp results (max 32 warps = 1024/32 or 1024/64)
  constexpr int kMaxWarps = 32;
  __shared__ AccT s_mean[kMaxWarps];
  __shared__ AccT s_m2[kMaxWarps];
  __shared__ int64_t s_n[kMaxWarps];
  __shared__ AccT s_nf[kMaxWarps];

  int tid = threadIdx.x;
  int lid = tid % warpSize;
  int wid = tid / warpSize;
  int num_warps = blockDim.x / warpSize;

  // First reduce within each warp
  val = WelfordWarpReduce(val);
  __syncthreads();

  // Warp leaders write to shared memory
  if (lid == 0) {
    s_mean[wid] = val.mean;
    s_m2[wid] = val.m2;
    s_n[wid] = val.n;
    s_nf[wid] = val.nf;
  }
  __syncthreads();

  // First warp reads and reduces across warps
  WelfordData<AccT> zero_val(0, 0, 0, 0);
  val = (tid < num_warps)
            ? WelfordData<AccT>{s_mean[lid], s_m2[lid], s_n[lid], s_nf[lid]}
            : zero_val;
  if (wid == 0) {
    val = WelfordWarpReduce(val);
  }
  return val;
}

// Constant matching PyTorch's kCUDABlockReduceNumThreads
constexpr int kBlockReduceNumThreads = 512;

// Phase 1: Welford moments kernel - computes mean and rstd per group
// Grid: N*G blocks. Each block processes D*HxW elements for one (n,g) pair.
// Matches PyTorch's RowwiseMomentsCUDAKernel exactly.
template <typename T, typename AccT>
__global__ void WelfordMomentsCUDAKernel(
    int64_t D_times_HxW, AccT eps, const T* X, AccT* mean_out, AccT* var_out) {
  const int64_t i = blockIdx.x;
  WelfordData<AccT> val(0, 0, 0, 0);

  for (int64_t j = threadIdx.x; j < D_times_HxW; j += blockDim.x) {
    const int64_t index = i * D_times_HxW + j;
    val = WelfordReduce(val, static_cast<AccT>(X[index]));
  }

  if (blockDim.x <= static_cast<unsigned>(warpSize)) {
    val = WelfordWarpReduce(val);
  } else {
    val = WelfordBlockReduce(val);
  }

  if (threadIdx.x == 0) {
    AccT final_mean = val.mean;
    AccT final_var = val.m2 / val.nf;
    mean_out[i] = final_mean;
    var_out[i] = final_var;
  }
}

// Phase 2a: Compute fused parameters a and b
// a[n*C + c] = rstd[n*G + c/D] * gamma[c]
// b[n*C + c] = -a[n*C + c] * mean[n*G + c/D] + beta[c]
// Matches PyTorch's ComputeFusedParamsCUDAKernel.
// Key: mean and rstd are first rounded through T (input dtype) to match
// PyTorch's behavior where mean/rstd are stored in T.
template <typename T, typename AccT>
__global__ void ComputeFusedParamsCUDAKernel(int64_t N,
                                             int64_t C,
                                             int64_t G,
                                             AccT eps,
                                             const AccT* mean,
                                             const AccT* var,
                                             const T* gamma,
                                             const T* beta,
                                             AccT* a,
                                             AccT* b) {
  const int64_t index =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < N * C) {
    const int64_t D = C / G;
    const int64_t ng = index / D;
    const int64_t c = index % C;
    // Round mean and rstd through T to match PyTorch's behavior
    // (PyTorch stores mean/rstd in input dtype T, then reads them back)
    AccT mean_val = static_cast<AccT>(static_cast<T>(mean[ng]));
    AccT rstd_val = static_cast<AccT>(static_cast<T>(rsqrt(var[ng] + eps)));
    const AccT scale =
        (gamma == nullptr) ? rstd_val : rstd_val * static_cast<AccT>(gamma[c]);
    a[index] = scale;
    b[index] =
        -scale * mean_val +
        ((beta == nullptr) ? static_cast<AccT>(0) : static_cast<AccT>(beta[c]));
  }
}

// Phase 2b: Element-wise normalization: Y = a * X + b
// Matches PyTorch's approach of Y = a[n*C+c] * float(X[idx]) + b[n*C+c].
template <typename T, typename AccT>
__global__ void GroupNormForwardElementwiseCUDAKernel(
    int64_t N_C, int64_t HxW, const T* X, const AccT* a, const AccT* b, T* Y) {
  const int64_t total = N_C * HxW;
  for (int64_t idx =
           static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < total;
       idx += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int64_t nc = idx / HxW;
    Y[idx] = static_cast<T>(a[nc] * static_cast<AccT>(X[idx]) + b[nc]);
  }
}

// Forward without gamma/beta - simple (X - mean) * rstd
// Rounds mean/rstd through T to match PyTorch's precision behavior.
template <typename T, typename AccT>
__global__ void GroupNormForwardNoScaleBiasCUDAKernel(int64_t N_G,
                                                      int64_t D_HxW,
                                                      const T* X,
                                                      const AccT* mean,
                                                      const AccT* var,
                                                      AccT eps,
                                                      T* Y) {
  const int64_t total = N_G * D_HxW;
  for (int64_t idx =
           static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < total;
       idx += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int64_t ng = idx / D_HxW;
    // Round through T to match PyTorch behavior
    AccT mean_val = static_cast<AccT>(static_cast<T>(mean[ng]));
    AccT rstd_val = static_cast<AccT>(static_cast<T>(rsqrt(var[ng] + eps)));
    Y[idx] = static_cast<T>((static_cast<AccT>(X[idx]) - mean_val) * rstd_val);
  }
}

// Phase 2b vectorized: float4 version for T=float, HxW divisible by 4
template <typename T, typename AccT>
__global__ void GroupNormForwardElementwiseVec4CUDAKernel(
    int64_t N_C,
    int64_t HxW_vec,
    const T* __restrict__ X,
    const AccT* __restrict__ a,
    const AccT* __restrict__ b,
    T* __restrict__ Y) {
  const int64_t total = N_C * HxW_vec;
  for (int64_t idx =
           static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < total;
       idx += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int64_t nc = idx / HxW_vec;
    const AccT a_val = a[nc];
    const AccT b_val = b[nc];
    float4 x4 = reinterpret_cast<const float4*>(X)[idx];
    float4 y4;
    y4.x = static_cast<float>(a_val * static_cast<AccT>(x4.x) + b_val);
    y4.y = static_cast<float>(a_val * static_cast<AccT>(x4.y) + b_val);
    y4.z = static_cast<float>(a_val * static_cast<AccT>(x4.z) + b_val);
    y4.w = static_cast<float>(a_val * static_cast<AccT>(x4.w) + b_val);
    reinterpret_cast<float4*>(Y)[idx] = y4;
  }
}

// Phase 2b vectorized: double2 version for T=double, HxW divisible by 2
template <typename T, typename AccT>
__global__ void GroupNormForwardElementwiseVec2DoubleCUDAKernel(
    int64_t N_C,
    int64_t HxW_vec,
    const T* __restrict__ X,
    const AccT* __restrict__ a,
    const AccT* __restrict__ b,
    T* __restrict__ Y) {
  const int64_t total = N_C * HxW_vec;
  for (int64_t idx =
           static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < total;
       idx += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int64_t nc = idx / HxW_vec;
    const AccT a_val = a[nc];
    const AccT b_val = b[nc];
    double2 x2 = reinterpret_cast<const double2*>(X)[idx];
    double2 y2;
    y2.x = static_cast<double>(a_val * static_cast<AccT>(x2.x) + b_val);
    y2.y = static_cast<double>(a_val * static_cast<AccT>(x2.y) + b_val);
    reinterpret_cast<double2*>(Y)[idx] = y2;
  }
}

// Forward without gamma/beta vectorized: float4 version
template <typename T, typename AccT>
__global__ void GroupNormForwardNoScaleBiasVec4CUDAKernel(
    int64_t N_G,
    int64_t D_HxW_vec,
    const T* __restrict__ X,
    const AccT* __restrict__ mean,
    const AccT* __restrict__ var,
    AccT eps,
    T* __restrict__ Y) {
  const int64_t total = N_G * D_HxW_vec;
  for (int64_t idx =
           static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < total;
       idx += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int64_t ng = idx / D_HxW_vec;
    // Round through T to match PyTorch behavior
    AccT mean_val = static_cast<AccT>(static_cast<T>(mean[ng]));
    AccT rstd_val = static_cast<AccT>(static_cast<T>(rsqrt(var[ng] + eps)));
    float4 x4 = reinterpret_cast<const float4*>(X)[idx];
    float4 y4;
    y4.x = static_cast<float>((static_cast<AccT>(x4.x) - mean_val) * rstd_val);
    y4.y = static_cast<float>((static_cast<AccT>(x4.y) - mean_val) * rstd_val);
    y4.z = static_cast<float>((static_cast<AccT>(x4.z) - mean_val) * rstd_val);
    y4.w = static_cast<float>((static_cast<AccT>(x4.w) - mean_val) * rstd_val);
    reinterpret_cast<float4*>(Y)[idx] = y4;
  }
}

// Forward without gamma/beta vectorized: double2 version
template <typename T, typename AccT>
__global__ void GroupNormForwardNoScaleBiasVec2DoubleCUDAKernel(
    int64_t N_G,
    int64_t D_HxW_vec,
    const T* __restrict__ X,
    const AccT* __restrict__ mean,
    const AccT* __restrict__ var,
    AccT eps,
    T* __restrict__ Y) {
  const int64_t total = N_G * D_HxW_vec;
  for (int64_t idx =
           static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < total;
       idx += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int64_t ng = idx / D_HxW_vec;
    AccT mean_val = static_cast<AccT>(static_cast<T>(mean[ng]));
    AccT rstd_val = static_cast<AccT>(static_cast<T>(rsqrt(var[ng] + eps)));
    double2 x2 = reinterpret_cast<const double2*>(X)[idx];
    double2 y2;
    y2.x = static_cast<double>((static_cast<AccT>(x2.x) - mean_val) * rstd_val);
    y2.y = static_cast<double>((static_cast<AccT>(x2.y) - mean_val) * rstd_val);
    reinterpret_cast<double2*>(Y)[idx] = y2;
  }
}

// Fused normalization kernel that computes Y on the fly without temp buffers
// Rounds mean/rstd through T to match PyTorch's precision behavior.
template <typename T, typename AccT>
__global__ void GroupNormForwardFusedNCHWKernel(int64_t N_C,
                                                int64_t HxW,
                                                int64_t C,
                                                int64_t G,
                                                const T* X,
                                                const AccT* mean,
                                                const AccT* var,
                                                const T* gamma,
                                                const T* beta,
                                                AccT eps,
                                                T* Y) {
  const int64_t D = C / G;
  const int64_t total = N_C * HxW;
  for (int64_t idx =
           static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < total;
       idx += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int64_t nc = idx / HxW;
    const int64_t c = nc % C;
    const int64_t ng = nc / D;
    // Round through T to match PyTorch behavior
    AccT mean_val = static_cast<AccT>(static_cast<T>(mean[ng]));
    AccT rstd_val = static_cast<AccT>(static_cast<T>(rsqrt(var[ng] + eps)));
    AccT scale_val =
        (gamma == nullptr) ? rstd_val : rstd_val * static_cast<AccT>(gamma[c]);
    AccT bias_val =
        (beta == nullptr) ? static_cast<AccT>(0) : static_cast<AccT>(beta[c]);
    Y[idx] = static_cast<T>(scale_val * static_cast<AccT>(X[idx]) +
                            (-scale_val * mean_val + bias_val));
  }
}

// Keep the old NHWC kernels for GroupNormForwardGetMeanAndVar and
// GroupNormForward (used by NHWC float32/64 path and
// GroupNormDirectCUDAFunctor)
template <typename T, typename AccT>
__global__ void GroupNormForwardGetMeanAndVar(const T* x,
                                              int64_t N,
                                              int64_t C,
                                              int64_t W,
                                              int64_t imsize,
                                              int groups,
                                              int64_t group_size,
                                              AccT* mean,
                                              AccT* var) {
  int64_t gid = blockIdx.y;
  for (int64_t cid = blockIdx.x; cid < group_size; cid += gridDim.x) {
    for (int64_t bid = blockIdx.z; bid < N; bid += gridDim.z) {
      int64_t H = imsize / W;
      int64_t number =
          min(group_size, static_cast<int64_t>(C - gid * group_size));
      int64_t ccid = gid * group_size + cid;
      if (ccid >= C) return;
      AccT x_mean = static_cast<AccT>(0);
      AccT x_var = static_cast<AccT>(0);
      for (int64_t imid = threadIdx.x; imid < imsize; imid += blockDim.x) {
        AccT val;
        int64_t hid = imid / W;
        int64_t wid = imid % W;
        val = static_cast<AccT>(x[(bid * H + hid) * W * C + wid * C + ccid]);

        x_mean += val;
        x_var += val * val;
      }
      x_mean /= number * imsize;
      x_var /= number * imsize;

#ifdef __NVCC__
      CudaAtomicAddWithWarp(&mean[bid * groups + gid], x_mean);
      CudaAtomicAddWithWarp(&var[bid * groups + gid], x_var);
#endif
#ifdef __HIPCC__
      if (blockDim.x < kps::details::kWarpSize) {
        CudaAtomicAdd(&mean[bid * groups + gid], x_mean);
        CudaAtomicAdd(&var[bid * groups + gid], x_var);
      } else {
        CudaAtomicAddWithWarp(&mean[bid * groups + gid], x_mean);
        CudaAtomicAddWithWarp(&var[bid * groups + gid], x_var);
      }
#endif
    }
  }
}

template <typename T, typename AccT, int flags>
__global__ void GroupNormForward(const T* x,
                                 const AccT* mean,
                                 const AccT* var,
                                 const T* scale,
                                 const T* bias,
                                 int64_t N,
                                 int64_t C,
                                 int64_t W,
                                 int64_t imsize,
                                 int groups,
                                 int64_t group_size,
                                 AccT epsilon,
                                 T* y,
                                 AccT* real_var,
                                 const DataLayout data_layout) {
  int64_t gid = blockIdx.y;
  for (int64_t cid = blockIdx.x; cid < group_size; cid += gridDim.x) {
    for (int64_t bid = blockIdx.z; bid < N; bid += gridDim.z) {
      int64_t H = imsize / W;
      int64_t ccid = gid * group_size + cid;
      if (ccid >= C) return;
      auto ng = bid * groups + gid;
      AccT x_mean = mean[ng];
      AccT x_var = var[ng];
      x_var = x_var - x_mean * x_mean;

      AccT var_inv = rsqrt(x_var + epsilon);
      if (cid == 0 && threadIdx.x == 0) {
        real_var[ng] = x_var;
      }
      for (int64_t imid = threadIdx.x; imid < imsize; imid += blockDim.x) {
        AccT val;
        int64_t hid, wid;
        int64_t index = (bid * C + ccid) * imsize + imid;
        if (data_layout == DataLayout::NCHW) {
          val = static_cast<AccT>(x[index]);
        } else {
          hid = imid / W;
          wid = imid % W;
          val = static_cast<AccT>(x[(bid * H + hid) * W * C + wid * C + ccid]);
        }
        val = (val - x_mean) * var_inv;
        if (flags & kHasScale) {
          val *= static_cast<AccT>(scale[ccid]);
        }
        if (flags & kHasBias) {
          val += static_cast<AccT>(bias[ccid]);
        }
        if (data_layout == DataLayout::NCHW) {
          y[index] = static_cast<T>(val);
        } else {
          y[(bid * H + hid) * W * C + wid * C + ccid] = static_cast<T>(val);
        }
      }
    }
  }
}

template <typename T, typename AccT>
void GroupNormDirectCUDAFunctor<T, AccT>::operator()(
    gpuStream_t stream,
    const T* input,
    std::vector<int> input_shape,
    const T* bias,
    const T* scale,
    AccT* temp_variance,
    int groups,
    float eps,
    T* output,
    AccT* mean,
    AccT* variance,
    const DataLayout data_layout) {
  const auto input_ddim = make_ddim(input_shape);
  const int64_t C =
      (data_layout == DataLayout::NCHW ? input_ddim[1]
                                       : input_ddim[input_ddim.size() - 1]);
  const int64_t group_size = C / groups;
  const int64_t W =
      (data_layout == DataLayout::NCHW ? input_ddim[input_ddim.size() - 1]
                                       : input_ddim[input_ddim.size() - 2]);

  int64_t image_size = 1;
  if (data_layout == DataLayout::NCHW) {
    for (int i = 2; i < input_ddim.size(); ++i) {
      image_size *= input_ddim[i];
    }
  } else {
    for (int i = 1; i < input_ddim.size() - 1; ++i) {
      image_size *= input_ddim[i];
    }
  }
  int block_size = std::min(static_cast<int64_t>(1024), image_size);
  int64_t max_grid_x = 65535;
  dim3 grid(std::min(group_size, max_grid_x),
            groups,
            std::min(input_ddim[0], max_grid_x));
  dim3 threads(block_size, 1, 1);
  if (data_layout == DataLayout::NCHW) {
    if (FLAGS_use_accuracy_compatible_kernel) {
      // =========================================================
      // PyTorch-compatible NCHW path using Welford algorithm
      // =========================================================
      const int64_t N = input_ddim[0];
      const int64_t G = groups;
      const int64_t D = C / G;
      const int64_t D_times_HxW = D * image_size;
      const int64_t num_threads =
          D_times_HxW < kBlockReduceNumThreads ? 32 : kBlockReduceNumThreads;

      // Phase 1: Welford moments -> mean and centered variance in temp_variance
      WelfordMomentsCUDAKernel<T, AccT><<<N * G, num_threads, 0, stream>>>(
          D_times_HxW, static_cast<AccT>(eps), input, mean, temp_variance);

      // Phase 2: Fused normalization using mean/var on the fly
      if (scale != nullptr || bias != nullptr) {
        const int64_t total = N * C * image_size;
        const int64_t elem_threads = 256;
        const int64_t elem_blocks =
            std::min((total + elem_threads - 1) / elem_threads, max_grid_x);
        GroupNormForwardFusedNCHWKernel<T, AccT>
            <<<elem_blocks, elem_threads, 0, stream>>>(N * C,
                                                       image_size,
                                                       C,
                                                       G,
                                                       input,
                                                       mean,
                                                       temp_variance,
                                                       scale,
                                                       bias,
                                                       static_cast<AccT>(eps),
                                                       output);
      } else {
        const int64_t total = N * G * D_times_HxW;
        const int64_t elem_threads = 256;
        const int64_t elem_blocks =
            std::min((total + elem_threads - 1) / elem_threads, max_grid_x);
        GroupNormForwardNoScaleBiasCUDAKernel<T, AccT>
            <<<elem_blocks, elem_threads, 0, stream>>>(N * G,
                                                       D_times_HxW,
                                                       input,
                                                       mean,
                                                       temp_variance,
                                                       static_cast<AccT>(eps),
                                                       output);
      }

      // Copy centered variance to variance output
#ifdef PADDLE_WITH_HIP
      hipMemcpyAsync(variance,
                     temp_variance,
                     N * G * sizeof(AccT),
                     hipMemcpyDeviceToDevice,
                     stream);
#else
      cudaMemcpyAsync(variance,
                      temp_variance,
                      N * G * sizeof(AccT),
                      cudaMemcpyDeviceToDevice,
                      stream);
#endif
    } else {
      // =========================================================
      // Original high-performance NCHW path
      // =========================================================
      constexpr int vec_size = sizeof(float4) / sizeof(T);
      int64_t size = group_size * image_size;
      const int max_num_threads = 1024;
      int max_block_size =
          std::min(static_cast<int>(size / vec_size), max_num_threads);
      int block_size_nchw = 1;
      while (block_size_nchw < max_block_size) {
        block_size_nchw *= 2;
      }
      block_size_nchw = std::max(block_size_nchw, kps::details::kWarpSize);
      int64_t n_groups = input_ddim[0] * static_cast<int64_t>(groups);
      dim3 grids(std::min(max_grid_x, n_groups));
      dim3 blocks(block_size_nchw);
      if (size < vec_size * block_size_nchw) {
        ScalarGetMeanAndVarNCHW<T, AccT><<<grids, blocks, 0, stream>>>(
            input, mean, temp_variance, size, n_groups);
      } else {
        VectorizedGetMeanAndVarNCHW<T, AccT, vec_size>
            <<<grids, blocks, 0, stream>>>(
                input, mean, temp_variance, size, n_groups);
      }
      GroupNormForward<T, AccT, 3>
          <<<grid, threads, 0, stream>>>(input,
                                         mean,
                                         temp_variance,
                                         scale,
                                         bias,
                                         input_ddim[0],
                                         C,
                                         W,
                                         image_size,
                                         groups,
                                         group_size,
                                         static_cast<AccT>(eps),
                                         output,
                                         variance,
                                         data_layout);
    }
  } else {
#ifdef PADDLE_WITH_HIP
    hipMemset(mean, 0, sizeof(AccT) * input_ddim[0] * groups);
    hipMemset(temp_variance, 0, sizeof(AccT) * input_ddim[0] * groups);
#else
    cudaMemset(mean, 0, sizeof(AccT) * input_ddim[0] * groups);
    cudaMemset(temp_variance, 0, sizeof(AccT) * input_ddim[0] * groups);
#endif

    GroupNormForwardGetMeanAndVar<T, AccT>
        <<<grid, threads, 0, stream>>>(input,
                                       input_ddim[0],
                                       C,
                                       W,
                                       image_size,
                                       groups,
                                       group_size,
                                       mean,
                                       temp_variance);
    GroupNormForward<T, AccT, 3>
        <<<grid, threads, 0, stream>>>(input,
                                       mean,
                                       temp_variance,
                                       scale,
                                       bias,
                                       input_ddim[0],
                                       C,
                                       W,
                                       image_size,
                                       groups,
                                       group_size,
                                       static_cast<AccT>(eps),
                                       output,
                                       variance,
                                       data_layout);
  }
}
template class PADDLE_API GroupNormDirectCUDAFunctor<float, float>;
#if defined(PADDLE_WITH_CUDA) && !defined(PADDLE_WITH_HIP)
template class PADDLE_API GroupNormDirectCUDAFunctor<half, float>;
#endif

template <typename T, typename Context>
void GroupNormGeneralCaseKernel(const Context& dev_ctx,
                                const DenseTensor& x,
                                const optional<DenseTensor>& scale,
                                const optional<DenseTensor>& bias,
                                double epsilon,
                                int groups,
                                const std::string& data_layout_str,
                                DenseTensor* y,
                                DenseTensor* mean,
                                DenseTensor* var) {
  using AccT = typename MPTypeTrait<T>::Type;
  const DataLayout data_layout = StringToDataLayout(data_layout_str);
  const auto scale_ptr = scale.get_ptr();
  const auto bias_ptr = bias.get_ptr();
  const auto x_dims = x.dims();
  const int64_t N = x_dims[0];
  const int64_t C =
      (data_layout == DataLayout::NCHW ? x_dims[1] : x_dims[x_dims.size() - 1]);
  const int64_t group_size = C / groups;
  const int64_t W =
      (data_layout == DataLayout::NCHW ? x_dims[x_dims.size() - 1]
                                       : x_dims[x_dims.size() - 2]);
  const int64_t G = groups;
  const int64_t D = C / G;

  dev_ctx.template Alloc<T>(y);
  dev_ctx.template Alloc<AccT>(mean);
  dev_ctx.template Alloc<AccT>(var);

  auto* x_data = x.data<T>();
  auto* y_data = y->data<T>();
  auto* mean_data = mean->data<AccT>();
  auto* var_data = var->data<AccT>();

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
    if (FLAGS_use_accuracy_compatible_kernel) {
      // =========================================================
      // PyTorch-compatible NCHW path using Welford algorithm
      // =========================================================
      const int64_t D_times_HxW = D * imsize;

      // Phase 1: Compute moments using Welford algorithm
      const int64_t num_threads =
          D_times_HxW < kBlockReduceNumThreads ? 32 : kBlockReduceNumThreads;

      WelfordMomentsCUDAKernel<T, AccT>
          <<<N * G, num_threads, 0, dev_ctx.stream()>>>(
              D_times_HxW,
              static_cast<AccT>(epsilon),
              x_data,
              mean_data,
              var_data);

      // Phase 2: Normalization
      if (scale_data != nullptr || bias_data != nullptr) {
        // Phase 2a: Compute fused params a and b
        DenseTensor a_tensor, b_tensor;
        a_tensor.Resize({N, C});
        b_tensor.Resize({N, C});
        AccT* a_data = dev_ctx.template Alloc<AccT>(&a_tensor);
        AccT* b_data = dev_ctx.template Alloc<AccT>(&b_tensor);

        constexpr int64_t kNumThreads = 256;
        const int64_t B = (N * C + kNumThreads - 1) / kNumThreads;
        ComputeFusedParamsCUDAKernel<T, AccT>
            <<<B, kNumThreads, 0, dev_ctx.stream()>>>(
                N,
                C,
                G,
                static_cast<AccT>(epsilon),
                mean_data,
                var_data,
                scale_data,
                bias_data,
                a_data,
                b_data);

        // Phase 2b: Element-wise Y = a * X + b
        constexpr int64_t kElemThreads = 256;
        int64_t max_grid_x = dev_ctx.GetCUDAMaxGridDimSize()[0];

        if (std::is_same<T, float>::value && (imsize % 4 == 0)) {
          const int64_t HxW_vec = imsize / 4;
          const int64_t total_vec = N * C * HxW_vec;
          const int64_t elem_blocks = std::min(
              (total_vec + kElemThreads - 1) / kElemThreads, max_grid_x);
          GroupNormForwardElementwiseVec4CUDAKernel<T, AccT>
              <<<elem_blocks, kElemThreads, 0, dev_ctx.stream()>>>(
                  N * C, HxW_vec, x_data, a_data, b_data, y_data);
        } else if (std::is_same<T, double>::value && (imsize % 2 == 0)) {
          const int64_t HxW_vec = imsize / 2;
          const int64_t total_vec = N * C * HxW_vec;
          const int64_t elem_blocks = std::min(
              (total_vec + kElemThreads - 1) / kElemThreads, max_grid_x);
          GroupNormForwardElementwiseVec2DoubleCUDAKernel<T, AccT>
              <<<elem_blocks, kElemThreads, 0, dev_ctx.stream()>>>(
                  N * C, HxW_vec, x_data, a_data, b_data, y_data);
        } else {
          const int64_t total = N * C * imsize;
          const int64_t elem_blocks =
              std::min((total + kElemThreads - 1) / kElemThreads, max_grid_x);
          GroupNormForwardElementwiseCUDAKernel<T, AccT>
              <<<elem_blocks, kElemThreads, 0, dev_ctx.stream()>>>(
                  N * C, imsize, x_data, a_data, b_data, y_data);
        }
      } else {
        // No scale/bias: Y = (X - mean) * rstd
        constexpr int64_t kElemThreads = 256;
        int64_t max_grid_x = dev_ctx.GetCUDAMaxGridDimSize()[0];

        if (std::is_same<T, float>::value && (D_times_HxW % 4 == 0)) {
          const int64_t D_HxW_vec = D_times_HxW / 4;
          const int64_t total_vec = N * G * D_HxW_vec;
          const int64_t elem_blocks = std::min(
              (total_vec + kElemThreads - 1) / kElemThreads, max_grid_x);
          GroupNormForwardNoScaleBiasVec4CUDAKernel<T, AccT>
              <<<elem_blocks, kElemThreads, 0, dev_ctx.stream()>>>(
                  N * G,
                  D_HxW_vec,
                  x_data,
                  mean_data,
                  var_data,
                  static_cast<AccT>(epsilon),
                  y_data);
        } else if (std::is_same<T, double>::value && (D_times_HxW % 2 == 0)) {
          const int64_t D_HxW_vec = D_times_HxW / 2;
          const int64_t total_vec = N * G * D_HxW_vec;
          const int64_t elem_blocks = std::min(
              (total_vec + kElemThreads - 1) / kElemThreads, max_grid_x);
          GroupNormForwardNoScaleBiasVec2DoubleCUDAKernel<T, AccT>
              <<<elem_blocks, kElemThreads, 0, dev_ctx.stream()>>>(
                  N * G,
                  D_HxW_vec,
                  x_data,
                  mean_data,
                  var_data,
                  static_cast<AccT>(epsilon),
                  y_data);
        } else {
          const int64_t D_total = N * G * D_times_HxW;
          const int64_t elem_blocks =
              std::min((D_total + kElemThreads - 1) / kElemThreads, max_grid_x);
          GroupNormForwardNoScaleBiasCUDAKernel<T, AccT>
              <<<elem_blocks, kElemThreads, 0, dev_ctx.stream()>>>(
                  N * G,
                  D_times_HxW,
                  x_data,
                  mean_data,
                  var_data,
                  static_cast<AccT>(epsilon),
                  y_data);
        }
      }
    } else {
      // =========================================================
      // Original high-performance NCHW path
      // =========================================================
      DenseTensor temp_var;
      temp_var.Resize(var->dims());
      dev_ctx.template Alloc<AccT>(&temp_var);
      auto* temp_var_data = temp_var.data<AccT>();

      constexpr int vec_size = sizeof(float4) / sizeof(T);
      int64_t size = group_size * imsize;
      const int max_num_threads = 1024;
      int max_block_size =
          std::min(static_cast<int>(size / vec_size), max_num_threads);
      int block_size_nchw = 1;
      while (block_size_nchw < max_block_size) {
        block_size_nchw *= 2;
      }
      block_size_nchw = std::max(block_size_nchw, kps::details::kWarpSize);
      int64_t n_groups = N * static_cast<int64_t>(groups);
      int64_t max_grid_x = dev_ctx.GetCUDAMaxGridDimSize()[0];
      dim3 grids(std::min(max_grid_x, n_groups));
      dim3 blocks(block_size_nchw);
      if (size < vec_size * block_size_nchw) {
        ScalarGetMeanAndVarNCHW<T, AccT>
            <<<grids, blocks, 0, dev_ctx.stream()>>>(
                x_data, mean_data, temp_var_data, size, n_groups);
      } else {
        VectorizedGetMeanAndVarNCHW<T, AccT, vec_size>
            <<<grids, blocks, 0, dev_ctx.stream()>>>(
                x_data, mean_data, temp_var_data, size, n_groups);
      }

      int64_t max_grid_z = dev_ctx.GetCUDAMaxGridDimSize()[2];
      int block_size_orig = std::min(static_cast<int64_t>(1024), imsize);
      dim3 grid(
          std::min(max_grid_x, group_size), groups, std::min(max_grid_z, N));
      dim3 threads(block_size_orig, 1, 1);
      int flags = (scale_data != nullptr) * kHasScale +
                  (bias_data != nullptr) * kHasBias;
      UNROLL_ALL_CASES(flags,
                       GroupNormForward,
                       x_data,
                       mean_data,
                       temp_var_data,
                       scale_data,
                       bias_data,
                       N,
                       C,
                       W,
                       imsize,
                       groups,
                       group_size,
                       static_cast<AccT>(epsilon),
                       y_data,
                       var_data,
                       data_layout);
    }
  } else {
    // =========================================================
    // NHWC float32/float64 path (legacy, kept as-is)
    // =========================================================
    DenseTensor temp_var;
    temp_var.Resize(var->dims());
    dev_ctx.template Alloc<AccT>(&temp_var);
    auto* temp_var_data = temp_var.data<AccT>();

    funcs::SetConstant<GPUContext, AccT> set_zero_AccT;
    set_zero_AccT(dev_ctx, mean, static_cast<AccT>(0));
    set_zero_AccT(dev_ctx, &temp_var, static_cast<AccT>(0));

    int block_size = std::min(static_cast<int64_t>(1024), imsize);
    int64_t max_grid_x = dev_ctx.GetCUDAMaxGridDimSize()[0];
    int64_t max_grid_z = dev_ctx.GetCUDAMaxGridDimSize()[2];
    dim3 grid(
        std::min(max_grid_x, group_size), groups, std::min(max_grid_z, N));
    dim3 threads(block_size, 1, 1);

    GroupNormForwardGetMeanAndVar<T,
                                  AccT><<<grid, threads, 0, dev_ctx.stream()>>>(
        x_data, N, C, W, imsize, groups, group_size, mean_data, temp_var_data);
    int flags =
        (scale_data != nullptr) * kHasScale + (bias_data != nullptr) * kHasBias;
    UNROLL_ALL_CASES(flags,
                     GroupNormForward,
                     x_data,
                     mean_data,
                     temp_var_data,
                     scale_data,
                     bias_data,
                     N,
                     C,
                     W,
                     imsize,
                     groups,
                     group_size,
                     static_cast<AccT>(epsilon),
                     y_data,
                     var_data,
                     data_layout);
  }
}

template <typename T, typename Context>
void GroupNormKernel(const Context& dev_ctx,
                     const DenseTensor& x,
                     const optional<DenseTensor>& scale,
                     const optional<DenseTensor>& bias,
                     double epsilon,
                     int groups,
                     const std::string& data_layout_str,
                     DenseTensor* y,
                     DenseTensor* mean,
                     DenseTensor* var) {
  if (y && y->numel() == 0) {
    dev_ctx.template Alloc<T>(y);
    if (mean) {
      Full<T, Context>(dev_ctx, mean->dims(), 0, mean);
    }
    if (var) {
      Full<T, Context>(dev_ctx, var->dims(), 0, var);
    }
    return;
  }
  using std::is_same;
  if (is_same<T, float16>::value && data_layout_str == "NHWC") {
    const optional<DenseTensor>& residual = optional<DenseTensor>(paddle::none);
    DenseTensor empty_tensor;
    GroupNormNDHWCKernel<float16, Context>(dev_ctx,
                                           x,
                                           residual,
                                           scale,
                                           bias,
                                           epsilon,
                                           groups,
                                           data_layout_str,
                                           "",
                                           y,
                                           &empty_tensor,
                                           mean,
                                           var);
    return;
  }

#ifdef PADDLE_CUDA_BF16
  if (is_same<T, bfloat16>::value && data_layout_str == "NHWC") {
    const optional<DenseTensor>& residual = optional<DenseTensor>(paddle::none);
    DenseTensor empty_tensor;
    GroupNormNDHWCKernel<bfloat16, Context>(dev_ctx,
                                            x,
                                            residual,
                                            scale,
                                            bias,
                                            epsilon,
                                            groups,
                                            data_layout_str,
                                            "",
                                            y,
                                            &empty_tensor,
                                            mean,
                                            var);
    return;
  }
#endif

  GroupNormGeneralCaseKernel<T, Context>(
      dev_ctx, x, scale, bias, epsilon, groups, data_layout_str, y, mean, var);
}
}  // namespace phi

PD_REGISTER_KERNEL(group_norm,
                   GPU,
                   ALL_LAYOUT,
                   phi::GroupNormKernel,
                   float,
                   double,
                   phi::bfloat16,
                   phi::float16) {
  if (kernel_key.dtype() == phi::DataType::BFLOAT16 ||
      kernel_key.dtype() == phi::DataType::FLOAT16) {
    kernel->OutputAt(1).SetDataType(phi::DataType::FLOAT32);
    kernel->OutputAt(2).SetDataType(phi::DataType::FLOAT32);
  }
}

PD_REGISTER_KERNEL(add_group_norm_silu,
                   GPU,
                   ALL_LAYOUT,
                   phi::GroupNormNDHWCKernel,
                   phi::bfloat16,
                   phi::float16) {
  kernel->OutputAt(2).SetDataType(phi::DataType::FLOAT32);
  kernel->OutputAt(3).SetDataType(phi::DataType::FLOAT32);
}
