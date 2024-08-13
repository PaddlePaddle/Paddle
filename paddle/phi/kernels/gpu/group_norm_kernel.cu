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
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/math_function.h"
#include "paddle/phi/kernels/gpu/group_norm_utils.h"

#include "paddle/phi/common/data_type.h"
#include "paddle/phi/common/float16.h"
#include "paddle/phi/core/device_context.h"

namespace phi {

static inline int32_t divUp(int32_t m, int32_t n) { return (m + n - 1) / n; }

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

static int32_t findMaxDivisor(int32_t n, int32_t maxAllowedDivisor) {
  int32_t maxDivisor = -1;
  for (int32_t i = 1; i <= std::sqrt(n); i++) {
    if (n % i == 0) {
      int32_t divisor1 = n / i;
      int32_t divisor2 = i;

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
  float src_data = phi::__2float<T>(*srcX);
  *sum += src_data;
  *sumSq += src_data * src_data;
}

template <typename T, int THREADS_PER_CHANNEL>
inline __device__ void UpdateSum(const T* srcX,
                                 const T* srcR,
                                 float* sum,
                                 float* sumSq) {
  float src_data = phi::__2float<T>(*srcX);
  float srcy_data = phi::__2float<T>(*srcR);
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
inline __device__ void UpdateSum<phi::dtype::float16, 2>(
    const phi::dtype::float16* srcX, float* sum, float* sumSq) {
  __half2 h2 = *reinterpret_cast<__half2 const*>(srcX);
  float2 f2 = __half22float2(h2);
  *sum += f2.x + f2.y;
  *sumSq += f2.x * f2.x + f2.y * f2.y;
}

template <>
inline __device__ void UpdateSum<phi::dtype::float16, 2>(
    const phi::dtype::float16* srcX,
    const phi::dtype::float16* srcR,
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
inline __device__ void UpdateSum<phi::dtype::bfloat16, 2>(
    const phi::dtype::bfloat16* srcX, float* sum, float* sumSq) {
  __nv_bfloat162 h2 = *reinterpret_cast<__nv_bfloat162 const*>(srcX);
  float2 f2 = phi::bfloat1622float2(h2);
  *sum += f2.x + f2.y;
  *sumSq += f2.x * f2.x + f2.y * f2.y;
}

template <>
inline __device__ void UpdateSum<phi::dtype::bfloat16, 2>(
    const phi::dtype::bfloat16* srcX,
    const phi::dtype::bfloat16* srcR,
    float* sum,
    float* sumSq) {
  __nv_bfloat162 h2 = *reinterpret_cast<__nv_bfloat162 const*>(srcX);
  __nv_bfloat162 h2_r = *reinterpret_cast<__nv_bfloat162 const*>(srcR);
  float2 f2 = phi::bfloat1622float2(h2);
  float2 f2_r = phi::bfloat1622float2(h2_r);
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
  int32_t ci = blockIdx.x * params.cPerBlock + threadIdx.x;
  if (ci >= params.c) {
    return;
  }
  // The first activation loaded by that block.
  int32_t dhwBegin = blockIdx.y * params.dhwPerBlock;
  // The last activation loaded by that block.
  int32_t dhwEnd = min(dhwBegin + params.dhwPerBlock, params.dhw);

  // The sums.
  float sum = 0.F;
  float sumSq = 0.F;

  for (int32_t dhwi = dhwBegin; dhwi < dhwEnd; ++dhwi) {
    // The offset.
    int64_t offset = static_cast<int64_t>(ni) * params.dhwc +
                     static_cast<int64_t>(dhwi) * params.c + ci;
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
  int32_t ci =
      blockIdx.x * params.cPerBlock + threadIdx.x * THREADS_PER_CHANNEL;
  if (ci >= params.c || threadIdx.x * THREADS_PER_CHANNEL >= params.cPerBlock) {
    return;
  }
  int32_t gj = ci / params.cPerGroup;
  int32_t cj = ci % params.cPerGroup;
  int32_t dhwBegin = blockIdx.y * params.dhwPerBlock;
  // The last activation loaded by that block.
  int32_t dhwEnd = min(dhwBegin + params.dhwPerBlock, params.dhw);

  // The sums.
  float sum = 0.F;
  float sumSq = 0.F;

  for (int32_t dhwi = dhwBegin; dhwi < dhwEnd; ++dhwi) {
    // The offset.
    int64_t offset = static_cast<int64_t>(ni) * params.dhwc +
                     static_cast<int64_t>(dhwi) * params.c + ci;
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
template class groupNormNDHWCSum<half>;

template <typename T, int THREADS_PER_CHANNEL>
inline __device__ void GroupNormCompute(int32_t dhwBegin,
                                        int32_t dhwEnd,
                                        int32_t ci,
                                        const GroupNormNDHWCParams<T>& params,
                                        float mean,
                                        float invStdDev) {
  float gamma =
      phi::__2float<T>(*(reinterpret_cast<T const*>(params.gamma) + ci));
  float beta =
      phi::__2float<T>(*(reinterpret_cast<T const*>(params.beta) + ci));
  for (int32_t dhwi = dhwBegin; dhwi < dhwEnd; ++dhwi) {
    // The src/dst offset.
    int64_t offset = (int64_t)blockIdx.z * params.dhwc + dhwi * params.c + ci;
    float src_data = phi::__2float<T>(params.srcX[offset]);
    if (params.srcR != nullptr) {
      auto gi = ci / params.cPerGroup;
      auto gj = ci % params.cPerGroup;
      int64_t g_offset =
          params.y_same_with_x ? offset : gi * params.cPerGroup + gj;
      src_data += phi::__2float<T>(params.srcR[g_offset]);
      *reinterpret_cast<T*>(&params.eleOut[offset]) = phi::__2dst<T>(src_data);
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
    *reinterpret_cast<T*>(&params.dst[offset]) = phi::__2dst<T>(dst_data);
  }
}

template <>
inline __device__ void GroupNormCompute<phi::dtype::float16, 2>(
    int32_t dhwBegin,
    int32_t dhwEnd,
    int32_t ci,
    const GroupNormNDHWCParams<phi::dtype::float16>& params,
    float mean,
    float invStdDev) {
  float2 gammaF2, betaF2;
  gammaF2 = __half22float2(*reinterpret_cast<__half2 const*>(
      reinterpret_cast<half const*>(params.gamma) + ci));
  betaF2 = __half22float2(*reinterpret_cast<__half2 const*>(
      reinterpret_cast<half const*>(params.beta) + ci));

  // Iterate over the activations to compute the sums.
  for (int32_t dhwi = dhwBegin; dhwi < dhwEnd; ++dhwi) {
    // The src/dst offset.
    int64_t offset = (int64_t)blockIdx.z * params.dhwc + dhwi * params.c + ci;

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
    int32_t dhwBegin,
    int32_t dhwEnd,
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
  for (int32_t dhwi = dhwBegin; dhwi < dhwEnd; ++dhwi) {
    // The src/dst offset.
    int64_t offset = (int64_t)blockIdx.z * params.dhwc + dhwi * params.c + ci;

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
inline __device__ void GroupNormCompute<phi::dtype::bfloat16, 2>(
    int32_t dhwBegin,
    int32_t dhwEnd,
    int32_t ci,
    const GroupNormNDHWCParams<phi::dtype::bfloat16>& params,
    float mean,
    float invStdDev) {
  float2 gammaF2, betaF2;
  gammaF2 = phi::bfloat1622float2(*reinterpret_cast<__nv_bfloat162 const*>(
      reinterpret_cast<__nv_bfloat16 const*>(params.gamma) + ci));
  betaF2 = phi::bfloat1622float2(*reinterpret_cast<__nv_bfloat162 const*>(
      reinterpret_cast<__nv_bfloat16 const*>(params.beta) + ci));

  // Iterate over the activations to compute the sums.
  for (int32_t dhwi = dhwBegin; dhwi < dhwEnd; ++dhwi) {
    // The src/dst offset.
    int64_t offset = (int64_t)blockIdx.z * params.dhwc + dhwi * params.c + ci;

    // Fetch two channels per thread.
    __nv_bfloat162 h2 =
        *reinterpret_cast<__nv_bfloat162 const*>(&params.srcX[offset]);

    // Extract the two half values.
    float2 f2 = phi::bfloat1622float2(h2);

    if (params.srcR != nullptr) {
      auto gi = ci / params.cPerGroup;
      auto gj = ci % params.cPerGroup;
      int64_t g_offset =
          params.y_same_with_x ? offset : gi * params.cPerGroup + gj;
      __nv_bfloat162 r2 =
          *reinterpret_cast<__nv_bfloat162 const*>(&params.srcR[g_offset]);
      float2 r_f2 = phi::bfloat1622float2(r2);
      f2.x += r_f2.x;
      f2.y += r_f2.y;
      *reinterpret_cast<__nv_bfloat162*>(&params.eleOut[offset]) =
          phi::float22bfloat162_rn(f2);
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
        phi::float22bfloat162_rn(f2);
  }
}
#endif

template <typename T, int THREADS_PER_CHANNEL>
__global__ void groupNormNDHWCScaleKernel(
    const GroupNormNDHWCParams<T> params) {
  // The instance in the batch.
  int32_t ni = blockIdx.z;
  // The channel loaded by that thread (2 channels per thread for F16x2).
  int32_t ci =
      blockIdx.x * params.cPerBlock + threadIdx.x * THREADS_PER_CHANNEL;

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
  int32_t dhwBegin = blockIdx.y * params.dhwPerBlock;
  // The last activation loaded by that block.
  int32_t dhwEnd = min(dhwBegin + params.dhwPerBlock, params.dhw);
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
template class groupNormNDHWCScale<half>;

template <typename T, typename Context>
void GroupNormNDHWCKernel(const Context& dev_ctx,
                          const DenseTensor& x,
                          const paddle::optional<DenseTensor>& residual,
                          const paddle::optional<DenseTensor>& scale,
                          const paddle::optional<DenseTensor>& bias,
                          float epsilon,
                          int groups,
                          const std::string& data_layout_str,
                          const std::string& activation,
                          DenseTensor* y,
                          DenseTensor* residual_out,
                          DenseTensor* mean,
                          DenseTensor* var) {
  const DataLayout data_layout = common::StringToDataLayout(data_layout_str);
  if (data_layout != DataLayout::kNHWC) {
    PD_THROW("data_layout only supports NHWC and NDHWC");
  }
  using AccT = typename phi::dtype::MPTypeTrait<T>::Type;
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
    int32_t r_dim = 1;
    for (size_t i = 0; i < r_dims.size(); i++) {
      r_dim *= r_dims[i];
    }
    params_.y_same_with_x =
        r_dim == params_.n * params_.c * params_.d * params_.h * params_.w
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
  params_.dhw = params_.d * params_.h * params_.w;
  const int32_t blocksPerDHW = findMaxDivisor(params_.dhw, maxBlocksPerDHW);
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
  phi::backends::gpu::GpuMemcpyAsync(mean_data,
                                     params_.redBuffer,
                                     params_.n * groups * sizeof(float),
                                     hipMemcpyDeviceToHost,
                                     stream);
#else
  phi::backends::gpu::GpuMemcpyAsync(mean_data,
                                     params_.redBuffer,
                                     params_.n * groups * sizeof(float),
                                     cudaMemcpyDeviceToHost,
                                     stream);
#endif
}

template <typename T, typename AccT>
__global__ void GroupNormForwardGetMeanAndVar(const T* x,
                                              int N,
                                              int C,
                                              int W,
                                              int imsize,
                                              int groups,
                                              int group_size,
                                              AccT* mean,
                                              AccT* var) {
  int gid = blockIdx.y;
  int cid = blockIdx.x;
  int bid = blockIdx.z;
  int H = imsize / W;
  int number = min(group_size, static_cast<int>(C - gid * group_size));
  int ccid = gid * group_size + cid;
  if (ccid >= C) return;
  AccT x_mean = static_cast<AccT>(0);
  AccT x_var = static_cast<AccT>(0);
  for (int imid = threadIdx.x; imid < imsize; imid += blockDim.x) {
    AccT val;
    int hid = imid / W;
    int wid = imid % W;
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
  // Note(wangyanpeng04): When the block size is less than the warp size,
  // WarpReduce will result in all zeros. It seems to be an internal problem of
  // hipcub on DCU.
  if (blockDim.x < phi::kps::details::kWarpSize) {
    phi::CudaAtomicAdd(&mean[bid * groups + gid], x_mean);
    phi::CudaAtomicAdd(&var[bid * groups + gid], x_var);
  } else {
    CudaAtomicAddWithWarp(&mean[bid * groups + gid], x_mean);
    CudaAtomicAddWithWarp(&var[bid * groups + gid], x_var);
  }
#endif
}

template <typename T, typename AccT, int flags>
__global__ void GroupNormForward(const T* x,
                                 const AccT* mean,
                                 const AccT* var,
                                 const T* scale,
                                 const T* bias,
                                 int N,
                                 int C,
                                 int W,
                                 int imsize,
                                 int groups,
                                 int group_size,
                                 AccT epsilon,
                                 T* y,
                                 AccT* real_var,
                                 const DataLayout data_layout) {
  int gid = blockIdx.y;
  int cid = blockIdx.x;
  int bid = blockIdx.z;
  int H = imsize / W;
  int ccid = gid * group_size + cid;
  if (ccid >= C) return;
  auto ng = bid * groups + gid;
  AccT x_mean = mean[ng];
  AccT x_var = var[ng];
  x_var = x_var - x_mean * x_mean;

  AccT var_inv = rsqrt(x_var + epsilon);
  if (cid == 0 && threadIdx.x == 0) {
    real_var[ng] = x_var;
  }
  for (int imid = threadIdx.x; imid < imsize; imid += blockDim.x) {
    AccT val;
    int hid, wid;
    int index = (bid * C + ccid) * imsize + imid;
    if (data_layout == DataLayout::kNCHW) {
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
    if (data_layout == DataLayout::kNCHW) {
      y[index] = static_cast<T>(val);
    } else {
      y[(bid * H + hid) * W * C + wid * C + ccid] = static_cast<T>(val);
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
  const auto input_ddim = common::make_ddim(input_shape);
  const int C =
      (data_layout == DataLayout::kNCHW ? input_ddim[1]
                                        : input_ddim[input_ddim.size() - 1]);
  const int group_size = C / groups;
  const int W =
      (data_layout == DataLayout::kNCHW ? input_ddim[input_ddim.size() - 1]
                                        : input_ddim[input_ddim.size() - 2]);

  int image_size = 1;
  if (data_layout == DataLayout::kNCHW) {
    for (int i = 2; i < input_ddim.size(); ++i) {
      image_size *= input_ddim[i];
    }
  } else {
    for (int i = 1; i < input_ddim.size() - 1; ++i) {
      image_size *= input_ddim[i];
    }
  }
  int block_size = std::min(1024, image_size);
  dim3 grid(group_size, groups, input_ddim[0]);
  dim3 threads(block_size, 1, 1);
  if (data_layout == DataLayout::kNCHW) {
    constexpr int vec_size = sizeof(float4) / sizeof(T);
    int size = group_size * image_size;  // group element size
    const int max_num_threads = 1024;
    int max_block_size = std::min(size / vec_size, max_num_threads);
    int block_size_nchw = 1;
    while (block_size_nchw < max_block_size) {
      block_size_nchw *= 2;
    }

    block_size_nchw = std::max(block_size_nchw, phi::kps::details::kWarpSize);
    dim3 grids(input_ddim[0] * groups);
    dim3 blocks(block_size_nchw);

    if (size < vec_size * block_size_nchw) {
      phi::ScalarGetMeanAndVarNCHW<T, AccT>
          <<<grids, blocks, 0, stream>>>(input, mean, temp_variance, size);
    } else {
      phi::VectorizedGetMeanAndVarNCHW<T, AccT, vec_size>
          <<<grids, blocks, 0, stream>>>(input, mean, temp_variance, size);
    }
  } else {
#ifdef PADDLE_WITH_HIP
    hipMemset(mean, 0, sizeof(AccT) * input_ddim[0] * groups);
    hipMemset(temp_variance, 0, sizeof(AccT) * input_ddim[0] * groups);
#else
    cudaMemset(mean, 0, sizeof(AccT) * input_ddim[0] * groups);
    cudaMemset(temp_variance, 0, sizeof(AccT) * input_ddim[0] * groups);
#endif

    phi::GroupNormForwardGetMeanAndVar<T, AccT>
        <<<grid, threads, 0, stream>>>(input,
                                       input_ddim[0],
                                       C,
                                       W,
                                       image_size,
                                       groups,
                                       group_size,
                                       mean,
                                       temp_variance);
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
template class GroupNormDirectCUDAFunctor<float, float>;
#if defined(PADDLE_WITH_CUDA) && !defined(PADDLE_WITH_HIP)
template class GroupNormDirectCUDAFunctor<half, float>;
#endif

template <typename T, typename Context>
void GroupNormGeneralCaseKernel(const Context& dev_ctx,
                                const DenseTensor& x,
                                const paddle::optional<DenseTensor>& scale,
                                const paddle::optional<DenseTensor>& bias,
                                float epsilon,
                                int groups,
                                const std::string& data_layout_str,
                                DenseTensor* y,
                                DenseTensor* mean,
                                DenseTensor* var) {
  using AccT = typename phi::dtype::MPTypeTrait<T>::Type;
  const DataLayout data_layout = common::StringToDataLayout(data_layout_str);
  const auto scale_ptr = scale.get_ptr();
  const auto bias_ptr = bias.get_ptr();
  const auto x_dims = x.dims();
  const int C = (data_layout == DataLayout::kNCHW ? x_dims[1]
                                                  : x_dims[x_dims.size() - 1]);
  const int group_size = C / groups;
  const int W = (data_layout == DataLayout::kNCHW ? x_dims[x_dims.size() - 1]
                                                  : x_dims[x_dims.size() - 2]);

  dev_ctx.template Alloc<T>(y);
  dev_ctx.template Alloc<AccT>(mean);
  dev_ctx.template Alloc<AccT>(var);
  // temp_var is used to calculate the mean^2
  DenseTensor temp_var;
  temp_var.Resize(var->dims());
  dev_ctx.template Alloc<AccT>(&temp_var);
  phi::funcs::SetConstant<GPUContext, T> set_zero;
  phi::funcs::SetConstant<GPUContext, AccT> set_zero_AccT;
  auto* x_data = x.data<T>();
  auto* y_data = y->data<T>();
  auto* mean_data = mean->data<AccT>();
  auto* var_data = var->data<AccT>();
  auto* temp_var_data = temp_var.data<AccT>();

  const T* scale_data = nullptr;
  if (scale_ptr) scale_data = scale_ptr->data<T>();
  const T* bias_data = nullptr;
  if (bias_ptr) bias_data = bias_ptr->data<T>();

  int imsize = 1;
  if (data_layout == DataLayout::kNCHW) {
    for (int i = 2; i < x_dims.size(); ++i) {
      imsize *= x_dims[i];
    }
  } else {
    for (int i = 1; i < x_dims.size() - 1; ++i) {
      imsize *= x_dims[i];
    }
  }

  int block_size = std::min(1024, imsize);

  dim3 grid(group_size, groups, x_dims[0]);
  dim3 threads(block_size, 1, 1);
  if (data_layout == DataLayout::kNCHW) {
    constexpr int vec_size = sizeof(float4) / sizeof(T);
    int size = group_size * imsize;
    const int max_num_threads = 1024;
    int max_block_size = std::min(size / vec_size, max_num_threads);
    int block_size_nchw = 1;
    while (block_size_nchw < max_block_size) {
      block_size_nchw *= 2;
    }
    block_size_nchw = std::max(block_size_nchw, kps::details::kWarpSize);
    dim3 grids(x_dims[0] * groups);
    dim3 blocks(block_size_nchw);
    if (size < vec_size * block_size_nchw) {
      ScalarGetMeanAndVarNCHW<T, AccT><<<grids, blocks, 0, dev_ctx.stream()>>>(
          x_data, mean_data, temp_var_data, size);
    } else {
      VectorizedGetMeanAndVarNCHW<T, AccT, vec_size>
          <<<grids, blocks, 0, dev_ctx.stream()>>>(
              x_data, mean_data, temp_var_data, size);
    }
  } else {
    set_zero_AccT(dev_ctx, mean, static_cast<AccT>(0));
    set_zero_AccT(dev_ctx, &temp_var, static_cast<AccT>(0));
    GroupNormForwardGetMeanAndVar<T, AccT>
        <<<grid, threads, 0, dev_ctx.stream()>>>(x_data,
                                                 x_dims[0],
                                                 C,
                                                 W,
                                                 imsize,
                                                 groups,
                                                 group_size,
                                                 mean_data,
                                                 temp_var_data);
  }
  int flags =
      (scale_data != nullptr) * kHasScale + (bias_data != nullptr) * kHasBias;
  UNROLL_ALL_CASES(flags,
                   GroupNormForward,
                   x_data,
                   mean_data,
                   temp_var_data,
                   scale_data,
                   bias_data,
                   x_dims[0],
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

template <typename T, typename Context>
void GroupNormKernel(const Context& dev_ctx,
                     const DenseTensor& x,
                     const paddle::optional<DenseTensor>& scale,
                     const paddle::optional<DenseTensor>& bias,
                     float epsilon,
                     int groups,
                     const std::string& data_layout_str,
                     DenseTensor* y,
                     DenseTensor* mean,
                     DenseTensor* var) {
  using std::is_same;
  if (is_same<T, phi::dtype::float16>::value && data_layout_str == "NHWC") {
    const paddle::optional<DenseTensor>& residual =
        paddle::optional<DenseTensor>(paddle::none);
    GroupNormNDHWCKernel<phi::dtype::float16, Context>(dev_ctx,
                                                       x,
                                                       residual,
                                                       scale,
                                                       bias,
                                                       epsilon,
                                                       groups,
                                                       data_layout_str,
                                                       "",
                                                       y,
                                                       new DenseTensor(),
                                                       mean,
                                                       var);
    return;
  }

#ifdef PADDLE_CUDA_BF16
  if (is_same<T, phi::dtype::bfloat16>::value && data_layout_str == "NHWC") {
    const paddle::optional<DenseTensor>& residual =
        paddle::optional<DenseTensor>(paddle::none);
    GroupNormNDHWCKernel<phi::dtype::bfloat16, Context>(dev_ctx,
                                                        x,
                                                        residual,
                                                        scale,
                                                        bias,
                                                        epsilon,
                                                        groups,
                                                        data_layout_str,
                                                        "",
                                                        y,
                                                        new DenseTensor(),
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
                   phi::dtype::bfloat16,
                   phi::dtype::float16) {
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
                   phi::dtype::bfloat16,
                   phi::dtype::float16) {
  kernel->OutputAt(2).SetDataType(phi::DataType::FLOAT32);
  kernel->OutputAt(3).SetDataType(phi::DataType::FLOAT32);
}
