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

#pragma once

#include "glog/logging.h"

#include "paddle/common/flags.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/common/amp_type_traits.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/kernels/funcs/aligned_vector.h"

COMMON_DECLARE_bool(use_fast_math);

namespace phi {

#if defined(__NVCC__) || defined(__HIPCC__)
template <bool FastMode>
static __device__ __forceinline__ float FP32FastTanh(float x) {
#if __CUDA_ARCH__ >= 750 && CUDA_VERSION >= 11000
  if (FastMode) {
    float y;
    asm("tanh.approx.f32 %0,%1; \n\t" : "=f"(y) : "f"(x));
    return y;
  }
#endif
  return tanhf(x);
}

template <typename T, bool FastMode>
static __device__ __forceinline__ T GeluFwd(T x) {
  constexpr float kBeta = 0.7978845608f;  // M_SQRT2 * M_2_SQRTPI * 0.5
  constexpr float kKappa = 0.044715f;
  const float cast_x = static_cast<float>(x);
  auto x_cube = cast_x * cast_x * cast_x;
  auto inner = kBeta * (cast_x + kKappa * x_cube);
  auto tanh_out = FP32FastTanh<FastMode>(inner);
  return static_cast<T>(0.5f * cast_x * (1.0f + tanh_out));
}

#ifdef PADDLE_WITH_HIP
template <bool FastMode>
static __device__ __forceinline__ __half GeluFwdHalf(__half x) {
  constexpr float kBeta = 0.7978845608f;
  constexpr float kKappa = 0.044715f;
  const float cast_x = __half2float(x);
  auto x_cube = cast_x * cast_x * cast_x;
  auto inner = kBeta * (cast_x + kKappa * x_cube);
  auto tanh_out = FP32FastTanh<FastMode>(inner);
  return __float2half(0.5f * cast_x * (1.0f + tanh_out));
}
#endif

template <bool FastMode>
static __device__ __forceinline__ float FP32GeluBwd(float x, float y_g) {
  constexpr float kBeta = 0.7978845608f;  // M_SQRT2 * M_2_SQRTPI * 0.5
  constexpr float kKappa = 0.044715f;
  auto x_sq = x * x;
  auto x_cube = x_sq * x;
  auto inner = kBeta * (x + kKappa * x_cube);
  auto tanh_inner = FP32FastTanh<FastMode>(inner);

  auto left = 0.5f * x;
  auto right = 1.0f + tanh_inner;

  auto left_derivative = 0.5f * right;
  auto tanh_derivative = 1.0f - tanh_inner * tanh_inner;
  auto inner_derivative = kBeta * (1.0f + 3.0f * kKappa * x_sq);
  auto right_derivative = left * tanh_derivative * inner_derivative;

  return y_g * (left_derivative + right_derivative);
}

template <int VecSize, bool FastMode>
static __global__ void FP16FastGeluFwdCUDAKernel(const __half* x,
                                                 __half* y,
                                                 size_t n) {
  size_t offset =
      static_cast<size_t>(threadIdx.x + blockIdx.x * blockDim.x) * VecSize;
  size_t stride = static_cast<size_t>(blockDim.x * gridDim.x) * VecSize;
  for (; offset < n; offset += stride) {
    using ArrT = AlignedVector<__half, VecSize>;
    ArrT in_arr = *reinterpret_cast<const ArrT*>(x + offset);
#pragma unroll
    for (int i = 0; i < VecSize; ++i) {
#ifdef PADDLE_WITH_HIP
      in_arr[i] = GeluFwdHalf<FastMode>(in_arr[i]);
#else
      in_arr[i] = GeluFwd<half, FastMode>(in_arr[i]);
#endif
    }
    *reinterpret_cast<ArrT*>(y + offset) = in_arr;
  }
}

template <int VecSize, bool FastMode>
static __global__ void FP16FastGeluBwdCUDAKernel(const __half* x,
                                                 const __half* y_g,
                                                 __half* x_g,
                                                 size_t n) {
  size_t offset =
      static_cast<size_t>(threadIdx.x + blockIdx.x * blockDim.x) * VecSize;
  size_t stride = static_cast<size_t>(blockDim.x * gridDim.x) * VecSize;
  for (; offset < n; offset += stride) {
    using ArrT = AlignedVector<__half, VecSize>;
    ArrT x_in_arr = *reinterpret_cast<const ArrT*>(x + offset);
    ArrT y_g_in_arr = *reinterpret_cast<const ArrT*>(y_g + offset);
#pragma unroll
    for (int i = 0; i < VecSize; ++i) {
      __half2 tmp_fp16_2;
#if defined(PADDLE_WITH_HIP) && HIP_VERSION < 60100000
      tmp_fp16_2.x = *reinterpret_cast<uint16_t*>(&x_in_arr[i]);
      tmp_fp16_2.y = *reinterpret_cast<uint16_t*>(&y_g_in_arr[i]);
#else
      tmp_fp16_2.x = x_in_arr[i];
      tmp_fp16_2.y = y_g_in_arr[i];
#endif
      float2 tmp_fp32_2 = __half22float2(tmp_fp16_2);
      x_in_arr[i] =
          __float2half(FP32GeluBwd<FastMode>(tmp_fp32_2.x, tmp_fp32_2.y));
    }
    *reinterpret_cast<ArrT*>(x_g + offset) = x_in_arr;
  }
}

static bool TryLaunchFP16FastGeluFwdVectorizeCUDAKernel(
    const GPUContext& dev_ctx, const __half* x, __half* y, size_t n) {
  auto is_aligned = [](const void* p, size_t alignment) {
    return reinterpret_cast<uintptr_t>(p) % alignment == 0;
  };

#define PD_LAUNCH_FP16_FAST_GELU_FWD_KERNEL(__vec_size, __use_fast_math)      \
  do {                                                                        \
    constexpr auto kAlignment = alignof(AlignedVector<__half, __vec_size>);   \
    if (n % __vec_size == 0 && is_aligned(x, kAlignment) &&                   \
        is_aligned(y, kAlignment)) {                                          \
      size_t thread = std::min<size_t>(512, dev_ctx.GetMaxThreadsPerBlock()); \
      size_t block = (n / __vec_size + thread - 1) / thread;                  \
      block = std::min<size_t>(block, dev_ctx.GetCUDAMaxGridDimSize()[0]);    \
      VLOG(10) << "Use FP16 fast gelu fwd kernel, block = " << block          \
               << " , thread = " << thread;                                   \
      FP16FastGeluFwdCUDAKernel<__vec_size, __use_fast_math>                  \
          <<<block, thread, 0, dev_ctx.stream()>>>(x, y, n);                  \
      return true;                                                            \
    }                                                                         \
  } while (0)

  if (FLAGS_use_fast_math) {
    PD_LAUNCH_FP16_FAST_GELU_FWD_KERNEL(8, true);
  } else {
    PD_LAUNCH_FP16_FAST_GELU_FWD_KERNEL(8, false);
  }

#undef PD_LAUNCH_FP16_FAST_GELU_FWD_KERNEL
  return false;
}

static bool TryLaunchFP16FastGeluBwdVectorizeCUDAKernel(
    const GPUContext& dev_ctx,
    const __half* x,
    const __half* y_g,
    __half* x_g,
    size_t n) {
  auto is_aligned = [](const void* p, size_t alignment) {
    return reinterpret_cast<uintptr_t>(p) % alignment == 0;
  };

#define PD_LAUNCH_FP16_FAST_GELU_BWD_KERNEL(__vec_size, __use_fast_math)      \
  do {                                                                        \
    constexpr auto kAlignment = alignof(AlignedVector<__half, __vec_size>);   \
    if (n % __vec_size == 0 && is_aligned(x, kAlignment) &&                   \
        is_aligned(x, kAlignment) && is_aligned(y_g, kAlignment) &&           \
        is_aligned(x_g, kAlignment)) {                                        \
      size_t thread = std::min<size_t>(512, dev_ctx.GetMaxThreadsPerBlock()); \
      size_t block = (n / __vec_size + thread - 1) / thread;                  \
      block = std::min<size_t>(block, dev_ctx.GetCUDAMaxGridDimSize()[0]);    \
      VLOG(10) << "Use FP16 fast gelu bwd kernel, block = " << block          \
               << " , thread = " << thread;                                   \
      FP16FastGeluBwdCUDAKernel<__vec_size, __use_fast_math>                  \
          <<<block, thread, 0, dev_ctx.stream()>>>(x, y_g, x_g, n);           \
      return true;                                                            \
    }                                                                         \
  } while (0)

  if (FLAGS_use_fast_math) {
    PD_LAUNCH_FP16_FAST_GELU_BWD_KERNEL(8, true);
  } else {
    PD_LAUNCH_FP16_FAST_GELU_BWD_KERNEL(8, false);
  }

#undef PD_LAUNCH_FP16_FAST_GELU_BWD_KERNEL
  return false;
}
#endif

}  // namespace phi
