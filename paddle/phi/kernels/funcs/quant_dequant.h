/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

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

#include <vector>
#include "paddle/common/hostdevice.h"
#include "paddle/phi/backends/gpu/gpu_launch_config.h"
#include "paddle/phi/common/float16.h"
#include "paddle/phi/common/transform.h"
#include "paddle/phi/kernels/funcs/aligned_vector.h"
#include "paddle/phi/kernels/funcs/blas/blas.h"

namespace phi {

using backends::gpu::GpuLaunchConfig;

constexpr int DequantKernelVecSize = 4;

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
inline HOSTDEVICE T roundWithTiesAwayFromZero(T x) {
  return static_cast<T>(x > 0 ? ceil(x) : floor(x));
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

template <typename T>
__forceinline__ __device__ int8_t
quant_helper_ties_to_even_or_away_from_zero(const T input,
                                            const float scale,
                                            const int round_type,
                                            const float max_bound,
                                            const float min_bound) {
  float quant_value = max_bound * scale * static_cast<float>(input);

  if (round_type == 0) {
    quant_value = static_cast<float>(roundWithTiesToEven(quant_value));
  } else {
    quant_value = static_cast<float>(roundWithTiesAwayFromZero(quant_value));
  }
  quant_value = quant_value > max_bound ? max_bound : quant_value;
  quant_value = quant_value < min_bound ? min_bound : quant_value;
  return static_cast<int8_t>(quant_value);
}

template <typename T>
__global__ void QuantKernel(const T* input,
                            char4* output,
                            const float scale,
                            const int m,
                            const int n,
                            const int round_type,
                            const float max_bound,
                            const float min_bound) {
  int n_id = (blockIdx.x * blockDim.x + threadIdx.x) << 2;
  int m_id = blockIdx.y * blockDim.y + threadIdx.y;

  bool check = ((m_id < m) && (n_id < n));
  if (check) {
    char4 tmp;
    tmp.x = quant_helper(
        input[m_id * n + n_id], scale, round_type, max_bound, min_bound);
    tmp.y = quant_helper(
        input[m_id * n + n_id + 1], scale, round_type, max_bound, min_bound);
    tmp.z = quant_helper(
        input[m_id * n + n_id + 2], scale, round_type, max_bound, min_bound);
    tmp.w = quant_helper(
        input[m_id * n + n_id + 3], scale, round_type, max_bound, min_bound);
    output[(m_id * n + n_id) >> 2] = tmp;
  }
}

template <typename T>
__global__ void QuantKernelWithVecSize(const T* input,
                                       char4* output,
                                       const float scale,
                                       const int m,
                                       const int n,
                                       const int round_type,
                                       const float max_bound,
                                       const float min_bound) {
  int n_id = (blockIdx.x * blockDim.x + threadIdx.x) << 2;
  int m_id = blockIdx.y * blockDim.y + threadIdx.y;

  bool check = ((m_id < m) && (n_id < n));
  if (check) {
    char4 tmp;
    tmp.x = quant_helper_ties_to_even_or_away_from_zero(
        input[m_id * n + n_id], scale, round_type, max_bound, min_bound);
    tmp.y = quant_helper_ties_to_even_or_away_from_zero(
        input[m_id * n + n_id + 1], scale, round_type, max_bound, min_bound);
    tmp.z = quant_helper_ties_to_even_or_away_from_zero(
        input[m_id * n + n_id + 2], scale, round_type, max_bound, min_bound);
    tmp.w = quant_helper_ties_to_even_or_away_from_zero(
        input[m_id * n + n_id + 3], scale, round_type, max_bound, min_bound);
    output[(m_id * n + n_id) >> 2] = tmp;
  }
}

template <typename T>
__global__ void QuantKernelWithVecSize(const T* input,
                                       char3* output,
                                       const float scale,
                                       const int m,
                                       const int n,
                                       const int round_type,
                                       const float max_bound,
                                       const float min_bound) {
  int n_id = (blockIdx.x * blockDim.x + threadIdx.x) * 3;
  int m_id = blockIdx.y * blockDim.y + threadIdx.y;

  bool check = ((m_id < m) && (n_id < n));
  if (check) {
    char3 tmp;
    tmp.x = quant_helper_ties_to_even_or_away_from_zero(
        input[m_id * n + n_id], scale, round_type, max_bound, min_bound);
    tmp.y = quant_helper_ties_to_even_or_away_from_zero(
        input[m_id * n + n_id + 1], scale, round_type, max_bound, min_bound);
    tmp.z = quant_helper_ties_to_even_or_away_from_zero(
        input[m_id * n + n_id + 2], scale, round_type, max_bound, min_bound);
    output[(m_id * n + n_id) / 3] = tmp;
  }
}

template <typename T>
__global__ void QuantKernelWithVecSize(const T* input,
                                       char2* output,
                                       const float scale,
                                       const int m,
                                       const int n,
                                       const int round_type,
                                       const float max_bound,
                                       const float min_bound) {
  int n_id = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
  int m_id = blockIdx.y * blockDim.y + threadIdx.y;

  bool check = ((m_id < m) && (n_id < n));
  if (check) {
    char2 tmp;
    tmp.x = quant_helper_ties_to_even_or_away_from_zero(
        input[m_id * n + n_id], scale, round_type, max_bound, min_bound);
    tmp.y = quant_helper_ties_to_even_or_away_from_zero(
        input[m_id * n + n_id + 1], scale, round_type, max_bound, min_bound);
    output[(m_id * n + n_id) >> 1] = tmp;
  }
}

template <typename T>
__global__ void QuantKernelWithVecSize(const T* input,
                                       char* output,
                                       const float scale,
                                       const int m,
                                       const int n,
                                       const int round_type,
                                       const float max_bound,
                                       const float min_bound) {
  int n_id = (blockIdx.x * blockDim.x + threadIdx.x);
  int m_id = blockIdx.y * blockDim.y + threadIdx.y;

  bool check = ((m_id < m) && (n_id < n));
  if (check) {
    char tmp;
    tmp = quant_helper_ties_to_even_or_away_from_zero(
        input[m_id * n + n_id], scale, round_type, max_bound, min_bound);
    output[m_id * n + n_id] = tmp;
  }
}

template <typename T>
void LaunchQuantKernel(const T* input,
                       int8_t* output,
                       const float scale,
                       const int m,
                       const int n,
                       const int round_type,
                       const float max_bound,
                       const float min_bound,
                       gpuStream_t stream) {
  // TODO(minghaoBD): optimize the kennel launch times when m==1 or n==1
  dim3 grid(((n >> 2) + 31) / 32, (m + 31) / 32);
  dim3 block(32, 32);

  QuantKernel<<<grid, block, 0, stream>>>(input,
                                          (char4*)output,  // NOLINT
                                          scale,
                                          m,
                                          n,
                                          round_type,
                                          max_bound,
                                          min_bound);
}

template <typename T>
void LaunchQuantKernelWithVecSize(const T* input,
                                  int8_t* output,
                                  const float scale,
                                  const int m,
                                  const int n,
                                  const int round_type,
                                  const float max_bound,
                                  const float min_bound,
                                  gpuStream_t stream) {
  int vec_size = 1;
  if (n % 4 == 0) {
    vec_size = 4;
  } else if (n % 3 == 0) {
    vec_size = 3;
  } else if (n % 2 == 0) {
    vec_size = 2;
  }

  dim3 grid(((n / vec_size) + 31) / 32, (m + 31) / 32);
  dim3 block(32, 32);

  switch (vec_size) {
    case 4:
      QuantKernelWithVecSize<<<grid, block, 0, stream>>>(
          input,
          reinterpret_cast<char4*>(output),
          scale,
          m,
          n,
          round_type,
          max_bound,
          min_bound);
      break;
    case 3:
      QuantKernelWithVecSize<<<grid, block, 0, stream>>>(
          input,
          reinterpret_cast<char3*>(output),
          scale,
          m,
          n,
          round_type,
          max_bound,
          min_bound);
      break;
    case 2:
      QuantKernelWithVecSize<<<grid, block, 0, stream>>>(
          input,
          reinterpret_cast<char2*>(output),
          scale,
          m,
          n,
          round_type,
          max_bound,
          min_bound);
      break;
    case 1:
      QuantKernelWithVecSize<<<grid, block, 0, stream>>>(
          input,
          reinterpret_cast<char*>(output),
          scale,
          m,
          n,
          round_type,
          max_bound,
          min_bound);
      break;
    default:
      return;
  }
}

template <typename T, int VecSize>
__global__ void DequantKernel(T* output,
                              const int32_t* input,
                              const int m,  // batch size
                              const int n,  // hidden
                              const float quant_in_scale,
                              const float* dequant_out_scale_data) {
  int numel = m * n;
  int stride = blockDim.x * gridDim.x * VecSize;
  int idx = (blockIdx.x * blockDim.x + threadIdx.x) * VecSize;
  int col_id = idx % n;

  phi::AlignedVector<int32_t, VecSize> in_vec;
  phi::AlignedVector<float, VecSize> out_scale_vec;
  phi::AlignedVector<T, VecSize> out_vec;

  for (; idx < numel; idx += stride) {
    phi::Load<int32_t, VecSize>(input + idx, &in_vec);
    phi::Load<float, VecSize>(dequant_out_scale_data + col_id, &out_scale_vec);

#pragma unroll
    for (int i = 0; i < VecSize; ++i) {
      out_vec[i] =
          static_cast<T>(static_cast<float>(in_vec[i]) * out_scale_vec[i]);
    }

    phi::Store<T, VecSize>(out_vec, output + idx);
  }
}

template <typename T>
void LaunchDequantKernel(const int32_t* input,
                         T* output,
                         const int m,  // m
                         const int n,  // n
                         gpuStream_t stream,
                         GpuLaunchConfig* gpu_config,
                         const float quant_in_scale,
                         const float* dequant_out_scale_data) {
  DequantKernel<T, DequantKernelVecSize>
      <<<gpu_config->block_per_grid, gpu_config->thread_per_block, 0, stream>>>(
          output, input, m, n, quant_in_scale, dequant_out_scale_data);
}

template <typename T, int VecSize>
__global__ void DequantKernelWithScaleOfInputAndWeight(
    T* output,
    const int32_t* input,
    const int m,  // batch size
    const int n,  // hidden
    const float quant_in_scale,
    const float* quant_weight_scale,
    float quant_max_bound) {
  int numel = m * n;
  int stride = blockDim.x * gridDim.x * VecSize;
  int idx = (blockIdx.x * blockDim.x + threadIdx.x) * VecSize;
  int col_id = idx % n;

  phi::AlignedVector<int32_t, VecSize> in_vec;
  phi::AlignedVector<float, VecSize> out_scale_vec;
  phi::AlignedVector<T, VecSize> out_vec;

  for (; idx < numel; idx += stride) {
    phi::Load<int32_t, VecSize>(input + idx, &in_vec);
    phi::Load<float, VecSize>(quant_weight_scale + col_id, &out_scale_vec);

#pragma unroll
    for (int i = 0; i < VecSize; ++i) {
      out_vec[i] = static_cast<T>(static_cast<float>(in_vec[i]) /
                                  (quant_max_bound * quant_max_bound *
                                   quant_in_scale * out_scale_vec[i]));
    }

    phi::Store<T, VecSize>(out_vec, output + idx);
  }
}

template <typename T>
void LaunchDequantKernelWithScaleOfInputAndWeight(
    const int32_t* input,
    T* output,
    const int m,  // m
    const int n,  // n
    gpuStream_t stream,
    GpuLaunchConfig* gpu_config,
    const float quant_in_scale,
    const float* quant_weight_scale,
    float quant_max_bound) {
  if (n % DequantKernelVecSize != 0) {
    DequantKernelWithScaleOfInputAndWeight<T, 1><<<gpu_config->block_per_grid,
                                                   gpu_config->thread_per_block,
                                                   0,
                                                   stream>>>(output,
                                                             input,
                                                             m,
                                                             n,
                                                             quant_in_scale,
                                                             quant_weight_scale,
                                                             quant_max_bound);
    return;
  }
  DequantKernelWithScaleOfInputAndWeight<T, DequantKernelVecSize>
      <<<gpu_config->block_per_grid, gpu_config->thread_per_block, 0, stream>>>(
          output,
          input,
          m,
          n,
          quant_in_scale,
          quant_weight_scale,
          quant_max_bound);
}

}  // namespace phi
