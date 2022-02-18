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

#include <string>

#ifdef PADDLE_WITH_CUDA
#include <cuda.h>
#include <curand_kernel.h>
#include "paddle/fluid/platform/dynload/curand.h"
#endif
#ifdef PADDLE_WITH_HIP
#include <hip/hip_runtime.h>
#include <hiprand_kernel.h>
#include "paddle/fluid/platform/dynload/hiprand.h"
#endif

#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/generator.h"
#include "paddle/fluid/framework/tensor_util.h"
#include "paddle/fluid/operators/dropout_impl_util.h"
#include "paddle/fluid/platform/aligned_vector.h"
#include "paddle/fluid/platform/device/gpu/gpu_info.h"

namespace paddle {
namespace operators {

template <typename T>
int get_vectorized_size(const Tensor& x) {
  const T* x_data = x.data<T>();
  uint64_t address = reinterpret_cast<uint64_t>(x_data);
  constexpr int vec4 = std::alignment_of<platform::AlignedVector<T, 4>>::value;
  constexpr int vec2 = std::alignment_of<platform::AlignedVector<T, 2>>::value;
  int vec_size = 1;
  if (address % vec4 == 0) {
    vec_size = 4;
  } else if (address % vec2 == 0) {
    vec_size = 2;
  }

  int64_t numel = x.numel();
  while ((numel % vec_size) && vec_size >= 1) {
    vec_size /= 2;
  }

  PADDLE_ENFORCE_GE(
      vec_size, 1,
      platform::errors::InvalidArgument(
          " Tensor vectorized size must greater than or equal to 1"));

  PADDLE_ENFORCE_LE(
      vec_size, 4, platform::errors::InvalidArgument(
                       " Tensor vectorized size must less than or equal to 4"));
  return vec_size;
}

template <typename T, typename MaskType>
__global__ void dropout_kernel(const size_t n, uint64_t seed,
                               const float dropout_prob, const T* src,
                               MaskType* mask, T* dst, bool is_upscale_in_train,
                               uint64_t increment) {
  using MT = typename details::MPTypeTrait<T>::Type;
  int thread_idx = blockDim.x * blockIdx.x + threadIdx.x;
#ifdef PADDLE_WITH_HIP
  hiprandStatePhilox4_32_10_t state;
  hiprand_init(seed, thread_idx, increment, &state);
#else
  curandStatePhilox4_32_10_t state;
  curand_init(seed, thread_idx, increment, &state);
#endif

  float p = 1.0 - dropout_prob;
  MT factor = static_cast<MT>(1.0 / p);

  size_t loop_times = 1;
  for (; thread_idx < n; thread_idx += blockDim.x * gridDim.x * 4) {
#ifdef PADDLE_WITH_HIP
    float4 rand = hiprand_uniform4(&state);
#else
    float4 rand = curand_uniform4(&state);
#endif
    for (size_t ii = 0; ii < 4; ii++) {
      size_t idx = thread_idx + ii * blockDim.x * gridDim.x;
      if (idx < n) {
        if ((&rand.x)[ii] < dropout_prob) {
          mask[idx] = 1;
          dst[idx] = is_upscale_in_train
                         ? static_cast<T>(static_cast<MT>(src[idx]) * factor)
                         : src[idx];
        } else {
          mask[idx] = 0;
          dst[idx] = 0;
        }
      }
    }
    if (loop_times > 1) {
      __syncthreads();
    }
    ++loop_times;
  }
}

template <typename T, typename MaskType, size_t VecSize>
__global__ void dropout_kernel_vec(const size_t n, uint64_t seed,
                                   const float dropout_prob, const T* src,
                                   MaskType* mask, T* dst,
                                   bool is_upscale_in_train,
                                   uint64_t increment) {
  using MT = typename details::MPTypeTrait<T>::Type;
  using LoadT = platform::AlignedVector<T, VecSize>;
  using MaskLoadT = platform::AlignedVector<MaskType, VecSize>;

#ifdef PADDLE_WITH_HIP
  int64_t idx = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
  hiprandStatePhilox4_32_10_t state;
  hiprand_init(seed, idx, increment, &state);
#else
  int64_t idx = blockDim.x * blockIdx.x + threadIdx.x;
  curandStatePhilox4_32_10_t state;
  curand_init(seed, idx, increment, &state);
#endif

  float p = 1.0 - dropout_prob;
  MT factor = static_cast<MT>(1.0 / p);

  size_t loop_times = 1;

  for (int i = idx * VecSize; i < n; i += blockDim.x * gridDim.x * VecSize) {
    LoadT src_val;
    platform::Load<T, VecSize>(&src[i], &src_val);

#ifdef PADDLE_WITH_HIP
    float4 rand = hiprand_uniform4(&state);
#else
    float4 rand = curand_uniform4(&state);
#endif

    LoadT dst_val;
    MaskLoadT mask_val;

#pragma unroll
    for (int j = 0; j < VecSize; j++) {
      if ((&rand.x)[j] < p) {
        mask_val[j] = 1;
        dst_val[j] = is_upscale_in_train
                         ? static_cast<T>(static_cast<MT>(src_val[j]) * factor)
                         : src_val[j];
      } else {
        mask_val[j] = 0;
        dst_val[j] = 0;
      }
    }

    platform::Store<T, VecSize>(dst_val, &dst[i]);
    platform::Store<MaskType, VecSize>(mask_val, &mask[i]);
    if (loop_times > 1) {
      __syncthreads();
    }
    ++loop_times;
  }
}

template <typename T>
void DropoutFwGPUKernelDriverV2(const platform::CUDADeviceContext& dev_ctx,
                                bool is_test,
                                const std::string dropout_implementation,
                                float dropout_prob, bool upscale_in_train,
                                bool is_fix_seed, int seed_val, const Tensor& x,
                                const Tensor* seed, Tensor* mask, Tensor* y) {
  auto& place = *dev_ctx.eigen_device();

  if (!is_test) {
    int64_t x_numel = x.numel();
    auto stream = dev_ctx.stream();
    auto* mask_data = mask->data<uint8_t>();
    size_t size = framework::product(mask->dims());

    auto* x_data = x.data<T>();
    auto* y_data = y->data<T>();
    if (dropout_prob == 1.0f) {
#ifdef PADDLE_WITH_HIP
      PADDLE_ENFORCE_GPU_SUCCESS(
          hipMemsetAsync(y_data, 0, x_numel * sizeof(T), stream));
      PADDLE_ENFORCE_GPU_SUCCESS(
          hipMemsetAsync(mask_data, 0, x_numel * sizeof(*mask_data), stream));
#else
      PADDLE_ENFORCE_GPU_SUCCESS(
          cudaMemsetAsync(y_data, 0, x_numel * sizeof(T), stream));
      PADDLE_ENFORCE_GPU_SUCCESS(
          cudaMemsetAsync(mask_data, 0, x_numel * sizeof(*mask_data), stream));
#endif
      return;
    }
    int vec_size = get_vectorized_size<T>(x);
    int64_t device_id = dev_ctx.GetPlace().GetDeviceId();

    auto gpu_config = GetGpuLaunchConfig1D(dev_ctx, x_numel, vec_size);

    size_t grid_size = gpu_config.GetGridSize();
    size_t block_size = gpu_config.GetBlockSize();

    const auto& prop = platform::GetDeviceProperties(device_id);
    size_t max_grid_size = prop.maxThreadsPerMultiProcessor *
                           prop.multiProcessorCount / block_size;

    if (grid_size > max_grid_size) {
      grid_size = max_grid_size;
    }

    size_t total_thread = grid_size * block_size;
    const int increment = ((x_numel - 1) / (4 * total_thread) + 1) * 4;

    uint64_t seed_data;
    uint64_t offset;
    GetSeedDataAndIncrement(dev_ctx, seed, is_fix_seed, seed_val, increment,
                            &seed_data, &offset);

    VLOG(0) << "grid_size: " << grid_size << "\n";
    VLOG(0) << "block_size: " << block_size << "\n";
// VLOG(0) << "max_grid_size: " << max_grid_size << "\n";
// VLOG(0) << "prop.multiProcessorCount: " << prop.multiProcessorCount << "\n";
// VLOG(0) << "prop.maxThreadsPerMultiProcessor " <<
// prop.maxThreadsPerMultiProcessor << "\n";
// VLOG(0) << "sm_count: " << sm_count << "\n";
// VLOG(0) << "increment: " << increment << "\n";
// VLOG(0) << "offset: " << offset << "\n";

#ifdef __HIPCC__
    switch (vec_size) {
      case 4:
        hipLaunchKernelGGL(HIP_KERNEL_NAME(dropout_kernel_vec<T, uint8_t, 4>),
                           grid_size, block_size, 0, stream, x_numel, seed_data,
                           dropout_prob, x_data, mask_data, y_data,
                           upscale_in_train, offset);
        break;
      case 2:
        hipLaunchKernelGGL(HIP_KERNEL_NAME(dropout_kernel_vec<T, uint8_t, 2>),
                           grid_size, block_size, 0, stream, x_numel, seed_data,
                           dropout_prob, x_data, mask_data, y_data,
                           upscale_in_train, offset);
        break;
      case 1:
        hipLaunchKernelGGL(HIP_KERNEL_NAME(dropout_kernel<T, uint8_t>),
                           grid_size, block_size, 0, stream, x_numel, seed_data,
                           dropout_prob, x_data, mask_data, y_data,
                           upscale_in_train, offset);
        break;
    }
#else
    switch (vec_size) {
      case 4:
        dropout_kernel_vec<T, uint8_t, 4><<<grid_size, block_size, 0, stream>>>(
            x_numel, seed_data, dropout_prob, x_data, mask_data, y_data,
            upscale_in_train, offset);
        break;
      case 2:
        dropout_kernel_vec<T, uint8_t, 2><<<grid_size, block_size, 0, stream>>>(
            x_numel, seed_data, dropout_prob, x_data, mask_data, y_data,
            upscale_in_train, offset);
        break;
      case 1:
        dropout_kernel<T, uint8_t><<<grid_size, block_size, 0, stream>>>(
            x_numel, seed_data, dropout_prob, x_data, mask_data, y_data,
            upscale_in_train, offset);
        break;
    }
#endif
  } else {
    auto X = EigenMatrix<T>::Reshape(x, 1);
    auto Y = EigenMatrix<T>::Reshape(*y, 1);
    if (upscale_in_train) {
      Y.device(place) = X;
    } else {
      Y.device(place) = X * static_cast<T>(1.0f - dropout_prob);
    }
  }
}

}  // namespace operators
}  // namespace paddle
