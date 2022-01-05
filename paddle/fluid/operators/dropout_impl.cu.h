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
#include "paddle/fluid/operators/dropout_op.h"
#include "paddle/fluid/platform/aligned_vector.h"
#include "paddle/fluid/platform/device/gpu/gpu_launch_config.h"
#include "paddle/pten/kernels/hybird/cuda/elementwise/elementwise_no_broadcast.cu.h"

namespace paddle {
namespace operators {

template <typename T, typename MaskType>
__global__ void RandomGenerator(const size_t n, uint64_t seed,
                                const float dropout_prob, const T* src,
                                MaskType* mask, T* dst,
                                bool is_upscale_in_train, uint64_t increment) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
#ifdef PADDLE_WITH_HIP
  hiprandStatePhilox4_32_10_t state;
  hiprand_init(seed, idx, increment, &state);
#else
  curandStatePhilox4_32_10_t state;
  curand_init(seed, idx, increment, &state);
#endif

  MaskType mask_val;
  T dst_val;
  T factor = static_cast<T>(1.0f / (1.0f - dropout_prob));
  for (; idx < n; idx += blockDim.x * gridDim.x) {
    T src_val = src[idx];
#ifdef PADDLE_WITH_HIP
    if (hiprand_uniform(&state) < dropout_prob) {
#else
    if (curand_uniform(&state) < dropout_prob) {
#endif
      mask_val = 0;
      dst_val = 0;
    } else {
      mask_val = 1;
      dst_val = is_upscale_in_train ? src_val * factor : src_val;
    }
    mask[idx] = mask_val;
    dst[idx] = dst_val;
  }
}

template <typename T, typename MaskType, int VecSize>
__global__ void VectorizedRandomGenerator(const size_t n, uint64_t seed,
                                          const float dropout_prob,
                                          const T* src, MaskType* mask, T* dst,
                                          bool is_upscale_in_train,
                                          uint64_t increment) {
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

  T factor = static_cast<T>(1.0f / (1.0f - dropout_prob));
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
      if ((&rand.x)[j] < dropout_prob) {
        dst_val[j] = 0;
        mask_val[j] = 0;
      } else {
        dst_val[j] = is_upscale_in_train ? src_val[j] * factor : src_val[j];
        mask_val[j] = 1;
      }
    }

    platform::Store<T, VecSize>(dst_val, &dst[i]);
    platform::Store<MaskType, VecSize>(mask_val, &mask[i]);
  }
}

template <typename T, typename MaskType, int VecSize>
__global__ void DropoutGradCUDAKernel(const T* dout, const MaskType* mask,
                                      const T factor, const int64_t size,
                                      T* dx) {
  using LoadT = platform::AlignedVector<T, VecSize>;
  using MaskLoadT = platform::AlignedVector<MaskType, VecSize>;

  int64_t idx = blockDim.x * blockIdx.x + threadIdx.x;
  for (int i = idx * VecSize; i < size; i += blockDim.x * gridDim.x * VecSize) {
    LoadT dout_val;
    platform::Load<T, VecSize>(&dout[i], &dout_val);

    MaskLoadT mask_val;
    platform::Load<MaskType, VecSize>(&mask[i], &mask_val);

    LoadT dx_val;

#pragma unroll
    for (int j = 0; j < VecSize; j++) {
      dx_val[j] = dout_val[j] * static_cast<T>(mask_val[j]) * factor;
    }

    platform::Store<T, VecSize>(dx_val, &dx[i]);
  }
}

template <typename T>
void DropoutFwGPUKernelDriver(const platform::CUDADeviceContext& dev_ctx,
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

    // increment is used to set the args(offset) of curand_init, which defines
    // offset in subsequence.
    // The detail:
    // https://docs.nvidia.com/cuda/curand/device-api-overview.html
    // Increment should be at least the number of curand() random numbers used
    // in each thread to avoid the random number generated this time being the
    // same as the previous calls.
    uint64_t seed_data;
    uint64_t increment;
    // VectorizedRandomGenerator use curand_uniform4, so we only support
    // vec_size is 4;
    int vec_size = (platform::GetVectorizedSize<T>(x_data) == 4) ? 4 : 1;
    int block_size = pten::GetThreadsConfig(dev_ctx, x_numel, vec_size);
    int grid_size =
        ((x_numel + vec_size - 1) / vec_size + block_size - 1) / block_size;

    auto offset =
        ((x_numel - 1) / (grid_size * block_size * vec_size) + 1) * vec_size;

    GetSeedDataAndIncrement(dev_ctx, seed, is_fix_seed, seed_val, offset,
                            &seed_data, &increment);

#ifdef __HIPCC__
    if (vec_size == 4 && size % 4 == 0) {
      hipLaunchKernelGGL(
          HIP_KERNEL_NAME(VectorizedRandomGenerator<T, uint8_t, 4>), grid_size,
          block_size, 0, stream, size, seed_data, dropout_prob, x_data,
          mask_data, y_data, upscale_in_train, increment);
    } else {
      hipLaunchKernelGGL(HIP_KERNEL_NAME(RandomGenerator<T, uint8_t>),
                         grid_size, block_size, 0, stream, size, seed_data,
                         dropout_prob, x_data, mask_data, y_data,
                         upscale_in_train, increment);
    }
#else
    if (vec_size == 4 && size % 4 == 0) {
      VectorizedRandomGenerator<T, uint8_t,
                                4><<<grid_size, block_size, 0, stream>>>(
          size, seed_data, dropout_prob, x_data, mask_data, y_data,
          upscale_in_train, increment);
    } else {
      RandomGenerator<T, uint8_t><<<grid_size, block_size, 0, stream>>>(
          size, seed_data, dropout_prob, x_data, mask_data, y_data,
          upscale_in_train, increment);
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

template <typename T>
void DropoutGradGPUKernelDriver(const platform::CUDADeviceContext& dev_ctx,
                                const std::string dropout_implementation,
                                float dropout_prob, const Tensor& grad_y,
                                const Tensor& mask, int64_t size,
                                Tensor* grad_x, bool is_test = false) {
  auto dX = EigenVector<T>::Flatten(*grad_x);
  auto dY = EigenVector<T>::Flatten(grad_y);

  auto& place = *dev_ctx.eigen_device();
  if (is_test) {
    if (dropout_implementation == "upscale_in_train") {
      dX.device(place) = static_cast<T>(1) * dY;
    } else {
      dX.device(place) = dY * static_cast<T>(1.0f - dropout_prob);
    }
  } else {
    auto M = EigenVector<uint8_t>::Flatten(mask);
    if (dropout_implementation == "upscale_in_train") {
      if (dropout_prob == 1.0f) {
        dX.device(place) = static_cast<T>(0) * dY;
      } else {
        int vec_size = platform::GetVectorizedSize<T>(grad_y.data<T>());
        if (vec_size == 4 && size % 4 == 0) {
          auto factor = static_cast<T>(1.0f / (1.0f - dropout_prob));
          auto stream = dev_ctx.stream();
          platform::GpuLaunchConfig config =
              platform::GetGpuLaunchConfig1D(dev_ctx, size);
          DropoutGradCUDAKernel<
              T, uint8_t,
              4><<<config.block_per_grid, config.thread_per_block, 0, stream>>>(
              grad_y.data<T>(), mask.data<uint8_t>(), factor, size,
              grad_x->data<T>());
        } else {
          dX.device(place) =
              dY * M.cast<T>() / static_cast<T>(1.0f - dropout_prob);
        }
      }
    } else {
      dX.device(place) = dY * M.cast<T>();
    }
  }
}

}  // namespace operators
}  // namespace paddle
