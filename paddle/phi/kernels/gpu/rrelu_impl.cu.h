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

// This whole file is modified based on the file 
// "paddle/fluid/operators/dropout_impl.cu.h"
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
#include "paddle/fluid/operators/amp/fp16_type_traits.h"
// This file "dropout_impl_util.h" is originally used in the implementation 
// of dropout op  and can be used in rrelu op without any modification.
#include "paddle/fluid/operators/dropout_impl_util.h"   
#include "paddle/fluid/operators/elementwise/elementwise_op_impl.cu.h"
#include "paddle/fluid/platform/aligned_vector.h"
#include "paddle/phi/backends/gpu/gpu_launch_config.h"
#include "paddle/phi/kernels/funcs/functors.h"

// Maybe it is better to put the following functions and classes in the namespace phi 
// TODO: remain to be done
namespace paddle {
namespace operators {

template <typename T>
__global__ void RandomGenerator(const size_t n, uint64_t seed,
                                bool is_test,
                                const float lower,
                                const float upper,
                                const T* src,
                                T* mask,
                                T* dst,
                                uint64_t increment) {
  using MT = typename details::MPTypeTrait<T>::Type;
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
#ifdef PADDLE_WITH_HIP
    hiprandStatePhilox4_32_10_t state;
    hiprand_init(seed, idx, increment, &state);
#else
    curandStatePhilox4_32_10_t state;
    curand_init(seed, idx, increment, &state);
#endif

  T mask_val;
  T dst_val;

  if (!is_test) {
    for (; idx < n; idx += blockDim.x * gridDim.x) {
      T src_val = src[idx];
      if (src_val < static_cast<T>(0)) {
#ifdef PADDLE_WITH_HIP
        // random_sampled_value should be in [0, 1]
        float random_sampled_value = static_cast<float>(hiprand_uniform(&state));
#else
        // random_sampled_value should be in [0, 1]
        float random_sampled_value = static_cast<float>(curand_uniform(&state));
#endif
        random_sampled_value = random_sampled_value * (upper - lower) + lower;
        mask_val = static_cast<T>(random_sampled_value);
        dst_val = static_cast<T>(static_cast<MT>(src_val) * static_cast<MT>(random_sampled_value));
      } else {
        mask_val = static_cast<T>(1.0f);
        dst_val = src_val;
      }
      mask[idx] = mask_val;
      dst[idx] = dst_val;
    }
  } else {
    float middle_value = (lower + upper) / 2.0f;
    for (; idx < n; idx += blockDim.x * gridDim.x) {
      T src_val = src[idx];
      if (src_val < static_cast<T>(0)) {
        mask_val = static_cast<T>(middle_value);
        dst_val = static_cast<T>(static_cast<MT>(src_val) * static_cast<MT>(middle_value));
      } else {
        mask_val = static_cast<T>(1.0f);
        dst_val = src_val;
      }
      mask[idx] = mask_val;
      dst[idx] = dst_val;
    }
  }
}

template <typename T, int VecSize>
__global__ void VectorizedRandomGenerator(const size_t n, uint64_t seed,
                                          bool is_test,
                                          const float lower,
                                          const float upper,
                                          const T* src, 
                                          T* mask,
                                          T* dst,
                                          uint64_t increment) {
  using MT = typename details::MPTypeTrait<T>::Type;
  using LoadT = phi::AlignedVector<T, VecSize>;

#ifdef PADDLE_WITH_HIP
  int64_t idx = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
  hiprandStatePhilox4_32_10_t state;
  hiprand_init(seed, idx, increment, &state);
#else
  int64_t idx = blockDim.x * blockIdx.x + threadIdx.x;
  curandStatePhilox4_32_10_t state;
  curand_init(seed, idx, increment, &state);
#endif

  if (!is_test) {
    for (int i = idx * VecSize; i < n; i += blockDim.x * gridDim.x * VecSize) {
      LoadT src_val;
      phi::Load<T, VecSize>(&src[i], &src_val);

  #ifdef PADDLE_WITH_HIP
      float4 rand = hiprand_uniform4(&state);
  #else
      float4 rand = curand_uniform4(&state);
  #endif

      LoadT dst_val;
      LoadT mask_val;

  #pragma unroll
      for (int j = 0; j < VecSize; j++) {
        if (src_val[j] >= static_cast<T>(0)) {
          dst_val[j] = src_val[j];
          mask_val[j] = static_cast<T>(1);
        } else {
          // random_sampled_value should be in [0, 1]
          float random_sampled_value = static_cast<float>((&rand.x)[j]);
          random_sampled_value = random_sampled_value * (upper - lower) + lower;
          dst_val[j] = static_cast<T>(static_cast<MT>(src_val[j]) * static_cast<MT>(random_sampled_value));
          mask_val[j] = static_cast<T>(random_sampled_value);
        }
      }
      phi::Store<T, VecSize>(dst_val, &dst[i]);
      phi::Store<T, VecSize>(mask_val, &mask[i]);
    }
  } else {
    float middle_value = (lower + upper) / 2.0f;
    for (int i = idx * VecSize; i < n; i += blockDim.x * gridDim.x * VecSize) {
      LoadT src_val;
      phi::Load<T, VecSize>(&src[i], &src_val);
      LoadT dst_val;
      LoadT mask_val;

  #pragma unroll
      for (int j = 0; j < VecSize; j++) {
        if (src_val[j] >= static_cast<T>(0)) {
          dst_val[j] = src_val[j];
          mask_val[j] = static_cast<T>(1);
        } else {
          dst_val[j] = static_cast<T>(static_cast<MT>(src_val[j]) * static_cast<MT>(middle_value));
          mask_val[j] = static_cast<T>(middle_value);
        }
      }
      phi::Store<T, VecSize>(dst_val, &dst[i]);
      phi::Store<T, VecSize>(mask_val, &mask[i]);
    }
  }
}

template <typename T>
struct CudaRReluGradFunctor {
  using MT = typename details::MPTypeTrait<T>::Type;

  explicit CudaRReluGradFunctor(const MT factor) : factor_(factor) {}

  __device__ __forceinline__ T operator()(const T dout, const T mask) const {
    return static_cast<T>(static_cast<MT>(dout) * static_cast<MT>(mask));
  }

 private:
 // factor_ is useless, but I am not sure how to delete it in a safe way
  MT factor_;
};

template <typename T>
void RReluFwGPUKernelDriver(const phi::GPUContext& dev_ctx, bool is_test,
                            float lower, float upper,
                            bool is_fix_seed, int seed_val,
                            const framework::Tensor& x,
                            const framework::Tensor* seed,
                            framework::Tensor* mask, framework::Tensor* y) {
  auto& place = *dev_ctx.eigen_device();
  uint64_t x_numel = x.numel();
  auto stream = dev_ctx.stream();
  auto* x_data = x.data<T>();
  auto* y_data = y->data<T>();
  auto* mask_data = mask->data<T>();
  size_t size = x_numel;

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
  int vec_size = (phi::GetVectorizedSize<T>(x_data) == 4) ? 4 : 1;
  auto gpu_config =
      phi::backends::gpu::GetGpuLaunchConfig1D(dev_ctx, x_numel, vec_size);
  auto offset =
      ((x_numel - 1) / (gpu_config.GetThreadNum() * vec_size) + 1) * vec_size;

  GetSeedDataAndIncrement(dev_ctx, seed, is_fix_seed, seed_val, offset,
                          &seed_data, &increment);

#ifdef __HIPCC__
  if (vec_size == 4 && size % 4 == 0) {
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(VectorizedRandomGenerator<T, 4>),
        gpu_config.GetGridSize(), gpu_config.GetBlockSize(), 0, stream, 
        size, seed_data, is_test, lower, upper, x_data, mask_data, y_data, increment);
  } else {
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(RandomGenerator<T>),
        gpu_config.GetGridSize(), gpu_config.GetBlockSize(), 0, stream, 
        size, seed_data, is_test, lower, upper, x_data, mask_data, y_data, increment);
  }
#else
  if (vec_size == 4 && size % 4 == 0) {
    VectorizedRandomGenerator<T, 4><<<
        gpu_config.block_per_grid, gpu_config.thread_per_block, 0, stream>>>(
        size, seed_data, is_test, lower, upper, x_data, mask_data, y_data, increment);
  } else {
    RandomGenerator<T><<<
        gpu_config.block_per_grid, gpu_config.thread_per_block, 0, stream>>>(
        size, seed_data, is_test, lower, upper, x_data, mask_data, y_data, increment);
  }
#endif
}

template <typename T>
void RReluGradGPUKernelDriver(const phi::GPUContext& dev_ctx,
                              const framework::Tensor& grad_y,
                              const framework::Tensor& mask, 
                              framework::Tensor* grad_x) {
  using MT = typename details::MPTypeTrait<T>::Type;
  MT factor = static_cast<MT>(1.0f);

  std::vector<const framework::Tensor*> ins = {&grad_y, &mask};
  std::vector<framework::Tensor*> outs = {grad_x};
  paddle::operators::LaunchSameDimsElementwiseCudaKernel<T>(
      dev_ctx, ins, &outs, CudaRReluGradFunctor<T>(factor));
}

}  // namespace operators
}  // namespace paddle
