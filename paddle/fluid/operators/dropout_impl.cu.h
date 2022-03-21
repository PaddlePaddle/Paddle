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
#include "paddle/fluid/operators/amp/fp16_type_traits.h"
#include "paddle/fluid/operators/dropout_impl_util.h"
#include "paddle/fluid/operators/elementwise/elementwise_op_impl.cu.h"
#include "paddle/fluid/platform/aligned_vector.h"
#include "paddle/phi/backends/gpu/gpu_launch_config.h"
#include "paddle/phi/kernels/funcs/functors.h"

DECLARE_bool(use_curand);

namespace paddle {
namespace operators {

template <typename T, typename MaskType>
__global__ void RandomGenerator(const size_t n, uint64_t seed,
                                const float dropout_prob, const T* src,
                                MaskType* mask, T* dst,
                                bool is_upscale_in_train, uint64_t increment) {
  using MT = typename details::MPTypeTrait<T>::Type;
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
  MT factor = static_cast<MT>(1.0f / (1.0f - dropout_prob));
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
      dst_val = is_upscale_in_train
                    ? static_cast<T>(static_cast<MT>(src_val) * factor)
                    : src_val;
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
  using MT = typename details::MPTypeTrait<T>::Type;
  using LoadT = phi::AlignedVector<T, VecSize>;
  using MaskLoadT = phi::AlignedVector<MaskType, VecSize>;

#ifdef PADDLE_WITH_HIP
  int64_t idx = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
  hiprandStatePhilox4_32_10_t state;
  hiprand_init(seed, idx, increment, &state);
#else
  int64_t idx = blockDim.x * blockIdx.x + threadIdx.x;
  curandStatePhilox4_32_10_t state;
  curand_init(seed, idx, increment, &state);
#endif

  MT factor = static_cast<MT>(1.0f / (1.0f - dropout_prob));
  for (int i = idx * VecSize; i < n; i += blockDim.x * gridDim.x * VecSize) {
    LoadT src_val;
    phi::Load<T, VecSize>(&src[i], &src_val);

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
        dst_val[j] = is_upscale_in_train
                         ? static_cast<T>(static_cast<MT>(src_val[j]) * factor)
                         : src_val[j];
        mask_val[j] = 1;
      }
    }

    phi::Store<T, VecSize>(dst_val, &dst[i]);
    phi::Store<MaskType, VecSize>(mask_val, &mask[i]);
  }
}

template <typename T, typename MaskType>
struct CudaDropoutGradFunctor {
  using MT = typename details::MPTypeTrait<T>::Type;

  explicit CudaDropoutGradFunctor(const MT factor) : factor_(factor) {}

  __device__ __forceinline__ T operator()(const T dout,
                                          const MaskType mask) const {
    return static_cast<T>(static_cast<MT>(dout) * static_cast<MT>(mask) *
                          factor_);
  }

 private:
  MT factor_;
};

template <typename T, typename MaskType, int VecSize>
__global__ void DropoutGradCUDAKernel(
    const T* dout, const MaskType* mask,
    const typename details::MPTypeTrait<T>::Type factor, const int64_t size,
    T* dx) {
  using MT = typename details::MPTypeTrait<T>::Type;
  using LoadT = phi::AlignedVector<T, VecSize>;
  using MaskLoadT = phi::AlignedVector<MaskType, VecSize>;

  int64_t idx = blockDim.x * blockIdx.x + threadIdx.x;
  for (int i = idx * VecSize; i < size; i += blockDim.x * gridDim.x * VecSize) {
    LoadT dout_val;
    phi::Load<T, VecSize>(&dout[i], &dout_val);

    MaskLoadT mask_val;
    phi::Load<MaskType, VecSize>(&mask[i], &mask_val);

    LoadT dx_val;

#pragma unroll
    for (int j = 0; j < VecSize; j++) {
      dx_val[j] = static_cast<T>(static_cast<MT>(dout_val[j]) *
                                 static_cast<MT>(mask_val[j]) * factor);
    }

    phi::Store<T, VecSize>(dx_val, &dx[i]);
  }
}

/*********************************************************************/
/***** Function for new implementation(2022/03/14) of dropout OP *****/
template <typename T>
int get_vectorized_size(const framework::Tensor& x) {
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
/***************************************************************/

template <typename T>
void DropoutFwGPUKernelDriver(const phi::GPUContext& dev_ctx, bool is_test,
                              const std::string dropout_implementation,
                              float dropout_prob, bool upscale_in_train,
                              bool is_fix_seed, int seed_val,
                              const framework::Tensor& x,
                              const framework::Tensor* seed,
                              framework::Tensor* mask, framework::Tensor* y) {
  auto& place = *dev_ctx.eigen_device();
  int64_t x_numel = x.numel();
  auto stream = dev_ctx.stream();
  auto* x_data = x.data<T>();
  auto* y_data = y->data<T>();

  if (!is_test) {
    auto* mask_data = mask->data<uint8_t>();
    size_t size = phi::product(mask->dims());

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

    uint64_t seed_data;
    uint64_t increment;

    int vec_size = (phi::GetVectorizedSize<T>(x_data) == 4) ? 4 : 1;
    if (FLAGS_use_curand) {
      vec_size = get_vectorized_size<T>(x);
    }

    auto gpu_config =
        phi::backends::gpu::GetGpuLaunchConfig1D(dev_ctx, x_numel, vec_size);
    size_t grid_size = gpu_config.GetGridSize();
    size_t block_size = gpu_config.GetBlockSize();
    if (FLAGS_use_curand) {
      int64_t device_id = dev_ctx.GetPlace().GetDeviceId();
      const auto& prop = platform::GetDeviceProperties(device_id);
      size_t max_grid_size = prop.maxThreadsPerMultiProcessor *
                             prop.multiProcessorCount / block_size;
      grid_size = std::min(grid_size, max_grid_size);
    }

    auto offset =
        ((x_numel - 1) / (grid_size * block_size * vec_size) + 1) * vec_size;
    GetSeedDataAndIncrement(dev_ctx, seed, is_fix_seed, seed_val, offset,
                            &seed_data, &increment);

    if (FLAGS_use_curand) {
#ifdef PADDLE_WITH_HIP
      switch (vec_size) {
        case 4:
          hipLaunchKernelGGL(HIP_KERNEL_NAME(dropout_kernel_vec<T, uint8_t, 4>),
                             grid_size, block_size, 0, stream, x_numel,
                             seed_data, dropout_prob, x_data, mask_data, y_data,
                             upscale_in_train, increment);
          break;
        case 2:
          hipLaunchKernelGGL(HIP_KERNEL_NAME(dropout_kernel_vec<T, uint8_t, 2>),
                             grid_size, block_size, 0, stream, x_numel,
                             seed_data, dropout_prob, x_data, mask_data, y_data,
                             upscale_in_train, increment);
          break;
        case 1:
          hipLaunchKernelGGL(HIP_KERNEL_NAME(dropout_kernel<T, uint8_t>),
                             grid_size, block_size, 0, stream, x_numel,
                             seed_data, dropout_prob, x_data, mask_data, y_data,
                             upscale_in_train, increment);
          break;
      }
#else
      switch (vec_size) {
        case 4:
          dropout_kernel_vec<T, uint8_t,
                             4><<<grid_size, block_size, 0, stream>>>(
              x_numel, seed_data, dropout_prob, x_data, mask_data, y_data,
              upscale_in_train, increment);
          break;
        case 2:
          dropout_kernel_vec<T, uint8_t,
                             2><<<grid_size, block_size, 0, stream>>>(
              x_numel, seed_data, dropout_prob, x_data, mask_data, y_data,
              upscale_in_train, increment);
          break;
        case 1:
          dropout_kernel<T, uint8_t><<<grid_size, block_size, 0, stream>>>(
              x_numel, seed_data, dropout_prob, x_data, mask_data, y_data,
              upscale_in_train, increment);
          break;
      }
#endif
      return;
    }

#ifdef __HIPCC__
    if (vec_size == 4 && size % 4 == 0) {
      hipLaunchKernelGGL(
          HIP_KERNEL_NAME(VectorizedRandomGenerator<T, uint8_t, 4>),
          gpu_config.GetGridSize(), gpu_config.GetBlockSize(), 0, stream, size,
          seed_data, dropout_prob, x_data, mask_data, y_data, upscale_in_train,
          increment);
    } else {
      hipLaunchKernelGGL(HIP_KERNEL_NAME(RandomGenerator<T, uint8_t>),
                         gpu_config.GetGridSize(), gpu_config.GetBlockSize(), 0,
                         stream, size, seed_data, dropout_prob, x_data,
                         mask_data, y_data, upscale_in_train, increment);
    }
#else
    if (vec_size == 4 && size % 4 == 0) {
      VectorizedRandomGenerator<T, uint8_t, 4><<<
          gpu_config.block_per_grid, gpu_config.thread_per_block, 0, stream>>>(
          size, seed_data, dropout_prob, x_data, mask_data, y_data,
          upscale_in_train, increment);
    } else {
      RandomGenerator<T, uint8_t><<<gpu_config.block_per_grid,
                                    gpu_config.thread_per_block, 0, stream>>>(
          size, seed_data, dropout_prob, x_data, mask_data, y_data,
          upscale_in_train, increment);
    }
#endif
  } else {
    if (upscale_in_train) {
// todo: can y share with data with x directly?
#ifdef PADDLE_WITH_HIP
      PADDLE_ENFORCE_GPU_SUCCESS(
          hipMemcpyAsync(y_data, x_data, sizeof(T) * x_numel,
                         hipMemcpyDeviceToDevice, stream));
#else
      PADDLE_ENFORCE_GPU_SUCCESS(
          cudaMemcpyAsync(y_data, x_data, sizeof(T) * x_numel,
                          cudaMemcpyDeviceToDevice, stream));
#endif
    } else {
      using MT = typename details::MPTypeTrait<T>::Type;
      MT factor = static_cast<MT>(1.0f - dropout_prob);
      std::vector<const framework::Tensor*> ins = {&x};
      std::vector<framework::Tensor*> outs = {y};
      auto functor = phi::funcs::ScaleFunctor<T>(factor);
      paddle::operators::LaunchSameDimsElementwiseCudaKernel<T>(dev_ctx, ins,
                                                                &outs, functor);
    }
  }
}

template <typename T>
void DropoutGradGPUKernelDriver(const phi::GPUContext& dev_ctx,
                                const std::string dropout_implementation,
                                float dropout_prob,
                                const framework::Tensor& grad_y,
                                const framework::Tensor& mask, int64_t size,
                                framework::Tensor* grad_x,
                                bool is_test = false) {
  using MT = typename details::MPTypeTrait<T>::Type;
  auto stream = dev_ctx.stream();
  MT factor;
  if (is_test) {
    if (dropout_implementation == "upscale_in_train") {
      factor = static_cast<MT>(1.0f);
    } else {
      factor = static_cast<MT>(1.0f - dropout_prob);
    }
    std::vector<const framework::Tensor*> ins = {&grad_y};
    std::vector<framework::Tensor*> outs = {grad_x};
    auto functor = phi::funcs::ScaleFunctor<T>(factor);
    paddle::operators::LaunchSameDimsElementwiseCudaKernel<T>(dev_ctx, ins,
                                                              &outs, functor);
  } else {
    std::vector<const framework::Tensor*> ins = {&grad_y, &mask};
    std::vector<framework::Tensor*> outs = {grad_x};
    if (dropout_implementation == "upscale_in_train") {
      if (dropout_prob == 1.0f) {
#ifdef PADDLE_WITH_HIP
        hipMemset(grad_x->data<T>(), 0, size * sizeof(T));
#else
        cudaMemset(grad_x->data<T>(), 0, size * sizeof(T));
#endif
      } else {
        factor = static_cast<MT>(1.0f / (1.0f - dropout_prob));
        paddle::operators::LaunchSameDimsElementwiseCudaKernel<T>(
            dev_ctx, ins, &outs, CudaDropoutGradFunctor<T, uint8_t>(factor));
      }
    } else {
      factor = static_cast<MT>(1.0f);
      paddle::operators::LaunchSameDimsElementwiseCudaKernel<T>(
          dev_ctx, ins, &outs, CudaDropoutGradFunctor<T, uint8_t>(factor));
    }
  }
}

}  // namespace operators
}  // namespace paddle
