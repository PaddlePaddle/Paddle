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
#include "paddle/phi/kernels/funcs/distribution_helper.h"
#include "paddle/phi/kernels/funcs/functors.h"

namespace paddle {
namespace operators {

template <typename T>
struct RandomGeneratorDstValFunctor {
  const float dropout_prob_;
  const bool is_upscale_in_train_;
  HOSTDEVICE inline RandomGeneratorDstValFunctor(const float dropout_prob,
                                                 const bool is_upscale_in_train)
      : dropout_prob_(dropout_prob),
        is_upscale_in_train_(is_upscale_in_train) {}

  HOSTDEVICE inline T operator()(const T src_val, const T rand) const {
    using MT = typename details::MPTypeTrait<T>::Type;
    MT factor = static_cast<MT>(1.0f / (1.0f - dropout_prob_));
    if ((rand) < dropout_prob_) {
      return 0;
    } else {
      return is_upscale_in_train_
                 ? static_cast<T>(static_cast<MT>(src_val) * factor)
                 : src_val;
    }
  }
};

template <typename T>
struct RandomGeneratorMaskValFunctor {
  const float dropout_prob_;
  HOSTDEVICE inline RandomGeneratorMaskValFunctor(const float dropout_prob)
      : dropout_prob_(dropout_prob) {}

  HOSTDEVICE inline T operator()(const T src_val, const T rand) const {
    // rand = static_cast<float4>(rand);
    using MT = typename details::MPTypeTrait<T>::Type;
    if (rand < dropout_prob_) {
      return 0;
    } else {
      return 1;
    }
  }
};

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

  size_t idx = static_cast<size_t>(BLOCK_ID_X * BLOCK_NUM_X);

#ifdef PADDLE_WITH_HIP
  hiprandStatePhilox4_32_10_t state;
  hiprand_init(seed, idx + THREAD_ID_X, increment, &state);
  using SType = hiprandStatePhilox4_32_10_t;
#else
  curandStatePhilox4_32_10_t state;
  curand_init(seed, idx + THREAD_ID_X, increment, &state);
  using SType = curandStatePhilox4_32_10_t;
#endif

  constexpr int kCount = VecSize;
  MT rands[kCount];
  MaskType mask_result[kCount];
  T dst_result[kCount];
  for (int i = idx * VecSize; i < n; i += BLOCK_NUM_X * GRID_NUM_X * VecSize) {
    T src_val[kCount];
    bool is_boundary = n - i > BLOCK_NUM_X * VecSize;
    if (is_boundary) {
      kps::ReadData<T, kCount, 1, 1, true>(
          src_val, &src[i], static_cast<int>(BLOCK_NUM_X * VecSize));
    } else {
      kps::ReadData<T, kCount, 1, 1, false>(src_val, &src[i], n - i);
    }
    MT src_mt[kCount];
    kps::ElementwiseUnary<T, MT, kCount, 1, 1, kps::IdentityFunctor<T>>(
        src_mt, src_val, kps::IdentityFunctor<T>());
    kps::ElementwiseRandom<SType, MT, kCount, 1,
                           phi::funcs::normal_distribution<float>>(
        rands, phi::funcs::normal_distribution<float>(), &state);

    T dst_val[kCount];
    MaskType mask_val[kCount];

    auto dst_val_functor =
        RandomGeneratorDstValFunctor<MT>(dropout_prob, is_upscale_in_train);
    auto mask_val_functor = RandomGeneratorMaskValFunctor<MT>(dropout_prob);

    kps::ElementwiseBinary<MT, T, kCount, 1, 1,
                           RandomGeneratorDstValFunctor<MT>>(
        dst_result, src_mt, rands, dst_val_functor);
    kps::ElementwiseBinary<MT, MaskType, kCount, 1, 1,
                           RandomGeneratorMaskValFunctor<MT>>(
        mask_result, src_mt, rands, mask_val_functor);
    if (is_boundary) {
      kps::WriteData<T, kCount, 1, 1, true>(
          dst_val, dst_result, static_cast<int>(BLOCK_NUM_X * VecSize));
      kps::WriteData<MaskType, kCount, 1, 1, true>(
          mask_val, &mask_result[0], static_cast<int>(BLOCK_NUM_X * VecSize));
    } else {
      kps::WriteData<T, kCount, 1, 1, false>(dst_val, dst_result, n - i);
      kps::WriteData<MaskType, kCount, 1, 1, false>(mask_val, &mask_result[0],
                                                    n - i);
    }
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
