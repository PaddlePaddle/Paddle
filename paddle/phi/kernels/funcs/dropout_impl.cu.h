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
#endif
#ifdef PADDLE_WITH_HIP
#include <hip/hip_runtime.h>
#include <hiprand_kernel.h>
#endif

#include "paddle/phi/kernels/funcs/dropout_impl_util.h"

#include "paddle/phi/backends/gpu/cuda/cuda_graph_with_memory_pool.h"
#include "paddle/phi/backends/gpu/gpu_launch_config.h"
#include "paddle/phi/kernels/funcs/broadcast_function.h"
#include "paddle/phi/kernels/funcs/distribution_helper.h"
#include "paddle/phi/kernels/funcs/functors.h"
#include "paddle/phi/kernels/primitive/compute_primitives.h"
#include "paddle/phi/kernels/primitive/datamover_primitives.h"
#include "paddle/phi/kernels/scale_kernel.h"

namespace phi {
namespace funcs {

template <typename T>
struct DstFunctor {
  using MT = typename phi::dtype::MPTypeTrait<T>::Type;

  HOSTDEVICE inline DstFunctor(const float retain_prob,
                               const bool is_upscale_in_train,
                               const int64_t num)
      : retain_prob_(retain_prob),
        is_upscale_in_train_(is_upscale_in_train),
        num_(num) {
    factor = static_cast<MT>(1.0f / retain_prob_);
  }

  HOSTDEVICE inline T operator()(const T src_val, const uint8_t mask) const {
    for (int i = 0; i < num_; i++) {
      if (mask == static_cast<uint8_t>(1)) {
        return is_upscale_in_train_
                   ? static_cast<T>(static_cast<MT>(src_val) * factor)
                   : static_cast<T>(src_val);
      } else {
        return static_cast<T>(0);
      }
    }
  }

 private:
  const float retain_prob_;
  const bool is_upscale_in_train_;
  const int64_t num_;
  MT factor;
};

template <typename T>
struct MaskFunctor {
  explicit MaskFunctor(const float retain_prob) : retain_prob_(retain_prob) {}

  HOSTDEVICE inline void operator()(T* dst, const float* rand, int num) const {
    static constexpr int kCount =
        phi::funcs::uniform_distribution<float>::kReturnsCount;
// 0 ~ kCount - 1 is dst, kCount ~ 2 * kCount - 1 is mask
#pragma unroll
    for (int i = 0; i < kCount; i++) {
      dst[i] = rand[i] < retain_prob_ ? static_cast<T>(1) : static_cast<T>(0);
    }
  }

 private:
  float retain_prob_;
};

template <typename T>
struct DstMaskFunctor {
  using MT = typename phi::dtype::MPTypeTrait<T>::Type;
  HOSTDEVICE inline DstMaskFunctor(const float retain_prob,
                                   const bool is_upscale_in_train)
      : retain_prob_(retain_prob), is_upscale_in_train_(is_upscale_in_train) {
    factor = static_cast<MT>(1.0f / retain_prob_);
  }

  HOSTDEVICE inline void operator()(T* dst,
                                    const T* src_val,
                                    const float* rand,
                                    int num) const {
    static constexpr int kCount =
        phi::funcs::uniform_distribution<float>::kReturnsCount;
// 0 ~ kCount - 1 is dst, kCount ~ 2 * kCount - 1 is mask
#pragma unroll
    for (int i = 0; i < kCount; i++) {
      if (rand[i] < retain_prob_) {
        dst[i] = is_upscale_in_train_
                     ? static_cast<T>(static_cast<MT>(src_val[i]) * factor)
                     : static_cast<T>(src_val[i]);
        dst[i + kCount] = static_cast<T>(1);
      } else {
        dst[i] = static_cast<T>(0);
        dst[i + kCount] = dst[i];
      }
    }
  }

 private:
  MT factor;
  float retain_prob_;
  bool is_upscale_in_train_;
};

template <typename T>
__global__ void VectorizedRandomGenerator(
    unsigned int
        identifier, /* This is used to relate kernel to cudaGraph nodes*/
    const size_t n,
    uint64_t seed,
    const float dropout_prob,
    const T* src,
    uint8_t* mask,
    T* dst,
    bool is_upscale_in_train,
    uint64_t increment,
    size_t main_offset) {
  size_t idx = static_cast<size_t>(BLOCK_ID_X * BLOCK_NUM_X);
  static constexpr int kCount =
      phi::funcs::uniform_distribution<float>::kReturnsCount;
  size_t stride = BLOCK_NUM_X * GRID_NUM_X * kCount;
#ifdef PADDLE_WITH_HIP
  hiprandStatePhilox4_32_10_t state;
  hiprand_init(seed, idx + THREAD_ID_X, increment, &state);
  using SType = hiprandStatePhilox4_32_10_t;
#else
  curandStatePhilox4_32_10_t state;
  curand_init(seed, idx + THREAD_ID_X, increment, &state);
  using SType = curandStatePhilox4_32_10_t;
#endif
  T dst_mask[kCount *
             2];  // 0 ~ kCount - 1 : dst,  kCount ~ 2 * kCount - 1: mask
  float rands[kCount];
  uint8_t mask_result[kCount];
  using Rand = phi::funcs::uniform_distribution<float>;
  using Cast = kps::IdentityFunctor<T>;
  int deal_size = BLOCK_NUM_X * kCount;

  size_t fix = idx * kCount;

  auto dst_functor =
      DstMaskFunctor<T>(1.0f - dropout_prob, is_upscale_in_train);
  for (; fix < main_offset; fix += stride) {
    kps::ReadData<T, kCount, 1, false>(&dst_mask[0], src + fix, deal_size);
    kps::ElementwiseRandom<SType, float, kCount, Rand>(
        &rands[0], Rand(), &state);
    // dst
    kps::OperatorTernary<T, float, T, DstMaskFunctor<T>>(
        &dst_mask[0], &dst_mask[0], &rands[0], dst_functor, kCount);
    kps::WriteData<T, kCount, 1, false>(dst + fix, &dst_mask[0], deal_size);
    // mask
    kps::ElementwiseUnary<T, uint8_t, kCount, 1, Cast>(
        &mask_result[0], &dst_mask[kCount], Cast());
    kps::WriteData<uint8_t, kCount, 1, false>(
        mask + fix, &mask_result[0], deal_size);
  }
  int remainder = n - fix;
  if (remainder > 0) {
    kps::ReadData<T, kCount, 1, true>(&dst_mask[0], src + fix, remainder);
    kps::ElementwiseRandom<SType, float, kCount, Rand>(
        &rands[0], Rand(), &state);
    // dst
    kps::OperatorTernary<T, float, T, DstMaskFunctor<T>>(
        &dst_mask[0], &dst_mask[0], &rands[0], dst_functor, kCount);
    kps::WriteData<T, kCount, 1, true>(dst + fix, &dst_mask[0], remainder);
    // mask
    kps::ElementwiseUnary<T, uint8_t, kCount, 1, Cast>(
        &mask_result[0], &dst_mask[kCount], Cast());
    kps::WriteData<uint8_t, kCount, 1, true>(
        mask + fix, &mask_result[0], remainder);
  }
}

template <typename T>
__global__ void VectorizedGeneratorMask(const size_t n,
                                        uint64_t seed,
                                        const float dropout_prob,
                                        const T* src,
                                        uint8_t* mask,
                                        uint64_t increment,
                                        size_t main_offset,
                                        MaskFunctor<T> mask_functor,

                                        const uint64_t* seed_ptr) {
  // Vectorized Generate Mask
  // kCount is 4 for curand_uniform4 is used
  if (seed_ptr) seed = seed_ptr[0];

  constexpr int kCount = phi::funcs::uniform_distribution<float>::kReturnsCount;
  size_t idx = static_cast<size_t>(BLOCK_ID_X * BLOCK_NUM_X);
  size_t stride = BLOCK_NUM_X * GRID_NUM_X * kCount;
#ifdef PADDLE_WITH_HIP
  hiprandStatePhilox4_32_10_t state;
  hiprand_init(seed, idx + THREAD_ID_X, increment, &state);
  using SType = hiprandStatePhilox4_32_10_t;
#else
  curandStatePhilox4_32_10_t state;
  curand_init(seed, idx + THREAD_ID_X, increment, &state);
  using SType = curandStatePhilox4_32_10_t;
#endif
  T dst_mask[kCount];  // 0 ~ kCount - 1 : dst,  kCount ~ 2 * kCount - 1: mask
  float rands[kCount];
  uint8_t mask_result[kCount];
  using Rand = phi::funcs::uniform_distribution<float>;
  using Cast = kps::IdentityFunctor<T>;
  int deal_size = BLOCK_NUM_X * kCount;

  size_t fix = idx * kCount;
  for (; fix < main_offset; fix += stride) {
    kps::ReadData<T, kCount, 1, false>(&dst_mask[0], src + fix, deal_size);
    kps::ElementwiseRandom<SType, float, kCount, Rand>(
        &rands[0], Rand(), &state);
    // dst
    kps::OperatorBinary<float, T, MaskFunctor<T>>(
        &dst_mask[0], &rands[0], mask_functor, kCount);

    // mask
    kps::ElementwiseUnary<T, uint8_t, kCount, 1, Cast>(
        &mask_result[0], &dst_mask[0], Cast());
    kps::WriteData<uint8_t, kCount, 1, false>(
        mask + fix, &mask_result[0], deal_size);
  }
  int remainder = n - fix;
  if (remainder > 0) {
    kps::ReadData<T, kCount, 1, true>(&dst_mask[0], src + fix, remainder);
    kps::ElementwiseRandom<SType, float, kCount, Rand>(
        &rands[0], Rand(), &state);
    // dst
    kps::OperatorBinary<float, T, MaskFunctor<T>>(
        &dst_mask[0], &rands[0], mask_functor, kCount);
    // mask
    kps::ElementwiseUnary<T, uint8_t, kCount, 1, Cast>(
        &mask_result[0], &dst_mask[0], Cast());
    kps::WriteData<uint8_t, kCount, 1, true>(
        mask + fix, &mask_result[0], remainder);
  }
}

template <typename T>
void DropoutFwGPUKernelDriver(
    const phi::GPUContext& dev_ctx,
    bool is_test,
    float dropout_prob,
    bool upscale_in_train,
    bool is_fix_seed,
    int seed_val,
    const phi::DenseTensor& x,
    const phi::DenseTensor* seed,
    phi::DenseTensor* mask,
    phi::DenseTensor* y,
    bool is_dropout_nd = false,
    const std::vector<int>& axis = std::vector<int>()) {
  int64_t x_numel = x.numel();
  auto stream = dev_ctx.stream();
  auto* x_data = x.data<T>();
  auto* y_data = y->data<T>();

  if (!is_test && mask) {
    auto* mask_data = mask->data<uint8_t>();
    size_t size = common::product(mask->dims());

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
    // VectorizedRandomGenerator use curand_uniform4, so kVecSize is 4;
    constexpr int kVecSize =
        phi::funcs::uniform_distribution<float>::kReturnsCount;
    auto gpu_config =
        phi::backends::gpu::GetGpuLaunchConfig1D(dev_ctx, x_numel, kVecSize);
    size_t grid_size = gpu_config.GetGridSize();
    size_t block_size = gpu_config.GetBlockSize();

    int64_t device_id = dev_ctx.GetPlace().GetDeviceId();
    const auto& prop = phi::backends::gpu::GetDeviceProperties(device_id);
    size_t max_grid_size = prop.maxThreadsPerMultiProcessor *
                           prop.multiProcessorCount / block_size;
    grid_size = std::min(grid_size, max_grid_size);

    auto offset =
        ((x_numel - 1) / (grid_size * block_size * kVecSize) + 1) * kVecSize;
    size_t main_offset =
        size / (block_size * kVecSize) * (block_size * kVecSize);

    if (is_dropout_nd) {
      auto mask_functor = MaskFunctor<T>(1.0f - dropout_prob);
      bool copy_in_kernel = GetSeedDataAndIncrement(dev_ctx,
                                                    seed,
                                                    is_fix_seed,
                                                    seed_val,
                                                    offset,
                                                    &seed_data,
                                                    &increment,
                                                    true);
      const uint64_t* seed_ptr =
          copy_in_kernel ? seed->data<uint64_t>() : nullptr;

      VectorizedGeneratorMask<T>
          <<<grid_size, block_size, 0, stream>>>(size,
                                                 seed_data,
                                                 dropout_prob,
                                                 x_data,
                                                 mask_data,
                                                 increment,
                                                 main_offset,
                                                 mask_functor,
                                                 seed_ptr);
      auto dst_functor =
          DstFunctor<T>(1.0f - dropout_prob, upscale_in_train, x_numel);
      std::vector<const phi::DenseTensor*> ins = {&x, mask};
      std::vector<phi::DenseTensor*> outs = {y};
      phi::funcs::BroadcastKernel<T>(dev_ctx, ins, &outs, dst_functor);
    } else {
      bool copy_in_kernel = GetSeedDataAndIncrement(
          dev_ctx, seed, is_fix_seed, seed_val, offset, &seed_data, &increment);
      const phi::GPUContext* dev_ctx_p = &dev_ctx;
      auto gen_cuda = dev_ctx.GetGenerator();
      auto state_index = gen_cuda->GetStateIndex();

      phi::backends::gpu::CUDAGraphNodeLauncher::parameterSetter_t
          parameterSetter = [offset, dev_ctx_p, state_index, is_fix_seed](
                                phi::backends::gpu::gpuKernelParams& params) {
            if (!is_fix_seed) {
          // we assume seed is null pointer
          // seed copy to cpu is meaningless here
#ifndef PADDLE_WITH_HIP
              assert(seed_tensor_ptr == nullptr);
#endif
              auto gen_cuda = dev_ctx_p->GetGenerator();
              // ensure the generator use correct state index
              gen_cuda->SetStateIndex(state_index);

              uint64_t seed, increment;
              std::tie(seed, increment) = gen_cuda->IncrementOffset(offset);

              params.As<uint64_t>(2) = seed;
              params.As<uint64_t>(8) = increment;

              VLOG(10) << "CUDA_GRAPH seed = " << seed
                       << ", increment = " << increment;
            }
          };

      phi::backends::gpu::CUDAGraphNodeLauncher::gpuKernelCallback_t
          cudaKernelCallback = [=](unsigned int id) {
            void* functionPtr =
                reinterpret_cast<void*>(&(VectorizedRandomGenerator<T>));
#ifdef PADDLE_WITH_HIP
            hipFunction_t cudaFunc =
                reinterpret_cast<hipFunction_t>(functionPtr);
#else
            cudaFunction_t cudaFunc;
            PADDLE_ENFORCE_GPU_SUCCESS(
                cudaGetFuncBySymbol(&cudaFunc, functionPtr));
#endif
            VLOG(10) << "[cudaKernelCallback] cudaFunc = " << cudaFunc
                     << " functionPtr = " << functionPtr;

            VectorizedRandomGenerator<T>
                <<<grid_size, block_size, 0, stream>>>(id,
                                                       size,
                                                       seed_data,
                                                       dropout_prob,
                                                       x_data,
                                                       mask_data,
                                                       y_data,
                                                       upscale_in_train,
                                                       increment,
                                                       main_offset);
            return cudaFunc;
          };
      phi::backends::gpu::CUDAGraphNodeLauncher::Instance().KernelNodeLaunch(
          parameterSetter, cudaKernelCallback);

      VLOG(10) << "NON_CUDA_GRAPH seed = " << seed_data
               << ", increment = " << increment;
    }
  } else {
    if (upscale_in_train) {
      // y = x
      phi::Copy(dev_ctx, x, dev_ctx.GetPlace(), false, y);
    } else {
      using MT = typename phi::dtype::MPTypeTrait<T>::Type;
      MT factor = static_cast<MT>(1.0f - dropout_prob);
      // y = factor * x
      phi::ScaleKernel<T, phi::GPUContext>(dev_ctx, x, factor, 0.0f, false, y);
    }
  }
}

template <typename T>
struct CudaDropoutGradFunctor {
  using MT = typename phi::dtype::MPTypeTrait<T>::Type;

  explicit CudaDropoutGradFunctor(const MT factor) : factor_(factor) {}

  __device__ __forceinline__ T operator()(const T dout,
                                          const uint8_t mask) const {
    return static_cast<T>(static_cast<MT>(dout) * static_cast<MT>(mask) *
                          factor_);
  }

 private:
  MT factor_;
};

template <typename T>
void DropoutGradGPUKernelDriver(const phi::GPUContext& dev_ctx,
                                bool is_test,
                                float dropout_prob,
                                bool upscale_in_train,
                                const phi::DenseTensor& grad_y,
                                const phi::DenseTensor& mask,
                                phi::DenseTensor* grad_x,
                                bool is_dropout_nd = false) {
  using MT = typename phi::dtype::MPTypeTrait<T>::Type;

  auto stream = dev_ctx.stream();
  if (is_test) {
    MT factor = static_cast<MT>(upscale_in_train ? 1.0f : 1.0f - dropout_prob);
    // y = factor * x
    phi::ScaleKernel<T, phi::GPUContext>(
        dev_ctx, grad_y, factor, 0.0f, false, grad_x);
  } else {
    if (upscale_in_train && dropout_prob == 1.0f) {
#ifdef PADDLE_WITH_HIP
      hipMemset(grad_x->data<T>(), 0, grad_x->numel() * sizeof(T));
#else
      cudaMemset(grad_x->data<T>(), 0, grad_x->numel() * sizeof(T));
#endif
    } else {
      MT factor = upscale_in_train
                      ? static_cast<MT>(1.0f / (1.0f - dropout_prob))
                      : static_cast<MT>(1.0f);

      std::vector<const phi::DenseTensor*> ins = {&grad_y, &mask};
      std::vector<phi::DenseTensor*> outs = {grad_x};
      if (is_dropout_nd) {
        phi::funcs::BroadcastKernel<T>(
            dev_ctx, ins, &outs, CudaDropoutGradFunctor<T>(factor));
      } else {
        phi::funcs::ElementwiseKernel<T>(
            dev_ctx, ins, &outs, CudaDropoutGradFunctor<T>(factor));
      }
    }
  }
}

}  // namespace funcs
}  // namespace phi
