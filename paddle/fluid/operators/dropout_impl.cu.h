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
#include "paddle/fluid/platform/cuda_graph_with_memory_pool.h"
#include "paddle/phi/backends/gpu/gpu_launch_config.h"
#include "paddle/phi/kernels/funcs/broadcast_function.h"
#include "paddle/phi/kernels/funcs/distribution_helper.h"
#include "paddle/phi/kernels/funcs/functors.h"

namespace paddle {
namespace operators {

template <typename T1, typename T2 = T1, typename OutT = T1>
struct DstMaskFunctor {
  const float retain_prob_;
  const bool is_upscale_in_train_;
  using MT = typename details::MPTypeTrait<T1>::Type;
  MT factor;
  HOSTDEVICE inline DstMaskFunctor(const float retain_prob,
                                   const bool is_upscale_in_train)
      : retain_prob_(retain_prob), is_upscale_in_train_(is_upscale_in_train) {
    factor = static_cast<MT>(1.0f / retain_prob_);
  }

  HOSTDEVICE inline void operator()(OutT* dst,
                                    const T1* src_val,
                                    const T2* rand,
                                    int num) const {
    static constexpr int kCount =
        phi::funcs::uniform_distribution<T2>::kReturnsCount;
// 0 ~ kCount -1 is dist , kCount ~ 2 * kCount - 1 is mask
#pragma unroll
    for (int i = 0; i < kCount; i++) {
      if (rand[i] < retain_prob_) {
        dst[i] = is_upscale_in_train_
                     ? static_cast<T1>(static_cast<MT>(src_val[i]) * factor)
                     : static_cast<T1>(src_val[i]);
        dst[i + kCount] = static_cast<T1>(1);
      } else {
        dst[i] = static_cast<T1>(0);
        dst[i + kCount] = dst[i];
      }
    }
  }
};

template <typename T, typename MaskType>
__global__ void VectorizedRandomGenerator(const size_t n,
                                          uint64_t seed,
                                          const float dropout_prob,
                                          const T* src,
                                          MaskType* mask,
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
  T dst_mask[kCount * 2];  // 0 ~ kCount -1 : dst;kCount ~ 2 * kCount - 1: mask
  float rands[kCount];
  MaskType mask_result[kCount];
  using Rand = phi::funcs::uniform_distribution<float>;
  using Cast = kps::IdentityFunctor<T>;
  int deal_size = BLOCK_NUM_X * kCount;

  size_t fix = idx * kCount;

  auto dst_functor =
      DstMaskFunctor<T, float>(1.0f - dropout_prob, is_upscale_in_train);
  for (; fix < main_offset; fix += stride) {
    kps::ReadData<T, kCount, 1, false>(&dst_mask[0], src + fix, deal_size);
    kps::ElementwiseRandom<SType, float, kCount, Rand>(
        &rands[0], Rand(), &state);
    // dst
    kps::OperatorTernary<T, float, T, DstMaskFunctor<T, float>>(
        &dst_mask[0], &dst_mask[0], &rands[0], dst_functor, kCount);
    kps::WriteData<T, kCount, 1, false>(dst + fix, &dst_mask[0], deal_size);
    // mask
    kps::ElementwiseUnary<T, MaskType, kCount, 1, Cast>(
        &mask_result[0], &dst_mask[kCount], Cast());
    kps::WriteData<MaskType, kCount, 1, false>(
        mask + fix, &mask_result[0], deal_size);
    if (fix > idx * kCount + 1) {
      __syncthreads();
    }
  }
  int remainder = n - fix;
  if (remainder > 0) {
    kps::ReadData<T, kCount, 1, true>(&dst_mask[0], src + fix, remainder);
    kps::ElementwiseRandom<SType, float, kCount, Rand>(
        &rands[0], Rand(), &state);
    // dst
    kps::OperatorTernary<T, float, T, DstMaskFunctor<T, float>>(
        &dst_mask[0], &dst_mask[0], &rands[0], dst_functor, kCount);
    kps::WriteData<T, kCount, 1, true>(dst + fix, &dst_mask[0], remainder);
    // mask
    kps::ElementwiseUnary<T, MaskType, kCount, 1, Cast>(
        &mask_result[0], &dst_mask[kCount], Cast());
    kps::WriteData<MaskType, kCount, 1, true>(
        mask + fix, &mask_result[0], remainder);
    __syncthreads();
  }
}

template <typename T1, typename T2 = T1, typename OutT = T1>
struct MaskFunctor {
  const float retain_prob_;
  using MT = typename details::MPTypeTrait<T1>::Type;
  MT factor;
  HOSTDEVICE inline MaskFunctor(const float retain_prob)
      : retain_prob_(retain_prob) {
    factor = static_cast<MT>(1.0f / retain_prob_);
  }

  HOSTDEVICE inline void operator()(OutT* dst, const T2* rand, int num) const {
    static constexpr int kCount =
        phi::funcs::uniform_distribution<T2>::kReturnsCount;
// 0 ~ kCount -1 is dist , kCount ~ 2 * kCount - 1 is mask
#pragma unroll
    for (int i = 0; i < kCount; i++) {
      if (rand[i] < retain_prob_) {
        dst[i] = static_cast<T1>(1);
      } else {
        dst[i] = static_cast<T1>(0);
      }
    }
  }
};

template <typename T, typename MaskType>
struct DstFunctor {
  using MT = typename details::MPTypeTrait<T>::Type;
  MT factor;
  HOSTDEVICE inline DstFunctor(const float retain_prob,
                               const bool is_upscale_in_train,
                               const int64_t num)
      : retain_prob_(retain_prob),
        is_upscale_in_train_(is_upscale_in_train),
        num_(num) {
    factor = static_cast<MT>(1.0f / retain_prob_);
  }

  HOSTDEVICE inline T operator()(const T src_val, const MaskType mask) const {
    for (int i = 0; i < num_; i++) {
      if (mask == static_cast<MaskType>(1)) {
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
};

template <typename T, typename MaskType>
__global__ void VectorizedGeneratorMask(const size_t n,
                                        uint64_t seed,
                                        const float dropout_prob,
                                        const T* src,
                                        MaskType* mask,
                                        uint64_t increment,
                                        size_t main_offset) {
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
  T dst_mask[kCount];  // 0 ~ kCount -1 : dst;kCount ~ 2 * kCount - 1: mask
  float rands[kCount];
  MaskType mask_result[kCount];
  using Rand = phi::funcs::uniform_distribution<float>;
  using Cast = kps::IdentityFunctor<T>;
  int deal_size = BLOCK_NUM_X * kCount;

  size_t fix = idx * kCount;

  auto mask_functor = MaskFunctor<T, float>(1.0f - dropout_prob);
  for (; fix < main_offset; fix += stride) {
    kps::ReadData<T, kCount, 1, false>(&dst_mask[0], src + fix, deal_size);
    kps::ElementwiseRandom<SType, float, kCount, Rand>(
        &rands[0], Rand(), &state);
    // dst
    kps::OperatorBinary<float, T, MaskFunctor<T, float>>(
        &dst_mask[0], &rands[0], mask_functor, kCount);

    // mask
    kps::ElementwiseUnary<T, MaskType, kCount, 1, Cast>(
        &mask_result[0], &dst_mask[0], Cast());
    kps::WriteData<MaskType, kCount, 1, false>(
        mask + fix, &mask_result[0], deal_size);
    if (fix > idx * kCount + 1) {
      __syncthreads();
    }
  }
  int remainder = n - fix;
  if (remainder > 0) {
    kps::ReadData<T, kCount, 1, true>(&dst_mask[0], src + fix, remainder);
    kps::ElementwiseRandom<SType, float, kCount, Rand>(
        &rands[0], Rand(), &state);
    // dst
    kps::OperatorBinary<float, T, MaskFunctor<T, float>>(
        &dst_mask[0], &rands[0], mask_functor, kCount);
    // mask
    kps::ElementwiseUnary<T, MaskType, kCount, 1, Cast>(
        &mask_result[0], &dst_mask[0], Cast());
    kps::WriteData<MaskType, kCount, 1, true>(
        mask + fix, &mask_result[0], remainder);
    __syncthreads();
  }
}

inline void CalcBroadcastedMask(const phi::GPUContext& dev_ctx,
                                const phi::DenseTensor& mask,
                                phi::DenseTensor* broadcasted_mask) {
  // The broadcast of mask can be combined to the following ElementwiseKernel
  // when the BroadcastKernel supports different input types.
  broadcasted_mask->mutable_data<uint8_t>(dev_ctx.GetPlace());

  std::vector<const phi::DenseTensor*> ins = {&mask};
  std::vector<phi::DenseTensor*> outs = {broadcasted_mask};
  phi::funcs::BroadcastKernel<phi::ElementwiseType::kUnary, uint8_t, uint8_t>(
      dev_ctx, ins, &outs, -1, kps::IdentityFunctor<uint8_t>());
}

template <typename T, typename MT>
void ScaleByDropoutFactor(const phi::GPUContext& dev_ctx,
                          const phi::DenseTensor& x,
                          phi::DenseTensor* y,
                          MT factor) {
  std::vector<const phi::DenseTensor*> ins = {&x};
  std::vector<phi::DenseTensor*> outs = {y};
  auto functor = phi::funcs::ScaleFunctor<T>(factor);
  phi::funcs::ElementwiseKernel<T>(dev_ctx, ins, &outs, functor);
}

template <typename T>
void DropoutFwGPUKernelDriver(const phi::GPUContext& dev_ctx,
                              bool is_test,
                              float dropout_prob,
                              bool upscale_in_train,
                              bool is_fix_seed,
                              int seed_val,
                              const phi::DenseTensor& x,
                              const phi::DenseTensor* seed,
                              phi::DenseTensor* mask,
                              phi::DenseTensor* y,
                              bool is_dropout_nd = false) {
  int64_t x_numel = x.numel();
  auto stream = dev_ctx.stream();
  auto* x_data = x.data<T>();
  auto* y_data = y->data<T>();

  if (!is_test && mask) {
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
    // VectorizedRandomGenerator use curand_uniform4, so kVecSize is 4;
    constexpr int kVecSize =
        phi::funcs::uniform_distribution<float>::kReturnsCount;
    auto gpu_config =
        phi::backends::gpu::GetGpuLaunchConfig1D(dev_ctx, x_numel, kVecSize);
    size_t grid_size = gpu_config.GetGridSize();
    size_t block_size = gpu_config.GetBlockSize();

    int64_t device_id = dev_ctx.GetPlace().GetDeviceId();
    const auto& prop = platform::GetDeviceProperties(device_id);
    size_t max_grid_size = prop.maxThreadsPerMultiProcessor *
                           prop.multiProcessorCount / block_size;
    grid_size = std::min(grid_size, max_grid_size);

    auto offset =
        ((x_numel - 1) / (grid_size * block_size * kVecSize) + 1) * kVecSize;
    GetSeedDataAndIncrement(
        dev_ctx, seed, is_fix_seed, seed_val, offset, &seed_data, &increment);
    size_t main_offset =
        size / (block_size * kVecSize) * (block_size * kVecSize);

    if (is_dropout_nd) {
      VectorizedGeneratorMask<T, uint8_t>
          <<<grid_size, block_size, 0, stream>>>(size,
                                                 seed_data,
                                                 dropout_prob,
                                                 x_data,
                                                 mask_data,
                                                 increment,
                                                 main_offset);

      phi::DenseTensor broadcasted_mask;
      broadcasted_mask.Resize(x.dims());
      CalcBroadcastedMask(dev_ctx, *mask, &broadcasted_mask);

      auto dst_functor = DstFunctor<T, uint8_t>(
          1.0f - dropout_prob, upscale_in_train, x_numel);
      std::vector<const phi::DenseTensor*> ins = {&x, &broadcasted_mask};
      std::vector<phi::DenseTensor*> outs = {y};
      phi::funcs::ElementwiseKernel<T>(dev_ctx, ins, &outs, dst_functor);
    } else {
#define PD_DROPOUT_KERNEL_NAME VectorizedRandomGenerator<T, uint8_t>
      PD_RECORD_CUDA_GRAPH_RANDOM_KERNEL(!is_fix_seed,
                                         PD_DROPOUT_KERNEL_NAME,
                                         grid_size,
                                         block_size,
                                         0,
                                         stream,
                                         offset,
                                         KERNEL_PARAMS.As<uint64_t>(1),
                                         KERNEL_PARAMS.As<uint64_t>(7),
                                         size,
                                         seed_data,
                                         dropout_prob,
                                         x_data,
                                         mask_data,
                                         y_data,
                                         upscale_in_train,
                                         increment,
                                         main_offset);
#undef PD_DROPOUT_KERNEL_NAME
    }
  } else {
    if (upscale_in_train) {
      // y = x
      framework::TensorCopy(x, dev_ctx.GetPlace(), dev_ctx, y);
    } else {
      using MT = typename details::MPTypeTrait<T>::Type;
      MT factor = static_cast<MT>(1.0f - dropout_prob);
      // y = factor * x
      ScaleByDropoutFactor<T, MT>(dev_ctx, x, y, factor);
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

template <typename T>
void DropoutGradGPUKernelDriver(const phi::GPUContext& dev_ctx,
                                bool is_test,
                                float dropout_prob,
                                bool upscale_in_train,
                                const phi::DenseTensor& grad_y,
                                const phi::DenseTensor& mask,
                                phi::DenseTensor* grad_x,
                                bool is_dropout_nd = false) {
  using MT = typename details::MPTypeTrait<T>::Type;

  auto stream = dev_ctx.stream();
  if (is_test) {
    MT factor = static_cast<MT>(upscale_in_train ? 1.0f : 1.0f - dropout_prob);
    // y = factor * x
    ScaleByDropoutFactor<T, MT>(dev_ctx, grad_y, grad_x, factor);
  } else {
    phi::DenseTensor broadcasted_mask;
    if (is_dropout_nd) {
      broadcasted_mask.Resize(grad_y.dims());
      CalcBroadcastedMask(dev_ctx, mask, &broadcasted_mask);
    }

    std::vector<const phi::DenseTensor*> ins = {
        &grad_y, is_dropout_nd ? &broadcasted_mask : &mask};
    std::vector<phi::DenseTensor*> outs = {grad_x};
    if (upscale_in_train) {
      if (dropout_prob == 1.0f) {
#ifdef PADDLE_WITH_HIP
        hipMemset(grad_x->data<T>(), 0, grad_x->numel() * sizeof(T));
#else
        cudaMemset(grad_x->data<T>(), 0, grad_x->numel() * sizeof(T));
#endif
      } else {
        MT factor = static_cast<MT>(1.0f / (1.0f - dropout_prob));
        phi::funcs::ElementwiseKernel<T>(
            dev_ctx, ins, &outs, CudaDropoutGradFunctor<T, uint8_t>(factor));
      }
    } else {
      MT factor = static_cast<MT>(1.0f);
      phi::funcs::ElementwiseKernel<T>(
          dev_ctx, ins, &outs, CudaDropoutGradFunctor<T, uint8_t>(factor));
    }
  }
}

}  // namespace operators
}  // namespace paddle
