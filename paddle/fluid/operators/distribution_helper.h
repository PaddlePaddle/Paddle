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

#ifdef __NVCC__
#include <curand_kernel.h>
#endif
#ifdef __HIPCC__
#include <hiprand_kernel.h>
#endif

#include "paddle/fluid/framework/generator.h"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/platform/device/gpu/gpu_info.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/for_range.h"
#include "paddle/phi/core/hostdevice.h"

#if defined(__NVCC__) || defined(__HIPCC__)
#include "paddle/phi/kernels/primitive/kernel_primitives.h"
#endif

#if !defined(_WIN32)
#define UNLIKELY(condition) __builtin_expect(static_cast<bool>(condition), 0)
#else
// there is no equivalent intrinsics in msvc.
#define UNLIKELY(condition) (condition)
#endif

namespace paddle {
namespace distribution {

using Tensor = framework::Tensor;

/********************* Transformation Function **********************/
template <typename T>
struct exponential_transform {
  explicit exponential_transform(T lambda) : lambda_(lambda) {}

  HOSTDEVICE inline T operator()(T val) const {
#if defined(__NVCC__) || defined(__HIPCC__)
    if (std::is_same<T, double>::value) {
      return static_cast<T>(-1.0) / lambda_ * log(val);
    } else {
      return static_cast<T>(-1.0) / lambda_ * __logf(val);
    }
#else
    return static_cast<T>(-1.0) / lambda_ * std::log(static_cast<T>(1.0) - val);
#endif
  }

 private:
  T lambda_;
};

template <typename T>
struct uniform_transform {
  explicit uniform_transform(T min, T max) : range_(max - min), min_(min) {}

  HOSTDEVICE inline T operator()(T val) const {
    if (UNLIKELY(val == static_cast<T>(1.0))) {
      return min_;
    } else {
      return val * range_ + min_;
    }
  }

 private:
  T range_;
  T min_;
};

template <typename T>
struct normal_transform {
  explicit normal_transform(T mean, T std) : mean_(mean), std_(std) {}

  HOSTDEVICE inline T operator()(T val) const { return val * std_ + mean_; }

 private:
  T mean_;
  T std_;
};

#if defined(__NVCC__) || defined(__HIPCC__)

namespace kps = phi::kps;

/*********************** Distribution Function *************************/
template <typename T>
struct uniform_distribution;

template <typename T>
struct normal_distribution;

#if defined(__NVCC__)
template <>
struct uniform_distribution<float> {
  __device__ inline float4 operator()(curandStatePhilox4_32_10_t *state) const {
    return curand_uniform4(state);
  }
  static constexpr int kReturnsCount = 4;
};

template <>
struct uniform_distribution<double> {
  __device__ inline double2 operator()(
      curandStatePhilox4_32_10_t *state) const {
    return curand_uniform2_double(state);
  }
  static constexpr int kReturnsCount = 2;
};

template <>
struct normal_distribution<float> {
  __device__ inline float4 operator()(curandStatePhilox4_32_10_t *state) const {
    return curand_normal4(state);
  }
  static constexpr int kReturnsCount = 4;
};

template <>
struct normal_distribution<double> {
  __device__ inline double2 operator()(
      curandStatePhilox4_32_10_t *state) const {
    return curand_normal2_double(state);
  }
  static constexpr int kReturnsCount = 2;
};

#else
template <>
struct uniform_distribution<float> {
  __device__ inline float4 operator()(
      hiprandStatePhilox4_32_10_t *state) const {
    return hiprand_uniform4(state);
  }
  static constexpr int kReturnsCount = 4;
};

template <>
struct uniform_distribution<double> {
  __device__ inline double2 operator()(
      hiprandStatePhilox4_32_10_t *state) const {
    return hiprand_uniform2_double(state);
  }
  static constexpr int kReturnsCount = 2;
};

template <>
struct normal_distribution<float> {
  __device__ inline float4 operator()(
      hiprandStatePhilox4_32_10_t *state) const {
    return hiprand_normal4(state);
  }
  static constexpr int kReturnsCount = 4;
};

template <>
struct normal_distribution<double> {
  __device__ inline double2 operator()(
      hiprandStatePhilox4_32_10_t *state) const {
    return hiprand_normal2_double(state);
  }
  static constexpr int kReturnsCount = 2;
};
#endif

/******** Launch GPU function of distribution and transformation *********/
template <typename T, typename DistOp, typename TransformOp>
__global__ void DistributionKernel(size_t size, uint64_t seed, uint64_t offset,
                                   DistOp dist, TransformOp trans, T *out_data,
                                   size_t stride) {
  size_t idx = static_cast<size_t>(BLOCK_ID_X * BLOCK_NUM_X);
  static constexpr int kCount = DistOp::kReturnsCount;
#if defined(__NVCC__)
  curandStatePhilox4_32_10_t state;
  curand_init(seed, idx + THREAD_ID_X, offset, &state);
  using SType = curandStatePhilox4_32_10_t;
#else
  hiprandStatePhilox4_32_10_t state;
  hiprand_init(seed, idx + THREAD_ID_X, offset, &state);
  using SType = hiprandStatePhilox4_32_10_t;
#endif
  size_t total_thread = GRID_NUM_X * BLOCK_NUM_X;
  T args[kCount];
  T result[kCount];
  for (size_t i = idx; i < size; i += total_thread * kCount) {
    kps::ElementwiseRandom<SType, T, kCount, 1, DistOp>(&args[0], dist, &state);
    kps::ElementwiseUnary<T, T, kCount, 1, 1, TransformOp>(&result[0], &args[0],
                                                           trans);
    kps::WriteData<T, T, kCount, 1, 1, true>(out_data + i, &result[0], size - i,
                                             1, stride, 1);
    __syncthreads();
  }
}

template <typename T, typename DistOp, typename TransformOp>
void distribution_and_transform(const platform::CUDADeviceContext &dev_ctx,
                                Tensor *out, DistOp dist, TransformOp trans) {
  T *out_data = out->mutable_data<T>(dev_ctx.GetPlace());
  auto size = out->numel();

  int64_t device_id = dev_ctx.GetPlace().GetDeviceId();
  auto gen_cuda = framework::GetDefaultCUDAGenerator(device_id);

  size_t block_size = 256;
  size_t expect_grid_size = (size + block_size - 1) / block_size;
  const auto &prop = platform::GetDeviceProperties(device_id);
  size_t max_grid_size = (prop.maxThreadsPerMultiProcessor / block_size) *
                         prop.multiProcessorCount;
  size_t grid_size =
      expect_grid_size > max_grid_size ? max_grid_size : expect_grid_size;

  size_t total_thread = block_size * grid_size;
  size_t curand4_loop_times =
      (size + 4 * total_thread - 1) / (4 * total_thread);
  // 'increment' shoulde be multiple of 4
  uint64_t increment = curand4_loop_times * 4;

  auto seed_offset = gen_cuda->IncrementOffset(increment);
  uint64_t seed = seed_offset.first;
  uint64_t offset = seed_offset.second;

  DistributionKernel<
      T, DistOp, TransformOp><<<grid_size, block_size, 0, dev_ctx.stream()>>>(
      size, seed, offset, dist, trans, out_data, total_thread);
}

#endif

}  // namespace distribution
}  // namespace paddle
