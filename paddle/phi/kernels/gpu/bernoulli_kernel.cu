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

#include "paddle/phi/kernels/bernoulli_kernel.h"

#ifdef __NVCC__
#include <curand_kernel.h>
#endif
#ifdef __HIPCC__
#include <hiprand_kernel.h>
#endif

#include <algorithm>
#include <vector>

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/backends/gpu/gpu_launch_config.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/distribution_helper.h"

namespace phi {

template <typename T>
__device__ T convert_to_T(float rand_value, T x_value);

template <>
__device__ phi::dtype::float16 convert_to_T<phi::dtype::float16>(float rand_value, phi::dtype::float16 x_value) {
  return static_cast<phi::dtype::float16>(rand_value <= static_cast<float>(x_value));
}

template <>
__device__ phi::dtype::bfloat16 convert_to_T<phi::dtype::bfloat16>(float rand_value, phi::dtype::bfloat16 x_value) {
  return static_cast<phi::dtype::bfloat16>(rand_value <= static_cast<float>(x_value));
}

template <>
__device__ float convert_to_T<float>(float rand_value, float x_value) {
  return static_cast<float>(rand_value <= x_value);
}

template <>
__device__ double convert_to_T<double>(float rand_value, double x_value) {
  return static_cast<double>(rand_value <= x_value);
}

// 'curand_uniform4/hiprand_uniform4' generate 4 random number each time
template <typename T>
__global__ void bernoulli_cuda_kernel(
    size_t size, uint64_t seed, uint64_t offset, const T* x_data, T* out_data) {
  size_t thread_idx =
      static_cast<size_t>(blockIdx.x * blockDim.x + threadIdx.x);

#if defined(__NVCC__)
  curandStatePhilox4_32_10_t state;
  curand_init(seed, thread_idx, offset, &state);
#else
  hiprandStatePhilox4_32_10_t state;
  hiprand_init(seed, thread_idx, offset, &state);
#endif

  size_t total_thread = gridDim.x * blockDim.x;
  for (size_t i = 4 * thread_idx; i < size; i += total_thread * 4) {
    funcs::uniform_distribution<float> dist;
    float4 rand = dist(&state);
#pragma unroll
    for (size_t j = 0; j < 4; j++) {
      size_t idx = i + j;
      if (idx < size) {
        out_data[idx] = convert_to_T<T>((&rand.x)[j], x_data[idx]);
      }
    }
  }
}

template <typename T, typename Context>
void BernoulliKernel(const Context& ctx,
                     const DenseTensor& x,
                     DenseTensor* out) {
  const T* x_data = x.data<T>();
  T* out_data = ctx.template Alloc<T>(out);
  auto numel = x.numel();

  auto gen_cuda = ctx.GetGenerator();

  auto seed_offset = gen_cuda->IncrementOffset(12);
  uint64_t seed = seed_offset.first;
  uint64_t offset = seed_offset.second;

  auto gpu_config = phi::backends::gpu::GetGpuLaunchConfig1D(ctx, numel, 4);
  size_t grid_size = gpu_config.GetGridSize();
  size_t block_size = gpu_config.GetBlockSize();

  bernoulli_cuda_kernel<<<grid_size, block_size, 0, ctx.stream()>>>(
      numel, seed, offset, x_data, out_data);
}

}  // namespace phi

PD_REGISTER_KERNEL(bernoulli,
                   GPU,
                   ALL_LAYOUT,
                   phi::BernoulliKernel,
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
                   float,
                   double) {}
