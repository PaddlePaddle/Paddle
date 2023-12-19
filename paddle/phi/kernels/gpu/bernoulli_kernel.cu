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
#include "paddle/phi/common/amp_type_traits.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/distribution_helper.h"

namespace phi {

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
    using MPType = typename phi::dtype::MPTypeTrait<T>::Type;
#pragma unroll
    for (size_t j = 0; j < 4; j++) {
      size_t idx = i + j;
      if (idx < size) {
        MPType p = static_cast<MPType>(x_data[idx]);
        PADDLE_ENFORCE(p >= 0 && p <= 1,
                       "The probability should be in [0, 1], but got %f",
                       p);
        out_data[idx] = static_cast<T>((&rand.x)[j] <= static_cast<MPType>(p));
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
