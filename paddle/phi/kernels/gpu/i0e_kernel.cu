/* Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/backends/gpu/gpu_launch_config.h"
#include "paddle/phi/core/kernel_registry.h"

#include "paddle/phi/kernels/i0e_kernel.h"
#include "paddle/phi/kernels/gpu/bessel_utils.h"

namespace phi {

template <typename T, typename Context>
void I0eKernel(const Context& ctx, const DenseTensor& x, DenseTensor* out) {
  const T* x_data = x.data<T>();
  T* out_data = ctx.template Alloc<T>(out);
  const int size = x.numel();
  const int kMaxBlockDim = 256;

  int block_size = std::min(kMaxBlockDim, ctx.GetMaxThreadsPerBlock());
  dim3 dim_block(block_size);
  dim3 dim_grid((size + block_size - 1) / block_size);
  phi::backends::gpu::LimitGridDim(ctx, &dim_grid);

  auto gen_cuda = ctx.GetGenerator();
  auto seed_offset = gen_cuda->IncrementOffset(20);
  uint64_t seed = seed_offset.first;
  uint64_t offset = seed_offset.second;
  CalcI0e<<<dim_grid, dim_block>>>(x_data, out_data, size, seed, offset);
}

}  // namespace phi

PD_REGISTER_KERNEL(
    i0e, GPU, ALL_LAYOUT, phi::I0eKernel, float, double) {}
