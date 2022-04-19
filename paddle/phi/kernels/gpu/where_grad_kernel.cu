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

#include "paddle/phi/kernels/where_grad_kernel.h"

namespace phi {

template <typename T>
__global__ void WhereGradCUDAKernel(
    const int N, const T* dout, const bool* cond, T* dx, T* dy) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  for (; idx < N; idx += blockDim.x * gridDim.x) {
    if (dx != nullptr) {
      dx[idx] = cond[idx] ? dout[idx] : 0.;
    }
    if (dy != nullptr) {
      dy[idx] = cond[idx] ? 0. : dout[idx];
    }
  }
}

template <typename T, typename Context>
void WhereGradKernel(const Context& ctx,
                     const DenseTensor& condition,
                     const DenseTensor& x,
                     const DenseTensor& y,
                     const DenseTensor& out_grad,
                     DenseTensor* x_grad,
                     DenseTensor* y_grad) {
  const bool* cond_data = condition.data<bool>();
  auto numel = condition.numel();
  auto* dout = out_grad.data<T>();

  T* dx = (x_grad != nullptr) ? ctx.template Alloc<T>(x_grad) : nullptr;
  T* dy = (y_grad != nullptr) ? ctx.template Alloc<T>(y_grad) : nullptr;

  auto stream = ctx.stream();
  auto config = backends::gpu::GetGpuLaunchConfig1D(ctx, numel);
  WhereGradCUDAKernel<
      T><<<config.block_per_grid.x, config.thread_per_block.x, 0, stream>>>(
      numel, dout, cond_data, dx, dy);
}

}  // namespace phi

PD_REGISTER_KERNEL(where_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::WhereGradKernel,
                   float,
                   double,
                   int,
                   int64_t) {}
