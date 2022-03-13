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

#include "paddle/phi/kernels/stack_kernel.h"

#include "paddle/fluid/memory/memory.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/backends/gpu/gpu_launch_config.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T, typename IntType>
__global__ void StackCUDAKernel(T** input_ptrs,
                                int split_size,
                                int rows,
                                int cols,
                                T* __restrict__ output) {
  IntType grid_x = blockIdx.x * blockDim.x + threadIdx.x;

  for (; grid_x < cols; grid_x += blockDim.x * gridDim.x) {
    IntType grid_y = blockIdx.y * blockDim.y + threadIdx.y;

    IntType split = grid_x / split_size;
    const T* input_ptr = input_ptrs[split];
    IntType col_offset = grid_x % split_size;
#pragma unroll
    for (; grid_y < rows; grid_y += blockDim.y * gridDim.y) {
      output[grid_y * cols + grid_x] =
          input_ptr[grid_y * split_size + col_offset];
    }
  }
}

template <typename T, typename Context>
void StackKernel(const Context& dev_ctx,
                 const std::vector<const DenseTensor*>& x,
                 int axis,
                 DenseTensor* out) {
  if (axis < 0) axis += (x[0]->dims().size() + 1);

  int n = static_cast<int>(x.size());
  T* y_data = dev_ctx.template Alloc<T>(out);
  std::vector<const T*> x_datas(n);
  for (int i = 0; i < n; i++) {
    x_datas[i] = x[i]->data<T>();
  }

  auto tmp_x_data = paddle::memory::Alloc(dev_ctx, x_datas.size() * sizeof(T*));
  paddle::memory::Copy(dev_ctx.GetPlace(),
                       tmp_x_data->ptr(),
                       phi::CPUPlace(),
                       reinterpret_cast<void*>(x_datas.data()),
                       x_datas.size() * sizeof(T*),
                       dev_ctx.stream());

  // Split x dim from axis to matrix
  int x_row = 1, x_col = 1;
  for (int i = 0; i < axis; ++i) {
    x_row *= x[0]->dims()[i];
  }
  x_col = x[0]->numel() / x_row;
  int out_col = x_col * n;

  auto config =
      phi::backends::gpu::GetGpuLaunchConfig2D(dev_ctx, out_col, x_row);

  if (out->numel() < std::numeric_limits<int32_t>::max()) {
    StackCUDAKernel<T, int32_t><<<config.block_per_grid,
                                  config.thread_per_block,
                                  0,
                                  dev_ctx.stream()>>>(
        reinterpret_cast<T**>(tmp_x_data->ptr()),
        x_col,
        x_row,
        out_col,
        y_data);
  } else {
    StackCUDAKernel<T, int64_t><<<config.block_per_grid,
                                  config.thread_per_block,
                                  0,
                                  dev_ctx.stream()>>>(
        reinterpret_cast<T**>(tmp_x_data->ptr()),
        x_col,
        x_row,
        out_col,
        y_data);
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(stack,
                   GPU,
                   ALL_LAYOUT,
                   phi::StackKernel,
                   float,
                   double,
                   int64_t,
                   int,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}
