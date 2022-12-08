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

constexpr int kWarpperSize = 256;
template <typename T>
struct DataWarpper {
  const T* data[kWarpperSize];
  HOSTDEVICE inline const T* operator[](int i) const { return data[i]; }
};

template <typename T, typename IntType, typename WarpT>
__global__ void StackCUDAKernel(WarpT input_ptrs,
                                IntType split_size,
                                IntType rows,
                                IntType cols,
                                T* __restrict__ output) {
  IntType grid_x = static_cast<IntType>(blockIdx.x) * blockDim.x + threadIdx.x;
  IntType grid_x_stride = static_cast<IntType>(blockDim.x) * gridDim.x;
  IntType grid_y_stride = static_cast<IntType>(blockDim.y) * gridDim.y;

  for (; grid_x < cols; grid_x += grid_x_stride) {
    IntType grid_y =
        static_cast<IntType>(blockIdx.y) * blockDim.y + threadIdx.y;

    IntType split = grid_x / split_size;
    const T* input_ptr = input_ptrs[split];
    IntType col_offset = grid_x % split_size;
#pragma unroll
    for (; grid_y < rows; grid_y += grid_y_stride) {
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
  
  // Split x dim from axis to matrix
  int64_t x_row = 1, x_col = 1;
  for (int i = 0; i < axis; ++i) {
      x_row *= x[0]->dims()[i];
  }
  x_col = x[0]->numel() / x_row;
  int64_t out_col = x_col * n;
  auto config =
      phi::backends::gpu::GetGpuLaunchConfig2D(dev_ctx, out_col, x_row);


#define IMPL_STACK_CUDA_KERNEL(index_t, input_data)       \
    StackCUDAKernel<T, index_t, decltype(input_data)><<<  \
                config.block_per_grid, config.thread_per_block, 0, dev_ctx.stream()>>>( \
                                input_data,                      \
                                static_cast<index_t>(x_col),     \
                                static_cast<index_t>(x_row),     \
                                static_cast<index_t>(out_col),   \
                                y_data);

  if (n <= kWarpperSize) {
    DataWarpper<T> data_warpper;
    for (auto i = 0; i < n; ++i) {
        data_warpper.data[i] = x[i]->data<T>();
    }
    if (out->numel() < std::numeric_limits<int32_t>::max()) {
        IMPL_STACK_CUDA_KERNEL(int32_t, data_warpper);
    } else {
        IMPL_STACK_CUDA_KERNEL(int64_t, data_warpper);
    }
  } else {
    std::vector<const T*> x_datas(n);
    for (int i = 0; i < n; i++) {
      x_datas[i] = x[i]->data<T>();
    }
    auto tmp_x_data = paddle::memory::Alloc(
        dev_ctx.GetPlace(),
        x_datas.size() * sizeof(T*),
        phi::Stream(reinterpret_cast<phi::StreamId>(dev_ctx.stream())));
    paddle::memory::Copy(dev_ctx.GetPlace(),
                         tmp_x_data->ptr(),
                         phi::CPUPlace(),
                         reinterpret_cast<void*>(x_datas.data()),
                         x_datas.size() * sizeof(T*),
                         dev_ctx.stream());

    if (out->numel() < std::numeric_limits<int32_t>::max()) {
        IMPL_STACK_CUDA_KERNEL(int32_t, reinterpret_cast<T**>(tmp_x_data->ptr()));
    } else {
        IMPL_STACK_CUDA_KERNEL(int64_t, reinterpret_cast<T**>(tmp_x_data->ptr()));
    }
  }
#undef IMPL_STACK_CUDA_KERNEL
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
