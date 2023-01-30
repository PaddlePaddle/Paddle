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
<<<<<<< HEAD
#include "paddle/fluid/memory/memory.h"
#include "paddle/phi/backends/gpu/gpu_launch_config.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/segmented_array.h"

namespace phi {

template <typename T, typename IndexT, typename ArrayT>
__global__ void StackCUDAKernel(ArrayT array,
                                funcs::GeneralDivMod<IndexT> divmoder,
                                IndexT split_size,
                                IndexT rows,
                                IndexT cols,
                                T* __restrict__ output) {
  IndexT grid_x = static_cast<IndexT>(blockIdx.x) * blockDim.x + threadIdx.x;
  IndexT grid_x_stride = static_cast<IndexT>(blockDim.x) * gridDim.x;
  IndexT grid_y_stride = static_cast<IndexT>(blockDim.y) * gridDim.y;

  for (; grid_x < cols; grid_x += grid_x_stride) {
    IndexT grid_y = static_cast<IndexT>(blockIdx.y) * blockDim.y + threadIdx.y;

    auto divmod_rslt = divmoder.div_mod(grid_x);
    IndexT split = divmod_rslt[0];       // grid_x / split_size
    IndexT col_offset = divmod_rslt[1];  // grid_x % split_size
    const T* input_ptr = array.data[split];
=======

#include "paddle/fluid/memory/memory.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/backends/gpu/gpu_launch_config.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T, typename IntType>
__global__ void StackCUDAKernel(T** input_ptrs,
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
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
#pragma unroll
    for (; grid_y < rows; grid_y += grid_y_stride) {
      output[grid_y * cols + grid_x] =
          input_ptr[grid_y * split_size + col_offset];
    }
  }
}

<<<<<<< HEAD
template <typename Context,
          typename T,
          typename IndexT,
          funcs::SegmentedArraySize Size>
void LaunchStackKernel(const Context& ctx,
                       const IndexT x_col,
                       const IndexT x_row,
                       const IndexT out_col,
                       const std::vector<const DenseTensor*>& x,
                       DenseTensor* out) {
  T* out_ptr = ctx.template Alloc<T>(out);
  auto config = phi::backends::gpu::GetGpuLaunchConfig2D(ctx, out_col, x_row);

  funcs::ConstPointerArraySetter<Context, T, Size> setter(ctx, x);
  funcs::GeneralDivMod<IndexT> divmoder(x_col);
  StackCUDAKernel<T, IndexT, decltype(setter.array)>
      <<<config.block_per_grid, config.thread_per_block, 0, ctx.stream()>>>(
          setter.array, divmoder, x_col, x_row, out_col, out_ptr);
}

template <typename T, typename Context>
void StackKernel(const Context& ctx,
=======
template <typename T, typename Context>
void StackKernel(const Context& dev_ctx,
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
                 const std::vector<const DenseTensor*>& x,
                 int axis,
                 DenseTensor* out) {
  if (axis < 0) axis += (x[0]->dims().size() + 1);
<<<<<<< HEAD
  int num = static_cast<int>(x.size());

  // Split x dim from axis to matrix
  int64_t x_row = 1;
  for (int i = 0; i < axis; ++i) {
    x_row *= x[0]->dims()[i];
  }
  int64_t x_col = x[0]->numel() / x_row;
  int64_t out_col = x_col * num;

  if (out->numel() < std::numeric_limits<int32_t>::max()) {
    switch (funcs::CalcArraySize(num)) {
      SEGMENTED_ARRAY_KERNEL_HELPER(
          LaunchStackKernel<Context, T, int32_t, kArraySize>(
              ctx, x_col, x_row, out_col, x, out));
    }
  } else {
    switch (funcs::CalcArraySize(num)) {
      SEGMENTED_ARRAY_KERNEL_HELPER(
          LaunchStackKernel<Context, T, int64_t, kArraySize>(
              ctx, x_col, x_row, out_col, x, out));
    }
=======

  int n = static_cast<int>(x.size());
  T* y_data = dev_ctx.template Alloc<T>(out);
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

  // Split x dim from axis to matrix
  int64_t x_row = 1, x_col = 1;
  for (int i = 0; i < axis; ++i) {
    x_row *= x[0]->dims()[i];
  }
  x_col = x[0]->numel() / x_row;
  int64_t out_col = x_col * n;

  auto config =
      phi::backends::gpu::GetGpuLaunchConfig2D(dev_ctx, out_col, x_row);

  if (out->numel() < std::numeric_limits<int32_t>::max()) {
    StackCUDAKernel<T, int32_t>
        <<<config.block_per_grid,
           config.thread_per_block,
           0,
           dev_ctx.stream()>>>(reinterpret_cast<T**>(tmp_x_data->ptr()),
                               static_cast<int32_t>(x_col),
                               static_cast<int32_t>(x_row),
                               static_cast<int32_t>(out_col),
                               y_data);
  } else {
    StackCUDAKernel<T, int64_t>
        <<<config.block_per_grid,
           config.thread_per_block,
           0,
           dev_ctx.stream()>>>(reinterpret_cast<T**>(tmp_x_data->ptr()),
                               x_col,
                               x_row,
                               out_col,
                               y_data);
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(stack,
                   GPU,
                   ALL_LAYOUT,
                   phi::StackKernel,
                   float,
                   double,
<<<<<<< HEAD
                   bool,
                   int64_t,
                   int,
                   uint8_t,
                   int8_t,
=======
                   int64_t,
                   int,
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}
