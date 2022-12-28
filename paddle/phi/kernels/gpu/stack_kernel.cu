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
#include "paddle/phi/kernels/funcs/fast_divmod.h"
#include "paddle/phi/kernels/funcs/pointer_array.h"

namespace phi {

template <typename IndexT>
struct DivmodWarpper {
 public:
  explicit DivmodWarpper(IndexT d) { divmoder = phi::funcs::FastDivMod(d); }
  __device__ inline phi::funcs::FastDivMod::DivModT div_mod(IndexT val) {
    return divmoder.Divmod(val);
  }

 private:
  phi::funcs::FastDivMod divmoder;
};

template <>
struct DivmodWarpper<int64_t> {
 public:
  using DivModT = phi::AlignedVector<int64_t, 2>;

  explicit DivmodWarpper(int64_t d) { divisor = d; }
  __device__ inline DivModT div_mod(int64_t val) {
    DivModT data;
    data[0] = val / divisor;
    data[1] = val - data[0] * divisor;
    return data;
  }

 private:
  int64_t divisor;
};

template <typename T, typename IndexT, funcs::SegmentedArraySize Size>
__global__ void StackCUDAKernel(funcs::ConstPointerArray<T, Size> array,
                                DivmodWarpper<IndexT> divmoder,
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
    const T* input_ptr = array.data[divmod_rslt[0]];
#pragma unroll
    for (; grid_y < rows; grid_y += grid_y_stride) {
      output[grid_y * cols + grid_x] =
          input_ptr[grid_y * split_size + divmod_rslt[1]];
    }
  }
}

template <typename Context, typename T, typename IndexT>
void LaunchStackKernel(const Context& ctx,
                       const std::vector<const DenseTensor*>& x,
                       const IndexT x_row,
                       const IndexT x_col,
                       const IndexT y_col,
                       DenseTensor* y) {
  T* y_ptr = ctx.template Alloc<T>(y);

  auto config = phi::backends::gpu::GetGpuLaunchConfig2D(ctx, y_col, x_row);
  DivmodWarpper<IndexT> divmoder(x_col);
  switch (funcs::CalcArraySize(x.size())) {
    POINTER_ARRAY_KERNEL_HELPER(
        StackCUDAKernel<T, IndexT, kArraySize>
        <<<config.block_per_grid, config.thread_per_block, 0, ctx.stream()>>>(
            setter.array, divmoder, x_col, x_row, y_col, y_ptr));
  }
}

template <typename T, typename Context>
void StackKernel(const Context& ctx,
                 const std::vector<const DenseTensor*>& x,
                 int axis,
                 DenseTensor* out) {
  if (axis < 0) axis += (x[0]->dims().size() + 1);
  int n = static_cast<int>(x.size());

  // Split x dim from axis to matrix
  int64_t x_row = 1;
  for (int i = 0; i < axis; ++i) {
    x_row *= x[0]->dims()[i];
  }
  int64_t x_col = x[0]->numel() / x_row;
  int64_t out_col = x_col * n;

  if (out->numel() < std::numeric_limits<int32_t>::max()) {
    LaunchStackKernel<Context, T, int32_t>(ctx, x, x_row, x_col, out_col, out);
  } else {
    LaunchStackKernel<Context, T, int64_t>(ctx, x, x_row, x_col, out_col, out);
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(stack,
                   GPU,
                   ALL_LAYOUT,
                   phi::StackKernel,
                   float,
                   double,
                   bool,
                   int64_t,
                   int,
                   uint8_t,
                   int8_t,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}
