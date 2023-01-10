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

#include "paddle/phi/kernels/stack_grad_kernel.h"

#include "paddle/fluid/memory/memory.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/backends/gpu/gpu_launch_config.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/segmented_array.h"

namespace phi {

template <typename T, typename IntType, typename ArrayT>
__global__ void UnstackCUDAKernel(const T* __restrict__ input,
                                  int pre_dim_size,
                                  int split_dim_size,
                                  int suf_dim_size,
                                  int num_split,
                                  ArrayT array) {
  assert(blockDim.y == 1);
  assert(blockDim.z == 1);
  // In this case they are equal
  assert(split_dim_size % num_split == 0);

  IntType size = pre_dim_size * split_dim_size * suf_dim_size;
  IntType each_dim_size = split_dim_size / num_split;

  for (IntType offset = blockIdx.x * blockDim.x + threadIdx.x; offset < size;
       offset += blockDim.x * gridDim.x) {
    IntType i = offset / (split_dim_size * suf_dim_size);
    IntType j = (offset % (split_dim_size * suf_dim_size)) / suf_dim_size;
    IntType k = offset % suf_dim_size;

    T* output = array.data[j / each_dim_size];
    if (output == nullptr) {
      return;
    }
    IntType output_ind = i * each_dim_size * suf_dim_size +
                         (j % each_dim_size) * suf_dim_size + k;
    *(output + output_ind) = input[offset];
  }
}

template <typename Context,
          typename T,
          typename IndexT,
          funcs::SegmentedArraySize Size>
void LaunchStackGradKernel(const Context& ctx,
                           const IndexT pre_dim,
                           const IndexT split_dim,
                           const IndexT suf_dim,
                           const IndexT num_splits,
                           const DenseTensor& out,
                           std::vector<DenseTensor*>* x_grad) {
  // each x_grad should have same shape
  auto dout_ptr = out.data<T>();
  auto config = phi::backends::gpu::GetGpuLaunchConfig1D(
      ctx, pre_dim * split_dim * suf_dim);

  funcs::PointerArraySetter<Context, T, Size> setter(ctx, x_grad);
  UnstackCUDAKernel<T, IndexT, decltype(setter.array)>
      <<<config.block_per_grid.x, config.thread_per_block.x, 0, ctx.stream()>>>(
          dout_ptr, pre_dim, split_dim, suf_dim, num_splits, setter.array);
}

template <typename T, typename Context>
void StackGradKernel(const Context& ctx,
                     const DenseTensor& out,
                     int axis,
                     std::vector<DenseTensor*> x_grad) {
  if (axis < 0) axis += out.dims().size();

  int64_t split_dim = out.dims()[axis];
  PADDLE_ENFORCE_EQ(
      split_dim,
      x_grad.size(),
      phi::errors::InvalidArgument(
          "Output x_grad size should be equal to the split_dim, but"
          " received split_dim is:%d x_grad size is:%d.",
          split_dim,
          x_grad.size()));

  auto dout_dims = out.dims();
  int64_t dout_pre = 1;
  for (int i = 0; i < axis; ++i) {
    dout_pre *= dout_dims[i];
  }
  int64_t dout_suf = out.numel() / (split_dim * dout_pre);

  if (out.numel() < std::numeric_limits<int32_t>::max()) {
    switch (funcs::CalcArraySize(split_dim)) {
      POINTER_ARRAY_KERNEL_HELPER(
          LaunchStackGradKernel<Context, T, int32_t, kArraySize>(
              ctx, dout_pre, split_dim, dout_suf, split_dim, out, &x_grad));
    }
  } else {
    switch (funcs::CalcArraySize(split_dim)) {
      POINTER_ARRAY_KERNEL_HELPER(
          LaunchStackGradKernel<Context, T, int64_t, kArraySize>(
              ctx, dout_pre, split_dim, dout_suf, split_dim, out, &x_grad));
    }
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(stack_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::StackGradKernel,
                   float,
                   double,
                   bool,
                   int64_t,
                   int,
                   uint8_t,
                   int8_t,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}
