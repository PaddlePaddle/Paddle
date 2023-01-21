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
#include "paddle/phi/backends/gpu/gpu_launch_config.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/segmented_array.h"

namespace phi {

template <typename T, typename IndexT, typename ArrayT>
__global__ void UnStackCudaKernel(const T* __restrict__ input,
                                  IndexT pre_dim_size,
                                  IndexT split_dim_size,
                                  IndexT suf_dim_size,
                                  IndexT num_split,
                                  ArrayT array) {
  assert(blockDim.y == 1);
  assert(blockDim.z == 1);
  // In this case they are equal
  assert(split_dim_size % num_split == 0);

  IndexT size = pre_dim_size * split_dim_size * suf_dim_size;
  IndexT each_dim_size = split_dim_size / num_split;

  for (IndexT offset = blockIdx.x * blockDim.x + threadIdx.x; offset < size;
       offset += blockDim.x * gridDim.x) {
    IndexT i = offset / (split_dim_size * suf_dim_size);
    IndexT j = (offset % (split_dim_size * suf_dim_size)) / suf_dim_size;
    IndexT k = offset % suf_dim_size;

    T* output = array.data[j / each_dim_size];
    if (output == nullptr) {
      return;
    }
    IndexT output_ind = i * each_dim_size * suf_dim_size +
                        (j % each_dim_size) * suf_dim_size + k;
    *(output + output_ind) = input[offset];
  }
}

template <typename T, typename IndexT, typename ArrayT>
__global__ void UnStackCudaKernelForLastDim(const T* __restrict__ in_data,
                                            const IndexT cols,
                                            const IndexT rows,
                                            const IndexT tile_x_num,
                                            ArrayT array) {
  constexpr int buffer_size = 512;
  __shared__ T s_buf[buffer_size];

  for (IndexT tile_x = blockIdx.x; tile_x < tile_x_num; tile_x += gridDim.x) {
    IndexT row_idx = tile_x * blockDim.x + threadIdx.x;
    IndexT col_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int s_idx = threadIdx.y * blockDim.x + threadIdx.x;
    bool is_valid = (col_idx < cols && row_idx < rows);

    if (is_valid) {
      T data = in_data[row_idx * cols + col_idx];
      s_buf[s_idx] = data;
    }
    __syncthreads();
    if (is_valid) {
      if (array.data[col_idx]) {
        array.data[col_idx][row_idx] = s_buf[s_idx];
      }
    }
  }
}

template <typename Context,
          typename T,
          typename IndexT,
          funcs::SegmentedArraySize Size>
void LaunchUnStackKernel(const Context& ctx,
                         const IndexT pre_dim,
                         const IndexT split_dim,
                         const IndexT suf_dim,
                         const IndexT num_splits,
                         const DenseTensor& out_grad,
                         std::vector<DenseTensor*>* x_grad) {
  // each x_grad should have same shape
  auto dout_ptr = out_grad.data<T>();
  funcs::PointerArraySetter<Context, T, Size> setter(ctx, x_grad);

  if (suf_dim == 1) {
    // For the case axis == (out_grad.dims().size() - 1)
    constexpr int kThreads = 512;
    constexpr int kWarpSize = 32;
    constexpr int kMaxOut = 16;

    int tid_x = 0, tid_y = 0, bid_x = 0, bid_y = 1;
    if (split_dim < kMaxOut) {
      tid_y = split_dim;
      tid_x =
          std::min(backends::gpu::RoundToNextHighPowOfTwo(pre_dim, kWarpSize),
                   kThreads / backends::gpu::RoundToNextHighPowOfTwo(tid_y));
    } else {
      tid_y = kMaxOut;
      tid_x = kWarpSize;
      bid_y = backends::gpu::DivUp<int>(split_dim, kMaxOut);
    }
    int tile_x_num = backends::gpu::DivUp<int>(pre_dim, tid_x);
    bid_x = std::min(tile_x_num, backends::gpu::kMultiDimslimit);
    dim3 blocks(tid_x, tid_y, 1);
    dim3 grids(bid_x, bid_y, 1);

    UnStackCudaKernelForLastDim<T, IndexT, decltype(setter.array)>
        <<<grids, blocks, 0, ctx.stream()>>>(
            dout_ptr, split_dim, pre_dim, tile_x_num, setter.array);
  } else {
    auto config = phi::backends::gpu::GetGpuLaunchConfig1D(
        ctx, pre_dim * split_dim * suf_dim);

    UnStackCudaKernel<T, IndexT, decltype(setter.array)>
        <<<config.block_per_grid.x,
           config.thread_per_block.x,
           0,
           ctx.stream()>>>(
            dout_ptr, pre_dim, split_dim, suf_dim, num_splits, setter.array);
  }
}

template <typename T, typename Context>
void StackGradKernel(const Context& ctx,
                     const DenseTensor& out_grad,
                     int axis,
                     std::vector<DenseTensor*> x_grad) {
  if (axis < 0) axis += out_grad.dims().size();

  int64_t split_dim = out_grad.dims()[axis];
  PADDLE_ENFORCE_EQ(
      split_dim,
      x_grad.size(),
      phi::errors::InvalidArgument(
          "Output x_grad size should be equal to the split_dim, but"
          " received split_dim is:%d x_grad size is:%d.",
          split_dim,
          x_grad.size()));

  auto dout_dims = out_grad.dims();
  int64_t dout_pre = 1;
  for (int i = 0; i < axis; ++i) {
    dout_pre *= dout_dims[i];
  }
  int64_t dout_suf = out_grad.numel() / (split_dim * dout_pre);

  if (out_grad.numel() < std::numeric_limits<int32_t>::max()) {
    switch (funcs::CalcArraySize(split_dim)) {
      SEGMENTED_ARRAY_KERNEL_HELPER(
          LaunchUnStackKernel<Context, T, int32_t, kArraySize>(ctx,
                                                               dout_pre,
                                                               split_dim,
                                                               dout_suf,
                                                               split_dim,
                                                               out_grad,
                                                               &x_grad));
    }
  } else {
    switch (funcs::CalcArraySize(split_dim)) {
      SEGMENTED_ARRAY_KERNEL_HELPER(
          LaunchUnStackKernel<Context, T, int64_t, kArraySize>(ctx,
                                                               dout_pre,
                                                               split_dim,
                                                               dout_suf,
                                                               split_dim,
                                                               out_grad,
                                                               &x_grad));
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
