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

namespace phi {

constexpr int kWarpSize = 32;
constexpr int kMaxOut = 16;

template <typename T, typename IntType>
__global__ void UnStackHelperCUDAKernel(const T* __restrict__ input,
                                        int pre_dim_size,
                                        int split_dim_size,
                                        int suf_dim_size,
                                        int num_split,
                                        T** output_ptrs) {
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

    T* output = output_ptrs[j / each_dim_size];
    if (output == nullptr) {
      return;
    }
    IntType output_ind = i * each_dim_size * suf_dim_size +
                         (j % each_dim_size) * suf_dim_size + k;
    *(output + output_ind) = input[offset];
  }
}

template <typename T, typename IndexT, typename PointT>
__global__ void StackGradKernelForLastDim(const T* __restrict__ in_data,
                                          const int split_num,
                                          const int num_per_block,
                                          const int num_tile_x,
                                          const IndexT col,
                                          const IndexT row,
                                          PointT out_datas) {
  constexpr int buffer_size = 512;
  __shared__ T s_buffer[buffer_size];
  for (int tile_y = blockIdx.y; tile_y < row; tile_y += gridDim.y) {
    int col_range = (blockIdx.x < num_tile_x)
                        ? kMaxOut
                        : split_num - (kMaxOut * num_tile_x);
    int read_size = kWarpSize * num_per_block;
    if (threadIdx.y < col_range) {
      IndexT row_idx = tile_y * read_size * split_num;
      IndexT col_idx = threadIdx.x + threadIdx.y * blockDim.x;
      T data = in_data[row_idx + col_idx + blockIdx.x * full_stride];
      int s_idx = col_idx s_buffer[s_idx] = data;
    }
    __syncthreads();

    for () {
      out_datas.val[] = s_buffer;
    }
  }
}

template <typename T, typename Context>
void StackGradKernel(const Context& dev_ctx,
                     const DenseTensor& out,
                     int axis,
                     std::vector<DenseTensor*> x_grad) {
  auto& dy_dims = out.dims();
  if (axis < 0) axis += dy_dims.size();
  int n = dy_dims[axis];
  PADDLE_ENFORCE_EQ(n,
                    x_grad.size(),
                    phi::errors::InvalidArgument(
                        "Output x_grad size should be equal to n, but"
                        " received n is:%d x_grad size is:%d.",
                        n,
                        x_grad.size()));
  auto dy_data = out.data<T>();
  bool use_int32 = out->numel() < std::numeric_limits<int32_t>::max();

  // x_grad is output, so save each data address, then copy each dy into dx_data
  std::vector<T*> outputs(n);
  for (size_t j = 0; j < x_grad.size(); ++j) {
    if (x_grad[j] == nullptr) {
      outputs[j] = nullptr;
      continue;
    }
    if (x_grad[j]->numel() != 0UL) {
      T* ptr = dev_ctx.template Alloc<T>(x_grad[j]);
      outputs[j] = ptr;
    } else {
      outputs[j] = nullptr;
    }
  }

  int64_t dy_pre = 1;
  for (int i = 0; i < n; ++i) {
    dy_pre *= dy_dims[i];
  }

  if (axis == (dy_dims.size() - 1)) {
    int tid_y = std::min(kMaxOut, GetLastPow2(n));
    int bid_x = n > kMaxOut ? (n + kMaxOut - 1) / kMaxOut : 1;
    int num_per_block = n <= (kMaxOut >> 1) ? kMaxOut / tid_y : 1;
    int tid_x = num_per_block * kWarpSize;
    int bid_y = (dy_pre + tid_x - 1) / tid_x;
    dim3 blocks(tid_x, tid_y, 1);
    dim3 grids(bid_x, bid_y, 1);
    if (use_int32) {
      StackGradKernelForLastDim<T, int32_t, decltype()>
          <<<blocks, grids, 0, ctx.stream()>>>(dy_data, n, num_per_block,

          )
    } else {
    }
  } else {
    int64_t dy_rest = out.numel() / (n * dy_pre);
  }

  // each x_grad should have same shape
  int dy_pre = 1, dy_suf = 1;
  int split_dim = n;
  for (int i = 0; i < axis; ++i) {
    dy_pre *= dy_dims[i];
  }
  dy_suf = out.numel() / (split_dim * dy_pre);

  auto tmp_out_data = paddle::memory::Alloc(
      dev_ctx.GetPlace(),
      outputs.size() * sizeof(T*),
      phi::Stream(reinterpret_cast<phi::StreamId>(dev_ctx.stream())));
  paddle::memory::Copy(dev_ctx.GetPlace(),
                       tmp_out_data->ptr(),
                       phi::CPUPlace(),
                       reinterpret_cast<void*>(outputs.data()),
                       outputs.size() * sizeof(T*),
                       dev_ctx.stream());

  auto config = phi::backends::gpu::GetGpuLaunchConfig1D(
      dev_ctx, dy_pre * split_dim * dy_suf);

  if (out.numel() < std::numeric_limits<int32_t>::max()) {
    UnStackHelperCUDAKernel<T, int32_t>
        <<<config.block_per_grid.x,
           config.thread_per_block.x,
           0,
           dev_ctx.stream()>>>(dy_data,
                               dy_pre,
                               split_dim,
                               dy_suf,
                               split_dim,
                               reinterpret_cast<T**>(tmp_out_data->ptr()));
  } else {
    UnStackHelperCUDAKernel<T, int64_t>
        <<<config.block_per_grid.x,
           config.thread_per_block.x,
           0,
           dev_ctx.stream()>>>(dy_data,
                               dy_pre,
                               split_dim,
                               dy_suf,
                               split_dim,
                               reinterpret_cast<T**>(tmp_out_data->ptr()));
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(stack_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::StackGradKernel,
                   float,
                   double,
                   int64_t,
                   int,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}
