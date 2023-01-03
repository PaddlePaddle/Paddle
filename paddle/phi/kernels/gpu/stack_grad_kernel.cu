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

template <typename T, typename IndexT>
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

  IndexT size = pre_dim_size * split_dim_size * suf_dim_size;
  IndexT each_dim_size = split_dim_size / num_split;

  for (IndexT offset = blockIdx.x * blockDim.x + threadIdx.x; offset < size;
       offset += blockDim.x * gridDim.x) {
    IndexT i = offset / (split_dim_size * suf_dim_size);
    IndexT j = (offset % (split_dim_size * suf_dim_size)) / suf_dim_size;
    IndexT k = offset % suf_dim_size;

    T* output = output_ptrs[j / each_dim_size];
    if (output == nullptr) {
      return;
    }
    IndexT output_ind = i * each_dim_size * suf_dim_size +
                         (j % each_dim_size) * suf_dim_size + k;
    *(output + output_ind) = input[offset];
  }
}

template <typename T, typename IndexT>
__global__ void StackGradKernelForLastDim(const T* __restrict__ in_data,
                                          const IndexT cols,
                                          const IndexT rows,
                                          const IndexT tile_num_y,
                                          T** out_datas) {
  constexpr int buffer_size = 512;
  __shared__ T s_buf[buffer_size];

  for (IndexT tile_y = blockIdx.y; tile_y < tile_num_y; tile_y += gridDim.y) {
    IndexT row_idx = tile_y * blockDim.y + threadIdx.y;
    IndexT col_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (col_idx < cols && row_idx < rows) {
      int s_idx = threadIdx.x * blockDim.y + threadIdx.y;
      T data = in_data[row_idx * cols + col_idx];
      s_buf[s_idx] = data;
      __syncthreads();
      
      int out_idx = threadIdx.x + blockIdx.x * blockDim.x;
      if (out_datas[threadIdx.x] == nullptr) {
        out_datas[out_idx][row_idx] = s_buf[s_idx];
      }
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
  int num = dy_dims[axis];
  PADDLE_ENFORCE_EQ(num,
                    x_grad.size(),
                    phi::errors::InvalidArgument(
                    "Output x_grad size should be equal to num, but"
                    " received num is:%d x_grad size is:%d.",
                    num,
                    x_grad.size()));
  auto dy_data = out.data<T>();
  bool use_int32 = out.numel() < std::numeric_limits<int32_t>::max();

  // x_grad is output, so save each data address, then copy each dy into dx_data
  std::vector<T*> outputs(num);
  for (size_t j = 0; j < x_grad.size(); ++j) {
    if (x_grad[j] == nullptr || x_grad[j]->numel() == 0UL) {
      outputs[j] = nullptr;
    } else {
      outputs[j] = dev_ctx.template Alloc<T>(x_grad[j]);
    }
  }

  int64_t dy_pre = 1;
  for (int i = 0; i < axis; ++i) {
    dy_pre *= dy_dims[i];
  }
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

  if (axis == (dy_dims.size() - 1)) {
    constexpr int threads = 512;
    int tid_x = std::min(kMaxOut, num);
    int tid_y = tid_x < kMaxOut ? threads / backends::gpu::RoundToNextHighPowOfTwo(tid_x) : kWarpSize;
    tid_y = std::min(static_cast<int>(backends::gpu::RoundToNextHighPowOfTwo(dy_pre)), tid_y);
    int bid_x = num > kMaxOut ? backends::gpu::DivUp<int>(num, kMaxOut): 1;
    int tile_y_num = backends::gpu::DivUp<int>(dy_pre, tid_y);
    int bid_y = std::min(tile_y_num, backends::gpu::kMultiDimslimit);
    printf("tid_x = %d\t, tid_y = %d\n", tid_x, tid_y);
    printf("bid_x = %d\t, bid_y = %d\n", bid_x, bid_y);
    dim3 blocks(tid_x, tid_y, 1);
    dim3 grids(bid_x, bid_y, 1);
    
    if (use_int32) {
      StackGradKernelForLastDim<T, int32_t>
          <<<blocks, grids, 0, dev_ctx.stream()>>>(dy_data, num, dy_pre, 
            tile_y_num, reinterpret_cast<T**>(tmp_out_data->ptr()));
    } else {
      StackGradKernelForLastDim<T, int64_t>
          <<<blocks, grids, 0, dev_ctx.stream()>>>(dy_data, num, dy_pre, 
            tile_y_num, reinterpret_cast<T**>(tmp_out_data->ptr()));
    }
    return;
  } else {
    int64_t dy_rest = out.numel() / (num * dy_pre);
  }

  // each x_grad should have same shape
  int dy_suf = 1;
  int split_dim = num;
  dy_suf = out.numel() / (split_dim * dy_pre);

  auto config = backends::gpu::GetGpuLaunchConfig1D(
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
