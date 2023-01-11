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

namespace phi {

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
                                          const IndexT tile_x_num,
                                          T** out_datas) {
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
      if (out_datas[col_idx] != nullptr) {
        out_datas[col_idx][row_idx] = s_buf[s_idx];
      }
    }
  }
}

template <typename Context, typename T, typename IndexT>
void LaunchStackGradCUDAKernel(const Context& ctx,
                               const DenseTensor& out,
                               std::vector<DenseTensor*>* x_grad_ptr,
                               const int axis,
                               const int64_t dy_pre) {
  auto x_grad = *x_grad_ptr;
  int out_num = out.dims()[axis];
  PADDLE_ENFORCE_EQ(
      out_num,
      x_grad.size(),
      phi::errors::InvalidArgument(
          "Output x_grad size shall be equal to output num, but output num "
          "received in stack_grad op is:%d, and x_grad size is:%d.",
          out_num,
          x_grad.size()));
  std::vector<T*> outputs(out_num);
  for (size_t j = 0; j < out_num; ++j) {
    if (x_grad[j] == nullptr || x_grad[j]->numel() == 0UL) {
      outputs[j] = nullptr;
    } else {
      outputs[j] = ctx.template Alloc<T>(x_grad[j]);
    }
  }

  auto tmp_out_data = paddle::memory::Alloc(
      ctx.GetPlace(),
      out_num * sizeof(T*),
      phi::Stream(reinterpret_cast<phi::StreamId>(ctx.stream())));
  paddle::memory::Copy(ctx.GetPlace(),
                       tmp_out_data->ptr(),
                       phi::CPUPlace(),
                       reinterpret_cast<void*>(outputs.data()),
                       out_num * sizeof(T*),
                       ctx.stream());

  if (axis == (out.dims().size() - 1)) {
    constexpr int kThreads = 512;
    constexpr int kWarpSize = 32;
    constexpr int kMaxOut = 16;
    int tid_x = 0, tid_y = 0, bid_x = 0, bid_y = 1;
    bool is_small_num = out_num < kMaxOut;

    if (is_small_num) {
      tid_y = out_num;
      tid_x =
          std::min(backends::gpu::RoundToNextHighPowOfTwo(dy_pre, kWarpSize),
                   kThreads / backends::gpu::RoundToNextHighPowOfTwo(tid_y));
    } else {
      tid_y = kMaxOut;
      tid_x = kWarpSize;
      bid_y = backends::gpu::DivUp<int>(out_num, kMaxOut);
    }
    int tile_x_num = backends::gpu::DivUp<int>(dy_pre, tid_x);
    bid_x = std::min(tile_x_num, backends::gpu::kMultiDimslimit);
    dim3 blocks(tid_x, tid_y, 1);
    dim3 grids(bid_x, bid_y, 1);

    StackGradKernelForLastDim<T, IndexT><<<grids, blocks, 0, ctx.stream()>>>(
        out.data<T>(),
        out_num,
        dy_pre,
        tile_x_num,
        reinterpret_cast<T**>(tmp_out_data->ptr()));
  } else {
    int dy_suf = out.numel() / (out_num * dy_pre);
    auto config =
        backends::gpu::GetGpuLaunchConfig1D(ctx, dy_pre * out_num * dy_suf);

    UnStackHelperCUDAKernel<T, IndexT>
        <<<config.block_per_grid, config.thread_per_block, 0, ctx.stream()>>>(
            out.data<T>(),
            dy_pre,
            out_num,
            dy_suf,
            out_num,
            reinterpret_cast<T**>(tmp_out_data->ptr()));
  }
}

template <typename T, typename Context>
void StackGradKernel(const Context& dev_ctx,
                     const DenseTensor& out,
                     int axis,
                     std::vector<DenseTensor*> x_grad) {
  const auto& dy_dims = out.dims();
  int actual_axis = axis < 0 ? axis + dy_dims.size() : axis;
  bool use_int32 = out.numel() < std::numeric_limits<int32_t>::max();

  int64_t dy_pre = 1;
  for (int i = 0; i < actual_axis; ++i) {
    dy_pre *= dy_dims[i];
  }
  if (use_int32) {
    LaunchStackGradCUDAKernel<Context, T, int32_t>(
        dev_ctx, out, &x_grad, actual_axis, dy_pre);
  } else {
    LaunchStackGradCUDAKernel<Context, T, int64_t>(
        dev_ctx, out, &x_grad, actual_axis, dy_pre);
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
