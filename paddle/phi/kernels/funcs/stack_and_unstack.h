// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#pragma once

#include "glog/logging.h"

#include "paddle/phi/backends/gpu/gpu_launch_config.h"
#include "paddle/phi/kernels/funcs/fast_divmod.h"
#include "paddle/phi/kernels/funcs/segmented_array.h"

namespace phi {
namespace funcs {

template <typename T, typename IndexT, typename ArrayT>
__global__ void StackCudaKernel(ArrayT array,
                                GeneralDivMod<IndexT> divmoder,
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
#pragma unroll
    for (; grid_y < rows; grid_y += grid_y_stride) {
      output[grid_y * cols + grid_x] =
          input_ptr[grid_y * split_size + col_offset];
    }
  }
}

template <typename Context,
          typename T,
          typename IndexT,
          SegmentedArraySize Size>
void LaunchStackKernel(const Context& ctx,
                       const IndexT x_col,
                       const IndexT x_row,
                       const IndexT out_col,
                       const std::vector<const DenseTensor*>& x,
                       DenseTensor* out) {
  T* out_ptr = ctx.template Alloc<T>(out);
  auto config = phi::backends::gpu::GetGpuLaunchConfig2D(ctx, out_col, x_row);

  ConstPointerArraySetter<Context, T, Size> setter(ctx, x);
  GeneralDivMod<IndexT> divmoder(x_col);
  StackCudaKernel<T, IndexT, decltype(setter.array)>
      <<<config.block_per_grid, config.thread_per_block, 0, ctx.stream()>>>(
          setter.array, divmoder, x_col, x_row, out_col, out_ptr);
}

template <typename T, typename Context>
void StackRawKernel(const Context& ctx,
                    const std::vector<const DenseTensor*>& x,
                    int axis,
                    DenseTensor* out) {
  if (axis < 0) axis += (x[0]->dims().size() + 1);
  int num = static_cast<int>(x.size());

  // zero sized tensor case
  if (x[0]->numel() == 0) {
    ctx.template Alloc<T>(out);
    auto out_dims = out->dims();
    out->Resize(out_dims);
    return;
  }
  // Split x dim from axis to matrix of shape [x_row, x_col], and the output
  // tensor's shape is [x_row, out_col].
  int64_t x_row = 1, x_row_bak = 1;
  for (int i = 0; i < axis; ++i) {
    x_row *= x[0]->dims()[i];
  }
  x_row_bak = x_row == 0 ? 1 : x_row;
  int64_t x_col = x[0]->numel() / x_row_bak;
  int64_t out_col = x_col * num;

  if (out->numel() < std::numeric_limits<int32_t>::max()) {
    switch (CalcArraySize(num)) {
      SEGMENTED_ARRAY_KERNEL_HELPER(
          LaunchStackKernel<Context, T, int32_t, kArraySize>(
              ctx, x_col, x_row, out_col, x, out));
    }
  } else {
    switch (CalcArraySize(num)) {
      SEGMENTED_ARRAY_KERNEL_HELPER(
          LaunchStackKernel<Context, T, int64_t, kArraySize>(
              ctx, x_col, x_row, out_col, x, out));
    }
  }
}

template <typename T, typename IndexT, typename ArrayT>
__global__ void UnStackCudaKernel(const T* __restrict__ input,
                                  IndexT out_row,
                                  IndexT split_dim,
                                  IndexT out_col,
                                  IndexT num_splits,
                                  GeneralDivMod<IndexT> col_divmoder,
                                  ArrayT array) {
  assert(blockDim.y == 1);
  assert(blockDim.z == 1);
  // In this case they are equal
  assert(split_dim % num_splits == 0);

  IndexT numel = out_row * split_dim * out_col;
  IndexT each_dim_size = split_dim / num_splits;
  IndexT split_dim_with_out_col = split_dim * out_col;

  IndexT offset = blockIdx.x * blockDim.x + threadIdx.x;
  if (each_dim_size == 1) {
    for (; offset < numel; offset += blockDim.x * gridDim.x) {
      auto col_divmod_rslt = col_divmoder.div_mod(offset);

      IndexT i = offset / split_dim_with_out_col;
      IndexT j = col_divmod_rslt[0] - i * split_dim;
      IndexT k = col_divmod_rslt[1];  // offset % out_col

      T* output = array.data[j];
      if (output) {
        IndexT output_idx = i * out_col + k;
        *(output + output_idx) = input[offset];
      }
    }
  } else {
    for (; offset < numel; offset += blockDim.x * gridDim.x) {
      auto col_divmod_rslt = col_divmoder.div_mod(offset);

      IndexT i = offset / split_dim_with_out_col;
      IndexT j = col_divmod_rslt[0] - i * split_dim;
      IndexT k = col_divmod_rslt[1];  // offset % out_col

      T* output = array.data[j / each_dim_size];
      if (output) {
        IndexT output_idx = (i + j % each_dim_size) * out_col + k;
        *(output + output_idx) = input[offset];
      }
    }
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
          SegmentedArraySize Size>
void LaunchUnStackKernel(const Context& ctx,
                         const IndexT out_row,
                         const IndexT split_dim,
                         const IndexT out_col,
                         const IndexT num_splits,
                         const DenseTensor& x,
                         std::vector<DenseTensor*>* outs) {
  // each tensor in outs should have same shape.
  VLOG(6) << "out_row=" << out_row << ", split_dim=" << split_dim
          << ", out_col=" << out_col << ", num_splits=" << num_splits;

  auto x_ptr = x.data<T>();
  PointerArraySetter<Context, T, Size> setter(ctx, outs, /*need_alloc=*/true);

  if (out_col == 1) {
    // For the case axis == (x.dims().size() - 1)
    constexpr int kThreads = 512;
    constexpr int kWarpSize = 32;
    constexpr int kMaxOut = 16;

    int tid_x = 0, tid_y = 0, bid_x = 0, bid_y = 1;
    if (split_dim < kMaxOut) {
      tid_y = split_dim;
      tid_x =
          std::min(backends::gpu::RoundToNextHighPowOfTwo(out_row, kWarpSize),
                   kThreads / backends::gpu::RoundToNextHighPowOfTwo(tid_y));
    } else {
      tid_y = kMaxOut;
      tid_x = kWarpSize;
      bid_y = backends::gpu::DivUp<int>(split_dim, kMaxOut);
    }
    int tile_x_num = backends::gpu::DivUp<int>(out_row, tid_x);
    bid_x = std::min(tile_x_num, backends::gpu::kMultiDimslimit);
    dim3 blocks(tid_x, tid_y, 1);
    dim3 grids(bid_x, bid_y, 1);

    UnStackCudaKernelForLastDim<T, IndexT, decltype(setter.array)>
        <<<grids, blocks, 0, ctx.stream()>>>(
            x_ptr, split_dim, out_row, tile_x_num, setter.array);
  } else {
    GeneralDivMod<IndexT> col_divmoder(out_col);
    auto config = phi::backends::gpu::GetGpuLaunchConfig1D(
        ctx, out_row * split_dim * out_col);

    UnStackCudaKernel<T, IndexT, decltype(setter.array)>
        <<<config.block_per_grid.x,
           config.thread_per_block.x,
           0,
           ctx.stream()>>>(x_ptr,
                           out_row,
                           split_dim,
                           out_col,
                           num_splits,
                           col_divmoder,
                           setter.array);
  }
}

template <typename T, typename Context>
void UnStackRawKernel(const Context& ctx,
                      const DenseTensor& x,
                      int axis,
                      std::vector<DenseTensor*>* outs) {
  auto x_dims = x.dims();

  // Input tensor is splited to split_dim tensors along split_dim dimension.
  int64_t split_dim = x_dims[axis];

  // zero sized tensor case
  if (x.numel() == 0) {
    for (int i = 0; i < split_dim; i++) {
      ctx.template Alloc<T>((*outs)[i]);
      auto x_grad_dim = (*outs)[i]->dims();
      (*outs)[i]->Resize(x_grad_dim);
    }
    return;
  }
  // Treat outs[i] as [out_row, out_col], and x as [out_row, split_dim,
  // out_col].
  int64_t out_row = 1;
  for (int i = 0; i < axis; ++i) {
    out_row *= x_dims[i];
  }

  int64_t out_col = x.numel() / (split_dim * out_row);

  if (x.numel() < std::numeric_limits<int32_t>::max()) {
    switch (CalcArraySize(split_dim)) {
      SEGMENTED_ARRAY_KERNEL_HELPER(
          LaunchUnStackKernel<Context, T, int32_t, kArraySize>(
              ctx, out_row, split_dim, out_col, split_dim, x, outs));
    }
  } else {
    switch (CalcArraySize(split_dim)) {
      SEGMENTED_ARRAY_KERNEL_HELPER(
          LaunchUnStackKernel<Context, T, int64_t, kArraySize>(
              ctx, out_row, split_dim, out_col, split_dim, x, outs));
    }
  }
}

}  // namespace funcs
}  // namespace phi
