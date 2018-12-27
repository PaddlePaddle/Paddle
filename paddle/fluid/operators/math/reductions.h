/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once
#include <algorithm>
#include <cub/cub.cuh>  // NOLINT
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/float16.h"
#include "paddle/fluid/platform/temporary_allocator.h"

namespace paddle {
namespace operators {
namespace math {

struct RowOffset {
  __host__ __device__ explicit RowOffset(const int& cols) : cols_(cols) {}

  __host__ __device__ int operator()(const int& x) const { return cols_ * x; }

  int cols_;
};

// maps a warp to each row
template <typename T, typename outT, typename Op>
static __global__ void RowReduceKernel(
    T in, outT out, int num_rows, int num_cols, Op op,
    typename std::iterator_traits<T>::value_type initVal) {
  typedef typename std::iterator_traits<T>::value_type value_type;
  const int row = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
  const int lane = threadIdx.x % 32;

  if (num_cols == 1) {
    int gid = threadIdx.x + blockIdx.x * blockDim.x;
    if (gid < num_rows) out[gid] = in[gid];
    return;
  }

  value_type sum = initVal;
  int col = lane;

  if (row < num_rows && col < num_cols) {
    for (; col < num_cols; col += 32) {
      sum = op(sum, in[row * num_cols + col]);
    }
  }

  typedef cub::WarpReduce<value_type> WarpReduce;

  __shared__ typename WarpReduce::TempStorage temp_storage;

  sum = WarpReduce(temp_storage).Reduce(sum, op, min(num_cols, 32));

  if (row < num_rows && lane == 0) out[row] = sum;
}

template <typename T, typename OUT_T, typename IN_T, typename Op>
void LaunchRowReduction(const platform::CUDADeviceContext* ctx, OUT_T out,
                        IN_T in, int num_rows, int num_cols, Op op, T init) {
  if (num_cols < 1024) {
    const int threads_per_block = 128;
    const int warps_per_block = threads_per_block / 32;
    int num_blocks = (num_rows + warps_per_block - 1) / warps_per_block;

    RowReduceKernel<<<num_blocks, threads_per_block, 0, ctx->stream()>>>(
        in, out, num_rows, num_cols, op, init);
    return;
  }

  // setup segment offsets with counting and transform iterator
  RowOffset row_offset_op(num_cols);
  cub::CountingInputIterator<int> counting_iter(0);
  cub::TransformInputIterator<int, RowOffset, cub::CountingInputIterator<int>>
      transform_iter(counting_iter, row_offset_op);

  std::size_t temp_storage_bytes = 0;
  uint8_t* tmp_allocation_ptr;

  for (int i = 0; i < 2; ++i) {
    auto success = cub::DeviceSegmentedReduce::Reduce(
        i == 0 ? nullptr : tmp_allocation_ptr, temp_storage_bytes, in, out,
        num_rows, transform_iter, transform_iter + 1, op, init, ctx->stream());

    PADDLE_ENFORCE_EQ(success, 0, "CUB segmented reduce error");
    if (i == 0) {
      tmp_allocation_ptr =
          platform::DeviceTemporaryAllocator::Instance().Get(ctx).Allocate(
              temp_storage_bytes * sizeof(uint8_t));
    }
  }
}

}  // namespace math
}  // namespace operators
}  // namespace paddle
