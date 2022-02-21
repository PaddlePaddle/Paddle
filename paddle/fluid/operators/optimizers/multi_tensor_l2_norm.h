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

#pragma once

#include "cub/cub.cuh"
#include "paddle/fluid/operators/amp/fp16_type_traits.h"
#include "paddle/fluid/platform/device_context.h"

namespace paddle {
namespace operators {

template <typename T, int NumTensor, int NumChunk>
struct TensorMeta {
  static constexpr int kTensorNum = NumTensor;
  static constexpr int kChunkNum = NumChunk;

  const T *ptrs[kTensorNum];
  int sizes[kTensorNum];
  int chunk_ids[kChunkNum];
  int tensor_ids[kChunkNum];
  int start_tensor_id;
};

template <typename T, int BlockDim, int NumTensor, int NumChunk>
static __global__ void MultiTensorL2NormCUDAKernel(
    TensorMeta<T, NumTensor, NumChunk> meta,
    typename details::MPTypeTrait<T>::Type *y, int chunk_size,
    int max_chunk_num) {
  int block_id = blockIdx.x;

  const int tensor_id = meta.tensor_ids[block_id];
  const int chunk_id = meta.chunk_ids[block_id];

  const T *ptr = meta.ptrs[tensor_id] + chunk_id * chunk_size;
  const int size =
      min(meta.sizes[tensor_id] - chunk_id * chunk_size, chunk_size);

  using MT = typename details::MPTypeTrait<T>::Type;
  using BlockReduce = cub::BlockReduce<MT, BlockDim>;
  __shared__ typename BlockReduce::TempStorage storage;
  MT square_sum = static_cast<MT>(0);
  for (int i = threadIdx.x; i < size; i += BlockDim) {
    auto tmp = static_cast<MT>(ptr[i]);
    square_sum += (tmp * tmp);
  }
  square_sum = BlockReduce(storage).Reduce(square_sum, cub::Sum());
  if (threadIdx.x == 0) {
    y[(tensor_id + meta.start_tensor_id) * max_chunk_num + chunk_id] =
        square_sum;
  }
}

template <typename T, int BlockDim>
static __global__ void MultiTensorL2NormCleanUpCUDAKernel(const T *x, T *y,
                                                          int max_chunk_num) {
  int tensor_id = blockIdx.x;
  x += (tensor_id * max_chunk_num);
  using BlockReduce = cub::BlockReduce<T, BlockDim>;
  __shared__ typename BlockReduce::TempStorage storage;
  T sum = static_cast<T>(0);
  for (int i = threadIdx.x; i < max_chunk_num; i += BlockDim) {
    sum += x[i];
  }
  sum = BlockReduce(storage).Reduce(sum, cub::Sum());
  if (threadIdx.x == 0) {
    y[blockIdx.x] = sum;
  }
}

template <typename T>
static void MultiTensorL2Norm(const platform::CUDADeviceContext &dev_ctx,
                              const T *x, const int *numel_offsets, int n,
                              typename details::MPTypeTrait<T>::Type *y,
                              int chunk_size = 2048 * 32) {
  if (n == 0) return;

  std::vector<int> lengths(n);
  int max_chunk_num = -1;
  for (int i = 0; i < n; ++i) {
    lengths[i] = numel_offsets[i + 1] - numel_offsets[i];
    auto tmp_chunk_num = (lengths[i] + chunk_size - 1) / chunk_size;
    max_chunk_num = std::max(max_chunk_num, tmp_chunk_num);
  }

  using MT = typename details::MPTypeTrait<T>::Type;

  auto place = dev_ctx.GetPlace();
  auto stream = dev_ctx.stream();
  memory::Buffer tmp_out(place);
  MT *tmp_out_ptr = tmp_out.Alloc<MT>(n * max_chunk_num);
  PADDLE_ENFORCE_GPU_SUCCESS(
      cudaMemsetAsync(tmp_out_ptr, 0, n * max_chunk_num * sizeof(MT), stream));

  TensorMeta<T, 110, 320> metas;
  constexpr int kNumTensor = decltype(metas)::kTensorNum;
  constexpr int kNumChunk = decltype(metas)::kChunkNum;
  constexpr int kBlockDim = 512;

  int tensor_id = 0;
  int chunk_id = 0;
  metas.start_tensor_id = 0;
  for (int i = 0; i < n; ++i) {
    metas.ptrs[tensor_id] = x;
    metas.sizes[tensor_id] = lengths[i];
    x += lengths[i];
    ++tensor_id;

    auto chunk_num = (lengths[i] + chunk_size - 1) / chunk_size;
    for (int j = 0; j < chunk_num; ++j) {
      metas.chunk_ids[chunk_id] = j;
      metas.tensor_ids[chunk_id] = tensor_id - 1;
      ++chunk_id;

      bool tensor_full = (tensor_id == kNumTensor && j + 1 == chunk_num);
      bool block_full = (chunk_id == kNumChunk);
      bool last_chunk = (i + 1 == n && j + 1 == chunk_num);

      if (tensor_full || block_full || last_chunk) {
        MultiTensorL2NormCUDAKernel<
            T, kBlockDim><<<chunk_id, kBlockDim, 0, stream>>>(
            metas, tmp_out_ptr, chunk_size, max_chunk_num);

        chunk_id = 0;
        if (j + 1 == chunk_num) {
          metas.start_tensor_id = i + 1;
          tensor_id = 0;
        } else {
          metas.ptrs[0] = metas.ptrs[tensor_id - 1];
          metas.sizes[0] = lengths[tensor_id - 1];
          metas.start_tensor_id = i;
          tensor_id = 1;
        }
      }
    }
  }

  MultiTensorL2NormCleanUpCUDAKernel<typename details::MPTypeTrait<T>::Type,
                                     kBlockDim><<<n, kBlockDim, 0, stream>>>(
      tmp_out_ptr, y, max_chunk_num);
}

}  // namespace operators
}  // namespace paddle
