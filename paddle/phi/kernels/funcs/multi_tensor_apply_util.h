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

#include <cstdint>

#include "math.h"  // NOLINT

#include "paddle/phi/core/cuda_stream.h"

namespace phi {
namespace funcs {

template <int MaxTensorNumPerLaunch, int MaxChunkNumPerLaunch>
struct TensorMetaList {
  static constexpr int kTensorNum = MaxTensorNumPerLaunch;
  static constexpr int kChunkNum = MaxChunkNumPerLaunch;

  static_assert(kTensorNum > 0 && kTensorNum < 256,
                "kTensorNum must be inside (0, 256).");
  static_assert(kChunkNum > 0 && kChunkNum < 65536,
                "kChunkNum must be inside (0, 65536).");

  /**
   * The tensor numel offset of each tensor.
   * The offsets[0] would be always 0 in the first launch,
   * and then offsets[0] >= 0 in the following other launches.
   * The numel of the i-th tensor would be offsets[i + 1] - offsets[i].
   */
  int offsets[kTensorNum + 1];

  /**
   * The tensor id of each chunk. The tensor_ids[0] is always 0.
   * Note that tensor_ids would be always in the ascending order.
   * The actual tensor id is start_tensor_id + tensor_ids[i].
   *
   * The reason why we assume that the actual tensor id is
   * start_tensor_id + tensor_ids[i] is to make tensor_ids to be
   * a uint8_t array instead of an int array, making sizeof(TensorMetaList)
   * smaller, so that kChunkNum can be larger.
   */
  uint8_t tensor_ids[kChunkNum];

  /**
   * The chunk id of the chunk inside each tensor. It would be
   * something like chunk_ids = [0, 1, 2, 0, 0, 1, 2, 3], meaning
   * that there are 3 tensors and each tensor contains 3, 1 and 4
   * chunks. Note that chunk_ids[0] is always 0 and the actual
   * chunk id of the first tensor is always start_chunk_id + chunk_ids[i].
   *
   * The reason why we assume that the actual chunk id of the first
   * tensor is always start_chunk_id + chunk_ids[i] is to make
   * chunk_ids to be a uint16_t array instead of an int array, making
   * sizeof(TensorMetaList) smaller, so that kChunkNum can be larger.
   */
  uint16_t chunk_ids[kChunkNum];

  /**
   * The tensor_ids offset.
   */
  int start_tensor_id;

  /**
   * The chunk_ids offset.
   */
  int start_chunk_id;
};

template <typename Functor,
          int MaxTensorNumPerLaunch,
          int MaxChunkNumPerLaunch,
          typename... Args>
static __global__ void MultiTensorApplyCUDAKernel(
    Functor functor,
    TensorMetaList<MaxTensorNumPerLaunch, MaxChunkNumPerLaunch> meta,
    int chunk_size,
    Args... args) {
  const int block_id = blockIdx.x;
  const int tensor_id = meta.tensor_ids[block_id];
  const int chunk_id = static_cast<int>(meta.chunk_ids[block_id]) +
                       (tensor_id == 0) * meta.start_chunk_id;
  const int prev_offset = meta.offsets[tensor_id];
  const int next_offset = meta.offsets[tensor_id + 1];
  const int ptr_offset = prev_offset + chunk_id * chunk_size;
  const int size = min(next_offset - ptr_offset, chunk_size);

  functor(
      tensor_id + meta.start_tensor_id, chunk_id, ptr_offset, size, args...);
}

template <int MaxTensorNumPerLaunch, int MaxChunkNumPerLaunch>
class MultiTensorLauncher {
 public:
  MultiTensorLauncher(
      const TensorMetaList<MaxTensorNumPerLaunch, MaxChunkNumPerLaunch> &meta,
      const int &chunk_id,
      const int &chunk_size,
      const int &block_dim,
      const gpuStream_t &stream)
      : meta_(meta),
        chunk_id_(chunk_id),
        chunk_size_(chunk_size),
        block_dim_(block_dim),
        stream_(stream) {}

  template <typename Functor, typename... Args>
  void Launch(Functor &&functor, Args &&...args) const {
    MultiTensorApplyCUDAKernel<Functor,
                               MaxTensorNumPerLaunch,
                               MaxChunkNumPerLaunch>
        <<<chunk_id_, block_dim_, 0, stream_>>>(
            functor, meta_, chunk_size_, args...);
  }

 private:
  const TensorMetaList<MaxTensorNumPerLaunch, MaxChunkNumPerLaunch> &meta_;
  const int &chunk_id_;
  const int &chunk_size_;
  const int &block_dim_;
  const gpuStream_t &stream_;
};

template <int MaxTensorNumPerLaunch,
          int MaxChunkNumPerLaunch,
          typename Callback>
static void MultiTensorApplyWithCallback(gpuStream_t stream,
                                         const int *offsets,
                                         int n,
                                         int chunk_size,
                                         int block_dim,
                                         Callback &&callback) {
  if (n == 0) return;

  constexpr auto NumTensor = MaxTensorNumPerLaunch;
  constexpr auto NumChunk = MaxChunkNumPerLaunch;
  TensorMetaList<NumTensor, NumChunk> metas;

  int tensor_id = 0;
  int chunk_id = 0;
  int numel_offset = 0;
  metas.start_tensor_id = 0;
  metas.start_chunk_id = 0;
  int launch_num = 0;

  MultiTensorLauncher<MaxTensorNumPerLaunch, MaxChunkNumPerLaunch> launcher(
      metas, chunk_id, chunk_size, block_dim, stream);

  for (int i = 0; i < n; ++i) {
    auto length = offsets[i + 1] - offsets[i];
    if (tensor_id == 0) {
      metas.start_tensor_id = i;
      metas.offsets[0] = numel_offset;
    }
    metas.offsets[tensor_id + 1] = metas.offsets[tensor_id] + length;
    ++tensor_id;
    numel_offset += length;

    auto chunk_num = (length + chunk_size - 1) / chunk_size;
    int last_launch_chunk_id = 0;
    for (int j = 0; j < chunk_num; ++j) {
      metas.chunk_ids[chunk_id] = j - last_launch_chunk_id;
      metas.tensor_ids[chunk_id] = tensor_id - 1;
      ++chunk_id;

      bool tensor_full = (tensor_id == NumTensor && j + 1 == chunk_num);
      bool block_full = (chunk_id == NumChunk);
      bool last_chunk = (i + 1 == n && j + 1 == chunk_num);

      if (tensor_full || block_full || last_chunk) {
        callback(launcher, launch_num);
        ++launch_num;
        chunk_id = 0;
        if (j + 1 == chunk_num) {  // chunk for the current tensor is full
          metas.start_chunk_id = 0;
          tensor_id = 0;
        } else {
          metas.offsets[0] = metas.offsets[tensor_id - 1];
          metas.offsets[1] = metas.offsets[tensor_id];
          metas.start_tensor_id = i;
          metas.start_chunk_id = j + 1;
          last_launch_chunk_id = j + 1;
          tensor_id = 1;
        }
      }
    }
  }
}

template <typename Functor,
          int MaxTensorNumPerLaunch,
          int MaxChunkNumPerLaunch,
          typename... Args>
static void MultiTensorApply(Functor functor,
                             gpuStream_t stream,
                             const int *offsets,
                             int n,
                             int chunk_size,
                             int block_dim,
                             Args &&...args) {
  auto callback = [&](const MultiTensorLauncher<MaxTensorNumPerLaunch,
                                                MaxChunkNumPerLaunch> &launcher,
                      int i) { launcher.Launch(functor, args...); };
  MultiTensorApplyWithCallback<MaxTensorNumPerLaunch, MaxChunkNumPerLaunch>(
      stream, offsets, n, chunk_size, block_dim, callback);
}

}  // namespace funcs
}  // namespace phi
