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

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/tensor_utils.h"

namespace phi {
namespace funcs {

// This code is referenced from apex's multi_tensor_apply.cuh.
// https://github.com/NVIDIA/apex

template <int N, int MaxTensorSize, int MaxBlockSize>
struct TensorAndBlockInfo {
  void *tensor_addrs[N - 1][MaxTensorSize];
  const void *grads[MaxTensorSize];
  int sizes[MaxTensorSize];
  uint8_t tensor_ids[MaxBlockSize];
  // int16
  uint16_t chunk_ids[MaxBlockSize];
  int start_chunk_id;

  DEVICE void GetChunkIdAndTensorId(int *chunk_id, int *tensor_id) const {
    int block_id = blockIdx.x;
    int tmp_tensor_id = tensor_ids[block_id];
    *chunk_id = static_cast<int>(chunk_ids[block_id]) +
                (tmp_tensor_id == 0) * start_chunk_id;
    *tensor_id = tmp_tensor_id;
  }
};

template <int N,
          int MaxTensorSize,
          int MaxBlockSize,
          typename Functor,
          typename... ArgTypes>
__global__ void MultiTensorApplyCudaKernel(
    int chunk_size,
    TensorAndBlockInfo<N, MaxTensorSize, MaxBlockSize> t_info,
    Functor functor,
    ArgTypes... args) {
  functor(chunk_size, t_info, args...);
}

template <int InputNum,
          int MaxTensorSize,
          int MaxBlockSize,
          typename Functor,
          typename Context,
          typename... ArgTypes>
void LaunchMultiTensorApplyKernel(
    const Context &dev_ctx,
    int block_size,
    int chunk_size,
    const std::vector<std::vector<DenseTensor *>> &input_vector,
    const std::vector<const DenseTensor *> &grads,
    Functor functor,
    ArgTypes... args) {
  PADDLE_ENFORCE_EQ(
      input_vector.size(),
      InputNum - 1,
      errors::InvalidArgument(
          "input_vector.size() != InputNum - 1, the input vector's size is "
          "unequal to InputNum - 1, please cheack grads, params, momemts1, "
          "moments2, and, master_params."));
  size_t length = input_vector[0].size();
  PADDLE_ENFORCE_GT(
      length,
      0,
      errors::InvalidArgument(
          "input_vector[0].size() is not > 0, please cheack params."));
  auto ctx_place = dev_ctx.GetPlace();
  PADDLE_ENFORCE_EQ(
      ctx_place.GetType() == AllocationType::GPU,
      true,
      errors::PreconditionNotMet(
          "Context place error, excepted GPUPlace, but actually %s.",
          ctx_place));
  auto place = input_vector[0][0]->place();
  for (size_t i = 0; i < input_vector.size(); i++) {
    PADDLE_ENFORCE_EQ(
        input_vector[i].size(),
        length,
        errors::InvalidArgument(
            "some input vectors' size mismatch other input vector."));
    for (size_t j = 0; j < input_vector[i].size(); j++) {
      PADDLE_ENFORCE_EQ(
          input_vector[i][j]->place(),
          place,
          errors::InvalidArgument(
              "A tensor was not on the same device as the first tensor"));
      PADDLE_ENFORCE_EQ(input_vector[i][j]->numel(),
                        input_vector[0][j]->numel(),
                        errors::InvalidArgument(
                            "The number of elements of Inputs must be equal."));
    }
  }

  size_t tensors_size = input_vector[0].size();

  TensorAndBlockInfo<InputNum, MaxTensorSize, MaxBlockSize> t_info;
  t_info.start_chunk_id = 0;

  auto stream = dev_ctx.stream();
  int block_id = 0;
  int tensor_id = 0;
  for (int t = 0; t < tensors_size; t++) {
    t_info.sizes[tensor_id] = input_vector[0][t]->numel();
    t_info.grads[tensor_id] = grads[t]->data();
    for (int d = 0; d < InputNum - 1; d++) {
      t_info.tensor_addrs[d][tensor_id] = input_vector[d][t]->data();
    }
    tensor_id++;
    int chunks_this_tensor =
        (input_vector[0][t]->numel() + chunk_size - 1) / chunk_size;

    constexpr auto kMaxChunkId = std::numeric_limits<uint16_t>::max();
    for (int chunk = 0; chunk < chunks_this_tensor; chunk++) {
      t_info.tensor_ids[block_id] = tensor_id - 1;
      auto saved_chunk_id =
          (tensor_id == 1 ? chunk - t_info.start_chunk_id : chunk);
      PADDLE_ENFORCE_GE(saved_chunk_id,
                        0,
                        errors::InvalidArgument(
                            "The chunk id is less than 0 in "
                            "MultiTensorApplyKernel. This may be a bug."));
      PADDLE_ENFORCE_LE(
          saved_chunk_id,
          kMaxChunkId,
          errors::InvalidArgument(
              "The chunk id exceeds maximum value %d. This may be a bug.",
              kMaxChunkId));
      t_info.chunk_ids[block_id] = saved_chunk_id;
      block_id++;
      bool reach_tensors_limit =
          (tensor_id == MaxTensorSize && chunk == chunks_this_tensor - 1);
      bool reach_blocks_limit = (block_id == MaxBlockSize);
      bool finish_compute =
          (t == tensors_size - 1 && chunk == chunks_this_tensor - 1);
      if (reach_tensors_limit || reach_blocks_limit || finish_compute) {
        MultiTensorApplyCudaKernel<InputNum,
                                   MaxTensorSize,
                                   MaxBlockSize,
                                   Functor,
                                   ArgTypes...>
            <<<block_id, block_size, 0, stream>>>(
                chunk_size, t_info, functor, args...);

        block_id = 0;
        if (chunk == chunks_this_tensor - 1) {
          tensor_id = 0;
          t_info.start_chunk_id = 0;
        } else {
          t_info.sizes[0] = t_info.sizes[tensor_id - 1];
          t_info.grads[0] = t_info.grads[tensor_id - 1];
          for (int d = 0; d < InputNum - 1; d++) {
            t_info.tensor_addrs[d][0] = t_info.tensor_addrs[d][tensor_id - 1];
          }
          tensor_id = 1;
          t_info.start_chunk_id = chunk + 1;
        }
      }
    }
  }
}

}  // namespace funcs
}  // namespace phi
