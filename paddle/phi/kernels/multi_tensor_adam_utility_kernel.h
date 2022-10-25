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

#include "paddle/fluid/framework/tensor_util.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/tensor_utils.h"

namespace phi {

constexpr int max_chunk_size = 65535;

template <int N, int MAXTENSORSIZE, int MAXBLOCKSIZE>
struct TensorAndBlockInfo {
  void *tensor_addrs[N - 1][MAXTENSORSIZE];
  const void *grads[MAXTENSORSIZE];
  int sizes[MAXTENSORSIZE];
  uint8_t tenosr_for_this_block[MAXBLOCKSIZE];
  // int16
  uint16_t chunk_for_this_block[MAXBLOCKSIZE];
  int start_chunk_this_tensor;
};

template <typename MT, typename T, typename U, typename... ArgTypes>
__global__ void UtilityKernel(int chunk_size,
                              T t_info,
                              U multi_tensor_adam_functor_op,
                              ArgTypes... args) {
  multi_tensor_adam_functor_op(chunk_size, t_info, args...);
}

template <int InputNum,
          int MAXTENSORSIZE,
          int MAXBLOCKSIZE,
          typename MT,
          typename T,
          typename Context,
          typename... ArgTypes>
void MultiTensorAdamUtilityKernel(
    const Context &dev_ctx,
    int block_size,  // 512
    int chunk_size,  // 2048*32
    const std::vector<std::vector<DenseTensor *>> &input_vector,
    const std::vector<const DenseTensor *> &g,
    T multi_tensor_adam_functor_op,
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
  auto place = input_vector[0][0]->place();
  PADDLE_ENFORCE_NE(
      place,
      CPUPlace(),
      errors::InvalidArgument(
          "expected input to be on gpu, but input is on cpu now."));
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

  TensorAndBlockInfo<InputNum, MAXTENSORSIZE, MAXBLOCKSIZE> t_info;

  auto stream = dev_ctx.stream();
  int block_id = 0;
  int tensor_id = 0;
  for (int t = 0; t < tensors_size; t++) {
    t_info.sizes[tensor_id] = input_vector[0][t]->numel();
    t_info.grads[tensor_id] = g[t]->data();
    for (int d = 0; d < InputNum - 1; d++)
      t_info.tensor_addrs[d][tensor_id] = input_vector[d][t]->data();
    tensor_id++;
    int chunks_this_tensor =
        (input_vector[0][t]->numel() + chunk_size - 1) / chunk_size;
    t_info.start_chunk_this_tensor = 0;
    int local_chunk = 0;

    for (int chunk = 0; chunk < chunks_this_tensor; chunk++) {
      t_info.tenosr_for_this_block[block_id] = tensor_id - 1;
      if (local_chunk > max_chunk_size) {
        t_info.start_chunk_this_tensor += max_chunk_size;
        local_chunk = 1;
      }
      t_info.chunk_for_this_block[block_id] = local_chunk;
      local_chunk++;
      block_id++;
      bool reach_tensors_limit =
          (tensor_id == MAXTENSORSIZE && chunk == chunks_this_tensor - 1);
      bool reach_blocks_limit = (block_id == MAXBLOCKSIZE);
      bool finish_compute =
          (t == tensors_size - 1 && chunk == chunks_this_tensor - 1);
      if (reach_tensors_limit || reach_blocks_limit || finish_compute) {
        UtilityKernel<MT>
            <<<block_id, block_size, 0, stream>>>(chunk_size,  // 2048*32
                                                  t_info,
                                                  multi_tensor_adam_functor_op,
                                                  args...);

        block_id = 0;
        if (chunk == chunks_this_tensor - 1) {
          tensor_id = 0;
        } else {
          t_info.sizes[0] = t_info.sizes[tensor_id - 1];
          t_info.grads[0] = t_info.grads[tensor_id - 1];
          for (int d = 0; d < InputNum - 1; d++)
            t_info.tensor_addrs[d][0] = t_info.tensor_addrs[d][tensor_id - 1];
          tensor_id = 1;
        }
      }
    }
  }
}

}  // namespace phi
