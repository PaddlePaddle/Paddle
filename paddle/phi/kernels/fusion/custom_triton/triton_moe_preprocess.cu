// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include <cuda_fp16.h>
#include <vector>
#include "paddle/extension.h"

__global__ void GetMaxTokenByOneExpert(const int *selected_experts_ptr,
                                       int *out,
                                       int valid_token_num,
                                       int num_experts,
                                       int top_k) {
  int tid = threadIdx.x;
  int expert_id = blockIdx.x;
  int token_num_this_expert_handle = 0;
  // 只让第0个cuda thread 干活
  if (tid == 0) {
    for (int i = 0; i < valid_token_num * top_k; i += 1) {
      if (selected_experts_ptr[i] == expert_id) {
        token_num_this_expert_handle++;
      }
    }
  }
  atomicMax(out, token_num_this_expert_handle);
}

template <typename T>
__global__ void PreProcess(const T *routing_weights,
                           const int *selected_experts_ptr,
                           int *token_ids_ptr,
                           T *token_weight_ptr,
                           int *expert_ids_ptr,
                           int num_experts,
                           int max_token_num_by_one_expert,
                           int top_k,
                           int valid_token_num) {
  int tid = threadIdx.x;
  int lain_id = tid % 32;
  int expert_id = blockIdx.x;
  int id_this_expert_handle = 0;
  // 只让第0个cuda thread 干活
  if (tid == 0) {
    for (int i = 0; i < valid_token_num * top_k; i += 1) {
      if (selected_experts_ptr[i] == expert_id) {
        int tmp =
            expert_id * max_token_num_by_one_expert + id_this_expert_handle;
        token_ids_ptr[tmp] = i;
        token_weight_ptr[tmp] = routing_weights[i];
        expert_ids_ptr[tmp] = expert_id;
        id_this_expert_handle++;
      }
    }
  }
}

std::vector<paddle::Tensor> TritonMoeProcess(
    const paddle::Tensor &routing_weights,
    const paddle::Tensor &selected_experts,
    int num_experts) {
  int valid_token_num = routing_weights.shape()[0];

  int top_k = selected_experts.shape()[1];

  auto max_token_num_by_one_expert_tensor =
      paddle::full({1}, 0, paddle::DataType::INT32, routing_weights.place());
  GetMaxTokenByOneExpert<<<num_experts, 32>>>(
      selected_experts.data<int>(),
      max_token_num_by_one_expert_tensor.data<int>(),
      valid_token_num,
      num_experts,
      top_k);

  int max_token_num_by_one_expert;
  cudaMemcpy(&max_token_num_by_one_expert,
             max_token_num_by_one_expert_tensor.data<int>(),
             sizeof(int),
             cudaMemcpyDeviceToHost);

  const int token_each_threadblock = 16;
  max_token_num_by_one_expert =
      (max_token_num_by_one_expert + token_each_threadblock - 1) /
      token_each_threadblock * token_each_threadblock;

  auto token_ids = paddle::full({num_experts * max_token_num_by_one_expert},
                                selected_experts.numel(),
                                paddle::DataType::INT32,
                                routing_weights.place());
  auto token_weight = paddle::full({num_experts * max_token_num_by_one_expert},
                                   0,
                                   paddle::DataType::FLOAT16,
                                   routing_weights.place());
  auto expert_ids_ptr =
      paddle::full({num_experts * max_token_num_by_one_expert},
                   -1,
                   paddle::DataType::INT32,
                   routing_weights.place());

  PreProcess<<<num_experts, 32>>>(routing_weights.data<phi::dtype::float16>(),
                                  selected_experts.data<int>(),
                                  token_ids.data<int>(),
                                  token_weight.data<phi::dtype::float16>(),
                                  expert_ids_ptr.data<int>(),
                                  num_experts,
                                  max_token_num_by_one_expert,
                                  top_k,
                                  valid_token_num);
  return {token_ids, token_weight, expert_ids_ptr};
}

PD_BUILD_OP(triton_moe_preposs)
    .Inputs({"routing_weights", "selected_experts"})
    .Outputs({"token_ids", "token_weight", "expert_ids_ptr"})
    .Attrs({"num_experts: int"})
    .SetKernelFn(PD_KERNEL(TritonMoeProcess));
