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
#include "generated/moe/moe.h"
#include "paddle/extension.h"

template <typename T>
__global__ void PreProcess(const T *routing_weights,
                           const int *selected_experts_ptr,
                           int *token_ids_ptr,
                           T *token_weight_ptr,
                           int *expert_ids_ptr,
                           int num_experts,
                           int max_token_num_by_expert,
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
        int tmp = expert_id * max_token_num_by_expert + id_this_expert_handle;
        token_ids_ptr[tmp] = i;
        token_weight_ptr[tmp] = routing_weights[i];
        expert_ids_ptr[tmp] = expert_id;
        id_this_expert_handle ++;
      }
    }
  }
}

std::vector<paddle::Tensor> TritonMoe(
    const paddle::Tensor& x,  // [*, K]
    const paddle::Tensor& all_experts_weight, // [E, K, N]
    const paddle::Tensor& routing_weights, // after paddle.topk
    const paddle::Tensor& selected_experts, // after paddle.topk
    int fake_top_k
    ) {
  
  int valid_token_num = x.shape()[0];
  int num_experts = all_experts_weight.shape()[0];
  int K = all_experts_weight.shape()[1];
  int N = all_experts_weight.shape()[2];
  int max_token_num_by_expert = 64;
  int top_k = selected_experts.shape()[1];
  
  auto token_ids = paddle::full({num_experts * max_token_num_by_expert}, selected_experts.numel(), paddle::DataType::INT32, x.place());
  auto token_weight = paddle::full({num_experts * max_token_num_by_expert}, 0, paddle::DataType::FLOAT16, x.place());
  auto expert_ids_ptr = paddle::full({num_experts * max_token_num_by_expert}, 0, paddle::DataType::INT32, x.place());

  PreProcess<<<num_experts, 32>>>(
    routing_weights.data<phi::dtype::float16>(),
    selected_experts.data<int>(),
    token_ids.data<int>(),
    token_weight.data<phi::dtype::float16>(),
    expert_ids_ptr.data<int>(),
    num_experts,
    max_token_num_by_expert,
    top_k,
    valid_token_num
  );
  
  auto c_out = paddle::full({valid_token_num, top_k, N}, 0, x.dtype(), x.place());

  auto num_tokens_post_padded = paddle::full({1}, token_ids.numel(), paddle::DataType::INT32, x.place());

  auto status = moe_kernel(
    c_out.stream(),
    (CUdeviceptr)(x.data<phi::dtype::float16>()),
    (CUdeviceptr)(all_experts_weight.data<phi::dtype::float16>()),
    (CUdeviceptr)(c_out.data<phi::dtype::float16>()),
    (CUdeviceptr)(token_weight.data<phi::dtype::float16>()),
    (CUdeviceptr)(token_ids.data<int>()),
    (CUdeviceptr)(expert_ids_ptr.data<int>()),
    (CUdeviceptr)(num_tokens_post_padded.data<int>()),
    N,
    K,
    token_ids.shape()[0],    // EM
    valid_token_num * top_k, // valid_token_num
    K, 1,
    K*N, N, 1,
    N, 1,
    1, -1, 0);
  assert(status == CUDA_SUCCESS);
  
  return {c_out};
  //return {token_ids, token_weight, expert_ids_ptr};
}

PD_BUILD_OP(triton_moe)
    .Inputs({"x", "all_experts_weight", "routing_weights", "selected_experts"})
    .Outputs({"c_out"})
    //.Outputs({"token_ids", "token_weight", "expert_ids_ptr"})
    .SetKernelFn(PD_KERNEL(TritonMoe))
    .Attrs({"fake_top_k: int"});
