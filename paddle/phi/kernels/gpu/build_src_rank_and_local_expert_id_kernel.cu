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

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T, typename U>
__global__ void build_srcrank_and_local_expert_id_kernel(
    T* src_rank,
    T* local_expert_id,
    const U* expert_num,
    int64_t total_num,
    int64_t num_total_experts,
    int64_t num_local_experts) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= num_total_experts) return;
  int64_t start = 0;
  int64_t end = 0;
  for (int64_t i = 0; i < num_total_experts; ++i) {
    end += expert_num[i];
    if (i == tid) {
      break;
    }
    start += expert_num[i];
  }
  for (int64_t i = start; i != end; ++i) {
    src_rank[i] = static_cast<T>(tid / num_local_experts);
    local_expert_id[i] = static_cast<T>(tid % num_local_experts);
  }
}

template <typename T, typename U>
void build_srcrank_and_local_expert_id(T* src_rank,
                                       T* local_expert_id,
                                       const U* expert_num,
                                       int64_t total_num,
                                       int64_t num_total_experts,
                                       int64_t num_local_experts,
                                       cudaStream_t stream) {
  int64_t threads_per_block = 32;
  int64_t blocks =
      (num_total_experts + threads_per_block - 1) / threads_per_block;
  build_srcrank_and_local_expert_id_kernel<T, U>
      <<<blocks, threads_per_block, 0, stream>>>(src_rank,
                                                 local_expert_id,
                                                 expert_num,
                                                 total_num,
                                                 num_total_experts,
                                                 num_local_experts);
}

template <typename T, typename Context>
void BuildSrcRankAndLocalExpertIdKernel(
    const Context& dev_ctx,
    const DenseTensor& expert_num_global_tensor,
    const std::vector<int64_t>& expert_num_global,
    int64_t num_local_experts,
    DenseTensor* src_rank,
    DenseTensor* local_expert_id) {
  int64_t token_num =
      std::accumulate(expert_num_global.begin(), expert_num_global.end(), 0);

  const int64_t* expert_num_global_tensor_data =
      expert_num_global_tensor.data<int64_t>();

  // Hard coded as ernie-core did.
  int* src_rank_data = dev_ctx.template Alloc<int>(src_rank);
  int* local_expert_id_data = dev_ctx.template Alloc<int>(local_expert_id);

  build_srcrank_and_local_expert_id<int, int64_t>(src_rank_data,
                                                  local_expert_id_data,
                                                  expert_num_global_tensor_data,
                                                  token_num,
                                                  expert_num_global.size(),
                                                  num_local_experts,
                                                  dev_ctx.stream());
}

}  // namespace phi

PD_REGISTER_KERNEL(build_src_rank_and_local_expert_id,
                   GPU,
                   ALL_LAYOUT,
                   phi::BuildSrcRankAndLocalExpertIdKernel,
                   int,
                   int64_t) {}
