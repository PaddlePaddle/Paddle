// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

// The file has been adapted from DeepSeek DeepEP project
// Copyright (c) 2025 DeepSeek
// Licensed under the MIT License -
// https://github.com/deepseek-ai/DeepEP/blob/main/LICENSE

#include "paddle/fluid/distributed/collective/deep_ep/kernels/buffer.cuh"
#include "paddle/fluid/distributed/collective/deep_ep/kernels/configs.cuh"
#include "paddle/fluid/distributed/collective/deep_ep/kernels/exception.cuh"
#include "paddle/fluid/distributed/collective/deep_ep/kernels/launch.cuh"
#include "paddle/fluid/distributed/collective/deep_ep/kernels/utils.cuh"

namespace deep_ep {

namespace internode {

// extern nvshmem_team_t cpu_rdma_team;

template <int kNumThreads, int kNumExpertsPerSM, int kNumRanksPerSM>
__global__ void __launch_bounds__(kNumThreads, 1)
    get_dispatch_layout(const int64_t* topk_idx,
                        int* num_tokens_per_rank,
                        int* num_tokens_per_rdma_rank,
                        int* num_tokens_per_expert,
                        bool* is_token_in_rank,
                        int num_tokens,
                        int num_topk,
                        int num_ranks,
                        int num_experts) {
  auto sm_id = static_cast<int>(blockIdx.x);
  auto thread_id = static_cast<int>(threadIdx.x);

  // Count expert statistics
  __shared__ int num_tokens_per_expert_per_thread[kNumThreads]
                                                 [kNumExpertsPerSM];
  int expert_begin_idx = sm_id * kNumExpertsPerSM,
      expert_end_idx = min(expert_begin_idx + kNumExpertsPerSM, num_experts);
  if (expert_begin_idx < expert_end_idx) {
// Per-thread count
#pragma unroll
    for (int i = 0; i < kNumExpertsPerSM; ++i)
      num_tokens_per_expert_per_thread[thread_id][i] = 0;
#pragma unroll
    for (int i = thread_id; i < num_tokens; i += kNumThreads) {
      auto shifted_topk_idx = topk_idx + i * num_topk;
#pragma unroll
      for (int j = 0, expert_idx; j < num_topk; ++j) {
        expert_idx = static_cast<int>(shifted_topk_idx[j]);
        if (expert_begin_idx <= expert_idx && expert_idx < expert_end_idx)
          ++num_tokens_per_expert_per_thread[thread_id]
                                            [expert_idx - expert_begin_idx];
      }
    }
    __syncthreads();

    // Sum up
    EP_STATIC_ASSERT(kNumExpertsPerSM <= kNumThreads,
                     "Too many experts per SM");
    if (expert_begin_idx + thread_id < expert_end_idx) {
      int sum = 0;
#pragma unroll
      for (int i = 0; i < kNumThreads; ++i)
        sum += num_tokens_per_expert_per_thread[i][thread_id];
      num_tokens_per_expert[expert_begin_idx + thread_id] = sum;
    }
    return;
  }

  if (num_tokens_per_rdma_rank != nullptr)
    EP_DEVICE_ASSERT(num_ranks % NUM_MAX_NVL_PEERS == 0 &&
                     num_ranks > NUM_MAX_NVL_PEERS);

  // Count rank statistics
  constexpr int kNumRDMARanksPerSM = kNumRanksPerSM / NUM_MAX_NVL_PEERS;
  __shared__ int num_tokens_per_rank_per_thread[kNumThreads][kNumRanksPerSM];
  __shared__ int num_tokens_per_rdma_rank_per_thread[kNumThreads]
                                                    [kNumRDMARanksPerSM];
  auto sm_begin = (num_experts + kNumExpertsPerSM - 1) / kNumExpertsPerSM;
  int rank_begin_idx = (sm_id - sm_begin) * kNumRanksPerSM,
      rank_end_idx = min(rank_begin_idx + kNumRanksPerSM, num_ranks);
  int rdma_rank_begin_idx = rank_begin_idx / NUM_MAX_NVL_PEERS,
      rdma_rank_end_idx = rank_end_idx / NUM_MAX_NVL_PEERS;
  if (rank_begin_idx < rank_end_idx) {
    const auto num_expert_per_rank = num_experts / num_ranks;
    auto expert_begin = rank_begin_idx * num_expert_per_rank;
    auto expert_end = rank_end_idx * num_expert_per_rank;

// Per-thread count
#pragma unroll
    for (int i = 0; i < kNumRanksPerSM; ++i)
      num_tokens_per_rank_per_thread[thread_id][i] = 0;
#pragma unroll
    for (int i = 0; i < kNumRDMARanksPerSM; ++i)
      num_tokens_per_rdma_rank_per_thread[thread_id][i] = 0;
#pragma unroll
    for (int i = thread_id; i < num_tokens; i += kNumThreads) {
      auto shifted_topk_idx = topk_idx + i * num_topk;
      int is_in_rank[kNumRanksPerSM] = {0},
          is_in_rdma_rank[kNumRDMARanksPerSM] = {0};
#pragma unroll
      for (int j = 0, expert_idx, rank_idx; j < num_topk; ++j) {
        expert_idx = static_cast<int>(shifted_topk_idx[j]);
        if (expert_begin <= expert_idx && expert_idx < expert_end) {
          // Count single rank
          rank_idx = expert_idx / num_expert_per_rank - rank_begin_idx;
          is_in_rank[rank_idx]++,
              is_in_rdma_rank[rank_idx / NUM_MAX_NVL_PEERS]++;
        }
      }

      auto shifted_is_token_in_rank = is_token_in_rank + i * num_ranks;
#pragma unroll
      for (int j = 0; j + rank_begin_idx < rank_end_idx; ++j) {
        shifted_is_token_in_rank[j + rank_begin_idx] = (is_in_rank[j] > 0);
        num_tokens_per_rank_per_thread[thread_id][j] += (is_in_rank[j] > 0);
      }

#pragma unroll
      for (int j = 0; j + rdma_rank_begin_idx < rdma_rank_end_idx; ++j)
        num_tokens_per_rdma_rank_per_thread[thread_id][j] +=
            (is_in_rdma_rank[j] > 0);
    }
    __syncthreads();

    // Sum up
    EP_STATIC_ASSERT(kNumRanksPerSM <= kNumThreads, "Too many ranks per SM");
    if (rank_begin_idx + thread_id < rank_end_idx) {
      int sum = 0;
#pragma unroll
      for (int i = 0; i < kNumThreads; ++i)
        sum += num_tokens_per_rank_per_thread[i][thread_id];
      num_tokens_per_rank[rank_begin_idx + thread_id] = sum;
    }

    if (num_tokens_per_rdma_rank != nullptr &&
        rdma_rank_begin_idx + thread_id < rdma_rank_end_idx) {
      int sum = 0;
#pragma unroll
      for (int i = 0; i < kNumThreads; ++i)
        sum += num_tokens_per_rdma_rank_per_thread[i][thread_id];
      num_tokens_per_rdma_rank[rdma_rank_begin_idx + thread_id] = sum;
    }
  }
}

void get_dispatch_layout(const int64_t* topk_idx,
                         int* num_tokens_per_rank,
                         int* num_tokens_per_rdma_rank,
                         int* num_tokens_per_expert,
                         bool* is_token_in_rank,
                         int num_tokens,
                         int num_topk,
                         int num_ranks,
                         int num_experts,
                         cudaStream_t stream) {
  constexpr int kNumThreads = 256, kNumExpertsPerSM = 32, kNumRanksPerSM = 8;
  int num_sms = ((num_experts + kNumExpertsPerSM - 1) / kNumExpertsPerSM) +
                (num_ranks + kNumRanksPerSM - 1) / kNumRanksPerSM;
  EP_STATIC_ASSERT(kNumExpertsPerSM % NUM_MAX_NVL_PEERS == 0,
                   "Invalid number of experts per SM");

  SETUP_LAUNCH_CONFIG(num_sms, kNumThreads, stream);
  LAUNCH_KERNEL(
      &cfg,
      (get_dispatch_layout<kNumThreads, kNumExpertsPerSM, kNumRanksPerSM>),
      topk_idx,
      num_tokens_per_rank,
      num_tokens_per_rdma_rank,
      num_tokens_per_expert,
      is_token_in_rank,
      num_tokens,
      num_topk,
      num_ranks,
      num_experts);
}

struct SourceMeta {
  int src_rdma_rank, is_token_in_nvl_rank_bits;

  EP_STATIC_ASSERT(NUM_MAX_NVL_PEERS == 8,
                   "Invalid number of maximum NVL peers");

  __forceinline__ SourceMeta() = default;

  // TODO(Hongqing-work): faster encoding
  __device__ __forceinline__ SourceMeta(int rdma_rank,
                                        const bool* is_token_in_nvl_ranks) {
    src_rdma_rank = rdma_rank;
    is_token_in_nvl_rank_bits = is_token_in_nvl_ranks[0];
#pragma unroll
    for (int i = 1; i < NUM_MAX_NVL_PEERS; ++i)
      is_token_in_nvl_rank_bits |= is_token_in_nvl_ranks[i] << i;
  }

  __device__ __forceinline__ bool is_token_in_nvl_rank(int nvl_rank) const {
    return (is_token_in_nvl_rank_bits >> nvl_rank) & 1;
  }
};

EP_STATIC_ASSERT(sizeof(SourceMeta) % sizeof(int) == 0,
                 "Invalid size of `SourceMeta`");

int get_source_meta_bytes() { return sizeof(SourceMeta); }

}  // namespace internode

}  // namespace deep_ep
