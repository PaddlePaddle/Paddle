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

#include <cstring>
#include <vector>

#ifdef PADDLE_WITH_NVSHMEM
// clang-format off
#include <nvshmem.h>
#include <nvshmemx.h>
#include <infiniband/mlx5dv.h>
#include <non_abi/device/threadgroup/nvshmemi_common_device_defines.cuh>
#include <device_host_transport/nvshmem_common_ibgda.h>
// clang-format on
#endif

#include "paddle/fluid/distributed/collective/flash_ep/kernels/configs.cuh"
#include "paddle/fluid/distributed/collective/flash_ep/kernels/exception.cuh"
#include "paddle/fluid/distributed/collective/flash_ep/kernels/launch.cuh"
#include "paddle/fluid/distributed/collective/flash_ep/kernels/utils.cuh"

#ifdef PADDLE_WITH_NVSHMEM
#include "paddle/fluid/distributed/collective/flash_ep/kernels/ibgda_device.cuh"
#endif

namespace flash_ep {

namespace intranode {

template <int kNumRanks>
__global__ void barrier(int** task_fifo_ptrs, int head, int rank) {
  barrier_device<kNumRanks>(task_fifo_ptrs, head, rank);
}

void barrier(int** task_fifo_ptrs,
             int head,
             int rank,
             int num_ranks,
             cudaStream_t stream) {
#define BARRIER_LAUNCH_CASE(ranks)                                 \
  LAUNCH_KERNEL(&cfg, barrier<ranks>, task_fifo_ptrs, head, rank); \
  break

  SETUP_LAUNCH_CONFIG(1, 32, stream);
  SWITCH_RANKS(BARRIER_LAUNCH_CASE);
#undef BARRIER_LAUNCH_CASE
}

}  // namespace intranode

namespace internode {

template <int kNumRDMARanks>
__global__ void get_flash_ep_coalesce_rdma_schedule_kernel(
    const int64_t* topk_idx,
    const int* local_expert_to_stage_map,
    int* dispatch_rdma_schedule_map,
    int* combine_rdma_schedule_map,
    int num_tokens,
    int num_topk,
    int num_ranks,
    int num_experts,
    int num_loop_stage) {
  int token_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (token_idx >= num_tokens) return;

  int dispatch_rdma_schedule_map_fragment[kNumRDMARanks];  // 第一次在某rdma
                                                           // rank出现的轮数
  int combine_rdma_schedule_map_fragment[kNumRDMARanks];  // 最后一次在某rdma
                                                          // rank出现的轮数

  auto shifted_topk_idx = topk_idx + token_idx * num_topk;

  int num_experts_per_rank = num_experts / num_ranks;
  int num_rdma_ranks = num_ranks / NUM_MAX_NVL_PEERS;

#pragma unroll
  for (int i = 0; i < kNumRDMARanks; ++i) {
    dispatch_rdma_schedule_map_fragment[i] = num_loop_stage;
    combine_rdma_schedule_map_fragment[i] = -1;
  }

  for (int i = 0; i < num_topk; ++i) {
    int expert_idx =
        static_cast<int>(shifted_topk_idx[i]);  // topk专家中的第i个
    if (expert_idx < 0 || expert_idx >= num_experts) continue;
    int rank_idx = (expert_idx / num_experts_per_rank) % num_ranks;
    int rdma_rank_idx = rank_idx / NUM_MAX_NVL_PEERS;
    int local_expert_idx = expert_idx % num_experts_per_rank;
    auto loop_idx = reinterpret_cast<const int2*>(
        local_expert_to_stage_map)[local_expert_idx];
    int dispatch_loop_idx = loop_idx.x;
    int combine_loop_idx = loop_idx.y;
    EP_DEVICE_ASSERT(dispatch_loop_idx >= 0 &&
                     dispatch_loop_idx < num_loop_stage);
    EP_DEVICE_ASSERT(combine_loop_idx >= 0 &&
                     combine_loop_idx < num_loop_stage);
    dispatch_rdma_schedule_map_fragment[rdma_rank_idx] = min(
        dispatch_loop_idx, dispatch_rdma_schedule_map_fragment[rdma_rank_idx]);
    combine_rdma_schedule_map_fragment[rdma_rank_idx] = max(
        combine_loop_idx, combine_rdma_schedule_map_fragment[rdma_rank_idx]);
  }
  for (int i = 0; i < num_rdma_ranks; ++i) {
    dispatch_rdma_schedule_map[token_idx * num_rdma_ranks + i] =
        dispatch_rdma_schedule_map_fragment[i];
    combine_rdma_schedule_map[token_idx * num_rdma_ranks + i] =
        combine_rdma_schedule_map_fragment[i];
  }
}

template <int kNumThreads,
          int kNumExpertsPerSM,
          int kNumRanksPerSM,
          int kNumRDMARanks>
__global__ void __launch_bounds__(kNumThreads, 1)
    get_flash_ep_coalesce_rdma_layout_kernel(
        const int64_t* topk_idx,
        const int* dispatch_rdma_schedule_map,
        const int* combine_rdma_schedule_map,
        int* num_tokens_per_rank,
        int* num_tokens_per_rdma_rank,
        int* num_tokens_per_expert,
        bool* is_token_in_rank,
        int num_tokens,
        int num_topk,
        int num_ranks,
        int num_experts,
        int num_sms_per_loop) {
  int sm_id = static_cast<int>(blockIdx.x);
  int loop_idx = sm_id / num_sms_per_loop;
  int sm_id_in_loop = sm_id % num_sms_per_loop;
  int thread_id = static_cast<int>(threadIdx.x);

  int num_rdma_ranks = num_ranks / NUM_MAX_NVL_PEERS;

  num_tokens_per_rank += loop_idx * 2 * num_ranks;
  num_tokens_per_rdma_rank += loop_idx * 2 * num_rdma_ranks;
  num_tokens_per_expert += loop_idx * 2 * num_experts;
  is_token_in_rank += loop_idx * 2 * num_tokens * num_ranks;

  // 统计专家级别发送指标, 每个sm负责一些experts
  __shared__ int dispatch_num_tokens_per_expert_per_thread
      [kNumThreads][kNumExpertsPerSM];  // 每个线程统计一部分token, 最后再聚合
  __shared__ int combine_num_tokens_per_expert_per_thread[kNumThreads]
                                                         [kNumExpertsPerSM];

  // 计算自己的这个block负责哪些目标expert
  int expert_begin_idx = sm_id_in_loop * kNumExpertsPerSM,
      expert_end_idx = min(expert_begin_idx + kNumExpertsPerSM, num_experts);

  int num_experts_per_rank = num_experts / num_ranks;
  bool vectorized_load_schedule_map = kNumRDMARanks % 4 == 0;

  if (expert_begin_idx < expert_end_idx) {
// Per-thread count
#pragma unroll
    for (int i = 0; i < kNumExpertsPerSM; ++i) {
      dispatch_num_tokens_per_expert_per_thread[thread_id][i] = 0;
      combine_num_tokens_per_expert_per_thread[thread_id][i] = 0;
    }

    int dispatch_schedule_map_fragment[kNumRDMARanks];
    int combine_schedule_map_fragment[kNumRDMARanks];

#pragma unroll
    for (int i = thread_id; i < num_tokens; i += kNumThreads) {
      auto shifted_topk_idx =
          topk_idx + i * num_topk;  // 这个token对应的topk_idx起始地址
      auto shifted_dispatch_schedule_map =
          dispatch_rdma_schedule_map + i * num_rdma_ranks;
      auto shifted_combine_schedule_map =
          combine_rdma_schedule_map + i * num_rdma_ranks;

      if (vectorized_load_schedule_map) {
        for (int j = 0; j < kNumRDMARanks / 4; j++) {
          const int4* dispatch_schedule_map_vec =
              reinterpret_cast<const int4*>(shifted_dispatch_schedule_map) + j;
          const int4* combine_schedule_map_vec =
              reinterpret_cast<const int4*>(shifted_combine_schedule_map) + j;
#pragma unroll
          for (int k = 0; k < 4; k++) {
            dispatch_schedule_map_fragment[j * 4 + k] =
                reinterpret_cast<const int*>(dispatch_schedule_map_vec)[k];
            combine_schedule_map_fragment[j * 4 + k] =
                reinterpret_cast<const int*>(combine_schedule_map_vec)[k];
          }
        }
      } else {
        for (int j = 0; j < kNumRDMARanks; j++) {
          dispatch_schedule_map_fragment[j] = shifted_dispatch_schedule_map[j];
          combine_schedule_map_fragment[j] = shifted_combine_schedule_map[j];
        }
      }

#pragma unroll
      for (int j = 0; j < num_topk; ++j) {
        int expert_idx =
            static_cast<int>(shifted_topk_idx[j]);  // topk专家中的第j个
        int local_expert_idx = expert_idx % num_experts_per_rank;
        int rank_idx = expert_idx / num_experts_per_rank;
        int rdma_rank_idx = rank_idx / NUM_MAX_NVL_PEERS;

        bool in_sm_range =
            expert_begin_idx <= expert_idx &&
            expert_idx < expert_end_idx;  // 表示这个专家是否是当前sm负责
        if (in_sm_range &&
            dispatch_schedule_map_fragment[rdma_rank_idx] == loop_idx) {
          ++dispatch_num_tokens_per_expert_per_thread[thread_id]
                                                     [expert_idx -
                                                      expert_begin_idx];
        }
        if (in_sm_range &&
            combine_schedule_map_fragment[rdma_rank_idx] == loop_idx) {
          ++combine_num_tokens_per_expert_per_thread[thread_id]
                                                    [expert_idx -
                                                     expert_begin_idx];
        }
      }
    }
    __syncthreads();

    // Sum up
    EP_STATIC_ASSERT(kNumExpertsPerSM <= kNumThreads,
                     "Too many experts per SM");

    if (expert_begin_idx + thread_id < expert_end_idx) {
      int dispatch_sum = 0;
      int combine_sum = 0;
#pragma unroll
      for (int i = 0; i < kNumThreads; ++i) {
        dispatch_sum += dispatch_num_tokens_per_expert_per_thread[i][thread_id];
        combine_sum += combine_num_tokens_per_expert_per_thread[i][thread_id];
      }
      num_tokens_per_expert[expert_begin_idx + thread_id] = dispatch_sum;
      num_tokens_per_expert[num_experts + expert_begin_idx + thread_id] =
          combine_sum;
    }
    return;
  }

  if (num_tokens_per_rdma_rank != nullptr)
    EP_DEVICE_ASSERT(num_ranks % NUM_MAX_NVL_PEERS == 0 &&
                     num_ranks > NUM_MAX_NVL_PEERS);

  // 统计rank级别发送指标, 每个sm负责一些ranks
  constexpr int kNumRDMARanksPerSM = kNumRanksPerSM / NUM_MAX_NVL_PEERS;
  __shared__ int dispatch_num_tokens_per_rank_per_thread[kNumThreads]
                                                        [kNumRanksPerSM];
  __shared__ int
      dispatch_num_tokens_per_rdma_rank_per_thread[kNumThreads]
                                                  [kNumRDMARanksPerSM];
  __shared__ int combine_num_tokens_per_rank_per_thread[kNumThreads]
                                                       [kNumRanksPerSM];
  __shared__ int
      combine_num_tokens_per_rdma_rank_per_thread[kNumThreads]
                                                 [kNumRDMARanksPerSM];

  auto sm_begin =
      (num_experts + kNumExpertsPerSM - 1) /
      kNumExpertsPerSM;  // 因为前kNumExpertsPerSM个线程前面就return了,
                         // 所以这里重新分配一下sm
  int rank_begin_idx = (sm_id_in_loop - sm_begin) * kNumRanksPerSM,
      rank_end_idx = rank_begin_idx + kNumRanksPerSM;
  EP_DEVICE_ASSERT(rank_end_idx <= num_ranks);
  int rdma_rank_begin_idx = rank_begin_idx / NUM_MAX_NVL_PEERS,
      rdma_rank_end_idx = rank_end_idx / NUM_MAX_NVL_PEERS;
  if (rank_begin_idx < rank_end_idx) {
    const auto num_expert_per_rank = num_experts / num_ranks;
    auto expert_begin =
        rank_begin_idx * num_expert_per_rank;  // 全局视角下的expert_id
    auto expert_end = rank_end_idx * num_expert_per_rank;

// Per-thread count
// 初始化置零
#pragma unroll
    for (int i = 0; i < kNumRanksPerSM; ++i) {
      dispatch_num_tokens_per_rank_per_thread[thread_id][i] = 0;
      combine_num_tokens_per_rank_per_thread[thread_id][i] = 0;
    }
#pragma unroll
    for (int i = 0; i < kNumRDMARanksPerSM; ++i) {
      dispatch_num_tokens_per_rdma_rank_per_thread[thread_id][i] = 0;
      combine_num_tokens_per_rdma_rank_per_thread[thread_id][i] = 0;
    }

    int dispatch_schedule_map_fragment[kNumRDMARanks];
    int combine_schedule_map_fragment[kNumRDMARanks];

#pragma unroll
    // 遍历所有token
    for (int i = thread_id; i < num_tokens; i += kNumThreads) {
      auto shifted_topk_idx =
          topk_idx + i * num_topk;  // 这个token对应的topk_idx起始地址
      auto shifted_dispatch_schedule_map =
          dispatch_rdma_schedule_map + i * num_rdma_ranks;
      auto shifted_combine_schedule_map =
          combine_rdma_schedule_map + i * num_rdma_ranks;

      if (vectorized_load_schedule_map) {
        for (int j = 0; j < kNumRDMARanks / 4; j++) {
          const int4* dispatch_schedule_map_vec =
              reinterpret_cast<const int4*>(shifted_dispatch_schedule_map) + j;
          const int4* combine_schedule_map_vec =
              reinterpret_cast<const int4*>(shifted_combine_schedule_map) + j;
#pragma unroll
          for (int k = 0; k < 4; k++) {
            dispatch_schedule_map_fragment[j * 4 + k] =
                reinterpret_cast<const int*>(dispatch_schedule_map_vec)[k];
            combine_schedule_map_fragment[j * 4 + k] =
                reinterpret_cast<const int*>(combine_schedule_map_vec)[k];
          }
        }
      } else {
        for (int j = 0; j < kNumRDMARanks; j++) {
          dispatch_schedule_map_fragment[j] = shifted_dispatch_schedule_map[j];
          combine_schedule_map_fragment[j] = shifted_combine_schedule_map[j];
        }
      }

      int dispatch_is_in_rank[kNumRanksPerSM] = {0};
      int combine_is_in_rank[kNumRanksPerSM] = {0};
      int dispatch_is_in_rdma_rank[kNumRDMARanksPerSM] = {0};
      int combine_is_in_rdma_rank[kNumRDMARanksPerSM] = {0};
#pragma unroll
      // 遍历这个token的topk
      for (int j = 0; j < num_topk; ++j) {
        int expert_idx = static_cast<int>(shifted_topk_idx[j]);
        bool in_sm_range =
            expert_begin <= expert_idx && expert_idx < expert_end;
        int rank_idx = expert_idx / num_expert_per_rank -
                       rank_begin_idx;  // SM负责的相对rank_idx
        int rdma_rank_idx =
            (expert_idx / num_experts_per_rank) / NUM_MAX_NVL_PEERS;
        if (in_sm_range &&
            dispatch_schedule_map_fragment[rdma_rank_idx] == loop_idx) {
          dispatch_is_in_rank[rank_idx]++;
          dispatch_is_in_rdma_rank[rank_idx / NUM_MAX_NVL_PEERS]++;
        }
        if (in_sm_range &&
            combine_schedule_map_fragment[rdma_rank_idx] == loop_idx) {
          combine_is_in_rank[rank_idx]++;
          combine_is_in_rdma_rank[rank_idx / NUM_MAX_NVL_PEERS]++;
        }
      }

      EP_STATIC_ASSERT(kNumRanksPerSM == 8,
                       "Not match the vectorized memory access");
      EP_DEVICE_ASSERT(rank_end_idx - rank_begin_idx == 8);
      auto dispatch_shifted_is_token_in_rank = is_token_in_rank + i * num_ranks;
      auto combine_shifted_is_token_in_rank =
          is_token_in_rank + num_tokens * num_ranks + i * num_ranks;

      int2* dispatch_shifted_is_token_in_rank_vec = reinterpret_cast<int2*>(
          dispatch_shifted_is_token_in_rank + rank_begin_idx);
      int2* combine_shifted_is_token_in_rank_vec = reinterpret_cast<int2*>(
          combine_shifted_is_token_in_rank + rank_begin_idx);
      int2 dispatch_vec, combine_vec;
      bool* dispatch_vec_ptr = reinterpret_cast<bool*>(&dispatch_vec);
      bool* combine_vec_ptr = reinterpret_cast<bool*>(&combine_vec);

#pragma unroll
      for (int j = 0; j + rank_begin_idx < rank_end_idx; ++j) {
        dispatch_vec_ptr[j] = (dispatch_is_in_rank[j] > 0);
        combine_vec_ptr[j] = (combine_is_in_rank[j] > 0);
      }
      *dispatch_shifted_is_token_in_rank_vec = dispatch_vec;
      *combine_shifted_is_token_in_rank_vec = combine_vec;

#pragma unroll
      for (int j = 0; j + rank_begin_idx < rank_end_idx; ++j) {
        dispatch_num_tokens_per_rank_per_thread[thread_id][j] +=
            (dispatch_is_in_rank[j] > 0);
        combine_num_tokens_per_rank_per_thread[thread_id][j] +=
            (combine_is_in_rank[j] > 0);
      }

#pragma unroll
      for (int j = 0; j + rdma_rank_begin_idx < rdma_rank_end_idx; ++j) {
        dispatch_num_tokens_per_rdma_rank_per_thread[thread_id][j] +=
            (dispatch_is_in_rdma_rank[j] > 0);
        combine_num_tokens_per_rdma_rank_per_thread[thread_id][j] +=
            (combine_is_in_rdma_rank[j] > 0);
      }
    }
    __syncthreads();

    // Sum up
    EP_STATIC_ASSERT(kNumRanksPerSM <= kNumThreads, "Too many ranks per SM");

    // 前kNumRanksPerSM个线程负责规约
    if (rank_begin_idx + thread_id < rank_end_idx) {
      int dispatch_sum = 0;
      int combine_sum = 0;
#pragma unroll
      for (int i = 0; i < kNumThreads; ++i) {
        dispatch_sum += dispatch_num_tokens_per_rank_per_thread[i][thread_id];
        combine_sum += combine_num_tokens_per_rank_per_thread[i][thread_id];
      }
      num_tokens_per_rank[rank_begin_idx + thread_id] = dispatch_sum;
      num_tokens_per_rank[num_ranks + rank_begin_idx + thread_id] = combine_sum;
    }

    if (num_tokens_per_rdma_rank != nullptr &&
        rdma_rank_begin_idx + thread_id < rdma_rank_end_idx) {
      int dispatch_sum = 0;
      int combine_sum = 0;
#pragma unroll
      for (int i = 0; i < kNumThreads; ++i) {
        dispatch_sum +=
            dispatch_num_tokens_per_rdma_rank_per_thread[i][thread_id];
        combine_sum +=
            combine_num_tokens_per_rdma_rank_per_thread[i][thread_id];
      }
      int num_rdma_ranks = num_ranks / NUM_MAX_NVL_PEERS;
      num_tokens_per_rdma_rank[rdma_rank_begin_idx + thread_id] = dispatch_sum;
      num_tokens_per_rdma_rank[num_rdma_ranks + rdma_rank_begin_idx +
                               thread_id] = combine_sum;
    }
  }
}

void get_flash_ep_coalesce_rdma_layout(const int64_t* topk_idx,
                                       const int* dispatch_rdma_schedule_map,
                                       const int* combine_rdma_schedule_map,
                                       int* num_tokens_per_rank,
                                       int* num_tokens_per_rdma_rank,
                                       int* num_tokens_per_expert,
                                       bool* is_token_in_rank,
                                       int num_tokens,
                                       int num_topk,
                                       int num_ranks,
                                       int num_experts,
                                       int num_loop_stage,
                                       cudaStream_t stream) {
  constexpr int kNumThreads = 256;
  constexpr int kNumExpertsPerSM = 8;
  constexpr int kNumRanksPerSM = 8;

  int num_sms_per_loop =
      ((num_experts + kNumExpertsPerSM - 1) / kNumExpertsPerSM) +
      (num_ranks + kNumRanksPerSM - 1) / kNumRanksPerSM;
  int num_sms = num_sms_per_loop * num_loop_stage;

  static_assert(kNumExpertsPerSM % NUM_MAX_NVL_PEERS == 0,
                "Invalid number of experts per SM");

#define GET_LAYOUT_LAUNCH_CASE(num_rdma_ranks)                     \
  {                                                                \
    auto layout_func =                                             \
        get_flash_ep_coalesce_rdma_layout_kernel<kNumThreads,      \
                                                 kNumExpertsPerSM, \
                                                 kNumRanksPerSM,   \
                                                 num_rdma_ranks>;  \
    LAUNCH_KERNEL(&cfg,                                            \
                  layout_func,                                     \
                  topk_idx,                                        \
                  dispatch_rdma_schedule_map,                      \
                  combine_rdma_schedule_map,                       \
                  num_tokens_per_rank,                             \
                  num_tokens_per_rdma_rank,                        \
                  num_tokens_per_expert,                           \
                  is_token_in_rank,                                \
                  num_tokens,                                      \
                  num_topk,                                        \
                  num_ranks,                                       \
                  num_experts,                                     \
                  num_sms_per_loop);                               \
  }                                                                \
  break

  SETUP_LAUNCH_CONFIG(num_sms, kNumThreads, stream);
  SWITCH_RDMA_RANKS(GET_LAYOUT_LAUNCH_CASE);

#undef GET_LAYOUT_LAUNCH_CASE
}

void get_flash_ep_coalesce_rdma_schedule(const int64_t* topk_idx,
                                         const int* local_expert_to_stage_map,
                                         int* dispatch_rdma_schedule_map,
                                         int* combine_rdma_schedule_map,
                                         const int num_ranks,
                                         const int num_experts,
                                         const int num_loop_stage,
                                         const int num_tokens,
                                         const int num_topk,
                                         cudaStream_t stream) {
  int num_experts_per_rank = num_experts / num_ranks;
  int num_rdma_ranks = num_ranks / NUM_MAX_NVL_PEERS;

  // 每个线程负责一个token
  constexpr int64_t kNumThreads = 256;
  const int64_t num_sms = (num_tokens + kNumThreads - 1) / kNumThreads;

#define GET_SCHEDULE_LAUNCH_CASE(num_rdma_ranks)                    \
  {                                                                 \
    auto schedule_func =                                            \
        get_flash_ep_coalesce_rdma_schedule_kernel<num_rdma_ranks>; \
    LAUNCH_KERNEL(&cfg,                                             \
                  schedule_func,                                    \
                  topk_idx,                                         \
                  local_expert_to_stage_map,                        \
                  dispatch_rdma_schedule_map,                       \
                  combine_rdma_schedule_map,                        \
                  num_tokens,                                       \
                  num_topk,                                         \
                  num_ranks,                                        \
                  num_experts,                                      \
                  num_loop_stage);                                  \
  }                                                                 \
  break

  SETUP_LAUNCH_CONFIG(num_sms, kNumThreads, stream);
  SWITCH_RDMA_RANKS(GET_SCHEDULE_LAUNCH_CASE);

#undef GET_SCHEDULE_LAUNCH_CASE
}

#ifdef PADDLE_WITH_NVSHMEM
nvshmem_team_t cpu_rdma_team = NVSHMEM_TEAM_INVALID;
nvshmem_team_config_t cpu_rdma_team_config;

std::vector<uint8_t> get_unique_id() {
  nvshmemx_uniqueid_t unique_id;
  nvshmemx_get_uniqueid(&unique_id);
  std::vector<uint8_t> result(sizeof(nvshmemx_uniqueid_t));
  std::memcpy(result.data(), &unique_id, sizeof(nvshmemx_uniqueid_t));
  return result;
}

int init(const std::vector<uint8_t>& root_unique_id_val,
         int rank,
         int num_ranks,
         bool low_latency_mode) {
  nvshmemx_uniqueid_t root_unique_id;
  nvshmemx_init_attr_t attr;
  std::memcpy(
      &root_unique_id, root_unique_id_val.data(), sizeof(nvshmemx_uniqueid_t));
  nvshmemx_set_attr_uniqueid_args(rank, num_ranks, &root_unique_id, &attr);
  nvshmemx_init_attr(NVSHMEMX_INIT_WITH_UNIQUEID, &attr);

  // Create sub-RDMA teams
  // NOTES: if `num_ranks <= NUM_MAX_NVL_PEERS` then only low-latency kernels
  // are used
  if (low_latency_mode && num_ranks > NUM_MAX_NVL_PEERS) {
    EP_HOST_ASSERT(cpu_rdma_team == NVSHMEM_TEAM_INVALID);
    EP_HOST_ASSERT(num_ranks % NUM_MAX_NVL_PEERS == 0);
    EP_HOST_ASSERT(nvshmem_team_split_strided(NVSHMEM_TEAM_WORLD,
                                              rank % NUM_MAX_NVL_PEERS,
                                              NUM_MAX_NVL_PEERS,
                                              num_ranks / NUM_MAX_NVL_PEERS,
                                              &cpu_rdma_team_config,
                                              0,
                                              &cpu_rdma_team) == 0);
    EP_HOST_ASSERT(cpu_rdma_team != NVSHMEM_TEAM_INVALID);
  }

  // TODO(DeepEP): we still use `nvshmem_barrier` under IBRC mode, which should
  // be switch to IBGDA mode later
  nvshmemi_device_host_state_t* dev_state_ptr = nullptr;
  CUDA_CHECK(cudaGetSymbolAddress(reinterpret_cast<void**>(&dev_state_ptr),
                                  nvshmemi_device_state_d));

  bool ibgda_is_initialized = false;
  CUDA_CHECK(cudaMemcpy(&dev_state_ptr->ibgda_is_initialized,
                        &ibgda_is_initialized,
                        sizeof(bool),
                        cudaMemcpyHostToDevice));
  nvshmem_barrier_all();
  return nvshmem_my_pe();
}

void* alloc(size_t size, size_t alignment) {
  return nvshmem_align(alignment, size);
}

void free(void* ptr) { nvshmem_free(ptr); }

void barrier() { nvshmem_barrier_all(); }

void finalize() {
  if (cpu_rdma_team != NVSHMEM_TEAM_INVALID) {
    nvshmem_team_destroy(cpu_rdma_team);
    cpu_rdma_team = NVSHMEM_TEAM_INVALID;
  }
  nvshmem_finalize();
}
#endif  // PADDLE_WITH_NVSHMEM

}  // namespace internode
}  // namespace flash_ep
