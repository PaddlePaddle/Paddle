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
    int num_pipeline_stages) {
  int token_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (token_idx >= num_tokens) return;

  int rmem_dispatch_rdma_schedule_map[kNumRDMARanks];  // 第一次在某rdma
                                                       // rank出现的轮数
  int rmem_combine_rdma_schedule_map[kNumRDMARanks];   // 最后一次在某rdma
                                                       // rank出现的轮数

  auto shifted_topk_idx = topk_idx + token_idx * num_topk;

  int num_experts_per_rank = num_experts / num_ranks;
  int num_rdma_ranks = num_ranks / NUM_MAX_NVL_PEERS;

#pragma unroll
  for (int i = 0; i < kNumRDMARanks; ++i) {
    rmem_dispatch_rdma_schedule_map[i] = num_pipeline_stages;
    rmem_combine_rdma_schedule_map[i] = -1;
  }

  for (int i = 0; i < num_topk; ++i) {
    int expert_idx =
        static_cast<int>(shifted_topk_idx[i]);  // topk专家中的第i个
    if (expert_idx < 0 || expert_idx >= num_experts) continue;
    int rank_idx = (expert_idx / num_experts_per_rank) % num_ranks;
    int rdma_rank_idx = rank_idx / NUM_MAX_NVL_PEERS;
    int local_expert_idx = expert_idx % num_experts_per_rank;
    auto stage_idx = reinterpret_cast<const int2*>(
        local_expert_to_stage_map)[local_expert_idx];
    int dispatch_stage_idx = stage_idx.x;
    int combine_stage_idx = stage_idx.y;
    EP_DEVICE_ASSERT(dispatch_stage_idx >= 0 &&
                     dispatch_stage_idx < num_pipeline_stages);
    EP_DEVICE_ASSERT(combine_stage_idx >= 0 &&
                     combine_stage_idx < num_pipeline_stages);
    rmem_dispatch_rdma_schedule_map[rdma_rank_idx] =
        min(dispatch_stage_idx, rmem_dispatch_rdma_schedule_map[rdma_rank_idx]);
    rmem_combine_rdma_schedule_map[rdma_rank_idx] =
        max(combine_stage_idx, rmem_combine_rdma_schedule_map[rdma_rank_idx]);
  }
  for (int i = 0; i < num_rdma_ranks; ++i) {
    dispatch_rdma_schedule_map[token_idx * num_rdma_ranks + i] =
        rmem_dispatch_rdma_schedule_map[i];
    combine_rdma_schedule_map[token_idx * num_rdma_ranks + i] =
        rmem_combine_rdma_schedule_map[i];
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
        int num_sms_per_loop,
        int num_pipeline_stages) {
  int sm_id = static_cast<int>(blockIdx.x);
  int stage_idx = sm_id / num_sms_per_loop;
  int sm_id_in_loop = sm_id % num_sms_per_loop;
  int thread_id = static_cast<int>(threadIdx.x);

  int num_rdma_ranks = num_ranks / NUM_MAX_NVL_PEERS;

  auto dispatch_num_tokens_per_rank =
      num_tokens_per_rank + stage_idx * num_ranks;
  auto dispatch_num_tokens_per_rdma_rank =
      num_tokens_per_rdma_rank + stage_idx * num_rdma_ranks;
  auto dispatch_num_tokens_per_expert =
      num_tokens_per_expert + stage_idx * num_experts;
  auto dispatch_is_token_in_rank =
      is_token_in_rank + stage_idx * num_tokens * num_ranks;
  auto combine_num_tokens_per_rank =
      num_tokens_per_rank + (num_pipeline_stages + stage_idx) * num_ranks;
  auto combine_num_tokens_per_rdma_rank =
      num_tokens_per_rdma_rank +
      (num_pipeline_stages + stage_idx) * num_rdma_ranks;
  auto combine_num_tokens_per_expert =
      num_tokens_per_expert + (num_pipeline_stages + stage_idx) * num_experts;
  auto combine_is_token_in_rank =
      is_token_in_rank +
      (num_pipeline_stages + stage_idx) * num_tokens * num_ranks;

  // 统计专家级别发送指标, 每个sm负责一些experts
  __shared__ int smem_dispatch_num_tokens_per_expert_per_thread
      [kNumThreads][kNumExpertsPerSM];  // 每个线程统计一部分token, 最后再聚合
  __shared__ int
      smem_combine_num_tokens_per_expert_per_thread[kNumThreads]
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
      smem_dispatch_num_tokens_per_expert_per_thread[thread_id][i] = 0;
      smem_combine_num_tokens_per_expert_per_thread[thread_id][i] = 0;
    }

    int rmem_dispatch_schedule_map[kNumRDMARanks];
    int rmem_combine_schedule_map[kNumRDMARanks];

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
            rmem_dispatch_schedule_map[j * 4 + k] =
                reinterpret_cast<const int*>(dispatch_schedule_map_vec)[k];
            rmem_combine_schedule_map[j * 4 + k] =
                reinterpret_cast<const int*>(combine_schedule_map_vec)[k];
          }
        }
      } else {
        for (int j = 0; j < kNumRDMARanks; j++) {
          rmem_dispatch_schedule_map[j] = shifted_dispatch_schedule_map[j];
          rmem_combine_schedule_map[j] = shifted_combine_schedule_map[j];
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
            rmem_dispatch_schedule_map[rdma_rank_idx] == stage_idx) {
          ++smem_dispatch_num_tokens_per_expert_per_thread[thread_id]
                                                          [expert_idx -
                                                           expert_begin_idx];
        }
        if (in_sm_range &&
            rmem_combine_schedule_map[rdma_rank_idx] == stage_idx) {
          ++smem_combine_num_tokens_per_expert_per_thread[thread_id]
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
        dispatch_sum +=
            smem_dispatch_num_tokens_per_expert_per_thread[i][thread_id];
        combine_sum +=
            smem_combine_num_tokens_per_expert_per_thread[i][thread_id];
      }
      dispatch_num_tokens_per_expert[expert_begin_idx + thread_id] =
          dispatch_sum;
      combine_num_tokens_per_expert[expert_begin_idx + thread_id] = combine_sum;
    }
    return;
  }

  if (num_tokens_per_rdma_rank != nullptr)
    EP_DEVICE_ASSERT(num_ranks % NUM_MAX_NVL_PEERS == 0 &&
                     num_ranks > NUM_MAX_NVL_PEERS);

  // 统计rank级别发送指标, 每个sm负责一些ranks
  constexpr int kNumRDMARanksPerSM = kNumRanksPerSM / NUM_MAX_NVL_PEERS;
  __shared__ int smem_dispatch_num_tokens_per_rank_per_thread[kNumThreads]
                                                             [kNumRanksPerSM];
  __shared__ int
      smem_dispatch_num_tokens_per_rdma_rank_per_thread[kNumThreads]
                                                       [kNumRDMARanksPerSM];
  __shared__ int smem_combine_num_tokens_per_rank_per_thread[kNumThreads]
                                                            [kNumRanksPerSM];
  __shared__ int
      smem_combine_num_tokens_per_rdma_rank_per_thread[kNumThreads]
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
      smem_dispatch_num_tokens_per_rank_per_thread[thread_id][i] = 0;
      smem_combine_num_tokens_per_rank_per_thread[thread_id][i] = 0;
    }
#pragma unroll
    for (int i = 0; i < kNumRDMARanksPerSM; ++i) {
      smem_dispatch_num_tokens_per_rdma_rank_per_thread[thread_id][i] = 0;
      smem_combine_num_tokens_per_rdma_rank_per_thread[thread_id][i] = 0;
    }

    int rmem_dispatch_schedule_map[kNumRDMARanks];
    int rmem_combine_schedule_map[kNumRDMARanks];

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
            rmem_dispatch_schedule_map[j * 4 + k] =
                reinterpret_cast<const int*>(dispatch_schedule_map_vec)[k];
            rmem_combine_schedule_map[j * 4 + k] =
                reinterpret_cast<const int*>(combine_schedule_map_vec)[k];
          }
        }
      } else {
        for (int j = 0; j < kNumRDMARanks; j++) {
          rmem_dispatch_schedule_map[j] = shifted_dispatch_schedule_map[j];
          rmem_combine_schedule_map[j] = shifted_combine_schedule_map[j];
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
            rmem_dispatch_schedule_map[rdma_rank_idx] == stage_idx) {
          dispatch_is_in_rank[rank_idx]++;
          dispatch_is_in_rdma_rank[rank_idx / NUM_MAX_NVL_PEERS]++;
        }
        if (in_sm_range &&
            rmem_combine_schedule_map[rdma_rank_idx] == stage_idx) {
          combine_is_in_rank[rank_idx]++;
          combine_is_in_rdma_rank[rank_idx / NUM_MAX_NVL_PEERS]++;
        }
      }

      EP_STATIC_ASSERT(kNumRanksPerSM == 8,
                       "Not match the vectorized memory access");
      EP_DEVICE_ASSERT(rank_end_idx - rank_begin_idx == 8);
      auto dispatch_shifted_is_token_in_rank =
          dispatch_is_token_in_rank + i * num_ranks;
      auto combine_shifted_is_token_in_rank =
          combine_is_token_in_rank + i * num_ranks;

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
        smem_dispatch_num_tokens_per_rank_per_thread[thread_id][j] +=
            (dispatch_is_in_rank[j] > 0);
        smem_combine_num_tokens_per_rank_per_thread[thread_id][j] +=
            (combine_is_in_rank[j] > 0);
      }

#pragma unroll
      for (int j = 0; j + rdma_rank_begin_idx < rdma_rank_end_idx; ++j) {
        smem_dispatch_num_tokens_per_rdma_rank_per_thread[thread_id][j] +=
            (dispatch_is_in_rdma_rank[j] > 0);
        smem_combine_num_tokens_per_rdma_rank_per_thread[thread_id][j] +=
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
        dispatch_sum +=
            smem_dispatch_num_tokens_per_rank_per_thread[i][thread_id];
        combine_sum +=
            smem_combine_num_tokens_per_rank_per_thread[i][thread_id];
      }
      dispatch_num_tokens_per_rank[rank_begin_idx + thread_id] = dispatch_sum;
      combine_num_tokens_per_rank[rank_begin_idx + thread_id] = combine_sum;
    }

    if (num_tokens_per_rdma_rank != nullptr &&
        rdma_rank_begin_idx + thread_id < rdma_rank_end_idx) {
      int dispatch_sum = 0;
      int combine_sum = 0;
#pragma unroll
      for (int i = 0; i < kNumThreads; ++i) {
        dispatch_sum +=
            smem_dispatch_num_tokens_per_rdma_rank_per_thread[i][thread_id];
        combine_sum +=
            smem_combine_num_tokens_per_rdma_rank_per_thread[i][thread_id];
      }
      int num_rdma_ranks = num_ranks / NUM_MAX_NVL_PEERS;
      dispatch_num_tokens_per_rdma_rank[rdma_rank_begin_idx + thread_id] =
          dispatch_sum;
      combine_num_tokens_per_rdma_rank[rdma_rank_begin_idx + thread_id] =
          combine_sum;
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
                                       int num_pipeline_stages,
                                       cudaStream_t stream) {
  constexpr int kNumThreads = 256;
  constexpr int kNumExpertsPerSM = 8;
  constexpr int kNumRanksPerSM = 8;

  int num_sms_per_loop =
      ((num_experts + kNumExpertsPerSM - 1) / kNumExpertsPerSM) +
      (num_ranks + kNumRanksPerSM - 1) / kNumRanksPerSM;
  int num_sms = num_sms_per_loop * num_pipeline_stages;

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
                  num_sms_per_loop,                                \
                  num_pipeline_stages);                            \
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
                                         const int num_pipeline_stages,
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
                  num_pipeline_stages);                             \
  }                                                                 \
  break

  SETUP_LAUNCH_CONFIG(num_sms, kNumThreads, stream);
  SWITCH_RDMA_RANKS(GET_SCHEDULE_LAUNCH_CASE);

#undef GET_SCHEDULE_LAUNCH_CASE
}

template <typename T, int64_t N>
struct alignas(16) VectorType {
  T data[N];
};

template <>
struct alignas(16) VectorType<float, 4> {
  float4 data;  // Built-in CUDA vector type
};

template <>
struct alignas(16) VectorType<__nv_bfloat16, 8> {
  __nv_bfloat16 data[8];
};

template <>
struct alignas(16) VectorType<__nv_fp8_e4m3, 16> {
  __nv_fp8_e4m3 data[16];
};

template <>
struct alignas(16) VectorType<uint8_t, 16> {
  uint8_t data[16];
};

template <>
struct alignas(16) VectorType<int32_t, 4> {
  int32_t data[4];
};

// Helper function to perform vectorized memory copy
template <typename T>
__device__ __forceinline__ void vectorized_memcpy(const T* src,
                                                  T* dst,
                                                  int64_t num_elements) {
  constexpr int64_t vector_size_in_bytes = 16;
  const int64_t elements_per_vector = vector_size_in_bytes / sizeof(T);

  int64_t num_vectors = num_elements / elements_per_vector;
  int64_t remaining_elements = num_elements % elements_per_vector;

  using VecType = VectorType<T, elements_per_vector>;
  const VecType* src_vec = reinterpret_cast<const VecType*>(src);
  VecType* dst_vec = reinterpret_cast<VecType*>(dst);

#pragma unroll
  for (int idx = threadIdx.x; idx < num_vectors; idx += blockDim.x) {
    dst_vec[idx] = src_vec[idx];
  }

  if (remaining_elements > 0) {
    int offset = num_vectors * elements_per_vector;
    for (int i = threadIdx.x; i < remaining_elements; i += blockDim.x) {
      dst[offset + i] = src[offset + i];
    }
  }
}

template <>
__device__ __forceinline__ void vectorized_memcpy<__nv_fp8_e4m3>(
    const __nv_fp8_e4m3* src, __nv_fp8_e4m3* dst, int64_t num_elements) {
  const int64_t elements_per_vector = 16;

  int64_t num_vectors = num_elements / elements_per_vector;
  int64_t remaining_elements = num_elements % elements_per_vector;

  const uint4* src_vec = reinterpret_cast<const uint4*>(src);
  uint4* dst_vec = reinterpret_cast<uint4*>(dst);

#pragma unroll
  for (int idx = threadIdx.x; idx < num_vectors; idx += blockDim.x) {
    dst_vec[idx] = __ldg(src_vec + idx);
  }

  if (remaining_elements > 0) {
    int offset = num_vectors * elements_per_vector;
    for (int i = threadIdx.x; i < remaining_elements; i += blockDim.x) {
      dst[offset + i] = src[offset + i];
    }
  }
}

constexpr int kCumsumInvalidTag = -1;

struct __custom_bfloat164 {
  __nv_bfloat16 x;
  __nv_bfloat16 y;
  __nv_bfloat16 z;
  __nv_bfloat16 w;
};

template <typename T>
__device__ __forceinline__ void vectorized_add(const T* src,
                                               float* dst,
                                               int64_t num_elements);

template <>
__device__ __forceinline__ void vectorized_add<float>(const float* src,
                                                      float* dst,
                                                      int64_t num_elements) {
  const float4* src_vec = reinterpret_cast<const float4*>(src);
  float4* dst_vec = reinterpret_cast<float4*>(dst);

  int64_t num_vectors = num_elements / 4;
  int64_t remaining_elements = num_elements % 4;

  for (int i = threadIdx.x; i < num_vectors; i += blockDim.x) {
    float4 s_vec_num = __ldg(src_vec + i);
    float4 d_vec_num = *(dst_vec + i);
    d_vec_num.x = __fadd_rn(s_vec_num.x, d_vec_num.x);
    d_vec_num.y = __fadd_rn(s_vec_num.y, d_vec_num.y);
    d_vec_num.z = __fadd_rn(s_vec_num.z, d_vec_num.z);
    d_vec_num.w = __fadd_rn(s_vec_num.w, d_vec_num.w);
    *(dst_vec + i) = d_vec_num;
  }

  if (remaining_elements > 0) {
    int offset = num_vectors * 4;
    for (int i = threadIdx.x; i < remaining_elements; i += blockDim.x) {
      dst[offset + i] += src[offset + i];
    }
  }
}

template <>
__device__ __forceinline__ void vectorized_add<__nv_bfloat16>(
    const __nv_bfloat16* src, float* dst, int64_t num_elements) {
  const uint64_t* src_vec = reinterpret_cast<const uint64_t*>(src);
  float4* dst_vec = reinterpret_cast<float4*>(dst);

  int64_t num_vectors = num_elements / 4;
  int64_t remaining_elements = num_elements % 4;

  for (int i = threadIdx.x; i < num_vectors; i += blockDim.x) {
    uint64_t s_vec_num_raw = __ldg(src_vec + i);
    __custom_bfloat164 s_vec_num =
        *reinterpret_cast<__custom_bfloat164*>(&s_vec_num_raw);
    float4 d_vec_num = *(dst_vec + i);
    d_vec_num.x = __fadd_rn(__bfloat162float(s_vec_num.x), d_vec_num.x);
    d_vec_num.y = __fadd_rn(__bfloat162float(s_vec_num.y), d_vec_num.y);
    d_vec_num.z = __fadd_rn(__bfloat162float(s_vec_num.z), d_vec_num.z);
    d_vec_num.w = __fadd_rn(__bfloat162float(s_vec_num.w), d_vec_num.w);
    *(dst_vec + i) = d_vec_num;
  }

  if (remaining_elements > 0) {
    int offset = num_vectors * 4;
    for (int i = threadIdx.x; i < remaining_elements; i += blockDim.x) {
      dst[offset + i] += __bfloat162float(src[offset + i]);
    }
  }
}

struct token_infos {
  int token_row_idx;

  __device__ __host__ token_infos() : token_row_idx(-1) {}
  __device__ __host__ token_infos(int idx) : token_row_idx(idx) {}

  __device__ __host__ token_infos& operator=(const token_infos& other) {
    token_row_idx = other.token_row_idx;
    return *this;
  }
};

// prefix_sum[i] ≤ t < prefix_sum[i+1]
__device__ __forceinline__ int findPtrIndex(const int* prefix_sum,
                                            int M,
                                            int t) {
  int low = 0, high = M;
  while (low < high) {
    int mid = (low + high) / 2;
    if (prefix_sum[mid] <= t) {
      low = mid + 1;
    } else {
      high = mid;
    }
  }
  return low - 1;
}

template <int kCumsumBlockSize, int kMaxNumExpertsPerRank>
__global__ void dispatch_moe_meta_kernel(
    const int32_t* input_topk_idx,
    int32_t* global_expertwise_block_cumsum,
    const int32_t topk,
    const int32_t all_token_num,
    const int32_t num_experts,
    int32_t* output_rowmap,
    int32_t* output_rowmap_offset) {
  int64_t local_cumsum = 0;
  const int64_t block_row_base =
      static_cast<int64_t>(blockIdx.x) * static_cast<int64_t>(kCumsumBlockSize);
  int64_t cumsum_offset = (blockIdx.x != 0) * kCumsumInvalidTag;
  __shared__ token_infos
      shared_token_infos[kCumsumBlockSize][kMaxNumExpertsPerRank];

  if (threadIdx.x < num_experts) {
    token_infos local_token_infos[kCumsumBlockSize];
    for (int64_t row = block_row_base; row < block_row_base + kCumsumBlockSize;
         row++) {
      if (row >= all_token_num) break;
      const int64_t internal_row = row - block_row_base;
#pragma unroll
      for (int64_t k = 0; k < topk; k++) {
        token_infos proposed = {input_topk_idx[row * topk + k]};
        if (proposed.token_row_idx == -1) continue;
        if (threadIdx.x == proposed.token_row_idx) {
          local_token_infos[internal_row] = {local_cumsum};
          local_cumsum += 1;
        }
      }
    }
    // Inter-block communication
    const int64_t anticipate_signal_idx =
        static_cast<int64_t>(blockIdx.x) * static_cast<int64_t>(num_experts) +
        threadIdx.x;
    const int64_t push_signal_idx = (static_cast<int64_t>(blockIdx.x) + 1) *
                                        static_cast<int64_t>(num_experts) +
                                    threadIdx.x;
    if (blockIdx.x != 0) {
      // signal receive from previous block, using light-weight atomicAdd(check)
      // this will not change any data, only do fetch in low-cost
      while ((cumsum_offset = atomicAdd(
                  &global_expertwise_block_cumsum[anticipate_signal_idx], 0)) ==
             kCumsumInvalidTag) {
      }
    }
    // signal send for next block, with current cumsum
    const int64_t proposed_offset = cumsum_offset + local_cumsum;
    global_expertwise_block_cumsum[push_signal_idx] = proposed_offset;
    if (blockIdx.x == gridDim.x - 1) {
      output_rowmap_offset[threadIdx.x] = proposed_offset;
    }
    // Intra-block communication;
#pragma unroll
    for (int64_t i = 0; i < kCumsumBlockSize; i++) {
      local_token_infos[i].token_row_idx =
          (local_token_infos[i].token_row_idx == -1)
              ? -1
              : local_token_infos[i].token_row_idx + cumsum_offset;
      shared_token_infos[i][threadIdx.x] = local_token_infos[i];
    }
  }
  __syncthreads();

  for (int64_t row = block_row_base; row < block_row_base + kCumsumBlockSize;
       row++) {
    if (row >= all_token_num) return;
    const int64_t internal_row = row - block_row_base;
    for (int64_t expert_id = 0; expert_id < num_experts; expert_id++) {
      const token_infos this_expert_token_info =
          shared_token_infos[internal_row][expert_id];
      const int64_t proposed_row_idx = this_expert_token_info.token_row_idx;
      if (proposed_row_idx == -1) continue;  // no memcpy
      if (threadIdx.x == 0)
        output_rowmap[row * static_cast<int64_t>(num_experts) + expert_id] =
            proposed_row_idx;
    }
  }
}

void dispatch_moe_meta(const int32_t* input_topk_idx,
                       int32_t* global_expertwise_block_cumsum,
                       const int32_t topk,
                       const int32_t all_token_num,
                       const int32_t num_experts,
                       int32_t* output_rowmap,
                       int32_t* output_rowmap_offset,
                       cudaStream_t stream) {
  constexpr int kCumsumBlockSize = 32;
  constexpr int kMaxNumExpertsPerRank = 8;
  constexpr int kNumThreads = 512;

  EP_HOST_ASSERT(num_experts <= kMaxNumExpertsPerRank);
  int num_sms = (all_token_num + kCumsumBlockSize - 1) / kCumsumBlockSize;

  dispatch_moe_meta_kernel<kCumsumBlockSize, kMaxNumExpertsPerRank>
      <<<num_sms, kNumThreads, 0, stream>>>(input_topk_idx,
                                            global_expertwise_block_cumsum,
                                            topk,
                                            all_token_num,
                                            num_experts,
                                            output_rowmap,
                                            output_rowmap_offset);
}

template <bool kIsForward,
          bool kMixedPrecision,
          int kCumsumBlockSize,
          int kMaxStageNum>
__global__ void local_dispatch_kernel(const void** input_x,
                                      const float** input_topk_weights,
                                      const int32_t** input_topk_idx,
                                      const int32_t** src_meta,
                                      const float** input_scales,
                                      const int32_t** output_rowmap,
                                      const int32_t** output_rowmap_offset,
                                      const int32_t* a2a_prefix_sum,
                                      const int32_t local_expert_id,
                                      const int32_t hidden_size,
                                      const int32_t topk,
                                      const int32_t a2a_num,
                                      const int32_t all_token_num,
                                      const int32_t scale_num,
                                      const int32_t num_experts,
                                      void* output_hidden,
                                      int32_t* output_top_idx,
                                      float* output_topk_weights,
                                      int32_t* output_src_meta,
                                      float* output_scales) {
  __shared__ const float* smem_topk_weights_ptrs[kMaxStageNum];
  __shared__ const int32_t* smem_topk_idx_ptrs[kMaxStageNum];
  __shared__ const void* smem_input_x_ptrs[kMaxStageNum];
  __shared__ const int32_t* smem_src_meta_ptrs[kMaxStageNum];
  __shared__ const float* smem_input_scales_ptrs[kMaxStageNum];
  EP_DEVICE_ASSERT(a2a_num < blockDim.x);
  if (threadIdx.x < a2a_num) {
    if (input_topk_weights)
      smem_topk_weights_ptrs[threadIdx.x] = input_topk_weights[threadIdx.x];
    if (input_topk_idx)
      smem_topk_idx_ptrs[threadIdx.x] = input_topk_idx[threadIdx.x];
    if (input_x) smem_input_x_ptrs[threadIdx.x] = input_x[threadIdx.x];
    if (src_meta) smem_src_meta_ptrs[threadIdx.x] = src_meta[threadIdx.x];
    if (input_scales)
      smem_input_scales_ptrs[threadIdx.x] = input_scales[threadIdx.x];
  }

  __syncthreads();

  int64_t rmem_expert_prefix_sum[kMaxStageNum];
  rmem_expert_prefix_sum[0] = 0;
  for (int i = 1; i < a2a_num; i++) {
    rmem_expert_prefix_sum[i] = rmem_expert_prefix_sum[i - 1] +
                                output_rowmap_offset[i - 1][local_expert_id];
  }

  const int64_t block_row_base = blockIdx.x * kCumsumBlockSize;
  for (int64_t row = block_row_base; row < block_row_base + kCumsumBlockSize;
       row++) {
    // OOB check
    if (row >= all_token_num) return;
    const int64_t a2a_idx = findPtrIndex(a2a_prefix_sum, a2a_num, row);
    const int64_t a2a_token_idx = row - a2a_prefix_sum[a2a_idx];
    int64_t tmp_proposed_row_idx =
        output_rowmap[a2a_idx]
                     [a2a_token_idx * static_cast<int64_t>(num_experts) +
                      static_cast<int64_t>(local_expert_id)];
    if (tmp_proposed_row_idx == -1) continue;  // no memcpy
    const int64_t proposed_row_idx =
        tmp_proposed_row_idx + rmem_expert_prefix_sum[a2a_idx];

    if (kIsForward) {
      if (threadIdx.x < topk) {
        int64_t tmp_idx =
            smem_topk_idx_ptrs[a2a_idx][static_cast<int64_t>(a2a_token_idx) *
                                            static_cast<int64_t>(topk) +
                                        static_cast<int64_t>(threadIdx.x)];
        if (tmp_idx == local_expert_id) {
          output_topk_weights[proposed_row_idx] =
              smem_topk_weights_ptrs[a2a_idx]
                                    [static_cast<int64_t>(a2a_token_idx) *
                                         static_cast<int64_t>(topk) +
                                     static_cast<int64_t>(threadIdx.x)];
        }
      }
    }

    if (threadIdx.x == 0) {
      if (!kIsForward) {
        for (int i = 0; i < topk; i++) {
          output_top_idx[proposed_row_idx * topk + i] =
              smem_topk_idx_ptrs[a2a_idx][static_cast<int64_t>(a2a_token_idx) *
                                              static_cast<int64_t>(topk) +
                                          static_cast<int64_t>(i)];
        }
      }
      const int4* src_vec = reinterpret_cast<const int4*>(
          &(smem_src_meta_ptrs[a2a_idx]
                              [static_cast<int64_t>(a2a_token_idx) * 4]));
      int4* dst_vec = reinterpret_cast<int4*>(
          &(output_src_meta[static_cast<int64_t>(proposed_row_idx) * 4]));
      dst_vec[0] = src_vec[0];
    }
    if (kMixedPrecision) {
      auto input_x_ptr =
          reinterpret_cast<const __nv_fp8_e4m3*>(smem_input_x_ptrs[a2a_idx]);
      auto output_hidden_ptr = reinterpret_cast<__nv_fp8_e4m3*>(output_hidden);
      vectorized_memcpy<__nv_fp8_e4m3>(
          &input_x_ptr[static_cast<int64_t>(a2a_token_idx) *
                       static_cast<int64_t>(hidden_size)],
          &output_hidden_ptr[static_cast<int64_t>(proposed_row_idx) *
                             static_cast<int64_t>(hidden_size)],
          hidden_size);
      vectorized_memcpy(
          &smem_input_scales_ptrs[a2a_idx][static_cast<int64_t>(a2a_token_idx) *
                                           static_cast<int64_t>(scale_num)],
          &output_scales[static_cast<int64_t>(proposed_row_idx) *
                         static_cast<int64_t>(scale_num)],
          scale_num);
    } else {
      auto input_x_ptr =
          reinterpret_cast<const __nv_bfloat16*>(smem_input_x_ptrs[a2a_idx]);
      auto output_hidden_ptr = reinterpret_cast<__nv_bfloat16*>(output_hidden);
      vectorized_memcpy<__nv_bfloat16>(
          &input_x_ptr[static_cast<int64_t>(a2a_token_idx) *
                       static_cast<int64_t>(hidden_size)],
          &output_hidden_ptr[static_cast<int64_t>(proposed_row_idx) *
                             static_cast<int64_t>(hidden_size)],
          hidden_size);
    }
  }
}

template <int kSMNum, int kMaxStageNum>
__global__ void combine_moe_kernel_forward(const __nv_bfloat16* input_x,
                                           const int32_t** gbl_channel_prefix,
                                           const int32_t* src_meta,
                                           const int32_t hidden_size,
                                           const int32_t num_pipeline_stages,
                                           const int64_t token_num,
                                           float** output_x,
                                           int num_channels) {
  EP_DEVICE_ASSERT(blockDim.x % 32 == 0);
  const int32_t* rmem_gbl_channel_prefix_ptrs[kMaxStageNum];
  float* rmem_output_x_ptrs[kMaxStageNum];
  for (int i = 0; i < num_pipeline_stages; i++) {
    rmem_gbl_channel_prefix_ptrs[i] = gbl_channel_prefix[i];
    rmem_output_x_ptrs[i] = output_x[i];
  }

  const int64_t block_row_base =
      static_cast<int64_t>(blockIdx.x) * static_cast<int64_t>(kSMNum);
  for (int64_t row = block_row_base; row < block_row_base + kSMNum; row++) {
    if (row >= token_num) return;
    const int4* vec_token_meta = reinterpret_cast<const int4*>(src_meta);
    const int4 meta_vec = __ldg(vec_token_meta + row);
    const int64_t channel_id = meta_vec.x;
    const int64_t nvl_head = meta_vec.y;
    const int64_t src_rank = meta_vec.z;
    const int64_t stage_idx_to_combine = meta_vec.w;

    const int64_t channel_offset =
        rmem_gbl_channel_prefix_ptrs[stage_idx_to_combine]
                                    [src_rank *
                                         static_cast<int64_t>(num_channels) +
                                     channel_id];
    const int64_t offset = channel_offset + nvl_head;
    float* token_out_ptr = rmem_output_x_ptrs[stage_idx_to_combine] +
                           offset * static_cast<int64_t>(hidden_size);
    const __nv_bfloat16* token_load_ptr =
        input_x + row * static_cast<int64_t>(hidden_size);
    vectorized_add(token_load_ptr, token_out_ptr, hidden_size);
  }
}

template <int kSMNum, int kMaxStageNum>
__global__ void combine_moe_kernel_backward(const __nv_bfloat16* input_x,
                                            const int32_t* topk_idx,
                                            const float* topk_weights,
                                            const int32_t** gbl_channel_prefix,
                                            const int32_t* src_meta,
                                            const int32_t hidden_size,
                                            const int32_t num_pipeline_stages,
                                            const int64_t token_num,
                                            const int32_t topk,
                                            const int32_t local_expert_id,
                                            float** output_x,
                                            float** output_topk_weights,
                                            int num_channels) {
  EP_DEVICE_ASSERT(blockDim.x % 32 == 0);
  const int32_t* rmem_gbl_channel_prefix_ptrs[kMaxStageNum];
  float* rmem_output_x_ptrs[kMaxStageNum];
  float* rmem_output_topk_weights_ptrs[kMaxStageNum];
  for (int i = 0; i < num_pipeline_stages; i++) {
    rmem_gbl_channel_prefix_ptrs[i] = gbl_channel_prefix[i];
    rmem_output_x_ptrs[i] = output_x[i];
    rmem_output_topk_weights_ptrs[i] = output_topk_weights[i];
  }

  const int64_t block_row_base =
      static_cast<int64_t>(blockIdx.x) * static_cast<int64_t>(kSMNum);
  for (int64_t row = block_row_base; row < block_row_base + kSMNum; row++) {
    if (row >= token_num) return;
    const int4* vec_token_meta = reinterpret_cast<const int4*>(src_meta);
    const int4 meta_vec = __ldg(vec_token_meta + row);
    const int64_t channel_id = meta_vec.x;
    const int64_t nvl_head = meta_vec.y;
    const int64_t src_rank = meta_vec.z;
    const int64_t stage_idx_to_combine = meta_vec.w;

    const int64_t channel_offset =
        rmem_gbl_channel_prefix_ptrs[stage_idx_to_combine]
                                    [src_rank *
                                         static_cast<int64_t>(num_channels) +
                                     channel_id];
    const int64_t offset = channel_offset + nvl_head;

    if (threadIdx.x < topk) {
      if (topk_idx[row * static_cast<int64_t>(topk) +
                   static_cast<int64_t>(threadIdx.x)] == local_expert_id) {
        rmem_output_topk_weights_ptrs[stage_idx_to_combine]
                                     [offset * static_cast<int64_t>(topk) +
                                      static_cast<int64_t>(threadIdx.x)] =
                                         topk_weights[row];
      }
    }

    float* token_out_ptr = rmem_output_x_ptrs[stage_idx_to_combine] +
                           offset * static_cast<int64_t>(hidden_size);
    const __nv_bfloat16* token_load_ptr =
        input_x + row * static_cast<int64_t>(hidden_size);
    vectorized_add(token_load_ptr, token_out_ptr, hidden_size);
  }
}

void local_dispatch(const void** input_x,
                    const float** input_topk_weights,
                    const int32_t** input_topk_idx,
                    const int32_t** src_meta,
                    const float** input_scales,
                    const int32_t** output_rowmap,
                    const int32_t** output_rowmap_offset,
                    const int32_t* a2a_prefix_sum,
                    const int32_t local_expert_id,
                    const int32_t hidden_size,
                    const int32_t topk,
                    const int32_t a2a_num,
                    const int32_t all_token_num,
                    const int32_t scale_num,
                    const int32_t num_experts,
                    void* output_hidden,
                    int32_t* output_top_idx,
                    float* output_topk_weights,
                    int32_t* output_src_meta,
                    float* output_scales,
                    cudaStream_t stream,
                    bool use_fp8,
                    bool forward,
                    int num_pipeline_stages) {
  constexpr int kNumThreads = 1024;
  constexpr int kCumsumBlockSize = 2;
  constexpr int kMaxStageNum = 8;

  int num_sms = (all_token_num + kCumsumBlockSize - 1) / kCumsumBlockSize;
  EP_HOST_ASSERT(!(use_fp8 && !forward));
  EP_HOST_ASSERT(num_pipeline_stages <= kMaxStageNum);

#define LAUNCH_LOCAL_DISPATCH_KERNEL(T, FORWARD, USE_FP8)                   \
  do {                                                                      \
    local_dispatch_kernel<FORWARD, USE_FP8, kCumsumBlockSize, kMaxStageNum> \
        <<<num_sms, kNumThreads, 0, stream>>>(input_x,                      \
                                              input_topk_weights,           \
                                              input_topk_idx,               \
                                              src_meta,                     \
                                              input_scales,                 \
                                              output_rowmap,                \
                                              output_rowmap_offset,         \
                                              a2a_prefix_sum,               \
                                              local_expert_id,              \
                                              hidden_size,                  \
                                              topk,                         \
                                              a2a_num,                      \
                                              all_token_num,                \
                                              scale_num,                    \
                                              num_experts,                  \
                                              output_hidden,                \
                                              output_top_idx,               \
                                              output_topk_weights,          \
                                              output_src_meta,              \
                                              output_scales);               \
  } while (0)

  if (use_fp8 && forward) {
    LAUNCH_LOCAL_DISPATCH_KERNEL(__nv_fp8_e4m3, true, true);
  } else if (!use_fp8 && forward) {
    LAUNCH_LOCAL_DISPATCH_KERNEL(__nv_bfloat16, true, false);
  } else {
    LAUNCH_LOCAL_DISPATCH_KERNEL(__nv_bfloat16, false, false);
  }
#undef LAUNCH_LOCAL_DISPATCH_KERNEL
}

static int LimitGridDim(int64_t n) {
  return static_cast<int>(std::min<int64_t>(n, 1024 * 1024));
}

void local_combine_forward(const __nv_bfloat16* input_x,
                           const int32_t** gbl_channel_prefix,
                           const int32_t* src_meta,
                           const int32_t hidden_size,
                           const int32_t num_pipeline_stages,
                           const int64_t token_num,
                           float** output_x,
                           int num_channels,
                           cudaStream_t stream) {
  constexpr int kNumThreads = 1024;
  constexpr int kSMNum = 4;
  constexpr int kMaxStageNum = 8;
  int num_sms = LimitGridDim((token_num + kSMNum - 1) / kSMNum);

  EP_HOST_ASSERT(num_pipeline_stages <= kMaxStageNum);

  combine_moe_kernel_forward<kSMNum, kMaxStageNum>
      <<<num_sms, kNumThreads, 0, stream>>>(input_x,
                                            gbl_channel_prefix,
                                            src_meta,
                                            hidden_size,
                                            num_pipeline_stages,
                                            token_num,
                                            output_x,
                                            num_channels);
}

void local_combine_backward(const __nv_bfloat16* input_x,
                            const int32_t* topk_idx,
                            const float* topk_weights,
                            const int32_t** gbl_channel_prefix,
                            const int32_t* src_meta,
                            const int32_t hidden_size,
                            const int32_t num_pipeline_stages,
                            const int64_t token_num,
                            const int32_t topk,
                            const int32_t local_expert_id,
                            float** output_x,
                            float** output_topk_weights,
                            int num_channels,
                            cudaStream_t stream) {
  constexpr int kNumThreads = 1024;
  constexpr int kSMNum = 4;
  int num_sms = LimitGridDim((token_num + kSMNum - 1) / kSMNum);
  constexpr int kMaxStageNum = 8;

  EP_HOST_ASSERT(num_pipeline_stages <= kMaxStageNum);

  combine_moe_kernel_backward<kSMNum, kMaxStageNum>
      <<<num_sms, kNumThreads, 0, stream>>>(input_x,
                                            topk_idx,
                                            topk_weights,
                                            gbl_channel_prefix,
                                            src_meta,
                                            hidden_size,
                                            num_pipeline_stages,
                                            token_num,
                                            topk,
                                            local_expert_id,
                                            output_x,
                                            output_topk_weights,
                                            num_channels);
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
