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

// clang-format off
#include <nvshmem.h>
#include <nvshmemx.h>
#include <infiniband/mlx5dv.h>
#include <non_abi/device/threadgroup/nvshmemi_common_device_defines.cuh>
#include <device_host_transport/nvshmem_common_ibgda.h>
// clang-format on

#include "paddle/fluid/distributed/collective/deep_ep/kernels/buffer.cuh"
#include "paddle/fluid/distributed/collective/deep_ep/kernels/configs.cuh"
#include "paddle/fluid/distributed/collective/deep_ep/kernels/exception.cuh"
#include "paddle/fluid/distributed/collective/deep_ep/kernels/ibgda_device.cuh"
#include "paddle/fluid/distributed/collective/deep_ep/kernels/launch.cuh"
#include "paddle/fluid/distributed/collective/deep_ep/kernels/utils.cuh"

namespace deep_ep {

namespace internode {

extern nvshmem_team_t cpu_rdma_team;

struct SourceMeta {
  int src_rdma_rank, is_token_in_nvl_rank_bits;

  EP_STATIC_ASSERT(NUM_MAX_NVL_PEERS == 8,
                   "Invalid number of maximum NVL peers");

  __forceinline__ SourceMeta() = default;

  // TODO(Xreki): faster encoding
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

__host__ __device__ __forceinline__ int get_num_bytes_per_rdma_token(
    int hidden_int4, int num_scales, int num_topk_idx, int num_topk_weights) {
  return static_cast<int>(
      align(hidden_int4 * sizeof(int4) + sizeof(SourceMeta) +
                num_scales * sizeof(float) + num_topk_idx * sizeof(int) +
                num_topk_weights * sizeof(float),
            sizeof(int4)));
}

__host__ __device__ __forceinline__ std::pair<int, int> get_rdma_clean_meta(
    int hidden_int4,
    int num_scales,
    int num_topk_idx,
    int num_topk_weights,
    int num_rdma_ranks,
    int num_rdma_recv_buffer_tokens,
    int num_sms) {
  // Return `int32_t` offset and count to clean
  return {(get_num_bytes_per_rdma_token(
               hidden_int4, num_scales, num_topk_idx, num_topk_weights) *
           num_rdma_recv_buffer_tokens * num_rdma_ranks * 2 * num_sms) /
              sizeof(int),
          (NUM_MAX_NVL_PEERS * 2 + 4) * num_rdma_ranks * 2 * num_sms};
}

__host__ __device__ __forceinline__ std::pair<int, int> get_nvl_clean_meta(
    int hidden_int4,
    int num_scales,
    int num_topk_idx,
    int num_topk_weights,
    int num_rdma_ranks,
    int num_nvl_ranks,
    int num_nvl_recv_buffer_tokens,
    int num_sms) {
  // Return `int32_t` offset and to clean
  EP_STATIC_ASSERT(sizeof(SourceMeta) % sizeof(int) == 0,
                   "Invalid size of `SourceMeta`");
  return {
      (num_nvl_recv_buffer_tokens *
       (hidden_int4 * sizeof(int4) + num_scales * sizeof(float) +
        num_topk_idx * sizeof(int) + num_topk_weights * sizeof(float) +
        sizeof(SourceMeta)) *
       num_nvl_ranks * num_sms) /
          sizeof(int),
      num_nvl_ranks * (2 * num_rdma_ranks + 2) * num_sms,
  };
}

template <bool kLowLatencyMode>
__forceinline__ __device__ int translate_dst_rdma_rank(const int dst_rdma_rank,
                                                       const int nvl_rank) {
  return kLowLatencyMode ? (dst_rdma_rank * NUM_MAX_NVL_PEERS + nvl_rank)
                         : dst_rdma_rank;
}

template <bool kLowLatencyMode>
__forceinline__ __device__ void nvshmem_barrier_with_same_gpu_idx(
    const nvshmem_team_t& rdma_team) {
  kLowLatencyMode ? void(nvshmem_barrier(rdma_team)) : nvshmem_barrier_all();
}

template <bool kLowLatencyMode, int kNumRDMARanks>
__global__ void notify_dispatch(const int* num_tokens_per_rank,
                                int* moe_recv_counter_mapped,
                                int num_ranks,
                                const int* num_tokens_per_rdma_rank,
                                int* moe_recv_rdma_counter_mapped,
                                const int* num_tokens_per_expert,
                                int* moe_recv_expert_counter_mapped,
                                int num_experts,
                                const bool* is_token_in_rank,
                                int num_tokens,
                                int num_channels,
                                int expert_alignment,
                                const int rdma_clean_offset,
                                const int rdma_num_int_clean,
                                const int nvl_clean_offset,
                                const int nvl_num_int_clean,
                                int* rdma_channel_prefix_matrix,
                                int* recv_rdma_rank_prefix_sum,
                                int* gbl_channel_prefix_matrix,
                                int* recv_gbl_rank_prefix_sum,
                                void* rdma_buffer_ptr,
                                void** buffer_ptrs,
                                int** task_fifo_ptrs,
                                int head,
                                int rank,
                                const nvshmem_team_t rdma_team) {
  auto sm_id = static_cast<int>(blockIdx.x);
  auto thread_id = static_cast<int>(threadIdx.x), warp_id = thread_id / 32,
       lane_id = get_lane_id();
  auto num_threads = static_cast<int>(blockDim.x), num_warps = num_threads / 32;

  auto rdma_rank = rank / NUM_MAX_NVL_PEERS,
       nvl_rank = rank % NUM_MAX_NVL_PEERS;
  auto num_rdma_experts = num_experts / kNumRDMARanks,
       num_nvl_experts = num_rdma_experts / NUM_MAX_NVL_PEERS;

  if (sm_id == 0) {
    // Communication with others
    // Global barrier: the first warp do intra-node sync, the second warp do
    // internode sync
    EP_DEVICE_ASSERT(num_warps > 1);
    EP_DEVICE_ASSERT(kNumRDMARanks <= num_threads);
    if (thread_id == 32)
      nvshmem_barrier_with_same_gpu_idx<kLowLatencyMode>(rdma_team);
    barrier_device<NUM_MAX_NVL_PEERS>(task_fifo_ptrs, head, nvl_rank);
    move_fifo_slots<NUM_MAX_NVL_PEERS>(head);
    __syncthreads();

    // Send numbers of tokens per rank/expert to RDMA ranks
    auto rdma_buffer_ptr_int = reinterpret_cast<int*>(rdma_buffer_ptr);
    auto rdma_recv_num_tokens_mixed =
        SymBuffer<int>(rdma_buffer_ptr,
                       NUM_MAX_NVL_PEERS + num_rdma_experts + 1,
                       kNumRDMARanks);

    // Clean up for later data dispatch
    EP_DEVICE_ASSERT(rdma_recv_num_tokens_mixed.total_bytes <=
                     rdma_clean_offset * sizeof(int));
#pragma unroll
    for (int i = thread_id; i < rdma_num_int_clean; i += num_threads)
      rdma_buffer_ptr_int[rdma_clean_offset + i] = 0;

// Copy to send buffer
#pragma unroll
    for (int i = thread_id; i < num_ranks; i += num_threads)
      rdma_recv_num_tokens_mixed.send_buffer(
          i / NUM_MAX_NVL_PEERS)[i % NUM_MAX_NVL_PEERS] =
          num_tokens_per_rank[i];
#pragma unroll
    for (int i = thread_id; i < num_experts; i += num_threads)
      rdma_recv_num_tokens_mixed.send_buffer(
          i / num_rdma_experts)[NUM_MAX_NVL_PEERS + i % num_rdma_experts] =
          num_tokens_per_expert[i];
    if (thread_id < kNumRDMARanks)
      rdma_recv_num_tokens_mixed.send_buffer(
          thread_id)[NUM_MAX_NVL_PEERS + num_rdma_experts] =
          num_tokens_per_rdma_rank[thread_id];
    __syncthreads();

    // Issue send
    // TODO(Xreki): more light fence or barrier or signaling
    // TODO(Xreki): overlap EP barrier and NVL cleaning
    if (thread_id < kNumRDMARanks) {
      nvshmem_int_put_nbi(
          rdma_recv_num_tokens_mixed.recv_buffer(rdma_rank),
          rdma_recv_num_tokens_mixed.send_buffer(thread_id),
          NUM_MAX_NVL_PEERS + num_rdma_experts + 1,
          translate_dst_rdma_rank<kLowLatencyMode>(thread_id, nvl_rank));
    }
    __syncthreads();
    if (thread_id == 0)
      nvshmem_barrier_with_same_gpu_idx<kLowLatencyMode>(rdma_team);
    __syncthreads();

    // NVL buffers
    auto nvl_send_buffer =
        thread_id < NUM_MAX_NVL_PEERS ? buffer_ptrs[thread_id] : nullptr;
    auto nvl_recv_buffer = buffer_ptrs[nvl_rank];
    auto nvl_reduced_num_tokens_per_expert =
        Buffer<int>(nvl_recv_buffer, num_rdma_experts)
            .advance_also(nvl_send_buffer);
    auto nvl_send_num_tokens_per_rank =
        AsymBuffer<int>(nvl_send_buffer, kNumRDMARanks, NUM_MAX_NVL_PEERS);
    auto nvl_send_num_tokens_per_expert =
        AsymBuffer<int>(nvl_send_buffer, num_nvl_experts, NUM_MAX_NVL_PEERS);
    auto nvl_recv_num_tokens_per_rank =
        AsymBuffer<int>(nvl_recv_buffer, kNumRDMARanks, NUM_MAX_NVL_PEERS);
    auto nvl_recv_num_tokens_per_expert =
        AsymBuffer<int>(nvl_recv_buffer, num_nvl_experts, NUM_MAX_NVL_PEERS);

    // Clean up for later data dispatch
    auto nvl_buffer_ptr_int = reinterpret_cast<int*>(buffer_ptrs[nvl_rank]);
    EP_DEVICE_ASSERT(nvl_reduced_num_tokens_per_expert.total_bytes +
                         nvl_send_num_tokens_per_rank.total_bytes +
                         nvl_send_num_tokens_per_expert.total_bytes <=
                     nvl_clean_offset * sizeof(int));
#pragma unroll
    for (int i = thread_id; i < nvl_num_int_clean; i += num_threads)
      nvl_buffer_ptr_int[nvl_clean_offset + i] = 0;

    // Reduce number of tokens per expert into the NVL send buffer
    // TODO(Xreki): may use NVSHMEM reduction
    EP_DEVICE_ASSERT(num_rdma_experts <= num_threads);
    if (thread_id < num_rdma_experts) {
      int sum = 0;
#pragma unroll
      for (int i = 0; i < kNumRDMARanks; ++i)
        sum += rdma_recv_num_tokens_mixed.recv_buffer(
            i)[NUM_MAX_NVL_PEERS + thread_id];
      nvl_reduced_num_tokens_per_expert[thread_id] = sum;
    }
    __syncthreads();

    // Reduce RDMA received tokens
    if (thread_id == 0) {
      int sum = 0;
#pragma unroll
      for (int i = 0; i < kNumRDMARanks; ++i) {
        sum += rdma_recv_num_tokens_mixed.recv_buffer(
            i)[NUM_MAX_NVL_PEERS + num_rdma_experts];
        recv_rdma_rank_prefix_sum[i] = sum;
      }
      while (ld_volatile_global(moe_recv_rdma_counter_mapped) != -1) {
      }
      *moe_recv_rdma_counter_mapped = sum;
    }

    // Send numbers of tokens per rank/expert to NVL ranks
    EP_DEVICE_ASSERT(NUM_MAX_NVL_PEERS <= num_threads);
    if (thread_id < NUM_MAX_NVL_PEERS) {
#pragma unroll
      for (int i = 0; i < kNumRDMARanks; ++i)
        nvl_send_num_tokens_per_rank.buffer(nvl_rank)[i] =
            rdma_recv_num_tokens_mixed.recv_buffer(i)[thread_id];
#pragma unroll
      for (int i = 0; i < num_nvl_experts; ++i)
        nvl_send_num_tokens_per_expert.buffer(nvl_rank)[i] =
            nvl_reduced_num_tokens_per_expert[thread_id * num_nvl_experts + i];
    }
    memory_fence();
    __syncthreads();
    barrier_device<NUM_MAX_NVL_PEERS>(task_fifo_ptrs, head, nvl_rank);
    move_fifo_slots<NUM_MAX_NVL_PEERS>(head);
    __syncthreads();

    // Reduce number of tokens per rank/expert
    EP_DEVICE_ASSERT(num_nvl_experts <= num_threads);
    if (thread_id == 0) {
      int sum = 0;
#pragma unroll
      for (int i = 0; i < num_ranks; ++i) {
        int src_rdma_rank = i / NUM_MAX_NVL_PEERS,
            src_nvl_rank = i % NUM_MAX_NVL_PEERS;
        sum += nvl_recv_num_tokens_per_rank.buffer(src_nvl_rank)[src_rdma_rank];
        recv_gbl_rank_prefix_sum[i] = sum;
      }
      while (ld_volatile_global(moe_recv_counter_mapped) != -1) {
      }
      *moe_recv_counter_mapped = sum;
    }
    if (thread_id < num_nvl_experts) {
      int sum = 0;
#pragma unroll
      for (int i = 0; i < NUM_MAX_NVL_PEERS; ++i)
        sum += nvl_recv_num_tokens_per_expert.buffer(i)[thread_id];
      sum = (sum + expert_alignment - 1) / expert_alignment * expert_alignment;
      while (ld_volatile_global(moe_recv_expert_counter_mapped + thread_id) !=
             -1) {
      }
      moe_recv_expert_counter_mapped[thread_id] = sum;
    }

    // Finally barrier
    __syncthreads();
    if (thread_id == 32)
      nvshmem_barrier_with_same_gpu_idx<kLowLatencyMode>(rdma_team);
    barrier_device<NUM_MAX_NVL_PEERS>(task_fifo_ptrs, head, nvl_rank);
    move_fifo_slots<NUM_MAX_NVL_PEERS>(head);
  } else {
    // Calculate meta data
    int dst_rdma_rank = sm_id - 1;
    for (int channel_id = warp_id; channel_id < num_channels;
         channel_id += num_warps) {
      int token_start_idx, token_end_idx;
      get_channel_task_range(
          num_tokens, num_channels, channel_id, token_start_idx, token_end_idx);

      // Iterate over tokens
      int total_count = 0, per_nvl_rank_count[NUM_MAX_NVL_PEERS] = {0};
      for (int64_t i = token_start_idx + lane_id; i < token_end_idx; i += 32) {
        EP_STATIC_ASSERT(NUM_MAX_NVL_PEERS * sizeof(bool) == sizeof(uint64_t),
                         "Invalid number of NVL peers");
        auto is_token_in_rank_uint64 = *reinterpret_cast<const uint64_t*>(
            is_token_in_rank + i * num_ranks +
            dst_rdma_rank * NUM_MAX_NVL_PEERS);
        auto is_token_in_rank_values =
            reinterpret_cast<const bool*>(&is_token_in_rank_uint64);
#pragma unroll
        for (int j = 0; j < NUM_MAX_NVL_PEERS; ++j)
          per_nvl_rank_count[j] += is_token_in_rank_values[j];
        total_count += (is_token_in_rank_uint64 != 0);
      }

      // Warp reduce
      total_count = warp_reduce_sum(total_count);
#pragma unroll
      for (int i = 0; i < NUM_MAX_NVL_PEERS; ++i)
        per_nvl_rank_count[i] = warp_reduce_sum(per_nvl_rank_count[i]);

      // Write into channel matrix
      if (lane_id == 0) {
#pragma unroll
        for (int i = 0; i < NUM_MAX_NVL_PEERS; ++i)
          gbl_channel_prefix_matrix[(dst_rdma_rank * NUM_MAX_NVL_PEERS + i) *
                                        num_channels +
                                    channel_id] = per_nvl_rank_count[i];
        rdma_channel_prefix_matrix[dst_rdma_rank * num_channels + channel_id] =
            total_count;
      }
    }

    // Calculate prefix sum
    __syncthreads();
    if (thread_id == 0) {
      auto prefix_row =
          rdma_channel_prefix_matrix + dst_rdma_rank * num_channels;
#pragma unroll
      for (int i = 1; i < num_channels; ++i) prefix_row[i] += prefix_row[i - 1];
    }

    EP_STATIC_ASSERT(NUM_MAX_NVL_PEERS <= 32, "Invalid number of NVL peers");
    if (thread_id < NUM_MAX_NVL_PEERS) {
      auto prefix_row =
          gbl_channel_prefix_matrix +
          (dst_rdma_rank * NUM_MAX_NVL_PEERS + thread_id) * num_channels;
#pragma unroll
      for (int i = 1; i < num_channels; ++i) prefix_row[i] += prefix_row[i - 1];
    }
  }
}

void notify_dispatch(const int* num_tokens_per_rank,
                     int* moe_recv_counter_mapped,
                     int num_ranks,
                     const int* num_tokens_per_rdma_rank,
                     int* moe_recv_rdma_counter_mapped,
                     const int* num_tokens_per_expert,
                     int* moe_recv_expert_counter_mapped,
                     int num_experts,
                     const bool* is_token_in_rank,
                     int num_tokens,
                     int num_channels,
                     int hidden_int4,
                     int num_scales,
                     int num_topk,
                     int expert_alignment,
                     int* rdma_channel_prefix_matrix,
                     int* recv_rdma_rank_prefix_sum,
                     int* gbl_channel_prefix_matrix,
                     int* recv_gbl_rank_prefix_sum,
                     void* rdma_buffer_ptr,
                     int num_max_rdma_chunked_recv_tokens,
                     void** buffer_ptrs,
                     int num_max_nvl_chunked_recv_tokens,
                     int** task_fifo_ptrs,
                     int head,
                     int rank,
                     cudaStream_t stream,
                     int64_t num_rdma_bytes,
                     int64_t num_nvl_bytes,
                     bool low_latency_mode) {
#define NOTIFY_DISPATCH_LAUNCH_CASE(num_rdma_ranks)                           \
  {                                                                           \
    auto notify_dispatch_func = low_latency_mode                              \
                                    ? notify_dispatch<true, num_rdma_ranks>   \
                                    : notify_dispatch<false, num_rdma_ranks>; \
    LAUNCH_KERNEL(&cfg,                                                       \
                  notify_dispatch_func,                                       \
                  num_tokens_per_rank,                                        \
                  moe_recv_counter_mapped,                                    \
                  num_ranks,                                                  \
                  num_tokens_per_rdma_rank,                                   \
                  moe_recv_rdma_counter_mapped,                               \
                  num_tokens_per_expert,                                      \
                  moe_recv_expert_counter_mapped,                             \
                  num_experts,                                                \
                  is_token_in_rank,                                           \
                  num_tokens,                                                 \
                  num_channels,                                               \
                  expert_alignment,                                           \
                  rdma_clean_meta.first,                                      \
                  rdma_clean_meta.second,                                     \
                  nvl_clean_meta.first,                                       \
                  nvl_clean_meta.second,                                      \
                  rdma_channel_prefix_matrix,                                 \
                  recv_rdma_rank_prefix_sum,                                  \
                  gbl_channel_prefix_matrix,                                  \
                  recv_gbl_rank_prefix_sum,                                   \
                  rdma_buffer_ptr,                                            \
                  buffer_ptrs,                                                \
                  task_fifo_ptrs,                                             \
                  head,                                                       \
                  rank,                                                       \
                  cpu_rdma_team);                                             \
  }                                                                           \
  break

  constexpr int kNumThreads = 512;
  const auto num_rdma_ranks = num_ranks / NUM_MAX_NVL_PEERS;

  // Get clean meta
  auto rdma_clean_meta = get_rdma_clean_meta(hidden_int4,
                                             num_scales,
                                             num_topk,
                                             num_topk,
                                             num_rdma_ranks,
                                             num_max_rdma_chunked_recv_tokens,
                                             num_channels);
  auto nvl_clean_meta = get_nvl_clean_meta(hidden_int4,
                                           num_scales,
                                           num_topk,
                                           num_topk,
                                           num_rdma_ranks,
                                           NUM_MAX_NVL_PEERS,
                                           num_max_nvl_chunked_recv_tokens,
                                           num_channels);
  EP_HOST_ASSERT((rdma_clean_meta.first + rdma_clean_meta.second) *
                     sizeof(int) <=
                 num_rdma_bytes);
  EP_HOST_ASSERT((nvl_clean_meta.first + nvl_clean_meta.second) * sizeof(int) <=
                 num_nvl_bytes);
  EP_HOST_ASSERT(num_rdma_bytes < std::numeric_limits<int>::max());
  EP_HOST_ASSERT(num_nvl_bytes < std::numeric_limits<int>::max());

  // Launch kernel
  SETUP_LAUNCH_CONFIG(1 + num_rdma_ranks, kNumThreads, stream);
  SWITCH_RDMA_RANKS(NOTIFY_DISPATCH_LAUNCH_CASE);
#undef NOTIFY_DISPATCH_LAUNCH_CASE
}

// At most 8 RDMA ranks to be sent
constexpr int get_num_topk_rdma_ranks(int num_rdma_ranks) {
  return num_rdma_ranks < 8 ? num_rdma_ranks : 8;
}

template <bool kLowLatencyMode,
          int kNumRDMARanks,
          bool kCachedMode,
          int kNumDispatchRDMASenderWarps,
          int kNumTopkRDMARanks = get_num_topk_rdma_ranks(kNumRDMARanks)>
__global__ void __launch_bounds__(
    ((kNumDispatchRDMASenderWarps + 1 + NUM_MAX_NVL_PEERS) * 32), 1)
    dispatch(int4* recv_x,
             float* recv_x_scales,
             int64_t* recv_topk_idx,
             float* recv_topk_weights,
             SourceMeta* recv_src_meta,
             const int4* x,
             const float* x_scales,
             const int64_t* topk_idx,
             const float* topk_weights,
             int* send_rdma_head,
             int* send_nvl_head,
             int* recv_rdma_channel_prefix_matrix,
             int* recv_gbl_channel_prefix_matrix,
             const int* rdma_channel_prefix_matrix,
             const int* recv_rdma_rank_prefix_sum,
             const int* gbl_channel_prefix_matrix,
             const int* recv_gbl_rank_prefix_sum,
             int num_tokens,
             int hidden_int4,
             int num_scales,
             int num_topk,
             int num_experts,
             const bool* is_token_in_rank,
             void* rdma_buffer_ptr,
             int num_max_rdma_chunked_send_tokens,
             int num_max_rdma_chunked_recv_tokens,
             void** buffer_ptrs,
             int num_max_nvl_chunked_send_tokens,
             int num_max_nvl_chunked_recv_tokens,
             int rank,
             int num_ranks) {
  enum class WarpRole {
    kRDMASender,
    kRDMASenderCoordinator,
    kRDMAAndNVLForwarder,
    kForwarderCoordinator,
    kNVLReceivers
  };

  const auto sm_id = static_cast<int>(blockIdx.x);
  const auto num_threads = static_cast<int>(blockDim.x),
             num_warps = num_threads / 32;
  const auto thread_id = static_cast<int>(threadIdx.x),
             warp_id = thread_id / 32, lane_id = get_lane_id();
  const auto num_channels = static_cast<int>(gridDim.x) / 2,
             channel_id = sm_id / 2;
  const bool is_forwarder = sm_id % 2 == 0;
  const auto rdma_rank = rank / NUM_MAX_NVL_PEERS,
             nvl_rank = rank % NUM_MAX_NVL_PEERS;

  EP_DEVICE_ASSERT(ibgda_get_state()->num_rc_per_pe >= num_channels);

  const auto role_meta = [=]() -> std::pair<WarpRole, int> {
    if (is_forwarder) {
      if (warp_id < NUM_MAX_NVL_PEERS) {
        return {WarpRole::kRDMAAndNVLForwarder,
                (warp_id + channel_id) % NUM_MAX_NVL_PEERS};
      } else {
        return {WarpRole::kForwarderCoordinator, warp_id - NUM_MAX_NVL_PEERS};
      }
    } else if (warp_id < kNumDispatchRDMASenderWarps) {
      return {WarpRole::kRDMASender, -1};
    } else if (warp_id == kNumDispatchRDMASenderWarps) {
      return {WarpRole::kRDMASenderCoordinator, -1};
    } else {
      return {WarpRole::kNVLReceivers,
              (warp_id + channel_id - kNumDispatchRDMASenderWarps) %
                  NUM_MAX_NVL_PEERS};
    }
  }();
  auto warp_role = role_meta.first;
  auto target_rank = role_meta.second;  // Not applicable for RDMA senders
  EP_DEVICE_ASSERT(num_warps ==
                   kNumDispatchRDMASenderWarps + 1 + NUM_MAX_NVL_PEERS);

  // Data checks
  EP_DEVICE_ASSERT(num_topk <= 32);

  // RDMA symmetric layout
  EP_STATIC_ASSERT(NUM_MAX_NVL_PEERS * sizeof(bool) == sizeof(uint64_t),
                   "Invalid number of NVL peers");
  auto hidden_bytes = hidden_int4 * sizeof(int4);
  auto num_bytes_per_rdma_token =
      get_num_bytes_per_rdma_token(hidden_int4, num_scales, num_topk, num_topk);
  auto rdma_channel_data = SymBuffer<int8_t>(
      rdma_buffer_ptr,
      num_max_rdma_chunked_recv_tokens * num_bytes_per_rdma_token,
      kNumRDMARanks,
      channel_id,
      num_channels);
  auto rdma_channel_meta = SymBuffer<int>(rdma_buffer_ptr,
                                          NUM_MAX_NVL_PEERS * 2 + 2,
                                          kNumRDMARanks,
                                          channel_id,
                                          num_channels);
  auto rdma_channel_head = SymBuffer<uint64_t, false>(
      rdma_buffer_ptr, 1, kNumRDMARanks, channel_id, num_channels);
  auto rdma_channel_tail = SymBuffer<uint64_t, false>(
      rdma_buffer_ptr, 1, kNumRDMARanks, channel_id, num_channels);

  // NVL buffer layouts
  // NOTES: `rs_wr_buffer_ptr` means "Read for Senders, Write for Receivers",
  // `ws_rr_buffer_ptr` means "Write for Senders, Read for Receivers"
  void *rs_wr_buffer_ptr = nullptr, *ws_rr_buffer_ptr = nullptr;
  int rs_wr_rank = 0, ws_rr_rank = 0;
  if (warp_role == WarpRole::kRDMAAndNVLForwarder)
    rs_wr_buffer_ptr = buffer_ptrs[nvl_rank],
    ws_rr_buffer_ptr = buffer_ptrs[target_rank], rs_wr_rank = nvl_rank,
    ws_rr_rank = target_rank;
  if (warp_role == WarpRole::kNVLReceivers)
    rs_wr_buffer_ptr = buffer_ptrs[target_rank],
    ws_rr_buffer_ptr = buffer_ptrs[nvl_rank], rs_wr_rank = target_rank,
    ws_rr_rank = nvl_rank;

  // Allocate buffers
  auto nvl_channel_x =
      AsymBuffer<int4>(ws_rr_buffer_ptr,
                       num_max_nvl_chunked_recv_tokens * hidden_int4,
                       NUM_MAX_NVL_PEERS,
                       channel_id,
                       num_channels,
                       rs_wr_rank)
          .advance_also(rs_wr_buffer_ptr);
  auto nvl_channel_src_meta =
      AsymBuffer<SourceMeta>(ws_rr_buffer_ptr,
                             num_max_nvl_chunked_recv_tokens,
                             NUM_MAX_NVL_PEERS,
                             channel_id,
                             num_channels,
                             rs_wr_rank)
          .advance_also(rs_wr_buffer_ptr);
  auto nvl_channel_x_scales =
      AsymBuffer<float>(ws_rr_buffer_ptr,
                        num_max_nvl_chunked_recv_tokens * num_scales,
                        NUM_MAX_NVL_PEERS,
                        channel_id,
                        num_channels,
                        rs_wr_rank)
          .advance_also(rs_wr_buffer_ptr);
  auto nvl_channel_topk_idx =
      AsymBuffer<int>(ws_rr_buffer_ptr,
                      num_max_nvl_chunked_recv_tokens * num_topk,
                      NUM_MAX_NVL_PEERS,
                      channel_id,
                      num_channels,
                      rs_wr_rank)
          .advance_also(rs_wr_buffer_ptr);
  auto nvl_channel_topk_weights =
      AsymBuffer<float>(ws_rr_buffer_ptr,
                        num_max_nvl_chunked_recv_tokens * num_topk,
                        NUM_MAX_NVL_PEERS,
                        channel_id,
                        num_channels,
                        rs_wr_rank)
          .advance_also(rs_wr_buffer_ptr);
  auto nvl_channel_prefix_start = AsymBuffer<int>(ws_rr_buffer_ptr,
                                                  kNumRDMARanks,
                                                  NUM_MAX_NVL_PEERS,
                                                  channel_id,
                                                  num_channels,
                                                  rs_wr_rank)
                                      .advance_also(rs_wr_buffer_ptr);
  auto nvl_channel_prefix_end = AsymBuffer<int>(ws_rr_buffer_ptr,
                                                kNumRDMARanks,
                                                NUM_MAX_NVL_PEERS,
                                                channel_id,
                                                num_channels,
                                                rs_wr_rank)
                                    .advance_also(rs_wr_buffer_ptr);
  auto nvl_channel_head = AsymBuffer<int>(rs_wr_buffer_ptr,
                                          1,
                                          NUM_MAX_NVL_PEERS,
                                          channel_id,
                                          num_channels,
                                          ws_rr_rank)
                              .advance_also(ws_rr_buffer_ptr);
  auto nvl_channel_tail = AsymBuffer<int>(ws_rr_buffer_ptr,
                                          1,
                                          NUM_MAX_NVL_PEERS,
                                          channel_id,
                                          num_channels,
                                          rs_wr_rank)
                              .advance_also(rs_wr_buffer_ptr);

  // RDMA sender warp synchronization
  __shared__ volatile int rdma_send_next_token_idx;
  __shared__ volatile int rdma_send_channel_tail[kNumRDMARanks];
  __shared__ volatile int rdma_send_channel_next_tail[kNumRDMARanks];
  auto sync_rdma_sender_smem = []() {
    asm volatile(
        "bar.sync 0, %0;" ::"r"((kNumDispatchRDMASenderWarps + 1) * 32));
  };

  // Forward warp synchronization
  __shared__ volatile int forward_channel_head[NUM_MAX_NVL_PEERS]
                                              [kNumRDMARanks];
  __shared__ volatile bool forward_channel_retired[NUM_MAX_NVL_PEERS];
  auto sync_forwarder_smem = []() {
    asm volatile("bar.sync 1, %0;" ::"r"((NUM_MAX_NVL_PEERS + 1) * 32));
  };

  if (warp_role == WarpRole::kRDMASender) {
    // Get tasks
    int token_start_idx, token_end_idx;
    get_channel_task_range(
        num_tokens, num_channels, channel_id, token_start_idx, token_end_idx);

    // Clean shared memory
    EP_STATIC_ASSERT(kNumRDMARanks <= 32, "Invalid number of RDMA ranks");
    (warp_id == 0 && lane_id == 0)
        ? (rdma_send_next_token_idx = token_start_idx)
        : 0;
    (warp_id == 0 && lane_id < kNumRDMARanks)
        ? (rdma_send_channel_tail[lane_id] = 0)
        : 0;
    (warp_id == 0 && lane_id < kNumRDMARanks)
        ? (rdma_send_channel_next_tail[lane_id] = 0)
        : 0;

    // Send number of tokens in this channel by `-value - 1`
    EP_STATIC_ASSERT(NUM_MAX_NVL_PEERS * 2 + 2 <= 32,
                     "Invalid number of NVL peers");
    for (int dst_rdma_rank = warp_id; dst_rdma_rank < kNumRDMARanks;
         dst_rdma_rank += kNumDispatchRDMASenderWarps) {
      auto dst_ptr = dst_rdma_rank == rdma_rank
                         ? rdma_channel_meta.recv_buffer(dst_rdma_rank)
                         : rdma_channel_meta.send_buffer(dst_rdma_rank);
      if (lane_id < NUM_MAX_NVL_PEERS) {
        dst_ptr[lane_id] =
            -(channel_id == 0
                  ? 0
                  : gbl_channel_prefix_matrix
                        [(dst_rdma_rank * NUM_MAX_NVL_PEERS + lane_id) *
                             num_channels +
                         channel_id - 1]) -
            1;
      } else if (lane_id < NUM_MAX_NVL_PEERS * 2) {
        dst_ptr[lane_id] =
            -gbl_channel_prefix_matrix[(dst_rdma_rank * NUM_MAX_NVL_PEERS +
                                        lane_id - NUM_MAX_NVL_PEERS) *
                                           num_channels +
                                       channel_id] -
            1;
      } else if (lane_id == NUM_MAX_NVL_PEERS * 2) {
        dst_ptr[lane_id] =
            -(channel_id == 0
                  ? 0
                  : rdma_channel_prefix_matrix[dst_rdma_rank * num_channels +
                                               channel_id - 1]) -
            1;
      } else if (lane_id == NUM_MAX_NVL_PEERS * 2 + 1) {
        dst_ptr[lane_id] =
            -rdma_channel_prefix_matrix[dst_rdma_rank * num_channels +
                                        channel_id] -
            1;
      }
      __syncwarp();
      // Issue RDMA for non-local ranks
      if (dst_rdma_rank != rdma_rank) {
        nvshmemi_ibgda_put_nbi_warp<true>(
            reinterpret_cast<uint64_t>(
                rdma_channel_meta.recv_buffer(rdma_rank)),
            reinterpret_cast<uint64_t>(
                rdma_channel_meta.send_buffer(dst_rdma_rank)),
            sizeof(int) * (NUM_MAX_NVL_PEERS * 2 + 2),
            translate_dst_rdma_rank<kLowLatencyMode>(dst_rdma_rank, nvl_rank),
            channel_id,
            lane_id,
            0);
      }
    }
    sync_rdma_sender_smem();

    // Iterate over tokens and copy into buffer
    int64_t token_idx;
    int cached_rdma_channel_head = 0, last_rdma_tail_idx = -1;
    auto send_buffer = lane_id == rdma_rank
                           ? rdma_channel_data.recv_buffer(lane_id)
                           : rdma_channel_data.send_buffer(lane_id);
    for (token_idx = token_start_idx + warp_id; token_idx < token_end_idx;
         token_idx += kNumDispatchRDMASenderWarps) {
      // Read RDMA rank existence
      uint64_t is_token_in_rank_uint64 = 0;
      if (lane_id < kNumRDMARanks)
        is_token_in_rank_uint64 = *reinterpret_cast<const uint64_t*>(
            is_token_in_rank + token_idx * num_ranks +
            lane_id * NUM_MAX_NVL_PEERS);

      // Acquire sequential lock
      while (lane_id == 0 && rdma_send_next_token_idx != token_idx) {
      }
      __syncwarp();

      // Acquire next tail
      int rdma_tail_idx = -1;
      if (is_token_in_rank_uint64 != 0) {
        rdma_tail_idx = rdma_send_channel_next_tail[lane_id]++;
        while (rdma_tail_idx - cached_rdma_channel_head >=
               num_max_rdma_chunked_recv_tokens)
          cached_rdma_channel_head = static_cast<int>(
              ld_volatile_global(rdma_channel_head.buffer(lane_id)));
      }
      __syncwarp();

      // Store RDMA head for combine
      if (lane_id < kNumRDMARanks && !kCachedMode)
        send_rdma_head[token_idx * kNumRDMARanks + lane_id] = rdma_tail_idx;

      // Update last token tail
      if (last_rdma_tail_idx >= 0)
        st_release_cta(const_cast<const int*>(rdma_send_channel_tail + lane_id),
                       last_rdma_tail_idx + 1);
      last_rdma_tail_idx = rdma_tail_idx;

      // Release sequential lock
      lane_id == 0 ? (rdma_send_next_token_idx += 1) : 0;

      // Broadcast tails
      SourceMeta src_meta;
      int num_topk_ranks = 0, topk_ranks[kNumTopkRDMARanks];
      void* dst_send_buffers[kNumTopkRDMARanks];
#pragma unroll
      for (int i = 0, slot_idx; i < kNumRDMARanks; ++i)
        if ((slot_idx = __shfl_sync(0xffffffff, rdma_tail_idx, i)) >= 0) {
          slot_idx = slot_idx % num_max_rdma_chunked_recv_tokens;
          topk_ranks[num_topk_ranks] = i;
          auto recv_is_token_in_rank_uint64 =
              broadcast(is_token_in_rank_uint64, i);
          auto recv_is_token_in_rank_values =
              reinterpret_cast<const bool*>(&recv_is_token_in_rank_uint64);
          if (lane_id == num_topk_ranks)
            src_meta = SourceMeta(rdma_rank, recv_is_token_in_rank_values);
          dst_send_buffers[num_topk_ranks++] =
              reinterpret_cast<uint8_t*>(broadcast(send_buffer, i)) +
              slot_idx * num_bytes_per_rdma_token;
        }
      EP_DEVICE_ASSERT(num_topk_ranks <= kNumTopkRDMARanks);

      // Copy `x` into symmetric send buffer
      auto st_broadcast = [=](const int key, const int4& value) {
#pragma unroll
        for (int j = 0; j < num_topk_ranks; ++j)
          st_na_global(reinterpret_cast<int4*>(dst_send_buffers[j]) + key,
                       value);
      };
      UNROLLED_WARP_COPY(5,
                         lane_id,
                         hidden_int4,
                         0,
                         x + token_idx * hidden_int4,
                         ld_nc_global,
                         st_broadcast);
#pragma unroll
      for (int i = 0; i < num_topk_ranks; ++i)
        dst_send_buffers[i] =
            reinterpret_cast<int4*>(dst_send_buffers[i]) + hidden_int4;

      // Copy source metadata into symmetric send buffer
      if (lane_id < num_topk_ranks)
        st_na_global(reinterpret_cast<SourceMeta*>(dst_send_buffers[lane_id]),
                     src_meta);
#pragma unroll
      for (int i = 0; i < num_topk_ranks; ++i)
        dst_send_buffers[i] =
            reinterpret_cast<SourceMeta*>(dst_send_buffers[i]) + 1;

// Copy `x_scales` into symmetric send buffer
#pragma unroll
      for (int i = lane_id; i < num_scales; i += 32) {
        auto value = ld_nc_global(x_scales + token_idx * num_scales + i);
#pragma unroll
        for (int j = 0; j < num_topk_ranks; ++j)
          st_na_global(reinterpret_cast<float*>(dst_send_buffers[j]) + i,
                       value);
      }
#pragma unroll
      for (int i = 0; i < num_topk_ranks; ++i)
        dst_send_buffers[i] =
            reinterpret_cast<float*>(dst_send_buffers[i]) + num_scales;

// Copy `topk_idx` and `topk_weights` into symmetric send buffer
#pragma unroll
      for (int i = lane_id; i < num_topk * num_topk_ranks; i += 32) {
        auto rank_idx = i / num_topk, copy_idx = i % num_topk;
        auto idx_value = static_cast<int>(
            ld_nc_global(topk_idx + token_idx * num_topk + copy_idx));
        auto weight_value =
            ld_nc_global(topk_weights + token_idx * num_topk + copy_idx);
        st_na_global(
            reinterpret_cast<int*>(dst_send_buffers[rank_idx]) + copy_idx,
            idx_value);
        st_na_global(reinterpret_cast<float*>(dst_send_buffers[rank_idx]) +
                         num_topk + copy_idx,
                     weight_value);
      }
    }

    // Epilogue
    // Acquire sequential lock
    while (lane_id == 0 && rdma_send_next_token_idx != token_idx) {
    }
    __syncwarp();

    // Update last token tail
    if (last_rdma_tail_idx >= 0)
      st_release_cta(const_cast<const int*>(rdma_send_channel_tail + lane_id),
                     last_rdma_tail_idx + 1);

    // Release sequential lock
    lane_id == 0 ? (rdma_send_next_token_idx += 1) : 0;
  } else if (warp_role == WarpRole::kRDMASenderCoordinator) {
    // NOTES: in case of splitting the issued put at the end of the buffer
    EP_DEVICE_ASSERT(num_max_rdma_chunked_recv_tokens %
                         num_max_rdma_chunked_send_tokens ==
                     0);

    // Synchronize shared memory
    sync_rdma_sender_smem();

    // Get number of tokens to send for each RDMA rank
    int num_tokens_to_send = 0;
    if (lane_id < kNumRDMARanks) {
      num_tokens_to_send =
          rdma_channel_prefix_matrix[lane_id * num_channels + channel_id];
      if (channel_id > 0)
        num_tokens_to_send -=
            rdma_channel_prefix_matrix[lane_id * num_channels + channel_id - 1];
    }

    // Iterate all RDMA ranks
    int last_issued_tail = 0;
    while (__any_sync(0xffffffff, num_tokens_to_send > 0)) {
      for (int i = 0, synced_num_tokens_to_send; i < kNumRDMARanks; ++i) {
        // To mitigate incast congestion, shuffle the starting index of target
        // rank for different ranks and channel
        int dst_rdma_rank = (i + channel_id + rdma_rank) % kNumRDMARanks;
        synced_num_tokens_to_send =
            __shfl_sync(0xffffffff, num_tokens_to_send, dst_rdma_rank);
        if (synced_num_tokens_to_send == 0) continue;

        // Read progress
        auto synced_last_issued_tail =
            __shfl_sync(0xffffffff, last_issued_tail, dst_rdma_rank);
        auto processed_tail = ld_acquire_cta(
            const_cast<const int*>(rdma_send_channel_tail + dst_rdma_rank));
        auto num_tokens_processed = processed_tail - synced_last_issued_tail;
        if (num_tokens_processed != synced_num_tokens_to_send &&
            num_tokens_processed < num_max_rdma_chunked_send_tokens)
          continue;

        // Issue RDMA send
        auto num_tokens_to_issue =
            min(num_tokens_processed, num_max_rdma_chunked_send_tokens);
        EP_DEVICE_ASSERT(num_tokens_to_issue >= 0 &&
                         num_tokens_to_issue <= synced_num_tokens_to_send);
        if (dst_rdma_rank != rdma_rank) {
          auto dst_slot_idx =
              synced_last_issued_tail % num_max_rdma_chunked_recv_tokens;
          EP_DEVICE_ASSERT(dst_slot_idx + num_tokens_to_issue <=
                           num_max_rdma_chunked_recv_tokens);
          const size_t num_bytes_per_msg =
              num_bytes_per_rdma_token * num_tokens_to_issue;
          const auto dst_ptr = reinterpret_cast<uint64_t>(
              rdma_channel_data.recv_buffer(rdma_rank) +
              dst_slot_idx * num_bytes_per_rdma_token);
          const auto src_ptr = reinterpret_cast<uint64_t>(
              rdma_channel_data.send_buffer(dst_rdma_rank) +
              dst_slot_idx * num_bytes_per_rdma_token);
          nvshmemi_ibgda_put_nbi_warp<true>(
              dst_ptr,
              src_ptr,
              num_bytes_per_msg,
              translate_dst_rdma_rank<kLowLatencyMode>(dst_rdma_rank, nvl_rank),
              channel_id,
              lane_id,
              0);
        } else {
          // Lighter fence for local RDMA rank
          memory_fence();
        }

        // Update tails
        __syncwarp();
        if (lane_id == dst_rdma_rank) {
          last_issued_tail += num_tokens_to_issue;
          num_tokens_to_send -= num_tokens_to_issue;
          nvshmemi_ibgda_amo_nonfetch_add(
              rdma_channel_tail.buffer(rdma_rank),
              num_tokens_to_issue,
              translate_dst_rdma_rank<kLowLatencyMode>(dst_rdma_rank, nvl_rank),
              channel_id,
              dst_rdma_rank == rdma_rank);
        }
      }
    }
  } else if (warp_role == WarpRole::kRDMAAndNVLForwarder) {
    // RDMA consumers and NVL producers
    const auto dst_nvl_rank = target_rank;
    const auto dst_rank = rdma_rank * NUM_MAX_NVL_PEERS + dst_nvl_rank;
    const auto dst_rank_expert_begin = dst_rank * (num_experts / num_ranks);
    const auto dst_rank_expert_end =
        dst_rank_expert_begin + (num_experts / num_ranks);

    // Wait counters to arrive
    int num_tokens_to_recv_from_rdma = 0, src_rdma_channel_prefix = 0;
    EP_DEVICE_ASSERT(kNumRDMARanks <= 32);
    auto start_time = clock64();
    if (lane_id < kNumRDMARanks) {
      while (true) {
        auto meta_0 = ld_volatile_global(
            rdma_channel_meta.recv_buffer(lane_id) + dst_nvl_rank);
        auto meta_1 =
            ld_volatile_global(rdma_channel_meta.recv_buffer(lane_id) +
                               NUM_MAX_NVL_PEERS + dst_nvl_rank);
        auto meta_2 = ld_volatile_global(
            rdma_channel_meta.recv_buffer(lane_id) + NUM_MAX_NVL_PEERS * 2);
        auto meta_3 = ld_volatile_global(
            rdma_channel_meta.recv_buffer(lane_id) + NUM_MAX_NVL_PEERS * 2 + 1);
        if (meta_0 < 0 && meta_1 < 0 && meta_2 < 0 && meta_3 < 0) {
          // Notify NVL ranks
          int start_sum = -meta_0 - 1, end_sum = -meta_1 - 1;
          EP_DEVICE_ASSERT(start_sum >= 0 && end_sum >= 0 &&
                           end_sum >= start_sum);
          st_relaxed_sys_global(nvl_channel_prefix_start.buffer() + lane_id,
                                -start_sum - 1);
          st_relaxed_sys_global(nvl_channel_prefix_end.buffer() + lane_id,
                                -end_sum - 1);

          // Save RDMA channel received token count
          src_rdma_channel_prefix = -meta_2 - 1;
          auto src_rdma_channel_prefix_1 = -meta_3 - 1;
          num_tokens_to_recv_from_rdma =
              src_rdma_channel_prefix_1 - src_rdma_channel_prefix;
          if (!kCachedMode)
            recv_rdma_channel_prefix_matrix[lane_id * num_channels +
                                            channel_id] =
                src_rdma_channel_prefix_1;
          src_rdma_channel_prefix +=
              lane_id == 0 ? 0 : recv_rdma_rank_prefix_sum[lane_id - 1];
          EP_DEVICE_ASSERT(num_tokens_to_recv_from_rdma >= 0);
          break;
        }

        // Timeout check
        if (clock64() - start_time > NUM_TIMEOUT_CYCLES) {
          printf(
              "DeepEP dispatch forwarder timeout (RDMA meta), channel: %d, "
              "RDMA: %d, nvl: %d, src RDMA lane: %d, dst NVL: %d, meta: %d, "
              "%d, %d, %d\n",
              channel_id,
              rdma_rank,
              nvl_rank,
              lane_id,
              dst_nvl_rank,
              meta_0,
              meta_1,
              meta_2,
              meta_3);
          trap();
        }
      }
    }
    __syncwarp();

    // Shift cached head
    send_nvl_head += src_rdma_channel_prefix * NUM_MAX_NVL_PEERS + dst_nvl_rank;

    // Wait shared memory to be cleaned
    sync_forwarder_smem();

    // Forward tokens from RDMA buffer
    // NOTES: always start from the local rank
    int src_rdma_rank = sm_id % kNumRDMARanks;
    int cached_rdma_channel_head = 0, cached_rdma_channel_tail = 0;
    int cached_nvl_channel_head = 0, cached_nvl_channel_tail = 0,
        rdma_nvl_token_idx = 0;
    while (__any_sync(0xffffffff, num_tokens_to_recv_from_rdma > 0)) {
      // Check destination queue emptiness, or wait a buffer to be released
      start_time = clock64();
      while (lane_id == 0) {
        int num_used_slots = cached_nvl_channel_tail - cached_nvl_channel_head;
        if (num_max_nvl_chunked_recv_tokens - num_used_slots >=
            num_max_nvl_chunked_send_tokens)
          break;
        cached_nvl_channel_head = ld_volatile_global(nvl_channel_head.buffer());

        // Timeout check
        if (clock64() - start_time > NUM_TIMEOUT_CYCLES) {
          printf(
              "DeepEP dispatch forwarder timeout (NVL check), channel: %d, "
              "RDMA: %d, nvl: %d, dst NVL: %d, head: %d, tail: %d\n",
              channel_id,
              rdma_rank,
              nvl_rank,
              dst_nvl_rank,
              ld_volatile_global(nvl_channel_head.buffer()),
              cached_nvl_channel_tail);
          trap();
        }
      }
      __syncwarp();

      // Find next source RDMA rank (round-robin)
      start_time = clock64();
      while (true) {
        src_rdma_rank = (src_rdma_rank + 1) % kNumRDMARanks;
        if (__shfl_sync(
                0xffffffff, num_tokens_to_recv_from_rdma, src_rdma_rank) > 0) {
          if (lane_id == src_rdma_rank &&
              cached_rdma_channel_head == cached_rdma_channel_tail)
            cached_rdma_channel_tail = static_cast<int>(
                ld_acquire_sys_global(rdma_channel_tail.buffer(src_rdma_rank)));
          if (__shfl_sync(0xffffffff,
                          cached_rdma_channel_tail > cached_rdma_channel_head,
                          src_rdma_rank))
            break;
        }

        // Timeout check
        if (clock64() - start_time > NUM_TIMEOUT_CYCLES &&
            lane_id < kNumRDMARanks) {
          printf(
              "DeepEP dispatch forwarder timeout (RDMA check), channel: %d, "
              "RDMA: %d, nvl: %d, dst NVL: %d, src RDMA lane: %d, head: %d, "
              "tail: %d, expected: %d\n",
              channel_id,
              rdma_rank,
              nvl_rank,
              dst_nvl_rank,
              lane_id,
              cached_rdma_channel_head,
              cached_rdma_channel_tail,
              num_tokens_to_recv_from_rdma);
          trap();
        }
      }
      auto src_rdma_head =
          __shfl_sync(0xffffffff, cached_rdma_channel_head, src_rdma_rank);
      auto src_rdma_tail =
          __shfl_sync(0xffffffff, cached_rdma_channel_tail, src_rdma_rank);

      // Iterate over every token from the RDMA buffer
      for (int i = src_rdma_head, num_tokens_sent = 0; i < src_rdma_tail; ++i) {
        auto rdma_slot_idx = i % num_max_rdma_chunked_recv_tokens;
        void* shifted = rdma_channel_data.recv_buffer(src_rdma_rank) +
                        rdma_slot_idx * num_bytes_per_rdma_token;
        auto src_meta = ld_nc_global(reinterpret_cast<SourceMeta*>(
            reinterpret_cast<int8_t*>(shifted) + hidden_bytes));
        lane_id == src_rdma_rank ? (num_tokens_to_recv_from_rdma -= 1) : 0;
        bool is_in_dst_nvl_rank = src_meta.is_token_in_nvl_rank(dst_nvl_rank);
        if (lane_id == src_rdma_rank) {
          auto cached_head = is_in_dst_nvl_rank ? rdma_nvl_token_idx : -1;
          rdma_nvl_token_idx += is_in_dst_nvl_rank;
          if (!kCachedMode) send_nvl_head[i * NUM_MAX_NVL_PEERS] = cached_head;
        }
        if (!is_in_dst_nvl_rank) continue;

        // Get an empty slot
        int dst_slot_idx =
            (cached_nvl_channel_tail++) % num_max_nvl_chunked_recv_tokens;

        // Copy data
        UNROLLED_WARP_COPY(5,
                           lane_id,
                           hidden_int4,
                           nvl_channel_x.buffer() + dst_slot_idx * hidden_int4,
                           reinterpret_cast<int4*>(shifted),
                           ld_nc_global,
                           st_na_global);
        shifted = reinterpret_cast<int4*>(shifted) + hidden_int4;

        // Copy source meta
        if (lane_id == 0)
          st_na_global(nvl_channel_src_meta.buffer() + dst_slot_idx, src_meta);
        shifted = reinterpret_cast<SourceMeta*>(shifted) + 1;

        // Copy `x_scales`
        UNROLLED_WARP_COPY(
            1,
            lane_id,
            num_scales,
            nvl_channel_x_scales.buffer() + dst_slot_idx * num_scales,
            reinterpret_cast<float*>(shifted),
            ld_nc_global,
            st_na_global);
        shifted = reinterpret_cast<float*>(shifted) + num_scales;

        // Copy `topk_idx` and `topk_weights`
        // NOTES: do not use `shifted` after this `if`, because only several
        // lanes are shifted
        if (lane_id < num_topk) {
          // Read
          auto idx_value =
              ld_nc_global(reinterpret_cast<int*>(shifted) + lane_id);
          shifted = reinterpret_cast<int*>(shifted) + num_topk;
          auto weight_value =
              ld_nc_global(reinterpret_cast<float*>(shifted) + lane_id);

          // Transform and write
          idx_value = (idx_value >= dst_rank_expert_begin &&
                       idx_value < dst_rank_expert_end)
                          ? idx_value - dst_rank_expert_begin
                          : -1;
          st_na_global(
              nvl_channel_topk_idx.buffer() + dst_slot_idx * num_topk + lane_id,
              idx_value);
          weight_value = idx_value >= 0 ? weight_value : 0.0f;
          st_na_global(nvl_channel_topk_weights.buffer() +
                           dst_slot_idx * num_topk + lane_id,
                       weight_value);
        }

        // In case of insufficient NVL buffers, early stopping
        if ((++num_tokens_sent) == num_max_nvl_chunked_send_tokens)
          src_rdma_tail = i + 1;
      }

      // Sync head index
      if (lane_id == src_rdma_rank)
        forward_channel_head[dst_nvl_rank][src_rdma_rank] =
            (cached_rdma_channel_head = src_rdma_tail);

      // Move tail index
      __syncwarp();
      if (lane_id == 0)
        st_release_sys_global(nvl_channel_tail.buffer(),
                              cached_nvl_channel_tail);
    }

    // Retired
    __syncwarp();
    if (lane_id == 0) forward_channel_retired[dst_nvl_rank] = true;
  } else if (warp_role == WarpRole::kForwarderCoordinator) {
    // Extra warps for forwarder coordinator should exit directly
    if (target_rank > 0) return;

    // Forward warp coordinator
    EP_STATIC_ASSERT(kNumRDMARanks <= 32, "Invalid number of RDMA peers");

    // Clean shared memory
    EP_STATIC_ASSERT(NUM_MAX_NVL_PEERS <= 32, "Invalid number of NVL peers");
#pragma unroll
    for (int i = lane_id; i < kNumRDMARanks * NUM_MAX_NVL_PEERS; i += 32)
      forward_channel_head[i % NUM_MAX_NVL_PEERS][i / NUM_MAX_NVL_PEERS] = 0;
    if (lane_id < NUM_MAX_NVL_PEERS) forward_channel_retired[lane_id] = false;
    sync_forwarder_smem();

    int last_head = 0, target_rdma = lane_id < kNumRDMARanks ? lane_id : 0;
    while (true) {
      // Find minimum head
      int min_head = std::numeric_limits<int>::max();
#pragma unroll
      for (int i = 0; i < NUM_MAX_NVL_PEERS; ++i)
        if (!forward_channel_retired[i])
          min_head = min(min_head, forward_channel_head[i][target_rdma]);
      if (__all_sync(0xffffffff, min_head == std::numeric_limits<int>::max()))
        break;

      // Update remote head
      if (min_head != std::numeric_limits<int>::max() &&
          min_head >= last_head + num_max_rdma_chunked_send_tokens &&
          lane_id < kNumRDMARanks) {
        nvshmemi_ibgda_amo_nonfetch_add(
            rdma_channel_head.buffer(rdma_rank),
            min_head - last_head,
            translate_dst_rdma_rank<kLowLatencyMode>(lane_id, nvl_rank),
            channel_id,
            lane_id == rdma_rank);
        last_head = min_head;
      }

      // Nanosleep and let other warps work
      __nanosleep(NUM_WAIT_NANOSECONDS);
    }
  } else {
    // NVL consumers
    // Retrieve rank offset from barrier results (each lane's register stores an
    // RDMA rank)
    int src_nvl_rank = target_rank, total_offset = 0;
    EP_STATIC_ASSERT(kNumRDMARanks <= 32, "Invalid number of RDMA peers");
    if (lane_id < kNumRDMARanks &&
        lane_id * NUM_MAX_NVL_PEERS + src_nvl_rank > 0)
      total_offset = recv_gbl_rank_prefix_sum[lane_id * NUM_MAX_NVL_PEERS +
                                              src_nvl_rank - 1];

    // Receive channel offsets
    int start_offset = 0, end_offset = 0, num_tokens_to_recv;
    auto start_time = clock64();
    while (lane_id < kNumRDMARanks) {
      start_offset =
          ld_volatile_global(nvl_channel_prefix_start.buffer() + lane_id);
      end_offset =
          ld_volatile_global(nvl_channel_prefix_end.buffer() + lane_id);
      if (start_offset < 0 && end_offset < 0) {
        start_offset = -start_offset - 1, end_offset = -end_offset - 1;
        total_offset += start_offset;
        break;
      }

      // Timeout check
      if (clock64() - start_time > NUM_TIMEOUT_CYCLES) {
        printf(
            "DeepEP dispatch NVL receiver timeout, channel: %d, RDMA: %d, nvl: "
            "%d, src RDMA: %d, src nvl: %d, start: %d, end: %d\n",
            channel_id,
            rdma_rank,
            nvl_rank,
            lane_id,
            src_nvl_rank,
            start_offset,
            end_offset);
        trap();
      }
    }
    num_tokens_to_recv = warp_reduce_sum(end_offset - start_offset);

    // Save for combine usage
    if (lane_id < kNumRDMARanks && !kCachedMode)
      recv_gbl_channel_prefix_matrix[(lane_id * NUM_MAX_NVL_PEERS +
                                      src_nvl_rank) *
                                         num_channels +
                                     channel_id] = total_offset;
    __syncwarp();

    int cached_channel_head_idx = 0, cached_channel_tail_idx = 0;
    while (num_tokens_to_recv > 0) {
      // Check channel status by lane 0
      start_time = clock64();
      while (lane_id == 0) {
        // Ready to copy
        if (cached_channel_head_idx != cached_channel_tail_idx) break;
        cached_channel_tail_idx =
            ld_acquire_sys_global(nvl_channel_tail.buffer());

        // Timeout check
        if (clock64() - start_time > NUM_TIMEOUT_CYCLES) {
          printf(
              "DeepEP dispatch NVL receiver timeout, channel: %d, RDMA: %d, "
              "nvl: %d, src NVL: %d, head: %d, tail: %d\n",
              channel_id,
              rdma_rank,
              nvl_rank,
              src_nvl_rank,
              cached_channel_head_idx,
              cached_channel_tail_idx);
          trap();
        }
      }

      // Sync queue tail
      cached_channel_tail_idx =
          __shfl_sync(0xffffffff, cached_channel_tail_idx, 0);

      // Copy data
      int num_recv_tokens = cached_channel_tail_idx - cached_channel_head_idx;
      for (int chunk_idx = 0; chunk_idx < num_recv_tokens;
           ++chunk_idx, --num_tokens_to_recv) {
        int token_idx_in_buffer =
            (cached_channel_head_idx++) % num_max_nvl_chunked_recv_tokens;
        auto meta =
            ld_nc_global(nvl_channel_src_meta.buffer() + token_idx_in_buffer);
        int64_t recv_token_idx =
            __shfl_sync(0xffffffff, total_offset, meta.src_rdma_rank);
        (lane_id == meta.src_rdma_rank) ? (total_offset += 1) : 0;

        // Copy data
        UNROLLED_WARP_COPY(
            5,
            lane_id,
            hidden_int4,
            recv_x + recv_token_idx * hidden_int4,
            nvl_channel_x.buffer() + token_idx_in_buffer * hidden_int4,
            ld_nc_global,
            st_na_global);

        // Copy source meta
        if (lane_id == 0 && !kCachedMode)
          st_na_global(recv_src_meta + recv_token_idx, meta);

        // Copy scales
        UNROLLED_WARP_COPY(
            1,
            lane_id,
            num_scales,
            recv_x_scales + recv_token_idx * num_scales,
            nvl_channel_x_scales.buffer() + token_idx_in_buffer * num_scales,
            ld_nc_global,
            st_na_global);

        // Copy `topk_idx` and `topk_weights`
        if (lane_id < num_topk) {
          auto recv_idx = recv_token_idx * num_topk + lane_id;
          auto buffer_idx = token_idx_in_buffer * num_topk + lane_id;
          st_na_global(recv_topk_idx + recv_idx,
                       static_cast<int64_t>(ld_nc_global(
                           nvl_channel_topk_idx.buffer() + buffer_idx)));
          st_na_global(
              recv_topk_weights + recv_idx,
              ld_nc_global(nvl_channel_topk_weights.buffer() + buffer_idx));
        }
      }

      // Move queue
      __syncwarp();
      if (lane_id == 0)
        st_relaxed_sys_global(nvl_channel_head.buffer(),
                              cached_channel_head_idx);
    }
  }
}

void dispatch(void* recv_x,
              float* recv_x_scales,
              int64_t* recv_topk_idx,
              float* recv_topk_weights,
              void* recv_src_meta,
              const void* x,
              const float* x_scales,
              const int64_t* topk_idx,
              const float* topk_weights,
              int* send_rdma_head,
              int* send_nvl_head,
              int* recv_rdma_channel_prefix_matrix,
              int* recv_gbl_channel_prefix_matrix,
              const int* rdma_channel_prefix_matrix,
              const int* recv_rdma_rank_prefix_sum,
              const int* gbl_channel_prefix_matrix,
              const int* recv_gbl_rank_prefix_sum,
              int num_tokens,
              int hidden_int4,
              int num_scales,
              int num_topk,
              int num_experts,
              const bool* is_token_in_rank,
              void* rdma_buffer_ptr,
              int num_max_rdma_chunked_send_tokens,
              int num_max_rdma_chunked_recv_tokens,
              void** buffer_ptrs,
              int num_max_nvl_chunked_send_tokens,
              int num_max_nvl_chunked_recv_tokens,
              int rank,
              int num_ranks,
              bool is_cached_dispatch,
              cudaStream_t stream,
              int num_channels,
              bool low_latency_mode) {
  constexpr int kNumDispatchRDMASenderWarps = 7;

#define DISPATCH_LAUNCH_CASE(num_rdma_ranks)                                \
  {                                                                         \
    auto dispatch_func =                                                    \
        low_latency_mode                                                    \
            ? (is_cached_dispatch ? dispatch<true,                          \
                                             num_rdma_ranks,                \
                                             true,                          \
                                             kNumDispatchRDMASenderWarps>   \
                                  : dispatch<true,                          \
                                             num_rdma_ranks,                \
                                             false,                         \
                                             kNumDispatchRDMASenderWarps>)  \
            : (is_cached_dispatch ? dispatch<false,                         \
                                             num_rdma_ranks,                \
                                             true,                          \
                                             kNumDispatchRDMASenderWarps>   \
                                  : dispatch<false,                         \
                                             num_rdma_ranks,                \
                                             false,                         \
                                             kNumDispatchRDMASenderWarps>); \
    LAUNCH_KERNEL(&cfg,                                                     \
                  dispatch_func,                                            \
                  reinterpret_cast<int4*>(recv_x),                          \
                  recv_x_scales,                                            \
                  recv_topk_idx,                                            \
                  recv_topk_weights,                                        \
                  reinterpret_cast<SourceMeta*>(recv_src_meta),             \
                  reinterpret_cast<const int4*>(x),                         \
                  x_scales,                                                 \
                  topk_idx,                                                 \
                  topk_weights,                                             \
                  send_rdma_head,                                           \
                  send_nvl_head,                                            \
                  recv_rdma_channel_prefix_matrix,                          \
                  recv_gbl_channel_prefix_matrix,                           \
                  rdma_channel_prefix_matrix,                               \
                  recv_rdma_rank_prefix_sum,                                \
                  gbl_channel_prefix_matrix,                                \
                  recv_gbl_rank_prefix_sum,                                 \
                  num_tokens,                                               \
                  hidden_int4,                                              \
                  num_scales,                                               \
                  num_topk,                                                 \
                  num_experts,                                              \
                  is_token_in_rank,                                         \
                  rdma_buffer_ptr,                                          \
                  num_max_rdma_chunked_send_tokens,                         \
                  num_max_rdma_chunked_recv_tokens,                         \
                  buffer_ptrs,                                              \
                  num_max_nvl_chunked_send_tokens,                          \
                  num_max_nvl_chunked_recv_tokens,                          \
                  rank,                                                     \
                  num_ranks);                                               \
  }                                                                         \
  break

  EP_HOST_ASSERT((topk_idx == nullptr) == (topk_weights == nullptr));
  EP_HOST_ASSERT((recv_topk_idx == nullptr) == (recv_topk_weights == nullptr));

  SETUP_LAUNCH_CONFIG(
      num_channels * 2,
      (kNumDispatchRDMASenderWarps + 1 + NUM_MAX_NVL_PEERS) * 32,
      stream);
  SWITCH_RDMA_RANKS(DISPATCH_LAUNCH_CASE);
#undef DISPATCH_LAUNCH_CASE
}

template <bool kLowLatencyMode>
__global__ void cached_notify(const int rdma_clean_offset,
                              const int rdma_num_int_clean,
                              const int nvl_clean_offset,
                              const int nvl_num_int_clean,
                              int* combined_rdma_head,
                              int num_combined_tokens,
                              int num_channels,
                              const int* rdma_channel_prefix_matrix,
                              const int* rdma_rank_prefix_sum,
                              int* combined_nvl_head,
                              void* rdma_buffer_ptr,
                              void** buffer_ptrs,
                              int** task_fifo_ptrs,
                              int head,
                              int rank,
                              int num_ranks,
                              bool is_cached_dispatch,
                              const nvshmem_team_t rdma_team) {
  auto sm_id = static_cast<int>(blockIdx.x);
  auto thread_id = static_cast<int>(threadIdx.x);
  auto num_threads = static_cast<int>(blockDim.x);
  auto num_warps = num_threads / 32;
  auto warp_id = thread_id / 32;
  auto lane_id = get_lane_id();

  auto nvl_rank = rank % NUM_MAX_NVL_PEERS;
  auto num_rdma_ranks = num_ranks / NUM_MAX_NVL_PEERS;

  // Using two SMs, which clean the RDMA/NVL buffer respectively
  if (sm_id == 0) {
    // Barrier for RDMA
    if (thread_id == 0)
      nvshmem_barrier_with_same_gpu_idx<kLowLatencyMode>(rdma_team);
    __syncthreads();

    // Clean
    auto rdma_buffer_ptr_int = reinterpret_cast<int*>(rdma_buffer_ptr);
#pragma unroll
    for (int i = thread_id; i < rdma_num_int_clean; i += num_threads)
      rdma_buffer_ptr_int[rdma_clean_offset + i] = 0;
    nvshmem_fence();
    __syncthreads();

    // Barrier again
    if (thread_id == 0)
      nvshmem_barrier_with_same_gpu_idx<kLowLatencyMode>(rdma_team);
  } else if (sm_id == 1) {
    // Barrier for NVL
    barrier_device<NUM_MAX_NVL_PEERS>(task_fifo_ptrs, head, nvl_rank);
    move_fifo_slots<NUM_MAX_NVL_PEERS>(head);
    __syncthreads();

    // Clean
    auto nvl_buffer_ptr_int = reinterpret_cast<int*>(buffer_ptrs[nvl_rank]);
#pragma unroll
    for (int i = thread_id; i < nvl_num_int_clean; i += num_threads)
      nvl_buffer_ptr_int[nvl_clean_offset + i] = 0;
    memory_fence();
    __syncthreads();

    // Barrier again
    barrier_device<NUM_MAX_NVL_PEERS>(task_fifo_ptrs, head, nvl_rank);
    move_fifo_slots<NUM_MAX_NVL_PEERS>(head);
  } else if (sm_id == 2) {
    if (is_cached_dispatch) return;

    EP_DEVICE_ASSERT(num_warps >= num_channels);
    EP_DEVICE_ASSERT(num_rdma_ranks <= 32);

    // Iterate in reverse order
    if (lane_id < num_rdma_ranks && warp_id < num_channels) {
      int token_start_idx, token_end_idx;
      get_channel_task_range(num_combined_tokens,
                             num_channels,
                             warp_id,
                             token_start_idx,
                             token_end_idx);

      // NOTES: `1 << 25` is a heuristic large number
      int last_head = 1 << 25;
      for (int token_idx = token_end_idx - 1; token_idx >= token_start_idx;
           --token_idx) {
        auto current_head =
            __ldg(combined_rdma_head + token_idx * num_rdma_ranks + lane_id);
        if (current_head < 0) {
          combined_rdma_head[token_idx * num_rdma_ranks + lane_id] =
              -last_head - 1;
        } else {
          last_head = current_head;
        }
      }
    }
  } else {
    if (is_cached_dispatch) return;

    EP_DEVICE_ASSERT(num_warps >= num_channels);
    EP_DEVICE_ASSERT(rdma_channel_prefix_matrix != nullptr &&
                     rdma_rank_prefix_sum != nullptr);
    EP_STATIC_ASSERT(NUM_MAX_NVL_PEERS <= 32, "Too many NVL peers");

    if (lane_id < NUM_MAX_NVL_PEERS && warp_id < num_channels) {
      for (int dst_rdma_rank = sm_id - 3; dst_rdma_rank < num_rdma_ranks;
           dst_rdma_rank += num_channels * 2 - 3) {
        // Iterate in reverse order
        int token_start_idx =
            warp_id == 0
                ? 0
                : rdma_channel_prefix_matrix[dst_rdma_rank * num_channels +
                                             warp_id - 1];
        int token_end_idx =
            rdma_channel_prefix_matrix[dst_rdma_rank * num_channels + warp_id];
        int shift =
            dst_rdma_rank == 0 ? 0 : rdma_rank_prefix_sum[dst_rdma_rank - 1];
        token_start_idx += shift, token_end_idx += shift;

        // NOTES: `1 << 25` is a heuristic large number
        int last_head = 1 << 25;
#pragma unroll
        for (int token_idx = token_end_idx - 1; token_idx >= token_start_idx;
             --token_idx) {
          auto current_head = __ldg(combined_nvl_head +
                                    token_idx * NUM_MAX_NVL_PEERS + lane_id);
          if (current_head < 0) {
            combined_nvl_head[token_idx * NUM_MAX_NVL_PEERS + lane_id] =
                -last_head - 1;
          } else {
            last_head = current_head;
          }
        }
      }
    }
  }
}

void cached_notify(int hidden_int4,
                   int num_scales,
                   int num_topk_idx,
                   int num_topk_weights,
                   int num_ranks,
                   int num_channels,
                   int num_combined_tokens,
                   int* combined_rdma_head,
                   const int* rdma_channel_prefix_matrix,
                   const int* rdma_rank_prefix_sum,
                   int* combined_nvl_head,
                   void* rdma_buffer_ptr,
                   int num_max_rdma_chunked_recv_tokens,
                   void** buffer_ptrs,
                   int num_max_nvl_chunked_recv_tokens,
                   int** task_fifo_ptrs,
                   int head,
                   int rank,
                   cudaStream_t stream,
                   int64_t num_rdma_bytes,
                   int64_t num_nvl_bytes,
                   bool is_cached_dispatch,
                   bool low_latency_mode) {
  const int num_threads = std::max(128, 32 * num_channels);
  const auto num_rdma_ranks = num_ranks / NUM_MAX_NVL_PEERS;

  // Get clean meta
  auto rdma_clean_meta = get_rdma_clean_meta(hidden_int4,
                                             num_scales,
                                             num_topk_idx,
                                             num_topk_weights,
                                             num_rdma_ranks,
                                             num_max_rdma_chunked_recv_tokens,
                                             num_channels);
  auto nvl_clean_meta = get_nvl_clean_meta(hidden_int4,
                                           num_scales,
                                           num_topk_idx,
                                           num_topk_weights,
                                           num_rdma_ranks,
                                           NUM_MAX_NVL_PEERS,
                                           num_max_nvl_chunked_recv_tokens,
                                           num_channels);
  EP_HOST_ASSERT((rdma_clean_meta.first + rdma_clean_meta.second) *
                     sizeof(int) <=
                 num_rdma_bytes);
  EP_HOST_ASSERT((nvl_clean_meta.first + nvl_clean_meta.second) * sizeof(int) <=
                 num_nvl_bytes);
  EP_HOST_ASSERT(num_rdma_bytes < std::numeric_limits<int>::max());
  EP_HOST_ASSERT(num_nvl_bytes < std::numeric_limits<int>::max());
  EP_HOST_ASSERT(num_channels * 2 > 3);

  // Launch kernel
  auto cached_notify_func =
      low_latency_mode ? cached_notify<true> : cached_notify<false>;
  SETUP_LAUNCH_CONFIG(num_channels * 2, num_threads, stream);
  LAUNCH_KERNEL(&cfg,
                cached_notify_func,
                rdma_clean_meta.first,
                rdma_clean_meta.second,
                nvl_clean_meta.first,
                nvl_clean_meta.second,
                combined_rdma_head,
                num_combined_tokens,
                num_channels,
                rdma_channel_prefix_matrix,
                rdma_rank_prefix_sum,
                combined_nvl_head,
                rdma_buffer_ptr,
                buffer_ptrs,
                task_fifo_ptrs,
                head,
                rank,
                num_ranks,
                is_cached_dispatch,
                cpu_rdma_team);
}

template <int kNumRanks,
          typename dtype_t,
          int kMaxNumRanks,
          typename ReceiveFn,
          typename ReceiveTWFn>
__device__ int combine_token(bool is_token_in_rank,
                             int head_idx,
                             int lane_id,
                             int hidden_int4,
                             int num_topk,
                             int4* combined_row,
                             float* combined_topk_weights,
                             int num_max_recv_tokens,
                             const ReceiveFn& recv_fn,
                             const ReceiveTWFn& recv_tw_fn) {
  constexpr auto kDtypePerInt4 = sizeof(int4) / sizeof(dtype_t);

  // Broadcast current heads
  // Lane `i` holds the head of rank `i` and `is_token_in_rank`
  EP_STATIC_ASSERT(kMaxNumRanks <= 32, "Too many ranks");
  int num_topk_ranks = 0, topk_ranks[kMaxNumRanks], slot_indices[kMaxNumRanks];
#pragma unroll
  for (int i = 0; i < kNumRanks; ++i)
    if (__shfl_sync(0xffffffff, is_token_in_rank, i)) {
      slot_indices[num_topk_ranks] =
          __shfl_sync(0xffffffff, head_idx, i) % num_max_recv_tokens;
      topk_ranks[num_topk_ranks++] = i;
    }
  EP_DEVICE_ASSERT(num_topk_ranks <= kMaxNumRanks);

// Reduce data
#pragma unroll
  for (int i = lane_id; i < hidden_int4; i += 32) {
    // Read buffers
    // TODO(Xreki): maybe too many registers here
    int4 recv_value_int4[kMaxNumRanks];
#pragma unroll
    for (int j = 0; j < num_topk_ranks; ++j)
      recv_value_int4[j] = recv_fn(topk_ranks[j], slot_indices[j], i);

    // Reduce all-to-all results
    float values[kDtypePerInt4] = {0};
#pragma unroll
    for (int j = 0; j < num_topk_ranks; ++j) {
      auto recv_value_dtypes =
          reinterpret_cast<const dtype_t*>(&recv_value_int4[j]);
#pragma unroll
      for (int k = 0; k < kDtypePerInt4; ++k)
        values[k] += static_cast<float>(recv_value_dtypes[k]);
    }

    // Cast back to `dtype_t` and write
    int4 out_int4;
    auto out_dtypes = reinterpret_cast<dtype_t*>(&out_int4);
#pragma unroll
    for (int j = 0; j < kDtypePerInt4; ++j)
      out_dtypes[j] = static_cast<dtype_t>(values[j]);
    st_na_global(combined_row + i, out_int4);
  }

  // Reduce `topk_weights`
  if (lane_id < num_topk) {
    float value = 0;
#pragma unroll
    for (int i = 0; i < num_topk_ranks; ++i)
      value += recv_tw_fn(topk_ranks[i], slot_indices[i], lane_id);
    st_na_global(combined_topk_weights + lane_id, value);
  }

  // Return the minimum top-k rank
  return topk_ranks[0];
}

template <
    bool kLowLatencyMode,
    int kNumRDMARanks,
    typename dtype_t,
    int kNumCombineForwarderWarps,
    int kNumTopkRDMARanks = get_num_topk_rdma_ranks(kNumRDMARanks),
    int kNumWarpsPerForwarder = (kNumCombineForwarderWarps / kNumRDMARanks > 0)
                                    ? kNumCombineForwarderWarps / kNumRDMARanks
                                    : 1,
    int kNumForwarders = kNumRDMARanks* kNumWarpsPerForwarder,
    int kNumRDMAReceivers = kNumForwarders + NUM_MAX_NVL_PEERS>
__global__ void __launch_bounds__((NUM_MAX_NVL_PEERS + 1 + kNumForwarders) * 32,
                                  1)
    combine(int4* combined_x,
            float* combined_topk_weights,
            const bool* is_combined_token_in_rank,
            const int4* x,
            const float* topk_weights,
            const int* combined_rdma_head,
            const int* combined_nvl_head,
            const SourceMeta* src_meta,
            const int* rdma_channel_prefix_matrix,
            const int* rdma_rank_prefix_sum,
            const int* gbl_channel_prefix_matrix,
            int num_tokens,
            int num_combined_tokens,
            int hidden,
            int num_topk,
            void* rdma_buffer_ptr,
            int num_max_rdma_chunked_send_tokens,
            int num_max_rdma_chunked_recv_tokens,
            void** buffer_ptrs,
            int num_max_nvl_chunked_send_tokens,
            int num_max_nvl_chunked_recv_tokens,
            int rank,
            int num_ranks) {
  enum class WarpRole {
    kNVLSender,
    kNVLAndRDMAForwarder,
    kRDMAReceiver,
    kCoordinator
  };

  const auto sm_id = static_cast<int>(blockIdx.x);
  const auto num_threads = static_cast<int>(blockDim.x),
             num_warps = num_threads / 32;
  const auto thread_id = static_cast<int>(threadIdx.x), lane_id = get_lane_id();
  const auto num_channels = static_cast<int>(gridDim.x) / 2,
             channel_id = sm_id / 2;
  const bool is_rdma_receiver_sm = sm_id % 2 == 1;

  EP_DEVICE_ASSERT(num_topk <= 32);
  EP_DEVICE_ASSERT(hidden % (sizeof(int4) / sizeof(dtype_t)) == 0);
  const auto hidden_int4 = hidden / (sizeof(int4) / sizeof(dtype_t));

  // NOTES: we decouple a channel into 2 SMs
  const auto rdma_rank = rank / NUM_MAX_NVL_PEERS,
             nvl_rank = rank % NUM_MAX_NVL_PEERS;
  auto role_meta = [=]() -> std::pair<WarpRole, int> {
    auto warp_id = thread_id / 32;
    if (!is_rdma_receiver_sm) {
      if (warp_id < NUM_MAX_NVL_PEERS) {
        auto shuffled_warp_id = warp_id;
        shuffled_warp_id = (shuffled_warp_id + channel_id) % NUM_MAX_NVL_PEERS;
        return {WarpRole::kNVLSender, shuffled_warp_id};
      } else if (warp_id < NUM_MAX_NVL_PEERS + kNumForwarders) {
        auto shuffled_warp_id = warp_id - NUM_MAX_NVL_PEERS;
        shuffled_warp_id = (shuffled_warp_id + channel_id) % kNumForwarders;
        return {WarpRole::kNVLAndRDMAForwarder, shuffled_warp_id};
      } else {
        return {WarpRole::kCoordinator, 0};
      }
    } else {
      if (warp_id < NUM_MAX_NVL_PEERS + kNumForwarders) {
        return {WarpRole::kRDMAReceiver, warp_id};
      } else {
        return {WarpRole::kCoordinator, 0};
      }
    }
  }();
  auto warp_role = role_meta.first;
  auto warp_id = role_meta.second;

  EP_DEVICE_ASSERT(num_warps == NUM_MAX_NVL_PEERS + kNumForwarders + 1);
  auto num_max_nvl_chunked_recv_tokens_per_rdma =
      num_max_nvl_chunked_recv_tokens / kNumRDMARanks;

  if (warp_role == WarpRole::kNVLSender) {
    // NVL producers
    const auto dst_nvl_rank = warp_id;

    // NVL layouts
    // NOTES: to avoid deadlocks, we use separate NVL buffers for different RDMA
    // sources
    auto dst_buffer_ptr = buffer_ptrs[dst_nvl_rank],
         local_buffer_ptr = buffer_ptrs[nvl_rank];
    auto nvl_channel_x =
        AsymBuffer<int4>(dst_buffer_ptr,
                         num_max_nvl_chunked_recv_tokens * hidden_int4,
                         NUM_MAX_NVL_PEERS,
                         channel_id,
                         num_channels,
                         nvl_rank)
            .advance_also(local_buffer_ptr);
    auto nvl_channel_src_meta =
        AsymBuffer<SourceMeta>(dst_buffer_ptr,
                               num_max_nvl_chunked_recv_tokens,
                               NUM_MAX_NVL_PEERS,
                               channel_id,
                               num_channels,
                               nvl_rank)
            .advance_also(local_buffer_ptr);
    auto nvl_channel_topk_weights =
        AsymBuffer<float>(dst_buffer_ptr,
                          num_max_nvl_chunked_recv_tokens * num_topk,
                          NUM_MAX_NVL_PEERS,
                          channel_id,
                          num_channels,
                          nvl_rank)
            .advance_also(local_buffer_ptr);
    auto nvl_channel_head = AsymBuffer<int>(local_buffer_ptr,
                                            kNumRDMARanks,
                                            NUM_MAX_NVL_PEERS,
                                            channel_id,
                                            num_channels,
                                            dst_nvl_rank)
                                .advance_also(dst_buffer_ptr);
    auto nvl_channel_tail = AsymBuffer<int>(dst_buffer_ptr,
                                            kNumRDMARanks,
                                            NUM_MAX_NVL_PEERS,
                                            channel_id,
                                            num_channels,
                                            nvl_rank)
                                .advance_also(local_buffer_ptr);

    // Get tasks for each RDMA lane
    int token_start_idx = 0, token_end_idx = 0;
    if (lane_id < kNumRDMARanks) {
      int prefix_idx =
          (lane_id * NUM_MAX_NVL_PEERS + dst_nvl_rank) * num_channels +
          channel_id;
      token_start_idx = gbl_channel_prefix_matrix[prefix_idx];
      token_end_idx = (prefix_idx == num_channels * num_ranks - 1)
                          ? num_tokens
                          : gbl_channel_prefix_matrix[prefix_idx + 1];
    }
    __syncwarp();

    // NOTES: here the cached value of each lane is only responsible for a
    // single RDMA buffer
    int cached_channel_head_idx = 0, cached_channel_tail_idx = 0;
    EP_STATIC_ASSERT(kNumRDMARanks <= 32, "Invalid number of RDMA peers");

    // Iterate over all tokens and send by chunks
    while (true) {
      // Exit if possible
      if (__all_sync(0xffffffff, token_start_idx >= token_end_idx)) break;

      // Decide next RDMA buffer to send
      bool is_lane_ready = false;
      auto start_time = clock64();
      while (true) {
        int num_used_slots = cached_channel_tail_idx - cached_channel_head_idx;
        is_lane_ready =
            lane_id < kNumRDMARanks && token_start_idx < token_end_idx &&
            num_max_nvl_chunked_recv_tokens_per_rdma - num_used_slots >=
                num_max_nvl_chunked_send_tokens;
        if (__any_sync(0xffffffff, is_lane_ready)) break;

        // Retry
        if (lane_id < kNumRDMARanks && token_start_idx < token_end_idx)
          cached_channel_head_idx =
              ld_volatile_global(nvl_channel_head.buffer() + lane_id);

        // Timeout check
        if (clock64() - start_time > NUM_TIMEOUT_CYCLES &&
            lane_id < kNumRDMARanks) {
          printf(
              "DeepEP combine NVL sender timeout, channel: %d, RDMA: %d, nvl: "
              "%d, dst NVL: %d, RDMA lane: %d, head: %d, tail: %d, start: %d, "
              "end: %d\n",
              channel_id,
              rdma_rank,
              nvl_rank,
              dst_nvl_rank,
              lane_id,
              ld_volatile_global(nvl_channel_head.buffer() + lane_id),
              cached_channel_tail_idx,
              token_start_idx,
              token_end_idx);
          trap();
        }
      }

      // Sync token start index and count
      for (int current_rdma_idx = 0; current_rdma_idx < kNumRDMARanks;
           ++current_rdma_idx) {
        if (__shfl_sync(0xffffffff,
                        (token_start_idx >= token_end_idx) || (!is_lane_ready),
                        current_rdma_idx))
          continue;

        // Sync token start index
        auto token_idx = static_cast<int64_t>(
            __shfl_sync(0xffffffff, token_start_idx, current_rdma_idx));
        int num_tokens_in_chunk =
            __shfl_sync(0xffffffff,
                        min(num_max_nvl_chunked_send_tokens,
                            token_end_idx - token_start_idx),
                        current_rdma_idx);

        // Send by chunk
        for (int chunk_idx = 0; chunk_idx < num_tokens_in_chunk;
             ++chunk_idx, ++token_idx) {
          // Get an empty slot
          int dst_slot_idx = 0;
          if (lane_id == current_rdma_idx) {
            dst_slot_idx = (cached_channel_tail_idx++) %
                           num_max_nvl_chunked_recv_tokens_per_rdma;
            dst_slot_idx =
                current_rdma_idx * num_max_nvl_chunked_recv_tokens_per_rdma +
                dst_slot_idx;
          }
          dst_slot_idx =
              __shfl_sync(0xffffffff, dst_slot_idx, current_rdma_idx);

          // Copy data
          auto shifted_x_buffers =
              nvl_channel_x.buffer() + dst_slot_idx * hidden_int4;
          auto shifted_x = x + token_idx * hidden_int4;
          UNROLLED_WARP_COPY(5,
                             lane_id,
                             hidden_int4,
                             shifted_x_buffers,
                             shifted_x,
                             ld_nc_global,
                             st_na_global);

          // Copy source meta
          if (lane_id == 0)
            st_na_global(nvl_channel_src_meta.buffer() + dst_slot_idx,
                         ld_nc_global(src_meta + token_idx));

          // Copy `topk_weights`
          if (lane_id < num_topk)
            st_na_global(
                nvl_channel_topk_weights.buffer() + dst_slot_idx * num_topk +
                    lane_id,
                ld_nc_global(topk_weights + token_idx * num_topk + lane_id));
        }
        lane_id == current_rdma_idx
            ? (token_start_idx = static_cast<int>(token_idx))
            : 0;
      }

      // Move queue tail
      __syncwarp();
      if (lane_id < kNumRDMARanks && is_lane_ready)
        st_release_sys_global(nvl_channel_tail.buffer() + lane_id,
                              cached_channel_tail_idx);
    }
  } else {
    // Combiners and coordinators
    // RDMA symmetric layout
    auto hidden_bytes = hidden_int4 * sizeof(int4);
    auto num_bytes_per_rdma_token =
        get_num_bytes_per_rdma_token(hidden_int4, 0, 0, num_topk);
    auto rdma_channel_data = SymBuffer<int8_t>(
        rdma_buffer_ptr,
        num_max_rdma_chunked_recv_tokens * num_bytes_per_rdma_token,
        kNumRDMARanks,
        channel_id,
        num_channels);
    auto rdma_channel_head = SymBuffer<uint64_t, false>(
        rdma_buffer_ptr, 1, kNumRDMARanks, channel_id, num_channels);
    auto rdma_channel_tail = SymBuffer<uint64_t, false>(
        rdma_buffer_ptr, 1, kNumRDMARanks, channel_id, num_channels);

    // NVL layouts
    void* local_nvl_buffer = buffer_ptrs[nvl_rank];
    void* nvl_buffers[NUM_MAX_NVL_PEERS];
#pragma unroll
    for (int i = 0; i < NUM_MAX_NVL_PEERS; ++i) nvl_buffers[i] = buffer_ptrs[i];
    auto nvl_channel_x =
        AsymBuffer<int4>(local_nvl_buffer,
                         num_max_nvl_chunked_recv_tokens * hidden_int4,
                         NUM_MAX_NVL_PEERS,
                         channel_id,
                         num_channels)
            .advance_also<NUM_MAX_NVL_PEERS>(nvl_buffers);
    auto nvl_channel_src_meta =
        AsymBuffer<SourceMeta>(local_nvl_buffer,
                               num_max_nvl_chunked_recv_tokens,
                               NUM_MAX_NVL_PEERS,
                               channel_id,
                               num_channels)
            .advance_also<NUM_MAX_NVL_PEERS>(nvl_buffers);
    auto nvl_channel_topk_weights =
        AsymBuffer<float>(local_nvl_buffer,
                          num_max_nvl_chunked_recv_tokens * num_topk,
                          NUM_MAX_NVL_PEERS,
                          channel_id,
                          num_channels)
            .advance_also<NUM_MAX_NVL_PEERS>(nvl_buffers);
    auto nvl_channel_head =
        AsymBuffer<int, NUM_MAX_NVL_PEERS>(nvl_buffers,
                                           kNumRDMARanks,
                                           NUM_MAX_NVL_PEERS,
                                           channel_id,
                                           num_channels,
                                           nvl_rank)
            .advance_also(local_nvl_buffer);
    auto nvl_channel_tail = AsymBuffer<int>(local_nvl_buffer,
                                            kNumRDMARanks,
                                            NUM_MAX_NVL_PEERS,
                                            channel_id,
                                            num_channels)
                                .advance_also<NUM_MAX_NVL_PEERS>(nvl_buffers);

    // Combiner warp synchronization
    __shared__ volatile int forwarder_nvl_head[kNumForwarders]
                                              [NUM_MAX_NVL_PEERS];
    __shared__ volatile bool forwarder_retired[kNumForwarders];
    __shared__ volatile int rdma_receiver_rdma_head[kNumRDMAReceivers]
                                                   [kNumRDMARanks];
    __shared__ volatile bool rdma_receiver_retired[kNumRDMAReceivers];
    auto sync_forwarder_smem = [=]() {
      asm volatile("bar.sync 0, %0;" ::"r"((kNumForwarders + 1) * 32));
    };
    auto sync_rdma_receiver_smem = [=]() {
      asm volatile("bar.sync 1, %0;" ::"r"((kNumRDMAReceivers + 1) * 32));
    };

    if (warp_role == WarpRole::kNVLAndRDMAForwarder) {
      // Receive from NVL ranks and forward to RDMA ranks
      // NOTES: this part is using "large warps" for each RDMA ranks
      const auto dst_rdma_rank = warp_id / kNumWarpsPerForwarder;
      const auto sub_warp_id = warp_id % kNumWarpsPerForwarder;
      auto send_buffer = dst_rdma_rank == rdma_rank
                             ? rdma_channel_data.recv_buffer(dst_rdma_rank)
                             : rdma_channel_data.send_buffer(dst_rdma_rank);
      auto sync_large_warp = [=]() {
        if (kNumWarpsPerForwarder == 1) {
          __syncwarp();
        } else {
          asm volatile("bar.sync %0, %1;" ::"r"(dst_rdma_rank + 2),
                       "r"(kNumWarpsPerForwarder * 32));
        }
      };
      EP_STATIC_ASSERT(kNumWarpsPerForwarder == 1 || kNumRDMARanks + 2 <= 16,
                       "Barriers are not enough");

      // Advance to the corresponding NVL buffer
      nvl_channel_x.advance(dst_rdma_rank *
                            num_max_nvl_chunked_recv_tokens_per_rdma *
                            hidden_int4);
      nvl_channel_src_meta.advance(dst_rdma_rank *
                                   num_max_nvl_chunked_recv_tokens_per_rdma);
      nvl_channel_topk_weights.advance(
          dst_rdma_rank * num_max_nvl_chunked_recv_tokens_per_rdma * num_topk);
      nvl_channel_head.advance(dst_rdma_rank);
      nvl_channel_tail.advance(dst_rdma_rank);

      // Clean shared memory and sync
      EP_STATIC_ASSERT(NUM_MAX_NVL_PEERS <= 32, "Invalid number of NVL peers");
      lane_id < NUM_MAX_NVL_PEERS ? (forwarder_nvl_head[warp_id][lane_id] = 0)
                                  : 0;
      lane_id == 0 ? (forwarder_retired[warp_id] = false) : false;
      sync_forwarder_smem();

      // Get count and cached head
      int cached_nvl_channel_tail_idx = 0;
      int num_tokens_to_combine =
          rdma_channel_prefix_matrix[dst_rdma_rank * num_channels + channel_id];
      int num_tokens_prefix =
          channel_id == 0
              ? 0
              : rdma_channel_prefix_matrix[dst_rdma_rank * num_channels +
                                           channel_id - 1];
      num_tokens_to_combine -= num_tokens_prefix;
      num_tokens_prefix +=
          dst_rdma_rank == 0 ? 0 : rdma_rank_prefix_sum[dst_rdma_rank - 1];
      combined_nvl_head += num_tokens_prefix * NUM_MAX_NVL_PEERS;

      // Iterate over all tokens and combine by chunks
      for (int token_start_idx = 0; token_start_idx < num_tokens_to_combine;
           token_start_idx += num_max_rdma_chunked_send_tokens) {
        // Check destination queue emptiness, or wait a buffer to be released
        auto token_end_idx =
            min(token_start_idx + num_max_rdma_chunked_send_tokens,
                num_tokens_to_combine);
        auto num_chunked_tokens = token_end_idx - token_start_idx;
        auto start_time = clock64();
        while (sub_warp_id == 0 && lane_id == 0) {
          // Inequality: `num_max_rdma_chunked_recv_tokens - (tail - head) >=
          // num_chunked_tokens` Here, `token_start_idx` is the actual tail
          int num_used_slots =
              token_start_idx -
              ld_volatile_global(rdma_channel_head.buffer(dst_rdma_rank));
          if (num_max_rdma_chunked_recv_tokens - num_used_slots >=
              num_chunked_tokens)
            break;

          // Timeout check
          if (clock64() - start_time > NUM_TIMEOUT_CYCLES) {
            printf(
                "DeepEP combine forwarder (RDMA check) timeout, channel: %d, "
                "RDMA: %d, nvl: %d, dst RDMA: %d, head: %ld, tail: %d, "
                "chunked: %d\n",
                channel_id,
                rdma_rank,
                nvl_rank,
                dst_rdma_rank,
                ld_volatile_global(rdma_channel_head.buffer(dst_rdma_rank)),
                token_start_idx,
                num_chunked_tokens);
            trap();
          }
        }
        sync_large_warp();

        // Combine and write to the RDMA buffer
        for (int token_idx = token_start_idx + sub_warp_id;
             token_idx < token_end_idx;
             token_idx += kNumWarpsPerForwarder) {
          // Read expected head
          EP_STATIC_ASSERT(kNumRDMARanks <= 32, "Invalid number of RDMA peers");
          int expected_head = -1;
          if (lane_id < NUM_MAX_NVL_PEERS)
            expected_head = ld_nc_global(
                combined_nvl_head + token_idx * NUM_MAX_NVL_PEERS + lane_id);

          // Wait lanes to be ready
          start_time = clock64();
          while (cached_nvl_channel_tail_idx <= expected_head) {
            cached_nvl_channel_tail_idx =
                ld_acquire_sys_global(nvl_channel_tail.buffer(lane_id));

            // Timeout check
            if (clock64() - start_time > NUM_TIMEOUT_CYCLES &&
                lane_id < NUM_MAX_NVL_PEERS) {
              printf(
                  "DeepEP combine forwarder (NVL check) timeout, channel: %d, "
                  "RDMA: %d, nvl: %d, src NVL: %d, dst RDMA: %d, tail: %d, "
                  "waiting: %d, total: %d, sub: %d, large: %d, expected: %d\n",
                  channel_id,
                  rdma_rank,
                  nvl_rank,
                  lane_id,
                  dst_rdma_rank,
                  cached_nvl_channel_tail_idx,
                  token_idx,
                  num_tokens_to_combine,
                  sub_warp_id,
                  kNumWarpsPerForwarder,
                  expected_head);
              trap();
            }
          }

          // Combine current token
          auto rdma_slot_idx = token_idx % num_max_rdma_chunked_recv_tokens;
          void* shifted =
              send_buffer + rdma_slot_idx * num_bytes_per_rdma_token;
          auto recv_fn =
              [&](int src_nvl_rank, int slot_idx, int hidden_int4_idx) -> int4 {
            return ld_nc_global(nvl_channel_x.buffer(src_nvl_rank) +
                                slot_idx * hidden_int4 + hidden_int4_idx);
          };
          auto recv_tw_fn =
              [&](int src_nvl_rank, int slot_idx, int topk_idx) -> float {
            return ld_nc_global(nvl_channel_topk_weights.buffer(src_nvl_rank) +
                                slot_idx * num_topk + topk_idx);
          };
          combine_token<NUM_MAX_NVL_PEERS, dtype_t, NUM_MAX_NVL_PEERS>(
              expected_head >= 0,
              expected_head,
              lane_id,
              hidden_int4,
              num_topk,
              reinterpret_cast<int4*>(shifted),
              reinterpret_cast<float*>(reinterpret_cast<int8_t*>(shifted) +
                                       hidden_bytes + sizeof(SourceMeta)),
              num_max_nvl_chunked_recv_tokens_per_rdma,
              recv_fn,
              recv_tw_fn);

          // Update head
          if (lane_id < NUM_MAX_NVL_PEERS)
            expected_head < 0
                ? (forwarder_nvl_head[warp_id][lane_id] = -expected_head - 1)
                : (forwarder_nvl_head[warp_id][lane_id] = expected_head + 1);
        }
        sync_large_warp();

        // Issue RDMA send
        if (sub_warp_id == kNumWarpsPerForwarder - 1) {
          if (dst_rdma_rank != rdma_rank) {
            auto rdma_slot_idx =
                token_start_idx % num_max_rdma_chunked_recv_tokens;
            const size_t num_bytes_per_msg =
                num_chunked_tokens * num_bytes_per_rdma_token;
            const auto dst_ptr = reinterpret_cast<uint64_t>(
                rdma_channel_data.recv_buffer(rdma_rank) +
                rdma_slot_idx * num_bytes_per_rdma_token);
            const auto src_ptr = reinterpret_cast<uint64_t>(
                rdma_channel_data.send_buffer(dst_rdma_rank) +
                rdma_slot_idx * num_bytes_per_rdma_token);
            nvshmemi_ibgda_put_nbi_warp<true>(
                dst_ptr,
                src_ptr,
                num_bytes_per_msg,
                translate_dst_rdma_rank<kLowLatencyMode>(dst_rdma_rank,
                                                         nvl_rank),
                channel_id,
                lane_id,
                0);
          } else {
            memory_fence();
          }

          // Write new RDMA tail
          __syncwarp();
          if (lane_id == 0)
            nvshmemi_ibgda_amo_nonfetch_add(
                rdma_channel_tail.buffer(rdma_rank),
                num_chunked_tokens,
                translate_dst_rdma_rank<kLowLatencyMode>(dst_rdma_rank,
                                                         nvl_rank),
                channel_id,
                dst_rdma_rank == rdma_rank);
        }
      }

      // Retired
      __syncwarp();
      if (lane_id == 0) forwarder_retired[warp_id] = true;
    } else if (warp_role == WarpRole::kRDMAReceiver) {
      // Receive from RDMA ranks and write to the output tensor
      // Clean shared memory and sync
      EP_DEVICE_ASSERT(kNumRDMARanks <= 32);
      lane_id < kNumRDMARanks ? (rdma_receiver_rdma_head[warp_id][lane_id] = 0)
                              : 0;
      lane_id == 0 ? (rdma_receiver_retired[warp_id] = false) : 0;
      sync_rdma_receiver_smem();

      // The same tokens as the dispatch process
      int token_start_idx, token_end_idx;
      get_channel_task_range(num_combined_tokens,
                             num_channels,
                             channel_id,
                             token_start_idx,
                             token_end_idx);

      // Iterate over all tokens and combine
      int cached_channel_tail_idx = 0;
      for (int64_t token_idx = token_start_idx + warp_id;
           token_idx < token_end_idx;
           token_idx += kNumRDMAReceivers) {
        // Read expected head
        EP_STATIC_ASSERT(kNumRDMARanks <= 32, "Invalid number of RDMA peers");
        int expected_head = -1;
        if (lane_id < kNumRDMARanks) {
          expected_head = ld_nc_global(combined_rdma_head +
                                       token_idx * kNumRDMARanks + lane_id);
          (expected_head < 0)
              ? (rdma_receiver_rdma_head[warp_id][lane_id] = -expected_head - 1)
              : (rdma_receiver_rdma_head[warp_id][lane_id] = expected_head);
        }

        // Wait lanes to be ready
        auto start_time = clock64();
        while (cached_channel_tail_idx <= expected_head) {
          cached_channel_tail_idx = static_cast<int>(
              ld_acquire_sys_global(rdma_channel_tail.buffer(lane_id)));

          // Timeout check
          if (clock64() - start_time > NUM_TIMEOUT_CYCLES) {
            printf(
                "DeepEP combine RDMA receiver timeout, channel: %d, RDMA: %d, "
                "nvl: %d, src RDMA: %d, tail: %d, waiting: %ld, expect: %d\n",
                channel_id,
                rdma_rank,
                nvl_rank,
                lane_id,
                cached_channel_tail_idx,
                token_idx,
                expected_head);
            trap();
          }
        }
        __syncwarp();

        // Combine current token
        auto recv_fn =
            [&](int src_rdma_rank, int slot_idx, int hidden_int4_idx) -> int4 {
          return ld_nc_global(reinterpret_cast<const int4*>(
                                  rdma_channel_data.recv_buffer(src_rdma_rank) +
                                  slot_idx * num_bytes_per_rdma_token) +
                              hidden_int4_idx);
        };
        auto recv_tw_fn =
            [&](int src_rdma_rank, int slot_idx, int topk_idx) -> float {
          return ld_nc_global(reinterpret_cast<const float*>(
                                  rdma_channel_data.recv_buffer(src_rdma_rank) +
                                  slot_idx * num_bytes_per_rdma_token +
                                  hidden_bytes + sizeof(SourceMeta)) +
                              topk_idx);
        };
        combine_token<kNumRDMARanks, dtype_t, kNumTopkRDMARanks>(
            expected_head >= 0,
            expected_head,
            lane_id,
            hidden_int4,
            num_topk,
            combined_x + token_idx * hidden_int4,
            combined_topk_weights + token_idx * num_topk,
            num_max_rdma_chunked_recv_tokens,
            recv_fn,
            recv_tw_fn);
      }

      // Retired
      __syncwarp();
      if (lane_id == 0) rdma_receiver_retired[warp_id] = true;
    } else {
      // Coordinator
      // Sync shared memory status
      is_rdma_receiver_sm ? sync_rdma_receiver_smem() : sync_forwarder_smem();
      const auto num_warps_per_rdma_rank = kNumForwarders / kNumRDMARanks;

      int last_rdma_head = 0;
      int last_nvl_head[kNumRDMARanks] = {0};
      int dst_rdma_rank = lane_id < kNumRDMARanks ? lane_id : 0;
      int dst_nvl_rank = lane_id < NUM_MAX_NVL_PEERS ? lane_id : 0;
      EP_STATIC_ASSERT(kNumCombineForwarderWarps <= 32,
                       "Invalid number of forwarder warps");
      while (true) {
        // Retired
        if (is_rdma_receiver_sm &&
            __all_sync(
                0xffffffff,
                lane_id >= kNumRDMAReceivers || rdma_receiver_retired[lane_id]))
          break;
        if (!is_rdma_receiver_sm &&
            __all_sync(0xffffffff,
                       lane_id >= kNumForwarders || forwarder_retired[lane_id]))
          break;

        // Find minimum head for RDMA ranks
        if (is_rdma_receiver_sm) {
          int min_head = std::numeric_limits<int>::max();
#pragma unroll
          for (int i = 0; i < kNumRDMAReceivers; ++i)
            if (!rdma_receiver_retired[i])
              min_head =
                  min(min_head, rdma_receiver_rdma_head[i][dst_rdma_rank]);
          if (min_head != std::numeric_limits<int>::max() &&
              min_head >= last_rdma_head + num_max_rdma_chunked_send_tokens &&
              lane_id < kNumRDMARanks) {
            nvshmemi_ibgda_amo_nonfetch_add(
                rdma_channel_head.buffer(rdma_rank),
                min_head - last_rdma_head,
                translate_dst_rdma_rank<kLowLatencyMode>(dst_rdma_rank,
                                                         nvl_rank),
                channel_id,
                dst_rdma_rank == rdma_rank);
            last_rdma_head = min_head;
          }
        } else {
// Find minimum head for NVL ranks
#pragma unroll
          for (int i = 0; i < kNumRDMARanks; ++i) {
            int min_head = std::numeric_limits<int>::max();
#pragma unroll
            for (int j = 0; j < num_warps_per_rdma_rank; ++j)
              if (!forwarder_retired[i * num_warps_per_rdma_rank + j])
                min_head = min(min_head,
                               forwarder_nvl_head[i * num_warps_per_rdma_rank +
                                                  j][dst_nvl_rank]);
            if (min_head != std::numeric_limits<int>::max() &&
                min_head > last_nvl_head[i] && lane_id < NUM_MAX_NVL_PEERS)
              st_relaxed_sys_global(
                  nvl_channel_head.buffer_by(dst_nvl_rank) + i,
                  last_nvl_head[i] = min_head);
          }
        }

        // Nanosleep and let other warps work
        __nanosleep(NUM_WAIT_NANOSECONDS);
      }
    }
  }
}

void combine(cudaDataType_t type,
             void* combined_x,
             float* combined_topk_weights,
             const bool* is_combined_token_in_rank,
             const void* x,
             const float* topk_weights,
             const int* combined_rdma_head,
             const int* combined_nvl_head,
             const void* src_meta,
             const int* rdma_channel_prefix_matrix,
             const int* rdma_rank_prefix_sum,
             const int* gbl_channel_prefix_matrix,
             int num_tokens,
             int num_combined_tokens,
             int hidden,
             int num_topk,
             void* rdma_buffer_ptr,
             int num_max_rdma_chunked_send_tokens,
             int num_max_rdma_chunked_recv_tokens,
             void** buffer_ptrs,
             int num_max_nvl_chunked_send_tokens,
             int num_max_nvl_chunked_recv_tokens,
             int rank,
             int num_ranks,
             cudaStream_t stream,
             int num_channels,
             bool low_latency_mode) {
  constexpr int kNumCombineForwarderWarps = 16;

#define COMBINE_LAUNCH_CASE(num_rdma_ranks)                                    \
  {                                                                            \
    auto combine_func = low_latency_mode ? combine<true,                       \
                                                   num_rdma_ranks,             \
                                                   nv_bfloat16,                \
                                                   kNumCombineForwarderWarps>  \
                                         : combine<false,                      \
                                                   num_rdma_ranks,             \
                                                   nv_bfloat16,                \
                                                   kNumCombineForwarderWarps>; \
    LAUNCH_KERNEL(&cfg,                                                        \
                  combine_func,                                                \
                  reinterpret_cast<int4*>(combined_x),                         \
                  combined_topk_weights,                                       \
                  is_combined_token_in_rank,                                   \
                  reinterpret_cast<const int4*>(x),                            \
                  topk_weights,                                                \
                  combined_rdma_head,                                          \
                  combined_nvl_head,                                           \
                  reinterpret_cast<const SourceMeta*>(src_meta),               \
                  rdma_channel_prefix_matrix,                                  \
                  rdma_rank_prefix_sum,                                        \
                  gbl_channel_prefix_matrix,                                   \
                  num_tokens,                                                  \
                  num_combined_tokens,                                         \
                  hidden,                                                      \
                  num_topk,                                                    \
                  rdma_buffer_ptr,                                             \
                  num_max_rdma_chunked_send_tokens,                            \
                  num_max_rdma_chunked_recv_tokens,                            \
                  buffer_ptrs,                                                 \
                  num_max_nvl_chunked_send_tokens,                             \
                  num_max_nvl_chunked_recv_tokens,                             \
                  rank,                                                        \
                  num_ranks);                                                  \
  }                                                                            \
  break

  int num_rdma_ranks = num_ranks / NUM_MAX_NVL_PEERS;
  auto num_warps_per_forwarder =
      std::max(kNumCombineForwarderWarps / num_rdma_ranks, 1);
  int num_forwarder_warps = num_rdma_ranks * num_warps_per_forwarder;
  EP_HOST_ASSERT(num_forwarder_warps > 0 &&
                 num_forwarder_warps % num_rdma_ranks == 0);
  EP_HOST_ASSERT(num_max_nvl_chunked_recv_tokens % num_rdma_ranks == 0);
  EP_HOST_ASSERT(num_max_nvl_chunked_recv_tokens / num_rdma_ranks >
                 std::max(num_max_rdma_chunked_send_tokens,
                          num_max_nvl_chunked_send_tokens));
  EP_HOST_ASSERT(type == CUDA_R_16BF);

  SETUP_LAUNCH_CONFIG(num_channels * 2,
                      (NUM_MAX_NVL_PEERS + num_forwarder_warps + 1) * 32,
                      stream);
  SWITCH_RDMA_RANKS(COMBINE_LAUNCH_CASE);
#undef COMBINE_LAUNCH_CASE
}

}  // namespace internode

}  // namespace deep_ep
