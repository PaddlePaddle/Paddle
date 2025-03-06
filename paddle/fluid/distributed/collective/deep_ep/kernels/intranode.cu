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

#include <limits>

#include "paddle/fluid/distributed/collective/deep_ep/kernels/buffer.cuh"
#include "paddle/fluid/distributed/collective/deep_ep/kernels/configs.cuh"
#include "paddle/fluid/distributed/collective/deep_ep/kernels/exception.cuh"
#include "paddle/fluid/distributed/collective/deep_ep/kernels/launch.cuh"
#include "paddle/fluid/distributed/collective/deep_ep/kernels/utils.cuh"

namespace deep_ep {

namespace intranode {

template <int kNumRanks>
__global__ void notify_dispatch(const int* num_tokens_per_rank,
                                int* moe_recv_counter_mapped,
                                const int* num_tokens_per_expert,
                                int* moe_recv_expert_counter_mapped,
                                int num_experts,
                                int num_tokens,
                                int num_channels,
                                const bool* is_token_in_rank,
                                int* channel_prefix_matrix,
                                int* rank_prefix_matrix_copy,
                                int num_memset_int,
                                int expert_alignment,
                                void** buffer_ptrs,
                                int** task_fifo_ptrs,
                                int head,
                                int rank) {
  auto sm_id = static_cast<int>(blockIdx.x);
  auto thread_id = static_cast<int>(threadIdx.x),
       num_threads = static_cast<int>(blockDim.x);
  auto lane_id = thread_id % 32, warp_id = thread_id / 32,
       num_warps = num_threads / 32;

  if (sm_id == 0) {
    // Barrier first
    barrier_device<kNumRanks>(task_fifo_ptrs, head, rank);
    move_fifo_slots<kNumRanks>(head);
    __syncthreads();

    int *per_rank_buffer, *per_expert_buffer;
    if (thread_id < kNumRanks) {
      per_rank_buffer = reinterpret_cast<int*>(buffer_ptrs[thread_id]);
      per_expert_buffer = per_rank_buffer + kNumRanks * kNumRanks;
    }

    // After this loop:
    //  - `per_rank_buffer[rank][i, j]` means the number of tokens from rank i
    //  to rank j
    //  - `per_expert_buffer[rank][i, j]` means the number of tokens from rank i
    //  to local expert j
    int num_experts_per_rank = num_experts / kNumRanks;
    if (thread_id < kNumRanks) {
#pragma unroll
      for (int i = 0; i < kNumRanks; ++i)
        per_rank_buffer[rank * kNumRanks + i] = num_tokens_per_rank[i];
#pragma unroll
      for (int i = 0; i < num_experts_per_rank; ++i)
        per_expert_buffer[rank * num_experts_per_rank + i] =
            num_tokens_per_expert[thread_id * num_experts_per_rank + i];
    }
    __syncthreads();

    // Wait for all ranks to be finished
    barrier_device<kNumRanks>(task_fifo_ptrs, head, rank);
    move_fifo_slots<kNumRanks>(head);
    __syncthreads();

    // Sum per-rank counts and return to CPU
    // Also pre-compute the prefix sum for data sending
    auto local_per_rank_buffer = reinterpret_cast<int*>(buffer_ptrs[rank]);
    if (thread_id < kNumRanks) {
#pragma unroll
      for (int i = 1; i < kNumRanks; ++i)
        local_per_rank_buffer[i * kNumRanks + thread_id] +=
            local_per_rank_buffer[(i - 1) * kNumRanks + thread_id];
      if (thread_id == rank)
        *moe_recv_counter_mapped =
            local_per_rank_buffer[(kNumRanks - 1) * kNumRanks + rank];
    }

    // Sum per-experts counts and return to CPU
    auto local_per_expert_buffer =
        local_per_rank_buffer + kNumRanks * kNumRanks;
    if (thread_id < num_experts_per_rank) {
      int sum = 0;
#pragma unroll
      for (int i = 0; i < kNumRanks; ++i)
        sum += local_per_expert_buffer[i * num_experts_per_rank + thread_id];
      sum = (sum + expert_alignment - 1) / expert_alignment * expert_alignment;
      moe_recv_expert_counter_mapped[thread_id] = sum;
    }
    __syncthreads();

// Copy rank size prefix matrix to another tensor
#pragma unroll
    for (int i = thread_id; i < kNumRanks * kNumRanks; i += num_threads)
      rank_prefix_matrix_copy[i] = local_per_rank_buffer[i];

// Extra memset for later communication queue
#pragma unroll
    for (int i = thread_id; i < num_memset_int; i += num_threads)
      local_per_expert_buffer[i] = 0;

    // Barrier
    memory_fence();
    __syncthreads();
    barrier_device<kNumRanks>(task_fifo_ptrs, head, rank);
  } else {
    int dst_rank = sm_id - 1;
    for (int channel_id = warp_id; channel_id < num_channels;
         channel_id += num_warps) {
      int token_start_idx, token_end_idx;
      get_channel_task_range(
          num_tokens, num_channels, channel_id, token_start_idx, token_end_idx);

      // Iterate over tokens
      int count = 0;
      for (int64_t i = token_start_idx + lane_id; i < token_end_idx; i += 32)
        count += is_token_in_rank[i * kNumRanks + dst_rank];
      count = warp_reduce_sum(count);
      if (lane_id == 0)
        channel_prefix_matrix[dst_rank * num_channels + channel_id] = count;
    }
    __syncthreads();

    // Pre-compute prefix sum for all channels
    if (thread_id == 0) {
#pragma unroll
      for (int i = 1; i < num_channels; ++i)
        channel_prefix_matrix[dst_rank * num_channels + i] +=
            channel_prefix_matrix[dst_rank * num_channels + i - 1];
    }
  }
}

void notify_dispatch(const int* num_tokens_per_rank,
                     int* moe_recv_counter_mapped,
                     int num_ranks,
                     const int* num_tokens_per_expert,
                     int* moe_recv_expert_counter_mapped,
                     int num_experts,
                     int num_tokens,
                     const bool* is_token_in_rank,
                     int* channel_prefix_matrix,
                     int* rank_prefix_matrix_copy,
                     int num_memset_int,
                     int expert_alignment,
                     void** buffer_ptrs,
                     int** task_fifo_ptrs,
                     int head,
                     int rank,
                     cudaStream_t stream,
                     int num_channels) {
#define NOTIFY_DISPATCH_LAUNCH_CASE(ranks)      \
  LAUNCH_KERNEL(&cfg,                           \
                notify_dispatch<ranks>,         \
                num_tokens_per_rank,            \
                moe_recv_counter_mapped,        \
                num_tokens_per_expert,          \
                moe_recv_expert_counter_mapped, \
                num_experts,                    \
                num_tokens,                     \
                num_channels,                   \
                is_token_in_rank,               \
                channel_prefix_matrix,          \
                rank_prefix_matrix_copy,        \
                num_memset_int,                 \
                expert_alignment,               \
                buffer_ptrs,                    \
                task_fifo_ptrs,                 \
                head,                           \
                rank);                          \
  break

  constexpr int kNumThreads = 128;
  EP_HOST_ASSERT(num_experts % num_ranks == 0);
  EP_HOST_ASSERT(num_experts / num_ranks <= kNumThreads &&
                 num_ranks <= kNumThreads);

  SETUP_LAUNCH_CONFIG(1 + num_ranks, kNumThreads, stream);
  SWITCH_RANKS(NOTIFY_DISPATCH_LAUNCH_CASE);
#undef NOTIFY_DISPATCH_LAUNCH_CASE
}

template <int kNumRanks>
__global__ void cached_notify_dispatch(const int* rank_prefix_matrix,
                                       int num_memset_int,
                                       void** buffer_ptrs,
                                       int** task_fifo_ptrs,
                                       int head,
                                       int rank) {
  // A simplified version for cached handles
  barrier_device<kNumRanks>(task_fifo_ptrs, head, rank);
  move_fifo_slots<kNumRanks>(head);
  __syncthreads();

  // Copy and clean
  auto thread_id = static_cast<int>(threadIdx.x),
       num_threads = static_cast<int>(blockDim.x);
  auto ptr = reinterpret_cast<int*>(buffer_ptrs[rank]);
#pragma unroll
  for (int i = thread_id; i < kNumRanks * kNumRanks; i += num_threads)
    ptr[i] = rank_prefix_matrix[i];
#pragma unroll
  for (int i = thread_id; i < num_memset_int; i += num_threads)
    ptr[kNumRanks * kNumRanks + i] = 0;
  memory_fence();
  __syncthreads();

  // Barrier after cleaning
  barrier_device<kNumRanks>(task_fifo_ptrs, head, rank);
}

void cached_notify_dispatch(const int* rank_prefix_matrix,
                            int num_memset_int,
                            void** buffer_ptrs,
                            int** task_fifo_ptrs,
                            int head,
                            int rank,
                            int num_ranks,
                            cudaStream_t stream) {
#define CACHED_NOTIFY_DISPATCH_LAUNCH_CASE(ranks) \
  LAUNCH_KERNEL(&cfg,                             \
                cached_notify_dispatch<ranks>,    \
                rank_prefix_matrix,               \
                num_memset_int,                   \
                buffer_ptrs,                      \
                task_fifo_ptrs,                   \
                head,                             \
                rank);                            \
  break

  SETUP_LAUNCH_CONFIG(1, 128, stream);
  SWITCH_RANKS(CACHED_NOTIFY_DISPATCH_LAUNCH_CASE);
#undef CACHED_NOTIFY_DISPATCH_LAUNCH_CASE
}

template <int kNumRanks>
__global__ void __launch_bounds__(kNumRanks * 32, 1)
    dispatch(int4* recv_x,
             float* recv_x_scales,
             int* recv_src_idx,
             int64_t* recv_topk_idx,
             float* recv_topk_weights,
             int* recv_channel_offset,
             int* send_head,
             const int4* x,
             const float* x_scales,
             const int64_t* topk_idx,
             const float* topk_weights,
             const bool* is_token_in_rank,
             const int* channel_prefix_matrix,
             int num_tokens,
             int hidden_int4,
             int num_topk,
             int num_experts,
             int num_scales,
             void** buffer_ptrs,
             int rank,
             int num_max_send_tokens,
             int num_recv_buffer_tokens) {
  const auto num_sms = static_cast<int>(gridDim.x),
             sm_id = static_cast<int>(blockIdx.x);
  const auto thread_id = static_cast<int>(threadIdx.x);
  const bool is_sender = sm_id % 2 == 0;
  EP_DEVICE_ASSERT(num_sms % 2 == 0);

  // Each warp is responsible for a single rank
  const auto num_channels = num_sms / 2;
  const auto responsible_rank = (static_cast<int>(thread_id)) / 32;

  // Even-numbered blocks for sending, odd-numbered blocks for receiving
  const auto responsible_channel = sm_id / 2;

  int num_experts_per_rank = num_experts / kNumRanks;
  EP_DEVICE_ASSERT(num_experts_per_rank > 0 || num_topk == 0);
  EP_DEVICE_ASSERT(num_topk <= 32);
  EP_DEVICE_ASSERT((topk_idx == nullptr) == (topk_weights == nullptr));
  EP_DEVICE_ASSERT((recv_topk_idx == nullptr) ==
                   (recv_topk_weights == nullptr));

  // Calculate pointers by the specific layout
  // `rank_prefix_matrix`: kNumRanks * kNumRanks * sizeof(int)
  auto ptr = reinterpret_cast<void*>(
      reinterpret_cast<int8_t*>(
          buffer_ptrs[is_sender ? responsible_rank : rank]) +
      kNumRanks * kNumRanks * sizeof(int));
  int target_rank = is_sender ? rank : responsible_rank;
  auto num_channels_total = num_channels * kNumRanks;
  auto channel_rank_offset = responsible_channel * kNumRanks + target_rank;

  // Channel buffer metadata
  // Senders are responsible for tails, and receivers are responsible for heads
  // Stored on the receiver side
  // The retired signals are actually boolean flags, but to align with 16 bytes,
  // we make it `int64_t` `start_offset`: kNumChannels * kNumRanks * sizeof(int)
  // `end_offset`: kNumChannels * kNumRanks * sizeof(int)
  // `head_idx`: kNumChannels * kNumRanks * sizeof(int)
  // `tail_idx`: kNumChannels * kNumRanks * sizeof(int)
  auto channel_start_offset =
      Buffer<int>(ptr, num_channels_total, channel_rank_offset);
  auto channel_end_offset =
      Buffer<int>(ptr, num_channels_total, channel_rank_offset);
  auto channel_head_idx =
      Buffer<int>(ptr, num_channels_total, channel_rank_offset);
  auto channel_tail_idx =
      Buffer<int>(ptr, num_channels_total, channel_rank_offset);

  // Channel data buffers, stored on the receiver side
  // `x_buffers`: kNumChannels * kNumRanks * num_recv_buffer_tokens *
  // hidden_int4 * sizeof(int4) `src_idx_buffers`: kNumChannels * kNumRanks *
  // num_recv_buffer_tokens * sizeof(int) `topk_idx_buffers`: kNumChannels *
  // kNumRanks * num_recv_buffer_tokens * num_topk * sizeof(int64_t)
  // `topk_weights_buffers`: kNumChannels * kNumRanks * num_recv_buffer_tokens *
  // num_topk * sizeof(float) `x_scales_buffers`: kNumChannels * kNumRanks *
  // num_recv_buffer_tokens * num_scales * sizeof(float)
  auto channel_x_buffers =
      Buffer<int4>(ptr,
                   num_channels_total * num_recv_buffer_tokens * hidden_int4,
                   channel_rank_offset * num_recv_buffer_tokens * hidden_int4);
  auto channel_src_idx_buffers =
      Buffer<int>(ptr,
                  num_channels_total * num_recv_buffer_tokens,
                  channel_rank_offset * num_recv_buffer_tokens);
  auto channel_topk_idx_buffers =
      Buffer<int64_t>(ptr,
                      num_channels_total * num_recv_buffer_tokens * num_topk,
                      channel_rank_offset * num_recv_buffer_tokens * num_topk);
  auto channel_topk_weights_buffers =
      Buffer<float>(ptr,
                    num_channels_total * num_recv_buffer_tokens * num_topk,
                    channel_rank_offset * num_recv_buffer_tokens * num_topk);
  auto channel_x_scales_buffers =
      Buffer<float>(ptr,
                    num_channels_total * num_recv_buffer_tokens * num_scales,
                    channel_rank_offset * num_recv_buffer_tokens * num_scales);

  if (is_sender) {
    // Workers for sending
    constexpr int num_send_warps = kNumRanks;
    const auto send_thread_id = thread_id;
    const auto send_warp_id = send_thread_id / 32;
    const auto send_lane_id = send_thread_id % 32;
    EP_DEVICE_ASSERT(kNumRanks <= 32);
    EP_DEVICE_ASSERT(num_send_warps == kNumRanks &&
                     send_warp_id == responsible_rank);

    // Send offset by `-value - 1`, e.g. 0 -> -1, 1 -> -2
    // NOTES: this is for distinguishing zero tokens
    if (send_lane_id == 0) {
      int value = responsible_channel > 0
                      ? channel_prefix_matrix[send_warp_id * num_channels +
                                              responsible_channel - 1]
                      : 0;
      st_relaxed_sys_global(channel_start_offset.buffer(), -value - 1);
      value = channel_prefix_matrix[send_warp_id * num_channels +
                                    responsible_channel];
      st_relaxed_sys_global(channel_end_offset.buffer(), -value - 1);
    }
    __syncwarp();

    // Get tasks
    int token_start_idx, token_end_idx;
    get_channel_task_range(num_tokens,
                           num_channels,
                           responsible_channel,
                           token_start_idx,
                           token_end_idx);

    // Iterate over all tokens and send by chunks
    int cached_channel_tail_idx = 0;
    for (int64_t token_idx = token_start_idx; token_idx < token_end_idx;) {
      // Check destination queue emptiness, or wait a buffer to be released
      // (rare cases)
      auto start_time = clock64();
      while (send_lane_id == 0) {
        // NOTES: we only consider the worst case, because counting the real
        // numbers are time-consuming
        int num_used_slots = cached_channel_tail_idx -
                             ld_volatile_global(channel_head_idx.buffer());
        if (num_recv_buffer_tokens - num_used_slots >= num_max_send_tokens)
          break;

        // Rare cases to loop again
        if (clock64() - start_time > NUM_TIMEOUT_CYCLES) {
          printf(
              "DeepEP timeout for dispatch senders, rank %d, "
              "responsible_channel = %d\n",
              rank,
              responsible_channel);
          trap();
        }
      }
      __syncwarp();

      int chunk_token_idx = 0;
      while (chunk_token_idx < num_max_send_tokens &&
             token_idx < token_end_idx) {
        if (send_lane_id == 0)
          send_head[token_idx * kNumRanks + send_warp_id] =
              is_token_in_rank[token_idx * kNumRanks + send_warp_id]
                  ? cached_channel_tail_idx
                  : -1;
        // Skip if not selected
        if (!is_token_in_rank[token_idx * kNumRanks + send_warp_id]) {
          token_idx++;
          continue;
        }

        // Get an empty slot
        int dst_slot_idx = (cached_channel_tail_idx++) % num_recv_buffer_tokens;

        // Copy data
        auto shifted_channel_x_buffers =
            channel_x_buffers.buffer() + dst_slot_idx * hidden_int4;
        auto shifted_x = x + token_idx * hidden_int4;
        UNROLLED_WARP_COPY(5,
                           send_lane_id,
                           hidden_int4,
                           shifted_channel_x_buffers,
                           shifted_x,
                           __ldg,
                           st_na_global);

        // Copy source index
        if (send_lane_id == 0)
          channel_src_idx_buffers[dst_slot_idx] = static_cast<int>(token_idx);

        // Copy `topk_idx` and `topk_weights` with transformed index
        if (send_lane_id < num_topk) {
          // Top-k index
          int recv_expert_begin = send_warp_id * num_experts_per_rank,
              recv_expert_end = (send_warp_id + 1) * num_experts_per_rank;
          auto idx_value =
              __ldg(topk_idx + token_idx * num_topk + send_lane_id);
          idx_value =
              (idx_value >= recv_expert_begin && idx_value < recv_expert_end)
                  ? idx_value - recv_expert_begin
                  : -1;
          channel_topk_idx_buffers[dst_slot_idx * num_topk + send_lane_id] =
              idx_value;

          // Top-k weights
          auto weight_value =
              __ldg(topk_weights + token_idx * num_topk + send_lane_id);
          weight_value = (idx_value >= 0) ? weight_value : 0.0f;
          channel_topk_weights_buffers[dst_slot_idx * num_topk + send_lane_id] =
              weight_value;
        }

// Copy `x_scales`
#pragma unroll
        for (int i = send_lane_id; i < num_scales; i += 32)
          channel_x_scales_buffers[dst_slot_idx * num_scales + i] =
              __ldg(x_scales + token_idx * num_scales + i);

        // Move token index
        chunk_token_idx++, token_idx++;
      }

      // Move tail index
      __syncwarp();
      if (send_lane_id == 0)
        st_release_sys_global(channel_tail_idx.buffer(),
                              cached_channel_tail_idx);
    }
  } else {
    // Workers for receiving and copying into buffer
    constexpr int num_recv_warps = kNumRanks;
    const auto recv_thread_id = thread_id;
    const auto recv_warp_id = recv_thread_id / 32;
    const auto recv_lane_id = recv_thread_id % 32;
    EP_DEVICE_ASSERT(kNumRanks <= 32 && recv_warp_id == responsible_rank);
    EP_DEVICE_ASSERT(recv_thread_id >= 0 && num_recv_warps == kNumRanks);

    // Calculate offset first
    auto rank_prefix_matrix = reinterpret_cast<int*>(buffer_ptrs[rank]);
    int rank_offset =
        recv_warp_id > 0
            ? rank_prefix_matrix[(recv_warp_id - 1) * kNumRanks + rank]
            : 0;

    // Receive channel offset
    int total_offset, num_tokens_to_recv;
    while (recv_lane_id == 0 && (total_offset = ld_volatile_global(
                                     channel_start_offset.buffer())) == 0) {
    }
    while (recv_lane_id == 0 && (num_tokens_to_recv = ld_volatile_global(
                                     channel_end_offset.buffer())) == 0) {
    }
    if (recv_lane_id == 0) {
      total_offset = -total_offset - 1,
      num_tokens_to_recv = -num_tokens_to_recv - 1;
      recv_channel_offset[recv_warp_id * num_channels + responsible_channel] =
          total_offset;
      num_tokens_to_recv -= total_offset;
    }
    total_offset = __shfl_sync(0xffffffff, total_offset, 0);
    total_offset += rank_offset;
    num_tokens_to_recv = __shfl_sync(0xffffffff, num_tokens_to_recv, 0);

    auto start_time = clock64();
    int cached_channel_head_idx = 0, cached_channel_tail_idx = 0;
    while (num_tokens_to_recv > 0) {
      // Check channel status by lane 0
      while (recv_lane_id == 0) {
        cached_channel_tail_idx =
            ld_acquire_sys_global(channel_tail_idx.buffer());

        // Ready to copy
        if (cached_channel_head_idx != cached_channel_tail_idx) break;

        // Timeout check
        if (clock64() - start_time > NUM_TIMEOUT_CYCLES) {
          printf(
              "DeepEP timeout for dispatch receivers, rank %d, "
              "responsible_channel = %d, tokens remained: %d\n",
              rank,
              responsible_channel,
              num_tokens_to_recv);
          trap();
        }
      }

      // Sync queue tail
      cached_channel_tail_idx =
          __shfl_sync(0xffffffff, cached_channel_tail_idx, 0);

      // Copy data
      int num_recv_tokens = cached_channel_tail_idx - cached_channel_head_idx;
      for (int chunk_idx = 0; chunk_idx < num_recv_tokens; ++chunk_idx) {
        int token_idx_in_buffer =
            (cached_channel_head_idx + chunk_idx) % num_recv_buffer_tokens;
        auto shifted_buffer_x_int4 =
            channel_x_buffers.buffer() + token_idx_in_buffer * hidden_int4;
        auto shifted_recv_x_int4 =
            recv_x +
            static_cast<int64_t>(total_offset + chunk_idx) * hidden_int4;
        UNROLLED_WARP_COPY(5,
                           recv_lane_id,
                           hidden_int4,
                           shifted_recv_x_int4,
                           shifted_buffer_x_int4,
                           ld_nc_global,
                           st_na_global);
      }

// Copy `src_idx`
#pragma unroll 4
      for (int chunk_idx = cached_channel_head_idx + recv_lane_id;
           chunk_idx < cached_channel_tail_idx;
           chunk_idx += 32)
        recv_src_idx[total_offset + chunk_idx - cached_channel_head_idx] =
            ld_nc_global(channel_src_idx_buffers.buffer() +
                         chunk_idx % num_recv_buffer_tokens);

// Copy `topk_idx` and `topk_weights`
#pragma unroll 4
      for (int idx = recv_lane_id; idx < num_recv_tokens * num_topk;
           idx += 32) {
        int chunk_idx = idx / num_topk, token_topk_idx = idx % num_topk;
        int token_idx_in_buffer =
            (cached_channel_head_idx + chunk_idx) % num_recv_buffer_tokens;
        auto recv_idx =
            static_cast<int64_t>(total_offset + chunk_idx) * num_topk +
            token_topk_idx;
        auto buffer_idx = token_idx_in_buffer * num_topk + token_topk_idx;
        recv_topk_idx[recv_idx] =
            ld_nc_global(channel_topk_idx_buffers.buffer() + buffer_idx);
        recv_topk_weights[recv_idx] =
            ld_nc_global(channel_topk_weights_buffers.buffer() + buffer_idx);
      }

// Copy `x_scales`
#pragma unroll 4
      for (int i = recv_lane_id; i < num_recv_tokens * num_scales; i += 32) {
        int chunk_idx = i / num_scales, scales_idx = i % num_scales;
        int token_idx_in_buffer =
            (cached_channel_head_idx + chunk_idx) % num_recv_buffer_tokens;
        recv_x_scales[static_cast<int64_t>(total_offset + chunk_idx) *
                          num_scales +
                      scales_idx] =
            ld_nc_global(channel_x_scales_buffers.buffer() +
                         token_idx_in_buffer * num_scales + scales_idx);
      }

      // Move queue
      cached_channel_head_idx += num_recv_tokens;
      total_offset += num_recv_tokens;
      __syncwarp();
      if (recv_lane_id == 0)
        st_relaxed_sys_global(channel_head_idx.buffer(),
                              cached_channel_head_idx);

      // Exit
      num_tokens_to_recv -= num_recv_tokens;
    }
  }
}

void dispatch(void* recv_x,
              float* recv_x_scales,
              int* recv_src_idx,
              int64_t* recv_topk_idx,
              float* recv_topk_weights,
              int* recv_channel_offset,
              int* send_head,
              const void* x,
              const float* x_scales,
              const int64_t* topk_idx,
              const float* topk_weights,
              const bool* is_token_in_rank,
              const int* channel_prefix_matrix,
              int num_tokens,
              int hidden_int4,
              int num_topk,
              int num_experts,
              int num_scales,
              void** buffer_ptrs,
              int rank,
              int num_ranks,
              cudaStream_t stream,
              int num_sms,
              int num_max_send_tokens,
              int num_recv_buffer_tokens) {
#define DISPATCH_LAUNCH_CASE(ranks)               \
  LAUNCH_KERNEL(&cfg,                             \
                dispatch<ranks>,                  \
                reinterpret_cast<int4*>(recv_x),  \
                recv_x_scales,                    \
                recv_src_idx,                     \
                recv_topk_idx,                    \
                recv_topk_weights,                \
                recv_channel_offset,              \
                send_head,                        \
                reinterpret_cast<const int4*>(x), \
                x_scales,                         \
                topk_idx,                         \
                topk_weights,                     \
                is_token_in_rank,                 \
                channel_prefix_matrix,            \
                num_tokens,                       \
                hidden_int4,                      \
                num_topk,                         \
                num_experts,                      \
                num_scales,                       \
                buffer_ptrs,                      \
                rank,                             \
                num_max_send_tokens,              \
                num_recv_buffer_tokens);          \
  break

  // Even-numbered blocks for sending, odd-numbered blocks for receiving.
  EP_HOST_ASSERT(num_sms % 2 == 0);
  SETUP_LAUNCH_CONFIG(num_sms, num_ranks * 32, stream);
  SWITCH_RANKS(DISPATCH_LAUNCH_CASE);
#undef DISPATCH_LAUNCH_CASE
}

template <int kNumRanks>
__global__ void cached_notify_combine(void** buffer_ptrs,
                                      int* send_head,
                                      int num_channels,
                                      int num_recv_tokens,
                                      int num_memset_int,
                                      int** task_fifo_ptrs,
                                      int head,
                                      int rank) {
  const auto sm_id = static_cast<int>(blockIdx.x);
  if (sm_id == 0) {
    // Barrier before cleaning
    barrier_device<kNumRanks>(task_fifo_ptrs, head, rank);
    move_fifo_slots<kNumRanks>(head);
    __syncthreads();

    // Clean
    auto thread_id = static_cast<int>(threadIdx.x),
         num_threads = static_cast<int>(blockDim.x);
    auto ptr = reinterpret_cast<int*>(buffer_ptrs[rank]);
#pragma unroll
    for (int i = thread_id; i < num_memset_int; i += num_threads) ptr[i] = 0;
    memory_fence();
    __syncthreads();

    // Barrier after cleaning
    barrier_device<kNumRanks>(task_fifo_ptrs, head, rank);
  } else {
    const auto channel_id = sm_id - 1;
    const auto thread_id = static_cast<int>(threadIdx.x);
    const auto rank_id = thread_id / 32;
    const auto lane_id = thread_id % 32;

    int token_start_idx, token_end_idx;
    get_channel_task_range(num_recv_tokens,
                           num_channels,
                           channel_id,
                           token_start_idx,
                           token_end_idx);

    // NOTES: `1 << 25` is a heuristic large number
    int last_head = 1 << 25;
#pragma unroll
    for (int token_idx_tail = token_end_idx - 1;
         token_idx_tail >= token_start_idx;
         token_idx_tail -= 32) {
      int token_idx = token_idx_tail - lane_id, expected_head = 0;
      auto current_head =
          (token_idx >= token_start_idx)
              ? __ldg(send_head + token_idx * kNumRanks + rank_id)
              : -1;
      for (int i = 0; i < min(32, token_idx_tail - token_start_idx + 1); ++i) {
        head = __shfl_sync(0xffffffff, current_head, i);
        if (head < 0) {
          if (lane_id == i) expected_head = -last_head - 1;
        } else {
          last_head = head;
        }
      }
      if (current_head < 0 && token_idx >= token_start_idx)
        send_head[token_idx * kNumRanks + rank_id] = expected_head;
    }
  }
}

void cached_notify_combine(void** buffer_ptrs,
                           int* send_head,
                           int num_channels,
                           int num_recv_tokens,
                           int num_memset_int,
                           int** task_fifo_ptrs,
                           int head,
                           int rank,
                           int num_ranks,
                           cudaStream_t stream) {
#define CACHED_NOTIFY_COMBINE(ranks)          \
  LAUNCH_KERNEL(&cfg,                         \
                cached_notify_combine<ranks>, \
                buffer_ptrs,                  \
                send_head,                    \
                num_channels,                 \
                num_recv_tokens,              \
                num_memset_int,               \
                task_fifo_ptrs,               \
                head,                         \
                rank);                        \
  break

  const int num_threads = std::max(128, 32 * num_ranks);
  EP_HOST_ASSERT(num_ranks <= num_threads);
  EP_HOST_ASSERT(num_threads <= 1024);
  EP_HOST_ASSERT(1 + num_channels <= num_channels * 2);
  SETUP_LAUNCH_CONFIG(1 + num_channels, num_threads, stream);
  SWITCH_RANKS(CACHED_NOTIFY_COMBINE);
#undef CACHED_NOTIFY_COMBINE
}

template <typename dtype_t, int kNumRanks, int kNumThreads>
__global__ void __launch_bounds__(kNumThreads, 1)
    combine(dtype_t* recv_x,
            float* recv_topk_weights,
            const dtype_t* x,
            const float* topk_weights,
            const int* src_idx,
            const int* rank_prefix_matrix,
            const int* channel_prefix_matrix,
            int* send_head,
            int num_tokens,
            int num_recv_tokens,
            int hidden,
            int num_topk,
            void** buffer_ptrs,
            int rank,
            int num_max_send_tokens,
            int num_recv_buffer_tokens) {
  const auto num_sms = static_cast<int>(gridDim.x);
  const auto thread_id = static_cast<int>(threadIdx.x);
  const auto sm_id = static_cast<int>(blockIdx.x);
  const auto num_channels = num_sms / 2;
  const bool is_sender = sm_id % 2 == 0;
  const int responsible_channel = sm_id / 2;
  EP_DEVICE_ASSERT(num_topk <= 32);

  constexpr int kDtypePerInt4 = sizeof(int4) / sizeof(dtype_t);
  int hidden_int4 = hidden * sizeof(dtype_t) / sizeof(int4);
  auto x_int4 = reinterpret_cast<const int4*>(x);
  auto recv_int4 = reinterpret_cast<int4*>(recv_x);

  if (is_sender) {
    // Workers for sending
    // Several warps are responsible for a single rank
    constexpr int num_send_warps = kNumThreads / 32;
    constexpr int num_send_warps_per_rank = num_send_warps / kNumRanks;
    const auto num_threads_per_rank = num_send_warps_per_rank * 32;
    const auto send_thread_id = thread_id;
    const auto send_lane_id = send_thread_id % 32;
    const auto send_rank_id = thread_id / num_threads_per_rank;
    const auto send_warp_id_in_rank =
        send_thread_id % num_threads_per_rank / 32;

    // Calculate pointers by the specific layout
    auto ptr = reinterpret_cast<void*>(
        reinterpret_cast<int8_t*>(buffer_ptrs[send_rank_id]));
    auto num_channels_total = num_channels * kNumRanks;
    auto channel_rank_offset = responsible_channel * kNumRanks + rank;

    // Channel meta data
    // `head_idx`: kNumChannels * kNumRanks * sizeof(int)
    // `tail_idx`: kNumChannels * kNumRanks * sizeof(int)
    // `x_buffers`: kNumChannels * kNumRanks * num_recv_buffer_tokens *
    // hidden_int4 * sizeof(int4) `src_idx_buffers`: kNumChannels * kNumRanks *
    // num_recv_buffer_tokens * sizeof(int) `topk_weights_buffers`: kNumChannels
    // * kNumRanks * num_recv_buffer_tokens * num_topk * sizeof(float)
    auto channel_head_idx =
        Buffer<int>(ptr, num_channels_total, channel_rank_offset);
    auto channel_tail_idx =
        Buffer<int>(ptr, num_channels_total, channel_rank_offset);
    auto channel_x_buffers = Buffer<int4>(
        ptr,
        num_channels_total * num_recv_buffer_tokens * hidden_int4,
        channel_rank_offset * num_recv_buffer_tokens * hidden_int4);
    auto channel_src_idx_buffers =
        Buffer<int>(ptr,
                    num_channels_total * num_recv_buffer_tokens,
                    channel_rank_offset * num_recv_buffer_tokens);
    auto channel_topk_weights_buffers =
        Buffer<float>(ptr,
                      num_channels_total * num_recv_buffer_tokens * num_topk,
                      channel_rank_offset * num_recv_buffer_tokens * num_topk);

    // Get tasks
    // NOTES: `channel_offset` is already shifted
    int rank_offset =
        send_rank_id > 0
            ? rank_prefix_matrix[(send_rank_id - 1) * kNumRanks + rank]
            : 0;
    int num_rank_tokens =
        rank_prefix_matrix[send_rank_id * kNumRanks + rank] - rank_offset;
    int channel_offset = channel_prefix_matrix[send_rank_id * num_channels +
                                               responsible_channel];
    int num_channel_tokens =
        (responsible_channel == num_channels - 1
             ? num_rank_tokens
             : channel_prefix_matrix[send_rank_id * num_channels +
                                     responsible_channel + 1]) -
        channel_offset;
    int token_start_idx = rank_offset + channel_offset,
        token_end_idx = rank_offset + channel_offset + num_channel_tokens;

    // Iterate over all tokens and send by chunks
    int current_channel_tail_idx = 0;
    for (int64_t token_idx = token_start_idx; token_idx < token_end_idx;) {
      // Check destination queue emptiness, or wait a buffer to be released
      // (rare cases)
      auto start_time = clock64();
      int num_round_tokens =
          min(num_max_send_tokens, token_end_idx - static_cast<int>(token_idx));
      while (send_lane_id == 0) {
        // NOTES: we only consider the worst case, because counting the real
        // numbers are time-consuming
        int num_used_slots = current_channel_tail_idx -
                             ld_volatile_global(channel_head_idx.buffer());
        if (num_recv_buffer_tokens - num_used_slots >= num_round_tokens) break;

        // Rare cases to loop again
        if (clock64() - start_time > NUM_TIMEOUT_CYCLES) {
          printf(
              "DeepEP timeout for combine senders, rank %d, "
              "responsible_channel = %d\n",
              rank,
              responsible_channel);
          trap();
        }
      }
      __syncwarp();

// Send by chunk
#pragma unroll
      for (int i = send_warp_id_in_rank; i < num_round_tokens;
           i += num_send_warps_per_rank) {
        // Get an empty slot
        int dst_slot_idx =
            (current_channel_tail_idx + i) % num_recv_buffer_tokens;

        // Copy data
        auto shifted_x_buffers =
            channel_x_buffers.buffer() + dst_slot_idx * hidden_int4;
        auto shifted_x = x_int4 + (token_idx + i) * hidden_int4;
        UNROLLED_WARP_COPY(4,
                           send_lane_id,
                           hidden_int4,
                           shifted_x_buffers,
                           shifted_x,
                           ld_nc_global,
                           st_na_global);

        // Send source index
        if (send_lane_id == 0)
          channel_src_idx_buffers[dst_slot_idx] =
              __ldg(src_idx + token_idx + i);

        // Send `topk_weights`
        if (num_topk > 0 && send_lane_id < num_topk)
          channel_topk_weights_buffers[dst_slot_idx * num_topk + send_lane_id] =
              __ldg(topk_weights + (token_idx + i) * num_topk + send_lane_id);
      }
      token_idx += num_round_tokens;
      current_channel_tail_idx += num_round_tokens;

      // Move tail index
      asm volatile("bar.sync %0, %1;" ::"r"(send_rank_id),
                   "r"(num_threads_per_rank));
      if (send_lane_id == 0 && send_warp_id_in_rank == 0)
        st_release_sys_global(channel_tail_idx.buffer(),
                              current_channel_tail_idx);
    }
  } else {
    // Workers for receiving
    // One warp for moving the queue head, others for reduction
    constexpr int num_recv_warps = kNumThreads / 32;
    const auto recv_warp_id = thread_id / 32;
    const auto recv_lane_id = thread_id % 32;
    EP_DEVICE_ASSERT(kNumRanks <= 32 && kNumThreads > 32);
    EP_DEVICE_ASSERT(thread_id >= 0 && kNumThreads % 32 == 0);

    // Shared head, tail and retired flags for receiver warps
    __shared__ volatile int warp_channel_head_idx[num_recv_warps][kNumRanks];
    __shared__ volatile int channel_tail_idx[kNumRanks];
    __shared__ volatile bool warp_retired[num_recv_warps];
    if (thread_id < num_recv_warps) warp_retired[thread_id] = false;
    if (recv_lane_id < kNumRanks)
      warp_channel_head_idx[recv_warp_id][recv_lane_id] = 0;
    if (thread_id < kNumRanks) channel_tail_idx[thread_id] = 0;
    asm volatile("bar.sync 0, %0;" ::"r"(kNumThreads));

    if (thread_id < 32) {
      int* channel_head_idx_ptr = reinterpret_cast<int*>(buffer_ptrs[rank]) +
                                  responsible_channel * kNumRanks +
                                  recv_lane_id;
      int* channel_tail_idx_ptr =
          channel_head_idx_ptr + num_channels * kNumRanks;

      // Queue head updater
      int last_head = 0;
      while (recv_lane_id < kNumRanks) {
        // Check retired
        bool retired = true;
#pragma unroll
        for (int i = 1; i < num_recv_warps; ++i)
          retired = retired && warp_retired[i];
        if (retired) break;

        // Update queue tail
        channel_tail_idx[recv_lane_id] =
            ld_acquire_sys_global(channel_tail_idx_ptr);

        // Update minimum head
        int min_head = std::numeric_limits<int>::max();
#pragma unroll
        for (int i = 1; i < num_recv_warps; ++i)
          if (!warp_retired[i])
            min_head = min(min_head, warp_channel_head_idx[i][recv_lane_id]);
        if (min_head != std::numeric_limits<int>::max() && min_head > last_head)
          st_relaxed_sys_global(channel_head_idx_ptr, last_head = min_head);
      }
    } else {
      // Receivers
      // Channel metadata
      // All lanes will use data buffer, but only rank lane will use
      // `head/tail/src_idx`
      Buffer<int4> channel_x_buffers[kNumRanks];
      Buffer<float> channel_topk_weights_buffers[kNumRanks];

// Calculate pointers by the specific layout
#pragma unroll
      for (int i = 0; i < kNumRanks; ++i) {
        auto channel_rank_offset = responsible_channel * kNumRanks + i;
        auto num_channels_total = num_channels * kNumRanks;
        // `head_idx` & `tail_idx`: kNumChannels * kNumRanks * sizeof(int)
        auto ptr = reinterpret_cast<void*>(
            reinterpret_cast<int8_t*>(buffer_ptrs[rank]) +
            2 * num_channels * kNumRanks * sizeof(int));

        // `x_buffers`: kNumChannels * kNumRanks * num_recv_buffer_tokens *
        // hidden_int4 * sizeof(int4)
        channel_x_buffers[i] = Buffer<int4>(
            ptr,
            num_channels_total * num_recv_buffer_tokens * hidden_int4,
            channel_rank_offset * num_recv_buffer_tokens * hidden_int4);

        // `src_idx_buffers`: kNumChannels * kNumRanks * num_recv_buffer_tokens
        // * sizeof(int)
        ptr = reinterpret_cast<void*>(reinterpret_cast<int8_t*>(ptr) +
                                      num_channels_total *
                                          num_recv_buffer_tokens * sizeof(int));

        // `topk_weights_buffers`: kNumChannels * kNumRanks *
        // num_recv_buffer_tokens * num_topk * sizeof(float)
        channel_topk_weights_buffers[i] = Buffer<float>(
            ptr,
            num_channels_total * num_recv_buffer_tokens * num_topk,
            channel_rank_offset * num_recv_buffer_tokens * num_topk);
      }

      // The same tokens as the dispatch process
      int token_start_idx, token_end_idx;
      get_channel_task_range(num_recv_tokens,
                             num_channels,
                             responsible_channel,
                             token_start_idx,
                             token_end_idx);

      // Iterate over all tokens and combine
      for (int64_t token_idx = token_start_idx + recv_warp_id - 1;
           token_idx < token_end_idx;
           token_idx += num_recv_warps - 1) {
        // Read expected head
        int expected_head = -1;
        if (recv_lane_id < kNumRanks) {
          expected_head =
              ld_nc_global(send_head + token_idx * kNumRanks + recv_lane_id);
          warp_channel_head_idx[recv_warp_id][recv_lane_id] =
              (expected_head < 0) ? -expected_head - 1 : expected_head + 1;
        }
        auto start_time = clock64();
        while (channel_tail_idx[recv_lane_id] <= expected_head &&
               expected_head >= 0) {
          // Timeout check
          if (clock64() - start_time > NUM_TIMEOUT_CYCLES) {
            printf(
                "DeepEP timeout for combine receivers, rank %d, "
                "responsible_channel = %d, expect = %d\n",
                rank,
                responsible_channel,
                expected_head);
            trap();
          }
        }
        __syncwarp();

        // Broadcast current heads
        int num_topk_ranks = 0, topk_ranks[kNumRanks], slot_indices[kNumRanks];
#pragma unroll
        for (int i = 0; i < kNumRanks; ++i) {
          auto expected_head_i = __shfl_sync(0xffffffff, expected_head, i);
          if (expected_head_i >= 0) {
            slot_indices[num_topk_ranks] =
                expected_head_i % num_recv_buffer_tokens;
            topk_ranks[num_topk_ranks++] = i;
          }
        }

// Reduce data
#pragma unroll
        for (int i = recv_lane_id; i < hidden_int4; i += 32) {
          // Read buffers
          int4 recv_value_int4[kNumRanks];
#pragma unroll
          for (int j = 0; j < num_topk_ranks; ++j)
            recv_value_int4[j] =
                ld_nc_global(channel_x_buffers[topk_ranks[j]].buffer() +
                             slot_indices[j] * hidden_int4 + i);

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
          recv_int4[token_idx * hidden_int4 + i] = out_int4;
        }

        // Reduce `topk_weights`
        if (recv_lane_id < num_topk) {
          float value = 0;
#pragma unroll
          for (int i = 0; i < num_topk_ranks; ++i)
            value += ld_nc_global(
                channel_topk_weights_buffers[topk_ranks[i]].buffer() +
                slot_indices[i] * num_topk + recv_lane_id);
          recv_topk_weights[token_idx * num_topk + recv_lane_id] = value;
        }
      }

      // Retired
      __syncwarp();
      if (recv_lane_id == 0) warp_retired[recv_warp_id] = true;
    }
  }
}

void combine(cudaDataType_t type,
             void* recv_x,
             float* recv_topk_weights,
             const void* x,
             const float* topk_weights,
             const int* src_idx,
             const int* rank_prefix_matrix,
             const int* channel_prefix_matrix,
             int* send_head,
             int num_tokens,
             int num_recv_tokens,
             int hidden,
             int num_topk,
             void** buffer_ptrs,
             int rank,
             int num_ranks,
             cudaStream_t stream,
             int num_sms,
             int num_max_send_tokens,
             int num_recv_buffer_tokens) {
  constexpr int kNumThreads = 768;

#define COMBINE_LAUNCH_CASE(dtype, ranks)             \
  LAUNCH_KERNEL(&cfg,                                 \
                (combine<dtype, ranks, kNumThreads>), \
                reinterpret_cast<dtype*>(recv_x),     \
                recv_topk_weights,                    \
                reinterpret_cast<const dtype*>(x),    \
                topk_weights,                         \
                src_idx,                              \
                rank_prefix_matrix,                   \
                channel_prefix_matrix,                \
                send_head,                            \
                num_tokens,                           \
                num_recv_tokens,                      \
                hidden,                               \
                num_topk,                             \
                buffer_ptrs,                          \
                rank,                                 \
                num_max_send_tokens,                  \
                num_recv_buffer_tokens);              \
  break
#define COMBINE_DTYPE_LAUNCH_CASE(dtype)               \
  SWITCH_RANKS_WITH_DTYPE(dtype, COMBINE_LAUNCH_CASE); \
  break

  // Even-numbered blocks for sending, odd-numbered blocks for receiving
  EP_HOST_ASSERT(num_sms % 2 == 0);
  EP_HOST_ASSERT(kNumThreads >= num_ranks * 32);
  SETUP_LAUNCH_CONFIG(num_sms, kNumThreads, stream);
  SWITCH_TYPES(COMBINE_DTYPE_LAUNCH_CASE);
#undef COMBINE_DTYPE_LAUNCH_CASE
#undef COMBINE_LAUNCH_CASE
}

}  // namespace intranode

}  // namespace deep_ep
