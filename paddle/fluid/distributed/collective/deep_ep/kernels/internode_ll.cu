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

#include "paddle/fluid/distributed/collective/deep_ep/kernels/configs.cuh"
#include "paddle/fluid/distributed/collective/deep_ep/kernels/exception.cuh"
#include "paddle/fluid/distributed/collective/deep_ep/kernels/ibgda_device.cuh"
#include "paddle/fluid/distributed/collective/deep_ep/kernels/launch.cuh"
#include "paddle/phi/kernels/funcs/aligned_vector.h"

namespace deep_ep {

namespace internode_ll {

__global__ void barrier_all() { nvshmemx_barrier_all_block(); }

void barrier_all(cudaStream_t stream) {
  constexpr int kNumThreads = 1;

  SETUP_LAUNCH_CONFIG(1, kNumThreads, stream);
  LAUNCH_KERNEL(&cfg, barrier_all);
}

template <int kNumThreads>
__launch_bounds__(kNumThreads, 1) __global__ void clean_low_latency_buffer(
    int* clean_0, int num_clean_int_0, int* clean_1, int num_clean_int_1) {
  // Barrier before cleaning (in case of unfinished chunked EP)
  nvshmemx_barrier_all_block();

  // Clean
  auto thread_id = static_cast<int>(threadIdx.x);
#pragma unroll
  for (int i = thread_id; i < num_clean_int_0; i += kNumThreads) clean_0[i] = 0;
#pragma unroll
  for (int i = thread_id; i < num_clean_int_1; i += kNumThreads) clean_1[i] = 0;

  // Barrier after cleaning (make sure low-latency mode work fine)
  nvshmemx_barrier_all_block();
}

void clean_low_latency_buffer(int* clean_0,
                              int num_clean_int_0,
                              int* clean_1,
                              int num_clean_int_1,
                              cudaStream_t stream) {
  constexpr int kNumThreads = 256;

  SETUP_LAUNCH_CONFIG(1, kNumThreads, stream);
  LAUNCH_KERNEL(&cfg,
                clean_low_latency_buffer<kNumThreads>,
                clean_0,
                num_clean_int_0,
                clean_1,
                num_clean_int_1);
}

template <bool kUseFP8, int kNumWarpGroups, int kNumWarpsPerGroup, int kHidden>
__global__ __launch_bounds__(
    kNumWarpGroups* kNumWarpsPerGroup * 32,
    1) void dispatch(void* packed_recv_x,
                     float* packed_recv_x_scales,
                     int* packed_recv_src_info,
                     int64_t* packed_recv_layout_range,
                     int* packed_recv_count,
                     void* rdma_recv_x,
                     int* rdma_recv_count,
                     void* rdma_x,
                     const void* x,
                     const int64_t* topk_idx,
                     const float* expertwise_scale,
                     int* atomic_counter_per_expert,
                     int* atomic_finish_counter_per_expert,
                     int* next_clean,
                     int num_next_clean_int,
                     int num_tokens,
                     int num_max_dispatch_tokens_per_rank,
                     int num_topk,
                     int num_experts,
                     int rank,
                     int num_ranks,
                     int phases) {
  const auto sm_id = static_cast<int>(blockIdx.x);
  const auto thread_id = static_cast<int>(threadIdx.x);
  const auto warp_id = thread_id / 32, lane_id = get_lane_id();
  const auto num_sms = static_cast<int>(gridDim.x);
  const auto num_warps = kNumWarpGroups * kNumWarpsPerGroup;
  const auto num_local_experts = num_experts / num_ranks;
  const auto warp_group_id = warp_id / kNumWarpsPerGroup;
  const auto sub_warp_id = warp_id % kNumWarpsPerGroup;
  const auto responsible_expert_idx = sm_id * kNumWarpGroups + warp_group_id;

  const bool use_expertwise_scale = expertwise_scale;

  // FP8 staffs
  constexpr int kNumPerChannels = 128;
  constexpr float kFP8Margin = 1e-4, kFP8Amax = 448,
                  kFP8AmaxInv = 1.0f / 448.0f;
  const int num_scales = kHidden / kNumPerChannels;
  size_t hidden_bytes =
      kHidden * (kUseFP8 ? sizeof(__nv_fp8_storage_t) : sizeof(nv_bfloat16));

  if (use_expertwise_scale) hidden_bytes = kHidden;

  const size_t hidden_int4 = hidden_bytes / sizeof(int4);

  // Message package: hidden data, FP8 scales, index at source
  // NOTES: currently we have 3 reserved int fields for future use
  using vec_t = typename std::conditional<kUseFP8, int2, int4>::type;
  size_t num_bytes_per_msg =
      sizeof(int4) + (kUseFP8 ? (kHidden + num_scales * sizeof(float))
                              : (kHidden * sizeof(nv_bfloat16)));
  if (use_expertwise_scale) num_bytes_per_msg = sizeof(int4) + kHidden;

  const size_t num_int4_per_msg = num_bytes_per_msg / sizeof(int4);
  EP_DEVICE_ASSERT(num_bytes_per_msg % sizeof(int4) == 0);

  // Sending phase
  if ((phases & LOW_LATENCY_SEND_PHASE) == 0) goto LOW_LATENCY_DISPATCH_RECV;

  // Expert counts
  __shared__ int shared_num_tokens_sent_per_expert[kNumWarpGroups];

  // There are 2 kinds of warps in this part:
  // 1. The first-kind warps for FP8 cast and sending top-k tokens
  // 2. The last warp for reading `topk_idx` and count for per-expert
  // information
  if (warp_id < num_warps - 1) {
    constexpr int kNumElemsPerRead = sizeof(int4) / sizeof(nv_bfloat16);
    EP_DEVICE_ASSERT(kHidden % kNumElemsPerRead == 0);
    EP_STATIC_ASSERT(kNumElemsPerRead * 32 % kNumPerChannels == 0,
                     "Invalid vectorization");
    const auto num_threads = (num_warps - 1) * 32;
    const size_t hidden_bf16_int4 = kHidden / kNumElemsPerRead;

    for (int token_idx = sm_id; token_idx < num_tokens; token_idx += num_sms) {
      const auto x_int4 =
          reinterpret_cast<const int4*>(x) + token_idx * hidden_bf16_int4;

      auto rdma_x_src_idx = reinterpret_cast<int*>(
          reinterpret_cast<uint8_t*>(rdma_x) + token_idx * num_bytes_per_msg);

      if (use_expertwise_scale) {
        // Each token needs to be quantified into different int8 data based on
        // the experts it is dispatched to
        rdma_x_src_idx = reinterpret_cast<int*>(
            reinterpret_cast<uint8_t*>(rdma_x) +
            (warp_id + token_idx * num_topk) * num_bytes_per_msg);
      }
      const auto rdma_x_vec = reinterpret_cast<vec_t*>(
          reinterpret_cast<uint8_t*>(rdma_x_src_idx) + sizeof(int4));
      const auto rdma_x_scales = reinterpret_cast<float*>(
          reinterpret_cast<uint8_t*>(rdma_x_vec) + hidden_bytes);

      // Overlap top-k index read and source token index write
      auto dst_expert_idx =
          warp_id < num_topk ? static_cast<int>(__ldg(
                                   topk_idx + token_idx * num_topk + warp_id))
                             : -1;
      thread_id == 0 ? (*rdma_x_src_idx = token_idx) : 0;

      if (use_expertwise_scale) {
        if (warp_id < num_topk) {
          lane_id == 0 ? (*rdma_x_src_idx = token_idx) : 0;
        }
      }
      // Note(zkk)
      // create a run_deepep_loop, so I need not modify Deepep's code any more.
      int run_deepep_loop = 1;
      if (use_expertwise_scale) {
        run_deepep_loop = 0;
        for (int ii = 0; ii < num_topk; ii++) {
          int tmp_id = topk_idx[ii + token_idx * num_topk];
          float scale = expertwise_scale[tmp_id];

          for (int i = thread_id; i < hidden_bf16_int4; i += num_threads) {
            auto int4_value = __ldg(x_int4 + i);
            auto bf16_values = reinterpret_cast<nv_bfloat16*>(&int4_value);
            int2 int2_value;
            phi::AlignedVector<int8_t, 8> res_vec;
            const float max_bound = 127.f;
            const float min_bound = -127.f;
            for (int j = 0; j < 8; j++) {
              float quant_value =
                  max_bound * scale * static_cast<float>(bf16_values[j]);
              quant_value = quant_value > max_bound ? max_bound : quant_value;
              quant_value = quant_value < min_bound ? min_bound : quant_value;
              res_vec[j] = static_cast<int8_t>(round(quant_value));
            }
            phi::Store(res_vec,
                       reinterpret_cast<int8_t*>(rdma_x) +
                           (ii + token_idx * num_topk) * num_bytes_per_msg +
                           sizeof(int4) + i * sizeof(res_vec));
          }
        }
      }

// FP8 cast
#pragma unroll
      for (int i = thread_id; i < hidden_bf16_int4 * run_deepep_loop;
           i += num_threads) {
        // Read
        auto int4_value = __ldg(x_int4 + i);

        if (kUseFP8) {
          // Calculate local amax
          auto bf16_values = reinterpret_cast<nv_bfloat16*>(&int4_value);
          float fp32_values[kNumElemsPerRead];
          float amax = kFP8Margin, scale, scale_inv;
#pragma unroll
          for (int j = 0; j < kNumElemsPerRead; ++j) {
            fp32_values[j] = static_cast<float>(bf16_values[j]);
            amax = fmaxf(amax, fabsf(fp32_values[j]));
          }

          // Reduce amax and scale
          EP_STATIC_ASSERT(kNumElemsPerRead * 32 / kNumPerChannels == 2,
                           "Invalid vectorization");
          amax = half_warp_reduce_max(amax), scale = kFP8Amax / amax,
          scale_inv = amax * kFP8AmaxInv;
          if (lane_id == 0 || lane_id == 16)
            rdma_x_scales[i * kNumElemsPerRead / 128] = scale_inv;

          // Cast into send buffer
          vec_t int2_value;
          auto fp8x2_values =
              reinterpret_cast<__nv_fp8x2_storage_t*>(&int2_value);
#pragma unroll
          for (int j = 0; j < kNumElemsPerRead; j += 2) {
            float2 fp32x2 = {fp32_values[j] * scale,
                             fp32_values[j + 1] * scale};
            fp8x2_values[j / 2] =
                __nv_cvt_float2_to_fp8x2(fp32x2, __NV_SATFINITE, __NV_E4M3);
          }
          rdma_x_vec[i] = int2_value;
        } else {
          // Reinterpret-cast is for C++14 compatibility
          rdma_x_vec[i] = *reinterpret_cast<vec_t*>(&int4_value);
        }
      }
      asm volatile("bar.sync 1, %0;" ::"r"(num_threads));

      // Issue IBGDA sends
      if (dst_expert_idx >= 0) {
        int slot_idx =
            lane_id == 0
                ? atomicAdd(atomic_counter_per_expert + dst_expert_idx, 1)
                : 0;
        slot_idx = __shfl_sync(0xffffffff, slot_idx, 0);
        const auto dst_rank = dst_expert_idx / num_local_experts;
        const auto dst_expert_local_idx = dst_expert_idx % num_local_experts;
        const auto src_ptr = reinterpret_cast<uint64_t>(rdma_x_src_idx);
        const auto dst_ptr =
            reinterpret_cast<uint64_t>(rdma_recv_x) +
            dst_expert_local_idx * num_ranks *
                num_max_dispatch_tokens_per_rank * num_bytes_per_msg +
            rank * num_max_dispatch_tokens_per_rank * num_bytes_per_msg +
            slot_idx * num_bytes_per_msg;
        if (dst_rank != rank) {
          void* peer_base_addr = reinterpret_cast<void*>(
              __ldg(reinterpret_cast<const uint64_t*>(
                        nvshmemi_device_state_d.peer_heap_base_p2p) +
                    dst_rank));
          if (peer_base_addr) {
            char* req_rptr_actual =
                reinterpret_cast<char*>(peer_base_addr) +
                (reinterpret_cast<char*>(dst_ptr) -
                 reinterpret_cast<char*>(nvshmemi_device_state_d.heap_base));
            const auto* src_int4_ptr = reinterpret_cast<const int4*>(src_ptr);
            const auto* dst_int4_ptr = reinterpret_cast<int4*>(req_rptr_actual);
            UNROLLED_WARP_COPY(8,
                               lane_id,
                               num_int4_per_msg,
                               dst_int4_ptr,
                               src_int4_ptr,
                               ld_nc_global,
                               st_na_global);
          } else {
            nvshmemi_ibgda_put_nbi_warp(dst_ptr,
                                        src_ptr,
                                        num_bytes_per_msg,
                                        dst_rank,
                                        dst_expert_local_idx,
                                        lane_id,
                                        slot_idx);
          }
        } else {
          // NOTES: only 2 load iterations for 7K hidden with 8 unrolls
          const auto* src_int4_ptr = reinterpret_cast<const int4*>(src_ptr);
          const auto* dst_int4_ptr = reinterpret_cast<int4*>(dst_ptr);
          UNROLLED_WARP_COPY(8,
                             lane_id,
                             num_int4_per_msg,
                             dst_int4_ptr,
                             src_int4_ptr,
                             ld_nc_global,
                             st_na_global);
        }

        // Increase counter after finishing
        __syncwarp();
        lane_id == 0 ? atomic_add_release_global(
                           atomic_finish_counter_per_expert + dst_expert_idx, 1)
                     : 0;
      }
    }
  } else if (warp_id == num_warps - 1) {
    EP_DEVICE_ASSERT(num_sms > 1);
    if (sm_id == 0) {
      // The first SM is also responsible for checking QPs
      EP_DEVICE_ASSERT(ibgda_get_state()->num_rc_per_pe == num_local_experts);

// The first SM is also responsible for cleaning the next buffer
#pragma unroll
      for (int i = lane_id; i < num_next_clean_int; i += 32) next_clean[i] = 0;

      // Notify before executing `int_p`
      __syncwarp();
#pragma unroll
      for (int i = lane_id; i < num_experts; i += 32)
        atomic_add_release_global(atomic_finish_counter_per_expert + i,
                                  FINISHED_SUM_TAG);
    }

    // This SM should be responsible for some destination experts, read
    // `topk_idx` for them
    int expert_count[kNumWarpGroups] = {0};
    const auto expert_begin_idx = sm_id * kNumWarpGroups;
    const auto expert_end_idx =
        min(expert_begin_idx + kNumWarpGroups, num_experts);

// Per lane count
#pragma unroll 8
    for (int i = lane_id; i < num_tokens * num_topk; i += 32) {
      auto idx = static_cast<int>(__ldg(topk_idx + i));
      if (idx >= expert_begin_idx && idx < expert_end_idx)
        expert_count[idx - expert_begin_idx]++;
    }

// Warp reduce
#pragma unroll
    for (int i = expert_begin_idx; i < expert_end_idx; ++i) {
      auto sum = warp_reduce_sum(expert_count[i - expert_begin_idx]);
      if (lane_id == 0) {
        shared_num_tokens_sent_per_expert[i - expert_begin_idx] = sum;
        atomic_add_release_global(atomic_finish_counter_per_expert + i,
                                  FINISHED_SUM_TAG - sum);
      }
    }
  }
  __syncthreads();

  // Issue count sends
  if (responsible_expert_idx < num_experts && sub_warp_id == 0 &&
      lane_id == 0) {
    const auto dst_rank = responsible_expert_idx / num_local_experts;
    const auto dst_expert_local_idx =
        responsible_expert_idx % num_local_experts;
    const auto num_tokens_sent =
        shared_num_tokens_sent_per_expert[responsible_expert_idx -
                                          sm_id * kNumWarpGroups];

    // Wait local sends issued and send expert counts
    while (ld_acquire_global(atomic_finish_counter_per_expert +
                             responsible_expert_idx) != FINISHED_SUM_TAG * 2) {
    }
    if (dst_rank != rank) {
      void* peer_base_addr = reinterpret_cast<void*>(
          __ldg(reinterpret_cast<const uint64_t*>(
                    nvshmemi_device_state_d.peer_heap_base_p2p) +
                dst_rank));
      if (peer_base_addr) {  // P2P enabled
        int* rptr_actual = reinterpret_cast<int*>(
            reinterpret_cast<char*>(peer_base_addr) +
            (reinterpret_cast<char*>(rdma_recv_count +
                                     dst_expert_local_idx * num_ranks + rank) -
             reinterpret_cast<char*>(nvshmemi_device_state_d.heap_base)));
        st_release_sys_global(rptr_actual, -num_tokens_sent - 1);
      } else {
        nvshmemi_ibgda_amo_nonfetch_add(
            rdma_recv_count + dst_expert_local_idx * num_ranks + rank,
            -num_tokens_sent - 1,
            dst_rank,
            dst_expert_local_idx);
      }
    } else {
      st_na_release(rdma_recv_count + dst_expert_local_idx * num_ranks + rank,
                    -num_tokens_sent - 1);
    }

    // Clean workspace for next use
    atomic_counter_per_expert[responsible_expert_idx] = 0;
    atomic_finish_counter_per_expert[responsible_expert_idx] = 0;

    // Clean `packed_recv_count`
    if (dst_rank == 0) packed_recv_count[dst_expert_local_idx] = 0;
  }
  __syncwarp();

// Receiving phase
LOW_LATENCY_DISPATCH_RECV:
  if ((phases & LOW_LATENCY_RECV_PHASE) == 0) return;

  // For send-and-recv kernels, we need a grid sync for making
  // `packed_recv_count` visible
  if (phases & LOW_LATENCY_SEND_PHASE) cg::this_grid().sync();

  // Receiving and packing
  if (responsible_expert_idx < num_experts) {
    const auto src_rank = responsible_expert_idx / num_local_experts;
    const auto local_expert_idx = responsible_expert_idx % num_local_experts;
    const auto rdma_recv_x_uint8 =
        reinterpret_cast<uint8_t*>(rdma_recv_x) +
        local_expert_idx * num_ranks * num_max_dispatch_tokens_per_rank *
            num_bytes_per_msg +
        src_rank * num_max_dispatch_tokens_per_rank * num_bytes_per_msg;
    const auto recv_x_int4 = reinterpret_cast<int4*>(packed_recv_x) +
                             local_expert_idx * num_ranks *
                                 num_max_dispatch_tokens_per_rank * hidden_int4;
    const auto recv_x_scales =
        packed_recv_x_scales + local_expert_idx * num_ranks *
                                   num_max_dispatch_tokens_per_rank *
                                   num_scales;
    const auto recv_src_info =
        packed_recv_src_info +
        local_expert_idx * num_ranks * num_max_dispatch_tokens_per_rank;
    const auto recv_range =
        packed_recv_layout_range + local_expert_idx * num_ranks;

    // Shared between sub-warps in warp groups
    __shared__ int shared_num_recv_tokens[kNumWarpGroups],
        shared_recv_token_begin_idx[kNumWarpGroups];

    // Wait tokens to arrive
    // NOTES: using sub-warp 1 to overlap with sub-warp 0
    int num_recv_tokens, recv_token_begin_idx;
    EP_STATIC_ASSERT(kNumWarpsPerGroup > 1,
                     "Requires more than one warp per group");
    if (sub_warp_id == 1 && lane_id == 0) {
      while ((num_recv_tokens = ld_acquire_global(
                  rdma_recv_count + local_expert_idx * num_ranks + src_rank)) ==
             0) {
      }
      num_recv_tokens = -num_recv_tokens - 1;
      recv_token_begin_idx =
          atomicAdd(packed_recv_count + local_expert_idx, num_recv_tokens);
      shared_num_recv_tokens[warp_group_id] = num_recv_tokens;
      shared_recv_token_begin_idx[warp_group_id] = recv_token_begin_idx;
      recv_range[src_rank] =
          pack2<int, int64_t>(num_recv_tokens, recv_token_begin_idx);
    }
    asm volatile("bar.sync %0, %1;" ::"r"(warp_group_id + 2),
                 "r"(kNumWarpsPerGroup * 32));
    num_recv_tokens = shared_num_recv_tokens[warp_group_id];
    recv_token_begin_idx = shared_recv_token_begin_idx[warp_group_id];

    // Copy tokens
    EP_DEVICE_ASSERT(num_scales <= 64);
    for (int i = sub_warp_id; i < num_recv_tokens; i += kNumWarpsPerGroup) {
      // Copy source info
      const auto src_src_idx =
          reinterpret_cast<int*>(rdma_recv_x_uint8 + i * num_bytes_per_msg);

      if (lane_id == 0)
        recv_src_info[recv_token_begin_idx + i] = ld_nc_global(src_src_idx);
      __syncwarp();

      // Copy data
      // NOTES: only 2 load iterations for 7K hidden with 7 unrolls
      const auto src_data = reinterpret_cast<int4*>(
          reinterpret_cast<uint8_t*>(src_src_idx) + sizeof(int4));
      const auto dst_data =
          recv_x_int4 + (recv_token_begin_idx + i) * hidden_int4;
      UNROLLED_WARP_COPY(7,
                         lane_id,
                         hidden_int4,
                         dst_data,
                         src_data,
                         ld_nc_global,
                         st_na_global);

      // Copy scales
      if (kUseFP8) {
        const auto src_scales = reinterpret_cast<float*>(
            reinterpret_cast<uint8_t*>(src_data) + hidden_bytes);
        const auto dst_scales =
            reinterpret_cast<float*>(recv_x_scales + recv_token_begin_idx + i);
        const auto scale_stride = num_ranks * num_max_dispatch_tokens_per_rank;
        auto scale_0 =
            lane_id < num_scales ? ld_nc_global(src_scales + lane_id) : 0;
        auto scale_1 = (lane_id + 32) < num_scales
                           ? ld_nc_global(src_scales + lane_id + 32)
                           : 0;
        lane_id < num_scales ? dst_scales[lane_id * scale_stride] = scale_0
                             : 0.0f;
        (lane_id + 32) < num_scales
            ? dst_scales[(lane_id + 32) * scale_stride] = scale_1
            : 0.0f;
      }
    }
  }
}

void dispatch(void* packed_recv_x,
              float* packed_recv_x_scales,
              int* packed_recv_src_info,
              int64_t* packed_recv_layout_range,
              int* packed_recv_count,
              void* rdma_recv_x,
              int* rdma_recv_count,
              void* rdma_x,
              const void* x,
              const int64_t* topk_idx,
              const float* expertwise_scale,
              int* next_clean,
              int num_next_clean_int,
              int num_tokens,
              int hidden,
              int num_max_dispatch_tokens_per_rank,
              int num_topk,
              int num_experts,
              int rank,
              int num_ranks,
              bool use_fp8,
              void* workspace,
              cudaStream_t stream,
              int phases) {
  constexpr int kNumMaxTopK = 9;
  constexpr int NUM_WARPS = 32;
  const int dev_id = 0;
  int sm_count;
  cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, dev_id);
  const int num_warp_groups = cell_div(num_experts, sm_count);
  const auto num_sms = max(sm_count, cell_div(num_experts, num_warp_groups));
  EP_HOST_ASSERT(num_topk <= kNumMaxTopK);

  // Workspace checks
  auto atomic_counter_per_expert = reinterpret_cast<int*>(workspace);
  auto atomic_finish_counter_per_expert =
      atomic_counter_per_expert + num_experts;
  EP_HOST_ASSERT(num_experts * sizeof(int) * 2 <= NUM_WORKSPACE_BYTES);

  DISPATCH_HIDDEN_SIZE(
      hidden,
      kHidden,
      {DISPATCH_NUM_WARP_GROUPS(num_warp_groups, kNumWarpGroups, {
        constexpr int kNumWarpsPerGroup = NUM_WARPS / kNumWarpGroups;
        EP_STATIC_ASSERT(kNumMaxTopK + 1 <= kNumWarpGroups * kNumWarpsPerGroup,
                         "Too many top-k selections");
        auto dispatch_func =
            use_fp8
                ? dispatch<true, kNumWarpGroups, kNumWarpsPerGroup, kHidden>
                : dispatch<false, kNumWarpGroups, kNumWarpsPerGroup, kHidden>;
        SETUP_LAUNCH_CONFIG(num_sms, NUM_WARPS * 32, stream);
        LAUNCH_KERNEL(&cfg,
                      dispatch_func,
                      packed_recv_x,
                      packed_recv_x_scales,
                      packed_recv_src_info,
                      packed_recv_layout_range,
                      packed_recv_count,
                      rdma_recv_x,
                      rdma_recv_count,
                      rdma_x,
                      x,
                      topk_idx,
                      expertwise_scale,
                      atomic_counter_per_expert,
                      atomic_finish_counter_per_expert,
                      next_clean,
                      num_next_clean_int,
                      num_tokens,
                      num_max_dispatch_tokens_per_rank,
                      num_topk,
                      num_experts,
                      rank,
                      num_ranks,
                      phases);
      })})
}

template <int kNumWarpGroups,
          int kNumWarpsPerGroup,
          int kHidden,
          int kNumMaxTopk>
__global__ __launch_bounds__(
    kNumWarpGroups* kNumWarpsPerGroup * 32,
    1) void combine(void* combined_x,
                    void* rdma_recv_x,
                    int* rdma_recv_flag,
                    void* rdma_send_x,
                    const void* x,
                    const int64_t* topk_idx,
                    const float* topk_weights,
                    const int* src_info,
                    const int64_t* layout_range,
                    int* next_clean,
                    int num_next_clean_int,
                    int* atomic_clean_flag,
                    int num_combined_tokens,
                    int hidden,
                    int num_topk,
                    int num_max_dispatch_tokens_per_rank,
                    int num_experts,
                    int rank,
                    int num_ranks,
                    int phases,
                    bool zero_copy) {
  const auto sm_id = static_cast<int>(blockIdx.x);
  const auto num_sms = static_cast<int>(gridDim.x);
  const auto thread_id = static_cast<int>(threadIdx.x);
  const auto num_threads = static_cast<int>(blockDim.x);
  const auto warp_id = thread_id / 32, lane_id = get_lane_id();
  const auto num_local_experts = num_experts / num_ranks;
  const auto warp_group_id = warp_id / kNumWarpsPerGroup;
  const auto sub_warp_id = warp_id % kNumWarpsPerGroup;
  const auto responsible_expert_idx = sm_id * kNumWarpGroups + warp_group_id;

  // Data type staffs
  constexpr int kNumElemsPerInt4 = sizeof(int4) / sizeof(nv_bfloat16);
  const size_t hidden_bf16_int4 = kHidden / kNumElemsPerInt4;

  // Message package
  // BF16 mode: always use BF16 for hidden data (ignoring the extra flag slot)
  constexpr size_t num_bytes_per_slot =
      sizeof(int4) + kHidden * sizeof(nv_bfloat16);
  EP_STATIC_ASSERT(num_bytes_per_slot % sizeof(int4) == 0,
                   "Invalid vectorization");

  // Sending phase
  if ((phases & LOW_LATENCY_SEND_PHASE) == 0) goto LOW_LATENCY_COMBINE_RECV;

  // Clean up next buffer
  if (sm_id == 0 && warp_group_id == 0 && sub_warp_id == 0) {
#pragma unroll
    for (int i = lane_id; i < num_next_clean_int; i += 32) next_clean[i] = 0;

    // Notify before executing `int_p`
    __syncwarp();
    if (lane_id == 0) atomic_add_release_global(atomic_clean_flag, num_experts);
  }

  // Issue IBGDA sends
  if (responsible_expert_idx < num_experts) {
    const auto dst_rank = responsible_expert_idx / num_local_experts;
    const auto local_expert_idx = responsible_expert_idx % num_local_experts;
    const auto global_expert_idx = rank * num_local_experts + local_expert_idx;
    const auto layout =
        __ldg(layout_range + local_expert_idx * num_ranks +
              dst_rank);  // num_recv_tokens, recv_token_begin_idx

    const auto local_x = reinterpret_cast<const int4*>(x) +
                         local_expert_idx * num_ranks *
                             num_max_dispatch_tokens_per_rank *
                             hidden_bf16_int4;
    const auto local_src_info = src_info + local_expert_idx * num_ranks *
                                               num_max_dispatch_tokens_per_rank;
    const auto rdma_send_x_vec = reinterpret_cast<uint8_t*>(rdma_send_x) +
                                 local_expert_idx * num_ranks *
                                     num_max_dispatch_tokens_per_rank *
                                     num_bytes_per_slot;

    // Unpack layout
    int offset, num_tokens_to_send;
    unpack2(layout, num_tokens_to_send, offset);

    // Issue IBGDA send
    for (int token_idx = offset + sub_warp_id;
         token_idx < offset + num_tokens_to_send;
         token_idx += kNumWarpsPerGroup) {
      const auto x_int4 = local_x + token_idx * hidden_bf16_int4;
      const auto rdma_send_type_row = reinterpret_cast<int*>(
          rdma_send_x_vec + token_idx * num_bytes_per_slot);
      const auto rdma_send_x_vec_row =
          reinterpret_cast<uint8_t*>(rdma_send_type_row);

      // Copy directly to local rank, or copy to buffer and issue RDMA
      auto src_idx = __ldg(local_src_info + token_idx);
      const auto buf_ptr = reinterpret_cast<int64_t>(rdma_send_x_vec_row);
      const auto dst_ptr =
          reinterpret_cast<uint64_t>(rdma_recv_x) +
          (global_expert_idx * num_max_dispatch_tokens_per_rank + src_idx) *
              num_bytes_per_slot;
      if (dst_rank == rank) {
        const auto dst_int4_ptr = reinterpret_cast<int4*>(dst_ptr);
        UNROLLED_WARP_COPY(7,
                           lane_id,
                           hidden_bf16_int4,
                           dst_int4_ptr,
                           x_int4,
                           ld_nc_global,
                           st_na_global);
      } else {
        const auto buf_int4_ptr = reinterpret_cast<int4*>(buf_ptr);
        if (!zero_copy)
          UNROLLED_WARP_COPY(7,
                             lane_id,
                             hidden_bf16_int4,
                             buf_int4_ptr,
                             x_int4,
                             ld_nc_global,
                             st_na_global);
        void* peer_base_addr = reinterpret_cast<void*>(
            __ldg(reinterpret_cast<const uint64_t*>(
                      nvshmemi_device_state_d.peer_heap_base_p2p) +
                  dst_rank));
        if (peer_base_addr) {
          char* req_rptr_actual =
              reinterpret_cast<char*>(peer_base_addr) +
              (reinterpret_cast<char*>(dst_ptr) -
               reinterpret_cast<char*>(nvshmemi_device_state_d.heap_base));
          const auto dst_int4_ptr = reinterpret_cast<int4*>(req_rptr_actual);
          UNROLLED_WARP_COPY(7,
                             lane_id,
                             hidden_bf16_int4,
                             dst_int4_ptr,
                             x_int4,
                             ld_nc_global,
                             st_na_global);
        } else {
          nvshmemi_ibgda_put_nbi_warp(dst_ptr,
                                      buf_ptr,
                                      hidden * sizeof(nv_bfloat16),
                                      dst_rank,
                                      local_expert_idx,
                                      lane_id,
                                      token_idx - offset);
        }
      }
    }

    // Put finishing flag
    EP_STATIC_ASSERT(kNumWarpsPerGroup > 1,
                     "Requires more than one warp per group");
    asm volatile("bar.sync %0, %1;" ::"r"(warp_group_id + 1),
                 "r"(kNumWarpsPerGroup * 32));
    if (sub_warp_id == 1 && lane_id == 0) {
      while (ld_acquire_global(atomic_clean_flag) == 0) {
      }
      if (dst_rank != rank) {
        void* peer_base_addr = reinterpret_cast<void*>(
            __ldg(reinterpret_cast<const uint64_t*>(
                      nvshmemi_device_state_d.peer_heap_base_p2p) +
                  dst_rank));
        if (peer_base_addr) {
          int* req_rptr_actual = reinterpret_cast<int*>(
              reinterpret_cast<char*>(peer_base_addr) +
              (reinterpret_cast<char*>(rdma_recv_flag + global_expert_idx) -
               reinterpret_cast<char*>(nvshmemi_device_state_d.heap_base)));
          st_release_sys_global(req_rptr_actual, 1);
        } else {
          nvshmemi_ibgda_amo_nonfetch_add(rdma_recv_flag + global_expert_idx,
                                          1,
                                          dst_rank,
                                          local_expert_idx);
        }
      } else {
        st_na_release(rdma_recv_flag + global_expert_idx, 1);
      }
      atomic_add_release_global(atomic_clean_flag, -1);
    }
    __syncwarp();
  }

// Receiving phase
LOW_LATENCY_COMBINE_RECV:
  if ((phases & LOW_LATENCY_RECV_PHASE) == 0) return;

  // Wait all ranks to arrive and notify PCIe usage
  if (responsible_expert_idx < num_experts) {
    EP_STATIC_ASSERT(kNumWarpsPerGroup > 1,
                     "Invalid number of warps per group");
    if (sub_warp_id == 0 && lane_id == 0)
      while (ld_acquire_global(rdma_recv_flag + responsible_expert_idx) == 0) {
      }
  }
  cg::this_grid().sync();

  // Reduce tokens with FP8 cast
  // EP_DEVICE_ASSERT(num_topk <= 32 && hidden_bf16_int4 <= num_threads);
  EP_STATIC_ASSERT(kHidden % (32 * kNumElemsPerInt4) == 0,
                   "Invalid vectorization");
  for (int g_id = thread_id; g_id < hidden_bf16_int4; g_id += num_threads) {
    for (int token_idx = sm_id; token_idx < num_combined_tokens;
         token_idx += num_sms) {
      // Read top-k indices and weights
      int reg_topk_idx[kNumMaxTopk];
      float reg_topk_weights[kNumMaxTopk];
#pragma unroll
      for (int i = 0; i < num_topk; ++i) {
        reg_topk_idx[i] =
            static_cast<int>(__ldg(topk_idx + token_idx * num_topk + i));
        reg_topk_weights[i] = __ldg(topk_weights + token_idx * num_topk + i);
      }

      float combined_values[kNumElemsPerInt4] = {0.0f};
#pragma unroll
      for (int i = 0; i < num_topk; ++i)
        if (reg_topk_idx[i] >= 0) {
          // Read from sources
          auto rdma_buffer_type = reinterpret_cast<const int*>(
              reinterpret_cast<uint8_t*>(rdma_recv_x) +
              (reg_topk_idx[i] * num_max_dispatch_tokens_per_rank + token_idx) *
                  num_bytes_per_slot);
          auto rdma_buffer_row =
              reinterpret_cast<const uint8_t*>(rdma_buffer_type);

          // Reduce
          auto x_vec = ld_nc_global(
              reinterpret_cast<const int4*>(rdma_buffer_row) + g_id);
          const auto x_bf16 = reinterpret_cast<nv_bfloat16*>(&x_vec);
#pragma unroll
          for (int j = 0; j < kNumElemsPerInt4; ++j)
            combined_values[j] +=
                static_cast<float>(x_bf16[j]) * reg_topk_weights[i];
        }

      // Write results
      int4& combined_int4 = *reinterpret_cast<int4*>(combined_values);
      auto combined_bf16 = reinterpret_cast<nv_bfloat16*>(&combined_values);
#pragma unroll
      for (int j = 0; j < kNumElemsPerInt4; ++j)
        combined_bf16[j] = static_cast<nv_bfloat16>(combined_values[j]);
      (reinterpret_cast<int4*>(combined_x) +
       token_idx * hidden_bf16_int4)[g_id] = combined_int4;
    }
  }
}

void combine(void* combined_x,
             void* rdma_recv_x,
             int* rdma_recv_flag,
             void* rdma_send_x,
             const void* x,
             const int64_t* topk_idx,
             const float* topk_weights,
             const int* src_info,
             const int64_t* layout_range,
             int* next_clean,
             int num_next_clean_int,
             int num_combined_tokens,
             int hidden,
             int num_max_dispatch_tokens_per_rank,
             int num_topk,
             int num_experts,
             int rank,
             int num_ranks,
             void* workspace,
             cudaStream_t stream,
             int phases,
             bool zero_copy) {
  constexpr int kNumMaxTopk = 9;
  constexpr int NUM_WARPS = 32;

  const int dev_id = 0;
  int sm_count;
  cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, dev_id);
  const int num_warp_groups = cell_div(num_experts, sm_count);
  const auto num_sms = max(sm_count, cell_div(num_experts, num_warp_groups));

  // Check workspace
  auto atomic_clean_flag = reinterpret_cast<int*>(workspace);
  EP_HOST_ASSERT(sizeof(int) <= NUM_WORKSPACE_BYTES);
  EP_HOST_ASSERT(num_topk <= kNumMaxTopk);

  DISPATCH_HIDDEN_SIZE(
      hidden,
      kHidden,
      {DISPATCH_NUM_WARP_GROUPS(num_warp_groups, kNumWarpGroups, {
        constexpr int kNumWarpsPerGroup = NUM_WARPS / kNumWarpGroups;
        auto combine_func =
            combine<kNumWarpGroups, kNumWarpsPerGroup, kHidden, kNumMaxTopk>;
        SETUP_LAUNCH_CONFIG(num_sms, NUM_WARPS * 32, stream);
        LAUNCH_KERNEL(&cfg,
                      combine_func,
                      combined_x,
                      rdma_recv_x,
                      rdma_recv_flag,
                      rdma_send_x,
                      x,
                      topk_idx,
                      topk_weights,
                      src_info,
                      layout_range,
                      next_clean,
                      num_next_clean_int,
                      atomic_clean_flag,
                      num_combined_tokens,
                      hidden,
                      num_topk,
                      num_max_dispatch_tokens_per_rank,
                      num_experts,
                      rank,
                      num_ranks,
                      phases,
                      zero_copy);
      })})
}

}  // namespace internode_ll

}  // namespace deep_ep
