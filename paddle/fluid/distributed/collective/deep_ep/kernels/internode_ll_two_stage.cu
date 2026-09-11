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

namespace deep_ep {

namespace internode_ll_two_stage {

constexpr size_t AlignUpElems(size_t n, size_t elems) {
  return (n + elems - 1) / elems * elems;
}

template <int kNumThreads>
__launch_bounds__(kNumThreads, 1) __global__
    void clean_low_latency_buffer_two_stage(void** buffer_ptrs_gpu,
                                            const size_t max_nvl_num_bytes,
                                            const size_t signal_bytes,
                                            const int nvl_rank,
                                            const int num_experts,
                                            int* clean_0,
                                            int num_clean_int_0,
                                            int* clean_1,
                                            int num_clean_int_1) {
  // Barrier before cleaning (in case of unfinished chunked EP)
  nvshmemx_barrier_all_block();

  auto thread_id = static_cast<int>(threadIdx.x);
  // Clean NVL Buffer
  int* buffer_ptrs_gpu_signal0 = reinterpret_cast<int*>(
      reinterpret_cast<uint8_t*>(buffer_ptrs_gpu[nvl_rank]) +
      max_nvl_num_bytes);
  int* buffer_ptrs_gpu_signal1 = reinterpret_cast<int*>(
      reinterpret_cast<uint8_t*>(buffer_ptrs_gpu[nvl_rank]) +
      (max_nvl_num_bytes * 2 + signal_bytes));
#pragma unroll
  for (int i = thread_id; i < num_experts; i += kNumThreads) {
    buffer_ptrs_gpu_signal0[i] = 0;
    buffer_ptrs_gpu_signal1[i] = 0;
  }

  // Clean RDMA Buffer
#pragma unroll
  for (int i = thread_id; i < num_clean_int_0; i += kNumThreads) clean_0[i] = 0;
#pragma unroll
  for (int i = thread_id; i < num_clean_int_1; i += kNumThreads) clean_1[i] = 0;

  // Barrier after cleaning (make sure low-latency mode work fine)
  nvshmemx_barrier_all_block();
}

void clean_low_latency_buffer_two_stage(void** buffer_ptrs_gpu,
                                        const size_t max_nvl_num_bytes,
                                        const size_t signal_bytes,
                                        const int nvl_rank,
                                        const int num_experts,
                                        int* clean_0,
                                        int num_clean_int_0,
                                        int* clean_1,
                                        int num_clean_int_1,
                                        cudaStream_t stream) {
  constexpr int kNumThreads = 512;

  SETUP_LAUNCH_CONFIG(1, kNumThreads, stream);
  LAUNCH_KERNEL(&cfg,
                clean_low_latency_buffer_two_stage<kNumThreads>,
                buffer_ptrs_gpu,
                max_nvl_num_bytes,
                signal_bytes,
                nvl_rank,
                num_experts,
                clean_0,
                num_clean_int_0,
                clean_1,
                num_clean_int_1);
}

template <bool kUseFP8,
          int kNumWarpGroups,
          int kNumWarpsPerGroup,
          int kHidden,
          int kNumRdmaRanks,
          int kNumExperts,
          int kTopk,
          int kRDMANumWarps,
          int kG2SNumWarps,
          int kNumPerChannels = 128>
__global__ __launch_bounds__(
    kNumWarpGroups* kNumWarpsPerGroup * 32,
    1) void dispatch_wp_kernel(void* packed_recv_x,
                              float* packed_recv_x_scales,
                              void* packed_rdma_recv_x,
                              int* packed_recv_src_info,
                              int64_t* packed_recv_layout_range,
                              int* packed_recv_count,
                              int* packed_rdma_recv_count,
                              bool* rdma_send_flags,  // kNumRdmaRanks
                              void* rdma_recv_x,
                              int* rdma_recv_count, // num_rdma_ranks * num_max_chunks
                              void* rdma_x,
                              void** nvl_recv_x,  // num_local_experts * dp_num *
                                                  // num_max_token_per_dp *
                                                  // hidden_size
                              const void* x,
                              const int64_t* topk_idx,
                              const float* topk_weights,
                              int* atomic_counter_per_expert,
                              int* atomic_counter_per_rdma,
                              int* atomic_finished_counter_per_rdma,
                              int* atomic_recv_tokens_per_rdma_expert,
                              int* atomic_nvl_sender_multi_sms,
                              int* atomic_nvl_sender_multi_sms_rdma,
                              int* next_clean,
                              int num_next_clean_int,  // Not used temporarily
                              int num_tokens,
                              int num_max_dispatch_tokens_per_rank,
                              int num_tokens_per_chunk,
                              int rank,
                              int phases,
                              int next_buffer_id) {
  constexpr int UNROLL_FACTOR = kHidden / 1024;
  constexpr int kNumRanks = kNumRdmaRanks * NUM_MAX_NVL_PEERS;
  constexpr int kNumLocalExperts = kNumExperts / kNumRanks;
  constexpr int kNumRdmaExperts = kNumLocalExperts * NUM_MAX_NVL_PEERS;
  constexpr int kRDMANumThreads = kRDMANumWarps * 32;
  constexpr int kG2SNumThreads = kG2SNumWarps * 32;
  const int nvl_buffer_id = next_buffer_id ^ 1;

  const int kNumMaxChunks = num_max_dispatch_tokens_per_rank;

  const auto rdma_rank = rank / NUM_MAX_NVL_PEERS,
             nvl_rank = rank % NUM_MAX_NVL_PEERS;

  const auto sm_id = static_cast<int>(blockIdx.x);
  const auto num_sms = static_cast<int>(gridDim.x);

  const int num_qps = num_sms * kNumRdmaRanks;
  const int qp_offset = sm_id * kNumRdmaRanks;

  // FP8 staffs
  constexpr float kFP8Margin = 1e-4, kFP8Amax = 448,
                  kFP8AmaxInv = 1.0f / 448.0f;
  constexpr int kNumScales =
      kNumPerChannels == -1 ? 1 : kHidden / kNumPerChannels;
  constexpr int kAlignElems = sizeof(int4) / sizeof(float);
  const size_t hidden_bytes =
      kHidden * (kUseFP8 ? sizeof(__nv_fp8_storage_t) : sizeof(nv_bfloat16));
  const size_t hidden_int4 = hidden_bytes / sizeof(int4);

  // index_source, hidden, (scale), nvl_num, nvl_rank0, dst_idx0, topk_weight0,
  // ..., nvl_rank7, dst_idx7, topk_weight7, ...
  using vec_t = typename std::conditional<kUseFP8, int2, int4>::type;
  const size_t num_bytes_per_msg =
      sizeof(int4) +
      (kNumRdmaRanks * (kTopk * 3 + 1) * sizeof(int) + sizeof(int4) - 1) /
          sizeof(int4) * sizeof(int4) +
      (kUseFP8
           ? (kHidden + AlignUpElems(kNumScales, kAlignElems) * sizeof(float))
           : (kHidden * sizeof(nv_bfloat16)));
  // rdma_index_source, hidden, (scale)
  const size_t num_bytes_per_msg_rdma_revecier_and_nvl_sender =
      sizeof(int4) +
      (kUseFP8
           ? (kHidden + AlignUpElems(kNumScales, kAlignElems) * sizeof(float))
           : (kHidden * sizeof(nv_bfloat16)));
  constexpr size_t combine_num_bytes_per_msg = kHidden * sizeof(nv_bfloat16);
  const size_t DISPATCH_NVL_BUFFER_X_BYTES =
      kNumLocalExperts * kNumRanks * num_max_dispatch_tokens_per_rank *
      num_bytes_per_msg_rdma_revecier_and_nvl_sender;
  const size_t COMBINE_NVL_BUFFER_X_BYTES = kNumRdmaExperts * kNumRdmaRanks *
                                            num_max_dispatch_tokens_per_rank *
                                            combine_num_bytes_per_msg;
  const size_t NVL_MAX_BUFFER_X_BYTES =
      ((DISPATCH_NVL_BUFFER_X_BYTES > COMBINE_NVL_BUFFER_X_BYTES
            ? DISPATCH_NVL_BUFFER_X_BYTES
            : COMBINE_NVL_BUFFER_X_BYTES) +
       NUM_BUFFER_ALIGNMENT_BYTES - 1) /
      NUM_BUFFER_ALIGNMENT_BYTES * NUM_BUFFER_ALIGNMENT_BYTES;
  constexpr size_t SIGNAL_BYTES = (kNumLocalExperts * kNumRanks * sizeof(int) +
                                   NUM_BUFFER_ALIGNMENT_BYTES - 1) /
                                  NUM_BUFFER_ALIGNMENT_BYTES *
                                  NUM_BUFFER_ALIGNMENT_BYTES;
  const size_t NVL_BUFFER_X_BYTES_PER_BUFFER =
      NVL_MAX_BUFFER_X_BYTES + SIGNAL_BYTES;
  const size_t NVL_BUFFER_OFFSET =
      nvl_buffer_id * NVL_BUFFER_X_BYTES_PER_BUFFER;
  const size_t num_bytes_per_msg_rdma_to_nvl =
      kUseFP8
          ? (kHidden + AlignUpElems(kNumScales, kAlignElems) * sizeof(float))
          : (kHidden * sizeof(nv_bfloat16));
  const size_t num_int4_per_msg = num_bytes_per_msg / sizeof(int4);
  const size_t num_int4_per_msg_rdma_revecier_and_nvl_sender =
      num_bytes_per_msg_rdma_revecier_and_nvl_sender / sizeof(int4);
  const size_t num_int4_per_msg_rdma_to_nvl =
      num_bytes_per_msg_rdma_to_nvl / sizeof(int4);
  EP_DEVICE_ASSERT(num_bytes_per_msg % sizeof(int4) == 0);
  EP_DEVICE_ASSERT(
      num_bytes_per_msg_rdma_revecier_and_nvl_sender % sizeof(int4) == 0);
  EP_DEVICE_ASSERT(num_bytes_per_msg_rdma_to_nvl % sizeof(int4) == 0);

  constexpr int kNumElemsPerRead = sizeof(int4) / sizeof(nv_bfloat16);
  EP_DEVICE_ASSERT(kHidden % kNumElemsPerRead == 0);
  EP_STATIC_ASSERT(kNumElemsPerRead * 32 % kNumPerChannels == 0,
                    "Invalid vectorization");
  const size_t hidden_bf16_int4 = kHidden / kNumElemsPerRead;
  
  const auto num_threads = static_cast<int>(blockDim.x),
             num_warps = num_threads / 32;
    
  const auto thread_id = static_cast<int>(threadIdx.x),
             warp_id = thread_id / 32, lane_id = get_lane_id();

  const auto warp_group_id = warp_id / kNumWarpsPerGroup;
  const auto sub_warp_id = warp_id % kNumWarpsPerGroup;
  const auto responsible_expert_idx = sm_id * kNumWarpGroups + warp_group_id;

  constexpr int kNumStages = 4;

  alignas(128) __shared__ uint8_t tmp_x[kNumStages][num_bytes_per_msg_rdma_to_nvl];
  alignas(128) __shared__ uint64_t full_barriers[kNumStages];
  alignas(128) __shared__ uint64_t o_full_barriers[kNumStages * kNumWarpGroups];
  alignas(16) __shared__ int shared_num_recv_tokens[1];

  if (thread_id < kNumStages * kNumWarpGroups) {
    mbarrier_init(o_full_barriers + thread_id, 1);
    if (thread_id < kNumStages) {
      mbarrier_init(full_barriers + thread_id, 1);
    }
    fence_barrier_init();
  }
  __syncthreads();

  uint32_t tma_phase = 0;
  uint32_t o_tma_phase = 0;

  // check
  if (sm_id == 0 && thread_id == 0) {
    EP_DEVICE_ASSERT(ibgda_get_state()->num_rc_per_pe >= 148 * (kNumRdmaRanks - 1));
  }

  const int num_chunks = cell_div(num_tokens, num_tokens_per_chunk);
  
  // Sending phase
  if ((phases & LOW_LATENCY_SEND_PHASE) == 0) goto LOW_LATENCY_DISPATCH_RECV;

  if (sm_id == 0) {
    #pragma unroll
    for (int i = thread_id; i < kNumRdmaRanks; i += num_threads) {
      packed_rdma_recv_count[i] = -1;
    }
    #pragma unroll
    for (int i = thread_id; i < kNumLocalExperts; i += num_threads) {
      packed_recv_count[i] = 0;
    }
    // clean next buffer
    #pragma unroll
    for (int i = thread_id; i < num_next_clean_int; i += num_threads) {
      next_clean[i] = 0;
    }
    // clean next nvl buffer
    #pragma unroll
    for (int i = thread_id; i < kNumExperts; i += num_threads) {
      *(reinterpret_cast<int*>(
            reinterpret_cast<uint8_t*>(nvl_recv_x[nvl_rank]) +
            next_buffer_id * NVL_BUFFER_X_BYTES_PER_BUFFER +
            NVL_MAX_BUFFER_X_BYTES) +
        i) = 0;
    }
  }
  cg::this_grid().sync();

  // 所以传输级别的flag加上chunk粒度，目的是chunk之间可以构建流水线
  if (warp_id < kRDMANumWarps) {
    // RDMA Sender
    const int wid = warp_id;
    const int tid = thread_id;
    // loop chunk
    for (int chunk_id = sm_id; chunk_id < num_chunks; chunk_id += num_sms) {
      const int chunk_offset = chunk_id * num_tokens_per_chunk;
      int dst_rdma_rank_tokens = 0;
      int dst_rdma_rank_token_id = 0;
      int dst_rdma_rank_token_offset = 0;
      // 统计该chunk需要发送的到目标rdma_rank的token个数
      for (int token_id = 0; token_id < num_tokens_per_chunk; token_id++) {
        const int token_offset = chunk_offset + token_id;
        if (token_offset > num_tokens) break;
        const int64_t* topk_idx_now = topk_idx + token_offset * kTopk;
        if (wid < kNumRdmaRanks) {
          const int dst_rdma_rank = wid;
          const int dst_rdma_expert_start = dst_rdma_rank * kNumRdmaExperts;
          const int dst_rdma_expert_end = (dst_rdma_rank + 1) * kNumRdmaExperts;
          for (int topk_i = 0; topk_i < kTopk; ++topk_i) {
            const int64_t expert_idx = topk_idx_now[topk_i];
            if (expert_idx >= dst_rdma_expert_start && expert_idx < dst_rdma_expert_end) {
              // 是否要发送到对应rdma_rank
              dst_rdma_rank_tokens += 1;
              break;
            }
          }
        }
      }
      // kRDMANumWarps >= kNumRdmaRanks
      // 一次申请连续的，后续流程需要根据这个决定循环大小
      if (wid < kNumRdmaRanks) {
        if (lane_id == 0) {
          dst_rdma_rank_token_offset = atomicAdd(&atomic_counter_per_rdma[wid], dst_rdma_rank_tokens);
        }
      }
      dst_rdma_rank_token_offset =
              __shfl_sync(0xffffffff, dst_rdma_rank_token_offset, 0);  // broadcast
      
      // loop token
      for (int token_id = 0; token_id < num_tokens_per_chunk; token_id++) {
        const int token_offset = chunk_offset + token_id;
        // token_offset越界时也需要设置flag，但不需要做量化相关的操作
        if (token_offset > num_tokens) break;
        const auto x_int4 =
          reinterpret_cast<const int4*>(x) + token_offset * hidden_bf16_int4;
        bool* rdma_send_flags_now = rdma_send_flags + token_offset * kNumRdmaRanks;
        
        // init rdma_send_flags
        #pragma unroll
        for (int flag_i = tid; flag_i < kNumRdmaRanks; flag_i += kRDMANumThreads) {
          rdma_send_flags_now[flag_i] = false;
        }
        const auto rdma_x_src_idx = reinterpret_cast<int*>(
            reinterpret_cast<uint8_t*>(rdma_x) + token_offset * num_bytes_per_msg);
        const auto rdma_x_vec = reinterpret_cast<vec_t*>(
            reinterpret_cast<uint8_t*>(rdma_x_src_idx) + sizeof(int4));
        const auto rdma_x_scales = reinterpret_cast<float*>(
            reinterpret_cast<uint8_t*>(rdma_x_vec) + hidden_bytes);
        const auto index_source = rdma_x_src_idx;
        const auto nvl_rank_meta = reinterpret_cast<int*>(
            rdma_x_scales +
            (kUseFP8 ? AlignUpElems(kNumScales, kAlignElems) : 0));
        const int64_t* topk_idx_now = topk_idx + token_offset * kTopk;
        const float* topk_weights_now = topk_weights + token_offset * kTopk;

        tid == 0 ? (*index_source = token_offset) : 0;
        
        // quant
        #pragma unroll
        for (int i = tid; i < hidden_bf16_int4; i += kRDMANumThreads) {
          // Read
          auto int4_value = __ldg(x_int4 + i);
          // convert int4 to float
          if constexpr (kUseFP8) {
            // Calculate local amax
            auto bf16_values = reinterpret_cast<nv_bfloat16*>(&int4_value);
            float fp32_values[kNumElemsPerRead];
            float amax = kFP8Margin, scale, scale_inv;
            #pragma unroll
            for (int j = 0; j < kNumElemsPerRead; ++j) {
              fp32_values[j] = static_cast<float>(bf16_values[j]);
              amax = fmaxf(amax, fabsf(fp32_values[j]));
            }

            // Reduce amax and scale, 8 * 32 / 128 = 2
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
            rdma_x_vec[i] = *reinterpret_cast<vec_t*>(&int4_value);
          }
        }
        asm volatile("bar.sync %0, %1;" ::"r"(1),
                     "r"(kRDMANumThreads));

        // RDMA Send
        if (wid < kNumRdmaRanks) {
          const int dst_rdma_rank = wid;
          const int qp_id = dst_rdma_rank + qp_offset;
          const int dst_rdma_expert_start = dst_rdma_rank * kNumRdmaExperts;
          const int dst_rdma_expert_end = (dst_rdma_rank + 1) * kNumRdmaExperts;
          const auto nvl_rank_nums =
            nvl_rank_meta + dst_rdma_rank * (kTopk * 3 + 1);
          const auto nvl_rank_meta_now = nvl_rank_nums + 1;
          int dst_nvl_count = 0;
          for (int topk_i = 0; topk_i < kTopk; ++topk_i) {
            const int64_t expert_idx = topk_idx_now[topk_i];
            const float topk_weight = topk_weights_now[topk_i];
            if (expert_idx >= dst_rdma_expert_start &&
                expert_idx < dst_rdma_expert_end) {
              if (lane_id == 0) {
                nvl_rank_meta_now[dst_nvl_count * 3] =
                    expert_idx % kNumRdmaExperts;  // dst_expert in dst_rdma_rank
                const int dst_index =
                    atomicAdd(&atomic_counter_per_expert[expert_idx], 1);
                nvl_rank_meta_now[dst_nvl_count * 3 + 1] =
                    dst_index;  // dst_index
                reinterpret_cast<float*>(
                    nvl_rank_meta_now)[dst_nvl_count * 3 + 2] = topk_weight;
              }
              dst_nvl_count += 1;
            }
          }
          lane_id == 0 ? (nvl_rank_nums[0] = dst_nvl_count) : 0;
          __syncwarp();
          // Send
          if (dst_nvl_count > 0) {
            lane_id == 0 ? (rdma_send_flags_now[dst_rdma_rank] = true) : 0;
            int dst_cum_index = dst_rdma_rank_token_offset + dst_rdma_rank_token_id;

            dst_rdma_rank_token_id++;
            const auto src_ptr = reinterpret_cast<uint64_t>(rdma_x_src_idx);
            auto dst_ptr =
                reinterpret_cast<uint64_t>(rdma_recv_x) +
                rdma_rank * num_max_dispatch_tokens_per_rank * num_bytes_per_msg +
                dst_cum_index * num_bytes_per_msg;
            if (rdma_rank == dst_rdma_rank) {
              // local copy
              const auto* src_int4_ptr = reinterpret_cast<const int4*>(src_ptr);
              auto* dst_int4_ptr = reinterpret_cast<int4*>(dst_ptr);
              UNROLLED_WARP_COPY(UNROLL_FACTOR,
                                 lane_id,
                                 num_int4_per_msg,
                                 dst_int4_ptr,
                                 src_int4_ptr,
                                 ld_nc_global,
                                 st_na_global);
            } else {
              nvshmemi_ibgda_put_nbi_warp<true, false>(
                  dst_ptr,
                  src_ptr,
                  num_bytes_per_msg,
                  dst_rdma_rank * NUM_MAX_NVL_PEERS + nvl_rank,
                  qp_id,
                  lane_id,
                  0);
            }
            __syncwarp();
          }
        }
      }

      if (wid < kNumRdmaRanks) {
        const int dst_rdma_rank = wid;
        const int qp_id = dst_rdma_rank + qp_offset;
        auto dst_ptr = reinterpret_cast<uint64_t>(
            rdma_recv_count + rdma_rank * kNumMaxChunks + chunk_id);
        const int flag_value = - ((dst_rdma_rank_token_offset << 16) + dst_rdma_rank_tokens) - 1;

        const int chunk_recv_offset_and_tokens = - flag_value - 1;
        const int new_dst_rdma_rank_token_offset = chunk_recv_offset_and_tokens >> 16;
        const int new_dst_rdma_rank_tokens = chunk_recv_offset_and_tokens & 0x0000FFFF;

        bool is_local_copy = dst_rdma_rank == rdma_rank;
        if (is_local_copy) {
          if (lane_id == 0) {
            st_na_release(
              reinterpret_cast<int*>(dst_ptr), 
              flag_value);
          }
        } else {
          nvshmemi_ibgda_amo_nonfetch_add(
              reinterpret_cast<int*>(dst_ptr),
              flag_value,
              dst_rdma_rank * NUM_MAX_NVL_PEERS + nvl_rank,
              qp_id);
        }
      }
    }
    asm volatile("bar.sync %0, %1;" ::"r"(2),
                 "r"(kRDMANumThreads));
    if (tid == 0) {
      atomic_add_release_global(
            atomic_nvl_sender_multi_sms_rdma, 1);
      if (sm_id == 0) {
        while (ld_acquire_global(atomic_nvl_sender_multi_sms_rdma) !=
                  num_sms) {
        }
        atomic_nvl_sender_multi_sms_rdma[0] = 0;
      }
    }
    asm volatile("bar.sync %0, %1;" ::"r"(2),
                 "r"(kRDMANumThreads));
    if (sm_id == 0) {
      // reset atomic_counter_per_rdma
      for (int i = tid; i < kNumRdmaRanks; i += kRDMANumThreads) {
        atomic_counter_per_rdma[i] = 0;
      }
      // reset atomic_counter_per_expert
      for (int i = tid; i < kNumExperts; i += kRDMANumThreads) {
        atomic_counter_per_expert[i] = 0;
      }
    }
  } else if (warp_id < (kRDMANumWarps + kG2SNumWarps)) {
    // RDMA Receiver And NVL Sender
    const int wid = warp_id - kRDMANumWarps;
    const int tid = thread_id - kRDMANumThreads;
    // loop rdma
    for (int chunk_id = sm_id; chunk_id < num_chunks; chunk_id += num_sms) {
        const int chunk_offset = chunk_id * num_tokens_per_chunk;
      // loop chunk
      for (int rdma_id = 0; rdma_id < kNumRdmaRanks; ++rdma_id) {
        const int src_rdma_rank = rdma_rank >= rdma_id ? rdma_rank - rdma_id : rdma_rank + kNumRdmaRanks - rdma_id;
        const int src_rank = src_rdma_rank * NUM_MAX_NVL_PEERS + nvl_rank;
        int chunk_recv_offset_and_tokens;
        // wait chunk ready
        if (tid == 0) {
          while ((chunk_recv_offset_and_tokens = ld_acquire_sys_global(
                    rdma_recv_count + src_rdma_rank * kNumMaxChunks + chunk_id)) ==
                0) {
          }
          chunk_recv_offset_and_tokens = - chunk_recv_offset_and_tokens - 1;
          shared_num_recv_tokens[0] = chunk_recv_offset_and_tokens;
        }

        asm volatile("bar.sync %0, %1;" ::"r"(3),
                     "r"(kG2SNumThreads));
        chunk_recv_offset_and_tokens = shared_num_recv_tokens[0];
        // split chunk_recv_offset_and_tokens to offset and token_num
        const int dst_rdma_rank_token_offset = chunk_recv_offset_and_tokens >> 16;
        const int dst_rdma_rank_tokens = chunk_recv_offset_and_tokens & 0x0000FFFF;
        if (tid == 0) {
          // sum dst_rdma_rank_tokens to packed_rdma_recv_count
          atomicAdd(packed_rdma_recv_count + src_rdma_rank, -dst_rdma_rank_tokens);
        }
        const int rdma_offset = (src_rdma_rank * num_max_dispatch_tokens_per_rank + dst_rdma_rank_token_offset) * num_bytes_per_msg;
        uint8_t* rdma_recv_x_uint8 = reinterpret_cast<uint8_t*>(rdma_recv_x) + rdma_offset;
        uint8_t* packed_rdma_recv_x_uint8 = reinterpret_cast<uint8_t*>(packed_rdma_recv_x) + rdma_offset;

        if (dst_rdma_rank_tokens > 0) {
          // prefetch
          if (wid == 0) {
            if (elect_one_sync()) {
              const int stage_idx = 0;
              uint8_t* rdma_recv_x_uint8_now = rdma_recv_x_uint8 + sizeof(int4);
              tma_load_1d(tmp_x[stage_idx], rdma_recv_x_uint8_now, full_barriers + stage_idx, num_bytes_per_msg_rdma_to_nvl);
              mbarrier_arrive_and_expect_tx(full_barriers + stage_idx, num_bytes_per_msg_rdma_to_nvl);
            }
            __syncwarp();
          }
          // loop dst_rdma_rank_tokens
          for (int dst_rdma_rank_token_id = 0, iter_idx = 0; dst_rdma_rank_token_id < dst_rdma_rank_tokens; ++dst_rdma_rank_token_id, ++iter_idx) {
            const int stage_idx = iter_idx % kNumStages;
            const int next_stage_idx = (iter_idx + 1) % kNumStages;

            const int tmp_offset = dst_rdma_rank_token_id * num_bytes_per_msg;
            uint8_t* rdma_recv_x_uint8_now = rdma_recv_x_uint8 + tmp_offset;
            uint8_t* packed_rdma_recv_x_uint8_now = packed_rdma_recv_x_uint8 + tmp_offset;

            const auto src_data = reinterpret_cast<int4*>(rdma_recv_x_uint8_now);
            const auto rdma_recv_x_scales = reinterpret_cast<float*>(
                reinterpret_cast<uint8_t*>(src_data) + sizeof(int4) + hidden_bytes);
            const auto rdma_recv_nvl_rank_meta = reinterpret_cast<int*>(
                rdma_recv_x_scales +
                (kUseFP8 ? AlignUpElems(kNumScales, kAlignElems) : 0));
            const int dst_nvl_experts =
                *(rdma_recv_nvl_rank_meta + rdma_rank * (kTopk * 3 + 1));
            const auto rdma_recv_nvl_rank_meta_now =
                rdma_recv_nvl_rank_meta + rdma_rank * (kTopk * 3 + 1) + 1;

            // copy next stage
            if (wid == 0 && (dst_rdma_rank_token_id + 1) < dst_rdma_rank_tokens) {
              if (elect_one_sync()) {
                tma_store_wait<2>();
                uint8_t *rdma_recv_x_uint8_next = rdma_recv_x_uint8_now + num_bytes_per_msg + sizeof(int4);
                tma_load_1d(tmp_x[next_stage_idx], rdma_recv_x_uint8_next, full_barriers + next_stage_idx, num_bytes_per_msg_rdma_to_nvl);
                mbarrier_arrive_and_expect_tx(full_barriers + next_stage_idx, num_bytes_per_msg_rdma_to_nvl);
              }
              __syncwarp();
            }

            // nvl sender
            if (wid == 0) {
              mbarrier_wait<true>(full_barriers + stage_idx, tma_phase, stage_idx);
              for (int loop_nvl_expert_i = 0; loop_nvl_expert_i < dst_nvl_experts; loop_nvl_expert_i++) {
                const int rdma_local_expert_idx =
                    rdma_recv_nvl_rank_meta_now[loop_nvl_expert_i * 3];
                const int rdma_local_expert_cumsum_index =
                    rdma_recv_nvl_rank_meta_now[loop_nvl_expert_i * 3 + 1];
                const int dst_nvl_rank = rdma_local_expert_idx / kNumLocalExperts;
                const int dst_nvl_local_expert =
                    rdma_local_expert_idx % kNumLocalExperts;
                const auto dst_data =
                    reinterpret_cast<int4*>(
                        reinterpret_cast<uint8_t*>(nvl_recv_x[dst_nvl_rank]) +
                        NVL_BUFFER_OFFSET) +
                    ((dst_nvl_local_expert * kNumRanks + src_rank) *
                        num_max_dispatch_tokens_per_rank +
                    rdma_local_expert_cumsum_index) *
                        num_int4_per_msg_rdma_revecier_and_nvl_sender;
                if (elect_one_sync()) {
                  tma_store_1d(
                    tmp_x[stage_idx], dst_data + 1, num_bytes_per_msg_rdma_to_nvl);
                }
              }
              asm volatile("cp.async.bulk.commit_group;");
              __syncwarp();
            }
            
            // used in combine, local copy
            if (wid == kG2SNumWarps - 1) {
              for (int loop_nvl_expert_i = lane_id; loop_nvl_expert_i < dst_nvl_experts; loop_nvl_expert_i += 32) {
                const int rdma_local_expert_idx =
                    rdma_recv_nvl_rank_meta_now[loop_nvl_expert_i * 3];
                const int rdma_local_expert_cumsum_index =
                    rdma_recv_nvl_rank_meta_now[loop_nvl_expert_i * 3 + 1];
                const int dst_nvl_rank = rdma_local_expert_idx / kNumLocalExperts;
                const int dst_nvl_local_expert =
                    rdma_local_expert_idx % kNumLocalExperts;
                const auto dst_data =
                    reinterpret_cast<int4*>(
                        reinterpret_cast<uint8_t*>(nvl_recv_x[dst_nvl_rank]) +
                        NVL_BUFFER_OFFSET) +
                    ((dst_nvl_local_expert * kNumRanks + src_rank) *
                        num_max_dispatch_tokens_per_rank +
                    rdma_local_expert_cumsum_index) *
                        num_int4_per_msg_rdma_revecier_and_nvl_sender;
                int* rdma_dst_cumsum_idx = reinterpret_cast<int*>(dst_data);
                st_na_global(rdma_dst_cumsum_idx, rdma_local_expert_cumsum_index);
                atomicAdd(atomic_recv_tokens_per_rdma_expert +
                          src_rdma_rank * kNumRdmaExperts +
                          rdma_local_expert_idx,
                          1);
              }
              UNROLLED_WARP_COPY(
                UNROLL_FACTOR,
                lane_id,
                num_int4_per_msg,
                reinterpret_cast<int4*>(packed_rdma_recv_x_uint8_now),
                reinterpret_cast<int4*>(rdma_recv_x_uint8_now),
                ld_nc_global,
                st_na_global);
              __syncwarp();
            }
            asm volatile("bar.sync %0, %1;" ::"r"(4),
                        "r"(kG2SNumThreads));
          }
          tma_store_wait<0>();
        }
        asm volatile("bar.sync %0, %1;" ::"r"(5),
                     "r"(kG2SNumThreads));
        if (tid == 0) {
          atomic_add_release_global(
            atomic_nvl_sender_multi_sms + src_rdma_rank, 1);
        }
      }
    }
    if (sm_id == 0) {
      // chunk finished, set flags
      for (int rdma_id = 0; rdma_id < kNumRdmaRanks; ++rdma_id) {
        const int src_rdma_rank = rdma_rank >= rdma_id ? rdma_rank - rdma_id : rdma_rank + kNumRdmaRanks - rdma_id;
        const int src_rank = src_rdma_rank * NUM_MAX_NVL_PEERS + nvl_rank;
        if (sm_id == 0 && tid == 0) {
          int tmp_value;
          while ((tmp_value = ld_acquire_global(atomic_nvl_sender_multi_sms + src_rdma_rank)) !=
                  num_chunks) {
          }
          atomic_nvl_sender_multi_sms[src_rdma_rank] = 0;
        }
        asm volatile("bar.sync %0, %1;" ::"r"(6),
                      "r"(kG2SNumThreads));
        for (int dst_rdma_local_expert_idx = tid;
              dst_rdma_local_expert_idx < NUM_MAX_NVL_PEERS * kNumLocalExperts;
              dst_rdma_local_expert_idx += kG2SNumThreads) {
          const int dst_nvl_rank = dst_rdma_local_expert_idx / kNumLocalExperts;
          const int dst_nvl_local_expert =
              dst_rdma_local_expert_idx % kNumLocalExperts;
          st_release_sys_global(
              reinterpret_cast<int*>(
                  reinterpret_cast<uint8_t*>(nvl_recv_x[dst_nvl_rank]) +
                  NVL_BUFFER_OFFSET + NVL_MAX_BUFFER_X_BYTES) +
                  dst_nvl_local_expert * kNumRanks + src_rank,
              -ld_acquire_global(atomic_recv_tokens_per_rdma_expert +
                                  src_rdma_rank * kNumRdmaExperts +
                                  dst_rdma_local_expert_idx) -
                  1);
          // reset
          *(atomic_recv_tokens_per_rdma_expert +
            src_rdma_rank * kNumRdmaExperts + dst_rdma_local_expert_idx) = 0;
        }
      }
    }
  }


  // Receiving phase
LOW_LATENCY_DISPATCH_RECV:
  if ((phases & LOW_LATENCY_RECV_PHASE) == 0) return;

  cg::this_grid().sync();

  {
    /* NVL Receiver */
    // Local PRMT
    if (responsible_expert_idx < kNumExperts) {
      const auto src_rank = responsible_expert_idx / kNumLocalExperts;
      const auto local_expert_idx = responsible_expert_idx % kNumLocalExperts;
      const auto nvl_recv_x_uint8 =
          reinterpret_cast<uint8_t*>(nvl_recv_x[nvl_rank]) + NVL_BUFFER_OFFSET +
          (local_expert_idx * kNumRanks + src_rank) *
              num_max_dispatch_tokens_per_rank *
              num_bytes_per_msg_rdma_revecier_and_nvl_sender;
      auto recv_x_int4 = reinterpret_cast<int4*>(packed_recv_x) +
                              local_expert_idx * kNumRanks *
                                  num_max_dispatch_tokens_per_rank * hidden_int4;
      const auto recv_x_scales =
          packed_recv_x_scales + local_expert_idx * kNumRanks *
                                    num_max_dispatch_tokens_per_rank *
                                    kNumScales;
      const auto recv_src_info =
          packed_recv_src_info +
          local_expert_idx * kNumRanks * num_max_dispatch_tokens_per_rank;
      const auto recv_range =
          packed_recv_layout_range + local_expert_idx * kNumRanks;

      // Shared between sub-warps in warp groups
      __shared__ int shared_num_recv_tokens[kNumWarpGroups],
                     shared_recv_token_begin_idx[kNumWarpGroups];

      // Wait tokens to arrive
      int num_recv_tokens, recv_token_begin_idx;
      EP_STATIC_ASSERT(kNumWarpsPerGroup > 1,
                      "Requires more than one warp per group");
      if (sub_warp_id == 1 && lane_id == 0) {
        while ((num_recv_tokens = ld_acquire_sys_global(
                    reinterpret_cast<int*>(
                        reinterpret_cast<uint8_t*>(nvl_recv_x[nvl_rank]) +
                        NVL_BUFFER_OFFSET + NVL_MAX_BUFFER_X_BYTES) +
                    local_expert_idx * kNumRanks + src_rank)) == 0) {
        }
        num_recv_tokens = -num_recv_tokens - 1;
        recv_token_begin_idx =
            atomicAdd(packed_recv_count + local_expert_idx, num_recv_tokens);
        shared_num_recv_tokens[warp_group_id] = num_recv_tokens;
        shared_recv_token_begin_idx[warp_group_id] = recv_token_begin_idx;
        recv_range[src_rank] =
            pack2<int, int64_t>(num_recv_tokens, recv_token_begin_idx);
      }
      asm volatile("bar.sync %0, %1;" ::"r"(warp_group_id + 7),
                  "r"(kNumWarpsPerGroup * 32));
      num_recv_tokens = shared_num_recv_tokens[warp_group_id];
      recv_token_begin_idx = shared_recv_token_begin_idx[warp_group_id];

      // Copy tokens
      EP_DEVICE_ASSERT(kNumScales <= 64);

      if (num_recv_tokens > 0) {
        if (sub_warp_id == 0) {
          // prefetch
          if (elect_one_sync()) {
            tma_load_1d(tmp_x[0], nvl_recv_x_uint8 + sizeof(int4), o_full_barriers + warp_group_id * kNumStages, hidden_bytes);
            mbarrier_arrive_and_expect_tx(o_full_barriers + warp_group_id * kNumStages, hidden_bytes);
          }
          __syncwarp();

          for (int i = 0, iter_idx = 0; i < num_recv_tokens; i++, iter_idx++) {
            const int stage_idx = iter_idx % kNumStages;
            const int next_stage_idx = (iter_idx + 1) % kNumStages;
            // copy next stage
            if ((i + 1) < num_recv_tokens) {
              if (elect_one_sync()) {
                tma_store_wait<2>();
                uint8_t *src_data = nvl_recv_x_uint8 + (i + 1) * num_bytes_per_msg_rdma_revecier_and_nvl_sender + sizeof(int4);
                tma_load_1d(tmp_x[next_stage_idx], src_data, o_full_barriers + warp_group_id * kNumStages + next_stage_idx, hidden_bytes);
                mbarrier_arrive_and_expect_tx(o_full_barriers + warp_group_id * kNumStages + next_stage_idx, hidden_bytes);
              }
              __syncwarp();
            }
            mbarrier_wait<true>(o_full_barriers + warp_group_id * kNumStages + stage_idx, o_tma_phase, stage_idx);
            if (elect_one_sync()) {
              auto dst_data =
                recv_x_int4 + (recv_token_begin_idx + i) * hidden_int4;
              tma_store_1d(
                tmp_x[stage_idx], dst_data, hidden_bytes);
              asm volatile("cp.async.bulk.commit_group;");
            }
            __syncwarp();
          }
        } else {
          for (int i = sub_warp_id - 1; i < num_recv_tokens; i += (kNumWarpsPerGroup - 1)) {
            // Copy source info
            const auto src_src_idx = reinterpret_cast<int*>(
                nvl_recv_x_uint8 +
                i * num_bytes_per_msg_rdma_revecier_and_nvl_sender);
            if (lane_id == 0)
              recv_src_info[recv_token_begin_idx + i] = ld_nc_global(src_src_idx);

            // Copy data
            const auto src_data = reinterpret_cast<int4*>(
                reinterpret_cast<uint8_t*>(src_src_idx) + sizeof(int4));

            // Copy scales
            if (kUseFP8) {
              const auto src_scales = reinterpret_cast<float*>(
                  reinterpret_cast<uint8_t*>(src_data) + hidden_bytes);
              const auto dst_scales =
                  reinterpret_cast<float*>(recv_x_scales + recv_token_begin_idx + i);
              const auto scale_stride = kNumRanks * num_max_dispatch_tokens_per_rank;
              if constexpr (kNumPerChannels == -1) {
                if (lane_id == 0) {
                  auto scale = ld_nc_global(src_scales);
                  dst_scales[0] = scale;
                }
              } else {
                auto scale_0 =
                    lane_id < kNumScales ? ld_nc_global(src_scales + lane_id) : 0;
                auto scale_1 = (lane_id + 32) < kNumScales
                                  ? ld_nc_global(src_scales + lane_id + 32)
                                  : 0;
                lane_id < kNumScales ? dst_scales[lane_id * scale_stride] = scale_0
                                    : 0.0f;
                (lane_id + 32) < kNumScales
                    ? dst_scales[(lane_id + 32) * scale_stride] = scale_1
                    : 0.0f;
              }
            }
          }
        }
      }
    }
  }
}

void dispatch(void* packed_recv_x,
              float* packed_recv_x_scales,
              void* packed_rdma_recv_x,
              int* packed_recv_src_info,
              int64_t* packed_recv_layout_range,
              int* packed_recv_count,
              int* packed_rdma_recv_count,
              bool* rdma_send_flags,
              void* rdma_recv_x,
              int* rdma_recv_count,
              void* rdma_x,
              void** nvl_recv_x,
              const void* x,
              const int64_t* topk_idx,
              const float* topk_weights,
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
              int phases,
              int next_buffer_id,
              int num_per_channel) {
  constexpr int kNumMaxTopK = 8;
  constexpr int NUM_WARPS = 32;

  const int dev_id = 0;
  int sm_count;
  cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, dev_id);
  const int num_warp_groups = cell_div(num_experts, sm_count);
  const int num_sms_local = min(sm_count, cell_div(num_experts, num_warp_groups));
  const auto num_sms = max(sm_count, num_sms_local);
  const int num_tokens_per_chunk = cell_div(num_tokens, num_sms);
  
  EP_HOST_ASSERT(num_topk <= kNumMaxTopK);
  const int num_rdma_ranks = num_ranks / NUM_MAX_NVL_PEERS;
  const int num_rdma_experts = num_experts / num_rdma_ranks;
  // Workspace checks
  auto atomic_counter_per_expert = reinterpret_cast<int*>(workspace);
  auto atomic_counter_per_rdma = atomic_counter_per_expert + num_experts;
  auto atomic_finished_counter_per_rdma =
      atomic_counter_per_rdma + num_rdma_ranks;
  auto atomic_recv_tokens_per_rdma_expert =
      atomic_finished_counter_per_rdma + num_rdma_ranks;
  auto atomic_nvl_sender_multi_sms =
      atomic_recv_tokens_per_rdma_expert +
      num_rdma_ranks * num_rdma_experts;  // num_rdma_ranks
  auto atomic_nvl_sender_multi_sms_rdma = atomic_nvl_sender_multi_sms + num_rdma_ranks;
  EP_HOST_ASSERT((num_experts + num_rdma_ranks * 3 + 1 + num_rdma_ranks * num_rdma_experts) *
                     sizeof(int) <=
                 NUM_WORKSPACE_BYTES);

  DISPATCH_HIDDEN_SIZE(
      hidden,
      kHidden,
      {DISPATCH_NUM_TOPK(
          num_topk,
          kTopk,
          {DISPATCH_RDMA_RANKS(
              num_rdma_ranks,
              kNumRdmaRanks,
              {DISPATCH_NUM_EXPERTS(
                  num_experts,
                  kNumExperts,
                  {DISPATCH_NUM_WARP_GROUPS(
                      num_warp_groups,
                      kNumWarpGroups,
                      {DISPATCH_NUM_PER_CHANNEL(
                          num_per_channel, kNumPerChannels, {
                            constexpr int kNumWarpsPerGroup =
                                NUM_WARPS / kNumWarpGroups;
                            assert(num_rdma_ranks <=
                                   kNumWarpGroups * kNumWarpsPerGroup);
                            EP_STATIC_ASSERT(
                                kNumMaxTopK + 1 <=
                                    kNumWarpGroups * kNumWarpsPerGroup,
                                "Too many top-k selections");
                            constexpr int kRDMANumWarps = 24;
                            auto dispatch_func = dispatch_wp_kernel<true,
                                                          kNumWarpGroups,
                                                          kNumWarpsPerGroup,
                                                          kHidden,
                                                          kNumRdmaRanks,
                                                          kNumExperts,
                                                          kTopk,
                                                          kRDMANumWarps, // kNumRdmaRanks, // kRDMANumWarps
                                                          NUM_WARPS - kRDMANumWarps,
                                                          kNumPerChannels>;
                            SETUP_LAUNCH_CONFIG(
                                num_sms,
                                kNumWarpGroups * kNumWarpsPerGroup * 32,
                                stream);
                            LAUNCH_KERNEL(&cfg,
                                          dispatch_func,
                                          packed_recv_x,
                                          packed_recv_x_scales,
                                          packed_rdma_recv_x,
                                          packed_recv_src_info,
                                          packed_recv_layout_range,
                                          packed_recv_count,
                                          packed_rdma_recv_count,
                                          rdma_send_flags,
                                          rdma_recv_x,
                                          rdma_recv_count,
                                          rdma_x,
                                          nvl_recv_x,
                                          x,
                                          topk_idx,
                                          topk_weights,
                                          atomic_counter_per_expert,
                                          atomic_counter_per_rdma,
                                          atomic_finished_counter_per_rdma,
                                          atomic_recv_tokens_per_rdma_expert,
                                          atomic_nvl_sender_multi_sms,
                                          atomic_nvl_sender_multi_sms_rdma,
                                          next_clean,
                                          num_next_clean_int,
                                          num_tokens,
                                          num_max_dispatch_tokens_per_rank,
                                          num_tokens_per_chunk,
                                          rank,
                                          phases,
                                          next_buffer_id);
                          })})})})})});
}

template <int kNumWarpGroups,
          int kNumWarpsPerGroup,
          int kHidden,
          int kNumRdmaRanks,
          int kNumExperts,
          int kTopk,
          bool kDispatchUseFP8,
          int kNumQPs,
          int kNumPerChannels = 128>
__global__ __launch_bounds__(
    kNumWarpGroups* kNumWarpsPerGroup * 32,
    1) void combine_kernel(void* combined_x,
                           void* rdma_recv_x,
                           int* rdma_recv_flag,
                           void* rdma_send_x,
                           void* dispatch_rdma_recv_x,
                           const int* dispatch_rdma_recv_count,
                           void** nvl_recv_buffer,
                           const void* x,
                           const int64_t* topk_idx,
                           const float* topk_weights,
                           const int* src_info,
                           const int64_t* layout_range,
                           const bool* rdma_send_flags,
                           int* next_clean,
                           int num_next_clean_int,  // Not used temporarily
                           int* atomic_clean_flag,
                           int* atomic_nvl_sender_multi_sms,
                           int num_combined_tokens,
                           int hidden,
                           int num_topk,
                           int num_max_dispatch_tokens_per_rank,
                           int num_experts,
                           int rank,
                           int num_ranks,
                           int phases,
                           int next_buffer_id) {
  constexpr int UNROLL_FACTOR = kHidden / 1024;
  constexpr int kNumRanks = kNumRdmaRanks * NUM_MAX_NVL_PEERS;
  constexpr int kNumLocalExperts = kNumExperts / kNumRanks;
  constexpr int kNumRdmaExperts = kNumLocalExperts * NUM_MAX_NVL_PEERS;
  constexpr int kAlignElems = sizeof(int4) / sizeof(float);
  constexpr int kNumScales =
      kNumPerChannels == -1 ? 1 : kHidden / kNumPerChannels;
  const int nvl_buffer_id = next_buffer_id ^ 1;

  const size_t num_bytes_per_msg_dispatch =
      sizeof(int4) +
      (kNumRdmaRanks * (kTopk * 3 + 1) * sizeof(int) + sizeof(int4) - 1) /
          sizeof(int4) * sizeof(int4) +
      (kDispatchUseFP8
           ? (kHidden + AlignUpElems(kNumScales, kAlignElems) * sizeof(float))
           : (kHidden * sizeof(nv_bfloat16)));
  const size_t num_bytes_per_msg_rdma_revecier_and_nvl_sender_dispatch =
      sizeof(int4) +
      (kDispatchUseFP8
           ? (kHidden + AlignUpElems(kNumScales, kAlignElems) * sizeof(float))
           : (kHidden * sizeof(nv_bfloat16)));

  const size_t dispatch_hidden_bytes =
      kHidden *
      (kDispatchUseFP8 ? sizeof(__nv_fp8_storage_t) : sizeof(nv_bfloat16));
  const size_t combine_hidden_bytes = kHidden * sizeof(nv_bfloat16);
  const size_t combine_hidden_int4_num = combine_hidden_bytes / sizeof(int4);

  const auto sm_id = static_cast<int>(blockIdx.x);
  const auto num_sms = static_cast<int>(gridDim.x);

  const int num_qps = num_sms * kNumRdmaRanks;
  const int qp_offset = sm_id * kNumRdmaRanks;

  const auto thread_id = static_cast<int>(threadIdx.x);
  const auto num_threads = static_cast<int>(blockDim.x),
             num_warps = num_threads / 32;
  const auto warp_id = thread_id / 32, lane_id = get_lane_id();
  const auto num_local_experts = num_experts / num_ranks;
  const auto warp_group_id = warp_id / kNumWarpsPerGroup;
  const auto sub_warp_id = warp_id % kNumWarpsPerGroup;
  const auto responsible_expert_idx = sm_id * kNumWarpGroups + warp_group_id;

  const auto rdma_rank = rank / NUM_MAX_NVL_PEERS,
             nvl_rank = rank % NUM_MAX_NVL_PEERS;

  constexpr int kNumElemsPerInt4 = sizeof(int4) / sizeof(nv_bfloat16);
  const size_t hidden_bf16_int4 = kHidden / kNumElemsPerInt4;
  if (sm_id == 0 && thread_id == 0) {
    EP_DEVICE_ASSERT(ibgda_get_state()->num_rc_per_pe >= kNumQPs);
  }

  constexpr size_t num_bytes_per_slot = kHidden * sizeof(nv_bfloat16);
  const size_t DISPATCH_NVL_BUFFER_X_BYTES =
      kNumLocalExperts * kNumRanks * num_max_dispatch_tokens_per_rank *
      num_bytes_per_msg_rdma_revecier_and_nvl_sender_dispatch;
  const size_t COMBINE_NVL_BUFFER_X_BYTES = kNumRdmaExperts * kNumRdmaRanks *
                                            num_max_dispatch_tokens_per_rank *
                                            num_bytes_per_slot;
  const size_t NVL_MAX_BUFFER_X_BYTES =
      ((DISPATCH_NVL_BUFFER_X_BYTES > COMBINE_NVL_BUFFER_X_BYTES
            ? DISPATCH_NVL_BUFFER_X_BYTES
            : COMBINE_NVL_BUFFER_X_BYTES) +
       NUM_BUFFER_ALIGNMENT_BYTES - 1) /
      NUM_BUFFER_ALIGNMENT_BYTES * NUM_BUFFER_ALIGNMENT_BYTES;
  constexpr size_t SIGNAL_BYTES = (kNumLocalExperts * kNumRanks * sizeof(int) +
                                   NUM_BUFFER_ALIGNMENT_BYTES - 1) /
                                  NUM_BUFFER_ALIGNMENT_BYTES *
                                  NUM_BUFFER_ALIGNMENT_BYTES;
  const size_t NVL_BUFFER_X_BYTES_PER_BUFFER =
      NVL_MAX_BUFFER_X_BYTES + SIGNAL_BYTES;
  const size_t NVL_BUFFER_OFFSET =
      nvl_buffer_id * NVL_BUFFER_X_BYTES_PER_BUFFER;

  // Sending phase
  if ((phases & LOW_LATENCY_SEND_PHASE) == 0) goto LOW_LATENCY_COMBINE_RECV;

  // Clean up next buffer
  if (sm_id == 0) {
    #pragma unroll
    for (int i = thread_id; i < num_next_clean_int; i += num_threads) {
      next_clean[i] = 0;
    }
    for (int i = thread_id; i < kNumExperts; i += num_threads) {
      // reset nvl_recv_buffer
      *(reinterpret_cast<int*>(
            reinterpret_cast<uint8_t*>(nvl_recv_buffer[nvl_rank]) +
            next_buffer_id * NVL_BUFFER_X_BYTES_PER_BUFFER +
            NVL_MAX_BUFFER_X_BYTES) +
        i) = 0;
    }
  }
  cg::this_grid().sync();
  
  /* NVL Sender */
  if (responsible_expert_idx < num_experts) {
    const auto dst_rank = responsible_expert_idx / num_local_experts;
    const auto dst_rdma_rank = dst_rank / NUM_MAX_NVL_PEERS;
    const auto dst_nvl_rank = dst_rank % NUM_MAX_NVL_PEERS;
    const auto local_expert_idx = responsible_expert_idx % num_local_experts;
    const auto global_rdma_expert_idx =
        nvl_rank * num_local_experts + local_expert_idx;
    const auto local_x = reinterpret_cast<const int4*>(x) +
                         local_expert_idx * num_ranks *
                             num_max_dispatch_tokens_per_rank *
                             hidden_bf16_int4;
    const auto local_src_info =
        src_info +
        local_expert_idx * num_ranks *
            num_max_dispatch_tokens_per_rank;  // [dst_rank_index_source,
                                               // dst_rdma_index, topk_weight]
    const auto layout =
        __ldg(layout_range + local_expert_idx * num_ranks + dst_rank);

    // Unpack layout
    int offset, num_tokens_to_send;
    unpack2(layout, num_tokens_to_send, offset);

    for (int token_idx = sub_warp_id; token_idx < num_tokens_to_send;
         token_idx += kNumWarpsPerGroup) {
      const int idx_now = token_idx + offset;
      const int* src_idxs = local_src_info + idx_now;
      const int dst_rdma_index = src_idxs[0];
      // nvl recv buffer
      const auto dst_ptr = reinterpret_cast<int4*>(
          reinterpret_cast<uint8_t*>(nvl_recv_buffer[dst_nvl_rank]) +
          NVL_BUFFER_OFFSET +
          ((global_rdma_expert_idx * kNumRdmaRanks + dst_rdma_rank) *
               num_max_dispatch_tokens_per_rank +
           dst_rdma_index) *
              num_bytes_per_slot);
      const auto x_int4 = local_x + idx_now * hidden_bf16_int4;
      UNROLLED_WARP_COPY(7,
                         lane_id,
                         hidden_bf16_int4,
                         dst_ptr,
                         x_int4,
                         ld_nc_global,
                         st_na_global);
      __syncwarp();
    }
    // Put nvl finished flag
    EP_STATIC_ASSERT(kNumWarpsPerGroup > 1,
                     "Requires more than one warp per group");
    asm volatile("bar.sync %0, %1;" ::"r"(warp_group_id + 1),
                 "r"(kNumWarpsPerGroup * 32));
    if (sub_warp_id == 1 && lane_id == 0) {
      auto dst_ptr =
          reinterpret_cast<int*>(
              reinterpret_cast<uint8_t*>(nvl_recv_buffer[dst_nvl_rank]) +
              NVL_BUFFER_OFFSET + NVL_MAX_BUFFER_X_BYTES) +
          global_rdma_expert_idx * kNumRdmaRanks + dst_rdma_rank;
      st_release_sys_global(dst_ptr, 1);
    }
    __syncwarp();
  }

  // Wait all nvl ranks to arrive
  if (responsible_expert_idx < num_experts) {
    EP_STATIC_ASSERT(kNumWarpsPerGroup > 1,
                     "Invalid number of warps per group");
    // if (sub_warp_id == 0 && lane_id == 0) {
    if (thread_id == 0) {
      while (ld_acquire_sys_global(
                 reinterpret_cast<int*>(
                     reinterpret_cast<uint8_t*>(nvl_recv_buffer[nvl_rank]) +
                     NVL_BUFFER_OFFSET + NVL_MAX_BUFFER_X_BYTES) +
                 responsible_expert_idx) == 0) {
      }
    }
  }
  cg::this_grid().sync();

  /* NVL Receiver / NVL Reducer */
  {
    for (int rdma_id = 0; rdma_id < kNumRdmaRanks; ++rdma_id) {
      // 先处理当前rdma rank的
      const int deal_rdma_rank = rdma_rank >= rdma_id ? rdma_rank - rdma_id : rdma_rank + kNumRdmaRanks - rdma_id;
      const int qp_id = qp_offset + deal_rdma_rank;
      const int num_tokens_to_deal =
          (-dispatch_rdma_recv_count[deal_rdma_rank] - 1);
      const auto dispatch_rdma_recv_x_this_rdma_rank =
          reinterpret_cast<uint8_t*>(dispatch_rdma_recv_x) +
          deal_rdma_rank * num_max_dispatch_tokens_per_rank *
              num_bytes_per_msg_dispatch;
      auto rdma_send_x_this_rdma_rank =
          reinterpret_cast<uint8_t*>(rdma_send_x) +
          deal_rdma_rank * num_max_dispatch_tokens_per_rank *
              combine_hidden_bytes;
      // reduce
      for (int rdma_recv_token_idx = sm_id; rdma_recv_token_idx < num_tokens_to_deal; rdma_recv_token_idx += num_sms) {
        const auto dispatch_rdma_recv_x_now =
            dispatch_rdma_recv_x_this_rdma_rank +
            rdma_recv_token_idx * num_bytes_per_msg_dispatch;
        const auto index_source =
            reinterpret_cast<const int*>(dispatch_rdma_recv_x_now)[0];
        const int* nvl_rank_meta = reinterpret_cast<const int*>(
            dispatch_rdma_recv_x_now + sizeof(int4) + dispatch_hidden_bytes +
            (kDispatchUseFP8
                 ? AlignUpElems(kNumScales, kAlignElems) * sizeof(float)
                 : 0));
        const int nvl_rank_nums =
            *(nvl_rank_meta + rdma_rank * (kTopk * 3 + 1));
        const int* nvl_rank_meta_now =
            nvl_rank_meta + rdma_rank * (kTopk * 3 + 1) + 1;
        int4* dst_ptr = reinterpret_cast<int4*>(
            rdma_send_x_this_rdma_rank + index_source * combine_hidden_bytes);
        for (int g_id = thread_id; g_id < hidden_bf16_int4;
             g_id += num_threads) {
          float combined_values[kNumElemsPerInt4] = {0.0f};
          for (int nvl_rank_idx = 0; nvl_rank_idx < nvl_rank_nums;
               nvl_rank_idx += 1) {
            const int dst_rdma_expert_idx = nvl_rank_meta_now[nvl_rank_idx * 3];
            const int dst_cum_index = nvl_rank_meta_now[nvl_rank_idx * 3 + 1];
            const float topk_weight = reinterpret_cast<const float*>(
                nvl_rank_meta_now)[nvl_rank_idx * 3 + 2];
            const int4* src_ptr = reinterpret_cast<int4*>(
                reinterpret_cast<uint8_t*>(nvl_recv_buffer[nvl_rank]) +
                NVL_BUFFER_OFFSET +
                ((dst_rdma_expert_idx * kNumRdmaRanks + deal_rdma_rank) *
                     num_max_dispatch_tokens_per_rank +
                 dst_cum_index) *
                    num_bytes_per_slot);
            auto x_vec = ld_nc_global(src_ptr + g_id);
            const auto x_bf16 = reinterpret_cast<nv_bfloat16*>(&x_vec);
            #pragma unroll
            for (int j = 0; j < kNumElemsPerInt4; ++j)
              combined_values[j] += static_cast<float>(x_bf16[j]) * topk_weight;
          }
          int4& combined_int4 = *reinterpret_cast<int4*>(combined_values);
          auto combined_bf16 = reinterpret_cast<nv_bfloat16*>(&combined_values);
          #pragma unroll
          for (int j = 0; j < kNumElemsPerInt4; ++j)
            combined_bf16[j] = static_cast<nv_bfloat16>(combined_values[j]);
          dst_ptr[g_id] = combined_int4;
        }
        __syncthreads();
        // issue copy to remote rdma per token
        if (warp_id == 0) {
          const auto src_ptr = reinterpret_cast<uint64_t>(
              rdma_send_x_this_rdma_rank + index_source * combine_hidden_bytes);
          const auto dst_ptr =
              reinterpret_cast<uint64_t>(rdma_recv_x) +
              (rdma_rank * num_max_dispatch_tokens_per_rank + index_source) *
                  combine_hidden_bytes;
          if (rdma_rank == deal_rdma_rank) {
            // local copy
            const auto* src_int4_ptr = reinterpret_cast<const int4*>(src_ptr);
            const auto* dst_int4_ptr = reinterpret_cast<int4*>(dst_ptr);
            UNROLLED_WARP_COPY(UNROLL_FACTOR,
                               lane_id,
                               combine_hidden_int4_num,
                               dst_int4_ptr,
                               src_int4_ptr,
                               ld_nc_global,
                               st_na_global);
          } else {
            nvshmemi_ibgda_put_nbi_warp<true>(
              dst_ptr,
              src_ptr,
              combine_hidden_bytes,
              deal_rdma_rank * NUM_MAX_NVL_PEERS + nvl_rank,
              qp_id,
              lane_id,
              0);
          }
          __syncwarp();
        }
      }
      cg::this_grid().sync();
      // set flag
      if (sm_id == 0 && thread_id == 0) {
        // notify remote rdma
        auto dst_rdma_flag = reinterpret_cast<uint64_t>(
            rdma_recv_flag + rdma_rank);
        bool is_local_copy = deal_rdma_rank == rdma_rank;
        if (is_local_copy) {
          st_na_release(rdma_recv_flag + rdma_rank, 1);
        } else {
          nvshmemi_ibgda_amo_nonfetch_add(
              reinterpret_cast<int*>(dst_rdma_flag),
              1,
              deal_rdma_rank * NUM_MAX_NVL_PEERS + nvl_rank,
              thread_id);
        }
      }
    }
  }

  // Receiving phase
LOW_LATENCY_COMBINE_RECV:
  if ((phases & LOW_LATENCY_RECV_PHASE) == 0) return;

  if (sm_id < kNumRdmaRanks) {
    if (thread_id == 0) {
      while (ld_acquire_sys_global(rdma_recv_flag + sm_id) == 0) {
      }
    }
  }
  cg::this_grid().sync();

  for (int token_idx = sm_id; token_idx < num_combined_tokens; token_idx += num_sms) {
    for (int g_id = thread_id; g_id < hidden_bf16_int4; g_id += num_threads) {
      float combined_values[kNumElemsPerInt4] = {0.0f};
      const bool* rdma_send_flags_now =
          rdma_send_flags + token_idx * kNumRdmaRanks;
      for (int rdma_rank_idx = 0; rdma_rank_idx < kNumRdmaRanks;
           ++rdma_rank_idx) {
        if (rdma_send_flags_now[rdma_rank_idx]) {
          const int4* src_ptr = reinterpret_cast<int4*>(
              reinterpret_cast<uint8_t*>(rdma_recv_x) +
              (rdma_rank_idx * num_max_dispatch_tokens_per_rank + token_idx) *
                  combine_hidden_bytes);
          auto x_vec = ld_nc_global(src_ptr + g_id);
          const auto x_bf16 = reinterpret_cast<nv_bfloat16*>(&x_vec);
          #pragma unroll
          for (int j = 0; j < kNumElemsPerInt4; ++j)
            combined_values[j] += static_cast<float>(x_bf16[j]);
        }
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
             void* dispatch_rdma_recv_x,
             const int* dispatch_rdma_recv_count,
             void** nvl_buffer,
             const void* x,  // num_local_experts * num_ranks * kHidden
             const int64_t* topk_idx,
             const float* topk_weights,
             const int* src_info,
             const int64_t* layout_range,
             const bool* rdma_send_flags,
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
             bool dispatch_use_fp8,
             int next_buffer_id,
             int num_per_channel) {
  constexpr int kNumMaxTopk = 8;
  constexpr int kNumQPs = 4;
  constexpr int NUM_WARPS = 32;

  const int dev_id = 0;
  int sm_count;
  cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, dev_id);
  const int num_warp_groups = cell_div(num_experts, sm_count);
  const auto num_sms = min(sm_count, cell_div(num_experts, num_warp_groups));
  const int num_rdma_ranks = num_ranks / NUM_MAX_NVL_PEERS;

  // Check workspace
  auto atomic_clean_flag = reinterpret_cast<int*>(workspace);
  auto atomic_nvl_sender_multi_sms = atomic_clean_flag + 1;
  EP_HOST_ASSERT((1 + num_rdma_ranks) * sizeof(int) <= NUM_WORKSPACE_BYTES);
  EP_HOST_ASSERT(num_topk <= kNumMaxTopk);

  DISPATCH_HIDDEN_SIZE(
      hidden,
      kHidden,
      {DISPATCH_NUM_TOPK(
          num_topk,
          kTopk,
          {DISPATCH_RDMA_RANKS(
              num_rdma_ranks,
              kNumRdmaRanks,
              {DISPATCH_NUM_EXPERTS(
                  num_experts,
                  kNumExperts,
                  {DISPATCH_NUM_WARP_GROUPS(
                      num_warp_groups,
                      kNumWarpGroups,
                      {DISPATCH_NUM_PER_CHANNEL(
                          num_per_channel, kNumPerChannels, {
                            constexpr int kNumWarpsPerGroup =
                                NUM_WARPS / kNumWarpGroups;
                            auto combine_func =
                                dispatch_use_fp8
                                    ? combine_kernel<kNumWarpGroups,
                                                     kNumWarpsPerGroup,
                                                     kHidden,
                                                     kNumRdmaRanks,
                                                     kNumExperts,
                                                     kTopk,
                                                     true,
                                                     kNumQPs,
                                                     kNumPerChannels>
                                    : combine_kernel<kNumWarpGroups,
                                                     kNumWarpsPerGroup,
                                                     kHidden,
                                                     kNumRdmaRanks,
                                                     kNumExperts,
                                                     kTopk,
                                                     false,
                                                     kNumQPs,
                                                     kNumPerChannels>;
                            SETUP_LAUNCH_CONFIG(
                                num_sms,
                                kNumWarpGroups * kNumWarpsPerGroup * 32,
                                stream);
                            LAUNCH_KERNEL(&cfg,
                                          combine_func,
                                          combined_x,
                                          rdma_recv_x,
                                          rdma_recv_flag,
                                          rdma_send_x,
                                          dispatch_rdma_recv_x,
                                          dispatch_rdma_recv_count,
                                          nvl_buffer,
                                          x,
                                          topk_idx,
                                          topk_weights,
                                          src_info,
                                          layout_range,
                                          rdma_send_flags,
                                          next_clean,
                                          num_next_clean_int,
                                          atomic_clean_flag,
                                          atomic_nvl_sender_multi_sms,
                                          num_combined_tokens,
                                          hidden,
                                          num_topk,
                                          num_max_dispatch_tokens_per_rank,
                                          num_experts,
                                          rank,
                                          num_ranks,
                                          phases,
                                          next_buffer_id);
                          })})})})})})
}

}  // namespace internode_ll_two_stage

}  // namespace deep_ep
