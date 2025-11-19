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

namespace m2n_ll_two_stage {

constexpr bool M2N_LL_DEBUG = false;
constexpr bool M2N_LL_ACC_DEBUG = false;
constexpr bool M2N_LL_HANG_DEBUG = true;
constexpr int64_t M2N_NUM_HANG_CYCLES = 2000000000;  // 345MHZ 5.8s;

template <bool kUseFP8,
          int kNumWarpGroups,
          int kNumWarpsPerGroup,
          int kHidden,
          int kNumRdmaRanks,
          int kNumExperts,
          int kTopk,
          int kNumQPs>
__global__ __launch_bounds__(
    kNumWarpGroups* kNumWarpsPerGroup * 32,
    1) void dispatch_kernel(void* packed_recv_x,
                            float* packed_recv_x_scales,
                            void* packed_rdma_recv_x,
                            int* packed_recv_src_info,
                            int64_t* packed_recv_layout_range,
                            int* packed_recv_count,
                            int* packed_rdma_recv_count,
                            bool* rdma_send_flags,  // kNumRdmaRanks
                            void* rdma_recv_x,
                            int* rdma_recv_count,
                            int* rdma_recv_complete,
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
                            int* atomic_counter_per_qp,
                            int num_tokens,
                            int num_max_dispatch_tokens_per_rank,
                            int rank,
                            int a_start_rank,
                            int a_num_ranks,
                            int e_start_rank,
                            int e_num_ranks,
                            int phases) {
  constexpr int UNROLL_FACTOR = kHidden / 1024;
  constexpr int kNumRanks = kNumRdmaRanks * NUM_MAX_NVL_PEERS;
  constexpr int kNumLocalExperts = kNumExperts / kNumRanks;
  constexpr int kNumRdmaExperts = kNumLocalExperts * NUM_MAX_NVL_PEERS;

  const auto sm_id = static_cast<int>(blockIdx.x);
  const auto num_sms = static_cast<int>(gridDim.x);
  const auto num_threads = static_cast<int>(blockDim.x),
             num_warps = num_threads / 32;
  const auto thread_id = static_cast<int>(threadIdx.x),
             warp_id = thread_id / 32, lane_id = get_lane_id();
  const auto warp_group_id = warp_id / kNumWarpsPerGroup;
  const auto sub_warp_id = warp_id % kNumWarpsPerGroup;
  const auto responsible_expert_idx = sm_id * kNumWarpGroups + warp_group_id;
  int a_start_rdma_rank = a_start_rank / NUM_MAX_NVL_PEERS;
  int a_num_rdma_ranks = a_num_ranks / NUM_MAX_NVL_PEERS;
  int e_start_rdma_rank = e_start_rank / NUM_MAX_NVL_PEERS;
  int e_num_rdma_ranks = e_num_ranks / NUM_MAX_NVL_PEERS;

  const auto rdma_rank = rank / NUM_MAX_NVL_PEERS,
             nvl_rank = rank % NUM_MAX_NVL_PEERS;
  const int qp_id = sm_id % kNumQPs;
  // check
  if (sm_id == 0 && thread_id == 0) {
    EP_DEVICE_ASSERT(ibgda_get_state()->num_rc_per_pe >= kNumQPs);
  }

  // FP8 staffs
  constexpr int kNumPerChannels = 128;
  constexpr float kFP8Margin = 1e-4, kFP8Amax = 448,
                  kFP8AmaxInv = 1.0f / 448.0f;
  constexpr int kNumScales = kHidden / kNumPerChannels;
  const size_t hidden_bytes =
      kHidden * (kUseFP8 ? sizeof(__nv_fp8_storage_t) : sizeof(nv_bfloat16));
  const size_t hidden_int4 = hidden_bytes / sizeof(int4);

  // index_source, hidden, (scale), nvl_valid_num, nvl_rank0, dst_idx0,
  // topk_weight0,
  // ..., nvl_rank8, dst_idx8, topk_weight8, ...
  using vec_t = typename std::conditional<kUseFP8, int2, int4>::type;
  const size_t num_bytes_per_msg =
      sizeof(int4) +
      (kNumRdmaRanks * (kTopk * 3 + 1) * sizeof(int) + sizeof(int4) - 1) /
          sizeof(int4) * sizeof(int4) +
      (kUseFP8 ? (kHidden + kNumScales * sizeof(float))
               : (kHidden * sizeof(nv_bfloat16)));
  // rdma_index_source, hidden, (scale)
  const size_t num_bytes_per_msg_rdma_revecier_and_nvl_sender =
      sizeof(int4) + (kUseFP8 ? (kHidden + kNumScales * sizeof(float))
                              : (kHidden * sizeof(nv_bfloat16)));
  const size_t NVL_BUFFER_X_BYTES =
      kNumLocalExperts * kNumRanks * num_max_dispatch_tokens_per_rank *
      num_bytes_per_msg_rdma_revecier_and_nvl_sender;
  const size_t num_bytes_per_msg_rdma_to_nvl =
      kUseFP8 ? (kHidden + kNumScales * sizeof(float))
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

  if ((phases & LOW_LATENCY_SEND_PHASE) == 0) goto LOW_LATENCY_DISPATCH_RECV;

  /* RDMA Sender */
  {
    constexpr int kNumElemsPerRead = sizeof(int4) / sizeof(nv_bfloat16);
    EP_DEVICE_ASSERT(kHidden % kNumElemsPerRead == 0);
    EP_STATIC_ASSERT(kNumElemsPerRead * 32 % kNumPerChannels == 0,
                     "Invalid vectorization");
    const size_t hidden_bf16_int4 = kHidden / kNumElemsPerRead;

    for (int token_idx = sm_id; token_idx < num_tokens; token_idx += num_sms) {
      const auto x_int4 =
          reinterpret_cast<const int4*>(x) + token_idx * hidden_bf16_int4;
      bool* rdma_send_flags_now = rdma_send_flags + token_idx * kNumRdmaRanks;

// init rdma_send_flags
#pragma unroll
      for (int flag_i = thread_id; flag_i < kNumRdmaRanks;
           flag_i += num_threads) {
        rdma_send_flags_now[flag_i] = false;
      }
      const auto rdma_x_src_idx = reinterpret_cast<int*>(
          reinterpret_cast<uint8_t*>(rdma_x) + token_idx * num_bytes_per_msg);
      const auto rdma_x_vec = reinterpret_cast<vec_t*>(
          reinterpret_cast<uint8_t*>(rdma_x_src_idx) + sizeof(int4));
      const auto rdma_x_scales = reinterpret_cast<float*>(
          reinterpret_cast<uint8_t*>(rdma_x_vec) + hidden_bytes);

      const auto nvl_rank_meta =
          reinterpret_cast<int*>(rdma_x_scales + (kUseFP8 ? kNumScales : 0));

      thread_id == 0 ? (*rdma_x_src_idx = token_idx) : 0;

#pragma unroll
      for (int i = thread_id; i < hidden_bf16_int4; i += num_threads) {
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
      __syncthreads();

      // Only need issue to MoE machine!
      if (warp_id < e_num_rdma_ranks) {
        const int dst_rdma_rank = warp_id + e_start_rdma_rank;
        const int dst_rdma_expert_start = dst_rdma_rank * kNumRdmaExperts;
        const int dst_rdma_expert_end = (dst_rdma_rank + 1) * kNumRdmaExperts;

        const int64_t* topk_idx_now = topk_idx + token_idx * kTopk;
        const float* topk_weights_now = topk_weights + token_idx * kTopk;

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

        // dst_nvl_count > 0 means should issue message to dst_rdma_rank!
        if (dst_nvl_count > 0) {
          lane_id == 0 ? (rdma_send_flags_now[dst_rdma_rank] = true) : 0;
          int slot_idx =
              lane_id == 0
                  ? atomicAdd(&atomic_counter_per_rdma[dst_rdma_rank], 1)
                  : 0;
          slot_idx = __shfl_sync(0xffffffff, slot_idx, 0);  // broadcast
          const auto src_ptr = reinterpret_cast<uint64_t>(rdma_x_src_idx);
          const auto dst_ptr =
              reinterpret_cast<uint64_t>(rdma_recv_x) +
              (rdma_rank * num_max_dispatch_tokens_per_rank + slot_idx) *
                  num_bytes_per_msg;

          // must run in RDMA!
          if constexpr (kNumQPs > 1) {
            nvshmemi_ibgda_put_nbi_warp<true>(
                dst_ptr,
                src_ptr,
                num_bytes_per_msg,
                dst_rdma_rank * NUM_MAX_NVL_PEERS + nvl_rank,
                qp_id,
                lane_id,
                0);
          } else {
            nvshmemi_ibgda_put_nbi_warp(
                dst_ptr,
                src_ptr,
                num_bytes_per_msg,
                dst_rdma_rank * NUM_MAX_NVL_PEERS + nvl_rank,
                qp_id,
                lane_id,
                slot_idx);
          }
          __syncwarp();
          lane_id == 0
              ? (atomic_add_release_global(
                    atomic_finished_counter_per_rdma + dst_rdma_rank, 1))
              : 0;
        }
      }
    }
  }
  if (sm_id == num_sms - 1) {
    for (int i = thread_id; i < kNumLocalExperts; i += num_threads) {
      packed_recv_count[i] = 0;
    }
  }
  cg::this_grid().sync();

  // Issue count sends
  if (sm_id < kNumRdmaRanks) {
    int dst_rdma_rank = sm_id;
    const auto num_tokens_sent =
        atomic_finished_counter_per_rdma[dst_rdma_rank];

    if (thread_id < kNumQPs) {
      auto dst_ptr = reinterpret_cast<uint64_t>(
          rdma_recv_count + rdma_rank * kNumQPs + thread_id);

      bool is_local_copy = dst_rdma_rank == rdma_rank;
      if (is_local_copy) {  // local copy
        st_na_release(rdma_recv_count + rdma_rank * kNumQPs + thread_id,
                      -num_tokens_sent - 1);
      } else {
        nvshmemi_ibgda_amo_nonfetch_add(
            reinterpret_cast<int*>(dst_ptr),
            -num_tokens_sent - 1,
            dst_rdma_rank * NUM_MAX_NVL_PEERS + nvl_rank,
            thread_id);
      }
    }
    __syncthreads();
    // clean
    if (thread_id == 0) {
      atomic_counter_per_rdma[dst_rdma_rank] = 0;
      atomic_finished_counter_per_rdma[dst_rdma_rank] = 0;
    }
  }
  if (sm_id == num_sms - 1) {
    for (int i = thread_id; i < kNumExperts; i += num_threads) {
      atomic_counter_per_expert[i] = 0;
    }
  }

LOW_LATENCY_DISPATCH_RECV:
  if ((phases & LOW_LATENCY_RECV_PHASE) == 0) return;

  // TODO(ZKK): only wait one rank complete, is need to wait all rank complete
  if (rank >= a_start_rank && rank < a_start_rank + a_num_ranks) {
    int e_num_rdma_rank = e_num_ranks / NUM_MAX_NVL_PEERS;
    int e_start_rdma_rank = e_start_rank / NUM_MAX_NVL_PEERS;

    // ==========
    const int sms_per_rdma = num_sms / kNumRdmaRanks;
    const int src_rdma_rank = sm_id / sms_per_rdma;
    if (src_rdma_rank < kNumRdmaRanks) {
      const int sub_rdma_rank = sm_id % sms_per_rdma;
      if (thread_id < kNumQPs) {
        if (thread_id == 0) {
          sub_rdma_rank == 0 ? packed_rdma_recv_count[src_rdma_rank] = -1 : 0;
        }
      }
    }

    // ========
    if (thread_id < kNumExperts && sm_id == 0) {
      const auto src_rank = thread_id / kNumLocalExperts;
      const auto local_expert_idx = thread_id % kNumLocalExperts;
      const auto recv_range =
          packed_recv_layout_range + local_expert_idx * kNumRanks;
      recv_range[src_rank] = pack2<int, int64_t>(0, 0);
    }

    if (sm_id < e_num_rdma_rank && thread_id < NUM_MAX_NVL_PEERS) {
      int src_rdma_rank = sm_id + e_start_rdma_rank;
      auto lsl_flag_before = ld_acquire_sys_global(
          rdma_recv_complete + src_rdma_rank * NUM_MAX_NVL_PEERS + thread_id);
      if (M2N_LL_DEBUG) {
        if (thread_id == 0) {
          printf(
              "[kernel][dispatch][wait] src_rdma_rank: %d, offset: %d, "
              "flag_before: %d\n",
              src_rdma_rank * NUM_MAX_NVL_PEERS + thread_id,
              src_rdma_rank * NUM_MAX_NVL_PEERS + thread_id,
              lsl_flag_before);
        }
      }

      auto start_time = clock64();
      auto wait_recv_cost = clock64();
      while ((ld_acquire_sys_global(rdma_recv_complete +
                                    src_rdma_rank * NUM_MAX_NVL_PEERS +
                                    thread_id)) == 0) {
        // debug info of dispatch wait
        if (M2N_LL_HANG_DEBUG) {
          wait_recv_cost = clock64() - start_time;
          if (wait_recv_cost > M2N_NUM_HANG_CYCLES) {
            if (thread_id == 0) {
              printf(
                  "[kernel][dispatch][wait] wait than clock cycles: %ld, "
                  "flags: ",
                  wait_recv_cost);
              for (int i = 0; i < a_num_ranks + e_num_ranks; i++) {
                auto lsl_flag_debug = ld_acquire_sys_global(
                    rdma_recv_complete + src_rdma_rank * NUM_MAX_NVL_PEERS + i);
                printf("%d, ", lsl_flag_debug);
              }
              printf("\n");
              start_time = clock64();
            }
            // break;
          }
        }
      }
      auto lsl_flag = ld_acquire_sys_global(
          rdma_recv_complete + src_rdma_rank * NUM_MAX_NVL_PEERS + thread_id);

      rdma_recv_complete[src_rdma_rank * NUM_MAX_NVL_PEERS + thread_id] = 0;
      if (M2N_LL_DEBUG) {
        if (thread_id == 0) {
          printf(
              "[kernel][dispatch][wait][complete] src_rdma_rank: %d, flag: "
              "%d\n",
              src_rdma_rank * NUM_MAX_NVL_PEERS + thread_id,
              lsl_flag);
        }
      }
    }
    return;
  }

  // below code are only executed by MoE machine!

  /* RDMA Receiver and NVL Sender */
  // we should guarantee data in rdma_recv_x are valid in MoE machine, by while
  // checking rdma_recv_count! and then do NVL send! rdma_recv_x's shape is
  // [kNumRdmaRanks, num_max_dispatch_tokens_per_rank] in unit of
  // num_bytes_per_msg! rdma_recv_count's shape is [kNumRdmaRanks, kNumQPs]

  {
    const int sms_per_rdma = num_sms / kNumRdmaRanks;
    const int src_rdma_rank = sm_id / sms_per_rdma;

    // atomic_recv_tokens_per_rdma_expert's shape is
    // [kNumRdmaRanks，kNumRdmaExperts] Now,
    // atomic_recv_tokens_per_rdma_expert's shape is [kNumRdmaExperts]!
    atomic_recv_tokens_per_rdma_expert =
        atomic_recv_tokens_per_rdma_expert + src_rdma_rank * kNumRdmaExperts;

    if (src_rdma_rank < kNumRdmaRanks) {
      const int sub_sm_id = sm_id % sms_per_rdma;
      const int src_rank = src_rdma_rank * NUM_MAX_NVL_PEERS + nvl_rank;

      const int rmda_offset =
          src_rdma_rank * num_max_dispatch_tokens_per_rank * num_bytes_per_msg;
      const auto rdma_recv_x_uint8 =
          reinterpret_cast<uint8_t*>(rdma_recv_x) + rmda_offset;
      const auto packed_rdma_recv_x_uint8 =
          reinterpret_cast<uint8_t*>(packed_rdma_recv_x) + rmda_offset;

      __shared__ int shared_num_recv_tokens[1];
      int num_recv_tokens_per_rdma = -1;
      if (thread_id < kNumQPs) {
        // only read flag of attn machine, if one machine is fast and one
        // machine is slow, this will have hang in the last micro batch
        if (src_rdma_rank >= a_start_rdma_rank &&
            src_rdma_rank < a_start_rdma_rank + a_num_rdma_ranks) {
          auto start_time = clock64();
          auto wait_recv_cost = clock64();
          while ((num_recv_tokens_per_rdma = ld_acquire_sys_global(
                      rdma_recv_count + src_rdma_rank * kNumQPs + thread_id)) ==
                 0) {
            if (M2N_LL_HANG_DEBUG) {
              if (thread_id == 0) {
                wait_recv_cost = clock64() - start_time;
                if (wait_recv_cost > M2N_NUM_HANG_CYCLES) {
                  printf(
                      "[kernel][dispatch][rdma_recv_count] wait than clock "
                      "cycles: %ld\n",
                      wait_recv_cost);
                  start_time = clock64();
                }
              }
            }
          }
        }

        if (thread_id == 0) {
          sub_sm_id == 0
              ? packed_rdma_recv_count[src_rdma_rank] = num_recv_tokens_per_rdma
              : 0;
          shared_num_recv_tokens[0] = -num_recv_tokens_per_rdma - 1;
        }
      }
      __syncthreads();
      num_recv_tokens_per_rdma = shared_num_recv_tokens[0];

      // data is valid, begin to send these tokens through nvlink!
      // remember these tokens are from src_rdma_rank!
      for (int rdma_recv_token_idx = sub_sm_id;
           rdma_recv_token_idx < num_recv_tokens_per_rdma;
           rdma_recv_token_idx += sms_per_rdma) {
        const int token_offset = rdma_recv_token_idx * num_bytes_per_msg;
        const auto rdma_recv_x_uint8_now = rdma_recv_x_uint8 + token_offset;
        const auto packed_rdma_recv_x_uint8_now =
            packed_rdma_recv_x_uint8 + token_offset;

        const auto src_data = reinterpret_cast<int4*>(rdma_recv_x_uint8_now);
        const auto rdma_recv_x_scales = reinterpret_cast<float*>(
            reinterpret_cast<uint8_t*>(src_data) + sizeof(int4) + hidden_bytes);
        const auto rdma_recv_nvl_rank_meta = reinterpret_cast<int*>(
            rdma_recv_x_scales + (kUseFP8 ? kNumScales : 0));

        // here must be rdma_rank!
        const int dst_nvl_experts =
            *(rdma_recv_nvl_rank_meta + rdma_rank * (kTopk * 3 + 1));
        const auto rdma_recv_nvl_rank_meta_now =
            rdma_recv_nvl_rank_meta + rdma_rank * (kTopk * 3 + 1) + 1;

        // Used in combine
        if (warp_id == num_warps - 1) {
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

        // nvl sender
        // we need send dst_nvl_experts times for this rdma_recv_token_idx token
        // using one sm!
        for (int loop_nvl_expert_i = warp_id;
             loop_nvl_expert_i < dst_nvl_experts;
             loop_nvl_expert_i += num_warps) {
          const int rdma_local_expert_idx =
              rdma_recv_nvl_rank_meta_now[loop_nvl_expert_i * 3];
          const int dst_nvl_rank = rdma_local_expert_idx / kNumLocalExperts;
          const int dst_nvl_local_expert =
              rdma_local_expert_idx % kNumLocalExperts;

          const int rdma_local_expert_cumsum_index =
              rdma_recv_nvl_rank_meta_now[loop_nvl_expert_i * 3 + 1];

          // write to nvl_recv_x[dst_nvl_rank]
          // whose‘s shape is [kNumLocalExperts, kNumRanks,
          // num_max_dispatch_tokens_per_rank] in unit of
          // num_int4_per_msg_rdma_revecier_and_nvl_sender! kNumRanks means for
          // each expert we need to know which rank this data is from!
          const auto dst_data =
              reinterpret_cast<int4*>(nvl_recv_x[dst_nvl_rank]) +
              ((dst_nvl_local_expert * kNumRanks + src_rank) *
                   num_max_dispatch_tokens_per_rank +
               rdma_local_expert_cumsum_index) *
                  num_int4_per_msg_rdma_revecier_and_nvl_sender;

          if (lane_id == 0) {
            st_na_global(reinterpret_cast<int*>(dst_data),
                         rdma_local_expert_cumsum_index);
          }

          UNROLLED_WARP_COPY(UNROLL_FACTOR,
                             lane_id,
                             num_int4_per_msg_rdma_to_nvl,
                             dst_data + 1,
                             src_data + 1,
                             ld_nc_global,
                             st_na_global);
          __syncwarp();
          // we need record how many tokens are sent to different experts in
          // this machine!
          lane_id == 0
              ? (atomic_add_release_global(
                    atomic_recv_tokens_per_rdma_expert + rdma_local_expert_idx,
                    1))
              : 0;
        }
      }
      __syncthreads();
      thread_id == 0 ? (atomic_add_release_global(
                           atomic_nvl_sender_multi_sms + src_rdma_rank, 1))
                     : 0;
      if (sub_sm_id == 0 && thread_id == 0) {
        auto start_time = clock64();
        auto wait_recv_cost = clock64();
        while (ld_acquire_global(atomic_nvl_sender_multi_sms + src_rdma_rank) !=
               sms_per_rdma) {
          if (M2N_LL_HANG_DEBUG) {
            if (thread_id == 0) {
              wait_recv_cost = clock64() - start_time;
              if (wait_recv_cost > M2N_NUM_HANG_CYCLES) {
                printf(
                    "[kernel][dispatch][atomic_nvl_sender_multi_sms] wait than "
                    "clock cycles: %ld\n",
                    wait_recv_cost);
                start_time = clock64();
              }
            }
          }
        }
        atomic_nvl_sender_multi_sms[src_rdma_rank] = 0;
      }
      __syncthreads();
      if (sub_sm_id == 0) {
        // need tell nvl receive how many tokens we have send from src_rdma_rank
        // machine!
        for (int dst_rdma_local_expert_idx = thread_id;
             dst_rdma_local_expert_idx < NUM_MAX_NVL_PEERS * kNumLocalExperts;
             dst_rdma_local_expert_idx += num_threads) {
          const int dst_nvl_rank = dst_rdma_local_expert_idx / kNumLocalExperts;
          const int dst_nvl_local_expert =
              dst_rdma_local_expert_idx % kNumLocalExperts;

          st_release_sys_global(
              reinterpret_cast<int*>(
                  reinterpret_cast<uint8_t*>(nvl_recv_x[dst_nvl_rank]) +
                  NVL_BUFFER_X_BYTES) +
                  dst_nvl_local_expert * kNumRanks + src_rank,
              -ld_acquire_global(atomic_recv_tokens_per_rdma_expert +
                                 dst_rdma_local_expert_idx) -
                  1);
          // reset
          *(atomic_recv_tokens_per_rdma_expert + dst_rdma_local_expert_idx) = 0;
        }
        for (int reset_i = thread_id; reset_i < kNumQPs;
             reset_i += num_threads) {
          rdma_recv_count[src_rdma_rank * kNumQPs + reset_i] = 0;
        }
      }
    }
  }

  /* NVL Receiver */
  if (responsible_expert_idx < kNumExperts) {
    const auto src_rank = responsible_expert_idx / kNumLocalExperts;
    const auto local_expert_idx = responsible_expert_idx % kNumLocalExperts;
    // local_expert_idx receiveom src_rank!
    const int recv_offset_this_warpgroup =
        local_expert_idx * kNumRanks + src_rank;

    const auto nvl_recv_x_uint8 =
        reinterpret_cast<uint8_t*>(nvl_recv_x[nvl_rank]) +
        recv_offset_this_warpgroup * num_max_dispatch_tokens_per_rank *
            num_bytes_per_msg_rdma_revecier_and_nvl_sender;
    const auto recv_x_int4 = reinterpret_cast<int4*>(packed_recv_x) +
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
      auto start_time = clock64();
      auto wait_recv_cost = clock64();
      while ((num_recv_tokens = ld_acquire_sys_global(
                  reinterpret_cast<int*>(
                      reinterpret_cast<uint8_t*>(nvl_recv_x[nvl_rank]) +
                      NVL_BUFFER_X_BYTES) +
                  recv_offset_this_warpgroup)) == 0) {
        if (M2N_LL_HANG_DEBUG) {
          if (thread_id == 0) {
            wait_recv_cost = clock64() - start_time;
            if (wait_recv_cost > M2N_NUM_HANG_CYCLES) {
              printf(
                  "[kernel][dispatch][nvl_recv_x] wait than clock cycles: "
                  "%ld\n",
                  wait_recv_cost);
              start_time = clock64();
            }
          }
        }
      }
      num_recv_tokens = -num_recv_tokens - 1;
      recv_token_begin_idx =
          atomicAdd(packed_recv_count + local_expert_idx, num_recv_tokens);
      shared_num_recv_tokens[warp_group_id] = num_recv_tokens;
      shared_recv_token_begin_idx[warp_group_id] = recv_token_begin_idx;
      recv_range[src_rank] =
          pack2<int, int64_t>(num_recv_tokens, recv_token_begin_idx);
      // reset nvl_recv_token_num
      *(reinterpret_cast<int*>(
            reinterpret_cast<uint8_t*>(nvl_recv_x[nvl_rank]) +
            NVL_BUFFER_X_BYTES) +
        recv_offset_this_warpgroup) = 0;
    }
    asm volatile("bar.sync %0, %1;" ::"r"(warp_group_id + 2),
                 "r"(kNumWarpsPerGroup * 32));
    num_recv_tokens = shared_num_recv_tokens[warp_group_id];
    recv_token_begin_idx = shared_recv_token_begin_idx[warp_group_id];

    // Copy tokens
    EP_DEVICE_ASSERT(kNumScales <= 64);
    for (int i = sub_warp_id; i < num_recv_tokens; i += kNumWarpsPerGroup) {
      // Copy source info
      const auto src_src_idx = reinterpret_cast<int*>(
          nvl_recv_x_uint8 +
          i * num_bytes_per_msg_rdma_revecier_and_nvl_sender);
      if (lane_id == 0)
        recv_src_info[recv_token_begin_idx + i] = ld_nc_global(src_src_idx);
      __syncwarp();

      // Copy data
      const auto src_data = reinterpret_cast<int4*>(
          reinterpret_cast<uint8_t*>(src_src_idx) + sizeof(int4));
      const auto dst_data =
          recv_x_int4 + (recv_token_begin_idx + i) * hidden_int4;
      UNROLLED_WARP_COPY(UNROLL_FACTOR,
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
        const auto scale_stride = kNumRanks * num_max_dispatch_tokens_per_rank;
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

  // 这里为啥需要加上这个？
  // 加上吧，放置出错啦！
  cg::this_grid().sync();

  // TODO(ZKK): Stuff.
  if (rank >= e_start_rank && rank < e_start_rank + e_num_ranks) {
    if (sm_id < a_num_rdma_ranks && thread_id < NUM_MAX_NVL_PEERS) {
      int dst_rdma_rank = sm_id + a_start_rdma_rank;
      auto dst_ptr = reinterpret_cast<uint64_t>(
          rdma_recv_complete + rdma_rank * NUM_MAX_NVL_PEERS + nvl_rank);

      nvshmemi_ibgda_amo_nonfetch_add(
          reinterpret_cast<int*>(dst_ptr),
          1,
          dst_rdma_rank * NUM_MAX_NVL_PEERS + thread_id,
          thread_id);
      if (M2N_LL_DEBUG) {
        if (thread_id == 0) {
          printf("[kernel][dispatch][complete] dst_rank: %d, offset: %d\n",
                 dst_rdma_rank * NUM_MAX_NVL_PEERS + thread_id,
                 rdma_rank * NUM_MAX_NVL_PEERS + nvl_rank);
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
              int* rdma_recv_complete,
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
              int a_start_rank,
              int a_num_ranks,
              int e_start_rank,
              int e_num_ranks,
              bool use_fp8,
              void* workspace,
              cudaStream_t stream,
              int phases) {
  constexpr int kNumMaxTopK = 8;
  constexpr int kNumQPs = 32;
  constexpr int NUM_WARPS = 32;

  const int dev_id = 0;
  int sm_count;
  cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, dev_id);
  sm_count = 24;
  int num_warp_groups = cell_div(num_experts, sm_count);
  num_warp_groups =
      (num_warp_groups % 2 == 1) ? num_warp_groups + 1 : num_warp_groups;
  const auto num_sms = max(sm_count, cell_div(num_experts, num_warp_groups));
  // const auto num_sms = 24;
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
  auto atomic_counter_per_qp =
      atomic_nvl_sender_multi_sms + num_rdma_ranks;  // num_rdma_ranks * kNumQPs
  EP_HOST_ASSERT((num_experts + num_rdma_ranks * 3 + num_rdma_experts +
                  num_rdma_ranks * kNumQPs) *
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
                  {DISPATCH_NUM_WARP_GROUPS(num_warp_groups, kNumWarpGroups, {
                    constexpr int kNumWarpsPerGroup =
                        NUM_WARPS / kNumWarpGroups;
                    assert(num_rdma_ranks <=
                           kNumWarpGroups * kNumWarpsPerGroup);
                    EP_STATIC_ASSERT(
                        kNumMaxTopK + 1 <= kNumWarpGroups * kNumWarpsPerGroup,
                        "Too many top-k selections");
                    auto dispatch_func =
                        use_fp8 ? dispatch_kernel<true,
                                                  kNumWarpGroups,
                                                  kNumWarpsPerGroup,
                                                  kHidden,
                                                  kNumRdmaRanks,
                                                  kNumExperts,
                                                  kTopk,
                                                  kNumQPs>
                                : dispatch_kernel<false,
                                                  kNumWarpGroups,
                                                  kNumWarpsPerGroup,
                                                  kHidden,
                                                  kNumRdmaRanks,
                                                  kNumExperts,
                                                  kTopk,
                                                  kNumQPs>;
                    SETUP_LAUNCH_CONFIG(num_sms,
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
                                  rdma_recv_complete,
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
                                  atomic_counter_per_qp,
                                  num_tokens,
                                  num_max_dispatch_tokens_per_rank,
                                  rank,
                                  a_start_rank,
                                  a_num_ranks,
                                  e_start_rank,
                                  e_num_ranks,
                                  phases);
                  })})})})});
}

template <int kNumWarpGroups,
          int kNumWarpsPerGroup,
          int kHidden,
          int kNumRdmaRanks,
          int kNumExperts,
          int kTopk,
          bool kDispatchUseFP8,
          int kNumQPs>
__global__ __launch_bounds__(
    kNumWarpGroups* kNumWarpsPerGroup * 32,
    1) void combine_kernel(void* combined_x,
                           void* rdma_recv_x,
                           int* rdma_recv_flag,
                           void* rdma_send_x,
                           int* rdma_recv_complete,
                           void* dispatch_rdma_recv_x,
                           const int* dispatch_rdma_recv_count,
                           void** nvl_recv_buffer,
                           const void* x,
                           const int64_t* topk_idx,
                           const float* topk_weights,
                           const int* src_info,
                           const int64_t* layout_range,
                           const bool* rdma_send_flags,
                           int* atomic_clean_flag,
                           int* atomic_nvl_sender_multi_sms,
                           int num_combined_tokens,
                           int hidden,
                           int num_topk,
                           int num_max_dispatch_tokens_per_rank,
                           int num_experts,
                           int rank,
                           int num_ranks,
                           int a_start_rank,
                           int a_num_ranks,
                           int e_start_rank,
                           int e_num_ranks,
                           int phases) {
  constexpr int UNROLL_FACTOR = kHidden / 1024;
  constexpr int kNumRanks = kNumRdmaRanks * NUM_MAX_NVL_PEERS;
  constexpr int kNumLocalExperts = kNumExperts / kNumRanks;
  constexpr int kNumRdmaExperts = kNumLocalExperts * NUM_MAX_NVL_PEERS;
  constexpr int kNumPerChannels = 128;
  constexpr int kNumScales = kHidden / kNumPerChannels;

  const size_t num_bytes_per_msg_dispatch =
      sizeof(int4) +
      (kNumRdmaRanks * (kTopk * 3 + 1) * sizeof(int) + sizeof(int4) - 1) /
          sizeof(int4) * sizeof(int4) +
      (kDispatchUseFP8 ? (kHidden + kNumScales * sizeof(float))
                       : (kHidden * sizeof(nv_bfloat16)));
  const size_t num_bytes_per_msg_rdma_revecier_and_nvl_sender_dispatch =
      sizeof(int4) + (kDispatchUseFP8 ? (kHidden + kNumScales * sizeof(float))
                                      : (kHidden * sizeof(nv_bfloat16)));

  const size_t dispatch_hidden_bytes =
      kHidden *
      (kDispatchUseFP8 ? sizeof(__nv_fp8_storage_t) : sizeof(nv_bfloat16));
  const size_t combine_hidden_bytes = kHidden * sizeof(nv_bfloat16);
  const size_t combine_hidden_int4_num = combine_hidden_bytes / sizeof(int4);

  const auto sm_id = static_cast<int>(blockIdx.x);
  const auto num_sms = static_cast<int>(gridDim.x);
  const auto thread_id = static_cast<int>(threadIdx.x);
  const auto num_threads = static_cast<int>(blockDim.x),
             num_warps = num_threads / 32;
  const auto warp_id = thread_id / 32, lane_id = get_lane_id();
  const auto num_local_experts = num_experts / num_ranks;
  const auto warp_group_id = warp_id / kNumWarpsPerGroup;
  const auto sub_warp_id = warp_id % kNumWarpsPerGroup;
  const auto responsible_expert_idx = sm_id * kNumWarpGroups + warp_group_id;
  int a_start_rdma_rank = a_start_rank / NUM_MAX_NVL_PEERS;
  int a_num_rdma_ranks = a_num_ranks / NUM_MAX_NVL_PEERS;
  int e_start_rdma_rank = e_start_rank / NUM_MAX_NVL_PEERS;
  int e_num_rdma_ranks = e_num_ranks / NUM_MAX_NVL_PEERS;

  if (sm_id == 0 && thread_id == 0) {
    EP_DEVICE_ASSERT(ibgda_get_state()->num_rc_per_pe >= kNumQPs);
  }

  const auto rdma_rank = rank / NUM_MAX_NVL_PEERS,
             nvl_rank = rank % NUM_MAX_NVL_PEERS;

  constexpr int kNumElemsPerInt4 = sizeof(int4) / sizeof(nv_bfloat16);
  const size_t hidden_bf16_int4 = kHidden / kNumElemsPerInt4;
  if (sm_id == 0 && thread_id == 0) {
    EP_DEVICE_ASSERT(ibgda_get_state()->num_rc_per_pe >= kNumQPs);
    // EP_DEVICE_ASSERT(num_threads >= hidden_bf16_int4); // TODO: lzy why
  }

  constexpr size_t num_bytes_per_slot = kHidden * sizeof(nv_bfloat16);
  const size_t DISPATCH_NVL_BUFFER_X_BYTES =
      kNumLocalExperts * kNumRanks * num_max_dispatch_tokens_per_rank *
          num_bytes_per_msg_rdma_revecier_and_nvl_sender_dispatch +
      kNumExperts * sizeof(int);
  const size_t COMBINE_NVL_BUFFER_X_BYTES = kNumRdmaExperts * kNumRdmaRanks *
                                            num_max_dispatch_tokens_per_rank *
                                            num_bytes_per_slot;
  const size_t NVL_BUFFER_X_BYTES =
      DISPATCH_NVL_BUFFER_X_BYTES + COMBINE_NVL_BUFFER_X_BYTES;

  if ((phases & LOW_LATENCY_SEND_PHASE) == 0) goto LOW_LATENCY_COMBINE_RECV;

  if (M2N_LL_ACC_DEBUG) {
    if (sm_id == 0 && thread_id == 0) {
      if (responsible_expert_idx < num_experts) {
        const auto dst_rank = responsible_expert_idx / num_local_experts;
        const auto dst_rdma_rank = dst_rank / NUM_MAX_NVL_PEERS;
        const auto dst_nvl_rank = dst_rank % NUM_MAX_NVL_PEERS;
        auto tmp = reinterpret_cast<int*>(nvl_recv_buffer[dst_nvl_rank] +
                                          NVL_BUFFER_X_BYTES);
        printf("nvl flag: ");
        for (int i = 0; i < num_local_experts * num_ranks; i++) {
          printf("%d, ", tmp[i]);
        }
        printf("\n");
      }
    }
  }

  /* NVL Sender */
  if (responsible_expert_idx < num_experts) {
    // we will send local_expert_idx partial result to dst_rank!
    // first
    // we need issue them to dst_nvl_rank through nvlink!
    // then rdma to dst_rdma_rank / dst_rank!

    const auto dst_rank = responsible_expert_idx / num_local_experts;
    const auto dst_rdma_rank = dst_rank / NUM_MAX_NVL_PEERS;
    const auto dst_nvl_rank = dst_rank % NUM_MAX_NVL_PEERS;
    const auto local_expert_idx = responsible_expert_idx % num_local_experts;
    // global_rdma_expert_idx means expert_ids in range of one machine!
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

    // Attention 卡上当然要是鸡蛋啦！
    // if (rank >= 0 && rank < 16) EP_DEVICE_ASSERT(num_tokens_to_send == 0);

    for (int token_idx = sub_warp_id; token_idx < num_tokens_to_send;
         token_idx += kNumWarpsPerGroup) {
      const int idx_now = token_idx + offset;
      const int* src_idxs = local_src_info + idx_now;
      const int dst_rdma_index = src_idxs[0];
      // nvl recv buffer
      const auto dst_ptr = reinterpret_cast<int4*>(
          reinterpret_cast<uint8_t*>(nvl_recv_buffer[dst_nvl_rank]) +
          DISPATCH_NVL_BUFFER_X_BYTES +
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
      auto dst_ptr = reinterpret_cast<int*>(reinterpret_cast<uint8_t*>(
                                                nvl_recv_buffer[dst_nvl_rank]) +
                                            NVL_BUFFER_X_BYTES) +
                     global_rdma_expert_idx * kNumRdmaRanks + dst_rdma_rank;
      st_release_sys_global(dst_ptr, 1);
    }
    __syncwarp();
  }

  // Wait all nvl ranks to arrive
  if (responsible_expert_idx < num_experts) {
    EP_STATIC_ASSERT(kNumWarpsPerGroup > 1,
                     "Invalid number of warps per group");
    if (rdma_rank >= e_start_rdma_rank &&
        rdma_rank < e_start_rdma_rank + e_num_rdma_ranks && sub_warp_id == 0 &&
        lane_id == 0) {
      // if (sub_warp_id == 0 && lane_id == 0) {
      auto start_time = clock64();
      auto wait_recv_cost = clock64();
      while (ld_acquire_sys_global(
                 reinterpret_cast<int*>(
                     reinterpret_cast<uint8_t*>(nvl_recv_buffer[nvl_rank]) +
                     NVL_BUFFER_X_BYTES) +
                 responsible_expert_idx) == 0) {
        if (M2N_LL_HANG_DEBUG) {
          if (thread_id == 0) {
            wait_recv_cost = clock64() - start_time;
            if (wait_recv_cost > M2N_NUM_HANG_CYCLES) {
              printf(
                  "[kernel][combine][nvl_recv_buffer] wait than clock cycles: "
                  "%ld\n",
                  wait_recv_cost);
              start_time = clock64();
            }
          }
        }
      }
      // reset nvl_recv_buffer
      *(reinterpret_cast<int*>(
            reinterpret_cast<uint8_t*>(nvl_recv_buffer[nvl_rank]) +
            NVL_BUFFER_X_BYTES) +
        responsible_expert_idx) = 0;
    }
  }
  cg::this_grid().sync();

  /* NVL Receiver / NVL Reducer */
  {
    // receive data from nvlink and do reduce!
    // then issue the result !
    const int sms_per_rdma = num_sms / kNumRdmaRanks;
    const int deal_rdma_rank = sm_id / sms_per_rdma;
    if (deal_rdma_rank < kNumRdmaRanks) {
      const int sub_deal_rdma_rank = sm_id % sms_per_rdma;
      const int qp_id = sub_deal_rdma_rank % kNumQPs;
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
      for (int rdma_recv_token_idx = sub_deal_rdma_rank;
           rdma_recv_token_idx < num_tokens_to_deal;
           rdma_recv_token_idx += sms_per_rdma) {
        const auto dispatch_rdma_recv_x_now =
            dispatch_rdma_recv_x_this_rdma_rank +
            rdma_recv_token_idx * num_bytes_per_msg_dispatch;
        const auto index_source =
            reinterpret_cast<const int*>(dispatch_rdma_recv_x_now)[0];
        const int* nvl_rank_meta = reinterpret_cast<const int*>(
            dispatch_rdma_recv_x_now + sizeof(int4) + dispatch_hidden_bytes +
            (kDispatchUseFP8 ? kNumScales * sizeof(float) : 0));
        const int nvl_rank_nums =
            *(nvl_rank_meta + rdma_rank * (kTopk * 3 + 1));
        const int* nvl_rank_meta_now =
            nvl_rank_meta + rdma_rank * (kTopk * 3 + 1) + 1;
        int4* dst_ptr = reinterpret_cast<int4*>(
            rdma_send_x_this_rdma_rank + index_source * combine_hidden_bytes);
        float combined_values[kNumElemsPerInt4] = {0.0f};
        for (int g_id = thread_id; g_id < hidden_bf16_int4;
             g_id += num_threads) {
          for (int nvl_rank_idx = 0; nvl_rank_idx < nvl_rank_nums;
               nvl_rank_idx += 1) {
            const int dst_rdma_expert_idx = nvl_rank_meta_now[nvl_rank_idx * 3];
            const int dst_cum_index = nvl_rank_meta_now[nvl_rank_idx * 3 + 1];
            const float topk_weight = reinterpret_cast<const float*>(
                nvl_rank_meta_now)[nvl_rank_idx * 3 + 2];
            const int4* src_ptr = reinterpret_cast<int4*>(
                reinterpret_cast<uint8_t*>(nvl_recv_buffer[nvl_rank]) +
                DISPATCH_NVL_BUFFER_X_BYTES +
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
            if constexpr (kNumQPs > 1) {
              nvshmemi_ibgda_put_nbi_warp<true>(
                  dst_ptr,
                  src_ptr,
                  combine_hidden_bytes,
                  deal_rdma_rank * NUM_MAX_NVL_PEERS + nvl_rank,
                  qp_id,
                  lane_id,
                  0);
            } else {
              nvshmemi_ibgda_put_nbi_warp(
                  dst_ptr,
                  src_ptr,
                  combine_hidden_bytes,
                  deal_rdma_rank * NUM_MAX_NVL_PEERS + nvl_rank,
                  qp_id,
                  lane_id,
                  rdma_recv_token_idx);
            }
          }
          __syncwarp();
        }
      }
      thread_id == 0 ? (atomic_add_release_global(
                           atomic_nvl_sender_multi_sms + deal_rdma_rank, 1))
                     : 0;
      // all sms reduce done
      if (sub_deal_rdma_rank == 0 && thread_id == 0) {
        auto start_time = clock64();
        auto wait_recv_cost = clock64();
        while (ld_acquire_global(atomic_nvl_sender_multi_sms +
                                 deal_rdma_rank) != sms_per_rdma) {
          if (M2N_LL_HANG_DEBUG) {
            if (thread_id == 0) {
              wait_recv_cost = clock64() - start_time;
              if (wait_recv_cost > M2N_NUM_HANG_CYCLES) {
                printf(
                    "[kernel][combine][atomic_nvl_sender_multi_sms] wait than "
                    "clock cycles: %ld\n",
                    wait_recv_cost);
                start_time = clock64();
              }
            }
          }
        }
        atomic_nvl_sender_multi_sms[deal_rdma_rank] = 0;
      }
      __syncthreads();
      // set flag
      if (sub_deal_rdma_rank == 0 && thread_id < kNumQPs) {
        // notify remote rdma
        auto dst_rdma_flag = reinterpret_cast<uint64_t>(
            rdma_recv_flag + rdma_rank * kNumQPs + thread_id);
        bool is_local_copy = deal_rdma_rank == rdma_rank;
        if (is_local_copy) {
          st_na_release(rdma_recv_flag + rdma_rank * kNumQPs + thread_id, 1);
        } else {
          nvshmemi_ibgda_amo_nonfetch_add(
              reinterpret_cast<int*>(dst_rdma_flag),
              1,
              deal_rdma_rank * NUM_MAX_NVL_PEERS + nvl_rank,
              qp_id);
        }
      }
    }
  }

LOW_LATENCY_COMBINE_RECV:
  if ((phases & LOW_LATENCY_RECV_PHASE) == 0) return;

  // TODO(ZKK): stuff.
  if (rank >= e_start_rank && rank < e_start_rank + e_num_ranks) {
    if (sm_id < a_num_rdma_ranks && thread_id < NUM_MAX_NVL_PEERS) {
      int src_rdma_rank = sm_id + a_start_rdma_rank;
      auto lsl_flag_before =
          ld_acquire_sys_global(rdma_recv_complete + num_ranks +
                                src_rdma_rank * NUM_MAX_NVL_PEERS + thread_id);
      if (M2N_LL_DEBUG) {
        if (thread_id == 0) {
          printf(
              "[kernel][combine][wait] src_rdma_rank: %d, offset: %d, "
              "flag_before: %d\n",
              src_rdma_rank * NUM_MAX_NVL_PEERS + thread_id,
              num_ranks + src_rdma_rank * NUM_MAX_NVL_PEERS + thread_id,
              lsl_flag_before);
        }
      }
      auto start_time = clock64();
      auto wait_recv_cost = clock64();
      while ((ld_acquire_sys_global(rdma_recv_complete + num_ranks +
                                    src_rdma_rank * NUM_MAX_NVL_PEERS +
                                    thread_id)) == 0) {
        if (M2N_LL_HANG_DEBUG) {
          if (thread_id == 0) {
            wait_recv_cost = clock64() - start_time;
            if (wait_recv_cost > M2N_NUM_HANG_CYCLES) {
              printf("[kernel][combine][wait] wait than clock cycles: %ld\n",
                     wait_recv_cost);
              start_time = clock64();
            }
          }
        }
      }
      auto lsl_flag =
          ld_acquire_sys_global(rdma_recv_complete + num_ranks +
                                src_rdma_rank * NUM_MAX_NVL_PEERS + thread_id);

      rdma_recv_complete[num_ranks + src_rdma_rank * NUM_MAX_NVL_PEERS +
                         thread_id] = 0;
      if (M2N_LL_DEBUG) {
        if (thread_id == 0) {
          printf(
              "[kernel][combine][wait][complete] src_rdma_rank: %d, flag: %d\n",
              src_rdma_rank * NUM_MAX_NVL_PEERS + thread_id,
              lsl_flag);
        }
      }
    }
    return;
  }

  /* RDMA Receiver / RDMA Reducer */
  // Wait all rdma ranks to arrive
  // only read flag of experts machine, if one machine is fast and one machine
  // is slow, this will have hang in the last micro batch

  if (sm_id >= e_start_rdma_rank &&
      sm_id < e_start_rdma_rank + e_num_rdma_ranks && sm_id < kNumRdmaRanks) {
    if (thread_id < kNumQPs) {
      auto start_time = clock64();
      auto wait_recv_cost = clock64();
      while (ld_acquire_sys_global(rdma_recv_flag + sm_id * kNumQPs +
                                   thread_id) == 0) {
        if (M2N_LL_HANG_DEBUG) {
          if (thread_id == 0) {
            wait_recv_cost = clock64() - start_time;
            if (wait_recv_cost > M2N_NUM_HANG_CYCLES) {
              printf(
                  "[kernel][combine][rdma_recv_flag] wait than clock cycles: "
                  "%ld\n",
                  wait_recv_cost);
              start_time = clock64();
            }
          }
        }
      }
      // reset
      rdma_recv_flag[sm_id * kNumQPs + thread_id] = 0;
    }
  }

  cg::this_grid().sync();

  for (int g_id = thread_id; g_id < hidden_bf16_int4; g_id += num_threads) {
    for (int token_idx = sm_id; token_idx < num_combined_tokens;
         token_idx += num_sms) {
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

  //
  cg::this_grid().sync();

  // TODO(ZKK): stuff.
  if (rank >= a_start_rank && rank < a_start_rank + a_num_ranks) {
    // int e_num_rdma_ranks = e_num_ranks / NUM_MAX_NVL_PEERS;
    // int e_start_rdma_rank = e_start_rank / NUM_MAX_NVL_PEERS;
    // int a_start_rdma_rank = a_start_rank / NUM_MAX_NVL_PEERS;
    if (sm_id < e_num_rdma_ranks && thread_id < NUM_MAX_NVL_PEERS) {
      int dst_rdma_rank = sm_id + e_start_rdma_rank;
      auto dst_ptr =
          reinterpret_cast<uint64_t>(rdma_recv_complete + num_ranks +
                                     rdma_rank * NUM_MAX_NVL_PEERS + nvl_rank);

      nvshmemi_ibgda_amo_nonfetch_add(
          reinterpret_cast<int*>(dst_ptr),
          1,
          dst_rdma_rank * NUM_MAX_NVL_PEERS + thread_id,
          thread_id);
      if (M2N_LL_DEBUG) {
        if (thread_id == 0) {
          printf("[kernel][combine][complete] dst_rank: %d, offset: %d\n",
                 dst_rdma_rank * NUM_MAX_NVL_PEERS + thread_id,
                 num_ranks + rdma_rank * NUM_MAX_NVL_PEERS + nvl_rank);
        }
      }
    }
  }
}

void combine(void* combined_x,
             void* rdma_recv_x,
             int* rdma_recv_flag,
             void* rdma_send_x,
             int* rdma_recv_complete,
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
             int a_start_rank,
             int a_num_ranks,
             int e_start_rank,
             int e_num_ranks,
             void* workspace,
             cudaStream_t stream,
             int phases,
             bool dispatch_use_fp8) {
  constexpr int kNumMaxTopk = 8;
  constexpr int kNumQPs = 4;
  constexpr int NUM_WARPS = 32;

  const int dev_id = 0;
  int sm_count;
  cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, dev_id);
  sm_count = 24;
  int num_warp_groups = cell_div(num_experts, sm_count);
  num_warp_groups =
      (num_warp_groups % 2 == 1) ? num_warp_groups + 1 : num_warp_groups;
  const auto num_sms = max(sm_count, cell_div(num_experts, num_warp_groups));
  // const auto num_sms = 24;
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
                  {DISPATCH_NUM_WARP_GROUPS(num_warp_groups, kNumWarpGroups, {
                    constexpr int kNumWarpsPerGroup =
                        NUM_WARPS / kNumWarpGroups;
                    auto combine_func = dispatch_use_fp8
                                            ? combine_kernel<kNumWarpGroups,
                                                             kNumWarpsPerGroup,
                                                             kHidden,
                                                             kNumRdmaRanks,
                                                             kNumExperts,
                                                             kTopk,
                                                             true,
                                                             kNumQPs>
                                            : combine_kernel<kNumWarpGroups,
                                                             kNumWarpsPerGroup,
                                                             kHidden,
                                                             kNumRdmaRanks,
                                                             kNumExperts,
                                                             kTopk,
                                                             false,
                                                             kNumQPs>;
                    SETUP_LAUNCH_CONFIG(num_sms,
                                        kNumWarpGroups * kNumWarpsPerGroup * 32,
                                        stream);
                    LAUNCH_KERNEL(&cfg,
                                  combine_func,
                                  combined_x,
                                  rdma_recv_x,
                                  rdma_recv_flag,
                                  rdma_send_x,
                                  rdma_recv_complete,
                                  dispatch_rdma_recv_x,
                                  dispatch_rdma_recv_count,
                                  nvl_buffer,
                                  x,
                                  topk_idx,
                                  topk_weights,
                                  src_info,
                                  layout_range,
                                  rdma_send_flags,
                                  atomic_clean_flag,
                                  atomic_nvl_sender_multi_sms,
                                  num_combined_tokens,
                                  hidden,
                                  num_topk,
                                  num_max_dispatch_tokens_per_rank,
                                  num_experts,
                                  rank,
                                  num_ranks,
                                  a_start_rank,
                                  a_num_ranks,
                                  e_start_rank,
                                  e_num_ranks,
                                  phases);
                  })})})})})
}

}  // namespace m2n_ll_two_stage

}  // namespace deep_ep
