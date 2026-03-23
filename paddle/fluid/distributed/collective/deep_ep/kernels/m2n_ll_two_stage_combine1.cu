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

// m2n_ll_two_stage combine implementation for hidden sizes: 1536, 4096, 5120

#include "paddle/fluid/distributed/collective/deep_ep/kernels/m2n_ll_two_stage_impl.cuh"

namespace deep_ep {

namespace m2n_ll_two_stage {

void combine_hidden_group1(void* combined_x,
                           void* rdma_recv_x,
                           int* rdma_recv_flag,
                           void* rdma_send_x,
                           int* rdma_recv_complete,
                           void* dispatch_rdma_recv_x,
                           const int* dispatch_rdma_recv_count,
                           void** nvl_buffer,
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
                           int num_max_dispatch_tokens_per_rank,
                           int num_topk,
                           int num_experts,
                           int rank,
                           int num_ranks,
                           int a_start_rank,
                           int a_num_ranks,
                           int e_start_rank,
                           int e_num_ranks,
                           int num_sms,
                           int num_warp_groups,
                           cudaStream_t stream,
                           int phases,
                           bool dispatch_use_fp8) {
  constexpr int kNumQPs = 4;
  constexpr int NUM_WARPS = 32;
  const int num_rdma_ranks = num_ranks / NUM_MAX_NVL_PEERS;

#define COMBINE_KERNEL(kHidden)                                               \
  DISPATCH_NUM_TOPK(                                                          \
      num_topk,                                                               \
      kTopk,                                                                  \
      {DISPATCH_RDMA_RANKS(                                                   \
          num_rdma_ranks,                                                     \
          kNumRdmaRanks,                                                      \
          {DISPATCH_NUM_EXPERTS(                                              \
              num_experts,                                                    \
              kNumExperts,                                                    \
              {DISPATCH_NUM_WARP_GROUPS(num_warp_groups, kNumWarpGroups, {    \
                constexpr int kNumWarpsPerGroup = NUM_WARPS / kNumWarpGroups; \
                auto combine_func = dispatch_use_fp8                          \
                                        ? combine_kernel<kNumWarpGroups,      \
                                                         kNumWarpsPerGroup,   \
                                                         kHidden,             \
                                                         kNumRdmaRanks,       \
                                                         kNumExperts,         \
                                                         kTopk,               \
                                                         true,                \
                                                         kNumQPs>             \
                                        : combine_kernel<kNumWarpGroups,      \
                                                         kNumWarpsPerGroup,   \
                                                         kHidden,             \
                                                         kNumRdmaRanks,       \
                                                         kNumExperts,         \
                                                         kTopk,               \
                                                         false,               \
                                                         kNumQPs>;            \
                SETUP_LAUNCH_CONFIG(                                          \
                    num_sms, kNumWarpGroups* kNumWarpsPerGroup * 32, stream); \
                LAUNCH_KERNEL(&cfg,                                           \
                              combine_func,                                   \
                              combined_x,                                     \
                              rdma_recv_x,                                    \
                              rdma_recv_flag,                                 \
                              rdma_send_x,                                    \
                              rdma_recv_complete,                             \
                              dispatch_rdma_recv_x,                           \
                              dispatch_rdma_recv_count,                       \
                              nvl_buffer,                                     \
                              x,                                              \
                              topk_idx,                                       \
                              topk_weights,                                   \
                              src_info,                                       \
                              layout_range,                                   \
                              rdma_send_flags,                                \
                              atomic_clean_flag,                              \
                              atomic_nvl_sender_multi_sms,                    \
                              num_combined_tokens,                            \
                              hidden,                                         \
                              num_topk,                                       \
                              num_max_dispatch_tokens_per_rank,               \
                              num_experts,                                    \
                              rank,                                           \
                              num_ranks,                                      \
                              a_start_rank,                                   \
                              a_num_ranks,                                    \
                              e_start_rank,                                   \
                              e_num_ranks,                                    \
                              phases);                                        \
              })})})})

  if (hidden == 1536) {
    COMBINE_KERNEL(1536);
  } else if (hidden == 4096) {
    COMBINE_KERNEL(4096);
  } else if (hidden == 5120) {
    COMBINE_KERNEL(5120);
  } else {
    EP_HOST_ASSERT(false && "Unsupported hidden size in group1");
  }

#undef COMBINE_KERNEL
}

}  // namespace m2n_ll_two_stage

}  // namespace deep_ep
