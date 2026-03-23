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

// m2n_ll_two_stage dispatch implementation for hidden sizes: 1536, 4096, 5120

#include "paddle/fluid/distributed/collective/deep_ep/kernels/m2n_ll_two_stage_impl.cuh"

namespace deep_ep {

namespace m2n_ll_two_stage {

void dispatch_hidden_group1(void* packed_recv_x,
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
                            int* atomic_counter_per_expert,
                            int* atomic_counter_per_rdma,
                            int* atomic_finished_counter_per_rdma,
                            int* atomic_recv_tokens_per_rdma_expert,
                            int* atomic_nvl_sender_multi_sms,
                            int* atomic_counter_per_qp,
                            int num_tokens,
                            int num_max_dispatch_tokens_per_rank,
                            int hidden,
                            int num_topk,
                            int num_experts,
                            int rank,
                            int num_ranks,
                            int a_start_rank,
                            int a_num_ranks,
                            int e_start_rank,
                            int e_num_ranks,
                            bool use_fp8,
                            int num_sms,
                            int num_warp_groups,
                            cudaStream_t stream,
                            int phases) {
  constexpr int kNumMaxTopK = 8;
  constexpr int kNumQPs = 32;
  constexpr int NUM_WARPS = 32;
  const int num_rdma_ranks = num_ranks / NUM_MAX_NVL_PEERS;

#define DISPATCH_KERNEL(kHidden)                                              \
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
                assert(num_rdma_ranks <= kNumWarpGroups * kNumWarpsPerGroup); \
                EP_STATIC_ASSERT(                                             \
                    kNumMaxTopK + 1 <= kNumWarpGroups * kNumWarpsPerGroup,    \
                    "Too many top-k selections");                             \
                auto dispatch_func = use_fp8                                  \
                                         ? dispatch_kernel<true,              \
                                                           kNumWarpGroups,    \
                                                           kNumWarpsPerGroup, \
                                                           kHidden,           \
                                                           kNumRdmaRanks,     \
                                                           kNumExperts,       \
                                                           kTopk,             \
                                                           kNumQPs>           \
                                         : dispatch_kernel<false,             \
                                                           kNumWarpGroups,    \
                                                           kNumWarpsPerGroup, \
                                                           kHidden,           \
                                                           kNumRdmaRanks,     \
                                                           kNumExperts,       \
                                                           kTopk,             \
                                                           kNumQPs>;          \
                SETUP_LAUNCH_CONFIG(                                          \
                    num_sms, kNumWarpGroups* kNumWarpsPerGroup * 32, stream); \
                LAUNCH_KERNEL(&cfg,                                           \
                              dispatch_func,                                  \
                              packed_recv_x,                                  \
                              packed_recv_x_scales,                           \
                              packed_rdma_recv_x,                             \
                              packed_recv_src_info,                           \
                              packed_recv_layout_range,                       \
                              packed_recv_count,                              \
                              packed_rdma_recv_count,                         \
                              rdma_send_flags,                                \
                              rdma_recv_x,                                    \
                              rdma_recv_count,                                \
                              rdma_recv_complete,                             \
                              rdma_x,                                         \
                              nvl_recv_x,                                     \
                              x,                                              \
                              topk_idx,                                       \
                              topk_weights,                                   \
                              atomic_counter_per_expert,                      \
                              atomic_counter_per_rdma,                        \
                              atomic_finished_counter_per_rdma,               \
                              atomic_recv_tokens_per_rdma_expert,             \
                              atomic_nvl_sender_multi_sms,                    \
                              atomic_counter_per_qp,                          \
                              num_tokens,                                     \
                              num_max_dispatch_tokens_per_rank,               \
                              rank,                                           \
                              a_start_rank,                                   \
                              a_num_ranks,                                    \
                              e_start_rank,                                   \
                              e_num_ranks,                                    \
                              phases);                                        \
              })})})})

  if (hidden == 1536) {
    DISPATCH_KERNEL(1536);
  } else if (hidden == 4096) {
    DISPATCH_KERNEL(4096);
  } else if (hidden == 5120) {
    DISPATCH_KERNEL(5120);
  } else {
    EP_HOST_ASSERT(false && "Unsupported hidden size in group1");
  }

#undef DISPATCH_KERNEL
}

}  // namespace m2n_ll_two_stage

}  // namespace deep_ep
