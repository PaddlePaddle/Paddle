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

// Main dispatch/combine entry points that delegate to split implementations

#include "paddle/fluid/distributed/collective/deep_ep/kernels/configs.cuh"
#include "paddle/fluid/distributed/collective/deep_ep/kernels/exception.cuh"

namespace deep_ep {

namespace m2n_ll_two_stage {

// Forward declarations for dispatch implementations
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
                            int phases);

void dispatch_hidden_group2(void* packed_recv_x,
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
                            int phases);

// Forward declarations for combine implementations
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
                           bool dispatch_use_fp8);

void combine_hidden_group2(void* combined_x,
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
                           bool dispatch_use_fp8);

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

  const int dev_id = 0;
  int sm_count;
  cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, dev_id);
  sm_count = 24;
  int num_warp_groups = cell_div(num_experts, sm_count);
  num_warp_groups =
      (num_warp_groups % 2 == 1) ? num_warp_groups + 1 : num_warp_groups;
  const auto num_sms = max(sm_count, cell_div(num_experts, num_warp_groups));
  EP_HOST_ASSERT(num_topk <= kNumMaxTopK);
  const int num_rdma_ranks = num_ranks / NUM_MAX_NVL_PEERS;
  const int num_rdma_experts = num_experts / num_rdma_ranks;

  // Workspace setup
  auto atomic_counter_per_expert = reinterpret_cast<int*>(workspace);
  auto atomic_counter_per_rdma = atomic_counter_per_expert + num_experts;
  auto atomic_finished_counter_per_rdma =
      atomic_counter_per_rdma + num_rdma_ranks;
  auto atomic_recv_tokens_per_rdma_expert =
      atomic_finished_counter_per_rdma + num_rdma_ranks;
  auto atomic_nvl_sender_multi_sms =
      atomic_recv_tokens_per_rdma_expert + num_rdma_ranks * num_rdma_experts;
  auto atomic_counter_per_qp = atomic_nvl_sender_multi_sms + num_rdma_ranks;
  EP_HOST_ASSERT((num_experts + num_rdma_ranks * 3 + num_rdma_experts +
                  num_rdma_ranks * kNumQPs) *
                     sizeof(int) <=
                 NUM_WORKSPACE_BYTES);

  // Dispatch to appropriate hidden size group
  if (hidden == 1536 || hidden == 4096 || hidden == 5120) {
    dispatch_hidden_group1(packed_recv_x,
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
                           hidden,
                           num_topk,
                           num_experts,
                           rank,
                           num_ranks,
                           a_start_rank,
                           a_num_ranks,
                           e_start_rank,
                           e_num_ranks,
                           use_fp8,
                           num_sms,
                           num_warp_groups,
                           stream,
                           phases);
  } else if (hidden == 6144 || hidden == 7168 || hidden == 8192) {
    dispatch_hidden_group2(packed_recv_x,
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
                           hidden,
                           num_topk,
                           num_experts,
                           rank,
                           num_ranks,
                           a_start_rank,
                           a_num_ranks,
                           e_start_rank,
                           e_num_ranks,
                           use_fp8,
                           num_sms,
                           num_warp_groups,
                           stream,
                           phases);
  } else {
    EP_HOST_ASSERT(false && "Unsupported hidden size");
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
             const void* x,
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

  const int dev_id = 0;
  int sm_count;
  cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, dev_id);
  sm_count = 24;
  int num_warp_groups = cell_div(num_experts, sm_count);
  num_warp_groups =
      (num_warp_groups % 2 == 1) ? num_warp_groups + 1 : num_warp_groups;
  const auto num_sms = max(sm_count, cell_div(num_experts, num_warp_groups));
  const int num_rdma_ranks = num_ranks / NUM_MAX_NVL_PEERS;

  // Check workspace
  auto atomic_clean_flag = reinterpret_cast<int*>(workspace);
  auto atomic_nvl_sender_multi_sms = atomic_clean_flag + 1;
  EP_HOST_ASSERT((1 + num_rdma_ranks) * sizeof(int) <= NUM_WORKSPACE_BYTES);
  EP_HOST_ASSERT(num_topk <= kNumMaxTopk);

  // Dispatch to appropriate hidden size group
  if (hidden == 1536 || hidden == 4096 || hidden == 5120) {
    combine_hidden_group1(combined_x,
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
                          num_max_dispatch_tokens_per_rank,
                          num_topk,
                          num_experts,
                          rank,
                          num_ranks,
                          a_start_rank,
                          a_num_ranks,
                          e_start_rank,
                          e_num_ranks,
                          num_sms,
                          num_warp_groups,
                          stream,
                          phases,
                          dispatch_use_fp8);
  } else if (hidden == 6144 || hidden == 7168 || hidden == 8192) {
    combine_hidden_group2(combined_x,
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
                          num_max_dispatch_tokens_per_rank,
                          num_topk,
                          num_experts,
                          rank,
                          num_ranks,
                          a_start_rank,
                          a_num_ranks,
                          e_start_rank,
                          e_num_ranks,
                          num_sms,
                          num_warp_groups,
                          stream,
                          phases,
                          dispatch_use_fp8);
  } else {
    EP_HOST_ASSERT(false && "Unsupported hidden size");
  }
}

}  // namespace m2n_ll_two_stage

}  // namespace deep_ep
