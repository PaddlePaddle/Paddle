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

#pragma once

#include <vector>

namespace deep_ep {

// Intranode runtime
namespace intranode {

void barrier(int** task_fifo_ptrs,
             int head,
             int rank,
             int num_ranks,
             cudaStream_t stream);

}  // namespace intranode

// Internode runtime
namespace internode {

#ifdef PADDLE_WITH_NVSHMEM
std::vector<uint8_t> get_unique_id();

int init(const std::vector<uint8_t>& root_unique_id_val,
         int rank,
         int num_ranks,
         bool low_latency_mode);

void* alloc(size_t size, size_t alignment);

void free(void* ptr);

void barrier();

void finalize();
#endif  // PADDLE_WITH_NVSHMEM

void get_dispatch_layout(const int64_t* topk_idx,
                         int* num_tokens_per_rank,
                         int* num_tokens_per_rdma_rank,
                         int* num_tokens_per_expert,
                         bool* is_token_in_rank,
                         int num_tokens,
                         int num_topk,
                         int num_ranks,
                         int num_experts,
                         cudaStream_t stream);

}  // namespace internode

// Intranode kernels
namespace intranode {

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
                     int num_sms);

void cached_notify_dispatch(const int* rank_prefix_matrix,
                            int num_memset_int,
                            void** buffer_ptrs,
                            int** task_fifo_ptrs,
                            int head,
                            int rank,
                            int num_ranks,
                            cudaStream_t stream);

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
              int num_recv_buffer_tokens);

void cached_notify_combine(void** buffer_ptrs,
                           int* send_head,
                           int num_channels,
                           int num_recv_tokens,
                           int num_memset_int,
                           int** task_fifo_ptrs,
                           int head,
                           int rank,
                           int num_ranks,
                           cudaStream_t stream);

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
             int num_recv_buffer_tokens);

}  // namespace intranode

#ifdef PADDLE_WITH_NVSHMEM
// Internode kernels
namespace internode {

int get_source_meta_bytes();

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
                     bool low_latency_mode);

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
              bool low_latency_mode);

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
                   bool low_latency_mode);

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
             bool low_latency_mode);

}  // namespace internode

// Internode low-latency kernels
namespace internode_ll {

void barrier_all(cudaStream_t stream);
void clean_low_latency_buffer(int* clean_0,
                              int num_clean_int_0,
                              int* clean_1,
                              int num_clean_int_1,
                              cudaStream_t stream);

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
              int phases);

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
             bool zero_copy);

}  // namespace internode_ll

namespace internode_ll_two_stage {

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
              int next_buffer_id);

void combine(void* combined_x,
             void* rdma_recv_x,
             int* rdma_recv_flag,
             void* rdma_send_x,
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
             void* workspace,
             cudaStream_t stream,
             int phases,
             bool dispatch_use_fp8,
             int next_buffer_id);

}  // namespace internode_ll_two_stage

#endif  // PADDLE_WITH_NVSHMEM

}  // namespace deep_ep
