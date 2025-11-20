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

namespace flash_ep {

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

}  // namespace internode

// Internode kernels
namespace internode {

void get_flash_ep_coalesce_rdma_schedule(const int64_t* topk_idx,
                                         const int* local_expert_to_stage_map,
                                         int* dispatch_rdma_schedule_map,
                                         int* combine_rdma_schedule_map,
                                         const int num_ranks,
                                         const int num_experts,
                                         const int num_loop_stage,
                                         const int num_tokens,
                                         const int num_topk,
                                         cudaStream_t stream);

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
                                       int num_loop_stage,
                                       cudaStream_t stream);

void local_dispatch(const void** dispatched_hidden_states,
                    const float** dispatched_topk_weights,
                    const int32_t** dispatched_topk_idx,
                    const int32_t** recv_src_meta,
                    const float** fp8_scales,
                    const int32_t* a2a_prefix_sum,
                    int32_t* global_expertwise_block_cumsum,
                    const int32_t local_expert_id,
                    const int32_t hidden_size,
                    const int32_t topk,
                    const int32_t a2a_num,
                    const int64_t all_token_num,
                    const int64_t output_token_num,
                    const int64_t scale_num,
                    void* output_hidden,
                    int32_t* output_top_idx,
                    float* output_top_probs,
                    int32_t* output_src_meta,
                    float* output_fp8_scale,
                    cudaStream_t stream,
                    bool use_fp8,
                    bool forward);

void local_combine_forward(const __nv_bfloat16* hidden_states,
                           const int32_t** recv_gbl_channel_prefix,
                           const int32_t* recv_src_meta,
                           const int32_t hidden_size,
                           const int32_t num_loop_stage,
                           const int64_t token_num,
                           float** output_hidden_states,
                           cudaStream_t stream);

void local_combine_backward(const __nv_bfloat16* hidden_states,
                            const int32_t* topk_idx,
                            const float* topk_weights,
                            const int32_t** recv_gbl_channel_prefix,
                            const int32_t* recv_src_meta,
                            const int32_t hidden_size,
                            const int32_t num_loop_stage,
                            const int64_t token_num,
                            const int32_t topk,
                            const int32_t local_expert_id,
                            float** output_hidden_states,
                            float** output_topk_weights,
                            cudaStream_t stream);

int get_source_meta_bytes();
int get_details_source_meta_bytes();

#ifdef PADDLE_WITH_NVSHMEM

void fused_notify(const int* dispatch_num_tokens_per_rank,
                  int* dispatch_moe_recv_counter_mapped,
                  const int* combine_num_tokens_per_rank,
                  int* combine_moe_recv_counter_mapped,
                  int num_ranks,
                  const int* dispatch_num_tokens_per_rdma_rank,
                  int* dispatch_moe_recv_rdma_counter_mapped,
                  const int* combine_num_tokens_per_rdma_rank,
                  int* combine_moe_recv_rdma_counter_mapped,
                  const int* dispatch_num_tokens_per_expert,
                  int* dispatch_moe_recv_expert_counter_mapped,
                  int num_experts,
                  const bool* dispatch_is_token_in_rank,
                  const bool* combine_is_token_in_rank,
                  int num_tokens,
                  int num_channels,
                  int hidden_int4,
                  int num_scales,
                  int num_topk,
                  int expert_alignment,
                  int* dispatch_rdma_channel_prefix_matrix,
                  int* dispatch_recv_rdma_rank_prefix_sum,
                  int* dispatch_gbl_channel_prefix_matrix,
                  int* dispatch_recv_gbl_rank_prefix_sum,
                  int* combine_rdma_channel_prefix_matrix,
                  int* combine_recv_rdma_rank_prefix_sum,
                  int* combine_gbl_channel_prefix_matrix,
                  int* combine_recv_gbl_rank_prefix_sum,
                  int* combine_recv_rdma_channel_prefix_matrix,
                  int* combine_recv_gbl_channel_prefix_matrix,
                  int* combine_send_rdma_head,
                  int* combine_send_nvl_head,
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

void notify_combine(const int* num_tokens_per_rank,
                    int* moe_recv_counter_mapped,
                    int num_ranks,
                    const int* num_tokens_per_rdma_rank,
                    int* moe_recv_rdma_counter_mapped,
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
                    int* recv_rdma_channel_prefix_matrix,
                    int* recv_gbl_channel_prefix_matrix,
                    int* send_rdma_head,
                    int* send_nvl_head,
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

void notify_combine_post_step(int num_ranks,
                              int num_channels,
                              const int* recv_gbl_rank_prefix_sum,
                              const int* rdma_channel_prefix_matrix,
                              const int* gbl_channel_prefix_matrix,
                              int* recv_rdma_channel_prefix_matrix,
                              int* recv_gbl_channel_prefix_matrix,
                              void* rdma_buffer_ptr,
                              void** buffer_ptrs,
                              int** task_fifo_ptrs,
                              int head,
                              int rank,
                              cudaStream_t stream,
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
              bool low_latency_mode,
              bool is_asymmetric_mode,
              const int* asymm_send_combine_schedule_map,
              const int* asymm_recv_rdma_counter_loop_prefix_sum,
              const int* asymm_recv_rdma_rank_prefix_sum,
              const int* asymm_recv_rdma_channel_prefix_matrix,
              const int* asymm_send_rdma_head,
              const int* asymm_send_nvl_head,
              int* asymm_aggregated_nvl_head);

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

void clear_buffer(int hidden_int4,
                  int num_scales,
                  int num_topk_idx,
                  int num_topk_weights,
                  int num_ranks,
                  int num_channels,
                  void* rdma_buffer_ptr,
                  int num_max_rdma_chunked_recv_tokens,
                  void** buffer_ptrs,
                  int num_max_nvl_chunked_recv_tokens,
                  int** task_fifo_ptrs,
                  int head,
                  int rank,
                  const bool is_start,
                  const bool is_end,
                  cudaStream_t stream,
                  int64_t num_rdma_bytes,
                  int64_t num_nvl_bytes);

void combine(cudaDataType_t type,
             void* combined_x,
             float* combined_topk_weights,
             const void* x,
             const float* topk_weights,
             const int* combined_rdma_head,
             const int* combined_nvl_head,
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
             bool low_latency_mode,
             bool inplace_float_combine);
#endif  // PADDLE_WITH_NVSHMEM

}  // namespace internode

}  // namespace flash_ep
