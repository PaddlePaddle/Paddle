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

// Forcibly disable NDEBUG
#ifdef NDEBUG
#undef NDEBUG
#endif

#ifndef PADDLE_NO_PYTHON
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#endif
#include <optional>
#include <tuple>
#include <vector>
#include "paddle/fluid/distributed/collective/flash_ep/include/types.h"

#include "paddle/fluid/distributed/collective/flash_ep/config.hpp"
#include "paddle/fluid/distributed/collective/flash_ep/event.hpp"
#include "paddle/fluid/distributed/collective/flash_ep/kernels/configs.cuh"
#include "paddle/fluid/distributed/collective/flash_ep/kernels/exception.cuh"
#include "paddle/phi/api/include/tensor.h"

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/distributed/nccl_comm_context.h"

namespace flash_ep {

constexpr int kCumsumBlockSize = 32;
constexpr int kCumsumInvalidTag = -1;

struct Buffer {
  EP_STATIC_ASSERT(NUM_MAX_NVL_PEERS == 8,
                   "The number of maximum NVLink peers must be 8");

 private:
  // Low-latency mode buffer
  int low_latency_buffer_idx = 0;
  bool low_latency_mode = false;

  // NVLink Buffer
  int64_t num_nvl_bytes;
  void* buffer_ptrs[NUM_MAX_NVL_PEERS] = {nullptr};
  void** buffer_ptrs_gpu = nullptr;

  // NVSHMEM Buffer
  int64_t num_rdma_bytes;
  void* rdma_buffer_ptr = nullptr;

  // Device info and communication
  int device_id;
  int rank, rdma_rank, nvl_rank;
  int num_ranks, num_rdma_ranks, num_nvl_ranks;
  cudaIpcMemHandle_t ipc_handles[NUM_MAX_NVL_PEERS];

  // Stream for communication
  // flash_ep::detail::CUDAStream comm_stream;
  cudaStream_t comm_stream;
  phi::distributed::NCCLCommContext* comm_ctx;
  phi::GPUContext* calc_ctx;

  // After IPC/NVSHMEM synchronization, this flag will be true
  bool available = false;

  // Task fifo
  int head = 0;
  int* task_fifo_ptrs[NUM_MAX_NVL_PEERS] = {nullptr};
  int** task_fifo_ptrs_gpu = nullptr;

  // Workspace
  void* workspace = nullptr;

  // Host-side MoE info
  volatile int* moe_recv_counter = nullptr;
  int* moe_recv_counter_mapped = nullptr;

  // Host-side expert-level MoE info
  volatile int* moe_recv_expert_counter = nullptr;
  int* moe_recv_expert_counter_mapped = nullptr;

  // Host-side RDMA-level MoE info
  volatile int* moe_recv_rdma_counter = nullptr;
  int* moe_recv_rdma_counter_mapped = nullptr;

  // Host-side MoE info
  volatile int* dispatch_moe_recv_counter = nullptr;
  int* dispatch_moe_recv_counter_mapped = nullptr;
  volatile int* combine_moe_recv_counter = nullptr;
  int* combine_moe_recv_counter_mapped = nullptr;

  // Host-side RDMA-level MoE info
  volatile int* dispatch_moe_recv_rdma_counter = nullptr;
  int* dispatch_moe_recv_rdma_counter_mapped = nullptr;
  volatile int* combine_moe_recv_rdma_counter = nullptr;
  int* combine_moe_recv_rdma_counter_mapped = nullptr;

  int num_loop_stage{1};

 private:
  void move_fifo_slots(int num_slots = 1);

 public:
  Buffer(int rank,
         int num_ranks,
         int num_loop_stage,
         int64_t num_nvl_bytes,
         int64_t num_rdma_bytes,
         bool low_latency_mode,
         int context_ring_id);

  ~Buffer() noexcept(false);

  bool is_available() const;

  bool is_internode_available() const;

  int get_num_rdma_ranks() const;

  int get_rdma_rank() const;

  int get_root_rdma_rank(bool global) const;

  int get_local_device_id() const;

  cudaStream_t get_comm_stream() const;

#ifndef PADDLE_NO_PYTHON
  pybind11::bytearray get_local_ipc_handle() const;

  pybind11::bytearray get_local_nvshmem_unique_id() const;

  void sync(const std::vector<int>& device_ids,
            const std::vector<std::optional<pybind11::bytearray>>&
                all_gathered_handles,
            const std::optional<pybind11::bytearray>& root_unique_id_opt);
#endif

#ifdef PADDLE_WITH_NVSHMEM
  void clear_buffer(const flash_ep::detail::Tensor& x,
                    const std::optional<flash_ep::detail::Tensor>& x_scales,
                    const std::optional<flash_ep::detail::Tensor>& topk_idx,
                    const bool is_start,
                    const bool is_end,
                    const Config& config);

  std::tuple<flash_ep::detail::Tensor,
             std::optional<flash_ep::detail::Tensor>,
             std::optional<flash_ep::detail::Tensor>,
             std::optional<flash_ep::detail::Tensor>,
             std::vector<int>,
             flash_ep::detail::Tensor,
             flash_ep::detail::Tensor,
             std::optional<flash_ep::detail::Tensor>,
             flash_ep::detail::Tensor,
             std::optional<flash_ep::detail::Tensor>,
             flash_ep::detail::Tensor,
             std::optional<flash_ep::detail::Tensor>,
             std::optional<flash_ep::detail::Tensor>,
             std::optional<flash_ep::detail::Tensor>,
             std::optional<EventHandle>>
  internode_dispatch(
      const flash_ep::detail::Tensor& x,
      const std::optional<flash_ep::detail::Tensor>& x_scales,
      const std::optional<flash_ep::detail::Tensor>& topk_idx,
      const std::optional<flash_ep::detail::Tensor>& topk_weights,
      const std::optional<flash_ep::detail::Tensor>& num_tokens_per_rank,
      const std::optional<flash_ep::detail::Tensor>& num_tokens_per_rdma_rank,
      const flash_ep::detail::Tensor& is_token_in_rank,
      const std::optional<flash_ep::detail::Tensor>& num_tokens_per_expert,
      int cached_num_recv_tokens,
      int cached_num_rdma_recv_tokens,
      const std::optional<flash_ep::detail::Tensor>&
          cached_rdma_channel_prefix_matrix,
      const std::optional<flash_ep::detail::Tensor>&
          cached_recv_rdma_rank_prefix_sum,
      const std::optional<flash_ep::detail::Tensor>&
          cached_gbl_channel_prefix_matrix,
      const std::optional<flash_ep::detail::Tensor>&
          cached_recv_gbl_rank_prefix_sum,
      const std::optional<flash_ep::detail::Tensor>&
          asymm_send_combine_schedule_map,
      const std::optional<flash_ep::detail::Tensor>&
          asymm_recv_rdma_counter_loop_prefix_sum,
      const std::optional<flash_ep::detail::Tensor>&
          asymm_recv_rdma_rank_prefix_sum,
      const std::optional<flash_ep::detail::Tensor>&
          asymm_recv_rdma_channel_prefix_matrix,
      const std::optional<flash_ep::detail::Tensor>& asymm_send_rdma_head,
      const std::optional<flash_ep::detail::Tensor>& asymm_send_nvl_head,
      const std::optional<flash_ep::detail::Tensor>& asymm_aggregated_nvl_head,
      int expert_alignment,
      const Config& config,
      std::optional<EventHandle>& previous_event,  // NOLINT
      bool async,
      bool allocate_on_comm_stream,
      int num_experts);

  std::tuple<std::optional<flash_ep::detail::Tensor>,
             std::optional<flash_ep::detail::Tensor>,
             std::optional<EventHandle>>
  internode_combine(
      const flash_ep::detail::Tensor& x,
      const std::optional<flash_ep::detail::Tensor>& topk_weights,
      const flash_ep::detail::Tensor& rdma_channel_prefix_matrix,
      const flash_ep::detail::Tensor& rdma_rank_prefix_sum,
      const flash_ep::detail::Tensor& gbl_channel_prefix_matrix,
      const flash_ep::detail::Tensor& combined_rdma_head,
      const flash_ep::detail::Tensor& combined_nvl_head,
      const std::optional<flash_ep::detail::Tensor>& combined_x,
      const std::optional<flash_ep::detail::Tensor>& combined_topk_weights,
      const Config& config,
      std::optional<EventHandle>& previous_event,  // NOLINT
      bool async,
      bool allocate_on_comm_stream);

#endif  // PADDLE_WITH_NVSHMEM

  std::tuple<paddle::Tensor,
             std::optional<paddle::Tensor>,
             std::optional<paddle::Tensor>,
             std::optional<paddle::Tensor>,
             std::vector<int>,
             paddle::Tensor,
             paddle::Tensor,
             std::optional<paddle::Tensor>,
             paddle::Tensor,
             std::optional<paddle::Tensor>,
             paddle::Tensor,
             std::optional<paddle::Tensor>,
             std::optional<paddle::Tensor>,
             std::optional<paddle::Tensor>,
             std::optional<EventHandle>>
  internode_dispatch_api(
      const paddle::Tensor& x,
      const std::optional<paddle::Tensor>& x_scales,
      const std::optional<paddle::Tensor>& topk_idx,
      const std::optional<paddle::Tensor>& topk_weights,
      const std::optional<paddle::Tensor>& num_tokens_per_rank,
      const std::optional<paddle::Tensor>& num_tokens_per_rdma_rank,
      const paddle::Tensor& is_token_in_rank,
      const std::optional<paddle::Tensor>& num_tokens_per_expert,
      int cached_num_recv_tokens,
      int cached_num_rdma_recv_tokens,
      const std::optional<paddle::Tensor>& cached_rdma_channel_prefix_matrix,
      const std::optional<paddle::Tensor>& cached_recv_rdma_rank_prefix_sum,
      const std::optional<paddle::Tensor>& cached_gbl_channel_prefix_matrix,
      const std::optional<paddle::Tensor>& cached_recv_gbl_rank_prefix_sum,
      const std::optional<paddle::Tensor>& asymm_send_combine_schedule_map,
      const std::optional<paddle::Tensor>&
          asymm_recv_rdma_counter_loop_prefix_sum,
      const std::optional<paddle::Tensor>& asymm_recv_rdma_rank_prefix_sum,
      const std::optional<paddle::Tensor>&
          asymm_recv_rdma_channel_prefix_matrix,
      const std::optional<paddle::Tensor>& asymm_send_rdma_head,
      const std::optional<paddle::Tensor>& asymm_send_nvl_head,
      const std::optional<paddle::Tensor>& asymm_aggregated_nvl_head,
      int expert_alignment,
      const Config& config,
      std::optional<EventHandle>& previous_event,  // NOLINT
      bool async,
      bool allocate_on_comm_stream,
      int num_experts);

  std::tuple<std::optional<paddle::Tensor>,
             std::optional<paddle::Tensor>,
             std::optional<EventHandle>>
  internode_combine_api(
      const paddle::Tensor& x,
      const std::optional<paddle::Tensor>& topk_weights,
      const paddle::Tensor& rdma_channel_prefix_matrix,
      const paddle::Tensor& rdma_rank_prefix_sum,
      const paddle::Tensor& gbl_channel_prefix_matrix,
      const paddle::Tensor& combined_rdma_head,
      const paddle::Tensor& combined_nvl_head,
      const std::optional<paddle::Tensor>& combined_x,
      const std::optional<paddle::Tensor>& combined_topk_weights,
      const Config& config,
      std::optional<EventHandle>& previous_event,  // NOLINT
      bool async,
      bool allocate_on_comm_stream);

  std::tuple<std::vector<std::vector<int>>,
             std::vector<int>,
             std::vector<int>,
             flash_ep::detail::Tensor,
             flash_ep::detail::Tensor,
             flash_ep::detail::Tensor,
             flash_ep::detail::Tensor,
             std::vector<int>,
             std::vector<int>,
             flash_ep::detail::Tensor,
             flash_ep::detail::Tensor,
             flash_ep::detail::Tensor,
             flash_ep::detail::Tensor,
             flash_ep::detail::Tensor,
             flash_ep::detail::Tensor>
  internode_fused_notify(
      const flash_ep::detail::Tensor& x,
      const std::optional<flash_ep::detail::Tensor>& x_scales,
      const std::optional<flash_ep::detail::Tensor>& topk_idx,
      const std::optional<flash_ep::detail::Tensor>&
          dispatch_num_tokens_per_rank,
      const std::optional<flash_ep::detail::Tensor>&
          dispatch_num_tokens_per_rdma_rank,
      const std::optional<flash_ep::detail::Tensor>&
          dispatch_num_tokens_per_expert,
      const flash_ep::detail::Tensor& dispatch_is_token_in_rank,
      const std::optional<flash_ep::detail::Tensor>&
          combine_num_tokens_per_rank,
      const std::optional<flash_ep::detail::Tensor>&
          combine_num_tokens_per_rdma_rank,
      const flash_ep::detail::Tensor& combine_is_token_in_rank,
      int expert_alignment,
      const Config& config);

  std::tuple<std::vector<std::vector<int>>,
             std::vector<int>,
             std::vector<int>,
             paddle::Tensor,
             paddle::Tensor,
             paddle::Tensor,
             paddle::Tensor,
             std::vector<int>,
             std::vector<int>,
             paddle::Tensor,
             paddle::Tensor,
             paddle::Tensor,
             paddle::Tensor,
             paddle::Tensor,
             paddle::Tensor>
  internode_fused_notify_api(
      const paddle::Tensor& x,
      const std::optional<paddle::Tensor>& x_scales,
      const std::optional<paddle::Tensor>& topk_idx,
      const std::optional<paddle::Tensor>& dispatch_num_tokens_per_rank,
      const std::optional<paddle::Tensor>& dispatch_num_tokens_per_rdma_rank,
      const std::optional<paddle::Tensor>& dispatch_num_tokens_per_expert,
      const paddle::Tensor& dispatch_is_token_in_rank,
      const std::optional<paddle::Tensor>& combine_num_tokens_per_rank,
      const std::optional<paddle::Tensor>& combine_num_tokens_per_rdma_rank,
      const paddle::Tensor& combine_is_token_in_rank,
      int expert_alignment,
      const Config& config);

  void clear_buffer_api(const paddle::Tensor& x,
                        const std::optional<paddle::Tensor>& x_scales,
                        const std::optional<paddle::Tensor>& topk_idx,
                        const bool is_start,
                        const bool is_end,
                        const Config& config);
};

std::tuple<paddle::Tensor,  // dispatch_rdma_schedule_map
           paddle::Tensor>  // combine_rdma_schedule_map
get_flash_ep_coalesce_rdma_schedule_api(
    const paddle::Tensor& topk_idx,
    const paddle::Tensor& local_expert_to_stage_map,
    const int num_ranks,
    const int num_experts,
    const int num_loop_stage);

std::tuple<paddle::Tensor,  // num_tokens_per_rank
           paddle::Tensor,  // num_tokens_per_rdma_rank
           paddle::Tensor,  // num_tokens_per_expert
           paddle::Tensor>  // is_token_in_rank
get_flash_ep_coalesce_rdma_layout_api(
    const paddle::Tensor& topk_idx,
    const paddle::Tensor& dispatch_rdma_schedule_map,
    const paddle::Tensor& combine_rdma_schedule_map,
    const int num_ranks,
    const int num_experts,
    const int num_loop_stage);

std::vector<paddle::Tensor> get_flashep_rowmap_api(
    const paddle::Tensor& topk_idx, const int64_t num_experts);

std::tuple<paddle::Tensor,
           paddle::Tensor,
           paddle::Tensor,
           std::optional<paddle::Tensor>>
local_dispatch_forward_api(
    const std::vector<paddle::Tensor>& hidden_states,
    const std::vector<paddle::Tensor>& topk_weights,
    const std::vector<paddle::Tensor>& topk_idx,
    const std::vector<paddle::Tensor>& recv_src_meta_per_a2a,
    const std::optional<std::vector<paddle::Tensor>>& fp8_scales,
    const std::vector<paddle::Tensor>& output_route_map,
    const std::vector<paddle::Tensor>& output_route_map_len,
    const int64_t num_experts,
    const int64_t local_expert_id,
    const int64_t ori_out_len,
    const int64_t padding_align,
    const int64_t num_loop_stage);

std::vector<paddle::Tensor> local_dispatch_backward_api(
    const std::vector<paddle::Tensor>& hidden_states,
    const std::vector<paddle::Tensor>& topk_idx,
    const std::vector<paddle::Tensor>& recv_src_meta_per_a2a,
    const std::vector<paddle::Tensor>& output_route_map,
    const std::vector<paddle::Tensor>& output_route_map_len,
    const int64_t num_experts,
    const int64_t local_expert_id,
    const int64_t ori_out_len,
    const int padding_align,
    const int64_t num_loop_stage);

void local_combine_forward_api(
    std::vector<paddle::Tensor>& combine_buffers,  // NOLINT
    const paddle::Tensor& hidden_states,
    const paddle::Tensor& recv_gbl_src_meta,
    const std::vector<paddle::Tensor>& recv_gbl_channel_prefix_matrix_list,
    const int64_t ori_len,
    const std::vector<int>& is_buffer_active);

void local_combine_backward_api(
    std::vector<paddle::Tensor>& combine_buffers,  // NOLINT
    std::vector<paddle::Tensor>& combine_probs,    // NOLINT
    const paddle::Tensor& hidden_states,
    const paddle::Tensor& topk_idx,
    const paddle::Tensor& topk_weights,
    const paddle::Tensor& recv_gbl_src_meta,
    const std::vector<paddle::Tensor>& recv_gbl_channel_prefix_matrix_list,
    const int64_t local_expert_id,
    const int64_t ori_len,
    const std::vector<int>& is_buffer_active);

flash_ep::detail::Tensor ConvertPaddleTensorToDetailTensor(
    const paddle::Tensor& tensor);
paddle::Tensor ConvertDetailTensorToPaddleTensor(
    const flash_ep::detail::Tensor& tensor);

std::optional<flash_ep::detail::Tensor>
ConvertOptionalPaddleTensorToDetailTensor(
    const std::optional<paddle::Tensor>& tensor);
std::optional<paddle::Tensor> ConvertOptionalDetailTensorToPaddleTensor(
    const std::optional<flash_ep::detail::Tensor>& tensor);

}  // namespace flash_ep
