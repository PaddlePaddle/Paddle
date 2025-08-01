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
#include "paddle/fluid/distributed/collective/deep_ep/include/types.h"

#include "paddle/fluid/distributed/collective/deep_ep/config.hpp"
#include "paddle/fluid/distributed/collective/deep_ep/event.hpp"
#include "paddle/fluid/distributed/collective/deep_ep/kernels/configs.cuh"
#include "paddle/fluid/distributed/collective/deep_ep/kernels/exception.cuh"
#include "paddle/phi/api/include/tensor.h"

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/distributed/nccl_comm_context.h"

namespace deep_ep {

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
  // deep_ep::detail::CUDAStream comm_stream;
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

 private:
  void move_fifo_slots(int num_slots = 1);

 public:
  Buffer(int rank,
         int num_ranks,
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

  std::tuple<deep_ep::detail::Tensor,
             std::optional<deep_ep::detail::Tensor>,
             deep_ep::detail::Tensor,
             deep_ep::detail::Tensor,
             std::optional<EventHandle>>
  get_dispatch_layout(const deep_ep::detail::Tensor& topk_idx,
                      int num_experts,
                      std::optional<EventHandle>& previous_event,  // NOLINT
                      bool async,
                      bool allocate_on_comm_stream);

  std::tuple<deep_ep::detail::Tensor,
             std::optional<deep_ep::detail::Tensor>,
             std::optional<deep_ep::detail::Tensor>,
             std::optional<deep_ep::detail::Tensor>,
             std::vector<int>,
             deep_ep::detail::Tensor,
             deep_ep::detail::Tensor,
             deep_ep::detail::Tensor,
             deep_ep::detail::Tensor,
             deep_ep::detail::Tensor,
             std::optional<EventHandle>>
  intranode_dispatch(
      const deep_ep::detail::Tensor& x,
      const std::optional<deep_ep::detail::Tensor>& x_scales,
      const std::optional<deep_ep::detail::Tensor>& topk_idx,
      const std::optional<deep_ep::detail::Tensor>& topk_weights,
      const std::optional<deep_ep::detail::Tensor>& num_tokens_per_rank,
      const deep_ep::detail::Tensor& is_token_in_rank,
      const std::optional<deep_ep::detail::Tensor>& num_tokens_per_expert,
      int cached_num_recv_tokens,
      const std::optional<deep_ep::detail::Tensor>& cached_rank_prefix_matrix,
      const std::optional<deep_ep::detail::Tensor>&
          cached_channel_prefix_matrix,
      int expert_alignment,
      const Config& config,
      std::optional<EventHandle>& previous_event,  // NOLINT
      bool async,
      bool allocate_on_comm_stream);

  std::tuple<deep_ep::detail::Tensor,
             std::optional<deep_ep::detail::Tensor>,
             std::optional<EventHandle>>
  intranode_combine(const deep_ep::detail::Tensor& x,
                    const std::optional<deep_ep::detail::Tensor>& topk_weights,
                    const deep_ep::detail::Tensor& src_idx,
                    const deep_ep::detail::Tensor& rank_prefix_matrix,
                    const deep_ep::detail::Tensor& channel_prefix_matrix,
                    const deep_ep::detail::Tensor& send_head,
                    const Config& config,
                    std::optional<EventHandle>& previous_event,  // NOLINT
                    bool async,
                    bool allocate_on_comm_stream);

#ifdef PADDLE_WITH_NVSHMEM
  std::tuple<deep_ep::detail::Tensor,
             std::optional<deep_ep::detail::Tensor>,
             std::optional<deep_ep::detail::Tensor>,
             std::optional<deep_ep::detail::Tensor>,
             std::vector<int>,
             deep_ep::detail::Tensor,
             deep_ep::detail::Tensor,
             std::optional<deep_ep::detail::Tensor>,
             deep_ep::detail::Tensor,
             std::optional<deep_ep::detail::Tensor>,
             deep_ep::detail::Tensor,
             std::optional<deep_ep::detail::Tensor>,
             std::optional<deep_ep::detail::Tensor>,
             std::optional<deep_ep::detail::Tensor>,
             std::optional<EventHandle>>
  internode_dispatch(
      const deep_ep::detail::Tensor& x,
      const std::optional<deep_ep::detail::Tensor>& x_scales,
      const std::optional<deep_ep::detail::Tensor>& topk_idx,
      const std::optional<deep_ep::detail::Tensor>& topk_weights,
      const std::optional<deep_ep::detail::Tensor>& num_tokens_per_rank,
      const std::optional<deep_ep::detail::Tensor>& num_tokens_per_rdma_rank,
      const deep_ep::detail::Tensor& is_token_in_rank,
      const std::optional<deep_ep::detail::Tensor>& num_tokens_per_expert,
      int cached_num_recv_tokens,
      int cached_num_rdma_recv_tokens,
      const std::optional<deep_ep::detail::Tensor>&
          cached_rdma_channel_prefix_matrix,
      const std::optional<deep_ep::detail::Tensor>&
          cached_recv_rdma_rank_prefix_sum,
      const std::optional<deep_ep::detail::Tensor>&
          cached_gbl_channel_prefix_matrix,
      const std::optional<deep_ep::detail::Tensor>&
          cached_recv_gbl_rank_prefix_sum,
      int expert_alignment,
      const Config& config,
      std::optional<EventHandle>& previous_event,  // NOLINT
      bool async,
      bool allocate_on_comm_stream);

  std::tuple<deep_ep::detail::Tensor,
             std::optional<deep_ep::detail::Tensor>,
             std::optional<EventHandle>>
  internode_combine(const deep_ep::detail::Tensor& x,
                    const std::optional<deep_ep::detail::Tensor>& topk_weights,
                    const deep_ep::detail::Tensor& src_meta,
                    const deep_ep::detail::Tensor& is_combined_token_in_rank,
                    const deep_ep::detail::Tensor& rdma_channel_prefix_matrix,
                    const deep_ep::detail::Tensor& rdma_rank_prefix_sum,
                    const deep_ep::detail::Tensor& gbl_channel_prefix_matrix,
                    const deep_ep::detail::Tensor& combined_rdma_head,
                    const deep_ep::detail::Tensor& combined_nvl_head,
                    const Config& config,
                    std::optional<EventHandle>& previous_event,  // NOLINT
                    bool async,
                    bool allocate_on_comm_stream);
#endif  // PADDLE_WITH_NVSHMEM

  void clean_low_latency_buffer(int num_max_dispatch_tokens_per_rank,
                                int hidden,
                                int num_experts);
  void barrier_all();

#ifdef PADDLE_WITH_NVSHMEM
  std::tuple<deep_ep::detail::Tensor,
             std::optional<deep_ep::detail::Tensor>,
             deep_ep::detail::Tensor,
             deep_ep::detail::Tensor,
             deep_ep::detail::Tensor,
             std::optional<EventHandle>,
             std::optional<std::function<void()>>>
  low_latency_dispatch(
      const deep_ep::detail::Tensor& x,
      const deep_ep::detail::Tensor& topk_idx,
      const std::optional<deep_ep::detail::Tensor>& expertwise_scale,
      int num_max_dispatch_tokens_per_rank,
      int num_experts,
      bool use_fp8,
      bool async,
      bool return_recv_hook);

  std::tuple<deep_ep::detail::Tensor,
             std::optional<EventHandle>,
             std::optional<std::function<void()>>>
  low_latency_combine(
      const deep_ep::detail::Tensor& x,
      const deep_ep::detail::Tensor& topk_idx,
      const deep_ep::detail::Tensor& topk_weights,
      const deep_ep::detail::Tensor& src_info,
      const deep_ep::detail::Tensor& layout_range,
      int num_max_dispatch_tokens_per_rank,
      int num_experts,
      bool zero_copy,
      bool async,
      bool return_recv_hook,
      const std::optional<deep_ep::detail::Tensor>& out = std::nullopt);

  std::tuple<deep_ep::detail::Tensor,
             std::optional<deep_ep::detail::Tensor>,
             deep_ep::detail::Tensor,
             deep_ep::detail::Tensor,
             deep_ep::detail::Tensor,
             deep_ep::detail::Tensor,
             deep_ep::detail::Tensor,
             deep_ep::detail::Tensor,
             std::optional<EventHandle>,
             std::optional<std::function<void()>>>
  low_latency_dispatch_two_stage(const deep_ep::detail::Tensor& x,
                                 const deep_ep::detail::Tensor& topk_idx,
                                 const deep_ep::detail::Tensor& topk_weights,
                                 int num_max_dispatch_tokens_per_rank,
                                 int num_experts,
                                 bool use_fp8,
                                 bool async,
                                 bool return_recv_hook);

  std::tuple<deep_ep::detail::Tensor,
             std::optional<EventHandle>,
             std::optional<std::function<void()>>>
  low_latency_combine_two_stage(
      const deep_ep::detail::Tensor& x,
      const deep_ep::detail::Tensor& rdma_recv_x,
      const deep_ep::detail::Tensor& topk_idx,
      const deep_ep::detail::Tensor& topk_weights,
      const deep_ep::detail::Tensor& src_info,
      const deep_ep::detail::Tensor& layout_range,
      const deep_ep::detail::Tensor& rdma_send_flags,
      const deep_ep::detail::Tensor& dispatch_rdma_recv_count,
      int num_max_dispatch_tokens_per_rank,
      int num_experts,
      bool dispatch_use_fp8,
      bool async,
      bool return_recv_hook,
      const std::optional<deep_ep::detail::Tensor>& out);

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
      int expert_alignment,
      const Config& config,
      std::optional<EventHandle>& previous_event,  // NOLINT
      bool async,
      bool allocate_on_comm_stream);

  std::tuple<paddle::Tensor,
             std::optional<paddle::Tensor>,
             std::optional<EventHandle>>
  internode_combine_api(const paddle::Tensor& x,
                        const std::optional<paddle::Tensor>& topk_weights,
                        const paddle::Tensor& src_meta,
                        const paddle::Tensor& is_combined_token_in_rank,
                        const paddle::Tensor& rdma_channel_prefix_matrix,
                        const paddle::Tensor& rdma_rank_prefix_sum,
                        const paddle::Tensor& gbl_channel_prefix_matrix,
                        const paddle::Tensor& combined_rdma_head,
                        const paddle::Tensor& combined_nvl_head,
                        const Config& config,
                        std::optional<EventHandle>& previous_event,  // NOLINT
                        bool async,
                        bool allocate_on_comm_stream);

  std::tuple<paddle::Tensor,
             std::optional<paddle::Tensor>,
             paddle::Tensor,
             paddle::Tensor,
             paddle::Tensor,
             std::optional<EventHandle>,
             std::optional<std::function<void()>>>
  low_latency_dispatch_api(
      const paddle::Tensor& x,
      const paddle::Tensor& topk_idx,
      const std::optional<paddle::Tensor>& expertwise_scale,
      int num_max_dispatch_tokens_per_rank,
      int num_experts,
      bool use_fp8,
      bool async,
      bool return_recv_hook);

  std::tuple<paddle::Tensor,
             std::optional<EventHandle>,
             std::optional<std::function<void()>>>
  low_latency_combine_api(const paddle::Tensor& x,
                          const paddle::Tensor& topk_idx,
                          const paddle::Tensor& topk_weights,
                          const paddle::Tensor& src_info,
                          const paddle::Tensor& layout_range,
                          int num_max_dispatch_tokens_per_rank,
                          int num_experts,
                          bool zero_copy,
                          bool async,
                          bool return_recv_hook,
                          const std::optional<paddle::Tensor>& out);

  std::tuple<paddle::Tensor,
             std::optional<paddle::Tensor>,
             paddle::Tensor,
             paddle::Tensor,
             paddle::Tensor,
             paddle::Tensor,
             paddle::Tensor,
             paddle::Tensor,
             std::optional<EventHandle>,
             std::optional<std::function<void()>>>
  low_latency_dispatch_two_stage_api(const paddle::Tensor& x,
                                     const paddle::Tensor& topk_idx,
                                     const paddle::Tensor& topk_weights,
                                     int num_max_dispatch_tokens_per_rank,
                                     int num_experts,
                                     bool use_fp8,
                                     bool async,
                                     bool return_recv_hook);

  std::tuple<paddle::Tensor,
             std::optional<EventHandle>,
             std::optional<std::function<void()>>>
  low_latency_combine_two_stage_api(
      const paddle::Tensor& x,
      const paddle::Tensor& rdma_recv_x,
      const paddle::Tensor& topk_idx,
      const paddle::Tensor& topk_weights,
      const paddle::Tensor& src_info,
      const paddle::Tensor& layout_range,
      const paddle::Tensor& rdma_send_flags,
      const paddle::Tensor& dispatch_rdma_recv_count,
      int num_max_dispatch_tokens_per_rank,
      int num_experts,
      bool dispatch_use_fp8,
      bool async,
      bool return_recv_hook,
      const std::optional<paddle::Tensor>& out);

  std::tuple<paddle::Tensor,
             std::optional<paddle::Tensor>,
             paddle::Tensor,
             paddle::Tensor,
             std::optional<EventHandle>>
  get_dispatch_layout_api(const paddle::Tensor& topk_idx,
                          int num_experts,
                          std::optional<EventHandle>& previous_event,  // NOLINT
                          bool async,
                          bool allocate_on_comm_stream);

  std::tuple<paddle::Tensor,
             std::optional<paddle::Tensor>,
             std::optional<paddle::Tensor>,
             std::optional<paddle::Tensor>,
             std::vector<int>,
             paddle::Tensor,
             paddle::Tensor,
             paddle::Tensor,
             paddle::Tensor,
             paddle::Tensor,
             std::optional<EventHandle>>
  intranode_dispatch_api(
      const paddle::Tensor& x,
      const std::optional<paddle::Tensor>& x_scales,
      const std::optional<paddle::Tensor>& topk_idx,
      const std::optional<paddle::Tensor>& topk_weights,
      const std::optional<paddle::Tensor>& num_tokens_per_rank,
      const paddle::Tensor& is_token_in_rank,
      const std::optional<paddle::Tensor>& num_tokens_per_expert,
      int cached_num_recv_tokens,
      const std::optional<paddle::Tensor>& cached_rank_prefix_matrix,
      const std::optional<paddle::Tensor>& cached_channel_prefix_matrix,
      int expert_alignment,
      const Config& config,
      std::optional<EventHandle>& previous_event,  // NOLINT
      bool async,
      bool allocate_on_comm_stream);

  std::tuple<paddle::Tensor,
             std::optional<paddle::Tensor>,
             std::optional<EventHandle>>
  intranode_combine_api(const paddle::Tensor& x,
                        const std::optional<paddle::Tensor>& topk_weights,
                        const paddle::Tensor& src_idx,
                        const paddle::Tensor& rank_prefix_matrix,
                        const paddle::Tensor& channel_prefix_matrix,
                        const paddle::Tensor& send_head,
                        const Config& config,
                        std::optional<EventHandle>& previous_event,  // NOLINT
                        bool async,
                        bool allocate_on_comm_stream);
};

deep_ep::detail::Tensor ConvertPaddleTensorToDetailTensor(
    const paddle::Tensor& tensor);
paddle::Tensor ConvertDetailTensorToPaddleTensor(
    const deep_ep::detail::Tensor& tensor);

std::optional<deep_ep::detail::Tensor>
ConvertOptionalPaddleTensorToDetailTensor(
    const std::optional<paddle::Tensor>& tensor);
std::optional<paddle::Tensor> ConvertOptionalDetailTensorToPaddleTensor(
    const std::optional<deep_ep::detail::Tensor>& tensor);

}  // namespace deep_ep
