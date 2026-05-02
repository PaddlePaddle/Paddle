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
#include "paddle/fluid/distributed/collective/deep_ep_xpu/include/types.h"

#include "paddle/fluid/distributed/collective/deep_ep_xpu/config.hpp"
#include "paddle/fluid/distributed/collective/deep_ep_xpu/event.hpp"
#include "paddle/fluid/distributed/collective/deep_ep_xpu/kernels/api.h"
#include "paddle/fluid/distributed/collective/deep_ep_xpu/kernels/configs.h"
#include "paddle/fluid/distributed/collective/deep_ep_xpu/kernels/exception.h"
#include "paddle/fluid/eager/api/generated/eager_generated/forwards/dygraph_functions.h"
#include "paddle/phi/api/include/tensor.h"

#include "paddle/phi/backends/xpu/xpu_context.h"
#include "paddle/phi/core/distributed/bkcl_comm_context.h"

namespace paddle::deep_ep {

struct Buffer {
  EP_STATIC_ASSERT(NUM_MAX_NVL_PEERS == 8,
                   "The number of maximum NVLink peers must be 8");

 private:
  // init
  bool init_low_latency_buffer = false;
  bool init_normal_buffer = false;

  // Low-latency mode buffer
  int low_latency_buffer_idx = 0;
  bool low_latency_mode = false;
  int m2n_ll_dispatch_workspace_idx = 0;
  int m2n_ll_combine_workspace_idx = 0;
  int m2n_ll_dispatch_recv_complete_idx = 0;
  int m2n_ll_combine_recv_complete_idx = 0;

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
  cudaStream_t comm_stream;
  phi::distributed::BKCLCommContext* comm_ctx;
  phi::XPUContext* calc_ctx;

  // After IPC/NVSHMEM synchronization, this flag will be true
  bool available = false;

  // Barrier signals
  int* barrier_signal_ptrs[NUM_MAX_NVL_PEERS] = {nullptr};
  int** barrier_signal_ptrs_gpu = nullptr;

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

  std::unique_ptr<DeepEPBuffer> ep_runtime;

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

  std::tuple<dTensor,
             std::optional<dTensor>,
             dTensor,
             dTensor,
             std::optional<EventHandle>>
  get_dispatch_layout(const dTensor& topk_idx,
                      int num_experts,
                      std::optional<EventHandle>& previous_event,  // NOLINT
                      bool async,
                      bool allocate_on_comm_stream);

  std::tuple<dTensor,
             std::optional<dTensor>,
             std::optional<dTensor>,
             std::optional<dTensor>,
             std::vector<int>,
             dTensor,
             dTensor,
             dTensor,
             dTensor,
             dTensor,
             std::optional<EventHandle>>
  intranode_dispatch(const dTensor& x,
                     const std::optional<dTensor>& x_scales,
                     const std::optional<dTensor>& topk_idx,
                     const std::optional<dTensor>& topk_weights,
                     const std::optional<dTensor>& num_tokens_per_rank,
                     const dTensor& is_token_in_rank,
                     const std::optional<dTensor>& num_tokens_per_expert,
                     int cached_num_recv_tokens,
                     const std::optional<dTensor>& cached_rank_prefix_matrix,
                     const std::optional<dTensor>& cached_channel_prefix_matrix,
                     int expert_alignment,
                     const Config& config,
                     std::optional<EventHandle>& previous_event,  // NOLINT
                     bool async,
                     bool allocate_on_comm_stream);

  std::tuple<dTensor, std::optional<dTensor>, std::optional<EventHandle>>
  intranode_combine(const dTensor& x,
                    const std::optional<dTensor>& topk_weights,
                    const dTensor& src_idx,
                    const dTensor& rank_prefix_matrix,
                    const dTensor& channel_prefix_matrix,
                    const dTensor& send_head,
                    const Config& config,
                    std::optional<EventHandle>& previous_event,  // NOLINT
                    bool async,
                    bool allocate_on_comm_stream);

  std::tuple<dTensor,
             std::optional<dTensor>,
             std::optional<dTensor>,
             std::optional<dTensor>,
             std::vector<int>,
             dTensor,
             dTensor,
             std::optional<dTensor>,
             dTensor,
             std::optional<dTensor>,
             dTensor,
             std::optional<dTensor>,
             std::optional<dTensor>,
             std::optional<dTensor>,
             std::optional<EventHandle>>
  internode_dispatch(
      const dTensor& x,
      const std::optional<dTensor>& x_scales,
      const std::optional<dTensor>& topk_idx,
      const std::optional<dTensor>& topk_weights,
      const std::optional<dTensor>& num_tokens_per_rank,
      const std::optional<dTensor>& num_tokens_per_rdma_rank,
      const dTensor& is_token_in_rank,
      const std::optional<dTensor>& num_tokens_per_expert,
      int cached_num_recv_tokens,
      int cached_num_rdma_recv_tokens,
      const std::optional<dTensor>& cached_rdma_channel_prefix_matrix,
      const std::optional<dTensor>& cached_recv_rdma_rank_prefix_sum,
      const std::optional<dTensor>& cached_gbl_channel_prefix_matrix,
      const std::optional<dTensor>& cached_recv_gbl_rank_prefix_sum,
      int expert_alignment,
      const Config& config,
      std::optional<EventHandle>& previous_event,  // NOLINT
      bool async,
      bool allocate_on_comm_stream);

  std::tuple<dTensor, std::optional<dTensor>, std::optional<EventHandle>>
  internode_combine(const dTensor& x,
                    const std::optional<dTensor>& topk_weights,
                    const dTensor& src_meta,
                    const dTensor& is_combined_token_in_rank,
                    const dTensor& rdma_channel_prefix_matrix,
                    const dTensor& rdma_rank_prefix_sum,
                    const dTensor& gbl_channel_prefix_matrix,
                    const dTensor& combined_rdma_head,
                    const dTensor& combined_nvl_head,
                    const Config& config,
                    std::optional<EventHandle>& previous_event,  // NOLINT
                    bool async,
                    bool allocate_on_comm_stream);

  void clean_low_latency_buffer(int num_max_dispatch_tokens_per_rank,
                                int hidden,
                                int num_experts);
  void clean_low_latency_two_stage_buffer(int num_max_dispatch_tokens_per_rank,
                                          int hidden,
                                          int num_experts,
                                          int num_topk,
                                          int num_ranks,
                                          bool use_fp8);
  void barrier_all();

  std::tuple<dTensor,
             std::optional<dTensor>,
             dTensor,
             dTensor,
             dTensor,
             std::optional<EventHandle>,
             std::optional<std::function<void()>>>
  low_latency_dispatch(const dTensor& x,
                       const dTensor& topk_idx,
                       const std::optional<dTensor>& expertwise_scale,
                       int num_max_dispatch_tokens_per_rank,
                       int num_experts,
                       bool use_fp8,
                       bool async,
                       bool return_recv_hook,
                       int num_per_channel);

  std::tuple<dTensor,
             std::optional<EventHandle>,
             std::optional<std::function<void()>>>
  low_latency_combine(const dTensor& x,
                      const dTensor& topk_idx,
                      const dTensor& topk_weights,
                      const dTensor& src_info,
                      const dTensor& layout_range,
                      int num_max_dispatch_tokens_per_rank,
                      int num_experts,
                      bool zero_copy,
                      bool async,
                      bool return_recv_hook,
                      const std::optional<dTensor>& out = std::nullopt);

  std::tuple<dTensor,
             std::optional<dTensor>,
             dTensor,
             dTensor,
             dTensor,
             dTensor,
             dTensor,
             dTensor,
             std::optional<EventHandle>,
             std::optional<std::function<void()>>>
  low_latency_dispatch_two_stage(const dTensor& x,
                                 const dTensor& topk_idx,
                                 const dTensor& topk_weights,
                                 int num_max_dispatch_tokens_per_rank,
                                 int num_experts,
                                 bool use_fp8,
                                 bool async,
                                 bool return_recv_hook);

  std::tuple<dTensor,
             std::optional<EventHandle>,
             std::optional<std::function<void()>>>
  low_latency_combine_two_stage(const dTensor& x,
                                const dTensor& rdma_recv_x,
                                const dTensor& topk_idx,
                                const dTensor& topk_weights,
                                const dTensor& src_info,
                                const dTensor& layout_range,
                                const dTensor& rdma_send_flags,
                                const dTensor& dispatch_rdma_recv_count,
                                int num_max_dispatch_tokens_per_rank,
                                int num_experts,
                                bool dispatch_use_fp8,
                                bool async,
                                bool return_recv_hook,
                                const std::optional<dTensor>& out);

  std::tuple<dTensor,
             std::optional<dTensor>,
             dTensor,
             dTensor,
             dTensor,
             dTensor,
             dTensor,
             dTensor,
             std::optional<EventHandle>,
             std::optional<std::function<EventHandle()>>>
  m2n_low_latency_dispatch_two_stage(const dTensor& x,
                                     const dTensor& topk_idx,
                                     const dTensor& topk_weights,
                                     int num_max_dispatch_tokens_per_rank,
                                     int num_experts,
                                     int a_start_rank,
                                     int a_num_ranks,
                                     int e_start_rank,
                                     int e_num_ranks,
                                     bool use_fp8,
                                     bool async,
                                     bool return_recv_hook);

  std::tuple<dTensor,
             std::optional<EventHandle>,
             std::optional<std::function<EventHandle()>>>
  m2n_low_latency_combine_two_stage(const dTensor& x,
                                    const dTensor& rdma_recv_x,
                                    const dTensor& topk_idx,
                                    const dTensor& topk_weights,
                                    const dTensor& src_info,
                                    const dTensor& layout_range,
                                    const dTensor& rdma_send_flags,
                                    const dTensor& dispatch_rdma_recv_count,
                                    int num_max_dispatch_tokens_per_rank,
                                    int num_experts,
                                    int a_start_rank,
                                    int a_num_ranks,
                                    int e_start_rank,
                                    int e_num_ranks,
                                    bool dispatch_use_fp8,
                                    bool async,
                                    bool return_recv_hook,
                                    const std::optional<dTensor>& out);

  std::tuple<Tensor,
             std::optional<Tensor>,
             std::optional<Tensor>,
             std::optional<Tensor>,
             std::vector<int>,
             Tensor,
             Tensor,
             std::optional<Tensor>,
             Tensor,
             std::optional<Tensor>,
             Tensor,
             std::optional<Tensor>,
             std::optional<Tensor>,
             std::optional<Tensor>,
             std::optional<EventHandle>>
  internode_dispatch_api(
      const Tensor& x,
      const std::optional<Tensor>& x_scales,
      const std::optional<Tensor>& topk_idx,
      const std::optional<Tensor>& topk_weights,
      const std::optional<Tensor>& num_tokens_per_rank,
      const std::optional<Tensor>& num_tokens_per_rdma_rank,
      const Tensor& is_token_in_rank,
      const std::optional<Tensor>& num_tokens_per_expert,
      int cached_num_recv_tokens,
      int cached_num_rdma_recv_tokens,
      const std::optional<Tensor>& cached_rdma_channel_prefix_matrix,
      const std::optional<Tensor>& cached_recv_rdma_rank_prefix_sum,
      const std::optional<Tensor>& cached_gbl_channel_prefix_matrix,
      const std::optional<Tensor>& cached_recv_gbl_rank_prefix_sum,
      int expert_alignment,
      const Config& config,
      std::optional<EventHandle>& previous_event,  // NOLINT
      bool async,
      bool allocate_on_comm_stream);

  std::tuple<Tensor, std::optional<Tensor>, std::optional<EventHandle>>
  internode_combine_api(const Tensor& x,
                        const std::optional<Tensor>& topk_weights,
                        const Tensor& src_meta,
                        const Tensor& is_combined_token_in_rank,
                        const Tensor& rdma_channel_prefix_matrix,
                        const Tensor& rdma_rank_prefix_sum,
                        const Tensor& gbl_channel_prefix_matrix,
                        const Tensor& combined_rdma_head,
                        const Tensor& combined_nvl_head,
                        const Config& config,
                        std::optional<EventHandle>& previous_event,  // NOLINT
                        bool async,
                        bool allocate_on_comm_stream);

  std::tuple<Tensor,
             std::optional<Tensor>,
             Tensor,
             Tensor,
             Tensor,
             std::optional<EventHandle>,
             std::optional<std::function<void()>>>
  low_latency_dispatch_api(const Tensor& x,
                           const Tensor& topk_idx,
                           const std::optional<Tensor>& expertwise_scale,
                           int num_max_dispatch_tokens_per_rank,
                           int num_experts,
                           bool use_fp8,
                           bool async,
                           bool return_recv_hook,
                           int num_per_channel);

  std::tuple<Tensor,
             std::optional<EventHandle>,
             std::optional<std::function<void()>>>
  low_latency_combine_api(const Tensor& x,
                          const Tensor& topk_idx,
                          const Tensor& topk_weights,
                          const Tensor& src_info,
                          const Tensor& layout_range,
                          int num_max_dispatch_tokens_per_rank,
                          int num_experts,
                          bool zero_copy,
                          bool async,
                          bool return_recv_hook,
                          const std::optional<Tensor>& out);

  std::tuple<Tensor,
             std::optional<Tensor>,
             Tensor,
             Tensor,
             Tensor,
             Tensor,
             Tensor,
             Tensor,
             std::optional<EventHandle>,
             std::optional<std::function<void()>>>
  low_latency_dispatch_two_stage_api(const Tensor& x,
                                     const Tensor& topk_idx,
                                     const Tensor& topk_weights,
                                     int num_max_dispatch_tokens_per_rank,
                                     int num_experts,
                                     bool use_fp8,
                                     bool async,
                                     bool return_recv_hook);

  std::tuple<Tensor,
             std::optional<EventHandle>,
             std::optional<std::function<void()>>>
  low_latency_combine_two_stage_api(const Tensor& x,
                                    const Tensor& rdma_recv_x,
                                    const Tensor& topk_idx,
                                    const Tensor& topk_weights,
                                    const Tensor& src_info,
                                    const Tensor& layout_range,
                                    const Tensor& rdma_send_flags,
                                    const Tensor& dispatch_rdma_recv_count,
                                    int num_max_dispatch_tokens_per_rank,
                                    int num_experts,
                                    bool dispatch_use_fp8,
                                    bool async,
                                    bool return_recv_hook,
                                    const std::optional<Tensor>& out);

  std::tuple<Tensor,
             std::optional<Tensor>,
             Tensor,
             Tensor,
             Tensor,
             Tensor,
             Tensor,
             Tensor,
             std::optional<EventHandle>,
             std::optional<std::function<EventHandle()>>>
  m2n_low_latency_dispatch_two_stage_api(const Tensor& x,
                                         const Tensor& topk_idx,
                                         const Tensor& topk_weights,
                                         int num_max_dispatch_tokens_per_rank,
                                         int num_experts,
                                         int a_start_rank,
                                         int a_num_ranks,
                                         int e_start_rank,
                                         int e_num_ranks,
                                         bool use_fp8,
                                         bool async,
                                         bool return_recv_hook);

  std::tuple<Tensor,
             std::optional<EventHandle>,
             std::optional<std::function<EventHandle()>>>
  m2n_low_latency_combine_two_stage_api(const Tensor& x,
                                        const Tensor& rdma_recv_x,
                                        const Tensor& topk_idx,
                                        const Tensor& topk_weights,
                                        const Tensor& src_info,
                                        const Tensor& layout_range,
                                        const Tensor& rdma_send_flags,
                                        const Tensor& dispatch_rdma_recv_count,
                                        int num_max_dispatch_tokens_per_rank,
                                        int num_experts,
                                        int a_start_rank,
                                        int a_num_ranks,
                                        int e_start_rank,
                                        int e_num_ranks,
                                        bool dispatch_use_fp8,
                                        bool async,
                                        bool return_recv_hook,
                                        const std::optional<Tensor>& out);

  std::tuple<Tensor,
             std::optional<Tensor>,
             Tensor,
             Tensor,
             std::optional<EventHandle>>
  get_dispatch_layout_api(const Tensor& topk_idx,
                          int num_experts,
                          std::optional<EventHandle>& previous_event,  // NOLINT
                          bool async,
                          bool allocate_on_comm_stream);

  std::tuple<Tensor,
             std::optional<Tensor>,
             std::optional<Tensor>,
             std::optional<Tensor>,
             std::vector<int>,
             Tensor,
             Tensor,
             Tensor,
             Tensor,
             Tensor,
             std::optional<EventHandle>>
  intranode_dispatch_api(
      const Tensor& x,
      const std::optional<Tensor>& x_scales,
      const std::optional<Tensor>& topk_idx,
      const std::optional<Tensor>& topk_weights,
      const std::optional<Tensor>& num_tokens_per_rank,
      const Tensor& is_token_in_rank,
      const std::optional<Tensor>& num_tokens_per_expert,
      int cached_num_recv_tokens,
      const std::optional<Tensor>& cached_rank_prefix_matrix,
      const std::optional<Tensor>& cached_channel_prefix_matrix,
      int expert_alignment,
      const Config& config,
      std::optional<EventHandle>& previous_event,  // NOLINT
      bool async,
      bool allocate_on_comm_stream);

  std::tuple<Tensor, std::optional<Tensor>, std::optional<EventHandle>>
  intranode_combine_api(const Tensor& x,
                        const std::optional<Tensor>& topk_weights,
                        const Tensor& src_idx,
                        const Tensor& rank_prefix_matrix,
                        const Tensor& channel_prefix_matrix,
                        const Tensor& send_head,
                        const Config& config,
                        std::optional<EventHandle>& previous_event,  // NOLINT
                        bool async,
                        bool allocate_on_comm_stream);
};

dTensor GetDetailTensor(const Tensor& tensor);
Tensor GetPaddleTensor(const dTensor& tensor);

std::optional<dTensor> GetDetailTensor(const std::optional<Tensor>& tensor);
std::optional<Tensor> GetPaddleTensor(const std::optional<dTensor>& tensor);

}  // namespace paddle::deep_ep
