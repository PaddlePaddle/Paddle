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

#include <cuda_runtime.h>
#include <atomic>
#include <chrono>
#include <memory>

#include "paddle/fluid/distributed/collective/flash_ep/flash_ep.hpp"
#include "paddle/fluid/distributed/collective/flash_ep/kernels/api.cuh"
#include "paddle/fluid/distributed/collective/flash_ep/kernels/configs.cuh"

#include "paddle/fluid/distributed/collective/flash_ep/include/CUDADataType.h"
#include "paddle/fluid/distributed/collective/flash_ep/include/ScalarType.h"
#include "paddle/fluid/distributed/collective/process_group_nccl.h"
#include "paddle/phi/api/include/api.h"
#include "paddle/phi/api/include/tensor_utils.h"
#include "paddle/phi/common/data_type.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/distributed/utils.h"
#include "paddle/phi/core/memory/allocation/allocator_facade.h"

namespace flash_ep {

namespace detail {
void SetAllocatorStreamForGPUContext(cudaStream_t stream,
                                     phi::GPUContext* ctx) {
  ctx->SetAllocator(paddle::memory::allocation::AllocatorFacade::Instance()
                        .GetAllocator(ctx->GetPlace(), stream)
                        .get());
}
}  // namespace detail

Buffer::Buffer(int rank,
               int num_ranks,
               int num_loop_stage,
               int64_t num_nvl_bytes,
               int64_t num_rdma_bytes,
               bool low_latency_mode,
               int context_ring_id)
    : rank(rank),
      num_ranks(num_ranks),
      num_nvl_bytes(num_nvl_bytes),
      num_rdma_bytes(num_rdma_bytes),
      low_latency_mode(low_latency_mode),
      num_loop_stage(num_loop_stage) {
  CUDA_CHECK(cudaGetDevice(&device_id));
  auto map = paddle::distributed::ProcessGroupMapFromGid::getInstance();
  paddle::distributed::ProcessGroup* pg = map->get(context_ring_id);
  const auto& place = phi::GPUPlace(device_id);
  comm_ctx =
      reinterpret_cast<paddle::distributed::ProcessGroupNCCL*>(pg)
          ->GetOrCreateCommContext(place, phi::distributed::CommType::ALLTOALL);
  comm_stream = comm_ctx->GetStream();
  calc_ctx = reinterpret_cast<phi::GPUContext*>(
      reinterpret_cast<paddle::distributed::ProcessGroupNCCL*>(pg)
          ->GetDeviceContext(place, true));
  // Task fifo memory
  int64_t fifo_bytes = sizeof(int) * NUM_MAX_FIFO_SLOTS;
  int64_t buffer_ptr_bytes = sizeof(void*) * NUM_MAX_NVL_PEERS;
  int64_t task_ptr_bytes = sizeof(int*) * NUM_MAX_NVL_PEERS;

  // Common checks
  EP_HOST_ASSERT(num_nvl_bytes % NUM_BUFFER_ALIGNMENT_BYTES == 0 &&
                 (num_nvl_bytes <= std::numeric_limits<int64_t>::max() ||
                  num_rdma_bytes == 0));
  EP_HOST_ASSERT(
      num_rdma_bytes % NUM_BUFFER_ALIGNMENT_BYTES == 0 &&
      (low_latency_mode || num_rdma_bytes <= std::numeric_limits<int>::max()));
  EP_HOST_ASSERT(0 <= rank && rank < num_ranks &&
                 (num_ranks <= NUM_MAX_NVL_PEERS * NUM_MAX_RDMA_PEERS ||
                  low_latency_mode));
  EP_HOST_ASSERT(num_ranks < NUM_MAX_NVL_PEERS ||
                 num_ranks % NUM_MAX_NVL_PEERS == 0);
  if (num_rdma_bytes > 0)
    EP_HOST_ASSERT(num_ranks > NUM_MAX_NVL_PEERS || low_latency_mode);

  // Get ranks
  // CUDA_CHECK(cudaGetDevice(&device_id));
  rdma_rank = rank / NUM_MAX_NVL_PEERS, nvl_rank = rank % NUM_MAX_NVL_PEERS;
  num_rdma_ranks = std::max(1, num_ranks / NUM_MAX_NVL_PEERS),
  num_nvl_ranks = std::min(num_ranks, NUM_MAX_NVL_PEERS);

  // Get device info
  cudaDeviceProp device_prop = {};
  CUDA_CHECK(cudaGetDeviceProperties(&device_prop, device_id));

  if (num_nvl_bytes > 0) {
    // Local IPC: alloc local memory and set local IPC handle
    CUDA_CHECK(cudaMalloc(
        &buffer_ptrs[nvl_rank],
        num_nvl_bytes + fifo_bytes + buffer_ptr_bytes + task_ptr_bytes));
    CUDA_CHECK(
        cudaIpcGetMemHandle(&ipc_handles[nvl_rank], buffer_ptrs[nvl_rank]));
    buffer_ptrs_gpu = reinterpret_cast<void**>(
        reinterpret_cast<uint8_t*>(buffer_ptrs[nvl_rank]) + num_nvl_bytes +
        fifo_bytes);

    // Set task fifo
    EP_HOST_ASSERT(NUM_MAX_FIFO_SLOTS % num_nvl_ranks == 0);
    task_fifo_ptrs[nvl_rank] = reinterpret_cast<int*>(
        reinterpret_cast<uint8_t*>(buffer_ptrs[nvl_rank]) + num_nvl_bytes);
    task_fifo_ptrs_gpu = reinterpret_cast<int**>(
        reinterpret_cast<uint8_t*>(buffer_ptrs[nvl_rank]) + num_nvl_bytes +
        fifo_bytes + buffer_ptr_bytes);

    // No need to synchronize, will do a full device sync during `sync`
    CUDA_CHECK(cudaMemsetAsync(
        buffer_ptrs[nvl_rank],
        0,
        num_nvl_bytes + fifo_bytes + buffer_ptr_bytes + task_ptr_bytes,
        comm_stream));
  }

  // Create 32 MiB workspace
  CUDA_CHECK(cudaMalloc(&workspace, NUM_WORKSPACE_BYTES));
  CUDA_CHECK(cudaMemsetAsync(workspace, 0, NUM_WORKSPACE_BYTES, comm_stream));

  // MoE counter
  CUDA_CHECK(cudaMallocHost(&dispatch_moe_recv_counter,
                            sizeof(int64_t) * num_loop_stage,
                            cudaHostAllocMapped));
  CUDA_CHECK(
      cudaHostGetDevicePointer(&dispatch_moe_recv_counter_mapped,
                               const_cast<int*>(dispatch_moe_recv_counter),
                               0));
  *dispatch_moe_recv_counter = -1;

  CUDA_CHECK(cudaMallocHost(&combine_moe_recv_counter,
                            sizeof(int64_t) * num_loop_stage,
                            cudaHostAllocMapped));
  CUDA_CHECK(
      cudaHostGetDevicePointer(&combine_moe_recv_counter_mapped,
                               const_cast<int*>(combine_moe_recv_counter),
                               0));
  *combine_moe_recv_counter = -1;

  // MoE expert-level counter
  CUDA_CHECK(
      cudaMallocHost(&moe_recv_expert_counter,
                     sizeof(int) * NUM_MAX_LOCAL_EXPERTS * num_loop_stage,
                     cudaHostAllocMapped));
  CUDA_CHECK(cudaHostGetDevicePointer(&moe_recv_expert_counter_mapped,
                                      const_cast<int*>(moe_recv_expert_counter),
                                      0));
  for (int i = 0; i < NUM_MAX_LOCAL_EXPERTS; ++i)
    moe_recv_expert_counter[i] = -1;

  // MoE RDMA-level counter
  if (num_rdma_ranks > 0) {
    CUDA_CHECK(cudaMallocHost(&dispatch_moe_recv_rdma_counter,
                              sizeof(int) * num_loop_stage,
                              cudaHostAllocMapped));
    CUDA_CHECK(cudaMallocHost(
        &dispatch_moe_recv_rdma_counter, sizeof(int), cudaHostAllocMapped));
    CUDA_CHECK(cudaHostGetDevicePointer(
        &dispatch_moe_recv_rdma_counter_mapped,
        const_cast<int*>(dispatch_moe_recv_rdma_counter),
        0));
    *dispatch_moe_recv_rdma_counter = -1;
    CUDA_CHECK(cudaMallocHost(&combine_moe_recv_rdma_counter,
                              sizeof(int) * num_loop_stage,
                              cudaHostAllocMapped));
    CUDA_CHECK(cudaHostGetDevicePointer(
        &combine_moe_recv_rdma_counter_mapped,
        const_cast<int*>(combine_moe_recv_rdma_counter),
        0));
    *combine_moe_recv_rdma_counter = -1;
  }
}

Buffer::~Buffer() noexcept(false) {
  // Synchronize
  CUDA_CHECK(cudaDeviceSynchronize());

  if (num_nvl_bytes > 0) {
    // Barrier
    intranode::barrier(
        task_fifo_ptrs_gpu, head, nvl_rank, num_nvl_ranks, comm_stream);
    move_fifo_slots();
    CUDA_CHECK(cudaDeviceSynchronize());

    // Close remote IPC
    if (is_available()) {
      for (int i = 0; i < num_nvl_ranks; ++i)
        if (i != nvl_rank) CUDA_CHECK(cudaIpcCloseMemHandle(buffer_ptrs[i]));
    }

    // Free local buffer and error flag
    CUDA_CHECK(cudaFree(buffer_ptrs[nvl_rank]));
  }

#ifdef PADDLE_WITH_NVSHMEM
  // Free NVSHMEM
  if (num_rdma_bytes > 0) {
    CUDA_CHECK(cudaDeviceSynchronize());
    internode::barrier();
    internode::free(rdma_buffer_ptr);
    internode::finalize();
  }
#endif

  // Free cuBLAS handle, workspace and MoE counter
  CUDA_CHECK(cudaFree(workspace));
  CUDA_CHECK(cudaFreeHost(const_cast<int*>(moe_recv_counter)));

  // Free chunked mode staffs
  CUDA_CHECK(cudaFreeHost(const_cast<int*>(moe_recv_expert_counter)));
}

void Buffer::move_fifo_slots(int num_slots) {
  head = (head + num_ranks * num_slots) % NUM_MAX_FIFO_SLOTS;
}

bool Buffer::is_available() const { return available; }

bool Buffer::is_internode_available() const {
#ifdef PADDLE_WITH_NVSHMEM
  return is_available() && num_ranks > NUM_MAX_NVL_PEERS;
#else
  return false;
#endif
}

int Buffer::get_num_rdma_ranks() const { return num_rdma_ranks; }

int Buffer::get_rdma_rank() const { return rdma_rank; }

int Buffer::get_root_rdma_rank(bool global) const {
  return global ? nvl_rank : 0;
}

int Buffer::get_local_device_id() const { return device_id; }

cudaStream_t Buffer::get_comm_stream() const { return comm_stream; }

#ifndef PADDLE_NO_PYTHON
pybind11::bytearray Buffer::get_local_ipc_handle() const {
  return {ipc_handles[nvl_rank].reserved, CUDA_IPC_HANDLE_SIZE};
}

pybind11::bytearray Buffer::get_local_nvshmem_unique_id() const {
#ifdef PADDLE_WITH_NVSHMEM
  EP_HOST_ASSERT(rdma_rank == 0 &&
                 "Only RDMA rank 0 can get NVSHMEM unique ID");
  auto unique_id = internode::get_unique_id();
#else
  LOG(ERROR) << "NVSHMEM is not enabled. You can enable it by setting cmake "
                "option WITH_NVSHMEM=ON.";
  std::vector<uint8_t> unique_id;
#endif
  return {reinterpret_cast<const char*>(unique_id.data()), unique_id.size()};
}

void Buffer::sync(
    const std::vector<int>& device_ids,
    const std::vector<std::optional<pybind11::bytearray>>& all_gathered_handles,
    const std::optional<pybind11::bytearray>& root_unique_id_opt) {
  EP_HOST_ASSERT(!is_available());

  // Sync IPC handles
  if (num_nvl_bytes > 0) {
    EP_HOST_ASSERT(num_ranks == static_cast<int64_t>(device_ids.size()));
    EP_HOST_ASSERT(device_ids.size() == all_gathered_handles.size());
    for (int i = 0, offset = rdma_rank * num_nvl_ranks; i < num_nvl_ranks;
         ++i) {
      EP_HOST_ASSERT(all_gathered_handles[offset + i].has_value());
      auto handle_str = std::string(all_gathered_handles[offset + i].value());
      EP_HOST_ASSERT(handle_str.size() == CUDA_IPC_HANDLE_SIZE);
      if (offset + i != rank) {
        std::memcpy(
            ipc_handles[i].reserved, handle_str.c_str(), CUDA_IPC_HANDLE_SIZE);
        CUDA_CHECK(cudaIpcOpenMemHandle(
            &buffer_ptrs[i], ipc_handles[i], cudaIpcMemLazyEnablePeerAccess));
        task_fifo_ptrs[i] = reinterpret_cast<int*>(
            reinterpret_cast<uint8_t*>(buffer_ptrs[i]) + num_nvl_bytes);
      } else {
        EP_HOST_ASSERT(std::memcmp(ipc_handles[i].reserved,
                                   handle_str.c_str(),
                                   CUDA_IPC_HANDLE_SIZE) == 0);
      }
    }

    // Copy all buffer and task pointers to GPU
    CUDA_CHECK(cudaMemcpy(buffer_ptrs_gpu,
                          buffer_ptrs,
                          sizeof(void*) * NUM_MAX_NVL_PEERS,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(task_fifo_ptrs_gpu,
                          task_fifo_ptrs,
                          sizeof(int*) * NUM_MAX_NVL_PEERS,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaDeviceSynchronize());
  }

#ifdef PADDLE_WITH_NVSHMEM
  // Sync NVSHMEM handles and allocate memory
  if (num_rdma_bytes > 0) {
    // Initialize NVSHMEM
    EP_HOST_ASSERT(root_unique_id_opt.has_value());
    std::vector<uint8_t> root_unique_id(root_unique_id_opt->size());
    auto root_unique_id_str = root_unique_id_opt->cast<std::string>();
    std::memcpy(root_unique_id.data(),
                root_unique_id_str.c_str(),
                root_unique_id_opt->size());
    auto nvshmem_rank = low_latency_mode ? rank : rdma_rank;
    auto num_nvshmem_ranks = low_latency_mode ? num_ranks : num_rdma_ranks;
    EP_HOST_ASSERT(nvshmem_rank == internode::init(root_unique_id,
                                                   nvshmem_rank,
                                                   num_nvshmem_ranks,
                                                   low_latency_mode));
    internode::barrier();

    // Allocate
    rdma_buffer_ptr =
        internode::alloc(num_rdma_bytes, NUM_BUFFER_ALIGNMENT_BYTES);

    // Clean buffer (mainly for low-latency mode)
    CUDA_CHECK(cudaMemset(rdma_buffer_ptr, 0, num_rdma_bytes));

    // Barrier
    internode::barrier();
    CUDA_CHECK(cudaDeviceSynchronize());
  }
#endif

  // Ready to use
  available = true;
}
#endif

#ifdef PADDLE_WITH_NVSHMEM
void Buffer::clear_buffer(
    const flash_ep::detail::Tensor& x,
    const std::optional<flash_ep::detail::Tensor>& x_scales,
    const std::optional<flash_ep::detail::Tensor>& topk_idx,
    const bool is_start,
    const bool is_end,
    const Config& config) {
  int hidden_int4 =
      static_cast<int>(x.size(1) * x.element_size() / sizeof(int4));
  int num_scales = 0;
  if (x_scales.has_value()) {
    EP_HOST_ASSERT(x.element_size() == 1);
    EP_HOST_ASSERT(x_scales->scalar_type() == flash_ep::detail::kFloat32);
    EP_HOST_ASSERT(x_scales->dim() > 0 && x_scales->dim() < 3 &&
                   x_scales->is_contiguous());
    num_scales = x_scales->dim() == 1 ? 1 : static_cast<int>(x_scales->size(1));
  }

  int num_topk = 0;
  if (topk_idx.has_value()) {
    num_topk = static_cast<int>(topk_idx->size(1));
  }

  const int num_channels = config.num_sms / 2;

  // Just a barrier and clean flags
  internode::clear_buffer(
      hidden_int4,
      num_scales,
      num_topk,
      num_topk,
      num_ranks,
      num_channels,
      rdma_buffer_ptr,
      config.num_max_rdma_chunked_recv_tokens,
      buffer_ptrs_gpu,
      config.num_max_nvl_chunked_recv_tokens,
      task_fifo_ptrs_gpu,
      head,
      rank,
      is_start,
      is_end,
      comm_stream,
      config.get_rdma_buffer_size_hint(hidden_int4 * sizeof(int4), num_ranks),
      num_nvl_bytes);
  move_fifo_slots(2);
}

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
Buffer::internode_dispatch(
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
    int num_experts) {
  // In dispatch, CPU will busy-wait until GPU receive tensor size metadata from
  // other ranks, which can be quite long. If users of DeepEP need to execute
  // other Python code on other threads, such as KV transfer, their code will
  // get stuck due to GIL unless we release GIL here.
  // pybind11::gil_scoped_release release;

  const int num_channels = config.num_sms / 2;
  EP_HOST_ASSERT(config.num_sms % 2 == 0);
  EP_HOST_ASSERT(0 < get_num_rdma_ranks() &&
                 get_num_rdma_ranks() <= NUM_MAX_RDMA_PEERS);

  bool cached_mode = cached_rdma_channel_prefix_matrix.has_value();
  if (cached_mode) {
    EP_HOST_ASSERT(cached_rdma_channel_prefix_matrix.has_value());
    EP_HOST_ASSERT(cached_recv_rdma_rank_prefix_sum.has_value());
    EP_HOST_ASSERT(cached_gbl_channel_prefix_matrix.has_value());
    EP_HOST_ASSERT(cached_recv_gbl_rank_prefix_sum.has_value());
  } else {
    EP_HOST_ASSERT(num_tokens_per_rank.has_value());
    EP_HOST_ASSERT(num_tokens_per_rdma_rank.has_value());
    EP_HOST_ASSERT(num_tokens_per_expert.has_value());
  }

  // Type checks
  if (cached_mode) {
    EP_HOST_ASSERT(cached_rdma_channel_prefix_matrix->scalar_type() ==
                   flash_ep::detail::kInt32);
    EP_HOST_ASSERT(cached_recv_rdma_rank_prefix_sum->scalar_type() ==
                   flash_ep::detail::kInt32);
    EP_HOST_ASSERT(cached_gbl_channel_prefix_matrix->scalar_type() ==
                   flash_ep::detail::kInt32);
    EP_HOST_ASSERT(cached_recv_gbl_rank_prefix_sum->scalar_type() ==
                   flash_ep::detail::kInt32);
  } else {
    EP_HOST_ASSERT(num_tokens_per_rank->scalar_type() ==
                   flash_ep::detail::kInt32);
    EP_HOST_ASSERT(num_tokens_per_rdma_rank->scalar_type() ==
                   flash_ep::detail::kInt32);
    EP_HOST_ASSERT(num_tokens_per_expert->scalar_type() ==
                   flash_ep::detail::kInt32);
  }

  // Shape and contiguous checks
  EP_HOST_ASSERT(x.dim() == 2 && x.is_contiguous());
  EP_HOST_ASSERT((x.size(1) * x.element_size()) % sizeof(int4) == 0);
  if (cached_mode) {
    EP_HOST_ASSERT(cached_rdma_channel_prefix_matrix->dim() == 2 &&
                   cached_rdma_channel_prefix_matrix->is_contiguous());
    EP_HOST_ASSERT(cached_rdma_channel_prefix_matrix->size(0) ==
                       num_rdma_ranks &&
                   cached_rdma_channel_prefix_matrix->size(1) == num_channels);
    EP_HOST_ASSERT(cached_recv_rdma_rank_prefix_sum->dim() == 1 &&
                   cached_recv_rdma_rank_prefix_sum->is_contiguous());
    EP_HOST_ASSERT(cached_recv_rdma_rank_prefix_sum->size(0) == num_rdma_ranks);
    EP_HOST_ASSERT(cached_gbl_channel_prefix_matrix->dim() == 2 &&
                   cached_gbl_channel_prefix_matrix->is_contiguous());
    EP_HOST_ASSERT(cached_gbl_channel_prefix_matrix->size(0) == num_ranks &&
                   cached_gbl_channel_prefix_matrix->size(1) == num_channels);
    EP_HOST_ASSERT(cached_recv_gbl_rank_prefix_sum->dim() == 1 &&
                   cached_recv_gbl_rank_prefix_sum->is_contiguous());
    EP_HOST_ASSERT(cached_recv_gbl_rank_prefix_sum->size(0) == num_ranks);
  } else {
    EP_HOST_ASSERT(num_tokens_per_rank->dim() == 1 &&
                   num_tokens_per_rank->is_contiguous());
    EP_HOST_ASSERT(num_tokens_per_rdma_rank->dim() == 1 &&
                   num_tokens_per_rdma_rank->is_contiguous());
    EP_HOST_ASSERT(num_tokens_per_expert->dim() == 1 &&
                   num_tokens_per_expert->is_contiguous());
    EP_HOST_ASSERT(num_tokens_per_rank->size(0) == num_ranks);
    EP_HOST_ASSERT(num_tokens_per_rdma_rank->size(0) == num_rdma_ranks);
    EP_HOST_ASSERT(num_tokens_per_expert->size(0) % num_ranks == 0);
    EP_HOST_ASSERT(num_tokens_per_expert->size(0) / num_ranks <=
                   NUM_MAX_LOCAL_EXPERTS);
  }

  auto num_tokens = static_cast<int>(x.size(0)),
       hidden = static_cast<int>(x.size(1)),
       hidden_int4 =
           static_cast<int>(x.size(1) * x.element_size() / sizeof(int4));

  // Top-k checks
  int num_topk = 0;
  int64_t* topk_idx_ptr = nullptr;
  float* topk_weights_ptr = nullptr;
  EP_HOST_ASSERT(topk_idx.has_value() == topk_weights.has_value());
  if (topk_idx.has_value()) {
    num_topk = static_cast<int>(topk_idx->size(1));
    EP_HOST_ASSERT(topk_idx->dim() == 2 && topk_idx->is_contiguous());
    EP_HOST_ASSERT(topk_weights->dim() == 2 && topk_weights->is_contiguous());
    EP_HOST_ASSERT(num_tokens == topk_idx->size(0) &&
                   num_tokens == topk_weights->size(0));
    EP_HOST_ASSERT(num_topk == topk_weights->size(1));
    EP_HOST_ASSERT(topk_weights->scalar_type() == flash_ep::detail::kFloat32);
    topk_idx_ptr = topk_idx->data_ptr<int64_t>();
    topk_weights_ptr = topk_weights->data_ptr<float>();
  } else {
    num_experts = cached_mode ? 0 : num_experts;
  }
  int num_local_experts = num_experts / num_ranks;

  // FP8 scales checks
  float* x_scales_ptr = nullptr;
  int num_scales = 0;
  if (x_scales.has_value()) {
    EP_HOST_ASSERT(x.element_size() == 1);
    EP_HOST_ASSERT(x_scales->scalar_type() == flash_ep::detail::kFloat32);
    EP_HOST_ASSERT(x_scales->dim() > 0 && x_scales->dim() < 3 &&
                   x_scales->is_contiguous());
    EP_HOST_ASSERT(x_scales->size(0) == num_tokens);
    num_scales = x_scales->dim() == 1 ? 1 : static_cast<int>(x_scales->size(1));
    x_scales_ptr = x_scales->data_ptr<float>();
  }

  // Allocate all tensors on comm stream if set
  // NOTES: do not allocate tensors upfront!
  auto compute_stream = calc_ctx->stream();
  if (allocate_on_comm_stream) {
    EP_HOST_ASSERT(previous_event.has_value() && async);
    flash_ep::detail::SetAllocatorStreamForGPUContext(comm_stream, calc_ctx);
  }

  // Wait previous tasks to be finished
  if (previous_event.has_value()) {
    stream_wait(comm_stream, previous_event.value());
  } else {
    stream_wait(comm_stream, compute_stream);
  }

  // Create handles (only return for non-cached mode)
  int num_recv_tokens = -1, num_rdma_recv_tokens = -1;
  auto rdma_channel_prefix_matrix = flash_ep::detail::Tensor();
  auto recv_rdma_rank_prefix_sum = flash_ep::detail::Tensor();
  auto gbl_channel_prefix_matrix = flash_ep::detail::Tensor();
  auto recv_gbl_rank_prefix_sum = flash_ep::detail::Tensor();
  std::vector<int> num_recv_tokens_per_expert_list;

  // Barrier or send sizes
  EP_HOST_ASSERT(cached_mode);
  num_recv_tokens = cached_num_recv_tokens;
  num_rdma_recv_tokens = cached_num_rdma_recv_tokens;
  rdma_channel_prefix_matrix = cached_rdma_channel_prefix_matrix.value();
  recv_rdma_rank_prefix_sum = cached_recv_rdma_rank_prefix_sum.value();
  gbl_channel_prefix_matrix = cached_gbl_channel_prefix_matrix.value();
  recv_gbl_rank_prefix_sum = cached_recv_gbl_rank_prefix_sum.value();

  // Just a barrier and clean flags
  internode::cached_notify(
      hidden_int4,
      num_scales,
      num_topk,
      num_topk,
      num_ranks,
      num_channels,
      0,
      nullptr,
      nullptr,
      nullptr,
      nullptr,
      rdma_buffer_ptr,
      config.num_max_rdma_chunked_recv_tokens,
      buffer_ptrs_gpu,
      config.num_max_nvl_chunked_recv_tokens,
      task_fifo_ptrs_gpu,
      head,
      rank,
      comm_stream,
      config.get_rdma_buffer_size_hint(hidden_int4 * sizeof(int4), num_ranks),
      num_nvl_bytes,
      true,
      low_latency_mode);
  move_fifo_slots(2);

  // Allocate new tensors
  auto recv_x = ConvertPaddleTensorToDetailTensor(paddle::experimental::empty(
      {num_recv_tokens, hidden}, x.dtype(), x.place()));
  auto recv_topk_idx = std::optional<flash_ep::detail::Tensor>(),
       recv_topk_weights = std::optional<flash_ep::detail::Tensor>(),
       recv_x_scales = std::optional<flash_ep::detail::Tensor>();
  auto recv_rdma_channel_prefix_matrix =
      std::optional<flash_ep::detail::Tensor>();
  auto send_rdma_head = std::optional<flash_ep::detail::Tensor>();
  auto send_nvl_head = std::optional<flash_ep::detail::Tensor>();
  auto recv_src_meta =
      ConvertPaddleTensorToDetailTensor(paddle::experimental::empty(
          {num_recv_tokens, internode::get_details_source_meta_bytes()},
          phi::DataType::INT8,
          phi::GPUPlace(device_id)));
  auto recv_gbl_channel_prefix_matrix = ConvertPaddleTensorToDetailTensor(
      paddle::experimental::empty({num_ranks, num_channels},
                                  phi::DataType::INT32,
                                  phi::GPUPlace(device_id)));
  if (!cached_mode) {
    recv_rdma_channel_prefix_matrix = ConvertPaddleTensorToDetailTensor(
        paddle::experimental::empty({num_rdma_ranks, num_channels},
                                    phi::DataType::INT32,
                                    phi::GPUPlace(device_id)));
    send_rdma_head = ConvertPaddleTensorToDetailTensor(
        paddle::experimental::empty({num_tokens, num_rdma_ranks},
                                    phi::DataType::INT32,
                                    phi::GPUPlace(device_id)));
    send_nvl_head = ConvertPaddleTensorToDetailTensor(
        paddle::experimental::empty({num_rdma_recv_tokens, NUM_MAX_NVL_PEERS},
                                    phi::DataType::INT32,
                                    phi::GPUPlace(device_id)));
  }

  // Assign pointers
  int64_t* recv_topk_idx_ptr = nullptr;
  float* recv_topk_weights_ptr = nullptr;
  float* recv_x_scales_ptr = nullptr;
  if (topk_idx.has_value()) {
    recv_topk_idx =
        ConvertPaddleTensorToDetailTensor(paddle::experimental::empty(
            {num_recv_tokens, num_topk}, topk_idx->dtype(), topk_idx->place()));
    recv_topk_weights = ConvertPaddleTensorToDetailTensor(
        paddle::experimental::empty({num_recv_tokens, num_topk},
                                    topk_weights->dtype(),
                                    topk_weights->place()));
    recv_topk_idx_ptr = recv_topk_idx->data_ptr<int64_t>();
    recv_topk_weights_ptr = recv_topk_weights->data_ptr<float>();
  }
  if (x_scales.has_value()) {
    recv_x_scales =
        x_scales->dim() == 1
            ? ConvertPaddleTensorToDetailTensor(paddle::experimental::empty(
                  {num_recv_tokens}, x_scales->dtype(), x_scales->place()))
            : ConvertPaddleTensorToDetailTensor(
                  paddle::experimental::empty({num_recv_tokens, num_scales},
                                              x_scales->dtype(),
                                              x_scales->place()));
    recv_x_scales_ptr = recv_x_scales->data_ptr<float>();
  }

  bool asymmetric_mode = asymm_send_combine_schedule_map.has_value();
  if (asymmetric_mode) {
    EP_HOST_ASSERT(cached_mode);
    EP_HOST_ASSERT(asymm_recv_rdma_counter_loop_prefix_sum.has_value());
    EP_HOST_ASSERT(asymm_recv_rdma_rank_prefix_sum.has_value());
    EP_HOST_ASSERT(asymm_recv_rdma_channel_prefix_matrix.has_value());
    EP_HOST_ASSERT(asymm_send_rdma_head.has_value());
    EP_HOST_ASSERT(asymm_send_nvl_head.has_value());
    EP_HOST_ASSERT(asymm_aggregated_nvl_head.has_value());
  }
  // Launch data dispatch
  // NOTES: the buffer size checks are moved into the `.cu` file
  internode::dispatch(
      recv_x.data_ptr(),
      recv_x_scales_ptr,
      recv_topk_idx_ptr,
      recv_topk_weights_ptr,
      recv_src_meta.data_ptr(),
      x.data_ptr(),
      x_scales_ptr,
      topk_idx_ptr,
      topk_weights_ptr,
      cached_mode ? nullptr : send_rdma_head->data_ptr<int>(),
      cached_mode ? nullptr : send_nvl_head->data_ptr<int>(),
      cached_mode ? nullptr : recv_rdma_channel_prefix_matrix->data_ptr<int>(),
      recv_gbl_channel_prefix_matrix.data_ptr<int>(),
      rdma_channel_prefix_matrix.data_ptr<int>(),
      recv_rdma_rank_prefix_sum.data_ptr<int>(),
      gbl_channel_prefix_matrix.data_ptr<int>(),
      recv_gbl_rank_prefix_sum.data_ptr<int>(),
      num_tokens,
      hidden_int4,
      num_scales,
      num_topk,
      num_experts,
      is_token_in_rank.data_ptr<bool>(),
      rdma_buffer_ptr,
      config.num_max_rdma_chunked_send_tokens,
      config.num_max_rdma_chunked_recv_tokens,
      buffer_ptrs_gpu,
      config.num_max_nvl_chunked_send_tokens,
      config.num_max_nvl_chunked_recv_tokens,
      rank,
      num_ranks,
      cached_mode,
      comm_stream,
      num_channels,
      low_latency_mode,
      asymmetric_mode,
      asymmetric_mode ? asymm_send_combine_schedule_map->data_ptr<int>()
                      : nullptr,
      asymmetric_mode ? asymm_recv_rdma_counter_loop_prefix_sum->data_ptr<int>()
                      : nullptr,
      asymmetric_mode ? asymm_recv_rdma_rank_prefix_sum->data_ptr<int>()
                      : nullptr,
      asymmetric_mode ? asymm_recv_rdma_channel_prefix_matrix->data_ptr<int>()
                      : nullptr,
      asymmetric_mode ? asymm_send_rdma_head->data_ptr<int>() : nullptr,
      asymmetric_mode ? asymm_send_nvl_head->data_ptr<int>() : nullptr,
      asymmetric_mode ? asymm_aggregated_nvl_head->data_ptr<int>() : nullptr);

  // Wait streams
  std::optional<EventHandle> event;
  if (async) {
    event = EventHandle(comm_stream);
    for (auto& t : {x,
                    is_token_in_rank,
                    recv_x,
                    rdma_channel_prefix_matrix,
                    recv_rdma_rank_prefix_sum,
                    gbl_channel_prefix_matrix,
                    recv_gbl_rank_prefix_sum,
                    recv_src_meta,
                    recv_gbl_channel_prefix_matrix}) {
      t.record_stream(comm_stream);
      if (allocate_on_comm_stream) t.record_stream(compute_stream);
    }
    for (auto& to : {x_scales,
                     topk_idx,
                     topk_weights,
                     num_tokens_per_rank,
                     num_tokens_per_rdma_rank,
                     num_tokens_per_expert,
                     cached_rdma_channel_prefix_matrix,
                     cached_recv_rdma_rank_prefix_sum,
                     cached_gbl_channel_prefix_matrix,
                     cached_recv_gbl_rank_prefix_sum,
                     recv_topk_idx,
                     recv_topk_weights,
                     recv_x_scales,
                     recv_rdma_channel_prefix_matrix,
                     send_rdma_head,
                     send_nvl_head}) {
      to.has_value() ? to->record_stream(comm_stream) : void();
      if (allocate_on_comm_stream)
        to.has_value() ? to->record_stream(compute_stream) : void();
    }
  } else {
    stream_wait(compute_stream, comm_stream);
  }

  // Switch back compute stream
  if (allocate_on_comm_stream) {
    flash_ep::detail::SetAllocatorStreamForGPUContext(compute_stream, calc_ctx);
  }

  // Return values
  return {recv_x,
          recv_x_scales,
          recv_topk_idx,
          recv_topk_weights,
          num_recv_tokens_per_expert_list,
          rdma_channel_prefix_matrix,
          gbl_channel_prefix_matrix,
          recv_rdma_channel_prefix_matrix,
          recv_rdma_rank_prefix_sum,
          recv_gbl_channel_prefix_matrix,
          recv_gbl_rank_prefix_sum,
          recv_src_meta,
          send_rdma_head,
          send_nvl_head,
          event};
}

std::tuple<std::optional<flash_ep::detail::Tensor>,
           std::optional<flash_ep::detail::Tensor>,
           std::optional<EventHandle>>
Buffer::internode_combine(
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
    bool allocate_on_comm_stream) {
  const int num_channels = config.num_sms / 2;
  EP_HOST_ASSERT(config.num_sms % 2 == 0);

  // Shape and contiguous checks
  EP_HOST_ASSERT(x.dim() == 2 && x.is_contiguous());
  EP_HOST_ASSERT(rdma_channel_prefix_matrix.dim() == 2 &&
                 rdma_channel_prefix_matrix.is_contiguous() &&
                 rdma_channel_prefix_matrix.scalar_type() ==
                     flash_ep::detail::kInt32);
  EP_HOST_ASSERT(
      rdma_rank_prefix_sum.dim() == 1 && rdma_rank_prefix_sum.is_contiguous() &&
      rdma_rank_prefix_sum.scalar_type() == flash_ep::detail::kInt32);
  EP_HOST_ASSERT(gbl_channel_prefix_matrix.dim() == 2 &&
                 gbl_channel_prefix_matrix.is_contiguous() &&
                 gbl_channel_prefix_matrix.scalar_type() ==
                     flash_ep::detail::kInt32);
  EP_HOST_ASSERT(combined_rdma_head.dim() == 2 &&
                 combined_rdma_head.is_contiguous() &&
                 combined_rdma_head.scalar_type() == flash_ep::detail::kInt32);
  EP_HOST_ASSERT(combined_nvl_head.dim() == 2 &&
                 combined_nvl_head.is_contiguous() &&
                 combined_nvl_head.scalar_type() == flash_ep::detail::kInt32);

  auto num_tokens = static_cast<int>(x.size(0)),
       hidden = static_cast<int>(x.size(1)),
       hidden_int4 =
           static_cast<int>(x.size(1) * x.element_size() / sizeof(int4));
  auto num_combined_tokens = static_cast<int>(combined_rdma_head.size(0));
  EP_HOST_ASSERT((hidden * x.element_size()) % sizeof(int4) == 0);
  EP_HOST_ASSERT(rdma_channel_prefix_matrix.size(0) == num_rdma_ranks &&
                 rdma_channel_prefix_matrix.size(1) == num_channels);
  EP_HOST_ASSERT(rdma_rank_prefix_sum.size(0) == num_rdma_ranks);
  EP_HOST_ASSERT(gbl_channel_prefix_matrix.size(0) == num_ranks &&
                 gbl_channel_prefix_matrix.size(1) == num_channels);
  EP_HOST_ASSERT(combined_rdma_head.dim() == 2 &&
                 combined_rdma_head.size(0) == num_combined_tokens &&
                 combined_rdma_head.size(1) == num_rdma_ranks);
  EP_HOST_ASSERT(combined_nvl_head.dim() == 2 &&
                 combined_nvl_head.size(1) == NUM_MAX_NVL_PEERS);

  // Allocate all tensors on comm stream if set
  // NOTES: do not allocate tensors upfront!
  auto compute_stream = calc_ctx->stream();
  if (allocate_on_comm_stream) {
    EP_HOST_ASSERT(previous_event.has_value() && async);
    flash_ep::detail::SetAllocatorStreamForGPUContext(comm_stream, calc_ctx);
  }

  // Wait previous tasks to be finished
  if (previous_event.has_value()) {
    stream_wait(comm_stream, previous_event.value());
  } else {
    stream_wait(comm_stream, compute_stream);
  }

  if (combined_topk_weights.has_value()) {
    EP_HOST_ASSERT(combined_x.has_value());
  }

  if (combined_x.has_value()) {
    EP_HOST_ASSERT(combined_topk_weights.has_value() ||
                   !topk_weights.has_value());
  }

  // Top-k checks
  int num_topk = 0;
  float* topk_weights_ptr = nullptr;
  float* combined_topk_weights_ptr = nullptr;
  auto in_combined_topk_weights = std::optional<flash_ep::detail::Tensor>();
  auto res_combined_topk_weights = std::optional<flash_ep::detail::Tensor>();
  if (topk_weights.has_value()) {
    EP_HOST_ASSERT(topk_weights->dim() == 2 && topk_weights->is_contiguous());
    EP_HOST_ASSERT(topk_weights->size(0) == num_tokens);
    EP_HOST_ASSERT(topk_weights->scalar_type() == flash_ep::detail::kFloat32);
    num_topk = static_cast<int>(topk_weights->size(1));
    topk_weights_ptr = topk_weights->data_ptr<float>();
    if (!combined_topk_weights.has_value()) {
      in_combined_topk_weights = ConvertPaddleTensorToDetailTensor(
          paddle::experimental::empty({num_combined_tokens, num_topk},
                                      topk_weights->dtype(),
                                      topk_weights->place()));
      res_combined_topk_weights = in_combined_topk_weights;
    } else {
      EP_HOST_ASSERT(combined_topk_weights->dim() == 2 &&
                     combined_topk_weights->is_contiguous());
      EP_HOST_ASSERT(combined_topk_weights->dtype() == topk_weights->dtype());
      in_combined_topk_weights = combined_topk_weights;
    }
    combined_topk_weights_ptr = in_combined_topk_weights->data_ptr<float>();
  }

  // Extra check for avoid-dead-lock design
  EP_HOST_ASSERT(config.num_max_nvl_chunked_recv_tokens % num_rdma_ranks == 0);
  EP_HOST_ASSERT(config.num_max_nvl_chunked_send_tokens <=
                 config.num_max_nvl_chunked_recv_tokens / num_rdma_ranks);

  // Launch barrier and reset queue head and tail
  internode::cached_notify(
      hidden_int4,
      0,
      0,
      num_topk,
      num_ranks,
      num_channels,
      num_combined_tokens,
      combined_rdma_head.data_ptr<int>(),
      rdma_channel_prefix_matrix.data_ptr<int>(),
      rdma_rank_prefix_sum.data_ptr<int>(),
      combined_nvl_head.data_ptr<int>(),
      rdma_buffer_ptr,
      config.num_max_rdma_chunked_recv_tokens,
      buffer_ptrs_gpu,
      config.num_max_nvl_chunked_recv_tokens,
      task_fifo_ptrs_gpu,
      head,
      rank,
      comm_stream,
      config.get_rdma_buffer_size_hint(hidden_int4 * sizeof(int4), num_ranks),
      num_nvl_bytes,
      false,
      low_latency_mode);
  move_fifo_slots(2);

  // Launch data combine
  bool inplace_float_combine = false;
  auto in_combined_x = std::optional<flash_ep::detail::Tensor>();
  auto res_combined_x = std::optional<flash_ep::detail::Tensor>();
  if (combined_x.has_value()) {
    inplace_float_combine = true;
    in_combined_x = combined_x;
    EP_HOST_ASSERT(in_combined_x->dim() == 2 &&
                   in_combined_x->is_contiguous() &&
                   in_combined_x->scalar_type() == flash_ep::detail::kFloat32 &&
                   in_combined_x->size(1) == hidden);
  } else {
    in_combined_x =
        ConvertPaddleTensorToDetailTensor(paddle::experimental::empty(
            {num_combined_tokens, hidden}, x.dtype(), x.place()));
    res_combined_x = in_combined_x;
  }
  internode::combine(
      flash_ep::detail::ScalarTypeToCudaDataType(x.scalar_type()),
      in_combined_x->data_ptr(),
      combined_topk_weights_ptr,
      x.data_ptr(),
      topk_weights_ptr,
      combined_rdma_head.data_ptr<int>(),
      combined_nvl_head.data_ptr<int>(),
      rdma_channel_prefix_matrix.data_ptr<int>(),
      rdma_rank_prefix_sum.data_ptr<int>(),
      gbl_channel_prefix_matrix.data_ptr<int>(),
      num_tokens,
      num_combined_tokens,
      hidden,
      num_topk,
      rdma_buffer_ptr,
      config.num_max_rdma_chunked_send_tokens,
      config.num_max_rdma_chunked_recv_tokens,
      buffer_ptrs_gpu,
      config.num_max_nvl_chunked_send_tokens,
      config.num_max_nvl_chunked_recv_tokens,
      rank,
      num_ranks,
      comm_stream,
      num_channels,
      low_latency_mode,
      inplace_float_combine);

  // Wait streams
  std::optional<EventHandle> event;
  if (async) {
    event = EventHandle(comm_stream);
    for (auto& t : {x,
                    rdma_channel_prefix_matrix,
                    rdma_rank_prefix_sum,
                    gbl_channel_prefix_matrix,
                    combined_rdma_head,
                    combined_nvl_head}) {
      t.record_stream(comm_stream);
      if (allocate_on_comm_stream) t.record_stream(compute_stream);
    }
    for (auto& to : {topk_weights,
                     combined_topk_weights,
                     in_combined_x,
                     res_combined_x,
                     in_combined_topk_weights,
                     res_combined_topk_weights}) {
      to.has_value() ? to->record_stream(comm_stream) : void();
      if (allocate_on_comm_stream)
        to.has_value() ? to->record_stream(compute_stream) : void();
    }
  } else {
    stream_wait(compute_stream, comm_stream);
  }

  // Switch back compute stream
  if (allocate_on_comm_stream) {
    flash_ep::detail::SetAllocatorStreamForGPUContext(compute_stream, calc_ctx);
  }

  // Return values
  return {res_combined_x, res_combined_topk_weights, event};
}

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
Buffer::internode_fused_notify(
    const flash_ep::detail::Tensor& x,
    const std::optional<flash_ep::detail::Tensor>& x_scales,
    const std::optional<flash_ep::detail::Tensor>& topk_idx,
    const std::optional<flash_ep::detail::Tensor>& dispatch_num_tokens_per_rank,
    const std::optional<flash_ep::detail::Tensor>&
        dispatch_num_tokens_per_rdma_rank,
    const std::optional<flash_ep::detail::Tensor>&
        dispatch_num_tokens_per_expert,
    const flash_ep::detail::Tensor& dispatch_is_token_in_rank,
    const std::optional<flash_ep::detail::Tensor>& combine_num_tokens_per_rank,
    const std::optional<flash_ep::detail::Tensor>&
        combine_num_tokens_per_rdma_rank,
    const flash_ep::detail::Tensor& combine_is_token_in_rank,
    int expert_alignment,
    const Config& config) {
  const int num_channels = config.num_sms / 2;
  EP_HOST_ASSERT(config.num_sms % 2 == 0);
  EP_HOST_ASSERT(0 < get_num_rdma_ranks() &&
                 get_num_rdma_ranks() <= NUM_MAX_RDMA_PEERS);
  EP_HOST_ASSERT(dispatch_num_tokens_per_rank->size(0) == num_loop_stage);
  EP_HOST_ASSERT(dispatch_num_tokens_per_rdma_rank->size(0) == num_loop_stage);
  EP_HOST_ASSERT(combine_num_tokens_per_rank->size(0) == num_loop_stage);

  auto num_tokens = static_cast<int>(x.size(0)),
       hidden = static_cast<int>(x.size(1)),
       hidden_int4 =
           static_cast<int>(x.size(1) * x.element_size() / sizeof(int4));

  // Top-k checks
  int num_topk = 0;
  int64_t* topk_idx_ptr = nullptr;
  if (topk_idx.has_value()) {
    num_topk = static_cast<int>(topk_idx->size(1));
    EP_HOST_ASSERT(topk_idx->dim() == 2 && topk_idx->is_contiguous());
    EP_HOST_ASSERT(num_tokens == topk_idx->size(0));
    EP_HOST_ASSERT(num_topk == topk_idx->size(1));
    topk_idx_ptr = topk_idx->data_ptr<int64_t>();
  }
  auto num_experts = static_cast<int>(dispatch_num_tokens_per_expert->size(1));
  int num_local_experts = num_experts / num_ranks;

  // FP8 scales checks
  float* x_scales_ptr = nullptr;
  int num_scales = 0;
  if (x_scales.has_value()) {
    EP_HOST_ASSERT(x.element_size() == 1);
    EP_HOST_ASSERT(x_scales->scalar_type() == flash_ep::detail::kFloat32);
    EP_HOST_ASSERT(x_scales->dim() > 0 && x_scales->dim() < 3 &&
                   x_scales->is_contiguous());
    EP_HOST_ASSERT(x_scales->size(0) == num_tokens);
    num_scales = x_scales->dim() == 1 ? 1 : static_cast<int>(x_scales->size(1));
    x_scales_ptr = x_scales->data_ptr<float>();
  }

  // notify dispatch
  auto dispatch_rdma_channel_prefix_matrix =
      ConvertPaddleTensorToDetailTensor(paddle::experimental::empty(
          {num_loop_stage, num_rdma_ranks, num_channels},
          phi::DataType::INT32,
          phi::GPUPlace(device_id)));
  auto dispatch_recv_rdma_rank_prefix_sum = ConvertPaddleTensorToDetailTensor(
      paddle::experimental::empty({num_loop_stage, num_rdma_ranks},
                                  phi::DataType::INT32,
                                  phi::GPUPlace(device_id)));
  auto dispatch_gbl_channel_prefix_matrix = ConvertPaddleTensorToDetailTensor(
      paddle::experimental::empty({num_loop_stage, num_ranks, num_channels},
                                  phi::DataType::INT32,
                                  phi::GPUPlace(device_id)));
  auto dispatch_recv_gbl_rank_prefix_sum = ConvertPaddleTensorToDetailTensor(
      paddle::experimental::empty({num_loop_stage, num_ranks},
                                  phi::DataType::INT32,
                                  phi::GPUPlace(device_id)));

  // notify combine
  auto combine_rdma_channel_prefix_matrix =
      ConvertPaddleTensorToDetailTensor(paddle::experimental::empty(
          {num_loop_stage, num_rdma_ranks, num_channels},
          phi::DataType::INT32,
          phi::GPUPlace(device_id)));
  auto combine_recv_rdma_rank_prefix_sum = ConvertPaddleTensorToDetailTensor(
      paddle::experimental::empty({num_loop_stage, num_rdma_ranks},
                                  phi::DataType::INT32,
                                  phi::GPUPlace(device_id)));
  auto combine_gbl_channel_prefix_matrix = ConvertPaddleTensorToDetailTensor(
      paddle::experimental::empty({num_loop_stage, num_ranks, num_channels},
                                  phi::DataType::INT32,
                                  phi::GPUPlace(device_id)));
  auto combine_recv_gbl_rank_prefix_sum = ConvertPaddleTensorToDetailTensor(
      paddle::experimental::empty({num_loop_stage, num_ranks},
                                  phi::DataType::INT32,
                                  phi::GPUPlace(device_id)));

  auto combine_recv_rdma_channel_prefix_matrix =
      ConvertPaddleTensorToDetailTensor(paddle::experimental::empty(
          {num_loop_stage, num_rdma_ranks, num_channels},
          phi::DataType::INT32,
          phi::GPUPlace(device_id)));
  auto combine_recv_gbl_channel_prefix_matrix =
      ConvertPaddleTensorToDetailTensor(
          paddle::experimental::empty({num_loop_stage, num_ranks, num_channels},
                                      phi::DataType::INT32,
                                      phi::GPUPlace(device_id)));

  auto combine_send_rdma_head =
      ConvertPaddleTensorToDetailTensor(paddle::experimental::empty(
          {num_loop_stage, num_tokens, num_ranks / NUM_MAX_NVL_PEERS},
          phi::DataType::INT32,
          phi::GPUPlace(device_id)));
  auto combine_send_nvl_head =
      ConvertPaddleTensorToDetailTensor(paddle::experimental::empty(
          {num_loop_stage, num_tokens, num_ranks / NUM_MAX_NVL_PEERS, 8},
          phi::DataType::INT32,
          phi::GPUPlace(device_id)));
  auto combine_num_rdma_recv_tokens_cumsum =
      ConvertPaddleTensorToDetailTensor(paddle::experimental::empty(
          {num_loop_stage}, phi::DataType::INT32, phi::GPUPlace(device_id)));

  auto compute_stream = calc_ctx->stream();
  stream_wait(comm_stream, compute_stream);

  // notify dispatch
  for (int s = 0; s < num_loop_stage; ++s) {
    dispatch_moe_recv_counter[s] = -1;
    dispatch_moe_recv_rdma_counter[s] = -1;
    combine_moe_recv_counter[s] = -1;
    combine_moe_recv_rdma_counter[s] = -1;
    for (int i = 0; i < num_local_experts; ++i)
      moe_recv_expert_counter[s * num_local_experts + i] = -1;
  }

  internode::fused_notify(
      dispatch_num_tokens_per_rank->data_ptr<int>(),
      dispatch_moe_recv_counter_mapped,
      combine_num_tokens_per_rank->data_ptr<int>(),
      combine_moe_recv_counter_mapped,
      num_ranks,
      dispatch_num_tokens_per_rdma_rank->data_ptr<int>(),
      dispatch_moe_recv_rdma_counter_mapped,
      combine_num_tokens_per_rdma_rank->data_ptr<int>(),
      combine_moe_recv_rdma_counter_mapped,
      dispatch_num_tokens_per_expert->data_ptr<int>(),
      moe_recv_expert_counter_mapped,
      num_experts,
      dispatch_is_token_in_rank.data_ptr<bool>(),
      combine_is_token_in_rank.data_ptr<bool>(),
      num_tokens,
      num_channels,
      hidden_int4,
      num_scales,
      num_topk,
      expert_alignment,
      dispatch_rdma_channel_prefix_matrix.data_ptr<int>(),
      dispatch_recv_rdma_rank_prefix_sum.data_ptr<int>(),
      dispatch_gbl_channel_prefix_matrix.data_ptr<int>(),
      dispatch_recv_gbl_rank_prefix_sum.data_ptr<int>(),
      combine_rdma_channel_prefix_matrix.data_ptr<int>(),
      combine_recv_rdma_rank_prefix_sum.data_ptr<int>(),
      combine_gbl_channel_prefix_matrix.data_ptr<int>(),
      combine_recv_gbl_rank_prefix_sum.data_ptr<int>(),
      combine_recv_rdma_channel_prefix_matrix.data_ptr<int>(),
      combine_recv_gbl_channel_prefix_matrix.data_ptr<int>(),
      combine_send_rdma_head.data_ptr<int>(),
      combine_send_nvl_head.data_ptr<int>(),
      rdma_buffer_ptr,
      config.num_max_rdma_chunked_recv_tokens,
      buffer_ptrs_gpu,
      config.num_max_nvl_chunked_recv_tokens,
      task_fifo_ptrs_gpu,
      head,
      rank,
      comm_stream,
      config.get_rdma_buffer_size_hint(hidden_int4 * sizeof(int4), num_ranks),
      num_nvl_bytes,
      low_latency_mode,
      num_loop_stage,
      combine_num_rdma_recv_tokens_cumsum.data_ptr<int>());
  move_fifo_slots(3);

  internode::fused_notify_combine_post_step(
      num_ranks,
      num_channels,
      num_loop_stage,
      combine_recv_gbl_rank_prefix_sum.data_ptr<int>(),
      combine_rdma_channel_prefix_matrix.data_ptr<int>(),
      combine_gbl_channel_prefix_matrix.data_ptr<int>(),
      combine_recv_rdma_channel_prefix_matrix.data_ptr<int>(),
      combine_recv_gbl_channel_prefix_matrix.data_ptr<int>(),
      rdma_buffer_ptr,
      buffer_ptrs_gpu,
      task_fifo_ptrs_gpu,
      head,
      rank,
      comm_stream,
      low_latency_mode);
  move_fifo_slots(3);

  // Synchronize total received tokens and tokens per expert
  auto start_time = std::chrono::high_resolution_clock::now();
  while (true) {
    bool ready = true;
    for (int s = 0; s < num_loop_stage; ++s) {
      ready = ready && (combine_moe_recv_counter[s] >= 0) &&
              (combine_moe_recv_rdma_counter[s] >= 0);
    }
    for (int s = 0; s < num_loop_stage && ready; ++s) {
      ready &= (dispatch_moe_recv_counter[s] >= 0) &&
               (dispatch_moe_recv_rdma_counter[s] >= 0);
      for (int i = 0; i < num_local_experts && ready; ++i)
        ready &= moe_recv_expert_counter[s * num_local_experts + i] >= 0;
    }

    if (ready) break;

    // Timeout check
    if (std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::high_resolution_clock::now() - start_time)
            .count() > NUM_CPU_TIMEOUT_SECS) {
      LOG(INFO) << "Global rank: " << rank
                << ", combine_moe_recv_counter: " << combine_moe_recv_counter[0]
                << ", combine_moe_recv_rdma_counter: "
                << combine_moe_recv_rdma_counter[0]
                << ", dispatch_moe_recv_counter: "
                << dispatch_moe_recv_counter[0]
                << ", dispatch_moe_recv_rdma_counter: "
                << dispatch_moe_recv_rdma_counter[0]
                << ", moe_recv_expert_counter: " << moe_recv_expert_counter[0];
      throw std::runtime_error(
          "FlashEP error: timeout (internode_fused_notify CPU)");
    }
  }
  std::vector<int> combine_num_recv_tokens(
      combine_moe_recv_counter, combine_moe_recv_counter + num_loop_stage);
  std::vector<int> combine_num_rdma_recv_tokens(
      combine_moe_recv_rdma_counter,
      combine_moe_recv_rdma_counter + num_loop_stage);
  std::vector<int> dispatch_num_recv_tokens(
      dispatch_moe_recv_counter, dispatch_moe_recv_counter + num_loop_stage);
  std::vector<int> dispatch_num_rdma_recv_tokens(
      dispatch_moe_recv_rdma_counter,
      dispatch_moe_recv_rdma_counter + num_loop_stage);

  std::vector<std::vector<int>> num_recv_tokens_per_expert_list;
  num_recv_tokens_per_expert_list.reserve(num_loop_stage);
  for (int s = 0; s < num_loop_stage; ++s) {
    num_recv_tokens_per_expert_list.emplace_back(
        moe_recv_expert_counter + s * num_local_experts,
        moe_recv_expert_counter + (s + 1) * num_local_experts);
  }

  stream_wait(compute_stream, comm_stream);

  return {num_recv_tokens_per_expert_list,
          dispatch_num_recv_tokens,
          dispatch_num_rdma_recv_tokens,
          dispatch_rdma_channel_prefix_matrix,
          dispatch_gbl_channel_prefix_matrix,
          dispatch_recv_rdma_rank_prefix_sum,
          dispatch_recv_gbl_rank_prefix_sum,
          combine_num_recv_tokens,
          combine_num_rdma_recv_tokens,
          combine_recv_rdma_rank_prefix_sum,
          combine_recv_rdma_channel_prefix_matrix,
          combine_recv_gbl_channel_prefix_matrix,
          combine_send_rdma_head,
          combine_send_nvl_head,
          combine_num_rdma_recv_tokens_cumsum};
}

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
Buffer::internode_dispatch_api(
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
    const std::optional<paddle::Tensor>& asymm_recv_rdma_channel_prefix_matrix,
    const std::optional<paddle::Tensor>& asymm_send_rdma_head,
    const std::optional<paddle::Tensor>& asymm_send_nvl_head,
    const std::optional<paddle::Tensor>& asymm_aggregated_nvl_head,
    int expert_alignment,
    const Config& config,
    std::optional<EventHandle>& previous_event,  // NOLINT
    bool async,
    bool allocate_on_comm_stream,
    int num_experts) {
#ifdef PADDLE_WITH_NVSHMEM
  const auto& x_ = ConvertPaddleTensorToDetailTensor(x);
  std::optional<flash_ep::detail::Tensor> x_scales_ =
      ConvertOptionalPaddleTensorToDetailTensor(x_scales);

  std::optional<flash_ep::detail::Tensor> topk_idx_ =
      ConvertOptionalPaddleTensorToDetailTensor(topk_idx);
  std::optional<flash_ep::detail::Tensor> topk_weights_ =
      ConvertOptionalPaddleTensorToDetailTensor(topk_weights);
  std::optional<flash_ep::detail::Tensor> num_tokens_per_rank_ =
      ConvertOptionalPaddleTensorToDetailTensor(num_tokens_per_rank);
  std::optional<flash_ep::detail::Tensor> num_tokens_per_rdma_rank_ =
      ConvertOptionalPaddleTensorToDetailTensor(num_tokens_per_rdma_rank);

  const auto& is_token_in_rank_ =
      ConvertPaddleTensorToDetailTensor(is_token_in_rank);
  std::optional<flash_ep::detail::Tensor> num_tokens_per_expert_ =
      ConvertOptionalPaddleTensorToDetailTensor(num_tokens_per_expert);

  std::optional<flash_ep::detail::Tensor> cached_rdma_channel_prefix_matrix_ =
      ConvertOptionalPaddleTensorToDetailTensor(
          cached_rdma_channel_prefix_matrix);
  std::optional<flash_ep::detail::Tensor> cached_recv_rdma_rank_prefix_sum_ =
      ConvertOptionalPaddleTensorToDetailTensor(
          cached_recv_rdma_rank_prefix_sum);
  std::optional<flash_ep::detail::Tensor> cached_gbl_channel_prefix_matrix_ =
      ConvertOptionalPaddleTensorToDetailTensor(
          cached_gbl_channel_prefix_matrix);
  std::optional<flash_ep::detail::Tensor> cached_recv_gbl_rank_prefix_sum_ =
      ConvertOptionalPaddleTensorToDetailTensor(
          cached_recv_gbl_rank_prefix_sum);
  std::optional<flash_ep::detail::Tensor> asymm_send_combine_schedule_map_ =
      ConvertOptionalPaddleTensorToDetailTensor(
          asymm_send_combine_schedule_map);
  std::optional<flash_ep::detail::Tensor>
      asymm_recv_rdma_counter_loop_prefix_sum_ =
          ConvertOptionalPaddleTensorToDetailTensor(
              asymm_recv_rdma_counter_loop_prefix_sum);
  std::optional<flash_ep::detail::Tensor> asymm_recv_rdma_rank_prefix_sum_ =
      ConvertOptionalPaddleTensorToDetailTensor(
          asymm_recv_rdma_rank_prefix_sum);
  std::optional<flash_ep::detail::Tensor>
      asymm_recv_rdma_channel_prefix_matrix_ =
          ConvertOptionalPaddleTensorToDetailTensor(
              asymm_recv_rdma_channel_prefix_matrix);
  std::optional<flash_ep::detail::Tensor> asymm_send_rdma_head_ =
      ConvertOptionalPaddleTensorToDetailTensor(asymm_send_rdma_head);
  std::optional<flash_ep::detail::Tensor> asymm_send_nvl_head_ =
      ConvertOptionalPaddleTensorToDetailTensor(asymm_send_nvl_head);
  std::optional<flash_ep::detail::Tensor> asymm_aggregated_nvl_head_ =
      ConvertOptionalPaddleTensorToDetailTensor(asymm_aggregated_nvl_head);

  auto res = internode_dispatch(x_,
                                x_scales_,
                                topk_idx_,
                                topk_weights_,
                                num_tokens_per_rank_,
                                num_tokens_per_rdma_rank_,
                                is_token_in_rank_,
                                num_tokens_per_expert_,
                                cached_num_recv_tokens,
                                cached_num_rdma_recv_tokens,
                                cached_rdma_channel_prefix_matrix_,
                                cached_recv_rdma_rank_prefix_sum_,
                                cached_gbl_channel_prefix_matrix_,
                                cached_recv_gbl_rank_prefix_sum_,
                                asymm_send_combine_schedule_map_,
                                asymm_recv_rdma_counter_loop_prefix_sum_,
                                asymm_recv_rdma_rank_prefix_sum_,
                                asymm_recv_rdma_channel_prefix_matrix_,
                                asymm_send_rdma_head_,
                                asymm_send_nvl_head_,
                                asymm_aggregated_nvl_head_,
                                expert_alignment,
                                config,
                                previous_event,
                                async,
                                allocate_on_comm_stream,
                                num_experts);

  auto recv_x_ = ConvertDetailTensorToPaddleTensor(std::get<0>(res));
  std::optional<paddle::Tensor> recv_x_scales_ =
      ConvertOptionalDetailTensorToPaddleTensor(std::get<1>(res));

  std::optional<paddle::Tensor> recv_topk_idx_ =
      ConvertOptionalDetailTensorToPaddleTensor(std::get<2>(res));
  std::optional<paddle::Tensor> recv_topk_weights_ =
      ConvertOptionalDetailTensorToPaddleTensor(std::get<3>(res));

  const auto& num_recv_tokens_per_expert_list = std::get<4>(res);

  auto rdma_channel_prefix_matrix_ =
      ConvertDetailTensorToPaddleTensor(std::get<5>(res));

  auto gbl_channel_prefix_matrix_ =
      ConvertDetailTensorToPaddleTensor(std::get<6>(res));

  std::optional<paddle::Tensor> recv_rdma_channel_prefix_matrix_ =
      ConvertOptionalDetailTensorToPaddleTensor(std::get<7>(res));
  auto recv_rdma_rank_prefix_sum_ =
      ConvertDetailTensorToPaddleTensor(std::get<8>(res));

  std::optional<paddle::Tensor> recv_gbl_channel_prefix_matrix_ =
      ConvertOptionalDetailTensorToPaddleTensor(std::get<9>(res));
  auto recv_gbl_rank_prefix_sum_ =
      ConvertDetailTensorToPaddleTensor(std::get<10>(res));

  std::optional<paddle::Tensor> recv_src_meta_ =
      ConvertOptionalDetailTensorToPaddleTensor(std::get<11>(res));

  std::optional<paddle::Tensor> send_rdma_head_ =
      ConvertOptionalDetailTensorToPaddleTensor(std::get<12>(res));
  std::optional<paddle::Tensor> send_nvl_head_ =
      ConvertOptionalDetailTensorToPaddleTensor(std::get<13>(res));

  const auto& event = std::get<14>(res);

  return {recv_x_,
          recv_x_scales_,
          recv_topk_idx_,
          recv_topk_weights_,
          num_recv_tokens_per_expert_list,
          rdma_channel_prefix_matrix_,
          gbl_channel_prefix_matrix_,
          recv_rdma_channel_prefix_matrix_,
          recv_rdma_rank_prefix_sum_,
          recv_gbl_channel_prefix_matrix_,
          recv_gbl_rank_prefix_sum_,
          recv_src_meta_,
          send_rdma_head_,
          send_nvl_head_,
          event};
#else
  LOG(ERROR) << "NVSHMEM is not enabled. You can enable it by setting cmake "
                "option WITH_NVSHMEM=ON.";
  return {};
#endif
}

std::tuple<std::optional<paddle::Tensor>,
           std::optional<paddle::Tensor>,
           std::optional<EventHandle>>
Buffer::internode_combine_api(
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
    bool allocate_on_comm_stream) {
#ifdef PADDLE_WITH_NVSHMEM
  const auto& x_ = ConvertPaddleTensorToDetailTensor(x);

  std::optional<flash_ep::detail::Tensor> topk_weights_ =
      ConvertOptionalPaddleTensorToDetailTensor(topk_weights);

  const auto& rdma_channel_prefix_matrix_ =
      ConvertPaddleTensorToDetailTensor(rdma_channel_prefix_matrix);
  const auto& rdma_rank_prefix_sum_ =
      ConvertPaddleTensorToDetailTensor(rdma_rank_prefix_sum);
  const auto& gbl_channel_prefix_matrix_ =
      ConvertPaddleTensorToDetailTensor(gbl_channel_prefix_matrix);

  const auto& combined_rdma_head_ =
      ConvertPaddleTensorToDetailTensor(combined_rdma_head);
  const auto& combined_nvl_head_ =
      ConvertPaddleTensorToDetailTensor(combined_nvl_head);

  std::optional<flash_ep::detail::Tensor> combined_x_ =
      ConvertOptionalPaddleTensorToDetailTensor(combined_x);
  std::optional<flash_ep::detail::Tensor> combined_topk_weights_ =
      ConvertOptionalPaddleTensorToDetailTensor(combined_topk_weights);

  auto res = internode_combine(x_,
                               topk_weights_,
                               rdma_channel_prefix_matrix_,
                               rdma_rank_prefix_sum_,
                               gbl_channel_prefix_matrix_,
                               combined_rdma_head_,
                               combined_nvl_head_,
                               combined_x_,
                               combined_topk_weights_,
                               config,
                               previous_event,
                               async,
                               allocate_on_comm_stream);

  auto res_combined_x_ =
      ConvertOptionalDetailTensorToPaddleTensor(std::get<0>(res));
  std::optional<paddle::Tensor> res_combined_topk_weights_ =
      ConvertOptionalDetailTensorToPaddleTensor(std::get<1>(res));

  const auto& event = std::get<2>(res);

  return {res_combined_x_, res_combined_topk_weights_, event};
#else
  LOG(ERROR) << "NVSHMEM is not enabled. You can enable it by setting cmake "
                "option WITH_NVSHMEM=ON.";
  return {};
#endif
}

void Buffer::clear_buffer_api(const paddle::Tensor& x,
                              const std::optional<paddle::Tensor>& x_scales,
                              const std::optional<paddle::Tensor>& topk_idx,
                              const bool is_start,
                              const bool is_end,
                              const Config& config) {
#ifdef PADDLE_WITH_NVSHMEM
  const auto& x_ = ConvertPaddleTensorToDetailTensor(x);
  std::optional<flash_ep::detail::Tensor> x_scales_ =
      ConvertOptionalPaddleTensorToDetailTensor(x_scales);
  std::optional<flash_ep::detail::Tensor> topk_idx_ =
      ConvertOptionalPaddleTensorToDetailTensor(topk_idx);
  clear_buffer(x_, x_scales_, topk_idx_, is_start, is_end, config);
#else
  LOG(ERROR) << "NVSHMEM is not enabled. You can enable it by setting cmake "
                "option WITH_NVSHMEM=ON.";
  return {};
#endif
}

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
Buffer::internode_fused_notify_api(
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
    const Config& config) {
#ifdef PADDLE_WITH_NVSHMEM
  const auto& x_ = ConvertPaddleTensorToDetailTensor(x);
  std::optional<flash_ep::detail::Tensor> x_scales_ =
      ConvertOptionalPaddleTensorToDetailTensor(x_scales);

  std::optional<flash_ep::detail::Tensor> topk_idx_ =
      ConvertOptionalPaddleTensorToDetailTensor(topk_idx);

  std::optional<flash_ep::detail::Tensor> dispatch_num_tokens_per_rank_ =
      ConvertOptionalPaddleTensorToDetailTensor(dispatch_num_tokens_per_rank);
  std::optional<flash_ep::detail::Tensor> dispatch_num_tokens_per_rdma_rank_ =
      ConvertOptionalPaddleTensorToDetailTensor(
          dispatch_num_tokens_per_rdma_rank);
  std::optional<flash_ep::detail::Tensor> dispatch_num_tokens_per_expert_ =
      ConvertOptionalPaddleTensorToDetailTensor(dispatch_num_tokens_per_expert);
  const auto& dispatch_is_token_in_rank_ =
      ConvertPaddleTensorToDetailTensor(dispatch_is_token_in_rank);

  std::optional<flash_ep::detail::Tensor> combine_num_tokens_per_rank_ =
      ConvertOptionalPaddleTensorToDetailTensor(combine_num_tokens_per_rank);
  std::optional<flash_ep::detail::Tensor> combine_num_tokens_per_rdma_rank_ =
      ConvertOptionalPaddleTensorToDetailTensor(
          combine_num_tokens_per_rdma_rank);
  const auto& combine_is_token_in_rank_ =
      ConvertPaddleTensorToDetailTensor(combine_is_token_in_rank);

  auto res = internode_fused_notify(x_,
                                    x_scales_,
                                    topk_idx_,
                                    dispatch_num_tokens_per_rank_,
                                    dispatch_num_tokens_per_rdma_rank_,
                                    dispatch_num_tokens_per_expert_,
                                    dispatch_is_token_in_rank_,
                                    combine_num_tokens_per_rank_,
                                    combine_num_tokens_per_rdma_rank_,
                                    combine_is_token_in_rank_,
                                    expert_alignment,
                                    config);
  auto num_recv_tokens_per_expert_list_ = std::get<0>(res);
  auto dispatch_num_recv_tokens_ = std::get<1>(res);
  auto dispatch_num_rdma_recv_tokens_ = std::get<2>(res);

  auto dispatch_rdma_channel_prefix_matrix_ =
      ConvertDetailTensorToPaddleTensor(std::get<3>(res));

  auto dispatch_gbl_channel_prefix_matrix_ =
      ConvertDetailTensorToPaddleTensor(std::get<4>(res));

  auto dispatch_recv_rdma_rank_prefix_sum_ =
      ConvertDetailTensorToPaddleTensor(std::get<5>(res));

  auto dispatch_recv_gbl_rank_prefix_sum_ =
      ConvertDetailTensorToPaddleTensor(std::get<6>(res));

  auto combine_num_recv_tokens_ = std::get<7>(res);
  auto combine_num_rdma_recv_tokens_ = std::get<8>(res);
  auto combine_recv_rdma_rank_prefix_sum_ =
      ConvertDetailTensorToPaddleTensor(std::get<9>(res));

  auto combine_recv_rdma_channel_prefix_matrix_ =
      ConvertDetailTensorToPaddleTensor(std::get<10>(res));

  auto combine_recv_gbl_channel_prefix_matrix_ =
      ConvertDetailTensorToPaddleTensor(std::get<11>(res));

  auto combine_send_rdma_head_ =
      ConvertDetailTensorToPaddleTensor(std::get<12>(res));
  auto combine_send_nvl_head_ =
      ConvertDetailTensorToPaddleTensor(std::get<13>(res));
  auto combine_num_rdma_recv_tokens_cumsum_ =
      ConvertDetailTensorToPaddleTensor(std::get<14>(res));

  return {num_recv_tokens_per_expert_list_,
          dispatch_num_recv_tokens_,
          dispatch_num_rdma_recv_tokens_,
          dispatch_rdma_channel_prefix_matrix_,
          dispatch_gbl_channel_prefix_matrix_,
          dispatch_recv_rdma_rank_prefix_sum_,
          dispatch_recv_gbl_rank_prefix_sum_,
          combine_num_recv_tokens_,
          combine_num_rdma_recv_tokens_,
          combine_recv_rdma_rank_prefix_sum_,
          combine_recv_rdma_channel_prefix_matrix_,
          combine_recv_gbl_channel_prefix_matrix_,
          combine_send_rdma_head_,
          combine_send_nvl_head_,
          combine_num_rdma_recv_tokens_cumsum_};
#else
  LOG(ERROR) << "NVSHMEM is not enabled. You can enable it by setting cmake "
                "option WITH_NVSHMEM=ON.";
  return {};
#endif
}

std::tuple<paddle::Tensor,  // dispatch_rdma_schedule_map
           paddle::Tensor>  // combine_rdma_schedule_map
get_flash_ep_coalesce_rdma_schedule_api(
    const paddle::Tensor& topk_idx,
    const paddle::Tensor& local_expert_to_stage_map,
    const int num_ranks,
    const int num_experts,
    const int num_loop_stage) {
  EP_HOST_ASSERT(topk_idx.shape().size() == 2);
  EP_HOST_ASSERT(topk_idx.dtype() == phi::DataType::INT64);
  EP_HOST_ASSERT(num_experts >= num_ranks);
  EP_HOST_ASSERT(num_experts % num_ranks == 0);
  EP_HOST_ASSERT(local_expert_to_stage_map.shape().size() == 2);
  EP_HOST_ASSERT(local_expert_to_stage_map.shape()[0] ==
                 num_experts / num_ranks);
  EP_HOST_ASSERT(local_expert_to_stage_map.shape()[1] == 2);
  EP_HOST_ASSERT(local_expert_to_stage_map.dtype() == phi::DataType::INT32);

  int num_experts_per_rank = num_experts / num_ranks;

  int num_tokens = topk_idx.shape()[0];
  int num_topk = topk_idx.shape()[1];
  int num_rdma_ranks = num_ranks / NUM_MAX_NVL_PEERS;

  auto place = topk_idx.place();
  auto stream = topk_idx.stream();

  // 表示了, 每个token发给每个rdma_rank时是在第几轮.
  // 取值范围是[-1, num_loop_stage], -1和num_loop_stage代表不发送
  paddle::Tensor dispatch_rdma_schedule_map = paddle::experimental::empty(
      {num_tokens, num_rdma_ranks}, phi::DataType::INT32, place);
  paddle::Tensor combine_rdma_schedule_map = paddle::experimental::empty(
      {num_tokens, num_rdma_ranks}, phi::DataType::INT32, place);

  flash_ep::internode::get_flash_ep_coalesce_rdma_schedule(
      topk_idx.data<int64_t>(),
      local_expert_to_stage_map.data<int>(),
      dispatch_rdma_schedule_map.data<int>(),
      combine_rdma_schedule_map.data<int>(),
      num_ranks,
      num_experts,
      num_loop_stage,
      num_tokens,
      num_topk,
      stream);

  return {dispatch_rdma_schedule_map, combine_rdma_schedule_map};
}

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
    const int num_loop_stage) {
  EP_HOST_ASSERT(topk_idx.shape().size() == 2);
  EP_HOST_ASSERT(topk_idx.dtype() == phi::DataType::INT64);
  EP_HOST_ASSERT(num_experts >= num_ranks);
  EP_HOST_ASSERT(num_experts % num_ranks == 0);

  int num_tokens = topk_idx.shape()[0];
  int num_topk = topk_idx.shape()[1];
  int num_rdma_ranks = std::max(1, num_ranks / NUM_MAX_NVL_PEERS);

  auto place = topk_idx.place();
  auto stream = topk_idx.stream();

  paddle::Tensor num_tokens_per_rank = paddle::experimental::empty(
      {2, num_loop_stage, num_ranks}, phi::DataType::INT32, place);
  paddle::Tensor num_tokens_per_rdma_rank = paddle::experimental::empty(
      {2, num_loop_stage, num_rdma_ranks}, phi::DataType::INT32, place);
  paddle::Tensor num_tokens_per_expert = paddle::experimental::empty(
      {2, num_loop_stage, num_experts}, phi::DataType::INT32, place);
  paddle::Tensor is_token_in_rank = paddle::experimental::empty(
      {2, num_loop_stage, num_tokens, num_ranks}, phi::DataType::BOOL, place);
  flash_ep::internode::get_flash_ep_coalesce_rdma_layout(
      topk_idx.data<int64_t>(),
      dispatch_rdma_schedule_map.data<int>(),
      combine_rdma_schedule_map.data<int>(),
      num_tokens_per_rank.data<int>(),
      num_tokens_per_rdma_rank.data<int>(),
      num_tokens_per_expert.data<int>(),
      is_token_in_rank.data<bool>(),
      num_tokens,
      num_topk,
      num_ranks,
      num_experts,
      num_loop_stage,
      stream);
  return {num_tokens_per_rank,
          num_tokens_per_rdma_rank,
          num_tokens_per_expert,
          is_token_in_rank};
}

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
    const int64_t local_expert_id,
    const int64_t ori_out_len,
    const int64_t padding_align) {
  EP_HOST_ASSERT(hidden_states[0].dtype() == paddle::DataType::FLOAT8_E4M3FN ||
                 hidden_states[0].dtype() == paddle::DataType::BFLOAT16);
  EP_HOST_ASSERT(hidden_states[0].shape().size() == 2);
  EP_HOST_ASSERT(topk_weights[0].shape().size() == 2);
  EP_HOST_ASSERT(topk_idx[0].shape().size() == 2);
  EP_HOST_ASSERT(hidden_states.size() == topk_weights.size());
  EP_HOST_ASSERT(topk_idx.size() == topk_weights.size());

  int64_t hidden_size = hidden_states[0].shape()[1];
  int64_t topk = topk_idx[0].shape()[1];
  int64_t a2a_num = hidden_states.size();
  bool use_fp8 = hidden_states[0].dtype() == phi::DataType::FLOAT8_E4M3FN;
  int64_t scale_num = hidden_size / 128;

  auto place = hidden_states[0].place();
  auto stream = hidden_states[0].stream();

  int token_out_len =
      ((ori_out_len + padding_align - 1) / padding_align) * padding_align;

  paddle::Tensor output_hidden_states = paddle::experimental::full(
      {token_out_len, hidden_size}, 0, hidden_states[0].dtype(), place);
  paddle::Tensor output_topk_probs = paddle::experimental::full(
      {token_out_len}, 0, topk_weights[0].dtype(), place);
  paddle::Tensor output_src_meta = paddle::experimental::full(
      {token_out_len, 4}, 0, phi::DataType::INT32, place);
  std::optional<paddle::Tensor> output_scale;
  float* output_scale_ptr = nullptr;
  if (use_fp8) {
    EP_HOST_ASSERT(fp8_scales.has_value());
    EP_HOST_ASSERT(fp8_scales.value().size() == hidden_states.size());
    output_scale = paddle::experimental::full(
        {token_out_len, scale_num}, 0, fp8_scales.value()[0].dtype(), place);
    output_scale_ptr = output_scale.value().data<float>();
  }

  const void** d_hidden_states_ptr;
  cudaMallocAsync(&d_hidden_states_ptr, a2a_num * sizeof(void*), stream);
  std::vector<const void*> host_hidden_states_ptrs;
  for (int64_t i = 0; i < a2a_num; ++i) {
    host_hidden_states_ptrs.push_back(hidden_states[i].data());
  }
  cudaMemcpyAsync(d_hidden_states_ptr,
                  host_hidden_states_ptrs.data(),
                  a2a_num * sizeof(void*),
                  cudaMemcpyHostToDevice,
                  stream);

  const float** d_topk_weights_ptr;
  cudaMallocAsync(&d_topk_weights_ptr, a2a_num * sizeof(float*), stream);
  std::vector<const float*> host_topk_weights_ptrs;
  for (int64_t i = 0; i < a2a_num; ++i) {
    host_topk_weights_ptrs.push_back(topk_weights[i].data<float>());
  }
  cudaMemcpyAsync(d_topk_weights_ptr,
                  host_topk_weights_ptrs.data(),
                  a2a_num * sizeof(float*),
                  cudaMemcpyHostToDevice,
                  stream);

  const int32_t** d_topk_idx_ptr;
  cudaMallocAsync(&d_topk_idx_ptr, a2a_num * sizeof(int32_t*), stream);
  std::vector<const int32_t*> host_topk_idx_ptrs;
  for (int32_t i = 0; i < a2a_num; ++i) {
    host_topk_idx_ptrs.push_back(topk_idx[i].data<int32_t>());
  }
  cudaMemcpyAsync(d_topk_idx_ptr,
                  host_topk_idx_ptrs.data(),
                  a2a_num * sizeof(int32_t*),
                  cudaMemcpyHostToDevice,
                  stream);

  const int32_t** d_recv_src_meta_per_a2a_ptr;
  cudaMallocAsync(
      &d_recv_src_meta_per_a2a_ptr, a2a_num * sizeof(int32_t*), stream);
  std::vector<const int32_t*> host_recv_src_meta_per_a2a_ptrs;
  for (int64_t i = 0; i < a2a_num; ++i) {
    host_recv_src_meta_per_a2a_ptrs.push_back(
        recv_src_meta_per_a2a[i].data<int32_t>());
  }
  cudaMemcpyAsync(d_recv_src_meta_per_a2a_ptr,
                  host_recv_src_meta_per_a2a_ptrs.data(),
                  a2a_num * sizeof(int32_t*),
                  cudaMemcpyHostToDevice,
                  stream);

  const float** d_fp8_scales_ptr = nullptr;
  if (use_fp8) {
    cudaMallocAsync(&d_fp8_scales_ptr, a2a_num * sizeof(float*), stream);
    std::vector<const float*> host_fp8_scales_ptrs;
    for (int64_t i = 0; i < a2a_num; ++i) {
      host_fp8_scales_ptrs.push_back(fp8_scales.value()[i].data<float>());
    }
    cudaMemcpyAsync(d_fp8_scales_ptr,
                    host_fp8_scales_ptrs.data(),
                    a2a_num * sizeof(float*),
                    cudaMemcpyHostToDevice,
                    stream);
  }

  // a2a_prefix_sum
  int64_t all_token_num = hidden_states[0].shape()[0];
  paddle::Tensor a2a_prefix_sum_tensor =
      paddle::experimental::empty({a2a_num}, phi::DataType::INT32, place);
  std::vector<int32_t> h_a2a_prefix_sum(a2a_num);
  h_a2a_prefix_sum[0] = 0;
  for (int64_t i = 1; i < a2a_num; i++) {
    h_a2a_prefix_sum[i] =
        h_a2a_prefix_sum[i - 1] + hidden_states[i - 1].shape()[0];
    all_token_num += hidden_states[i].shape()[0];
  }
  cudaMemcpyAsync(a2a_prefix_sum_tensor.data<int32_t>(),
                  h_a2a_prefix_sum.data(),
                  a2a_num * sizeof(int32_t),
                  cudaMemcpyHostToDevice,
                  stream);

  const int cumsum_blocknum =
      (all_token_num + kCumsumBlockSize - 1) / kCumsumBlockSize;

  paddle::Tensor global_expertwise_block_cumsum = paddle::experimental::full(
      {cumsum_blocknum + 1}, kCumsumInvalidTag, phi::DataType::INT32, place);

  flash_ep::internode::local_dispatch(
      d_hidden_states_ptr,
      d_topk_weights_ptr,
      d_topk_idx_ptr,
      d_recv_src_meta_per_a2a_ptr,
      d_fp8_scales_ptr,
      a2a_prefix_sum_tensor.data<int32_t>(),
      global_expertwise_block_cumsum.data<int32_t>(),
      local_expert_id,
      hidden_size,
      topk,
      a2a_num,
      all_token_num,
      ori_out_len,
      scale_num,
      output_hidden_states.data(),
      nullptr,
      output_topk_probs.data<float>(),
      output_src_meta.data<int32_t>(),
      output_scale_ptr,
      stream,
      use_fp8,
      true);
  return {
      output_hidden_states, output_topk_probs, output_src_meta, output_scale};
}

std::vector<paddle::Tensor> local_dispatch_backward_api(
    const std::vector<paddle::Tensor>& hidden_states,
    const std::vector<paddle::Tensor>& topk_idx,
    const std::vector<paddle::Tensor>& recv_src_meta_per_a2a,
    const int64_t local_expert_id,
    const int64_t ori_out_len,
    const int64_t padding_align) {
  EP_HOST_ASSERT(hidden_states[0].dtype() == paddle::DataType::BFLOAT16);
  EP_HOST_ASSERT(hidden_states[0].shape().size() == 2);
  EP_HOST_ASSERT(topk_idx[0].shape().size() == 2);
  EP_HOST_ASSERT(hidden_states.size() == topk_idx.size());

  int64_t hidden_size = hidden_states[0].shape()[1];
  int64_t topk = topk_idx[0].shape()[1];
  int64_t a2a_num = hidden_states.size();

  auto place = hidden_states[0].place();
  auto stream = hidden_states[0].stream();

  int token_out_len =
      ((ori_out_len + padding_align - 1) / padding_align) * padding_align;

  paddle::Tensor output_hidden_states = paddle::experimental::full(
      {token_out_len, hidden_size}, 0, hidden_states[0].dtype(), place);
  paddle::Tensor output_topk_idx = paddle::experimental::full(
      {token_out_len, topk}, -1, phi::DataType::INT32, place);
  paddle::Tensor output_src_meta = paddle::experimental::full(
      {token_out_len, 4}, 0, phi::DataType::INT32, place);

  const void** d_hidden_states_ptr;
  cudaMallocAsync(&d_hidden_states_ptr, a2a_num * sizeof(void*), stream);
  std::vector<const void*> host_hidden_states_ptrs;
  for (int64_t i = 0; i < a2a_num; ++i) {
    host_hidden_states_ptrs.push_back(hidden_states[i].data());
  }
  cudaMemcpyAsync(d_hidden_states_ptr,
                  host_hidden_states_ptrs.data(),
                  a2a_num * sizeof(void*),
                  cudaMemcpyHostToDevice,
                  stream);

  const int32_t** d_topk_idx_ptr;
  cudaMallocAsync(&d_topk_idx_ptr, a2a_num * sizeof(int32_t*), stream);
  std::vector<const int32_t*> host_topk_idx_ptrs;
  for (int32_t i = 0; i < a2a_num; ++i) {
    host_topk_idx_ptrs.push_back(topk_idx[i].data<int32_t>());
  }
  cudaMemcpyAsync(d_topk_idx_ptr,
                  host_topk_idx_ptrs.data(),
                  a2a_num * sizeof(int32_t*),
                  cudaMemcpyHostToDevice,
                  stream);

  const int32_t** d_recv_src_meta_per_a2a_ptr;
  cudaMallocAsync(
      &d_recv_src_meta_per_a2a_ptr, a2a_num * sizeof(int32_t*), stream);
  std::vector<const int32_t*> host_recv_src_meta_per_a2a_ptrs;
  for (int64_t i = 0; i < a2a_num; ++i) {
    host_recv_src_meta_per_a2a_ptrs.push_back(
        recv_src_meta_per_a2a[i].data<int32_t>());
  }
  cudaMemcpyAsync(d_recv_src_meta_per_a2a_ptr,
                  host_recv_src_meta_per_a2a_ptrs.data(),
                  a2a_num * sizeof(int32_t*),
                  cudaMemcpyHostToDevice,
                  stream);

  // a2a_prefix_sum
  int64_t all_token_num = hidden_states[0].shape()[0];
  paddle::Tensor a2a_prefix_sum_tensor =
      paddle::experimental::empty({a2a_num}, phi::DataType::INT32, place);
  std::vector<int32_t> h_a2a_prefix_sum(a2a_num);
  h_a2a_prefix_sum[0] = 0;
  for (int64_t i = 1; i < a2a_num; i++) {
    h_a2a_prefix_sum[i] =
        h_a2a_prefix_sum[i - 1] + hidden_states[i - 1].shape()[0];
    all_token_num += hidden_states[i].shape()[0];
  }
  cudaMemcpyAsync(a2a_prefix_sum_tensor.data<int32_t>(),
                  h_a2a_prefix_sum.data(),
                  a2a_num * sizeof(int32_t),
                  cudaMemcpyHostToDevice,
                  stream);

  const int cumsum_blocknum =
      (all_token_num + kCumsumBlockSize - 1) / kCumsumBlockSize;

  paddle::Tensor global_expertwise_block_cumsum = paddle::experimental::full(
      {cumsum_blocknum + 1}, kCumsumInvalidTag, phi::DataType::INT32, place);

  flash_ep::internode::local_dispatch(
      d_hidden_states_ptr,
      nullptr,
      d_topk_idx_ptr,
      d_recv_src_meta_per_a2a_ptr,
      nullptr,
      a2a_prefix_sum_tensor.data<int32_t>(),
      global_expertwise_block_cumsum.data<int32_t>(),
      local_expert_id,
      hidden_size,
      topk,
      a2a_num,
      all_token_num,
      ori_out_len,
      -1,
      output_hidden_states.data(),
      output_topk_idx.data<int32_t>(),
      nullptr,
      output_src_meta.data<int32_t>(),
      nullptr,
      stream,
      false,
      false);
  return {output_hidden_states, output_topk_idx, output_src_meta};
}

void local_combine_forward_api(
    std::vector<paddle::Tensor>& combine_buffers,  // NOLINT
    const paddle::Tensor& hidden_states,
    const paddle::Tensor& recv_gbl_src_meta,
    const std::vector<paddle::Tensor>& recv_gbl_channel_prefix_matrix_list,
    const int64_t ori_len,
    const std::vector<int>& is_buffer_active) {
  int hidden_size = hidden_states.shape()[1];
  int token_num = ori_len;
  int num_loop_stage = recv_gbl_channel_prefix_matrix_list.size();

  EP_HOST_ASSERT(hidden_states.shape().size() == 2);
  EP_HOST_ASSERT(hidden_states.dtype() == phi::DataType::BFLOAT16);
  EP_HOST_ASSERT(is_buffer_active.size() == combine_buffers.size());
  EP_HOST_ASSERT(is_buffer_active.size() == num_loop_stage);

  auto stream = hidden_states.stream();

  const int32_t** d_recv_gbl_channel_prefix_ptr;
  cudaMallocAsync(&d_recv_gbl_channel_prefix_ptr,
                  num_loop_stage * sizeof(int32_t*),
                  stream);
  std::vector<const int32_t*> host_recv_gbl_channel_prefix_ptrs;
  for (int i = 0; i < num_loop_stage; ++i) {
    host_recv_gbl_channel_prefix_ptrs.push_back(
        recv_gbl_channel_prefix_matrix_list[i].data<int32_t>());
  }
  cudaMemcpyAsync(d_recv_gbl_channel_prefix_ptr,
                  host_recv_gbl_channel_prefix_ptrs.data(),
                  num_loop_stage * sizeof(int32_t*),
                  cudaMemcpyHostToDevice,
                  stream);

  float** d_out_combine_ptr;
  cudaMallocAsync(&d_out_combine_ptr, num_loop_stage * sizeof(float*), stream);
  std::vector<float*> host_out_combine_ptrs;
  for (int i = 0; i < num_loop_stage; ++i) {
    if (is_buffer_active[i]) {
      host_out_combine_ptrs.push_back(combine_buffers[i].data<float>());
    } else {
      host_out_combine_ptrs.push_back(nullptr);
    }
  }
  cudaMemcpyAsync(d_out_combine_ptr,
                  host_out_combine_ptrs.data(),
                  num_loop_stage * sizeof(float*),
                  cudaMemcpyHostToDevice,
                  stream);

  flash_ep::internode::local_combine_forward(
      reinterpret_cast<const __nv_bfloat16*>(hidden_states.data()),
      d_recv_gbl_channel_prefix_ptr,
      recv_gbl_src_meta.data<int32_t>(),
      hidden_size,
      num_loop_stage,
      token_num,
      d_out_combine_ptr,
      stream);
}

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
    const std::vector<int>& is_buffer_active) {
  EP_HOST_ASSERT(hidden_states.shape().size() == 2);
  EP_HOST_ASSERT(hidden_states.dtype() == phi::DataType::BFLOAT16);
  EP_HOST_ASSERT(topk_idx.shape().size() == 2);
  EP_HOST_ASSERT(topk_idx.dtype() == phi::DataType::INT32);
  EP_HOST_ASSERT(topk_weights.shape().size() == 1);
  EP_HOST_ASSERT(topk_weights.dtype() == phi::DataType::FLOAT32);
  EP_HOST_ASSERT(topk_weights.shape()[0] == hidden_states.shape()[0]);
  EP_HOST_ASSERT(is_buffer_active.size() == combine_buffers.size());

  int hidden_size = hidden_states.shape()[1];
  int topk = topk_idx.shape()[1];
  int num_loop_stage = recv_gbl_channel_prefix_matrix_list.size();

  EP_HOST_ASSERT(is_buffer_active.size() == num_loop_stage);

  auto stream = hidden_states.stream();

  const int32_t** d_recv_gbl_channel_prefix_ptr;
  cudaMallocAsync(&d_recv_gbl_channel_prefix_ptr,
                  num_loop_stage * sizeof(int32_t*),
                  stream);
  std::vector<const int32_t*> host_recv_gbl_channel_prefix_ptrs;
  for (int i = 0; i < num_loop_stage; ++i) {
    host_recv_gbl_channel_prefix_ptrs.push_back(
        recv_gbl_channel_prefix_matrix_list[i].data<int32_t>());
  }
  cudaMemcpyAsync(d_recv_gbl_channel_prefix_ptr,
                  host_recv_gbl_channel_prefix_ptrs.data(),
                  num_loop_stage * sizeof(int32_t*),
                  cudaMemcpyHostToDevice,
                  stream);

  float** d_out_combine_ptr;
  cudaMallocAsync(&d_out_combine_ptr, num_loop_stage * sizeof(float*), stream);
  std::vector<float*> host_out_combine_ptrs;
  for (int i = 0; i < num_loop_stage; ++i) {
    if (is_buffer_active[i]) {
      host_out_combine_ptrs.push_back(combine_buffers[i].data<float>());
    } else {
      host_out_combine_ptrs.push_back(nullptr);
    }
  }
  cudaMemcpyAsync(d_out_combine_ptr,
                  host_out_combine_ptrs.data(),
                  num_loop_stage * sizeof(float*),
                  cudaMemcpyHostToDevice,
                  stream);

  float** d_out_probs_ptr;
  cudaMallocAsync(&d_out_probs_ptr, num_loop_stage * sizeof(float*), stream);
  std::vector<float*> host_out_probs_ptrs;
  for (int i = 0; i < num_loop_stage; ++i) {
    if (is_buffer_active[i]) {
      host_out_probs_ptrs.push_back(combine_probs[i].data<float>());
    } else {
      host_out_probs_ptrs.push_back(nullptr);
    }
  }
  cudaMemcpyAsync(d_out_probs_ptr,
                  host_out_probs_ptrs.data(),
                  num_loop_stage * sizeof(float*),
                  cudaMemcpyHostToDevice,
                  stream);
  flash_ep::internode::local_combine_backward(
      reinterpret_cast<const __nv_bfloat16*>(hidden_states.data()),
      topk_idx.data<int32_t>(),
      topk_weights.data<float>(),
      d_recv_gbl_channel_prefix_ptr,
      recv_gbl_src_meta.data<int32_t>(),
      hidden_size,
      num_loop_stage,
      ori_len,
      topk,
      local_expert_id,
      d_out_combine_ptr,
      d_out_probs_ptr,
      stream);
}

flash_ep::detail::Tensor ConvertPaddleTensorToDetailTensor(
    const paddle::Tensor& tensor) {
  flash_ep::detail::Tensor res(tensor);
  return res;
}

paddle::Tensor ConvertDetailTensorToPaddleTensor(
    const flash_ep::detail::Tensor& tensor) {
  return tensor.raw_tensor();
}

std::optional<flash_ep::detail::Tensor>
ConvertOptionalPaddleTensorToDetailTensor(
    const std::optional<paddle::Tensor>& tensor) {
  std::optional<flash_ep::detail::Tensor> res;
  if (tensor.has_value()) {
    res = ConvertPaddleTensorToDetailTensor(tensor.value());
  }
  return res;
}

std::optional<paddle::Tensor> ConvertOptionalDetailTensorToPaddleTensor(
    const std::optional<flash_ep::detail::Tensor>& tensor) {
  std::optional<paddle::Tensor> res;
  if (tensor.has_value()) {
    res = ConvertDetailTensorToPaddleTensor(tensor.value());
  }
  return res;
}

}  // namespace flash_ep
