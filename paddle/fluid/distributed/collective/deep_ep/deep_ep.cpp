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
#include <pybind11/functional.h>
#include <atomic>
#include <chrono>
#include <memory>

#include "paddle/fluid/distributed/collective/deep_ep/deep_ep.hpp"
#include "paddle/fluid/distributed/collective/deep_ep/kernels/api.cuh"
#include "paddle/fluid/distributed/collective/deep_ep/kernels/configs.cuh"

#include "paddle/fluid/distributed/collective/deep_ep/include/CUDADataType.h"
#include "paddle/fluid/distributed/collective/deep_ep/include/ScalarType.h"
#include "paddle/fluid/distributed/collective/process_group_nccl.h"
#include "paddle/phi/api/include/api.h"
#include "paddle/phi/common/data_type.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/distributed/utils.h"

namespace deep_ep {

Buffer::Buffer(int rank,
               int num_ranks,
               int64_t num_nvl_bytes,
               int64_t num_rdma_bytes,
               bool low_latency_mode,
               int context_ring_id)
    : rank(rank),
      num_ranks(num_ranks),
      num_nvl_bytes(num_nvl_bytes),
      num_rdma_bytes(num_rdma_bytes),
      low_latency_mode(low_latency_mode) {
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
                 (num_nvl_bytes <= std::numeric_limits<int>::max() ||
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
    CUDA_CHECK(
        cudaMemsetAsync(task_fifo_ptrs[nvl_rank], 0, fifo_bytes, comm_stream));
  }

  // Create 32 MiB workspace
  CUDA_CHECK(cudaMalloc(&workspace, NUM_WORKSPACE_BYTES));
  CUDA_CHECK(cudaMemsetAsync(workspace, 0, NUM_WORKSPACE_BYTES, comm_stream));

  // MoE counter
  CUDA_CHECK(
      cudaMallocHost(&moe_recv_counter, sizeof(int64_t), cudaHostAllocMapped));
  CUDA_CHECK(cudaHostGetDevicePointer(
      &moe_recv_counter_mapped, const_cast<int*>(moe_recv_counter), 0));
  *moe_recv_counter = -1;

  // MoE expert-level counter
  CUDA_CHECK(cudaMallocHost(&moe_recv_expert_counter,
                            sizeof(int) * NUM_MAX_LOCAL_EXPERTS,
                            cudaHostAllocMapped));
  CUDA_CHECK(cudaHostGetDevicePointer(&moe_recv_expert_counter_mapped,
                                      const_cast<int*>(moe_recv_expert_counter),
                                      0));
  for (int i = 0; i < NUM_MAX_LOCAL_EXPERTS; ++i)
    moe_recv_expert_counter[i] = -1;

  // MoE RDMA-level counter
  if (num_rdma_ranks > 0) {
    CUDA_CHECK(cudaMallocHost(
        &moe_recv_rdma_counter, sizeof(int), cudaHostAllocMapped));
    CUDA_CHECK(cudaHostGetDevicePointer(&moe_recv_rdma_counter_mapped,
                                        const_cast<int*>(moe_recv_rdma_counter),
                                        0));
    *moe_recv_rdma_counter = -1;
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

  // Free NVSHMEM
  if (num_rdma_bytes > 0) {
    // CUDA_CHECK(cudaDeviceSynchronize());
    // internode::barrier();
    // internode::free(rdma_buffer_ptr);
    // internode::finalize();
    LOG(FATAL) << "not supported yet.";
  }

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
  return is_available() && num_ranks > NUM_MAX_NVL_PEERS;
}

int Buffer::get_num_rdma_ranks() const { return num_rdma_ranks; }

int Buffer::get_rdma_rank() const { return rdma_rank; }

int Buffer::get_root_rdma_rank(bool global) const {
  return global ? nvl_rank : 0;
}

int Buffer::get_local_device_id() const { return device_id; }

pybind11::bytearray Buffer::get_local_ipc_handle() const {
  return {ipc_handles[nvl_rank].reserved, CUDA_IPC_HANDLE_SIZE};
}

void Buffer::sync(
    const std::vector<int>& device_ids,
    const std::vector<std::optional<pybind11::bytearray>>& all_gathered_handles,
    const std::optional<pybind11::bytearray>& root_unique_id_opt) {
  EP_HOST_ASSERT(!is_available());

  // Sync IPC handles
  if (num_nvl_bytes > 0) {
    EP_HOST_ASSERT(num_ranks == device_ids.size());
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

  // Sync NVSHMEM handles and allocate memory
  if (num_rdma_bytes > 0) {
    // // Initialize NVSHMEM
    // EP_HOST_ASSERT(root_unique_id_opt.has_value());
    // std::vector<uint8_t> root_unique_id(root_unique_id_opt->size());
    // auto root_unique_id_str = root_unique_id_opt->cast<std::string>();
    // std::memcpy(root_unique_id.data(), root_unique_id_str.c_str(),
    // root_unique_id_opt->size()); auto nvshmem_rank = low_latency_mode ? rank
    // : rdma_rank; auto num_nvshmem_ranks = low_latency_mode ? num_ranks :
    // num_rdma_ranks; EP_HOST_ASSERT(nvshmem_rank ==
    // internode::init(root_unique_id, nvshmem_rank, num_nvshmem_ranks,
    // low_latency_mode)); internode::barrier();

    // // Allocate
    // rdma_buffer_ptr = internode::alloc(num_rdma_bytes,
    // NUM_BUFFER_ALIGNMENT_BYTES);

    // // Clean buffer (mainly for low-latency mode)
    // CUDA_CHECK(cudaMemset(rdma_buffer_ptr, 0, num_rdma_bytes));

    // // Barrier
    // internode::barrier();
    // CUDA_CHECK(cudaDeviceSynchronize());
    LOG(FATAL) << "Not implemented yet";
  }

  // Ready to use
  available = true;
}

std::tuple<deep_ep::detail::Tensor,
           std::optional<deep_ep::detail::Tensor>,
           deep_ep::detail::Tensor,
           deep_ep::detail::Tensor,
           std::optional<EventHandle>>
Buffer::get_dispatch_layout(const deep_ep::detail::Tensor& topk_idx,
                            int num_experts,
                            std::optional<EventHandle>& previous_event,
                            bool async,
                            bool allocate_on_comm_stream) {
  EP_HOST_ASSERT(topk_idx.dim() == 2);
  EP_HOST_ASSERT(topk_idx.is_contiguous());
  EP_HOST_ASSERT(num_experts > 0);

  // Allocate all tensors on comm stream if set
  // NOTES: do not allocate tensors upfront!
  // auto compute_stream = deep_ep::detail::getCurrentCUDAStream();
  auto compute_stream = calc_ctx->stream();
  if (allocate_on_comm_stream) {
    EP_HOST_ASSERT(previous_event.has_value() && async);
    deep_ep::detail::setCurrentCUDAStream(comm_stream);
  }

  // Wait previous tasks to be finished
  if (previous_event.has_value()) {
    stream_wait(comm_stream, previous_event.value());
  } else {
    stream_wait(comm_stream, compute_stream);
  }

  auto num_tokens = static_cast<int>(topk_idx.size(0)),
       num_topk = static_cast<int>(topk_idx.size(1));
  auto num_tokens_per_rank =
      ConvertPaddleTensorToDetailTensor(paddle::experimental::empty(
          {num_ranks}, phi::DataType::INT32, phi::GPUPlace(device_id)));
  auto num_tokens_per_rdma_rank = std::optional<deep_ep::detail::Tensor>();
  auto num_tokens_per_expert =
      ConvertPaddleTensorToDetailTensor(paddle::experimental::empty(
          {num_experts}, phi::DataType::INT32, phi::GPUPlace(device_id)));
  auto is_token_in_rank = ConvertPaddleTensorToDetailTensor(
      paddle::experimental::empty({num_tokens, num_ranks},
                                  phi::DataType::BOOL,
                                  phi::GPUPlace(device_id)));
  if (is_internode_available())
    num_tokens_per_rdma_rank =
        ConvertPaddleTensorToDetailTensor(paddle::experimental::empty(
            {num_rdma_ranks}, phi::DataType::INT32, phi::GPUPlace(device_id)));

  internode::get_dispatch_layout(
      topk_idx.data_ptr<int64_t>(),
      num_tokens_per_rank.data_ptr<int>(),
      num_tokens_per_rdma_rank.has_value()
          ? num_tokens_per_rdma_rank.value().data_ptr<int>()
          : nullptr,
      num_tokens_per_expert.data_ptr<int>(),
      is_token_in_rank.data_ptr<bool>(),
      num_tokens,
      num_topk,
      num_ranks,
      num_experts,
      comm_stream);

  // Wait streams
  std::optional<EventHandle> event;
  if (async) {
    event = EventHandle(comm_stream);
    for (auto& t : {topk_idx,
                    num_tokens_per_rank,
                    num_tokens_per_expert,
                    is_token_in_rank}) {
      t.record_stream(comm_stream);
      if (allocate_on_comm_stream) t.record_stream(compute_stream);
    }
    for (auto& to : {num_tokens_per_rdma_rank}) {
      to.has_value() ? to->record_stream(comm_stream) : void();
      if (allocate_on_comm_stream)
        to.has_value() ? to->record_stream(compute_stream) : void();
    }
  } else {
    stream_wait(compute_stream, comm_stream);
  }

  // Switch back compute stream
  if (allocate_on_comm_stream)
    deep_ep::detail::setCurrentCUDAStream(compute_stream);

  return {num_tokens_per_rank,
          num_tokens_per_rdma_rank,
          num_tokens_per_expert,
          is_token_in_rank,
          event};
}

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
Buffer::intranode_dispatch(
    const deep_ep::detail::Tensor& x,
    const std::optional<deep_ep::detail::Tensor>& x_scales,
    const std::optional<deep_ep::detail::Tensor>& topk_idx,
    const std::optional<deep_ep::detail::Tensor>& topk_weights,
    const std::optional<deep_ep::detail::Tensor>& num_tokens_per_rank,
    const deep_ep::detail::Tensor& is_token_in_rank,
    const std::optional<deep_ep::detail::Tensor>& num_tokens_per_expert,
    int cached_num_recv_tokens,
    const std::optional<deep_ep::detail::Tensor>& cached_rank_prefix_matrix,
    const std::optional<deep_ep::detail::Tensor>& cached_channel_prefix_matrix,
    int expert_alignment,
    const Config& config,
    std::optional<EventHandle>& previous_event,  // NOLINT
    bool async,
    bool allocate_on_comm_stream) {
  bool cached_mode = cached_rank_prefix_matrix.has_value();

  // One channel use two blocks, even-numbered blocks for sending, odd-numbered
  // blocks for receiving.
  EP_HOST_ASSERT(config.num_sms % 2 == 0);
  int num_channels = config.num_sms / 2;
  if (cached_mode) {
    EP_HOST_ASSERT(cached_rank_prefix_matrix.has_value());
    EP_HOST_ASSERT(cached_channel_prefix_matrix.has_value());
  } else {
    EP_HOST_ASSERT(num_tokens_per_rank.has_value());
    EP_HOST_ASSERT(num_tokens_per_expert.has_value());
  }

  // Type checks
  EP_HOST_ASSERT(is_token_in_rank.scalar_type() == deep_ep::detail::kBool);
  if (cached_mode) {
    EP_HOST_ASSERT(cached_rank_prefix_matrix->scalar_type() ==
                   deep_ep::detail::kInt32);
    EP_HOST_ASSERT(cached_channel_prefix_matrix->scalar_type() ==
                   deep_ep::detail::kInt32);
  } else {
    EP_HOST_ASSERT(num_tokens_per_expert->scalar_type() ==
                   deep_ep::detail::kInt32);
    EP_HOST_ASSERT(num_tokens_per_rank->scalar_type() ==
                   deep_ep::detail::kInt32);
  }

  // Shape and contiguous checks
  EP_HOST_ASSERT(x.dim() == 2 && x.is_contiguous());
  EP_HOST_ASSERT((x.size(1) * x.element_size()) % sizeof(int4) == 0);
  EP_HOST_ASSERT(is_token_in_rank.dim() == 2 &&
                 is_token_in_rank.is_contiguous());
  EP_HOST_ASSERT(is_token_in_rank.size(0) == x.size(0) &&
                 is_token_in_rank.size(1) == num_ranks);
  if (cached_mode) {
    EP_HOST_ASSERT(cached_rank_prefix_matrix->dim() == 2 &&
                   cached_rank_prefix_matrix->is_contiguous());
    EP_HOST_ASSERT(cached_rank_prefix_matrix->size(0) == num_ranks &&
                   cached_rank_prefix_matrix->size(1) == num_ranks);
    EP_HOST_ASSERT(cached_channel_prefix_matrix->dim() == 2 &&
                   cached_channel_prefix_matrix->is_contiguous());
    EP_HOST_ASSERT(cached_channel_prefix_matrix->size(0) == num_ranks &&
                   cached_channel_prefix_matrix->size(1) == num_channels);
  } else {
    EP_HOST_ASSERT(num_tokens_per_expert->dim() == 1 &&
                   num_tokens_per_expert->is_contiguous());
    EP_HOST_ASSERT(num_tokens_per_expert->size(0) % num_ranks == 0);
    EP_HOST_ASSERT(num_tokens_per_expert->size(0) / num_ranks <=
                   NUM_MAX_LOCAL_EXPERTS);
    EP_HOST_ASSERT(num_tokens_per_rank->dim() == 1 &&
                   num_tokens_per_rank->is_contiguous());
    EP_HOST_ASSERT(num_tokens_per_rank->size(0) == num_ranks);
  }

  auto num_tokens = static_cast<int>(x.size(0)),
       hidden = static_cast<int>(x.size(1));
  auto num_experts =
           cached_mode ? 0 : static_cast<int>(num_tokens_per_expert->size(0)),
       num_local_experts = num_experts / num_ranks;

  // Top-k checks
  int num_topk = 0;
  int64_t* topk_idx_ptr = nullptr;
  float* topk_weights_ptr = nullptr;
  EP_HOST_ASSERT(topk_idx.has_value() == topk_weights.has_value());
  if (topk_idx.has_value()) {
    num_topk = static_cast<int>(topk_idx->size(1));
    EP_HOST_ASSERT(num_experts > 0);
    EP_HOST_ASSERT(topk_idx->dim() == 2 && topk_idx->is_contiguous());
    EP_HOST_ASSERT(topk_weights->dim() == 2 && topk_weights->is_contiguous());
    EP_HOST_ASSERT(num_tokens == topk_idx->size(0) &&
                   num_tokens == topk_weights->size(0));
    EP_HOST_ASSERT(num_topk == topk_weights->size(1));
    EP_HOST_ASSERT(topk_weights->scalar_type() == deep_ep::detail::kFloat32);
    topk_idx_ptr = topk_idx->data_ptr<int64_t>();
    topk_weights_ptr = topk_weights->data_ptr<float>();
  }

  // FP8 scales checks
  float* x_scales_ptr = nullptr;
  int num_scales = 0;
  if (x_scales.has_value()) {
    EP_HOST_ASSERT(x.element_size() == 1);
    EP_HOST_ASSERT(x_scales->scalar_type() == deep_ep::detail::kFloat32);
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
    deep_ep::detail::setCurrentCUDAStream(comm_stream);
  }

  // Wait previous tasks to be finished
  if (previous_event.has_value()) {
    stream_wait(comm_stream, previous_event.value());
  } else {
    stream_wait(comm_stream, compute_stream);
  }

  // Create handles (only return for non-cached mode)
  int num_recv_tokens = -1;
  auto rank_prefix_matrix = deep_ep::detail::Tensor();
  auto channel_prefix_matrix = deep_ep::detail::Tensor();
  std::vector<int> num_recv_tokens_per_expert_list;

  // Barrier or send sizes
  // To clean: channel start/end offset, head and tail
  int num_memset_int = num_channels * num_ranks * 4;
  if (cached_mode) {
    num_recv_tokens = cached_num_recv_tokens;
    rank_prefix_matrix = cached_rank_prefix_matrix.value();
    channel_prefix_matrix = cached_channel_prefix_matrix.value();

    // Copy rank prefix matrix and clean flags
    intranode::cached_notify_dispatch(rank_prefix_matrix.data_ptr<int>(),
                                      num_memset_int,
                                      buffer_ptrs_gpu,
                                      task_fifo_ptrs_gpu,
                                      head,
                                      rank,
                                      num_ranks,
                                      comm_stream);
    move_fifo_slots(2);
  } else {
    rank_prefix_matrix = ConvertPaddleTensorToDetailTensor(
        paddle::experimental::empty({num_ranks, num_ranks},
                                    phi::DataType::INT32,
                                    phi::GPUPlace(device_id)));
    channel_prefix_matrix = ConvertPaddleTensorToDetailTensor(
        paddle::experimental::empty({num_ranks, num_channels},
                                    phi::DataType::INT32,
                                    phi::GPUPlace(device_id)));

    // Send sizes
    // Meta information:
    //  - Size prefix by ranks, shaped as `[num_ranks, num_ranks]`
    //  - Size prefix by experts (not used later), shaped as `[num_ranks,
    //  num_local_experts]`
    // NOTES: no more token dropping in this version
    *moe_recv_counter = -1;
    for (int i = 0; i < num_local_experts; ++i) moe_recv_expert_counter[i] = -1;
    EP_HOST_ASSERT(num_ranks * (num_ranks + num_local_experts) * sizeof(int) <=
                   num_nvl_bytes);
    intranode::notify_dispatch(num_tokens_per_rank->data_ptr<int>(),
                               moe_recv_counter_mapped,
                               num_ranks,
                               num_tokens_per_expert->data_ptr<int>(),
                               moe_recv_expert_counter_mapped,
                               num_experts,
                               num_tokens,
                               is_token_in_rank.data_ptr<bool>(),
                               channel_prefix_matrix.data_ptr<int>(),
                               rank_prefix_matrix.data_ptr<int>(),
                               num_memset_int,
                               expert_alignment,
                               buffer_ptrs_gpu,
                               task_fifo_ptrs_gpu,
                               head,
                               rank,
                               comm_stream,
                               num_channels);
    move_fifo_slots(3);

    // Synchronize total received tokens and tokens per expert
    auto start_time = std::chrono::high_resolution_clock::now();
    while (true) {
      // Read total count
      num_recv_tokens = static_cast<int>(*moe_recv_counter);

      // Read per-expert count
      bool ready = (num_recv_tokens >= 0);
      for (int i = 0; i < num_local_experts && ready; ++i)
        ready &= moe_recv_expert_counter[i] >= 0;

      if (ready) break;

      // Timeout check
      if (std::chrono::duration_cast<std::chrono::seconds>(
              std::chrono::high_resolution_clock::now() - start_time)
              .count() > NUM_CPU_TIMEOUT_SECS)
        throw std::runtime_error("DeepEP error: CPU recv timeout");
    }
    num_recv_tokens_per_expert_list = std::vector<int>(
        moe_recv_expert_counter, moe_recv_expert_counter + num_local_experts);
  }

  // Allocate new tensors
  auto recv_x = ConvertPaddleTensorToDetailTensor(paddle::experimental::empty(
      {num_recv_tokens, hidden}, x.dtype(), x.place()));
  auto recv_src_idx =
      ConvertPaddleTensorToDetailTensor(paddle::experimental::empty(
          {num_recv_tokens}, phi::DataType::INT32, phi::GPUPlace(device_id)));
  auto recv_topk_idx = std::optional<deep_ep::detail::Tensor>(),
       recv_topk_weights = std::optional<deep_ep::detail::Tensor>(),
       recv_x_scales = std::optional<deep_ep::detail::Tensor>();
  auto recv_channel_prefix_matrix = ConvertPaddleTensorToDetailTensor(
      paddle::experimental::empty({num_ranks, num_channels},
                                  phi::DataType::INT32,
                                  phi::GPUPlace(device_id)));
  auto send_head = ConvertPaddleTensorToDetailTensor(
      paddle::experimental::empty({num_tokens, num_ranks},
                                  phi::DataType::INT32,
                                  phi::GPUPlace(device_id)));

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
                                    topk_idx->place()));
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

  // Dispatch
  EP_HOST_ASSERT(
      num_ranks * num_ranks * sizeof(int) +             // Size prefix matrix
          num_channels * num_ranks * sizeof(int) +      // Channel start offset
          num_channels * num_ranks * sizeof(int) +      // Channel end offset
          num_channels * num_ranks * sizeof(int) * 2 +  // Queue head and tail
          num_channels * num_ranks * config.num_max_nvl_chunked_recv_tokens *
              hidden * recv_x.element_size() +  // Data buffer
          num_channels * num_ranks * config.num_max_nvl_chunked_recv_tokens *
              sizeof(int) +  // Source index buffer
          num_channels * num_ranks * config.num_max_nvl_chunked_recv_tokens *
              num_topk * sizeof(int64_t) +  // Top-k index buffer
          num_channels * num_ranks * config.num_max_nvl_chunked_recv_tokens *
              num_topk * sizeof(float) +  // Top-k weight buffer
          num_channels * num_ranks * config.num_max_nvl_chunked_recv_tokens *
              sizeof(float) * num_scales  // FP8 scale buffer
      <= num_nvl_bytes);
  intranode::dispatch(
      recv_x.data_ptr(),
      recv_x_scales_ptr,
      recv_src_idx.data_ptr<int>(),
      recv_topk_idx_ptr,
      recv_topk_weights_ptr,
      recv_channel_prefix_matrix.data_ptr<int>(),
      send_head.data_ptr<int>(),
      x.data_ptr(),
      x_scales_ptr,
      topk_idx_ptr,
      topk_weights_ptr,
      is_token_in_rank.data_ptr<bool>(),
      channel_prefix_matrix.data_ptr<int>(),
      num_tokens,
      static_cast<int>(hidden * recv_x.element_size() / sizeof(int4)),
      num_topk,
      num_experts,
      num_scales,
      buffer_ptrs_gpu,
      rank,
      num_ranks,
      comm_stream,
      config.num_sms,
      config.num_max_nvl_chunked_send_tokens,
      config.num_max_nvl_chunked_recv_tokens);

  // Wait streams
  std::optional<EventHandle> event;
  if (async) {
    event = EventHandle(comm_stream);
    for (auto& t : {x,
                    is_token_in_rank,
                    rank_prefix_matrix,
                    channel_prefix_matrix,
                    recv_x,
                    recv_src_idx,
                    recv_channel_prefix_matrix,
                    send_head}) {
      t.record_stream(comm_stream);
      if (allocate_on_comm_stream) t.record_stream(compute_stream);
    }
    for (auto& to : {x_scales,
                     topk_idx,
                     topk_weights,
                     num_tokens_per_rank,
                     num_tokens_per_expert,
                     cached_channel_prefix_matrix,
                     cached_rank_prefix_matrix,
                     recv_topk_idx,
                     recv_topk_weights,
                     recv_x_scales}) {
      to.has_value() ? to->record_stream(comm_stream) : void();
      if (allocate_on_comm_stream)
        to.has_value() ? to->record_stream(compute_stream) : void();
    }
  } else {
    stream_wait(compute_stream, comm_stream);
  }

  // Switch back compute stream
  if (allocate_on_comm_stream)
    deep_ep::detail::setCurrentCUDAStream(compute_stream);

  // Return values
  return {recv_x,
          recv_x_scales,
          recv_topk_idx,
          recv_topk_weights,
          num_recv_tokens_per_expert_list,
          rank_prefix_matrix,
          channel_prefix_matrix,
          recv_channel_prefix_matrix,
          recv_src_idx,
          send_head,
          event};
}

std::tuple<deep_ep::detail::Tensor,
           std::optional<deep_ep::detail::Tensor>,
           std::optional<EventHandle>>
Buffer::intranode_combine(
    const deep_ep::detail::Tensor& x,
    const std::optional<deep_ep::detail::Tensor>& topk_weights,
    const deep_ep::detail::Tensor& src_idx,
    const deep_ep::detail::Tensor& rank_prefix_matrix,
    const deep_ep::detail::Tensor& channel_prefix_matrix,
    const deep_ep::detail::Tensor& send_head,
    const Config& config,
    std::optional<EventHandle>& previous_event,
    bool async,
    bool allocate_on_comm_stream) {
  EP_HOST_ASSERT(x.dim() == 2 && x.is_contiguous());
  EP_HOST_ASSERT(src_idx.dim() == 1 && src_idx.is_contiguous() &&
                 src_idx.scalar_type() == deep_ep::detail::kInt32);
  EP_HOST_ASSERT(send_head.dim() == 2 && send_head.is_contiguous() &&
                 send_head.scalar_type() == deep_ep::detail::kInt32);
  EP_HOST_ASSERT(rank_prefix_matrix.dim() == 2 &&
                 rank_prefix_matrix.is_contiguous() &&
                 rank_prefix_matrix.scalar_type() == deep_ep::detail::kInt32);
  EP_HOST_ASSERT(channel_prefix_matrix.dim() == 2 &&
                 channel_prefix_matrix.is_contiguous() &&
                 channel_prefix_matrix.scalar_type() ==
                     deep_ep::detail::kInt32);

  // One channel use two blocks, even-numbered blocks for sending, odd-numbered
  // blocks for receiving.
  EP_HOST_ASSERT(config.num_sms % 2 == 0);
  int num_channels = config.num_sms / 2;

  auto num_tokens = static_cast<int>(x.size(0)),
       hidden = static_cast<int>(x.size(1));
  auto num_recv_tokens = static_cast<int>(send_head.size(0));
  EP_HOST_ASSERT(src_idx.size(0) == num_tokens);
  EP_HOST_ASSERT(send_head.size(1) == num_ranks);
  EP_HOST_ASSERT(rank_prefix_matrix.size(0) == num_ranks &&
                 rank_prefix_matrix.size(1) == num_ranks);
  EP_HOST_ASSERT(channel_prefix_matrix.size(0) == num_ranks &&
                 channel_prefix_matrix.size(1) == num_channels);
  EP_HOST_ASSERT((hidden * x.element_size()) % sizeof(int4) == 0);

  // Allocate all tensors on comm stream if set
  // NOTES: do not allocate tensors upfront!
  auto compute_stream = calc_ctx->stream();
  if (allocate_on_comm_stream) {
    EP_HOST_ASSERT(previous_event.has_value() && async);
    deep_ep::detail::setCurrentCUDAStream(comm_stream);
  }

  // Wait previous tasks to be finished
  if (previous_event.has_value()) {
    stream_wait(comm_stream, previous_event.value());
  } else {
    stream_wait(comm_stream, compute_stream);
  }

  int num_topk = 0;
  auto recv_topk_weights = std::optional<deep_ep::detail::Tensor>();
  float* topk_weights_ptr = nullptr;
  float* recv_topk_weights_ptr = nullptr;
  if (topk_weights.has_value()) {
    EP_HOST_ASSERT(topk_weights->dim() == 2 && topk_weights->is_contiguous());
    EP_HOST_ASSERT(topk_weights->size(0) == num_tokens);
    EP_HOST_ASSERT(topk_weights->scalar_type() == deep_ep::detail::kFloat32);
    num_topk = static_cast<int>(topk_weights->size(1));
    topk_weights_ptr = topk_weights->data_ptr<float>();
    recv_topk_weights = ConvertPaddleTensorToDetailTensor(
        paddle::experimental::empty({num_recv_tokens, num_topk},
                                    topk_weights->dtype(),
                                    topk_weights->place()));
    recv_topk_weights_ptr = recv_topk_weights->data_ptr<float>();
  }

  // Launch barrier and reset queue head and tail
  EP_HOST_ASSERT(num_channels * num_ranks * sizeof(int) * 2 <= num_nvl_bytes);
  intranode::cached_notify_combine(buffer_ptrs_gpu,
                                   send_head.data_ptr<int>(),
                                   num_channels,
                                   num_recv_tokens,
                                   num_channels * num_ranks * 2,
                                   task_fifo_ptrs_gpu,
                                   head,
                                   rank,
                                   num_ranks,
                                   comm_stream);

  // NOTES: this function uses two FIFO slots (barrier before and after)
  move_fifo_slots(2);

  // Combine data
  auto recv_x = ConvertPaddleTensorToDetailTensor(paddle::experimental::empty(
      {num_recv_tokens, hidden}, x.dtype(), x.place()));
  EP_HOST_ASSERT(
      num_channels * num_ranks * sizeof(int) * 2 +  // Queue head and tail
          num_channels * num_ranks * config.num_max_nvl_chunked_recv_tokens *
              hidden * x.element_size() +  // Data buffer
          num_channels * num_ranks * config.num_max_nvl_chunked_recv_tokens *
              sizeof(int) +  // Source index buffer
          num_channels * num_ranks * config.num_max_nvl_chunked_recv_tokens *
              num_topk * sizeof(float)  // Top-k weight buffer
      <= num_nvl_bytes);
  intranode::combine(deep_ep::detail::ScalarTypeToCudaDataType(x.scalar_type()),
                     recv_x.data_ptr(),
                     recv_topk_weights_ptr,
                     x.data_ptr(),
                     topk_weights_ptr,
                     src_idx.data_ptr<int>(),
                     rank_prefix_matrix.data_ptr<int>(),
                     channel_prefix_matrix.data_ptr<int>(),
                     send_head.data_ptr<int>(),
                     num_tokens,
                     num_recv_tokens,
                     hidden,
                     num_topk,
                     buffer_ptrs_gpu,
                     rank,
                     num_ranks,
                     comm_stream,
                     config.num_sms,
                     config.num_max_nvl_chunked_send_tokens,
                     config.num_max_nvl_chunked_recv_tokens);

  // Wait streams
  std::optional<EventHandle> event;
  if (async) {
    event = EventHandle(comm_stream);
    for (auto& t : {x,
                    src_idx,
                    send_head,
                    rank_prefix_matrix,
                    channel_prefix_matrix,
                    recv_x}) {
      t.record_stream(comm_stream);
      if (allocate_on_comm_stream) t.record_stream(compute_stream);
    }
    for (auto& to : {topk_weights, recv_topk_weights}) {
      to.has_value() ? to->record_stream(comm_stream) : void();
      if (allocate_on_comm_stream)
        to.has_value() ? to->record_stream(compute_stream) : void();
    }
  } else {
    stream_wait(compute_stream, comm_stream);
  }

  // Switch back compute stream
  if (allocate_on_comm_stream)
    deep_ep::detail::setCurrentCUDAStream(compute_stream);

  return {recv_x, recv_topk_weights, event};
}

std::tuple<paddle::Tensor,
           std::optional<paddle::Tensor>,
           paddle::Tensor,
           paddle::Tensor,
           std::optional<EventHandle>>
Buffer::get_dispatch_layout_api(const paddle::Tensor& topk_idx,
                                int num_experts,
                                std::optional<EventHandle>& previous_event,
                                bool async,
                                bool allocate_on_comm_stream) {
  const auto& topk_idx_ = ConvertPaddleTensorToDetailTensor(topk_idx);
  auto res = get_dispatch_layout(
      topk_idx_, num_experts, previous_event, async, allocate_on_comm_stream);
  const auto& num_tokens_per_rank = std::get<0>(res);
  const auto& num_tokens_per_rdma_rank = std::get<1>(res);
  const auto& num_tokens_per_expert = std::get<2>(res);
  const auto& is_token_in_rank = std::get<3>(res);
  const auto& event = std::get<4>(res);
  auto num_tokens_per_rank_ =
      ConvertDetailTensorToPaddleTensor(num_tokens_per_rank);
  std::optional<paddle::Tensor> num_tokens_per_rdma_rank_ = std::nullopt;
  if (num_tokens_per_rdma_rank.has_value()) {
    num_tokens_per_rdma_rank_ =
        ConvertDetailTensorToPaddleTensor(num_tokens_per_rdma_rank.value());
  }
  auto num_tokens_per_expert_ =
      ConvertDetailTensorToPaddleTensor(num_tokens_per_expert);
  auto is_token_in_rank_ = ConvertDetailTensorToPaddleTensor(is_token_in_rank);
  return {num_tokens_per_rank_,
          num_tokens_per_rdma_rank_,
          num_tokens_per_expert_,
          is_token_in_rank_,
          event};
}

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
Buffer::intranode_dispatch_api(
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
    bool allocate_on_comm_stream) {
  const auto& x_ = ConvertPaddleTensorToDetailTensor(x);
  std::optional<deep_ep::detail::Tensor> x_scales_;
  if (x_scales.has_value()) {
    x_scales_ = ConvertPaddleTensorToDetailTensor(x_scales.value());
  }
  std::optional<deep_ep::detail::Tensor> topk_idx_;
  if (topk_idx.has_value()) {
    topk_idx_ = ConvertPaddleTensorToDetailTensor(topk_idx.value());
  }
  std::optional<deep_ep::detail::Tensor> topk_weights_;
  if (topk_weights.has_value()) {
    topk_weights_ = ConvertPaddleTensorToDetailTensor(topk_weights.value());
  }
  std::optional<deep_ep::detail::Tensor> num_tokens_per_rank_;
  if (num_tokens_per_rank.has_value()) {
    num_tokens_per_rank_ =
        ConvertPaddleTensorToDetailTensor(num_tokens_per_rank.value());
  }
  const auto& is_token_in_rank_ =
      ConvertPaddleTensorToDetailTensor(is_token_in_rank);
  std::optional<deep_ep::detail::Tensor> num_tokens_per_expert_;
  if (num_tokens_per_expert.has_value()) {
    num_tokens_per_expert_ =
        ConvertPaddleTensorToDetailTensor(num_tokens_per_expert.value());
  }
  std::optional<deep_ep::detail::Tensor> cached_rank_prefix_matrix_;
  if (cached_rank_prefix_matrix.has_value()) {
    cached_rank_prefix_matrix_ =
        ConvertPaddleTensorToDetailTensor(cached_rank_prefix_matrix.value());
  }
  std::optional<deep_ep::detail::Tensor> cached_channel_prefix_matrix_;
  if (cached_channel_prefix_matrix.has_value()) {
    cached_channel_prefix_matrix_ =
        ConvertPaddleTensorToDetailTensor(cached_channel_prefix_matrix.value());
  }

  auto res = intranode_dispatch(x_,
                                x_scales_,
                                topk_idx_,
                                topk_weights_,
                                num_tokens_per_rank_,
                                is_token_in_rank_,
                                num_tokens_per_expert_,
                                cached_num_recv_tokens,
                                cached_rank_prefix_matrix_,
                                cached_channel_prefix_matrix_,
                                expert_alignment,
                                config,
                                previous_event,
                                async,
                                allocate_on_comm_stream);

  //   {recv_x, recv_x_scales, recv_topk_idx, recv_topk_weights,
  //   num_recv_tokens_per_expert_list, rank_prefix_matrix,
  //   channel_prefix_matrix, recv_channel_prefix_matrix, recv_src_idx,
  //   send_head, event};
  const auto& recv_x = std::get<0>(res);
  const auto& recv_x_scales = std::get<1>(res);
  const auto& recv_topk_idx = std::get<2>(res);
  const auto& recv_topk_weights = std::get<3>(res);
  const auto& num_recv_tokens_per_expert_list = std::get<4>(res);
  const auto& rank_prefix_matrix = std::get<5>(res);
  const auto& channel_prefix_matrix = std::get<6>(res);
  const auto& recv_channel_prefix_matrix = std::get<7>(res);
  const auto& recv_src_idx = std::get<8>(res);
  const auto& send_head = std::get<9>(res);
  const auto& event = std::get<10>(res);

  auto recv_x_ = ConvertDetailTensorToPaddleTensor(recv_x);
  std::optional<paddle::Tensor> recv_x_scales_;
  if (recv_x_scales.has_value()) {
    recv_x_scales_ = ConvertDetailTensorToPaddleTensor(recv_x_scales.value());
  }
  std::optional<paddle::Tensor> recv_topk_idx_;
  if (recv_topk_idx.has_value()) {
    recv_topk_idx_ = ConvertDetailTensorToPaddleTensor(recv_topk_idx.value());
  }
  std::optional<paddle::Tensor> recv_topk_weights_;
  if (recv_topk_weights.has_value()) {
    recv_topk_weights_ =
        ConvertDetailTensorToPaddleTensor(recv_topk_weights.value());
  }
  auto rank_prefix_matrix_ =
      ConvertDetailTensorToPaddleTensor(rank_prefix_matrix);
  auto channel_prefix_matrix_ =
      ConvertDetailTensorToPaddleTensor(channel_prefix_matrix);
  auto recv_channel_prefix_matrix_ =
      ConvertDetailTensorToPaddleTensor(recv_channel_prefix_matrix);
  auto recv_src_idx_ = ConvertDetailTensorToPaddleTensor(recv_src_idx);
  auto send_head_ = ConvertDetailTensorToPaddleTensor(send_head);
  return {recv_x_,
          recv_x_scales_,
          recv_topk_idx_,
          recv_topk_weights_,
          num_recv_tokens_per_expert_list,
          rank_prefix_matrix_,
          channel_prefix_matrix_,
          recv_channel_prefix_matrix_,
          recv_src_idx_,
          send_head_,
          event};
}

std::tuple<paddle::Tensor,
           std::optional<paddle::Tensor>,
           std::optional<EventHandle>>
Buffer::intranode_combine_api(const paddle::Tensor& x,
                              const std::optional<paddle::Tensor>& topk_weights,
                              const paddle::Tensor& src_idx,
                              const paddle::Tensor& rank_prefix_matrix,
                              const paddle::Tensor& channel_prefix_matrix,
                              const paddle::Tensor& send_head,
                              const Config& config,
                              std::optional<EventHandle>& previous_event,
                              bool async,
                              bool allocate_on_comm_stream) {
  const auto& x_ = ConvertPaddleTensorToDetailTensor(x);
  std::optional<deep_ep::detail::Tensor> topk_weights_;
  if (topk_weights.has_value()) {
    topk_weights_ = ConvertPaddleTensorToDetailTensor(topk_weights.value());
  }
  const auto& src_idx_ = ConvertPaddleTensorToDetailTensor(src_idx);
  const auto& rank_prefix_matrix_ =
      ConvertPaddleTensorToDetailTensor(rank_prefix_matrix);
  const auto& channel_prefix_matrix_ =
      ConvertPaddleTensorToDetailTensor(channel_prefix_matrix);
  const auto& send_head_ = ConvertPaddleTensorToDetailTensor(send_head);

  auto res = intranode_combine(x_,
                               topk_weights_,
                               src_idx_,
                               rank_prefix_matrix_,
                               channel_prefix_matrix_,
                               send_head_,
                               config,
                               previous_event,
                               async,
                               allocate_on_comm_stream);

  const auto& recv_x = std::get<0>(res);
  const auto& recv_topk_weights = std::get<1>(res);
  const auto& event = std::get<2>(res);

  auto recv_x_ = ConvertDetailTensorToPaddleTensor(recv_x);
  std::optional<paddle::Tensor> recv_topk_weights_;
  if (recv_topk_weights.has_value()) {
    recv_topk_weights_ =
        ConvertDetailTensorToPaddleTensor(recv_topk_weights.value());
  }
  auto event_ = event;
  return {recv_x_, recv_topk_weights_, event_};
}

deep_ep::detail::Tensor ConvertPaddleTensorToDetailTensor(
    const paddle::Tensor& tensor) {
  deep_ep::detail::Tensor res(tensor);
  return res;
}

paddle::Tensor ConvertDetailTensorToPaddleTensor(
    const deep_ep::detail::Tensor& tensor) {
  return tensor.raw_tensor();
}

}  // namespace deep_ep
