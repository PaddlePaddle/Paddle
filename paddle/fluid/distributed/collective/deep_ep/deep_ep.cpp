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

#include "paddle/fluid/distributed/collective/deep_ep/deep_ep.hpp"
#include "paddle/fluid/distributed/collective/deep_ep/kernels/api.cuh"
#include "paddle/fluid/distributed/collective/deep_ep/kernels/configs.cuh"

#include "paddle/fluid/distributed/collective/deep_ep/include/CUDADataType.h"
#include "paddle/fluid/distributed/collective/deep_ep/include/ScalarType.h"
#include "paddle/fluid/distributed/collective/process_group_nccl.h"
#include "paddle/phi/api/include/api.h"
#include "paddle/phi/api/include/tensor_utils.h"
#include "paddle/phi/common/data_type.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/distributed/utils.h"
#include "paddle/phi/core/memory/allocation/allocator_facade.h"

namespace deep_ep {

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
  auto compute_stream = calc_ctx->stream();
  if (allocate_on_comm_stream) {
    EP_HOST_ASSERT(previous_event.has_value() && async);
    deep_ep::detail::SetAllocatorStreamForGPUContext(comm_stream, calc_ctx);
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

  // get_dispatch_layout is used for both intranode and internode.
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
  if (allocate_on_comm_stream) {
    deep_ep::detail::SetAllocatorStreamForGPUContext(compute_stream, calc_ctx);
  }

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
    deep_ep::detail::SetAllocatorStreamForGPUContext(comm_stream, calc_ctx);
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
    EP_HOST_ASSERT(num_ranks * (num_ranks + num_local_experts) *
                       static_cast<int64_t>(sizeof(int)) <=
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
      num_ranks * num_ranks *
              static_cast<int64_t>(sizeof(int)) +  // prefix matrix
          num_channels * num_ranks *
              static_cast<int64_t>(sizeof(int)) +  // Channel start offset
          num_channels * num_ranks *
              static_cast<int64_t>(sizeof(int)) +  // Channel end offset
          num_channels * num_ranks * static_cast<int64_t>(sizeof(int)) *
              2 +  // Queue head and tail
          num_channels * num_ranks * config.num_max_nvl_chunked_recv_tokens *
              hidden * recv_x.element_size() +  // Data buffer
          num_channels * num_ranks * config.num_max_nvl_chunked_recv_tokens *
              static_cast<int64_t>(sizeof(int)) +  // Source index buffer
          num_channels * num_ranks * config.num_max_nvl_chunked_recv_tokens *
              num_topk *
              static_cast<int64_t>(sizeof(int64_t)) +  // Top-k index buffer
          num_channels * num_ranks * config.num_max_nvl_chunked_recv_tokens *
              num_topk *
              static_cast<int64_t>(sizeof(float)) +  // Top-k weight buffer
          num_channels * num_ranks * config.num_max_nvl_chunked_recv_tokens *
              static_cast<int64_t>(sizeof(float)) *
              num_scales  // FP8 scale buffer
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
  if (allocate_on_comm_stream) {
    deep_ep::detail::SetAllocatorStreamForGPUContext(compute_stream, calc_ctx);
  }

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
    deep_ep::detail::SetAllocatorStreamForGPUContext(comm_stream, calc_ctx);
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
  EP_HOST_ASSERT(num_channels * num_ranks * static_cast<int64_t>(sizeof(int)) *
                     2 <=
                 num_nvl_bytes);
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
      num_channels * num_ranks * static_cast<int64_t>(sizeof(int)) *
              2 +  // Queue head and tail
          num_channels * num_ranks * config.num_max_nvl_chunked_recv_tokens *
              hidden * x.element_size() +  // Data buffer
          num_channels * num_ranks * config.num_max_nvl_chunked_recv_tokens *
              static_cast<int64_t>(sizeof(int)) +  // Source index buffer
          num_channels * num_ranks * config.num_max_nvl_chunked_recv_tokens *
              num_topk *
              static_cast<int64_t>(sizeof(float))  // Top-k weight buffer
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
  if (allocate_on_comm_stream) {
    deep_ep::detail::SetAllocatorStreamForGPUContext(compute_stream, calc_ctx);
  }

  return {recv_x, recv_topk_weights, event};
}

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
Buffer::internode_dispatch(
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
    bool allocate_on_comm_stream) {
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
                   deep_ep::detail::kInt32);
    EP_HOST_ASSERT(cached_recv_rdma_rank_prefix_sum->scalar_type() ==
                   deep_ep::detail::kInt32);
    EP_HOST_ASSERT(cached_gbl_channel_prefix_matrix->scalar_type() ==
                   deep_ep::detail::kInt32);
    EP_HOST_ASSERT(cached_recv_gbl_rank_prefix_sum->scalar_type() ==
                   deep_ep::detail::kInt32);
  } else {
    EP_HOST_ASSERT(num_tokens_per_rank->scalar_type() ==
                   deep_ep::detail::kInt32);
    EP_HOST_ASSERT(num_tokens_per_rdma_rank->scalar_type() ==
                   deep_ep::detail::kInt32);
    EP_HOST_ASSERT(num_tokens_per_expert->scalar_type() ==
                   deep_ep::detail::kInt32);
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
    deep_ep::detail::SetAllocatorStreamForGPUContext(comm_stream, calc_ctx);
  }

  // Wait previous tasks to be finished
  if (previous_event.has_value()) {
    stream_wait(comm_stream, previous_event.value());
  } else {
    stream_wait(comm_stream, compute_stream);
  }

  // Create handles (only return for non-cached mode)
  int num_recv_tokens = -1, num_rdma_recv_tokens = -1;
  auto rdma_channel_prefix_matrix = deep_ep::detail::Tensor();
  auto recv_rdma_rank_prefix_sum = deep_ep::detail::Tensor();
  auto gbl_channel_prefix_matrix = deep_ep::detail::Tensor();
  auto recv_gbl_rank_prefix_sum = deep_ep::detail::Tensor();
  std::vector<int> num_recv_tokens_per_expert_list;

  // Barrier or send sizes
  if (cached_mode) {
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
  } else {
    rdma_channel_prefix_matrix = ConvertPaddleTensorToDetailTensor(
        paddle::experimental::empty({num_rdma_ranks, num_channels},
                                    phi::DataType::INT32,
                                    phi::GPUPlace(device_id)));
    recv_rdma_rank_prefix_sum =
        ConvertPaddleTensorToDetailTensor(paddle::experimental::empty(
            {num_rdma_ranks}, phi::DataType::INT32, phi::GPUPlace(device_id)));
    gbl_channel_prefix_matrix = ConvertPaddleTensorToDetailTensor(
        paddle::experimental::empty({num_ranks, num_channels},
                                    phi::DataType::INT32,
                                    phi::GPUPlace(device_id)));
    recv_gbl_rank_prefix_sum =
        ConvertPaddleTensorToDetailTensor(paddle::experimental::empty(
            {num_ranks}, phi::DataType::INT32, phi::GPUPlace(device_id)));

    // Send sizes
    *moe_recv_counter = -1, *moe_recv_rdma_counter = -1;
    for (int i = 0; i < num_local_experts; ++i) moe_recv_expert_counter[i] = -1;
    internode::notify_dispatch(
        num_tokens_per_rank->data_ptr<int>(),
        moe_recv_counter_mapped,
        num_ranks,
        num_tokens_per_rdma_rank->data_ptr<int>(),
        moe_recv_rdma_counter_mapped,
        num_tokens_per_expert->data_ptr<int>(),
        moe_recv_expert_counter_mapped,
        num_experts,
        is_token_in_rank.data_ptr<bool>(),
        num_tokens,
        num_channels,
        hidden_int4,
        num_scales,
        num_topk,
        expert_alignment,
        rdma_channel_prefix_matrix.data_ptr<int>(),
        recv_rdma_rank_prefix_sum.data_ptr<int>(),
        gbl_channel_prefix_matrix.data_ptr<int>(),
        recv_gbl_rank_prefix_sum.data_ptr<int>(),
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
        low_latency_mode);
    move_fifo_slots(3);

    // Synchronize total received tokens and tokens per expert
    auto start_time = std::chrono::high_resolution_clock::now();
    while (true) {
      // Read total count
      num_recv_tokens = static_cast<int>(*moe_recv_counter);
      num_rdma_recv_tokens = static_cast<int>(*moe_recv_rdma_counter);

      // Read per-expert count
      bool ready = (num_recv_tokens >= 0) && (num_rdma_recv_tokens >= 0);
      for (int i = 0; i < num_local_experts && ready; ++i)
        ready &= moe_recv_expert_counter[i] >= 0;

      if (ready) break;

      // Timeout check
      if (std::chrono::duration_cast<std::chrono::seconds>(
              std::chrono::high_resolution_clock::now() - start_time)
              .count() > NUM_CPU_TIMEOUT_SECS) {
        LOG(INFO) << "Global rank: " << rank
                  << ", num_recv_tokens: " << num_recv_tokens
                  << ", num_rdma_recv_tokens: " << num_rdma_recv_tokens;
        for (int i = 0; i < num_local_experts; ++i)
          LOG(INFO) << "moe_recv_expert_counter[" << i
                    << "]: " << moe_recv_expert_counter[i];
        throw std::runtime_error("DeepEP error: timeout (dispatch CPU)");
      }
    }
    num_recv_tokens_per_expert_list = std::vector<int>(
        moe_recv_expert_counter, moe_recv_expert_counter + num_local_experts);
  }

  // Allocate new tensors
  auto recv_x = ConvertPaddleTensorToDetailTensor(paddle::experimental::empty(
      {num_recv_tokens, hidden}, x.dtype(), x.place()));
  auto recv_topk_idx = std::optional<deep_ep::detail::Tensor>(),
       recv_topk_weights = std::optional<deep_ep::detail::Tensor>(),
       recv_x_scales = std::optional<deep_ep::detail::Tensor>();
  auto recv_src_meta = std::optional<deep_ep::detail::Tensor>();
  auto recv_rdma_channel_prefix_matrix =
      std::optional<deep_ep::detail::Tensor>();
  auto recv_gbl_channel_prefix_matrix =
      std::optional<deep_ep::detail::Tensor>();
  auto send_rdma_head = std::optional<deep_ep::detail::Tensor>();
  auto send_nvl_head = std::optional<deep_ep::detail::Tensor>();
  if (!cached_mode) {
    recv_src_meta =
        ConvertPaddleTensorToDetailTensor(paddle::experimental::empty(
            {num_recv_tokens, internode::get_source_meta_bytes()},
            phi::DataType::INT8,
            phi::GPUPlace(device_id)));
    recv_rdma_channel_prefix_matrix = ConvertPaddleTensorToDetailTensor(
        paddle::experimental::empty({num_rdma_ranks, num_channels},
                                    phi::DataType::INT32,
                                    phi::GPUPlace(device_id)));
    recv_gbl_channel_prefix_matrix = ConvertPaddleTensorToDetailTensor(
        paddle::experimental::empty({num_ranks, num_channels},
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

  // Launch data dispatch
  // NOTES: the buffer size checks are moved into the `.cu` file
  internode::dispatch(
      recv_x.data_ptr(),
      recv_x_scales_ptr,
      recv_topk_idx_ptr,
      recv_topk_weights_ptr,
      cached_mode ? nullptr : recv_src_meta->data_ptr(),
      x.data_ptr(),
      x_scales_ptr,
      topk_idx_ptr,
      topk_weights_ptr,
      cached_mode ? nullptr : send_rdma_head->data_ptr<int>(),
      cached_mode ? nullptr : send_nvl_head->data_ptr<int>(),
      cached_mode ? nullptr : recv_rdma_channel_prefix_matrix->data_ptr<int>(),
      cached_mode ? nullptr : recv_gbl_channel_prefix_matrix->data_ptr<int>(),
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
      low_latency_mode);

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
                    recv_gbl_rank_prefix_sum}) {
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
                     recv_gbl_channel_prefix_matrix,
                     send_rdma_head,
                     send_nvl_head,
                     recv_src_meta}) {
      to.has_value() ? to->record_stream(comm_stream) : void();
      if (allocate_on_comm_stream)
        to.has_value() ? to->record_stream(compute_stream) : void();
    }
  } else {
    stream_wait(compute_stream, comm_stream);
  }

  // Switch back compute stream
  if (allocate_on_comm_stream) {
    deep_ep::detail::SetAllocatorStreamForGPUContext(compute_stream, calc_ctx);
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

std::tuple<deep_ep::detail::Tensor,
           std::optional<deep_ep::detail::Tensor>,
           std::optional<EventHandle>>
Buffer::internode_combine(
    const deep_ep::detail::Tensor& x,
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
    bool allocate_on_comm_stream) {
  const int num_channels = config.num_sms / 2;
  EP_HOST_ASSERT(config.num_sms % 2 == 0);

  // Shape and contiguous checks
  EP_HOST_ASSERT(x.dim() == 2 && x.is_contiguous());
  EP_HOST_ASSERT(src_meta.dim() == 2 && src_meta.is_contiguous() &&
                 src_meta.scalar_type() == deep_ep::detail::kByte);
  EP_HOST_ASSERT(is_combined_token_in_rank.dim() == 2 &&
                 is_combined_token_in_rank.is_contiguous() &&
                 is_combined_token_in_rank.scalar_type() ==
                     deep_ep::detail::kBool);
  EP_HOST_ASSERT(rdma_channel_prefix_matrix.dim() == 2 &&
                 rdma_channel_prefix_matrix.is_contiguous() &&
                 rdma_channel_prefix_matrix.scalar_type() ==
                     deep_ep::detail::kInt32);
  EP_HOST_ASSERT(rdma_rank_prefix_sum.dim() == 1 &&
                 rdma_rank_prefix_sum.is_contiguous() &&
                 rdma_rank_prefix_sum.scalar_type() == deep_ep::detail::kInt32);
  EP_HOST_ASSERT(gbl_channel_prefix_matrix.dim() == 2 &&
                 gbl_channel_prefix_matrix.is_contiguous() &&
                 gbl_channel_prefix_matrix.scalar_type() ==
                     deep_ep::detail::kInt32);
  EP_HOST_ASSERT(combined_rdma_head.dim() == 2 &&
                 combined_rdma_head.is_contiguous() &&
                 combined_rdma_head.scalar_type() == deep_ep::detail::kInt32);
  EP_HOST_ASSERT(combined_nvl_head.dim() == 2 &&
                 combined_nvl_head.is_contiguous() &&
                 combined_nvl_head.scalar_type() == deep_ep::detail::kInt32);

  auto num_tokens = static_cast<int>(x.size(0)),
       hidden = static_cast<int>(x.size(1)),
       hidden_int4 =
           static_cast<int>(x.size(1) * x.element_size() / sizeof(int4));
  auto num_combined_tokens =
      static_cast<int>(is_combined_token_in_rank.size(0));
  EP_HOST_ASSERT((hidden * x.element_size()) % sizeof(int4) == 0);
  EP_HOST_ASSERT(src_meta.size(1) == internode::get_source_meta_bytes());
  EP_HOST_ASSERT(is_combined_token_in_rank.size(1) == num_ranks);
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
    deep_ep::detail::SetAllocatorStreamForGPUContext(comm_stream, calc_ctx);
  }

  // Wait previous tasks to be finished
  if (previous_event.has_value()) {
    stream_wait(comm_stream, previous_event.value());
  } else {
    stream_wait(comm_stream, compute_stream);
  }

  // Top-k checks
  int num_topk = 0;
  auto combined_topk_weights = std::optional<deep_ep::detail::Tensor>();
  float* topk_weights_ptr = nullptr;
  float* combined_topk_weights_ptr = nullptr;
  if (topk_weights.has_value()) {
    EP_HOST_ASSERT(topk_weights->dim() == 2 && topk_weights->is_contiguous());
    EP_HOST_ASSERT(topk_weights->size(0) == num_tokens);
    EP_HOST_ASSERT(topk_weights->scalar_type() == deep_ep::detail::kFloat32);
    num_topk = static_cast<int>(topk_weights->size(1));
    topk_weights_ptr = topk_weights->data_ptr<float>();
    combined_topk_weights = ConvertPaddleTensorToDetailTensor(
        paddle::experimental::empty({num_combined_tokens, num_topk},
                                    topk_weights->dtype(),
                                    topk_weights->place()));
    combined_topk_weights_ptr = combined_topk_weights->data_ptr<float>();
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
  auto combined_x =
      ConvertPaddleTensorToDetailTensor(paddle::experimental::empty(
          {num_combined_tokens, hidden}, x.dtype(), x.place()));
  internode::combine(deep_ep::detail::ScalarTypeToCudaDataType(x.scalar_type()),
                     combined_x.data_ptr(),
                     combined_topk_weights_ptr,
                     is_combined_token_in_rank.data_ptr<bool>(),
                     x.data_ptr(),
                     topk_weights_ptr,
                     combined_rdma_head.data_ptr<int>(),
                     combined_nvl_head.data_ptr<int>(),
                     src_meta.data_ptr(),
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
                     low_latency_mode);

  // Wait streams
  std::optional<EventHandle> event;
  if (async) {
    event = EventHandle(comm_stream);
    for (auto& t : {x,
                    src_meta,
                    is_combined_token_in_rank,
                    rdma_channel_prefix_matrix,
                    rdma_rank_prefix_sum,
                    gbl_channel_prefix_matrix,
                    combined_x,
                    combined_rdma_head,
                    combined_nvl_head}) {
      t.record_stream(comm_stream);
      if (allocate_on_comm_stream) t.record_stream(compute_stream);
    }
    for (auto& to : {topk_weights, combined_topk_weights}) {
      to.has_value() ? to->record_stream(comm_stream) : void();
      if (allocate_on_comm_stream)
        to.has_value() ? to->record_stream(compute_stream) : void();
    }
  } else {
    stream_wait(compute_stream, comm_stream);
  }

  // Switch back compute stream
  if (allocate_on_comm_stream) {
    deep_ep::detail::SetAllocatorStreamForGPUContext(compute_stream, calc_ctx);
  }

  // Return values
  return {combined_x, combined_topk_weights, event};
}
#endif  // PADDLE_WITH_NVSHMEM

void Buffer::clean_low_latency_buffer(int num_max_dispatch_tokens_per_rank,
                                      int hidden,
                                      int num_experts) {
#ifdef PADDLE_WITH_NVSHMEM
  EP_HOST_ASSERT(low_latency_mode);

  auto layout = LowLatencyLayout(rdma_buffer_ptr,
                                 num_max_dispatch_tokens_per_rank,
                                 hidden,
                                 num_ranks,
                                 num_experts);
  auto clean_meta_0 = layout.buffers[0].clean_meta();
  auto clean_meta_1 = layout.buffers[1].clean_meta();

  auto check_boundary = [=](void* ptr, size_t num_bytes) {
    auto offset = reinterpret_cast<int64_t>(ptr) -
                  reinterpret_cast<int64_t>(rdma_buffer_ptr);
    EP_HOST_ASSERT(0 <= offset &&
                   offset + static_cast<int64_t>(num_bytes) <= num_rdma_bytes);
  };
  check_boundary(clean_meta_0.first, clean_meta_0.second * sizeof(int));
  check_boundary(clean_meta_1.first, clean_meta_1.second * sizeof(int));

  internode_ll::clean_low_latency_buffer(clean_meta_0.first,
                                         clean_meta_0.second,
                                         clean_meta_1.first,
                                         clean_meta_1.second,
                                         calc_ctx->stream());
#else
  LOG(ERROR) << "NVSHMEM is not enabled. You can enable it by setting cmake "
                "option WITH_NVSHMEM=ON.";
#endif
}

void Buffer::barrier_all() { internode_ll::barrier_all(calc_ctx->stream()); }

#ifdef PADDLE_WITH_NVSHMEM
std::tuple<deep_ep::detail::Tensor,
           std::optional<deep_ep::detail::Tensor>,
           deep_ep::detail::Tensor,
           deep_ep::detail::Tensor,
           deep_ep::detail::Tensor,
           std::optional<EventHandle>,
           std::optional<std::function<void()>>>
Buffer::low_latency_dispatch(const deep_ep::detail::Tensor& x,
                             const deep_ep::detail::Tensor& topk_idx,
                             int num_max_dispatch_tokens_per_rank,
                             int num_experts,
                             bool use_fp8,
                             bool async,
                             bool return_recv_hook) {
  EP_HOST_ASSERT(low_latency_mode);

  // Tensor checks
  // By default using `ptp128c` FP8 cast
  EP_HOST_ASSERT(x.dim() == 2 && x.is_contiguous() &&
                 x.scalar_type() == deep_ep::detail::kBFloat16);
  EP_HOST_ASSERT(x.size(1) % sizeof(int4) == 0 && x.size(1) % 128 == 0);
  EP_HOST_ASSERT(topk_idx.dim() == 2 && topk_idx.is_contiguous());
  EP_HOST_ASSERT(x.size(0) == topk_idx.size(0) &&
                 x.size(0) <= num_max_dispatch_tokens_per_rank);
  EP_HOST_ASSERT(topk_idx.scalar_type() == deep_ep::detail::kInt64);
  EP_HOST_ASSERT(num_experts % num_ranks == 0);

  auto num_tokens = static_cast<int>(x.size(0)),
       hidden = static_cast<int>(x.size(1));
  auto num_scales = hidden / 128, num_topk = static_cast<int>(topk_idx.size(1));
  int num_local_experts = num_experts / num_ranks;

  // Buffer control
  LowLatencyLayout layout(rdma_buffer_ptr,
                          num_max_dispatch_tokens_per_rank,
                          hidden,
                          num_ranks,
                          num_experts);
  EP_HOST_ASSERT(static_cast<int64_t>(layout.total_bytes) <= num_rdma_bytes);
  auto buffer = layout.buffers[low_latency_buffer_idx];
  auto next_buffer = layout.buffers[low_latency_buffer_idx ^= 1];

  // Wait previous tasks to be finished
  // NOTES: the hook mode will always use the default stream
  auto compute_stream = calc_ctx->stream();
  auto launch_stream = return_recv_hook ? compute_stream : comm_stream;
  EP_HOST_ASSERT(!(async && return_recv_hook));
  if (!return_recv_hook) stream_wait(launch_stream, compute_stream);

  // Allocate packed tensors
  auto packed_recv_x =
      ConvertPaddleTensorToDetailTensor(paddle::experimental::empty(
          {num_local_experts,
           num_ranks * num_max_dispatch_tokens_per_rank,
           hidden},
          use_fp8 ? phi::DataType::FLOAT8_E4M3FN : phi::DataType::BFLOAT16,
          x.place()));
  auto packed_recv_src_info =
      ConvertPaddleTensorToDetailTensor(paddle::experimental::empty(
          {num_local_experts, num_ranks * num_max_dispatch_tokens_per_rank},
          phi::DataType::INT32,
          phi::GPUPlace(device_id)));
  auto packed_recv_layout_range = ConvertPaddleTensorToDetailTensor(
      paddle::experimental::empty({num_local_experts, num_ranks},
                                  phi::DataType::INT64,
                                  phi::GPUPlace(device_id)));
  auto packed_recv_count =
      ConvertPaddleTensorToDetailTensor(paddle::experimental::empty(
          {num_local_experts}, phi::DataType::INT32, phi::GPUPlace(device_id)));

  // Allocate column-majored scales
  auto packed_recv_x_scales = std::optional<deep_ep::detail::Tensor>();

  float* packed_recv_x_scales_ptr = nullptr;

  if (use_fp8) {
    EP_HOST_ASSERT((num_ranks * num_max_dispatch_tokens_per_rank) % 4 == 0 &&
                   "TMA requires the number of tokens to be multiple of 4");
    packed_recv_x_scales =
        ConvertPaddleTensorToDetailTensor(paddle::experimental::empty(
            {num_local_experts,
             num_scales,
             num_ranks * num_max_dispatch_tokens_per_rank},
            phi::DataType::FLOAT32,
            phi::GPUPlace(device_id)));
    packed_recv_x_scales =
        ConvertPaddleTensorToDetailTensor(paddle::experimental::transpose(
            ConvertDetailTensorToPaddleTensor(packed_recv_x_scales.value()),
            std::vector<int>{0, 2, 1}));
    packed_recv_x_scales_ptr = packed_recv_x_scales.value().data_ptr<float>();
  }

  // Kernel launch
  auto next_clean_meta = next_buffer.clean_meta();
  auto launcher = [=](int phases) {
    internode_ll::dispatch(packed_recv_x.data_ptr(),
                           packed_recv_x_scales_ptr,
                           packed_recv_src_info.data_ptr<int>(),
                           packed_recv_layout_range.data_ptr<int64_t>(),
                           packed_recv_count.data_ptr<int>(),
                           buffer.dispatch_rdma_recv_data_buffer,
                           buffer.dispatch_rdma_recv_count_buffer,
                           buffer.dispatch_rdma_send_buffer,
                           x.data_ptr(),
                           topk_idx.data_ptr<int64_t>(),
                           next_clean_meta.first,
                           next_clean_meta.second,
                           num_tokens,
                           hidden,
                           num_max_dispatch_tokens_per_rank,
                           num_topk,
                           num_experts,
                           rank,
                           num_ranks,
                           use_fp8,
                           workspace,
                           launch_stream,
                           phases);
  };
  launcher(return_recv_hook
               ? LOW_LATENCY_SEND_PHASE
               : (LOW_LATENCY_SEND_PHASE | LOW_LATENCY_RECV_PHASE));

  // Wait streams
  std::optional<EventHandle> event;
  if (async) {
    // NOTES: we must ensure the all tensors will not be deallocated before the
    // stream-wait happens, so in Python API, we must wrap all tensors into the
    // event handle.
    event = EventHandle(launch_stream);
  } else if (!return_recv_hook) {
    stream_wait(compute_stream, launch_stream);
  }

  // Receiver callback
  std::optional<std::function<void()>> recv_hook = std::nullopt;
  if (return_recv_hook) recv_hook = [=]() { launcher(LOW_LATENCY_RECV_PHASE); };

  // Return values
  return {packed_recv_x,
          packed_recv_x_scales,
          packed_recv_count,
          packed_recv_src_info,
          packed_recv_layout_range,
          event,
          recv_hook};
}

std::tuple<deep_ep::detail::Tensor,
           std::optional<EventHandle>,
           std::optional<std::function<void()>>>
Buffer::low_latency_combine(const deep_ep::detail::Tensor& x,
                            const deep_ep::detail::Tensor& topk_idx,
                            const deep_ep::detail::Tensor& topk_weights,
                            const deep_ep::detail::Tensor& src_info,
                            const deep_ep::detail::Tensor& layout_range,
                            int num_max_dispatch_tokens_per_rank,
                            int num_experts,
                            bool async,
                            bool return_recv_hook) {
  EP_HOST_ASSERT(low_latency_mode);

  // Tensor checks
  EP_HOST_ASSERT(x.dim() == 3 && x.is_contiguous() &&
                 x.scalar_type() == deep_ep::detail::kBFloat16);
  EP_HOST_ASSERT(x.size(0) == num_experts / num_ranks);
  EP_HOST_ASSERT(x.size(1) == num_ranks * num_max_dispatch_tokens_per_rank);
  EP_HOST_ASSERT(x.size(2) % sizeof(int4) == 0 && x.size(2) % 128 == 0);
  EP_HOST_ASSERT(topk_idx.dim() == 2 && topk_idx.is_contiguous());
  EP_HOST_ASSERT(topk_idx.size(0) == topk_weights.size(0) &&
                 topk_idx.size(1) == topk_weights.size(1));
  EP_HOST_ASSERT(topk_idx.scalar_type() == deep_ep::detail::kInt64);
  EP_HOST_ASSERT(topk_weights.dim() == 2 && topk_weights.is_contiguous());
  EP_HOST_ASSERT(topk_weights.size(0) <= num_max_dispatch_tokens_per_rank);
  EP_HOST_ASSERT(topk_weights.scalar_type() == deep_ep::detail::kFloat32);
  EP_HOST_ASSERT(src_info.dim() == 2 && src_info.is_contiguous());
  EP_HOST_ASSERT(src_info.scalar_type() == deep_ep::detail::kInt32 &&
                 x.size(0) == src_info.size(0));
  EP_HOST_ASSERT(layout_range.dim() == 2 && layout_range.is_contiguous());
  EP_HOST_ASSERT(layout_range.scalar_type() == deep_ep::detail::kInt64);
  EP_HOST_ASSERT(layout_range.size(0) == num_experts / num_ranks &&
                 layout_range.size(1) == num_ranks);
  auto hidden = static_cast<int>(x.size(2));
  auto num_local_experts = num_experts / num_ranks,
       num_topk = static_cast<int>(topk_weights.size(1));
  (void)num_local_experts;
  auto num_combined_tokens = static_cast<int>(topk_weights.size(0));

  // Buffer control
  LowLatencyLayout layout(rdma_buffer_ptr,
                          num_max_dispatch_tokens_per_rank,
                          hidden,
                          num_ranks,
                          num_experts);
  EP_HOST_ASSERT(static_cast<int64_t>(layout.total_bytes) <= num_rdma_bytes);
  auto buffer = layout.buffers[low_latency_buffer_idx];
  auto next_buffer = layout.buffers[low_latency_buffer_idx ^= 1];

  // Wait previous tasks to be finished
  // NOTES: the hook mode will always use the default stream
  auto compute_stream = calc_ctx->stream();
  auto launch_stream = return_recv_hook ? compute_stream : comm_stream;
  EP_HOST_ASSERT(!(async && return_recv_hook));
  if (!return_recv_hook) stream_wait(launch_stream, compute_stream);

  // Allocate output tensor
  auto combined_x =
      ConvertPaddleTensorToDetailTensor(paddle::experimental::empty(
          {num_combined_tokens, hidden}, x.dtype(), x.place()));

  // Kernel launch
  auto next_clean_meta = next_buffer.clean_meta();
  auto launcher = [=](int phases) {
    internode_ll::combine(combined_x.data_ptr(),
                          buffer.combine_rdma_recv_data_buffer,
                          buffer.combine_rdma_recv_flag_buffer,
                          buffer.combine_rdma_send_buffer,
                          x.data_ptr(),
                          topk_idx.data_ptr<int64_t>(),
                          topk_weights.data_ptr<float>(),
                          src_info.data_ptr<int>(),
                          layout_range.data_ptr<int64_t>(),
                          next_clean_meta.first,
                          next_clean_meta.second,
                          num_combined_tokens,
                          hidden,
                          num_max_dispatch_tokens_per_rank,
                          num_topk,
                          num_experts,
                          rank,
                          num_ranks,
                          workspace,
                          launch_stream,
                          phases);
  };
  launcher(return_recv_hook
               ? LOW_LATENCY_SEND_PHASE
               : (LOW_LATENCY_SEND_PHASE | LOW_LATENCY_RECV_PHASE));

  // Wait streams
  std::optional<EventHandle> event;
  if (async) {
    // NOTES: we must ensure the all tensors will not be deallocated before the
    // stream-wait happens, so in Python API, we must wrap all tensors into the
    // event handle.
    event = EventHandle(launch_stream);
  } else if (!return_recv_hook) {
    stream_wait(compute_stream, launch_stream);
  }

  // Receiver callback
  std::optional<std::function<void()>> recv_hook = std::nullopt;
  if (return_recv_hook) recv_hook = [=]() { launcher(LOW_LATENCY_RECV_PHASE); };

  // Return values
  return std::tuple<deep_ep::detail::Tensor,
                    std::optional<EventHandle>,
                    std::optional<std::function<void()>>>{
      deep_ep::detail::Tensor{combined_x}, event, recv_hook};
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
    int expert_alignment,
    const Config& config,
    std::optional<EventHandle>& previous_event,  // NOLINT
    bool async,
    bool allocate_on_comm_stream) {
#ifdef PADDLE_WITH_NVSHMEM
  const auto& x_ = ConvertPaddleTensorToDetailTensor(x);
  std::optional<deep_ep::detail::Tensor> x_scales_ =
      ConvertOptionalPaddleTensorToDetailTensor(x_scales);

  std::optional<deep_ep::detail::Tensor> topk_idx_ =
      ConvertOptionalPaddleTensorToDetailTensor(topk_idx);
  std::optional<deep_ep::detail::Tensor> topk_weights_ =
      ConvertOptionalPaddleTensorToDetailTensor(topk_weights);
  std::optional<deep_ep::detail::Tensor> num_tokens_per_rank_ =
      ConvertOptionalPaddleTensorToDetailTensor(num_tokens_per_rank);
  std::optional<deep_ep::detail::Tensor> num_tokens_per_rdma_rank_ =
      ConvertOptionalPaddleTensorToDetailTensor(num_tokens_per_rdma_rank);

  const auto& is_token_in_rank_ =
      ConvertPaddleTensorToDetailTensor(is_token_in_rank);
  std::optional<deep_ep::detail::Tensor> num_tokens_per_expert_ =
      ConvertOptionalPaddleTensorToDetailTensor(num_tokens_per_expert);

  std::optional<deep_ep::detail::Tensor> cached_rdma_channel_prefix_matrix_ =
      ConvertOptionalPaddleTensorToDetailTensor(
          cached_rdma_channel_prefix_matrix);
  std::optional<deep_ep::detail::Tensor> cached_recv_rdma_rank_prefix_sum_ =
      ConvertOptionalPaddleTensorToDetailTensor(
          cached_recv_rdma_rank_prefix_sum);
  std::optional<deep_ep::detail::Tensor> cached_gbl_channel_prefix_matrix_ =
      ConvertOptionalPaddleTensorToDetailTensor(
          cached_gbl_channel_prefix_matrix);
  std::optional<deep_ep::detail::Tensor> cached_recv_gbl_rank_prefix_sum_ =
      ConvertOptionalPaddleTensorToDetailTensor(
          cached_recv_gbl_rank_prefix_sum);

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
                                expert_alignment,
                                config,
                                previous_event,
                                async,
                                allocate_on_comm_stream);

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

std::tuple<paddle::Tensor,
           std::optional<paddle::Tensor>,
           std::optional<EventHandle>>
Buffer::internode_combine_api(
    const paddle::Tensor& x,
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
    bool allocate_on_comm_stream) {
#ifdef PADDLE_WITH_NVSHMEM
  const auto& x_ = ConvertPaddleTensorToDetailTensor(x);

  std::optional<deep_ep::detail::Tensor> topk_weights_ =
      ConvertOptionalPaddleTensorToDetailTensor(topk_weights);

  const auto& src_meta_ = ConvertPaddleTensorToDetailTensor(src_meta);
  const auto& is_combined_token_in_rank_ =
      ConvertPaddleTensorToDetailTensor(is_combined_token_in_rank);

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

  auto res = internode_combine(x_,
                               topk_weights_,
                               src_meta_,
                               is_combined_token_in_rank_,
                               rdma_channel_prefix_matrix_,
                               rdma_rank_prefix_sum_,
                               gbl_channel_prefix_matrix_,
                               combined_rdma_head_,
                               combined_nvl_head_,
                               config,
                               previous_event,
                               async,
                               allocate_on_comm_stream);

  auto combined_x_ = ConvertDetailTensorToPaddleTensor(std::get<0>(res));
  std::optional<paddle::Tensor> combined_topk_weights_ =
      ConvertOptionalDetailTensorToPaddleTensor(std::get<1>(res));

  const auto& event = std::get<2>(res);

  return {combined_x_, combined_topk_weights_, event};
#else
  LOG(ERROR) << "NVSHMEM is not enabled. You can enable it by setting cmake "
                "option WITH_NVSHMEM=ON.";
  return {};
#endif
}

std::tuple<paddle::Tensor,
           std::optional<paddle::Tensor>,
           paddle::Tensor,
           paddle::Tensor,
           paddle::Tensor,
           std::optional<EventHandle>,
           std::optional<std::function<void()>>>
Buffer::low_latency_dispatch_api(const paddle::Tensor& x,
                                 const paddle::Tensor& topk_idx,
                                 int num_max_dispatch_tokens_per_rank,
                                 int num_experts,
                                 bool use_fp8,
                                 bool async,
                                 bool return_recv_hook) {
#ifdef PADDLE_WITH_NVSHMEM
  const auto& x_ = ConvertPaddleTensorToDetailTensor(x);
  const auto& topk_idx_ = ConvertPaddleTensorToDetailTensor(topk_idx);

  auto res = low_latency_dispatch(x_,
                                  topk_idx_,
                                  num_max_dispatch_tokens_per_rank,
                                  num_experts,
                                  use_fp8,
                                  async,
                                  return_recv_hook);

  auto packed_recv_x_ = ConvertDetailTensorToPaddleTensor(std::get<0>(res));

  std::optional<paddle::Tensor> packed_recv_x_scales_;
  if (std::get<1>(res).has_value()) {
    packed_recv_x_scales_ =
        ConvertDetailTensorToPaddleTensor(std::get<1>(res).value());
  }

  auto packed_recv_count_ = ConvertDetailTensorToPaddleTensor(std::get<2>(res));
  auto packed_recv_src_info_ =
      ConvertDetailTensorToPaddleTensor(std::get<3>(res));
  auto packed_recv_layout_range_ =
      ConvertDetailTensorToPaddleTensor(std::get<4>(res));

  const auto& event = std::get<5>(res);
  auto recv_hook = std::get<6>(res);

  return {packed_recv_x_,
          packed_recv_x_scales_,
          packed_recv_count_,
          packed_recv_src_info_,
          packed_recv_layout_range_,
          event,
          recv_hook};
#else
  LOG(ERROR) << "NVSHMEM is not enabled. You can enable it by setting cmake "
                "option WITH_NVSHMEM=ON.";
  return {};
#endif
}

std::tuple<paddle::Tensor,
           std::optional<EventHandle>,
           std::optional<std::function<void()>>>
Buffer::low_latency_combine_api(const paddle::Tensor& x,
                                const paddle::Tensor& topk_idx,
                                const paddle::Tensor& topk_weights,
                                const paddle::Tensor& src_info,
                                const paddle::Tensor& layout_range,
                                int num_max_dispatch_tokens_per_rank,
                                int num_experts,
                                bool async,
                                bool return_recv_hook) {
#ifdef PADDLE_WITH_NVSHMEM
  const auto& x_ = ConvertPaddleTensorToDetailTensor(x);
  const auto& topk_idx_ = ConvertPaddleTensorToDetailTensor(topk_idx);
  const auto& topk_weights_ = ConvertPaddleTensorToDetailTensor(topk_weights);
  const auto& src_info_ = ConvertPaddleTensorToDetailTensor(src_info);
  const auto& layout_range_ = ConvertPaddleTensorToDetailTensor(layout_range);

  auto res = low_latency_combine(x_,
                                 topk_idx_,
                                 topk_weights_,
                                 src_info_,
                                 layout_range_,
                                 num_max_dispatch_tokens_per_rank,
                                 num_experts,
                                 async,
                                 return_recv_hook);

  auto combined_x_ = ConvertDetailTensorToPaddleTensor(std::get<0>(res));
  const auto& event = std::get<1>(res);
  auto recv_hook = std::get<2>(res);

  return {combined_x_, event, recv_hook};
#else
  LOG(ERROR) << "NVSHMEM is not enabled. You can enable it by setting cmake "
                "option WITH_NVSHMEM=ON.";
  return {};
#endif
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

std::optional<deep_ep::detail::Tensor>
ConvertOptionalPaddleTensorToDetailTensor(
    const std::optional<paddle::Tensor>& tensor) {
  std::optional<deep_ep::detail::Tensor> res;
  if (tensor.has_value()) {
    res = ConvertPaddleTensorToDetailTensor(tensor.value());
  }
  return res;
}

std::optional<paddle::Tensor> ConvertOptionalDetailTensorToPaddleTensor(
    const std::optional<deep_ep::detail::Tensor>& tensor) {
  std::optional<paddle::Tensor> res;
  if (tensor.has_value()) {
    res = ConvertDetailTensorToPaddleTensor(tensor.value());
  }
  return res;
}

}  // namespace deep_ep
