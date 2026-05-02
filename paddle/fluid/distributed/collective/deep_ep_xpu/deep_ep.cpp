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

#include "paddle/fluid/distributed/collective/deep_ep_xpu/deep_ep.hpp"
#include "paddle/fluid/distributed/collective/deep_ep_xpu/kernels/configs.h"

#include "paddle/fluid/distributed/collective/deep_ep_xpu/include/CUDADataType.h"
#include "paddle/fluid/distributed/collective/deep_ep_xpu/include/ScalarType.h"
#include "paddle/fluid/distributed/collective/process_group_bkcl.h"
#include "paddle/phi/api/include/api.h"
#include "paddle/phi/api/include/tensor_utils.h"
#include "paddle/phi/common/data_type.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/distributed/utils.h"
#include "paddle/phi/core/memory/allocation/allocator_facade.h"

COMMON_DECLARE_int64(deep_ep_comm_prealloc_in_mb);

namespace paddle::deep_ep {
std::once_flag pre_alloc_once_flag;

namespace detail {
void SetAllocatorStreamForGPUContext(cudaStream_t stream,
                                     phi::XPUContext* ctx) {
  ctx->SetAllocator(
      paddle::memory::allocation::AllocatorFacade::Instance()
          .GetAllocator(ctx->GetPlace(), reinterpret_cast<XPUStream>(stream))
          .get());
}
}  // namespace detail

void PreAlloc(Tensor tensor, cudaStream_t stream) {
  int64_t numel = tensor.numel();
  auto alloc_size = FLAGS_deep_ep_comm_prealloc_in_mb * 1000000;
  std::cout << "alloc once here, size: " << alloc_size << " numel: " << numel
            << std::endl;
  std::cout << tensor.place() << "\t" << stream << std::endl;
  paddle::memory::allocation::AllocatorFacade::Instance()
      .GetAllocator(tensor.place(), stream)
      ->Allocate(alloc_size);
}

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
  const auto& place = XPUPlace(device_id);
  comm_ctx =
      reinterpret_cast<paddle::distributed::ProcessGroupBKCL*>(pg)
          ->GetOrCreateCommContext(place, phi::distributed::CommType::ALLTOALL);
  comm_stream = reinterpret_cast<cudaStream_t>(comm_ctx->GetStream());
  calc_ctx = reinterpret_cast<phi::XPUContext*>(
      reinterpret_cast<paddle::distributed::ProcessGroupBKCL*>(pg)
          ->GetDeviceContext(place, true));

  ep_runtime = std::make_unique<DeepEPBuffer>(comm_ctx->GetBKCLComm());

  VLOG(3) << "DeepEP buffer device_id " << device_id << " context_ring_id "
          << context_ring_id << " comm_stream "
          << reinterpret_cast<void*>(comm_stream) << " compute_stream "
          << reinterpret_cast<cudaStream_t>(calc_ctx->stream());

  // Task fifo memory
  int64_t fifo_bytes = sizeof(int) * NUM_MAX_FIFO_SLOTS;
  int64_t buffer_ptr_bytes = sizeof(void*) * NUM_MAX_NVL_PEERS;
  int64_t task_ptr_bytes = sizeof(int*) * NUM_MAX_NVL_PEERS;

  EP_HOST_ASSERT(
      num_rdma_bytes % NUM_BUFFER_ALIGNMENT_BYTES == 0 &&
      (low_latency_mode || num_rdma_bytes <= std::numeric_limits<int>::max()));
  EP_HOST_ASSERT(0 <= rank && rank < num_ranks &&
                 (num_ranks <= NUM_MAX_NVL_PEERS * NUM_MAX_RDMA_PEERS ||
                  low_latency_mode));
  EP_HOST_ASSERT(num_ranks < NUM_MAX_NVL_PEERS ||
                 num_ranks % NUM_MAX_NVL_PEERS == 0);

  // Get ranks
  rdma_rank = rank / NUM_MAX_NVL_PEERS, nvl_rank = rank % NUM_MAX_NVL_PEERS;
  num_rdma_ranks = std::max(1, num_ranks / NUM_MAX_NVL_PEERS),
  num_nvl_ranks = std::min(num_ranks, NUM_MAX_NVL_PEERS);
  available = true;
  VLOG(3) << "DeepEP buffer init end, rdma_rank " << rdma_rank << " nvl_rank "
          << nvl_rank << " num_rdma_ranks " << num_rdma_ranks
          << " num_nvl_ranks " << num_nvl_ranks;
}

Buffer::~Buffer() noexcept(false) { CUDA_CHECK(cudaDeviceSynchronize()); }

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

cudaStream_t Buffer::get_comm_stream() const { return comm_stream; }

#ifndef PADDLE_NO_PYTHON
pybind11::bytearray Buffer::get_local_ipc_handle() const {
  return {ipc_handles[nvl_rank].reserved, CUDA_IPC_HANDLE_SIZE};
}

pybind11::bytearray Buffer::get_local_nvshmem_unique_id() const {
  return {reinterpret_cast<const char*>(""), sizeof(BKCLUniqueId)};
}

void Buffer::sync(
    const std::vector<int>& device_ids,
    const std::vector<std::optional<pybind11::bytearray>>& all_gathered_handles,
    const std::optional<pybind11::bytearray>& root_unique_id_opt) {
  if (num_rdma_ranks > 1 || low_latency_mode) {
    int ret = bkcl_xshmem_init(comm_ctx->GetBKCLComm());
    EP_HOST_ASSERT(ret == 0 && "bkcl_xshmem_init failed");
  }
}
#endif

std::tuple<dTensor,
           std::optional<dTensor>,
           dTensor,
           dTensor,
           std::optional<EventHandle>>
Buffer::get_dispatch_layout(const dTensor& topk_idx,
                            int num_experts,
                            std::optional<EventHandle>& previous_event,
                            bool async,
                            bool allocate_on_comm_stream) {
  EP_HOST_ASSERT(topk_idx.dim() == 2);
  EP_HOST_ASSERT(topk_idx.is_contiguous());
  EP_HOST_ASSERT(num_experts > 0);

  // Allocate all tensors on comm stream if set
  // NOTES: do not allocate tensors upfront!
  auto compute_stream = reinterpret_cast<cudaStream_t>(calc_ctx->stream());
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
  auto num_tokens_per_rank = GetDetailTensor(paddle::experimental::empty(
      {num_ranks}, DataType::INT32, XPUPlace(device_id)));
  auto num_tokens_per_rdma_rank = std::optional<dTensor>();
  auto num_tokens_per_expert = GetDetailTensor(paddle::experimental::empty(
      {num_experts}, DataType::INT32, XPUPlace(device_id)));
  auto is_token_in_rank = GetDetailTensor(paddle::experimental::empty(
      {num_tokens, num_ranks}, DataType::BOOL, XPUPlace(device_id)));
  if (is_internode_available()) {
    num_tokens_per_rdma_rank = GetDetailTensor(paddle::experimental::empty(
        {num_rdma_ranks}, DataType::INT32, XPUPlace(device_id)));
  }

  // Wait streams
  std::optional<EventHandle> event;
  if (async) {
    event = EventHandle(comm_stream);
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
Buffer::intranode_dispatch(
    const dTensor& x,
    const std::optional<dTensor>& x_scales,
    const std::optional<dTensor>& topk_idx,
    const std::optional<dTensor>& topk_weights,
    const std::optional<dTensor>& num_tokens_per_rank,
    const dTensor& is_token_in_rank,
    const std::optional<dTensor>& num_tokens_per_expert,
    int cached_num_recv_tokens,  // num_experts in cache mode
    const std::optional<dTensor>&
        cached_rank_prefix_matrix,  // topk_idx in cache mode
    const std::optional<dTensor>&
        cached_channel_prefix_matrix,  // took_weights in cache mode
    int expert_alignment,
    const Config& config,
    std::optional<EventHandle>& previous_event,  // NOLINT
    bool async,
    bool allocate_on_comm_stream) {
  int curr_num_experts;
  std::optional<dTensor> curr_topk_idx;
  std::optional<dTensor> curr_topk_weights;

  if (topk_idx.has_value()) {
    EP_HOST_ASSERT(topk_weights.has_value() &&
                   num_tokens_per_expert.has_value());
    curr_topk_idx =
        GetDetailTensor(cast_ad_func(topk_idx->raw_tensor(), DataType::INT32));
    curr_topk_weights =
        GetDetailTensor(assign_ad_func(topk_weights->raw_tensor()));
    curr_num_experts = static_cast<int>(num_tokens_per_expert->size(0));

  } else {  // cache mode
    EP_HOST_ASSERT(cached_rank_prefix_matrix.has_value() &&
                   cached_channel_prefix_matrix.has_value());
    curr_topk_idx = cached_rank_prefix_matrix;
    curr_topk_weights = cached_channel_prefix_matrix;
    curr_num_experts = cached_num_recv_tokens;
  }
  EP_HOST_ASSERT(curr_num_experts != 0);

  // Shape and contiguous checks
  EP_HOST_ASSERT(x.dim() == 2 && x.is_contiguous());
  auto num_tokens = static_cast<int>(x.size(0));
  int hidden_size = static_cast<int>(x.size(1));
  int num_topk = static_cast<int>(curr_topk_idx->size(1));
  auto num_local_experts = curr_num_experts / num_ranks;
  int ret = 0;

  // For int8 dispatch, the corresponding combine would be bf16,
  // so we must init buffer with bf16 here to avoid buffer overflow of combine.
  if (!init_normal_buffer) {
    ret = ep_runtime->init_normal_buffer(hidden_size, num_ranks, BKCL_BFLOAT16);
    EP_HOST_ASSERT(ret == 0 && "bkcl_init_normal_buffer failed");
    init_normal_buffer = true;
  }

  int num_scales = 0;
  bool use_int8 = false;
  if (x_scales.has_value()) {
    num_scales = static_cast<int>(x_scales->size(1));
    use_int8 = true;
  }

  // Allocate all tensors on comm stream if set
  // NOTES: do not allocate tensors upfront!
  auto compute_stream = reinterpret_cast<cudaStream_t>(calc_ctx->stream());

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

  auto d_num_recv_tokens_per_expert_list =
      GetDetailTensor(paddle::experimental::empty(
          {num_local_experts}, DataType::INT32, x.place()));
  auto h_num_recv_tokens_per_expert_list =
      std::vector<int>(num_local_experts, 0);
  auto rank_prefix_matrix = curr_topk_idx;
  auto channel_prefix_matrix = curr_topk_weights;
  auto recv_channel_prefix_matrix = GetDetailTensor(
      paddle::experimental::empty({num_ranks, 12}, DataType::INT32, x.place()));
  auto recv_src_idx = GetDetailTensor(paddle::experimental::empty(
      {curr_num_experts}, DataType::INT32, x.place()));
  auto send_head = GetDetailTensor(paddle::experimental::empty(
      {num_tokens, num_ranks}, DataType::INT32, x.place()));

  int num_recv_tokens =
      ep_runtime
          ->notify_dispatch_standard_with_num_recv_tokens_per_expert_list_cpu(
              x.data_ptr(),
              curr_topk_idx->data_ptr<int>(),
              curr_topk_weights->data_ptr<float>(),
              num_scales,
              hidden_size,
              num_tokens,
              num_topk,
              curr_num_experts,
              d_num_recv_tokens_per_expert_list
                  .data_ptr<int>(),  // should not be nullptr
              h_num_recv_tokens_per_expert_list.data(),
              ToBKCLDataType(x.dtype()),
              use_int8,
              async ? reinterpret_cast<XPUStream>(comm_stream)
                    : reinterpret_cast<XPUStream>(compute_stream));
  // num_tokens maybe 0, and num_recv_tokens also can be 0.
  EP_HOST_ASSERT(num_recv_tokens >= 0 &&
                 "bkcl_notify_dispatch_standard failed");

  auto recv_x = GetDetailTensor(paddle::experimental::empty(
      {num_recv_tokens, hidden_size}, x.dtype(), x.place()));
  std::optional<dTensor> recv_topk_idx =
      GetDetailTensor(paddle::experimental::empty({num_recv_tokens, num_topk},
                                                  curr_topk_idx->dtype(),
                                                  curr_topk_idx->place()));
  std::optional<dTensor> recv_topk_weights =
      GetDetailTensor(paddle::experimental::empty({num_recv_tokens, num_topk},
                                                  curr_topk_weights->dtype(),
                                                  curr_topk_weights->place()));

  auto recv_x_scales = std::optional<dTensor>();
  float* x_scales_ptr = nullptr;
  float* recv_x_scales_ptr = nullptr;

  if (x_scales.has_value()) {
    x_scales_ptr = const_cast<float*>(x_scales->data_ptr<float>());
    recv_x_scales = GetDetailTensor(paddle::experimental::empty(
        {num_recv_tokens, num_scales}, x_scales->dtype(), x_scales->place()));
    recv_x_scales_ptr = recv_x_scales->data_ptr<float>();
  }

  VLOG(3) << "DeepEP intranode_dispatch num_local_experts " << num_local_experts
          << " num_scales " << num_scales << " hidden_size " << hidden_size
          << " num_tokens " << num_tokens << " curr_num_experts "
          << curr_num_experts << " num_recv_tokens " << num_recv_tokens;
  VLOG(3) << "DeepEP intranode_dispatch x dim " << x.dim()
          << " curr_topk_idx dim " << curr_topk_idx->dim()
          << " curr_topk_weights dim " << curr_topk_weights->dim();

  ret = ep_runtime->normal_dispatch_standard(
      x.data_ptr(),  // sendbuf
      x_scales_ptr,
      curr_topk_idx->data_ptr<int>(),
      curr_topk_weights->data_ptr<float>(),
      recv_x.data_ptr(),
      recv_x_scales_ptr,
      recv_topk_idx->data_ptr<int>(),
      recv_topk_weights->data_ptr<float>(),
      num_scales,
      -1,  // UNUSED
      hidden_size,
      num_tokens,
      num_topk,
      curr_num_experts,
      ToBKCLDataType(x.dtype()),
      use_int8,
      async ? reinterpret_cast<XPUStream>(comm_stream)
            : reinterpret_cast<XPUStream>(compute_stream));
  EP_HOST_ASSERT(ret == 0 && "bkcl_normal_dispatch_standard failed");

  // Wait streams
  std::optional<EventHandle> event;
  if (async) {
    event = EventHandle(comm_stream);
    for (auto& t : {x, recv_x}) {
      t.record_stream(comm_stream);
      if (allocate_on_comm_stream) t.record_stream(compute_stream);
    }
    for (auto& to : {x_scales,
                     topk_idx,
                     topk_weights,
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
          h_num_recv_tokens_per_expert_list,
          rank_prefix_matrix.value(),     // topk_idx in cache mode
          channel_prefix_matrix.value(),  // topk_weights in cache mode
          recv_channel_prefix_matrix,
          recv_src_idx,  // num_experts in cache mode
          send_head,
          event};
}

std::tuple<dTensor, std::optional<dTensor>, std::optional<EventHandle>>
Buffer::intranode_combine(const dTensor& x,
                          const std::optional<dTensor>& topk_weights,
                          const dTensor& src_idx,
                          const dTensor& rank_prefix_matrix,
                          const dTensor& channel_prefix_matrix,
                          const dTensor& send_head,
                          const Config& config,
                          std::optional<EventHandle>& previous_event,
                          bool async,
                          bool allocate_on_comm_stream) {
  EP_HOST_ASSERT(x.dim() == 2 && x.is_contiguous());
  auto num_tokens = static_cast<int>(x.size(0)),
       hidden_size = static_cast<int>(x.size(1));
  auto num_combined_tokens = static_cast<int>(send_head.size(0));

  int ret = BKCL_SUCCESS;
  if (!init_normal_buffer) {
    ret = ep_runtime->init_normal_buffer(hidden_size, num_ranks, BKCL_BFLOAT16);
    EP_HOST_ASSERT(ret == 0 && "bkcl_init_normal_buffer failed");
    init_normal_buffer = true;
  }

  // Allocate all tensors on comm stream if set
  // NOTES: do not allocate tensors upfront!
  auto compute_stream = reinterpret_cast<cudaStream_t>(calc_ctx->stream());

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
  auto combined_topk_weights = std::optional<dTensor>();
  float* topk_weights_ptr = nullptr;
  float* combined_topk_weights_ptr = nullptr;
  if (topk_weights.has_value()) {
    EP_HOST_ASSERT(topk_weights->dim() == 2 && topk_weights->is_contiguous());
    EP_HOST_ASSERT(topk_weights->size(0) == num_tokens);
    EP_HOST_ASSERT(topk_weights->scalar_type() == deep_ep::detail::kFloat32);
    num_topk = static_cast<int>(topk_weights->size(1));
    topk_weights_ptr = topk_weights->data_ptr<float>();
    combined_topk_weights = GetDetailTensor(
        paddle::experimental::empty({num_combined_tokens, num_topk},
                                    topk_weights->dtype(),
                                    topk_weights->place()));
    combined_topk_weights_ptr = combined_topk_weights->data_ptr<float>();
  }

  // Combine data
  auto recv_x = GetDetailTensor(paddle::experimental::empty(
      {num_combined_tokens, hidden_size}, x.dtype(), x.place()));

  VLOG(3) << "DeepEP intranode_combine x.dim " << x.dim() << " num_tokens "
          << num_tokens << " num_combined_tokens " << num_combined_tokens
          << " num_topk " << num_topk << " topk_weights_ptr "
          << topk_weights_ptr << " combined_topk_weights_ptr "
          << combined_topk_weights_ptr;

  ret = ep_runtime->normal_combine_standard(
      x.data_ptr(),
      topk_weights_ptr,
      recv_x.data_ptr(),
      combined_topk_weights_ptr,
      hidden_size,
      num_tokens,
      num_combined_tokens,
      num_topk,
      0 /*num_experts*/,
      ToBKCLDataType(x.scalar_type()),
      async ? reinterpret_cast<XPUStream>(comm_stream)
            : reinterpret_cast<XPUStream>(compute_stream));

  // Wait streams
  std::optional<EventHandle> event;
  if (async) {
    event = EventHandle(comm_stream);
    for (auto& to : {x, recv_x}) {
      to.record_stream(comm_stream);
      if (allocate_on_comm_stream) to.record_stream(compute_stream);
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

  return {recv_x, combined_topk_weights, event};
}

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
Buffer::internode_dispatch(
    const dTensor& x,
    const std::optional<dTensor>& x_scales,
    const std::optional<dTensor>& topk_idx,
    const std::optional<dTensor>& topk_weights,
    const std::optional<dTensor>& num_tokens_per_rank,
    const std::optional<dTensor>& num_tokens_per_rdma_rank,
    const dTensor& is_token_in_rank,
    const std::optional<dTensor>& num_tokens_per_expert,
    int cached_num_recv_tokens,  // num_experts in cache mode
    int cached_num_rdma_recv_tokens,
    const std::optional<dTensor>& cached_rdma_channel_prefix_matrix,
    const std::optional<dTensor>&
        cached_recv_rdma_rank_prefix_sum,  // topk_weights in cache mode
    const std::optional<dTensor>& cached_gbl_channel_prefix_matrix,
    const std::optional<dTensor>&
        cached_recv_gbl_rank_prefix_sum,  // topk_idx in cache mode
    int expert_alignment,
    const Config& config,
    std::optional<EventHandle>& previous_event,  // NOLINT
    bool async,
    bool allocate_on_comm_stream) {
  int curr_num_experts;
  std::optional<dTensor> curr_topk_idx;
  std::optional<dTensor> curr_topk_weights;

  if (topk_idx.has_value()) {
    EP_HOST_ASSERT(topk_weights.has_value() &&
                   num_tokens_per_expert.has_value());
    curr_topk_idx =
        GetDetailTensor(cast_ad_func(topk_idx->raw_tensor(), DataType::INT32));
    curr_topk_weights =
        GetDetailTensor(assign_ad_func(topk_weights->raw_tensor()));
    curr_num_experts = static_cast<int>(num_tokens_per_expert->size(0));
  } else {  // cache mode
    EP_HOST_ASSERT(cached_recv_gbl_rank_prefix_sum.has_value() &&
                   cached_recv_rdma_rank_prefix_sum.has_value());
    curr_topk_idx = cached_recv_gbl_rank_prefix_sum;
    curr_topk_weights = cached_recv_rdma_rank_prefix_sum;
    curr_num_experts = cached_num_recv_tokens;
  }
  EP_HOST_ASSERT(curr_num_experts != 0);

  // Shape and contiguous checks
  EP_HOST_ASSERT(x.dim() == 2 && x.is_contiguous());
  auto num_tokens = static_cast<int>(x.size(0));
  int hidden_size = static_cast<int>(x.size(1));
  int num_topk = static_cast<int>(curr_topk_idx->size(1));
  auto num_local_experts = curr_num_experts / num_ranks;
  int ret = 0;

  // For int8 dispatch, the corresponding combine would be bf16,
  // so we must init buffer with bf16 here to avoid buffer overflow of combine.
  if (!init_normal_buffer) {
    ret = ep_runtime->init_normal_buffer(hidden_size, num_ranks, BKCL_BFLOAT16);
    EP_HOST_ASSERT(ret == 0 && "bkcl_init_normal_buffer failed");
    init_normal_buffer = true;
  }

  int num_scales = 0;
  bool use_int8 = false;
  if (x_scales.has_value()) {
    num_scales = static_cast<int>(x_scales->size(1));
    use_int8 = true;
  }

  // Allocate all tensors on comm stream if set
  // NOTES: do not allocate tensors upfront!
  auto compute_stream = reinterpret_cast<cudaStream_t>(calc_ctx->stream());

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

  auto d_num_recv_tokens_per_expert_list =
      GetDetailTensor(paddle::experimental::empty(
          {num_local_experts}, DataType::INT32, x.place()));
  auto h_num_recv_tokens_per_expert_list =
      std::vector<int>(num_local_experts, 0);

  // unsupported yet
  auto rdma_channel_prefix_matrix = GetDetailTensor(
      paddle::experimental::empty({10}, DataType::INT32, x.place()));
  auto gbl_channel_prefix_matrix = GetDetailTensor(
      paddle::experimental::empty({10}, DataType::INT32, x.place()));
  auto recv_rdma_rank_prefix_sum = curr_topk_weights;
  auto recv_gbl_rank_prefix_sum = curr_topk_idx;

  int num_recv_tokens =
      ep_runtime
          ->notify_dispatch_standard_with_num_recv_tokens_per_expert_list_cpu(
              x.data_ptr(),  // x
              curr_topk_idx->data_ptr<int>(),
              curr_topk_weights->data_ptr<float>(),  // topk_weight
              num_scales,
              hidden_size,
              num_tokens,
              num_topk,
              curr_num_experts,
              d_num_recv_tokens_per_expert_list
                  .data_ptr<int>(),  // should not be nullptr
              h_num_recv_tokens_per_expert_list.data(),
              ToBKCLDataType(x.dtype()),
              use_int8,
              async ? reinterpret_cast<XPUStream>(comm_stream)
                    : reinterpret_cast<XPUStream>(compute_stream));
  // num_tokens maybe 0, and num_recv_tokens also can be 0.
  EP_HOST_ASSERT(num_recv_tokens >= 0 &&
                 "bkcl_notify_dispatch_standard failed");

  std::optional<dTensor> recv_rdma_channel_prefix_matrix =
      GetDetailTensor(paddle::experimental::empty(
          {1, 1}, DataType::INT32, curr_topk_idx->place()));
  std::optional<dTensor> recv_gbl_channel_prefix_matrix =
      GetDetailTensor(paddle::experimental::empty(
          {1, 1}, DataType::INT32, curr_topk_idx->place()));
  std::optional<dTensor> recv_src_meta =
      GetDetailTensor(paddle::experimental::empty(
          {curr_num_experts, 1}, DataType::INT32, curr_topk_idx->place()));
  std::optional<dTensor> send_rdma_head =
      GetDetailTensor(paddle::experimental::empty(
          {1, 1}, DataType::INT32, curr_topk_idx->place()));
  std::optional<dTensor> send_nvl_head =
      GetDetailTensor(paddle::experimental::empty(
          {1, 1}, DataType::INT32, curr_topk_idx->place()));

  auto recv_x = GetDetailTensor(paddle::experimental::empty(
      {num_recv_tokens, hidden_size}, x.dtype(), x.place()));
  std::optional<dTensor> recv_topk_idx =
      GetDetailTensor(paddle::experimental::empty({num_recv_tokens, num_topk},
                                                  curr_topk_idx->dtype(),
                                                  curr_topk_idx->place()));
  std::optional<dTensor> recv_topk_weights =
      GetDetailTensor(paddle::experimental::empty({num_recv_tokens, num_topk},
                                                  curr_topk_weights->dtype(),
                                                  curr_topk_weights->place()));

  auto recv_x_scales = std::optional<dTensor>();
  float* x_scales_ptr = nullptr;
  float* recv_x_scales_ptr = nullptr;

  if (x_scales.has_value()) {
    x_scales_ptr = const_cast<float*>(x_scales->data_ptr<float>());
    recv_x_scales = GetDetailTensor(paddle::experimental::empty(
        {num_recv_tokens, num_scales}, x_scales->dtype(), x_scales->place()));
    recv_x_scales_ptr = recv_x_scales->data_ptr<float>();
  }

  VLOG(3) << "DeepEP internode_dispatch num_local_experts " << num_local_experts
          << " num_scales " << num_scales << " hidden_size " << hidden_size
          << " num_tokens " << num_tokens << " curr_num_experts "
          << curr_num_experts << " num_recv_tokens " << num_recv_tokens;

  ret = ep_runtime->normal_dispatch_standard(
      x.data_ptr(),  // sendbuf
      x_scales_ptr,
      curr_topk_idx->data_ptr<int>(),
      curr_topk_weights->data_ptr<float>(),
      recv_x.data_ptr(),
      recv_x_scales_ptr,
      recv_topk_idx->data_ptr<int>(),
      recv_topk_weights->data_ptr<float>(),
      num_scales,
      -1,  // UNUSED
      hidden_size,
      num_tokens,
      num_topk,
      curr_num_experts,
      ToBKCLDataType(x.dtype()),
      use_int8,
      async ? reinterpret_cast<XPUStream>(comm_stream)
            : reinterpret_cast<XPUStream>(compute_stream));
  EP_HOST_ASSERT(ret == 0 && "bkcl_normal_dispatch_standard failed");

  // Wait streams
  std::optional<EventHandle> event;
  if (async) {
    event = EventHandle(comm_stream);
    for (auto& t : {x, recv_x}) {
      t.record_stream(comm_stream);
      if (allocate_on_comm_stream) t.record_stream(compute_stream);
    }
    for (auto& to : {x_scales,
                     topk_idx,
                     topk_weights,
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
          h_num_recv_tokens_per_expert_list,
          rdma_channel_prefix_matrix,
          gbl_channel_prefix_matrix,
          recv_rdma_channel_prefix_matrix,
          recv_rdma_rank_prefix_sum.value(),  // topk_weights in cache mode
          recv_gbl_channel_prefix_matrix,
          recv_gbl_rank_prefix_sum.value(),  // topk_idx in cache mode
          recv_src_meta,                     // num_experts in cache mode
          send_rdma_head,
          send_nvl_head,
          event};
}

std::tuple<dTensor, std::optional<dTensor>, std::optional<EventHandle>>
Buffer::internode_combine(const dTensor& x,
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
                          bool allocate_on_comm_stream) {
  EP_HOST_ASSERT(x.dim() == 2 && x.is_contiguous());
  auto num_tokens = static_cast<int>(x.size(0)),
       hidden_size = static_cast<int>(x.size(1));
  auto num_combined_tokens =
      static_cast<int>(is_combined_token_in_rank.size(0));

  int ret = BKCL_SUCCESS;
  if (!init_normal_buffer) {
    ret = ep_runtime->init_normal_buffer(hidden_size, num_ranks, BKCL_BFLOAT16);
    EP_HOST_ASSERT(ret == 0 && "bkcl_init_normal_buffer failed");
    init_normal_buffer = true;
  }

  // Allocate all tensors on comm stream if set
  // NOTES: do not allocate tensors upfront!
  auto compute_stream = reinterpret_cast<cudaStream_t>(calc_ctx->stream());

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
  auto combined_topk_weights = std::optional<dTensor>();
  float* topk_weights_ptr = nullptr;
  float* combined_topk_weights_ptr = nullptr;
  if (topk_weights.has_value()) {
    EP_HOST_ASSERT(topk_weights->dim() == 2 && topk_weights->is_contiguous());
    EP_HOST_ASSERT(topk_weights->size(0) == num_tokens);
    EP_HOST_ASSERT(topk_weights->scalar_type() == deep_ep::detail::kFloat32);
    num_topk = static_cast<int>(topk_weights->size(1));
    topk_weights_ptr = topk_weights->data_ptr<float>();
    combined_topk_weights = GetDetailTensor(
        paddle::experimental::empty({num_combined_tokens, num_topk},
                                    topk_weights->dtype(),
                                    topk_weights->place()));
    combined_topk_weights_ptr = combined_topk_weights->data_ptr<float>();
  }

  // Combine data
  auto recv_x = GetDetailTensor(paddle::experimental::empty(
      {num_combined_tokens, hidden_size}, x.dtype(), x.place()));

  VLOG(3) << "DeepEP intranode_combine x.dim " << x.dim() << " num_tokens "
          << num_tokens << " num_combined_tokens " << num_combined_tokens
          << " num_topk " << num_topk << " topk_weights_ptr "
          << topk_weights_ptr << " combined_topk_weights_ptr "
          << combined_topk_weights_ptr;

  ret = ep_runtime->normal_combine_standard(
      x.data_ptr(),
      topk_weights_ptr,
      recv_x.data_ptr(),
      combined_topk_weights_ptr,
      hidden_size,
      num_tokens,
      num_combined_tokens,
      num_topk,
      0 /*num_experts*/,
      ToBKCLDataType(x.scalar_type()),
      async ? reinterpret_cast<XPUStream>(comm_stream)
            : reinterpret_cast<XPUStream>(compute_stream));

  // Wait streams
  std::optional<EventHandle> event;
  if (async) {
    event = EventHandle(comm_stream);
    for (auto& to : {x, recv_x}) {
      to.record_stream(comm_stream);
      if (allocate_on_comm_stream) to.record_stream(compute_stream);
    }

    for (auto& to : {
             topk_weights,
             combined_topk_weights,
         }) {
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

  return {recv_x, combined_topk_weights, event};
}

void Buffer::clean_low_latency_buffer(int num_max_dispatch_tokens_per_rank,
                                      int hidden,
                                      int num_experts) {
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
}

void Buffer::clean_low_latency_two_stage_buffer(
    int num_max_dispatch_tokens_per_rank,
    int hidden,
    int num_experts,
    int num_topk,
    int num_ranks,
    bool use_fp8) {
  return;
}

void Buffer::barrier_all() {}

std::tuple<dTensor,
           std::optional<dTensor>,
           dTensor,
           dTensor,
           dTensor,
           std::optional<EventHandle>,
           std::optional<std::function<void()>>>
Buffer::low_latency_dispatch(const dTensor& x,
                             const dTensor& topk_idx,
                             const std::optional<dTensor>& expertwise_scale,
                             int num_max_dispatch_tokens_per_rank,
                             int num_experts,
                             bool use_fp8,
                             bool async,
                             bool return_recv_hook,
                             int num_per_channel) {
  EP_HOST_ASSERT(low_latency_mode);
  auto num_tokens = static_cast<int>(x.size(0)),
       hidden_size = static_cast<int>(x.size(1));
  auto num_scales = num_per_channel == -1 ? 1 : hidden_size / 128,
       num_topk = static_cast<int>(topk_idx.size(1));
  int num_local_experts = num_experts / num_ranks;

  if (!init_low_latency_buffer) {
    int ret = ep_runtime->init_low_latency_buffer(
        num_max_dispatch_tokens_per_rank, hidden_size, num_ranks, num_experts);
    EP_HOST_ASSERT(ret == 0 && "bkcl_init_low_latency_buffer failed");
    init_low_latency_buffer = true;
  }

  // Allocate packed tensors
  auto packed_recv_x = GetDetailTensor(paddle::experimental::empty(
      {num_local_experts,
       num_ranks * num_max_dispatch_tokens_per_rank,
       hidden_size},
      use_fp8 ? paddle::DataType::INT8 : paddle::DataType::BFLOAT16,
      x.place()));
  auto packed_recv_src_info = GetDetailTensor(paddle::experimental::empty(
      {num_local_experts, num_ranks * num_max_dispatch_tokens_per_rank},
      DataType::INT32,
      x.place()));
  auto packed_recv_layout_range = GetDetailTensor(paddle::experimental::empty(
      {2, num_local_experts, num_ranks}, DataType::INT32, x.place()));

  // Allocate column-majored scales
  auto packed_recv_x_scales = std::optional<dTensor>();
  float* packed_recv_x_scales_ptr = nullptr;

  if (use_fp8 && !expertwise_scale.has_value()) {
    EP_HOST_ASSERT((num_ranks * num_max_dispatch_tokens_per_rank) % 4 == 0 &&
                   "TMA requires the number of tokens to be multiple of 4");
    packed_recv_x_scales = GetDetailTensor(paddle::experimental::empty(
        {num_local_experts, num_ranks * num_max_dispatch_tokens_per_rank, 1},
        paddle::DataType::FLOAT32,
        x.place()));
    packed_recv_x_scales_ptr = packed_recv_x_scales.value().data_ptr<float>();
  }

  float* expertwise_scale_ptr = nullptr;
  if (expertwise_scale.has_value()) {
    expertwise_scale_ptr = expertwise_scale.value().data_ptr<float>();
  }

  // Wait previous tasks to be finished
  // NOTES: the hook mode will always use the default stream
  auto compute_stream = reinterpret_cast<cudaStream_t>(calc_ctx->stream());

  auto launch_stream = return_recv_hook ? compute_stream : comm_stream;
  EP_HOST_ASSERT(!(async && return_recv_hook));
  if (!return_recv_hook) stream_wait(launch_stream, compute_stream);

  const int* h_recv_count_ptr = nullptr;
  void* recv_count_ptr = nullptr;
  std::function<void()> recv_hook = [=]() {};

  std::tie(recv_count_ptr, recv_hook) = ep_runtime->low_latency_dispatch(
      const_cast<void*>(x.data_ptr()),
      num_tokens,
      const_cast<int*>(topk_idx.data_ptr<int>()),
      num_max_dispatch_tokens_per_rank,
      hidden_size,
      num_experts,
      num_topk,
      packed_recv_x.data_ptr(),
      packed_recv_x_scales_ptr,
      reinterpret_cast<int*>(
          const_cast<int*>(packed_recv_src_info.data_ptr<int>())),
      reinterpret_cast<int*>(
          const_cast<int*>(packed_recv_layout_range.data_ptr<int>())),
      use_fp8,
      return_recv_hook,
      expertwise_scale_ptr,
      reinterpret_cast<XPUStream>(launch_stream),
      nullptr);

  auto packed_recv_count =
      GetDetailTensor(paddle::from_blob(recv_count_ptr,
                                        {num_local_experts},
                                        paddle::DataType::INT32,
                                        phi::DataLayout::NCHW,
                                        x.place()));

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

  std::optional<std::function<void()>> opt_recv_hook =
      std::make_optional(recv_hook);

  return {packed_recv_x,
          packed_recv_x_scales,
          packed_recv_count,
          packed_recv_src_info,
          packed_recv_layout_range,
          event,
          opt_recv_hook};
}

std::tuple<dTensor,
           std::optional<EventHandle>,
           std::optional<std::function<void()>>>
Buffer::low_latency_combine(const dTensor& x,
                            const dTensor& topk_idx,
                            const dTensor& topk_weights,
                            const dTensor& src_info,
                            const dTensor& layout_range,
                            int num_max_dispatch_tokens_per_rank,
                            int num_experts,
                            bool zero_copy,
                            bool async,
                            bool return_recv_hook,
                            const std::optional<dTensor>& out) {
  auto hidden_size = static_cast<int>(x.size(2));
  auto num_local_experts = num_experts / num_ranks,
       num_topk = static_cast<int>(topk_weights.size(1));
  auto num_combined_tokens = static_cast<int>(topk_weights.size(0));

  if (!init_low_latency_buffer) {
    int ret = ep_runtime->init_low_latency_buffer(
        num_max_dispatch_tokens_per_rank, hidden_size, num_ranks, num_experts);
    EP_HOST_ASSERT(ret == 0 && "bkcl_init_low_latency_buffer failed");
    init_low_latency_buffer = true;
  }

  // Wait previous tasks to be finished
  // NOTES: the hook mode will always use the default stream
  auto compute_stream = reinterpret_cast<cudaStream_t>(calc_ctx->stream());

  auto launch_stream = return_recv_hook ? compute_stream : comm_stream;
  EP_HOST_ASSERT(!(async && return_recv_hook));
  if (!return_recv_hook) stream_wait(launch_stream, compute_stream);

  // Allocate output tensor
  dTensor combined_x;
  if (out.has_value()) {
    EP_HOST_ASSERT(out->dim() == 2 && out->is_contiguous());
    EP_HOST_ASSERT(out->size(0) == num_combined_tokens &&
                   out->size(1) == hidden_size);
    EP_HOST_ASSERT(out->scalar_type() == x.scalar_type());
    combined_x = out.value();
  } else {
    combined_x = GetDetailTensor(paddle::experimental::empty(
        {num_combined_tokens, hidden_size}, x.dtype(), x.place()));
  }

  std::function<void()> recv_hook = [=]() {};
  recv_hook = ep_runtime->low_latency_combine(
      const_cast<void*>(x.data_ptr()),
      const_cast<int*>(topk_idx.data_ptr<int>()),
      const_cast<float*>(topk_weights.data_ptr<float>()),
      num_combined_tokens,
      const_cast<int*>(src_info.data_ptr<int>()),
      const_cast<int*>(layout_range.data_ptr<int>()),
      num_max_dispatch_tokens_per_rank,
      hidden_size,
      num_experts,
      num_topk,
      combined_x.data_ptr(),
      return_recv_hook,
      zero_copy,
      reinterpret_cast<XPUStream>(launch_stream));

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

  // Return values
  return std::tuple<dTensor,
                    std::optional<EventHandle>,
                    std::optional<std::function<void()>>>{
      dTensor{combined_x}, event, recv_hook};
}

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
Buffer::low_latency_dispatch_two_stage(const dTensor& x,
                                       const dTensor& topk_idx,
                                       const dTensor& topk_weights,
                                       int num_max_dispatch_tokens_per_rank,
                                       int num_experts,
                                       bool use_fp8,
                                       bool async,
                                       bool return_recv_hook) {
  return {dTensor{},
          std::nullopt,
          dTensor{},
          dTensor{},
          dTensor{},
          dTensor{},
          dTensor{},
          dTensor{},
          std::nullopt,
          std::nullopt};
}

std::tuple<dTensor,
           std::optional<EventHandle>,
           std::optional<std::function<void()>>>
Buffer::low_latency_combine_two_stage(const dTensor& x,
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
                                      const std::optional<dTensor>& out) {
  return {dTensor{}, std::nullopt, std::nullopt};
}

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
Buffer::m2n_low_latency_dispatch_two_stage(const dTensor& x,
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
                                           bool return_recv_hook) {
  return {
      dTensor{},
      std::nullopt,
      dTensor{},
      dTensor{},
      dTensor{},
      dTensor{},
      dTensor{},
      dTensor{},
      std::nullopt,
      std::nullopt,
  };
}

std::tuple<dTensor,
           std::optional<EventHandle>,
           std::optional<std::function<EventHandle()>>>
Buffer::m2n_low_latency_combine_two_stage(
    const dTensor& x,
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
    const std::optional<dTensor>& out) {
  return {
      dTensor{},
      std::nullopt,
      std::nullopt,
  };
}

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
Buffer::internode_dispatch_api(
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
    bool allocate_on_comm_stream) {
  const auto& x_ = GetDetailTensor(x);
  std::optional<dTensor> x_scales_ = GetDetailTensor(x_scales);

  std::optional<dTensor> topk_idx_ = GetDetailTensor(topk_idx);
  std::optional<dTensor> topk_weights_ = GetDetailTensor(topk_weights);
  std::optional<dTensor> num_tokens_per_rank_ =
      GetDetailTensor(num_tokens_per_rank);
  std::optional<dTensor> num_tokens_per_rdma_rank_ =
      GetDetailTensor(num_tokens_per_rdma_rank);

  const auto& is_token_in_rank_ = GetDetailTensor(is_token_in_rank);
  std::optional<dTensor> num_tokens_per_expert_ =
      GetDetailTensor(num_tokens_per_expert);

  std::optional<dTensor> cached_rdma_channel_prefix_matrix_ =
      GetDetailTensor(cached_rdma_channel_prefix_matrix);
  std::optional<dTensor> cached_recv_rdma_rank_prefix_sum_ =
      GetDetailTensor(cached_recv_rdma_rank_prefix_sum);
  std::optional<dTensor> cached_gbl_channel_prefix_matrix_ =
      GetDetailTensor(cached_gbl_channel_prefix_matrix);
  std::optional<dTensor> cached_recv_gbl_rank_prefix_sum_ =
      GetDetailTensor(cached_recv_gbl_rank_prefix_sum);

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

  auto recv_x_ = GetPaddleTensor(std::get<0>(res));
  std::optional<Tensor> recv_x_scales_ = GetPaddleTensor(std::get<1>(res));

  std::optional<Tensor> recv_topk_idx_ = GetPaddleTensor(std::get<2>(res));
  std::optional<Tensor> recv_topk_weights_ = GetPaddleTensor(std::get<3>(res));

  const auto& num_recv_tokens_per_expert_list = std::get<4>(res);

  auto rdma_channel_prefix_matrix_ = GetPaddleTensor(std::get<5>(res));

  auto gbl_channel_prefix_matrix_ = GetPaddleTensor(std::get<6>(res));

  std::optional<Tensor> recv_rdma_channel_prefix_matrix_ =
      GetPaddleTensor(std::get<7>(res));
  auto recv_rdma_rank_prefix_sum_ = GetPaddleTensor(std::get<8>(res));

  std::optional<Tensor> recv_gbl_channel_prefix_matrix_ =
      GetPaddleTensor(std::get<9>(res));
  auto recv_gbl_rank_prefix_sum_ = GetPaddleTensor(std::get<10>(res));

  std::optional<Tensor> recv_src_meta_ = GetPaddleTensor(std::get<11>(res));

  std::optional<Tensor> send_rdma_head_ = GetPaddleTensor(std::get<12>(res));
  std::optional<Tensor> send_nvl_head_ = GetPaddleTensor(std::get<13>(res));

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
}

std::tuple<Tensor, std::optional<Tensor>, std::optional<EventHandle>>
Buffer::internode_combine_api(
    const Tensor& x,
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
    bool allocate_on_comm_stream) {
  const auto& x_ = GetDetailTensor(x);

  std::optional<dTensor> topk_weights_ = GetDetailTensor(topk_weights);

  const auto& src_meta_ = GetDetailTensor(src_meta);
  const auto& is_combined_token_in_rank_ =
      GetDetailTensor(is_combined_token_in_rank);

  const auto& rdma_channel_prefix_matrix_ =
      GetDetailTensor(rdma_channel_prefix_matrix);
  const auto& rdma_rank_prefix_sum_ = GetDetailTensor(rdma_rank_prefix_sum);
  const auto& gbl_channel_prefix_matrix_ =
      GetDetailTensor(gbl_channel_prefix_matrix);

  const auto& combined_rdma_head_ = GetDetailTensor(combined_rdma_head);
  const auto& combined_nvl_head_ = GetDetailTensor(combined_nvl_head);

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

  auto combined_x_ = GetPaddleTensor(std::get<0>(res));
  std::optional<Tensor> combined_topk_weights_ =
      GetPaddleTensor(std::get<1>(res));

  const auto& event = std::get<2>(res);

  return {combined_x_, combined_topk_weights_, event};
}

std::tuple<Tensor,
           std::optional<Tensor>,
           Tensor,
           Tensor,
           Tensor,
           std::optional<EventHandle>,
           std::optional<std::function<void()>>>
Buffer::low_latency_dispatch_api(const Tensor& x,
                                 const Tensor& topk_idx,
                                 const std::optional<Tensor>& expertwise_scale,
                                 int num_max_dispatch_tokens_per_rank,
                                 int num_experts,
                                 bool use_fp8,
                                 bool async,
                                 bool return_recv_hook,
                                 int num_per_channel) {
  const auto& x_ = GetDetailTensor(x);
  const auto& topk_idx_ =
      GetDetailTensor(cast_ad_func(topk_idx, DataType::INT32));

  std::optional<dTensor> expertwise_scale_;
  if (expertwise_scale.has_value()) {
    expertwise_scale_ = GetDetailTensor(expertwise_scale.value());
  }

  auto res = low_latency_dispatch(x_,
                                  topk_idx_,
                                  expertwise_scale_,
                                  num_max_dispatch_tokens_per_rank,
                                  num_experts,
                                  use_fp8,
                                  async,
                                  return_recv_hook,
                                  num_per_channel);

  auto packed_recv_x_ = GetPaddleTensor(std::get<0>(res));

  std::optional<Tensor> packed_recv_x_scales_;
  if (std::get<1>(res).has_value()) {
    packed_recv_x_scales_ = GetPaddleTensor(std::get<1>(res).value());
  }

  auto packed_recv_count_ = GetPaddleTensor(std::get<2>(res));
  auto packed_recv_src_info_ = GetPaddleTensor(std::get<3>(res));
  auto packed_recv_layout_range_ = GetPaddleTensor(std::get<4>(res));

  const auto& event = std::get<5>(res);
  auto recv_hook = std::get<6>(res);

  return {packed_recv_x_,
          packed_recv_x_scales_,
          packed_recv_count_,
          packed_recv_src_info_,
          packed_recv_layout_range_,
          event,
          recv_hook};
}

std::tuple<Tensor,
           std::optional<EventHandle>,
           std::optional<std::function<void()>>>
Buffer::low_latency_combine_api(const Tensor& x,
                                const Tensor& topk_idx,
                                const Tensor& topk_weights,
                                const Tensor& src_info,
                                const Tensor& layout_range,
                                int num_max_dispatch_tokens_per_rank,
                                int num_experts,
                                bool zero_copy,
                                bool async,
                                bool return_recv_hook,
                                const std::optional<Tensor>& out) {
  const auto& x_ = GetDetailTensor(x);
  const auto& topk_idx_ =
      GetDetailTensor(cast_ad_func(topk_idx, DataType::INT32));
  const auto& topk_weights_ = GetDetailTensor(topk_weights);
  const auto& src_info_ = GetDetailTensor(src_info);
  const auto& layout_range_ = GetDetailTensor(layout_range);
  std::optional<dTensor> out_ = std::nullopt;
  if (out.has_value()) {
    out_ = GetDetailTensor(out.value());
  }

  auto res = low_latency_combine(x_,
                                 topk_idx_,
                                 topk_weights_,
                                 src_info_,
                                 layout_range_,
                                 num_max_dispatch_tokens_per_rank,
                                 num_experts,
                                 zero_copy,
                                 async,
                                 return_recv_hook,
                                 out_);

  auto combined_x_ = GetPaddleTensor(std::get<0>(res));
  const auto& event = std::get<1>(res);
  auto recv_hook = std::get<2>(res);

  return {combined_x_, event, recv_hook};
}

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
Buffer::low_latency_dispatch_two_stage_api(const Tensor& x,
                                           const Tensor& topk_idx,
                                           const Tensor& topk_weights,
                                           int num_max_dispatch_tokens_per_rank,
                                           int num_experts,
                                           bool use_fp8,
                                           bool async,
                                           bool return_recv_hook) {
  const auto& x_ = GetDetailTensor(x);
  const auto& topk_idx_ = GetDetailTensor(topk_idx);
  const auto& topk_weights_ = GetDetailTensor(topk_weights);

  auto res = low_latency_dispatch_two_stage(x_,
                                            topk_idx_,
                                            topk_weights_,
                                            num_max_dispatch_tokens_per_rank,
                                            num_experts,
                                            use_fp8,
                                            async,
                                            return_recv_hook);

  auto packed_recv_x_ = GetPaddleTensor(std::get<0>(res));

  std::optional<Tensor> packed_recv_x_scales_;
  if (std::get<1>(res).has_value()) {
    packed_recv_x_scales_ = GetPaddleTensor(std::get<1>(res).value());
  }
  auto packed_recv_rdma_x_ = GetPaddleTensor(std::get<2>(res));

  auto packed_recv_count_ = GetPaddleTensor(std::get<3>(res));
  auto packed_rdma_recv_count_ = GetPaddleTensor(std::get<4>(res));
  auto packed_recv_src_info_ = GetPaddleTensor(std::get<5>(res));
  auto packed_recv_layout_range_ = GetPaddleTensor(std::get<6>(res));
  auto rdma_send_flags_ = GetPaddleTensor(std::get<7>(res));

  const auto& event = std::get<8>(res);
  auto recv_hook = std::get<9>(res);

  return {packed_recv_x_,
          packed_recv_x_scales_,
          packed_recv_rdma_x_,
          packed_recv_count_,
          packed_rdma_recv_count_,
          packed_recv_src_info_,
          packed_recv_layout_range_,
          rdma_send_flags_,
          event,
          recv_hook};
}

std::tuple<Tensor,
           std::optional<EventHandle>,
           std::optional<std::function<void()>>>
Buffer::low_latency_combine_two_stage_api(
    const Tensor& x,
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
    const std::optional<Tensor>& out) {
  const auto& x_ = GetDetailTensor(x);
  const auto& rdma_recv_x_ = GetDetailTensor(rdma_recv_x);
  const auto& topk_idx_ = GetDetailTensor(topk_idx);
  const auto& topk_weights_ = GetDetailTensor(topk_weights);
  const auto& src_info_ = GetDetailTensor(src_info);
  const auto& layout_range_ = GetDetailTensor(layout_range);
  const auto& rdma_send_flags_ = GetDetailTensor(rdma_send_flags);
  const auto& dispatch_rdma_recv_count_ =
      GetDetailTensor(dispatch_rdma_recv_count);

  std::optional<dTensor> out_ = std::nullopt;
  if (out.has_value()) {
    out_ = GetDetailTensor(out.value());
  }

  auto res = low_latency_combine_two_stage(x_,
                                           rdma_recv_x_,
                                           topk_idx_,
                                           topk_weights_,
                                           src_info_,
                                           layout_range_,
                                           rdma_send_flags_,
                                           dispatch_rdma_recv_count_,
                                           num_max_dispatch_tokens_per_rank,
                                           num_experts,
                                           dispatch_use_fp8,
                                           async,
                                           return_recv_hook,
                                           out_);

  auto combined_x_ = GetPaddleTensor(std::get<0>(res));
  const auto& event = std::get<1>(res);
  auto recv_hook = std::get<2>(res);

  return {combined_x_, event, recv_hook};
}

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
Buffer::m2n_low_latency_dispatch_two_stage_api(
    const Tensor& x,
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
    bool return_recv_hook) {
  const auto& x_ = GetDetailTensor(x);
  const auto& topk_idx_ = GetDetailTensor(topk_idx);
  const auto& topk_weights_ = GetDetailTensor(topk_weights);

  auto res =
      m2n_low_latency_dispatch_two_stage(x_,
                                         topk_idx_,
                                         topk_weights_,
                                         num_max_dispatch_tokens_per_rank,
                                         num_experts,
                                         a_start_rank,
                                         a_num_ranks,
                                         e_start_rank,
                                         e_num_ranks,
                                         use_fp8,
                                         async,
                                         return_recv_hook);

  auto packed_recv_x_ = GetPaddleTensor(std::get<0>(res));

  std::optional<Tensor> packed_recv_x_scales_;
  if (std::get<1>(res).has_value()) {
    packed_recv_x_scales_ = GetPaddleTensor(std::get<1>(res).value());
  }
  auto packed_recv_rdma_x_ = GetPaddleTensor(std::get<2>(res));
  auto packed_recv_count_ = GetPaddleTensor(std::get<3>(res));
  auto packed_rdma_recv_count_ = GetPaddleTensor(std::get<4>(res));
  auto packed_recv_src_info_ = GetPaddleTensor(std::get<5>(res));
  auto packed_recv_layout_range_ = GetPaddleTensor(std::get<6>(res));
  auto rdma_send_flags_ = GetPaddleTensor(std::get<7>(res));

  const auto& event = std::get<8>(res);
  auto recv_hook = std::get<9>(res);

  return {packed_recv_x_,
          packed_recv_x_scales_,
          packed_recv_rdma_x_,
          packed_recv_count_,
          packed_rdma_recv_count_,
          packed_recv_src_info_,
          packed_recv_layout_range_,
          rdma_send_flags_,
          event,
          recv_hook};
}

std::tuple<Tensor,
           std::optional<EventHandle>,
           std::optional<std::function<EventHandle()>>>
Buffer::m2n_low_latency_combine_two_stage_api(
    const Tensor& x,
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
    const std::optional<Tensor>& out) {
  const auto& x_ = GetDetailTensor(x);
  const auto& rdma_recv_x_ = GetDetailTensor(rdma_recv_x);
  const auto& topk_idx_ = GetDetailTensor(topk_idx);
  const auto& topk_weights_ = GetDetailTensor(topk_weights);
  const auto& src_info_ = GetDetailTensor(src_info);
  const auto& layout_range_ = GetDetailTensor(layout_range);
  const auto& rdma_send_flags_ = GetDetailTensor(rdma_send_flags);
  const auto& dispatch_rdma_recv_count_ =
      GetDetailTensor(dispatch_rdma_recv_count);

  std::optional<dTensor> out_ = std::nullopt;
  if (out.has_value()) {
    out_ = GetDetailTensor(out.value());
  }

  auto res = m2n_low_latency_combine_two_stage(x_,
                                               rdma_recv_x_,
                                               topk_idx_,
                                               topk_weights_,
                                               src_info_,
                                               layout_range_,
                                               rdma_send_flags_,
                                               dispatch_rdma_recv_count_,
                                               num_max_dispatch_tokens_per_rank,
                                               num_experts,
                                               a_start_rank,
                                               a_num_ranks,
                                               e_start_rank,
                                               e_num_ranks,
                                               dispatch_use_fp8,
                                               async,
                                               return_recv_hook,
                                               out_);

  auto combined_x_ = GetPaddleTensor(std::get<0>(res));
  const auto& event = std::get<1>(res);
  auto recv_hook = std::get<2>(res);

  return {combined_x_, event, recv_hook};
}

std::tuple<Tensor,
           std::optional<Tensor>,
           Tensor,
           Tensor,
           std::optional<EventHandle>>
Buffer::get_dispatch_layout_api(const Tensor& topk_idx,
                                int num_experts,
                                std::optional<EventHandle>& previous_event,
                                bool async,
                                bool allocate_on_comm_stream) {
  const auto& topk_idx_ = GetDetailTensor(topk_idx);
  auto res = get_dispatch_layout(
      topk_idx_, num_experts, previous_event, async, allocate_on_comm_stream);
  const auto& num_tokens_per_rank = std::get<0>(res);
  const auto& num_tokens_per_rdma_rank = std::get<1>(res);
  const auto& num_tokens_per_expert = std::get<2>(res);
  const auto& is_token_in_rank = std::get<3>(res);
  const auto& event = std::get<4>(res);
  auto num_tokens_per_rank_ = GetPaddleTensor(num_tokens_per_rank);
  std::optional<Tensor> num_tokens_per_rdma_rank_ = std::nullopt;
  if (num_tokens_per_rdma_rank.has_value()) {
    num_tokens_per_rdma_rank_ =
        GetPaddleTensor(num_tokens_per_rdma_rank.value());
  }
  auto num_tokens_per_expert_ = GetPaddleTensor(num_tokens_per_expert);
  auto is_token_in_rank_ = GetPaddleTensor(is_token_in_rank);
  return {num_tokens_per_rank_,
          num_tokens_per_rdma_rank_,
          num_tokens_per_expert_,
          is_token_in_rank_,
          event};
}

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
Buffer::intranode_dispatch_api(
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
    bool allocate_on_comm_stream) {
  const auto& x_ = GetDetailTensor(x);
  std::optional<dTensor> x_scales_;
  if (x_scales.has_value()) {
    x_scales_ = GetDetailTensor(x_scales.value());
  }
  std::optional<dTensor> topk_idx_;
  if (topk_idx.has_value()) {
    topk_idx_ = GetDetailTensor(topk_idx.value());
  }
  std::optional<dTensor> topk_weights_;
  if (topk_weights.has_value()) {
    topk_weights_ = GetDetailTensor(topk_weights.value());
  }
  std::optional<dTensor> num_tokens_per_rank_;
  if (num_tokens_per_rank.has_value()) {
    num_tokens_per_rank_ = GetDetailTensor(num_tokens_per_rank.value());
  }
  const auto& is_token_in_rank_ = GetDetailTensor(is_token_in_rank);
  std::optional<dTensor> num_tokens_per_expert_;
  if (num_tokens_per_expert.has_value()) {
    num_tokens_per_expert_ = GetDetailTensor(num_tokens_per_expert.value());
  }
  std::optional<dTensor> cached_rank_prefix_matrix_;
  if (cached_rank_prefix_matrix.has_value()) {
    cached_rank_prefix_matrix_ =
        GetDetailTensor(cached_rank_prefix_matrix.value());
  }
  std::optional<dTensor> cached_channel_prefix_matrix_;
  if (cached_channel_prefix_matrix.has_value()) {
    cached_channel_prefix_matrix_ =
        GetDetailTensor(cached_channel_prefix_matrix.value());
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

  auto recv_x_ = GetPaddleTensor(recv_x);
  std::optional<Tensor> recv_x_scales_;
  if (recv_x_scales.has_value()) {
    recv_x_scales_ = GetPaddleTensor(recv_x_scales.value());
  }
  std::optional<Tensor> recv_topk_idx_;
  if (recv_topk_idx.has_value()) {
    recv_topk_idx_ = GetPaddleTensor(recv_topk_idx.value());
  }
  std::optional<Tensor> recv_topk_weights_;
  if (recv_topk_weights.has_value()) {
    recv_topk_weights_ = GetPaddleTensor(recv_topk_weights.value());
  }
  auto rank_prefix_matrix_ = GetPaddleTensor(rank_prefix_matrix);
  auto channel_prefix_matrix_ = GetPaddleTensor(channel_prefix_matrix);
  auto recv_channel_prefix_matrix_ =
      GetPaddleTensor(recv_channel_prefix_matrix);
  auto recv_src_idx_ = GetPaddleTensor(recv_src_idx);
  auto send_head_ = GetPaddleTensor(send_head);

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

std::tuple<Tensor, std::optional<Tensor>, std::optional<EventHandle>>
Buffer::intranode_combine_api(const Tensor& x,
                              const std::optional<Tensor>& topk_weights,
                              const Tensor& src_idx,
                              const Tensor& rank_prefix_matrix,
                              const Tensor& channel_prefix_matrix,
                              const Tensor& send_head,
                              const Config& config,
                              std::optional<EventHandle>& previous_event,
                              bool async,
                              bool allocate_on_comm_stream) {
  const auto& x_ = GetDetailTensor(x);
  std::optional<dTensor> topk_weights_;
  if (topk_weights.has_value()) {
    topk_weights_ = GetDetailTensor(topk_weights.value());
  }
  const auto& src_idx_ = GetDetailTensor(src_idx);
  const auto& rank_prefix_matrix_ = GetDetailTensor(rank_prefix_matrix);
  const auto& channel_prefix_matrix_ = GetDetailTensor(channel_prefix_matrix);
  const auto& send_head_ = GetDetailTensor(send_head);

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

  auto recv_x_ = GetPaddleTensor(recv_x);
  std::optional<Tensor> recv_topk_weights_;
  if (recv_topk_weights.has_value()) {
    recv_topk_weights_ = GetPaddleTensor(recv_topk_weights.value());
  }
  auto event_ = event;
  return {recv_x_, recv_topk_weights_, event_};
}

dTensor GetDetailTensor(const Tensor& tensor) {
  dTensor res(tensor);
  return res;
}

Tensor GetPaddleTensor(const dTensor& tensor) { return tensor.raw_tensor(); }

std::optional<dTensor> GetDetailTensor(const std::optional<Tensor>& tensor) {
  std::optional<dTensor> res;
  if (tensor.has_value()) {
    res = GetDetailTensor(tensor.value());
  }
  return res;
}

std::optional<Tensor> GetPaddleTensor(const std::optional<dTensor>& tensor) {
  std::optional<Tensor> res;
  if (tensor.has_value()) {
    res = GetPaddleTensor(tensor.value());
  }
  return res;
}

}  // namespace paddle::deep_ep
