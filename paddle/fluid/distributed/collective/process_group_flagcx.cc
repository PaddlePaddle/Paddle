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

#include "paddle/fluid/distributed/collective/process_group_flagcx.h"
#include "paddle/common/flags.h"
#include "paddle/fluid/distributed/collective/common.h"
#include "paddle/phi/api/lib/utils/allocator.h"
#include "paddle/phi/backends/gpu/gpu_info.h"
#include "paddle/phi/common/memory_utils.h"
#include "paddle/phi/core/distributed/check/static_check.h"
#include "paddle/phi/core/distributed/comm_context_manager.h"
#include "paddle/phi/core/distributed/comm_task_manager.h"
#include "paddle/phi/core/distributed/flagcx_tools.h"
#include "paddle/phi/core/distributed/utils.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/platform/cuda_device_guard.h"
#include "paddle/phi/core/platform/device/gpu/nccl_helper.h"
#include "paddle/phi/core/utils/data_type.h"

COMMON_DECLARE_bool(flagcx_blocking_wait);
COMMON_DECLARE_bool(enable_async_trace);
COMMON_DECLARE_bool(eager_communication_connection);

// set this flag to `true` and recompile to enable dynamic checks
// constexpr bool FLAGS_enable_nccl_dynamic_check = false;
constexpr int64_t kWaitBlockTImeout = 10;

namespace paddle::distributed {

using phi::distributed::CheckSizeOnEachRank;
using phi::distributed::FlagcxDTypeToString;
using phi::distributed::FlagcxRedTypeToString;
using phi::distributed::IsP2POP;
using phi::distributed::SerializeFlagcxUniqueId;
using phi::distributed::ToFlagcxRedType;

uint64_t ProcessGroupFlagcx::s_group_call_counter = 0;

ProcessGroupFlagcx::FlagcxTask::FlagcxTask(const Place& place,
                                           int rank,
                                           CommType comm_type,
                                           bool sync_op,
                                           bool use_calc_stream,
                                           int gid)
    : TaskStream(rank, comm_type, sync_op, use_calc_stream),
      task_place_(place),
      gid_(gid) {
  if (!use_calc_stream) {
    comm_event_ = std::make_shared<platform::DeviceEvent>(
        place, platform::GenerateDeviceEventFlag());
  }
}

ProcessGroupFlagcx::FlagcxTask::~FlagcxTask() = default;

bool ProcessGroupFlagcx::FlagcxTask::IsCompleted() {
  if (comm_event_) {
    return comm_event_->Query();
  } else {
    return true;
  }
}

void ProcessGroupFlagcx::FlagcxTask::UpdateWaitChain(
    const phi::DeviceContext& ctx) {
  if (comm_event_) {
    comm_event_->Record(&ctx);
  }
}

void ProcessGroupFlagcx::FlagcxTask::RemoveHolderStreamInGroup() {
  auto map = distributed::ProcessGroupMapFromGid::getInstance();
  distributed::ProcessGroup* pg = map->get(gid_);
  if (!pg) return;
  auto* pg_flagcx = dynamic_cast<ProcessGroupFlagcx*>(pg);
  if (!pg_flagcx) return;
  pg_flagcx->EraseTensorHolders();
}

// TODO(sheniang03): Add timeout for wait, now timeout unused
bool ProcessGroupFlagcx::FlagcxTask::Wait(std::chrono::milliseconds timeout) {
  // Warning here when use calc stream but also invoke waiting explicitly.
  if (UseCalcStream()) {
    VLOG(5) << "Warning: The communication is on calc stream, wait here is "
               "useless.";
    return true;
  }

  const auto* calc_ctx =
      platform::DeviceContextPool::Instance().Get(task_place_);
  if (comm_event_) {
    comm_event_->Wait(platform::Place2DeviceType(task_place_), calc_ctx);
  }

  if (FLAGS_flagcx_blocking_wait) {
    // NOTE(shenliang03): It will block host for sync
    while (!IsCompleted()) {
      std::this_thread::sleep_for(std::chrono::milliseconds(kWaitBlockTImeout));
    }
  }
  RemoveHolderStreamInGroup();
  return true;
}

// Same as Wait
void ProcessGroupFlagcx::FlagcxTask::Synchronize() { Wait(kWaitTimeout); }

ProcessGroupFlagcx::ProcessGroupFlagcx(
    const std::shared_ptr<phi::distributed::Store>& store,
    int rank,
    int size,
    int gid,
    int64_t timeout,
    int flagcx_comm_init_option)
    : ProcessGroupWithStream(rank, size, gid),
      store_(store),
      place_to_calc_event_(),
      place_to_calc_ctx_(),
      place_to_comm_ctx_(),
      p2p_comm_seq_(),
      place_to_group_key_(),
      pg_timeout_(timeout),
      flagcx_comm_init_option_(flagcx_comm_init_option),
      allocation_stream_pairs_() {
  LOG(INFO) << "ProcessGroupFlagcx pg_timeout_ " << pg_timeout_;
  LOG(INFO) << "ProcessGroupFlagcx flagcx_comm_init_option_ "
            << flagcx_comm_init_option_;
  if (FLAGS_eager_communication_connection) {
    EagerConnect();
  }
}
ProcessGroupFlagcx::~ProcessGroupFlagcx() {
  LOG(INFO) << "ProcessGroupFlagcx destruct ";
}

void ProcessGroupFlagcx::GroupStart() {
  if (flagcx_comm_ != nullptr) {
    FLAGCX_CHECK(phi::dynload::flagcxGroupStart(flagcx_comm_));
    ++s_group_call_counter;
  }
}

void ProcessGroupFlagcx::GroupEnd() {
  if (flagcx_comm_ != nullptr) {
    FLAGCX_CHECK(phi::dynload::flagcxGroupEnd(flagcx_comm_));
    --s_group_call_counter;
  }
}

phi::DeviceContext* ProcessGroupFlagcx::GetDeviceContext(
    const Place& place) const {
  return GetDeviceContext(place, /*use_calc_stream*/ false);
}

// NOTE(shenliang03): GetDeviceContext is only used for collective, it can't
// be used for p2p op.
phi::DeviceContext* ProcessGroupFlagcx::GetDeviceContext(
    const Place& place, bool use_calc_stream) const {
  const std::string& key = GetKeyFromPlace(place);
  if (use_calc_stream) {
    const auto& iter = place_to_calc_ctx_.find(key);
    return iter->second;
  } else {
    const auto& iter = place_to_comm_ctx_.find(key);
    PADDLE_ENFORCE_NE(
        iter,
        place_to_comm_ctx_.end(),
        common::errors::NotFound(
            "Cannot find the device context in this process group."));
    return iter->second.get();
  }
}

void ProcessGroupFlagcx::EagerConnect() {
  const auto deviceId = phi::backends::gpu::GetCurrentDeviceId();
  const auto& place = phi::GPUPlace(deviceId);
  const auto key = GetKeyFromPlace(place);

  platform::CUDADeviceGuard cuda_guard(place);
  std::string store_key;
  GetStoreKey(key, CommType::ALLREDUCE, &store_key);

  auto it = place_to_comm_ctx_.find(key);
  if (it == place_to_comm_ctx_.end()) {
    CreateFlagcxEnvCache(place, key, store_key, CommType::ALLREDUCE);
  }
}

void ProcessGroupFlagcx::EagerConnectRingExchange() {
  std::vector<std::pair<int, int>> peers;
  const auto& place = phi::GPUPlace(phi::backends::gpu::GetCurrentDeviceId());

  for (int rank = 0; rank < size_; rank++) {
    auto peer_rank = rank + 1 >= size_ ? 0 : rank + 1;
    peers.push_back(std::make_pair(rank, peer_rank));
  }

  for (auto& peer : peers) {
    int f_rank = peer.first;
    int s_rank = peer.second;

    int peer_rank = 0;
    int cur_rank = rank_;
    if (rank_ == f_rank) {
      peer_rank = s_rank;
    } else if (rank_ == s_rank) {
      peer_rank = f_rank;
    } else {
      continue;
    }

    int low_rank = cur_rank < peer_rank ? cur_rank : peer_rank;
    int high_rank = cur_rank < peer_rank ? peer_rank : cur_rank;
    std::string key =
        std::to_string(low_rank) + "->" + std::to_string(high_rank);

    auto p2p_rank = rank_ < peer_rank ? 0 : 1;
    platform::CUDADeviceGuard cuda_guard(place);
    std::string store_key;
    GetStoreKey(key, CommType::SEND, &store_key);
    if (place_to_comm_ctx_.find(key) == place_to_comm_ctx_.end()) {
      CreateFlagcxEnvCache(place, key, store_key, CommType::SEND, p2p_rank);
    }
  }
}

std::shared_ptr<ProcessGroup::Task> ProcessGroupFlagcx::Collective(
    std::function<void(phi::distributed::FlagcxCommContext*, flagcxStream_t)>
        fn,
    const std::vector<phi::DenseTensor>& tensors,
    CommType comm_type,
    bool sync_op,
    bool use_calc_stream) {
  CheckTensorContiguous(tensors);

  VLOG(3) << "flagcx debug: collective start";
  comm_seq_++;
  PADDLE_ENFORCE_GT(
      tensors.size(),
      0,
      common::errors::InvalidArgument("Num of tensors must be greater than 0"));
  const auto& place = tensors[0].place();
  const auto& key = GetKeyFromPlace(place);

  platform::CUDADeviceGuard cuda_guard(place);

  std::string store_key;
  GetStoreKey(key, comm_type, &store_key);

  if (place_to_comm_ctx_.find(key) == place_to_comm_ctx_.end()) {
    CreateFlagcxEnvCache(place, key, store_key, comm_type);
  }

  if (!use_calc_stream) {
    SyncCalcStream(place, key);
  }

  auto task =
      CreateTask(place, rank_, comm_type, sync_op, use_calc_stream, gid_);

  const auto& comm_ctx = place_to_comm_ctx_.at(key);
  const auto* calc_ctx = place_to_calc_ctx_.at(key);

  auto flagcx_comm_ctx = this->GetCommContext(&store_key);

  flagcxStream_t flagcx_stream;
  if (use_calc_stream) {
    auto calc_stream = calc_ctx->stream();
    flagcx_comm_ctx->flagcx_handler_->devHandle->streamCopy(
        &flagcx_stream, reinterpret_cast<void*>(&calc_stream));
  } else {
    auto comm_stream = comm_ctx->stream();
    flagcx_comm_ctx->flagcx_handler_->devHandle->streamCopy(
        &flagcx_stream, reinterpret_cast<void*>(&comm_stream));
  }

  if (!FLAGS_enable_async_trace) {
    fn(flagcx_comm_ctx, flagcx_stream);
  }

  if (!use_calc_stream) {
    if (!is_coalescing_) {
      task->UpdateWaitChain(*comm_ctx);
      for (size_t i = 0; i < tensors.size(); ++i) {
        allocation_stream_pairs_.emplace_back(
            tensors[i].Holder(),
            *reinterpret_cast<gpuStream_t*>(flagcx_stream));
      }
    } else {
      for (size_t i = 0; i < tensors.size(); ++i) {
        coalescing_tensors_.emplace_back(
            std::make_shared<phi::DenseTensor>(tensors[i]));
      }
      coalescing_place_keys_.push_back(key);
    }
  }

  if (sync_op) {
    task->Wait();
  }

  flagcx_comm_ctx->flagcx_handler_->devHandle->streamFree(flagcx_stream);

  return task;
}

std::shared_ptr<ProcessGroup::Task> ProcessGroupFlagcx::Collective(
    std::function<void(phi::distributed::FlagcxCommContext*, flagcxStream_t)>
        fn,
    const phi::DenseTensor& tensor,
    CommType comm_type,
    bool sync_op,
    bool use_calc_stream) {
  const std::vector<phi::DenseTensor> tensors = {tensor};
  return Collective(fn, tensors, comm_type, sync_op, use_calc_stream);
}

phi::distributed::FlagcxCommContext* ProcessGroupFlagcx::GetCommContext(
    const std::string* key) {
  std::string store_key = std::to_string(this->gid_);
  if (key && !key->empty()) {
    store_key = *key;
  }
  const auto& comm_context_manager =
      phi::distributed::CommContextManager::GetInstance();
  auto comm_context = static_cast<phi::distributed::FlagcxCommContext*>(
      comm_context_manager.Get(store_key));
  PADDLE_ENFORCE_NE(
      comm_context,
      nullptr,
      common::errors::Unavailable("FlagcxCommContext is nullptr"));
  return comm_context;
}

void ProcessGroupFlagcx::StartCoalescing() {
  PADDLE_ENFORCE_EQ(is_coalescing_,
                    false,
                    common::errors::PreconditionNotMet(
                        "Coalescing is on, please call EndCoalesce."));
  is_coalescing_ = true;
  this->GroupStart();
}

void ProcessGroupFlagcx::EndCoalescing(
    std::optional<std::vector<std::shared_ptr<ProcessGroup::Task>>> tasks_opt) {
  this->GroupEnd();

  // NOTE(shenliang03): If using calculate stream, no need to record stream and
  // update task.
  if (!tasks_opt.has_value() || coalescing_tensors_.empty()) {
    is_coalescing_ = false;
    return;
  }

  auto& tasks = tasks_opt.value();

  PADDLE_ENFORCE_EQ(
      tasks.size(),
      coalescing_tensors_.size(),
      common::errors::PreconditionNotMet(
          "Number of tasks[%d] do not match number of collectives[%d].",
          tasks.size(),
          coalescing_tensors_.size()));

  for (size_t i = 0; i < tasks.size(); ++i) {
    auto* flagcx_task =
        static_cast<ProcessGroupFlagcx::FlagcxTask*>(tasks[i].get());
    const auto& tensor = coalescing_tensors_[i];
    const auto& key = coalescing_place_keys_[i];
    const auto& comm_ctx = place_to_comm_ctx_.at(key);
    auto flagcx_comm_ctx = this->GetCommContext(&store_key_);
    auto comm_stream = comm_ctx->stream();
    flagcxStream_t flagcx_stream;
    flagcx_comm_ctx->flagcx_handler_->devHandle->streamCopy(
        &flagcx_stream, reinterpret_cast<void*>(&comm_stream));

    flagcx_task->UpdateWaitChain(*comm_ctx);
    allocation_stream_pairs_.emplace_back(
        tensor->Holder(), *reinterpret_cast<gpuStream_t*>(flagcx_stream));
    flagcx_comm_ctx->flagcx_handler_->devHandle->streamFree(flagcx_stream);
  }

  is_coalescing_ = false;
  coalescing_tensors_.clear();
  coalescing_place_keys_.clear();
}

}  // namespace paddle::distributed
