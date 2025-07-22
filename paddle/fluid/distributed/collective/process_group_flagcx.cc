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
#include "paddle/utils/string/string_helper.h"

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

flagcxComm_t ProcessGroupFlagcx::FlagcxComm(const Place& place) const {
  PADDLE_ENFORCE_NOT_NULL(
      flagcx_comm_,
      ::common::errors::InvalidArgument("flagcx_comm_ is nullptr"));
  return flagcx_comm_;
}

phi::distributed::FlagcxCommContext* ProcessGroupFlagcx::GetOrCreateCommContext(
    const Place& place, CommType comm_type) {
  const auto& key = GetKeyFromPlace(place);
  std::string store_key;
  GetStoreKey(key, comm_type, &store_key);
  if (place_to_comm_ctx_.find(key) == place_to_comm_ctx_.end()) {
    CreateFlagcxEnvCache(place, key, store_key, comm_type);
  }
  return GetCommContext(&store_key);
}

std::shared_ptr<ProcessGroup::Task> ProcessGroupFlagcx::AllGather(
    phi::DenseTensor* out_tensor,
    const phi::DenseTensor& in_tensor,
    int64_t offset,
    int64_t numel,
    bool sync_op,
    bool use_calc_stream) {
  CheckTensorContiguous(in_tensor);
  CheckTensorContiguous(*out_tensor);

  // numel > 0 indicates the tensor need to be sliced
  const phi::DenseTensor& in_tensor_maybe_partial =
      numel > 0 ? GetPartialTensor(in_tensor, offset, numel) : in_tensor;
  return Collective(
      [&](phi::distributed::FlagcxCommContext* comm_context,
          flagcxStream_t stream) {
        VLOG(3) << "[flagcxAllGather] "
                << "sendbuff: " << in_tensor_maybe_partial.data()
                << ", recvbuff: " << out_tensor->data()
                << ", count: " << in_tensor_maybe_partial.numel()
                << ", datatype: "
                << FlagcxDTypeToString(
                       phi::ToFlagcxDataType(in_tensor_maybe_partial.dtype()))
                << ", flagcxcomm: " << comm_context->GetFlagcxComm()
                << ", stream: " << stream << ", rank_in_group: " << rank_
                << ", nranks: " << size_ << ", offset: " << offset
                << ", sync_op: " << sync_op
                << ", use_calc_stream: " << use_calc_stream << ", "
                << GetGroupMessage();
        comm_context->AllGather(out_tensor, in_tensor_maybe_partial, stream);
      },
      in_tensor_maybe_partial,
      CommType::ALLGATHER,
      sync_op,
      use_calc_stream);
}

std::shared_ptr<ProcessGroup::Task> ProcessGroupFlagcx::AllReduce(
    phi::DenseTensor* out_tensor,
    const phi::DenseTensor& in_tensor,
    const AllreduceOptions& opts,
    bool sync_op,
    bool use_calc_stream) {
  CheckTensorContiguous(in_tensor);
  CheckTensorContiguous(*out_tensor);

  return Collective(
      [&](phi::distributed::FlagcxCommContext* comm_context,
          flagcxStream_t stream) {
        VLOG(3) << "[flagcxAllReduce] "
                << "sendbuff: " << in_tensor.data()
                << ", recvbuff: " << out_tensor->data()
                << ", count: " << in_tensor.numel() << ", datatype: "
                << FlagcxDTypeToString(phi::ToFlagcxDataType(in_tensor.dtype()))
                << ", redop: "
                << FlagcxRedTypeToString(ToFlagcxRedType(opts.reduce_op))
                << ", flagcxcomm: " << comm_context->GetFlagcxComm()
                << ", stream: " << stream << ", rank_in_group: " << rank_
                << ", nranks: " << size_ << ", sync_op: " << sync_op
                << ", use_calc_stream: " << use_calc_stream << ", "
                << GetGroupMessage();

        comm_context->AllReduce(
            out_tensor, in_tensor, ToFlagcxRedType(opts.reduce_op), stream);
      },
      in_tensor,
      CommType::ALLREDUCE,
      sync_op,
      use_calc_stream);
}

std::shared_ptr<ProcessGroup::Task> ProcessGroupFlagcx::AllToAll(
    phi::DenseTensor* out_tensor,
    const phi::DenseTensor& in_tensor,
    const std::vector<int64_t>& out_size_each_rank,
    const std::vector<int64_t>& in_size_each_rank,
    bool sync_op,
    bool use_calc_stream) {
  CheckTensorContiguous(in_tensor);
  CheckTensorContiguous(*out_tensor);

  std::vector<int64_t> out_split_sizes;
  std::vector<int64_t> in_split_sizes;
  if (out_size_each_rank.empty() && in_size_each_rank.empty()) {
    out_split_sizes =
        std::vector<int64_t>(size_, out_tensor->dims()[0] / size_);
    in_split_sizes = std::vector<int64_t>(size_, in_tensor.dims()[0] / size_);
  } else {
    out_split_sizes = out_size_each_rank;
    in_split_sizes = in_size_each_rank;
  }

  const phi::DDim& out_dim = out_tensor->dims();
  const phi::DDim& in_dim = in_tensor.dims();
  // CheckSizeOnEachRank(out_dim, out_size_each_rank, size_);
  // CheckSizeOnEachRank(in_dim, in_size_each_rank, size_);
  CheckSizeOnEachRank(out_dim, out_split_sizes, size_);
  CheckSizeOnEachRank(in_dim, in_split_sizes, size_);

  return Collective(
      [&](phi::distributed::FlagcxCommContext* comm_context,
          flagcxStream_t stream) {
        int64_t in_row_size =
            in_dim[0] == 0 ? 0 : in_tensor.numel() / in_dim[0];
        int64_t out_row_size =
            out_dim[0] == 0 ? 0 : out_tensor->numel() / out_dim[0];
        int64_t in_offset = 0, in_numel = 0, out_offset = 0, out_numel = 0;
        phi::DenseTensor input_partial, output_partial;

        VLOG(3) << "[AllToAll] "
                << "sendbuff: " << in_tensor.data()
                << ", recvbuff: " << out_tensor->data()
                << ", count: " << in_tensor.numel() << ", datatype: "
                << FlagcxDTypeToString(phi::ToFlagcxDataType(in_tensor.dtype()))
                << ", flagcxcomm: " << comm_context->GetFlagcxComm()
                << ", stream: " << stream << ", rank_in_group: " << rank_
                << ", nranks: " << size_ << ", out_split_sizes: "
                << string::join_strings(out_split_sizes, ',')
                << ", in_split_sizes: "
                << string::join_strings(in_split_sizes, ',')
                << ", sync_op: " << sync_op
                << ", use_calc_stream: " << use_calc_stream << ", "
                << GetGroupMessage();

        GroupStart();
        for (auto i = 0; i < size_; i++) {
          in_numel = in_split_sizes[i] * in_row_size;

          if (in_numel > 0) {
            input_partial = GetPartialTensor(in_tensor, in_offset, in_numel);
            comm_context->Send(input_partial, in_numel, i, stream);
          }
          in_offset += in_numel;
          out_numel = out_split_sizes[i] * out_row_size;
          if (out_numel > 0) {
            output_partial =
                GetPartialTensor(*out_tensor, out_offset, out_numel);
            comm_context->Recv(&output_partial, out_numel, i, stream);
          }
          out_offset += out_numel;
        }
        GroupEnd();
      },
      in_tensor,
      CommType::ALLTOALL,
      sync_op,
      use_calc_stream);
}

std::shared_ptr<ProcessGroup::Task> ProcessGroupFlagcx::AllToAll(
    std::vector<phi::DenseTensor>* out_tensors,
    const std::vector<phi::DenseTensor>& in_tensors,
    bool sync_op,
    bool use_calc_stream) {
  CheckTensorContiguous(in_tensors);
  CheckTensorContiguous(*out_tensors);
  CheckTensorSamePlace(in_tensors);
  CheckTensorSamePlace(*out_tensors);
  phi::distributed::CommStaticCheck::CheckDataType(*out_tensors, in_tensors);

  PADDLE_ENFORCE_EQ(
      out_tensors->size(),
      size_,
      common::errors::InvalidArgument(
          "Number of out tensors[%d] do not match the world size[%d].",
          out_tensors->size(),
          size_));
  PADDLE_ENFORCE_EQ(
      in_tensors.size(),
      size_,
      common::errors::InvalidArgument(
          "Number of in tensors[%d] do not match the world size[%d].",
          in_tensors.size(),
          size_));

  return Collective(
      [&](phi::distributed::FlagcxCommContext* comm_context,
          flagcxStream_t stream) {
        VLOG(3) << "[AllToAll] "
                << "sendbuff: "
                << string::join_strings(GetTensorPtrs(in_tensors), ',')
                << ", recvbuff: "
                << string::join_strings(GetTensorPtrs(*out_tensors), ',')
                << ", datatype: "
                << FlagcxDTypeToString(
                       phi::ToFlagcxDataType(in_tensors[0].dtype()))
                << ", flagcxcomm: " << comm_context->GetFlagcxComm()
                << ", stream: " << stream << ", rank_in_group: " << rank_
                << ", nranks: " << size_ << ", out_split_sizes: "
                << string::join_strings(GetAllToAllSplitSizes(*out_tensors),
                                        ',')
                << ", in_split_sizes: "
                << string::join_strings(GetAllToAllSplitSizes(in_tensors), ',')
                << ", sync_op: " << sync_op
                << ", use_calc_stream: " << use_calc_stream << ", "
                << GetGroupMessage();

        GroupStart();
        for (auto i = 0; i < size_; i++) {
          int64_t in_numel = in_tensors[i].numel();
          int64_t out_numel = (*out_tensors)[i].numel();

          if (in_numel > 0) {
            comm_context->Send(in_tensors[i], in_numel, i, stream);
          }

          if (out_numel > 0) {
            comm_context->Recv(&(*out_tensors)[i], out_numel, i, stream);
          }
        }
        GroupEnd();
      },
      in_tensors,
      CommType::ALLTOALL,
      sync_op,
      use_calc_stream);
}

std::shared_ptr<ProcessGroup::Task> ProcessGroupFlagcx::Barrier(
    const BarrierOptions& opts) {
  PADDLE_ENFORCE_GE(opts.device_id,
                    0,
                    common::errors::PreconditionNotMet(
                        "The barrier device id must greater or equal than 0."));
  phi::GPUPlace place(opts.device_id);
  auto allocator = std::unique_ptr<phi::Allocator>(
      new paddle::experimental::DefaultAllocator(place));
  phi::DenseTensorMeta meta(phi::DataType::FLOAT32, phi::DDim{1});
  phi::DenseTensor barrier_tensor{allocator.get(), meta};

  VLOG(3) << "[Barrier] "
          << "barrier opt: " << opts.device_id;

  auto task = AllReduce(&barrier_tensor,
                        barrier_tensor,
                        {},
                        /*sync_op*/ true,
                        /*use_calc_stream*/ false);
  auto flagcx_task = dynamic_cast<FlagcxTask*>(task.get());
  flagcx_task->SetBlockCPUInWait();
  return task;
}

std::shared_ptr<ProcessGroup::Task> ProcessGroupFlagcx::Broadcast(
    phi::DenseTensor* out_tensor,
    const phi::DenseTensor& in_tensor,
    const BroadcastOptions& opts,
    bool sync_op,
    bool use_calc_stream) {
  CheckTensorContiguous(in_tensor);
  CheckTensorContiguous(*out_tensor);

  return Collective(
      [&](phi::distributed::FlagcxCommContext* comm_context,
          flagcxStream_t stream) {
        int root = opts.source_rank + opts.source_root;

        VLOG(3) << "[flagcxBroadcast] "
                << "sendbuff: " << in_tensor.data()
                << ", recvbuff: " << out_tensor->data()
                << ", count: " << in_tensor.numel() << ", datatype: "
                << FlagcxDTypeToString(phi::ToFlagcxDataType(in_tensor.dtype()))
                << ", root: " << root
                << ", flagcxcomm: " << comm_context->GetFlagcxComm()
                << ", stream: " << stream << ", rank_in_group: " << rank_
                << ", nranks: " << size_ << ", sync_op: " << sync_op
                << ", use_calc_stream: " << use_calc_stream << ", "
                << GetGroupMessage();
        comm_context->Broadcast(out_tensor, in_tensor, root, stream);
      },
      in_tensor,
      CommType::BROADCAST,
      sync_op,
      use_calc_stream);
}

std::shared_ptr<ProcessGroup::Task> ProcessGroupFlagcx::Reduce(
    phi::DenseTensor* out_tensor,
    const phi::DenseTensor& in_tensor,
    const ReduceOptions& opts,
    bool sync_op,
    bool use_calc_stream) {
  CheckTensorContiguous(in_tensor);
  CheckTensorContiguous(*out_tensor);

  return Collective(
      [&](phi::distributed::FlagcxCommContext* comm_context,
          flagcxStream_t stream) {
        VLOG(3) << "[flagcxReduce] "
                << "sendbuff: " << in_tensor.data()
                << ", recvbuff: " << out_tensor->data()
                << ", count: " << in_tensor.numel() << ", datatype: "
                << FlagcxDTypeToString(phi::ToFlagcxDataType(in_tensor.dtype()))
                << ", redop: "
                << FlagcxRedTypeToString(ToFlagcxRedType(opts.reduce_op))
                << ", root: " << opts.root_rank
                << ", flagcxcomm: " << comm_context->GetFlagcxComm()
                << ", stream: " << stream << ", rank_in_group: " << rank_
                << ", nranks: " << size_ << ", sync_op: " << sync_op
                << ", use_calc_stream: " << use_calc_stream << ", "
                << GetGroupMessage();
        comm_context->Reduce(out_tensor,
                             in_tensor,
                             ToFlagcxRedType(opts.reduce_op),
                             opts.root_rank,
                             stream);
      },
      in_tensor,
      CommType::REDUCE,
      sync_op,
      use_calc_stream);
}

std::shared_ptr<ProcessGroup::Task> ProcessGroupFlagcx::ReduceScatter(
    phi::DenseTensor* out_tensor,
    const phi::DenseTensor& in_tensor,
    const ReduceScatterOptions& opts,
    bool sync_op,
    bool use_calc_stream) {
  CheckTensorContiguous(in_tensor);
  CheckTensorContiguous(*out_tensor);

  return Collective(
      [&](phi::distributed::FlagcxCommContext* comm_context,
          flagcxStream_t stream) {
        VLOG(3) << "[flagcxReduceScatter] "
                << "sendbuff: " << in_tensor.data()
                << ", recvbuff: " << out_tensor->data()
                << ", count: " << in_tensor.numel() << ", datatype: "
                << FlagcxDTypeToString(phi::ToFlagcxDataType(in_tensor.dtype()))
                << ", redop: "
                << FlagcxRedTypeToString(ToFlagcxRedType(opts.reduce_op))
                << ", flagcxcomm: " << comm_context->GetFlagcxComm()
                << ", stream: " << stream << ", rank_in_group: " << rank_
                << ", nranks: " << size_ << ", sync_op: " << sync_op
                << ", use_calc_stream: " << use_calc_stream << ", "
                << GetGroupMessage();
        comm_context->ReduceScatter(
            out_tensor, in_tensor, ToFlagcxRedType(opts.reduce_op), stream);
      },
      in_tensor,
      CommType::REDUCE_SCATTER,
      sync_op,
      use_calc_stream);
}

std::shared_ptr<ProcessGroup::Task> ProcessGroupFlagcx::Scatter(
    phi::DenseTensor* out_tensor,
    const phi::DenseTensor& in_tensor,
    const ScatterOptions& opts,
    bool sync_op,
    bool use_calc_stream) {
  CheckTensorContiguous(in_tensor);
  CheckTensorContiguous(*out_tensor);

  phi::distributed::CommStaticCheck::ScatterLikeShape(
      *out_tensor,
      in_tensor,
      /*dst_rank*/ opts.root_rank,
      /*cur_rank*/ rank_,
      size_);
  return Collective(
      [&](phi::distributed::FlagcxCommContext* comm_context,
          flagcxStream_t stream) {
        VLOG(3) << "[Scatter] "
                << "sendbuff: " << in_tensor.data()
                << ", recvbuff: " << out_tensor->data()
                << ", count: " << in_tensor.numel() << ", datatype: "
                << FlagcxDTypeToString(phi::ToFlagcxDataType(in_tensor.dtype()))
                << ", root: " << opts.root_rank
                << ", flagcxcomm: " << comm_context->GetFlagcxComm()
                << ", stream: " << stream << ", rank_in_group: " << rank_
                << ", nranks: " << size_ << ", sync_op: " << sync_op
                << ", use_calc_stream: " << use_calc_stream << ", "
                << GetGroupMessage();

        int64_t numel = in_tensor.numel() / size_;
        if (rank_ == opts.root_rank) {
          int64_t offset = 0;
          phi::DenseTensor partial_tensor;
          this->GroupStart();
          for (auto i = 0; i < size_; i++) {
            partial_tensor = GetPartialTensor(in_tensor, offset, numel);
            comm_context->Send(partial_tensor, numel, i, stream);
            offset += numel;
          }
          comm_context->Recv(out_tensor, numel, opts.root_rank, stream);
          this->GroupEnd();
        } else {
          comm_context->Recv(out_tensor, numel, opts.root_rank, stream);
        }
      },
      in_tensor,
      CommType::SCATTER,
      sync_op,
      use_calc_stream);
}

std::shared_ptr<ProcessGroup::Task> ProcessGroupFlagcx::Gather(
    phi::DenseTensor* out_tensor,
    const phi::DenseTensor& in_tensor,
    const GatherOptions& opts,
    bool sync_op,
    bool use_calc_stream) {
  CheckTensorContiguous(in_tensor);
  CheckTensorContiguous(*out_tensor);

  std::vector<phi::DenseTensor> partial_tensors;
  if (rank_ == opts.root_rank) {
    partial_tensors.reserve(size_);
    size_t offset = 0;
    size_t numel = out_tensor->numel() / size_;
    for (auto i = 0; i < size_; i++) {
      partial_tensors.push_back(GetPartialTensor(*out_tensor,
                                                 static_cast<int64_t>(offset),
                                                 static_cast<int64_t>(numel)));
      offset += numel;
    }
  }
  return Gather(&partial_tensors, in_tensor, opts, sync_op, use_calc_stream);
}

std::shared_ptr<ProcessGroup::Task> ProcessGroupFlagcx::Gather(
    std::vector<phi::DenseTensor>* gather_tensors_ptr,
    const phi::DenseTensor& in_tensor,
    const GatherOptions& opts,
    bool sync_op,
    bool use_calc_stream) {
  CheckTensorContiguous(in_tensor);
  CheckTensorContiguous(*gather_tensors_ptr);

  auto& gather_tensors = *gather_tensors_ptr;
  PADDLE_ENFORCE_GT(size_,
                    opts.root_rank,
                    common::errors::InvalidArgument(
                        "root world size [%d]  is less than root rank [%d]",
                        size_,
                        opts.root_rank));
  auto gather_func = [&](phi::distributed::FlagcxCommContext* comm_context,
                         flagcxStream_t stream) {
    VLOG(3) << "[Gather] "
            << "sendbuff: " << in_tensor.data()
            << ", count: " << in_tensor.numel() << ", datatype: "
            << FlagcxDTypeToString(phi::ToFlagcxDataType(in_tensor.dtype()))
            << ", root: " << opts.root_rank
            << ", flagcxcomm: " << comm_context->GetFlagcxComm()
            << ", stream: " << stream << ", rank_in_group: " << rank_
            << ", nranks: " << size_ << ", sync_op: " << sync_op
            << ", use_calc_stream: " << use_calc_stream << ", "
            << ", " << GetGroupMessage();

    this->GroupStart();
    // root receive from all devices
    if (rank_ == opts.root_rank) {
      for (auto i = 0; i < size_; i++) {
        auto& gather_tensor = gather_tensors[i];
        comm_context->Recv(&gather_tensor, gather_tensor.numel(), i, stream);
      }
    }
    // send to root
    comm_context->Send(in_tensor, in_tensor.numel(), opts.root_rank, stream);
    this->GroupEnd();
  };
  return Collective(
      gather_func, in_tensor, CommType::GATHER, sync_op, use_calc_stream);
}

std::shared_ptr<ProcessGroup::Task> ProcessGroupFlagcx::Recv(
    phi::DenseTensor* tensor,
    int src_rank,
    int64_t offset,
    int64_t numel,
    bool sync_op,
    bool use_calc_stream) {
  CheckTensorContiguous(*tensor);
  // numel > 0 indicates the tensor need to be sliced
  phi::DenseTensor partial_tensor;
  if (numel > 0) {
    partial_tensor = GetPartialTensor(*tensor, offset, numel);
    tensor = &partial_tensor;
  }

  return Point2Point(
      [&](phi::distributed::FlagcxCommContext* comm_context,
          flagcxStream_t stream,
          int rank_in_group) {
        VLOG(3) << "[flagcxRecv] "
                << "recvbuff: " << tensor->data()
                << ", count: " << tensor->numel() << ", datatype: "
                << FlagcxDTypeToString(phi::ToFlagcxDataType(tensor->dtype()))
                << ", src_in_group: " << src_rank
                << ", flagcxcomm: " << comm_context->GetFlagcxComm()
                << ", stream: " << stream
                << ", rank_in_group: " << rank_in_group << ", nranks: " << size_
                << ", offset: " << offset << ", sync_op: " << sync_op
                << ", use_calc_stream: " << use_calc_stream << ", "
                << GetGroupMessage();

        comm_context->Recv(tensor, tensor->numel(), rank_in_group, stream);
      },
      src_rank,
      *tensor,
      CommType::RECV,
      sync_op,
      use_calc_stream);
}

std::shared_ptr<ProcessGroup::Task> ProcessGroupFlagcx::Send(
    const phi::DenseTensor& tensor,
    int dst_rank,
    int64_t offset,
    int64_t numel,
    bool sync_op,
    bool use_calc_stream) {
  CheckTensorContiguous(tensor);
  // numel > 0 indicates the tensor need to be sliced
  const phi::DenseTensor& tensor_maybe_partial =
      numel > 0 ? GetPartialTensor(tensor, offset, numel) : tensor;

  return Point2Point(
      [&](phi::distributed::FlagcxCommContext* comm_context,
          flagcxStream_t stream,
          int rank_in_group) {
        VLOG(3) << "[flagcxSend] "
                << "sendbuff: " << tensor_maybe_partial.data()
                << ", count: " << tensor_maybe_partial.numel() << ", datatype: "
                << FlagcxDTypeToString(
                       phi::ToFlagcxDataType(tensor_maybe_partial.dtype()))
                << ", dst_in_group: " << dst_rank
                << ", flagcxcomm: " << comm_context->GetFlagcxComm()
                << ", stream: " << stream
                << ", rank_in_group: " << rank_in_group << ", nranks: " << size_
                << ", offset: " << offset << ", sync_op: " << sync_op
                << ", use_calc_stream: " << use_calc_stream << ", "
                << GetGroupMessage();

        comm_context->Send(tensor_maybe_partial,
                           tensor_maybe_partial.numel(),
                           rank_in_group,
                           stream);
      },
      dst_rank,
      tensor_maybe_partial,
      CommType::SEND,
      sync_op,
      use_calc_stream);
}

std::shared_ptr<ProcessGroupFlagcx::FlagcxTask> ProcessGroupFlagcx::CreateTask(
    const Place& place,
    int rank,
    CommType comm_type,
    bool is_sync,
    bool use_calc_stream,
    int gid) {
  return std::make_shared<ProcessGroupFlagcx::FlagcxTask>(
      place, rank, comm_type, is_sync, use_calc_stream, gid);
}

void ProcessGroupFlagcx::GetStoreKey(const std::string& place_key,
                                     CommType comm_type,
                                     std::string* store_key) {
  *store_key = "flagcx_ids/" + std::to_string(gid_) + "/0";

  place_to_group_key_[place_key] = *store_key;
}

void ProcessGroupFlagcx::CreateFlagcxEnvCache(const Place& place,
                                              const std::string& place_key,
                                              const std::string& store_key,
                                              CommType comm_type,
                                              int p2p_rank) {
  // TODO(changtao): we only support one flagcx comm ctx
  if (flagcx_comm_ != nullptr) {
    return;
  }
  VLOG(3) << "init flagcx rank_in_group: " << rank_ << ", nranks: " << size_
          << ", gid: " << gid_ << ", place key: " << place_key
          << ", store_key: " << store_key;
  store_key_ = store_key;

  phi::distributed::CommContextManager::CreateFlagcxCommContext(
      store_, store_key, rank_, size_, "");

  auto flagcx_comm_ctx = this->GetCommContext(&store_key);
  VLOG(3) << "Get flagcx comm: " << flagcx_comm_ctx->GetFlagcxComm();
  flagcx_comm_ = flagcx_comm_ctx->GetFlagcxComm();
  auto comm_ctx = std::make_unique<phi::GPUContext>(place);

  auto* calc_ctx = static_cast<phi::GPUContext*>(
      phi::DeviceContextPool::Instance().Get(place));

  place_to_calc_event_.emplace(
      place_key,
      platform::DeviceEvent(place, platform::GenerateDeviceEventFlag()));
  place_to_calc_ctx_.emplace(place_key, calc_ctx);
  place_to_comm_ctx_.emplace(place_key, std::move(comm_ctx));
}

void ProcessGroupFlagcx::SyncCalcStream(const Place& place,
                                        const std::string& place_key) {
  auto& calc_event = place_to_calc_event_.at(place_key);
  const auto* calc_ctx = place_to_calc_ctx_.at(place_key);
  const auto* comm_ctx = place_to_comm_ctx_.at(place_key).get();
  calc_event.Record(calc_ctx);
  calc_event.Wait(platform::Place2DeviceType(place), comm_ctx);
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

std::shared_ptr<ProcessGroup::Task> ProcessGroupFlagcx::Point2Point(
    std::function<
        void(phi::distributed::FlagcxCommContext*, flagcxStream_t, int)> fn,
    int peer,
    const phi::DenseTensor& tensor,
    CommType comm_type,
    bool sync_op,
    bool use_calc_stream) {
  CheckTensorContiguous(tensor);

  const auto& place = tensor.place();

  int p2p_rank = 0;
  int p2p_target_rank = 0;
  bool is_batch_p2p = s_group_call_counter > 0;
  std::string key = "";

  if (is_batch_p2p) {
    key = GetKeyFromPlace(place);
    p2p_rank = rank_;
    p2p_target_rank = peer;
  } else {
    int low_rank = rank_ < peer ? rank_ : peer;
    int high_rank = rank_ < peer ? peer : rank_;
    key = std::to_string(low_rank) + "->" + std::to_string(high_rank);
    p2p_rank = rank_ < peer ? 0 : 1;
    p2p_target_rank = 1 - p2p_rank;
  }

  platform::CUDADeviceGuard cuda_guard(place);

  std::string store_key;
  GetStoreKey(key, comm_type, &store_key);

  if (place_to_comm_ctx_.find(key) == place_to_comm_ctx_.end()) {
    CreateFlagcxEnvCache(place, key, store_key, comm_type, p2p_rank);
  }
  if (p2p_comm_seq_.find(key) == p2p_comm_seq_.end()) {
    p2p_comm_seq_[key] = 0;
  }
  p2p_comm_seq_[key]++;

  if (!use_calc_stream) {
    SyncCalcStream(place, key);
  }

  auto task =
      CreateTask(place, rank_, comm_type, sync_op, use_calc_stream, gid_);
  const auto* calc_ctx = place_to_calc_ctx_.at(key);
  const auto& comm_ctx = place_to_comm_ctx_.at(key);

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
    fn(flagcx_comm_ctx, flagcx_stream, p2p_target_rank);
  }

  if (!use_calc_stream) {
    if (!is_coalescing_) {
      task->UpdateWaitChain(*comm_ctx);
      allocation_stream_pairs_.emplace_back(
          tensor.Holder(), *reinterpret_cast<gpuStream_t*>(flagcx_stream));
    } else {
      coalescing_tensors_.emplace_back(
          std::make_shared<phi::DenseTensor>(tensor));
      coalescing_place_keys_.push_back(key);
    }
  }

  if (sync_op) {
    task->Wait();
  }

  flagcx_comm_ctx->flagcx_handler_->devHandle->streamFree(flagcx_stream);
  return task;
}

std::shared_ptr<ProcessGroupFlagcx>
ProcessGroupFlagcx::CreateProcessGroupFlagcx(
    const std::shared_ptr<phi::distributed::Store>& store,
    int rank,
    int size,
    int gid,
    int64_t timeout,
    int flagcx_comm_init_option) {
  auto process_group = std::make_shared<ProcessGroupFlagcx>(
      store, rank, size, gid, timeout, flagcx_comm_init_option);
  ProcessGroupIdMap::GetInstance().emplace(gid, process_group);
  return process_group;
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
