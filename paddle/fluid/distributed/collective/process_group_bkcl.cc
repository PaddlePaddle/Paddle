// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/distributed/collective/process_group_bkcl.h"

#include "paddle/fluid/distributed/collective/bkcl_tools.h"
#include "paddle/fluid/distributed/collective/common.h"
#include "paddle/fluid/distributed/collective/utils.h"
#include "paddle/fluid/platform/device/xpu/bkcl_helper.h"
#include "paddle/fluid/platform/device/xpu/xpu_info.h"
#include "paddle/phi/api/lib/utils/allocator.h"
#include "paddle/phi/core/device_context.h"
#include "paddle/phi/core/distributed/check/static_check.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/errors.h"

namespace paddle {
namespace distributed {
using XPUDeviceContext = paddle::platform::XPUDeviceContext;

ProcessGroupBKCL::BKCLTask::BKCLTask(const Place& place,
                                     int rank,
                                     CommType comm_type,
                                     bool sync_op,
                                     bool use_calc_stream)
    : TaskStream(rank, comm_type, sync_op, use_calc_stream), place_(place) {
  comm_event_ = std::make_shared<XPUEventManager>();
}

ProcessGroupBKCL::BKCLTask::~BKCLTask() {}

bool ProcessGroupBKCL::BKCLTask::IsCompleted() {
  LOG_FIRST_N(WARNING, 1) << "XPU do not support event query now.";
  return true;
}

// TODO(sheniang03): Add timeout for wait, now timeout unused
bool ProcessGroupBKCL::BKCLTask::Wait(std::chrono::milliseconds timeout) {
  // Warning here when use calc stream but also invoke waiting explicitly.
  if (UseCalcStream()) {
    VLOG(3) << "Warning: The communication is on calc stream, wait here is "
               "useless.";
    return true;
  }

  const auto* calc_ctx = static_cast<XPUContext*>(
      platform::DeviceContextPool::Instance().Get(place_));
  comm_event_->Block(*calc_ctx);

  if (barrier_) {
    // If we use the work to do barrier, we should block cpu

    // TODO(zhangxiaoci) There is no such function that can sync entire device
    // for xpu (for now), so all we can do is sync whatever stream that we know
    // and hope for the best. Note that for correctness the communication stream
    // needs to be in sync mode.
    platform::XPUDeviceGuard guard(place_.GetDeviceId());
    xpu_wait();
    calc_ctx->Wait();
  }
  return true;
}

// Same as Wait
void ProcessGroupBKCL::BKCLTask::Synchronize() { Wait(kWaitTimeout); }

ProcessGroupBKCL::ProcessGroupBKCL(
    const std::shared_ptr<phi::distributed::Store>& store,
    int rank,
    int size,
    int gid)
    : ProcessGroupWithStream(rank, size, gid), store_(store) {}

void ProcessGroupBKCL::GroupStart() {
  VLOG(3) << "bkcl_group_start";
  PADDLE_ENFORCE_XPU_SUCCESS(bkcl_group_start());
}

void ProcessGroupBKCL::GroupEnd() {
  VLOG(3) << "bkcl_group_end";
  PADDLE_ENFORCE_XPU_SUCCESS(bkcl_group_end());
}

std::shared_ptr<ProcessGroup::Task> ProcessGroupBKCL::Recv(
    phi::DenseTensor* tensor,
    int src_rank,
    int64_t offset,
    int64_t numel,
    bool sync_op,
    bool use_calc_stream) {
  // numel > 0 indicates the tensor need to be sliced
  phi::DenseTensor partial_tensor;
  if (numel > 0) {
    partial_tensor = GetPartialTensor(*tensor, offset, numel);
    tensor = &partial_tensor;
  }

  return Collective(
      tensor,
      // have to pass a tensor here
      // TODO(zhangxiaoci) catch up with nccl's api
      *tensor,
      [&](phi::DenseTensor* output,
          const phi::DenseTensor& input,
          BKCLContext_t comm,
          const XPUStream& stream) {
        VLOG(3) << "calling bkcl_recv"
                << ", rank_id: " << platform::GetBKCLRankID(comm)
                << ", dev_id: " << platform::GetBKCLDevID(comm)
                << ", nranks: " << platform::GetBKCLNRanks(comm)
                << ", src_rank: " << src_rank << ", numel: " << output->numel()
                << ", dtype: " << output->type() << ", sync_op: " << sync_op
                << ", use_calc_stream: " << use_calc_stream;
        int r = bkcl_recv(comm,
                          output->data(),
                          output->numel(),
                          src_rank,
                          platform::ToBKCLDataType(
                              framework::TransToProtoVarType(output->type())),
                          stream);
        return r;
      },
      CommType::RECV,
      sync_op,
      use_calc_stream);
}

std::shared_ptr<ProcessGroup::Task> ProcessGroupBKCL::Send(
    const phi::DenseTensor& tensor,
    int dst_rank,
    int64_t offset,
    int64_t numel,
    bool sync_op,
    bool use_calc_stream) {
  // numel > 0 indicates the tensor need to be sliced
  const phi::DenseTensor& tensor_maybe_partial =
      numel > 0 ? GetPartialTensor(tensor, offset, numel) : tensor;

  return Collective(
      nullptr,
      tensor_maybe_partial,
      [&](phi::DenseTensor* output,
          const phi::DenseTensor& input,
          BKCLContext_t comm,
          const XPUStream& stream) {
        VLOG(3) << "calling bkcl_send"
                << ", rank_id: " << platform::GetBKCLRankID(comm)
                << ", dev_id: " << platform::GetBKCLDevID(comm)
                << ", nranks: " << platform::GetBKCLNRanks(comm)
                << ", dst_rank: " << dst_rank
                << ", input numel: " << input.numel()
                << ", dtype: " << input.type() << ", sync_op: " << sync_op
                << ", use_calc_stream: " << use_calc_stream;
        int r = bkcl_send(comm,
                          input.data(),
                          input.numel(),
                          dst_rank,
                          platform::ToBKCLDataType(
                              framework::TransToProtoVarType(input.type())),
                          stream);
        return r;
      },
      CommType::SEND,
      sync_op,
      use_calc_stream);
}

std::shared_ptr<ProcessGroupBKCL::BKCLTask> ProcessGroupBKCL::CreateTask(
    const Place& place,
    int rank,
    CommType comm_type,
    bool is_sync,
    bool use_calc_stream) {
  return std::make_shared<ProcessGroupBKCL::BKCLTask>(
      place, rank, comm_type, is_sync, use_calc_stream);
}

void ProcessGroupBKCL::BroadcastUniqueBKCLID(BKCLUniqueId* bkcl_id) {
  auto key = "ProcessGroupBKCL/bkcl_ids/" + std::to_string(gid_) + "/0";
  if (rank_ == 0) {
    auto id = std::vector<uint8_t>(
        reinterpret_cast<uint8_t*>(bkcl_id),
        reinterpret_cast<uint8_t*>(bkcl_id) + BKCL_UNIQUE_ID_BYTES);
    store_->set(key, id);
  } else {
    const auto& ret = store_->get(key);
    std::memcpy(bkcl_id, ret.data(), ret.size());
  }
}

void ProcessGroupBKCL::CreateBKCLEnvCache(const Place& place,
                                          const std::string& place_key) {
  platform::XPUDeviceGuard guard(place.GetDeviceId());
  BKCLUniqueId bkcl_id;
  if (rank_ == 0) {
    PADDLE_ENFORCE_XPU_SUCCESS(bkcl_get_unique_id(&bkcl_id));
  }
  BroadcastUniqueBKCLID(&bkcl_id);

  VLOG(3) << "init bkcl rank: " << rank_ << ", nranks: " << size_
          << ", place: " << place_key
          << ", bkcl uniqueid: " << SerializeBKCLUniqueId(bkcl_id);

  calc_event_ = std::make_shared<XPUEventManager>();
  auto* calc_ctx = static_cast<phi::XPUContext*>(
      platform::DeviceContextPool::Instance().Get(place));
  // must use XPUDeviceContext here to make sure XPUContext::Init() is called
  auto comm_ctx = std::make_unique<XPUDeviceContext>(place);
  // set allocator
  comm_ctx->SetAllocator(memory::allocation::AllocatorFacade::Instance()
                             .GetAllocator(place)
                             .get());

  BKCLContext_t bkcl_comm;
  BKCLCHECK(bkcl_init_rank(&bkcl_comm, GetRank(), GetSize(), &bkcl_id));
  comm_ctx->SetBkclContext(bkcl_comm);
  // comm context creates a separate XPU stream for communication
  comm_ctx->CreateStream();

  place_to_calc_ctx_[place_key] = calc_ctx;
  place_to_comm_ctx_[place_key] = std::move(comm_ctx);
}

void ProcessGroupBKCL::SyncCalcStream(const Place& place) {
  const std::string& key = GetKeyFromPlace(place);
  const auto* calc_ctx = place_to_calc_ctx_[key];
  const auto* comm_ctx = place_to_comm_ctx_[key].get();
  calc_event_->Record(*calc_ctx);
  calc_event_->Block(*comm_ctx);
}

template <typename Fn>
std::shared_ptr<ProcessGroup::Task> ProcessGroupBKCL::Collective(
    phi::DenseTensor* out_tensor,
    const phi::DenseTensor& in_tensor,
    Fn fn,
    CommType op_type,
    bool sync_op,
    bool use_calc_stream) {
  const auto& place = in_tensor.place();
  const auto& key = GetKeyFromPlace(place);

  if (!calc_event_ ||
      (place_to_comm_ctx_.find(key) == place_to_comm_ctx_.end())) {
    CreateBKCLEnvCache(place, key);
  }

  if (!use_calc_stream) {
    SyncCalcStream(place);
  }

  auto task = CreateTask(place, rank_, op_type, sync_op, use_calc_stream);

  const auto* calc_ctx = place_to_calc_ctx_[key];
  const auto& comm_ctx = place_to_comm_ctx_[key];
  auto bkcl_stream = use_calc_stream ? calc_ctx->stream() : comm_ctx->stream();
  PADDLE_ENFORCE_XPU_SUCCESS(
      fn(out_tensor, in_tensor, comm_ctx->bkcl_context(), bkcl_stream));

  if (!use_calc_stream) {
    PADDLE_ENFORCE_NOT_NULL(
        comm_ctx.get(), platform::errors::Fatal("comm context is nullptr."));
    task->comm_event_->Record(*comm_ctx.get());
  }

  return task;
}

std::shared_ptr<ProcessGroup::Task> ProcessGroupBKCL::AllReduce(
    phi::DenseTensor* out_tensor,
    const phi::DenseTensor& in_tensor,
    const AllreduceOptions& opts,
    bool sync_op,
    bool use_calc_stream) {
  return Collective(
      out_tensor,
      in_tensor,
      [&](phi::DenseTensor* output,
          const phi::DenseTensor& input,
          BKCLContext_t comm,
          const XPUStream& stream) {
        VLOG(3) << "calling bkcl_all_reduce"
                << ", rank_id: " << platform::GetBKCLRankID(comm)
                << ", dev_id: " << platform::GetBKCLDevID(comm)
                << ", nranks: " << platform::GetBKCLNRanks(comm)
                << ", numel: " << input.numel() << ", dtype: " << input.type()
                << ", reduce_type: " << ToBKCLRedType(opts.reduce_op)
                << ", sync_op: " << sync_op
                << ", use_calc_stream: " << use_calc_stream;
        int r =
            bkcl_all_reduce(comm,
                            input.data(),
                            output->data(),
                            input.numel(),
                            platform::ToBKCLDataType(
                                framework::TransToProtoVarType(input.type())),
                            ToBKCLRedType(opts.reduce_op),
                            stream);
        return r;
      },
      CommType::ALLREDUCE,
      sync_op,
      use_calc_stream);
}

std::shared_ptr<ProcessGroup::Task> ProcessGroupBKCL::Broadcast(
    phi::DenseTensor* out_tensor,
    const phi::DenseTensor& in_tensor,
    const BroadcastOptions& opts,
    bool sync_op,
    bool use_calc_stream) {
  return Collective(
      out_tensor,
      in_tensor,
      [&](phi::DenseTensor* output,
          const phi::DenseTensor& input,
          BKCLContext_t comm,
          const XPUStream& stream) {
        int root = opts.source_rank + opts.source_root;
        VLOG(3) << "calling bkcl_broadcast"
                << ", rank_id: " << platform::GetBKCLRankID(comm)
                << ", dev_id: " << platform::GetBKCLDevID(comm)
                << ", nranks: " << platform::GetBKCLNRanks(comm)
                << ", root: " << root << ", numel: " << input.numel()
                << ", dtype: " << input.type() << ", sync_op: " << sync_op
                << ", use_calc_stream: " << use_calc_stream;
        if (framework::TransToProtoVarType(input.dtype()) ==
            framework::proto::VarType::INT64) {
          // special for int64_t, send as int32_t with DOUBLE NUMEL
          int r = bkcl_broadcast(
              comm,
              input.data(),
              output->data(),
              input.numel() * 2,
              platform::ToBKCLDataType(framework::proto::VarType::INT32),
              root,
              stream);
          return r;
        } else {
          int r =
              bkcl_broadcast(comm,
                             input.data(),
                             output->data(),
                             input.numel(),
                             platform::ToBKCLDataType(
                                 framework::TransToProtoVarType(input.type())),
                             root,
                             stream);
          return r;
        }
      },
      CommType::BROADCAST,
      sync_op,
      use_calc_stream);
}

std::shared_ptr<ProcessGroup::Task> ProcessGroupBKCL::AllGather(
    phi::DenseTensor* out_tensor,
    const phi::DenseTensor& in_tensor,
    int64_t offset,
    int64_t numel,
    bool sync_op,
    bool use_calc_stream) {
  const phi::DenseTensor& in_tensor_maybe_partial =
      numel > 0 ? GetPartialTensor(in_tensor, offset, numel) : in_tensor;
  phi::distributed::CommStaticCheck::GatherLikeShape(*out_tensor,
                                                     in_tensor_maybe_partial,
                                                     /*dst_rank*/ rank_,
                                                     /*cur_rank*/ rank_,
                                                     size_,
                                                     phi::AllocationType::XPU);
  return Collective(
      out_tensor,
      in_tensor_maybe_partial,
      [&](phi::DenseTensor* output,
          const phi::DenseTensor& input,
          BKCLContext_t comm,
          const XPUStream& stream) {
        VLOG(3) << "calling bkcl_all_gather"
                << ", rank_id: " << platform::GetBKCLRankID(comm)
                << ", dev_id: " << platform::GetBKCLDevID(comm)
                << ", nranks: " << platform::GetBKCLNRanks(comm)
                << ", numel: " << in_tensor_maybe_partial.numel()
                << ", dtype: " << input.type() << ", sync_op: " << sync_op
                << ", use_calc_stream: " << use_calc_stream;
        int r =
            bkcl_all_gather(comm,
                            in_tensor_maybe_partial.data(),
                            in_tensor_maybe_partial.numel(),
                            output->data(),
                            platform::ToBKCLDataType(
                                framework::TransToProtoVarType(input.type())),
                            stream);
        return r;
      },
      CommType::ALLGATHER,
      sync_op,
      use_calc_stream);
}

std::shared_ptr<ProcessGroup::Task> ProcessGroupBKCL::Reduce(
    phi::DenseTensor* out_tensor,
    const phi::DenseTensor& in_tensor,
    const ReduceOptions& opts,
    bool sync_op,
    bool use_calc_stream) {
  return Collective(
      out_tensor,
      in_tensor,
      [&](phi::DenseTensor* output,
          const phi::DenseTensor& input,
          BKCLContext_t comm,
          const XPUStream& stream) {
        VLOG(3) << "calling bkcl_reduce"
                << ", rank_id: " << platform::GetBKCLRankID(comm)
                << ", dev_id: " << platform::GetBKCLDevID(comm)
                << ", nranks: " << platform::GetBKCLNRanks(comm)
                << ", root: " << opts.root_rank << ", numel: " << input.numel()
                << ", dtype: " << input.type()
                << ", reduce_type: " << ToBKCLRedType(opts.reduce_op)
                << ", sync_op: " << sync_op
                << ", use_calc_stream: " << use_calc_stream;
        int r = bkcl_reduce(comm,
                            input.data(),
                            output->data(),
                            input.numel(),
                            platform::ToBKCLDataType(
                                framework::TransToProtoVarType(input.type())),
                            ToBKCLRedType(opts.reduce_op),
                            opts.root_rank,
                            stream);
        return r;
      },
      CommType::REDUCE,
      sync_op,
      use_calc_stream);
}

std::shared_ptr<ProcessGroup::Task> ProcessGroupBKCL::ReduceScatter(
    phi::DenseTensor* out_tensor,
    const phi::DenseTensor& in_tensor,
    const ReduceScatterOptions& opts,
    bool sync_op,
    bool use_calc_stream) {
  return Collective(
      out_tensor,
      in_tensor,
      [&](phi::DenseTensor* output,
          const phi::DenseTensor& input,
          BKCLContext_t comm,
          const XPUStream& stream) {
        VLOG(3) << "calling bkcl_reduce_scatter"
                << ", rank_id: " << platform::GetBKCLRankID(comm)
                << ", dev_id: " << platform::GetBKCLDevID(comm)
                << ", nranks: " << platform::GetBKCLNRanks(comm)
                << ", numel: " << output->numel() << ", dtype: " << input.type()
                << ", reduce_type: " << ToBKCLRedType(opts.reduce_op)
                << ", sync_op: " << sync_op
                << ", use_calc_stream: " << use_calc_stream;
        int r = bkcl_reduce_scatter(
            comm,
            input.data(),
            output->data(),
            output->numel(),
            platform::ToBKCLDataType(
                framework::TransToProtoVarType(input.type())),
            ToBKCLRedType(opts.reduce_op),
            stream);
        return r;
      },
      CommType::REDUCE_SCATTER,
      sync_op,
      use_calc_stream);
}

std::shared_ptr<ProcessGroup::Task> ProcessGroupBKCL::Barrier(
    const BarrierOptions& opts) {
  PADDLE_ENFORCE_GE(opts.device_id,
                    0,
                    platform::errors::PreconditionNotMet(
                        "The barrier device id must greater or equal than 0."));
  platform::XPUPlace place(opts.device_id);
  auto allocator = std::unique_ptr<phi::Allocator>(
      new paddle::experimental::DefaultAllocator(place));
  phi::DenseTensorMeta meta(phi::DataType::FLOAT32, phi::DDim{1});
  phi::DenseTensor barrier_tensor{allocator.get(), meta};

  auto task = AllReduce(&barrier_tensor,
                        barrier_tensor,
                        {},
                        /*sync_op*/ true,
                        /*use_calc_stream*/ false);
  auto bkcl_task = dynamic_cast<BKCLTask*>(task.get());
  bkcl_task->barrier_ = true;
  return task;
}

phi::DeviceContext* ProcessGroupBKCL::GetDeviceContext(
    const Place& place) const {
  return GetDeviceContext(place, /*use_calc_stream*/ false);
}

phi::DeviceContext* ProcessGroupBKCL::GetDeviceContext(
    const Place& place, bool use_calc_stream) const {
  const std::string& key = GetKeyFromPlace(place);
  if (use_calc_stream) {
    const auto& iter = place_to_calc_ctx_.find(key);
    return iter->second;
  } else {
    const auto& iter = place_to_comm_ctx_.find(key);
    PADDLE_ENFORCE_NE(iter,
                      place_to_comm_ctx_.end(),
                      platform::errors::InvalidArgument(
                          "Cannot find device context in process group."));
    return iter->second.get();
  }
}

// below are old apis
std::shared_ptr<ProcessGroup::Task> ProcessGroupBKCL::AllReduce(
    std::vector<phi::DenseTensor>& in_tensors,
    std::vector<phi::DenseTensor>& out_tensors,
    const AllreduceOptions& opts) {
  PADDLE_ENFORCE_EQ(
      in_tensors.size(),
      1,
      platform::errors::InvalidArgument(
          "BKCL only support single tensor collective communication."));
  PADDLE_ENFORCE_EQ(
      out_tensors.size(),
      1,
      platform::errors::InvalidArgument(
          "BKCL only support single tensor collective communication."));
  PADDLE_ENFORCE_EQ(
      CheckTensorsInXPUPlace(in_tensors),
      true,
      platform::errors::InvalidArgument("All inputs should be in XPUPlace."));
  return Collective(
      &out_tensors[0],
      in_tensors[0],
      [&](phi::DenseTensor* output,
          const phi::DenseTensor& input,
          BKCLContext_t comm,
          const XPUStream& stream) {
        VLOG(3) << "calling bkcl_all_reduce"
                << ", rank_id: " << platform::GetBKCLRankID(comm)
                << ", dev_id: " << platform::GetBKCLDevID(comm)
                << ", nranks: " << platform::GetBKCLNRanks(comm)
                << ", numel: " << input.numel() << ", dtype: " << input.type()
                << ", reduce_type: " << ToBKCLRedType(opts.reduce_op)
                << ", sync_op: " << true << ", use_calc_stream: " << false;
        int r =
            bkcl_all_reduce(comm,
                            input.data(),
                            output->data(),
                            input.numel(),
                            platform::ToBKCLDataType(
                                framework::TransToProtoVarType(input.type())),
                            ToBKCLRedType(opts.reduce_op),
                            stream);
        return r;
      },
      CommType::ALLREDUCE,
      /*sync_op*/ true,
      /*use_calc_stream*/ false);
}

std::shared_ptr<ProcessGroup::Task> ProcessGroupBKCL::AllReduce(
    std::vector<phi::DenseTensor>& in_tensors,
    std::vector<phi::DenseTensor>& out_tensors,
    const AllreduceOptions& opts,
    bool sync_op) {
  PADDLE_ENFORCE_EQ(
      in_tensors.size(),
      1,
      platform::errors::InvalidArgument(
          "BKCL only support single tensor collective communication."));
  PADDLE_ENFORCE_EQ(
      out_tensors.size(),
      1,
      platform::errors::InvalidArgument(
          "BKCL only support single tensor collective communication."));
  PADDLE_ENFORCE_EQ(
      CheckTensorsInXPUPlace(in_tensors),
      true,
      platform::errors::InvalidArgument("All inputs should be in XPUPlace."));
  return Collective(
      &out_tensors[0],
      in_tensors[0],
      [&](phi::DenseTensor* output,
          const phi::DenseTensor& input,
          BKCLContext_t comm,
          const XPUStream& stream) {
        VLOG(3) << "calling bkcl_all_reduce"
                << ", rank_id: " << platform::GetBKCLRankID(comm)
                << ", dev_id: " << platform::GetBKCLDevID(comm)
                << ", nranks: " << platform::GetBKCLNRanks(comm)
                << ", numel: " << input.numel() << ", dtype: " << input.type()
                << ", reduce_type: " << ToBKCLRedType(opts.reduce_op)
                << ", sync_op: " << sync_op << ", use_calc_stream: " << false;
        int r =
            bkcl_all_reduce(comm,
                            input.data(),
                            output->data(),
                            input.numel(),
                            platform::ToBKCLDataType(
                                framework::TransToProtoVarType(input.type())),
                            ToBKCLRedType(opts.reduce_op),
                            stream);
        return r;
      },
      CommType::ALLREDUCE,
      sync_op,
      /*use_calc_stream*/ false);
}

std::shared_ptr<ProcessGroup::Task> ProcessGroupBKCL::Broadcast(
    std::vector<phi::DenseTensor>& in_tensors,
    std::vector<phi::DenseTensor>& out_tensors,
    const BroadcastOptions& opts) {
  PADDLE_ENFORCE_EQ(
      in_tensors.size(),
      1,
      platform::errors::InvalidArgument(
          "BKCL only support single tensor collective communication."));
  PADDLE_ENFORCE_EQ(
      out_tensors.size(),
      1,
      platform::errors::InvalidArgument(
          "BKCL only support single tensor collective communication."));
  PADDLE_ENFORCE_EQ(
      CheckTensorsInXPUPlace(in_tensors),
      true,
      platform::errors::InvalidArgument("All inputs should be in XPUPlace."));

  return Collective(
      &out_tensors[0],
      in_tensors[0],
      [&](phi::DenseTensor* output,
          const phi::DenseTensor& input,
          BKCLContext_t comm,
          const XPUStream& stream) {
        const auto root =
            opts.source_rank * in_tensors.size() + opts.source_root;
        VLOG(3) << "calling bkcl_broadcast"
                << ", rank_id: " << platform::GetBKCLRankID(comm)
                << ", dev_id: " << platform::GetBKCLDevID(comm)
                << ", nranks: " << platform::GetBKCLNRanks(comm)
                << ", root: " << root << ", numel: " << input.numel()
                << ", dtype: " << input.type() << ", sync_op: " << true
                << ", use_calc_stream: " << false;
        int r =
            bkcl_broadcast(comm,
                           input.data(),
                           output->data(),
                           input.numel(),
                           platform::ToBKCLDataType(
                               framework::TransToProtoVarType(input.type())),
                           root,
                           stream);
        return r;
      },
      CommType::BROADCAST,
      /*sync_op*/ true,
      /*use_calc_stream*/ false);
}

std::shared_ptr<ProcessGroup::Task> ProcessGroupBKCL::Broadcast(
    std::vector<phi::DenseTensor>& in_tensors,
    std::vector<phi::DenseTensor>& out_tensors,
    const BroadcastOptions& opts,
    bool sync_op) {
  PADDLE_ENFORCE_EQ(
      in_tensors.size(),
      1,
      platform::errors::InvalidArgument(
          "BKCL only support single tensor collective communication."));
  PADDLE_ENFORCE_EQ(
      out_tensors.size(),
      1,
      platform::errors::InvalidArgument(
          "BKCL only support single tensor collective communication."));
  PADDLE_ENFORCE_EQ(
      CheckTensorsInXPUPlace(in_tensors),
      true,
      platform::errors::InvalidArgument("All inputs should be in XPUPlace."));

  return Collective(
      &out_tensors[0],
      in_tensors[0],
      [&](phi::DenseTensor* output,
          const phi::DenseTensor& input,
          BKCLContext_t comm,
          const XPUStream& stream) {
        const auto root =
            opts.source_rank * in_tensors.size() + opts.source_root;
        VLOG(3) << "calling bkcl_broadcast"
                << ", rank_id: " << platform::GetBKCLRankID(comm)
                << ", dev_id: " << platform::GetBKCLDevID(comm)
                << ", nranks: " << platform::GetBKCLNRanks(comm)
                << ", root: " << root << ", numel: " << input.numel()
                << ", dtype: " << input.type() << ", sync_op: " << sync_op
                << ", use_calc_stream: " << false;
        int r =
            bkcl_broadcast(comm,
                           input.data(),
                           output->data(),
                           input.numel(),
                           platform::ToBKCLDataType(
                               framework::TransToProtoVarType(input.type())),
                           root,
                           stream);
        return r;
      },
      CommType::BROADCAST,
      sync_op,
      /*use_calc_stream*/ false);
}

std::shared_ptr<ProcessGroup::Task> ProcessGroupBKCL::AllGather(
    std::vector<phi::DenseTensor>& in_tensors,
    std::vector<phi::DenseTensor>& out_tensors) {
  PADDLE_ENFORCE_EQ(
      in_tensors.size(),
      1,
      platform::errors::InvalidArgument(
          "BKCL only support single tensor collective communication."));
  PADDLE_ENFORCE_EQ(
      out_tensors.size(),
      1,
      platform::errors::InvalidArgument(
          "BKCL only support single tensor collective communication."));
  PADDLE_ENFORCE_EQ(
      CheckTensorsInXPUPlace(in_tensors),
      true,
      platform::errors::InvalidArgument("All inputs should be in XPUPlace."));
  PADDLE_ENFORCE_EQ(
      CheckTensorsInXPUPlace(out_tensors),
      true,
      platform::errors::InvalidArgument("All outputs should be in XPUPlace."));
  return Collective(
      &out_tensors[0],
      in_tensors[0],
      [&](phi::DenseTensor* output,
          const phi::DenseTensor& input,
          BKCLContext_t comm,
          const XPUStream& stream) {
        VLOG(3) << "calling bkcl_all_gather"
                << ", rank_id: " << platform::GetBKCLRankID(comm)
                << ", dev_id: " << platform::GetBKCLDevID(comm)
                << ", nranks: " << platform::GetBKCLNRanks(comm)
                << ", numel: " << input.numel() << ", dtype: " << input.type()
                << ", sync_op: " << true << ", use_calc_stream: " << false;
        int r =
            bkcl_all_gather(comm,
                            input.data(),
                            input.numel(),
                            output->data(),
                            platform::ToBKCLDataType(
                                framework::TransToProtoVarType(input.type())),
                            stream);
        return r;
      },
      CommType::ALLGATHER,
      /*sync_op*/ true,
      /*use_calc_stream*/ false);
}

std::shared_ptr<ProcessGroup::Task> ProcessGroupBKCL::AllGather(
    std::vector<phi::DenseTensor>& in_tensors,
    std::vector<phi::DenseTensor>& out_tensors,
    bool sync_op) {
  PADDLE_ENFORCE_EQ(
      in_tensors.size(),
      1,
      platform::errors::InvalidArgument(
          "BKCL only support single tensor collective communication."));
  PADDLE_ENFORCE_EQ(
      out_tensors.size(),
      1,
      platform::errors::InvalidArgument(
          "BKCL only support single tensor collective communication."));
  PADDLE_ENFORCE_EQ(
      CheckTensorsInXPUPlace(out_tensors),
      true,
      platform::errors::InvalidArgument("All outputs should be in XPUPlace."));
  return Collective(
      &out_tensors[0],
      in_tensors[0],
      [&](phi::DenseTensor* output,
          const phi::DenseTensor& input,
          BKCLContext_t comm,
          const XPUStream& stream) {
        VLOG(3) << "calling bkcl_all_gather"
                << ", rank_id: " << platform::GetBKCLRankID(comm)
                << ", dev_id: " << platform::GetBKCLDevID(comm)
                << ", nranks: " << platform::GetBKCLNRanks(comm)
                << ", numel: " << input.numel() << ", dtype: " << input.type()
                << ", sync_op: " << sync_op << ", use_calc_stream: " << false;
        int r =
            bkcl_all_gather(comm,
                            input.data(),
                            input.numel(),
                            output->data(),
                            platform::ToBKCLDataType(
                                framework::TransToProtoVarType(input.type())),
                            stream);
        return r;
      },
      CommType::ALLGATHER,
      sync_op,
      /*use_calc_stream*/ false);
}

std::shared_ptr<ProcessGroupBKCL> ProcessGroupBKCL::CreateProcessGroupBKCL(
    const std::shared_ptr<phi::distributed::Store>& store,
    int rank,
    int size,
    int gid) {
  auto process_group =
      std::make_shared<ProcessGroupBKCL>(store, rank, size, gid);
  ProcessGroupIdMap::GetInstance().emplace(gid, process_group);
  return process_group;
}

}  //  namespace distributed
}  //  namespace paddle
