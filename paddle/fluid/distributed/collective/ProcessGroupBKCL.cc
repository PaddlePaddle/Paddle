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

#include "paddle/fluid/distributed/collective/ProcessGroupBKCL.h"

#include "paddle/fluid/distributed/collective/BKCLTools.h"
#include "paddle/fluid/distributed/collective/Common.h"
#include "paddle/fluid/platform/device/xpu/bkcl_helper.h"
#include "paddle/fluid/platform/device/xpu/xpu_info.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/place.h"
#include "paddle/phi/common/place.h"

namespace paddle {
namespace distributed {
using XPUDeviceContext = paddle::platform::XPUDeviceContext;

void SyncDefaultStream(
    const std::vector<Place>& places,
    std::vector<XPUEventManager>& BKCLEvents,             // NOLINT
    std::vector<std::unique_ptr<XPUContext>>& dev_ctx) {  // NOLINT
  for (size_t i = 0; i < places.size(); ++i) {
    auto* default_ctx = static_cast<XPUContext*>(
        platform::DeviceContextPool::Instance().Get(places[i]));
    BKCLEvents[i].Record(*default_ctx);
    BKCLEvents[i].Block(*dev_ctx[i]);
  }
}

std::shared_ptr<ProcessGroupBKCL::BKCLTask> ProcessGroupBKCL::CreateTask(
    std::vector<Place> places,
    int rank,
    CommType comm_type,
    const std::vector<phi::DenseTensor>& inputs) {
  return std::make_shared<ProcessGroupBKCL::BKCLTask>(
      places, rank, comm_type, inputs);
}

ProcessGroupBKCL::BKCLTask::BKCLTask(
    const std::vector<Place>& places,
    int rank,
    CommType CommType,
    const std::vector<phi::DenseTensor>& inputs)
    : TaskStream(rank, inputs, CommType), places_(places) {
  control_events_.resize(places.size());
}

ProcessGroupBKCL::BKCLTask::~BKCLTask() {}

void ProcessGroupBKCL::BKCLTask::SynchronizeStreams() {
  for (size_t i = 0; i < places_.size(); ++i) {
    auto* default_ctx = static_cast<XPUContext*>(
        platform::DeviceContextPool::Instance().Get(places_[i]));
    PADDLE_ENFORCE_XPU_SUCCESS(xpu_stream_wait_event(
        default_ctx->stream(), control_events_[i].GetRawXpuEvent()));
  }
}

bool ProcessGroupBKCL::BKCLTask::IsCompleted() {
  LOG_FIRST_N(WARNING, 1) << "XPU do not support event query now.";
  return true;
}

// TODO(sheniang03): Add timeout for wait, now timeout unused
bool ProcessGroupBKCL::BKCLTask::Wait(std::chrono::milliseconds timeout) {
  SynchronizeStreams();

  if (!barrierTensors_.empty()) {
    // If we use the work to do barrier, we should block cpu
    for (auto& place : places_) {
      platform::XPUDeviceGuard guard(place.GetDeviceId());
      xpu_wait();
    }
  }
  return true;
}

// Same as Wait
void ProcessGroupBKCL::BKCLTask::Synchronize() { Wait(kWaitTimeout); }

ProcessGroupBKCL::ProcessGroupBKCL(const std::shared_ptr<Store>& store,
                                   int rank,
                                   int size,
                                   const platform::Place& place,
                                   int gid)
    : ProcessGroupStream(rank, size, place, gid), store_(store) {
  platform::SetXPUDeviceId(place_.device);
}

void ProcessGroupBKCL::BroadcastUniqueBKCLID(
    std::vector<BKCLUniqueId>& bkcl_ids) {  // NOLINT
  if (rank_ == 0) {
    for (size_t i = 0; i < bkcl_ids.size(); i++) {
      auto key = "ProcessGroupBKCL/bkcl_ids/" + std::to_string(gid_) + "/" +
                 std::to_string(i);
      auto bkcl_id = std::vector<uint8_t>(
          reinterpret_cast<uint8_t*>(&bkcl_ids[i]),
          reinterpret_cast<uint8_t*>(&bkcl_ids[i]) + BKCL_UNIQUE_ID_BYTES);
      store_->set(key, bkcl_id);
    }
  } else {
    for (size_t i = 0; i < bkcl_ids.size(); i++) {
      auto key = "ProcessGroupBKCL/bkcl_ids/" + std::to_string(gid_) + "/" +
                 std::to_string(i);
      auto ret = store_->get(key);
      std::memcpy(&bkcl_ids[i], ret.data(), ret.size());
    }
  }
}

// create BKCLManager cache for places_key
void ProcessGroupBKCL::CreateBKCLManagerCache(
    const std::string& places_key, const std::vector<Place>& places) {
  PADDLE_ENFORCE_EQ(places_key.empty(),
                    false,
                    platform::errors::PreconditionNotMet(
                        "Not able to create/get the BKCL Communicator since "
                        "the XPU place are not known"));

  std::vector<std::shared_ptr<BKCLCommManager>> bkcl_comms;
  bkcl_comms.resize(places.size());

  // using vector just for broadcast
  std::vector<BKCLUniqueId> bkcl_ids;
  bkcl_ids.resize(1);
  auto& bkcl_id = bkcl_ids.front();

  if (rank_ == 0) {
    PADDLE_ENFORCE_XPU_SUCCESS(bkcl_get_unique_id(&bkcl_id));
  }
  BroadcastUniqueBKCLID(bkcl_ids);

  VLOG(3) << "init bkcl rank: " << rank_ << ", nranks: " << size_
          << ", place: " << places_key
          << ", bkcl uniqueid: " << SerializeBKCLUniqueId(bkcl_id);

  std::vector<std::unique_ptr<XPUContext>> dev_ctx;
  dev_ctx.resize(places.size());

  PADDLE_ENFORCE_XPU_SUCCESS(bkcl_group_start());
  for (size_t i = 0; i < places.size(); ++i) {
    platform::XPUDeviceGuard guard(places[i].GetDeviceId());
    bkcl_comms[i] = BKCLCommManager::Create(GetSize(), GetRank(), bkcl_id);
    XPUDeviceContext* tmp = new XPUDeviceContext(places[i]);
    dev_ctx[i].reset(static_cast<XPUContext*>(tmp));
  }
  PADDLE_ENFORCE_XPU_SUCCESS(bkcl_group_end());

  std::vector<XPUEventManager> events;
  events.resize(places.size());

  // These caches will be useful to process sync/wait/communicate
  places_to_events_.emplace(places_key, std::move(events));
  places_to_bkclcomm_.emplace(places_key, std::move(bkcl_comms));
  places_to_ctx_.emplace(places_key, std::move(dev_ctx));
}

template <typename Fn>
std::shared_ptr<ProcessGroup::Task> ProcessGroupBKCL::Collective(
    std::vector<phi::DenseTensor>& inputs,
    std::vector<phi::DenseTensor>& outputs,
    Fn fn,
    CommType op_type) {
  const auto places = GetPlaceList(inputs);
  const auto key = GetKeyFromPlaces(places);

  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (places_to_bkclcomm_.find(key) == places_to_bkclcomm_.end()) {
      CreateBKCLManagerCache(key, places);
    }
  }

  auto& bkcl_comms = places_to_bkclcomm_[key];

  SyncDefaultStream(places, places_to_events_[key], places_to_ctx_[key]);

  auto task = CreateTask(places, rank_, op_type, inputs);

  // construct uninitialize guard for device
  {
    platform::BKCLGroupGuard bkcl_guard;
    for (size_t i = 0; i < inputs.size(); ++i) {
      platform::XPUDeviceGuard xpu_guard(places[i].GetDeviceId());
      const auto& bkcl_stream = places_to_ctx_[key][i]->stream();
      fn(inputs[i], outputs[i], bkcl_comms[i]->GetBkclComm(), bkcl_stream);
    }
  }

  for (size_t i = 0; i < inputs.size(); ++i) {
    platform::XPUDeviceGuard xpu_guard(places[i].GetDeviceId());
    task->control_events_[i].Record(*places_to_ctx_[key][i]);
  }
  return task;
}

std::shared_ptr<ProcessGroup::Task> ProcessGroupBKCL::AllReduce(
    std::vector<phi::DenseTensor>& in_tensors,
    std::vector<phi::DenseTensor>& out_tensors,
    const AllreduceOptions& opts) {
  PADDLE_ENFORCE_EQ(
      CheckTensorsInXPUPlace(in_tensors),
      true,
      platform::errors::InvalidArgument("All inputs should be in XPUPlace."));
  return Collective(
      in_tensors,
      out_tensors,
      [&](const phi::DenseTensor& input,
          phi::DenseTensor& output,
          BKCLContext_t comm,
          const XPUStream& stream) {
        return bkcl_all_reduce(
            comm,
            input.data(),
            output.data(),
            input.numel(),
            platform::ToBKCLDataType(
                framework::TransToProtoVarType(input.type())),
            ToBKCLRedType(opts.reduce_op),
            stream);
      },
      CommType::ALLREDUCE);
}

std::shared_ptr<ProcessGroup::Task> ProcessGroupBKCL::AllReduce(
    std::vector<phi::DenseTensor>& in_tensors,
    std::vector<phi::DenseTensor>& out_tensors,
    const AllreduceOptions& opts,
    bool sync_op) {
  PADDLE_ENFORCE_EQ(
      CheckTensorsInXPUPlace(in_tensors),
      true,
      platform::errors::InvalidArgument("All inputs should be in XPUPlace."));
  return Collective(
      in_tensors,
      out_tensors,
      [&](const phi::DenseTensor& input,
          phi::DenseTensor& output,
          BKCLContext_t comm,
          const XPUStream& stream) {
        return bkcl_all_reduce(
            comm,
            input.data(),
            output.data(),
            input.numel(),
            platform::ToBKCLDataType(
                framework::TransToProtoVarType(input.type())),
            ToBKCLRedType(opts.reduce_op),
            stream);
      },
      CommType::ALLREDUCE);
}

std::shared_ptr<ProcessGroup::Task> ProcessGroupBKCL::Broadcast(
    std::vector<phi::DenseTensor>& in_tensors,
    std::vector<phi::DenseTensor>& out_tensors,
    const BroadcastOptions& opts) {
  PADDLE_ENFORCE_EQ(
      CheckTensorsInXPUPlace(in_tensors),
      true,
      platform::errors::InvalidArgument("All inputs should be in XPUPlace."));

  return Collective(
      in_tensors,
      out_tensors,
      [&](phi::DenseTensor& input,
          phi::DenseTensor& output,
          BKCLContext_t comm,
          const XPUStream& stream) {
        const auto root =
            opts.source_rank * in_tensors.size() + opts.source_root;
        return bkcl_broadcast(comm,
                              input.data(),
                              output.data(),
                              input.numel(),
                              platform::ToBKCLDataType(
                                  framework::TransToProtoVarType(input.type())),
                              root,
                              stream);
      },
      CommType::BROADCAST);
}

std::shared_ptr<ProcessGroup::Task> ProcessGroupBKCL::Broadcast(
    std::vector<phi::DenseTensor>& in_tensors,
    std::vector<phi::DenseTensor>& out_tensors,
    const BroadcastOptions& opts,
    bool sync_op) {
  PADDLE_ENFORCE_EQ(
      CheckTensorsInXPUPlace(in_tensors),
      true,
      platform::errors::InvalidArgument("All inputs should be in XPUPlace."));

  return Collective(
      in_tensors,
      out_tensors,
      [&](phi::DenseTensor& input,
          phi::DenseTensor& output,
          BKCLContext_t comm,
          const XPUStream& stream) {
        const auto root =
            opts.source_rank * in_tensors.size() + opts.source_root;
        return bkcl_broadcast(comm,
                              input.data(),
                              output.data(),
                              input.numel(),
                              platform::ToBKCLDataType(
                                  framework::TransToProtoVarType(input.type())),
                              root,
                              stream);
      },
      CommType::BROADCAST);
}

std::shared_ptr<ProcessGroup::Task> ProcessGroupBKCL::AllGather(
    std::vector<phi::DenseTensor>& in_tensors,
    std::vector<phi::DenseTensor>& out_tensors) {
  PADDLE_ENFORCE_EQ(
      CheckTensorsInXPUPlace(in_tensors),
      true,
      platform::errors::InvalidArgument("All inputs should be in XPUPlace."));
  PADDLE_ENFORCE_EQ(
      CheckTensorsInXPUPlace(out_tensors),
      true,
      platform::errors::InvalidArgument("All outputs should be in XPUPlace."));
  return Collective(
      in_tensors,
      out_tensors,
      [&](const phi::DenseTensor& input,
          phi::DenseTensor& output,
          BKCLContext_t comm,
          const XPUStream& stream) {
        return bkcl_all_gather(
            comm,
            input.data(),
            input.numel(),
            output.data(),
            platform::ToBKCLDataType(
                framework::TransToProtoVarType(input.type())),
            stream);
      },
      CommType::ALLGATHER);
}

std::shared_ptr<ProcessGroup::Task> ProcessGroupBKCL::AllGather(
    std::vector<phi::DenseTensor>& in_tensors,
    std::vector<phi::DenseTensor>& out_tensors,
    bool sync_op) {
  PADDLE_ENFORCE_EQ(
      CheckTensorsInXPUPlace(in_tensors),
      true,
      platform::errors::InvalidArgument("All inputs should be in XPUPlace."));
  PADDLE_ENFORCE_EQ(
      CheckTensorsInXPUPlace(out_tensors),
      true,
      platform::errors::InvalidArgument("All outputs should be in XPUPlace."));
  return Collective(
      in_tensors,
      out_tensors,
      [&](const phi::DenseTensor& input,
          phi::DenseTensor& output,
          BKCLContext_t comm,
          const XPUStream& stream) {
        return bkcl_all_gather(
            comm,
            input.data(),
            input.numel(),
            output.data(),
            platform::ToBKCLDataType(
                framework::TransToProtoVarType(input.type())),
            stream);
      },
      CommType::ALLGATHER);
}

std::shared_ptr<ProcessGroup::Task> ProcessGroupBKCL::Barrier(
    const BarrierOptions& opts) {
  // Only support single card single process
  std::vector<phi::XPUPlace> places = {place_};

  std::vector<phi::DenseTensor> barrierTensors;
  barrierTensors.reserve(places.size());

  for (auto& place : places) {
    platform::XPUDeviceGuard guard(place.GetDeviceId());
    phi::DenseTensorMeta meta(phi::DataType::FLOAT32, phi::DDim({1}));
    auto allocator = std::unique_ptr<phi::Allocator>(
        new paddle::experimental::DefaultAllocator(place));
    barrierTensors.emplace_back(allocator.get(), meta);
  }
  auto task = ProcessGroupBKCL::AllReduce(
      barrierTensors, barrierTensors, AllreduceOptions());
  auto bkcl_task = dynamic_cast<ProcessGroupBKCL::BKCLTask*>(task.get());
  bkcl_task->barrierTensors_ = std::move(barrierTensors);
  return task;
}

const phi::DeviceContext& ProcessGroupBKCL::GetDeviceContext(
    const Place& place) const {
  std::vector<Place> places = {place};
  const auto& iter = places_to_ctx_.find(GetKeyFromPlaces(places));
  PADDLE_ENFORCE_NE(iter,
                    places_to_ctx_.end(),
                    platform::errors::InvalidArgument(
                        "Cannot find device context in process group."));
  return *iter->second[0];
}

}  //  namespace distributed
}  //  namespace paddle
