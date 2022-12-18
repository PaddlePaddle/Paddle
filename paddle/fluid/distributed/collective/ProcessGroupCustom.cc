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

#include "paddle/fluid/distributed/collective/ProcessGroupCustom.h"

#include "paddle/fluid/distributed/collective/CustomCCLTools.h"
#include "paddle/fluid/distributed/collective/common.h"
#include "paddle/fluid/memory/malloc.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/place.h"
#include "paddle/phi/common/place.h"

DECLARE_bool(xccl_blocking_wait);

constexpr int64_t kWaitBlockTImeout = 10;

namespace paddle {
namespace distributed {

void SyncDefaultStream(
    const std::vector<Place>& places,
    std::vector<CustomEventManager>& cclEvents,                    // NOLINT
    std::vector<std::unique_ptr<CustomDeviceContext>>& dev_ctx) {  // NOLINT
  for (size_t i = 0; i < places.size(); ++i) {
    auto* default_ctx = static_cast<platform::CustomDeviceContext*>(
        platform::DeviceContextPool::Instance().Get(places[i]));
    cclEvents[i].Record(*dev_ctx[i]);
    cclEvents[i].Block(*default_ctx);
  }
}

std::shared_ptr<ProcessGroupCustom::CustomTask> ProcessGroupCustom::CreateTask(
    std::vector<Place> places,
    int rank,
    CommType comm_type,
    const std::vector<phi::DenseTensor>& inputs) {
  return std::make_shared<ProcessGroupCustom::CustomTask>(
      places, rank, comm_type, inputs);
}

ProcessGroupCustom::CustomTask::CustomTask(
    const std::vector<Place>& places,
    int rank,
    CommType CommType,
    const std::vector<phi::DenseTensor>& inputs)
    : Task(rank, inputs, CommType), places_(places) {
  control_events_.resize(places.size());
  cclComms_.resize(places.size());
}

ProcessGroupCustom::CustomTask::~CustomTask() {}

void ProcessGroupCustom::CustomTask::SetOutputs(
    std::vector<phi::DenseTensor>& outputs) {  // NOLINT
  outputs_ = std::make_shared<std::vector<phi::DenseTensor>>(outputs);
}

void ProcessGroupCustom::CustomTask::SynchronizeStreams() {
  for (size_t i = 0; i < places_.size(); ++i) {
    auto* default_ctx = static_cast<platform::CustomDeviceContext*>(
        platform::DeviceContextPool::Instance().Get(places_[i]));
    phi::DeviceGuard guard(default_ctx->GetPlace());
    phi::stream::Stream stream(default_ctx->GetPlace(), default_ctx->stream());
    stream.WaitEvent(control_events_[i].GetCustomEvent());
  }
}

bool ProcessGroupCustom::CustomTask::IsCompleted() {
  for (size_t i = 0; i < places_.size(); ++i) {
    if (!control_events_[i].Query()) {
      return false;
    }
  }

  return true;
}

bool ProcessGroupCustom::CustomTask::Wait(std::chrono::milliseconds timeout) {
  SynchronizeStreams();
  while (!IsCompleted()) {
    std::this_thread::sleep_for(std::chrono::milliseconds(kWaitBlockTImeout));
  }
  return true;
}

// Same as Wait
void ProcessGroupCustom::CustomTask::Synchronize() { Wait(kWaitTimeout); }

ProcessGroupCustom::ProcessGroupCustom(const std::shared_ptr<Store>& store,
                                       const std::string& device_type,
                                       int rank,
                                       int size,
                                       int gid)
    : ProcessGroup(rank, size, gid), store_(store), device_type_(device_type) {}

void ProcessGroupCustom::BroadcastUniqueCustomID(
    std::vector<phi::ccl::CCLRootId>& ccl_ids) {  // NOLINT
  if (rank_ == 0) {
    for (size_t i = 0; i < ccl_ids.size(); i++) {
      auto key = "ProcessGroupCustom/ccl_ids/" + std::to_string(i);
      store_->set(key, ccl_ids[i]);
    }
  } else {
    for (size_t i = 0; i < ccl_ids.size(); i++) {
      auto key = "ProcessGroupCustom/ccl_ids/" + std::to_string(i);
      ccl_ids[i] = store_->get(key);
    }
  }
}

// create CustomCCLManager cache for places_key
void ProcessGroupCustom::CreateCustomManagerCache(
    const std::string& places_key, const std::vector<Place>& places) {
  PADDLE_ENFORCE_EQ(places_key.empty(),
                    false,
                    platform::errors::PreconditionNotMet(
                        "Not able to create/get the HCCL Communicator since "
                        "the NPU place are not known"));
  const std::string device_type = places.back().GetDeviceType();

  std::vector<std::shared_ptr<CustomCCLCommManager>> ccl_comms;
  ccl_comms.resize(places.size());

  // using vector just for broadcast
  std::vector<phi::ccl::CCLRootId> ccl_ids;
  ccl_ids.resize(1);
  auto& ccl_id = ccl_ids.front();

  if (rank_ == 0) {
    phi::DeviceManager::CCLGetUniqueId(device_type, &ccl_id);
  }
  BroadcastUniqueCustomID(ccl_ids);

  VLOG(3) << "init custom ccl rank: " << rank_ << ", nranks: " << size_
          << ", place: " << places_key
          << ", custom ccl uniqueid: " << SerializeCustomCCLUniqueId(ccl_id);

  std::vector<std::unique_ptr<CustomDeviceContext>> dev_ctx;
  dev_ctx.resize(places.size());

  std::unique_ptr<phi::ccl::CCLComm> comms(
      new phi::ccl::CCLComm[places.size()]);
  for (size_t i = 0; i < places.size(); ++i) {
    phi::DeviceGuard guard(places[i]);
    ccl_comms[i] = CustomCCLCommManager::Create(
        device_type, GetSize(), GetRank(), &ccl_id, comms.get() + i);
    dev_ctx[i].reset(new CustomDeviceContext(places[i]));
  }

  std::vector<CustomEventManager> events;
  events.resize(places.size());

  // These caches will be useful to process sync/wait/communicate
  places_to_events_.emplace(places_key, std::move(events));
  places_to_customcomm_.emplace(places_key, std::move(ccl_comms));
  places_to_ctx_.emplace(places_key, std::move(dev_ctx));
}

template <typename Fn>
std::shared_ptr<ProcessGroup::Task> ProcessGroupCustom::Collective(
    std::vector<phi::DenseTensor>& inputs,
    std::vector<phi::DenseTensor>& outputs,
    Fn fn,
    CommType op_type) {
  const auto places = GetPlaceList(inputs);
  const auto key = GetKeyFromPlaces(places);

  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (places_to_customcomm_.find(key) == places_to_customcomm_.end()) {
      CreateCustomManagerCache(key, places);
    }
  }

  auto& ccl_comms = places_to_customcomm_[key];
  SyncDefaultStream(places, places_to_events_[key], places_to_ctx_[key]);
  auto task = CreateTask(places, rank_, op_type, inputs);
  task->SetOutputs(outputs);

  for (size_t i = 0; i < inputs.size(); ++i) {
    phi::DeviceGuard guard(places[i]);
    const auto& ccl_stream = places_to_ctx_[key][i]->stream();
    phi::stream::Stream stream(places[i], ccl_stream);
    fn(inputs[i], outputs[i], ccl_comms[i]->GetCustomCCLComm(), stream);
  }

  for (size_t i = 0; i < inputs.size(); ++i) {
    phi::DeviceGuard guard(places[i]);
    task->control_events_[i].Record(*places_to_ctx_[key][i]);
  }
  return task;
}

void* XcclGetPointerByOffset(void* raw_pointer,
                             size_t offset,
                             experimental::DataType type) {
  if (type == experimental::DataType::FLOAT32) {
    return reinterpret_cast<void*>(reinterpret_cast<float*>(raw_pointer) +
                                   offset);
  } else if (type == experimental::DataType::FLOAT64) {
    return reinterpret_cast<void*>(reinterpret_cast<double*>(raw_pointer) +
                                   offset);
  } else if (type == experimental::DataType::INT32) {
    return reinterpret_cast<void*>(reinterpret_cast<int32_t*>(raw_pointer) +
                                   offset);
  } else if (type == experimental::DataType::INT64) {
    return reinterpret_cast<void*>(reinterpret_cast<int64_t*>(raw_pointer) +
                                   offset);
  } else if (type == experimental::DataType::FLOAT16) {
    return reinterpret_cast<void*>(reinterpret_cast<int16_t*>(raw_pointer) +
                                   offset);
  } else {
    PADDLE_THROW(platform::errors::Unimplemented(
        "This datatype in xccl is not supported."));
  }
  return nullptr;
}

std::shared_ptr<ProcessGroup::Task> ProcessGroupCustom::AllGather(
    phi::DenseTensor* out_tensor,
    const phi::DenseTensor& in_tensor,
    int64_t offset,
    int64_t numel,
    bool sync_op  // for compatibility, no use now
) {
  std::vector<phi::DenseTensor> in_wrapper{in_tensor};
  std::vector<phi::DenseTensor> out_wrapper{*out_tensor};
  return Collective(
      in_wrapper,
      out_wrapper,
      [&](phi::DenseTensor& input,
          phi::DenseTensor& output,
          phi::ccl::CCLComm comm,
          const phi::stream::Stream& stream) {
        return phi::DeviceManager::CCLAllGather(
            device_type_,
            XcclGetPointerByOffset(input.data(), offset, input.dtype()),
            output.data(),
            numel,
            phi::ccl::ToCCLDataType(input.dtype()),
            comm,
            stream);
      },
      CommType::ALLGATHER);
}

std::shared_ptr<ProcessGroup::Task> ProcessGroupCustom::AllReduce(
    phi::DenseTensor* out_tensor,
    const phi::DenseTensor& in_tensor,
    const AllreduceOptions& opts,
    bool sync_op  // for compatibility, no use now
) {
  std::vector<phi::DenseTensor> in_wrapper{in_tensor};
  std::vector<phi::DenseTensor> out_wrapper{*out_tensor};
  return AllReduce(in_wrapper, out_wrapper, opts);
}

std::shared_ptr<ProcessGroup::Task> ProcessGroupCustom::Broadcast(
    phi::DenseTensor* out_tensor,
    const phi::DenseTensor& in_tensor,
    const BroadcastOptions& opts,
    bool sync_op  // for compatibility, no use now
) {
  std::vector<phi::DenseTensor> in_wrapper{in_tensor};
  std::vector<phi::DenseTensor> out_wrapper{*out_tensor};
  return Broadcast(in_wrapper, out_wrapper, opts);
}

std::shared_ptr<ProcessGroup::Task> ProcessGroupCustom::Barrier(
    const BarrierOptions& opts) {
  // Only support single card single process
  PADDLE_ENFORCE_GE(opts.device_id,
                    0,
                    platform::errors::PreconditionNotMet(
                        "The barrier device id must greater or equal than 0."));
  platform::CustomPlace place(device_type_, opts.device_id);
  auto allocator = std::unique_ptr<phi::Allocator>(
      new paddle::experimental::DefaultAllocator(place));
  phi::DenseTensorMeta meta(phi::DataType::FLOAT32, phi::DDim{1});
  phi::DenseTensor barrier_tensor{allocator.get(), meta};

  auto task = ProcessGroupCustom::AllReduce(&barrier_tensor,
                                            barrier_tensor,
                                            {},
                                            /*sync_op*/ true);
  auto xccl_task = dynamic_cast<ProcessGroupCustom::CustomTask*>(task.get());
  xccl_task->barrierTensors_ = {barrier_tensor};
  return task;
}

phi::DeviceContext* ProcessGroupCustom::GetDeviceContext(
    const Place& place) const {
  const std::string key = GetKeyFromPlace(place);
  const auto& iter = places_to_ctx_.find(key);
  PADDLE_ENFORCE_NE(
      iter,
      places_to_ctx_.end(),
      platform::errors::NotFound(
          "Cannot find the device context in this process group."));
  return iter->second[0].get();
}

phi::ccl::CCLComm ProcessGroupCustom::CustomCCLComm(const Place& place) const {
  std::vector<Place> places = {place};
  const auto& iter = places_to_customcomm_.find(GetKeyFromPlaces(places));
  PADDLE_ENFORCE_NE(iter,
                    places_to_customcomm_.end(),
                    platform::errors::InvalidArgument(
                        "Cannot find nccl comm in process group."));
  return iter->second[0]->GetCustomCCLComm();
}

// TODO(sunyilun): methods below will be removed later
std::shared_ptr<ProcessGroup::Task> ProcessGroupCustom::AllGather(
    std::vector<phi::DenseTensor>& in_tensors,
    std::vector<phi::DenseTensor>& out_tensors) {
  PADDLE_ENFORCE_EQ(
      CheckTensorsInCustomPlace(in_tensors, device_type_),
      true,
      platform::errors::InvalidArgument(
          "All inputs should be in CustomPlace(%s).", device_type_));
  PADDLE_ENFORCE_EQ(
      CheckTensorsInCustomPlace(out_tensors, device_type_),
      true,
      platform::errors::InvalidArgument(
          "All outputs should be in CustomPlace(%s).", device_type_));
  return Collective(
      in_tensors,
      out_tensors,
      [&](phi::DenseTensor& input,
          phi::DenseTensor& output,
          phi::ccl::CCLComm comm,
          const phi::stream::Stream& stream) {
        return phi::DeviceManager::CCLAllGather(
            device_type_,
            input.data(),
            output.data(),
            input.numel(),
            phi::ccl::ToCCLDataType(input.dtype()),
            comm,
            stream);
      },
      CommType::ALLGATHER);
}

std::shared_ptr<ProcessGroup::Task> ProcessGroupCustom::AllReduce(
    std::vector<phi::DenseTensor>& in_tensors,   // NOLINT
    std::vector<phi::DenseTensor>& out_tensors,  // NOLINT
    const AllreduceOptions& opts) {
  PADDLE_ENFORCE_EQ(
      CheckTensorsInCustomPlace(in_tensors, device_type_),
      true,
      platform::errors::InvalidArgument(
          "All inputs should be in CustomPlace(%s).", device_type_));
  PADDLE_ENFORCE_EQ(
      CheckTensorsInCustomPlace(out_tensors, device_type_),
      true,
      platform::errors::InvalidArgument(
          "All outputs should be in CustomPlace(%s).", device_type_));
  return Collective(
      in_tensors,
      out_tensors,
      [&](phi::DenseTensor& input,
          phi::DenseTensor& output,
          phi::ccl::CCLComm comm,
          const phi::stream::Stream& stream) {
        return phi::DeviceManager::CCLAllReduce(
            device_type_,
            input.data(),
            output.data(),
            input.numel(),
            phi::ccl::ToCCLDataType(input.dtype()),
            ToCustomCCLRedType(opts.reduce_op),
            comm,
            stream);
      },
      CommType::ALLREDUCE);
}

std::shared_ptr<ProcessGroup::Task> ProcessGroupCustom::Broadcast(
    std::vector<phi::DenseTensor>& in_tensors,   // NOLINT
    std::vector<phi::DenseTensor>& out_tensors,  // NOLINT
    const BroadcastOptions& opts) {
  PADDLE_ENFORCE_EQ(
      CheckTensorsInCustomPlace(in_tensors, device_type_),
      true,
      platform::errors::InvalidArgument(
          "All inputs should be in CustomPlace(%s).", device_type_));
  PADDLE_ENFORCE_EQ(
      CheckTensorsInCustomPlace(out_tensors, device_type_),
      true,
      platform::errors::InvalidArgument(
          "All outputs should be in CustomPlace(%s).", device_type_));
  return Collective(
      in_tensors,
      out_tensors,
      [&](phi::DenseTensor& input,
          phi::DenseTensor& output,
          phi::ccl::CCLComm comm,
          const phi::stream::Stream& stream) {
        int root = opts.source_rank * in_tensors.size() + opts.source_root;
        if (rank_ == root) {
          return phi::DeviceManager::CCLBroadcast(
              device_type_,
              input.data(),
              input.numel(),
              phi::ccl::ToCCLDataType(input.dtype()),
              root,
              comm,
              stream);
        } else {
          return phi::DeviceManager::CCLBroadcast(
              device_type_,
              output.data(),
              output.numel(),
              phi::ccl::ToCCLDataType(output.dtype()),
              root,
              comm,
              stream);
        }
      },
      CommType::BROADCAST);
}

std::shared_ptr<ProcessGroupCustom>
ProcessGroupCustom::CreateProcessGroupCustom(
    const std::shared_ptr<Store>& store,
    const std::string& device_type,
    int rank,
    int size,
    int gid) {
  auto process_group =
      std::make_shared<ProcessGroupCustom>(store, device_type, rank, size, gid);
  ProcessGroupIdMap::GetInstance().emplace(gid, process_group);
  return process_group;
}

}  //  namespace distributed
}  //  namespace paddle
