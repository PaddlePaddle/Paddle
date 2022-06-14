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

#include "paddle/fluid/distributed/collective/ProcessGroupHCCL.h"
#include "paddle/fluid/distributed/collective/Common.h"
#include "paddle/fluid/distributed/collective/HCCLTools.h"
#include "paddle/fluid/memory/malloc.h"
#include "paddle/fluid/platform/device/npu/hccl_helper.h"
#include "paddle/fluid/platform/device/npu/npu_info.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/place.h"
#include "paddle/phi/api/include/api.h"
#include "paddle/phi/common/place.h"

DECLARE_bool(hccl_blocking_wait);
// DECLARE_bool(use_stream_safe_npu_allocator);

constexpr int64_t kWaitBlockTImeout = 10;

namespace paddle {
namespace distributed {

void SyncDefaultStream(
    const std::vector<Place>& places,
    std::vector<NPUEventManager>& hcclEvents,                   // NOLINT
    std::vector<std::unique_ptr<NPUDeviceContext>>& dev_ctx) {  // NOLINT
  for (size_t i = 0; i < places.size(); ++i) {
    auto* default_ctx = static_cast<platform::NPUDeviceContext*>(
        platform::DeviceContextPool::Instance().Get(places[i]));
    hcclEvents[i].Record(*dev_ctx[i]);
    hcclEvents[i].Block(*default_ctx);
  }
}

std::shared_ptr<ProcessGroupHCCL::HCCLTask> ProcessGroupHCCL::CreateTask(
    std::vector<Place> places, int rank, CommType comm_type,
    const std::vector<phi::DenseTensor>& inputs) {
  return std::make_shared<ProcessGroupHCCL::HCCLTask>(places, rank, comm_type,
                                                      inputs);
}

ProcessGroupHCCL::HCCLTask::HCCLTask(
    const std::vector<Place>& places, int rank, CommType CommType,
    const std::vector<phi::DenseTensor>& inputs)
    : Task(rank, inputs, CommType), places_(places) {
  control_events_.resize(places.size());
  hcclComms_.resize(places.size());
}

ProcessGroupHCCL::HCCLTask::~HCCLTask() {}

void ProcessGroupHCCL::HCCLTask::SetOutputs(
    std::vector<phi::DenseTensor>& outputs) {  // NOLINT
  outputs_ = std::make_shared<std::vector<phi::DenseTensor>>(outputs);
}

void ProcessGroupHCCL::HCCLTask::SynchronizeStreams() {
  for (size_t i = 0; i < places_.size(); ++i) {
    auto* default_ctx = static_cast<platform::NPUDeviceContext*>(
        platform::DeviceContextPool::Instance().Get(places_[i]));
    platform::NPUStreamWaitEvent(default_ctx->stream(),
                                 control_events_[i].GetRawNPUEvent());
  }
}

bool ProcessGroupHCCL::HCCLTask::IsCompleted() {
  for (size_t i = 0; i < places_.size(); ++i) {
    if (!control_events_[i].Query()) {
      return false;
    }
  }

  return true;
}

// TODO(sandyhouse): Add timeout for wait, now timeout unused
bool ProcessGroupHCCL::HCCLTask::Wait(std::chrono::milliseconds timeout) {
  SynchronizeStreams();
  // NOTE(sandyhouse): It will block host for sync
  while (!IsCompleted()) {
    std::this_thread::sleep_for(std::chrono::milliseconds(kWaitBlockTImeout));
  }
  return true;
}

// Same as Wait
void ProcessGroupHCCL::HCCLTask::Synchronize() { Wait(kWaitTimeout); }

ProcessGroupHCCL::ProcessGroupHCCL(const std::shared_ptr<Store>& store,
                                   int rank, int size,
                                   const platform::Place& place, int gid)
    : ProcessGroup(rank, size, place, gid), store_(store) {
  platform::SetNPUDeviceId(place_.device);
}

void ProcessGroupHCCL::BroadcastUniqueHCCLID(
    std::vector<HcclRootInfo>& hccl_ids) {  // NOLINT
  if (rank_ == 0) {
    for (size_t i = 0; i < hccl_ids.size(); i++) {
      auto key = "ProcessGroupHCCL/hccl_ids/" + std::to_string(i);
      auto hccl_id = std::vector<uint8_t>(
          reinterpret_cast<uint8_t*>(&hccl_ids[i]),
          reinterpret_cast<uint8_t*>(&hccl_ids[i]) + sizeof(HcclRootInfo));
      store_->set(key, hccl_id);
    }
  } else {
    for (size_t i = 0; i < hccl_ids.size(); i++) {
      auto key = "ProcessGroupHCCL/hccl_ids/" + std::to_string(i);
      auto ret = store_->get(key);
      std::memcpy(&hccl_ids[i], ret.data(), ret.size());
    }
  }
}

// create HCCLManager cache for places_key
void ProcessGroupHCCL::CreateHCCLManagerCache(
    const std::string& places_key, const std::vector<Place>& places) {
  PADDLE_ENFORCE_EQ(places_key.empty(), false,
                    platform::errors::PreconditionNotMet(
                        "Not able to create/get the HCCL Communicator since "
                        "the NPU place are not known"));

  std::vector<std::shared_ptr<HCCLCommManager>> hccl_comms;
  hccl_comms.resize(places.size());

  // using vector just for broadcast
  std::vector<HcclRootInfo> hccl_ids;
  hccl_ids.resize(1);
  auto& hccl_id = hccl_ids.front();

  if (rank_ == 0) {
    PADDLE_ENFORCE_NPU_SUCCESS(platform::dynload::HcclGetRootInfo(&hccl_id));
  }
  BroadcastUniqueHCCLID(hccl_ids);

  VLOG(3) << "init hccl rank: " << rank_ << ", nranks: " << size_
          << ", place: " << places_key
          << ", hccl uniqueid: " << SerializeHCCLUniqueId(hccl_id);

  std::vector<std::unique_ptr<NPUDeviceContext>> dev_ctx;
  dev_ctx.resize(places.size());

  std::unique_ptr<HcclComm[]> comms(new HcclComm[places.size()]);
  for (size_t i = 0; i < places.size(); ++i) {
    platform::NPUDeviceGuard guard(places[i].GetDeviceId());
    hccl_comms[i] = HCCLCommManager::Create(GetSize(), GetRank(), &hccl_id,
                                            comms.get() + i);
    dev_ctx[i].reset(new NPUDeviceContext(places[i]));
  }

  std::vector<NPUEventManager> events;
  events.resize(places.size());

  // These caches will be useful to process sync/wait/communicate
  places_to_events_.emplace(places_key, std::move(events));
  places_to_hcclcomm_.emplace(places_key, std::move(hccl_comms));
  places_to_ctx_.emplace(places_key, std::move(dev_ctx));
}

template <typename Fn>
std::shared_ptr<ProcessGroup::Task> ProcessGroupHCCL::Collective(
    std::vector<phi::DenseTensor>& inputs,
    std::vector<phi::DenseTensor>& outputs, Fn fn, CommType op_type) {
  const auto places = GetPlaceList(inputs);
  const auto key = GetKeyFromPlaces(places);

  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (places_to_hcclcomm_.find(key) == places_to_hcclcomm_.end()) {
      CreateHCCLManagerCache(key, places);
    }
  }

  auto& hccl_comms = places_to_hcclcomm_[key];

  SyncDefaultStream(places, places_to_events_[key], places_to_ctx_[key]);

  auto task = CreateTask(places, rank_, op_type, inputs);
  task->SetOutputs(outputs);

  // if (FLAGS_use_stream_safe_npu_allocator) {
  //   for (size_t i = 0; i < inputs.size(); ++i) {
  //     platform::NPUDeviceGuard guard(places[i].GetDeviceId());
  //     auto dense_tensor =
  //         std::dynamic_pointer_cast<phi::DenseTensor>(inputs[i].impl());
  //     memory::RecordStream(dense_tensor->Holder(),
  //                          places_to_ctx_[key][i]->stream());
  //   }
  // }

  for (size_t i = 0; i < inputs.size(); ++i) {
    platform::NPUDeviceGuard guard(places[i].GetDeviceId());
    const auto& hccl_stream = places_to_ctx_[key][i]->stream();
    fn(inputs[i], outputs[i], hccl_comms[i]->GetHcclComm(), hccl_stream);
  }

  for (size_t i = 0; i < inputs.size(); ++i) {
    platform::NPUDeviceGuard guard(places[i].GetDeviceId());
    task->control_events_[i].Record(*places_to_ctx_[key][i]);
  }
  return task;
}

std::shared_ptr<ProcessGroup::Task> ProcessGroupHCCL::AllReduce(
    std::vector<phi::DenseTensor>& in_tensors,   // NOLINT
    std::vector<phi::DenseTensor>& out_tensors,  // NOLINT
    const AllreduceOptions& opts) {
  return Collective(in_tensors, out_tensors,
                    [&](phi::DenseTensor& input, phi::DenseTensor& output,
                        HcclComm comm, const aclrtStream& stream) {
                      return platform::dynload::HcclAllReduce(
                          input.data(), output.data(), input.numel(),
                          platform::ToHCCLDataType(input.dtype()),
                          ToHCCLRedType(opts.reduce_op), comm, stream);
                    },
                    CommType::ALLREDUCE);
}

std::shared_ptr<ProcessGroup::Task> ProcessGroupHCCL::Broadcast(
    std::vector<phi::DenseTensor>& in_tensors,   // NOLINT
    std::vector<phi::DenseTensor>& out_tensors,  // NOLINT
    const BroadcastOptions& opts) {
  // PADDLE_ENFORCE_EQ(
  //     CheckTensorsInNPUPlace(tensors), true,
  //     platform::errors::InvalidArgument("All inputs should be in
  //     CudaPlace."));

  return Collective(
      in_tensors, out_tensors,
      [&](phi::DenseTensor& input, phi::DenseTensor& output, HcclComm comm,
          const aclrtStream& stream) {
        int root = opts.source_rank * in_tensors.size() + opts.source_root;
        if (rank_ == root) {
          return platform::dynload::HcclBroadcast(
              input.data(), input.numel(),
              platform::ToHCCLDataType(input.dtype()), root, comm, stream);
        } else {
          return platform::dynload::HcclBroadcast(
              output.data(), output.numel(),
              platform::ToHCCLDataType(output.dtype()), root, comm, stream);
        }
      },
      CommType::BROADCAST);
}

}  //  namespace distributed
}  //  namespace paddle
