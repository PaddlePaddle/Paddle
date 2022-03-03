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

#include "paddle/fluid/distributed/collective/ProcessGroupNCCL.h"
#include "paddle/fluid/platform/device/gpu/nccl_helper.h"
#include "paddle/fluid/platform/place.h"
#include "paddle/phi/api/include/api.h"
#include "paddle/phi/common/place.h"

DECLARE_bool(nccl_blocking_wait);
DECLARE_bool(use_stream_safe_cuda_allocator);

constexpr int64_t kWaitBlockTImeout = 10;

namespace paddle {
namespace distributed {

static ncclRedOp_t ToNCCLRedType(ReduceOp reduction) {
  static const std::map<ReduceOp, ncclRedOp_t> red_type = {
      {ReduceOp::MIN, ncclMin},
      {ReduceOp::MAX, ncclMax},
      {ReduceOp::SUM, ncclSum},
      {ReduceOp::PRODUCT, ncclProd},
  };
  auto it = red_type.find(reduction);
  PADDLE_ENFORCE_EQ(it != red_type.end(), true,
                    platform::errors::InvalidArgument(
                        "Invalid nccl reduction. Must be ncclMin | ncclMax | "
                        "ncclProd | ncclSum"));
  return it->second;
}

std::string SerializeNCCLUniqueId(const ncclUniqueId& ncclID) {
  const uint8_t* bytes = reinterpret_cast<const uint8_t*>(&ncclID);
  std::ostringstream oss;
  for (auto i = 0; i < NCCL_UNIQUE_ID_BYTES; ++i) {
    oss << std::hex << static_cast<int>(bytes[i]);
  }
  return oss.str();
}

// Get the list of devices from list of tensors
std::vector<Place> GetPlaceList(const std::vector<Tensor>& tensors) {
  std::vector<Place> places;
  places.reserve(tensors.size());
  for (auto& tensor : tensors) {
    places.push_back(tensor.inner_place());
  }
  return places;
}

// Get the deviceList String from the list of devices
std::string GetKeyFromPlaces(const std::vector<Place>& places) {
  std::string placeList;
  for (auto& place : places) {
    std::stringstream tmp;
    tmp << place;
    if (placeList.empty()) {
      placeList += tmp.str();
    } else {
      placeList += "," + tmp.str();
    }
  }
  return placeList;
}

bool CheckTensorsInCudaPlace(const std::vector<Tensor>& tensors) {
  return std::all_of(tensors.cbegin(), tensors.cend(), [&](const Tensor& t) {
    return t.place() == PlaceType::kGPU;
  });
}

void SyncDefaultStream(
    const std::vector<Place>& places,
    std::vector<EventManager>& ncclEvents,                       // NOLINT
    std::vector<std::unique_ptr<CUDADeviceContext>>& dev_ctx) {  // NOLINT
  for (size_t i = 0; i < places.size(); ++i) {
    auto* default_ctx = static_cast<platform::CUDADeviceContext*>(
        platform::DeviceContextPool::Instance().Get(places[i]));
    ncclEvents[i].Record(*dev_ctx[i]);
    ncclEvents[i].Block(*default_ctx);
  }
}

std::shared_ptr<ProcessGroupNCCL::NCCLTask> ProcessGroupNCCL::CreateTask(
    std::vector<Place> places, int rank, CommType comm_type,
    const std::vector<Tensor>& inputs) {
  return std::make_shared<ProcessGroupNCCL::NCCLTask>(places, rank, comm_type,
                                                      inputs);
}

ProcessGroupNCCL::NCCLTask::NCCLTask(const std::vector<Place>& places, int rank,
                                     CommType CommType,
                                     const std::vector<Tensor>& inputs)
    : Task(rank, inputs, CommType), places_(places) {
  control_events_.resize(places.size());
  ncclComms_.resize(places.size());
}

ProcessGroupNCCL::NCCLTask::~NCCLTask() {}

void ProcessGroupNCCL::NCCLTask::SetOutputs(
    std::vector<Tensor>& outputs) {  // NOLINT
  outputs_ = std::make_shared<std::vector<Tensor>>(outputs);
}

void ProcessGroupNCCL::NCCLTask::SynchronizeStreams() {
  for (size_t i = 0; i < places_.size(); ++i) {
    auto* default_ctx = static_cast<platform::CUDADeviceContext*>(
        platform::DeviceContextPool::Instance().Get(places_[i]));
    default_ctx->WaitEvent(control_events_[i].GetRawCudaEvent());
  }
}

bool ProcessGroupNCCL::NCCLTask::IsCompleted() {
  for (size_t i = 0; i < places_.size(); ++i) {
    if (!control_events_[i].Query()) {
      return false;
    }
  }

  return true;
}

// TODO(sheniang03): Add timeout for wait, now timeout unused
bool ProcessGroupNCCL::NCCLTask::Wait(std::chrono::milliseconds timeout) {
  SynchronizeStreams();
  if (FLAGS_nccl_blocking_wait) {
    // NOTE(shenliang03): It will block host for sync
    while (!IsCompleted()) {
      std::this_thread::sleep_for(std::chrono::milliseconds(kWaitBlockTImeout));
    }
  }

  if (!barrierTensors_.empty()) {
    // If we use the work to do barrier, we should block cpu
    for (auto& place : places_) {
      platform::CUDADeviceGuard gpuGuard(place);
      PADDLE_ENFORCE_GPU_SUCCESS(cudaDeviceSynchronize());
    }
  }
  return true;
}

// Same as Wait
void ProcessGroupNCCL::NCCLTask::Synchronize() { Wait(kWaitTimeout); }

ProcessGroupNCCL::ProcessGroupNCCL(const ProcessGroupStrategy& strategy,
                                   int rank, int size)
    : ProcessGroup(rank, size), strategy_(strategy) {}

void ProcessGroupNCCL::BcastNCCLId(
    std::vector<ncclUniqueId>& nccl_ids,  // NOLINT
    int root, int server_fd) {
  if (strategy_.local_rank_ == root) {
    std::vector<std::string> other_trainers;
    for (auto& ep : strategy_.trainer_endpoints_) {
      if (ep != strategy_.current_endpoint_) {
        other_trainers.push_back(ep);
      }
    }
    platform::SendBroadCastCommID(other_trainers, &nccl_ids);
  } else {
    platform::RecvBroadCastCommID(server_fd, strategy_.current_endpoint_,
                                  &nccl_ids);
  }
}

void ProcessGroupNCCL::BroadcastUniqueNCCLID(
    std::vector<ncclUniqueId>& nccl_ids) {  // NOLINT

  int server_fd = -1;
  if (rank_ != 0) {
    server_fd = platform::SocketServer::GetInstance(strategy_.current_endpoint_)
                    .socket();
  }
  BcastNCCLId(nccl_ids, 0, server_fd);
}

// create NCCLManager cache for places_key
void ProcessGroupNCCL::CreateNCCLManagerCache(
    const std::string& places_key, const std::vector<Place>& places) {
  PADDLE_ENFORCE_EQ(places_key.empty(), false,
                    platform::errors::PreconditionNotMet(
                        "Not able to create/get the NCCL Communicator since "
                        "the GPU place are not known"));

  std::vector<std::shared_ptr<NCCLCommManager>> nccl_comms;
  nccl_comms.resize(places.size());

  // using vector just for broadcast
  std::vector<ncclUniqueId> nccl_ids;
  nccl_ids.resize(1);
  auto& nccl_id = nccl_ids.front();

  for (auto& place : places) {
    used_place_ids_.insert(place.GetDeviceId());
  }

  if (rank_ == 0) {
    PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclGetUniqueId(&nccl_id));
  }
  BroadcastUniqueNCCLID(nccl_ids);

  VLOG(3) << "init nccl rank: " << strategy_.local_rank_
          << ", nranks: " << strategy_.nranks_ << ", place: " << places_key
          << ", nccl uniqueid: " << SerializeNCCLUniqueId(nccl_id);

  std::vector<std::unique_ptr<CUDADeviceContext>> dev_ctx;
  dev_ctx.resize(places.size());

  PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclGroupStart());

  for (size_t i = 0; i < places.size(); ++i) {
    platform::CUDADeviceGuard guard(places[i]);
    nccl_comms[i] = NCCLCommManager::Create(GetSize(), GetRank(), nccl_id);
    dev_ctx[i].reset(new CUDADeviceContext(places[i]));
  }

  PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclGroupEnd());

  std::vector<EventManager> events;
  events.resize(places.size());

  // These caches will be useful to process sync/wait/communicate
  places_to_events_.emplace(places_key, std::move(events));
  places_to_ncclcomm_.emplace(places_key, std::move(nccl_comms));
  places_to_ctx_.emplace(places_key, std::move(dev_ctx));
}

template <typename Fn>
std::shared_ptr<ProcessGroup::Task> ProcessGroupNCCL::Collective(
    std::vector<Tensor>& inputs, std::vector<Tensor>& outputs, Fn fn,
    CommType op_type) {
  const auto places = GetPlaceList(inputs);
  const auto key = GetKeyFromPlaces(places);

  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (places_to_ncclcomm_.find(key) == places_to_ncclcomm_.end()) {
      CreateNCCLManagerCache(key, places);
    }
  }

  auto& nccl_comms = places_to_ncclcomm_[key];

  SyncDefaultStream(places, places_to_events_[key], places_to_ctx_[key]);

  auto task = CreateTask(places, rank_, op_type, inputs);
  task->SetOutputs(outputs);

  // construct uninitialize guard for device
  platform::CUDADeviceGuard cuda_guard;

  if (FLAGS_use_stream_safe_cuda_allocator) {
    for (size_t i = 0; i < inputs.size(); ++i) {
      cuda_guard.SetDevice(places[i]);
      auto dense_tensor =
          std::dynamic_pointer_cast<phi::DenseTensor>(inputs[i].impl());
      memory::RecordStream(dense_tensor->Holder(),
                           places_to_ctx_[key][i]->stream());
    }
  }

  {
    platform::NCCLGroupGuard nccl_guard;
    for (size_t i = 0; i < inputs.size(); ++i) {
      cuda_guard.SetDevice(places[i]);
      const auto& nccl_stream = places_to_ctx_[key][i]->stream();
      fn(inputs[i], outputs[i], nccl_comms[i]->GetNcclComm(), nccl_stream);
    }
  }

  for (size_t i = 0; i < inputs.size(); ++i) {
    cuda_guard.SetDevice(places[i]);
    task->control_events_[i].Record(*places_to_ctx_[key][i]);
  }
  return task;
}

template <typename Fn>
std::shared_ptr<ProcessGroup::Task> ProcessGroupNCCL::PointToPoint(
    std::vector<Tensor>& tensors, Fn fn, int dst_rank, CommType op_type) {
  const auto places = GetPlaceList(tensors);
  const auto key = GetKeyFromPlaces(places);

  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (places_to_ncclcomm_.find(key) == places_to_ncclcomm_.end()) {
      CreateNCCLManagerCache(key, places);
    }
  }

  auto& nccl_comms = places_to_ncclcomm_[key];

  SyncDefaultStream(places, places_to_events_[key], places_to_ctx_[key]);

  auto task = CreateTask(places, rank_, op_type, tensors);

  // construct uninitialize guard for device
  platform::CUDADeviceGuard cuda_guard;

  if (FLAGS_use_stream_safe_cuda_allocator) {
    for (size_t i = 0; i < tensors.size(); ++i) {
      cuda_guard.SetDevice(places[i]);
      auto dense_tensor =
          std::dynamic_pointer_cast<phi::DenseTensor>(tensors[i].impl());
      memory::RecordStream(dense_tensor->Holder(),
                           places_to_ctx_[key][i]->stream());
    }
  }

  {
    platform::NCCLGroupGuard nccl_guard;
    for (size_t i = 0; i < tensors.size(); ++i) {
      cuda_guard.SetDevice(places[i]);
      const auto& nccl_stream = places_to_ctx_[key][i]->stream();
      fn(tensors[i], nccl_comms[i]->GetNcclComm(), nccl_stream, dst_rank);
    }
  }

  for (size_t i = 0; i < tensors.size(); ++i) {
    cuda_guard.SetDevice(places[i]);
    task->control_events_[i].Record(*places_to_ctx_[key][i]);
  }
  return task;
}

std::shared_ptr<ProcessGroup::Task> ProcessGroupNCCL::AllReduce(
    std::vector<Tensor>& tensors, const AllreduceOptions& opts) {
  PADDLE_ENFORCE_EQ(
      CheckTensorsInCudaPlace(tensors), true,
      platform::errors::InvalidArgument("All inputs should be in CudaPlace."));
  return Collective(
      tensors, tensors,
      [&](const Tensor& input, Tensor& output, ncclComm_t comm,
          const gpuStream_t& stream) {
        auto input_tensor =
            std::dynamic_pointer_cast<phi::DenseTensor>(input.impl());
        auto output_tensor =
            std::dynamic_pointer_cast<phi::DenseTensor>(output.impl());
        return platform::dynload::ncclAllReduce(
            input_tensor->data(), output_tensor->data(), input_tensor->numel(),
            platform::ToNCCLDataType(input.type()),
            ToNCCLRedType(opts.reduce_op), comm, stream);
      },
      CommType::ALLREDUCE);
}

std::shared_ptr<ProcessGroup::Task> ProcessGroupNCCL::Broadcast(
    std::vector<Tensor>& tensors, const BroadcastOptions& opts) {
  PADDLE_ENFORCE_EQ(
      CheckTensorsInCudaPlace(tensors), true,
      platform::errors::InvalidArgument("All inputs should be in CudaPlace."));

  return Collective(
      tensors, tensors,
      [&](Tensor& input, Tensor& output, ncclComm_t comm,
          const gpuStream_t& stream) {
        const auto root = opts.source_rank * tensors.size() + opts.source_root;
        auto input_tensor =
            std::dynamic_pointer_cast<phi::DenseTensor>(input.impl());
        auto output_tensor =
            std::dynamic_pointer_cast<phi::DenseTensor>(output.impl());
        return platform::dynload::ncclBcast(
            input_tensor->data(), input_tensor->numel(),
            platform::ToNCCLDataType(input.type()), root, comm, stream);
      },
      CommType::BROADCAST);
}

std::shared_ptr<ProcessGroup::Task> ProcessGroupNCCL::Barrier(
    const BarrierOptions& opts) {
  std::vector<phi::GPUPlace> places;

  if (!opts.place_ids.empty()) {
    for (auto place_id : opts.place_ids) {
      places.emplace_back(place_id);
    }
  } else if (!used_place_ids_.empty()) {
    for (auto place_id : used_place_ids_) {
      places.emplace_back(place_id);
    }
  } else {
    auto numGPUs = GetSize();
    int place_id = static_cast<int>(rank_ % numGPUs);
    places.emplace_back(place_id);
  }

  std::vector<Tensor> barrierTensors;
  barrierTensors.reserve(places.size());

  platform::CUDADeviceGuard gpuGuard;
  for (auto& place : places) {
    gpuGuard.SetDeviceIndex(place.GetDeviceId());
    auto dt = full({1}, 0, phi::DataType::FLOAT32, phi::Backend::GPU);
    barrierTensors.push_back(dt);
  }
  auto task = ProcessGroupNCCL::AllReduce(barrierTensors);
  auto nccl_task = dynamic_cast<ProcessGroupNCCL::NCCLTask*>(task.get());
  nccl_task->barrierTensors_ = std::move(barrierTensors);
  return task;
}

void CheckTensorsInDifferentDevices(const std::vector<Tensor>& tensors,
                                    const size_t num_devices) {
  PADDLE_ENFORCE_EQ(
      tensors.size() == 0, false,
      platform::errors::InvalidArgument("Tensor list must be nonempty."));
  PADDLE_ENFORCE_LE(
      tensors.size(), num_devices,
      platform::errors::InvalidArgument(
          "Tensor list mustn't be larger than the number of available GPUs."));

  std::set<Place> used_devices;

  for (const auto& t : tensors) {
    PADDLE_ENFORCE_EQ(t.is_cuda() && t.is_dense_tensor(), true,
                      platform::errors::InvalidArgument(
                          "Tensors must be CUDA and dense tensor."));

    const auto inserted = used_devices.insert(t.inner_place()).second;
    PADDLE_ENFORCE_EQ(inserted, true,
                      platform::errors::InvalidArgument(
                          "Tensors must be on distinct GPU devices."));
  }
}

std::shared_ptr<ProcessGroup::Task> ProcessGroupNCCL::Send(
    std::vector<Tensor>& tensors, int dst_rank) {
  CheckTensorsInDifferentDevices(tensors, static_cast<size_t>(GetSize()));

  auto task = PointToPoint(
      tensors,
      [&](Tensor& input, ncclComm_t comm, const gpuStream_t& stream,
          int dst_rank) {
        auto input_tensor =
            std::dynamic_pointer_cast<phi::DenseTensor>(input.impl());
        return platform::dynload::ncclSend(
            input_tensor->data(), input_tensor->numel(),
            platform::ToNCCLDataType(input.type()), dst_rank, comm, stream);
      },
      dst_rank, CommType::SEND);
  return task;
}

std::shared_ptr<ProcessGroup::Task> ProcessGroupNCCL::Recv(
    std::vector<Tensor>& tensors, int src_rank) {
  CheckTensorsInDifferentDevices(tensors, static_cast<size_t>(GetSize()));

  auto task = PointToPoint(
      tensors,
      [&](Tensor& output, ncclComm_t comm, const gpuStream_t& stream,
          int src_rank) {
        auto output_tensor =
            std::dynamic_pointer_cast<phi::DenseTensor>(output.impl());
        return platform::dynload::ncclRecv(
            output_tensor->data(), output_tensor->numel(),
            platform::ToNCCLDataType(output.type()), src_rank, comm, stream);
      },
      src_rank, CommType::RECV);
  return task;
}

}  //  namespace distributed
}  //  namespace paddle
