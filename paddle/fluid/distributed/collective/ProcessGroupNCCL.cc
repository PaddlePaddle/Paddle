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

#include "paddle/fluid/distributed/collective/Common.h"
#include "paddle/fluid/platform/device/gpu/gpu_info.h"
#include "paddle/fluid/platform/device/gpu/nccl_helper.h"
#include "paddle/fluid/platform/place.h"
#include "paddle/phi/api/lib/utils/allocator.h"
#include "paddle/phi/common/place.h"

DECLARE_bool(nccl_blocking_wait);
DECLARE_bool(use_stream_safe_cuda_allocator);

constexpr int64_t kWaitBlockTImeout = 10;

namespace paddle {
namespace distributed {

void SyncDefaultStream(
    const std::vector<Place>& places,
    std::vector<EventManager>& ncclEvents,                     // NOLINT
    std::vector<std::unique_ptr<phi::GPUContext>>& dev_ctx) {  // NOLINT
  for (size_t i = 0; i < places.size(); ++i) {
    auto* default_ctx = static_cast<phi::GPUContext*>(
        platform::DeviceContextPool::Instance().Get(places[i]));
    ncclEvents[i].Record(*default_ctx);
    ncclEvents[i].Block(*dev_ctx[i]);
  }
}

std::shared_ptr<ProcessGroupNCCL::NCCLTask> ProcessGroupNCCL::CreateTask(
    std::vector<Place> places,
    int rank,
    CommType comm_type,
    const std::vector<phi::DenseTensor>& inputs) {
  return std::make_shared<ProcessGroupNCCL::NCCLTask>(
      places, rank, comm_type, inputs);
}

ProcessGroupNCCL::NCCLTask::NCCLTask(
    const std::vector<Place>& places,
    int rank,
    CommType CommType,
    const std::vector<phi::DenseTensor>& inputs)
    : TaskStream(rank, inputs, CommType), places_(places) {
  control_events_.resize(places.size());
  ncclComms_.resize(places.size());
}

ProcessGroupNCCL::NCCLTask::NCCLTask(
    const std::vector<Place>& places,
    int rank,
    CommType comm_type,
    const std::vector<phi::DenseTensor>& inputs,
    bool sync_op,
    bool use_calc_stream)
    : TaskStream(rank, inputs, comm_type, sync_op, use_calc_stream),
      places_(places) {
  control_events_.resize(places.size());
  ncclComms_.resize(places.size());
}

ProcessGroupNCCL::NCCLTask::~NCCLTask() {}

void ProcessGroupNCCL::NCCLTask::SetOutputs(
    std::vector<phi::DenseTensor>& outputs) {  // NOLINT
  outputs_ = std::make_shared<std::vector<phi::DenseTensor>>(outputs);
}

void ProcessGroupNCCL::NCCLTask::SynchronizeStreams() {
  for (size_t i = 0; i < places_.size(); ++i) {
    auto* default_ctx = static_cast<phi::GPUContext*>(
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

void ProcessGroupNCCL::CheckSplitSizes(std::vector<int64_t>* split_sizes,
                                       std::vector<int64_t> tensor_shape) {
  int64_t len_size = (*split_sizes).size();
  if (len_size == 0) {
    PADDLE_ENFORCE_EQ(tensor_shape[0] % size_ == 0,
                      true,
                      platform::errors::InvalidArgument(
                          "Tensor's dim[0] must be divisible by group size "
                          "when split_sizes not given."));
    (*split_sizes)
        .insert((*split_sizes).end(),
                size_,
                static_cast<int64_t>(tensor_shape[0] / size_));
  } else {
    PADDLE_ENFORCE_EQ(
        len_size == size_,
        true,
        platform::errors::InvalidArgument(
            "The length of split_sizes must be equal to group size."));
    auto sum_size = std::accumulate(
        (*split_sizes).begin(), (*split_sizes).end(), static_cast<int64_t>(0));
    PADDLE_ENFORCE_EQ(
        sum_size == tensor_shape[0],
        true,
        platform::errors::InvalidArgument(
            "The sum of split_sizes must be equal to tensor's dim[0]."));
  }
}

// TODO(sheniang03): Add timeout for wait, now timeout unused
bool ProcessGroupNCCL::NCCLTask::Wait(std::chrono::milliseconds timeout) {
  // Warning here when use calc stream but also invoke waiting explicitly.
  if (UseCalcStream()) {
    VLOG(3) << "Warning: The communication is on calc stream, wait here is "
               "useless.";
    return true;
  }

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
#ifdef PADDLE_WITH_CUDA
      PADDLE_ENFORCE_GPU_SUCCESS(cudaDeviceSynchronize());
#else
      PADDLE_ENFORCE_GPU_SUCCESS(hipDeviceSynchronize());
#endif
    }
  }
  return true;
}

// Same as Wait
void ProcessGroupNCCL::NCCLTask::Synchronize() { Wait(kWaitTimeout); }

ProcessGroupNCCL::ProcessGroupNCCL(const std::shared_ptr<Store>& store,
                                   int rank,
                                   int size,
                                   const platform::Place& place,
                                   int gid)
    : ProcessGroupStream(rank, size, place, gid), store_(store) {
  platform::SetDeviceId(place_.device);
}

void ProcessGroupNCCL::BroadcastUniqueNCCLID(
    std::vector<ncclUniqueId>& nccl_ids) {  // NOLINT
  if (rank_ == 0) {
    for (size_t i = 0; i < nccl_ids.size(); i++) {
      auto key = "ProcessGroupNCCL/nccl_ids/" + std::to_string(gid_) + "/" +
                 std::to_string(i);
      auto nccl_id = std::vector<uint8_t>(
          reinterpret_cast<uint8_t*>(&nccl_ids[i]),
          reinterpret_cast<uint8_t*>(&nccl_ids[i]) + NCCL_UNIQUE_ID_BYTES);
      store_->set(key, nccl_id);
    }
  } else {
    for (size_t i = 0; i < nccl_ids.size(); i++) {
      auto key = "ProcessGroupNCCL/nccl_ids/" + std::to_string(gid_) + "/" +
                 std::to_string(i);
      auto ret = store_->get(key);
      std::memcpy(&nccl_ids[i], ret.data(), ret.size());
    }
  }
}

// create NCCLManager cache for places_key
void ProcessGroupNCCL::CreateNCCLManagerCache(
    const std::string& places_key, const std::vector<Place>& places) {
  PADDLE_ENFORCE_EQ(places_key.empty(),
                    false,
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

  VLOG(3) << "init nccl rank: " << rank_ << ", nranks: " << size_
          << ", place: " << places_key
          << ", nccl uniqueid: " << SerializeNCCLUniqueId(nccl_id);

  std::vector<std::unique_ptr<phi::GPUContext>> dev_ctx;
  dev_ctx.resize(places.size());

  PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclGroupStart());

  for (size_t i = 0; i < places.size(); ++i) {
    platform::CUDADeviceGuard guard(places[i]);
    nccl_comms[i] = NCCLCommManager::Create(GetSize(), GetRank(), nccl_id);
    dev_ctx[i].reset(new phi::GPUContext(places[i]));
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
    std::vector<phi::DenseTensor>& inputs,
    std::vector<phi::DenseTensor>& outputs,
    Fn fn,
    CommType comm_type,
    bool sync_op,
    bool use_calc_stream) {
  const auto& places = GetPlaceList(inputs);
  const auto& key = GetKeyFromPlaces(places);

  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (places_to_ncclcomm_.find(key) == places_to_ncclcomm_.end()) {
      CreateNCCLManagerCache(key, places);
    }
  }

  auto& nccl_comms = places_to_ncclcomm_[key];

  SyncDefaultStream(places, places_to_events_[key], places_to_ctx_[key]);

  auto task = std::make_shared<ProcessGroupNCCL::NCCLTask>(
      places, rank_, comm_type, inputs, sync_op, use_calc_stream);

  platform::CUDADeviceGuard cuda_guard;

  {
    platform::NCCLGroupGuard nccl_guard;
    for (size_t i = 0; i < inputs.size(); ++i) {
      cuda_guard.SetDevice(places[i]);

      gpuStream_t nccl_stream;
      if (use_calc_stream) {
        nccl_stream =
            static_cast<phi::GPUContext*>(
                platform::DeviceContextPool::Instance().Get(places[i]))
                ->stream();
      } else {
        nccl_stream = places_to_ctx_[key][i]->stream();
      }

      fn(inputs[i], outputs[i], nccl_comms[i]->GetNcclComm(), nccl_stream);
    }
  }

  if (FLAGS_use_stream_safe_cuda_allocator) {
    for (size_t i = 0; i < inputs.size(); ++i) {
      cuda_guard.SetDevice(places[i]);

      gpuStream_t nccl_stream;
      if (use_calc_stream) {
        nccl_stream =
            static_cast<phi::GPUContext*>(
                platform::DeviceContextPool::Instance().Get(places[i]))
                ->stream();
      } else {
        nccl_stream = places_to_ctx_[key][i]->stream();
      }

      memory::RecordStream(inputs[i].Holder(), nccl_stream);
    }
  }

  // Adding stream event dependency only when use comm stream
  if (!use_calc_stream) {
    for (size_t i = 0; i < inputs.size(); ++i) {
      cuda_guard.SetDevice(places[i]);
      task->control_events_[i].Record(*places_to_ctx_[key][i]);
    }
  }

  return task;
}

template <typename Fn>
std::shared_ptr<ProcessGroup::Task> ProcessGroupNCCL::Collective(
    std::vector<phi::DenseTensor>& inputs,
    std::vector<phi::DenseTensor>& outputs,
    Fn fn,
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

  // construct uninitialize guard for device
  platform::CUDADeviceGuard cuda_guard;

  {
    platform::NCCLGroupGuard nccl_guard;
    for (size_t i = 0; i < inputs.size(); ++i) {
      cuda_guard.SetDevice(places[i]);
      const auto& nccl_stream = places_to_ctx_[key][i]->stream();
      fn(inputs[i], outputs[i], nccl_comms[i]->GetNcclComm(), nccl_stream);
    }
  }

  if (FLAGS_use_stream_safe_cuda_allocator) {
    for (size_t i = 0; i < inputs.size(); ++i) {
      cuda_guard.SetDevice(places[i]);
      memory::RecordStream(inputs[i].Holder(),
                           places_to_ctx_[key][i]->stream());
    }
  }

  for (size_t i = 0; i < inputs.size(); ++i) {
    cuda_guard.SetDevice(places[i]);
    task->control_events_[i].Record(*places_to_ctx_[key][i]);
  }
  return task;
}

template <typename Fn>
void ProcessGroupNCCL::Collective(const phi::DenseTensor* in,
                                  phi::DenseTensor* out,
                                  Fn fn,
                                  CommType op_type) {
  std::vector<Place> places;
  places.push_back(in->place());
  const auto key = GetKeyFromPlaces(places);

  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (places_to_ncclcomm_.find(key) == places_to_ncclcomm_.end()) {
      CreateNCCLManagerCache(key, places);
    }
  }

  auto& nccl_comms = places_to_ncclcomm_[key];

  SyncDefaultStream(places, places_to_events_[key], places_to_ctx_[key]);

  // construct uninitialize guard for device
  platform::CUDADeviceGuard cuda_guard;

  if (FLAGS_use_stream_safe_cuda_allocator) {
    cuda_guard.SetDevice(places[0]);
    memory::RecordStream(in->Holder(), places_to_ctx_[key][0]->stream());
  }

  {
    platform::NCCLGroupGuard nccl_guard;
    cuda_guard.SetDevice(places[0]);
    const auto& nccl_stream = places_to_ctx_[key][0]->stream();
    fn(in, out, nccl_comms[0]->GetNcclComm(), nccl_stream);
  }

  cuda_guard.SetDevice(places[0]);
}

template <typename Fn>
std::shared_ptr<ProcessGroup::Task> ProcessGroupNCCL::PointToPoint(
    std::vector<phi::DenseTensor>& tensors,
    Fn fn,
    int dst_rank,
    CommType op_type) {
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
      memory::RecordStream(tensors[i].Holder(),
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
    std::vector<phi::DenseTensor>& in_tensors,
    std::vector<phi::DenseTensor>& out_tensors,
    const AllreduceOptions& opts) {
  PADDLE_ENFORCE_EQ(
      CheckTensorsInCudaPlace(in_tensors),
      true,
      platform::errors::InvalidArgument("All inputs should be in CudaPlace."));
  return Collective(
      in_tensors,
      out_tensors,
      [&](const phi::DenseTensor& input,
          phi::DenseTensor& output,
          ncclComm_t comm,
          const gpuStream_t& stream) {
        return platform::dynload::ncclAllReduce(
            input.data(),
            output.data(),
            input.numel(),
            platform::ToNCCLDataType(input.type()),
            ToNCCLRedType(opts.reduce_op),
            comm,
            stream);
      },
      CommType::ALLREDUCE);
}

std::shared_ptr<ProcessGroup::Task> ProcessGroupNCCL::AllReduce(
    std::vector<phi::DenseTensor>& in_tensors,
    std::vector<phi::DenseTensor>& out_tensors,
    const AllreduceOptions& opts,
    bool sync_op,
    bool use_calc_stream) {
  PADDLE_ENFORCE_EQ(
      CheckTensorsInCudaPlace(in_tensors),
      true,
      platform::errors::InvalidArgument("All inputs should be in CudaPlace."));
  return Collective(
      in_tensors,
      out_tensors,
      [&](const phi::DenseTensor& input,
          phi::DenseTensor& output,
          ncclComm_t comm,
          const gpuStream_t& stream) {
        return platform::dynload::ncclAllReduce(
            input.data(),
            output.data(),
            input.numel(),
            platform::ToNCCLDataType(input.type()),
            ToNCCLRedType(opts.reduce_op),
            comm,
            stream);
      },
      CommType::ALLREDUCE,
      sync_op,
      use_calc_stream);
}

std::shared_ptr<ProcessGroup::Task> ProcessGroupNCCL::Broadcast(
    std::vector<phi::DenseTensor>& in_tensors,
    std::vector<phi::DenseTensor>& out_tensors,
    const BroadcastOptions& opts) {
  PADDLE_ENFORCE_EQ(
      CheckTensorsInCudaPlace(in_tensors),
      true,
      platform::errors::InvalidArgument("All inputs should be in CudaPlace."));

  return Collective(
      in_tensors,
      out_tensors,
      [&](phi::DenseTensor& input,
          phi::DenseTensor& output,
          ncclComm_t comm,
          const gpuStream_t& stream) {
        const auto root =
            opts.source_rank * in_tensors.size() + opts.source_root;
        return platform::dynload::ncclBroadcast(
            input.data(),
            output.data(),
            input.numel(),
            platform::ToNCCLDataType(input.type()),
            root,
            comm,
            stream);
      },
      CommType::BROADCAST);
}

std::shared_ptr<ProcessGroup::Task> ProcessGroupNCCL::Barrier(
    const BarrierOptions& opts) {
  // Only support single card single process
  std::vector<phi::GPUPlace> places = {place_};

  std::vector<phi::DenseTensor> barrierTensors;
  barrierTensors.reserve(places.size());

  platform::CUDADeviceGuard gpuGuard;
  for (auto& place : places) {
    gpuGuard.SetDeviceIndex(place.GetDeviceId());
    phi::DenseTensorMeta meta(phi::DataType::FLOAT32, phi::DDim({1}));
    auto allocator = std::unique_ptr<phi::Allocator>(
        new paddle::experimental::DefaultAllocator(place));
    barrierTensors.emplace_back(allocator.get(), meta);
  }
  auto task = ProcessGroupNCCL::AllReduce(
      barrierTensors, barrierTensors, AllreduceOptions());
  auto nccl_task = dynamic_cast<ProcessGroupNCCL::NCCLTask*>(task.get());
  nccl_task->barrierTensors_ = std::move(barrierTensors);
  return task;
}

void CheckTensorsInDifferentDevices(
    const std::vector<phi::DenseTensor>& tensors, const size_t num_devices) {
  PADDLE_ENFORCE_EQ(
      tensors.size() == 0,
      false,
      platform::errors::InvalidArgument("Tensor list must be nonempty."));
  PADDLE_ENFORCE_LE(
      tensors.size(),
      num_devices,
      platform::errors::InvalidArgument(
          "Tensor list mustn't be larger than the number of available GPUs."));

  std::set<Place> used_devices;

  for (const auto& t : tensors) {
    PADDLE_ENFORCE_EQ(platform::is_gpu_place(t.place()),
                      true,
                      platform::errors::InvalidArgument(
                          "Tensors must be CUDA and dense tensor."));

    const auto inserted = used_devices.insert(t.place()).second;
    PADDLE_ENFORCE_EQ(inserted,
                      true,
                      platform::errors::InvalidArgument(
                          "Tensors must be on distinct GPU devices."));
  }
}

std::shared_ptr<ProcessGroup::Task> ProcessGroupNCCL::Send(
    std::vector<phi::DenseTensor>& tensors, int dst_rank) {
  CheckTensorsInDifferentDevices(tensors, static_cast<size_t>(GetSize()));

  auto task = PointToPoint(
      tensors,
      [&](phi::DenseTensor& input,
          ncclComm_t comm,
          const gpuStream_t& stream,
          int dst_rank) {
        return platform::dynload::ncclSend(
            input.data(),
            input.numel(),
            platform::ToNCCLDataType(input.dtype()),
            dst_rank,
            comm,
            stream);
      },
      dst_rank,
      CommType::SEND);
  return task;
}

std::shared_ptr<ProcessGroup::Task> ProcessGroupNCCL::Recv(
    std::vector<phi::DenseTensor>& tensors, int src_rank) {
  CheckTensorsInDifferentDevices(tensors, static_cast<size_t>(GetSize()));

  auto task = PointToPoint(
      tensors,
      [&](phi::DenseTensor& output,
          ncclComm_t comm,
          const gpuStream_t& stream,
          int src_rank) {
        return platform::dynload::ncclRecv(
            output.data(),
            output.numel(),
            platform::ToNCCLDataType(output.dtype()),
            src_rank,
            comm,
            stream);
      },
      src_rank,
      CommType::RECV);
  return task;
}

std::shared_ptr<ProcessGroup::Task> ProcessGroupNCCL::Send_Partial(
    phi::DenseTensor& tensors, int dst_rank, int offset, int length) {
  // CheckTensorsInDifferentDevices(tensors, static_cast<size_t>(GetSize()));

  phi::DenseTensor flatten_tensor;
  flatten_tensor.ShareDataWith(tensors).Resize({tensors.numel()});

  phi::DenseTensor shared_input = flatten_tensor.Slice(offset, offset + length);

  std::vector<phi::DenseTensor> shared_tensors;
  shared_tensors.push_back(shared_input);

  auto task = PointToPoint(
      shared_tensors,
      [&](phi::DenseTensor& input,
          ncclComm_t comm,
          const gpuStream_t& stream,
          int dst_rank) {
        return platform::dynload::ncclSend(
            input.data(),
            input.numel(),
            platform::ToNCCLDataType(input.dtype()),
            dst_rank,
            comm,
            stream);
      },
      dst_rank,
      CommType::SEND);
  return task;
}

std::shared_ptr<ProcessGroup::Task> ProcessGroupNCCL::Recv_Partial(
    phi::DenseTensor& tensors, int src_rank, int offset, int length) {
  // phi::DenseTensor shared_input = tensors.Slice(offset, offset+length);

  phi::DenseTensor flatten_tensor;
  flatten_tensor.ShareDataWith(tensors).Resize({tensors.numel()});
  phi::DenseTensor shared_input = flatten_tensor.Slice(offset, offset + length);

  std::vector<phi::DenseTensor> shared_tensors;
  shared_tensors.push_back(shared_input);

  auto task = PointToPoint(
      shared_tensors,
      [&](phi::DenseTensor& output,
          ncclComm_t comm,
          const gpuStream_t& stream,
          int src_rank) {
        return platform::dynload::ncclRecv(
            output.data(),
            output.numel(),
            platform::ToNCCLDataType(output.dtype()),
            src_rank,
            comm,
            stream);
      },
      src_rank,
      CommType::RECV);
  return task;
}

std::shared_ptr<ProcessGroup::Task> ProcessGroupNCCL::AllGather(
    std::vector<phi::DenseTensor>& in_tensors,
    std::vector<phi::DenseTensor>& out_tensors) {
  PADDLE_ENFORCE_EQ(
      CheckTensorsInCudaPlace(in_tensors),
      true,
      platform::errors::InvalidArgument("All inputs should be in CudaPlace."));
  PADDLE_ENFORCE_EQ(
      CheckTensorsInCudaPlace(out_tensors),
      true,
      platform::errors::InvalidArgument("All outputs should be in CudaPlace."));
  return Collective(
      in_tensors,
      out_tensors,
      [&](const phi::DenseTensor& input,
          phi::DenseTensor& output,
          ncclComm_t comm,
          const gpuStream_t& stream) {
        return platform::dynload::ncclAllGather(
            input.data(),
            output.data(),
            input.numel(),
            platform::ToNCCLDataType(input.dtype()),
            comm,
            stream);
      },
      CommType::ALLGATHER);
}

void* GetPointerByOffset(void* raw_pointer,
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
        "This datatype in nccl is not supported."));
  }
  return nullptr;
}

std::shared_ptr<ProcessGroup::Task> ProcessGroupNCCL::AllGather_Partial(
    std::vector<phi::DenseTensor>& in_tensors,
    std::vector<phi::DenseTensor>& out_tensors,
    int offset,
    int length) {
  PADDLE_ENFORCE_EQ(
      CheckTensorsInCudaPlace(in_tensors),
      true,
      platform::errors::InvalidArgument("All inputs should be in CudaPlace."));
  PADDLE_ENFORCE_EQ(
      CheckTensorsInCudaPlace(out_tensors),
      true,
      platform::errors::InvalidArgument("All outputs should be in CudaPlace."));
  return Collective(
      in_tensors,
      out_tensors,
      [&](phi::DenseTensor& input,
          phi::DenseTensor& output,
          ncclComm_t comm,
          const gpuStream_t& stream) {
        return platform::dynload::ncclAllGather(
            GetPointerByOffset(input.data(), offset, input.dtype()),
            output.data(),
            length,
            platform::ToNCCLDataType(input.dtype()),
            comm,
            stream);
      },
      CommType::ALLGATHER);
}

std::shared_ptr<ProcessGroup::Task> ProcessGroupNCCL::AllToAll(
    std::vector<phi::DenseTensor>& in_tensors,
    std::vector<phi::DenseTensor>& out_tensors) {
  PADDLE_ENFORCE_EQ(
      CheckTensorsInCudaPlace(in_tensors),
      true,
      platform::errors::InvalidArgument("All inputs should be in CudaPlace."));
  PADDLE_ENFORCE_EQ(
      CheckTensorsInCudaPlace(out_tensors),
      true,
      platform::errors::InvalidArgument("All inputs should be in CudaPlace."));
  return Collective(
      in_tensors,
      out_tensors,
      [&](phi::DenseTensor& input,
          phi::DenseTensor& output,
          ncclComm_t comm,
          const gpuStream_t& stream) {
        size_t offset = 0;
        PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclGroupStart());
        for (auto i = 0; i < size_; i++) {
          PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclSend(
              GetPointerByOffset(input.data(), offset, input.dtype()),
              input.numel() / size_,
              platform::ToNCCLDataType(input.dtype()),
              i,
              comm,
              stream));
          PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclRecv(
              GetPointerByOffset(output.data(), offset, input.dtype()),
              input.numel() / size_,
              platform::ToNCCLDataType(input.dtype()),
              i,
              comm,
              stream));
          offset += input.numel() / size_;
        }
        PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclGroupEnd());
      },
      CommType::ALLTOALL);
}

std::shared_ptr<ProcessGroup::Task> ProcessGroupNCCL::AllToAll_Single(
    std::vector<phi::DenseTensor>& in_tensors,
    std::vector<phi::DenseTensor>& out_tensors,
    std::vector<int64_t>& in_sizes,
    std::vector<int64_t>& out_sizes) {
  PADDLE_ENFORCE_EQ(
      CheckTensorsInCudaPlace(in_tensors),
      true,
      platform::errors::InvalidArgument("All inputs should be in CudaPlace."));
  PADDLE_ENFORCE_EQ(
      CheckTensorsInCudaPlace(out_tensors),
      true,
      platform::errors::InvalidArgument("All inputs should be in CudaPlace."));
  return Collective(
      in_tensors,
      out_tensors,
      [&](phi::DenseTensor& input,
          phi::DenseTensor& output,
          ncclComm_t comm,
          const gpuStream_t& stream) {
        PADDLE_ENFORCE_EQ(input.dtype() == output.dtype(),
                          true,
                          platform::errors::InvalidArgument(
                              "The dtypes of input and output must be equal."));

        std::vector<int64_t> in_dims = phi::vectorize(input.dims());
        std::vector<int64_t> out_dims = phi::vectorize(output.dims());
        CheckSplitSizes(&in_sizes, in_dims);
        CheckSplitSizes(&out_sizes, out_dims);

        size_t in_offset = 0, out_offset = 0;
        size_t in_length = 0, out_length = 0;
        size_t in_row_size = input.numel() / in_dims[0];
        size_t out_row_size = output.numel() / out_dims[0];

        PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclGroupStart());
        for (auto i = 0; i < size_; i++) {
          in_length = in_sizes[i] * in_row_size;
          PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclSend(
              GetPointerByOffset(input.data(), in_offset, input.dtype()),
              in_length,
              platform::ToNCCLDataType(input.dtype()),
              i,
              comm,
              stream));
          in_offset += in_length;

          out_length = out_sizes[i] * out_row_size;
          PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclRecv(
              GetPointerByOffset(output.data(), out_offset, input.dtype()),
              out_length,
              platform::ToNCCLDataType(input.dtype()),
              i,
              comm,
              stream));
          out_offset += out_length;
        }
        PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclGroupEnd());
      },
      CommType::ALLTOALL_SINGLE);
}

std::shared_ptr<ProcessGroup::Task> ProcessGroupNCCL::Reduce(
    std::vector<phi::DenseTensor>& in_tensors,
    std::vector<phi::DenseTensor>& out_tensors,
    const ReduceOptions& opts) {
  PADDLE_ENFORCE_EQ(
      CheckTensorsInCudaPlace(in_tensors),
      true,
      platform::errors::InvalidArgument("All inputs should be in CudaPlace."));
  return Collective(
      in_tensors,
      out_tensors,
      [&](const phi::DenseTensor& input,
          phi::DenseTensor& output,
          ncclComm_t comm,
          const gpuStream_t& stream) {
        PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclReduce(
            input.data(),
            output.data(),
            input.numel(),
            platform::ToNCCLDataType(input.dtype()),
            ToNCCLRedType(opts.reduce_op),
            opts.root_rank,
            comm,
            stream));
      },
      CommType::REDUCE);
}

std::shared_ptr<ProcessGroup::Task> ProcessGroupNCCL::Scatter(
    std::vector<phi::DenseTensor>& in_tensors,
    std::vector<phi::DenseTensor>& out_tensors,
    const ScatterOptions& opts) {
  PADDLE_ENFORCE_EQ(
      CheckTensorsInCudaPlace(in_tensors),
      true,
      platform::errors::InvalidArgument("All inputs should be in CudaPlace."));
  PADDLE_ENFORCE_EQ(
      CheckTensorsInCudaPlace(out_tensors),
      true,
      platform::errors::InvalidArgument("All inputs should be in CudaPlace."));
  return Collective(
      in_tensors,
      out_tensors,
      [&](phi::DenseTensor& input,
          phi::DenseTensor& output,
          ncclComm_t comm,
          const gpuStream_t& stream) {
        size_t offset = 0;
        if (rank_ == opts.root_rank) {
          PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclGroupStart());
          for (auto i = 0; i < size_; i++) {
            PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclSend(
                GetPointerByOffset(input.data(), offset, input.dtype()),
                input.numel() / size_,
                platform::ToNCCLDataType(input.dtype()),
                i,
                comm,
                stream));
            offset += input.numel() / size_;
          }
          PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclRecv(
              output.data(),
              input.numel() / size_,
              platform::ToNCCLDataType(input.dtype()),
              opts.root_rank,
              comm,
              stream));
          PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclGroupEnd());
        } else {
          PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclRecv(
              output.data(),
              input.numel() / size_,
              platform::ToNCCLDataType(input.dtype()),
              opts.root_rank,
              comm,
              stream));
        }
      },
      CommType::SCATTER);
}

std::shared_ptr<ProcessGroup::Task> ProcessGroupNCCL::_ReduceScatterBase(
    phi::DenseTensor& out_tensor,
    phi::DenseTensor& in_tensor,
    const ReduceScatterOptions& opts) {
  // auto tensor = out_tensors.back();
  PADDLE_ENFORCE_EQ(
      out_tensor.dtype(),
      in_tensor.dtype(),
      platform::errors::InvalidArgument(
          "Input tensor and output tensor should be same dtype."));

  PADDLE_ENFORCE_EQ(
      out_tensor.numel() * size_,
      in_tensor.numel(),
      platform::errors::InvalidArgument("input tensor must be the same size as "
                                        "output tensor size times world_size"));

  auto inputs = std::vector<phi::DenseTensor>{in_tensor};
  auto outputs = std::vector<phi::DenseTensor>{out_tensor};

  return Collective(
      inputs,
      outputs,
      [&](phi::DenseTensor& input,
          phi::DenseTensor& output,
          ncclComm_t comm,
          const gpuStream_t& stream) {
        if (FLAGS_use_stream_safe_cuda_allocator) {
          platform::CUDADeviceGuard cuda_guard;
          cuda_guard.SetDevice(output.place());
          memory::RecordStream(output.Holder(), stream);
        }
        PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclReduceScatter(
            input.data(),
            output.data(),
            output.numel(),
            platform::ToNCCLDataType(input.dtype()),
            ToNCCLRedType(opts.reduce_op),
            comm,
            stream));
      },
      CommType::REDUCE_SCATTER);
}

void ProcessGroupNCCL::GroupStart() {
  PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclGroupStart());
}

void ProcessGroupNCCL::GroupEnd() {
  PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclGroupEnd());
}

ncclComm_t ProcessGroupNCCL::NCCLComm(const Place& place) const {
  std::vector<Place> places = {place};
  const auto& iter = places_to_ncclcomm_.find(GetKeyFromPlaces(places));
  PADDLE_ENFORCE_NE(iter,
                    places_to_ncclcomm_.end(),
                    platform::errors::InvalidArgument(
                        "Cannot find nccl comm in process group."));
  return iter->second[0]->GetNcclComm();
}

}  //  namespace distributed
}  //  namespace paddle
