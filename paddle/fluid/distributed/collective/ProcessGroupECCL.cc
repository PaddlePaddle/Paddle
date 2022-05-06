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

#include "paddle/fluid/distributed/collective/ProcessGroupECCL.h"
#include <string>
#include "paddle/fluid/distributed/collective/Common.h"
#include "paddle/fluid/platform/device/gpu/eccl_helper.h"
#include "paddle/fluid/platform/place.h"
#include "paddle/phi/api/include/api.h"
#include "paddle/phi/common/place.h"
#include

DECLARE_bool(eccl_blocking_wait);
DECLARE_bool(use_stream_safe_cuda_allocator);

constexpr int64_t kWaitBlockTImeout = 10;

namespace paddle {
namespace distributed {

void SyncDefaultStream(
    const std::vector<Place>& places,
    std::vector<EventManager>& ecclEvents,                       // NOLINT
    std::vector<std::unique_ptr<CUDADeviceContext>>& dev_ctx) {  // NOLINT
  for (size_t i = 0; i < places.size(); ++i) {
    auto* default_ctx = static_cast<platform::CUDADeviceContext*>(
        platform::DeviceContextPool::Instance().Get(places[i]));
    ecclEvents[i].Record(*default_ctx);
    ecclEvents[i].Block(*dev_ctx[i]);
  }
}

std::shared_ptr<ProcessGroupECCL::ECCLTask> ProcessGroupECCL::CreateTask(
    std::vector<Place> places, int rank, CommType comm_type,
    const std::vector<phi::DenseTensor>& inputs) {
  return std::make_shared<ProcessGroupECCL::ECCLTask>(places, rank, comm_type,
                                                      inputs);
}

ProcessGroupECCL::ECCLTask::ECCLTask(
    const std::vector<Place>& places, int rank, CommType CommType,
    const std::vector<phi::DenseTensor>& inputs)
    : Task(rank, inputs, CommType), places_(places) {
  control_events_.resize(places.size());
  ecclComms_.resize(places.size());
}

ProcessGroupECCL::ECCLTask::~ECCLTask() {}

void ProcessGroupECCL::ECCLTask::SetOutputs(
    std::vector<phi::DenseTensor>& outputs) {  // NOLINT
  outputs_ = std::make_shared<std::vector<phi::DenseTensor>>(outputs);
}

void ProcessGroupECCL::ECCLTask::SynchronizeStreams() {
  for (size_t i = 0; i < places_.size(); ++i) {
    auto* default_ctx = static_cast<platform::CUDADeviceContext*>(
        platform::DeviceContextPool::Instance().Get(places_[i]));
    default_ctx->WaitEvent(control_events_[i].GetRawCudaEvent());
  }
}

bool ProcessGroupECCL::ECCLTask::IsCompleted() {
  for (size_t i = 0; i < places_.size(); ++i) {
    if (!control_events_[i].Query()) {
      return false;
    }
  }

  return true;
}

bool ProcessGroupECCL::ECCLTask::Wait(std::chrono::milliseconds timeout) {
  SynchronizeStreams();
  if (FLAGS_eccl_blocking_wait) {
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
void ProcessGroupECCL::ECCLTask::Synchronize() { Wait(kWaitTimeout); }

ProcessGroupECCL::ProcessGroupECCL(const std::shared_ptr<Store>& store,
                                   int rank, int size, int gid)
    : ProcessGroup(rank, size, gid), store_(store) {}

// create ECCLManager cache for places_key
void ProcessGroupECCL::CreateECCLManagerCache(
    const std::string& places_key, const std::vector<Place>& places) {
  PADDLE_ENFORCE_EQ(places_key.empty(), false,
                    platform::errors::PreconditionNotMet(
                        "Not able to create/get the ECCL Communicator since "
                        "the GPU place are not known"));

  std::vector<std::shared_ptr<ECCLCommManager>> eccl_comms;
  eccl_comms.resize(places.size());

  // using vector just for broadcast
  std::vector<EcclCommGroupIdType> eccl_ids;
  eccl_ids.resize(1);
  auto& eccl_id = eccl_ids.front();

  for (auto& place : places) {
    used_place_ids_.insert(place.GetDeviceId());
  }

  std::vector<std::unique_ptr<CUDADeviceContext>> dev_ctx;
  dev_ctx.resize(places.size());
  std::string bootstrap_endpoint = std::string(getenv("BOOTSTRAP_ENDPOINT"));
  if (str.empty()) {
    bootstrap_endpoint = "127.0.0.1:9999"
  }

  for (size_t i = 0; i < places.size(); ++i) {
    platform::CUDADeviceGuard guard(places[i]);
    // split_index, bootstrap_endpoint没有
    ECCL_ENSURE_SUCCESS(
        eccl_gen_unique_id(GetRank(), bootstrap_endpoint.c_str(), GetSize(), 0,
                           eccl_comms[i]->GetEcclComm()));
    // my_device_type is not define!
    ECCL_ENSURE_SUCCESS(eccl_init_comm_global(GetSize(), GetRank(), NVIDIA_GPU,
                                              used_place_ids_[i],
                                              eccl_comms[i]->GetEcclComm()));

    dev_ctx[i].reset(new CUDADeviceContext(places[i]));
  }

  std::vector<EventManager> events;
  events.resize(places.size());

  // These caches will be useful to process sync/wait/communicate
  places_to_events_.emplace(places_key, std::move(events));
  places_to_ecclcomm_.emplace(places_key, std::move(eccl_comms));
  places_to_ctx_.emplace(places_key, std::move(dev_ctx));
}

template <typename Fn>
std::shared_ptr<ProcessGroup::Task> ProcessGroupECCL::Collective(
    std::vector<phi::DenseTensor>& inputs,
    std::vector<phi::DenseTensor>& outputs, Fn fn, CommType op_type) {
  const auto places = GetPlaceList(inputs);
  const auto key = GetKeyFromPlaces(places);

  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (places_to_ecclcomm_.find(key) == places_to_ecclcomm_.end()) {
      CreateECCLManagerCache(key, places);
    }
  }

  auto& eccl_comms = places_to_ecclcomm_[key];

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
    platform::ECCLGroupGuard eccl_guard;
    for (size_t i = 0; i < inputs.size(); ++i) {
      cuda_guard.SetDevice(places[i]);
      const auto& eccl_stream = places_to_ctx_[key][i]->stream();
      fn(inputs[i], outputs[i], eccl_comms[i]->GetEcclComm(), eccl_stream);
    }
  }

  for (size_t i = 0; i < inputs.size(); ++i) {
    cuda_guard.SetDevice(places[i]);
    task->control_events_[i].Record(*places_to_ctx_[key][i]);
  }
  return task;
}

template <typename Fn>
std::shared_ptr<ProcessGroup::Task> ProcessGroupECCL::PointToPoint(
    std::vector<phi::DenseTensor>& tensors, Fn fn, int dst_rank,
    CommType op_type) {
  const auto places = GetPlaceList(tensors);
  const auto key = GetKeyFromPlaces(places);

  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (places_to_ecclcomm_.find(key) == places_to_ecclcomm_.end()) {
      CreateECCLManagerCache(key, places);
    }
  }

  auto& eccl_comms = places_to_ecclcomm_[key];

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
    platform::ECCLGroupGuard eccl_guard;
    for (size_t i = 0; i < tensors.size(); ++i) {
      cuda_guard.SetDevice(places[i]);
      const auto& eccl_stream = places_to_ctx_[key][i]->stream();
      fn(tensors[i], eccl_comms[i]->GetEcclComm(), eccl_stream, dst_rank);
    }
  }

  for (size_t i = 0; i < tensors.size(); ++i) {
    cuda_guard.SetDevice(places[i]);
    task->control_events_[i].Record(*places_to_ctx_[key][i]);
  }
  return task;
}

std::shared_ptr<ProcessGroup::Task> ProcessGroupECCL::AllReduce(
    std::vector<phi::DenseTensor>& tensors, const AllreduceOptions& opts) {
  PADDLE_ENFORCE_EQ(
      CheckTensorsInCudaPlace(tensors), true,
      platform::errors::InvalidArgument("All inputs should be in CudaPlace."));
  return Collective(
      tensors, tensors,
      [&](const Tensor& input, Tensor& output, ecclComm_t comm,
          const gpuStream_t& stream) {
        auto input_tensor =
            std::dynamic_pointer_cast<phi::DenseTensor>(input.impl());
        auto output_tensor =
            std::dynamic_pointer_cast<phi::DenseTensor>(output.impl());
        return platform::dynload::ecclAllReduce(
            input_tensor->data(), output_tensor->data(), input_tensor->numel(),
            platform::ToECCLDataType(input.type()),
            ToECCLRedType(opts.reduce_op), comm, stream);
      },
      CommType::ALLREDUCE);
}

std::shared_ptr<ProcessGroup::Task> ProcessGroupECCL::Broadcast(
    std::vector<phi::DenseTensor>& tensors, const BroadcastOptions& opts) {
  PADDLE_ENFORCE_EQ(
      CheckTensorsInCudaPlace(tensors), true,
      platform::errors::InvalidArgument("All inputs should be in CudaPlace."));

  return Collective(
      tensors, tensors,
      [&](Tensor& input, Tensor& output, ecclComm_t comm,
          const gpuStream_t& stream) {
        const auto root = opts.source_rank * tensors.size() + opts.source_root;
        auto input_tensor =
            std::dynamic_pointer_cast<phi::DenseTensor>(input.impl());
        auto output_tensor =
            std::dynamic_pointer_cast<phi::DenseTensor>(output.impl());
        return platform::dynload::ecclBcast(
            input_tensor->data(), input_tensor->numel(),
            platform::ToECCLDataType(input.type()), root, comm, stream);
      },
      CommType::BROADCAST);
}

std::shared_ptr<ProcessGroup::Task> ProcessGroupECCL::Barrier(
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

  std::vector<phi::DenseTensor> barrierTensors;
  barrierTensors.reserve(places.size());

  platform::CUDADeviceGuard gpuGuard;
  for (auto& place : places) {
    gpuGuard.SetDeviceIndex(place.GetDeviceId());
    auto dt = full({1}, 0, phi::DataType::FLOAT32, phi::GPUPlace());
    barrierTensors.push_back(dt);
  }
  auto task = ProcessGroupECCL::AllReduce(barrierTensors);
  auto eccl_task = dynamic_cast<ProcessGroupECCL::ECCLTask*>(task.get());
  eccl_task->barrierTensors_ = std::move(barrierTensors);
  return task;
}

void CheckTensorsInDifferentDevices(
    const std::vector<phi::DenseTensor>& tensors, const size_t num_devices) {
  PADDLE_ENFORCE_EQ(
      tensors.size() == 0, false,
      platform::errors::InvalidArgument("Tensor list must be nonempty."));
  PADDLE_ENFORCE_LE(
      tensors.size(), num_devices,
      platform::errors::InvalidArgument(
          "Tensor list mustn't be larger than the number of available GPUs."));

  std::set<Place> used_devices;

  for (const auto& t : tensors) {
    PADDLE_ENFORCE_EQ(t.is_gpu() && t.is_dense_tensor(), true,
                      platform::errors::InvalidArgument(
                          "Tensors must be CUDA and dense tensor."));

    const auto inserted = used_devices.insert(t.inner_place()).second;
    PADDLE_ENFORCE_EQ(inserted, true,
                      platform::errors::InvalidArgument(
                          "Tensors must be on distinct GPU devices."));
  }
}

std::shared_ptr<ProcessGroup::Task> ProcessGroupECCL::Send(
    std::vector<phi::DenseTensor>& tensors, int dst_rank) {
  CheckTensorsInDifferentDevices(tensors, static_cast<size_t>(GetSize()));

  auto task = PointToPoint(
      tensors,
      [&](Tensor& input, ecclComm_t comm, const gpuStream_t& stream,
          int dst_rank) {
        auto input_tensor =
            std::dynamic_pointer_cast<phi::DenseTensor>(input.impl());
        return platform::dynload::ecclSend(
            input_tensor->data(), input_tensor->numel(),
            platform::ToECCLDataType(input.type()), dst_rank, comm, stream);
      },
      dst_rank, CommType::SEND);
  return task;
}

std::shared_ptr<ProcessGroup::Task> ProcessGroupECCL::Recv(
    std::vector<phi::DenseTensor>& tensors, int src_rank) {
  CheckTensorsInDifferentDevices(tensors, static_cast<size_t>(GetSize()));

  auto task = PointToPoint(
      tensors,
      [&](Tensor& output, ecclComm_t comm, const gpuStream_t& stream,
          int src_rank) {
        auto output_tensor =
            std::dynamic_pointer_cast<phi::DenseTensor>(output.impl());
        return platform::dynload::ecclRecv(
            output_tensor->data(), output_tensor->numel(),
            platform::ToECCLDataType(output.type()), src_rank, comm, stream);
      },
      src_rank, CommType::RECV);
  return task;
}

std::shared_ptr<ProcessGroup::Task> ProcessGroupECCL::AllGather(
    std::vector<phi::DenseTensor>& in_tensors,
    std::vector<phi::DenseTensor>& out_tensors) {
  PADDLE_ENFORCE_EQ(
      CheckTensorsInCudaPlace(in_tensors), true,
      platform::errors::InvalidArgument("All inputs should be in CudaPlace."));
  PADDLE_ENFORCE_EQ(
      CheckTensorsInCudaPlace(out_tensors), true,
      platform::errors::InvalidArgument("All outputs should be in CudaPlace."));
  return Collective(
      in_tensors, out_tensors,
      [&](const Tensor& input, Tensor& output, ecclComm_t comm,
          const gpuStream_t& stream) {
        auto input_tensor =
            std::dynamic_pointer_cast<phi::DenseTensor>(input.impl());
        auto output_tensor =
            std::dynamic_pointer_cast<phi::DenseTensor>(output.impl());
        return platform::dynload::ecclAllGather(
            input_tensor->data(), output_tensor->data(), input_tensor->numel(),
            platform::ToECCLDataType(input.type()), comm, stream);
      },
      CommType::ALLGATHER);
}

void* GetPointerByOffset(void* raw_pointer, size_t offset,
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
        "This datatype in eccl is not supported."));
  }
}

std::shared_ptr<ProcessGroup::Task> ProcessGroupECCL::AllToAll(
    std::vector<phi::DenseTensor>& in_tensors,
    std::vector<phi::DenseTensor>& out_tensors) {
  PADDLE_ENFORCE_EQ(
      CheckTensorsInCudaPlace(in_tensors), true,
      platform::errors::InvalidArgument("All inputs should be in CudaPlace."));
  PADDLE_ENFORCE_EQ(
      CheckTensorsInCudaPlace(out_tensors), true,
      platform::errors::InvalidArgument("All inputs should be in CudaPlace."));
  return Collective(
      in_tensors, out_tensors,
      [&](const Tensor& input, Tensor& output, ecclComm_t comm,
          const gpuStream_t& stream) {
        auto input_tensor =
            std::dynamic_pointer_cast<phi::DenseTensor>(input.impl());
        auto output_tensor =
            std::dynamic_pointer_cast<phi::DenseTensor>(output.impl());
        size_t offset = 0;
        PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ecclGroupStart());
        for (auto i = 0; i < size_; i++) {
          PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ecclSend(
              GetPointerByOffset(input_tensor->data(), offset, input.type()),
              input_tensor->numel() / size_,
              platform::ToECCLDataType(input.type()), i, comm, stream));
          PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ecclRecv(
              GetPointerByOffset(output_tensor->data(), offset, input.type()),
              input_tensor->numel() / size_,
              platform::ToECCLDataType(input.type()), i, comm, stream));
          offset += input_tensor->numel() / size_;
        }
        PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ecclGroupEnd());
      },
      CommType::ALLREDUCE);
}

std::shared_ptr<ProcessGroup::Task> ProcessGroupECCL::Reduce(
    std::vector<phi::DenseTensor>& tensors, const ReduceOptions& opts) {
  PADDLE_ENFORCE_EQ(
      CheckTensorsInCudaPlace(tensors), true,
      platform::errors::InvalidArgument("All inputs should be in CudaPlace."));
  return Collective(
      tensors, tensors,
      [&](const Tensor& input, Tensor& output, ecclComm_t comm,
          const gpuStream_t& stream) {
        auto input_tensor =
            std::dynamic_pointer_cast<phi::DenseTensor>(input.impl());
        auto output_tensor =
            std::dynamic_pointer_cast<phi::DenseTensor>(output.impl());
        PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ecclReduce(
            input_tensor->data(), output_tensor->data(), input.numel(),
            platform::ToECCLDataType(input.type()),
            ToECCLRedType(opts.reduce_op), opts.root_rank, comm, stream));
      },
      CommType::REDUCE);
}

std::shared_ptr<ProcessGroup::Task> ProcessGroupECCL::Scatter(
    std::vector<phi::DenseTensor>& in_tensors,
    std::vector<phi::DenseTensor>& out_tensors, const ScatterOptions& opts) {
  PADDLE_ENFORCE_EQ(
      CheckTensorsInCudaPlace(in_tensors), true,
      platform::errors::InvalidArgument("All inputs should be in CudaPlace."));
  PADDLE_ENFORCE_EQ(
      CheckTensorsInCudaPlace(out_tensors), true,
      platform::errors::InvalidArgument("All inputs should be in CudaPlace."));
  return Collective(
      in_tensors, out_tensors,
      [&](const Tensor& input, Tensor& output, ecclComm_t comm,
          const gpuStream_t& stream) {
        auto input_tensor =
            std::dynamic_pointer_cast<phi::DenseTensor>(input.impl());
        auto output_tensor =
            std::dynamic_pointer_cast<phi::DenseTensor>(output.impl());
        size_t offset = 0;
        if (rank_ == opts.root_rank) {
          for (auto i = 0; i < size_; i++) {
            PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ecclSend(
                GetPointerByOffset(input_tensor->data(), offset, input.type()),
                input_tensor->numel() / size_,
                platform::ToECCLDataType(input.type()), i, comm, stream));
            offset += input_tensor->numel() / size_;
          }
          PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ecclRecv(
              output_tensor->data(), input_tensor->numel() / size_,
              platform::ToECCLDataType(input.type()), opts.root_rank, comm,
              stream));
        } else {
          PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ecclRecv(
              output_tensor->data(), input_tensor->numel() / size_,
              platform::ToECCLDataType(input.type()), opts.root_rank, comm,
              stream));
        }
      },
      CommType::SCATTER);
}

}  //  namespace distributed
}  //  namespace paddle
