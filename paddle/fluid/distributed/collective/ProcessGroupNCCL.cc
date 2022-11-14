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
#include "paddle/fluid/distributed/collective/utils.h"
#include "paddle/fluid/platform/device/gpu/nccl_helper.h"
#include "paddle/fluid/platform/place.h"
#include "paddle/phi/api/lib/utils/allocator.h"

DECLARE_bool(nccl_blocking_wait);
DECLARE_bool(use_stream_safe_cuda_allocator);

constexpr int64_t kWaitBlockTImeout = 10;

namespace paddle {
namespace distributed {

ProcessGroupNCCL::NCCLTask::NCCLTask(const Place& place,
                                     int rank,
                                     CommType comm_type,
                                     bool sync_op,
                                     bool use_calc_stream)
    : TaskStream(rank, comm_type, sync_op, use_calc_stream),
      comm_event_(place),
      task_place_(place) {}

ProcessGroupNCCL::NCCLTask::~NCCLTask() {}

bool ProcessGroupNCCL::NCCLTask::IsCompleted() { return comm_event_.Query(); }

void ProcessGroupNCCL::NCCLTask::UpdateWaitChain(
    const phi::DeviceContext& ctx) {
  comm_event_.Record(&ctx);
}

// TODO(sheniang03): Add timeout for wait, now timeout unused
bool ProcessGroupNCCL::NCCLTask::Wait(std::chrono::milliseconds timeout) {
  // Warning here when use calc stream but also invoke waiting explicitly.
  if (UseCalcStream()) {
    VLOG(3) << "Warning: The communication is on calc stream, wait here is "
               "useless.";
    return true;
  }

  const auto* calc_ctx =
      platform::DeviceContextPool::Instance().Get(task_place_);
  comm_event_.Wait(platform::Place2DeviceType(task_place_), calc_ctx);

  if (FLAGS_nccl_blocking_wait) {
    // NOTE(shenliang03): It will block host for sync
    while (!IsCompleted()) {
      std::this_thread::sleep_for(std::chrono::milliseconds(kWaitBlockTImeout));
    }
  }

  if (IsBlockCPUInWait()) {
    // If we use the work to do barrier, we should block cpu
#ifdef PADDLE_WITH_CUDA
    PADDLE_ENFORCE_GPU_SUCCESS(cudaDeviceSynchronize());
#else
    PADDLE_ENFORCE_GPU_SUCCESS(hipDeviceSynchronize());
#endif
  }
  return true;
}

// Same as Wait
void ProcessGroupNCCL::NCCLTask::Synchronize() { Wait(kWaitTimeout); }

ProcessGroupNCCL::ProcessGroupNCCL(const std::shared_ptr<Store>& store,
                                   int rank,
                                   int size,
                                   int gid)
    : ProcessGroupStream(rank, size, gid), store_(store) {}

void ProcessGroupNCCL::GroupStart() {
  PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclGroupStart());
}

void ProcessGroupNCCL::GroupEnd() {
  PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclGroupEnd());
}

const phi::DeviceContext& ProcessGroupNCCL::GetDeviceContext(
    const Place& place) const {
  return GetDeviceContext(place, /*use_calc_stream*/ false);
}

const phi::DeviceContext& ProcessGroupNCCL::GetDeviceContext(
    const Place& place, bool use_calc_stream) const {
  const std::string& key = GetKeyFromPlace(place);
  if (use_calc_stream) {
    const auto& iter = place_to_calc_ctx_.find(key);
    return *iter->second;
  } else {
    const auto& iter = place_to_comm_ctx_.find(key);
    PADDLE_ENFORCE_NE(
        iter,
        place_to_comm_ctx_.end(),
        platform::errors::NotFound(
            "Cannot find the device context in this process group."));
    return *iter->second;
  }
}

ncclComm_t ProcessGroupNCCL::NCCLComm(const Place& place) const {
  const std::string& key = GetKeyFromPlace(place);
  const auto& iter = place_to_comm_ctx_.find(key);
  PADDLE_ENFORCE_NE(
      iter,
      place_to_comm_ctx_.end(),
      platform::errors::NotFound(
          "Cannot find the NCCL commmunicator in this process group."));
  return iter->second->nccl_comm();
}

std::shared_ptr<ProcessGroup::Task> ProcessGroupNCCL::AllGather(
    phi::DenseTensor* out_tensor,
    const phi::DenseTensor& in_tensor,
    int64_t offset,
    int64_t numel,
    bool sync_op,
    bool use_calc_stream) {
  // numel > 0 indicates the tensor need to be sliced
  const phi::DenseTensor& in_tensor_maybe_partial =
      numel > 0 ? GetPartialTensor(in_tensor, offset, numel) : in_tensor;
  return Collective(
      out_tensor,
      in_tensor_maybe_partial,
      [](phi::DenseTensor* output,
         const phi::DenseTensor& input,
         ncclComm_t comm,
         gpuStream_t stream) {
        return platform::dynload::ncclAllGather(
            input.data(),
            output->data(),
            input.numel(),
            platform::ToNCCLDataType(input.dtype()),
            comm,
            stream);
      },
      CommType::ALLGATHER,
      sync_op,
      use_calc_stream);
}

std::shared_ptr<ProcessGroup::Task> ProcessGroupNCCL::AllReduce(
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
          ncclComm_t comm,
          gpuStream_t stream) {
        return platform::dynload::ncclAllReduce(
            input.data(),
            output->data(),
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

std::shared_ptr<ProcessGroup::Task> ProcessGroupNCCL::Barrier(
    const BarrierOptions& opts) {
  PADDLE_ENFORCE_GE(opts.device_id,
                    0,
                    platform::errors::PreconditionNotMet(
                        "The barrier device id must greater or equal than 0."));
  platform::CUDAPlace place(opts.device_id);
  auto allocator = std::unique_ptr<phi::Allocator>(
      new paddle::experimental::DefaultAllocator(place));
  phi::DenseTensorMeta meta(phi::DataType::FLOAT32, phi::DDim{1});
  phi::DenseTensor barrier_tensor{allocator.get(), meta};

  auto task = AllReduce(&barrier_tensor,
                        barrier_tensor,
                        {},
                        /*sync_op*/ true,
                        /*use_calc_stream*/ false);
  auto nccl_task = dynamic_cast<NCCLTask*>(task.get());
  nccl_task->SetBlockCPUInWait();
  return task;
}

std::shared_ptr<ProcessGroup::Task> ProcessGroupNCCL::Broadcast(
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
          ncclComm_t comm,
          gpuStream_t stream) {
        int root = opts.source_rank + opts.source_root;
        return platform::dynload::ncclBroadcast(
            input.data(),
            output->data(),
            input.numel(),
            platform::ToNCCLDataType(input.type()),
            root,
            comm,
            stream);
      },
      CommType::BROADCAST,
      sync_op,
      use_calc_stream);
}

std::shared_ptr<ProcessGroup::Task> ProcessGroupNCCL::Recv(
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
  return PointToPoint(
      tensor,
      src_rank,
      [](phi::DenseTensor* output,
         int src,
         ncclComm_t comm,
         gpuStream_t stream) {
        return platform::dynload::ncclRecv(
            output->data(),
            output->numel(),
            platform::ToNCCLDataType(output->dtype()),
            src,
            comm,
            stream);
      },
      CommType::RECV,
      sync_op,
      use_calc_stream);
}

std::shared_ptr<ProcessGroup::Task> ProcessGroupNCCL::Send(
    phi::DenseTensor* tensor,
    int dst_rank,
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
  return PointToPoint(
      tensor,
      dst_rank,
      [](phi::DenseTensor* input,
         int dst,
         ncclComm_t comm,
         gpuStream_t stream) {
        return platform::dynload::ncclSend(
            input->data(),
            input->numel(),
            platform::ToNCCLDataType(input->dtype()),
            dst,
            comm,
            stream);
      },
      CommType::SEND,
      sync_op,
      use_calc_stream);
}

std::shared_ptr<ProcessGroupNCCL::NCCLTask> ProcessGroupNCCL::CreateTask(
    const Place& place,
    int rank,
    CommType comm_type,
    bool is_sync,
    bool use_calc_stream) {
  return std::make_shared<ProcessGroupNCCL::NCCLTask>(
      place, rank, comm_type, is_sync, use_calc_stream);
}

void ProcessGroupNCCL::BroadcastUniqueNCCLID(ncclUniqueId* nccl_id) {
  const std::string key =
      "ProcessGroupNCCL/nccl_ids/" + std::to_string(gid_) + "/0";
  if (rank_ == 0) {
    std::vector<uint8_t> nccl_id_wrapper(
        reinterpret_cast<uint8_t*>(nccl_id),
        reinterpret_cast<uint8_t*>(nccl_id) + NCCL_UNIQUE_ID_BYTES);
    store_->set(key, nccl_id_wrapper);
  } else {
    const auto& nccl_id_wrapper = store_->get(key);
    std::memcpy(nccl_id, nccl_id_wrapper.data(), nccl_id_wrapper.size());
  }
}

void ProcessGroupNCCL::CreateNCCLEnvCache(const Place& place,
                                          const std::string& place_key) {
  if (place_to_comm_ctx_.size() > 0) {
    VLOG(3) << "Warning: Tensors from multiple devices are not supported yet.";
  }

  ncclUniqueId nccl_id;
  if (rank_ == 0) {
    PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclGetUniqueId(&nccl_id));
  }
  BroadcastUniqueNCCLID(&nccl_id);

  VLOG(3) << "init nccl rank: " << rank_ << ", nranks: " << size_
          << ", place: " << place_key
          << ", nccl uniqueid: " << SerializeNCCLUniqueId(nccl_id);

  auto* calc_ctx = static_cast<phi::GPUContext*>(
      platform::DeviceContextPool::Instance().Get(place));
  auto comm_ctx = std::make_unique<phi::GPUContext>(place);
  ncclComm_t nccl_comm;
  NCCLCHECK(platform::dynload::ncclCommInitRank(
      &nccl_comm, GetSize(), nccl_id, GetRank()));
  comm_ctx->set_nccl_comm(nccl_comm);

  place_to_calc_event_.emplace(place_key, place);
  place_to_calc_ctx_.emplace(place_key, calc_ctx);
  place_to_comm_ctx_.emplace(place_key, std::move(comm_ctx));

  // TODO(sunyilun): for compatibility, will be removed later
  std::vector<phi::GPUContext*> comm_ctx_wrapper{
      place_to_comm_ctx_[place_key].get()};
  places_to_ctx_.emplace(place_key, comm_ctx_wrapper);
}

void ProcessGroupNCCL::SyncCalcStream(const Place& place) {
  const std::string& key = GetKeyFromPlace(place);
  auto& calc_event = place_to_calc_event_.at(key);
  const auto* calc_ctx = place_to_calc_ctx_.at(key);
  const auto* comm_ctx = place_to_comm_ctx_.at(key).get();
  calc_event.Record(calc_ctx);
  calc_event.Wait(platform::Place2DeviceType(place), comm_ctx);
}

template <typename Fn>
std::shared_ptr<ProcessGroup::Task> ProcessGroupNCCL::Collective(
    phi::DenseTensor* out_tensor,
    const phi::DenseTensor& in_tensor,
    Fn fn,
    CommType comm_type,
    bool sync_op,
    bool use_calc_stream) {
  const auto& place = in_tensor.place();
  const auto& key = GetKeyFromPlace(place);

  platform::CUDADeviceGuard cuda_guard(place);

  if (place_to_comm_ctx_.find(key) == place_to_comm_ctx_.end()) {
    CreateNCCLEnvCache(place, key);
  }

  if (!use_calc_stream) {
    SyncCalcStream(place);
  }

  auto task = CreateTask(place, rank_, comm_type, sync_op, use_calc_stream);

  const auto* calc_ctx = place_to_calc_ctx_.at(key);
  const auto& comm_ctx = place_to_comm_ctx_.at(key);
  auto nccl_comm = comm_ctx->nccl_comm();
  auto nccl_stream = use_calc_stream ? calc_ctx->stream() : comm_ctx->stream();
  fn(out_tensor, in_tensor, nccl_comm, nccl_stream);

  if (!use_calc_stream) {
    if (FLAGS_use_stream_safe_cuda_allocator) {
      memory::RecordStream(in_tensor.Holder(), nccl_stream);
    }
    task->UpdateWaitChain(*comm_ctx);
  }

  return task;
}

template <typename Fn>
std::shared_ptr<ProcessGroup::Task> ProcessGroupNCCL::PointToPoint(
    phi::DenseTensor* tensor,
    int rank,
    Fn fn,
    CommType comm_type,
    bool sync_op,
    bool use_calc_stream) {
  const auto& place = tensor->place();
  const auto& key = GetKeyFromPlace(place);

  platform::CUDADeviceGuard cuda_guard(place);

  if (place_to_comm_ctx_.find(key) == place_to_comm_ctx_.end()) {
    CreateNCCLEnvCache(place, key);
  }

  if (!use_calc_stream) {
    SyncCalcStream(place);
  }

  auto task = CreateTask(place, rank_, comm_type, sync_op, use_calc_stream);

  const auto* calc_ctx = place_to_calc_ctx_.at(key);
  const auto& comm_ctx = place_to_comm_ctx_.at(key);
  auto nccl_comm = comm_ctx->nccl_comm();
  auto nccl_stream = use_calc_stream ? calc_ctx->stream() : comm_ctx->stream();
  fn(tensor, rank, nccl_comm, nccl_stream);

  if (!use_calc_stream) {
    if (FLAGS_use_stream_safe_cuda_allocator) {
      memory::RecordStream(tensor->Holder(), nccl_stream);
    }
    task->UpdateWaitChain(*comm_ctx);
  }

  return task;
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

// TODO(sunyilun): methods below will be removed later
void SyncDefaultStream(const std::vector<Place>& places,
                       platform::DeviceEvent& nccl_event,         // NOLINT
                       std::vector<phi::GPUContext*>& dev_ctx) {  // NOLINT
  for (size_t i = 0; i < places.size(); ++i) {
    auto* default_ctx = static_cast<phi::GPUContext*>(
        platform::DeviceContextPool::Instance().Get(places[i]));
    nccl_event.Record(default_ctx);
    nccl_event.Wait(platform::Place2DeviceType(places[i]), dev_ctx[i]);
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

std::shared_ptr<ProcessGroupNCCL::NCCLTask> ProcessGroupNCCL::CreateTask(
    const std::vector<Place>& places,
    int rank,
    CommType comm_type,
    const std::vector<phi::DenseTensor>& inputs,
    bool is_sync,
    bool use_calc_stream) {
  return std::make_shared<ProcessGroupNCCL::NCCLTask>(
      places, rank, comm_type, inputs, is_sync, use_calc_stream);
}

ProcessGroupNCCL::NCCLTask::NCCLTask(
    const std::vector<Place>& places,
    int rank,
    CommType CommType,
    const std::vector<phi::DenseTensor>& inputs)
    : TaskStream(rank, inputs, CommType),
      comm_event_(places[0]),
      task_place_(places[0]) {}

ProcessGroupNCCL::NCCLTask::NCCLTask(
    const std::vector<Place>& places,
    int rank,
    CommType comm_type,
    const std::vector<phi::DenseTensor>& inputs,
    bool sync_op,
    bool use_calc_stream)
    : TaskStream(rank, inputs, comm_type, sync_op, use_calc_stream),
      comm_event_(places[0]),
      task_place_(places[0]) {}

// create NCCLManager cache for places_key
void ProcessGroupNCCL::CreateNCCLManagerCache(
    const std::string& places_key, const std::vector<Place>& places) {
  PADDLE_ENFORCE_EQ(places_key.empty(),
                    false,
                    platform::errors::PreconditionNotMet(
                        "Not able to create/get the NCCL Communicator since "
                        "the GPU place are not known"));

  ncclUniqueId nccl_id;
  if (rank_ == 0) {
    PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclGetUniqueId(&nccl_id));
  }
  BroadcastUniqueNCCLID(&nccl_id);

  VLOG(3) << "init nccl rank: " << rank_ << ", nranks: " << size_
          << ", place: " << places_key
          << ", nccl uniqueid: " << SerializeNCCLUniqueId(nccl_id);

  std::vector<std::unique_ptr<phi::GPUContext>> dev_ctx;
  dev_ctx.resize(places.size());

  std::vector<phi::GPUContext*> dev_ctx_raw;
  dev_ctx_raw.resize(places.size());

  PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclGroupStart());

  for (size_t i = 0; i < places.size(); ++i) {
    platform::CUDADeviceGuard guard(places[i]);

    dev_ctx[i].reset(new phi::GPUContext(places[i]));
    ncclComm_t nccl_comm;
    NCCLCHECK(platform::dynload::ncclCommInitRank(
        &nccl_comm, GetSize(), nccl_id, GetRank()));
    dev_ctx[i]->set_nccl_comm(nccl_comm);
    dev_ctx_raw[i] = dev_ctx[i].get();
  }

  PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclGroupEnd());

  // TODO(sunyilun): for compatibility, will be removed later
  place_to_calc_event_.emplace(places_key, places[0]);
  place_to_calc_ctx_.emplace(
      places_key,
      static_cast<phi::GPUContext*>(
          platform::DeviceContextPool::Instance().Get(places[0])));
  place_to_comm_ctx_.emplace(places_key, std::move(dev_ctx[0]));

  // These caches will be useful to process sync/wait/communicate
  places_to_ctx_.emplace(places_key, std::move(dev_ctx_raw));
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
    if (place_to_comm_ctx_.find(key) == place_to_comm_ctx_.end()) {
      CreateNCCLManagerCache(key, places);
    }
  }

  if (!use_calc_stream) {
    SyncDefaultStream(
        places, place_to_calc_event_.at(key), places_to_ctx_.at(key));
  }

  auto task =
      CreateTask(places, rank_, comm_type, inputs, sync_op, use_calc_stream);

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
        nccl_stream = places_to_ctx_.at(key)[i]->stream();
      }

      fn(inputs[i],
         outputs[i],
         places_to_ctx_.at(key)[i]->nccl_comm(),
         nccl_stream);
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
        nccl_stream = places_to_ctx_.at(key)[i]->stream();
      }

      memory::RecordStream(inputs[i].Holder(), nccl_stream);
    }
  }

  // Adding stream event dependency only when use comm stream
  if (!use_calc_stream) {
    for (size_t i = 0; i < inputs.size(); ++i) {
      cuda_guard.SetDevice(places[i]);
      task->UpdateWaitChain(*places_to_ctx_.at(key)[i]);
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
    if (place_to_comm_ctx_.find(key) == place_to_comm_ctx_.end()) {
      CreateNCCLManagerCache(key, places);
    }
  }

  SyncDefaultStream(
      places, place_to_calc_event_.at(key), places_to_ctx_.at(key));

  auto task = CreateTask(places, rank_, op_type, inputs);

  // construct uninitialize guard for device
  platform::CUDADeviceGuard cuda_guard;

  {
    platform::NCCLGroupGuard nccl_guard;
    for (size_t i = 0; i < inputs.size(); ++i) {
      cuda_guard.SetDevice(places[i]);
      const auto& nccl_stream = places_to_ctx_.at(key)[i]->stream();
      fn(inputs[i],
         outputs[i],
         places_to_ctx_.at(key)[i]->nccl_comm(),
         nccl_stream);
    }
  }

  if (FLAGS_use_stream_safe_cuda_allocator) {
    for (size_t i = 0; i < inputs.size(); ++i) {
      cuda_guard.SetDevice(places[i]);
      memory::RecordStream(inputs[i].Holder(),
                           places_to_ctx_.at(key)[i]->stream());
    }
  }

  for (size_t i = 0; i < inputs.size(); ++i) {
    cuda_guard.SetDevice(places[i]);
    task->UpdateWaitChain(*places_to_ctx_.at(key)[i]);
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
  const std::string& key = GetKeyFromPlaces(places);

  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (place_to_comm_ctx_.find(key) == place_to_comm_ctx_.end()) {
      CreateNCCLManagerCache(key, places);
    }
  }

  SyncDefaultStream(
      places, place_to_calc_event_.at(key), places_to_ctx_.at(key));

  // construct uninitialize guard for device
  platform::CUDADeviceGuard cuda_guard;

  if (FLAGS_use_stream_safe_cuda_allocator) {
    cuda_guard.SetDevice(places[0]);
    memory::RecordStream(in->Holder(), places_to_ctx_.at(key)[0]->stream());
  }

  {
    platform::NCCLGroupGuard nccl_guard;
    cuda_guard.SetDevice(places[0]);
    const auto& nccl_stream = places_to_ctx_.at(key)[0]->stream();
    fn(in, out, places_to_ctx_.at(key)[0]->nccl_comm(), nccl_stream);
  }

  cuda_guard.SetDevice(places[0]);
}

template <typename Fn>
std::shared_ptr<ProcessGroup::Task> ProcessGroupNCCL::PointToPoint(
    std::vector<phi::DenseTensor>& tensors,
    Fn fn,
    int dst_rank,
    CommType op_type,
    bool sync_op,
    bool use_calc_stream) {
  const auto& places = GetPlaceList(tensors);
  const auto& key = GetKeyFromPlaces(places);

  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (place_to_comm_ctx_.find(key) == place_to_comm_ctx_.end()) {
      CreateNCCLManagerCache(key, places);
    }
  }

  if (!use_calc_stream) {
    SyncDefaultStream(
        places, place_to_calc_event_.at(key), places_to_ctx_.at(key));
  }

  auto task =
      CreateTask(places, rank_, op_type, tensors, sync_op, use_calc_stream);

  platform::CUDADeviceGuard cuda_guard;

  {
    platform::NCCLGroupGuard nccl_guard;
    for (size_t i = 0; i < tensors.size(); ++i) {
      cuda_guard.SetDevice(places[i]);
      gpuStream_t nccl_stream;
      if (use_calc_stream) {
        nccl_stream =
            static_cast<phi::GPUContext*>(
                platform::DeviceContextPool::Instance().Get(places[i]))
                ->stream();
      } else {
        nccl_stream = places_to_ctx_.at(key)[i]->stream();
      }
      fn(tensors[i],
         places_to_ctx_.at(key)[i]->nccl_comm(),
         nccl_stream,
         dst_rank);
    }
  }

  if (FLAGS_use_stream_safe_cuda_allocator) {
    for (size_t i = 0; i < tensors.size(); ++i) {
      cuda_guard.SetDevice(places[i]);
      gpuStream_t nccl_stream;
      if (use_calc_stream) {
        nccl_stream =
            static_cast<phi::GPUContext*>(
                platform::DeviceContextPool::Instance().Get(places[i]))
                ->stream();
      } else {
        nccl_stream = places_to_ctx_.at(key)[i]->stream();
      }
      memory::RecordStream(tensors[i].Holder(), nccl_stream);
    }
  }

  if (!use_calc_stream) {
    for (size_t i = 0; i < tensors.size(); ++i) {
      cuda_guard.SetDevice(places[i]);
      task->UpdateWaitChain(*places_to_ctx_.at(key)[i]);
    }
  }

  return task;
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
    if (place_to_comm_ctx_.find(key) == place_to_comm_ctx_.end()) {
      CreateNCCLManagerCache(key, places);
    }
  }

  SyncDefaultStream(
      places, place_to_calc_event_.at(key), places_to_ctx_.at(key));

  auto task = CreateTask(places, rank_, op_type, tensors);

  // construct uninitialize guard for device
  platform::CUDADeviceGuard cuda_guard;

  {
    platform::NCCLGroupGuard nccl_guard;
    for (size_t i = 0; i < tensors.size(); ++i) {
      cuda_guard.SetDevice(places[i]);
      const auto& nccl_stream = places_to_ctx_.at(key)[i]->stream();
      fn(tensors[i],
         places_to_ctx_.at(key)[i]->nccl_comm(),
         nccl_stream,
         dst_rank);
    }
  }

  if (FLAGS_use_stream_safe_cuda_allocator) {
    for (size_t i = 0; i < tensors.size(); ++i) {
      cuda_guard.SetDevice(places[i]);
      memory::RecordStream(tensors[i].Holder(),
                           places_to_ctx_.at(key)[i]->stream());
    }
  }

  for (size_t i = 0; i < tensors.size(); ++i) {
    cuda_guard.SetDevice(places[i]);
    task->UpdateWaitChain(*places_to_ctx_.at(key)[i]);
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
  } else if (type == experimental::DataType::FLOAT16) {
    return reinterpret_cast<void*>(reinterpret_cast<int16_t*>(raw_pointer) +
                                   offset);
  } else if (type == experimental::DataType::INT32) {
    return reinterpret_cast<void*>(reinterpret_cast<int32_t*>(raw_pointer) +
                                   offset);
  } else if (type == experimental::DataType::INT64) {
    return reinterpret_cast<void*>(reinterpret_cast<int64_t*>(raw_pointer) +
                                   offset);
  } else if (type == experimental::DataType::INT8) {
    return reinterpret_cast<void*>(reinterpret_cast<int8_t*>(raw_pointer) +
                                   offset);
  } else if (type == experimental::DataType::UINT8) {
    return reinterpret_cast<void*>(reinterpret_cast<uint8_t*>(raw_pointer) +
                                   offset);
  } else if (type == experimental::DataType::BOOL) {
    return reinterpret_cast<void*>(reinterpret_cast<bool*>(raw_pointer) +
                                   offset);
  } else if (type == experimental::DataType::BFLOAT16) {
    return reinterpret_cast<void*>(reinterpret_cast<uint16_t*>(raw_pointer) +
                                   offset);
  } else {
    PADDLE_THROW(platform::errors::Unimplemented(
        "Datatype %s in NCCL is not supported.", type));
  }
  return nullptr;
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

std::shared_ptr<ProcessGroup::Task> ProcessGroupNCCL::AllToAll(
    std::vector<phi::DenseTensor>& in_tensors,
    std::vector<phi::DenseTensor>& out_tensors,
    bool sync_op,
    bool use_calc_stream) {
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
      CommType::ALLTOALL,
      sync_op,
      use_calc_stream);
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

std::shared_ptr<ProcessGroup::Task> ProcessGroupNCCL::AllToAllSingle(
    std::vector<phi::DenseTensor>& in_tensors,
    std::vector<phi::DenseTensor>& out_tensors,
    std::vector<int64_t>& in_sizes,
    std::vector<int64_t>& out_sizes,
    bool sync_op,
    bool use_calc_stream) {
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
      CommType::ALLTOALL_SINGLE,
      sync_op,
      use_calc_stream);
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

std::shared_ptr<ProcessGroup::Task> ProcessGroupNCCL::Reduce(
    std::vector<phi::DenseTensor>& in_tensors,
    std::vector<phi::DenseTensor>& out_tensors,
    const ReduceOptions& opts,
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
      CommType::REDUCE,
      sync_op,
      use_calc_stream);
}

std::shared_ptr<ProcessGroup::Task> ProcessGroupNCCL::ReduceScatter(
    std::vector<phi::DenseTensor>& in_tensors,
    std::vector<phi::DenseTensor>& out_tensors,
    const ReduceScatterOptions& opts,
    bool sync_op,
    bool use_calc_stream) {
  return Collective(
      in_tensors,
      out_tensors,
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
      CommType::REDUCE_SCATTER,
      sync_op,
      use_calc_stream);
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

std::shared_ptr<ProcessGroup::Task> ProcessGroupNCCL::Scatter(
    std::vector<phi::DenseTensor>& in_tensors,
    std::vector<phi::DenseTensor>& out_tensors,
    const ScatterOptions& opts,
    bool sync_op,
    bool use_calc_stream) {
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
        PADDLE_ENFORCE_EQ(
            output.numel(),
            input.numel() / size_,
            platform::errors::InvalidArgument(
                "Input and output tensors should have the same shape."));
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
      CommType::SCATTER,
      sync_op,
      use_calc_stream);
}

}  //  namespace distributed
}  //  namespace paddle
