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

#include "paddle/fluid/distributed/collective/process_group_nccl.h"

#include "paddle/fluid/distributed/collective/common.h"
#include "paddle/fluid/distributed/collective/nccl_tools.h"
#include "paddle/fluid/distributed/collective/utils.h"
#include "paddle/fluid/platform/cuda_device_guard.h"
#include "paddle/fluid/platform/device/gpu/nccl_helper.h"
#include "paddle/phi/api/lib/utils/allocator.h"
#include "paddle/phi/core/distributed/check/nccl_dynamic_check.h"
#include "paddle/phi/core/distributed/check/static_check.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/flags.h"
#include "paddle/phi/core/utils/data_type.h"

#include "paddle/phi/core/distributed/comm_context_manager.h"

PHI_DECLARE_bool(nccl_blocking_wait);
PD_DECLARE_bool(use_stream_safe_cuda_allocator);

// set this flag to `true` and recompile to enable dynamic checks
constexpr bool FLAGS_enable_nccl_dynamic_check = false;
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

ProcessGroupNCCL::NCCLTask::~NCCLTask() = default;

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

ProcessGroupNCCL::ProcessGroupNCCL(
    const std::shared_ptr<phi::distributed::Store>& store,
    int rank,
    int size,
    int gid)
    : ProcessGroupWithStream(rank, size, gid), store_(store) {}

void ProcessGroupNCCL::GroupStart() {
  NCCL_CHECK(phi::dynload::ncclGroupStart());
}

void ProcessGroupNCCL::GroupEnd() { NCCL_CHECK(phi::dynload::ncclGroupEnd()); }

phi::DeviceContext* ProcessGroupNCCL::GetDeviceContext(
    const Place& place) const {
  return GetDeviceContext(place, /*use_calc_stream*/ false);
}

phi::DeviceContext* ProcessGroupNCCL::GetDeviceContext(
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
        phi::errors::NotFound(
            "Cannot find the device context in this process group."));
    return iter->second.get();
  }
}

ncclComm_t ProcessGroupNCCL::NCCLComm(const Place& place) const {
  const std::string& key = GetKeyFromPlace(place);
  const auto& iter = place_to_comm_ctx_.find(key);
  PADDLE_ENFORCE_NE(
      iter,
      place_to_comm_ctx_.end(),
      phi::errors::NotFound(
          "Cannot find the NCCL communicator in this process group."));
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
  return RunFnInNCCLEnv(
      [&](gpuStream_t stream) {
        auto comm_context = this->GetCommContext();
        comm_context->AllGather(out_tensor, in_tensor_maybe_partial, stream);
      },
      in_tensor_maybe_partial,
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
  return RunFnInNCCLEnv(
      [&](gpuStream_t stream) {
        auto comm_context = this->GetCommContext();
        comm_context->AllReduce(
            out_tensor, in_tensor, ToNCCLRedType(opts.reduce_op), stream);
      },
      in_tensor,
      CommType::ALLREDUCE,
      sync_op,
      use_calc_stream);
}

std::shared_ptr<ProcessGroup::Task> ProcessGroupNCCL::AllToAll(
    phi::DenseTensor* out_tensor,
    const phi::DenseTensor& in_tensor,
    const std::vector<int64_t>& out_size_each_rank,
    const std::vector<int64_t>& in_size_each_rank,
    bool sync_op,
    bool use_calc_stream) {
  const phi::DDim& out_dim = out_tensor->dims();
  const phi::DDim& in_dim = in_tensor.dims();
  CheckSizeOnEachRank(out_dim, out_size_each_rank, size_);
  CheckSizeOnEachRank(in_dim, in_size_each_rank, size_);

  // NOTE: Since `all_to_all` needs other processes' participation, it cannot
  // simply be covered by static checks. Factors are set to 0 here to skip the
  // shape check. Its shape check will be done by dynamic checks with
  // FLAGS_enable_nccl_dynamic_check.
  return RunFnInNCCLEnv(
      [&](gpuStream_t stream) {
        auto comm_context = this->GetCommContext();
        if (FLAGS_enable_nccl_dynamic_check) {
          phi::distributed::NCCLDynamicCheck::CheckShape(
              *out_tensor,
              in_tensor,
              in_size_each_rank,
              rank_,
              size_,
              comm_context->GetNcclComm());
        }

        int64_t in_row_size = in_tensor.numel() / in_dim[0],
                out_row_size = out_tensor->numel() / out_dim[0];
        int64_t in_offset = 0, in_numel = 0, out_offset = 0, out_numel = 0;
        phi::DenseTensor input_partial, output_partial;

        comm_context->GroupStart();
        for (auto i = 0; i < size_; i++) {
          in_numel = in_size_each_rank[i] * in_row_size;
          input_partial = GetPartialTensor(in_tensor, in_offset, in_numel);
          comm_context->Send(input_partial, in_numel, i, stream);
          in_offset += in_numel;

          out_numel = out_size_each_rank[i] * out_row_size;
          output_partial = GetPartialTensor(*out_tensor, out_offset, out_numel);
          comm_context->Recv(&output_partial, out_numel, i, stream);
          out_offset += out_numel;
        }
        comm_context->GroupEnd();
      },
      in_tensor,
      CommType::ALLTOALL,
      sync_op,
      use_calc_stream);
}

std::shared_ptr<ProcessGroup::Task> ProcessGroupNCCL::Barrier(
    const BarrierOptions& opts) {
  PADDLE_ENFORCE_GE(opts.device_id,
                    0,
                    phi::errors::PreconditionNotMet(
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
  return RunFnInNCCLEnv(
      [&](gpuStream_t stream) {
        int root = opts.source_rank + opts.source_root;
        auto comm_context = this->GetCommContext();
        comm_context->Broadcast(out_tensor, in_tensor, root, stream);
      },
      in_tensor,
      CommType::BROADCAST,
      sync_op,
      use_calc_stream);
}

std::shared_ptr<ProcessGroup::Task> ProcessGroupNCCL::Reduce(
    phi::DenseTensor* out_tensor,
    const phi::DenseTensor& in_tensor,
    const ReduceOptions& opts,
    bool sync_op,
    bool use_calc_stream) {
  return RunFnInNCCLEnv(
      [&](gpuStream_t stream) {
        auto comm_context = this->GetCommContext();
        comm_context->Reduce(out_tensor,
                             in_tensor,
                             ToNCCLRedType(opts.reduce_op),
                             opts.root_rank,
                             stream);
      },
      in_tensor,
      CommType::REDUCE,
      sync_op,
      use_calc_stream);
}

std::shared_ptr<ProcessGroup::Task> ProcessGroupNCCL::ReduceScatter(
    phi::DenseTensor* out_tensor,
    const phi::DenseTensor& in_tensor,
    const ReduceScatterOptions& opts,
    bool sync_op,
    bool use_calc_stream) {
  return RunFnInNCCLEnv(
      [&](gpuStream_t stream) {
        auto comm_context = this->GetCommContext();
        comm_context->ReduceScatter(
            out_tensor, in_tensor, ToNCCLRedType(opts.reduce_op), stream);
      },
      in_tensor,
      CommType::REDUCE_SCATTER,
      sync_op,
      use_calc_stream);
}

std::shared_ptr<ProcessGroup::Task> ProcessGroupNCCL::Scatter(
    phi::DenseTensor* out_tensor,
    const phi::DenseTensor& in_tensor,
    const ScatterOptions& opts,
    bool sync_op,
    bool use_calc_stream) {
  phi::distributed::CommStaticCheck::ScatterLikeShape(
      *out_tensor,
      in_tensor,
      /*dst_rank*/ opts.root_rank,
      /*cur_rank*/ rank_,
      size_);
  return RunFnInNCCLEnv(
      [&](gpuStream_t stream) {
        auto comm_context = this->GetCommContext();
        if (FLAGS_enable_nccl_dynamic_check) {
          phi::distributed::NCCLDynamicCheck::CheckShape(
              *out_tensor,
              /*root_rank*/ opts.root_rank,
              rank_,
              comm_context->GetNcclComm());
        }

        int64_t numel = in_tensor.numel() / size_;
        if (rank_ == opts.root_rank) {
          int64_t offset = 0;
          phi::DenseTensor partial_tensor;
          comm_context->GroupStart();
          for (auto i = 0; i < size_; i++) {
            partial_tensor = GetPartialTensor(in_tensor, offset, numel);
            comm_context->Send(partial_tensor, numel, i, stream);
            offset += numel;
          }
          comm_context->Recv(out_tensor, numel, opts.root_rank, stream);
          comm_context->GroupEnd();
        } else {
          comm_context->Recv(out_tensor, numel, opts.root_rank, stream);
        }
      },
      in_tensor,
      CommType::SCATTER,
      sync_op,
      use_calc_stream);
}

std::shared_ptr<ProcessGroup::Task> ProcessGroupNCCL::Gather(
    phi::DenseTensor* out_tensor,
    const phi::DenseTensor& in_tensor,
    const GatherOptions& opts,
    bool sync_op,
    bool use_calc_stream) {
  std::vector<phi::DenseTensor> partial_tensors;
  if (rank_ == opts.root_rank) {
    partial_tensors.reserve(size_);
    size_t offset = 0;
    size_t numel = out_tensor->numel() / size_;
    for (auto i = 0; i < size_; i++) {
      partial_tensors.push_back(GetPartialTensor(*out_tensor, offset, numel));
      offset += numel;
    }
  }
  return Gather(&partial_tensors, in_tensor, opts, sync_op, use_calc_stream);
}

std::shared_ptr<ProcessGroup::Task> ProcessGroupNCCL::Gather(
    std::vector<phi::DenseTensor>* gather_tensors_ptr,
    const phi::DenseTensor& in_tensor,
    const GatherOptions& opts,
    bool sync_op,
    bool use_calc_stream) {
  auto& gather_tensors = *gather_tensors_ptr;
  PADDLE_ENFORCE_GT(size_,
                    opts.root_rank,
                    phi::errors::InvalidArgument(
                        "root world size [%d]  is less than root rank [%d]",
                        size_,
                        opts.root_rank));
  auto gather_func = [&](gpuStream_t stream) {
    auto comm_context = this->GetCommContext();
    // shape check
    if (FLAGS_enable_nccl_dynamic_check) {
      phi::distributed::NCCLDynamicCheck::CheckGatherShape(
          in_tensor,
          gather_tensors,
          opts.root_rank,
          rank_,
          size_,
          comm_context->GetNcclComm());
    }

    comm_context->GroupStart();
    // root receive from all devices
    if (rank_ == opts.root_rank) {
      for (auto i = 0; i < size_; i++) {
        auto& gather_tensor = gather_tensors[i];
        comm_context->Recv(&gather_tensor, gather_tensor.numel(), i, stream);
      }
    }
    // send to root
    comm_context->Send(in_tensor, in_tensor.numel(), opts.root_rank, stream);
    comm_context->GroupEnd();
  };
  return RunFnInNCCLEnv(
      gather_func, in_tensor, CommType::GATHER, sync_op, use_calc_stream);
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

  return RunFnInNCCLEnv(
      [&](gpuStream_t stream) {
        auto comm_context = this->GetCommContext();
        comm_context->Recv(tensor, tensor->numel(), src_rank, stream);
      },
      *tensor,
      CommType::RECV,
      sync_op,
      use_calc_stream);
}

std::shared_ptr<ProcessGroup::Task> ProcessGroupNCCL::Send(
    const phi::DenseTensor& tensor,
    int dst_rank,
    int64_t offset,
    int64_t numel,
    bool sync_op,
    bool use_calc_stream) {
  // numel > 0 indicates the tensor need to be sliced
  const phi::DenseTensor& tensor_maybe_partial =
      numel > 0 ? GetPartialTensor(tensor, offset, numel) : tensor;

  return RunFnInNCCLEnv(
      [&](gpuStream_t stream) {
        auto comm_context = this->GetCommContext();
        comm_context->Send(tensor_maybe_partial,
                           tensor_maybe_partial.numel(),
                           dst_rank,
                           stream);
      },
      tensor_maybe_partial,
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
  if (!place_to_comm_ctx_.empty()) {
    VLOG(3) << "Warning: Tensors from multiple devices are not supported yet.";
  }

  VLOG(3) << "init nccl rank: " << rank_ << ", nranks: " << size_
          << ", place: " << place_key;

  phi::distributed::CommContextManager::CreateNCCLCommContext(
      store_, std::to_string(gid_), rank_, size_);

  auto* calc_ctx = static_cast<phi::GPUContext*>(
      platform::DeviceContextPool::Instance().Get(place));
  auto comm_ctx = std::make_unique<phi::GPUContext>(place);
  auto nccl_comm_ctx = this->GetCommContext();
  comm_ctx->set_nccl_comm(nccl_comm_ctx->GetNcclComm());

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

std::shared_ptr<ProcessGroup::Task> ProcessGroupNCCL::RunFnInNCCLEnv(
    std::function<void(gpuStream_t)> fn,
    const phi::DenseTensor& tensor,
    CommType comm_type,
    bool sync_op,
    bool use_calc_stream) {
  const auto& place = tensor.place();
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
  auto nccl_stream = use_calc_stream ? calc_ctx->stream() : comm_ctx->stream();
  fn(nccl_stream);

  if (!use_calc_stream) {
    if (FLAGS_use_stream_safe_cuda_allocator) {
      memory::RecordStream(tensor.Holder(), nccl_stream);
    }
    task->UpdateWaitChain(*comm_ctx);
  }

  if (FLAGS_enable_nccl_dynamic_check) {
    task->SetBlockCPUInWait();
    task->Wait();
  }
  return task;
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

ProcessGroupNCCL::NCCLTask::NCCLTask(
    const std::vector<Place>& places,
    int rank,
    CommType CommType,
    const std::vector<phi::DenseTensor>& inputs)
    : TaskStream(rank, inputs, CommType),
      comm_event_(places[0]),
      task_place_(places[0]) {}

// create NCCLManager cache for places_key
void ProcessGroupNCCL::CreateNCCLManagerCache(
    const std::string& places_key, const std::vector<Place>& places) {
  PADDLE_ENFORCE_EQ(places_key.empty(),
                    false,
                    phi::errors::PreconditionNotMet(
                        "Not able to create/get the NCCL Communicator since "
                        "the GPU place are not known"));

  ncclUniqueId nccl_id;
  if (rank_ == 0) {
    NCCL_CHECK(phi::dynload::ncclGetUniqueId(&nccl_id));
  }
  BroadcastUniqueNCCLID(&nccl_id);

  VLOG(3) << "init nccl rank: " << rank_ << ", nranks: " << size_
          << ", place: " << places_key
          << ", nccl uniqueid: " << SerializeNCCLUniqueId(nccl_id);

  std::vector<std::unique_ptr<phi::GPUContext>> dev_ctx;
  dev_ctx.resize(places.size());

  std::vector<phi::GPUContext*> dev_ctx_raw;
  dev_ctx_raw.resize(places.size());

  GroupStart();

  for (size_t i = 0; i < places.size(); ++i) {
    platform::CUDADeviceGuard guard(places[i]);

    dev_ctx[i] = std::make_unique<phi::GPUContext>(places[i]);
    ncclComm_t nccl_comm;
    NCCL_CHECK(phi::dynload::ncclCommInitRank(
        &nccl_comm, GetSize(), nccl_id, GetRank()));
    dev_ctx[i]->set_nccl_comm(nccl_comm);
    dev_ctx_raw[i] = dev_ctx[i].get();
  }

  GroupEnd();

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
      phi::errors::InvalidArgument("All inputs should be in CudaPlace."));
  return Collective(
      in_tensors,
      out_tensors,
      [&](const phi::DenseTensor& input,
          phi::DenseTensor& output,
          ncclComm_t comm,
          const gpuStream_t& stream) {
        auto comm_context = this->GetCommContext();
        comm_context->AllReduce(
            &output, input, ToNCCLRedType(opts.reduce_op), stream);
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
      phi::errors::InvalidArgument("All inputs should be in CudaPlace."));

  return Collective(
      in_tensors,
      out_tensors,
      [&](phi::DenseTensor& input,
          phi::DenseTensor& output,
          ncclComm_t comm,
          const gpuStream_t& stream) {
        const auto root =
            opts.source_rank * in_tensors.size() + opts.source_root;
        auto comm_context = this->GetCommContext();
        comm_context->Broadcast(&output, input, root, stream);
      },
      CommType::BROADCAST);
}

void CheckTensorsInDifferentDevices(
    const std::vector<phi::DenseTensor>& tensors, const size_t num_devices) {
  PADDLE_ENFORCE_EQ(
      tensors.empty(),
      false,
      phi::errors::InvalidArgument("Tensor list must be nonempty."));
  PADDLE_ENFORCE_LE(
      tensors.size(),
      num_devices,
      phi::errors::InvalidArgument(
          "Tensor list mustn't be larger than the number of available GPUs."));

  std::set<Place> used_devices;

  for (const auto& t : tensors) {
    PADDLE_ENFORCE_EQ(
        platform::is_gpu_place(t.place()),
        true,
        phi::errors::InvalidArgument("Tensors must be CUDA and dense tensor."));

    const auto inserted = used_devices.insert(t.place()).second;
    PADDLE_ENFORCE_EQ(inserted,
                      true,
                      phi::errors::InvalidArgument(
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
        auto comm_context = this->GetCommContext();
        comm_context->Send(input, input.numel(), dst_rank, stream);
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
        auto comm_context = this->GetCommContext();
        comm_context->Recv(&output, output.numel(), src_rank, stream);
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
      phi::errors::InvalidArgument("All inputs should be in CudaPlace."));
  PADDLE_ENFORCE_EQ(
      CheckTensorsInCudaPlace(out_tensors),
      true,
      phi::errors::InvalidArgument("All outputs should be in CudaPlace."));
  return Collective(
      in_tensors,
      out_tensors,
      [&](const phi::DenseTensor& input,
          phi::DenseTensor& output,
          ncclComm_t comm,
          const gpuStream_t& stream) {
        auto comm_context = this->GetCommContext();
        comm_context->AllGather(&output, input, stream);
      },
      CommType::ALLGATHER);
}

std::shared_ptr<ProcessGroup::Task> ProcessGroupNCCL::AllToAll(
    std::vector<phi::DenseTensor>& in_tensors,
    std::vector<phi::DenseTensor>& out_tensors) {
  PADDLE_ENFORCE_EQ(
      CheckTensorsInCudaPlace(in_tensors),
      true,
      phi::errors::InvalidArgument("All inputs should be in CudaPlace."));
  PADDLE_ENFORCE_EQ(
      CheckTensorsInCudaPlace(out_tensors),
      true,
      phi::errors::InvalidArgument("All inputs should be in CudaPlace."));
  return Collective(
      in_tensors,
      out_tensors,
      [&](phi::DenseTensor& input,
          phi::DenseTensor& output,
          ncclComm_t comm,
          const gpuStream_t& stream) {
        size_t offset = 0;
        size_t count = input.numel() / size_;
        auto comm_context = this->GetCommContext();
        comm_context->GroupStart();
        for (auto i = 0; i < size_; i++) {
          auto input_data = GetPartialTensor(input, offset, count);
          comm_context->Send(input_data, count, i, stream);
          auto output_data = GetPartialTensor(output, offset, count);
          comm_context->Recv(&output_data, count, i, stream);
          offset += count;
        }
        comm_context->GroupEnd();
      },
      CommType::ALLTOALL);
}

std::shared_ptr<ProcessGroup::Task> ProcessGroupNCCL::Reduce(
    std::vector<phi::DenseTensor>& in_tensors,
    std::vector<phi::DenseTensor>& out_tensors,
    const ReduceOptions& opts) {
  PADDLE_ENFORCE_EQ(
      CheckTensorsInCudaPlace(in_tensors),
      true,
      phi::errors::InvalidArgument("All inputs should be in CudaPlace."));
  return Collective(
      in_tensors,
      out_tensors,
      [&](const phi::DenseTensor& input,
          phi::DenseTensor& output,
          ncclComm_t comm,
          const gpuStream_t& stream) {
        auto comm_context = this->GetCommContext();
        comm_context->Reduce(&output,
                             input,
                             ToNCCLRedType(opts.reduce_op),
                             opts.root_rank,
                             stream);
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
      phi::errors::InvalidArgument("All inputs should be in CudaPlace."));
  PADDLE_ENFORCE_EQ(
      CheckTensorsInCudaPlace(out_tensors),
      true,
      phi::errors::InvalidArgument("All inputs should be in CudaPlace."));
  return Collective(
      in_tensors,
      out_tensors,
      [&](phi::DenseTensor& input,
          phi::DenseTensor& output,
          ncclComm_t comm,
          const gpuStream_t& stream) {
        auto comm_context = this->GetCommContext();
        size_t offset = 0;
        size_t count = input.numel() / size_;
        if (rank_ == opts.root_rank) {
          comm_context->GroupStart();
          for (auto i = 0; i < size_; i++) {
            auto input_data = reinterpret_cast<phi::DenseTensor*>(
                GetPointerByOffset(input.data(), offset, input.dtype()));
            comm_context->Send(*input_data, count, i, stream);
            offset += count;
          }
          comm_context->Recv(&output, count, opts.root_rank, stream);
          comm_context->GroupEnd();
        } else {
          comm_context->Recv(&output, count, opts.root_rank, stream);
        }
      },
      CommType::SCATTER);
}

std::shared_ptr<ProcessGroupNCCL> ProcessGroupNCCL::CreateProcessGroupNCCL(
    const std::shared_ptr<phi::distributed::Store>& store,
    int rank,
    int size,
    int gid) {
  auto process_group =
      std::make_shared<ProcessGroupNCCL>(store, rank, size, gid);
  ProcessGroupIdMap::GetInstance().emplace(gid, process_group);
  return process_group;
}

phi::distributed::NCCLCommContext* ProcessGroupNCCL::GetCommContext() {
  const auto& comm_context_manager =
      phi::distributed::CommContextManager::GetInstance();
  auto comm_context = static_cast<phi::distributed::NCCLCommContext*>(
      comm_context_manager.Get(std::to_string(this->gid_)));
  PADDLE_ENFORCE_NE(comm_context,
                    nullptr,
                    phi::errors::Unavailable("NCCLCommContext is nullptr"));
  return comm_context;
}

}  //  namespace distributed
}  //  namespace paddle
