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

#include "paddle/fluid/distributed/collective/check.h"
#include "paddle/fluid/distributed/collective/common.h"
#include "paddle/fluid/distributed/collective/nccl_tools.h"
#include "paddle/fluid/distributed/collective/utils.h"
#include "paddle/fluid/platform/device/gpu/nccl_helper.h"
#include "paddle/fluid/platform/place.h"
#include "paddle/phi/api/lib/utils/allocator.h"

DECLARE_bool(nccl_blocking_wait);
DECLARE_bool(use_stream_safe_cuda_allocator);

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
        platform::errors::NotFound(
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
  CommStaticCheck::GatherLikeShape(*out_tensor,
                                   in_tensor_maybe_partial,
                                   /*dst_rank*/ rank_,
                                   /*cur_rank*/ rank_,
                                   size_);
  return RunFnInNCCLEnv(
      [&](ncclComm_t comm, gpuStream_t stream) {
        if (FLAGS_enable_nccl_dynamic_check) {
          CommDynamicCheck::CheckShape(*out_tensor,
                                       /*root_rank*/ 0,
                                       rank_,
                                       comm);
        }
        NCCL_CHECK(phi::dynload::ncclAllGather(
            in_tensor_maybe_partial.data(),
            out_tensor->data(),
            in_tensor_maybe_partial.numel(),
            platform::ToNCCLDataType(in_tensor_maybe_partial.dtype()),
            comm,
            stream));
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
  CommStaticCheck::SameShape(*out_tensor,
                             in_tensor,
                             /*dst_rank*/ rank_,
                             /*cur_rank*/ rank_,
                             size_);
  return RunFnInNCCLEnv(
      [&](ncclComm_t comm, gpuStream_t stream) {
        if (FLAGS_enable_nccl_dynamic_check) {
          CommDynamicCheck::CheckShape(*out_tensor,
                                       /*root_rank*/ 0,
                                       rank_,
                                       comm);
        }
        NCCL_CHECK(phi::dynload::ncclAllReduce(
            in_tensor.data(),
            out_tensor->data(),
            in_tensor.numel(),
            platform::ToNCCLDataType(in_tensor.dtype()),
            ToNCCLRedType(opts.reduce_op),
            comm,
            stream));
      },
      in_tensor,
      CommType::ALLREDUCE,
      sync_op,
      use_calc_stream);
}

void CheckSizeOnEachRank(const phi::DDim& tensor_dim,
                         const std::vector<int64_t>& size_on_each_rank,
                         int world_size) {
  int length_size_on_each_rank = size_on_each_rank.size();
  PADDLE_ENFORCE_EQ(
      length_size_on_each_rank,
      world_size,
      platform::errors::InvalidArgument(
          "The length of size_on_each_rank must be equal to world_size."));

  int64_t sum_size_on_each_rank =
      std::accumulate(size_on_each_rank.begin(), size_on_each_rank.end(), 0);
  PADDLE_ENFORCE_EQ(
      sum_size_on_each_rank,
      tensor_dim[0],
      platform::errors::InvalidArgument(
          "The sum of size_on_each_rank must be equal to tensor's dim[0]."));
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
  CommStaticCheck::CheckShape(*out_tensor,
                              in_tensor,
                              /*dst_rank*/ rank_,
                              /*cur_rank*/ rank_,
                              size_,
                              /*out_size_factor*/ 0,
                              /*in_size_factor*/ 0);
  return RunFnInNCCLEnv(
      [&](ncclComm_t comm, gpuStream_t stream) {
        if (FLAGS_enable_nccl_dynamic_check) {
          CommDynamicCheck::CheckShape(
              *out_tensor, in_tensor, in_size_each_rank, rank_, size_, comm);
        }
        int64_t in_row_size = in_tensor.numel() / in_dim[0],
                out_row_size = out_tensor->numel() / out_dim[0];
        int64_t in_offset = 0, in_numel = 0, out_offset = 0, out_numel = 0;
        phi::DenseTensor input_partial, output_partial;

        GroupStart();
        for (auto i = 0; i < size_; i++) {
          in_numel = in_size_each_rank[i] * in_row_size;
          input_partial = GetPartialTensor(in_tensor, in_offset, in_numel);
          NCCL_CHECK(phi::dynload::ncclSend(
              input_partial.data(),
              in_numel,
              platform::ToNCCLDataType(input_partial.dtype()),
              i,
              comm,
              stream));
          in_offset += in_numel;

          out_numel = out_size_each_rank[i] * out_row_size;
          output_partial = GetPartialTensor(*out_tensor, out_offset, out_numel);
          NCCL_CHECK(phi::dynload::ncclRecv(
              output_partial.data(),
              out_numel,
              platform::ToNCCLDataType(output_partial.dtype()),
              i,
              comm,
              stream));
          out_offset += out_numel;
        }
        GroupEnd();
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
  CommStaticCheck::SameShape(*out_tensor,
                             in_tensor,
                             /*dst_rank*/ rank_,
                             /*cur_rank*/ rank_,
                             size_);
  return RunFnInNCCLEnv(
      [&](ncclComm_t comm, gpuStream_t stream) {
        int root = opts.source_rank + opts.source_root;
        if (FLAGS_enable_nccl_dynamic_check) {
          CommDynamicCheck::CheckShape(*out_tensor, root, rank_, comm);
        }
        NCCL_CHECK(phi::dynload::ncclBroadcast(
            in_tensor.data(),
            out_tensor->data(),
            in_tensor.numel(),
            platform::ToNCCLDataType(in_tensor.dtype()),
            root,
            comm,
            stream));
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
  CommStaticCheck::SameShape(*out_tensor,
                             in_tensor,
                             /*dst_rank*/ opts.root_rank,
                             /*cur_rank*/ rank_,
                             size_);
  return RunFnInNCCLEnv(
      [&](ncclComm_t comm, gpuStream_t stream) {
        if (FLAGS_enable_nccl_dynamic_check) {
          CommDynamicCheck::CheckShape(*out_tensor,
                                       /*root_rank*/ opts.root_rank,
                                       rank_,
                                       comm);
        }
        NCCL_CHECK(phi::dynload::ncclReduce(
            in_tensor.data(),
            out_tensor->data(),
            in_tensor.numel(),
            platform::ToNCCLDataType(in_tensor.dtype()),
            ToNCCLRedType(opts.reduce_op),
            opts.root_rank,
            comm,
            stream));
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
  CommStaticCheck::ScatterLikeShape(*out_tensor,
                                    in_tensor,
                                    /*dst_rank*/ rank_,
                                    /*cur_rank*/ rank_,
                                    size_);
  return RunFnInNCCLEnv(
      [&](ncclComm_t comm, gpuStream_t stream) {
        if (FLAGS_enable_nccl_dynamic_check) {
          CommDynamicCheck::CheckShape(*out_tensor,
                                       /*root_rank*/ 0,
                                       rank_,
                                       comm);
        }
        NCCL_CHECK(phi::dynload::ncclReduceScatter(
            in_tensor.data(),
            out_tensor->data(),
            out_tensor->numel(),
            platform::ToNCCLDataType(in_tensor.dtype()),
            ToNCCLRedType(opts.reduce_op),
            comm,
            stream));
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
  CommStaticCheck::ScatterLikeShape(*out_tensor,
                                    in_tensor,
                                    /*dst_rank*/ opts.root_rank,
                                    /*cur_rank*/ rank_,
                                    size_);
  return RunFnInNCCLEnv(
      [&](ncclComm_t comm, gpuStream_t stream) {
        if (FLAGS_enable_nccl_dynamic_check) {
          CommDynamicCheck::CheckShape(*out_tensor,
                                       /*root_rank*/ opts.root_rank,
                                       rank_,
                                       comm);
        }
        int64_t numel = in_tensor.numel() / size_;
        if (rank_ == opts.root_rank) {
          int64_t offset = 0;
          phi::DenseTensor partial_tensor;
          GroupStart();
          for (auto i = 0; i < size_; i++) {
            partial_tensor = GetPartialTensor(in_tensor, offset, numel);
            NCCL_CHECK(phi::dynload::ncclSend(
                partial_tensor.data(),
                numel,
                platform::ToNCCLDataType(partial_tensor.dtype()),
                i,
                comm,
                stream));
            offset += numel;
          }
          NCCL_CHECK(phi::dynload::ncclRecv(
              out_tensor->data(),
              numel,
              platform::ToNCCLDataType(out_tensor->dtype()),
              opts.root_rank,
              comm,
              stream));
          GroupEnd();
        } else {
          NCCL_CHECK(phi::dynload::ncclRecv(
              out_tensor->data(),
              numel,
              platform::ToNCCLDataType(out_tensor->dtype()),
              opts.root_rank,
              comm,
              stream));
        }
      },
      in_tensor,
      CommType::SCATTER,
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

  CommStaticCheck::CheckShape(*tensor, rank_, size_);
  return RunFnInNCCLEnv(
      [&](ncclComm_t comm, gpuStream_t stream) {
        if (FLAGS_enable_nccl_dynamic_check) {
          CommDynamicCheck::CheckShape(*tensor,
                                       /*root_rank*/ src_rank,
                                       rank_,
                                       comm);
        }
        NCCL_CHECK(
            phi::dynload::ncclRecv(tensor->data(),
                                   tensor->numel(),
                                   platform::ToNCCLDataType(tensor->dtype()),
                                   src_rank,
                                   comm,
                                   stream));
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

  CommStaticCheck::CheckShape(tensor_maybe_partial, rank_, size_);
  return RunFnInNCCLEnv(
      [&](ncclComm_t comm, gpuStream_t stream) {
        if (FLAGS_enable_nccl_dynamic_check) {
          CommDynamicCheck::CheckShape(tensor_maybe_partial,
                                       /*root_rank*/ rank_,
                                       rank_,
                                       comm);
        }
        NCCL_CHECK(phi::dynload::ncclSend(
            tensor_maybe_partial.data(),
            tensor_maybe_partial.numel(),
            platform::ToNCCLDataType(tensor_maybe_partial.dtype()),
            dst_rank,
            comm,
            stream));
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
  if (place_to_comm_ctx_.size() > 0) {
    VLOG(3) << "Warning: Tensors from multiple devices are not supported yet.";
  }

  ncclUniqueId nccl_id;
  if (rank_ == 0) {
    NCCL_CHECK(phi::dynload::ncclGetUniqueId(&nccl_id));
  }
  BroadcastUniqueNCCLID(&nccl_id);

  VLOG(3) << "init nccl rank: " << rank_ << ", nranks: " << size_
          << ", place: " << place_key
          << ", nccl uniqueid: " << SerializeNCCLUniqueId(nccl_id);

  auto* calc_ctx = static_cast<phi::GPUContext*>(
      platform::DeviceContextPool::Instance().Get(place));
  auto comm_ctx = std::make_unique<phi::GPUContext>(place);
  ncclComm_t nccl_comm;
  NCCL_CHECK(phi::dynload::ncclCommInitRank(
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

std::shared_ptr<ProcessGroup::Task> ProcessGroupNCCL::RunFnInNCCLEnv(
    std::function<void(ncclComm_t, gpuStream_t)> fn,
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
  auto nccl_comm = comm_ctx->nccl_comm();
  auto nccl_stream = use_calc_stream ? calc_ctx->stream() : comm_ctx->stream();
  fn(nccl_comm, nccl_stream);

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
                    platform::errors::PreconditionNotMet(
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

    dev_ctx[i].reset(new phi::GPUContext(places[i]));
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
      platform::errors::InvalidArgument("All inputs should be in CudaPlace."));
  return Collective(
      in_tensors,
      out_tensors,
      [&](const phi::DenseTensor& input,
          phi::DenseTensor& output,
          ncclComm_t comm,
          const gpuStream_t& stream) {
        return phi::dynload::ncclAllReduce(
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
        return phi::dynload::ncclBroadcast(
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
        return phi::dynload::ncclSend(input.data(),
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
        return phi::dynload::ncclRecv(output.data(),
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
        return phi::dynload::ncclAllGather(
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
        GroupStart();
        for (auto i = 0; i < size_; i++) {
          PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::ncclSend(
              GetPointerByOffset(input.data(), offset, input.dtype()),
              input.numel() / size_,
              platform::ToNCCLDataType(input.dtype()),
              i,
              comm,
              stream));
          PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::ncclRecv(
              GetPointerByOffset(output.data(), offset, input.dtype()),
              input.numel() / size_,
              platform::ToNCCLDataType(input.dtype()),
              i,
              comm,
              stream));
          offset += input.numel() / size_;
        }
        GroupEnd();
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
      platform::errors::InvalidArgument("All inputs should be in CudaPlace."));
  return Collective(
      in_tensors,
      out_tensors,
      [&](const phi::DenseTensor& input,
          phi::DenseTensor& output,
          ncclComm_t comm,
          const gpuStream_t& stream) {
        PADDLE_ENFORCE_GPU_SUCCESS(
            phi::dynload::ncclReduce(input.data(),
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
          GroupStart();
          for (auto i = 0; i < size_; i++) {
            PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::ncclSend(
                GetPointerByOffset(input.data(), offset, input.dtype()),
                input.numel() / size_,
                platform::ToNCCLDataType(input.dtype()),
                i,
                comm,
                stream));
            offset += input.numel() / size_;
          }
          PADDLE_ENFORCE_GPU_SUCCESS(
              phi::dynload::ncclRecv(output.data(),
                                     input.numel() / size_,
                                     platform::ToNCCLDataType(input.dtype()),
                                     opts.root_rank,
                                     comm,
                                     stream));
          GroupEnd();
        } else {
          PADDLE_ENFORCE_GPU_SUCCESS(
              phi::dynload::ncclRecv(output.data(),
                                     input.numel() / size_,
                                     platform::ToNCCLDataType(input.dtype()),
                                     opts.root_rank,
                                     comm,
                                     stream));
        }
      },
      CommType::SCATTER);
}

std::shared_ptr<ProcessGroupNCCL> ProcessGroupNCCL::CreateProcessGroupNCCL(
    const std::shared_ptr<Store>& store, int rank, int size, int gid) {
  auto process_group =
      std::make_shared<ProcessGroupNCCL>(store, rank, size, gid);
  ProcessGroupIdMap::GetInstance().emplace(gid, process_group);
  return process_group;
}

}  //  namespace distributed
}  //  namespace paddle
