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

#include "paddle/fluid/distributed/collective/process_group_custom.h"

#include "paddle/common/flags.h"
#include "paddle/fluid/distributed/collective/common.h"
#include "paddle/fluid/distributed/collective/custom_ccl_tools.h"
#include "paddle/phi/api/lib/utils/allocator.h"
#include "paddle/phi/core/distributed/check/static_check.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/utils/data_type.h"

#include "paddle/phi/core/distributed/comm_context_manager.h"

constexpr int64_t kWaitBlockTImeout = 10;

PD_DECLARE_bool(use_stream_safe_cuda_allocator);

namespace paddle {
namespace distributed {

using phi::distributed::CheckSizeOnEachRank;
using phi::distributed::GetPointerByOffset;
static std::mutex g_unfinished_xccl_task_events_mutex;
static std::list<std::unique_ptr<phi::event::Event>>
    g_unfinished_xccl_task_events;

ProcessGroupCustom::XCCLTask::XCCLTask(const Place& place,
                                       int rank,
                                       CommType comm_type,
                                       bool sync_op,
                                       bool use_calc_stream)
    : TaskStream(rank, comm_type, sync_op, use_calc_stream),
      task_place_(place),
      comm_event_(std::make_unique<phi::event::Event>()) {
  comm_event_->Init(task_place_);
}

ProcessGroupCustom::XCCLTask::XCCLTask(
    const std::vector<Place>& places,
    int rank,
    CommType CommType,
    const std::vector<phi::DenseTensor>& inputs)
    : TaskStream(rank, inputs, CommType),
      task_place_(places[0]),
      comm_event_(std::make_unique<phi::event::Event>()) {
  comm_event_->Init(task_place_);
}

ProcessGroupCustom::XCCLTask::~XCCLTask() {
  if (!IsCompleted()) {
    std::lock_guard<std::mutex> lock(g_unfinished_xccl_task_events_mutex);
    g_unfinished_xccl_task_events.push_back(std::move(comm_event_));
  }
}

bool ProcessGroupCustom::XCCLTask::IsCompleted() {
  return comm_event_->Query();
}

void ProcessGroupCustom::XCCLTask::UpdateWaitChain(
    const phi::DeviceContext& ctx) {
  {
    std::lock_guard<std::mutex> lock(g_unfinished_xccl_task_events_mutex);
    for (auto iter = g_unfinished_xccl_task_events.begin();
         iter != g_unfinished_xccl_task_events.end();) {
      if ((*iter)->Query()) {
        iter = g_unfinished_xccl_task_events.erase(iter);
      } else {
        iter++;
      }
    }
  }
  comm_event_->Record(
      reinterpret_cast<const phi::CustomContext&>(ctx).GetStream().get());
}

bool ProcessGroupCustom::XCCLTask::Wait(std::chrono::milliseconds timeout) {
  // Warning here when use calc stream but also invoke waiting explicitly.
  if (UseCalcStream()) {
    VLOG(3) << "Warning: The communication is on calc stream, wait here is "
               "useless.";
    return true;
  }

  const auto* calc_ctx = reinterpret_cast<phi::CustomContext*>(
      phi::DeviceContextPool::Instance().Get(task_place_));
  calc_ctx->GetStream()->WaitEvent(comm_event_.get());

  if (IsBlockCPUInWait()) {
    // If we use the work to do barrier, we should block cpu
    phi::DeviceManager::SynchronizeDevice(task_place_);
  }
  return true;
}

// Same as Wait
void ProcessGroupCustom::XCCLTask::Synchronize() { Wait(kWaitTimeout); }

ProcessGroupCustom::ProcessGroupCustom(
    const std::shared_ptr<phi::distributed::Store>& store,
    const std::string& device_type,
    int rank,
    int size,
    int gid)
    : ProcessGroupWithStream(rank, size, gid),
      store_(store),
      device_type_(device_type) {}

void ProcessGroupCustom::GroupStart(const std::string& dev_type) {
  phi::DeviceManager::CCLGroupStart(dev_type);
}

void ProcessGroupCustom::GroupEnd(const std::string& dev_type) {
  phi::DeviceManager::CCLGroupEnd(dev_type);
}

phi::DeviceContext* ProcessGroupCustom::GetDeviceContext(
    const Place& place) const {
  return GetDeviceContext(place, /*use_calc_stream*/ false);
}

phi::DeviceContext* ProcessGroupCustom::GetDeviceContext(
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
        common::errors::NotFound(
            "Cannot find the device context in this process group."));
    return iter->second.get();
  }
}

phi::ccl::CCLComm ProcessGroupCustom::XCCLComm(const Place& place) const {
  const std::string& key = GetKeyFromPlace(place);
  const auto& iter = place_to_comm_ctx_.find(key);
  PADDLE_ENFORCE_NE(
      iter,
      place_to_comm_ctx_.end(),
      common::errors::NotFound(
          "Cannot find the XCCL communicator in this process group."));
  return iter->second->xccl_comm();
}

std::string ProcessGroupCustom::GetCommName(int rank) {
  PADDLE_ENFORCE_GE(rank,
                    0,
                    common::errors::PreconditionNotMet(
                        "The rank must greater or equal than 0!"));
  auto num_devices = phi::DeviceManager::GetDeviceCount(device_type_);
  PADDLE_ENFORCE_GT(
      num_devices,
      0,
      common::errors::InvalidArgument("The num_devices must greater than 0!"));

  auto place_id = rank % num_devices;
  phi::CustomPlace place(device_type_, place_id);
  const auto& key = GetKeyFromPlace(place);
  phi::DeviceGuard guard(place);
  if (place_to_comm_ctx_.find(key) == place_to_comm_ctx_.end()) {
    CreateXCCLEnvCache(place, key);
  }

  char comm_name[128];
  phi::DeviceManager::CCLCommName(
      device_type_, this->GetCommContext()->GetXcclComm(), comm_name);
  std::string name_str(comm_name);
  return name_str;
}

std::shared_ptr<ProcessGroup::Task> ProcessGroupCustom::AllGather(
    phi::DenseTensor* out_tensor,
    const phi::DenseTensor& in_tensor,
    int64_t offset,
    int64_t numel,
    bool sync_op,
    bool use_calc_stream) {
  CheckTensorContiguous(in_tensor);
  CheckTensorContiguous(*out_tensor);

  // numel > 0 indicates the tensor need to be sliced
  const phi::DenseTensor& in_tensor_maybe_partial =
      numel > 0 ? GetPartialTensor(in_tensor, offset, numel) : in_tensor;
  return RunFnInXCCLEnv(
      [&](const phi::stream::Stream& stream) {
        auto comm_context = this->GetCommContext();
        comm_context->AllGather(out_tensor, in_tensor_maybe_partial, stream);
      },
      in_tensor_maybe_partial,
      CommType::ALLGATHER,
      sync_op,
      use_calc_stream);
}

std::shared_ptr<ProcessGroup::Task> ProcessGroupCustom::AllReduce(
    phi::DenseTensor* out_tensor,
    const phi::DenseTensor& in_tensor,
    const AllreduceOptions& opts,
    bool sync_op,
    bool use_calc_stream) {
  CheckTensorContiguous(in_tensor);
  CheckTensorContiguous(*out_tensor);

  return RunFnInXCCLEnv(
      [&](const phi::stream::Stream& stream) {
        auto comm_context = this->GetCommContext();
        comm_context->AllReduce(
            out_tensor,
            in_tensor,
            paddle::distributed::ToXCCLRedType(opts.reduce_op),
            stream);
      },
      in_tensor,
      CommType::ALLREDUCE,
      sync_op,
      use_calc_stream);
}

std::shared_ptr<ProcessGroup::Task> ProcessGroupCustom::AllToAll(
    phi::DenseTensor* out_tensor,
    const phi::DenseTensor& in_tensor,
    const std::vector<int64_t>& out_size_each_rank,
    const std::vector<int64_t>& in_size_each_rank,
    bool sync_op,
    bool use_calc_stream) {
  CheckTensorContiguous(in_tensor);
  CheckTensorContiguous(*out_tensor);

  const phi::DDim& out_dim = out_tensor->dims();
  const phi::DDim& in_dim = in_tensor.dims();
  CheckSizeOnEachRank(out_dim, out_size_each_rank, size_);
  CheckSizeOnEachRank(in_dim, in_size_each_rank, size_);

  // NOTE: Since `all_to_all` needs other processes' participation, it cannot
  // simply be covered by static checks. Factors are set to 0 here to skip the
  // shape check. Its shape check will be done by dynamic checks with
  // FLAGS_enable_xccl_dynamic_check.
  return RunFnInXCCLEnv(
      [&](const phi::stream::Stream& stream) {
        auto comm_context = this->GetCommContext();

        int64_t in_row_size = in_tensor.numel() / in_dim[0],
                out_row_size = out_tensor->numel() / out_dim[0];
        int64_t in_offset = 0, in_numel = 0, out_offset = 0, out_numel = 0;
        phi::DenseTensor input_partial, output_partial;

        std::vector<void*> send_buf, recv_buf;
        std::vector<size_t> send_count, recv_count;
        std::vector<phi::DataType> send_dtype, recv_dtype;
        for (auto i = 0; i < size_; i++) {
          in_numel = in_size_each_rank[i] * in_row_size;
          input_partial = GetPartialTensor(in_tensor, in_offset, in_numel);
          out_numel = out_size_each_rank[i] * out_row_size;
          output_partial = GetPartialTensor(*out_tensor, out_offset, out_numel);
          in_offset += in_numel;
          out_offset += out_numel;
          send_buf.push_back(input_partial.data());
          recv_buf.push_back(output_partial.data());
          send_count.push_back(in_numel);
          recv_count.push_back(out_numel);
          send_dtype.push_back(input_partial.dtype());
          recv_dtype.push_back(output_partial.dtype());
        }

        phi::DeviceManager::CCLAllToAll(
            device_type_,
            const_cast<const void**>(send_buf.data()),
            send_count.data(),
            send_dtype.data(),
            recv_buf.data(),
            recv_count.data(),
            recv_dtype.data(),
            rank_,
            size_,
            comm_context->GetXcclComm(),
            stream);
      },
      in_tensor,
      CommType::ALLTOALL,
      sync_op,
      use_calc_stream);
}

std::shared_ptr<ProcessGroup::Task> ProcessGroupCustom::Barrier(
    const BarrierOptions& opts) {
  PADDLE_ENFORCE_GE(opts.device_id,
                    0,
                    common::errors::PreconditionNotMet(
                        "The barrier device id must greater or equal than 0."));
  phi::CustomPlace place(device_type_, opts.device_id);
  auto allocator = std::unique_ptr<phi::Allocator>(
      new paddle::experimental::DefaultAllocator(place));
  phi::DenseTensorMeta meta(phi::DataType::FLOAT32, phi::DDim{1});
  phi::DenseTensor barrier_tensor{allocator.get(), meta};

  auto task = AllReduce(&barrier_tensor,
                        barrier_tensor,
                        {},
                        /*sync_op*/ true,
                        /*use_calc_stream*/ false);
  auto xccl_task = dynamic_cast<XCCLTask*>(task.get());
  xccl_task->SetBlockCPUInWait();
  return task;
}

std::shared_ptr<ProcessGroup::Task> ProcessGroupCustom::Broadcast(
    phi::DenseTensor* out_tensor,
    const phi::DenseTensor& in_tensor,
    const BroadcastOptions& opts,
    bool sync_op,
    bool use_calc_stream) {
  CheckTensorContiguous(in_tensor);
  CheckTensorContiguous(*out_tensor);

  return RunFnInXCCLEnv(
      [&](const phi::stream::Stream& stream) {
        int root = opts.source_rank + opts.source_root;
        auto comm_context = this->GetCommContext();
        comm_context->Broadcast(out_tensor, in_tensor, root, stream);
      },
      in_tensor,
      CommType::BROADCAST,
      sync_op,
      use_calc_stream);
}

std::shared_ptr<ProcessGroup::Task> ProcessGroupCustom::Reduce(
    phi::DenseTensor* out_tensor,
    const phi::DenseTensor& in_tensor,
    const ReduceOptions& opts,
    bool sync_op,
    bool use_calc_stream) {
  CheckTensorContiguous(in_tensor);
  CheckTensorContiguous(*out_tensor);

  return RunFnInXCCLEnv(
      [&](const phi::stream::Stream& stream) {
        auto comm_context = this->GetCommContext();
        comm_context->Reduce(out_tensor,
                             in_tensor,
                             paddle::distributed::ToXCCLRedType(opts.reduce_op),
                             opts.root_rank,
                             stream);
      },
      in_tensor,
      CommType::REDUCE,
      sync_op,
      use_calc_stream);
}

std::shared_ptr<ProcessGroup::Task> ProcessGroupCustom::ReduceScatter(
    phi::DenseTensor* out_tensor,
    const phi::DenseTensor& in_tensor,
    const ReduceScatterOptions& opts,
    bool sync_op,
    bool use_calc_stream) {
  CheckTensorContiguous(in_tensor);
  CheckTensorContiguous(*out_tensor);

  return RunFnInXCCLEnv(
      [&](const phi::stream::Stream& stream) {
        auto comm_context = this->GetCommContext();
        comm_context->ReduceScatter(
            out_tensor,
            in_tensor,
            paddle::distributed::ToXCCLRedType(opts.reduce_op),
            stream);
      },
      in_tensor,
      CommType::REDUCE_SCATTER,
      sync_op,
      use_calc_stream);
}

std::shared_ptr<ProcessGroup::Task> ProcessGroupCustom::Scatter(
    phi::DenseTensor* out_tensor,
    const phi::DenseTensor& in_tensor,
    const ScatterOptions& opts,
    bool sync_op,
    bool use_calc_stream) {
  CheckTensorContiguous(in_tensor);
  CheckTensorContiguous(*out_tensor);

  phi::distributed::CommStaticCheck::ScatterLikeShape(
      *out_tensor,
      in_tensor,
      /*dst_rank*/ opts.root_rank,
      /*cur_rank*/ rank_,
      size_,
      phi::AllocationType::CUSTOM);
  return RunFnInXCCLEnv(
      [&](const phi::stream::Stream& stream) {
        auto comm_context = this->GetCommContext();

        int64_t numel = in_tensor.numel() / size_;
        if (rank_ == opts.root_rank) {
          int64_t offset = 0;
          phi::DenseTensor partial_tensor;
          for (auto i = 0; i < size_; i++) {
            partial_tensor = GetPartialTensor(in_tensor, offset, numel);
            if (i != rank_) {
              comm_context->Send(partial_tensor, numel, i, stream);
            } else {
              phi::DeviceManager::GetDeviceWithPlace(stream.GetPlace())
                  ->MemoryCopyD2D(out_tensor->data(),
                                  partial_tensor.data(),
                                  numel * phi::SizeOf(partial_tensor.dtype()),
                                  &stream);
            }
            offset += numel;
          }
        } else {
          comm_context->Recv(out_tensor, numel, opts.root_rank, stream);
        }
      },
      in_tensor,
      CommType::SCATTER,
      sync_op,
      use_calc_stream);
}

std::shared_ptr<ProcessGroup::Task> ProcessGroupCustom::Gather(
    phi::DenseTensor* out_tensor,
    const phi::DenseTensor& in_tensor,
    const GatherOptions& opts,
    bool sync_op,
    bool use_calc_stream) {
  CheckTensorContiguous(in_tensor);
  CheckTensorContiguous(*out_tensor);

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

std::shared_ptr<ProcessGroup::Task> ProcessGroupCustom::Gather(
    std::vector<phi::DenseTensor>* gather_tensors_ptr,
    const phi::DenseTensor& in_tensor,
    const GatherOptions& opts,
    bool sync_op,
    bool use_calc_stream) {
  CheckTensorContiguous(in_tensor);
  CheckTensorContiguous(*gather_tensors_ptr);

  auto& gather_tensors = *gather_tensors_ptr;
  PADDLE_ENFORCE_GT(size_,
                    opts.root_rank,
                    common::errors::InvalidArgument(
                        "root world size [%d]  is less than root rank [%d]",
                        size_,
                        opts.root_rank));
  auto gather_func = [&](const phi::stream::Stream& stream) {
    auto comm_context = this->GetCommContext();
    // root receive from all devices
    if (rank_ == opts.root_rank) {
      for (auto i = 0; i < size_; i++) {
        auto& gather_tensor = gather_tensors[i];
        if (i != rank_) {
          comm_context->Recv(&gather_tensor, gather_tensor.numel(), i, stream);
        } else {
          phi::DeviceManager::GetDeviceWithPlace(stream.GetPlace())
              ->MemoryCopyD2D(
                  gather_tensor.data(),
                  in_tensor.data(),
                  in_tensor.numel() * phi::SizeOf(in_tensor.dtype()),
                  &stream);
        }
      }
    } else {
      // send to root
      comm_context->Send(in_tensor, in_tensor.numel(), opts.root_rank, stream);
    }
  };
  return RunFnInXCCLEnv(
      gather_func, in_tensor, CommType::GATHER, sync_op, use_calc_stream);
}

std::shared_ptr<ProcessGroup::Task> ProcessGroupCustom::Recv(
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

  return RunFnInXCCLEnv(
      [&](const phi::stream::Stream& stream) {
        auto comm_context = this->GetCommContext();
        comm_context->Recv(tensor, tensor->numel(), src_rank, stream);
      },
      *tensor,
      CommType::RECV,
      sync_op,
      use_calc_stream);
}

std::shared_ptr<ProcessGroup::Task> ProcessGroupCustom::Send(
    const phi::DenseTensor& tensor,
    int dst_rank,
    int64_t offset,
    int64_t numel,
    bool sync_op,
    bool use_calc_stream) {
  CheckTensorContiguous(tensor);

  // numel > 0 indicates the tensor need to be sliced
  const phi::DenseTensor& tensor_maybe_partial =
      numel > 0 ? GetPartialTensor(tensor, offset, numel) : tensor;

  return RunFnInXCCLEnv(
      [&](const phi::stream::Stream& stream) {
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

std::shared_ptr<ProcessGroupCustom::XCCLTask> ProcessGroupCustom::CreateTask(
    const Place& place,
    int rank,
    CommType comm_type,
    bool is_sync,
    bool use_calc_stream) {
  return std::make_shared<ProcessGroupCustom::XCCLTask>(
      place, rank, comm_type, is_sync, use_calc_stream);
}

void ProcessGroupCustom::BroadcastUniqueXCCLID(
    phi::ccl::CCLRootId* xccl_root_id) {
  const std::string key =
      "ProcessGroupCustom/xccl_ids/" + std::to_string(gid_) + "/0";
  if (rank_ == 0) {
    store_->set(key, *xccl_root_id);
  } else {
    *xccl_root_id = store_->get(key);
  }
}

void ProcessGroupCustom::CreateXCCLEnvCache(const Place& place,
                                            const std::string& place_key) {
  if (!place_to_comm_ctx_.empty()) {
    VLOG(3) << "Warning: Tensors from multiple devices are not supported yet.";
  }

  VLOG(3) << "init xccl rank: " << rank_ << ", nranks: " << size_
          << ", place: " << place_key;

  phi::distributed::CommContextManager::CreateXCCLCommContext(
      store_, std::to_string(gid_), place, rank_, size_);

  auto* calc_ctx = static_cast<phi::CustomContext*>(
      phi::DeviceContextPool::Instance().Get(place));
  auto comm_ctx = std::make_unique<phi::CustomContext>(place);
  comm_ctx->SetAllocator(
      &(phi::DeviceContextPool::Instance().Get(place)->GetAllocator()));
  comm_ctx->SetHostAllocator(
      &(phi::DeviceContextPool::Instance().Get(place)->GetHostAllocator()));
  comm_ctx->SetZeroAllocator(
      &(phi::DeviceContextPool::Instance().Get(place)->GetZeroAllocator()));
  comm_ctx->SetHostZeroAllocator(
      &(phi::DeviceContextPool::Instance().Get(place)->GetHostZeroAllocator()));

  auto xccl_comm_ctx = this->GetCommContext();
  comm_ctx->set_xccl_comm(xccl_comm_ctx->GetXcclComm());

  auto xccl_event = std::make_unique<phi::event::Event>();
  xccl_event->Init(place);
  place_to_calc_event_.emplace(place_key, std::move(xccl_event));
  place_to_calc_ctx_.emplace(place_key, calc_ctx);
  place_to_comm_ctx_.emplace(place_key, std::move(comm_ctx));

  // TODO(sunyilun): for compatibility, will be removed later
  std::vector<phi::CustomContext*> comm_ctx_wrapper{
      place_to_comm_ctx_[place_key].get()};
  places_to_ctx_.emplace(place_key, comm_ctx_wrapper);
}

void ProcessGroupCustom::SyncCalcStream(const Place& place) {
  const std::string& key = GetKeyFromPlace(place);
  auto& calc_event = place_to_calc_event_.at(key);
  const auto* calc_ctx = place_to_calc_ctx_.at(key);
  const auto* comm_ctx = place_to_comm_ctx_.at(key).get();
  calc_event->Record(calc_ctx->GetStream().get());
  comm_ctx->GetStream()->WaitEvent(calc_event.get());
}

std::shared_ptr<ProcessGroup::Task> ProcessGroupCustom::RunFnInXCCLEnv(
    std::function<void(const phi::stream::Stream&)> fn,
    const phi::DenseTensor& tensor,
    CommType comm_type,
    bool sync_op,
    bool use_calc_stream) {
  const auto& place = tensor.place();
  const auto& key = GetKeyFromPlace(place);

  phi::DeviceGuard guard(place);

  if (place_to_comm_ctx_.find(key) == place_to_comm_ctx_.end()) {
    CreateXCCLEnvCache(place, key);
  }

  if (!use_calc_stream) {
    SyncCalcStream(place);
  }

  auto task = CreateTask(place, rank_, comm_type, sync_op, use_calc_stream);

  const auto* calc_ctx = place_to_calc_ctx_.at(key);
  const auto& comm_ctx = place_to_comm_ctx_.at(key);
  auto& xccl_stream =
      use_calc_stream ? *calc_ctx->GetStream() : *comm_ctx->GetStream();
  fn(xccl_stream);

  if (!use_calc_stream) {
    if (FLAGS_use_stream_safe_cuda_allocator) {
      memory::RecordStream(tensor.Holder(), xccl_stream.raw_stream());
    }
    task->UpdateWaitChain(*comm_ctx);
  }

  return task;
}

// TODO(sunyilun): methods below will be removed later
void SyncDefaultStream(const std::vector<Place>& places,
                       phi::event::Event& xccl_event,                // NOLINT
                       std::vector<phi::CustomContext*>& dev_ctx) {  // NOLINT
  for (size_t i = 0; i < places.size(); ++i) {
    auto* default_ctx = static_cast<phi::CustomContext*>(
        phi::DeviceContextPool::Instance().Get(places[i]));
    xccl_event.Record(default_ctx->GetStream().get());
    dev_ctx[i]->GetStream()->WaitEvent(&xccl_event);
  }
}

std::shared_ptr<ProcessGroupCustom::XCCLTask> ProcessGroupCustom::CreateTask(
    std::vector<Place> places,
    int rank,
    CommType comm_type,
    const std::vector<phi::DenseTensor>& inputs) {
  return std::make_shared<ProcessGroupCustom::XCCLTask>(
      places, rank, comm_type, inputs);
}

// create XCCLManager cache for places_key
void ProcessGroupCustom::CreateXCCLManagerCache(
    const std::string& places_key, const std::vector<Place>& places) {
  PADDLE_ENFORCE_EQ(places_key.empty(),
                    false,
                    common::errors::PreconditionNotMet(
                        "Not able to create/get the XCCL Communicator since "
                        "the CustomPlace are not known"));

  phi::ccl::CCLRootId xccl_root_id;
  if (rank_ == 0) {
    phi::DeviceManager::CCLGetUniqueId(device_type_, &xccl_root_id);
  }
  BroadcastUniqueXCCLID(&xccl_root_id);

  VLOG(3) << "init xccl rank: " << rank_ << ", nranks: " << size_
          << ", place: " << places_key << ", xccl uniqueid: "
          << phi::ccl::SerializeXCCLUniqueId(xccl_root_id);

  std::vector<std::unique_ptr<phi::CustomContext>> dev_ctx;
  dev_ctx.resize(places.size());

  std::vector<phi::CustomContext*> dev_ctx_raw;
  dev_ctx_raw.resize(places.size());

  GroupStart(device_type_);

  for (size_t i = 0; i < places.size(); ++i) {
    phi::DeviceGuard guard(places[i]);

    dev_ctx[i] = std::make_unique<phi::CustomContext>(places[i]);
    dev_ctx[i]->SetAllocator(
        &(phi::DeviceContextPool::Instance().Get(places[i])->GetAllocator()));
    dev_ctx[i]->SetHostAllocator(&(
        phi::DeviceContextPool::Instance().Get(places[i])->GetHostAllocator()));
    dev_ctx[i]->SetZeroAllocator(&(
        phi::DeviceContextPool::Instance().Get(places[i])->GetZeroAllocator()));
    dev_ctx[i]->SetHostZeroAllocator(&(phi::DeviceContextPool::Instance()
                                           .Get(places[i])
                                           ->GetHostZeroAllocator()));

    phi::ccl::CCLComm xccl_comm;
    phi::DeviceManager::CCLCommInitRank(
        device_type_, GetSize(), &xccl_root_id, GetRank(), &xccl_comm);

    dev_ctx[i]->set_xccl_comm(xccl_comm);
    dev_ctx_raw[i] = dev_ctx[i].get();
  }

  GroupEnd(device_type_);

  // TODO(sunyilun): for compatibility, will be removed later
  auto xccl_event = std::make_unique<phi::event::Event>();
  xccl_event->Init(places[0]);
  place_to_calc_event_.emplace(places_key, std::move(xccl_event));
  place_to_calc_ctx_.emplace(
      places_key,
      static_cast<phi::CustomContext*>(
          phi::DeviceContextPool::Instance().Get(places[0])));
  place_to_comm_ctx_.emplace(places_key, std::move(dev_ctx[0]));

  // These caches will be useful to process sync/wait/communicate
  places_to_ctx_.emplace(places_key, std::move(dev_ctx_raw));
}

template <typename Fn>
std::shared_ptr<ProcessGroup::Task> ProcessGroupCustom::Collective(
    std::vector<phi::DenseTensor>& inputs,
    std::vector<phi::DenseTensor>& outputs,
    Fn fn,
    CommType op_type) {
  CheckTensorContiguous(inputs);
  CheckTensorContiguous(outputs);

  const auto places = GetPlaceList(inputs);
  const auto key = GetKeyFromPlaces(places);

  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (place_to_comm_ctx_.find(key) == place_to_comm_ctx_.end()) {
      CreateXCCLEnvCache(places[0], key);
    }
  }

  SyncDefaultStream(
      places, *place_to_calc_event_.at(key), places_to_ctx_.at(key));

  auto task = CreateTask(places, rank_, op_type, inputs);

  // construct uninitialize guard for device
  {
    GroupStart(device_type_);
    for (size_t i = 0; i < inputs.size(); ++i) {
      phi::DeviceGuard guard(places[i]);
      const auto& xccl_stream = *places_to_ctx_.at(key)[i]->GetStream();
      fn(inputs[i],
         outputs[i],
         places_to_ctx_.at(key)[i]->xccl_comm(),
         xccl_stream);
    }
    GroupEnd(device_type_);
  }

  if (FLAGS_use_stream_safe_cuda_allocator) {
    for (size_t i = 0; i < inputs.size(); ++i) {
      phi::DeviceGuard guard(places[i]);
      memory::RecordStream(inputs[i].Holder(),
                           places_to_ctx_.at(key)[i]->stream());
    }
  }

  for (size_t i = 0; i < inputs.size(); ++i) {
    phi::DeviceGuard guard(places[i]);
    task->UpdateWaitChain(*places_to_ctx_.at(key)[i]);
  }
  return task;
}

template <typename Fn>
std::shared_ptr<ProcessGroup::Task> ProcessGroupCustom::PointToPoint(
    std::vector<phi::DenseTensor>& tensors,
    Fn fn,
    int dst_rank,
    CommType op_type) {
  CheckTensorContiguous(tensors);

  const auto places = GetPlaceList(tensors);
  const auto key = GetKeyFromPlaces(places);

  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (place_to_comm_ctx_.find(key) == place_to_comm_ctx_.end()) {
      CreateXCCLManagerCache(key, places);
    }
  }

  SyncDefaultStream(
      places, *place_to_calc_event_.at(key), places_to_ctx_.at(key));

  auto task = CreateTask(places, rank_, op_type, tensors);

  // construct uninitialize guard for device

  {
    GroupStart(device_type_);
    for (size_t i = 0; i < tensors.size(); ++i) {
      phi::DeviceGuard guard(places[i]);

      const auto& xccl_stream = *places_to_ctx_.at(key)[i]->GetStream();
      fn(tensors[i],
         places_to_ctx_.at(key)[i]->xccl_comm(),
         xccl_stream,
         dst_rank);
    }
    GroupEnd(device_type_);
  }

  if (FLAGS_use_stream_safe_cuda_allocator) {
    for (size_t i = 0; i < tensors.size(); ++i) {
      phi::DeviceGuard guard(places[i]);
      memory::RecordStream(tensors[i].Holder(),
                           places_to_ctx_.at(key)[i]->stream());
    }
  }

  for (size_t i = 0; i < tensors.size(); ++i) {
    phi::DeviceGuard guard(places[i]);
    task->UpdateWaitChain(*places_to_ctx_.at(key)[i]);
  }
  return task;
}

std::shared_ptr<ProcessGroup::Task> ProcessGroupCustom::AllReduce(
    std::vector<phi::DenseTensor>& in_tensors,
    std::vector<phi::DenseTensor>& out_tensors,
    const AllreduceOptions& opts) {
  CheckTensorContiguous(in_tensors);
  CheckTensorContiguous(out_tensors);

  PADDLE_ENFORCE_EQ(
      CheckTensorsInCustomPlace(in_tensors, device_type_),
      true,
      common::errors::InvalidArgument("All inputs should be in CustomPlace."));
  return Collective(
      in_tensors,
      out_tensors,
      [&](const phi::DenseTensor& input,
          phi::DenseTensor& output,
          const phi::ccl::CCLComm& comm,
          const phi::stream::Stream& stream) {
        auto comm_context = this->GetCommContext();
        comm_context->AllReduce(
            &output,
            input,
            paddle::distributed::ToXCCLRedType(opts.reduce_op),
            stream);
      },
      CommType::ALLREDUCE);
}

std::shared_ptr<ProcessGroup::Task> ProcessGroupCustom::Broadcast(
    std::vector<phi::DenseTensor>& in_tensors,
    std::vector<phi::DenseTensor>& out_tensors,
    const BroadcastOptions& opts) {
  CheckTensorContiguous(in_tensors);
  CheckTensorContiguous(out_tensors);

  PADDLE_ENFORCE_EQ(
      CheckTensorsInCustomPlace(in_tensors, device_type_),
      true,
      common::errors::InvalidArgument("All inputs should be in CustomPlace."));

  return Collective(
      in_tensors,
      out_tensors,
      [&](phi::DenseTensor& input,
          phi::DenseTensor& output,
          const phi::ccl::CCLComm& comm,
          const phi::stream::Stream& stream) {
        const auto root =
            opts.source_rank * in_tensors.size() + opts.source_root;
        auto comm_context = this->GetCommContext();
        comm_context->Broadcast(&output, input, root, stream);
      },
      CommType::BROADCAST);
}

inline void CheckTensorsInDifferentDevices(
    const std::vector<phi::DenseTensor>& tensors, const size_t num_devices) {
  PADDLE_ENFORCE_EQ(
      tensors.empty(),
      false,
      common::errors::InvalidArgument("Tensor list must be nonempty."));
  PADDLE_ENFORCE_LE(
      tensors.size(),
      num_devices,
      common::errors::InvalidArgument("Tensor list mustn't be larger than the "
                                      "number of available CustomDevices."));

  std::set<Place> used_devices;

  for (const auto& t : tensors) {
    PADDLE_ENFORCE_EQ(phi::is_custom_place(t.place()),
                      true,
                      common::errors::InvalidArgument(
                          "Tensors must be CustomDevice and dense tensor."));

    const auto inserted = used_devices.insert(t.place()).second;
    PADDLE_ENFORCE_EQ(inserted,
                      true,
                      common::errors::InvalidArgument(
                          "Tensors must be on distinct CustomDevice devices."));
  }
}

std::shared_ptr<ProcessGroup::Task> ProcessGroupCustom::Send(
    std::vector<phi::DenseTensor>& tensors, int dst_rank) {
  CheckTensorContiguous(tensors);

  CheckTensorsInDifferentDevices(tensors, static_cast<size_t>(GetSize()));

  auto task = PointToPoint(
      tensors,
      [&](phi::DenseTensor& input,
          const phi::ccl::CCLComm& comm,
          const phi::stream::Stream& stream,
          int dst_rank) {
        auto comm_context = this->GetCommContext();
        comm_context->Send(input, input.numel(), dst_rank, stream);
      },
      dst_rank,
      CommType::SEND);
  return task;
}

std::shared_ptr<ProcessGroup::Task> ProcessGroupCustom::Recv(
    std::vector<phi::DenseTensor>& tensors, int src_rank) {
  CheckTensorContiguous(tensors);

  CheckTensorsInDifferentDevices(tensors, static_cast<size_t>(GetSize()));

  auto task = PointToPoint(
      tensors,
      [&](phi::DenseTensor& output,
          const phi::ccl::CCLComm& comm,
          const phi::stream::Stream& stream,
          int src_rank) {
        auto comm_context = this->GetCommContext();
        comm_context->Recv(&output, output.numel(), src_rank, stream);
      },
      src_rank,
      CommType::RECV);
  return task;
}

std::shared_ptr<ProcessGroup::Task> ProcessGroupCustom::AllGather(
    std::vector<phi::DenseTensor>& in_tensors,
    std::vector<phi::DenseTensor>& out_tensors) {
  CheckTensorContiguous(in_tensors);
  CheckTensorContiguous(out_tensors);

  PADDLE_ENFORCE_EQ(
      CheckTensorsInCustomPlace(in_tensors, device_type_),
      true,
      common::errors::InvalidArgument("All inputs should be in CustomPlace."));
  PADDLE_ENFORCE_EQ(
      CheckTensorsInCustomPlace(out_tensors, device_type_),
      true,
      common::errors::InvalidArgument("All outputs should be in CustomPlace."));
  return Collective(
      in_tensors,
      out_tensors,
      [&](const phi::DenseTensor& input,
          phi::DenseTensor& output,
          const phi::ccl::CCLComm& comm,
          const phi::stream::Stream& stream) {
        auto comm_context = this->GetCommContext();
        comm_context->AllGather(&output, input, stream);
      },
      CommType::ALLGATHER);
}

std::shared_ptr<ProcessGroup::Task> ProcessGroupCustom::AllToAll(
    std::vector<phi::DenseTensor>& in_tensors,
    std::vector<phi::DenseTensor>& out_tensors) {
  CheckTensorContiguous(in_tensors);
  CheckTensorContiguous(out_tensors);

  PADDLE_ENFORCE_EQ(
      CheckTensorsInCustomPlace(in_tensors, device_type_),
      true,
      common::errors::InvalidArgument("All inputs should be in CustomPlace."));
  PADDLE_ENFORCE_EQ(
      CheckTensorsInCustomPlace(out_tensors, device_type_),
      true,
      common::errors::InvalidArgument("All inputs should be in CustomPlace."));
  return Collective(
      in_tensors,
      out_tensors,
      [&](phi::DenseTensor& input,
          phi::DenseTensor& output,
          const phi::ccl::CCLComm& comm,
          const phi::stream::Stream& stream) {
        auto comm_context = this->GetCommContext();

        size_t offset = 0;
        std::vector<void*> send_buf, recv_buf;
        std::vector<size_t> send_count(size_, input.numel() / size_),
            recv_count(size_, input.numel() / size_);
        std::vector<phi::DataType> send_dtype(size_, input.dtype()),
            recv_dtype(size_, input.dtype());
        for (auto i = 0; i < size_; i++) {
          send_buf.push_back(
              GetPointerByOffset(input.data(), offset, input.dtype()));
          recv_buf.push_back(
              GetPointerByOffset(output.data(), offset, input.dtype()));
          offset += input.numel() / size_;
        }
        phi::DeviceManager::CCLAllToAll(
            device_type_,
            const_cast<const void**>(send_buf.data()),
            send_count.data(),
            send_dtype.data(),
            recv_buf.data(),
            recv_count.data(),
            recv_dtype.data(),
            rank_,
            size_,
            comm_context->GetXcclComm(),
            stream);
      },
      CommType::ALLTOALL);
}

std::shared_ptr<ProcessGroup::Task> ProcessGroupCustom::Reduce(
    std::vector<phi::DenseTensor>& in_tensors,
    std::vector<phi::DenseTensor>& out_tensors,
    const ReduceOptions& opts) {
  CheckTensorContiguous(in_tensors);
  CheckTensorContiguous(out_tensors);

  PADDLE_ENFORCE_EQ(
      CheckTensorsInCustomPlace(in_tensors, device_type_),
      true,
      common::errors::InvalidArgument("All inputs should be in CustomPlace."));
  return Collective(
      in_tensors,
      out_tensors,
      [&](const phi::DenseTensor& input,
          phi::DenseTensor& output,
          const phi::ccl::CCLComm& comm,
          const phi::stream::Stream& stream) {
        auto comm_context = this->GetCommContext();
        comm_context->Reduce(&output,
                             input,
                             paddle::distributed::ToXCCLRedType(opts.reduce_op),
                             opts.root_rank,
                             stream);
      },
      CommType::REDUCE);
}

std::shared_ptr<ProcessGroup::Task> ProcessGroupCustom::Scatter(
    std::vector<phi::DenseTensor>& in_tensors,
    std::vector<phi::DenseTensor>& out_tensors,
    const ScatterOptions& opts) {
  CheckTensorContiguous(in_tensors);
  CheckTensorContiguous(out_tensors);

  PADDLE_ENFORCE_EQ(
      CheckTensorsInCustomPlace(in_tensors, device_type_),
      true,
      common::errors::InvalidArgument("All inputs should be in CustomPlace."));
  PADDLE_ENFORCE_EQ(
      CheckTensorsInCustomPlace(out_tensors, device_type_),
      true,
      common::errors::InvalidArgument("All inputs should be in CustomPlace."));
  return Collective(
      in_tensors,
      out_tensors,
      [&](phi::DenseTensor& input,
          phi::DenseTensor& output,
          const phi::ccl::CCLComm& comm,
          const phi::stream::Stream& stream) {
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

std::shared_ptr<ProcessGroupCustom>
ProcessGroupCustom::CreateProcessGroupCustom(
    const std::shared_ptr<phi::distributed::Store>& store,
    const std::string& device_type,
    int rank,
    int size,
    int gid) {
  auto process_group =
      std::make_shared<ProcessGroupCustom>(store, device_type, rank, size, gid);
  ProcessGroupIdMap::GetInstance().emplace(gid, process_group);
  return process_group;
}

phi::distributed::XCCLCommContext* ProcessGroupCustom::GetCommContext() {
  const auto& comm_context_manager =
      phi::distributed::CommContextManager::GetInstance();
  auto comm_context = static_cast<phi::distributed::XCCLCommContext*>(
      comm_context_manager.Get(std::to_string(this->gid_)));
  PADDLE_ENFORCE_NE(comm_context,
                    nullptr,
                    common::errors::Unavailable("XCCLCommContext is nullptr"));
  return comm_context;
}

}  //  namespace distributed
}  //  namespace paddle
