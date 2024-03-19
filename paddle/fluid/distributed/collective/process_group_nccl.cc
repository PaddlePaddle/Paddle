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
#include "paddle/fluid/platform/cuda_device_guard.h"
#include "paddle/fluid/platform/device/gpu/nccl_helper.h"
#include "paddle/phi/api/lib/utils/allocator.h"
#include "paddle/phi/backends/gpu/gpu_info.h"
#include "paddle/phi/core/distributed/check/nccl_dynamic_check.h"
#include "paddle/phi/core/distributed/check/static_check.h"
#include "paddle/phi/core/distributed/comm_context_manager.h"
#include "paddle/phi/core/distributed/comm_task_manager.h"
#include "paddle/phi/core/distributed/nccl_comm_task.h"
#include "paddle/phi/core/distributed/nccl_tools.h"
#include "paddle/phi/core/distributed/trace_utils.h"
#include "paddle/phi/core/distributed/utils.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/utils/data_type.h"

DECLARE_bool(benchmark);
DECLARE_bool(benchmark_nccl);
DECLARE_bool(nccl_blocking_wait);
DECLARE_bool(use_stream_safe_cuda_allocator);
DECLARE_bool(enable_async_trace);

// set this flag to `true` and recompile to enable dynamic checks
constexpr bool FLAGS_enable_nccl_dynamic_check = false;
constexpr int64_t kWaitBlockTImeout = 10;

namespace paddle {
namespace distributed {

using phi::distributed::GetTraceEndKey;
using phi::distributed::GetTraceStartKey;
using phi::distributed::IsP2POP;
using phi::distributed::NCCLDTypeToString;
using phi::distributed::NCCLRedTypeToString;
using phi::distributed::SerializeNCCLUniqueId;
using phi::distributed::ToNCCLRedType;
uint64_t ProcessGroupNCCL::s_group_call_counter = 0;

ProcessGroupNCCL::NCCLTask::NCCLTask(const Place& place,
                                     int rank,
                                     CommType comm_type,
                                     bool sync_op,
                                     bool use_calc_stream,
                                     int gid)
    : TaskStream(rank, comm_type, sync_op, use_calc_stream),
      comm_event_(place, platform::GenerateDeviceEventFlag()),
      task_place_(place),
      gid_(gid) {}

ProcessGroupNCCL::NCCLTask::~NCCLTask() {}

bool ProcessGroupNCCL::NCCLTask::IsCompleted() { return comm_event_.Query(); }

void ProcessGroupNCCL::NCCLTask::UpdateWaitChain(
    const phi::DeviceContext& ctx) {
  comm_event_.Record(&ctx);
}

void ProcessGroupNCCL::NCCLTask::RemoveHolderStreamInGroup() {
  auto map = distributed::ProcessGroupMapFromGid::getInstance();
  distributed::ProcessGroup* pg = map->get(gid_);
  if (!pg) return;
  auto* pg_nccl = dynamic_cast<ProcessGroupNCCL*>(pg);
  if (!pg_nccl) return;
  pg_nccl->EraseTensorHolders();
}

// TODO(sheniang03): Add timeout for wait, now timeout unused
bool ProcessGroupNCCL::NCCLTask::Wait(std::chrono::milliseconds timeout) {
  // Warning here when use calc stream but also invoke waiting explicitly.
  if (UseCalcStream()) {
    VLOG(5) << "Warning: The communication is on calc stream, wait here is "
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
  RemoveHolderStreamInGroup();
  return true;
}

// Same as Wait
void ProcessGroupNCCL::NCCLTask::Synchronize() { Wait(kWaitTimeout); }

ProcessGroupNCCL::ProcessGroupNCCL(
    const std::shared_ptr<phi::distributed::Store>& store,
    int rank,
    int size,
    int gid,
    int64_t timeout,
    int nccl_comm_init_option)
    : ProcessGroupWithStream(rank, size, gid),
      store_(store),
      pg_timeout_(timeout),
      nccl_comm_init_option_(nccl_comm_init_option) {
  LOG(INFO) << "ProcessGroupNCCL pg_timeout_ " << pg_timeout_;
  LOG(INFO) << "ProcessGroupNCCL nccl_comm_init_option_ "
            << nccl_comm_init_option_;
}
ProcessGroupNCCL::~ProcessGroupNCCL() {
  LOG(INFO) << "ProcessGroupNCCL destruct ";
  auto& comm_task_manager = phi::distributed::CommTaskManager::GetInstance();
  comm_task_manager.Stop();
}

void ProcessGroupNCCL::GroupStart() {
  NCCL_CHECK(phi::dynload::ncclGroupStart());
  ++s_group_call_counter;
}

void ProcessGroupNCCL::GroupEnd() {
  NCCL_CHECK(phi::dynload::ncclGroupEnd());
  --s_group_call_counter;
  // NOTE: This is to sync the calc stream and comm stream for debug using
  // batch_isend_irecv
  if (FLAGS_benchmark || FLAGS_benchmark_nccl) {
    PADDLE_ENFORCE_GPU_SUCCESS(cudaDeviceSynchronize());
  }
}

phi::DeviceContext* ProcessGroupNCCL::GetDeviceContext(
    const Place& place) const {
  return GetDeviceContext(place, /*use_calc_stream*/ false);
}

// NOTE(shenliang03): GetDeviceContext is only used for collective, it can't
// be used for p2p op.
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
  phi::distributed::CommStaticCheck::GatherLikeShape(*out_tensor,
                                                     in_tensor_maybe_partial,
                                                     /*dst_rank*/ rank_,
                                                     /*cur_rank*/ rank_,
                                                     size_);
  return Collective(
      [&](ncclComm_t comm, gpuStream_t stream) {
        if (FLAGS_enable_nccl_dynamic_check) {
          phi::distributed::NCCLDynamicCheck::CheckShape(*out_tensor,
                                                         /*root_rank*/ 0,
                                                         rank_,
                                                         comm);
        }

        VLOG(3) << "[ncclAllGather] "
                << "sendbuff: " << in_tensor_maybe_partial.data()
                << ", recvbuff: " << out_tensor->data()
                << ", count: " << in_tensor_maybe_partial.numel()
                << ", datatype: "
                << NCCLDTypeToString(
                       phi::ToNCCLDataType(in_tensor_maybe_partial.dtype()))
                << ", ncclcomm: " << comm << ", stream: " << stream
                << ", offset: " << offset << ", sync_op: " << sync_op
                << ", use_calc_stream: " << use_calc_stream << ", "
                << GetGroupMessage();

        NCCL_CHECK(phi::dynload::ncclAllGather(
            in_tensor_maybe_partial.data(),
            out_tensor->data(),
            in_tensor_maybe_partial.numel(),
            phi::ToNCCLDataType(in_tensor_maybe_partial.dtype()),
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
  phi::distributed::CommStaticCheck::SameShape(*out_tensor,
                                               in_tensor,
                                               /*dst_rank*/ rank_,
                                               /*cur_rank*/ rank_,
                                               size_);
  return Collective(
      [&](ncclComm_t comm, gpuStream_t stream) {
        if (FLAGS_enable_nccl_dynamic_check) {
          phi::distributed::NCCLDynamicCheck::CheckShape(*out_tensor,
                                                         /*root_rank*/ 0,
                                                         rank_,
                                                         comm);
        }

        VLOG(3) << "[ncclAllReduce] "
                << "sendbuff: " << in_tensor.data()
                << ", recvbuff: " << out_tensor->data()
                << ", count: " << in_tensor.numel() << ", datatype: "
                << NCCLDTypeToString(phi::ToNCCLDataType(in_tensor.dtype()))
                << ", redop: "
                << NCCLRedTypeToString(ToNCCLRedType(opts.reduce_op))
                << ", ncclcomm: " << comm << ", stream: " << stream
                << ", sync_op: " << sync_op
                << ", use_calc_stream: " << use_calc_stream << ", "
                << GetGroupMessage();

        NCCL_CHECK(
            phi::dynload::ncclAllReduce(in_tensor.data(),
                                        out_tensor->data(),
                                        in_tensor.numel(),
                                        phi::ToNCCLDataType(in_tensor.dtype()),
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
      phi::errors::InvalidArgument(
          "The length of size_on_each_rank must be equal to world_size."));

  int64_t sum_size_on_each_rank =
      std::accumulate(size_on_each_rank.begin(), size_on_each_rank.end(), 0);
  PADDLE_ENFORCE_EQ(
      sum_size_on_each_rank,
      tensor_dim[0],
      phi::errors::InvalidArgument(
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
  VLOG(3) << "[AllToAll] start";
  CheckSizeOnEachRank(out_dim, out_size_each_rank, size_);
  CheckSizeOnEachRank(in_dim, in_size_each_rank, size_);

  // NOTE: Since `all_to_all` needs other processes' participation, it cannot
  // simply be covered by static checks. Factors are set to 0 here to skip the
  // shape check. Its shape check will be done by dynamic checks with
  // FLAGS_enable_nccl_dynamic_check.
  if (out_tensor->numel()==0)
    VLOG(3) << "[AllToAll] rank[" << rank_ << "] do not recv data";

  if (in_tensor.numel()==0)
    VLOG(3) << "[AllToAll] rank[" << rank_ << "] do not send data";

  if (out_tensor->numel() != 0 && in_tensor.numel() != 0){
    phi::distributed::CommStaticCheck::CheckShape(*out_tensor,
                                                  in_tensor,
                                                  /*dst_rank*/ rank_,
                                                  /*cur_rank*/ rank_,
                                                  size_,
                                                  /*out_size_factor*/ 0,
                                                  /*in_size_factor*/ 0);
  }
  return Collective(
      [&](ncclComm_t comm, gpuStream_t stream) {
        if (FLAGS_enable_nccl_dynamic_check) {
          phi::distributed::NCCLDynamicCheck::CheckShape(
              *out_tensor, in_tensor, in_size_each_rank, rank_, size_, comm);
        }
        int64_t in_row_size = in_dim[0] != 0 ? in_tensor.numel() / in_dim[0] : 0;
        int64_t out_row_size = out_dim[0] != 0 ? out_tensor->numel() / out_dim[0]: 0;
        int64_t in_offset = 0, in_numel = 0, out_offset = 0, out_numel = 0;
        phi::DenseTensor input_partial, output_partial;

        VLOG(3) << "[AllToAll] "
                << "sendbuff: " << in_tensor.data()
                << ", recvbuff: " << out_tensor->data()
                << ", count: " << in_tensor.numel() << ", datatype: "
                << NCCLDTypeToString(phi::ToNCCLDataType(in_tensor.dtype()))
                << ", ncclcomm: " << comm << ", stream: " << stream
                << ", out_size_each_rank: "
                << string::join_strings(out_size_each_rank, ',')
                << ", in_size_each_rank: "
                << string::join_strings(in_size_each_rank, ',')
                << ", sync_op: " << sync_op
                << ", use_calc_stream: " << use_calc_stream << ", "
                << GetGroupMessage();

        GroupStart();
        for (auto i = 0; i < size_; i++) {
          in_numel = in_size_each_rank[i] * in_row_size;
          if (in_numel != 0){
            input_partial = GetPartialTensor(in_tensor, in_offset, in_numel);
            VLOG(3) << "[AllToAll] skip send empty data to rank:" << i;
            NCCL_CHECK(
                phi::dynload::ncclSend(input_partial.data(),
                                      in_numel,
                                      phi::ToNCCLDataType(input_partial.dtype()),
                                      i,
                                      comm,
                                      stream));
            in_offset += in_numel;
          }

          out_numel = out_size_each_rank[i] * out_row_size;
          if (out_numel != 0){
            output_partial = GetPartialTensor(*out_tensor, out_offset, out_numel);
            VLOG(3) << "[AllToAll] skip recv empty data from rank:" << i;
            NCCL_CHECK(phi::dynload::ncclRecv(
                output_partial.data(),
                out_numel,
                phi::ToNCCLDataType(output_partial.dtype()),
                i,
                comm,
                stream));
            out_offset += out_numel;
          }
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
                    phi::errors::PreconditionNotMet(
                        "The barrier device id must greater or equal than 0."));
  platform::CUDAPlace place(opts.device_id);
  auto allocator = std::unique_ptr<phi::Allocator>(
      new paddle::experimental::DefaultAllocator(place));
  phi::DenseTensorMeta meta(phi::DataType::FLOAT32, phi::DDim{1});
  phi::DenseTensor barrier_tensor{allocator.get(), meta};

  VLOG(3) << "[Barrier] "
          << "barrier opt: " << opts.device_id;

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
  phi::distributed::CommStaticCheck::SameShape(*out_tensor,
                                               in_tensor,
                                               /*dst_rank*/ rank_,
                                               /*cur_rank*/ rank_,
                                               size_);
  return Collective(
      [&](ncclComm_t comm, gpuStream_t stream) {
        int root = opts.source_rank + opts.source_root;
        if (FLAGS_enable_nccl_dynamic_check) {
          phi::distributed::NCCLDynamicCheck::CheckShape(
              *out_tensor, root, rank_, comm);
        }

        VLOG(3) << "[ncclBroadcast] "
                << "sendbuff: " << in_tensor.data()
                << ", recvbuff: " << out_tensor->data()
                << ", count: " << in_tensor.numel() << ", datatype: "
                << NCCLDTypeToString(phi::ToNCCLDataType(in_tensor.dtype()))
                << ", root: " << root << ", ncclcomm: " << comm
                << ", stream: " << stream << ", sync_op: " << sync_op
                << ", use_calc_stream: " << use_calc_stream << ", "
                << GetGroupMessage();

        NCCL_CHECK(
            phi::dynload::ncclBroadcast(in_tensor.data(),
                                        out_tensor->data(),
                                        in_tensor.numel(),
                                        phi::ToNCCLDataType(in_tensor.dtype()),
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
  phi::distributed::CommStaticCheck::SameShape(*out_tensor,
                                               in_tensor,
                                               /*dst_rank*/ opts.root_rank,
                                               /*cur_rank*/ rank_,
                                               size_);
  return Collective(
      [&](ncclComm_t comm, gpuStream_t stream) {
        if (FLAGS_enable_nccl_dynamic_check) {
          phi::distributed::NCCLDynamicCheck::CheckShape(
              *out_tensor,
              /*root_rank*/ opts.root_rank,
              rank_,
              comm);
        }

        VLOG(3) << "[ncclReduce] "
                << "sendbuff: " << in_tensor.data()
                << ", recvbuff: " << out_tensor->data()
                << ", count: " << in_tensor.numel() << ", datatype: "
                << NCCLDTypeToString(phi::ToNCCLDataType(in_tensor.dtype()))
                << ", redop: "
                << NCCLRedTypeToString(ToNCCLRedType(opts.reduce_op))
                << ", root: " << opts.root_rank << ", ncclcomm: " << comm
                << ", stream: " << stream << ", sync_op: " << sync_op
                << ", use_calc_stream: " << use_calc_stream << ", "
                << GetGroupMessage();

        NCCL_CHECK(
            phi::dynload::ncclReduce(in_tensor.data(),
                                     out_tensor->data(),
                                     in_tensor.numel(),
                                     phi::ToNCCLDataType(in_tensor.dtype()),
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
  phi::distributed::CommStaticCheck::ScatterLikeShape(*out_tensor,
                                                      in_tensor,
                                                      /*dst_rank*/ rank_,
                                                      /*cur_rank*/ rank_,
                                                      size_);
  return Collective(
      [&](ncclComm_t comm, gpuStream_t stream) {
        if (FLAGS_enable_nccl_dynamic_check) {
          phi::distributed::NCCLDynamicCheck::CheckShape(*out_tensor,
                                                         /*root_rank*/ 0,
                                                         rank_,
                                                         comm);
        }

        VLOG(3) << "[ncclReduceScatter] "
                << "sendbuff: " << in_tensor.data()
                << ", recvbuff: " << out_tensor->data()
                << ", count: " << in_tensor.numel() << ", datatype: "
                << NCCLDTypeToString(phi::ToNCCLDataType(in_tensor.dtype()))
                << ", redop: "
                << NCCLRedTypeToString(ToNCCLRedType(opts.reduce_op))
                << ", ncclcomm: " << comm << ", stream: " << stream
                << ", sync_op: " << sync_op
                << ", use_calc_stream: " << use_calc_stream << ", "
                << GetGroupMessage();

        NCCL_CHECK(phi::dynload::ncclReduceScatter(
            in_tensor.data(),
            out_tensor->data(),
            out_tensor->numel(),
            phi::ToNCCLDataType(in_tensor.dtype()),
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
  phi::distributed::CommStaticCheck::ScatterLikeShape(
      *out_tensor,
      in_tensor,
      /*dst_rank*/ opts.root_rank,
      /*cur_rank*/ rank_,
      size_);
  return Collective(
      [&](ncclComm_t comm, gpuStream_t stream) {
        if (FLAGS_enable_nccl_dynamic_check) {
          phi::distributed::NCCLDynamicCheck::CheckShape(
              *out_tensor,
              /*root_rank*/ opts.root_rank,
              rank_,
              comm);
        }

        VLOG(3) << "[Scatter] "
                << "sendbuff: " << in_tensor.data()
                << ", recvbuff: " << out_tensor->data()
                << ", count: " << in_tensor.numel() << ", datatype: "
                << NCCLDTypeToString(phi::ToNCCLDataType(in_tensor.dtype()))
                << ", root: " << opts.root_rank << ", ncclcomm: " << comm
                << ", stream: " << stream << ", sync_op: " << sync_op
                << ", use_calc_stream: " << use_calc_stream << ", "
                << GetGroupMessage();

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
                phi::ToNCCLDataType(partial_tensor.dtype()),
                i,
                comm,
                stream));
            offset += numel;
          }
          NCCL_CHECK(
              phi::dynload::ncclRecv(out_tensor->data(),
                                     numel,
                                     phi::ToNCCLDataType(out_tensor->dtype()),
                                     opts.root_rank,
                                     comm,
                                     stream));
          GroupEnd();
        } else {
          NCCL_CHECK(
              phi::dynload::ncclRecv(out_tensor->data(),
                                     numel,
                                     phi::ToNCCLDataType(out_tensor->dtype()),
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
  auto gather_func = [&](ncclComm_t comm, gpuStream_t stream) {
    // shape check
    if (FLAGS_enable_nccl_dynamic_check) {
      phi::distributed::NCCLDynamicCheck::CheckGatherShape(
          in_tensor, gather_tensors, opts.root_rank, rank_, size_, comm);
    }

    VLOG(3) << "[Gather] "
            << "sendbuff: " << in_tensor.data()
            << ", count: " << in_tensor.numel() << ", datatype: "
            << NCCLDTypeToString(phi::ToNCCLDataType(in_tensor.dtype()))
            << ", root: " << opts.root_rank << ", ncclcomm: " << comm
            << ", stream: " << stream << ", sync_op: " << sync_op
            << ", use_calc_stream: " << use_calc_stream << ", "
            << GetGroupMessage();

    GroupStart();
    // root receive from all devices
    if (rank_ == opts.root_rank) {
      for (auto i = 0; i < size_; i++) {
        auto& gather_tensor = gather_tensors[i];
        NCCL_CHECK(
            phi::dynload::ncclRecv(gather_tensor.data(),
                                   gather_tensor.numel(),
                                   phi::ToNCCLDataType(gather_tensor.dtype()),
                                   i,
                                   comm,
                                   stream));
      }
    }
    // send to root
    NCCL_CHECK(phi::dynload::ncclSend(in_tensor.data(),
                                      in_tensor.numel(),
                                      phi::ToNCCLDataType(in_tensor.dtype()),
                                      opts.root_rank,
                                      comm,
                                      stream));
    GroupEnd();
  };
  return Collective(
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

  phi::distributed::CommStaticCheck::CheckShape(*tensor, rank_, size_);
  return Point2Point(
      [&](ncclComm_t comm, gpuStream_t stream, int rank_in_group) {
        if (FLAGS_enable_nccl_dynamic_check) {
          phi::distributed::NCCLDynamicCheck::CheckShape(*tensor,
                                                         /*root_rank*/ src_rank,
                                                         rank_,
                                                         comm);
        }

        VLOG(3) << "[ncclRecv] "
                << "recvbuff: " << tensor->data()
                << ", count: " << tensor->numel() << ", datatype: "
                << NCCLDTypeToString(phi::ToNCCLDataType(tensor->dtype()))
                << ", src_in_group: " << src_rank << ", ncclcomm: " << comm
                << ", stream: " << stream << ", offset: " << offset
                << ", sync_op: " << sync_op
                << ", use_calc_stream: " << use_calc_stream << ", "
                << GetGroupMessage();

        NCCL_CHECK(phi::dynload::ncclRecv(tensor->data(),
                                          tensor->numel(),
                                          phi::ToNCCLDataType(tensor->dtype()),
                                          rank_in_group,
                                          comm,
                                          stream));
      },
      src_rank,
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

  phi::distributed::CommStaticCheck::CheckShape(
      tensor_maybe_partial, rank_, size_);
  return Point2Point(
      [&](ncclComm_t comm, gpuStream_t stream, int rank_in_group) {
        if (FLAGS_enable_nccl_dynamic_check) {
          phi::distributed::NCCLDynamicCheck::CheckShape(tensor_maybe_partial,
                                                         /*root_rank*/ rank_,
                                                         rank_,
                                                         comm);
        }

        VLOG(3) << "[ncclSend] "
                << "sendbuff: " << tensor_maybe_partial.data()
                << ", count: " << tensor_maybe_partial.numel() << ", datatype: "
                << NCCLDTypeToString(
                       phi::ToNCCLDataType(tensor_maybe_partial.dtype()))
                << ", dst_in_group: " << dst_rank << ", ncclcomm: " << comm
                << ", stream: " << stream << ", offset: " << offset
                << ", sync_op: " << sync_op
                << ", use_calc_stream: " << use_calc_stream << ", "
                << GetGroupMessage();

        NCCL_CHECK(phi::dynload::ncclSend(
            tensor_maybe_partial.data(),
            tensor_maybe_partial.numel(),
            phi::ToNCCLDataType(tensor_maybe_partial.dtype()),
            rank_in_group,
            comm,
            stream));
      },
      dst_rank,
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
    bool use_calc_stream,
    int gid) {
  return std::make_shared<ProcessGroupNCCL::NCCLTask>(
      place, rank, comm_type, is_sync, use_calc_stream, gid);
}

void ProcessGroupNCCL::BroadcastUniqueNCCLID(ncclUniqueId* nccl_id,
                                             bool is_p2p_op,
                                             const std::string& p2p_key,
                                             int p2p_rank) {
  std::string store_key;

  if (!is_p2p_op) {
    store_key = "ProcessGroupNCCL/nccl_ids/" + std::to_string(gid_) + "/0";
  } else {
    store_key =
        "ProcessGroupNCCL/nccl_ids/" + std::to_string(gid_) + "/" + p2p_key;
  }

  if (rank_ == 0 || (is_p2p_op && p2p_rank == 0)) {
    std::vector<uint8_t> nccl_id_wrapper(
        reinterpret_cast<uint8_t*>(nccl_id),
        reinterpret_cast<uint8_t*>(nccl_id) + NCCL_UNIQUE_ID_BYTES);
    store_->set(store_key, nccl_id_wrapper);
  } else {
    const auto& nccl_id_wrapper = store_->get(store_key);
    std::memcpy(nccl_id, nccl_id_wrapper.data(), nccl_id_wrapper.size());
  }
  place_to_group_key_[p2p_key] = store_key;
}

void ProcessGroupNCCL::CreateNCCLEnvCache(const Place& place,
                                          const std::string& place_key,
                                          CommType comm_type,
                                          int p2p_rank) {
  ncclUniqueId nccl_id;

  bool is_batch_p2p = s_group_call_counter > 0;
  bool is_p2p_op = IsP2POP(comm_type, is_batch_p2p);

  if (rank_ == 0 || (is_p2p_op && p2p_rank == 0)) {
    NCCL_CHECK(phi::dynload::ncclGetUniqueId(&nccl_id));
  }

  BroadcastUniqueNCCLID(&nccl_id, is_p2p_op, place_key, p2p_rank);

  VLOG(3) << "init nccl rank_in_group: " << rank_ << ", nranks: " << size_
          << ", gid: " << gid_ << ", place key: " << place_key
          << ", nccl uniqueid: " << SerializeNCCLUniqueId(nccl_id);

  for (size_t i = 0; i < s_group_call_counter; ++i) {
    NCCL_CHECK(phi::dynload::ncclGroupEnd());
  }

  int num_ranks = is_p2p_op ? 2 : GetSize();
  int rank = is_p2p_op ? p2p_rank : GetRank();

  NCCL_CHECK(phi::dynload::ncclGroupStart());
  ncclComm_t nccl_comm;
  if (nccl_comm_init_option_ > 0 && phi::dynload::ncclCommInitRank2.IsValid()) {
    LOG(WARNING) << "Creating modified qp with ncclCommInitRank2.";
    NCCL_CHECK(phi::dynload::ncclCommInitRank2(
        &nccl_comm, num_ranks, nccl_id, rank, nccl_comm_init_option_));
  } else {
    if (nccl_comm_init_option_ > 0) {
      LOG(WARNING) << "ncclCommInitRank2 is not supported.";
    }
    NCCL_CHECK(
        phi::dynload::ncclCommInitRank(&nccl_comm, num_ranks, nccl_id, rank));
  }
  NCCL_CHECK(phi::dynload::ncclGroupEnd());

  VLOG(3) << "Get nccl comm: " << nccl_comm << " for place_key: " << place_key
          << " on rank_in_group: " << rank << " nranks: " << num_ranks
          << " gid: " << gid_;

  auto comm_ctx = std::make_unique<phi::GPUContext>(place);
  comm_ctx->set_nccl_comm(nccl_comm);

  if (FLAGS_enable_async_trace) {
    // gather global ranks in current group
    int* gpu_global_rank = nullptr;
    size_t gpu_global_rank_size = sizeof(int);
    CUDA_CHECK(cudaMalloc(&gpu_global_rank, gpu_global_rank_size));

    CUDA_CHECK(cudaMemcpy(gpu_global_rank,
                          &global_rank_,
                          gpu_global_rank_size,
                          cudaMemcpyHostToDevice));

    int* gpu_global_ranks = nullptr;
    size_t gpu_global_ranks_size = num_ranks * sizeof(int);
    CUDA_CHECK(cudaMalloc(&gpu_global_ranks, gpu_global_ranks_size));

    NCCL_CHECK(phi::dynload::ncclAllGather(gpu_global_rank,
                                           gpu_global_ranks,
                                           1,
                                           ncclInt,
                                           nccl_comm,
                                           comm_ctx->stream()));

    std::vector<int> global_ranks(num_ranks);
    CUDA_CHECK(cudaMemcpy(global_ranks.data(),
                          gpu_global_ranks,
                          gpu_global_ranks_size,
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(gpu_global_rank));
    CUDA_CHECK(cudaFree(gpu_global_ranks));

    // store global_ranks in current group_key
    std::once_flag flag;
    std::call_once(flag, [this]() {
      phi::distributed::CommContextManager::GetInstance().SetStore(store_);
      phi::distributed::CommTaskManager::GetInstance().SetTimeout(pg_timeout_);
    });

    std::string group_key = place_to_group_key_.at(place_key);
    phi::distributed::CommContextManager::GetInstance().AddGroupRanks(
        group_key, global_ranks);
  }

  auto* calc_ctx = static_cast<phi::GPUContext*>(
      platform::DeviceContextPool::Instance().Get(place));
  place_to_calc_event_.emplace(
      place_key,
      platform::DeviceEvent(place, platform::GenerateDeviceEventFlag()));
  place_to_calc_ctx_.emplace(place_key, calc_ctx);
  place_to_comm_ctx_.emplace(place_key, std::move(comm_ctx));

  for (size_t i = 0; i < s_group_call_counter; ++i) {
    NCCL_CHECK(phi::dynload::ncclGroupStart());
  }
}

void ProcessGroupNCCL::SyncCalcStream(const Place& place,
                                      const std::string& place_key) {
  auto& calc_event = place_to_calc_event_.at(place_key);
  const auto* calc_ctx = place_to_calc_ctx_.at(place_key);
  const auto* comm_ctx = place_to_comm_ctx_.at(place_key).get();
  calc_event.Record(calc_ctx);
  calc_event.Wait(platform::Place2DeviceType(place), comm_ctx);
}

std::shared_ptr<ProcessGroup::Task> ProcessGroupNCCL::Collective(
    std::function<void(ncclComm_t, gpuStream_t)> fn,
    const phi::DenseTensor& tensor,
    CommType comm_type,
    bool sync_op,
    bool use_calc_stream) {
  comm_seq_++;
  const auto& place = tensor.place();
  const auto& key = GetKeyFromPlace(place);

  platform::CUDADeviceGuard cuda_guard(place);

  if (place_to_comm_ctx_.find(key) == place_to_comm_ctx_.end()) {
    CreateNCCLEnvCache(place, key, comm_type);
  }

  if (!use_calc_stream) {
    SyncCalcStream(place, key);
  }

  auto task =
      CreateTask(place, rank_, comm_type, sync_op, use_calc_stream, gid_);

  const auto* calc_ctx = place_to_calc_ctx_.at(key);
  const auto& comm_ctx = place_to_comm_ctx_.at(key);
  auto nccl_comm = comm_ctx->nccl_comm();
  auto nccl_stream = use_calc_stream ? calc_ctx->stream() : comm_ctx->stream();

  if (!FLAGS_enable_async_trace) {
    fn(nccl_comm, nccl_stream);
  } else {
    std::string group_key = place_to_group_key_.at(key);
    auto comm_task =
        std::make_shared<phi::distributed::NCCLCommTask>(place,
                                                         group_key,
                                                         rank_,
                                                         size_,
                                                         gid_,
                                                         comm_seq_,
                                                         tensor.numel(),
                                                         sync_op,
                                                         use_calc_stream,
                                                         nccl_comm,
                                                         nccl_stream,
                                                         comm_type,
                                                         pg_timeout_);
    comm_task->StartRecord();
    fn(nccl_comm, nccl_stream);
    comm_task->EndRecord();
    comm_task->SetStore(store_);

    auto& comm_task_manager = phi::distributed::CommTaskManager::GetInstance();
    comm_task_manager.CommTaskEnqueue(std::move(comm_task));
  }

  if (!use_calc_stream) {
    if (FLAGS_use_stream_safe_cuda_allocator) {
      memory::RecordStream(tensor.Holder(), nccl_stream);
    }
    task->UpdateWaitChain(*comm_ctx);
    allocation_stream_pairs.emplace_back(tensor.Holder(), nccl_stream);
  }

  if (FLAGS_enable_nccl_dynamic_check) {
    task->SetBlockCPUInWait();
    task->Wait();
  }

  if (sync_op) {
    task->Wait();
  }

  if (FLAGS_benchmark || FLAGS_benchmark_nccl) {
    PADDLE_ENFORCE_GPU_SUCCESS(cudaDeviceSynchronize());
  }

  return task;
}

std::shared_ptr<ProcessGroup::Task> ProcessGroupNCCL::Point2Point(
    std::function<void(ncclComm_t, gpuStream_t, int)> fn,
    int peer,
    const phi::DenseTensor& tensor,
    CommType comm_type,
    bool sync_op,
    bool use_calc_stream) {
  const auto& place = tensor.place();

  int p2p_rank = 0;
  int p2p_target_rank = 0;
  bool is_batch_p2p = s_group_call_counter > 0;
  std::string key = "";

  int p2p_nrank = 0;
  if (is_batch_p2p) {
    key = GetKeyFromPlace(place);
    p2p_rank = rank_;
    p2p_target_rank = peer;
    p2p_nrank = GetSize();
  } else {
    int low_rank = rank_ < peer ? rank_ : peer;
    int high_rank = rank_ < peer ? peer : rank_;
    key = std::to_string(low_rank) + "->" + std::to_string(high_rank);
    p2p_rank = rank_ < peer ? 0 : 1;
    p2p_target_rank = 1 - p2p_rank;
    p2p_nrank = 2;
  }

  platform::CUDADeviceGuard cuda_guard(place);
  if (place_to_comm_ctx_.find(key) == place_to_comm_ctx_.end()) {
    CreateNCCLEnvCache(place, key, comm_type, p2p_rank);
  }
  if (p2p_comm_seq_.find(key) == p2p_comm_seq_.end()) {
    p2p_comm_seq_[key] = 0;
  }
  p2p_comm_seq_[key]++;

  if (!use_calc_stream) {
    SyncCalcStream(place, key);
  }

  auto task =
      CreateTask(place, rank_, comm_type, sync_op, use_calc_stream, gid_);
  const auto* calc_ctx = place_to_calc_ctx_.at(key);
  const auto& comm_ctx = place_to_comm_ctx_.at(key);

  auto nccl_comm = comm_ctx->nccl_comm();
  auto nccl_stream = use_calc_stream ? calc_ctx->stream() : comm_ctx->stream();

  std::string group_key = place_to_group_key_.at(key);
  auto comm_task =
      std::make_shared<phi::distributed::NCCLCommTask>(place,
                                                       group_key,
                                                       p2p_rank,
                                                       p2p_nrank,
                                                       gid_,
                                                       p2p_comm_seq_[key],
                                                       tensor.numel(),
                                                       sync_op,
                                                       use_calc_stream,
                                                       nccl_comm,
                                                       nccl_stream,
                                                       comm_type,
                                                       pg_timeout_);

  if (!FLAGS_enable_async_trace) {
    fn(nccl_comm, nccl_stream, p2p_target_rank);
  } else {
    comm_task->StartRecord();
    fn(nccl_comm, nccl_stream, p2p_target_rank);
    comm_task->EndRecord();
    comm_task->SetStore(store_);

    auto& comm_task_manager = phi::distributed::CommTaskManager::GetInstance();
    comm_task_manager.CommTaskEnqueue(std::move(comm_task));
  }

  if (!use_calc_stream) {
    if (FLAGS_use_stream_safe_cuda_allocator) {
      memory::RecordStream(tensor.Holder(), nccl_stream);
    }
    task->UpdateWaitChain(*comm_ctx);
    allocation_stream_pairs.emplace_back(tensor.Holder(), nccl_stream);
  }

  if (FLAGS_enable_nccl_dynamic_check) {
    task->SetBlockCPUInWait();
    task->Wait();
  }

  if (sync_op) {
    task->Wait();
  }

  if (!is_batch_p2p && (FLAGS_benchmark || FLAGS_benchmark_nccl)) {
    PADDLE_ENFORCE_GPU_SUCCESS(cudaDeviceSynchronize());
  }

  return task;
}

std::shared_ptr<ProcessGroupNCCL> ProcessGroupNCCL::CreateProcessGroupNCCL(
    const std::shared_ptr<phi::distributed::Store>& store,
    int rank,
    int size,
    int gid,
    int64_t timeout,
    int nccl_comm_init_option) {
  auto process_group = std::make_shared<ProcessGroupNCCL>(
      store, rank, size, gid, timeout, nccl_comm_init_option);
  ProcessGroupIdMap::GetInstance().emplace(gid, process_group);
  return process_group;
}

}  //  namespace distributed
}  //  namespace paddle
