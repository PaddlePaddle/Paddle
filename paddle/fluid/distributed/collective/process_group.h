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

#pragma once

#include <chrono>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "paddle/fluid/eager/api/utils/tensor_utils.h"  // NOTE: this header is required somewhere
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/device_context.h"
#include "paddle/phi/core/distributed/types.h"
#include "paddle/phi/core/distributed/utils.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/errors.h"

#define NCCL_CHECK(cmd)                            \
  do {                                             \
    ncclResult_t r = cmd;                          \
    if (r != ncclSuccess) {                        \
      printf("Failed, NCCL error %s:%d '%s'\n",    \
             __FILE__,                             \
             __LINE__,                             \
             phi::dynload::ncclGetErrorString(r)); \
      exit(EXIT_FAILURE);                          \
    }                                              \
  } while (0)

constexpr auto kWaitTimeout = std::chrono::milliseconds(0);

namespace paddle {
namespace distributed {

using phi::distributed::AllreduceOptions;
using phi::distributed::BarrierOptions;
using phi::distributed::BroadcastOptions;
using phi::distributed::CommType;
using phi::distributed::GatherOptions;
using phi::distributed::GetPartialTensor;
using phi::distributed::ReduceOp;
using phi::distributed::ReduceOptions;
using phi::distributed::ReduceScatterOptions;
using phi::distributed::ScatterOptions;
constexpr int kIgnoreId = -1;

class ProcessGroup {
 public:
  class Task {
   public:
    Task(int rank, CommType comm_type, bool sync_op)
        : rank_(rank), comm_type_(comm_type), sync_op_(sync_op) {}
    virtual ~Task() = default;

    virtual bool IsCompleted();
    virtual bool Wait(std::chrono::milliseconds timeout = kWaitTimeout) {
      return false;
    }
    virtual void Synchronize() {}
    virtual void UpdateWaitChain(const phi::DeviceContext& ctx) {}
    bool IsSync() const { return sync_op_; }

    // TODO(sunyilun): methods below will be removed later
    Task(int rank,
         const std::vector<phi::DenseTensor>& inputs,
         CommType comm_type)
        : rank_(rank), comm_type_(comm_type) {}
    Task(int rank,
         const std::vector<phi::DenseTensor>& inputs,
         CommType comm_type,
         bool sync_op)
        : rank_(rank), comm_type_(comm_type), sync_op_(sync_op) {}

   protected:
    const int rank_;
    CommType comm_type_{CommType::UNKNOWN};
    std::mutex mutex_;
    bool is_completed_{false};

   private:
    bool sync_op_{true};
  };

 public:
  ProcessGroup(int rank, int size, int gid);
  virtual ~ProcessGroup() = default;

  int GetRank() const { return rank_; }

  int GetSize() const { return size_; }

  virtual std::string GetBackendName() const = 0;

  virtual phi::DeviceContext* GetDeviceContext(const Place& place) const {
    PADDLE_THROW(phi::errors::Unimplemented(
        "ProcessGroup%s does not support get device_context.",
        GetBackendName()));
  }

  virtual phi::DeviceContext* GetDeviceContext(const Place& place,
                                               bool use_calc_stream) const {
    PADDLE_THROW(phi::errors::Unimplemented(
        "ProcessGroup%s does not support get device_context.",
        GetBackendName()));
  }

  // without stream APIs
  virtual std::shared_ptr<ProcessGroup::Task> AllGather(
      phi::DenseTensor* out_tensor,
      const phi::DenseTensor& in_tensor,
      bool sync_op) {
    PADDLE_THROW(phi::errors::Unimplemented(
        "ProcessGroup%s does not support all_gather with sync_op flag.",
        GetBackendName()));
  }

  virtual std::shared_ptr<ProcessGroup::Task> AllGather(
      phi::DenseTensor* out_tensor,
      const phi::DenseTensor& in_tensor,
      int64_t offset,
      int64_t numel,
      bool sync_op) {
    PADDLE_THROW(phi::errors::Unimplemented(
        "ProcessGroup%s does not support all_gather with sync_op flag.",
        GetBackendName()));
  }

  virtual std::shared_ptr<ProcessGroup::Task> AllReduce(
      phi::DenseTensor* out_tensor,
      const phi::DenseTensor& in_tensor,
      const AllreduceOptions& opts,
      bool sync_op) {
    PADDLE_THROW(phi::errors::Unimplemented(
        "ProcessGroup%s does not support all_reduce with sync_op flag.",
        GetBackendName()));
  }

  virtual std::shared_ptr<ProcessGroup::Task> AllToAll(
      phi::DenseTensor* out_tensor,
      const phi::DenseTensor& in_tensor,
      const std::vector<int64_t>& out_size_each_rank,
      const std::vector<int64_t>& in_size_each_rank,
      bool sync_op) {
    PADDLE_THROW(phi::errors::Unimplemented(
        "ProcessGroup%s does not support all_to_all with sync_op flag.",
        GetBackendName()));
  }

  virtual std::shared_ptr<ProcessGroup::Task> Barrier(
      const BarrierOptions& = BarrierOptions()) {
    PADDLE_THROW(phi::errors::Unimplemented(
        "ProcessGroup%s does not support barrier.", GetBackendName()));
  }

  virtual std::shared_ptr<ProcessGroup::Task> Broadcast(
      phi::DenseTensor* out_tensor,
      const phi::DenseTensor& in_tensor,
      const BroadcastOptions& opts,
      bool sync_op) {
    PADDLE_THROW(phi::errors::Unimplemented(
        "ProcessGroup%s does not support broadcast with sync_op flag",
        GetBackendName()));
  }

  virtual std::shared_ptr<ProcessGroup::Task> Reduce(
      phi::DenseTensor* out_tensor,
      const phi::DenseTensor& in_tensor,
      const ReduceOptions& opts,
      bool sync_op) {
    PADDLE_THROW(phi::errors::Unimplemented(
        "ProcessGroup%s does not support reduce with sync_op flag.",
        GetBackendName()));
  }

  virtual std::shared_ptr<ProcessGroup::Task> ReduceScatter(
      phi::DenseTensor* out_tensor,
      const phi::DenseTensor& in_tensor,
      const ReduceScatterOptions& opts,
      bool sync_op) {
    PADDLE_THROW(phi::errors::Unimplemented(
        "ProcessGroup%s does not support reduce_scatter with sync_op flag.",
        GetBackendName()));
  }

  virtual std::shared_ptr<ProcessGroup::Task> Scatter(
      phi::DenseTensor* out_tensor,
      const phi::DenseTensor& in_tensor,
      const ScatterOptions& opts,
      bool sync_op) {
    PADDLE_THROW(phi::errors::Unimplemented(
        "ProcessGroup%s does not support scatter with sync_op flag.",
        GetBackendName()));
  }

  virtual std::shared_ptr<ProcessGroup::Task> Recv(phi::DenseTensor* tensor,
                                                   int src_rank,
                                                   bool sync_op) {
    PADDLE_THROW(phi::errors::Unimplemented(
        "ProcessGroup%s does not support recv with sync_op flag.",
        GetBackendName()));
  }

  virtual std::shared_ptr<ProcessGroup::Task> Recv(phi::DenseTensor* tensor,
                                                   int src_rank,
                                                   int64_t offset,
                                                   int64_t numel,
                                                   bool sync_op) {
    PADDLE_THROW(phi::errors::Unimplemented(
        "ProcessGroup%s does not support recv with sync_op flag.",
        GetBackendName()));
  }

  virtual std::shared_ptr<ProcessGroup::Task> Send(
      const phi::DenseTensor& tensor, int dst_rank, bool sync_op) {
    PADDLE_THROW(phi::errors::Unimplemented(
        "ProcessGroup%s does not support send with sync_op flag.",
        GetBackendName()));
  }

  virtual std::shared_ptr<ProcessGroup::Task> Send(
      const phi::DenseTensor& tensor,
      int dst_rank,
      int64_t offset,
      int64_t numel,
      bool sync_op) {
    PADDLE_THROW(phi::errors::Unimplemented(
        "ProcessGroup%s does not support send with sync_op flag.",
        GetBackendName()));
  }

  // stream APIs
  virtual std::shared_ptr<ProcessGroup::Task> AllGather(
      phi::DenseTensor* out_tensor,
      const phi::DenseTensor& in_tensor,
      bool sync_op,
      bool use_calc_stream) {
    PADDLE_THROW(
        phi::errors::Unimplemented("ProcessGroup%s does not support all_gather "
                                   "with sync_op and use_calc_stream flag.",
                                   GetBackendName()));
  }

  virtual std::shared_ptr<ProcessGroup::Task> AllGather(
      phi::DenseTensor* out_tensor,
      const phi::DenseTensor& in_tensor,
      int64_t offset,
      int64_t numel,
      bool sync_op,
      bool use_calc_stream) {
    PADDLE_THROW(
        phi::errors::Unimplemented("ProcessGroup%s does not support all_gather "
                                   "with sync_op and use_calc_stream flag.",
                                   GetBackendName()));
  }

  virtual std::shared_ptr<ProcessGroup::Task> AllReduce(
      phi::DenseTensor* out_tensor,
      const phi::DenseTensor& in_tensor,
      const AllreduceOptions& opts,
      bool sync_op,
      bool use_calc_stream) {
    PADDLE_THROW(
        phi::errors::Unimplemented("ProcessGroup%s does not support all_reduce "
                                   "with sync_op and use_calc_stream flag.",
                                   GetBackendName()));
  }

  virtual std::shared_ptr<ProcessGroup::Task> AllToAll(
      phi::DenseTensor* out_tensor,
      const phi::DenseTensor& in_tensor,
      const std::vector<int64_t>& out_size_each_rank,
      const std::vector<int64_t>& in_size_each_rank,
      bool sync_op,
      bool use_calc_stream) {
    PADDLE_THROW(
        phi::errors::Unimplemented("ProcessGroup%s does not support all_to_all "
                                   "with sync_op and use_calc_stream flag.",
                                   GetBackendName()));
  }

  virtual std::shared_ptr<ProcessGroup::Task> Broadcast(
      phi::DenseTensor* out_tensor,
      const phi::DenseTensor& in_tensor,
      const BroadcastOptions& opts,
      bool sync_op,
      bool use_calc_stream) {
    PADDLE_THROW(
        phi::errors::Unimplemented("ProcessGroup%s does not support broadcast "
                                   "with sync_op and use_calc_stream flag.",
                                   GetBackendName()));
  }

  virtual std::shared_ptr<ProcessGroup::Task> Reduce(
      phi::DenseTensor* out_tensor,
      const phi::DenseTensor& in_tensor,
      const ReduceOptions& opts,
      bool sync_op,
      bool use_calc_stream) {
    PADDLE_THROW(
        phi::errors::Unimplemented("ProcessGroup%s does not support reduce "
                                   "with sync_op and use_calc_stream flag.",
                                   GetBackendName()));
  }

  virtual std::shared_ptr<ProcessGroup::Task> ReduceScatter(
      phi::DenseTensor* out_tensor,
      const phi::DenseTensor& in_tensor,
      const ReduceScatterOptions& opts,
      bool sync_op,
      bool use_calc_stream) {
    PADDLE_THROW(phi::errors::Unimplemented(
        "ProcessGroup%s does not support reduce_scatter "
        "with sync_op and use_calc_stream flag.",
        GetBackendName()));
  }

  virtual std::shared_ptr<ProcessGroup::Task> Scatter(
      phi::DenseTensor* out_tensor,
      const phi::DenseTensor& in_tensor,
      const ScatterOptions& opts,
      bool sync_op,
      bool use_calc_stream) {
    PADDLE_THROW(
        phi::errors::Unimplemented("ProcessGroup%s does not support scatter "
                                   "with sync_op and use_calc_stream flag.",
                                   GetBackendName()));
  }

  virtual std::shared_ptr<ProcessGroup::Task> Gather(
      phi::DenseTensor* out_tensor,
      const phi::DenseTensor& in_tensor,
      const GatherOptions& opts,
      bool sync_op,
      bool use_calc_stream) {
    PADDLE_THROW(
        phi::errors::Unimplemented("ProcessGroup%s does not support gather "
                                   "with sync_op and use_calc_stream flag.",
                                   GetBackendName()));
  }

  virtual std::shared_ptr<ProcessGroup::Task> Gather(
      std::vector<phi::DenseTensor>* gather_tensors_ptr,
      const phi::DenseTensor& in_tensor,
      const GatherOptions& opts,
      bool sync_op,
      bool use_calc_stream) {
    PADDLE_THROW(
        phi::errors::Unimplemented("ProcessGroup%s does not support gather "
                                   "with sync_op and use_calc_stream flag.",
                                   GetBackendName()));
  }

  virtual std::shared_ptr<ProcessGroup::Task> Recv(phi::DenseTensor* tensor,
                                                   int src_rank,
                                                   bool sync_op,
                                                   bool use_calc_stream) {
    PADDLE_THROW(
        phi::errors::Unimplemented("ProcessGroup%s does not support recv with "
                                   "sync_op and use_calc_stream flag.",
                                   GetBackendName()));
  }

  virtual std::shared_ptr<ProcessGroup::Task> Recv(phi::DenseTensor* tensor,
                                                   int src_rank,
                                                   int64_t offset,
                                                   int64_t numel,
                                                   bool sync_op,
                                                   bool use_calc_stream) {
    PADDLE_THROW(
        phi::errors::Unimplemented("ProcessGroup%s does not support recv "
                                   "with sync_op and use_calc_stream flag.",
                                   GetBackendName()));
  }

  virtual std::shared_ptr<ProcessGroup::Task> Send(
      const phi::DenseTensor& tensor,
      int dst_rank,
      bool sync_op,
      bool use_calc_stream) {
    PADDLE_THROW(
        phi::errors::Unimplemented("ProcessGroup%s does not support send "
                                   "with sync_op and use_calc_stream flag.",
                                   GetBackendName()));
  }

  virtual std::shared_ptr<ProcessGroup::Task> Send(
      const phi::DenseTensor& tensor,
      int dst_rank,
      int64_t offset,
      int64_t numel,
      bool sync_op,
      bool use_calc_stream) {
    PADDLE_THROW(
        phi::errors::Unimplemented("ProcessGroup%s does not support send "
                                   "with sync_op and use_calc_stream flag.",
                                   GetBackendName()));
  }

  // legacy APIs
  // TODO(liyurui): This API will be moved later
  virtual std::shared_ptr<ProcessGroup::Task> AllReduce(
      std::vector<phi::DenseTensor>& inputs,   // NOLINT
      std::vector<phi::DenseTensor>& outputs,  // NOLINT
      const AllreduceOptions& options = AllreduceOptions()) {
    return AllReduce(outputs.data(), inputs.front(), options, false);
  }

  virtual std::shared_ptr<ProcessGroup::Task> AllReduce(
      std::vector<phi::DenseTensor>& inputs,   // NOLINT
      std::vector<phi::DenseTensor>& outputs,  // NOLINT
      const AllreduceOptions& options,
      bool sync_op) {
    return AllReduce(outputs.data(), inputs.front(), options, sync_op);
  }

  // TODO(sunyilun): methods below will be removed later
  virtual std::shared_ptr<ProcessGroup::Task> Broadcast(
      std::vector<phi::DenseTensor>& inputs,   // NOLINT
      std::vector<phi::DenseTensor>& outputs,  // NOLINT
      const BroadcastOptions& options = BroadcastOptions()) {
    return Broadcast(outputs.data(), inputs.front(), options, false);
  }

  virtual std::shared_ptr<ProcessGroup::Task> Broadcast(
      std::vector<phi::DenseTensor>& inputs,   // NOLINT
      std::vector<phi::DenseTensor>& outputs,  // NOLINT
      const BroadcastOptions& options,
      bool sync_op) {
    return Broadcast(outputs.data(), inputs.front(), options, sync_op);
  }

  virtual std::shared_ptr<ProcessGroup::Task> Send(
      std::vector<phi::DenseTensor>& tensors, int dst_rank) {  // NOLINT
    return Send(tensors.front(), dst_rank, false);
  }

  virtual std::shared_ptr<ProcessGroup::Task> Recv(
      std::vector<phi::DenseTensor>& tensors, int src_rank) {  // NOLINT
    return Recv(&tensors.front(), src_rank, false);
  }

  virtual std::shared_ptr<ProcessGroup::Task> AllGather(
      std::vector<phi::DenseTensor>& in_tensors,     // NOLINT
      std::vector<phi::DenseTensor>& out_tensors) {  // NOLINT
    return AllGather(out_tensors.data(), in_tensors.front(), false);
  }

  virtual std::shared_ptr<ProcessGroup::Task> AllGather(
      std::vector<phi::DenseTensor>& in_tensors,   // NOLINT
      std::vector<phi::DenseTensor>& out_tensors,  // NOLINT
      bool sync_op) {
    return AllGather(out_tensors.data(), in_tensors.front(), sync_op);
  }

  virtual std::shared_ptr<ProcessGroup::Task> AllToAll(
      std::vector<phi::DenseTensor>&,    // NOLINT
      std::vector<phi::DenseTensor>&) {  // NOLINT
    PADDLE_THROW(phi::errors::InvalidArgument(
        "ProcessGroup%s does not support AllToAll", GetBackendName()));
  }

  virtual std::shared_ptr<ProcessGroup::Task> Reduce(
      std::vector<phi::DenseTensor>& ins,   // NOLINT
      std::vector<phi::DenseTensor>& outs,  // NOLINT
      const ReduceOptions& opts) {
    return Reduce(outs.data(), ins.front(), opts, false);
  }

  virtual std::shared_ptr<ProcessGroup::Task> Scatter(
      std::vector<phi::DenseTensor>& ins,   // NOLINT
      std::vector<phi::DenseTensor>& outs,  // NOLINT
      const ScatterOptions& opts) {
    return Scatter(outs.data(), ins.front(), opts, false);
  }

 protected:
  int rank_;
  int size_;
  int gid_;
};

class ProcessGroupIdMap
    : public std::unordered_map<int, std::shared_ptr<ProcessGroup>> {
 public:
  static ProcessGroupIdMap& GetInstance();
  static void DestroyProcessGroup();
};

// TODO(dev): The following method will be removed soon.
class ProcessGroupMapFromGid {
 public:
  bool has(int gid) { return map_.find(gid) != map_.end(); }

  void insert(int gid, ProcessGroup* pg) { map_[gid] = pg; }

  ProcessGroup* get(int gid) { return map_.find(gid)->second; }

  static std::shared_ptr<ProcessGroupMapFromGid> getInstance() {
    static auto s_instance = std::make_shared<ProcessGroupMapFromGid>();
    return s_instance;
  }

  ProcessGroupMapFromGid() = default;
  ~ProcessGroupMapFromGid() = default;

 private:
  std::unordered_map<int, ProcessGroup*> map_;
};

}  //  namespace distributed
}  //  namespace paddle
