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
#include <vector>

#include "paddle/fluid/distributed/collective/Types.h"
#include "paddle/fluid/eager/api/utils/tensor_utils.h"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/framework/variable.h"
#include "paddle/fluid/platform/enforce.h"

constexpr auto kWaitTimeout = std::chrono::milliseconds(0);

namespace paddle {
namespace distributed {

constexpr int IGNORE_ID = -1;
using Tensor = paddle::experimental::Tensor;

enum class CommType : std::uint8_t {
  BROADCAST = 0,
  ALLREDUCE = 1,
  ALLREDUCE_SPARSE = 2,  // TODO(shenliang03): to support sparse in allreduce
  REDUCE = 3,
  ALLGATHER = 4,
  GATHER = 5,
  SCATTER = 6,
  REDUCE_SCATTER = 7,
  ALLTOALL = 8,
  SEND = 9,
  RECV = 10,
  BARRIER = 11,
  ALLTOALL_SINGLE = 12,
  UNKNOWN = 100,
};

class ProcessGroup {
 public:
  class Task {
   public:
    Task(int rank, CommType comm_type, bool sync_op);

    virtual ~Task();
    virtual bool IsCompleted();
    virtual bool Wait(std::chrono::milliseconds timeout = kWaitTimeout);
    virtual void Synchronize();
    virtual void UpdateWaitChain(const phi::DeviceContext& ctx);
    bool IsSync() const { return sync_op_; }

    // TODO(sunyilun): methods below will be removed later
    Task(int rank,
         const std::vector<phi::DenseTensor>& inputs,
         CommType comm_type);
    Task(int rank,
         const std::vector<phi::DenseTensor>& inputs,
         CommType comm_type,
         bool sync_op);

   protected:
    const int rank_;
    CommType comm_type_{CommType::UNKNOWN};
    std::mutex mutex_;
    bool is_completed_{false};

   private:
    bool sync_op_{true};
  };

 public:
  explicit ProcessGroup(int rank, int size, int gid);
  virtual ~ProcessGroup() = default;
  // TODO(dev): This constructor will be removed later.
  explicit ProcessGroup(int rank,
                        int size,
                        const platform::Place& place,
                        int gid);

  int GetRank() const { return rank_; }

  int GetSize() const { return size_; }

  virtual std::string GetBackendName() const = 0;

  virtual const phi::DeviceContext& GetDeviceContext(const Place& place) const {
    PADDLE_THROW(platform::errors::Unimplemented(
        "ProcessGroup%s does not support get device_context.",
        GetBackendName()));
  }

  virtual std::shared_ptr<ProcessGroup::Task> AllGather(
      phi::DenseTensor* out_tensor,
      const phi::DenseTensor& in_tensor,
      int64_t offset,
      int64_t numel,
      bool sync_op) {
    PADDLE_THROW(platform::errors::Unimplemented(
        "ProcessGroup%s does not support all_gather with sync_op flag.",
        GetBackendName()));
  }

  virtual std::shared_ptr<ProcessGroup::Task> AllReduce(
      phi::DenseTensor* out_tensor,
      const phi::DenseTensor& in_tensor,
      const AllreduceOptions& opts,
      bool sync_op) {
    PADDLE_THROW(platform::errors::Unimplemented(
        "ProcessGroup%s does not support all_reduce with sync_op flag.",
        GetBackendName()));
  }

  virtual std::shared_ptr<ProcessGroup::Task> Barrier(
      const BarrierOptions& = BarrierOptions()) {
    PADDLE_THROW(platform::errors::Unimplemented(
        "ProcessGroup%s does not support barrier.", GetBackendName()));
  }

  virtual std::shared_ptr<ProcessGroup::Task> Broadcast(
      phi::DenseTensor* out_tensor,
      const phi::DenseTensor& in_tensor,
      const BroadcastOptions& opts,
      bool sync_op) {
    PADDLE_THROW(platform::errors::Unimplemented(
        "ProcessGroup%s does not support broadcast with sync_op flag",
        GetBackendName()));
  }

  virtual std::shared_ptr<ProcessGroup::Task> Recv(phi::DenseTensor* tensor,
                                                   int src_rank,
                                                   int64_t offset,
                                                   int64_t numel,
                                                   bool sync_op) {
    PADDLE_THROW(platform::errors::Unimplemented(
        "ProcessGroup%s does not support recv with sync_op flag.",
        GetBackendName()));
  }

  virtual std::shared_ptr<ProcessGroup::Task> Send(phi::DenseTensor*,
                                                   int dst_rank,
                                                   int64_t offset,
                                                   int64_t numel,
                                                   bool sync_op) {
    PADDLE_THROW(platform::errors::Unimplemented(
        "ProcessGroup%s does not support send with sync_op flag.",
        GetBackendName()));
  }

  // TODO(liyurui): This API will be moved later
  virtual std::shared_ptr<ProcessGroup::Task> AllReduce(
      std::vector<phi::DenseTensor>& /* input tensors */,   // NOLINT
      std::vector<phi::DenseTensor>& /* output tensors */,  // NOLINT
      const AllreduceOptions& = AllreduceOptions()) {
    PADDLE_THROW(platform::errors::InvalidArgument(
        "ProcessGroup%s does not support allreduce", GetBackendName()));
  }

  virtual std::shared_ptr<ProcessGroup::Task> AllReduce(
      std::vector<phi::DenseTensor>& /* input tensors */,   // NOLINT
      std::vector<phi::DenseTensor>& /* output tensors */,  // NOLINT
      const AllreduceOptions&,
      bool) {
    PADDLE_THROW(platform::errors::InvalidArgument(
        "ProcessGroup%s does not support allreduce with sync_op flag",
        GetBackendName()));
  }

  // TODO(sunyilun): methods below will be removed later
  virtual std::shared_ptr<ProcessGroup::Task> Broadcast(
      std::vector<phi::DenseTensor>& /* input tensors */,   // NOLINT
      std::vector<phi::DenseTensor>& /* output tensors */,  // NOLINT
      const BroadcastOptions& = BroadcastOptions()) {
    PADDLE_THROW(platform::errors::InvalidArgument(
        "ProcessGroup%s does not support broadcast", GetBackendName()));
  }

  virtual std::shared_ptr<ProcessGroup::Task> Broadcast(
      std::vector<phi::DenseTensor>& /* input tensors */,   // NOLINT
      std::vector<phi::DenseTensor>& /* output tensors */,  // NOLINT
      const BroadcastOptions&,
      bool) {
    PADDLE_THROW(platform::errors::InvalidArgument(
        "ProcessGroup%s does not support broadcast with sync_op flag",
        GetBackendName()));
  }

  virtual std::shared_ptr<ProcessGroup::Task> Send(
      std::vector<phi::DenseTensor>&, int) {  // NOLINT
    PADDLE_THROW(platform::errors::InvalidArgument(
        "ProcessGroup%s does not support send", GetBackendName()));
  }

  virtual std::shared_ptr<ProcessGroup::Task> Send(
      std::vector<phi::DenseTensor>&, int, bool) {  // NOLINT
    PADDLE_THROW(platform::errors::InvalidArgument(
        "ProcessGroup%s does not support send with sync_op flag",
        GetBackendName()));
  }

  virtual std::shared_ptr<ProcessGroup::Task> Recv(
      std::vector<phi::DenseTensor>&, int) {  // NOLINT
    PADDLE_THROW(platform::errors::InvalidArgument(
        "ProcessGroup%s does not support recv", GetBackendName()));
  }

  virtual std::shared_ptr<ProcessGroup::Task> Recv(
      std::vector<phi::DenseTensor>&, int, bool) {  // NOLINT
    PADDLE_THROW(platform::errors::InvalidArgument(
        "ProcessGroup%s does not support recv with sync_op flag",
        GetBackendName()));
  }

  virtual std::shared_ptr<ProcessGroup::Task> AllGather(
      std::vector<phi::DenseTensor>&,    // NOLINT
      std::vector<phi::DenseTensor>&) {  // NOLINT
    PADDLE_THROW(platform::errors::InvalidArgument(
        "ProcessGroup%s does not support all_gather", GetBackendName()));
  }

  virtual std::shared_ptr<ProcessGroup::Task> AllGather(
      std::vector<phi::DenseTensor>&,  // NOLINT
      std::vector<phi::DenseTensor>&,  // NOLINT
      bool) {
    PADDLE_THROW(platform::errors::InvalidArgument(
        "ProcessGroup%s does not support all_gather with sync_op flag",
        GetBackendName()));
  }

  virtual std::shared_ptr<ProcessGroup::Task> AllToAll(
      std::vector<phi::DenseTensor>&,    // NOLINT
      std::vector<phi::DenseTensor>&) {  // NOLINT
    PADDLE_THROW(platform::errors::InvalidArgument(
        "ProcessGroup%s does not support AllToAll", GetBackendName()));
  }

  virtual std::shared_ptr<ProcessGroup::Task> AllToAll(
      std::vector<phi::DenseTensor>&,  // NOLINT
      std::vector<phi::DenseTensor>&,  // NOLINT
      bool) {
    PADDLE_THROW(platform::errors::InvalidArgument(
        "ProcessGroup%s does not support alltoall", GetBackendName()));
  }

  virtual std::shared_ptr<ProcessGroup::Task> AllToAll_Single(
      std::vector<phi::DenseTensor>&,  // NOLINT
      std::vector<phi::DenseTensor>&,  // NOLINT
      std::vector<int64_t>&,
      std::vector<int64_t>&) {
    PADDLE_THROW(platform::errors::InvalidArgument(
        "ProcessGroup%s does not support AllToAll_Single", GetBackendName()));
  }

  virtual std::shared_ptr<ProcessGroup::Task> AllToAllSingle(
      std::vector<phi::DenseTensor>&,  // NOLINT
      std::vector<phi::DenseTensor>&,  // NOLINT
      std::vector<int64_t>&,
      std::vector<int64_t>&,
      bool) {
    PADDLE_THROW(platform::errors::InvalidArgument(
        "ProcessGroup%s does not support alltoall_single", GetBackendName()));
  }

  virtual std::shared_ptr<ProcessGroup::Task> Reduce(
      std::vector<phi::DenseTensor>&,  // NOLINT
      std::vector<phi::DenseTensor>&,  // NOLINT
      const ReduceOptions& opts) {
    PADDLE_THROW(platform::errors::InvalidArgument(
        "ProcessGroup%s does not support reduce", GetBackendName()));
  }

  virtual std::shared_ptr<ProcessGroup::Task> Reduce(
      std::vector<phi::DenseTensor>& /* input tensors */,   // NOLINT
      std::vector<phi::DenseTensor>& /* output tensors */,  // NOLINT
      const ReduceOptions&,
      bool) {
    PADDLE_THROW(platform::errors::InvalidArgument(
        "ProcessGroup%s does not support reduce with sync_op flag",
        GetBackendName()));
  }

  virtual std::shared_ptr<ProcessGroup::Task> Scatter(
      std::vector<phi::DenseTensor>&,  // NOLINT
      std::vector<phi::DenseTensor>&,  // NOLINT
      const ScatterOptions&) {
    PADDLE_THROW(platform::errors::InvalidArgument(
        "ProcessGroup%s does not support scatter", GetBackendName()));
  }

  virtual std::shared_ptr<ProcessGroup::Task> Scatter(
      std::vector<phi::DenseTensor>&,  // NOLINT
      std::vector<phi::DenseTensor>&,  // NOLINT
      const ScatterOptions&,
      bool) {
    PADDLE_THROW(platform::errors::InvalidArgument(
        "ProcessGroup%s does not support scatter with sync_op flag",
        GetBackendName()));
  }

  virtual std::shared_ptr<ProcessGroup::Task> ReduceScatter(
      std::vector<phi::DenseTensor>&,  // NOLINT
      std::vector<phi::DenseTensor>&,  // NOLINT
      const ReduceScatterOptions&,
      bool) {
    PADDLE_THROW(platform::errors::InvalidArgument(
        "ProcessGroup%s does not support reduce_scatter with sync_op flag",
        GetBackendName()));
  }

 protected:
  const int rank_;
  const int size_;
  const platform::Place place_;
  const int gid_;
};

class ProcessGroupMapFromGid {
 public:
  bool has(int gid) {
    auto it = map_.find(gid);
    return it != map_.end();
  }

  void insert(int gid, ProcessGroup* pg) {
    // TODO(sandyhouse): address ut and uncomment the following codes
    // PADDLE_ENFORCE_EQ(has(gid), false,
    //                   platform::errors::PreconditionNotMet(
    //                       "The process group with id %d doesnot exist.",
    //                       gid));
    map_[gid] = pg;
  }

  ProcessGroup* get(int gid) {
    // TODO(sandyhouse): address ut and uncomment the following codes
    // PADDLE_ENFORCE_EQ(has(gid), true,
    //                   platform::errors::PreconditionNotMet(
    //                       "The process group with id %d doesnot exist.",
    //                       gid));
    return map_.find(gid)->second;
  }

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
