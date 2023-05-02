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

#include "paddle/fluid/distributed/collective/types.h"
#include "paddle/fluid/eager/api/utils/tensor_utils.h"  // NOTE: this header is required somewhere
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/device_context.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/errors.h"

constexpr auto kWaitTimeout = std::chrono::milliseconds(0);

namespace paddle {
namespace distributed {

constexpr int kIgnoreId = -1;

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
  UNKNOWN = 100,
};

class ProcessGroup {
 public:
  class Task {
   public:
    Task(int rank, CommType comm_type, bool sync_op)
        : rank_(rank), comm_type_(comm_type), sync_op_(sync_op) {}
    virtual ~Task() = default;

    virtual bool IsCompleted();
    virtual bool Wait(std::chrono::milliseconds timeout UNUSED = kWaitTimeout) {
      return false;
    }
    virtual void Synchronize() {}
    virtual void UpdateWaitChain(const phi::DeviceContext& ctx UNUSED) {}
    bool IsSync() const { return sync_op_; }

    // TODO(sunyilun): methods below will be removed later
    Task(int rank,
         const std::vector<phi::DenseTensor>& inputs UNUSED,
         CommType comm_type)
        : rank_(rank), comm_type_(comm_type) {}
    Task(int rank,
         const std::vector<phi::DenseTensor>& inputs UNUSED,
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

  virtual phi::DeviceContext* GetDeviceContext(
      const Place& place UNUSED) const {
    PADDLE_THROW(phi::errors::Unimplemented(
        "ProcessGroup%s does not support get device_context.",
        GetBackendName()));
  }

  virtual phi::DeviceContext* GetDeviceContext(
      const Place& place UNUSED, bool use_calc_stream UNUSED) const {
    PADDLE_THROW(phi::errors::Unimplemented(
        "ProcessGroup%s does not support get device_context.",
        GetBackendName()));
  }

  // without stream APIs
  virtual std::shared_ptr<ProcessGroup::Task> AllGather(
      phi::DenseTensor* out_tensor UNUSED,
      const phi::DenseTensor& in_tensor UNUSED,
      bool sync_op UNUSED) {
    PADDLE_THROW(phi::errors::Unimplemented(
        "ProcessGroup%s does not support all_gather with sync_op flag.",
        GetBackendName()));
  }

  virtual std::shared_ptr<ProcessGroup::Task> AllGather(
      phi::DenseTensor* out_tensor UNUSED,
      const phi::DenseTensor& in_tensor UNUSED,
      int64_t offset UNUSED,
      int64_t numel UNUSED,
      bool sync_op UNUSED) {
    PADDLE_THROW(phi::errors::Unimplemented(
        "ProcessGroup%s does not support all_gather with sync_op flag.",
        GetBackendName()));
  }

  virtual std::shared_ptr<ProcessGroup::Task> AllReduce(
      phi::DenseTensor* out_tensor UNUSED,
      const phi::DenseTensor& in_tensor UNUSED,
      const AllreduceOptions& opts UNUSED,
      bool sync_op UNUSED) {
    PADDLE_THROW(phi::errors::Unimplemented(
        "ProcessGroup%s does not support all_reduce with sync_op flag.",
        GetBackendName()));
  }

  virtual std::shared_ptr<ProcessGroup::Task> AllToAll(
      phi::DenseTensor* out_tensor UNUSED,
      const phi::DenseTensor& in_tensor UNUSED,
      const std::vector<int64_t>& out_size_each_rank UNUSED,
      const std::vector<int64_t>& in_size_each_rank UNUSED,
      bool sync_op UNUSED) {
    PADDLE_THROW(phi::errors::Unimplemented(
        "ProcessGroup%s does not support all_to_all with sync_op flag.",
        GetBackendName()));
  }

  virtual std::shared_ptr<ProcessGroup::Task> Barrier(
      const BarrierOptions& UNUSED = BarrierOptions()) {
    PADDLE_THROW(phi::errors::Unimplemented(
        "ProcessGroup%s does not support barrier.", GetBackendName()));
  }

  virtual std::shared_ptr<ProcessGroup::Task> Broadcast(
      phi::DenseTensor* out_tensor UNUSED,
      const phi::DenseTensor& in_tensor UNUSED,
      const BroadcastOptions& opts UNUSED,
      bool sync_op UNUSED) {
    PADDLE_THROW(phi::errors::Unimplemented(
        "ProcessGroup%s does not support broadcast with sync_op flag",
        GetBackendName()));
  }

  virtual std::shared_ptr<ProcessGroup::Task> Reduce(
      phi::DenseTensor* out_tensor UNUSED,
      const phi::DenseTensor& in_tensor UNUSED,
      const ReduceOptions& opts UNUSED,
      bool sync_op UNUSED) {
    PADDLE_THROW(phi::errors::Unimplemented(
        "ProcessGroup%s does not support reduce with sync_op flag.",
        GetBackendName()));
  }

  virtual std::shared_ptr<ProcessGroup::Task> ReduceScatter(
      phi::DenseTensor* out_tensor UNUSED,
      const phi::DenseTensor& in_tensor UNUSED,
      const ReduceScatterOptions& opts UNUSED,
      bool sync_op UNUSED) {
    PADDLE_THROW(phi::errors::Unimplemented(
        "ProcessGroup%s does not support reduce_scatter with sync_op flag.",
        GetBackendName()));
  }

  virtual std::shared_ptr<ProcessGroup::Task> Scatter(
      phi::DenseTensor* out_tensor UNUSED,
      const phi::DenseTensor& in_tensor UNUSED,
      const ScatterOptions& opts UNUSED,
      bool sync_op UNUSED) {
    PADDLE_THROW(phi::errors::Unimplemented(
        "ProcessGroup%s does not support scatter with sync_op flag.",
        GetBackendName()));
  }

  virtual std::shared_ptr<ProcessGroup::Task> Recv(phi::DenseTensor* tensor
                                                       UNUSED,
                                                   int src_rank UNUSED,
                                                   bool sync_op UNUSED) {
    PADDLE_THROW(phi::errors::Unimplemented(
        "ProcessGroup%s does not support recv with sync_op flag.",
        GetBackendName()));
  }

  virtual std::shared_ptr<ProcessGroup::Task> Recv(phi::DenseTensor* tensor
                                                       UNUSED,
                                                   int src_rank UNUSED,
                                                   int64_t offset UNUSED,
                                                   int64_t numel UNUSED,
                                                   bool sync_op UNUSED) {
    PADDLE_THROW(phi::errors::Unimplemented(
        "ProcessGroup%s does not support recv with sync_op flag.",
        GetBackendName()));
  }

  virtual std::shared_ptr<ProcessGroup::Task> Send(
      const phi::DenseTensor& tensor UNUSED,
      int dst_rank UNUSED,
      bool sync_op UNUSED) {
    PADDLE_THROW(phi::errors::Unimplemented(
        "ProcessGroup%s does not support send with sync_op flag.",
        GetBackendName()));
  }

  virtual std::shared_ptr<ProcessGroup::Task> Send(
      const phi::DenseTensor& tensor UNUSED,
      int dst_rank UNUSED,
      int64_t offset UNUSED,
      int64_t numel UNUSED,
      bool sync_op UNUSED) {
    PADDLE_THROW(phi::errors::Unimplemented(
        "ProcessGroup%s does not support send with sync_op flag.",
        GetBackendName()));
  }

  // stream APIs
  virtual std::shared_ptr<ProcessGroup::Task> AllGather(
      phi::DenseTensor* out_tensor UNUSED,
      const phi::DenseTensor& in_tensor UNUSED,
      bool sync_op UNUSED,
      bool use_calc_stream UNUSED) {
    PADDLE_THROW(
        phi::errors::Unimplemented("ProcessGroup%s does not support all_gather "
                                   "with sync_op and use_calc_stream flag.",
                                   GetBackendName()));
  }

  virtual std::shared_ptr<ProcessGroup::Task> AllGather(
      phi::DenseTensor* out_tensor UNUSED,
      const phi::DenseTensor& in_tensor UNUSED,
      int64_t offset UNUSED,
      int64_t numel UNUSED,
      bool sync_op UNUSED,
      bool use_calc_stream UNUSED) {
    PADDLE_THROW(
        phi::errors::Unimplemented("ProcessGroup%s does not support all_gather "
                                   "with sync_op and use_calc_stream flag.",
                                   GetBackendName()));
  }

  virtual std::shared_ptr<ProcessGroup::Task> AllReduce(
      phi::DenseTensor* out_tensor UNUSED,
      const phi::DenseTensor& in_tensor UNUSED,
      const AllreduceOptions& opts UNUSED,
      bool sync_op UNUSED,
      bool use_calc_stream UNUSED) {
    PADDLE_THROW(
        phi::errors::Unimplemented("ProcessGroup%s does not support all_reduce "
                                   "with sync_op and use_calc_stream flag.",
                                   GetBackendName()));
  }

  virtual std::shared_ptr<ProcessGroup::Task> AllToAll(
      phi::DenseTensor* out_tensor UNUSED,
      const phi::DenseTensor& in_tensor UNUSED,
      const std::vector<int64_t>& out_size_each_rank UNUSED,
      const std::vector<int64_t>& in_size_each_rank UNUSED,
      bool sync_op UNUSED,
      bool use_calc_stream UNUSED) {
    PADDLE_THROW(
        phi::errors::Unimplemented("ProcessGroup%s does not support all_to_all "
                                   "with sync_op and use_calc_stream flag.",
                                   GetBackendName()));
  }

  virtual std::shared_ptr<ProcessGroup::Task> Broadcast(
      phi::DenseTensor* out_tensor UNUSED,
      const phi::DenseTensor& in_tensor UNUSED,
      const BroadcastOptions& opts UNUSED,
      bool sync_op UNUSED,
      bool use_calc_stream UNUSED) {
    PADDLE_THROW(
        phi::errors::Unimplemented("ProcessGroup%s does not support broadcast "
                                   "with sync_op and use_calc_stream flag.",
                                   GetBackendName()));
  }

  virtual std::shared_ptr<ProcessGroup::Task> Reduce(
      phi::DenseTensor* out_tensor UNUSED,
      const phi::DenseTensor& in_tensor UNUSED,
      const ReduceOptions& opts UNUSED,
      bool sync_op UNUSED,
      bool use_calc_stream UNUSED) {
    PADDLE_THROW(
        phi::errors::Unimplemented("ProcessGroup%s does not support reduce "
                                   "with sync_op and use_calc_stream flag.",
                                   GetBackendName()));
  }

  virtual std::shared_ptr<ProcessGroup::Task> ReduceScatter(
      phi::DenseTensor* out_tensor UNUSED,
      const phi::DenseTensor& in_tensor UNUSED,
      const ReduceScatterOptions& opts UNUSED,
      bool sync_op UNUSED,
      bool use_calc_stream UNUSED) {
    PADDLE_THROW(phi::errors::Unimplemented(
        "ProcessGroup%s does not support reduce_scatter "
        "with sync_op and use_calc_stream flag.",
        GetBackendName()));
  }

  virtual std::shared_ptr<ProcessGroup::Task> Scatter(
      phi::DenseTensor* out_tensor UNUSED,
      const phi::DenseTensor& in_tensor UNUSED,
      const ScatterOptions& opts UNUSED,
      bool sync_op UNUSED,
      bool use_calc_stream UNUSED) {
    PADDLE_THROW(
        phi::errors::Unimplemented("ProcessGroup%s does not support scatter "
                                   "with sync_op and use_calc_stream flag.",
                                   GetBackendName()));
  }

  virtual std::shared_ptr<ProcessGroup::Task> Gather(
      phi::DenseTensor* out_tensor UNUSED,
      const phi::DenseTensor& in_tensor UNUSED,
      const GatherOptions& opts UNUSED,
      bool sync_op UNUSED,
      bool use_calc_stream UNUSED) {
    PADDLE_THROW(
        phi::errors::Unimplemented("ProcessGroup%s does not support gather "
                                   "with sync_op and use_calc_stream flag.",
                                   GetBackendName()));
  }

  virtual std::shared_ptr<ProcessGroup::Task> Gather(
      std::vector<phi::DenseTensor>* gather_tensors_ptr UNUSED,
      const phi::DenseTensor& in_tensor UNUSED,
      const GatherOptions& opts UNUSED,
      bool sync_op UNUSED,
      bool use_calc_stream UNUSED) {
    PADDLE_THROW(
        phi::errors::Unimplemented("ProcessGroup%s does not support gather "
                                   "with sync_op and use_calc_stream flag.",
                                   GetBackendName()));
  }

  virtual std::shared_ptr<ProcessGroup::Task> Recv(
      phi::DenseTensor* tensor UNUSED,
      int src_rank UNUSED,
      bool sync_op UNUSED,
      bool use_calc_stream UNUSED) {
    PADDLE_THROW(
        phi::errors::Unimplemented("ProcessGroup%s does not support recv with "
                                   "sync_op and use_calc_stream flag.",
                                   GetBackendName()));
  }

  virtual std::shared_ptr<ProcessGroup::Task> Recv(
      phi::DenseTensor* tensor UNUSED,
      int src_rank UNUSED,
      int64_t offset UNUSED,
      int64_t numel UNUSED,
      bool sync_op UNUSED,
      bool use_calc_stream UNUSED) {
    PADDLE_THROW(
        phi::errors::Unimplemented("ProcessGroup%s does not support recv "
                                   "with sync_op and use_calc_stream flag.",
                                   GetBackendName()));
  }

  virtual std::shared_ptr<ProcessGroup::Task> Send(
      const phi::DenseTensor& tensor UNUSED,
      int dst_rank UNUSED,
      bool sync_op UNUSED,
      bool use_calc_stream UNUSED) {
    PADDLE_THROW(
        phi::errors::Unimplemented("ProcessGroup%s does not support send "
                                   "with sync_op and use_calc_stream flag.",
                                   GetBackendName()));
  }

  virtual std::shared_ptr<ProcessGroup::Task> Send(
      const phi::DenseTensor& tensor UNUSED,
      int dst_rank UNUSED,
      int64_t offset UNUSED,
      int64_t numel UNUSED,
      bool sync_op UNUSED,
      bool use_calc_stream UNUSED) {
    PADDLE_THROW(
        phi::errors::Unimplemented("ProcessGroup%s does not support send "
                                   "with sync_op and use_calc_stream flag.",
                                   GetBackendName()));
  }

  // legacy APIs
  // TODO(liyurui): This API will be moved later
  virtual std::shared_ptr<ProcessGroup::Task> AllReduce(
      std::vector<phi::DenseTensor>& /* input tensors */,   // NOLINT
      std::vector<phi::DenseTensor>& /* output tensors */,  // NOLINT
      const AllreduceOptions& UNUSED = AllreduceOptions()) {
    PADDLE_THROW(phi::errors::InvalidArgument(
        "ProcessGroup%s does not support allreduce", GetBackendName()));
  }

  virtual std::shared_ptr<ProcessGroup::Task> AllReduce(
      std::vector<phi::DenseTensor>& /* input tensors */,   // NOLINT
      std::vector<phi::DenseTensor>& /* output tensors */,  // NOLINT
      const AllreduceOptions& UNUSED,
      bool) {
    PADDLE_THROW(phi::errors::InvalidArgument(
        "ProcessGroup%s does not support allreduce with sync_op flag",
        GetBackendName()));
  }

  // TODO(sunyilun): methods below will be removed later
  virtual std::shared_ptr<ProcessGroup::Task> Broadcast(
      std::vector<phi::DenseTensor>& /* input tensors */,   // NOLINT
      std::vector<phi::DenseTensor>& /* output tensors */,  // NOLINT
      const BroadcastOptions& UNUSED = BroadcastOptions()) {
    PADDLE_THROW(phi::errors::InvalidArgument(
        "ProcessGroup%s does not support broadcast", GetBackendName()));
  }

  virtual std::shared_ptr<ProcessGroup::Task> Broadcast(
      std::vector<phi::DenseTensor>& /* input tensors */,   // NOLINT
      std::vector<phi::DenseTensor>& /* output tensors */,  // NOLINT
      const BroadcastOptions& UNUSED,
      bool) {
    PADDLE_THROW(phi::errors::InvalidArgument(
        "ProcessGroup%s does not support broadcast with sync_op flag",
        GetBackendName()));
  }

  virtual std::shared_ptr<ProcessGroup::Task> Send(
      std::vector<phi::DenseTensor>&, int) {  // NOLINT
    PADDLE_THROW(phi::errors::InvalidArgument(
        "ProcessGroup%s does not support send", GetBackendName()));
  }

  virtual std::shared_ptr<ProcessGroup::Task> Recv(
      std::vector<phi::DenseTensor>&, int) {  // NOLINT
    PADDLE_THROW(phi::errors::InvalidArgument(
        "ProcessGroup%s does not support recv", GetBackendName()));
  }

  virtual std::shared_ptr<ProcessGroup::Task> AllGather(
      std::vector<phi::DenseTensor>&,    // NOLINT
      std::vector<phi::DenseTensor>&) {  // NOLINT
    PADDLE_THROW(phi::errors::InvalidArgument(
        "ProcessGroup%s does not support all_gather", GetBackendName()));
  }

  virtual std::shared_ptr<ProcessGroup::Task> AllGather(
      std::vector<phi::DenseTensor>&,  // NOLINT
      std::vector<phi::DenseTensor>&,  // NOLINT
      bool) {
    PADDLE_THROW(phi::errors::InvalidArgument(
        "ProcessGroup%s does not support all_gather with sync_op flag",
        GetBackendName()));
  }

  virtual std::shared_ptr<ProcessGroup::Task> AllToAll(
      std::vector<phi::DenseTensor>&,    // NOLINT
      std::vector<phi::DenseTensor>&) {  // NOLINT
    PADDLE_THROW(phi::errors::InvalidArgument(
        "ProcessGroup%s does not support AllToAll", GetBackendName()));
  }

  virtual std::shared_ptr<ProcessGroup::Task> Reduce(
      std::vector<phi::DenseTensor>&,  // NOLINT
      std::vector<phi::DenseTensor>&,  // NOLINT
      const ReduceOptions& opts UNUSED) {
    PADDLE_THROW(phi::errors::InvalidArgument(
        "ProcessGroup%s does not support reduce", GetBackendName()));
  }

  virtual std::shared_ptr<ProcessGroup::Task> Scatter(
      std::vector<phi::DenseTensor>&,  // NOLINT
      std::vector<phi::DenseTensor>&,  // NOLINT
      const ScatterOptions&) {
    PADDLE_THROW(phi::errors::InvalidArgument(
        "ProcessGroup%s does not support scatter", GetBackendName()));
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
