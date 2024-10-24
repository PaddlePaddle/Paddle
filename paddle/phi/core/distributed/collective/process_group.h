// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "paddle/common/errors.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/device_context.h"
#include "paddle/phi/core/distributed/types.h"
#include "paddle/phi/core/distributed/utils.h"
#include "paddle/phi/core/enforce.h"

constexpr auto kWaitTimeout = std::chrono::milliseconds(0);

namespace phi {
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

  int GetGid() const { return gid_; }

  std::string GetGroupMessage() const {
    return std::string("rank_in_group: ") + std::to_string(rank_) +
           std::string(", nranks: ") + std::to_string(size_) +
           std::string(", gid: ") + std::to_string(gid_) +
           std::string(", backend: ") + GetBackendName();
  }

  virtual std::string GetBackendName() const = 0;

  virtual phi::DeviceContext* GetDeviceContext(
      const Place& place UNUSED) const {
    PADDLE_THROW(common::errors::Unimplemented(
        "ProcessGroup%s does not support get device_context.",
        GetBackendName()));
  }

  virtual phi::DeviceContext* GetDeviceContext(
      const Place& place UNUSED, bool use_calc_stream UNUSED) const {
    PADDLE_THROW(common::errors::Unimplemented(
        "ProcessGroup%s does not support get device_context.",
        GetBackendName()));
  }

  virtual void StartCoalescing() {}

  virtual void EndCoalescing(
      std::optional<std::vector<std::shared_ptr<ProcessGroup::Task>>>
          tasks_opt = std::nullopt) {}

  virtual void EagerConnect() {}

  virtual void EagerConnectRingExchange() {}

  // without stream APIs
  virtual std::shared_ptr<ProcessGroup::Task> AllGather(
      phi::DenseTensor* out_tensor UNUSED,
      const phi::DenseTensor& in_tensor UNUSED,
      bool sync_op UNUSED) {
    PADDLE_THROW(common::errors::Unimplemented(
        "ProcessGroup%s does not support all_gather with sync_op flag.",
        GetBackendName()));
  }

  virtual std::shared_ptr<ProcessGroup::Task> AllGather(
      phi::DenseTensor* out_tensor UNUSED,
      const phi::DenseTensor& in_tensor UNUSED,
      int64_t offset UNUSED,
      int64_t numel UNUSED,
      bool sync_op UNUSED) {
    PADDLE_THROW(common::errors::Unimplemented(
        "ProcessGroup%s does not support all_gather with sync_op flag.",
        GetBackendName()));
  }

  virtual std::shared_ptr<ProcessGroup::Task> AllReduce(
      phi::DenseTensor* out_tensor UNUSED,
      const phi::DenseTensor& in_tensor UNUSED,
      const AllreduceOptions& opts UNUSED,
      bool sync_op UNUSED) {
    PADDLE_THROW(common::errors::Unimplemented(
        "ProcessGroup%s does not support all_reduce with sync_op flag.",
        GetBackendName()));
  }

  virtual std::shared_ptr<ProcessGroup::Task> AllToAll(
      phi::DenseTensor* out_tensor UNUSED,
      const phi::DenseTensor& in_tensor UNUSED,
      const std::vector<int64_t>& out_size_each_rank UNUSED,
      const std::vector<int64_t>& in_size_each_rank UNUSED,
      bool sync_op UNUSED) {
    PADDLE_THROW(common::errors::Unimplemented(
        "ProcessGroup%s does not support all_to_all with sync_op flag.",
        GetBackendName()));
  }

  virtual std::shared_ptr<ProcessGroup::Task> Barrier(
      const BarrierOptions& UNUSED = BarrierOptions()) {
    PADDLE_THROW(common::errors::Unimplemented(
        "ProcessGroup%s does not support barrier.", GetBackendName()));
  }

  virtual std::shared_ptr<ProcessGroup::Task> Broadcast(
      phi::DenseTensor* out_tensor UNUSED,
      const phi::DenseTensor& in_tensor UNUSED,
      const BroadcastOptions& opts UNUSED,
      bool sync_op UNUSED) {
    PADDLE_THROW(common::errors::Unimplemented(
        "ProcessGroup%s does not support broadcast with sync_op flag",
        GetBackendName()));
  }

  virtual std::shared_ptr<ProcessGroup::Task> Reduce(
      phi::DenseTensor* out_tensor UNUSED,
      const phi::DenseTensor& in_tensor UNUSED,
      const ReduceOptions& opts UNUSED,
      bool sync_op UNUSED) {
    PADDLE_THROW(common::errors::Unimplemented(
        "ProcessGroup%s does not support reduce with sync_op flag.",
        GetBackendName()));
  }

  virtual std::shared_ptr<ProcessGroup::Task> ReduceScatter(
      phi::DenseTensor* out_tensor UNUSED,
      const phi::DenseTensor& in_tensor UNUSED,
      const ReduceScatterOptions& opts UNUSED,
      bool sync_op UNUSED) {
    PADDLE_THROW(common::errors::Unimplemented(
        "ProcessGroup%s does not support reduce_scatter with sync_op flag.",
        GetBackendName()));
  }

  virtual std::shared_ptr<ProcessGroup::Task> Scatter(
      phi::DenseTensor* out_tensor UNUSED,
      const phi::DenseTensor& in_tensor UNUSED,
      const ScatterOptions& opts UNUSED,
      bool sync_op UNUSED) {
    PADDLE_THROW(common::errors::Unimplemented(
        "ProcessGroup%s does not support scatter with sync_op flag.",
        GetBackendName()));
  }

  virtual std::shared_ptr<ProcessGroup::Task> Recv(phi::DenseTensor* tensor
                                                       UNUSED,
                                                   int src_rank UNUSED,
                                                   bool sync_op UNUSED) {
    PADDLE_THROW(common::errors::Unimplemented(
        "ProcessGroup%s does not support recv with sync_op flag.",
        GetBackendName()));
  }

  virtual std::shared_ptr<ProcessGroup::Task> Recv(phi::DenseTensor* tensor
                                                       UNUSED,
                                                   int src_rank UNUSED,
                                                   int64_t offset UNUSED,
                                                   int64_t numel UNUSED,
                                                   bool sync_op UNUSED) {
    PADDLE_THROW(common::errors::Unimplemented(
        "ProcessGroup%s does not support recv with sync_op flag.",
        GetBackendName()));
  }

  virtual std::shared_ptr<ProcessGroup::Task> Send(
      const phi::DenseTensor& tensor UNUSED,
      int dst_rank UNUSED,
      bool sync_op UNUSED) {
    PADDLE_THROW(common::errors::Unimplemented(
        "ProcessGroup%s does not support send with sync_op flag.",
        GetBackendName()));
  }

  virtual std::shared_ptr<ProcessGroup::Task> Send(
      const phi::DenseTensor& tensor UNUSED,
      int dst_rank UNUSED,
      int64_t offset UNUSED,
      int64_t numel UNUSED,
      bool sync_op UNUSED) {
    PADDLE_THROW(common::errors::Unimplemented(
        "ProcessGroup%s does not support send with sync_op flag.",
        GetBackendName()));
  }

  // stream APIs
  virtual std::shared_ptr<ProcessGroup::Task> AllGather(
      phi::DenseTensor* out_tensor UNUSED,
      const phi::DenseTensor& in_tensor UNUSED,
      bool sync_op UNUSED,
      bool use_calc_stream UNUSED) {
    PADDLE_THROW(common::errors::Unimplemented(
        "ProcessGroup%s does not support all_gather "
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
    PADDLE_THROW(common::errors::Unimplemented(
        "ProcessGroup%s does not support all_gather "
        "with sync_op and use_calc_stream flag.",
        GetBackendName()));
  }

  virtual std::shared_ptr<ProcessGroup::Task> AllReduce(
      phi::DenseTensor* out_tensor UNUSED,
      const phi::DenseTensor& in_tensor UNUSED,
      const AllreduceOptions& opts UNUSED,
      bool sync_op UNUSED,
      bool use_calc_stream UNUSED) {
    PADDLE_THROW(common::errors::Unimplemented(
        "ProcessGroup%s does not support all_reduce "
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
    PADDLE_THROW(common::errors::Unimplemented(
        "ProcessGroup%s does not support all_to_all "
        "with sync_op and use_calc_stream flag.",
        GetBackendName()));
  }

  virtual std::shared_ptr<ProcessGroup::Task> Broadcast(
      phi::DenseTensor* out_tensor UNUSED,
      const phi::DenseTensor& in_tensor UNUSED,
      const BroadcastOptions& opts UNUSED,
      bool sync_op UNUSED,
      bool use_calc_stream) {
    PADDLE_THROW(common::errors::Unimplemented(
        "ProcessGroup%s does not support broadcast "
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
        common::errors::Unimplemented("ProcessGroup%s does not support reduce "
                                      "with sync_op and use_calc_stream flag.",
                                      GetBackendName()));
  }

  virtual std::shared_ptr<ProcessGroup::Task> ReduceScatter(
      phi::DenseTensor* out_tensor UNUSED,
      const phi::DenseTensor& in_tensor UNUSED,
      const ReduceScatterOptions& opts UNUSED,
      bool sync_op UNUSED,
      bool use_calc_stream UNUSED) {
    PADDLE_THROW(common::errors::Unimplemented(
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
        common::errors::Unimplemented("ProcessGroup%s does not support scatter "
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
        common::errors::Unimplemented("ProcessGroup%s does not support gather "
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
        common::errors::Unimplemented("ProcessGroup%s does not support gather "
                                      "with sync_op and use_calc_stream flag.",
                                      GetBackendName()));
  }

  virtual std::shared_ptr<ProcessGroup::Task> Recv(
      phi::DenseTensor* tensor UNUSED,
      int src_rank UNUSED,
      bool sync_op UNUSED,
      bool use_calc_stream UNUSED) {
    PADDLE_THROW(common::errors::Unimplemented(
        "ProcessGroup%s does not support recv with "
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
        common::errors::Unimplemented("ProcessGroup%s does not support recv "
                                      "with sync_op and use_calc_stream flag.",
                                      GetBackendName()));
  }

  virtual std::shared_ptr<ProcessGroup::Task> Send(
      const phi::DenseTensor& tensor UNUSED,
      int dst_rank UNUSED,
      bool sync_op UNUSED,
      bool use_calc_stream UNUSED) {
    PADDLE_THROW(
        common::errors::Unimplemented("ProcessGroup%s does not support send "
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
        common::errors::Unimplemented("ProcessGroup%s does not support send "
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
    PADDLE_THROW(common::errors::InvalidArgument(
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
  int global_rank_{-1};
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

  ProcessGroup* get(int gid) {
    auto it = map_.find(gid);
    if (it == map_.end()) {
      return nullptr;
    }
    return it->second;
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

static void CheckTensorContiguous(const phi::DenseTensor& tensor) {
  if (!tensor.meta().is_contiguous()) {
    PADDLE_THROW(
        common::errors::InvalidArgument("The tensor must be contiguous"));
  }
}

static void CheckTensorContiguous(const std::vector<phi::DenseTensor>& inputs) {
  for (const auto& tensor : inputs) {
    if (!tensor.meta().is_contiguous()) {
      PADDLE_THROW(
          common::errors::InvalidArgument("The tensor must be contiguous"));
    }
  }
}

}  //  namespace distributed
}  // namespace phi
