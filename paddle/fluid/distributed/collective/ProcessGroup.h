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
#include "paddle/fluid/distributed/store/tcp_utils.h"
#include "paddle/fluid/eager/api/utils/tensor_utils.h"

#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/framework/variable.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace distributed {

using Tensor = paddle::experimental::Tensor;
constexpr auto kProcessGroupNoTimeout = std::chrono::milliseconds::zero();

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

std::string commTypeToString(CommType commType);

class ProcessGroup {
 public:
  class Task {
   public:
    explicit Task(int rank, CommType opType = CommType::UNKNOWN);

    virtual ~Task();
    virtual bool IsCompleted();
    virtual bool Wait(
        std::chrono::milliseconds timeout = kProcessGroupNoTimeout);
    virtual void Synchronize();

   protected:
    const int rank_;
    CommType comm_type_;
    mutable std::mutex mutex_;
    bool is_completed_ = false;
  };

  explicit ProcessGroup(int rank, int size);
  virtual ~ProcessGroup() {}

  int GetRank() const { return rank_; }

  int GetSize() const { return size_; }

  virtual const std::string GetBackendName() const = 0;

  virtual std::shared_ptr<ProcessGroup::Task> AllReduce(
      std::vector<Tensor>& /* tensors */,
      const AllreduceOptions& = AllreduceOptions()) {
    PADDLE_THROW(platform::errors::Fatal(
        "ProcessGroup%s does not support allreduce", GetBackendName()));
  }

  virtual std::shared_ptr<ProcessGroup::Task> Broadcast(
      std::vector<Tensor>& /* tensors */,
      const BroadcastOptions& = BroadcastOptions()) {
    PADDLE_THROW(platform::errors::Fatal(
        "ProcessGroup%s does not support broadcast", GetBackendName()));
  }

  virtual std::shared_ptr<ProcessGroup::Task> Barrier(
      const BarrierOptions& = BarrierOptions()) {
    PADDLE_THROW(platform::errors::Fatal(
        "ProcessGroup%s does not support barrier", GetBackendName()));
  }

  virtual std::shared_ptr<ProcessGroup::Task> Send(
      std::vector<Tensor>& tensors /* tensors */, int dst_rank) {  // NOLINT
    PADDLE_THROW(platform::errors::Fatal("ProcessGroup%s does not support send",
                                         GetBackendName()));
  }

  virtual std::shared_ptr<ProcessGroup::Task> Recv(
      std::vector<Tensor>& tensors /* tensors */, int src_rank) {  // NOLINT
    PADDLE_THROW(platform::errors::Fatal(
        "ProcessGroup%s does not support receive", GetBackendName()));
  }

  virtual std::shared_ptr<ProcessGroup::Task> AllGather(
      std::vector<Tensor>& in_tensors /* tensors */,    // NOLINT
      std::vector<std::vector<Tensor>>& out_tensors) {  // NOLINT
    PADDLE_THROW(platform::errors::Fatal(
        "ProcessGroup%s does not support AllGather", GetBackendName()));
  }

  virtual std::shared_ptr<ProcessGroup::Task> AllToAll(
      std::vector<Tensor>& in /* tensors */,     // NOLINT
      std::vector<Tensor>& out /* tensors */) {  // NOLINT
    PADDLE_THROW(platform::errors::Fatal(
        "ProcessGroup%s does not support AllToAll", GetBackendName()));
  }

  virtual std::shared_ptr<ProcessGroup::Task> Reduce(
      std::vector<Tensor>& tensors /* tensors */,  // NOLINT
      const ReduceOptions& opts) {                 // NOLINT
    PADDLE_THROW(platform::errors::Fatal(
        "ProcessGroup%s does not support Reduce", GetBackendName()));
  }

  virtual std::shared_ptr<ProcessGroup::Task> Scatter(
      std::vector<std::vector<Tensor>>& in_tensors /* tensors */,  // NOLINT
      std::vector<Tensor>& out_tensors /* tensors */,              // NOLINT
      const ScatterOptions&) {                                     // NOLINT
    PADDLE_THROW(platform::errors::Fatal(
        "ProcessGroup%s does not support Scatter", GetBackendName()));
  }

 protected:
  const int rank_;
  const int size_;
};

}  //  namespace distributed
}  //  namespace paddle
