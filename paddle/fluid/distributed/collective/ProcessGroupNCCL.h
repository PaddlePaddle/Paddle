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
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "paddle/fluid/distributed/collective/ProcessGroup.h"
#include "paddle/fluid/platform/cuda_device_guard.h"
#include "paddle/fluid/platform/device_context.h"

#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/gen_comm_id_helper.h"
#include "paddle/fluid/platform/place.h"
#include "paddle/fluid/platform/stream/cuda_stream.h"

#if defined(PADDLE_WITH_NCCL)
#include "paddle/fluid/distributed/collective/NCCLTools.h"
#include "paddle/fluid/platform/dynload/nccl.h"
#endif

constexpr const char* NCCL_BACKEND_NAME = "NCCL";

namespace paddle {
namespace distributed {

using Place = paddle::platform::Place;
using CUDAStream = platform::stream::CUDAStream;
using CUDADeviceContext = paddle::platform::CUDADeviceContext;

class ProcessGroupNCCL : public ProcessGroup {
 public:
  class NCCLTask : public ProcessGroup::Task,
                   public std::enable_shared_from_this<NCCLTask> {
   public:
    NCCLTask(const std::vector<Place>& places, int rank, CommType CommType,
             const std::vector<Tensor>& inputs);

    bool IsCompleted();

    void SynchronizeStreams();

    bool Wait(std::chrono::milliseconds timeout = kWaitTimeout);

    void Synchronize();

    void SetOutputs(std::vector<Tensor>& outputs);  // NOLINT

    virtual ~NCCLTask();

    std::vector<EventManager> control_events_;
    std::vector<Tensor> barrierTensors_;

   protected:
    std::vector<Place> places_;
    std::vector<std::shared_ptr<NCCLCommManager>> ncclComms_;
    std::shared_ptr<std::vector<Tensor>> outputs_;

   private:
  };

  ProcessGroupNCCL(const ProcessGroupStrategy& strategy, int rank, int size);

  const std::string GetBackendName() const override {
    return std::string(NCCL_BACKEND_NAME);
  }

  std::shared_ptr<ProcessGroup::Task> AllReduce(
      std::vector<Tensor>& tensors,
      const AllreduceOptions& = AllreduceOptions()) override;

  std::shared_ptr<ProcessGroup::Task> Broadcast(
      std::vector<Tensor>& tensors,
      const BroadcastOptions& = BroadcastOptions()) override;

  std::shared_ptr<ProcessGroup::Task> Barrier(
      const BarrierOptions& = BarrierOptions()) override;

  std::shared_ptr<ProcessGroup::Task> Send(std::vector<Tensor>& tensors,
                                           int dst_rank) override;

  std::shared_ptr<ProcessGroup::Task> Recv(std::vector<Tensor>& tensors,
                                           int src_rank) override;

 protected:
  virtual std::shared_ptr<ProcessGroupNCCL::NCCLTask> CreateTask(
      std::vector<Place> places, int rank, CommType opType,
      const std::vector<Tensor>& inputs);

 protected:
  ProcessGroupStrategy strategy_;
  std::shared_ptr<NCCLCommManager> nccl_comm_;
  std::mutex mutex_;
  std::unordered_map<std::string, std::vector<std::shared_ptr<NCCLCommManager>>>
      places_to_ncclcomm_;

  std::unordered_map<std::string, std::vector<EventManager>> places_to_events_;

  std::unordered_map<std::string,
                     std::vector<std::unique_ptr<CUDADeviceContext>>>
      places_to_ctx_;

  std::set<int> used_place_ids_;

 private:
  void BcastNCCLId(std::vector<ncclUniqueId>& nccl_ids, int root,  // NOLINT
                   int server_fd);

  void BroadcastUniqueNCCLID(std::vector<ncclUniqueId>& nccl_ids);  // NOLINT

  template <typename Fn>
  std::shared_ptr<ProcessGroup::Task> Collective(
      std::vector<Tensor>& inputs,   // NOLINT
      std::vector<Tensor>& outputs,  // NOLINT
      Fn fn, CommType op_type);

  template <typename Fn>
  std::shared_ptr<ProcessGroup::Task> PointToPoint(
      std::vector<Tensor>& tensors,  // NOLINT
      Fn fn, int dst_rank, CommType op_type);

  void CreateNCCLManagerCache(const std::string& places_key,
                              const std::vector<Place>& places);
};

}  //  namespace distributed
}  //  namespace paddle
