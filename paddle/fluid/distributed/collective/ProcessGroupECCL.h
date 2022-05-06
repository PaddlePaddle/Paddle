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
#if defined(PADDLE_WITH_ECCL)
#include <eccl.h>
#include <eccl_device.h>
#include <eccl_types.h>
#endif

#include "paddle/fluid/distributed/collective/ProcessGroup.h"
#include "paddle/fluid/platform/cuda_device_guard.h"
#include "paddle/fluid/platform/device_context.h"

#include "paddle/fluid/distributed/store/store.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/gen_comm_id_helper.h"
#include "paddle/fluid/platform/place.h"
#include "paddle/fluid/platform/stream/cuda_stream.h"

#if defined(PADDLE_WITH_ECCL)
#include "paddle/fluid/distributed/collective/ECCLTools.h"
#include "paddle/fluid/platform/dynload/eccl.h"
#endif

constexpr const char* ECCL_BACKEND_NAME = "ECCL";

namespace paddle {
namespace distributed {

using Place = paddle::platform::Place;
using CUDAStream = platform::stream::CUDAStream;
using CUDADeviceContext = paddle::platform::CUDADeviceContext;

class ProcessGroupECCL : public ProcessGroup {
 public:
  class ECCLTask : public ProcessGroup::Task,
                   public std::enable_shared_from_this<ECCLTask> {
   public:
    ECCLTask(const std::vector<Place>& places, int rank, CommType CommType,
             const std::vector<phi::DenseTensor>& inputs);

    bool IsCompleted();

    void SynchronizeStreams();

    bool Wait(std::chrono::milliseconds timeout = kWaitTimeout);

    void Synchronize();

    void SetOutputs(std::vector<phi::DenseTensor>& outputs);  // NOLINT

    virtual ~ECCLTask();

    std::vector<EventManager> control_events_;
    std::vector<phi::DenseTensor> barrierTensors_;

   protected:
    std::vector<Place> places_;
    std::vector<std::shared_ptr<ECCLCommManager>> ecclComms_;
    std::shared_ptr<std::vector<phi::DenseTensor>> outputs_;

   private:
  };

  ProcessGroupECCL(const std::shared_ptr<Store>& store, int rank, int size,
                   int gid);

  const std::string GetBackendName() const override {
    return std::string(ECCL_BACKEND_NAME);
  }

  std::shared_ptr<ProcessGroup::Task> AllReduce(
      std::vector<phi::DenseTensor>& tensors,
      const AllreduceOptions& = AllreduceOptions()) override;

  std::shared_ptr<ProcessGroup::Task> Broadcast(
      std::vector<phi::DenseTensor>& tensors,
      const BroadcastOptions& = BroadcastOptions()) override;

  std::shared_ptr<ProcessGroup::Task> Barrier(
      const BarrierOptions& = BarrierOptions()) override;

  std::shared_ptr<ProcessGroup::Task> Send(
      std::vector<phi::DenseTensor>& tensors, int dst_rank) override;

  std::shared_ptr<ProcessGroup::Task> Recv(
      std::vector<phi::DenseTensor>& tensors, int src_rank) override;

  std::shared_ptr<ProcessGroup::Task> AllGather(
      std::vector<phi::DenseTensor>& in_tensors,
      std::vector<phi::DenseTensor>& out_tensors) override;

  std::shared_ptr<ProcessGroup::Task> AllToAll(
      std::vector<phi::DenseTensor>& in,
      std::vector<phi::DenseTensor>& out) override;

  std::shared_ptr<ProcessGroup::Task> Reduce(
      std::vector<phi::DenseTensor>& tensors,
      const ReduceOptions& opts) override;

  std::shared_ptr<ProcessGroup::Task> Scatter(
      std::vector<phi::DenseTensor>& in_tensors,
      std::vector<phi::DenseTensor>& out_tensors,
      const ScatterOptions&) override;

 protected:
  virtual std::shared_ptr<ProcessGroupECCL::ECCLTask> CreateTask(
      std::vector<Place> places, int rank, CommType opType,
      const std::vector<phi::DenseTensor>& inputs);

 protected:
  std::shared_ptr<Store> store_;
  std::shared_ptr<ECCLCommManager> eccl_comm_;
  std::mutex mutex_;
  std::unordered_map<std::string, std::vector<std::shared_ptr<ECCLCommManager>>>
      places_to_ecclcomm_;

  std::unordered_map<std::string, std::vector<EventManager>> places_to_events_;

  std::unordered_map<std::string,
                     std::vector<std::unique_ptr<CUDADeviceContext>>>
      places_to_ctx_;

  std::set<int> used_place_ids_;
  int device_id_;

 private:
  void BcastECCLId(std::vector<ecclUniqueId>& eccl_ids, int root,  // NOLINT
                   int server_fd);

  void BroadcastUniqueECCLID(std::vector<ecclUniqueId>& eccl_ids);  // NOLINT

  template <typename Fn>
  std::shared_ptr<ProcessGroup::Task> Collective(
      std::vector<phi::DenseTensor>& inputs,   // NOLINT
      std::vector<phi::DenseTensor>& outputs,  // NOLINT
      Fn fn, CommType op_type);

  template <typename Fn>
  std::shared_ptr<ProcessGroup::Task> PointToPoint(
      std::vector<phi::DenseTensor>& tensors,  // NOLINT
      Fn fn, int dst_rank, CommType op_type);

  void CreateECCLManagerCache(const std::string& places_key,
                              const std::vector<Place>& places);
};

}  //  namespace distributed
}  //  namespace paddle
