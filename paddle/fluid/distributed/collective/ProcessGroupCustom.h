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

#include "paddle/fluid/distributed/collective/CustomCCLTools.h"
#include "paddle/fluid/distributed/collective/ProcessGroup.h"
#include "paddle/fluid/distributed/store/store.h"
#include "paddle/fluid/platform/device/npu/npu_stream.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/gen_comm_id_helper.h"
#include "paddle/fluid/platform/place.h"

namespace paddle {
namespace distributed {
using Place = paddle::platform::Place;
using CustomDeviceContext = paddle::platform::CustomDeviceContext;
class ProcessGroupCustom : public ProcessGroup {
 public:
  class CustomTask : public ProcessGroup::Task,
                     public std::enable_shared_from_this<CustomTask> {
   public:
    CustomTask(const std::vector<Place>& places,
               int rank,
               CommType CommType,
               const std::vector<phi::DenseTensor>& inputs);

    bool IsCompleted();
    void SynchronizeStreams();
    bool Wait(std::chrono::milliseconds timeout = kWaitTimeout);
    void Synchronize();
    void SetOutputs(std::vector<phi::DenseTensor>& outputs);  // NOLINT
    virtual ~CustomTask();

    std::vector<CustomEventManager> control_events_;
    std::vector<phi::DenseTensor> barrierTensors_;

   protected:
    std::vector<Place> places_;
    std::vector<std::shared_ptr<CustomCCLCommManager>> cclComms_;
    std::shared_ptr<std::vector<phi::DenseTensor>> outputs_;

   private:
    const std::string device_type_;
  };

  ProcessGroupCustom(const std::shared_ptr<Store>& store,
                     const std::string& device_type,
                     int rank,
                     int size,
                     int gid);

  std::string GetBackendName() const override { return "XCCL_" + device_type_; }

  std::shared_ptr<ProcessGroup::Task> AllGather(
      phi::DenseTensor* out_tensor,
      const phi::DenseTensor& in_tensor,
      int64_t offset,
      int64_t numel,
      bool sync_op) override;

  std::shared_ptr<ProcessGroup::Task> AllReduce(
      phi::DenseTensor* out_tensor,
      const phi::DenseTensor& in_tensor,
      const AllreduceOptions& opts,
      bool sync_op) override;

  std::shared_ptr<ProcessGroup::Task> Broadcast(
      phi::DenseTensor* out_tensor,
      const phi::DenseTensor& in_tensor,
      const BroadcastOptions& opts,
      bool sync_op) override;

  std::shared_ptr<ProcessGroup::Task> Barrier(
      const BarrierOptions& = BarrierOptions()) override;

  const phi::DeviceContext& GetDeviceContext(const Place& place) const override;

  phi::ccl::CCLComm CustomCCLComm(const Place& place) const;

  // TODO(sunyilun): methods below will be removed later
  std::shared_ptr<ProcessGroup::Task> AllGather(
      std::vector<phi::DenseTensor>& in_tensors,
      std::vector<phi::DenseTensor>& out_tensors) override;

  std::shared_ptr<ProcessGroup::Task> AllReduce(
      std::vector<phi::DenseTensor>& in_tensors,
      std::vector<phi::DenseTensor>& out_tensors,
      const AllreduceOptions& = AllreduceOptions()) override;

  std::shared_ptr<ProcessGroup::Task> Broadcast(
      std::vector<phi::DenseTensor>& in_tensors,
      std::vector<phi::DenseTensor>& out_tensors,
      const BroadcastOptions& = BroadcastOptions()) override;

 protected:
  virtual std::shared_ptr<ProcessGroupCustom::CustomTask> CreateTask(
      std::vector<Place> places,
      int rank,
      CommType opType,
      const std::vector<phi::DenseTensor>& inputs);

  std::shared_ptr<Store> store_;
  std::shared_ptr<CustomCCLCommManager> custom_comm_;
  std::mutex mutex_;
  std::unordered_map<std::string,
                     std::vector<std::shared_ptr<CustomCCLCommManager>>>
      places_to_customcomm_;
  std::unordered_map<std::string, std::vector<CustomEventManager>>
      places_to_events_;
  std::unordered_map<std::string,
                     std::vector<std::unique_ptr<CustomDeviceContext>>>
      places_to_ctx_;
  std::set<int> used_place_ids_;

 private:
  void BcastCustomId(std::vector<phi::ccl::CCLRootId>& ccl_ids,  // NOLINT
                     int root,
                     int server_fd);

  void BroadcastUniqueCustomID(
      std::vector<phi::ccl::CCLRootId>& custom_ccl_ids);  // NOLINT

  template <typename Fn>
  std::shared_ptr<ProcessGroup::Task> Collective(
      std::vector<phi::DenseTensor>& inputs,   // NOLINT
      std::vector<phi::DenseTensor>& outputs,  // NOLINT
      Fn fn,
      CommType op_type);

  void CreateCustomManagerCache(const std::string& places_key,
                                const std::vector<Place>& places);
  const std::string device_type_;
};
}  //  namespace distributed
}  //  namespace paddle
