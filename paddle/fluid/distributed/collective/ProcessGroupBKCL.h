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

#include "paddle/fluid/distributed/collective/BKCLTools.h"
#include "paddle/fluid/distributed/collective/ProcessGroup.h"
#include "paddle/fluid/distributed/store/store.h"
#include "paddle/fluid/platform/device/xpu/xpu_header.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/gen_comm_id_helper.h"
#include "paddle/fluid/platform/place.h"

constexpr const char* BKCL_BACKEND_NAME = "BKCL";

namespace paddle {
namespace distributed {

using Place = paddle::platform::Place;

// BKCL funcs use separate communication stream by default
class ProcessGroupBKCL : public ProcessGroup {
 public:
  class BKCLTask : public ProcessGroup::Task,
                   public std::enable_shared_from_this<BKCLTask> {
   public:
    BKCLTask(const std::vector<Place>& places,
             int rank,
             CommType CommType,
             const std::vector<phi::DenseTensor>& inputs);

    // TODO(zhangxiaoci): XPU do not support event query for now
    // bool IsCompleted();

    void SynchronizeStreams();

    bool Wait(std::chrono::milliseconds timeout = kWaitTimeout);

    void Synchronize();

    void SetOutputs(std::vector<phi::DenseTensor>& outputs);  // NOLINT

    virtual ~BKCLTask();

    std::vector<XPUEventManager> control_events_;
    std::vector<phi::DenseTensor> barrierTensors_;

   protected:
    std::vector<Place> places_;
    std::shared_ptr<std::vector<phi::DenseTensor>> outputs_;
  };

  ProcessGroupBKCL(const std::shared_ptr<Store>& store,
                   int rank,
                   int size,
                   const platform::Place& place,
                   int gid);

  const std::string GetBackendName() const override {
    return std::string(BKCL_BACKEND_NAME);
  }

  const phi::DeviceContext& GetDeviceContext(const Place& place) const override;

  std::shared_ptr<ProcessGroup::Task> AllReduce(
      std::vector<phi::DenseTensor>& in_tensors,
      std::vector<phi::DenseTensor>& out_tensors,
      const AllreduceOptions& = AllreduceOptions()) override;

  std::shared_ptr<ProcessGroup::Task> AllReduce(
      std::vector<phi::DenseTensor>& in_tensors,
      std::vector<phi::DenseTensor>& out_tensors,
      const AllreduceOptions& options,
      bool sync_op) override;

  std::shared_ptr<ProcessGroup::Task> Broadcast(
      std::vector<phi::DenseTensor>& in_tensors,
      std::vector<phi::DenseTensor>& out_tensors,
      const BroadcastOptions& = BroadcastOptions()) override;

  std::shared_ptr<ProcessGroup::Task> Broadcast(
      std::vector<phi::DenseTensor>& in_tensors,
      std::vector<phi::DenseTensor>& out_tensors,
      const BroadcastOptions&,
      bool sync_op) override;

  std::shared_ptr<ProcessGroup::Task> AllGather(
      std::vector<phi::DenseTensor>& in_tensors,
      std::vector<phi::DenseTensor>& out_tensors) override;

  std::shared_ptr<ProcessGroup::Task> AllGather(
      std::vector<phi::DenseTensor>& in_tensors,
      std::vector<phi::DenseTensor>& out_tensors,
      bool sync_op) override;

  std::shared_ptr<ProcessGroup::Task> Barrier(
      const BarrierOptions& = BarrierOptions()) override;

 protected:
  virtual std::shared_ptr<ProcessGroupBKCL::BKCLTask> CreateTask(
      std::vector<Place> places,
      int rank,
      CommType op_type,
      const std::vector<phi::DenseTensor>& inputs);

  std::shared_ptr<Store> store_;
  std::shared_ptr<BKCLCommManager> BKCL_comm_;
  std::mutex mutex_;
  std::unordered_map<std::string, std::vector<std::shared_ptr<BKCLCommManager>>>
      places_to_bkclcomm_;

  std::unordered_map<std::string, std::vector<XPUEventManager>>
      places_to_events_;

  std::unordered_map<std::string, std::vector<std::unique_ptr<XPUContext>>>
      places_to_ctx_;

  std::set<int> used_place_ids_;

 private:
  void BcastBKCLId(std::vector<BKCLUniqueId>& BKCL_ids,  // NOLINT
                   int root,                             // NOLINT
                   int server_fd);

  void BroadcastUniqueBKCLID(std::vector<BKCLUniqueId>& BKCL_ids);  // NOLINT

  template <typename Fn>
  std::shared_ptr<ProcessGroup::Task> Collective(
      std::vector<phi::DenseTensor>& inputs,   // NOLINT
      std::vector<phi::DenseTensor>& outputs,  // NOLINT
      Fn fn,
      CommType op_type);

  void CreateBKCLManagerCache(const std::string& places_key,
                              const std::vector<Place>& places);
};

}  //  namespace distributed
}  //  namespace paddle
