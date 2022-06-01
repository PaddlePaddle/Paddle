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
#include "paddle/fluid/platform/device/npu/npu_stream.h"
#include "paddle/fluid/platform/device_context.h"

#include "paddle/fluid/distributed/collective/HCCLTools.h"
#include "paddle/fluid/distributed/store/store.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/gen_comm_id_helper.h"
#include "paddle/fluid/platform/place.h"

constexpr const char* HCCL_BACKEND_NAME = "HCCL";

namespace paddle {
namespace distributed {

using Place = paddle::platform::Place;
using NPUStream = platform::stream::NPUStream;
using NPUDeviceContext = paddle::platform::NPUDeviceContext;

class ProcessGroupHCCL : public ProcessGroup {
 public:
  class HCCLTask : public ProcessGroup::Task,
                   public std::enable_shared_from_this<HCCLTask> {
   public:
    HCCLTask(const std::vector<Place>& places, int rank, CommType CommType,
             const std::vector<phi::DenseTensor>& inputs);

    bool IsCompleted();

    void SynchronizeStreams();

    bool Wait(std::chrono::milliseconds timeout = kWaitTimeout);

    void Synchronize();

    void SetOutputs(std::vector<phi::DenseTensor>& outputs);  // NOLINT

    virtual ~HCCLTask();

    std::vector<NPUEventManager> control_events_;

   protected:
    std::vector<Place> places_;
    std::vector<std::shared_ptr<HCCLCommManager>> hcclComms_;
    std::shared_ptr<std::vector<phi::DenseTensor>> outputs_;

   private:
  };

  ProcessGroupHCCL(const std::shared_ptr<Store>& store, int rank, int size,
                   const platform::Place& place, int gid);

  const std::string GetBackendName() const override {
    return std::string(HCCL_BACKEND_NAME);
  }

  std::shared_ptr<ProcessGroup::Task> AllReduce(
      std::vector<phi::DenseTensor>& in_tensors,
      std::vector<phi::DenseTensor>& out_tensors,
      const AllreduceOptions& = AllreduceOptions()) override;

  std::shared_ptr<ProcessGroup::Task> Broadcast(
      std::vector<phi::DenseTensor>& in_tensors,
      std::vector<phi::DenseTensor>& out_tensors,
      const BroadcastOptions& = BroadcastOptions()) override;

 protected:
  virtual std::shared_ptr<ProcessGroupHCCL::HCCLTask> CreateTask(
      std::vector<Place> places, int rank, CommType opType,
      const std::vector<phi::DenseTensor>& inputs);

  std::shared_ptr<Store> store_;
  std::shared_ptr<HCCLCommManager> hccl_comm_;
  std::mutex mutex_;
  std::unordered_map<std::string, std::vector<std::shared_ptr<HCCLCommManager>>>
      places_to_hcclcomm_;

  std::unordered_map<std::string, std::vector<NPUEventManager>>
      places_to_events_;

  std::unordered_map<std::string,
                     std::vector<std::unique_ptr<NPUDeviceContext>>>
      places_to_ctx_;

  std::set<int> used_place_ids_;

 private:
  void BcastHCCLId(std::vector<HcclRootInfo>& hccl_ids, int root,  // NOLINT
                   int server_fd);

  void BroadcastUniqueHCCLID(std::vector<HcclRootInfo>& hccl_ids);  // NOLINT

  template <typename Fn>
  std::shared_ptr<ProcessGroup::Task> Collective(
      std::vector<phi::DenseTensor>& inputs,   // NOLINT
      std::vector<phi::DenseTensor>& outputs,  // NOLINT
      Fn fn, CommType op_type);

  void CreateHCCLManagerCache(const std::string& places_key,
                              const std::vector<Place>& places);
};

}  //  namespace distributed
}  //  namespace paddle
