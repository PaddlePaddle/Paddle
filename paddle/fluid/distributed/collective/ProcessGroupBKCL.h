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

#include "paddle/fluid/distributed/collective/process_group_stream.h"
#include "paddle/fluid/distributed/store/store.h"
#include "paddle/fluid/platform/device/xpu/xpu_header.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/gen_comm_id_helper.h"
#include "paddle/fluid/platform/place.h"
#include "paddle/phi/core/device_context.h"

#if defined(PADDLE_WITH_XPU)
#include "paddle/fluid/distributed/collective/BKCLTools.h"
#endif

constexpr const char* BKCL_BACKEND_NAME = "BKCL";

namespace paddle {
namespace distributed {

using Place = paddle::platform::Place;

// BKCL funcs use separate communication stream by default
class ProcessGroupBKCL : public ProcessGroupStream {
 public:
  class BKCLTask final : public ProcessGroupStream::TaskStream,
                         public std::enable_shared_from_this<BKCLTask> {
   public:
    BKCLTask(const Place& place,
             int rank,
             CommType CommType,
             bool sync_op,
             bool use_calc_stream);
    virtual ~BKCLTask();

    // TODO(zhangxiaoci): XPU do not support event query for now
    bool IsCompleted() override;
    bool Wait(std::chrono::milliseconds timeout = kWaitTimeout) override;
    void Synchronize() override;

    void SynchronizeStreams();

   public:
    bool barrier_{false};
    std::shared_ptr<XPUEventManager> comm_event_;  // event on comm stream

   private:
    Place place_;
  };

 public:
  ProcessGroupBKCL(const std::shared_ptr<Store>& store,
                   int rank,
                   int size,
                   int gid);

  static std::shared_ptr<ProcessGroupBKCL> CreateProcessGroupBKCL(
      const std::shared_ptr<Store>& store, int rank, int size, int gid);

  std::string GetBackendName() const override {
    return std::string(BKCL_BACKEND_NAME);
  }

  phi::DeviceContext* GetDeviceContext(const Place& place) const override;

  phi::DeviceContext* GetDeviceContext(const Place& place,
                                       bool use_calc_stream) const override;

  std::shared_ptr<ProcessGroup::Task> AllReduce(
      phi::DenseTensor* out_tensor,
      const phi::DenseTensor& in_tensor,
      const AllreduceOptions& opts,
      bool sync_op,
      bool use_calc_stream) override;

  std::shared_ptr<ProcessGroup::Task> Broadcast(
      phi::DenseTensor* out_tensor,
      const phi::DenseTensor& in_tensor,
      const BroadcastOptions& opts,
      bool sync_op,
      bool use_calc_stream) override;

  std::shared_ptr<ProcessGroup::Task> AllGather(
      phi::DenseTensor* out_tensor,
      const phi::DenseTensor& in_tensor,
      int64_t offset,  // for compatibility, no use now
      int64_t numel,   // for compatibility, no use now
      bool sync_op,
      bool use_calc_stream) override;

  std::shared_ptr<ProcessGroup::Task> Reduce(phi::DenseTensor* out_tensor,
                                             const phi::DenseTensor& in_tensor,
                                             const ReduceOptions& opts,
                                             bool sync_op,
                                             bool use_calc_stream) override;

  std::shared_ptr<ProcessGroup::Task> Barrier(
      const BarrierOptions& = BarrierOptions()) override;

  static void GroupStart();

  static void GroupEnd();

  BKCLContext_t BKCLComm(const Place& place) const;

  // below are old apis
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

 private:
  std::shared_ptr<ProcessGroupBKCL::BKCLTask> CreateTask(const Place& place,
                                                         int rank,
                                                         CommType op_type,
                                                         bool sync_op,
                                                         bool use_calc_stream);

  void BroadcastUniqueBKCLID(BKCLUniqueId* bkcl_id);  // NOLINT

  void CreateBKCLEnvCache(const Place& place, const std::string& place_key);

  template <typename Fn>
  std::shared_ptr<ProcessGroupStream::Task> Collective(
      phi::DenseTensor* out_tensor,
      const phi::DenseTensor& in_tensor,
      Fn fn,
      CommType comm_type,
      bool sync_op,
      bool use_calc_stream);

  void SyncCalcStream(const Place& place);

 private:
  std::shared_ptr<Store> store_;
  std::mutex mutex_;
  std::shared_ptr<XPUEventManager> calc_event_;  // event on calc stream
  std::unordered_map<std::string, phi::XPUContext*> place_to_calc_ctx_;
  std::unordered_map<std::string, std::unique_ptr<phi::XPUContext>>
      place_to_comm_ctx_;
};

}  //  namespace distributed
}  //  namespace paddle
