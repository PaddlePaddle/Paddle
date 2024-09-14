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

#include "paddle/fluid/distributed/collective/process_group.h"
#include "paddle/fluid/distributed/collective/process_group_with_stream.h"
#include "paddle/phi/backends/custom/custom_context.h"
#include "paddle/phi/backends/device_manager.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/device_context.h"
#include "paddle/phi/core/distributed/store/store.h"
#include "paddle/phi/core/distributed/xccl_comm_context.h"

namespace paddle {
namespace distributed {

using Place = phi::Place;

class ProcessGroupCustom final : public ProcessGroupWithStream {
 public:
  class XCCLTask final : public ProcessGroupWithStream::TaskStream,
                         public std::enable_shared_from_this<XCCLTask> {
   public:
    XCCLTask(const Place& place,
             int rank,
             CommType comm_type,
             bool sync_op,
             bool use_calc_stream);
    virtual ~XCCLTask();

    bool IsCompleted() override;
    bool Wait(std::chrono::milliseconds timeout = kWaitTimeout) override;
    void Synchronize() override;
    void UpdateWaitChain(const phi::DeviceContext& ctx) override;

    bool IsBlockCPUInWait() const { return block_cpu_in_wait_; }
    void SetBlockCPUInWait() { block_cpu_in_wait_ = true; }

    // TODO(sunyilun): methods below will be removed later
    XCCLTask(const std::vector<Place>& places,
             int rank,
             CommType CommType,
             const std::vector<phi::DenseTensor>& inputs);

   private:
    bool block_cpu_in_wait_{false};
    Place task_place_;
    std::unique_ptr<phi::event::Event> comm_event_;  // event on comm stream
  };

 public:
  static std::shared_ptr<ProcessGroupCustom> CreateProcessGroupCustom(
      const std::shared_ptr<phi::distributed::Store>& store,
      const std::string& device_type,
      int rank,
      int size,
      int gid);

  ProcessGroupCustom(const std::shared_ptr<phi::distributed::Store>& store,
                     const std::string& device_type,
                     int rank,
                     int size,
                     int gid);

  std::string GetBackendName() const override { return "XCCL"; }

  std::string GetCommName(int rank);

  phi::DeviceContext* GetDeviceContext(const Place& place) const override;

  phi::DeviceContext* GetDeviceContext(const Place& place,
                                       bool use_calc_stream) const override;

  std::shared_ptr<ProcessGroup::Task> AllGather(
      phi::DenseTensor* out_tensor,
      const phi::DenseTensor& in_tensor,
      int64_t offset,
      int64_t numel,
      bool sync_op,
      bool use_calc_stream) override;

  std::shared_ptr<ProcessGroup::Task> AllReduce(
      phi::DenseTensor* out_tensor,
      const phi::DenseTensor& in_tensor,
      const AllreduceOptions& opts,
      bool sync_op,
      bool use_calc_stream) override;

  std::shared_ptr<ProcessGroup::Task> AllToAll(
      phi::DenseTensor* out_tensor,
      const phi::DenseTensor& in_tensor,
      const std::vector<int64_t>& out_size_each_rank,
      const std::vector<int64_t>& in_size_each_rank,
      bool sync_op,
      bool use_calc_stream) override;

  std::shared_ptr<ProcessGroup::Task> Barrier(
      const BarrierOptions& = BarrierOptions()) override;

  std::shared_ptr<ProcessGroup::Task> Broadcast(
      phi::DenseTensor* out_tensor,
      const phi::DenseTensor& in_tensor,
      const BroadcastOptions& opts,
      bool sync_op,
      bool use_calc_stream) override;

  std::shared_ptr<ProcessGroup::Task> Reduce(phi::DenseTensor* out_tensor,
                                             const phi::DenseTensor& in_tensor,
                                             const ReduceOptions& opts,
                                             bool sync_op,
                                             bool use_calc_stream) override;

  std::shared_ptr<ProcessGroup::Task> ReduceScatter(
      phi::DenseTensor* out_tensor,
      const phi::DenseTensor& in_tensor,
      const ReduceScatterOptions& opts,
      bool sync_op,
      bool use_calc_stream) override;

  std::shared_ptr<ProcessGroup::Task> Scatter(phi::DenseTensor* out_tensor,
                                              const phi::DenseTensor& in_tensor,
                                              const ScatterOptions& opts,
                                              bool sync_op,
                                              bool use_calc_stream) override;

  std::shared_ptr<ProcessGroup::Task> Gather(phi::DenseTensor* out_tensor,
                                             const phi::DenseTensor& in_tensor,
                                             const GatherOptions& opts,
                                             bool sync_op,
                                             bool use_calc_stream) override;

  std::shared_ptr<ProcessGroup::Task> Gather(
      std::vector<phi::DenseTensor>* gather_tensors_ptr,
      const phi::DenseTensor& in_tensor,
      const GatherOptions& opts,
      bool sync_op,
      bool use_calc_stream) override;

  std::shared_ptr<ProcessGroup::Task> Recv(phi::DenseTensor* tensor,
                                           int src_rank,
                                           int64_t offset,
                                           int64_t numel,
                                           bool sync_op,
                                           bool use_calc_stream) override;

  std::shared_ptr<ProcessGroup::Task> Send(const phi::DenseTensor& tensor,
                                           int dst_rank,
                                           int64_t offset,
                                           int64_t numel,
                                           bool sync_op,
                                           bool use_calc_stream) override;

  static void GroupStart(const std::string& dev_type);

  static void GroupEnd(const std::string& dev_type);

  phi::ccl::CCLComm XCCLComm(const Place& place) const;

  // TODO(liyurui): This API will be moved later
  std::shared_ptr<ProcessGroup::Task> AllReduce(
      std::vector<phi::DenseTensor>& in_tensors,
      std::vector<phi::DenseTensor>& out_tensors,
      const AllreduceOptions& = AllreduceOptions()) override;

  // TODO(sunyilun): methods below will be removed later
  std::shared_ptr<ProcessGroup::Task> Broadcast(
      std::vector<phi::DenseTensor>& in_tensors,
      std::vector<phi::DenseTensor>& out_tensors,
      const BroadcastOptions& = BroadcastOptions()) override;

  std::shared_ptr<ProcessGroup::Task> Send(
      std::vector<phi::DenseTensor>& tensors, int dst_rank) override;

  std::shared_ptr<ProcessGroup::Task> Recv(
      std::vector<phi::DenseTensor>& tensors, int src_rank) override;

  std::shared_ptr<ProcessGroup::Task> AllGather(
      std::vector<phi::DenseTensor>& in_tensors,
      std::vector<phi::DenseTensor>& out_tensors) override;

  std::shared_ptr<ProcessGroup::Task> AllToAll(
      std::vector<phi::DenseTensor>& in_tensors,
      std::vector<phi::DenseTensor>& out_tensors) override;

  std::shared_ptr<ProcessGroup::Task> Reduce(
      std::vector<phi::DenseTensor>& tensors,
      std::vector<phi::DenseTensor>& out_tensors,
      const ReduceOptions& opts) override;

  std::shared_ptr<ProcessGroup::Task> Scatter(
      std::vector<phi::DenseTensor>& in_tensors,
      std::vector<phi::DenseTensor>& out_tensors,
      const ScatterOptions& opts) override;

 private:
  std::shared_ptr<ProcessGroupCustom::XCCLTask> CreateTask(
      const Place& place,
      int rank,
      CommType op_type,
      bool sync_op,
      bool use_calc_stream);

  void BroadcastUniqueXCCLID(phi::ccl::CCLRootId* nccl_id);

  void CreateXCCLEnvCache(const Place& place, const std::string& place_key);

  void SyncCalcStream(const Place& place);

  std::shared_ptr<ProcessGroup::Task> RunFnInXCCLEnv(
      std::function<void(const phi::stream::Stream&)> fn,
      const phi::DenseTensor& tensor,
      CommType comm_type,
      bool sync_op,
      bool use_calc_stream);

  // TODO(sunyilun): methods below will be removed later
  std::shared_ptr<ProcessGroupCustom::XCCLTask> CreateTask(
      std::vector<Place> places,
      int rank,
      CommType op_type,
      const std::vector<phi::DenseTensor>& inputs);

  template <typename Fn>
  std::shared_ptr<ProcessGroup::Task> Collective(
      std::vector<phi::DenseTensor>& inputs,   // NOLINT
      std::vector<phi::DenseTensor>& outputs,  // NOLINT
      Fn fn,
      CommType op_type);

  template <typename Fn>
  std::shared_ptr<ProcessGroup::Task> PointToPoint(
      std::vector<phi::DenseTensor>& tensors,  // NOLINT
      Fn fn,
      int dst_rank,
      CommType op_type);

  void CreateXCCLManagerCache(const std::string& places_key,
                              const std::vector<Place>& places);

  phi::distributed::XCCLCommContext* GetCommContext();

 private:
  std::shared_ptr<phi::distributed::Store> store_;
  std::string device_type_;

  std::unordered_map<std::string, std::unique_ptr<phi::event::Event>>
      place_to_calc_event_;  // event on calc stream
  std::unordered_map<std::string, phi::CustomContext*> place_to_calc_ctx_;
  std::unordered_map<std::string, std::unique_ptr<phi::CustomContext>>
      place_to_comm_ctx_;

  // TODO(sunyilun): attrs below will be removed later
  std::mutex mutex_;
  std::unordered_map<std::string, std::vector<phi::CustomContext*>>
      places_to_ctx_;
};

}  //  namespace distributed
}  //  namespace paddle
