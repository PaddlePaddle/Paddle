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
#include "paddle/phi/backends/gpu/forwards.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/device_context.h"
#include "paddle/phi/core/distributed/nccl_comm_context.h"
#include "paddle/phi/core/distributed/store/store.h"
#include "paddle/phi/core/platform/device_event.h"

namespace paddle {
namespace distributed {

using Place = phi::Place;

class ProcessGroupNCCL final : public ProcessGroupWithStream {
 public:
  class NCCLTask final : public ProcessGroupWithStream::TaskStream,
                         public std::enable_shared_from_this<NCCLTask> {
   public:
    NCCLTask(const Place& place,
             int rank,
             CommType comm_type,
             bool sync_op,
             bool use_calc_stream,
             int gid);
    virtual ~NCCLTask();

    bool IsCompleted() override;
    bool Wait(std::chrono::milliseconds timeout = kWaitTimeout) override;
    void Synchronize() override;
    void UpdateWaitChain(const phi::DeviceContext& ctx) override;

    bool IsBlockCPUInWait() const { return block_cpu_in_wait_; }
    void SetBlockCPUInWait() { block_cpu_in_wait_ = true; }

    // TODO(sunyilun): methods below will be removed later
    NCCLTask(const std::vector<Place>& places,
             int rank,
             CommType CommType,
             const std::vector<phi::DenseTensor>& inputs);

    void RemoveHolderStreamInGroup();

   private:
    bool block_cpu_in_wait_{false};
    std::shared_ptr<platform::DeviceEvent> comm_event_;  // event on comm stream
    Place task_place_;
    int gid_;
  };

 public:
  static std::shared_ptr<ProcessGroupNCCL> CreateProcessGroupNCCL(
      const std::shared_ptr<phi::distributed::Store>& store,
      int rank,
      int size,
      int gid,
      int64_t timeout,
      int nccl_comm_init_option);

  ProcessGroupNCCL(const std::shared_ptr<phi::distributed::Store>& store,
                   int rank,
                   int size,
                   int gid,
                   int64_t timeout = 30 * 60 * 1000,
                   int nccl_comm_init_option = 0);
  ~ProcessGroupNCCL();

  std::string GetBackendName() const override { return "NCCL"; }

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

  static void GroupStart();

  static void GroupEnd();

  ncclComm_t NCCLComm(const Place& place) const;

  const bool GetNCCLCommInitOption() { return nccl_comm_init_option_; }

  phi::distributed::NCCLCommContext* GetOrCreateCommContext(
      const Place& place, CommType comm_type = CommType::UNKNOWN);

 private:
  std::shared_ptr<ProcessGroupNCCL::NCCLTask> CreateTask(const Place& place,
                                                         int rank,
                                                         CommType op_type,
                                                         bool sync_op,
                                                         bool use_calc_stream,
                                                         int gid);

  void GetStoreKey(const std::string& place_key,
                   CommType comm_type,
                   std::string* store_key);

  void CreateNCCLEnvCache(const Place& place,
                          const std::string& place_key,
                          const std::string& store_key,
                          CommType comm_type,
                          int p2p_rank = 0);

  void SyncCalcStream(const Place& place, const std::string& place_key);

  std::shared_ptr<ProcessGroup::Task> Collective(
      std::function<void(phi::distributed::NCCLCommContext*, gpuStream_t)> fn,
      const phi::DenseTensor& tensor,
      CommType comm_type,
      bool sync_op,
      bool use_calc_stream);

  std::shared_ptr<ProcessGroup::Task> Point2Point(
      std::function<void(phi::distributed::NCCLCommContext*, gpuStream_t, int)>
          fn,
      int peer,
      const phi::DenseTensor& tensor,
      CommType comm_type,
      bool sync_op,
      bool use_calc_stream);

  phi::distributed::NCCLCommContext* GetCommContext(
      const std::string* key = nullptr);

  void EraseTensorHolders() {
    for (const auto& allocation_stream : allocation_stream_pairs_) {
      auto holder_ptr = allocation_stream.first.lock();
      if (holder_ptr) {
        memory::EraseStream(holder_ptr, allocation_stream.second);
      }
    }
    VLOG(5) << "After task wait/synchronize, totoal "
            << allocation_stream_pairs_.size()
            << " tensor(s) allocation stream have been removed.";
    allocation_stream_pairs_.clear();
  }

  virtual void StartCoalescing();

  virtual void EndCoalescing(
      std::optional<std::vector<std::shared_ptr<ProcessGroup::Task>>>
          tasks_opt = std::nullopt);

  void EagerConnect();

  void EagerConnectRingExchange();

 private:
  std::shared_ptr<phi::distributed::Store> store_;

  std::unordered_map<std::string, platform::DeviceEvent>
      place_to_calc_event_;  // event on calc stream
  std::unordered_map<std::string, phi::GPUContext*> place_to_calc_ctx_;
  std::unordered_map<std::string, std::unique_ptr<phi::GPUContext>>
      place_to_comm_ctx_;

  uint64_t comm_seq_{0};
  std::unordered_map<std::string, uint64_t> p2p_comm_seq_;
  std::unordered_map<std::string, std::string> place_to_group_key_;

  // TODO(sunyilun): attrs below will be removed later
  std::mutex mutex_;
  static uint64_t s_group_call_counter;
  // default 30 minutes
  int64_t pg_timeout_;
  int nccl_comm_init_option_;

  // optimize memory for process_group
  std::vector<std::pair<std::weak_ptr<phi::Allocation>, gpuStream_t>>
      allocation_stream_pairs_;

  // For colaescing tensors processing (eg. batch_isend_irecv)
  bool is_coalescing_{false};
  std::vector<std::shared_ptr<phi::DenseTensor>> colaescing_tensors_;
  std::vector<std::string> colaescing_place_keys_;
};

}  //  namespace distributed
}  //  namespace paddle
