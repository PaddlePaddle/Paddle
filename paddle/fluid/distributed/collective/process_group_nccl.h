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
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "paddle/fluid/distributed/collective/gin_overlap/kernels/comm_args.h"
#include "paddle/fluid/distributed/collective/process_group.h"
#include "paddle/fluid/distributed/collective/process_group_with_stream.h"
#include "paddle/phi/backends/gpu/forwards.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/device_context.h"
#include "paddle/phi/core/distributed/comm_context_manager.h"
#include "paddle/phi/core/distributed/nccl_comm_context.h"
#include "paddle/phi/core/distributed/nccl_config.h"
#include "paddle/phi/core/distributed/store/store.h"
#include "paddle/phi/core/platform/device_event.h"

namespace paddle {
namespace distributed {

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
             const std::vector<DenseTensor>& inputs);

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
      int nccl_comm_init_option,
      std::shared_ptr<phi::distributed::NCCLConfig> nccl_config = nullptr);

  ProcessGroupNCCL(
      const std::shared_ptr<phi::distributed::Store>& store,
      int rank,
      int size,
      int gid,
      int64_t timeout = 30 * 60 * 1000,
      int nccl_comm_init_option = 0,
      std::shared_ptr<phi::distributed::NCCLConfig> nccl_config = nullptr);
  ~ProcessGroupNCCL();

  void EraseStream(const DenseTensor& tensor) const override;

  std::string GetBackendName() const override { return "NCCL"; }

  phi::DeviceContext* GetDeviceContext(const Place& place) const override;

  phi::DeviceContext* GetDeviceContext(const Place& place,
                                       bool use_calc_stream) const override;

  std::shared_ptr<ProcessGroup::Task> AllGather(DenseTensor* out_tensor,
                                                const DenseTensor& in_tensor,
                                                int64_t offset,
                                                int64_t numel,
                                                bool sync_op,
                                                bool use_calc_stream) override;

  std::shared_ptr<ProcessGroup::Task> AllReduce(DenseTensor* out_tensor,
                                                const DenseTensor& in_tensor,
                                                const AllreduceOptions& opts,
                                                bool sync_op,
                                                bool use_calc_stream) override;

  std::shared_ptr<ProcessGroup::Task> AllToAll(
      DenseTensor* out_tensor,
      const DenseTensor& in_tensor,
      const std::vector<int64_t>& out_size_each_rank,
      const std::vector<int64_t>& in_size_each_rank,
      bool sync_op,
      bool use_calc_stream) override;

  std::shared_ptr<ProcessGroup::Task> AllToAll(
      std::vector<DenseTensor>* out_tensors,
      const std::vector<DenseTensor>& in_tensors,
      bool sync_op,
      bool use_calc_stream) override;

  std::shared_ptr<ProcessGroup::Task> Barrier(
      const BarrierOptions& = BarrierOptions()) override;

  std::shared_ptr<ProcessGroup::Task> Broadcast(DenseTensor* out_tensor,
                                                const DenseTensor& in_tensor,
                                                const BroadcastOptions& opts,
                                                bool sync_op,
                                                bool use_calc_stream) override;

  std::shared_ptr<ProcessGroup::Task> Reduce(DenseTensor* out_tensor,
                                             const DenseTensor& in_tensor,
                                             const ReduceOptions& opts,
                                             bool sync_op,
                                             bool use_calc_stream) override;

  std::shared_ptr<ProcessGroup::Task> ReduceScatter(
      DenseTensor* out_tensor,
      const DenseTensor& in_tensor,
      const ReduceScatterOptions& opts,
      bool sync_op,
      bool use_calc_stream) override;

  std::shared_ptr<ProcessGroup::Task> Scatter(DenseTensor* out_tensor,
                                              const DenseTensor& in_tensor,
                                              const ScatterOptions& opts,
                                              bool sync_op,
                                              bool use_calc_stream) override;

  std::shared_ptr<ProcessGroup::Task> Gather(DenseTensor* out_tensor,
                                             const DenseTensor& in_tensor,
                                             const GatherOptions& opts,
                                             bool sync_op,
                                             bool use_calc_stream) override;

  std::shared_ptr<ProcessGroup::Task> Gather(
      std::vector<DenseTensor>* gather_tensors_ptr,
      const DenseTensor& in_tensor,
      const GatherOptions& opts,
      bool sync_op,
      bool use_calc_stream) override;

  std::shared_ptr<ProcessGroup::Task> Recv(DenseTensor* tensor,
                                           int src_rank,
                                           int64_t offset,
                                           int64_t numel,
                                           bool sync_op,
                                           bool use_calc_stream) override;

  std::shared_ptr<ProcessGroup::Task> Send(const DenseTensor& tensor,
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

  phi::DeviceContext* GetOrCreateCalcContext(const Place& place,
                                             bool use_calc_stream);

  void Shutdown();
  void Restart();
  phi::CUDAStream GetStream(const Place& place);
  void SetOuterEventWait(bool outer_wait);

  void EagerConnectRingExchange(
      std::shared_ptr<phi::distributed::NCCLConfig> nccl_config);

  // GIN primitives. They need IsGinSupported() and raise Unavailable
  // otherwise. They act on the NCCL communicator of the current device, and
  // the stream arguments are CUDA streams of that device.
  //
  // Same LSA team: the LSA team of a rank is the run of lsa_size consecutive
  // ranks that contains it (NCCLDevCommInfo), all on one node. Peers in this
  // rank's LSA team are reached by GPU loads and stores through LSA pointers,
  // other peers through GIN (the NIC).

  // True when Paddle was built with PADDLE_WITH_GIN_OVERLAP, the current GPU
  // has compute capability 9.0 or newer, and the loaded NCCL provides the host
  // entry points of the NCCL device (LSA/GIN) API.
  static bool IsGinSupported();

  // Creates the device communicator on the first call (collective, see
  // NCCLCommContext::GetOrCreateDevComm()) and returns its properties. Every
  // primitive except AllocHostWindow needs it. It raises when
  // the LSA team of the device communicator does not match node-major rank
  // order (lsa_rank != rank % lsa_size).
  phi::distributed::NCCLDevCommInfo GetOrCreateDevComm(int gin_context_count);

  // Properties of the device communicator. Raises when GetOrCreateDevComm()
  // has not been called.
  phi::distributed::NCCLDevCommInfo GetDevCommInfo();

  // Allocates and registers a pinned host memory window (collective, see
  // NCCLCommContext::AllocHostWindow()).
  phi::distributed::NCCLHostWindow AllocHostWindow(size_t bytes);

  // Issues GIN puts on `stream`. Each put increments its counter after its
  // data has been written, so a peer that sees its counter reach a value knows
  // the data of that many puts has arrived. The source must not be changed
  // before Flush() on a later point of the stream. num_blocks is the number of
  // thread blocks per launch; 0 means one block per put. A launch takes at
  // most 32 puts; longer lists are split into launches on the same stream.
  void Put(const std::vector<PutArgs>& puts,
           gpuStream_t stream,
           int num_blocks = 0);

  // Increments peer counters on `stream`: through an LSA atomic add
  // (__threadfence_system() then atomicAdd_system) for peers in this rank's
  // LSA team, through a GIN signal for the others. Work queued on `stream`
  // before this call has finished before an increment is made. lsa_counter of
  // the arguments is ignored and filled in here.
  void IncrementPeerCounter(
      const std::vector<IncrementPeerCounterArgs>& increments,
      gpuStream_t stream);

  // Copies to windows of peers in this rank's LSA team on `stream`, with one
  // cudaMemcpyBatchAsync for the whole list.
  void CopyToPeers(const std::vector<CopyToPeersArgs>& copies,
                   gpuStream_t stream);

  // Flushes every GIN context on `stream`: after it, the sources of the puts
  // issued so far may be reused.
  void Flush(gpuStream_t stream);

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

  void CreateNCCLEnvCache(
      const Place& place,
      const std::string& place_key,
      const std::string& store_key,
      CommType comm_type,
      int p2p_rank = 0,
      std::shared_ptr<phi::distributed::NCCLConfig> nccl_config = nullptr);

  void SyncCalcStream(const Place& place, const std::string& place_key);

  std::shared_ptr<ProcessGroup::Task> Collective(
      std::function<void(phi::distributed::NCCLCommContext*, gpuStream_t)> fn,
      const std::vector<DenseTensor>& tensors,
      CommType comm_type,
      bool sync_op,
      bool use_calc_stream);

  std::shared_ptr<ProcessGroup::Task> Collective(
      std::function<void(phi::distributed::NCCLCommContext*, gpuStream_t)> fn,
      const DenseTensor& tensor,
      CommType comm_type,
      bool sync_op,
      bool use_calc_stream);

  std::shared_ptr<ProcessGroup::Task> Point2Point(
      std::function<void(phi::distributed::NCCLCommContext*, gpuStream_t, int)>
          fn,
      int peer,
      const DenseTensor& tensor,
      CommType comm_type,
      bool sync_op,
      bool use_calc_stream);

  phi::distributed::NCCLCommContext* GetCommContext(
      const std::string* key = nullptr);

  void EraseTensorHolders();

  virtual void StartCoalescing();

  virtual void EndCoalescing(
      std::optional<std::vector<std::shared_ptr<ProcessGroup::Task>>>
          tasks_opt = std::nullopt);

  void EagerConnect();

  void EagerConnectRingExchange();

  // The NCCLCommContext of the current device. Raises when it has not been
  // created, instead of creating it (creation is collective).
  phi::distributed::NCCLCommContext* GetGinCommContext();

  // Converts a group rank to its index in this rank's LSA team, or -1 when it
  // is in another LSA team. Raises when `peer` is not a rank of this group.
  int PeerToLsaIndex(const phi::distributed::NCCLDevCommInfo& dev_comm_info,
                     int peer) const;

 private:
  std::shared_ptr<phi::distributed::Store> store_;

  std::unordered_map<std::string, platform::DeviceEvent>
      place_to_calc_event_;  // event on calc stream
  std::unordered_map<std::string, phi::GPUContext*> place_to_calc_ctx_;
  std::unordered_map<std::string, std::unique_ptr<phi::GPUContext>>
      place_to_comm_ctx_;

  uint64_t comm_seq_{0};
  std::unordered_map<std::string, uint64_t> p2p_comm_seq_;
  std::map<std::string, std::string> place_to_group_key_;

  // TODO(sunyilun): attrs below will be removed later
  std::mutex mutex_;
  static uint64_t s_group_call_counter;
  // default 30 minutes
  int64_t pg_timeout_;
  int nccl_comm_init_option_;

  // optimize memory for process_group
  std::vector<std::pair<std::weak_ptr<phi::Allocation>, gpuStream_t>>
      allocation_stream_pairs_;

  // For coalescing tensors processing (eg. batch_isend_irecv)
  bool is_coalescing_{false};
  std::vector<std::shared_ptr<DenseTensor>> coalescing_tensors_;
  std::vector<std::string> coalescing_place_keys_;

  std::unordered_map<std::string, phi::distributed::P2POption>
      place_to_p2p_opts_;
  int64_t create_count_;
  std::shared_ptr<phi::distributed::NCCLConfig> nccl_config_ptr_;

  bool outer_wait_{false};
};

}  //  namespace distributed
}  //  namespace paddle
