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

#include <future>
#include <memory>
#include <mutex>

#include "paddle/fluid/distributed/collective/process_group.h"
#include "paddle/fluid/distributed/collective/process_group_without_stream.h"
#include "paddle/phi/backends/context_pool.h"
#include "paddle/phi/core/distributed/gloo_comm_context.h"
#include "paddle/phi/core/distributed/store/store.h"
#include "paddle/phi/core/distributed/store/tcp_store.h"

namespace paddle {
namespace distributed {

class ProcessGroupGloo : public ProcessGroupWithoutStream {
 public:
  class GlooTask : public ProcessGroup::Task,
                   public std::enable_shared_from_this<GlooTask> {
   public:
    explicit GlooTask(int rank,
                      const std::vector<phi::DenseTensor>& input_tensors,
                      CommType comm_type);

    ~GlooTask() = default;

    virtual void Run() = 0;
    bool Wait(std::chrono::milliseconds timeout) override { return true; }
    bool IsCompleted() override { return true; }
    void Synchronize() override {}

   protected:
    friend class ProcessGroupGloo;
  };

  class GlooStore : public ::gloo::rendezvous::Store {
   public:
    explicit GlooStore(const std::shared_ptr<phi::distributed::Store>& store)
        : _store(store) {}

    ~GlooStore() = default;

    std::vector<char> get(const std::string& key) override;
    void wait(const std::vector<std::string>& keys) override;

    void set(const std::string& key, const std::vector<char>& value) override;

    void wait(const std::vector<std::string>& keys,
              const std::chrono::milliseconds& timeout) override;

   protected:
    std::shared_ptr<phi::distributed::Store> _store;
  };

  class GlooOptions {
   public:
    GlooOptions() = default;
    ~GlooOptions() = default;
    static std::shared_ptr<GlooOptions> create() {
      return std::make_shared<GlooOptions>();
    }
    std::shared_ptr<::gloo::transport::Device> device;
  };

  ProcessGroupGloo(const std::shared_ptr<phi::distributed::Store>& store,
                   int rank,
                   int world_size,
                   int gid,
                   std::shared_ptr<GlooOptions> options);

  static std::shared_ptr<ProcessGroupGloo> CreateProcessGroupGloo(
      const std::shared_ptr<phi::distributed::Store>& store,
      int rank,
      int world_size,
      int gid);

  ~ProcessGroupGloo() = default;

  std::shared_ptr<ProcessGroup::Task> AllGather(
      phi::DenseTensor* out_tensor,
      const phi::DenseTensor& in_tensor,
      int64_t /*offset*/,  // for compatibility, no use now
      int64_t /*numel*/,   // for compatibility, no use now
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

  std::shared_ptr<ProcessGroup::Task> Send(const phi::DenseTensor& tensor,
                                           int dst_rank,
                                           bool sync_op) override;

  std::shared_ptr<ProcessGroup::Task> Recv(phi::DenseTensor* tensor,
                                           int src_rank,
                                           bool sync_op) override;

  std::shared_ptr<ProcessGroup::Task> Reduce(phi::DenseTensor* out_tensor,
                                             const phi::DenseTensor& in_tensor,
                                             const ReduceOptions& opts,
                                             bool sync_op) override;

  std::shared_ptr<ProcessGroup::Task> Scatter(phi::DenseTensor* out_tensor,
                                              const phi::DenseTensor& in_tensor,
                                              const ScatterOptions& opts,
                                              bool sync_op) override;

  std::shared_ptr<ProcessGroup::Task> Gather(phi::DenseTensor* out_tensor,
                                             const phi::DenseTensor& in_tensor,
                                             const GatherOptions& opts,
                                             bool sync_op,
                                             bool use_calc_stream) override;

  // TODO(sunyilun): methods below will be removed later
  std::shared_ptr<ProcessGroup::Task> Broadcast(
      std::vector<phi::DenseTensor>& inputs,
      std::vector<phi::DenseTensor>& outputs,
      const BroadcastOptions& = BroadcastOptions()) override;

  std::shared_ptr<ProcessGroup::Task> Broadcast(
      std::vector<phi::DenseTensor>& inputs,
      std::vector<phi::DenseTensor>& outputs,
      const BroadcastOptions& opts,
      bool sync_op) override;

  std::shared_ptr<ProcessGroup::Task> Send(
      std::vector<phi::DenseTensor>& inputs, int dst_rank) override;

  std::shared_ptr<ProcessGroup::Task> Recv(
      std::vector<phi::DenseTensor>& outputs, int src_rank) override;

  std::shared_ptr<ProcessGroup::Task> AllReduce(
      std::vector<phi::DenseTensor>& inputs,
      std::vector<phi::DenseTensor>& outputs,
      const AllreduceOptions& opts = AllreduceOptions()) override;

  std::shared_ptr<ProcessGroup::Task> AllReduce(
      std::vector<phi::DenseTensor>& inputs,
      std::vector<phi::DenseTensor>& outputs,
      const AllreduceOptions& opts,
      bool sync_op) override;

  std::shared_ptr<ProcessGroup::Task> Barrier(
      const BarrierOptions& = BarrierOptions()) override;

  std::shared_ptr<ProcessGroup::Task> AllGather(
      std::vector<phi::DenseTensor>& in_tensors,
      std::vector<phi::DenseTensor>& out_tensors) override;

  std::shared_ptr<ProcessGroup::Task> AllGather(
      std::vector<phi::DenseTensor>& in_tensors,
      std::vector<phi::DenseTensor>& out_tensors,
      bool sync_op) override;

  std::shared_ptr<ProcessGroup::Task> Reduce(
      std::vector<phi::DenseTensor>& in_tensors,
      std::vector<phi::DenseTensor>& out_tensors,
      const ReduceOptions& opts) override;

  std::shared_ptr<ProcessGroup::Task> Scatter(
      std::vector<phi::DenseTensor>& in_tensors,
      std::vector<phi::DenseTensor>& out_tensors,
      const ScatterOptions&) override;

  std::shared_ptr<::gloo::Context> get_context() { return _context; }
  uint64_t next_tag() { return _tag++; }

  std::string GetBackendName() const override { return "GLOO"; }

  phi::DeviceContext* GetDeviceContext(const Place& place) const override {
    return phi::DeviceContextPool::Instance().Get(place);
  }

  phi::DeviceContext* GetDeviceContext(const Place& place,
                                       bool use_calc_stream) const override {
    PADDLE_ENFORCE_NE(
        use_calc_stream,
        true,
        common::errors::InvalidArgument("Gloo cannot use use_calc_stream."));
    return GetDeviceContext(place);
  }

  phi::distributed::GlooCommContext* GetCommContext();

  // Helper functions for Gloo.
  static std::shared_ptr<::gloo::transport::Device> createDeviceForHostname(
      const std::string& hostname);
  static std::shared_ptr<::gloo::transport::Device> createDeviceForInterface(
      const std::string& ifname);
  static std::shared_ptr<::gloo::transport::Device> createDefaultDevice();

 private:
  uint32_t _tag;
  std::shared_ptr<gloo::rendezvous::Context> _context;
  std::shared_ptr<::gloo::rendezvous::Store> _store;
};

}  // namespace distributed
}  // namespace paddle
