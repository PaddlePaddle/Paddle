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
#include <mutex>

#include "paddle/fluid/distributed/collective/ProcessGroup.h"

#ifdef PADDLE_WITH_GLOO
#include "paddle/fluid/framework/fleet/gloo_wrapper.h"
#endif

#include "paddle/fluid/distributed/store/store.h"
#include "paddle/fluid/distributed/store/tcp_store.h"

constexpr const char* GLOO_BACKEND_NAME = "GLOO";

namespace paddle {
namespace distributed {

class ProcessGroupGloo : public ProcessGroup {
 public:
  class GlooTask : public ProcessGroup::Task,
                   public std::enable_shared_from_this<GlooTask> {
   public:
    explicit GlooTask(int rank, const std::vector<Tensor>& input_tensors,
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
    explicit GlooStore(const std::shared_ptr<paddle::distributed::Store>& store)
        : _store(store) {}

    ~GlooStore() = default;

    std::vector<char> get(const std::string& key) override {
      VLOG(3) << "GlooStore::get";
      auto value = _store->get(key);
      return std::vector<char>(value.begin(), value.end());
    }

    void wait(const std::vector<std::string>& keys) override {
      VLOG(3) << "GlooStore::wait";
      for (auto& key : keys) {
        _store->wait(key);
      }
    }

    void set(const std::string& key, const std::vector<char>& value) override {
      VLOG(3) << "GlooStore::set";
      std::vector<uint8_t> tmp(value.begin(), value.end());
      _store->set(key, tmp);
    }

    void wait(const std::vector<std::string>& keys,
              const std::chrono::milliseconds& timeout) override {
      VLOG(3) << "GlooStore::wait";
      for (auto& key : keys) {
        _store->wait(key);
      }
      // wait(keys);
    }

   protected:
    std::shared_ptr<paddle::distributed::Store> _store;
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

  explicit ProcessGroupGloo(
      const std::shared_ptr<paddle::distributed::Store>& store, int rank,
      int world_size, std::shared_ptr<GlooOptions> options);

  ~ProcessGroupGloo() = default;

  std::shared_ptr<ProcessGroup::Task> Broadcast(
      std::vector<Tensor>& inputs,
      const BroadcastOptions& = BroadcastOptions()) override;

  std::shared_ptr<ProcessGroup::Task> AllReduce(
      std::vector<Tensor>& inputs,
      const AllreduceOptions& opts = AllreduceOptions()) override;

  std::shared_ptr<ProcessGroup::Task> Barrier(
      const BarrierOptions& = BarrierOptions()) override;

  std::shared_ptr<ProcessGroup::Task> AllGather(
      std::vector<Tensor>& in_tensors,
      std::vector<Tensor>& out_tensors) override;

  std::shared_ptr<ProcessGroup::Task> Reduce(
      std::vector<Tensor>& tensors, const ReduceOptions& opts) override;

  std::shared_ptr<ProcessGroup::Task> Scatter(std::vector<Tensor>& in_tensors,
                                              std::vector<Tensor>& out_tensors,
                                              const ScatterOptions&) override;

  std::shared_ptr<::gloo::Context> get_context() { return _context; }
  uint64_t next_tag() { return _tag++; }

  const std::string GetBackendName() const override {
    return GLOO_BACKEND_NAME;
  }

  // Helper functions for Gloo.
  static std::shared_ptr<::gloo::transport::Device> createDeviceForHostname(
      const std::string& hostname);
  static std::shared_ptr<::gloo::transport::Device> createDeviceForInterface(
      const std::string& ifname);
  static std::shared_ptr<::gloo::transport::Device> createDefaultDevice();

 protected:
  uint32_t _tag;
  std::shared_ptr<gloo::rendezvous::Context> _context;
  std::shared_ptr<::gloo::rendezvous::Store> _store;
};

}  // namespace distributed
}  // namespace paddle
