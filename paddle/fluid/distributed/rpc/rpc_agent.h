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
#include <string>
#include <unordered_map>
#include <vector>

#include "brpc/channel.h"
#include "brpc/server.h"
#include "paddle/common/macros.h"
#include "paddle/fluid/distributed/rpc/python_rpc_handler.h"
#include "paddle/fluid/distributed/rpc/rpc.pb.h"
#include "paddle/fluid/distributed/rpc/rpc_service.h"

namespace paddle {
namespace distributed {
struct WorkerInfo {
  std::string name_;
  uint32_t id_;
  std::string ip_;
  uint32_t port_;
  WorkerInfo(std::string name, uint32_t id, std::string ip, uint32_t port)
      : name_(std::move(name)), id_(id), ip_(std::move(ip)), port_(port) {}

  std::string to_string() const {
    std::string info = "{name: " + name_ + ", rank: " + std::to_string(id_) +
                       ", ip: " + ip_ + ", port: " + std::to_string(port_) +
                       "}";
    return info;
  }
};

class OnRpcDone : public google::protobuf::Closure {
 public:
  OnRpcDone() { promise_ = std::make_shared<std::promise<std::string>>(); }
  // process callback of response
  void Run();
  std::future<std::string> GetFuture() {
    return std::future<std::string>(promise_->get_future());
  }
  RpcResponse response_;
  RpcRequest request_;
  brpc::Controller cntl_;
  std::shared_ptr<std::promise<std::string>> promise_;
};

class RpcAgent {
 public:
  static std::shared_ptr<RpcAgent> RpcAgentInstance();
  static void SetAgentInstance(std::shared_ptr<RpcAgent> agent);
  // init RpcAgent instance and get information of all services
  RpcAgent(std::string name, std::vector<WorkerInfo> infos);
  ~RpcAgent() {}

  const WorkerInfo &GetWorkerInfo(const std::string &name) const {
    auto it = name_to_infos_.find(name);
    return it->second;
  }
  const WorkerInfo &GetWorkerInfoById(uint32_t id) const {
    auto it = id_to_infos_.find(id);
    return it->second;
  }
  const WorkerInfo &GetCurrentWorkerInfo() const {
    return GetWorkerInfo(name_);
  }
  const std::vector<WorkerInfo> &GetAllWorkerInfos() const {
    return this->infos_;
  }

  uint32_t Rank() { return this->rank_; }

  uint32_t WorldSize() { return infos_.size(); }

  int StartWorker();
  // build connection from client to all servers
  int StartClient();
  int Stop();

  std::future<std::string> InvokeRpc(const std::string &msg,
                                     const std::string &to,
                                     int timeout_ms);

 private:
  DISABLE_COPY_AND_ASSIGN(RpcAgent);
  static std::shared_ptr<RpcAgent> rpc_agent_instance_;
  brpc::Server server_;
  std::shared_ptr<RpcService> rpc_service_;
  std::vector<std::shared_ptr<brpc::Channel>> channels_;
  std::string name_;
  uint32_t rank_;
  std::unordered_map<std::string, WorkerInfo> name_to_infos_;
  std::unordered_map<uint32_t, WorkerInfo> id_to_infos_;
  std::vector<WorkerInfo> infos_;
};
}  // namespace distributed
}  // namespace paddle
