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
#include "glog/logging.h"
#include "paddle/fluid/distributed/rpc/python_rpc_handler.h"
#include "paddle/fluid/distributed/rpc/rpc.pb.h"
#include "paddle/fluid/distributed/rpc/rpc_service.h"

namespace paddle {
namespace distributed {
struct ServiceInfo {
  std::string name_;
  uint32_t id_;
  std::string ip_;
  uint32_t port_;
  ServiceInfo(std::string name, uint32_t id, std::string ip, uint32_t port)
      : name_(std::move(name)), id_(id), ip_(std::move(ip)), port_(port) {}
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
  RpcAgent(std::string name, std::vector<ServiceInfo> infos);
  ~RpcAgent() {}
  RpcAgent(const RpcAgent &) = delete;
  RpcAgent &operator=(const RpcAgent &) = delete;

  const ServiceInfo &GetServiceInfo(const std::string &name) const {
    auto it = nameToInfos_.find(name);
    return it->second;
  }
  const ServiceInfo &GetServiceInfoById(uint32_t id) const {
    auto it = idToInfos_.find(id);
    return it->second;
  }
  const ServiceInfo &GetCurrentServiceInfo() const {
    return GetServiceInfo(name_);
  }
  const std::vector<ServiceInfo> &GetAllServiceInfos() const {
    return this->infos_;
  }

  uint32_t Rank() { return this->rank_; }

  uint32_t WorldSize() { return infos_.size(); }

  int StartServer();
  // build connection from client to all servers
  int StartClient();
  int Stop();

  std::string Send(const std::string &msg, const std::string &to);

  std::future<std::string> InvokeRpc(const std::string &msg,
                                     const std::string &to,
                                     int time_out_ms);

 private:
  static std::shared_ptr<RpcAgent> rpcAgentInstance_;
  brpc::Server server_;
  std::shared_ptr<RpcService> rpcService_;
  std::vector<std::shared_ptr<brpc::Channel>> channels_;
  std::string name_;
  uint32_t rank_;
  std::unordered_map<std::string, ServiceInfo> nameToInfos_;
  std::unordered_map<uint32_t, ServiceInfo> idToInfos_;
  std::vector<ServiceInfo> infos_;
};
}  // namespace distributed
}  // namespace paddle
