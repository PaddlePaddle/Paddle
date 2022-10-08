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

#include "paddle/fluid/distributed/rpc/rpc_agent.h"

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace distributed {

const int kTimeoutMs = 500000;
const int kConnectTimeoutMs = 10000;
const int kMaxRetry = 5;
const int kCloseWaitMs = 1000;
std::shared_ptr<RpcAgent> RpcAgent::rpcAgentInstance_ = nullptr;

RpcAgent::RpcAgent(std::string name, std::vector<WorkerInfo> infos) {
  name_ = std::move(name);
  for (auto info : infos) {
    nameToInfos_.insert({info.name_, info});
    idToInfos_.insert({info.id_, info});
  }
  this->infos_ = std::move(infos);
  auto it = nameToInfos_.find(name_);
  this->rank_ = it->second.id_;
  rpcService_ = std::make_shared<RpcService>();
  if (server_.AddService(rpcService_.get(), brpc::SERVER_DOESNT_OWN_SERVICE) !=
      0) {
    LOG(ERROR) << "Fail to add service: " << name;
  }
}

int RpcAgent::StartServer() {
  auto info = GetWorkerInfo(name_);
  // Start the server.
  int port = info.port_;
  brpc::ServerOptions options;
  if (server_.Start(port, &options) != 0) {
    LOG(ERROR) << "Fail to start Server: " << name_;
    return -1;
  }
  LOG(INFO) << "Start Server : " << name_;
  return 0;
}

int RpcAgent::StartClient() {
  // Initialize the channel, NULL means using default options.
  brpc::ChannelOptions channelOptions;
  channelOptions.protocol = "baidu_std";
  channelOptions.timeout_ms = kTimeoutMs;
  channelOptions.connection_type = "pooled";
  channelOptions.connect_timeout_ms = kConnectTimeoutMs;
  channelOptions.max_retry = kMaxRetry;
  channels_.resize(nameToInfos_.size());
  // build connection from client to all servers
  for (std::size_t i = 0; i < channels_.size(); i++) {
    auto info = idToInfos_.find(i)->second;
    channels_[i].reset(new brpc::Channel());
    if (channels_[i]->Init(info.ip_.c_str(), info.port_, &channelOptions) !=
        0) {
      LOG(ERROR) << "Fail to initialize channel: " << i << ", ip: " << info.ip_
                 << ", port: " << info.port_;
      return -1;
    }
  }
  LOG(INFO) << "Init Channels: " << name_;
  return 0;
}

int RpcAgent::Stop() {
  LOG(INFO) << "Start stopping server: " << name_;
  server_.Stop(kCloseWaitMs);
  server_.Join();
  rpcAgentInstance_ = nullptr;
  LOG(INFO) << "Server " << name_ << " stoppped";
  return 0;
}
void OnRpcDone::Run() {
  // delete this after Run
  std::unique_ptr<OnRpcDone> self_guard(this);
  if (!cntl_.Failed()) {
    promise_->set_value(response_.message());
    VLOG(2) << "Received response from " << cntl_.remote_side() << " to "
            << cntl_.local_side()
            << " (attached=" << cntl_.response_attachment() << ")"
            << " latency=" << cntl_.latency_us() << "us";
  } else {
    LOG(ERROR) << cntl_.ErrorText();
  }
}
std::string RpcAgent::Send(const std::string &msg, const std::string &to) {
  auto it = nameToInfos_.find(to);
  uint32_t id = channels_.size();
  if (it != nameToInfos_.end()) {
    id = it->second.id_;
  } else {
    PADDLE_ENFORCE_NE(
        id,
        channels_.size(),
        platform::errors::OutOfRange("Worker %s doesn't exist!", to));
  }
  auto channel = channels_[id];
  brpc::Controller cntl;
  RpcResponse response;
  RpcRequest request;
  request.set_message(msg);
  RpcBaseService_Stub stub(channel.get());
  stub.Send(&cntl, &request, &response, NULL);
  if (cntl.Failed()) {
    LOG(ERROR) << "ERROR Send message: " << msg << " from "
               << cntl.remote_side() << " to " << cntl.local_side();
    return "ERROR";
  } else {
    return response.message();
  }
}

std::future<std::string> RpcAgent::InvokeRpc(const std::string &py_func,
                                             const std::string &to,
                                             int timeout_ms = kTimeoutMs) {
  auto it = nameToInfos_.find(to);
  uint32_t id = channels_.size();
  if (it != nameToInfos_.end()) {
    id = it->second.id_;
  } else {
    PADDLE_ENFORCE_NE(
        id,
        channels_.size(),
        platform::errors::OutOfRange("Worker %s doesn't exist!", to));
  }
  auto channel = channels_[id];
  // `done` must be allocated on the heap because its life cycle is after
  // calling done.Run().
  OnRpcDone *done = new OnRpcDone;
  done->cntl_.set_timeout_ms(timeout_ms);
  done->request_.set_message(py_func);
  std::future<std::string> fut = done->GetFuture();
  RpcBaseService_Stub stub(channel.get());
  stub.InvokeRpc(&done->cntl_, &done->request_, &done->response_, done);
  return fut;
}

std::shared_ptr<RpcAgent> RpcAgent::RpcAgentInstance() {
  PADDLE_ENFORCE_NE(rpcAgentInstance_,
                    nullptr,
                    platform::errors::Fatal(
                        "RpcAgent is not set, please calling "
                        "paddle.distributed.rpc.int_rpc() to init rpc agent."));
  return rpcAgentInstance_;
}
void RpcAgent::SetAgentInstance(std::shared_ptr<RpcAgent> agent) {
  PADDLE_ENFORCE_EQ(
      rpcAgentInstance_,
      nullptr,
      platform::errors::Fatal(
          "RpcAgent has been set, please don't set rpc agent repeatly."));
  rpcAgentInstance_ = agent;
}
}  // namespace distributed
}  // namespace paddle
