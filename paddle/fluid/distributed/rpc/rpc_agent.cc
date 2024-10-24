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

namespace paddle::distributed {

const int kTimeoutMs = 500000;
const int kConnectTimeoutMs = 10000;
const int kMaxRetry = 5;
const int kCloseWaitMs = 1000;
std::shared_ptr<RpcAgent> RpcAgent::rpc_agent_instance_ = nullptr;

RpcAgent::RpcAgent(std::string name, std::vector<WorkerInfo> infos) {
  name_ = std::move(name);
  for (const auto &info : infos) {
    name_to_infos_.insert({info.name_, info});
    id_to_infos_.insert({info.id_, info});
  }
  this->infos_ = std::move(infos);
  auto it = name_to_infos_.find(name_);
  this->rank_ = it->second.id_;
  rpc_service_ = std::make_shared<RpcService>();
  PADDLE_ENFORCE_EQ(
      server_.AddService(rpc_service_.get(), brpc::SERVER_DOESNT_OWN_SERVICE),
      0,
      common::errors::Fatal("Fail to add service: %s", name_));
}

int RpcAgent::StartWorker() {
  auto info = GetWorkerInfo(name_);
  // Start the server.
  int port = info.port_;
  brpc::ServerOptions options;
  PADDLE_ENFORCE_EQ(server_.Start(port, &options),
                    0,
                    common::errors::Fatal("Fail to start worker: %s", name_));
  VLOG(0) << "Start worker : " << name_;
  return 0;
}

int RpcAgent::StartClient() {
  // Initialize the channel, NULL means using default options.
  brpc::ChannelOptions channel_options;
  channel_options.protocol = "baidu_std";
  channel_options.timeout_ms = kTimeoutMs;
  channel_options.connection_type = "pooled";
  channel_options.connect_timeout_ms = kConnectTimeoutMs;
  channel_options.max_retry = kMaxRetry;
  channels_.resize(name_to_infos_.size());
  // build connection from client to all servers
  for (std::size_t i = 0; i < channels_.size(); i++) {
    auto info = id_to_infos_.find(i)->second;
    channels_[i].reset(new brpc::Channel());
    PADDLE_ENFORCE_EQ(
        channels_[i]->Init(info.ip_.c_str(), info.port_, &channel_options),
        0,
        common::errors::Fatal(
            "Fail to initialize channel: %d, ip: %s, port: %d",
            i,
            info.ip_,
            info.port_));
  }
  VLOG(0) << "Init Channels: " << name_;
  return 0;
}

int RpcAgent::Stop() {
  VLOG(0) << "Worker: " << name_ << " is going to stop.";
  server_.Stop(kCloseWaitMs);
  server_.Join();
  rpc_agent_instance_ = nullptr;
  VLOG(0) << "Worker: " << name_ << " has stopped";
  return 0;
}
void OnRpcDone::Run() {
  // delete this after Run
  std::unique_ptr<OnRpcDone> self_guard(this);
  PADDLE_ENFORCE_EQ(
      cntl_.Failed(), false, common::errors::Fatal(cntl_.ErrorText()));
  promise_->set_value(response_.message());
  VLOG(2) << "Received response from " << cntl_.remote_side() << " to "
          << cntl_.local_side() << " (attached=" << cntl_.response_attachment()
          << ")"
          << " latency=" << cntl_.latency_us() << "us";
}

std::future<std::string> RpcAgent::InvokeRpc(const std::string &py_func,
                                             const std::string &to,
                                             int timeout_ms = kTimeoutMs) {
  auto it = name_to_infos_.find(to);
  PADDLE_ENFORCE_NE(it,
                    name_to_infos_.end(),
                    common::errors::OutOfRange("Worker %s doesn't exist!", to));
  uint32_t id = it->second.id_;
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
  PADDLE_ENFORCE_NE(rpc_agent_instance_,
                    nullptr,
                    common::errors::Fatal(
                        "RpcAgent is not set, please calling "
                        "paddle.distributed.rpc.int_rpc() to init rpc agent."));
  return rpc_agent_instance_;
}
void RpcAgent::SetAgentInstance(std::shared_ptr<RpcAgent> agent) {
  PADDLE_ENFORCE_EQ(
      rpc_agent_instance_,
      nullptr,
      common::errors::Fatal(
          "RpcAgent has been set, please don't set rpc agent repeatedly."));
  rpc_agent_instance_ = agent;
}
}  // namespace paddle::distributed
