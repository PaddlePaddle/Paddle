/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once
#include <atomic>
#include <ctime>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

#include "brpc/channel.h"
#include "brpc/controller.h"
#include "brpc/server.h"
#include "paddle/common/macros.h"  // for DISABLE_COPY_AND_ASSIGN
#include "paddle/fluid/distributed/ps/service/brpc_ps_client.h"
#include "paddle/fluid/distributed/ps/service/brpc_utils.h"
#include "paddle/fluid/distributed/ps/service/sendrecv.pb.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/framework/variable_helper.h"
#include "paddle/utils/string/split.h"

namespace paddle {
namespace framework {
class Scope;
}  // namespace framework
}  // namespace paddle

namespace paddle {
namespace distributed {
PD_DECLARE_int32(pserver_timeout_ms);
using MultiVarMsg = ::paddle::distributed::MultiVariableMessage;
using VarMsg = ::paddle::distributed::VariableMessage;

typedef std::function<void(void*)> HeterRpcCallbackFunc;

class OnHeterRpcDone : public google::protobuf::Closure {
 public:
  explicit OnHeterRpcDone(HeterRpcCallbackFunc func) : handler_(func) {}
  virtual ~OnHeterRpcDone() {}
  void Run() { handler_(this); }

  void add_promise(std::shared_ptr<std::promise<int32_t>>& promise) {  // NOLINT
    _promises.push_back(promise);
  }

  void set_promise_value(int value) {
    for (auto& promise : _promises) {
      promise->set_value(value);
    }
  }
  int CheckResponse() { return 0; }
  std::vector<std::shared_ptr<std::promise<int32_t>>> _promises;
  HeterRpcCallbackFunc handler_;

  MultiVariableMessage request;
  MultiVariableMessage response;

  PsResponseMessage ps_response;

  brpc::Controller cntl;
  // PsRequestMessage *request(size_t i) { return &_requests[i]; }
  // PsResponseMessage *response(size_t i) { return &_responses[i]; }
  // std::vector<PsRequestMessage> _requests;
  // std::vector<PsResponseMessage> _responses;
  // std::vector<std::shared_ptr<brpc::Controller>> _cntls;
};

class HeterClient {
 public:
  virtual ~HeterClient() {}

  void InitClientChannels(bool need_encrypt,
                          const std::vector<std::string>& node_list,
                          int32_t peer_role) {
    brpc::ChannelOptions options;
    options.protocol = "baidu_std";
    options.connection_type = "single";
    options.timeout_ms = FLAGS_pserver_timeout_ms;
    std::vector<std::shared_ptr<brpc::Channel>>* client_channels = nullptr;
    if (peer_role == PEER_ROLE_IS_SWITCH) {
#ifdef PADDLE_WITH_ARM_BRPC
      if (need_encrypt) {
        options.mutable_ssl_options();
      }
      options.connection_type = "";
      VLOG(4) << "ssl enabled in arm";
#else
      if (need_encrypt) {
        options.mutable_ssl_options();
      }
#endif
      client_channels = &peer_switch_channels_;
    } else if (peer_role == PEER_ROLE_IS_WORKER) {
      client_channels = &peer_worker_channels_;
    } else {
      LOG(ERROR) << "init switch client failed, peer_role not valid";
    }
    (*client_channels).resize(node_list.size());
    for (size_t i = 0; i < node_list.size(); ++i) {
      (*client_channels)[i].reset(new brpc::Channel());
      if ((*client_channels)[i]->Init(node_list[i].c_str(), "", &options) !=
          0) {
        VLOG(0) << "client channel init failed! try again";
        auto ip_port = ::paddle::string::Split(node_list[i], ':');
        std::string ip = ip_port[0];
        int port = std::stoi(ip_port[1]);
        std::string int_ip_port = GetIntTypeEndpoint(ip, port);
        if ((*client_channels)[i]->Init(int_ip_port.c_str(), "", &options) !=
            0) {
          LOG(ERROR) << "client channel init failed! peer ip_port = "
                     << int_ip_port;
        }
      }
    }
    VLOG(4) << "InitClientChannels success";
  }

  void CreateClient2XpuConnection();

  void SendAndRecvAsync(const phi::DeviceContext& ctx,
                        const framework::Scope& scope,
                        const std::string& message_name,
                        const std::vector<std::string>& send_var_name,
                        const std::vector<std::string>& recv_var_name,
                        const std::string& mode = "forward");

  int Send(int group_id,
           const std::vector<std::string>& var_names,
           const std::vector<int64_t>& vars_len,
           void* data_ptr,
           int64_t data_size);

  int Send(const phi::DeviceContext& ctx,
           const framework::Scope& scope,
           const std::string& message_name,
           const std::vector<std::string>& send_var_names);

  int Recv(int group_id,
           const std::vector<std::string>& var_names,
           void* data_ptr,
           int64_t data_size);

  int Recv(const phi::DeviceContext& ctx,
           framework::Scope& recv_scope,  // NOLINT
           const std::string& message_name,
           const std::vector<std::string>& recv_var_names);

  // HeterClient singleton
  static std::shared_ptr<HeterClient> GetInstance(
      const std::vector<std::string>& endpoints,
      const std::vector<std::string>& previous_endpoints,
      const int& trainer_id) {
    if (NULL == s_instance_) {
      s_instance_.reset(new HeterClient());
      s_instance_->SetXpuList(endpoints);
      s_instance_->SetPreviousXpuList(previous_endpoints);
      s_instance_->SetTrainerID(trainer_id);
      s_instance_->CreateClient2XpuConnection();
    }
    return s_instance_;
  }

  // switch client singleton
  static std::shared_ptr<HeterClient> GetSwitchInstance(
      const std::vector<std::string>& peer_endpoints, int32_t peer_role) {
    std::unique_lock<std::mutex> lock(mtx_);
    if (peer_endpoints.empty()) {
      VLOG(4) << "init switch client failed, null peer_endpoints";
    }
    VLOG(4) << "peer role is: " << peer_role
            << ", addr is: " << peer_endpoints[0];
    if (switch_s_instance_ == nullptr) {
      switch_s_instance_.reset(new HeterClient());
      switch_s_instance_->SetPeerSwitchList(peer_endpoints);
      switch_s_instance_->InitClientChannels(false, peer_endpoints, peer_role);
    }
    return switch_s_instance_;
  }

  void SetPeerSwitchList(const std::vector<std::string>& peer_endpoints) {
    peer_switch_list_ = peer_endpoints;
  }

  void SetPeerWorkerList(const std::vector<std::string>& worker_endpoints) {
    peer_worker_list_ = worker_endpoints;
  }

  void Stop();

  std::future<int32_t> SendCmd(uint32_t table_id,
                               int cmd_id,
                               const std::vector<std::string>& params);

  std::future<int32_t> StartProfiler();

  std::future<int32_t> StopProfiler();
  std::future<int32_t> StopHeterWorker();

  std::vector<std::string>& GetXpuList() { return xpu_list_; }

  void SetXpuList(const std::vector<std::string>& xpu_list) {
    xpu_list_ = xpu_list;
  }

  void SetPreviousXpuList(const std::vector<std::string>& xpu_list) {
    previous_xpu_list_ = xpu_list;
  }

  void SetTrainerID(const int& trainer_id) { trainer_id_ = trainer_id; }

 public:
  std::vector<std::string> send_switch_list_;
  std::vector<std::string> recv_switch_list_;

  std::vector<std::string> peer_switch_list_;
  std::vector<std::string> peer_worker_list_;
  std::vector<std::shared_ptr<brpc::Channel>> send_switch_channels_;
  std::vector<std::shared_ptr<brpc::Channel>> recv_switch_channels_;

  std::vector<std::shared_ptr<brpc::Channel>> peer_switch_channels_;
  std::vector<std::shared_ptr<brpc::Channel>> peer_worker_channels_;

 private:
  HeterClient() {}
  HeterClient& operator=(const HeterClient&);
  HeterClient(const HeterClient&);

  static std::shared_ptr<HeterClient> s_instance_;
  static std::mutex mtx_;
  static std::shared_ptr<HeterClient> switch_s_instance_;
  std::vector<std::shared_ptr<brpc::Channel>> xpu_channels_;
  std::vector<std::shared_ptr<brpc::Channel>> previous_xpu_channels_;

  // DISABLE_COPY_AND_ASSIGN(HeterClient);
  std::vector<std::string> xpu_list_;
  std::vector<std::string> previous_xpu_list_;

  int trainer_id_;
};

}  // namespace distributed
}  // namespace paddle
