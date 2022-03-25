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
#include "paddle/fluid/distributed/ps/service/brpc_ps_client.h"
#include "paddle/fluid/distributed/ps/service/brpc_utils.h"
#include "paddle/fluid/distributed/ps/service/sendrecv.pb.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/framework/variable_helper.h"
#include "paddle/fluid/platform/macros.h"  // for DISABLE_COPY_AND_ASSIGN
#include "paddle/fluid/string/split.h"

namespace paddle {
namespace framework {
class Scope;
}  // namespace framework
}  // namespace paddle
DECLARE_int32(pserver_timeout_ms);
namespace paddle {
namespace distributed {

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
      options.ssl_options.enable = need_encrypt;
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
        auto ip_port = paddle::string::Split(node_list[i], ':');
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

  void SendAndRecvAsync(const platform::DeviceContext& ctx,
                        const framework::Scope& scope,
                        const std::string& message_name,
                        const std::vector<std::string>& send_var_name,
                        const std::vector<std::string>& recv_var_name,
                        const std::string& mode = "forward");

  int Send(const platform::DeviceContext& ctx, const framework::Scope& scope,
           const std::string& message_name,
           const std::vector<std::string>& send_var_names) {
    const framework::Scope* p_scope = &scope;  // 注意是 const
    OnHeterRpcDone* closure = new OnHeterRpcDone([](void* done) {
      auto* closure = reinterpret_cast<OnHeterRpcDone*>(done);
      int ret = 0;
      closure->set_promise_value(ret);
      PADDLE_ENFORCE_NE(
          closure->cntl.Failed(), true,
          platform::errors::Unimplemented(
              "HeterClient::SendToSwitch meets brpc error, error message is %s",
              closure->cntl.ErrorText()));
    });

    closure->cntl.set_timeout_ms(FLAGS_pserver_timeout_ms);
    auto& request_io_buffer = closure->cntl.request_attachment();

    distributed::MultiVarMsg request;
    // 1. set req message_name(string)
    request.set_message_name(message_name);

    // 2. set req send_var_names(<string>)
    for (auto& send_var_name : send_var_names) {
      request.add_send_var_names(send_var_name);
    }

    // 3. set req var_messages(<VarMessage>)
    for (auto& send_var_name : send_var_names) {
      auto* send_var_msg = request.add_var_messages();
      send_var_msg->set_varname(send_var_name);
      framework::Variable* var = p_scope->FindVar(send_var_name);
      butil::IOBuf temp_iobuf;
      if (var->IsType<framework::LoDTensor>()) {
        SerializeLodTensor(var, ctx, send_var_msg, &temp_iobuf);
      } else if (var->IsType<phi::SelectedRows>()) {
        SerializeSelectedRows(var, ctx, send_var_msg, &temp_iobuf);
      }
      request_io_buffer.append(temp_iobuf);
    }
    auto promise = std::make_shared<std::promise<int32_t>>();
    closure->add_promise(promise);
    std::future<int> fut = promise->get_future();
    if (send_switch_channels_.empty()) {
      LOG(ERROR) << "send_switch_channels_ is null, get xpu_channels_[0]";
      if (xpu_channels_.empty()) {
        LOG(ERROR) << "xpu_channels_ is null";
      }
      send_switch_channels_.push_back(xpu_channels_[0]);
    }
    brpc::Channel* channel = send_switch_channels_[0].get();
    // brpc::Channel* channel = xpu_channels_[0].get();
    ::paddle::distributed::PsService_Stub stub(channel);
    stub.SendToSwitch(&closure->cntl, &request, &closure->ps_response, closure);
    VLOG(4) << "waiting SendToSwitch response result......";
    fut.wait();
    VLOG(4) << "Send done";
    return 0;
  }

  int Recv(const platform::DeviceContext& ctx,
           framework::Scope& recv_scope,  // NOLINT
           const std::string& message_name,
           const std::vector<std::string>& recv_var_names) {
    OnHeterRpcDone* closure = new OnHeterRpcDone([](void* done) {
      auto* closure = reinterpret_cast<OnHeterRpcDone*>(done);
      VLOG(4) << "Recv service call done";
      int ret = 0;
      closure->set_promise_value(ret);
      PADDLE_ENFORCE_NE(
          closure->cntl.Failed(), true,
          platform::errors::Unimplemented("HeterClient::RecvFromSwitch meets "
                                          "brpc error, error message is %s",
                                          closure->cntl.ErrorText()));
    });

    closure->cntl.set_timeout_ms(FLAGS_pserver_timeout_ms);

    distributed::MultiVarMsg request;
    // 1. set req message_name(string)
    request.set_message_name(message_name);

    // 2. set req recv_var_names(<string>)
    for (auto& recv_var_name : recv_var_names) {
      request.add_recv_var_names(recv_var_name);
    }
    auto promise = std::make_shared<std::promise<int32_t>>();
    closure->add_promise(promise);
    std::future<int> fut = promise->get_future();
    if (recv_switch_channels_.empty()) {
      LOG(ERROR) << "peer_switch_channels_ is null, get xpu_channels_[1]";
      if (xpu_channels_.size() < 2) {
        LOG(ERROR) << "xpu_channels_ is null";
      }
      recv_switch_channels_.push_back(xpu_channels_[1]);
    }
    brpc::Channel* channel = recv_switch_channels_[0].get();
    ::paddle::distributed::PsService_Stub stub(channel);
    stub.RecvFromSwitch(&closure->cntl, &request, &closure->response, closure);
    fut.wait();
    VLOG(4) << "RecvFromSwitch done";
    // save in worker
    platform::DeviceContextPool& pool = platform::DeviceContextPool::Instance();
    platform::CPUPlace cpu_place;
    auto& cpu_dev_ctx = *pool.Get(cpu_place);
    auto& res_io_buffer = closure->cntl.response_attachment();
    VLOG(4) << "entering DeserializeFromMultiVarMsgAndIOBuf";
    distributed::DeserializeFromMultiVarMsgAndIOBuf(
        closure->response, &res_io_buffer, cpu_dev_ctx, &recv_scope);
    VLOG(4) << "Recv done";
    return 0;
  }

  // HeterClient singleton
  static std::shared_ptr<HeterClient> GetInstance(
      const std::vector<std::string>& endpoint,
      const std::vector<std::string>& previous_endpoint,
      const int& trainer_id) {
    if (NULL == s_instance_) {
      s_instance_.reset(new HeterClient());
      s_instance_->SetXpuList(endpoint);
      s_instance_->SetPreviousXpuList(previous_endpoint);
      s_instance_->SetTrainerID(trainer_id);
      s_instance_->CreateClient2XpuConnection();
    }
    return s_instance_;
  }

  // switch client singleton
  static HeterClient& GetSwitchInstance(
      const std::vector<std::string>& peer_endpoints, int32_t peer_role) {
    static HeterClient switch_s_instance_;
    if (peer_endpoints.empty()) {
      LOG(ERROR) << "init switch client failed, null peer_endpoints";
    }
    VLOG(4) << "peer role is: " << peer_role
            << ", addr is: " << peer_endpoints[0];
    switch_s_instance_.SetPeerSwitchList(peer_endpoints);
    switch_s_instance_.InitClientChannels(false, peer_endpoints, peer_role);
    return switch_s_instance_;
  }

  void SetPeerSwitchList(const std::vector<std::string>& peer_endpoints) {
    peer_switch_list_ = peer_endpoints;
  }

  void SetPeerWorkerList(const std::vector<std::string>& worker_endpoints) {
    peer_worker_list_ = worker_endpoints;
  }

  void Stop();

  std::future<int32_t> SendCmd(uint32_t table_id, int cmd_id,
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
  std::vector<std::shared_ptr<brpc::Channel>> xpu_channels_;
  std::vector<std::shared_ptr<brpc::Channel>> previous_xpu_channels_;

  // DISABLE_COPY_AND_ASSIGN(HeterClient);
  std::vector<std::string> xpu_list_;
  std::vector<std::string> previous_xpu_list_;

  int trainer_id_;
};

}  // end namespace distributed
}  // end namespace paddle
