/*Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <unistd.h>
#include <limits>
#include <memory>
#include <string>

#include "paddle/fluid/operators/distributed/grpc/grpc_serde.h"
#include "paddle/fluid/operators/distributed/grpc/grpc_server.h"

namespace grpc {
class ChannelArguments;
}  // namespace grpc
namespace paddle {
namespace framework {
class Variable;
}  // namespace framework
namespace operators {
namespace distributed {
class GRPCVariableResponse;
}  // namespace distributed
}  // namespace operators
}  // namespace paddle

using ::grpc::ServerAsyncResponseWriter;

DECLARE_bool(rpc_disable_reuse_port);
DECLARE_int32(rpc_retry_bind_port);

namespace paddle {
namespace operators {
namespace distributed {

// reference:
// https://stackoverflow.com/questions/41732884/grpc-multiple-services-in-cpp-async-server
class OriginRequestBase {
 public:
  explicit OriginRequestBase(sendrecv::SendRecvService::AsyncService* service,
                             ::grpc::ServerCompletionQueue* cq,
                             RequestHandler* request_handler, int req_id)
      : service_(service),
        cq_(cq),
        status_(PROCESS),
        request_handler_(request_handler),
        req_id_(req_id) {
    PADDLE_ENFORCE_NOT_NULL(cq_, platform::errors::InvalidArgument(
                                     "ServerCompletionQueue cq are empty"));
  }
  virtual ~OriginRequestBase() {}
  virtual void Process() = 0;

  std::string Status2String(const std::string& method) {
    std::string status = "Process";
    if (status_ == FINISH) {
      status = "Finish";
    }

    std::ostringstream s;
    s << method << " name:[" << GetReqName() << "]"
      << ", ep:[" << ctx_.peer() << "]"
      << " " << status << " using req_id:" << req_id_;
    return s.str();
  }

  CallStatus Status() const {
    std::lock_guard<std::mutex> l(status_mu_);
    return status_;
  }

  template <typename T>
  void Finish(const T& reply, ServerAsyncResponseWriter<T>* responder) {
    std::lock_guard<std::mutex> l(status_mu_);
    status_ = FINISH;
    responder->Finish(reply, ::grpc::Status::OK,
                      reinterpret_cast<void*>(static_cast<intptr_t>(req_id_)));
  }
  virtual std::string GetReqName() = 0;

 protected:
  mutable std::mutex status_mu_;
  ::grpc::ServerContext ctx_;
  sendrecv::SendRecvService::AsyncService* service_;
  ::grpc::ServerCompletionQueue* cq_;
  CallStatus status_;
  RequestHandler* request_handler_;
  int req_id_;
};

class RequestSendAndRecv final : public OriginRequestBase {
 public:
  // sendrecv::SendRecvService::Service
  explicit RequestSendAndRecv(sendrecv::SendRecvService::AsyncService* service,
                              ::grpc::ServerCompletionQueue* cq,
                              RequestHandler* request_handler, int req_id)
      : OriginRequestBase(service, cq, request_handler, req_id),
        responder_(&ctx_) {
    request_helper_.reset(new GRPCMultiVariableResponseHelper(
        request_handler->scope(), request_handler->dev_ctx(), true));
    service_->RequestSendAndRecvVariable(
        &ctx_, &request_, &responder_, cq_, cq_,
        reinterpret_cast<void*>(static_cast<intptr_t>(req_id)));
    VLOG(1) << "RequestSendAndRecv finish SendAndRecvVariable";
  }

  virtual ~RequestSendAndRecv() {}
  std::string GetReqName() override { return request_.message_name(); }

  void Process() override {
    VLOG(1) << "RequestSendAndRecv finish GetMultiVariableMessage";
    std::string message_name = request_.message_name();
    std::vector<std::string> in_var_names(request_.send_var_names_size());
    std::vector<std::string> out_var_names(request_.recv_var_names_size());

    for (int in_var_index = 0; in_var_index < request_.send_var_names_size();
         ++in_var_index) {
      in_var_names[in_var_index] = request_.send_var_names(in_var_index);
    }

    for (int out_var_index = 0; out_var_index < request_.recv_var_names_size();
         ++out_var_index) {
      out_var_names[out_var_index] = request_.recv_var_names(out_var_index);
    }

    VLOG(1) << "RequestSendAndRecv, message_name: " << message_name;
    int trainer_id = 0;
    DeserializeFromMultiVarMsg(request_, *request_handler_->dev_ctx(),
                               request_helper_->GetMutableLocalScope(),
                               &trainer_id);
    VLOG(1) << "RequestSendAndRecv finish DeserializeFromMultiVarMsg";
    request_handler_->SetMultiVarNames(in_var_names, out_var_names);
    framework::Variable* fake_in_var = nullptr;
    framework::Variable* fake_out_var = nullptr;
    request_handler_->Handle(message_name,
                             request_helper_->GetMutableLocalScope(),
                             fake_in_var, &fake_out_var, trainer_id);
    VLOG(1) << "RequestSendAndRecv finish Handle";
    SerializeToMultiVarMsg(
        message_name, in_var_names, out_var_names, *request_handler_->dev_ctx(),
        request_helper_->GetMutableLocalScope(), &reply_, trainer_id);
    VLOG(4) << "RequestSendAndRecv finish SerializeToMultiVarMsg";
    Finish(reply_, &responder_);
    // std::lock_guard<std::mutex> l(status_mu_);
    // status_ = FINISH;
    // responder_.Finish(reply_, ::grpc::Status::OK,
    //                   reinterpret_cast<void*>(static_cast<intptr_t>(req_id_)));
    VLOG(1) << "RequestSendAndRecv Finish";
  }

 protected:
  MultiVarMsg request_;
  MultiVarMsg reply_;
  ServerAsyncResponseWriter<MultiVarMsg> responder_;
  std::shared_ptr<GRPCMultiVariableResponseHelper> request_helper_;
  framework::Scope* scope_;
  platform::DeviceContext* dev_ctx_;
};

void HeterAsyncGRPCServer::WaitServerReady() {
  VLOG(4) << "AsyncGRPCServer is waiting server ready";
  std::unique_lock<std::mutex> lock(this->mutex_ready_);
  condition_ready_.wait(lock, [=] { return this->ready_ == 1; });
  VLOG(4) << "AsyncGRPCServer WaitSeverReady";
}

void HeterAsyncGRPCServer::StartServer() {
  for (int i = 0; i < FLAGS_rpc_retry_bind_port; i++) {
    ::grpc::ServerBuilder builder;
    std::unique_ptr<sendrecv::SendRecvService::AsyncService> service(
        new sendrecv::SendRecvService::AsyncService());
    builder.AddListeningPort(bind_address_, ::grpc::InsecureServerCredentials(),
                             &selected_port_);

    builder.SetMaxSendMessageSize(std::numeric_limits<int>::max());
    builder.SetMaxReceiveMessageSize(std::numeric_limits<int>::max());
    if (FLAGS_rpc_disable_reuse_port) {
      builder.SetOption(
          std::unique_ptr<::grpc::ServerBuilderOption>(new NoReusePortOption));
      LOG(INFO) << "set FLAGS_rpc_disable_reuse_port";
    }
    builder.RegisterService(service.get());

    for (auto t : rpc_call_map_) {
      rpc_cq_[t.first].reset(builder.AddCompletionQueue().release());
    }

    server_ = builder.BuildAndStart();
    if (selected_port_ != 0) {
      LOG(INFO) << "Server listening on " << bind_address_
                << " successful, selected port: " << selected_port_;
      // service_.reset(service.release());
      break;
    }

    LOG(WARNING) << "Server listening on " << bind_address_
                 << " failed, selected port: " << selected_port_
                 << ", retry after 3 seconds!";
    sleep(3);
  }

  PADDLE_ENFORCE_NE(
      selected_port_, 0,
      platform::errors::Unavailable("can't bind to address:%s", bind_address_));

  std::function<void(const std::string&, int)> f =
      std::bind(&HeterAsyncGRPCServer::TryToRegisterNewOne, this,
                std::placeholders::_1, std::placeholders::_2);

  for (auto& t : rpc_call_map_) {
    auto& rpc_name = t.first;
    auto& cq = rpc_cq_[rpc_name];
    auto threadnum = rpc_thread_num_[rpc_name];
    auto& reqs = rpc_reqs_[rpc_name];

    reqs.reserve(kRequestBufSize);

    for (int i = 0; i < kRequestBufSize; i++) {
      VLOG(6) << "TryToRegisterNewOne on RPC NAME: " << rpc_name << " I: " << i;
      TryToRegisterNewOne(rpc_name, i);
    }

    for (int i = 0; i < threadnum; i++) {
      rpc_threads_[rpc_name].emplace_back(new std::thread(std::bind(
          &HeterAsyncGRPCServer::HandleRequest, this, cq.get(), rpc_name, f)));
      VLOG(4) << t.first << " creates threads!";
    }
  }

  {
    std::lock_guard<std::mutex> lock(this->mutex_ready_);
    ready_ = 1;
  }
  condition_ready_.notify_all();

  // wait server
  server_->Wait();

  for (auto& t : rpc_threads_) {
    auto& threads = t.second;
    for (size_t i = 0; i < threads.size(); ++i) {
      threads[i]->join();
      VLOG(4) << t.first << " threads ends!";
    }
  }
}

void HeterAsyncGRPCServer::ShutdownQueue() {
  for (auto& t : rpc_cq_) {
    t.second->Shutdown();
    VLOG(4) << t.first << " queue shutdown!";
  }
}

void HeterAsyncGRPCServer::ShutDownImpl() {
  std::unique_lock<std::mutex> lock(cq_mutex_);
  is_heter_shut_down_ = true;
  ShutdownQueue();

  VLOG(4) << "server_ shutdown!";
  server_->Shutdown();
}

void HeterAsyncGRPCServer::TryToRegisterNewOne(const std::string& rpc_name,
                                               int req_id) {
  std::unique_lock<std::mutex> lock(cq_mutex_);
  if (is_heter_shut_down_) {
    VLOG(4) << "shutdown, do not TryToRegisterNewSendOne";
    return;
  }

  VLOG(4) << "TryToRegisterNewOne on RPC NAME: " << rpc_name
          << " REQ ID: " << req_id;

  auto& reqs = rpc_reqs_[rpc_name];
  auto& handler = rpc_call_map_[rpc_name];
  auto& cq = rpc_cq_[rpc_name];

  OriginRequestBase* b = nullptr;
  if (rpc_name == kRequestSendAndRecv) {
    b = new RequestSendAndRecv(service_.get(), cq.get(), handler, req_id);
  } else {
    PADDLE_THROW(
        platform::errors::InvalidArgument("not supported rpc: %s", rpc_name));
  }

  reqs[req_id] = b;

  VLOG(4) << "TryToRegisterNewOne status:" << b->Status();
}

void HeterAsyncGRPCServer::HandleRequest(
    ::grpc::ServerCompletionQueue* cq, const std::string& rpc_name,
    std::function<void(const std::string&, int)> TryToRegisterNewOne) {
  void* tag = NULL;
  bool ok = false;

  while (true) {
    VLOG(4) << "HandleRequest " << rpc_name << " wait next";
    if (!cq->Next(&tag, &ok)) {
      VLOG(4) << "CompletionQueue " << rpc_name << " shutdown!";
      break;
    }

    int req_id = static_cast<int>(reinterpret_cast<intptr_t>(tag));
    VLOG(4) << "HandleRequest " << rpc_name << ", req_id:" << req_id
            << " get next";

    auto& reqs = rpc_reqs_[rpc_name];
    OriginRequestBase* base = nullptr;
    {
      PADDLE_ENFORCE_EQ(
          (req_id >= 0 && req_id < kRequestBufSize), true,
          platform::errors::OutOfRange("request id: %s out of bounds: [0, %s)",
                                       req_id, kRequestBufSize));
      std::unique_lock<std::mutex> lock(cq_mutex_);
      base = reqs[req_id];
    }

    VLOG(3) << base->Status2String(rpc_name);

    // reference:
    // https://github.com/tensorflow/tensorflow/issues/5596
    // https://groups.google.com/forum/#!topic/grpc-io/xftlRy-IQwM
    // https://groups.google.com/forum/#!topic/grpc-io/ywATt88Ef_I
    if (!ok) {
      VLOG(4) << "completion queue:" << rpc_name << " recv no regular event"
              << " context:" << base->Status2String(rpc_name);
      TryToRegisterNewOne(rpc_name, req_id);
      delete base;
      continue;
    }

    switch (base->Status()) {
      case PROCESS: {
        base->Process();
        break;
      }
      case FINISH: {
        TryToRegisterNewOne(rpc_name, req_id);
        delete base;
        break;
      }
      default: { assert(false); }
    }
  }
}

}  // namespace distributed
}  // namespace operators
}  // namespace paddle
