/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/operators/detail/grpc_server.h"

using grpc::ServerAsyncResponseWriter;

namespace paddle {
namespace operators {
namespace detail {

enum CallStatus { PROCESS = 0, FINISH };

// reference:
// https://stackoverflow.com/questions/41732884/grpc-multiple-services-in-cpp-async-server
class RequestBase {
 public:
  explicit RequestBase(sendrecv::SendRecvService::AsyncService* service,
                       grpc::ServerCompletionQueue* cq)
      : service_(service), cq_(cq), status_(PROCESS) {
    assert(cq_);
  }
  virtual ~RequestBase() {}
  virtual void Process() { assert(false); }

  CallStatus Status() { return status_; }
  void SetStatus(CallStatus status) { status_ = status; }
  virtual std::string GetReqName() { assert(false); }

 protected:
  grpc::ServerContext ctx_;
  sendrecv::SendRecvService::AsyncService* service_;
  grpc::ServerCompletionQueue* cq_;
  CallStatus status_;
};

typedef std::pair<std::string, sendrecv::VariableMessage> MessageWithName;

class RequestSend final : public RequestBase {
 public:
  explicit RequestSend(sendrecv::SendRecvService::AsyncService* service,
                       grpc::ServerCompletionQueue* cq,
                       SimpleBlockQueue<MessageWithName>* queue)
      : RequestBase(service, cq), queue_(queue), responder_(&ctx_) {
    service_->RequestSendVariable(&ctx_, &request_, &responder_, cq_, cq_,
                                  this);
  }

  virtual ~RequestSend() {}

  virtual std::string GetReqName() { return request_.varname(); }

  virtual void Process() {
    MessageWithName msg_with_name =
        std::make_pair(request_.varname(), std::move(request_));
    queue_->Push(std::move(msg_with_name));
    // TODO(gongwb): check var's info.
    responder_.Finish(reply_, grpc::Status::OK, this);
    status_ = FINISH;
  }

 protected:
  sendrecv::VariableMessage request_;
  sendrecv::VoidMessage reply_;
  SimpleBlockQueue<MessageWithName>* queue_;
  ServerAsyncResponseWriter<sendrecv::VoidMessage> responder_;
};

class RequestGet final : public RequestBase {
 public:
  explicit RequestGet(sendrecv::SendRecvService::AsyncService* service,
                      grpc::ServerCompletionQueue* cq, framework::Scope* scope)
      : RequestBase(service, cq), responder_(&ctx_), scope_(scope) {
    service_->RequestGetVariable(&ctx_, &request_, &responder_, cq_, cq_, this);
  }

  virtual ~RequestGet() {}

  virtual std::string GetReqName() { return request_.varname(); }

  virtual void Process() {
    // proc request.
    std::string var_name = request_.varname();
    auto* var = scope_->FindVar(var_name);
    SerializeToMessage(var_name, var, platform::CPUDeviceContext(), &reply_);
    // TODO(gongwb): check var's info.
    responder_.Finish(reply_, grpc::Status::OK, this);
    status_ = FINISH;
  }

 protected:
  sendrecv::VariableMessage request_;
  sendrecv::VariableMessage reply_;
  ServerAsyncResponseWriter<sendrecv::VariableMessage> responder_;
  framework::Scope* scope_;
};

void AsyncGRPCServer::RunSyncUpdate() {
  grpc::ServerBuilder builder;
  builder.AddListeningPort(address_, grpc::InsecureServerCredentials());
  builder.SetMaxSendMessageSize(std::numeric_limits<int>::max());
  builder.SetMaxReceiveMessageSize(std::numeric_limits<int>::max());
  builder.RegisterService(&service_);

  cq_send_ = builder.AddCompletionQueue();
  cq_get_ = builder.AddCompletionQueue();
  server_ = builder.BuildAndStart();
  LOG(INFO) << "Server listening on " << address_ << std::endl;

  std::function<void()> send_register =
      std::bind(&AsyncGRPCServer::TryToRegisterNewSendOne, this);
  std::function<void()> get_register =
      std::bind(&AsyncGRPCServer::TryToRegisterNewGetOne, this);

  t_send_.reset(
      new std::thread(std::bind(&AsyncGRPCServer::HandleRequest, this, false,
                                cq_send_.get(), "cq_send", send_register)));

  t_get_.reset(
      new std::thread(std::bind(&AsyncGRPCServer::HandleRequest, this, true,
                                cq_get_.get(), "cq_get", get_register)));

  // wait server
  server_->Wait();
  t_send_->join();
  t_get_->join();
}

void AsyncGRPCServer::ShutdownQueue() {
  std::unique_lock<std::mutex> lock(cq_mutex_);
  cq_send_->Shutdown();
  cq_get_->Shutdown();
  is_shut_down_ = true;
}

// This URL explains why shutdown is complicate:
// https://stackoverflow.com/questions/35708348/grpc-what-is-the-recommended-way-to-shut-down-an-asynchronous-server-in-c
void AsyncGRPCServer::ShutDown() {
  server_->Shutdown();
  ShutdownQueue();
}

void AsyncGRPCServer::TryToRegisterNewSendOne() {
  std::unique_lock<std::mutex> lock(cq_mutex_);
  if (is_shut_down_) {
    return;
  }
  RequestSend* send =
      new RequestSend(&service_, cq_send_.get(), &var_recv_queue_);
  VLOG(4) << "create RequestSend status:" << send->Status();
}

void AsyncGRPCServer::TryToRegisterNewGetOne() {
  std::unique_lock<std::mutex> lock(cq_mutex_);
  if (is_shut_down_) {
    return;
  }
  RequestGet* get = new RequestGet(&service_, cq_get_.get(), scope_);
  VLOG(4) << "create Requestget status:" << get->Status();
}

void AsyncGRPCServer::HandleRequest(bool wait, grpc::ServerCompletionQueue* cq,
                                    std::string cq_name,
                                    std::function<void()> TryToRegisterNewOne) {
  TryToRegisterNewOne();

  void* tag = NULL;
  bool ok = false;
  while (true) {
    if (!cq->Next(&tag, &ok)) {
      LOG(INFO) << cq_name << " get CompletionQueue shutdown!";
      break;
    }

    assert(tag);
    if (wait && !done_) {
      Wait();
    }

    RequestBase* base = (RequestBase*)tag;
    // reference:
    // https://github.com/tensorflow/tensorflow/issues/5596
    // https://groups.google.com/forum/#!topic/grpc-io/xftlRy-IQwM
    if (!ok) {
      LOG(WARNING) << cq_name << " recv no regular event:argument name"
                   << base->GetReqName();
      // FIXME(gongwb): delete the old one?
      TryToRegisterNewOne();
      continue;
    }

    switch (base->Status()) {
      case PROCESS: {
        VLOG(4) << cq_name << " status:" << base->Status();
        TryToRegisterNewOne();
        base->Process();
        break;
      }
      case FINISH: {
        VLOG(4) << cq_name << " status:" << base->Status();
        delete base;
        break;
      }
      default: { assert(false); }
    }
  }
}

void AsyncGRPCServer::Wait() {
  std::unique_lock<std::mutex> lock(this->mutex_);
  condition_.wait(lock, [=] { return this->done_ == true; });
}

void AsyncGRPCServer::Reset() {
  std::lock_guard<std::mutex> lock(this->mutex_);
  done_ = false;
}

void AsyncGRPCServer::Done() {
  {
    std::lock_guard<std::mutex> lock(this->mutex_);
    done_ = true;
  }
  condition_.notify_all();
}

}  // namespace detail
}  // namespace operators
}  // namespace paddle
